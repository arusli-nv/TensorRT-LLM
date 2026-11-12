# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run-time rank mask and abort of the fence MoE All-to-All ops.

Every logical peer lives in one allocation and one rank issues ops, so the tests check masking and polling only.
"""

import ctypes
import os
import subprocess
import sys
import time

import pytest
import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm.bindings import internal
from tensorrt_llm.mapping import Mapping

_CAPACITY, _HIDDEN = 8, 128
# The abort state holds ``requested`` and ``status``, then per peer the flag value a long wait on it expects.
_ABORT_WORDS = int(internal.thop.MOE_A2A_ABORT_STATE_WORDS)


def _offset(meta: torch.Tensor, name: str) -> int:
    return int(meta[int(getattr(internal.thop, f"MOE_A2A_{name}_OFFSET_INDEX"))])


def _word(workspace: torch.Tensor, offset: int, row: int = 0) -> torch.Tensor:
    return workspace[row, offset : offset + 4].view(torch.int32)


def _mask_words(mask: int) -> torch.Tensor:
    """Return the rank mask as the kernels' four little-endian uint64 words."""
    return torch.tensor([(mask >> (64 * w)) & (2**64 - 1) for w in range(4)], dtype=torch.uint64)


def _rank_mask_controls(mask: int) -> tuple[dict, torch.Tensor]:
    """Return the rank-mask controls of a dispatch or combine and their abort state.

    The kernels read the device mask at execution time. The abort state lives in pinned host memory.
    """
    abort = torch.zeros(_ABORT_WORDS, dtype=torch.int32).pin_memory()
    words = _mask_words(mask).cuda()
    return {"enable_rank_mask": True, "active_rank_mask": words, "abort_state": abort}, abort


def _setup(ep_size: int, tokens: int = _CAPACITY) -> tuple[torch.Tensor, ...]:
    meta = NVLinkOneSided._make_workspace_layout(
        ep_size, 1, tokens, _HIDDEN, torch.bfloat16, None, 32, False, False, rank_mask=True
    )
    size = int(meta[int(internal.thop.MOE_A2A_WORKSPACE_SIZE_INDEX)])
    workspace = torch.zeros((ep_size, size), dtype=torch.uint8, device="cuda")
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, 0, ep_size, barrier=False)
    experts = torch.zeros((tokens, 1), dtype=torch.int32, device="cuda")
    payload = torch.arange(tokens * _HIDDEN, dtype=torch.bfloat16, device="cuda")
    return workspace, meta, experts, payload.reshape(tokens, _HIDDEN)


def _dispatch(workspace, meta, experts, controls, payload, tokens=_CAPACITY, rank=0):
    """Dispatch [payload, experts] like the MoE layer as ``rank``.

    Returns:
        The received tensors of shape [ep_size, tokens, ...], the combine offset and the gathered EPLB statistics.
    """
    ep_size = workspace.size(0)
    args = (experts, [payload, experts], workspace, meta, tokens, rank, ep_size, 1, ep_size, None)
    return torch.ops.trtllm.moe_a2a_dispatch(*args, **controls)


def _combine(workspace, meta, payload, offset, controls, tokens=_CAPACITY):
    """Combine like the MoE layer as rank 0."""
    args = (payload, tokens, workspace, meta, tokens, 0, workspace.size(0), 1, offset, False)
    return torch.ops.trtllm.moe_a2a_combine(*args, **controls)


def _next_parity(workspace: torch.Tensor, meta: torch.Tensor) -> tuple[int, int]:
    """Return the flag value and receive-count parity of rank 0's next round."""
    expected_flag = _word(workspace, _offset(meta, "FLAG_VAL")).item() + 1
    return expected_flag, ((expected_flag - 1) >> 1) & 1


def _finish(abort: torch.Tensor, failure: str, timeout_s: float = 10) -> None:
    """Wait for the work queued so far.

    At the deadline, release the work through the abort state so the process does not hang, and fail with
    ``failure``.
    """
    done = torch.cuda.Event()
    done.record()
    deadline = time.monotonic() + timeout_s
    while not done.query():
        if time.monotonic() > deadline:
            abort[0] = 1
            torch.cuda.synchronize()
            pytest.fail(failure)
        time.sleep(0.01)


def test_cancel_and_exclusion_on_replay() -> None:
    """One captured graph observes the abort state and the mask written before each replay."""
    workspace, meta, experts, payload = _setup(2)
    controls, abort = _rank_mask_controls(3)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        received, offset, _ = _dispatch(workspace, meta, experts, controls, payload)
        output = _combine(workspace, meta, received[0], offset, controls)

    # Peer 1 never signals. A pending abort request releases the dispatch's wait, and the combine of the
    # incomplete round takes only this rank's own rows.
    abort[0] = 1
    graph.replay()
    torch.cuda.synchronize()
    assert abort[:2].tolist() == [1, 1]
    assert _word(workspace, _offset(meta, "INCOMPLETE_ROUND")).item() == 1
    torch.testing.assert_close(output, payload, rtol=0, atol=0)

    # Until the workspace is reset, rank 0 publishes round flags only to itself, so peer 1 never combines
    # outputs computed from the incomplete round. Withdrawing the abort request alone does not change that.
    def peer_flags() -> list[int]:
        """Return rank 0's dispatch and combine flags in peer 1's row."""
        regions = ("DISPATCH_COMPLETION_FLAGS", "COMBINE_COMPLETION_FLAGS")
        return [_word(workspace, _offset(meta, region), row=1).item() for region in regions]

    published = peer_flags()
    assert published == [1, 0]
    # Peer 1 takes part in the next round with no rows for rank 0.
    expected_flag, parity = _next_parity(workspace, meta)
    _word(workspace, _offset(meta, "RECV_COUNTERS") + 4 * (parity * 2 + 1)).fill_(0)
    _word(workspace, _offset(meta, "DISPATCH_COMPLETION_FLAGS") + 4).fill_(expected_flag)
    _word(workspace, _offset(meta, "COMBINE_COMPLETION_FLAGS") + 4).fill_(expected_flag + 1)
    abort.zero_()
    graph.replay()
    _finish(abort, "the round after the incomplete one did not complete with peer 1's flags")
    assert peer_flags() == published

    # Rearmed with peer 1 excluded, the same graph completes without it.
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, 0, 2, barrier=False)
    abort.zero_()
    controls["active_rank_mask"].copy_(_mask_words(1))
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, payload, rtol=0, atol=0)
        assert torch.all(received[1][0] == 0)
    assert abort[:2].tolist() == [0, 0]


@pytest.mark.parametrize("ep_size, rank", [(2, 0), (72, 70)], ids=["first_word", "second_word"])
def test_no_peer_writes_after_incomplete_round(ep_size: int, rank: int) -> None:
    """Until the workspace reset, a dispatch writes nothing into the rows of a peer that may still read them.

    Rank 70 of 72 has its own bit in the second mask word.
    """
    workspace, meta, experts, payload = _setup(ep_size)
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, rank, ep_size, barrier=False)
    controls, abort = _rank_mask_controls((1 << ep_size) - 1)
    abort[0] = 1  # No peer signals, so the round stays incomplete.
    _dispatch(workspace, meta, experts, controls, payload, rank=rank)
    torch.cuda.synchronize()
    assert _word(workspace, _offset(meta, "INCOMPLETE_ROUND"), row=rank).item() == 1

    abort.zero_()
    peers = torch.cat([workspace[:rank], workspace[rank + 1 :]])
    # Route the tokens to peers of both mask words.
    experts.copy_(
        torch.tensor([0, 65, 71, 64, 3, 70, 69, 1], dtype=torch.int32).view(-1, 1) % ep_size
    )
    _dispatch(workspace, meta, experts, controls, payload, rank=rank)
    _finish(abort, "the dispatch after the incomplete round waited for a peer")
    assert torch.equal(torch.cat([workspace[:rank], workspace[rank + 1 :]]), peers)


def test_no_workspace_after_membership_change() -> None:
    """Workspace creation waits for every rank of the startup group, which may include a failed process."""
    health = EPGroupHealth(1)
    health.mark_failed(0)
    health.mark_active(0)
    mapping = Mapping(world_size=1, rank=0, tp_size=1, moe_ep_size=1)
    with pytest.raises(RuntimeError, match="only before the EP membership changes"):
        NVLinkOneSided(
            mapping,
            num_slots=2,
            top_k=1,
            max_num_tokens_per_rank=3,
            hidden_size=_HIDDEN,
            dtype=torch.bfloat16,
            ep_group_health=health,
        )


def _fence_timeout() -> None:
    """Check that a fence wait on a silent peer still fails at the timeout."""
    torch.ops.trtllm.moe_a2a_set_timeout(1)
    workspace, meta, experts, payload = _setup(2)
    controls, _ = _rank_mask_controls(3)
    _dispatch(workspace, meta, experts, controls, payload)
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        # Exit at once, flushing the device printf C stdio holds on a pipe.
        print("fence timeout trapped", flush=True)
        ctypes.CDLL(None).fflush(None)
        os._exit(0)
    raise AssertionError("dispatch completed without its peer")


def test_fence_timeout_still_traps() -> None:
    # The child traps on purpose, so it runs in a fresh interpreter without the launcher's environment.
    launcher_prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(launcher_prefixes)}
    result = subprocess.run(
        [sys.executable, __file__], env=env, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # The kernel's own report, not just any device error.
    assert "dispatch: rank 0 timed out waiting for rank 1 flag=" in result.stdout, result.stdout
    assert "fence timeout trapped" in result.stdout, result.stdout


if __name__ == "__main__":
    _fence_timeout()

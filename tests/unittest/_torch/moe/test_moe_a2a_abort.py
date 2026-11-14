# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run-time rank mask and abort of the fence MoE All-to-All ops.

Every logical peer lives in one allocation on one GPU, and the tests issue the ops of each rank in
turn, so they check masking, polling and the order of the ranks' steps only.
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
from tensorrt_llm.bindings import internal

_CAPACITY, _HIDDEN = 8, 128
# The abort state holds ``requested`` and ``status``, then the flag value awaited from each peer.
_ABORT_WORDS = int(internal.thop.MOE_A2A_ABORT_STATE_WORDS)


def _offset(meta: torch.Tensor, name: str) -> int:
    return int(meta[int(getattr(internal.thop, f"MOE_A2A_{name}_OFFSET_INDEX"))])


def _word(workspace: torch.Tensor, offset: int, row: int = 0) -> torch.Tensor:
    return workspace[row, offset : offset + 4].view(torch.int32)


def _mask_words(mask: int) -> torch.Tensor:
    """Return the rank mask as the kernels' four little-endian uint64 words."""
    return torch.tensor([(mask >> (64 * w)) & (2**64 - 1) for w in range(4)], dtype=torch.uint64)


def _rank_mask_controls(mask: int) -> tuple[dict, torch.Tensor]:
    """Return the rank-mask controls of a dispatch or combine and their pinned abort state."""
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
    """Dispatch [payload, experts] like the MoE layer as ``rank``."""
    ep_size = workspace.size(0)
    args = (experts, [payload, experts], workspace, meta, tokens, rank, ep_size, 1, ep_size, None)
    return torch.ops.trtllm.moe_a2a_dispatch(*args, **controls)


def _combine(workspace, meta, payload, offset, controls, tokens=_CAPACITY, rank=0):
    """Combine like the MoE layer as ``rank``."""
    args = (payload, tokens, workspace, meta, tokens, rank, workspace.size(0), 1, offset, False)
    return torch.ops.trtllm.moe_a2a_combine(*args, **controls)


def _next_parity(workspace: torch.Tensor, meta: torch.Tensor, rank: int = 0) -> tuple[int, int]:
    """Return the flag value and receive-count parity of ``rank``'s next round."""
    expected_flag = _word(workspace, _offset(meta, "FLAG_VAL"), row=rank).item() + 1
    return expected_flag, ((expected_flag - 1) >> 1) & 1


def _finish(abort: torch.Tensor, failure: str, timeout_s: float = 10) -> None:
    """Wait for the queued work. At the deadline, release it through the abort state and fail."""
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

    # Peer 1 never signals. A pending abort request releases the dispatch's wait, and the combine of
    # the incomplete round takes only this rank's own rows.
    abort[0] = 1
    graph.replay()
    _finish(abort, "the abort request did not release the dispatch's wait")
    assert abort[:2].tolist() == [1, 1]
    assert _word(workspace, _offset(meta, "INCOMPLETE_ROUND")).item() == 1
    torch.testing.assert_close(output, payload, rtol=0, atol=0)

    # Until the workspace is reset, rank 0 publishes round flags only to itself, so peer 1 never
    # combines outputs of the incomplete round. Withdrawing the abort request does not change that.
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
        _finish(abort, "the replay with peer 1 excluded waited for it")
        torch.testing.assert_close(output, payload, rtol=0, atol=0)
        assert torch.all(received[1][0] == 0)
    assert abort[:2].tolist() == [0, 0]


@pytest.mark.parametrize("ep_size, rank", [(2, 0), (72, 70)], ids=["first_word", "second_word"])
def test_no_peer_writes_after_incomplete_round(ep_size: int, rank: int) -> None:
    """Until the workspace reset, a dispatch writes nothing into rows a peer may still read."""
    workspace, meta, experts, payload = _setup(ep_size)
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, rank, ep_size, barrier=False)
    controls, abort = _rank_mask_controls((1 << ep_size) - 1)
    abort[0] = 1  # No peer signals, so the round stays incomplete.
    _dispatch(workspace, meta, experts, controls, payload, rank=rank)
    _finish(abort, "the abort request did not release the dispatch's wait")
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


@pytest.mark.parametrize(
    "ep_size, rank, peer", [(2, 0, 1), (72, 34, 2)], ids=["two_ranks", "rank_34_of_72"]
)
def test_aborted_dispatch_uses_no_rows_of_unconfirmed_peers(
    ep_size: int, rank: int, peer: int
) -> None:
    """An aborted round uses no rows of a peer whose completion flag it did not see.

    The peer writes its receive count but never its flag, and again after the abort. Rank 34 of 72
    waits on peer 2 and then on itself in the same lane, and keeps its own rows.
    """
    workspace, meta, experts, payload = _setup(ep_size)
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, rank, ep_size, barrier=False)
    experts.fill_(rank)
    controls, abort = _rank_mask_controls((1 << ep_size) - 1)
    _, parity = _next_parity(workspace, meta, rank)
    counts = _offset(meta, "RECV_COUNTERS") + 4 * parity * ep_size
    count = _word(workspace, counts + 4 * peer, row=rank)
    count.fill_(_CAPACITY)
    abort[0] = 1
    received, offset, _ = _dispatch(workspace, meta, experts, controls, payload, rank=rank)
    _finish(abort, "the abort request did not release the dispatch's wait")
    assert count.item() == 0
    assert _word(workspace, counts + 4 * rank, row=rank).item() == _CAPACITY

    count.fill_(_CAPACITY)
    expert_ids = received[1]
    expert_ids[peer].fill_(12345)  # an expert id that no rank owns
    torch.ops.trtllm.moe_a2a_sanitize_expert_ids(expert_ids, workspace, meta, rank, -1)
    assert torch.all(expert_ids[peer] == -1)
    assert torch.equal(expert_ids[rank], experts)

    # The rows of the peer in this rank's combine input region, which the peer reads.
    stride = _HIDDEN * payload.element_size()
    first = offset + peer * _CAPACITY * stride
    peer_rows = workspace[rank, first : first + _CAPACITY * stride]
    peer_rows.fill_(0x7F)
    output = _combine(workspace, meta, received[0], offset, controls, rank=rank)
    _finish(abort, "the combine of the incomplete round waited for the peer")
    assert torch.all(peer_rows == 0x7F)
    torch.testing.assert_close(output, payload, rtol=0, atol=0)


@pytest.mark.parametrize("peer_aborts", [False, True], ids=["healthy", "peer_aborts"])
def test_combine_cancels_on_overwritten_peer_rows(peer_aborts: bool) -> None:
    """A combine that read rows a peer overwrote after an aborted dispatch records a cancel.

    Rank 0's tokens go to rank 1, whose MoE output is its combine input region, as with zero-copy
    combine. Rank 1 completes its round before rank 0 combines it. With ``peer_aborts``, rank 1
    then aborts its next dispatch and overwrites that region like its next MoE.
    """
    workspace, meta, experts, payload = _setup(2)
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, 1, 2, barrier=False)
    controls, abort = _rank_mask_controls(3)
    peer_controls, peer_abort = _rank_mask_controls(3)
    experts.fill_(1)
    # Rank 1's dispatch flag for rank 0, and rank 0's combine flag for rank 1.
    dispatch_flag, _ = _next_parity(workspace, meta)
    _word(workspace, _offset(meta, "DISPATCH_COMPLETION_FLAGS") + 4).fill_(dispatch_flag)
    _word(workspace, _offset(meta, "COMBINE_COMPLETION_FLAGS"), row=1).fill_(dispatch_flag + 1)
    received, offset, _ = _dispatch(workspace, meta, experts, controls, payload)
    peer_received, _, _ = _dispatch(workspace, meta, experts, peer_controls, payload, rank=1)
    moe_output = torch.ops.trtllm.moe_a2a_get_combine_payload_tensor(
        workspace, meta, 1, 2, _CAPACITY, payload.dtype, _HIDDEN
    ).view(2, _CAPACITY, _HIDDEN)
    moe_output.copy_(peer_received[0])
    _combine(workspace, meta, moe_output, offset, peer_controls, rank=1)
    if peer_aborts:
        peer_abort[0] = 1
        _dispatch(workspace, meta, experts, peer_controls, payload, rank=1)
        moe_output.fill_(-1)
    _finish(peer_abort, "rank 1's round waited for rank 0")
    output = _combine(workspace, meta, received[0], offset, controls)
    _finish(abort, "rank 0's combine waited for rank 1")
    expected = torch.full_like(payload, -1) if peer_aborts else payload
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert abort[:2].tolist() == [0, int(peer_aborts)]


def test_incomplete_dispatch_marks_active_peers() -> None:
    """A dispatch whose round is incomplete marks the rounds of the active peers incomplete."""
    workspace, meta, experts, payload = _setup(3)
    controls, abort = _rank_mask_controls(0b011)  # rank 2 is inactive
    incomplete_round = _offset(meta, "INCOMPLETE_ROUND")
    # The incomplete-round value of a peer's aborted dispatch with flag value 5.
    _word(workspace, incomplete_round).fill_(5)
    _dispatch(workspace, meta, experts, controls, payload)
    _finish(abort, "the dispatch of an incomplete round waited for a peer")
    assert [_word(workspace, incomplete_round, row=r).item() for r in range(3)] == [5, 5, 0]


@pytest.mark.parametrize(
    "missing, inactive, waited",
    [([32, 64], [], [32, 64]), ([1, 33, 65], [33], [1, 65])],
    ids=["all_active", "inactive_peer"],
)
def test_long_wait_publishes_every_missing_peer_of_the_lane(missing, inactive, waited) -> None:
    """A lane that waits on a peer also publishes the lane's later active peers that are missing.

    Every other peer of rank 0 of 72 delivers its dispatch flag. A peer the mask clears is silent.
    """
    ep_size = 72
    workspace, meta, experts, payload = _setup(ep_size)
    mask = (1 << ep_size) - 1
    for peer in inactive:
        mask &= ~(1 << peer)
    controls, abort = _rank_mask_controls(mask)
    expected_flag, _ = _next_parity(workspace, meta)
    flags = _offset(meta, "DISPATCH_COMPLETION_FLAGS")
    for peer in range(1, ep_size):
        if peer not in missing:
            _word(workspace, flags + 4 * peer).fill_(expected_flag)
    _dispatch(workspace, meta, experts, controls, payload)
    deadline = time.monotonic() + 10
    while abort[2 + waited[-1]].item() == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    # Nothing withdraws a published flag before the abort, so one scan after the last publish sees
    # them all.
    waits = [peer for peer in range(ep_size) if abort[2 + peer].item() != 0]
    abort[0] = 1
    _finish(abort, "the abort request did not release the dispatch's wait")
    assert waits == waited
    assert abort[2:].tolist() == [0] * (_ABORT_WORDS - 2)


def test_rank_mask_mode_takes_a_cpu_mask() -> None:
    """The ops also take the rank mask as a CPU tensor, and refuse one without the local rank."""
    workspace, meta, experts, payload = _setup(2)
    controls, abort = _rank_mask_controls(1)
    controls["active_rank_mask"] = _mask_words(1)
    received, offset, _ = _dispatch(workspace, meta, experts, controls, payload)
    output = _combine(workspace, meta, received[0], offset, controls)
    _finish(abort, "the round without peer 1 waited for it")
    torch.testing.assert_close(output, payload, rtol=0, atol=0)

    controls["active_rank_mask"] = _mask_words(2)
    with pytest.raises(RuntimeError, match=r"must mark the local ep_rank \(0\) as active"):
        _dispatch(workspace, meta, experts, controls, payload)


def test_cpu_mask_written_only_on_change() -> None:
    """A CPU rank mask reaches the workspace's mask words only when it differs from the last one."""
    workspace, meta, experts, payload = _setup(2)
    controls, abort = _rank_mask_controls(3)
    # With the abort requested up front, every round returns at once, so a mask may name peer 1.
    abort[0] = 1
    offset = _offset(meta, "ACTIVE_RANK_MASK")
    device_words = workspace[0, offset : offset + 32].view(torch.int64)

    def dispatch(mask: int) -> list[int]:
        controls["active_rank_mask"] = _mask_words(mask)
        _dispatch(workspace, meta, experts, controls, payload)
        _finish(abort, "a dispatch with the abort requested kept waiting")
        return device_words.tolist()

    assert dispatch(1) == [1, 0, 0, 0]
    assert dispatch(3) == [3, 0, 0, 0]
    # An unchanged mask is not written again, so words changed behind the op's back stay.
    device_words.copy_(torch.tensor([1, 0, 0, 0]))
    assert dispatch(3) == [1, 0, 0, 0]
    # Initializing the row zeroes its mask words, so the next mask is written again.
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, 0, 2, barrier=False)
    assert dispatch(3) == [3, 0, 0, 0]


def test_cpu_mask_refused_during_capture() -> None:
    """A captured graph would not see later changes of a CPU rank mask, so capture refuses one."""
    workspace, meta, experts, payload = _setup(2)
    controls, _ = _rank_mask_controls(1)
    controls["active_rank_mask"] = _mask_words(1)
    with pytest.raises(RuntimeError, match="during CUDA graph capture"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            _dispatch(workspace, meta, experts, controls, payload)


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
    # The child traps on purpose, in a fresh interpreter without the launcher's environment.
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

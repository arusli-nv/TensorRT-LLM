# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run-time rank mask and abort of the fence MoE All-to-All ops.

Run in a singleton MPI process with CUDA available. Multi-rank cases place every logical
peer in one allocation and only rank 0 issues ops, so they check the native layout,
masking and polling, not peer transport.
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


def _offset(meta: torch.Tensor, name: str) -> int:
    return int(meta[int(getattr(internal.thop, f"MOE_A2A_{name}_OFFSET_INDEX"))])


def _word(workspace: torch.Tensor, offset: int) -> torch.Tensor:
    return workspace[0, offset : offset + 4].view(torch.int32)


def _rank_mask_controls(mask: int) -> tuple[dict, torch.Tensor]:
    """Return the rank-mask controls of a dispatch or combine (the device mask, read at execution time, and the
    abort record in pinned host memory) and the record."""
    abort = torch.zeros(2, dtype=torch.int32).pin_memory()
    words = torch.tensor([mask, 0, 0, 0], dtype=torch.uint64, device="cuda")
    return {"enable_rank_mask": True, "active_rank_mask": words, "abort_state": abort}, abort


def _set_mask(controls: dict, mask: int) -> None:
    controls["active_rank_mask"].copy_(torch.tensor([mask, 0, 0, 0], dtype=torch.uint64))


def _setup(
    ep_size: int,
    tokens: int = _CAPACITY,
    eplb_stats_num_experts=None,
    extra_bytes: int = 32,
    capacity: int = _CAPACITY,
    top_k: int = 1,
) -> tuple[torch.Tensor, ...]:
    meta = NVLinkOneSided._make_workspace_layout(
        ep_size,
        top_k,
        capacity,
        _HIDDEN,
        torch.bfloat16,
        eplb_stats_num_experts,
        extra_bytes,
        False,
        False,
        rank_mask=True,
    )
    size = int(meta[int(internal.thop.MOE_A2A_WORKSPACE_SIZE_INDEX)])
    workspace = torch.zeros((ep_size, size), dtype=torch.uint8, device="cuda")
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, 0, ep_size, barrier=False)
    experts = torch.zeros((tokens, top_k), dtype=torch.int32, device="cuda")
    payload = torch.arange(tokens * _HIDDEN, dtype=torch.bfloat16, device="cuda")
    return workspace, meta, experts, payload.reshape(tokens, _HIDDEN)


def _controls(controls: dict | None) -> dict:
    return {} if controls is None else controls


def _dispatch(workspace, meta, experts, payloads, controls, tokens=_CAPACITY, stats=None):
    """Dispatch like the MoE layer: the received tensors, [ep_size, tokens, ...] (source s's rows at [s]), the
    combine offset and the gathered EPLB statistics."""
    ep_size = workspace.size(0)
    args = (experts, payloads, workspace, meta, tokens, 0, ep_size, 1, ep_size, stats)
    if controls is None:
        return torch.ops.trtllm.moe_a2a_dispatch(
            *args, expert_id_payload_index=1, invalid_token_expert_id=-1
        )
    return torch.ops.trtllm.moe_a2a_dispatch(*args, **controls)


def _combine(workspace, meta, payload, offset, controls, tokens=_CAPACITY, in_workspace=False):
    """Combine like the MoE layer, with the rank-mask controls if any."""
    ep_size = workspace.size(0)
    args = (payload, tokens, workspace, meta, tokens, 0, ep_size, 1, offset, in_workspace)
    return torch.ops.trtllm.moe_a2a_combine(*args, **_controls(controls))


def _round(workspace, meta, experts, payload, controls):
    received, offset, _ = _dispatch(workspace, meta, experts, [payload, experts], controls)
    return received, _combine(workspace, meta, received[0], offset, controls)


def _recv_count(workspace: torch.Tensor, meta: torch.Tensor, parity: int, source: int) -> int:
    """The receive count ``source`` published in rank 0's row for a round of ``parity``."""
    ep_size = workspace.size(0)
    return _word(workspace, _offset(meta, "RECV_COUNTERS") + 4 * (parity * ep_size + source)).item()


def _next_parity(workspace: torch.Tensor, meta: torch.Tensor) -> tuple[int, int]:
    """The flag value and receive-count parity of rank 0's next round."""
    expected_flag = _word(workspace, _offset(meta, "FLAG_VAL")).item() + 1
    return expected_flag, ((expected_flag - 1) >> 1) & 1


def test_cancel_and_exclusion_on_replay() -> None:
    """Does one captured graph observe the abort record and the mask written before each replay?"""
    workspace, meta, experts, payload = _setup(2)
    controls, abort = _rank_mask_controls(3)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        received, output = _round(workspace, meta, experts, payload, controls)

    # Peer 1 never signals: a pending request releases both waits instead of a timeout, and the incomplete
    # round combines nothing.
    abort[0] = 1
    graph.replay()
    torch.cuda.synchronize()
    assert abort.tolist() == [1, 1]
    assert _word(workspace, _offset(meta, "FLAG_VAL") + 4).item() == 1
    assert torch.count_nonzero(output) == 0

    # Rearmed with peer 1 excluded, the same graph completes without it.
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, 0, 2, barrier=False)
    abort.zero_()
    _set_mask(controls, 1)
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, payload, rtol=0, atol=0)
        assert torch.all(received[1][0] == 0)
    assert abort.tolist() == [0, 0]


@pytest.mark.parametrize("row_bytes", [16, 24, 6, 3])
def test_graphs_of_different_batch_sizes_leave_an_excluded_source_s_state_untouched(
    row_bytes: int,
) -> None:
    """Do graphs of different batch sizes write only this rank's own rows of the dispatch region while a peer
    is excluded, and nothing of the peer's flags and statistics?"""
    ep_size = 2
    workspace, meta, experts, payload = _setup(ep_size, eplb_stats_num_experts=ep_size)
    byte_payload = torch.arange(_CAPACITY * row_bytes, dtype=torch.uint8, device="cuda")
    byte_payload = byte_payload.reshape(_CAPACITY, row_bytes)
    controls, abort = _rank_mask_controls(1)
    widths = (_HIDDEN * 2, 4, row_bytes)
    dispatch_start = _offset(meta, "DISPATCH_PAYLOAD")
    dispatch_end = dispatch_start + int(
        meta[int(internal.thop.MOE_A2A_DISPATCH_PAYLOAD_SIZE_INDEX)]
    )
    combine_input = _offset(meta, "COMBINE_INPUT")

    def own_rows(bucket: int) -> list[tuple[int, int]]:
        """Source 0's rows [0, bucket) of each payload, whose [ep_size, bucket] rows moeA2ADispatchOp
        places in turn from the dispatch region's start, cacheline aligned."""
        rows, start = [], dispatch_start
        for width in widths:
            rows.append((start, start + bucket * width))
            start += (ep_size * bucket * width + 127) // 128 * 128
        return rows

    # Poison the whole dispatch region and the peer's flags and statistics row: a graph of either batch
    # size may write only its own rows of the region.
    workspace[0, dispatch_start:dispatch_end].fill_(0xA5)
    tombstones = []
    for name in ("DISPATCH_COMPLETION_FLAGS", "COMBINE_COMPLETION_FLAGS"):
        tombstones.append(_word(workspace, _offset(meta, name) + 4))
    stats = _offset(meta, "EPLB_GATHERED_STATS") + ep_size * 4
    tombstones.append(workspace[0, stats : stats + ep_size * 4].view(torch.int32))
    for value in tombstones:
        value.fill_(2**31 - 1)
    tombstone_values = [torch.full_like(value, 2**31 - 1) for value in tombstones]
    untouched = torch.ones(dispatch_end - dispatch_start, dtype=torch.bool, device="cuda")
    for bucket in (4, 8):
        for lo, hi in own_rows(bucket):
            untouched[lo - dispatch_start : hi - dispatch_start] = False

    def transfer(bucket: int) -> tuple[list[torch.Tensor], torch.Tensor]:
        ids = experts[:bucket]
        payloads = [payload[:bucket], ids, byte_payload[:bucket]]
        received, offset, _ = _dispatch(workspace, meta, ids, payloads, controls, bucket)
        assert offset == combine_input
        for tensor, (lo, _) in zip(received, own_rows(bucket)):
            assert tensor.shape[:2] == (ep_size, bucket)
            assert tensor.data_ptr() == workspace[0].data_ptr() + lo
        expert_output = torch.ops.trtllm.moe_a2a_get_combine_payload_tensor(
            workspace, meta, 0, ep_size, bucket, payload.dtype, _HIDDEN
        ).view(ep_size, bucket, _HIDDEN)
        expert_output.copy_(received[0])
        output = _combine(workspace, meta, expert_output, offset, controls, bucket, True)
        return received, output

    captures = {}
    for bucket in (4, 8):
        transfer(bucket)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = transfer(bucket)
        captures[bucket] = graph, *outputs

    for bucket in (8, 4, 8, 4):
        graph, received, output = captures[bucket]
        graph.replay()
        torch.cuda.synchronize()
        expected = payload[:bucket]
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        # Source 0's rows hold its tokens in the order the dispatch assigned them.
        order = received[0][0, :, 0].argsort()
        torch.testing.assert_close(received[0][0][order], expected, rtol=0, atol=0)
        torch.testing.assert_close(received[2][0][order], byte_payload[:bucket], rtol=0, atol=0)
        assert torch.all(received[1][0] == 0)
        assert bool((workspace[0, dispatch_start:dispatch_end][untouched] == 0xA5).all())
        assert [value.tolist() for value in tombstones] == [
            value.tolist() for value in tombstone_values
        ]
        assert abort.tolist() == [0, 0]


def test_a_masked_source_reads_as_empty_even_if_it_arrived() -> None:
    """A masked peer's completed round is still never read: its receive count reads 0."""
    workspace, meta, experts, payload = _setup(2)
    controls, _ = _rank_mask_controls(1)
    expected_flag, parity = _next_parity(workspace, meta)
    _word(workspace, _offset(meta, "DISPATCH_COMPLETION_FLAGS") + 4).fill_(expected_flag)
    _word(workspace, _offset(meta, "RECV_COUNTERS") + 4 * (parity * 2 + 1)).fill_(3)
    _dispatch(workspace, meta, experts, [payload, experts], controls)
    torch.cuda.synchronize()
    assert [_recv_count(workspace, meta, parity, source) for source in (0, 1)] == [_CAPACITY, 0]


def test_dispatch_waits_for_a_source_that_arrives_late() -> None:
    """Does the dispatch take a source whose count and flag arrive while it waits, without an abort?"""
    workspace, meta, experts, payload = _setup(2)
    controls, abort = _rank_mask_controls(3)
    expected_flag = _word(workspace, _offset(meta, "FLAG_VAL")).item() + 1
    parity = ((expected_flag - 1) >> 1) & 1
    count = _word(workspace, _offset(meta, "RECV_COUNTERS") + 4 * (parity * 2 + 1))
    flag = _word(workspace, _offset(meta, "DISPATCH_COMPLETION_FLAGS") + 4)
    torch.cuda.synchronize()
    _dispatch(workspace, meta, experts, [payload, experts], controls)
    done = torch.cuda.Event()
    done.record()
    time.sleep(0.2)
    assert not done.query(), "the dispatch completed before peer 1 published its count"
    # Stand in for peer 1 on another stream while the dispatch waits: its count, then its flag.
    # The stream must run beside the waiting kernel (not with CUDA_DEVICE_MAX_CONNECTIONS=1).
    late = torch.cuda.Stream()
    with torch.cuda.stream(late):
        count.fill_(3)
        flag.fill_(expected_flag)
    deadline = time.monotonic() + 10
    while not done.query():
        if time.monotonic() > deadline:
            # Release the dispatch so that the failure does not hang the process.
            abort[0] = 1
            torch.cuda.synchronize()
            pytest.fail("the dispatch did not take peer 1's late count and flag")
        time.sleep(0.01)
    assert abort.tolist() == [0, 0]
    assert _word(workspace, _offset(meta, "FLAG_VAL") + 4).item() == 0
    assert _recv_count(workspace, meta, parity, 1) == 3


def test_an_aborted_round_without_a_completion_flag_is_incomplete() -> None:
    """Does an aborted round whose source's count arrived but whose completion flag did not return as
    cancelled and mark the round incomplete, so this rank publishes no further round flag?"""
    workspace, meta, experts, payload = _setup(2)
    controls, abort = _rank_mask_controls(3)
    _, parity = _next_parity(workspace, meta)
    _word(workspace, _offset(meta, "RECV_COUNTERS") + 4 * (parity * 2 + 1)).fill_(3)
    abort[0] = 1
    _dispatch(workspace, meta, experts, [payload, experts], controls)
    torch.cuda.synchronize()
    assert abort.tolist() == [1, 1]
    assert _word(workspace, _offset(meta, "FLAG_VAL") + 4).item() == 1


def test_replay_reads_the_slot_table_and_mask_written_before_it() -> None:
    """One captured dispatch routes by the expert -> slot table and the rank mask copied into their buffers before
    each replay: it sends each token to its expert's current slot, with that slot id, and nothing to a masked peer."""
    ep_size, num_slots = 2, 4  # Slots 0 and 1 live on rank 0, slots 2 and 3 on rank 1.
    workspace, meta, _, payload = _setup(ep_size)
    tokens = torch.arange(_CAPACITY, dtype=torch.int32, device="cuda").view(_CAPACITY, 1)
    experts = tokens % 2
    table = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    controls, abort = _rank_mask_controls(1)

    def dispatch() -> list[torch.Tensor]:
        received, *_ = torch.ops.trtllm.moe_a2a_dispatch(
            experts,
            [payload, experts, tokens],
            workspace,
            meta,
            _CAPACITY,
            0,
            ep_size,
            1,
            num_slots,
            None,
            expert_id_payload_index=1,
            **controls,
            expert_to_slot=table,
        )
        return received

    received = dispatch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dispatch()

    def peer_word(offset: int) -> torch.Tensor:
        return workspace[1, offset : offset + 4].view(torch.int32)

    def peer_rows(tensor: torch.Tensor) -> torch.Tensor:
        """Peer 1's copy of one of rank 0's int32 receive regions."""
        offset = tensor.data_ptr() - workspace[0].data_ptr()
        region = workspace[1, offset : offset + tensor.numel() * 4]
        return region.view(torch.int32).view(tensor.shape)

    def sent(slot_ids: torch.Tensor, token_ids: torch.Tensor) -> tuple[list, list]:
        """The slot ids and tokens of source 0's first _CAPACITY // 2 rows, in token order."""
        rows = _CAPACITY // 2
        order = token_ids[0, :rows, 0].argsort()
        return slot_ids[0, :rows, 0][order].tolist(), token_ids[0, :rows, 0][order].tolist()

    # Expert 0 moves from slot 1 (rank 0) to slot 3 (rank 1) and expert 1 from slot 2 to slot 0; peer 1 is masked
    # while it holds a routed slot. Even tokens choose expert 0, odd ones expert 1.
    for slots, mask in (((1, 2), 1), ((3, 0), 3), ((1, 2), 1)):
        table.copy_(torch.tensor(slots, dtype=torch.int32))
        _set_mask(controls, mask)
        expected_flag, parity = _next_parity(workspace, meta)
        if mask == 3:
            # Peer 1 sends rank 0 nothing this round.
            _word(workspace, _offset(meta, "RECV_COUNTERS") + 4 * (parity * 2 + 1)).fill_(0)
            _word(workspace, _offset(meta, "DISPATCH_COMPLETION_FLAGS") + 4).fill_(expected_flag)
        peer_before = workspace[1].clone()
        graph.replay()
        torch.cuda.synchronize()

        local = next(e for e in (0, 1) if slots[e] < 2)
        assert [_recv_count(workspace, meta, parity, source) for source in (0, 1)] == [
            _CAPACITY // 2,
            0,
        ]
        assert sent(received[1], received[2]) == (
            [slots[local]] * (_CAPACITY // 2),
            list(range(local, _CAPACITY, 2)),
        )
        if mask == 1:
            assert torch.equal(workspace[1], peer_before)
        else:
            remote = 1 - local
            assert sent(peer_rows(received[1]), peer_rows(received[2])) == (
                [slots[remote]] * (_CAPACITY // 2),
                list(range(remote, _CAPACITY, 2)),
            )
            peer_count = peer_word(_offset(meta, "RECV_COUNTERS") + 4 * parity * 2)
            assert peer_count.item() == _CAPACITY // 2
            assert peer_word(_offset(meta, "DISPATCH_COMPLETION_FLAGS")).item() == expected_flag
    assert abort.tolist() == [0, 0]


@pytest.mark.parametrize("local", [0, 5])
def test_each_source_s_receive_count_is_what_it_sent_and_0_when_masked(local: int) -> None:
    """Is every source's receive count right: self, a peer with rows, a peer with none and an absent peer that
    arrived?"""
    ep_size = 4
    workspace, meta, experts, payload = _setup(ep_size, tokens=local)
    controls, abort = _rank_mask_controls(0b0111)
    # Stand in for peers 1-3: each publishes its count and completion flag for the next round. Peer 3 is
    # absent (masked out), so its arrival must not count.
    expected_flag, parity = _next_parity(workspace, meta)
    for peer, count in {1: 3, 2: 0, 3: 2}.items():
        _word(workspace, _offset(meta, "RECV_COUNTERS") + 4 * (parity * ep_size + peer)).fill_(
            count
        )
        _word(workspace, _offset(meta, "DISPATCH_COMPLETION_FLAGS") + 4 * peer).fill_(expected_flag)
    _dispatch(workspace, meta, experts, [payload, experts], controls)
    torch.cuda.synchronize()
    counts = [_recv_count(workspace, meta, parity, source) for source in range(ep_size)]
    assert counts == [local, 3, 0, 0]
    assert abort.tolist() == [0, 0]


def test_dispatch_rejects_runtime_over_capacity() -> None:
    workspace, meta, experts, payload = _setup(1, _CAPACITY + 1)
    controls, _ = _rank_mask_controls(1)
    with pytest.raises(RuntimeError, match="allocation-time token capacity"):
        _dispatch(workspace, meta, experts, [experts], controls, _CAPACITY + 1)


def test_the_abort_record_needs_the_rank_mask() -> None:
    workspace, meta, experts, payload = _setup(1)
    kwargs = {"enable_rank_mask": False, "abort_state": _rank_mask_controls(1)[1]}
    with pytest.raises(RuntimeError, match="the abort record needs the rank-mask fence path"):
        torch.ops.trtllm.moe_a2a_dispatch(
            experts, [payload, experts], workspace, meta, _CAPACITY, 0, 1, 1, 1, **kwargs
        )


@pytest.mark.parametrize("masked", [False, True])
def test_fake_dispatch_matches_the_real_shapes(masked: bool) -> None:
    """Dispatch receives the runtime token count per source, with or without the rank-mask controls, as the
    fake op computes."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    tokens = _CAPACITY // 2
    workspace, meta, experts, payload = _setup(1, tokens=tokens)
    controls = _rank_mask_controls(1)[0] if masked else None
    received, offset, _ = _dispatch(workspace, meta, experts, [payload, experts], controls, tokens)
    output = _combine(workspace, meta, received[0], offset, controls, tokens)
    torch.cuda.synchronize()
    assert received[1].shape == (1, tokens, 1)
    assert output.shape == (tokens, _HIDDEN)

    real = (workspace, meta, experts, payload, controls)
    with FakeTensorMode(shape_env=ShapeEnv()):
        workspace, meta, experts, payload, controls = torch.utils._pytree.tree_map_only(
            torch.Tensor, lambda t: torch.empty(t.shape, dtype=t.dtype, device=t.device), real
        )
        received, offset, _ = _dispatch(
            workspace, meta, experts, [payload, experts], controls, tokens
        )
        output = _combine(workspace, meta, received[0], offset, controls, tokens)
    assert received[1].shape == (1, tokens, 1)
    assert output.shape == (tokens, _HIDDEN) and output.dtype == torch.bfloat16


def _exit_trapped(message: str) -> None:
    """Print ``message`` and exit at once, flushing the device printf C stdio holds on a pipe."""
    print(message, flush=True)
    ctypes.CDLL(None).fflush(None)
    os._exit(0)


def _fence_timeout() -> None:
    """Does a fence wait on a silent peer still fail stop at the timeout?"""
    torch.ops.trtllm.moe_a2a_set_timeout(1)
    workspace, meta, experts, payload = _setup(2)
    controls, _ = _rank_mask_controls(3)
    _dispatch(workspace, meta, experts, [payload, experts], controls)
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        _exit_trapped("fence timeout trapped")
    raise AssertionError("dispatch completed without its peer")


def _run_trapping_child(*args: str) -> str:
    """Run this file's child with ``args``; it traps on purpose, so it runs in a fresh interpreter, kept off the
    enclosing launcher's process-management interface. Returns its stdout."""
    launcher_prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(launcher_prefixes)}
    try:
        result = subprocess.run(
            [sys.executable, __file__, *args], env=env, capture_output=True, text=True, timeout=300
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"Pending trapping subprocess deadline: {error.stdout!r} {error.stderr!r}")
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_fence_timeout_still_traps() -> None:
    output = _run_trapping_child()
    # The kernel's own report, not just any device error.
    assert "dispatch: rank 0 timed out waiting for rank 1 flag=" in output, output
    assert "fence timeout trapped" in output, output


if __name__ == "__main__":
    _fence_timeout()

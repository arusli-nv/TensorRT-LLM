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
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm.bindings import internal
from tensorrt_llm.mapping import Mapping

_CAPACITY, _HIDDEN = 8, 128
# The abort state: ``requested`` and ``status``, then per peer the flag value a long wait on it expects (wait
# evidence).
_ABORT_WORDS = int(internal.thop.MOE_A2A_ABORT_STATE_WORDS)
_EVIDENCE = int(internal.thop.MOE_A2A_ABORT_WAIT_EVIDENCE_INDEX)


def _offset(meta: torch.Tensor, name: str) -> int:
    return int(meta[int(getattr(internal.thop, f"MOE_A2A_{name}_OFFSET_INDEX"))])


def _word(workspace: torch.Tensor, offset: int, row: int = 0) -> torch.Tensor:
    return workspace[row, offset : offset + 4].view(torch.int32)


def _mask_words(mask: int) -> torch.Tensor:
    """The rank mask as the kernels' four little-endian uint64 words (rank r is bit r % 64 of word r // 64)."""
    return torch.tensor([(mask >> (64 * w)) & (2**64 - 1) for w in range(4)], dtype=torch.uint64)


def _rank_mask_controls(mask: int) -> tuple[dict, torch.Tensor]:
    """Return the rank-mask controls of a dispatch or combine (the device mask, read at execution time, and the
    abort state in pinned host memory) and the record."""
    abort = torch.zeros(_ABORT_WORDS, dtype=torch.int32).pin_memory()
    words = _mask_words(mask).cuda()
    return {"enable_rank_mask": True, "active_rank_mask": words, "abort_state": abort}, abort


def _set_mask(controls: dict, mask: int) -> None:
    controls["active_rank_mask"].copy_(_mask_words(mask))


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
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, 0, ep_size)
    experts = torch.zeros((tokens, top_k), dtype=torch.int32, device="cuda")
    payload = torch.arange(tokens * _HIDDEN, dtype=torch.bfloat16, device="cuda")
    return workspace, meta, experts, payload.reshape(tokens, _HIDDEN)


def _controls(controls: dict | None) -> dict:
    return {} if controls is None else controls


def _dispatch(workspace, meta, experts, payloads, controls, tokens=_CAPACITY, stats=None, rank=0):
    """Dispatch like the MoE layer as ``rank``: the received tensors, [ep_size, tokens, ...] (source s's rows at
    [s]), the combine offset and the gathered EPLB statistics."""
    ep_size = workspace.size(0)
    args = (
        experts,
        payloads,
        workspace,
        meta,
        tokens,
        rank,
        ep_size,
        experts.size(1),
        ep_size,
        stats,
    )
    if controls is None:
        return torch.ops.trtllm.moe_a2a_dispatch(
            *args, expert_id_payload_index=1, invalid_token_expert_id=-1
        )
    return torch.ops.trtllm.moe_a2a_dispatch(*args, **controls)


def _combine(
    workspace, meta, payload, offset, controls, tokens=_CAPACITY, in_workspace=False, top_k=1
):
    """Combine like the MoE layer as rank 0, with the rank-mask controls if any."""
    ep_size = workspace.size(0)
    args = (payload, tokens, workspace, meta, tokens, 0, ep_size, top_k, offset, in_workspace)
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


def _finish(abort: torch.Tensor, failure: str, timeout_s: float = 10) -> None:
    """Wait for the work queued so far. If it still waits at the deadline, release it through the abort state, so
    that the failure does not hang the process, and fail with ``failure``."""
    done = torch.cuda.Event()
    done.record()
    deadline = time.monotonic() + timeout_s
    while not done.query():
        if time.monotonic() > deadline:
            abort[0] = 1
            torch.cuda.synchronize()
            pytest.fail(failure)
        time.sleep(0.01)


def _evidence_while_waiting(
    abort: torch.Tensor, peer: int, expected: int, timeout_s: float = 10
) -> bool:
    """Whether the wait evidence on ``peer`` reaches ``expected`` while the queued work waits."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if abort[_EVIDENCE + peer].item() == expected:
            return True
        time.sleep(0.01)
    return False


def test_cancel_and_exclusion_on_replay() -> None:
    """Does one captured graph observe the abort state and the mask written before each replay?"""
    workspace, meta, experts, payload = _setup(2)
    controls, abort = _rank_mask_controls(3)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        received, output = _round(workspace, meta, experts, payload, controls)

    # Peer 1 never signals: a pending request releases the dispatch's wait instead of a timeout, and the
    # combine of the incomplete round takes this rank's own rows only (every token goes to rank 0's expert), never
    # a peer's.
    abort[0] = 1
    graph.replay()
    torch.cuda.synchronize()
    assert abort[:2].tolist() == [1, 1]
    assert _word(workspace, _offset(meta, "INCOMPLETE_ROUND")).item() == 1
    torch.testing.assert_close(output, payload, rtol=0, atol=0)

    # The incomplete round is not consumed: until the workspace is reset, rank 0 publishes round flags to itself
    # only, so its rounds complete but peer 1 never combines outputs computed from it. Peer 1 stands in for the
    # next round; withdrawing the request alone does not make rank 0 publish its flags of that round.
    def peer_flags() -> list[int]:
        """Rank 0's dispatch and combine flags in peer 1's row."""
        regions = ("DISPATCH_COMPLETION_FLAGS", "COMBINE_COMPLETION_FLAGS")
        return [
            workspace[1, _offset(meta, r) : _offset(meta, r) + 4].view(torch.int32).item()
            for r in regions
        ]

    published = peer_flags()
    assert published == [1, 0]
    expected_flag, parity = _next_parity(workspace, meta)
    _word(workspace, _offset(meta, "RECV_COUNTERS") + 4 * (parity * 2 + 1)).fill_(0)
    _word(workspace, _offset(meta, "DISPATCH_COMPLETION_FLAGS") + 4).fill_(expected_flag)
    _word(workspace, _offset(meta, "COMBINE_COMPLETION_FLAGS") + 4).fill_(expected_flag + 1)
    abort.zero_()
    graph.replay()
    _finish(abort, "the round after the incomplete one did not complete with peer 1's flags")
    assert peer_flags() == published

    # Rearmed with peer 1 excluded, the same graph completes without it.
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, 0, 2)
    abort.zero_()
    _set_mask(controls, 1)
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, payload, rtol=0, atol=0)
        assert torch.all(received[1][0] == 0)
    assert abort[:2].tolist() == [0, 0]


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
        assert abort[:2].tolist() == [0, 0]


def test_dispatch_waits_for_a_source_that_arrives_late() -> None:
    """Does the dispatch take a source whose count and flag arrive while it waits, without an abort, and publish
    wait evidence on that source only while it waits?"""
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
    evidence_seen = _evidence_while_waiting(abort, 1, expected_flag)
    evidence_on_self = abort[_EVIDENCE].item()
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
    assert evidence_seen and evidence_on_self == 0
    assert abort[:2].tolist() == [0, 0]
    assert torch.count_nonzero(abort[_EVIDENCE:]) == 0
    assert _word(workspace, _offset(meta, "INCOMPLETE_ROUND")).item() == 0
    assert _recv_count(workspace, meta, parity, 1) == 3


def test_abort_releases_a_waiting_combine_and_withdraws_its_wait_evidence() -> None:
    """Does an abort requested while a combine waits on a silent peer release it, with zero output and a
    cancelled status, after the combine published wait evidence on that peer?"""
    workspace, meta, experts, payload = _setup(2)
    controls, abort = _rank_mask_controls(3)
    # Peer 1 takes part in the dispatch (no rows for rank 0), then never signals the combine.
    expected_flag, parity = _next_parity(workspace, meta)
    _word(workspace, _offset(meta, "RECV_COUNTERS") + 4 * (parity * 2 + 1)).fill_(0)
    _word(workspace, _offset(meta, "DISPATCH_COMPLETION_FLAGS") + 4).fill_(expected_flag)
    torch.cuda.synchronize()
    _, output = _round(workspace, meta, experts, payload, controls)
    evidence_seen = _evidence_while_waiting(abort, 1, expected_flag + 1)
    status_while_waiting = abort[:2].tolist()

    abort[0] = 1
    _finish(abort, "the combine kept waiting after the abort request")
    assert evidence_seen and status_while_waiting == [0, 0]
    assert abort[:2].tolist() == [1, 1]
    assert torch.count_nonzero(abort[_EVIDENCE:]) == 0
    assert torch.count_nonzero(output) == 0
    # An incomplete combine marks the round incomplete as a dispatch does, with its own flag value (bit 0 set).
    assert _word(workspace, _offset(meta, "INCOMPLETE_ROUND")).item() == (expected_flag + 1) | 1


def test_after_an_incomplete_round_nothing_is_written_into_a_peer_s_rows() -> None:
    """An active peer may still read its rows of the round this rank left: until the workspace reset, a dispatch of
    tokens routed to that peer writes nothing into its row and completes without it."""
    workspace, meta, experts, payload = _setup(2)
    controls, abort = _rank_mask_controls(3)
    abort[0] = 1  # peer 1 never signals: the round is left incomplete
    _round(workspace, meta, experts, payload, controls)
    torch.cuda.synchronize()
    assert _word(workspace, _offset(meta, "INCOMPLETE_ROUND")).item() == 1

    abort.zero_()
    peer_row = workspace[1].clone()
    experts.fill_(1)  # every token to peer 1's expert
    _dispatch(workspace, meta, experts, [payload, experts], controls)
    _finish(abort, "the dispatch after the incomplete round waited for peer 1")
    assert torch.equal(workspace[1], peer_row)


def test_the_combine_of_an_incomplete_round_takes_this_rank_s_rows_only() -> None:
    """The routes saved by an incomplete dispatch still name peer 1, whose rows of the round may never be completed
    (NaN here): the round's combine takes this rank's rows only."""
    workspace, meta, experts, payload = _setup(2, top_k=2)
    experts.copy_(torch.tensor([[0, 1]] * _CAPACITY, dtype=torch.int32))
    controls, abort = _rank_mask_controls(3)
    torch.ops.trtllm.moe_a2a_get_combine_payload_tensor(
        workspace, meta, 1, 2, _CAPACITY, payload.dtype, _HIDDEN
    ).fill_(float("nan"))
    abort[0] = 1  # peer 1 never signals: the dispatch is left incomplete
    received, offset, _ = _dispatch(workspace, meta, experts, [payload, experts], controls)
    torch.cuda.synchronize()
    assert _word(workspace, _offset(meta, "INCOMPLETE_ROUND")).item() == 1

    abort.zero_()
    output = _combine(workspace, meta, received[0], offset, controls, top_k=2)
    _finish(abort, "the combine of the incomplete round waited for peer 1")
    torch.testing.assert_close(output, payload, rtol=0, atol=0)
    assert abort[:2].tolist() == [0, 0]


def test_a_rank_in_mask_word_1_writes_into_no_peer_s_rows_after_an_incomplete_round() -> None:
    """The test above for rank 70 of 72 (NVL72), whose own bit is in the second mask word."""
    ep_size, rank = 72, 70
    workspace, meta, experts, payload = _setup(ep_size)
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, rank, ep_size)
    controls, abort = _rank_mask_controls(2**ep_size - 1)
    abort[0] = 1  # no peer signals: the round is left incomplete
    _dispatch(workspace, meta, experts, [payload, experts], controls, rank=rank)
    torch.cuda.synchronize()
    assert _word(workspace, _offset(meta, "INCOMPLETE_ROUND"), row=rank).item() == 1

    abort.zero_()
    peers = torch.cat([workspace[:rank], workspace[rank + 1 :]])
    experts.copy_(torch.tensor([0, 65, 71, 64, 3, 70, 69, 1], dtype=torch.int32).view(-1, 1))
    _dispatch(workspace, meta, experts, [payload, experts], controls, rank=rank)
    _finish(abort, "the dispatch after the incomplete round waited for a peer")
    assert torch.equal(torch.cat([workspace[:rank], workspace[rank + 1 :]]), peers)


def test_an_incomplete_combine_zeroes_the_tokens_of_blocks_that_start_after_others_returned() -> (
    None
):
    """The blocks of a combine that start after others returned early on an abort, and so stored the incomplete-round
    value, still wait for the silent peer and return on the request with a zero output: they do not take the value
    for an earlier collective's and combine the peer's rows without its flag."""
    # One block per token, more than the GPU holds at once, so later blocks start after the first ones returned.
    tokens = 64 * torch.cuda.get_device_properties(0).multi_processor_count
    workspace, meta, experts, payload = _setup(2, tokens=tokens, capacity=tokens)
    controls, abort = _rank_mask_controls(3)
    experts.fill_(1)  # every token to peer 1
    # Peer 1 takes part in the dispatch, then never signals the combine; its combine input holds nonzero rows.
    expected_flag, parity = _next_parity(workspace, meta)
    _word(workspace, _offset(meta, "RECV_COUNTERS") + 4 * (parity * 2 + 1)).fill_(0)
    _word(workspace, _offset(meta, "DISPATCH_COMPLETION_FLAGS") + 4).fill_(expected_flag)
    torch.ops.trtllm.moe_a2a_get_combine_payload_tensor(
        workspace, meta, 1, 2, tokens, payload.dtype, _HIDDEN
    ).fill_(1)
    torch.cuda.synchronize()
    received, offset, _ = _dispatch(workspace, meta, experts, [payload, experts], controls, tokens)
    output = _combine(workspace, meta, received[0], offset, controls, tokens)
    assert _evidence_while_waiting(abort, 1, expected_flag + 1), (
        "the combine did not wait for peer 1"
    )

    abort[0] = 1
    _finish(abort, "the combine kept waiting after the abort request")
    assert abort[:2].tolist() == [1, 1]
    assert torch.count_nonzero(output) == 0
    assert _word(workspace, _offset(meta, "INCOMPLETE_ROUND")).item() == (expected_flag + 1) | 1


def test_72_ranks_with_masked_ranks_in_both_mask_words() -> None:
    """With EP size 72 (NVL72: each lane waits for several peers, ranks 64-71 in the second mask word), are the
    routes to masked ranks of either word dropped and their rows left untouched, and every active peer's count,
    flag, EPLB statistics and combine input taken?"""
    ep_size, top_k, stats_experts = 72, 2, 4
    masked = {3, 65}
    targets = [(0, 70), (65, 1), (64, 3), (71, 65), (70, 71), (2, 64), (3, 66), (1, 0)]
    workspace, meta, _, payload = _setup(ep_size, eplb_stats_num_experts=stats_experts, top_k=top_k)
    experts = torch.tensor(targets, dtype=torch.int32, device="cuda")
    stats = torch.arange(11, 11 + stats_experts, dtype=torch.int32, device="cuda")
    controls, abort = _rank_mask_controls((2**ep_size - 1) & ~sum(1 << rank for rank in masked))
    active = [peer for peer in range(1, ep_size) if peer not in masked]
    # Every peer publishes its count and dispatch flag in rank 0's row; a masked peer's must not count.
    expected_flag, parity = _next_parity(workspace, meta)
    for peer in range(1, ep_size):
        count = _offset(meta, "RECV_COUNTERS") + 4 * (parity * ep_size + peer)
        _word(workspace, count).fill_(peer % 5 + 1)
        _word(workspace, _offset(meta, "DISPATCH_COMPLETION_FLAGS") + 4 * peer).fill_(expected_flag)
    masked_rows = {rank: workspace[rank].clone() for rank in masked}
    received, offset, _ = _dispatch(
        workspace, meta, experts, [payload, experts], controls, stats=stats
    )
    _finish(abort, "the dispatch waited for a peer")

    counts = [_recv_count(workspace, meta, parity, source) for source in range(ep_size)]
    assert counts == [2] + [0 if peer in masked else peer % 5 + 1 for peer in range(1, ep_size)]
    dispatch_rows = _offset(meta, "DISPATCH_PAYLOAD")
    for peer in active:
        sent = [t for t, pair in enumerate(targets) if peer in pair]
        assert (
            _word(workspace, _offset(meta, "DISPATCH_COMPLETION_FLAGS"), peer).item()
            == expected_flag
        )
        assert _word(
            workspace, _offset(meta, "RECV_COUNTERS") + 4 * parity * ep_size, peer
        ).item() == len(sent)
        gathered = _offset(meta, "EPLB_GATHERED_STATS")
        assert (
            workspace[peer, gathered : gathered + 4 * stats_experts].view(torch.int32).equal(stats)
        )
        rows = workspace[peer, dispatch_rows : dispatch_rows + _CAPACITY * _HIDDEN * 2]
        rows = rows.view(torch.bfloat16).view(_CAPACITY, _HIDDEN)
        order = rows[: len(sent), 0].float().argsort()
        torch.testing.assert_close(rows[: len(sent)][order], payload[sent], rtol=0, atol=0)

    # Every active peer signals the combine and, as an identity expert, returns rank 0's rows as it received them.
    for peer in active:
        _word(workspace, _offset(meta, "COMBINE_COMPLETION_FLAGS") + 4 * peer).fill_(
            expected_flag + 1
        )
        rows = workspace[peer, dispatch_rows : dispatch_rows + _CAPACITY * _HIDDEN * 2]
        torch.ops.trtllm.moe_a2a_get_combine_payload_tensor(
            workspace, meta, peer, ep_size, _CAPACITY, payload.dtype, _HIDDEN
        )[:_CAPACITY].copy_(rows.view(torch.bfloat16).view(_CAPACITY, _HIDDEN))
    output = _combine(workspace, meta, received[0], offset, controls, top_k=top_k)
    _finish(abort, "the combine waited for a peer")
    active = torch.tensor([sum(t not in masked for t in pair) for pair in targets], device="cuda")
    expected = (payload.float() * active[:, None]).to(payload.dtype)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert all(torch.equal(workspace[rank], row) for rank, row in masked_rows.items())
    assert abort[:2].tolist() == [0, 0]


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
    assert abort[:2].tolist() == [0, 0]


def test_no_workspace_is_created_once_the_membership_changed() -> None:
    """Creating a workspace exchanges handles with, and waits at a barrier for, every rank of the startup group,
    which after a membership change includes a failed process."""
    health = EPGroupHealth(1)
    health.mark_failed(0)
    health.mark_active(0)  # failed and replaced
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


def test_the_abort_record_needs_the_rank_mask() -> None:
    workspace, meta, experts, payload = _setup(1)
    kwargs = {"enable_rank_mask": False, "abort_state": _rank_mask_controls(1)[1]}
    with pytest.raises(RuntimeError, match="the abort state needs the rank-mask fence path"):
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

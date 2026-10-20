# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run-time rank mask, abort and fixed source slots of the fence MoE All-to-All ops.

Run in a singleton MPI process with CUDA available. Multi-rank cases place every logical
peer in one allocation and only rank 0 issues ops, so they check the native layout,
masking and polling, not peer transport.
"""

import os
import subprocess
import sys

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
    ep_size: int, tokens: int = _CAPACITY, eplb_stats_num_experts=None, extra_bytes: int = 32
) -> tuple[torch.Tensor, ...]:
    meta = NVLinkOneSided._make_workspace_layout(
        ep_size,
        1,
        _CAPACITY,
        _HIDDEN,
        torch.bfloat16,
        eplb_stats_num_experts,
        extra_bytes,
        False,
        False,
    )
    size = int(meta[int(internal.thop.MOE_A2A_WORKSPACE_SIZE_INDEX)])
    workspace = torch.zeros((ep_size, size), dtype=torch.uint8, device="cuda")
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, 0, ep_size, barrier=False)
    experts = torch.zeros((tokens, 1), dtype=torch.int32, device="cuda")
    payload = torch.arange(tokens * _HIDDEN, dtype=torch.bfloat16, device="cuda")
    return workspace, meta, experts, payload.reshape(tokens, _HIDDEN)


def _controls(controls: dict | None) -> dict:
    return {} if controls is None else controls


def _dispatch(workspace, meta, experts, payloads, controls, tokens=_CAPACITY, stats=None):
    """Dispatch like the MoE layer: the received tensors, the combine offset, the gathered EPLB
    statistics, and with fixed slots the rows of each source's slot that hold this round's tokens."""
    ep_size = workspace.size(0)
    args = (experts, payloads, workspace, meta, tokens, 0, ep_size, 1, ep_size, stats)
    if controls is None:
        received = torch.ops.trtllm.moe_a2a_dispatch(
            *args, expert_id_payload_index=1, invalid_token_expert_id=-1
        )
        return (*received, None)
    # Stale values the dispatch must overwrite.
    counts = torch.full((ep_size,), -7, dtype=torch.int32, device="cuda")
    received = torch.ops.trtllm.moe_a2a_dispatch(
        *args, **_controls(controls), source_valid_counts=counts
    )
    return (*received, counts)


def _combine(workspace, meta, payload, offset, controls, tokens=_CAPACITY, in_workspace=False):
    ep_size = workspace.size(0)
    args = (payload, tokens, workspace, meta, tokens, 0, ep_size, 1, offset, in_workspace)
    return torch.ops.trtllm.moe_a2a_combine(*args, **_controls(controls))


def _round(workspace, meta, experts, payload, controls):
    received, offset, _, counts = _dispatch(workspace, meta, experts, [payload, experts], controls)
    return received[1], _combine(workspace, meta, received[0], offset, controls), counts


def test_cancel_and_exclusion_on_replay() -> None:
    """Does one captured graph observe the abort record and the mask written before each replay?"""
    workspace, meta, experts, payload = _setup(2)
    controls, abort = _rank_mask_controls(3)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, output, counts = _round(workspace, meta, experts, payload, controls)

    # Peer 1 never signals: a pending request releases both waits instead of a timeout. Only
    # sources whose completion flag arrived count as received.
    abort[0] = 1
    graph.replay()
    torch.cuda.synchronize()
    assert abort.tolist() == [1, 1]
    assert counts.tolist() == [_CAPACITY, 0]
    assert torch.count_nonzero(output) == 0

    # Rearmed with peer 1 excluded, the same graph completes without it.
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, 0, 2, barrier=False)
    abort.zero_()
    _set_mask(controls, 1)
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, payload, rtol=0, atol=0)
        assert counts.tolist() == [_CAPACITY, 0]
    assert abort.tolist() == [0, 0]


@pytest.mark.parametrize("row_bytes", [16, 24, 6, 3])
def test_registered_source_slots_preserve_excluded_state(row_bytes: int) -> None:
    """Do graphs of different batch sizes share fixed source slots and leave excluded ones untouched?"""
    ep_size = 2
    workspace, meta, experts, payload = _setup(ep_size, eplb_stats_num_experts=ep_size)
    byte_payload = torch.arange(_CAPACITY * row_bytes, dtype=torch.uint8, device="cuda")
    byte_payload = byte_payload.reshape(_CAPACITY, row_bytes)
    controls, abort = _rank_mask_controls(1)

    def padded(size: int) -> int:
        return (size + 127) // 128 * 128

    # Each payload has a region of a fixed slot per source, sized for the capacity.
    widths = (_HIDDEN * 2, 4, row_bytes)
    regions = [_offset(meta, "DISPATCH_PAYLOAD")]
    for width in widths:
        regions.append(regions[-1] + padded(ep_size * _CAPACITY * width))
    combine_input = _offset(meta, "COMBINE_INPUT")
    assert regions[-1] <= combine_input
    fixed = torch.ops.trtllm.moe_a2a_get_combine_payload_tensor(
        workspace, meta, 0, ep_size, _CAPACITY, payload.dtype, _HIDDEN
    ).view(-1, _HIDDEN)

    # Poison everything source rank 1 owns in rank 0's workspace: its slot in every payload region.
    # Rank 0 itself resets its receive counters for source 1 each round and never reads them while
    # source 1 is masked, so they are not part of the quarantined state.
    slots = [
        (start + _CAPACITY * w, start + 2 * _CAPACITY * w) for start, w in zip(regions, widths)
    ]
    for lo, hi in slots:
        workspace[0, lo:hi].fill_(0xA5)
    tombstones = []
    for name in ("DISPATCH_COMPLETION_FLAGS", "COMBINE_COMPLETION_FLAGS"):
        tombstones.append(_word(workspace, _offset(meta, name) + 4))
    stats = _offset(meta, "EPLB_GATHERED_STATS") + ep_size * 4
    tombstones.append(workspace[0, stats : stats + ep_size * 4].view(torch.int32))
    for value in tombstones:
        value.fill_(2**31 - 1)
    tombstone_values = [torch.full_like(value, 2**31 - 1) for value in tombstones]

    def transfer(bucket: int) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        ids = experts[:bucket]
        payloads = [payload[:bucket], ids, byte_payload[:bucket]]
        received, offset, _, counts = _dispatch(workspace, meta, ids, payloads, controls, bucket)
        assert offset == combine_input
        # Every payload is the slots themselves: source s's row i is row s * capacity + i.
        for tensor, region in zip(received, regions):
            assert tensor.shape[:2] == (ep_size, _CAPACITY)
            assert tensor.data_ptr() == workspace[0].data_ptr() + region
        expert_output = fixed[: ep_size * bucket]
        expert_output.copy_(received[0][:, :bucket].reshape(-1, _HIDDEN))
        expert_output = expert_output.view(ep_size, bucket, _HIDDEN)
        output = _combine(workspace, meta, expert_output, offset, controls, bucket, True)
        return received, counts, output

    captures = {}
    for bucket in (4, 8):
        transfer(bucket)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = transfer(bucket)
        captures[bucket] = graph, *outputs

    for bucket in (8, 4, 8, 4):
        graph, received, counts, output = captures[bucket]
        graph.replay()
        torch.cuda.synchronize()
        expected = payload[:bucket]
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        order = received[0][0][:bucket, 0].argsort()
        torch.testing.assert_close(received[0][0][:bucket][order], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            received[2][0][:bucket][order], byte_payload[:bucket], rtol=0, atol=0
        )
        assert torch.all(received[1][0][:bucket] == 0)
        assert counts.tolist() == [bucket, 0]
        assert all(torch.all(workspace[0, lo:hi] == 0xA5) for lo, hi in slots)
        assert [value.tolist() for value in tombstones] == [
            value.tolist() for value in tombstone_values
        ]
        assert abort.tolist() == [0, 0]


def test_valid_counts_skip_a_masked_source_even_if_it_arrived() -> None:
    """A masked peer's completed round is still never read."""
    workspace, meta, experts, payload = _setup(2)
    controls, _ = _rank_mask_controls(1)
    expected_flag = _word(workspace, _offset(meta, "FLAG_VAL")).item() + 1
    parity = ((expected_flag - 1) >> 1) & 1
    _word(workspace, _offset(meta, "DISPATCH_COMPLETION_FLAGS") + 4).fill_(expected_flag)
    _word(workspace, _offset(meta, "RECV_COUNTERS") + 4 * (parity * 2 + 1)).fill_(3)
    _, _, _, counts = _dispatch(workspace, meta, experts, [payload, experts], controls)
    torch.cuda.synchronize()
    assert counts.tolist() == [_CAPACITY, 0]


@pytest.mark.parametrize("count,valid", [(3, 3), (_CAPACITY + 1, 0), (-5, 0)])
def test_valid_counts_bound_the_received_count(count: int, valid: int) -> None:
    """Does the dispatch only trust a peer's received count within [0, runtime tokens]?"""
    workspace, meta, experts, payload = _setup(2)
    controls, _ = _rank_mask_controls(3)
    # Stand in for peer 1: signal the next dispatch round and publish its count.
    expected_flag = _word(workspace, _offset(meta, "FLAG_VAL")).item() + 1
    parity = ((expected_flag - 1) >> 1) & 1
    _word(workspace, _offset(meta, "DISPATCH_COMPLETION_FLAGS") + 4).fill_(expected_flag)
    _word(workspace, _offset(meta, "RECV_COUNTERS") + 4 * (parity * 2 + 1)).fill_(count)
    _, _, _, counts = _dispatch(workspace, meta, experts, [payload, experts], controls)
    torch.cuda.synchronize()
    assert counts.tolist() == [_CAPACITY, valid]


def test_valid_counts_wait_for_a_source_that_arrives_late() -> None:
    """Does a source count as received when its count and flag arrive after the dispatch started?"""
    workspace, meta, experts, payload = _setup(2)
    controls, _ = _rank_mask_controls(3)
    expected_flag = _word(workspace, _offset(meta, "FLAG_VAL")).item() + 1
    parity = ((expected_flag - 1) >> 1) & 1
    torch.cuda.synchronize()
    # Stand in for peer 1 on another stream: publish its count, then its flag, well after the dispatch
    # started waiting for it.
    late = torch.cuda.Stream()
    with torch.cuda.stream(late):
        torch.cuda._sleep(50_000_000)
        _word(workspace, _offset(meta, "RECV_COUNTERS") + 4 * (parity * 2 + 1)).fill_(3)
        _word(workspace, _offset(meta, "DISPATCH_COMPLETION_FLAGS") + 4).fill_(expected_flag)
    _, _, _, counts = _dispatch(workspace, meta, experts, [payload, experts], controls)
    torch.cuda.synchronize()
    assert counts.tolist() == [_CAPACITY, 3]


def test_valid_counts_ignore_a_count_without_its_completion_flag() -> None:
    """Does an aborted round drop a source whose count arrived but whose completion flag did not?"""
    workspace, meta, experts, payload = _setup(2)
    controls, abort = _rank_mask_controls(3)
    expected_flag = _word(workspace, _offset(meta, "FLAG_VAL")).item() + 1
    parity = ((expected_flag - 1) >> 1) & 1
    _word(workspace, _offset(meta, "RECV_COUNTERS") + 4 * (parity * 2 + 1)).fill_(3)
    abort[0] = 1
    _, _, _, counts = _dispatch(workspace, meta, experts, [payload, experts], controls)
    torch.cuda.synchronize()
    assert counts.tolist() == [_CAPACITY, 0]


@pytest.mark.parametrize("peer_active", [True, False])
def test_fixed_slots_gather_eplb_statistics_of_active_sources(peer_active: bool) -> None:
    """With fixed slots, eplb_stats_out holds each active source's statistics row and 0 for a masked one."""
    stats_experts = 2  # at most num_experts (2 here)
    workspace, meta, experts, payload = _setup(2, eplb_stats_num_experts=stats_experts)
    ids = experts.clone()
    weights = torch.ones((_CAPACITY, 1), device="cuda")
    local_stats = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    controls, _ = _rank_mask_controls(3 if peer_active else 1)
    # Stand in for peer 1: its statistics row in rank 0's gathered buffer, then its count and flag.
    gathered = _offset(meta, "EPLB_GATHERED_STATS")
    workspace[0, gathered + stats_experts * 4 : gathered + 2 * stats_experts * 4].view(
        torch.int32
    ).copy_(torch.tensor([10, 20], dtype=torch.int32))
    expected_flag = _word(workspace, _offset(meta, "FLAG_VAL")).item() + 1
    parity = ((expected_flag - 1) >> 1) & 1
    _word(workspace, _offset(meta, "RECV_COUNTERS") + 4 * (parity * 2 + 1)).fill_(0)
    _word(workspace, _offset(meta, "DISPATCH_COMPLETION_FLAGS") + 4).fill_(expected_flag)
    stats_out = torch.full((2, stats_experts), -7, dtype=torch.int32, device="cuda")
    torch.ops.trtllm.moe_a2a_dispatch(
        experts,
        [payload, ids, weights],
        workspace,
        meta,
        _CAPACITY,
        0,
        2,
        1,
        2,
        local_stats,
        expert_id_payload_index=1,
        **controls,
        source_valid_counts=torch.zeros(2, dtype=torch.int32, device="cuda"),
        compact_expert_ids=torch.empty((2 * _CAPACITY, 1), dtype=torch.int32, device="cuda"),
        compact_weights=torch.empty((2 * _CAPACITY, 1), device="cuda"),
        eplb_stats_out=stats_out,
    )
    torch.cuda.synchronize()
    assert stats_out[0].tolist() == [1, 2]
    assert stats_out[1].tolist() == ([10, 20] if peer_active else [0, 0])


def test_an_aborted_source_gets_a_masked_sources_rows() -> None:
    """A source an abort released gets -1 / 0 compact rows and a 0 statistics row, never stale buffer contents."""
    stats_experts = 2
    workspace, meta, experts, payload = _setup(2, eplb_stats_num_experts=stats_experts)
    weights = torch.ones((_CAPACITY, 1), device="cuda")
    controls, abort = _rank_mask_controls(3)
    # Peer 1 wrote its statistics row and count but never released its flag.
    gathered = _offset(meta, "EPLB_GATHERED_STATS")
    workspace[0, gathered + stats_experts * 4 : gathered + 2 * stats_experts * 4].view(
        torch.int32
    ).copy_(torch.tensor([10, 20], dtype=torch.int32))
    expected_flag = _word(workspace, _offset(meta, "FLAG_VAL")).item() + 1
    parity = ((expected_flag - 1) >> 1) & 1
    _word(workspace, _offset(meta, "RECV_COUNTERS") + 4 * (parity * 2 + 1)).fill_(3)
    abort[0] = 1
    ids = torch.full((2 * _CAPACITY, 1), 99, dtype=torch.int32, device="cuda")
    compact_weights = torch.full((2 * _CAPACITY, 1), 9.0, device="cuda")
    stats_out = torch.full((2, stats_experts), -7, dtype=torch.int32, device="cuda")
    torch.ops.trtllm.moe_a2a_dispatch(
        experts,
        [payload, experts.clone(), weights],
        workspace,
        meta,
        _CAPACITY,
        0,
        2,
        1,
        2,
        torch.tensor([1, 2], dtype=torch.int32, device="cuda"),
        expert_id_payload_index=1,
        **controls,
        source_valid_counts=torch.zeros(2, dtype=torch.int32, device="cuda"),
        compact_expert_ids=ids,
        compact_weights=compact_weights,
        eplb_stats_out=stats_out,
    )
    torch.cuda.synchronize()
    assert stats_out[1].tolist() == [0, 0]
    assert ids[_CAPACITY:].view(-1).tolist() == [-1] * _CAPACITY
    assert compact_weights[_CAPACITY:].view(-1).tolist() == [0.0] * _CAPACITY


def _compact_round(workspace, meta, experts, payloads, controls):
    """Dispatch with compact routing lists (expert ids at payload 1, router weights at payload 2)."""
    ep_size = workspace.size(0)
    counts = torch.full((ep_size,), -7, dtype=torch.int32, device="cuda")
    ids = torch.full((ep_size * _CAPACITY, 1), 99, dtype=torch.int32, device="cuda")
    weights = torch.full((ep_size * _CAPACITY, 1), 9.0, device="cuda")
    received, *_ = torch.ops.trtllm.moe_a2a_dispatch(
        experts,
        payloads,
        workspace,
        meta,
        _CAPACITY,
        0,
        ep_size,
        1,
        ep_size,
        None,
        expert_id_payload_index=1,
        **controls,
        source_valid_counts=counts,
        compact_expert_ids=ids,
        compact_weights=weights,
    )
    return received, counts, ids, weights


@pytest.mark.parametrize("peer_active", [True, False])
def test_compact_routing_copies_the_rows_each_source_sent(peer_active: bool) -> None:
    """The dispatch's compact lists hold each active source's sent rows, -1 / 0 elsewhere."""
    local = 5
    workspace, meta, experts, payload = _setup(2, tokens=local)
    sent_ids = torch.arange(100, 100 + local, dtype=torch.int32, device="cuda").view(local, 1)
    sent_weights = torch.arange(local, dtype=torch.float32, device="cuda").view(local, 1) + 0.5
    payloads = [payload, sent_ids, sent_weights]
    controls, _ = _rank_mask_controls(1)
    # A first round to learn where peer 1's slot rows live in rank 0's receive buffers.
    received, *_ = _compact_round(workspace, meta, experts, payloads, controls)
    torch.cuda.synchronize()
    # Stand in for peer 1: its 3 rows in its slot, then its count and completion flag for the next round.
    received[1][1, :3] = torch.tensor([[200], [201], [202]], dtype=torch.int32, device="cuda")
    received[2][1, :3] = torch.tensor([[7.0], [8.0], [9.0]], device="cuda")
    expected_flag = _word(workspace, _offset(meta, "FLAG_VAL")).item() + 1
    parity = ((expected_flag - 1) >> 1) & 1
    _word(workspace, _offset(meta, "RECV_COUNTERS") + 4 * (parity * 2 + 1)).fill_(3)
    _word(workspace, _offset(meta, "DISPATCH_COMPLETION_FLAGS") + 4).fill_(expected_flag)
    _set_mask(controls, 3 if peer_active else 1)
    _, counts, ids, weights = _compact_round(workspace, meta, experts, payloads, controls)
    torch.cuda.synchronize()

    peer_rows = 3 if peer_active else 0
    assert counts.tolist() == [local, peer_rows]
    rows = list(zip(ids.view(-1).tolist(), weights.view(-1).tolist()))
    # Rank 0's own tokens land in its slot in arrival order; each keeps its weight.
    assert sorted(rows[:local]) == [(100 + i, i + 0.5) for i in range(local)]
    assert rows[local:_CAPACITY] == [(-1, 0.0)] * (_CAPACITY - local)
    peer = [(200, 7.0), (201, 8.0), (202, 9.0)][:peer_rows]
    assert rows[_CAPACITY:] == peer + [(-1, 0.0)] * (_CAPACITY - peer_rows)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("local", [0, 5])
def test_each_source_gets_its_own_valid_count(local: int, compact: bool) -> None:
    """Is every source's count right: self, a peer with rows, a peer with none and a departed peer that arrived?"""
    ep_size = 4
    workspace, meta, experts, payload = _setup(ep_size, tokens=local)
    weights = torch.ones((local, 1), device="cuda")
    controls, abort = _rank_mask_controls(0b0111)
    # Stand in for peers 1-3: each publishes its count and completion flag for the next round. Peer 3 has
    # departed (masked out), so its arrival must not count.
    expected_flag = _word(workspace, _offset(meta, "FLAG_VAL")).item() + 1
    parity = ((expected_flag - 1) >> 1) & 1
    peer_counts = {1: 3, 2: 0, 3: 2}
    for peer, count in peer_counts.items():
        counter = _offset(meta, "RECV_COUNTERS") + 4 * (parity * ep_size + peer)
        _word(workspace, counter).fill_(count)
        _word(workspace, _offset(meta, "DISPATCH_COMPLETION_FLAGS") + 4 * peer).fill_(expected_flag)
    payloads = [payload, experts, weights]
    if compact:
        _, counts, ids, _ = _compact_round(workspace, meta, experts, payloads, controls)
    else:
        counts = torch.full((ep_size,), -7, dtype=torch.int32, device="cuda")
        torch.ops.trtllm.moe_a2a_dispatch(
            experts,
            payloads,
            workspace,
            meta,
            _CAPACITY,
            0,
            ep_size,
            1,
            ep_size,
            None,
            **controls,
            source_valid_counts=counts,
        )
    torch.cuda.synchronize()

    assert counts.tolist() == [local, 3, 0, 0]
    assert abort.tolist() == [0, 0]
    departed_counter = _offset(meta, "RECV_COUNTERS") + 4 * (parity * ep_size + 3)
    assert _word(workspace, departed_counter).item() == 0
    if compact:
        rows = ids.view(ep_size, _CAPACITY).tolist()
        assert all(row == [-1] * _CAPACITY for row in rows[2:])
        assert rows[0][local:] == [-1] * (_CAPACITY - local)


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"source_valid_counts": None}, "compact routing lists need source_valid_counts"),
        ({"compact_weights": None}, "go together"),
        (
            {"compact_expert_ids": torch.zeros((3, 1), dtype=torch.int32)},
            "compact_expert_ids and compact_weights must",
        ),
        ({"invalid_token_expert_id": -1}, "exclude expert-id sanitization"),
    ],
)
def test_compact_routing_rejects_inconsistent_options(extra: dict, match: str) -> None:
    workspace, meta, experts, payload = _setup(1)
    weights = torch.zeros((_CAPACITY, 1), dtype=torch.float32, device="cuda")
    kwargs = {
        **_rank_mask_controls(1)[0],
        "expert_id_payload_index": 1,
        "source_valid_counts": torch.zeros(1, dtype=torch.int32, device="cuda"),
        "compact_expert_ids": torch.zeros((_CAPACITY, 1), dtype=torch.int32, device="cuda"),
        "compact_weights": torch.zeros((_CAPACITY, 1), dtype=torch.float32, device="cuda"),
    }
    kwargs.update({k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in extra.items()})
    with pytest.raises(RuntimeError, match=match):
        torch.ops.trtllm.moe_a2a_dispatch(
            experts, [payload, experts, weights], workspace, meta, _CAPACITY, 0, 1, 1, 1, **kwargs
        )


def test_source_slots_reject_runtime_over_capacity() -> None:
    workspace, meta, experts, payload = _setup(1, _CAPACITY + 1)
    controls, _ = _rank_mask_controls(1)
    with pytest.raises(RuntimeError, match="allocation-time token capacity"):
        _dispatch(workspace, meta, experts, [experts], controls, _CAPACITY + 1)


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"use_cft_counted_writes": True}, "fence path"),
        ({"expert_id_payload_index": 1, "invalid_token_expert_id": -1}, "source_valid_counts"),
        ({"enable_rank_mask": False, "active_rank_mask": None, "abort_state": None}, "fence path"),
    ],
)
def test_source_slots_reject_unsupported_dispatch_options(extra: dict, match: str) -> None:
    workspace, meta, experts, payload = _setup(1)
    kwargs = {
        **_rank_mask_controls(1)[0],
        "source_valid_counts": torch.zeros(1, dtype=torch.int32, device="cuda"),
        **extra,
    }
    with pytest.raises(RuntimeError, match=match):
        torch.ops.trtllm.moe_a2a_dispatch(
            experts, [payload, experts], workspace, meta, _CAPACITY, 0, 1, 1, 1, **kwargs
        )


def test_schema_and_fake_signatures() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        workspace = torch.empty((1, 1 << 20), dtype=torch.uint8, device="cuda")
        meta = torch.empty(15, dtype=torch.int64)
        experts = torch.empty((_CAPACITY, 1), dtype=torch.int32, device="cuda")
        payload = torch.empty((_CAPACITY, _HIDDEN), dtype=torch.bfloat16, device="cuda")
        controls = {
            "enable_rank_mask": True,
            "active_rank_mask": torch.empty(4, dtype=torch.uint64, device="cuda"),
            "abort_state": torch.empty(2, dtype=torch.int32),
        }
        for value in (None, controls):
            ids, output, _ = _round(workspace, meta, experts, payload, value)
            assert ids.shape == (1, _CAPACITY, 1)
            assert output.shape == (_CAPACITY, _HIDDEN) and output.dtype == torch.bfloat16


def _fence_timeout() -> None:
    """Does a fence wait on a silent peer still fail stop at the timeout?"""
    torch.ops.trtllm.moe_a2a_set_timeout(1)
    workspace, meta, experts, payload = _setup(2)
    controls, _ = _rank_mask_controls(3)
    _dispatch(workspace, meta, experts, [payload, experts], controls)
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        print("fence timeout trapped", flush=True)
        os._exit(0)
    raise AssertionError("dispatch completed without its peer")


def test_fence_timeout_still_traps() -> None:
    # The child traps on purpose, so it runs in a fresh interpreter, kept off the enclosing launcher's
    # process-management interface.
    launcher_prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(launcher_prefixes)}
    try:
        result = subprocess.run(
            [sys.executable, __file__], env=env, capture_output=True, text=True, timeout=300
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"Pending fence subprocess deadline: {error.stdout!r} {error.stderr!r}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fence timeout trapped" in result.stdout, result.stdout + result.stderr


if __name__ == "__main__":
    _fence_timeout()

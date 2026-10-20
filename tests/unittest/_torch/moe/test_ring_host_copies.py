# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EPLB host copies of a subset of the experts, as WideEP fault tolerance keeps them in a ring of nodes."""

import functools
import threading
from multiprocessing import shared_memory
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    HostMoeTensorSharer,
    MoeLoadBalancer,
    ring_copy_nodes,
)
from tensorrt_llm.bindings import internal

pytestmark = pytest.mark.cpu_only


def _sharer(local_rank, local_size, expert_count=8, expert_ids=None):
    comm = Mock(Get_rank=Mock(return_value=local_rank), Get_size=Mock(return_value=local_size))
    sharer = HostMoeTensorSharer(0, expert_count, comm)
    if expert_ids is not None:
        sharer.keep_experts(expert_ids)
    return sharer


@pytest.mark.parametrize("expert_count, local_size", [(8, 4), (10, 4), (3, 4)])
def test_sharer_splits_every_expert_by_default_as_before(expert_count, local_size):
    for local_rank in range(local_size):
        sharer = _sharer(local_rank, local_size, expert_count)
        start = local_rank * expert_count // local_size
        end = (local_rank + 1) * expert_count // local_size
        assert sharer.own_expert_ids == list(range(start, end))


def test_sharer_splits_the_experts_it_holds_over_the_node_s_ranks():
    held = [13, 5, 6, 7, 12]
    parts = [_sharer(rank, 2, 16, held).own_expert_ids for rank in range(2)]
    assert parts == [[5, 6], [7, 12, 13]]
    sharer = _sharer(1, 2, 16, held)
    assert sharer.rank_expert_ids(0) == [5, 6]
    # Another rank's experts and experts the node does not hold are only registered by shape.
    for expert in (5, 0):
        sharer.pre_register_host_tensor_with_shape(expert, "w", torch.float32, (4,))
    with pytest.raises(AssertionError):
        sharer.pre_register_host_tensor_with_shape(7, "w", torch.float32, (4,))
    with pytest.raises(AssertionError):
        sharer.share_host_tensor_with_shape(0, "w", torch.zeros(4))


def test_a_rank_that_shares_no_expert_creates_no_shared_memory():
    # Three experts over four ranks: local rank 0 loads none.
    sharers = [_sharer(rank, 4, expert_count=3) for rank in range(4)]
    assert sharers[0].own_expert_ids == []
    for sharer in sharers:
        sharer.set_shared_memory_base_name(f"test_ring_host_copies_{id(sharers)}")
        for expert in range(3):
            if expert in sharer.own_expert_ids:
                sharer.share_host_tensor_with_shape(expert, "w", torch.full((4,), float(expert)))
            else:
                sharer.pre_register_host_tensor_with_shape(expert, "w", torch.float32, (4,))
    try:
        for sharer in sharers:
            sharer.finalize_layer_weights()
        assert sharers[0].own_shm is None
        for sharer in sharers:
            weights = {}
            sharer.finalize_host_tensor_sharing(
                lambda expert, name, t: weights.__setitem__(expert, t.tolist())
            )
            assert weights == {expert: [float(expert)] * 4 for expert in range(3)}
        for sharer in sharers:
            sharer.unlink_shared_memory()
        # The names are gone; the mappings still hold the weights.
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=sharers[1].get_shared_memory_name())
        # Local rank 0 maps local rank 3's segment, which holds expert 2, last.
        assert (
            bytes(sharers[0].imported_shms[-1].buf[:16]) == torch.full((4,), 2.0).numpy().tobytes()
        )
    finally:
        for sharer in sharers:
            sharer.pre_shutdown_cleanup()
        for sharer in sharers:
            sharer.post_shutdown_cleanup()


def test_a_layer_that_does_not_fit_in_shared_memory_fails_before_creating_it(monkeypatch):
    # Two local ranks share four experts of 300 bytes aligned to 512: this rank's part needs 1024 bytes.
    sharer = _sharer(0, 2, expert_count=4)
    sharer.set_shared_memory_base_name(f"test_ring_host_copies_{id(sharer)}")
    for expert in range(4):
        if expert in sharer.own_expert_ids:
            sharer.share_host_tensor_with_shape(expert, "w", torch.zeros(300, dtype=torch.uint8))
        else:
            sharer.pre_register_host_tensor_with_shape(expert, "w", torch.uint8, (300,))
    statvfs = Mock(return_value=Mock(f_bavail=1023, f_frsize=1, f_blocks=4096))
    monkeypatch.setattr("os.statvfs", statvfs)
    with pytest.raises(RuntimeError, match=r"/dev/shm has .* MoE layer 0 needs"):
        sharer.finalize_layer_weights()
    assert sharer.own_shm is None
    statvfs.return_value.f_bavail = 1024
    try:
        sharer.finalize_layer_weights()
        assert sharer.own_shm is not None
    finally:
        sharer.pre_shutdown_cleanup()
        sharer.post_shutdown_cleanup()


def test_a_sharer_leaves_a_segment_it_did_not_create_in_place():
    sharer = _sharer(0, 1, expert_count=1)
    sharer.set_shared_memory_base_name(f"test_ring_host_copies_{id(sharer)}")
    sharer.share_host_tensor_with_shape(0, "w", torch.ones(4))
    other = shared_memory.SharedMemory(name=sharer.get_shared_memory_name(), create=True, size=8)
    try:
        other.buf[0] = 7
        with pytest.raises(RuntimeError, match="did not create"):
            sharer.finalize_layer_weights()
        assert other.buf[0] == 7
    finally:
        other.close()
        other.unlink()


def test_load_balancers_name_their_shared_memory_apart(monkeypatch):
    monkeypatch.setattr(
        MoeLoadBalancer,
        "_setup_mpi_comm",
        lambda self: setattr(
            self, "shared_mpi_comm", Mock(Get_rank=Mock(return_value=0), bcast=lambda token: token)
        ),
    )
    with patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"):
        names = {MoeLoadBalancer(0, 1, 0, "base").shared_memory_base_name for _ in range(2)}
    assert len(names) == 2 and all(name.startswith("base_") for name in names)


_MLB = "tensorrt_llm._torch.moe.fused_moe.moe_load_balancer"
# A launcher that maps ranks round-robin: node "a" runs ranks 0 and 2, node "b" ranks 1 and 3.
_HOSTS = ["a", "b", "a", "b"]
_EXPERTS = 8
# Two EPLB slots per rank and a ghost slot; the placement after FT emptied every replica.
_PLACEMENT = [[0, 1, -1], [2, 3, -1], [4, 5, -1], [6, 7, -1]]
# As MoE modules set it: (ep_rank * E // ep_size + local_slot) % E, ghost slots holding replicas until emptied.
_ASSIGNMENTS = [0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 0]
# Slot weights by name: w1 300 bytes, w2 200 bytes.
_WEIGHTS = {"w1": (300, torch.uint8), "w2": (100, torch.float16)}


def _no_writes(rank: int) -> list:
    """Write ranges of a workspace no all-to-all writes."""
    return []


def test_ring_keeps_each_node_s_copies_on_the_node_before_it():
    hosts = ["b", "a", "b", "a", "c", "c"]
    # Nodes by first rank: b (0, 2), a (1, 3), c (4, 5); b's copies live on c, a's on b, c's on a.
    assert ring_copy_nodes(hosts) == [(4, 5), (0, 2), (4, 5), (0, 2), (1, 3), (1, 3)]
    assert ring_copy_nodes(["a", "b"]) == [(1,), (0,)]


def test_a_single_node_keeps_its_own_copies():
    assert ring_copy_nodes(["a"] * 4) == [(0, 1, 2, 3)] * 4


def _ring_load_balancer(
    monkeypatch,
    ep_rank=0,
    layers=1,
    layer_updates_per_iter=0,
    placement=_PLACEMENT,
    assignments=_ASSIGNMENTS,
):
    """A MoeLoadBalancer with WideEP peer-loss recovery over native layers showing ``placement`` (initial weight
    ``assignments``); with a static placement (no ``layer_updates_per_iter``) it keeps ring host copies."""
    slots = len(placement[0])
    monkeypatch.setattr(f"{_MLB}.peer_loss_recovery_enabled", lambda: True)
    monkeypatch.setattr(MoeLoadBalancer, "_gather_copy_nodes", lambda self: ring_copy_nodes(_HOSTS))
    monkeypatch.setattr(MoeLoadBalancer, "_gather_largest_loss", lambda self: 2)
    with (
        patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"),
        patch(f"{_MLB}.HostMoeTensorSharer") as sharer,
    ):
        load_balancer = MoeLoadBalancer(ep_rank, len(_HOSTS), layer_updates_per_iter)
        load_balancer.load_balancer_impl.add_layer.side_effect = [
            Mock(
                get_layer_id=Mock(return_value=i),
                get_old_rank_expert_ids=Mock(return_value=placement),
            )
            for i in range(layers)
        ]
        for _ in range(layers):
            layer = load_balancer.add_layer(_EXPERTS, 2, slots)
            layer.set_initial_weight_assignments(assignments)
            for slot in range(slots):
                for name, (count, dtype) in _WEIGHTS.items():
                    layer.weight_slots[(slot, name)] = torch.zeros(count, dtype=dtype)
    return load_balancer, sharer


def test_online_eplb_keeps_every_expert_s_host_copy(monkeypatch):
    load_balancer, sharer = _ring_load_balancer(monkeypatch, layer_updates_per_iter=1)
    (layer,) = load_balancer.single_layer_load_balancers
    assert load_balancer.copy_nodes is None and layer.host_copy_ranks is None
    sharer.return_value.keep_experts.assert_not_called()


@pytest.mark.parametrize("ep_rank, held", [(0, [2, 3, 6, 7]), (1, [0, 1, 4, 5])])
def test_a_node_shares_the_home_experts_of_the_next_node(monkeypatch, ep_rank, held):
    load_balancer, sharer = _ring_load_balancer(monkeypatch, ep_rank)
    (layer,) = load_balancer.single_layer_load_balancers
    # Home ranks: each expert's slot after the replicas are emptied.
    assert layer.home_ranks.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    # Node a (ranks 0, 2) keeps node b's (ranks 1, 3) experts and the other way round.
    assert layer.host_copy_ranks == ([1, 3] if ep_rank == 0 else [0, 2])
    sharer.assert_called_once_with(0, _EXPERTS, load_balancer.shared_mpi_comm)
    sharer.return_value.keep_experts.assert_called_once_with(held)
    # Setting the weights again after loading keeps the experts.
    layer.set_initial_weight_assignments(_ASSIGNMENTS)
    assert sharer.return_value.keep_experts.call_count == 1


def test_an_expert_whose_copy_node_is_lost_too_is_not_restored(monkeypatch):
    load_balancer, _ = _ring_load_balancer(monkeypatch)
    # Losing node b's rank 3: its experts' copies are on node a.
    assert load_balancer.covers_loss_of([3])
    # Experts 0 and 1 (home rank 0) have their copies on node b; with all of it lost too, nothing restores them.
    with pytest.raises(ValueError, match=r"\{0: \[0, 1, 3, 6, 7\]\}"):
        load_balancer.recovery_placements([0, 1, 3])


def test_the_copy_plan_is_the_same_on_every_rank_and_spreads_sources(monkeypatch):
    plans = []
    for ep_rank in range(len(_HOSTS)):
        load_balancer, _ = _ring_load_balancer(monkeypatch, ep_rank)
        placements = load_balancer.recovery_placements([3])
        plan = load_balancer.restore_copy_plan([3], placements, 2048, _no_writes)
        plans.append((plan, load_balancer.placement_digest([3], placements, plan)))
    assert all((plan == plans[0][0]).all() and digest == plans[0][1] for plan, digest in plans)
    # Rank 3's experts 6 and 7 go to the ghost slots of ranks 0 and 1, copied by node a's ranks 0 and 2 in turn.
    # Columns: chunk, layer, target, slot, expert, weight, source, offset, offset in the weight, bytes.
    assert plans[0][0].tolist() == [
        [0, 0, 0, 2, 6, 0, 0, 0, 0, 300],
        [0, 0, 0, 2, 6, 1, 0, 512, 0, 200],
        [0, 0, 1, 2, 7, 0, 2, 0, 0, 300],
        [0, 0, 1, 2, 7, 1, 2, 512, 0, 200],
    ]
    # Another plan digests differently.
    assert load_balancer.placement_digest([3], placements, plan[:, ::-1].copy()) != plans[0][1]


def test_the_copy_plan_alternates_workspace_halves_between_chunks(monkeypatch):
    load_balancer, _ = _ring_load_balancer(monkeypatch, layers=2)
    placements = load_balancer.recovery_placements([3])
    # A 1024-byte workspace: halves of 512 bytes. A weight the rest of a half cannot take continues in the next
    # chunk, in the other half.
    plan = load_balancer.restore_copy_plan([3], placements, 1024, _no_writes)
    target_0 = plan[plan[:, 2] == 0]
    # Columns: chunk, layer, weight, offset, offset in the weight, bytes.
    assert target_0[:, [0, 1, 5, 7, 8, 9]].tolist() == [
        [0, 0, 0, 0, 0, 300],
        [1, 0, 1, 512, 0, 200],
        [1, 1, 0, 768, 0, 256],
        [2, 1, 0, 0, 256, 44],
        [2, 1, 1, 256, 0, 200],
    ]
    assert (plan[1:, 0] >= plan[:-1, 0]).all()
    # 256 bytes: the second half has no aligned room.
    room = r"\[128, 0\] bytes per half .* TRTLLM_NVLINK_ONE_SIDED_A2A_WORKSPACE_MB"
    with pytest.raises(ValueError, match=room):
        load_balancer.restore_copy_plan([3], placements, 256, _no_writes)


def test_the_copy_plan_refuses_to_load_a_routed_slot(monkeypatch):
    load_balancer, _ = _ring_load_balancer(monkeypatch)
    (placement,) = load_balancer.recovery_placements([3])
    # Rank 0's slot 1 still routes expert 1.
    placement[0, 1] = 6
    with pytest.raises(ValueError, match="would overwrite a routed one"):
        load_balancer.restore_copy_plan([3], [placement], 2048, _no_writes)


def _expected(layer_idx, expert, name):
    count, dtype = _WEIGHTS[name]
    generator = torch.Generator().manual_seed(layer_idx * 1000 + expert * 10 + len(name))
    return torch.randint(0, 256, (count * dtype.itemsize,), dtype=torch.uint8, generator=generator)


def test_staged_restores_load_every_slot_bit_exactly(monkeypatch):
    """Survivor ranks as threads over one host-memory "workspace": copies, barriers and halves interleave."""
    dead, survivors = [3], [0, 1, 2]
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: Mock())
    workspace = torch.zeros(len(_HOSTS), 1024, dtype=torch.uint8)
    ranks = {}
    for ep_rank in survivors:
        load_balancer, _ = _ring_load_balancer(monkeypatch, ep_rank, layers=2)
        for layer_idx, layer in enumerate(load_balancer.single_layer_load_balancers):
            for expert in range(_EXPERTS):
                if ep_rank in load_balancer.copy_nodes[layer.home_ranks[expert]]:
                    for name in _WEIGHTS:
                        layer.ring_host_weights[(expert, name)] = _expected(
                            layer_idx, expert, name
                        ).view(_WEIGHTS[name][1])
        placements = load_balancer.recovery_placements(dead)
        ranks[ep_rank] = (load_balancer, placements)
    plan = ranks[0][0].restore_copy_plan(dead, ranks[0][1], workspace.size(1), _no_writes)
    assert plan[-1, 0] >= 2
    barrier = threading.Barrier(len(survivors))
    errors = []

    def run(load_balancer):
        try:
            load_balancer.stage_restored_slots(plan, workspace, lambda: barrier.wait(timeout=30))
        except BaseException as error:  # noqa: BLE001 - re-raised by the test thread
            errors.append(error)
            barrier.abort()

    threads = [threading.Thread(target=run, args=(ranks[r][0],)) for r in survivors]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    for ep_rank in survivors:
        load_balancer, placements = ranks[ep_rank]
        for layer_idx, layer in enumerate(load_balancer.single_layer_load_balancers):
            for slot in range(3):
                expert = int(placements[layer_idx][ep_rank, slot])
                if _PLACEMENT[ep_rank][slot] == expert:
                    continue
                for name in _WEIGHTS:
                    loaded = layer.weight_slots[(slot, name)].view(torch.uint8)
                    assert torch.equal(loaded, _expected(layer_idx, expert, name)), (ep_rank, slot)


def test_ring_installs_the_staged_slots_without_loading_them(monkeypatch):
    load_balancer, _ = _ring_load_balancer(monkeypatch)
    load_balancer.install_recovery_placement([3], load_balancer.recovery_placements([3]))
    _, kwargs = load_balancer.load_balancer_impl.restore_placement.call_args
    assert kwargs == {"slots_loaded": True}


# Bound NVLinkOneSided layouts small enough that ring restores take several chunks of _WEIGHTS.
_TOKENS, _HIDDEN = 2, 128


def _layout(ep_size, top_k, eplb_stats_num_experts, low_precision_combine=False):
    return NVLinkOneSided._make_workspace_layout(
        ep_size,
        top_k,
        _TOKENS,
        _HIDDEN,
        torch.bfloat16,
        eplb_stats_num_experts,
        0,
        False,
        low_precision_combine,
    )


def _field(meta, name):
    return int(meta[int(getattr(internal.thop, f"MOE_A2A_{name}_INDEX"))])


def _kernel_writes(meta, source):
    """The byte ranges of a peer's row that the fence kernels with fixed slots write for ``source``.

    Per payload list a bound dispatch may send (bf16, fp8 or NVFP4 activations with block scales, then expert
    ids and router weights), moeA2ADispatchOp places each payload's ``[ep_size, capacity]`` rows in turn,
    cacheline aligned; the kernels write the source's rows, its receive count of either parity, its dispatch and
    combine completion flags and its EPLB statistics row. A fence combine only reads peers' combine input.
    The payload rows only check that source_write_ranges is a superset of this reading of the op;
    test_moe_a2a_late_writes.py checks it against what a killed rank's kernels actually write.
    """
    ep_size, capacity, top_k, experts = (
        _field(meta, name)
        for name in ("EP_SIZE", "MAX_NUM_TOKENS", "TOP_K", "EPLB_STATS_NUM_EXPERTS")
    )
    payload_end = _field(meta, "DISPATCH_PAYLOAD_OFFSET") + _field(meta, "DISPATCH_PAYLOAD_SIZE")
    writes = []
    for activations in ([_HIDDEN * 2], [_HIDDEN], [_HIDDEN // 2, _HIDDEN // 16]):
        offset = _field(meta, "DISPATCH_PAYLOAD_OFFSET")
        for width in activations + [4 * top_k, 4 * top_k]:
            writes.append(
                (offset + source * capacity * width, offset + (source + 1) * capacity * width)
            )
            offset = (offset + ep_size * capacity * width + 127) // 128 * 128
        assert offset <= payload_end, "the layout cannot hold these payloads"
    recv = _field(meta, "RECV_COUNTERS_OFFSET")
    words = [recv + 4 * source, recv + 4 * (ep_size + source)] + [
        _field(meta, f"{name}_OFFSET") + 4 * source
        for name in ("DISPATCH_COMPLETION_FLAGS", "COMBINE_COMPLETION_FLAGS")
    ]
    stats = _field(meta, "EPLB_GATHERED_STATS_OFFSET") + 4 * source * experts
    return writes + [(word, word + 4) for word in words] + [(stats, stats + 4 * experts)]


@pytest.mark.parametrize(
    "ep_size, top_k, eplb_stats_num_experts", [(1, 1, None), (4, 2, 8), (8, 8, None), (72, 8, 256)]
)
def test_source_write_ranges_cover_every_write_of_the_source(
    ep_size, top_k, eplb_stats_num_experts
):
    meta = _layout(ep_size, top_k, eplb_stats_num_experts)
    for source in range(ep_size):
        ranges = NVLinkOneSided.source_write_ranges(meta, source)
        for begin, end in _kernel_writes(meta, source):
            assert begin == end or any(b <= begin and end <= e for b, e in ranges), (
                source,
                begin,
                end,
            )


# Two ghost slots per rank, so the survivors of a two-rank loss can hold every expert.
_GHOSTS_PLACEMENT = [[0, 1, -1, -1], [2, 3, -1, -1], [4, 5, -1, -1], [6, 7, -1, -1]]
_GHOSTS_ASSIGNMENTS = [(2 * rank + slot) % _EXPERTS for rank in range(4) for slot in range(4)]


@pytest.mark.parametrize("dead", [[0], [3], [0, 3], [1, 3], [0, 2]])
@pytest.mark.parametrize(
    "top_k, eplb_stats_num_experts, low_precision_combine",
    [(1, None, False), (2, _EXPERTS, False), (8, _EXPERTS, True)],
)
def test_the_copy_plan_never_stages_where_a_dead_rank_writes(
    monkeypatch, dead, top_k, eplb_stats_num_experts, low_precision_combine
):
    meta = _layout(len(_HOSTS), top_k, eplb_stats_num_experts, low_precision_combine)
    row_bytes = _field(meta, "WORKSPACE_SIZE")
    write_ranges = functools.partial(NVLinkOneSided.source_write_ranges, meta)
    plans = []
    for ep_rank in range(len(_HOSTS)):
        load_balancer, _ = _ring_load_balancer(
            monkeypatch,
            ep_rank,
            layers=2,
            placement=_GHOSTS_PLACEMENT,
            assignments=_GHOSTS_ASSIGNMENTS,
        )
        placements = load_balancer.recovery_placements(dead)
        plans.append(load_balancer.restore_copy_plan(dead, placements, row_bytes, write_ranges))
    plan = plans[0]
    assert all((other == plan).all() for other in plans)
    # More than one chunk per target, so the halves alternate.
    assert plan[:, 0].max() >= 1
    _assert_pieces_load_every_weight(plan, 2 * 2 * len(dead))
    dead_writes = [
        (begin, end) for rank in dead for begin, end in _kernel_writes(meta, rank) if begin < end
    ]
    _assert_staged_apart(plan, row_bytes, lambda target: dead_writes)


def _assert_pieces_load_every_weight(plan, slots):
    """Each weight of each of ``slots`` loaded slots is staged whole, in consecutive pieces."""
    pieces = {}
    for row in plan.tolist():
        pieces.setdefault((row[1], row[2], row[3], row[5]), []).append((row[8], row[9]))
    assert len(pieces) == 2 * slots
    for (_, _, _, weight), parts in pieces.items():
        starts = [0] + list(np.cumsum([size for _, size in sorted(parts)]))
        assert [start for start, _ in sorted(parts)] == starts[:-1]
        assert starts[-1] == [300, 200][weight]


def _assert_staged_apart(plan, row_bytes, taken):
    """Every row of ``plan`` lies in the row and outside ``taken(target)``; rows of a target's chunk and of the
    chunk before it (still being read) never share a byte."""
    rows = plan.tolist()
    for i, (chunk, _, target, _, _, _, _, offset, _, size) in enumerate(rows):
        assert 0 <= offset and offset + size <= row_bytes
        assert not any(offset < end and begin < offset + size for begin, end in taken(target))
        for j, other in enumerate(rows):
            if j != i and other[2] == target and other[0] in (chunk - 1, chunk):
                assert other[7] + other[9] <= offset or offset + size <= other[7]


def test_a_replan_keeps_off_the_copies_a_rank_lost_while_staging_was_given(monkeypatch):
    meta = _layout(len(_HOSTS), 1, None)
    row_bytes = _field(meta, "WORKSPACE_SIZE")
    write_ranges = functools.partial(NVLinkOneSided.source_write_ranges, meta)
    load_balancer, _ = _ring_load_balancer(
        monkeypatch, layers=2, placement=_GHOSTS_PLACEMENT, assignments=_GHOSTS_ASSIGNMENTS
    )
    first = load_balancer.restore_copy_plan(
        [3], load_balancer.recovery_placements([3]), row_bytes, write_ranges
    )
    # Rank 2 sourced copies of the first plan, then is lost while they may still be in flight.
    late = first[first[:, 6] == 2].tolist()
    assert late
    placements = load_balancer.recovery_placements([2, 3])

    def late_copies(target):
        return [(row[7], row[7] + row[9]) for row in late if row[2] == target]

    plan = load_balancer.restore_copy_plan([2, 3], placements, row_bytes, write_ranges, [first])
    _assert_pieces_load_every_weight(plan, 2 * 2 * 2)
    _assert_staged_apart(plan, row_bytes, late_copies)
    # Without the first plan, the new plan would stage over those copies.
    unaware = load_balancer.restore_copy_plan([2, 3], placements, row_bytes, write_ranges)
    with pytest.raises(AssertionError):
        _assert_staged_apart(unaware, row_bytes, late_copies)


# DeepSeek-R1 at EP32 on eight nodes of four GPUs: 256 experts on 8 + 2 static EPLB slots per rank.
_R1_EP, _R1_GPUS_PER_NODE, _R1_EXPERTS, _R1_SLOTS = 32, 4, 256, 10
_R1_HOSTS = [f"node{rank // _R1_GPUS_PER_NODE}" for rank in range(_R1_EP)]


def _r1_node(node):
    return list(range(node * _R1_GPUS_PER_NODE, (node + 1) * _R1_GPUS_PER_NODE))


def _r1_static_load_balancer(monkeypatch):
    """A static R1 MoeLoadBalancer with ring host copies, showing the placement after FT emptied the replicas."""
    assignments = [
        (rank * _R1_EXPERTS // _R1_EP + slot) % _R1_EXPERTS
        for rank in range(_R1_EP)
        for slot in range(_R1_SLOTS)
    ]
    placement = np.array(assignments, dtype=np.int32).reshape(_R1_EP, _R1_SLOTS)
    placement[:, 8:] = -1
    monkeypatch.setattr(f"{_MLB}.peer_loss_recovery_enabled", lambda: True)
    monkeypatch.setattr(
        MoeLoadBalancer, "_gather_copy_nodes", lambda self: ring_copy_nodes(_R1_HOSTS)
    )
    with (
        patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"),
        patch(f"{_MLB}.HostMoeTensorSharer"),
    ):
        load_balancer = MoeLoadBalancer(0, _R1_EP, 0)
        native = Mock(
            get_layer_id=Mock(return_value=0),
            get_old_rank_expert_ids=Mock(return_value=placement.tolist()),
        )
        load_balancer.load_balancer_impl.add_layer.return_value = native
        load_balancer.add_layer(_R1_EXPERTS, 8, _R1_SLOTS).set_initial_weight_assignments(
            assignments
        )
    return load_balancer, native


def _assert_serves_every_expert_once(placement, dead):
    assert (placement[dead] == -1).all()
    assert (np.bincount(placement[placement >= 0], minlength=_R1_EXPERTS) == 1).all()
    assert (placement >= 0).sum(axis=1).max() <= _R1_SLOTS


@pytest.mark.parametrize("node", range(1, 8))
def test_r1_static_restores_a_lost_node_and_then_a_rank(monkeypatch, node):
    load_balancer, native = _r1_static_load_balancer(monkeypatch)
    dead = _r1_node(node)
    (placement,) = load_balancer.recovery_placements(dead)
    # 32 lost experts, one copy each, into the 56 empty slots of the 28 survivors.
    _assert_serves_every_expert_once(placement, dead)
    survivors = np.delete(np.arange(_R1_EP), dead)
    assert (placement[survivors] != load_balancer.current_placements()[0][survivors]).sum() == 32
    native.get_old_rank_expert_ids.return_value = placement.tolist()
    # Then any rank but one of the next node, whose home experts' copies were on the lost node.
    unprotected = _r1_node((node + 1) % 8)
    for rank in range(1, _R1_EP):
        if rank in dead:
            continue
        lost = dead + [rank]
        assert load_balancer.covers_loss_of(lost) == (rank not in unprotected)
        if rank not in unprotected:
            _assert_serves_every_expert_once(load_balancer.recovery_placements(lost)[0], lost)


def test_r1_static_two_nodes_leave_too_few_slots(monkeypatch):
    load_balancer, _ = _r1_static_load_balancer(monkeypatch)
    # 24 survivors hold 240 slots for 256 experts.
    with pytest.raises(ValueError, match="No surviving replica"):
        load_balancer.recovery_placements(_r1_node(2) + _r1_node(5))


def test_r1_ring_a_node_loss_leaves_the_next_node_without_host_copies(monkeypatch):
    load_balancer, native = _r1_static_load_balancer(monkeypatch)
    lost_node = _r1_node(1)
    native.get_old_rank_expert_ids.return_value = load_balancer.recovery_placements(lost_node)[
        0
    ].tolist()
    # Rank 8 (node 2) kept its home experts 64..71 resident; their only host copies were on node 1.
    assert not load_balancer.covers_loss_of(lost_node + [8])
    with pytest.raises(ValueError, match=r"\{0: \[64, 65, 66, 67, 68, 69, 70, 71\]\}"):
        load_balancer.recovery_placements(lost_node + [8])

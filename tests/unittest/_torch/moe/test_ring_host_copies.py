# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EPLB host copies of a subset of the experts, as WideEP fault tolerance keeps them in a ring of nodes."""

import threading
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    HostMoeTensorSharer,
    MoeLoadBalancer,
    ring_copy_nodes,
)

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
    finally:
        for sharer in sharers:
            sharer.pre_shutdown_cleanup()
        for sharer in sharers:
            sharer.post_shutdown_cleanup()


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


def test_ring_keeps_each_node_s_copies_on_the_node_before_it():
    hosts = ["b", "a", "b", "a", "c", "c"]
    # Nodes by first rank: b (0, 2), a (1, 3), c (4, 5); b's copies live on c, a's on b, c's on a.
    assert ring_copy_nodes(hosts) == [(4, 5), (0, 2), (4, 5), (0, 2), (1, 3), (1, 3)]
    assert ring_copy_nodes(["a", "b"]) == [(1,), (0,)]


def test_a_single_node_keeps_its_own_copies():
    assert ring_copy_nodes(["a"] * 4) == [(0, 1, 2, 3)] * 4


def _ring_load_balancer(monkeypatch, ep_rank=0, layers=1, layer_updates_per_iter=0):
    """A MoeLoadBalancer with WideEP peer-loss recovery over native layers showing _PLACEMENT; with a static
    placement (no ``layer_updates_per_iter``) it keeps ring host copies."""
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
                get_old_rank_expert_ids=Mock(return_value=_PLACEMENT),
            )
            for i in range(layers)
        ]
        for _ in range(layers):
            layer = load_balancer.add_layer(_EXPERTS, 2, 3)
            layer.set_initial_weight_assignments(_ASSIGNMENTS)
            for slot in range(3):
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
        plan = load_balancer.restore_copy_plan([3], placements, 2048)
        plans.append((plan, load_balancer.placement_digest([3], placements, plan)))
    assert all((plan == plans[0][0]).all() and digest == plans[0][1] for plan, digest in plans)
    # Rank 3's experts 6 and 7 go to the ghost slots of ranks 0 and 1, copied by node a's ranks 0 and 2 in turn.
    # Columns: chunk, layer, target, slot, expert, weight, source, offset, bytes.
    assert plans[0][0].tolist() == [
        [0, 0, 0, 2, 6, 0, 0, 0, 300],
        [0, 0, 0, 2, 6, 1, 0, 512, 200],
        [0, 0, 1, 2, 7, 0, 2, 0, 300],
        [0, 0, 1, 2, 7, 1, 2, 512, 200],
    ]
    # Another plan digests differently.
    assert load_balancer.placement_digest([3], placements, plan[:, ::-1].copy()) != plans[0][1]


def test_the_copy_plan_alternates_workspace_halves_between_chunks(monkeypatch):
    load_balancer, _ = _ring_load_balancer(monkeypatch, layers=2)
    placements = load_balancer.recovery_placements([3])
    # A 1024-byte workspace: chunks of 512 bytes, the second at offset 512, the third back at 0.
    plan = load_balancer.restore_copy_plan([3], placements, 1024)
    target_0 = plan[plan[:, 2] == 0]
    assert target_0[:, [0, 1, 5, 7]].tolist() == [
        [0, 0, 0, 0],
        [1, 0, 1, 512],
        [2, 1, 0, 0],
        [3, 1, 1, 512],
    ]
    assert (plan[1:, 0] >= plan[:-1, 0]).all()
    with pytest.raises(ValueError, match="300-byte expert weight of MoE layer 0 does not fit"):
        load_balancer.restore_copy_plan([3], placements, 512)


def test_the_copy_plan_refuses_to_load_a_routed_slot(monkeypatch):
    load_balancer, _ = _ring_load_balancer(monkeypatch)
    (placement,) = load_balancer.recovery_placements([3])
    # Rank 0's slot 1 still routes expert 1.
    placement[0, 1] = 6
    with pytest.raises(ValueError, match="would overwrite a routed one"):
        load_balancer.restore_copy_plan([3], [placement], 2048)


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
    plan = ranks[0][0].restore_copy_plan(dead, ranks[0][1], workspace.size(1))
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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EPLB placement after a WideEP rank loss: resident replicas and experts restored from host weights."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    MoeLoadBalancer,
    restore_lost_experts,
)
from tensorrt_llm.executor.recovery.config import FaultToleranceConfig

pytestmark = pytest.mark.cpu_only

WORLD_SIZE = 4
FAILED_RANK = 2
EXPERTS = 4
SLOTS_PER_RANK = 2

# Every expert has two replicas.
_REDUNDANT = [[0, 1], [2, 3], [1, 0], [3, 2]]
# Experts 2 and 3 live only on rank 1.
_SOLE_REPLICAS = [[0, 1], [2, 3], [0, 1], [0, 1]]


def _load_balancer(*placements, host_weights=False, experts=EXPERTS):
    """A MoeLoadBalancer over native layers with the given slot tables (with host weights: online EPLB)."""
    with patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"):
        load_balancer = MoeLoadBalancer(0, WORLD_SIZE, 0)
    native_layers = [
        Mock(
            get_layer_id=Mock(return_value=layer_idx),
            get_old_rank_expert_ids=Mock(return_value=placement),
        )
        for layer_idx, placement in enumerate(placements)
    ]
    load_balancer.load_balancer_impl.add_layer.side_effect = native_layers
    for _ in placements:
        load_balancer.add_layer(experts, 2, SLOTS_PER_RANK).host_copies = host_weights
    return load_balancer


@pytest.mark.parametrize("recovery", [False, True])
def test_static_routing_keeps_host_copies_only_with_peer_loss_recovery(recovery):
    with (
        patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"),
        patch(
            "tensorrt_llm._torch.moe.fused_moe.moe_load_balancer.peer_loss_recovery_enabled",
            return_value=recovery,
        ),
        patch("tensorrt_llm._torch.moe.fused_moe.moe_load_balancer.HostMoeTensorSharer"),
        patch.object(MoeLoadBalancer, "_gather_copy_nodes", return_value=[(1, 3), (0, 2)] * 2),
    ):
        load_balancer = MoeLoadBalancer(0, WORLD_SIZE, 0)
        layer = load_balancer.add_layer(EXPERTS, 2, SLOTS_PER_RANK)
    # Routing stays static either way; only the host copies (ring ones) and loadable slots follow recovery.
    assert (load_balancer.copy_nodes is not None) == recovery
    assert layer.is_static_routing()
    assert layer.need_load_shared_weights() == recovery
    assert (layer.host_tensor_sharer is not None) == recovery


def _digest(load_balancer, dead):
    return load_balancer.placement_digest(dead, load_balancer.recovery_placements(dead))


def test_placement_digest_is_deterministic_and_mask_sensitive():
    load_balancer = _load_balancer(_REDUNDANT, _REDUNDANT)
    digest = _digest(load_balancer, {FAILED_RANK})
    assert digest == _digest(_load_balancer(_REDUNDANT, _REDUNDANT), [FAILED_RANK])
    assert digest != _digest(load_balancer, {1})
    assert digest != _digest(load_balancer, set())


def test_without_host_weights_a_loss_is_covered_only_if_every_expert_keeps_a_replica():
    load_balancer = _load_balancer(_REDUNDANT, _SOLE_REPLICAS)
    assert [load_balancer.covers_loss_of([rank]) for rank in range(WORLD_SIZE)] == [
        True,
        False,
        True,
        True,
    ]
    with pytest.raises(ValueError, match=r"\{1: \[2, 3\]\}"):
        load_balancer.recovery_placements({1})
    with pytest.raises(ValueError, match="out of range"):
        load_balancer.recovery_placements({WORLD_SIZE})


def test_restore_evicts_the_most_replicated_expert_spread_over_ranks():
    placement = np.array([[0, 1], [-1, -1], [0, 1], [0, 1]], dtype=np.int32)
    assert restore_lost_experts(placement, [2, 3], [1]) == []
    # Expert 2 takes rank 0's first replica of expert 0 (three copies); expert 3 then takes a copy of expert 1
    # (still three) on rank 2, the lowest rank that has not taken a restored expert yet.
    assert placement.tolist() == [[2, 1], [-1, -1], [0, 3], [0, 1]]


def test_restore_never_evicts_a_last_replica():
    placement = np.array([[0], [1], [-1], [3]], dtype=np.int32)
    assert restore_lost_experts(placement, [2], [2]) == [2]
    assert placement.tolist() == [[0], [1], [-1], [3]]


def test_restore_fills_empty_slots_before_evicting_replicas():
    # Rank 2 is dead; ranks 0 and 3 each keep an empty slot, rank 3 also a replica of expert 0.
    placement = np.array([[0, -1], [1, 0], [-1, -1], [3, -1]], dtype=np.int32)
    assert restore_lost_experts(placement, [2], [2]) == []
    assert placement.tolist() == [[0, 2], [1, 0], [-1, -1], [3, -1]]


def test_empty_replica_slots_keeps_one_routed_slot_per_expert():
    load_balancer = _load_balancer(_REDUNDANT, host_weights=True)
    assert load_balancer.empty_replica_slots() == 4
    # Slot 0 of every rank is kept (each expert's lowest local slot); every second copy is emptied.
    load_balancer.load_balancer_impl.restore_placement.assert_called_once_with(
        [], [[[0, -1], [2, -1], [1, -1], [3, -1]]], slots_loaded=True
    )


def test_survivors_with_different_placements_agree_from_a_common_base():
    # Rank 3's last online-EPLB update swapped two slots on one survivor only.
    diverged = [[0, 1], [2, 3], [1, 0], [2, 3]]
    first = _load_balancer(_REDUNDANT, host_weights=True)
    second = _load_balancer(diverged, host_weights=True)
    base = first.current_placements()

    placements = [lb.recovery_placements([FAILED_RANK], base) for lb in (first, second)]

    assert [p.tolist() for p in placements[0]] == [p.tolist() for p in placements[1]]
    assert MoeLoadBalancer.placement_digest(
        [FAILED_RANK], placements[0]
    ) == MoeLoadBalancer.placement_digest([FAILED_RANK], placements[1])
    # Without the base they would disagree.
    assert second.recovery_placements([FAILED_RANK])[0].tolist() != placements[0][0].tolist()


def test_recovery_placements_reject_a_base_with_another_layer_count():
    load_balancer = _load_balancer(_REDUNDANT, host_weights=True)
    with pytest.raises(ValueError, match="layers"):
        load_balancer.recovery_placements([FAILED_RANK], base=[])


@pytest.mark.parametrize("ep_rank, ghost", [(0, [1]), (2, [1])])
def test_ghost_mode_unbacks_the_slots_that_empty_mode_empties(monkeypatch, ep_rank, ghost):
    """Before the weights move to their final storage, each layer learns its slots that will be empty."""
    monkeypatch.setattr(
        "tensorrt_llm._torch.moe.fused_moe.moe_load_balancer.fault_tolerance_config",
        lambda: FaultToleranceConfig(extra_slots="ghost"),
    )
    load_balancer = _load_balancer(_REDUNDANT, host_weights=True)
    load_balancer.ep_rank = ep_rank
    layer = load_balancer.single_layer_load_balancers[0]
    layer.initial_weight_assignments = [expert for rank in _REDUNDANT for expert in rank]
    with patch(
        "tensorrt_llm._torch.moe.fused_moe.moe_load_balancer.peer_loss_recovery_enabled",
        return_value=True,
    ):
        load_balancer.register_weight_slots_after_to_cuda()
    assert layer.ghost_slots == ghost
    # Static placements keep ghost slots in device memory: restores copy on the GPU.
    load_balancer.load_balancer_impl.set_use_gpu_memcpy.assert_called_once_with(True)


@pytest.mark.parametrize("layer_updates_per_iter, mode", [(0, "ghost"), (1, "replicas")])
def test_online_eplb_keeps_replicas_in_the_extra_slots(layer_updates_per_iter, mode):
    with patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"):
        load_balancer = MoeLoadBalancer(0, WORLD_SIZE, layer_updates_per_iter)
    with patch(
        "tensorrt_llm._torch.moe.fused_moe.moe_load_balancer.peer_loss_recovery_enabled",
        return_value=True,
    ):
        assert load_balancer.extra_slots_mode() == mode
    assert load_balancer.extra_slots_mode() is None


_MLB = "tensorrt_llm._torch.moe.fused_moe.moe_load_balancer"


def _online_load_balancer(
    monkeypatch,
    placement=None,
    *,
    largest_loss,
    experts,
    eplb_slots,
    extra_slots="ghost",
    host_accessible=True,
):
    """An online-EPLB MoeLoadBalancer with WideEP peer-loss recovery over one native layer showing ``placement``.

    ``largest_loss``: the most ranks one failure domain holds; ``host_accessible``: whether ghost slot memory can be.
    """
    monkeypatch.setattr(f"{_MLB}.peer_loss_recovery_enabled", lambda: True)
    monkeypatch.setattr(f"{_MLB}._ghost_slot_memory_host_accessible", lambda: host_accessible)
    monkeypatch.setattr(
        f"{_MLB}.fault_tolerance_config", lambda: FaultToleranceConfig(extra_slots=extra_slots)
    )
    monkeypatch.setattr(MoeLoadBalancer, "_gather_largest_loss", lambda self: largest_loss)
    with (
        patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"),
        patch(f"{_MLB}.HostMoeTensorSharer"),
        patch("torch.cuda.Event"),
    ):
        load_balancer = MoeLoadBalancer(0, WORLD_SIZE, 1)
        load_balancer.load_balancer_impl.add_layer.return_value = Mock(
            get_layer_id=Mock(return_value=0),
            get_old_rank_expert_ids=Mock(return_value=placement),
        )
        load_balancer.add_layer(experts, 2, eplb_slots, aux_stream=Mock())
    return load_balancer


@pytest.mark.parametrize(
    "largest_loss, eplb_slots, ghost_slots, extra_slots, host_accessible, mode",
    [
        # A node of two: two survivors need two slots each for four experts.
        (2, 1, 1, "ghost", True, "replicas+ghost"),
        (2, 1, 1, "empty", True, "replicas+empty"),
        # By default ghost where ghost slot memory can be host accessible, else empty.
        (2, 1, 1, None, True, "replicas+ghost"),
        (2, 1, 1, None, False, "replicas+empty"),
        # One rank: three survivors still need two slots each.
        (1, 1, 1, "ghost", True, "replicas+ghost"),
        # EPLB's slots already hold every expert without a node: nothing is appended.
        (2, 2, 0, "ghost", True, "replicas"),
    ],
)
def test_online_eplb_gets_ghost_slots_only_where_its_slots_fall_short(
    monkeypatch, largest_loss, eplb_slots, ghost_slots, extra_slots, host_accessible, mode
):
    load_balancer = _online_load_balancer(
        monkeypatch,
        largest_loss=largest_loss,
        experts=EXPERTS,
        eplb_slots=eplb_slots,
        extra_slots=extra_slots,
        host_accessible=host_accessible,
    )
    (layer,) = load_balancer.single_layer_load_balancers
    load_balancer.load_balancer_impl.add_layer.assert_called_once_with(
        EXPERTS, 2, eplb_slots + ghost_slots
    )
    assert (layer.slot_count_per_rank, layer.ghost_slot_count) == (
        eplb_slots + ghost_slots,
        ghost_slots,
    )
    assert load_balancer.extra_slots_mode() == mode


def test_online_eplb_without_peer_loss_recovery_gets_no_ghost_slots(monkeypatch):
    def gather(self):
        raise AssertionError("gathered hosts without peer-loss recovery")

    monkeypatch.setattr(MoeLoadBalancer, "_gather_largest_loss", gather)
    with (
        patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"),
        patch(f"{_MLB}.HostMoeTensorSharer"),
        patch("torch.cuda.Event"),
    ):
        load_balancer = MoeLoadBalancer(0, WORLD_SIZE, 1)
        layer = load_balancer.add_layer(EXPERTS, 2, 1, aux_stream=Mock())
    load_balancer.load_balancer_impl.add_layer.assert_called_once_with(EXPERTS, 2, 1)
    assert layer.ghost_slot_count == 0
    assignments = [0, 1, 2, 3]
    assert layer.with_ghost_slots(assignments) is assignments
    layer.set_initial_weight_assignments(assignments)
    layer.single_layer_load_balancer_impl.set_initial_weight_assignments.assert_called_once_with(
        assignments
    )
    assert load_balancer.extra_slots_mode() is None


@pytest.mark.parametrize("host_accessible", [True, False])
def test_ghost_slots_start_empty_and_unbacked(monkeypatch, host_accessible):
    load_balancer = _online_load_balancer(
        monkeypatch, largest_loss=2, experts=EXPERTS, eplb_slots=1, host_accessible=host_accessible
    )
    (layer,) = load_balancer.single_layer_load_balancers
    impl = layer.single_layer_load_balancer_impl

    # Each ghost slot loads its rank's first expert into the weight tensor...
    assignments = layer.with_ghost_slots([0, 1, 2, 3])
    assert assignments == [0, 0, 1, 1, 2, 2, 3, 3]
    # ...but routing and EPLB's updates start with it empty, also when the weights are set again after loading.
    for _ in range(2):
        layer.set_initial_weight_assignments(assignments)
        impl.set_initial_weight_assignments.assert_called_with([0, -1, 1, -1, 2, -1, 3, -1])
    load_balancer.register_weight_slots_after_to_cuda()
    assert layer.ghost_slots == [1]
    # Online EPLB's updates load host-accessible ghost slots with CPU copies, others with GPU copies.
    set_use_gpu_memcpy = load_balancer.load_balancer_impl.set_use_gpu_memcpy
    if host_accessible:
        set_use_gpu_memcpy.assert_not_called()
    else:
        set_use_gpu_memcpy.assert_called_once_with(True)
    weights = Mock()
    with patch(f"{_MLB}.torch") as torch:
        layer.make_tensor_host_accessible(weights)
    torch.ops.trtllm.moe_ghost_slot_storage.assert_called_once_with(weights, [1], host_accessible)


# Six experts on two EPLB slots per rank (experts 0 and 2 twice) and one ghost slot per rank.
_MIXED = [[0, 1, -1], [2, 3, -1], [4, 5, -1], [0, 2, -1]]


def test_mixed_mode_restores_into_ghost_slots_spread_over_ranks(monkeypatch):
    load_balancer = _online_load_balancer(
        monkeypatch, _MIXED, largest_loss=2, experts=6, eplb_slots=2
    )
    assert load_balancer.extra_slots_mode() == "replicas+ghost"
    # Losing rank 2: its experts take ghost slots on two ranks, one each, and every replica stays.
    (placement,) = load_balancer.recovery_placements([FAILED_RANK])
    assert placement.tolist() == [[0, 1, 4], [2, 3, 5], [-1, -1, -1], [0, 2, -1]]
    # Losing the node of ranks 2 and 3 leaves no replica: the ghost slots take the lost experts.
    assert load_balancer.covers_loss_of(_NODE)
    (placement,) = load_balancer.recovery_placements(_NODE)
    assert placement.tolist() == [[0, 1, 4], [2, 3, 5], [-1, -1, -1], [-1, -1, -1]]


def test_appended_empty_slots_take_lost_experts_first(monkeypatch):
    load_balancer = _online_load_balancer(
        monkeypatch, _MIXED, largest_loss=2, experts=6, eplb_slots=2, extra_slots="empty"
    )
    # Empty slots hold their memory: filling them costs no map and keeps every replica.
    (placement,) = load_balancer.recovery_placements([FAILED_RANK])
    assert placement.tolist() == [[0, 1, 4], [2, 3, 5], [-1, -1, -1], [0, 2, -1]]


def test_with_host_weights_lost_experts_are_restored():
    load_balancer = _load_balancer(_SOLE_REPLICAS, host_weights=True)
    assert all(load_balancer.covers_loss_of([rank]) for rank in range(WORLD_SIZE))
    (placement,) = load_balancer.recovery_placements({1})
    assert placement.tolist() == [[2, 1], [-1, -1], [0, 3], [0, 1]]


# With two GPUs per node, node 1 (ranks 2 and 3) holds the only replicas of experts 2 and 3.
_NODE = (2, 3)
_NODE_EXPERTS = [[0, 1], [0, 1], [2, 3], [2, 3]]


def test_with_host_weights_a_node_s_experts_are_restored_into_the_survivors():
    load_balancer = _load_balancer(_NODE_EXPERTS, host_weights=True)
    assert load_balancer.covers_loss_of(_NODE)
    (placement,) = load_balancer.recovery_placements(set(_NODE))
    # Each lost expert evicts a replica of expert 0 or 1, one per surviving rank.
    assert placement.tolist() == [[2, 1], [0, 3], [-1, -1], [-1, -1]]
    assert not _load_balancer(_NODE_EXPERTS).covers_loss_of(_NODE)


def test_a_node_loss_is_not_covered_when_the_survivors_have_too_few_slots():
    # Five experts, but the two surviving ranks hold four slots.
    load_balancer = _load_balancer([[0, 1], [2, 3], [4, 0], [1, 2]], host_weights=True, experts=5)
    assert load_balancer.covers_loss_of([2]) and load_balancer.covers_loss_of([3])
    assert not load_balancer.covers_loss_of(_NODE)
    with pytest.raises(ValueError, match=r"\{0: \[4\]\}"):
        load_balancer.recovery_placements(_NODE)


def test_install_masks_only_when_no_expert_moves():
    load_balancer = _load_balancer(_SOLE_REPLICAS, host_weights=True)
    impl = load_balancer.load_balancer_impl

    load_balancer.install_recovery_placement([3], load_balancer.recovery_placements([3]))
    impl.reconfigure_mask_only.assert_called_once_with([3])
    impl.restore_placement.assert_not_called()

    load_balancer.install_recovery_placement([1], load_balancer.recovery_placements([1]))
    impl.restore_placement.assert_called_once_with(
        [1], [[[2, 1], [-1, -1], [0, 3], [0, 1]]], slots_loaded=False
    )

    load_balancer.in_iter = True
    with pytest.raises(RuntimeError, match="iteration is active"):
        load_balancer.install_recovery_placement([1], load_balancer.recovery_placements([1]))


# DeepSeek-R1 at EP32 on eight nodes of four GPUs: 256 experts on 9 online EPLB slots per rank (288).
_R1_EP, _R1_EXPERTS, _R1_EPLB_SLOTS = 32, 256, 9


@pytest.mark.parametrize("clumped", [False, True])
def test_r1_online_restores_every_node_or_rank_loss(monkeypatch, clumped):
    monkeypatch.setattr(f"{_MLB}.peer_loss_recovery_enabled", lambda: True)
    monkeypatch.setattr(f"{_MLB}.fault_tolerance_config", lambda: FaultToleranceConfig())
    monkeypatch.setattr(MoeLoadBalancer, "_gather_largest_loss", lambda self: 4)
    with (
        patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"),
        patch(f"{_MLB}.HostMoeTensorSharer"),
        patch("torch.cuda.Event"),
    ):
        load_balancer = MoeLoadBalancer(0, _R1_EP, 1)
        native = Mock(get_layer_id=Mock(return_value=0))
        load_balancer.load_balancer_impl.add_layer.return_value = native
        layer = load_balancer.add_layer(_R1_EXPERTS, 8, _R1_EPLB_SLOTS, aux_stream=Mock())
    # 28 survivors of a node need ceil(256 / 28) = 10 slots each: one ghost slot per rank.
    assert layer.ghost_slot_count == 1
    rng = np.random.default_rng(0)
    for _ in range(100):
        # Online EPLB's 32 replicas go to the most loaded experts, spread at random or clumped on few ranks.
        load = rng.pareto(1.5, _R1_EXPERTS) + 0.01
        copies = np.ones(_R1_EXPERTS, dtype=np.int64)
        for _ in range(_R1_EP * _R1_EPLB_SLOTS - _R1_EXPERTS):
            copies[np.argmax(load / copies)] += 1
        slots = np.repeat(np.arange(_R1_EXPERTS), copies)
        if not clumped:
            rng.shuffle(slots)
        placement = np.full((_R1_EP, _R1_EPLB_SLOTS + 1), -1, dtype=np.int32)
        placement[:, :_R1_EPLB_SLOTS] = slots.reshape(_R1_EP, _R1_EPLB_SLOTS)
        native.get_old_rank_expert_ids.return_value = placement.tolist()
        node = int(rng.integers(1, 8))
        for dead in (list(range(4 * node, 4 * node + 4)), [int(rng.integers(1, _R1_EP))]):
            (restored,) = load_balancer.recovery_placements(dead)
            assert (restored[dead] == -1).all()
            # Every expert keeps at least one copy; only lost experts and evicted replicas moved.
            assert (np.bincount(restored[restored >= 0], minlength=_R1_EXPERTS) >= 1).all()
            survivors = np.delete(np.arange(_R1_EP), dead)
            kept = restored[survivors] == placement[survivors]
            evicted = placement[survivors][~kept]
            assert ((evicted < 0) | (copies[np.maximum(evicted, 0)] >= 2)).all()

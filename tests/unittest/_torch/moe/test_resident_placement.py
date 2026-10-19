# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EPLB placement after a WideEP rank loss: resident replicas and experts restored from host weights."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    EXTRA_SLOTS_ENV,
    MoeLoadBalancer,
    extra_slots_mode,
    restore_lost_experts,
)

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
    ):
        load_balancer = MoeLoadBalancer(0, WORLD_SIZE, 0)
        layer = load_balancer.add_layer(EXPERTS, 2, SLOTS_PER_RANK)
    # Routing stays static either way; only the host copies (and loadable slots) follow recovery.
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
        [], [[[0, -1], [2, -1], [1, -1], [3, -1]]]
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


def test_extra_slots_mode_rejects_an_unknown_mode(monkeypatch):
    monkeypatch.setenv(EXTRA_SLOTS_ENV, "spare")
    with pytest.raises(ValueError, match=EXTRA_SLOTS_ENV):
        extra_slots_mode()


@pytest.mark.parametrize("ep_rank, borrowed", [(0, [1]), (2, [1])])
def test_borrowed_mode_borrows_the_slots_that_empty_mode_empties(monkeypatch, ep_rank, borrowed):
    """Before the weights move to their final storage, each layer learns its slots that will be empty."""
    monkeypatch.setenv(EXTRA_SLOTS_ENV, "borrowed")
    load_balancer = _load_balancer(_REDUNDANT, host_weights=True)
    load_balancer.ep_rank = ep_rank
    layer = load_balancer.single_layer_load_balancers[0]
    layer.initial_weight_assignments = [expert for rank in _REDUNDANT for expert in rank]
    with patch(
        "tensorrt_llm._torch.moe.fused_moe.moe_load_balancer.peer_loss_recovery_enabled",
        return_value=True,
    ):
        load_balancer.register_weight_slots_after_to_cuda()
    assert layer.borrowed_slots == borrowed
    # Borrowed slots are not host accessible: restores copy on the GPU.
    load_balancer.load_balancer_impl.set_use_gpu_memcpy.assert_called_once_with(True)


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
    impl.restore_placement.assert_called_once_with([1], [[[2, 1], [-1, -1], [0, 3], [0, 1]]])

    load_balancer.in_iter = True
    with pytest.raises(RuntimeError, match="iteration is active"):
        load_balancer.install_recovery_placement([1], load_balancer.recovery_placements([1]))

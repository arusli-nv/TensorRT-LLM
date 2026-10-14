# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resident-replica EPLB placement after a WideEP rank loss."""

from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import MoeExecutionState
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import MoeLoadBalancer

pytestmark = pytest.mark.cpu_only

WORLD_SIZE = 4
FAILED_RANK = 2
EXPERTS = 4
SLOTS_PER_RANK = 2

# Every expert has two replicas.
_REDUNDANT = [[0, 1], [2, 3], [1, 0], [3, 2]]
# Experts 2 and 3 live only on rank 1.
_SOLE_REPLICAS = [[0, 1], [2, 3], [0, 1], [0, 1]]


def _load_balancer(*placements):
    """A static-routing MoeLoadBalancer over native layers with the given slot tables."""
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
        load_balancer.add_layer(EXPERTS, 2, SLOTS_PER_RANK)
    return load_balancer


def test_masked_placement_digest_is_deterministic_and_mask_sensitive():
    load_balancer = _load_balancer(_REDUNDANT, _REDUNDANT)
    digest = load_balancer.masked_placement_digest({FAILED_RANK})
    assert digest == _load_balancer(_REDUNDANT, _REDUNDANT).masked_placement_digest([FAILED_RANK])
    assert digest != load_balancer.masked_placement_digest({1})
    assert digest != load_balancer.masked_placement_digest(set())


def test_a_loss_is_covered_only_if_every_expert_keeps_a_replica():
    load_balancer = _load_balancer(_REDUNDANT, _SOLE_REPLICAS)
    assert [load_balancer.covers_loss_of(rank) for rank in range(WORLD_SIZE)] == [
        True,
        False,
        True,
        True,
    ]
    with pytest.raises(ValueError, match=r"\{1: \[2, 3\]\}"):
        load_balancer.masked_placement_digest({1})
    with pytest.raises(ValueError, match="out of range"):
        load_balancer.masked_placement_digest({WORLD_SIZE})


def test_install_resident_placement_reroutes_before_masking():
    events = []
    load_balancer = Mock()
    load_balancer.reconfigure_mask_only.side_effect = lambda dead: events.append(("routed", dead))
    health = EPGroupHealth(WORLD_SIZE)
    state = Mock(health=health)
    state.install_mask.side_effect = lambda: events.append(("mask", health.get_mask_words()))

    MoeExecutionState.install_resident_placement(state, load_balancer, {FAILED_RANK})

    assert health.get_failed_ranks() == frozenset({FAILED_RANK})
    # The kernels skip routes to a masked rank, so routing must exclude it first.
    assert events == [("routed", [FAILED_RANK]), ("mask", (0b1011, 0, 0, 0))]
    state.rearm.assert_called_once()
    with pytest.raises(RuntimeError, match="already failed"):
        MoeExecutionState.install_resident_placement(state, load_balancer, {FAILED_RANK})

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
"""Resident-replica EPLB placement after a WideEP rank loss."""

from unittest.mock import Mock, PropertyMock, patch

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import MoeLoadBalancer
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

WORLD_SIZE = 4
FAILED_RANK = 2


class _FakeLayer:
    def __init__(self, layer_idx, placement, expert_count):
        self._layer_idx = layer_idx
        self.placement = placement
        self.expert_count = expert_count

    def get_layer_idx(self):
        return self._layer_idx

    def get_old_rank_expert_ids(self):
        return [list(row) for row in self.placement]

    def route(self, token_selected_experts):
        """Round-robin over each expert's live replicas, like the device routing."""
        slots = {}
        for rank, row in enumerate(self.placement):
            for local_slot, expert in enumerate(row):
                if expert >= 0:
                    slots.setdefault(expert, []).append(rank * len(row) + local_slot)
        flat = token_selected_experts.flatten().tolist()
        seen = {}
        routed = []
        for expert in flat:
            replicas = slots[expert]
            routed.append(replicas[seen.get(expert, 0) % len(replicas)])
            seen[expert] = seen.get(expert, 0) + 1
        return torch.tensor(routed, dtype=torch.int32).view_as(token_selected_experts)


def _load_balancer(placements, expert_count=4):
    load_balancer = MoeLoadBalancer.__new__(MoeLoadBalancer)
    load_balancer.is_shutdown = True
    load_balancer.ep_rank = 0
    load_balancer.ep_size = len(placements[0])
    load_balancer.in_iter = False
    load_balancer.single_layer_load_balancers = [
        _FakeLayer(idx, placement, expert_count) for idx, placement in enumerate(placements)
    ]
    load_balancer._layer_routing_shapes = [(2, len(placements[0][0]))] * len(placements)
    return load_balancer


# Two slots per rank over four ranks; every expert has two replicas.
_REDUNDANT = [[0, 1], [2, 3], [1, 0], [3, 2]]


@pytest.mark.cpu_only
def test_masked_placement_digest_is_deterministic_and_mask_sensitive():
    load_balancer = _load_balancer([_REDUNDANT, _REDUNDANT])
    digest = load_balancer.masked_placement_digest({FAILED_RANK})
    assert digest == _load_balancer([_REDUNDANT, _REDUNDANT]).masked_placement_digest([FAILED_RANK])
    assert digest != load_balancer.masked_placement_digest({1})
    assert digest != load_balancer.masked_placement_digest(set())


@pytest.mark.cpu_only
def test_masked_placement_digest_rejects_lost_expert_and_bad_rank():
    sole_replicas = [[0, 1], [2, 3], [0, 1], [0, 1]]
    load_balancer = _load_balancer([_REDUNDANT, sole_replicas])
    with pytest.raises(ValueError, match=r"layer 1 .*experts \[2, 3\]"):
        load_balancer.masked_placement_digest({1})
    with pytest.raises(ValueError, match="out of range"):
        load_balancer.masked_placement_digest({WORLD_SIZE})


@pytest.fixture
def no_cuda_device():
    with (
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.cuda.set_device"),
    ):
        yield


@pytest.mark.cpu_only
def test_install_resident_placement_commits_mask(no_cuda_device):
    load_balancer = _load_balancer([_REDUNDANT])
    digest = load_balancer.masked_placement_digest({FAILED_RANK})

    def reconfigure(dead_ranks):
        layer = load_balancer.single_layer_load_balancers[0]
        layer.placement = [[-1, -1] if r in dead_ranks else row for r, row in enumerate(_REDUNDANT)]

    load_balancer.reconfigure_mask_only = reconfigure
    events = []
    load_balancer.check_routing_excludes = lambda dead: events.append(("checked", set(dead)))
    health = EPGroupHealth(WORLD_SIZE)
    execution_state = Mock(health=health)
    execution_state.install_mask.side_effect = lambda: events.append(
        ("mask", health.get_mask_words())
    )
    engine = PyTorchModelEngine.__new__(PyTorchModelEngine)
    engine.moe_load_balancer = load_balancer

    with patch.object(
        PyTorchModelEngine,
        "moe_execution_state",
        new_callable=PropertyMock,
        return_value=execution_state,
    ):
        engine.install_resident_placement(frozenset({FAILED_RANK}), digest)

    assert health.get_failed_ranks() == frozenset({FAILED_RANK})
    # The kernels skip routes to a masked rank, so routing must exclude it first.
    assert events == [("checked", {FAILED_RANK}), ("mask", (0b1011, 0, 0, 0))]


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@requires_cuda
def test_check_routing_excludes_accepts_routing_around_departed_rank():
    load_balancer = _load_balancer([_REDUNDANT, _REDUNDANT])
    for layer in load_balancer.single_layer_load_balancers:
        layer.placement = [
            [-1, -1] if r == FAILED_RANK else row for r, row in enumerate(_REDUNDANT)
        ]
    load_balancer.check_routing_excludes({FAILED_RANK})


@requires_cuda
def test_check_routing_excludes_rejects_route_to_departed_rank():
    load_balancer = _load_balancer([_REDUNDANT, _REDUNDANT])
    layer = load_balancer.single_layer_load_balancers[1]
    layer.placement = [[-1, -1] if r == FAILED_RANK else row for r, row in enumerate(_REDUNDANT)]
    # Layer 0 keeps routing to the departed rank's replicas of experts 1 and 0.
    with pytest.raises(
        RuntimeError, match=r"layer 0 still routes experts \[0, 1\] to departed ranks \[2\]"
    ):
        load_balancer.check_routing_excludes({FAILED_RANK})


@pytest.mark.cpu_only
def test_install_resident_placement_rejects_divergent_placement(no_cuda_device):
    load_balancer = _load_balancer([_REDUNDANT])
    load_balancer.reconfigure_mask_only = Mock()
    engine = PyTorchModelEngine.__new__(PyTorchModelEngine)
    engine.moe_load_balancer = load_balancer
    with pytest.raises(RuntimeError, match="differs from the agreed placement"):
        engine.install_resident_placement(frozenset({FAILED_RANK}), "0" * 64)

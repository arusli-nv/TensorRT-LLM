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


def _load_balancer(placements, expert_count=4):
    load_balancer = MoeLoadBalancer.__new__(MoeLoadBalancer)
    load_balancer.is_shutdown = True
    load_balancer.ep_rank = 0
    load_balancer.ep_size = len(placements[0])
    load_balancer.in_iter = False
    load_balancer.single_layer_load_balancers = [
        _FakeLayer(idx, placement, expert_count) for idx, placement in enumerate(placements)
    ]
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
    events = []
    load_balancer.reconfigure_mask_only = lambda dead: events.append(("routed", set(dead)))
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
        engine.install_resident_placement(frozenset({FAILED_RANK}))

    assert health.get_failed_ranks() == frozenset({FAILED_RANK})
    # The kernels skip routes to a masked rank, so routing must exclude it first.
    assert events == [("routed", {FAILED_RANK}), ("mask", (0b1011, 0, 0, 0))]

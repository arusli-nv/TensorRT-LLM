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
"""Attention-DP routing and admission over the ranks still serving."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.request_utils import get_from_waiting_queue
from tensorrt_llm._torch.pyexecutor.sampler.sampler_common import SampleType
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import DefaultADPRouter, RankState

pytestmark = pytest.mark.cpu_only

WORLD_SIZE = 4
FAILED_RANK = 2
SURVIVORS = (0, 1, 3)


def _router(tp_rank=0, tp_size=WORLD_SIZE):
    dist = MagicMock()
    dist.tp_rank = tp_rank
    dist.tp_size = tp_size
    dist.has_cp_helix = False
    return DefaultADPRouter(dist=dist)


def _item(req_id, target=None, relax=True, num_tokens=4):
    item = MagicMock()
    item.id = req_id
    item.child_req_ids = None
    item.request.num_input_tokens = num_tokens
    item.request.py_scheduling_params = Mock(attention_dp_rank=target, attention_dp_relax=relax)
    return item


def _survivor_states(active=None):
    active = active or {}
    return [
        RankState(rank=rank, num_active_requests=active.get(rank, 0), num_active_tokens=0)
        for rank in SURVIVORS
    ]


def test_gather_all_rank_states_omits_failed_ranks():
    router = _router()
    router.dist.tp_allgather.return_value = [
        RankState(rank=0, num_active_requests=1).serialize(),
        RankState(rank=1).serialize(),
        None,
        RankState(rank=3, num_active_requests=2).serialize(),
    ]
    states = router.gather_all_rank_states([])
    assert [state.rank for state in states] == list(SURVIVORS)
    assert [state.num_active_requests for state in states] == [1, 0, 2]


def test_default_router_balances_over_survivors_by_logical_rank():
    router = _router()
    requests = [_item(i) for i in range(6)]
    routed, expected = router.route_requests(_survivor_states({3: 1}), requests, 8)

    assert set(routed) == set(SURVIVORS)
    assert sum(len(reqs) for reqs in routed.values()) == 6
    assert len(routed[3]) <= expected


def test_only_strict_pins_to_failed_rank_are_unservable():
    pinned = _item(0, target=FAILED_RANK, relax=False)
    relaxed = _item(1, target=FAILED_RANK, relax=True)
    live = _item(2, target=3, relax=False)
    routable, unservable = DefaultADPRouter.split_unservable_requests(
        [pinned, relaxed, live], _survivor_states()
    )
    assert routable == [relaxed, live] and unservable == [pinned]


def test_relaxed_pin_to_failed_rank_is_rebalanced():
    router = _router()
    relaxed = _item(0, target=FAILED_RANK, relax=True)
    routed, _ = router.route_requests(_survivor_states(), [relaxed], 8)
    assert set(routed) == set(SURVIVORS)
    assert [req for reqs in routed.values() for req in reqs] == [relaxed]


def test_cuda_graph_vote_ignores_failed_rank():
    gathered = np.array([[1, 2, 0], [1, 4, 0], [0, 0, 0], [1, 1, 0]], dtype=np.int64)
    dist = Mock(live_ranks=SURVIVORS)
    dist.tp_allgather_int64.return_value = gathered
    runner = SimpleNamespace(config=SimpleNamespace(dist=dist))
    votes = CUDAGraphRunner._gather_adp_graph_batch_info(
        runner, SimpleNamespace(batch_size=2), True, SampleType.FAST
    )
    assert votes == [(True, 2, 0), (True, 4, 0), (True, 1, 0)]


def test_waiting_queue_releases_strict_pin_to_failed_rank():
    waiting_queue = FCFSWaitingQueue()
    pinned = _item(0, target=FAILED_RANK, relax=False)
    full = _item(1, target=1, relax=False)
    waiting_queue.add_requests([pinned, full])
    counts = {rank: 0 for rank in SURVIVORS}
    counts[1] = 8

    popped = get_from_waiting_queue(
        waiting_queue,
        max_req_count=4,
        enable_attention_dp=True,
        max_num_active_requests=8,
        all_ranks_num_active_requests=counts,
    )
    assert popped == [pinned]
    assert [item.id for item in waiting_queue] == [1]
    assert counts[1] == 8


def test_fail_unservable_requests_responds_from_rank0_only():
    for rank, expect_response in ((0, True), (1, False)):
        executor = PyExecutor.__new__(PyExecutor)
        executor.dist = Mock(rank=rank)
        executor.gather_all_responses = False
        executor._enqueue_responses = Mock()
        item = _item(7, target=FAILED_RANK, relax=False)
        item.request.client_id = 11

        executor._fail_unservable_requests([item])

        (responses,), _ = executor._enqueue_responses.call_args
        if expect_response:
            ((req_id, response),) = responses
            assert req_id == 7 and response.client_id == 11
            assert f"rank {FAILED_RANK} is not serving" in response.error_msg
        else:
            assert responses == []

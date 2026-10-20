# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention-DP routing and admission over the ranks still serving."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.common import (
    get_all_rank_ctx_requests,
    get_all_rank_num_tokens,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputPreparer
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.request_utils import get_from_waiting_queue
from tensorrt_llm._torch.pyexecutor.sampler.sampler_common import SampleType
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import DefaultADPRouter, RankState
from tensorrt_llm.executor.utils import is_peer_loss_retryable_message

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
    # After a loss the gather raises unless told what a departed rank counts as.
    assert router.dist.tp_allgather.call_args.kwargs == {"departed": None}
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


def test_strict_pin_outside_the_group_is_unservable_without_a_loss():
    states = [RankState(rank=rank) for rank in range(WORLD_SIZE)]
    pinned = _item(0, target=WORLD_SIZE, relax=False)
    assert DefaultADPRouter.split_unservable_requests([pinned], states) == ([], [pinned])


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


class _DepartedRankDist:
    """The int64 all-gather of MPIDist after FAILED_RANK left: every survivor sends ``peer_values``."""

    live_ranks = SURVIVORS
    tp_size = WORLD_SIZE

    def __init__(self, peer_values):
        self._peer_values = peer_values

    def tp_allgather_int64(self, values, *, departed=None):
        if departed is None:
            raise RuntimeError("tp_allgather_int64 needs a departed")
        rows = np.empty((WORLD_SIZE, len(values)), dtype=np.int64)
        rows[:] = departed
        rows[list(SURVIVORS)] = self._peer_values
        return rows


def test_every_per_step_vote_survives_a_departed_rank():
    """Each per-step int64 vote treats the departed rank as neutral instead of failing or vetoing."""
    # Admission: survivors have work, so the group can queue.
    executor = PyExecutor.__new__(PyExecutor)
    executor.enable_attention_dp = True
    executor.dist = _DepartedRankDist([2])
    assert executor._can_queue(SimpleNamespace(batch_size=2)) == (True, True)

    # Token and context-request counts: the departed rank contributes nothing.
    metadata = SimpleNamespace(
        get_adp_token_counts=lambda: [7], set_adp_token_counts=lambda counts: counts[0]
    )
    mapping = SimpleNamespace(has_cp_helix=lambda: False)
    assert get_all_rank_num_tokens(
        metadata, enable_attention_dp=True, mapping=mapping, dist=_DepartedRankDist([7])
    ) == [7, 7, 0, 7]
    assert get_all_rank_ctx_requests(1, enable_attention_dp=True, dist=_DepartedRankDist([1])) == [
        1,
        1,
        0,
        1,
    ]
    # With speculative decoding the token counts travel with the draft counts (tokens, requests, generations).
    runner = SimpleNamespace(
        _config=SimpleNamespace(enable_attention_dp=True),
        mapping=SimpleNamespace(cp_size=1, has_cp_helix=lambda: False),
        dist=_DepartedRankDist([7, 8, 2, 2]),
    )
    spec = SimpleNamespace(dp_num_tokens=lambda: 8, seq_lens=[4, 4], num_generations=2)
    assert DecoderInputPreparer._get_all_rank_num_tokens_and_spec_counts(
        runner, metadata, spec
    ) == (
        [7, 7, 0, 7],
        [[8, 8, 0, 8], [2, 2, 0, 2], [2, 2, 0, 2]],
    )

    # CUDA graph agreement: the departed rank does not vote.
    runner = SimpleNamespace(config=SimpleNamespace(dist=_DepartedRankDist([1, 2, 0])))
    votes = CUDAGraphRunner._gather_adp_graph_batch_info(
        runner, SimpleNamespace(batch_size=2), True, SampleType.FULL
    )
    assert votes == [(True, 2, 0)] * len(SURVIVORS)


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
            assert is_peer_loss_retryable_message(response.error_msg)
        else:
            assert responses == []

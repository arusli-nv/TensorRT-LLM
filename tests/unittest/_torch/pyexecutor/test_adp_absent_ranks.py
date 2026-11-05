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
from tensorrt_llm.executor.utils import is_rank_failure_retryable_message

pytestmark = pytest.mark.cpu_only

_GATED_MODULES = (
    "tensorrt_llm._torch.pyexecutor.request_utils",
    "tensorrt_llm._torch.pyexecutor.scheduler.adp_router",
)

WORLD_SIZE = 4
FAILED_RANK = 2
LIVE_RANKS = (0, 1, 3)


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


def _live_rank_states(active=None):
    active = active or {}
    return [
        RankState(rank=rank, num_active_requests=active.get(rank, 0), num_active_tokens=0)
        for rank in LIVE_RANKS
    ]


@pytest.fixture
def recovery(monkeypatch):
    """Rank failure recovery is enabled."""
    for module in _GATED_MODULES:
        monkeypatch.setattr(f"{module}.rank_failure_recovery_enabled", lambda: True)


@pytest.fixture
def no_recovery(monkeypatch):
    """Fault tolerance is off, so no rank is absent and every rank counts."""
    for module in _GATED_MODULES:
        monkeypatch.setattr(f"{module}.rank_failure_recovery_enabled", lambda: False)


def test_gather_all_rank_states_omits_failed_ranks():
    router = _router()
    router.dist.tp_allgather.return_value = [
        RankState(rank=0, num_active_requests=1).serialize(),
        RankState(rank=1).serialize(),
        None,
        RankState(rank=3, num_active_requests=2).serialize(),
    ]
    states = router.gather_all_rank_states([])
    # After a rank failure the gather raises unless told what an absent rank counts as.
    assert router.dist.tp_allgather.call_args.kwargs == {"absent": None}
    assert [state.rank for state in states] == list(LIVE_RANKS)
    assert [state.num_active_requests for state in states] == [1, 0, 2]


def test_default_router_balances_over_live_ranks_by_logical_rank():
    router = _router()
    requests = [_item(i) for i in range(6)]
    routed, expected = router.route_requests(_live_rank_states({3: 1}), requests, 8)

    assert set(routed) == set(LIVE_RANKS)
    assert sum(len(reqs) for reqs in routed.values()) == 6
    assert len(routed[3]) <= expected


def test_only_strict_pins_to_failed_rank_are_unservable():
    pinned = _item(0, target=FAILED_RANK, relax=False)
    relaxed = _item(1, target=FAILED_RANK, relax=True)
    live = _item(2, target=3, relax=False)
    routable, unservable = DefaultADPRouter.split_unservable_requests(
        [pinned, relaxed, live], _live_rank_states()
    )
    assert routable == [relaxed, live] and unservable == [pinned]


def test_strict_pin_outside_the_group_is_unservable_without_a_rank_failure():
    states = [RankState(rank=rank) for rank in range(WORLD_SIZE)]
    pinned = _item(0, target=WORLD_SIZE, relax=False)
    assert DefaultADPRouter.split_unservable_requests([pinned], states) == ([], [pinned])


def test_relaxed_pin_to_failed_rank_is_rebalanced(recovery):
    router = _router()
    relaxed = _item(0, target=FAILED_RANK, relax=True)
    routed, _ = router.route_requests(_live_rank_states(), [relaxed], 8)
    assert set(routed) == set(LIVE_RANKS)
    assert [req for reqs in routed.values() for req in reqs] == [relaxed]


def test_cuda_graph_vote_ignores_failed_rank():
    gathered = np.array([[1, 2, 0], [1, 4, 0], [0, 0, 0], [1, 1, 0]], dtype=np.int64)
    dist = Mock(live_ranks=LIVE_RANKS)
    dist.tp_allgather_int64.return_value = gathered
    runner = SimpleNamespace(config=SimpleNamespace(dist=dist))
    votes = CUDAGraphRunner._gather_adp_graph_batch_info(
        runner, SimpleNamespace(batch_size=2), True, SampleType.FAST
    )
    assert votes == [(True, 2, 0), (True, 4, 0), (True, 1, 0)]


class _AbsentRankDist:
    """The int64 all-gather of MPIDist after FAILED_RANK left: every live rank sends ``peer_values``."""

    live_ranks = LIVE_RANKS
    tp_size = WORLD_SIZE
    tp_rank = 0
    cp_size = 1

    def __init__(self, peer_values):
        self._peer_values = peer_values

    def tp_allgather_int64(self, values, *, absent=None):
        if absent is None:
            raise RuntimeError("tp_allgather_int64 needs an absent value")
        rows = np.empty((WORLD_SIZE, len(values)), dtype=np.int64)
        rows[:] = absent
        rows[list(LIVE_RANKS)] = self._peer_values
        return rows

    def tp_cp_allgather_int64(self, values, *, absent=None):
        return self.tp_allgather_int64(values, absent=absent)


def test_every_per_step_vote_survives_an_absent_rank(recovery):
    """Each per-step int64 vote treats the absent rank as neutral instead of failing or vetoing."""
    # Admission: live ranks have work, so the group can queue.
    executor = PyExecutor.__new__(PyExecutor)
    executor.model_engine = SimpleNamespace()
    executor.enable_attention_dp = True
    executor.dist = _AbsentRankDist([2])
    assert executor._can_queue(SimpleNamespace(batch_size=2)) == (True, True)

    # Token and context-request counts: the absent rank contributes nothing.
    metadata = SimpleNamespace(
        get_adp_token_counts=lambda: [7], set_adp_token_counts=lambda counts: counts[0]
    )
    mapping = SimpleNamespace(has_cp_helix=lambda: False)
    assert get_all_rank_num_tokens(
        metadata, enable_attention_dp=True, mapping=mapping, dist=_AbsentRankDist([7])
    ) == [7, 7, 0, 7]
    assert get_all_rank_ctx_requests(1, enable_attention_dp=True, dist=_AbsentRankDist([1])) == [
        1,
        1,
        0,
        1,
    ]
    # With speculative decoding the token counts travel with the draft counts (tokens, requests, generations).
    runner = SimpleNamespace(
        _config=SimpleNamespace(enable_attention_dp=True),
        mapping=SimpleNamespace(cp_size=1, has_cp_helix=lambda: False),
        dist=_AbsentRankDist([7, 8, 2, 2]),
    )
    spec = SimpleNamespace(dp_num_tokens=lambda: 8, seq_lens=[4, 4], num_generations=2)
    assert DecoderInputPreparer._get_all_rank_num_tokens_and_spec_counts(
        runner, metadata, spec
    ) == (
        [7, 7, 0, 7],
        [[8, 8, 0, 8], [2, 2, 0, 2], [2, 2, 0, 2]],
    )

    # CUDA graph agreement: the absent rank does not vote.
    runner = SimpleNamespace(config=SimpleNamespace(dist=_AbsentRankDist([1, 2, 0])))
    votes = CUDAGraphRunner._gather_adp_graph_batch_info(
        runner, SimpleNamespace(batch_size=2), True, SampleType.FULL
    )
    assert votes == [(True, 2, 0)] * len(LIVE_RANKS)


@pytest.mark.parametrize("tp_size", [1, 4])
def test_without_rank_failure_recovery_token_and_spec_counts_gather_unchanged(no_recovery, tp_size):
    """Without recovery the counts go through tp_cp_allgather_int64: no collective at tp_size 1.

    With recovery they take the same call; its ``absent`` fills the rows of ranks that left."""
    dist = Mock(spec=["tp_cp_allgather_int64", "tp_allgather_int64"])
    dist.tp_cp_allgather_int64.side_effect = lambda values, absent: np.tile(
        np.asarray(values, dtype=np.int64), (tp_size, 1)
    )
    runner = SimpleNamespace(
        _config=SimpleNamespace(enable_attention_dp=True),
        mapping=SimpleNamespace(cp_size=1, has_cp_helix=lambda: False),
        dist=dist,
    )
    metadata = SimpleNamespace(
        get_adp_token_counts=lambda: [7], set_adp_token_counts=lambda counts: counts[0]
    )
    spec = SimpleNamespace(dp_num_tokens=lambda: 8, seq_lens=[4, 4], num_generations=2)
    assert DecoderInputPreparer._get_all_rank_num_tokens_and_spec_counts(
        runner, metadata, spec
    ) == ([7] * tp_size, [[8] * tp_size, [2] * tp_size, [2] * tp_size])
    dist.tp_cp_allgather_int64.assert_called_once_with([7, 8, 2, 2], absent=0)
    dist.tp_allgather_int64.assert_not_called()


def test_waiting_queue_releases_strict_pin_to_failed_rank(recovery):
    waiting_queue = FCFSWaitingQueue()
    pinned = _item(0, target=FAILED_RANK, relax=False)
    full = _item(1, target=1, relax=False)
    waiting_queue.add_requests([pinned, full])
    counts = {rank: 0 for rank in LIVE_RANKS}
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


def _pop_pinned(target, full_rank=None):
    """Pop a request strictly pinned to ``target``; ``full_rank`` is at capacity. Returns the popped requests."""
    counts = {rank: 8 if rank == full_rank else 0 for rank in range(WORLD_SIZE)}
    waiting_queue = FCFSWaitingQueue()
    waiting_queue.add_requests([_item(0, target=target, relax=False)])
    popped = get_from_waiting_queue(
        waiting_queue,
        max_req_count=4,
        enable_attention_dp=True,
        max_num_active_requests=8,
        all_ranks_num_active_requests=counts,
    )
    return popped


def test_without_rank_failure_recovery_a_pin_past_the_last_rank_raises_unchanged(no_recovery):
    """Per-rank lists are indexed: IndexError at admission and in the router."""
    with pytest.raises(IndexError):
        _pop_pinned(WORLD_SIZE)
    states = [RankState(rank=rank) for rank in range(WORLD_SIZE)]
    for relax in (False, True):
        with pytest.raises(IndexError):
            _router().route_requests(states, [_item(0, target=WORLD_SIZE, relax=relax)], 8)


def test_without_rank_failure_recovery_a_negative_pin_is_handled_unchanged(no_recovery):
    """It is admitted against the last rank's count, then the router raises placing it (KeyError)."""
    assert len(_pop_pinned(-1)) == 1
    assert len(_pop_pinned(-1, full_rank=0)) == 1
    assert _pop_pinned(-1, full_rank=WORLD_SIZE - 1) == []
    states = [RankState(rank=rank) for rank in range(WORLD_SIZE)]
    for relax in (False, True):
        with pytest.raises(KeyError):
            _router().route_requests(states, [_item(0, target=-1, relax=relax)], 8)


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
            assert not is_rank_failure_retryable_message(response.error_msg)
        else:
            assert responses == []

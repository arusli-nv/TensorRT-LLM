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
"""Scheduler-side WideEP peer-loss recovery with fake distributed, model, and authority."""

import threading
import time
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import pytest

from tensorrt_llm._torch.distributed.communicator import HostCollectiveInterrupted
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import MoeLoadBalancer
from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    ExecutorRequestQueue,
    RequestAdmissionState,
)
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.request_utils import get_from_waiting_queue
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import DefaultADPRouter, RankState
from tensorrt_llm.executor.recovery.client import RecoveryAborted
from tensorrt_llm.executor.recovery.protocol import (
    Abort,
    Commit,
    Install,
    Installed,
    Propose,
    Ready,
    Stopped,
)

pytestmark = pytest.mark.cpu_only

WORLD_SIZE = 4
FAILED_RANK = 2
SURVIVORS = (0, 1, 3)


# ---------------------------------------------------------------------------
# Request queue admission
# ---------------------------------------------------------------------------


def _request_queue(rank=0):
    return ExecutorRequestQueue(
        dist=Mock(rank=rank),
        max_batch_size=8,
        enable_iter_perf_stats=False,
        batch_wait_timeout_ms=0,
    )


def test_request_queue_recovering_closes_and_reopens_admission():
    queue = _request_queue()
    queue.begin_recovery()
    assert queue.get_admission_state() is RequestAdmissionState.RECOVERING
    assert not queue.can_enqueue_request()
    with pytest.raises(RuntimeError, match="recovering"):
        queue.enqueue_request(Mock())

    queue.complete_recovery()
    assert queue.get_admission_state() is RequestAdmissionState.RUNNING
    assert queue.can_enqueue_request()


def test_request_queue_recovery_requires_running_admission():
    queue = _request_queue()
    queue.begin_sleep_transition(["weights"])
    with pytest.raises(RuntimeError, match="Invalid executor admission transition"):
        queue.begin_recovery()
    with pytest.raises(RuntimeError, match="Invalid executor admission transition"):
        queue.complete_recovery()


# ---------------------------------------------------------------------------
# ADP routing over survivors
# ---------------------------------------------------------------------------


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
    assert DefaultADPRouter.unservable_requests(routed, _survivor_states()) == []


def test_default_router_honours_strict_pin_to_live_logical_rank():
    router = _router()
    pinned = _item(0, target=3, relax=False)
    routed, _ = router.route_requests(_survivor_states(), [pinned], 8)
    assert routed[3] == [pinned]


def test_strict_pin_to_failed_rank_is_unservable_and_others_still_route():
    router = _router()
    pinned = _item(0, target=FAILED_RANK, relax=False)
    others = [_item(i) for i in range(1, 4)]
    routed, _ = router.route_requests(_survivor_states(), [pinned, *others], 8)

    unservable = DefaultADPRouter.unservable_requests(routed, _survivor_states())
    assert unservable == [pinned]
    assert set(routed) == set(SURVIVORS)
    assert sorted(req.id for reqs in routed.values() for req in reqs) == [1, 2, 3]


def test_relaxed_pin_to_failed_rank_is_rebalanced():
    router = _router()
    relaxed = _item(0, target=FAILED_RANK, relax=True)
    routed, _ = router.route_requests(_survivor_states(), [relaxed], 8)
    assert DefaultADPRouter.unservable_requests(routed, _survivor_states()) == []
    assert [req for reqs in routed.values() for req in reqs] == [relaxed]


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


# ---------------------------------------------------------------------------
# Placement helpers
# ---------------------------------------------------------------------------


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


def test_masked_placement_digest_is_deterministic_and_mask_sensitive():
    load_balancer = _load_balancer([_REDUNDANT, _REDUNDANT])
    digest = load_balancer.masked_placement_digest({FAILED_RANK})
    assert digest == _load_balancer([_REDUNDANT, _REDUNDANT]).masked_placement_digest([FAILED_RANK])
    assert digest != load_balancer.masked_placement_digest({1})
    assert digest != load_balancer.masked_placement_digest(set())


def test_masked_placement_digest_matches_installed_mask():
    load_balancer = _load_balancer([_REDUNDANT])
    digest = load_balancer.masked_placement_digest({FAILED_RANK})
    installed = np.array(_REDUNDANT)
    installed[FAILED_RANK] = -1
    load_balancer.single_layer_load_balancers[0].placement = installed.tolist()
    assert load_balancer.masked_placement_digest({FAILED_RANK}) == digest


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


def test_reconfigure_mask_only_with_timeout_bounds_native_wait(no_cuda_device):
    load_balancer = _load_balancer([_REDUNDANT])
    release = threading.Event()
    load_balancer.reconfigure_mask_only = lambda dead_ranks: release.wait(5)
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        load_balancer.reconfigure_mask_only_with_timeout([FAILED_RANK], 0.05)
    assert time.monotonic() - start < 2
    release.set()


def test_reconfigure_mask_only_with_timeout_propagates_errors(no_cuda_device):
    load_balancer = _load_balancer([_REDUNDANT])
    calls = []

    def reconfigure(dead_ranks):
        calls.append(dead_ranks)
        raise RuntimeError("no surviving replica")

    load_balancer.reconfigure_mask_only = reconfigure
    with pytest.raises(RuntimeError, match="no surviving replica"):
        load_balancer.reconfigure_mask_only_with_timeout([FAILED_RANK], 5)
    assert calls == [[FAILED_RANK]]


def test_install_resident_placement_commits_mask(no_cuda_device):
    load_balancer = _load_balancer([_REDUNDANT])
    digest = load_balancer.masked_placement_digest({FAILED_RANK})

    def reconfigure(dead_ranks):
        layer = load_balancer.single_layer_load_balancers[0]
        layer.placement = [[-1, -1] if r in dead_ranks else row for r, row in enumerate(_REDUNDANT)]

    load_balancer.reconfigure_mask_only = reconfigure
    health = EPGroupHealth(WORLD_SIZE)
    execution_state = Mock(health=health)
    engine = PyTorchModelEngine.__new__(PyTorchModelEngine)
    engine.moe_load_balancer = load_balancer

    with patch.object(
        PyTorchModelEngine,
        "moe_execution_state",
        new_callable=PropertyMock,
        return_value=execution_state,
    ):
        assert engine.prepare_resident_placement(frozenset({FAILED_RANK})) == digest
        engine.install_resident_placement(frozenset({FAILED_RANK}), digest, time.monotonic() + 5)

    assert health.get_failed_ranks() == frozenset({FAILED_RANK})
    execution_state.install_mask.assert_called_once_with(health.get_mask_words())


def test_install_resident_placement_rejects_divergent_placement(no_cuda_device):
    load_balancer = _load_balancer([_REDUNDANT])
    load_balancer.reconfigure_mask_only = Mock()
    engine = PyTorchModelEngine.__new__(PyTorchModelEngine)
    engine.moe_load_balancer = load_balancer
    with pytest.raises(RuntimeError, match="differs from the agreed placement"):
        engine.install_resident_placement(frozenset({FAILED_RANK}), "0" * 64, time.monotonic() + 5)


# ---------------------------------------------------------------------------
# Executor recovery sequence
# ---------------------------------------------------------------------------


class _FakeClient:
    """Scripted authority replies; records what the worker sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    def send(self, message):
        self.sent.append(message)

    def wait(self, message_type, deadline):
        assert deadline > time.monotonic()
        reply = self.replies.pop(0)
        if isinstance(reply, Abort):
            raise RecoveryAborted(reply.reason)
        assert isinstance(reply, message_type)
        return reply


def _happy_replies(generation=1):
    return [
        Propose(generation=generation, failed_rank=FAILED_RANK, survivors=SURVIVORS),
        Install(generation=generation),
        Commit(generation=generation),
    ]


def _executor(client, rank=1, failed=frozenset({FAILED_RANK})):
    executor = PyExecutor.__new__(PyExecutor)
    events = []
    dist = Mock(rank=rank)
    dist.revoke.side_effect = lambda: events.append("revoke")
    dist.shrink.side_effect = lambda: events.append("shrink") or failed
    dist.install_survivors.side_effect = lambda failed: events.append(("survivors", failed))
    engine = Mock()
    engine.prepare_resident_placement.return_value = "digest"
    engine.install_resident_placement.side_effect = lambda failed, digest, deadline: events.append(
        ("placement", failed, digest)
    )
    executor.dist = dist
    executor.model_engine = engine
    executor._recovery_client = client
    executor._peer_loss_recovery_timeout_s = 30.0
    executor.active_requests = []
    executor.waiting_queue = FCFSWaitingQueue()
    executor._unbroadcast_requests = []
    executor.previous_batch = None
    executor.executor_request_queue = _request_queue(rank)
    executor._join_device_work = Mock()
    executor._events = events
    return executor


def _interrupted(failed=frozenset({FAILED_RANK})):
    return HostCollectiveInterrupted("tp_allgather", failed)


def test_recovery_runs_protocol_and_installs_in_order():
    client = _FakeClient(_happy_replies(generation=1))
    executor = _executor(client)

    executor._recover_from_peer_loss(_interrupted())

    assert client.sent == [
        Stopped(rank=1, failed_rank=FAILED_RANK),
        Ready(rank=1, generation=1, placement_digest="digest"),
        Installed(rank=1, generation=1),
    ]
    failed = frozenset({FAILED_RANK})
    assert executor._events == [
        "revoke",
        "shrink",
        ("survivors", failed),
        ("placement", failed, "digest"),
    ]
    executor.model_engine.prepare_resident_placement.assert_called_once_with(failed)
    assert executor.executor_request_queue.get_admission_state() is RequestAdmissionState.RUNNING


def test_recovery_learns_failed_rank_from_shrink_after_revocation():
    """A survivor that only saw ERR_REVOKED still reports the agreed failed rank."""
    client = _FakeClient(_happy_replies())
    executor = _executor(client)

    executor._recover_from_peer_loss(_interrupted(frozenset()))

    assert client.sent[0] == Stopped(rank=1, failed_rank=FAILED_RANK)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda ex: ex.active_requests.append(Mock()),
        lambda ex: ex.waiting_queue.add_request(Mock()),
        lambda ex: ex._unbroadcast_requests.append(Mock()),
        lambda ex: setattr(ex, "previous_batch", Mock()),
    ],
    ids=["active", "waiting", "unbroadcast", "previous_batch"],
)
def test_recovery_requires_idle_executor(mutate):
    client = _FakeClient(_happy_replies())
    executor = _executor(client)
    mutate(executor)
    with pytest.raises(RuntimeError, match="requires an idle executor"):
        executor._recover_from_peer_loss(_interrupted())
    assert client.sent == []
    assert executor._events == ["revoke", "shrink"]


@pytest.mark.parametrize("failed", [frozenset(), frozenset({1, 2})])
def test_recovery_requires_exactly_one_departed_rank(failed):
    client = _FakeClient(_happy_replies())
    executor = _executor(client, rank=0, failed=failed)
    with pytest.raises(RuntimeError, match="exactly one departed rank"):
        executor._recover_from_peer_loss(_interrupted())
    assert client.sent == []


@pytest.mark.parametrize(
    "replies, match",
    [
        ([Propose(generation=1, failed_rank=3, survivors=(0, 1, 2))], "authority proposed"),
        ([Propose(generation=1, failed_rank=FAILED_RANK, survivors=(0, 3))], "authority proposed"),
        (_happy_replies()[:1] + [Install(generation=2)], "Expected recovery generation 1"),
        (_happy_replies()[:2] + [Commit(generation=2)], "Expected recovery generation 1"),
        (_happy_replies()[:1] + [Abort("second failure")], "second failure"),
    ],
    ids=["wrong_failed_rank", "not_a_survivor", "install_gen", "commit_gen", "abort"],
)
def test_recovery_fails_closed_on_protocol_mismatch(replies, match):
    executor = _executor(_FakeClient(replies))
    with pytest.raises(RuntimeError, match=match):
        executor._recover_from_peer_loss(_interrupted())
    assert executor.executor_request_queue.get_admission_state() is (
        RequestAdmissionState.RECOVERING
    )


def test_recovery_deadline_bounds_authority_wait():
    client = Mock()
    client.wait.side_effect = TimeoutError("Timed out waiting for Propose")
    executor = _executor(client)
    with pytest.raises(TimeoutError):
        executor._recover_from_peer_loss(_interrupted())
    (_, deadline), _ = client.wait.call_args
    assert deadline - time.monotonic() <= executor._peer_loss_recovery_timeout_s


def test_run_event_loop_resumes_after_recovery():
    executor = _executor(_FakeClient([]))
    error = _interrupted()
    executor.event_loop = Mock(side_effect=[error, None])
    executor._recover_from_peer_loss = Mock()

    executor._run_event_loop()

    assert executor.event_loop.call_count == 2
    executor._recover_from_peer_loss.assert_called_once_with(error)


def test_run_event_loop_without_recovery_reraises():
    executor = _executor(None)
    executor.event_loop = Mock(side_effect=_interrupted())
    executor._recover_from_peer_loss = Mock()
    with pytest.raises(HostCollectiveInterrupted):
        executor._run_event_loop()
    executor._recover_from_peer_loss.assert_not_called()

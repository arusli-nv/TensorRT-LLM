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
import torch

from tensorrt_llm._torch.distributed.communicator import HostCollectiveInterrupted
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import MoeLoadBalancer
from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    ExecutorRequestQueue,
    RequestQueueItem,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmResponse
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.peer_loss_recovery import PeerLossRecovery
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.request_utils import get_from_waiting_queue
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import DefaultADPRouter, RankState
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
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
    events = []
    load_balancer.check_routing_excludes = lambda dead: events.append(("checked", set(dead)))
    health = EPGroupHealth(WORLD_SIZE)
    execution_state = Mock(health=health)
    execution_state.install_mask.side_effect = lambda words: events.append(("mask", words))
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
    # The kernels skip routes to a masked rank, so routing must exclude it first.
    assert events == [("checked", {FAILED_RANK}), ("mask", health.get_mask_words())]


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


def _executor(rank=1, failed=frozenset({FAILED_RANK})):
    executor = PyExecutor.__new__(PyExecutor)
    events = []
    dist = Mock(rank=rank)
    dist.revoke.side_effect = lambda: events.append("revoke")
    dist.shrink.side_effect = lambda: events.append("shrink") or failed
    dist.install_survivors.side_effect = lambda failed: events.append(("survivors", failed))
    dist.watch_peer_failures.side_effect = lambda callback: events.append("watch") or Mock()
    dist.broadcast.side_effect = lambda value, root: value
    engine = Mock()
    engine.prepare_resident_placement.return_value = "digest"
    engine.install_resident_placement.side_effect = lambda failed, digest, deadline: events.append(
        ("placement", failed, digest)
    )
    executor.dist = dist
    executor.model_engine = engine
    executor.active_requests = []
    executor.waiting_queue = FCFSWaitingQueue()
    executor._unbroadcast_requests = []
    executor.request_accumulated = []
    executor._terminate_request = Mock()
    executor._enqueue_responses = Mock()
    executor.gather_all_responses = False
    executor.previous_batch = None
    executor._events = events
    return executor


def _recovery(client, rank=1, failed=frozenset({FAILED_RANK})):
    executor = _executor(rank, failed)
    recovery = PeerLossRecovery(executor, client)
    recovery._join_device_work = Mock()
    recovery._watch = Mock()
    executor._peer_loss_recovery = recovery
    return recovery


def _interrupted(failed=frozenset({FAILED_RANK})):
    return HostCollectiveInterrupted("tp_allgather", failed)


def test_recovery_runs_protocol_and_installs_in_order():
    client = _FakeClient(_happy_replies(generation=1))
    recovery = _recovery(client)

    recovery._recover(_interrupted())

    assert client.sent == [
        Stopped(rank=1, failed_rank=FAILED_RANK),
        Ready(rank=1, generation=1, placement_digest="digest"),
        Installed(rank=1, generation=1),
    ]
    failed = frozenset({FAILED_RANK})
    assert recovery._executor._events == [
        "revoke",
        "shrink",
        ("survivors", failed),
        ("placement", failed, "digest"),
        "watch",
    ]
    recovery._executor.model_engine.prepare_resident_placement.assert_called_once_with(failed)


def test_recovery_learns_failed_rank_from_shrink_after_revocation():
    """A survivor that only saw ERR_REVOKED still reports the agreed failed rank."""
    client = _FakeClient(_happy_replies())
    recovery = _recovery(client)

    recovery._recover(_interrupted(frozenset()))

    assert client.sent[0] == Stopped(rank=1, failed_rank=FAILED_RANK)


def _queued(request_id):
    return RequestQueueItem(request_id, Mock())


@pytest.mark.parametrize("rank", [0, 1])
def test_recovery_keeps_unscheduled_requests_on_rank0_only(rank):
    """Rank 0 re-broadcasts received-but-unscheduled requests after recovery."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=rank)
    executor = recovery._executor
    executor.request_accumulated = [_queued(5)]
    executor.waiting_queue.add_requests([_queued(1), _queued(2)])
    executor._unbroadcast_requests = [_queued(3)]

    recovery._recover(_interrupted())

    assert len(executor.waiting_queue) == 0
    assert executor._unbroadcast_requests == []
    expected = [1, 2, 3, 5] if rank == 0 else [5]
    assert [item.id for item in executor.request_accumulated] == expected


@pytest.mark.parametrize("failed", [frozenset(), frozenset({1, 2})])
def test_recovery_requires_exactly_one_departed_rank(failed):
    client = _FakeClient(_happy_replies())
    recovery = _recovery(client, rank=0, failed=failed)
    with pytest.raises(RuntimeError, match="exactly one departed rank"):
        recovery._recover(_interrupted())
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
    recovery = _recovery(_FakeClient(replies))
    with pytest.raises(RuntimeError, match=match):
        recovery._recover(_interrupted())


def test_recovery_deadline_bounds_authority_wait():
    client = Mock()
    client.wait.side_effect = TimeoutError("Timed out waiting for Propose")
    recovery = _recovery(client)
    with pytest.raises(TimeoutError):
        recovery._recover(_interrupted())
    (_, deadline), _ = client.wait.call_args
    assert deadline - time.monotonic() <= recovery._timeout_s


def test_run_discards_the_step_then_recovers_and_resumes():
    recovery = _recovery(_FakeClient([]))
    error = _interrupted()
    event_loop = Mock(side_effect=[error, None])
    calls = []
    batch = Mock(name="batch")
    recovery.step_prepared(batch)
    recovery._executor._discard_uncommitted_step = lambda step: calls.append(("discard", step))
    recovery._recover = lambda err: calls.append(("recover", err))

    recovery.run(event_loop)

    assert event_loop.call_count == 2
    assert calls == [("discard", batch), ("recover", error)]
    assert recovery._uncommitted_step is None


def test_run_watches_peers_only_while_the_loop_runs():
    recovery = _recovery(_FakeClient([]))
    recovery.run(Mock(return_value=None))
    recovery._executor.dist.watch_peer_failures.assert_called_once_with(recovery._on_peer_failure)
    recovery._watch.stop.assert_called_once()


def test_run_does_not_recover_during_kv_cache_estimation():
    recovery = _recovery(_FakeClient([]))
    recovery._executor.gather_all_responses = True
    with pytest.raises(HostCollectiveInterrupted):
        recovery.run(Mock(side_effect=_interrupted()))
    recovery._executor.dist.watch_peer_failures.assert_not_called()


def test_event_loop_without_recovery_reraises():
    executor = _executor()
    executor._peer_loss_recovery = None
    executor.event_loop = Mock(side_effect=_interrupted())
    with pytest.raises(HostCollectiveInterrupted):
        executor._run_event_loop()


def test_recovery_keeps_active_requests_and_rearms_failure_watch():
    """Requests of a rolled-back step stay active and run again after recovery."""
    recovery = _recovery(_FakeClient(_happy_replies()))
    request = Mock(is_attention_dp_dummy=False)
    recovery._executor.active_requests.append(request)
    recovery._peer_failed.set()

    recovery._recover(_interrupted())

    assert recovery._executor.active_requests == [request]
    assert not recovery._peer_failed.is_set()
    assert recovery._executor._events[-1] == "watch"


def _gated_recovery():
    recovery = _recovery(_FakeClient([]))
    recovery.step_prepared(Mock(name="batch"))
    return recovery, Mock()


def test_commit_step_accepts_step_without_peer_failure():
    recovery, sample_state = _gated_recovery()
    recovery.commit_step(sample_state)
    sample_state.sampler_event.synchronize.assert_called_once()
    assert recovery._uncommitted_step is None


def test_commit_step_discards_step_after_peer_failure():
    recovery, sample_state = _gated_recovery()
    batch = recovery._uncommitted_step
    recovery._peer_failed.set()
    with pytest.raises(HostCollectiveInterrupted):
        recovery.commit_step(sample_state)
    sample_state.sampler_event.synchronize.assert_called_once()
    assert recovery._uncommitted_step is batch


def test_commit_step_without_prepared_step_does_not_wait():
    recovery = _recovery(_FakeClient([]))
    sample_state = Mock()
    recovery.commit_step(sample_state)
    sample_state.sampler_event.synchronize.assert_not_called()


def test_peer_failure_flags_before_aborting_collectives():
    recovery = _recovery(_FakeClient([]))
    seen = []
    engine = recovery._executor.model_engine
    engine.moe_execution_state.request_abort.side_effect = lambda: seen.append(
        recovery._peer_failed.is_set()
    )
    recovery._on_peer_failure()
    assert seen == [True]


def test_discard_returns_step_resources_and_padding_request():
    executor = _executor()
    executor.kv_cache_manager = Mock()
    executor.inflight_req_ids = Mock()
    executor._pending_adp_dummy_request = None
    batch = Mock(name="batch")
    request = Mock(is_attention_dp_dummy=False)
    dummy = Mock(is_attention_dp_dummy=True, py_request_id=99)
    executor.active_requests.extend([request, dummy])

    executor._discard_uncommitted_step(batch)

    executor.kv_cache_manager.revert_prepare_resources.assert_called_once_with(batch)
    executor._terminate_request.assert_called_once_with(dummy)
    assert executor.active_requests == [request]


def test_discard_before_prepare_releases_the_tentative_padding_request():
    """A loss in the can-queue exchange precedes prepare_resources."""
    executor = _executor()
    executor.kv_cache_manager = Mock()
    executor.inflight_req_ids = Mock()
    executor.resource_manager = Mock()
    executor.resource_manager.get_resource_manager.return_value = None
    request = Mock(is_attention_dp_dummy=False)
    dummy = Mock(is_attention_dp_dummy=True, py_request_id=99, py_seq_slot=None)
    executor.active_requests.extend([request, dummy])
    executor._pending_adp_dummy_request = dummy

    executor._discard_uncommitted_step(None)

    executor.kv_cache_manager.free_resources.assert_called_once_with(dummy)
    executor.kv_cache_manager.revert_prepare_resources.assert_not_called()
    executor._terminate_request.assert_not_called()
    assert executor.active_requests == [request]


def test_publish_defers_terminations_when_the_gather_is_interrupted():
    recovery = _recovery(_FakeClient([]))
    executor = recovery._executor
    executor._enqueue_responses = Mock(side_effect=_interrupted())
    finished = Mock(name="finished")
    with pytest.raises(HostCollectiveInterrupted):
        executor._publish_responses([], [finished])
    executor._terminate_request.assert_not_called()
    assert recovery._undelivered_terminations == [finished]


def test_step_errors_after_a_peer_failure_become_interruptions():
    recovery = _recovery(_FakeClient([]))
    executor = recovery._executor
    executor._raise_if_peer_failed("forward", RuntimeError("garbage"))
    recovery._peer_failed.set()
    with pytest.raises(HostCollectiveInterrupted):
        executor._raise_if_peer_failed("forward", RuntimeError("garbage"))


def test_revert_prepare_resources_undoes_v1_reservations():
    """Each reservation prepare_resources made for the step is returned."""
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.kv_cache_type = CacheTypeCpp.SELF
    manager.mapping = Mock(has_cp_helix=Mock(return_value=False))
    manager.impl = Mock()
    manager._kv_reserve_draft_tokens = 0
    manager._preprepared_dummy_request_ids = {7}
    manager.kv_connector_manager = None
    first_chunk = Mock(py_request_id=1, is_first_context_chunk=True, py_beam_width=1)
    later_chunk = Mock(py_request_id=2, is_first_context_chunk=False, py_beam_width=1)
    prepared_dummy = Mock(py_request_id=7, is_first_context_chunk=True, py_beam_width=1)
    generating = Mock(py_request_id=3, py_draft_tokens=None)
    batch = Mock(
        context_requests=[first_chunk, later_chunk, prepared_dummy],
        generation_requests=[generating],
    )
    manager.revert_prepare_resources(batch)
    manager.impl.remove_sequence.assert_called_once_with(1, first_chunk, False)
    manager.impl.rewind_kv_cache.assert_called_once_with(3, 1)


def _sent_responses(recovery):
    sent = []
    recovery._executor._enqueue_responses = lambda responses: sent.append(list(responses))
    return sent


def test_routed_requests_are_tracked_until_their_final_response():
    recovery = _recovery(_FakeClient([]), rank=0)
    item = Mock(id=11, request=Mock(client_id="c11"))
    recovery.requests_routed({FAILED_RANK: [item]})
    assert recovery._request_owners == {11: (FAILED_RANK, "c11")}
    streaming = Mock(spec=LlmResponse, result=Mock(is_final=False))
    streaming.has_error.return_value = False
    recovery.responses_gathered([(11, streaming)])
    assert 11 in recovery._request_owners
    final = Mock(spec=LlmResponse, result=Mock(is_final=True))
    final.has_error.return_value = False
    recovery.responses_gathered([(11, final)])
    assert recovery._request_owners == {}


def test_recovery_fails_requests_owned_by_departed_rank():
    """Rank 0 fails the requests whose KV cache left with the departed rank, once."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=0)
    recovery._request_owners = {11: (FAILED_RANK, "c11"), 12: (3, "c12")}
    sent = _sent_responses(recovery)

    recovery._recover(_interrupted())

    ((request_id, response),) = sent[-1]
    assert request_id == 11 and response.client_id == "c11"
    assert f"rank {FAILED_RANK} failed" in response.error_msg
    assert recovery._request_owners == {12: (3, "c12")}


def test_survivors_join_departed_request_responses():
    """Ranks without the owner table still take part in the response collective."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=1)
    assert recovery._request_owners is None
    sent = _sent_responses(recovery)
    recovery._recover(_interrupted())
    assert sent == [[]]


@pytest.mark.parametrize("processed", [0, 1, 2])
def test_recovery_resends_gathers_rank0_did_not_process(processed):
    """A rank resends its parts of every gather rank 0 did not process."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=1)
    first, second = [(5, Mock(name="first"))], [(6, Mock(name="second"))]
    recovery.gathering_responses(first)
    recovery.gathering_responses(second)
    recovery._executor.dist.broadcast.side_effect = lambda value, root: processed
    finished = Mock(name="finished")
    recovery.defer_terminations([finished])
    order = []
    executor = recovery._executor
    executor._enqueue_responses = lambda responses: order.append(("send", list(responses)))
    executor._terminate_request = lambda request: order.append(("terminate", request))

    recovery._recover(_interrupted())

    expected = [first + second, second, []][processed]
    assert order == [("send", expected), ("terminate", finished)]
    assert recovery._undelivered_terminations == []


def _supported_executor():
    from types import SimpleNamespace

    load_balancer = Mock()
    load_balancer.is_static_routing.return_value = True
    load_balancer.covers_loss_of.side_effect = lambda rank: rank in (1, 2)
    return SimpleNamespace(
        model_engine=SimpleNamespace(
            moe_load_balancer=load_balancer, moe_execution_state=Mock(), spec_config=None
        ),
        kv_cache_manager=SimpleNamespace(enable_block_reuse=False),
        enable_attention_dp=True,
        _disable_mpi=False,
        dist=Mock(rank=0, world_size=4, pp_size=1, cp_size=1),
        adp_router=DefaultADPRouter.__new__(DefaultADPRouter),
        drafter=None,
        draft_model_engine=None,
        guided_decoder=None,
        max_beam_width=1,
        kv_cache_transceiver=None,
        kv_connector_manager=None,
        _is_kv_manager_v2=False,
        disable_overlap_scheduler=True,
        llm_args=SimpleNamespace(sleep_config=None),
    )


@pytest.mark.parametrize(
    "unsupported, requirement",
    [
        (lambda ex: setattr(ex, "drafter", Mock()), "no speculative decoding"),
        (lambda ex: setattr(ex.model_engine, "spec_config", Mock()), "no speculative decoding"),
        (lambda ex: setattr(ex, "guided_decoder", Mock()), "no guided decoding"),
        (lambda ex: setattr(ex, "max_beam_width", 2), "no beam search"),
        (lambda ex: setattr(ex.kv_cache_manager, "enable_block_reuse", True), "block reuse"),
        (lambda ex: setattr(ex, "kv_connector_manager", Mock()), "no KV connector"),
        (lambda ex: setattr(ex, "disable_overlap_scheduler", False), "no overlap scheduler"),
        (lambda ex: setattr(ex, "_is_kv_manager_v2", True), "KV cache manager v1"),
    ],
)
def test_create_rejects_configurations_a_discarded_step_cannot_rerun(
    monkeypatch, unsupported, requirement
):
    from tensorrt_llm._torch.pyexecutor import peer_loss_recovery

    monkeypatch.setattr(peer_loss_recovery, "recovery_enabled", lambda: True)
    executor = _supported_executor()
    unsupported(executor)
    with pytest.raises(ValueError, match=requirement):
        PeerLossRecovery.create(executor)
    executor.dist.enable_peer_loss_recovery.assert_not_called()


def test_create_registers_the_ranks_whose_loss_the_placement_covers(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import peer_loss_recovery

    monkeypatch.setattr(peer_loss_recovery, "recovery_enabled", lambda: True)
    registered = []
    monkeypatch.setattr(
        peer_loss_recovery,
        "_register",
        lambda rank, recoverable: registered.append((rank, recoverable)) or (Mock(), frozenset()),
    )
    executor = _supported_executor()
    assert isinstance(PeerLossRecovery.create(executor), PeerLossRecovery)
    assert registered == [(0, (1, 2))]
    executor.dist.enable_peer_loss_recovery.assert_called_once()

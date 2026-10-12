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

import signal
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from tensorrt_llm._torch.distributed.communicator import HostCollectiveInterrupted
from tensorrt_llm._torch.pyexecutor import peer_loss_recovery
from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmResponse
from tensorrt_llm._torch.pyexecutor.peer_loss_recovery import PeerLossRecovery
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import DefaultADPRouter
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
    dist.allgather.side_effect = lambda value: [value] * WORLD_SIZE
    engine = Mock()
    engine.moe_load_balancer.masked_placement_digest.return_value = "digest"
    engine.install_resident_placement.side_effect = lambda failed, digest: events.append(
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
    executor.hang_detector = MagicMock()
    executor._events = events
    return executor


def _recovery(client, rank=1, failed=frozenset({FAILED_RANK})):
    executor = _executor(rank, failed)
    recovery = PeerLossRecovery(executor, client, frozenset({1, FAILED_RANK, 3}))
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
    masked_placement_digest = (
        recovery._executor.model_engine.moe_load_balancer.masked_placement_digest
    )
    masked_placement_digest.assert_called_once_with(failed)


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


def _fail(error):
    def raise_error(*args, **kwargs):
        raise error

    return raise_error


@pytest.mark.parametrize(
    "inject",
    [
        lambda r: setattr(r._client, "replies", [Abort("authority stopped")]),
        lambda r: setattr(r._client, "replies", _happy_replies()[:1] + [Abort("stalled")]),
        lambda r: setattr(r._client, "replies", _happy_replies()[:2] + [Abort("stalled")]),
        lambda r: setattr(r._executor.dist.shrink, "side_effect", _fail(RuntimeError("shrink"))),
        lambda r: setattr(
            r._executor.model_engine.moe_load_balancer.masked_placement_digest,
            "side_effect",
            _fail(RuntimeError("placement")),
        ),
        lambda r: setattr(
            r._executor.dist.install_survivors, "side_effect", _fail(RuntimeError("survivors"))
        ),
        lambda r: setattr(
            r._executor.model_engine.install_resident_placement,
            "side_effect",
            _fail(RuntimeError("install")),
        ),
    ],
    ids=[
        "abort_at_propose",
        "abort_at_install",
        "abort_at_commit",
        "shrink",
        "prepare",
        "install_survivors",
        "install_placement",
    ],
)
def test_a_failure_at_any_transition_ends_recovery_without_resuming(inject):
    """Fault injection at every transition: the worker raises and resumes nothing."""
    recovery = _recovery(_FakeClient(_happy_replies()))
    recovery._executor._requeue_unscheduled_requests = Mock()
    recovery._peer_failed.set()
    inject(recovery)
    with pytest.raises(RuntimeError):
        recovery._recover(_interrupted())
    assert "watch" not in recovery._executor._events
    recovery._executor._requeue_unscheduled_requests.assert_not_called()
    assert recovery.peer_failed


def test_run_surfaces_a_failed_recovery_after_stopping_the_watch():
    recovery = _recovery(_FakeClient([Abort("authority stopped")]))
    recovery._executor._discard_uncommitted_step = Mock()
    watch = recovery._watch
    recovery._executor.dist.watch_peer_failures.side_effect = lambda callback: watch
    with pytest.raises(RecoveryAborted, match="authority stopped"):
        recovery.run(Mock(side_effect=_interrupted()))
    watch.stop.assert_called()


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
    # The hang detector bounds recovery.
    hang_detector = recovery._executor.hang_detector
    hang_detector.__enter__.assert_called_once()
    hang_detector.__enter__.return_value.checkpoint.assert_called_once()
    hang_detector.__exit__.assert_called_once()


def test_only_an_admitted_rank_outside_recovery_exits_alone():
    assert not _recovery(_FakeClient([]), rank=0).survives_own_loss
    recovery = _recovery(_FakeClient([]))
    assert recovery.survives_own_loss
    recovery._executor._discard_uncommitted_step = Mock()
    recovery._recover = Mock()
    recovery.run(Mock(side_effect=[_interrupted(), None]))
    # A second loss is terminal, so this rank no longer exits alone.
    assert not recovery.survives_own_loss


def test_crashed_loop_exits_only_this_rank_when_peers_can_recover(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import py_executor

    executor = _executor()
    executor._peer_loss_recovery = Mock(survives_own_loss=True)
    executor._run_event_loop = Mock(side_effect=RuntimeError("illegal address"))
    executor._is_warmup = True
    executor._event_loop_error_delivered = Mock()
    executor.garbage_collection_gen0_threshold = None
    kills = []
    monkeypatch.setattr(py_executor.os, "kill", lambda pid, sig: kills.append(sig))
    monkeypatch.setattr(py_executor, "hard_kill_on_rank_crash", Mock())
    monkeypatch.setattr(py_executor, "start_rank_crash_kill_watchdog", Mock(return_value=None))
    executor._executor_loop_cleanup = Mock()
    with pytest.raises(RuntimeError):
        executor._event_loop_wrapper()
    assert kills == [signal.SIGKILL]


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


@pytest.mark.parametrize("peer_failed", [False, True])
def test_commit_step_waits_then_accepts_or_discards(peer_failed):
    recovery = _recovery(_FakeClient([]))
    batch = Mock(name="batch")
    recovery.step_prepared(batch)
    sample_state = Mock()
    if peer_failed:
        recovery._peer_failed.set()
        with pytest.raises(HostCollectiveInterrupted):
            recovery.commit_step(sample_state)
    else:
        recovery.commit_step(sample_state)
    sample_state.sampler_event.synchronize.assert_called_once()
    assert recovery._uncommitted_step is (batch if peer_failed else None)


def test_commit_step_without_prepared_step_does_not_wait():
    recovery = _recovery(_FakeClient([]))
    sample_state = Mock()
    recovery.commit_step(sample_state)
    sample_state.sampler_event.synchronize.assert_not_called()


def test_peer_failure_flags_aborts_collectives_then_revokes_host_communicators():
    recovery = _recovery(_FakeClient([]))
    seen = []
    engine = recovery._executor.model_engine
    engine.moe_execution_state.request_abort.side_effect = lambda: seen.append(
        recovery._peer_failed.is_set()
    )
    recovery._on_peer_failure()
    assert seen == [True]
    assert recovery._executor._events == ["revoke"]


def test_peer_failure_revokes_even_if_the_abort_request_fails():
    recovery = _recovery(_FakeClient([]))
    engine = recovery._executor.model_engine
    engine.moe_execution_state.request_abort.side_effect = RuntimeError("context error")
    with pytest.raises(RuntimeError):
        recovery._on_peer_failure()
    assert recovery._executor._events == ["revoke"]


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
    with pytest.raises(HostCollectiveInterrupted):
        executor._raise_if_peer_failed("forward", _interrupted())
    recovery._peer_failed.set()
    with pytest.raises(HostCollectiveInterrupted):
        executor._raise_if_peer_failed("forward", RuntimeError("garbage"))


def _sent_responses(recovery):
    sent = []
    recovery._executor._enqueue_responses = lambda responses: sent.append(list(responses))
    return sent


def test_routed_requests_are_tracked_until_their_final_response():
    recovery = _recovery(_FakeClient([]), rank=0)
    item = Mock(id=11, request=Mock(client_id="c11"))
    recovery.requests_routed({FAILED_RANK: [item]})
    assert recovery._request_owners == {11: (FAILED_RANK, item)}
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
    item_12 = Mock(id=12)
    recovery._request_owners = {
        11: (FAILED_RANK, Mock(id=11, request=Mock(client_id="c11"))),
        12: (3, item_12),
    }
    recovery._executor.active_requests.append(
        Mock(is_child=False, py_request_id=12, is_attention_dp_dummy=False)
    )
    sent = _sent_responses(recovery)

    recovery._recover(_interrupted())

    ((request_id, response),) = sent[-1]
    assert request_id == 11 and response.client_id == "c11"
    assert f"rank {FAILED_RANK} failed" in response.error_msg
    assert recovery._request_owners == {12: (3, item_12)}


def test_recovery_readmits_requests_that_never_reached_their_rank():
    """A torn broadcast can leave a routed request unknown to its live owner."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=0)
    torn, delivered = Mock(id=21), Mock(id=22)
    recovery._request_owners = {21: (3, torn), 22: (1, delivered)}
    held = {1: {22}, 3: set()}
    recovery._executor.dist.allgather.side_effect = lambda value: [
        held.get(rank, value) for rank in range(WORLD_SIZE)
    ]
    _sent_responses(recovery)

    recovery._recover(_interrupted())

    assert recovery._executor.request_accumulated[:1] == [torn]
    assert recovery._request_owners == {22: (1, delivered)}


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


def _supported_executor(kv_cache_manager_cls=KVCacheManager):
    load_balancer = Mock()
    load_balancer.is_static_routing.return_value = True
    load_balancer.covers_loss_of.side_effect = lambda rank: rank in (1, 2)
    kv_cache_manager = kv_cache_manager_cls.__new__(kv_cache_manager_cls)
    return SimpleNamespace(
        model_engine=SimpleNamespace(
            moe_load_balancer=load_balancer, moe_execution_state=Mock(), spec_config=None
        ),
        kv_cache_manager=kv_cache_manager,
        resource_manager=Mock(get_resource_manager=Mock(return_value=None)),
        _disable_mpi=False,
        dist=Mock(
            rank=0,
            world_size=4,
            mapping=SimpleNamespace(enable_lm_head_tp_in_adp=False, moe_tp_size=1),
        ),
        adp_router=DefaultADPRouter.__new__(DefaultADPRouter),
        drafter=None,
        draft_model_engine=None,
        guided_decoder=None,
        max_beam_width=1,
        kv_cache_transceiver=None,
        kv_connector_manager=None,
        disable_overlap_scheduler=True,
        attention_dp_enable_balance=False,
        hang_detector=SimpleNamespace(timeout=300),
        llm_args=SimpleNamespace(sleep_config=None),
    )


@pytest.mark.parametrize(
    "unsupported, requirement",
    [
        (lambda ex: setattr(ex, "drafter", Mock()), "no speculative decoding"),
        (lambda ex: setattr(ex.model_engine, "spec_config", Mock()), "no speculative decoding"),
        (lambda ex: setattr(ex, "guided_decoder", Mock()), "no guided decoding"),
        (lambda ex: setattr(ex, "max_beam_width", 2), "no beam search"),
        (lambda ex: setattr(ex, "kv_connector_manager", Mock()), "no KV connector"),
        (lambda ex: setattr(ex, "disable_overlap_scheduler", False), "no overlap scheduler"),
        (
            lambda ex: setattr(ex, "kv_cache_manager", Mock()),
            "KV cache manager V1 or V2",
        ),
        (
            lambda ex: setattr(ex.resource_manager.get_resource_manager, "return_value", Mock()),
            "no PEFT",
        ),
        (
            lambda ex: setattr(ex.dist.mapping, "enable_lm_head_tp_in_adp", True),
            "no LM head TP",
        ),
        (lambda ex: setattr(ex.dist.mapping, "moe_tp_size", 2), "expert TP 1"),
        (lambda ex: setattr(ex, "attention_dp_enable_balance", True), "no attention DP balancing"),
        (lambda ex: setattr(ex.hang_detector, "timeout", 60), "hang detection timeout"),
    ],
)
def test_create_rejects_configurations_a_discarded_step_cannot_rerun(
    monkeypatch, unsupported, requirement
):
    monkeypatch.setattr(peer_loss_recovery, "recovery_enabled", lambda: True)
    executor = _supported_executor()
    unsupported(executor)
    with pytest.raises(ValueError, match=requirement):
        PeerLossRecovery.create(executor)
    executor.dist.enable_peer_loss_recovery.assert_not_called()


def test_create_rejects_a_single_device_connection(monkeypatch):
    monkeypatch.setattr(peer_loss_recovery, "recovery_enabled", lambda: True)
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    with pytest.raises(ValueError, match="CUDA_DEVICE_MAX_CONNECTIONS"):
        PeerLossRecovery.create(_supported_executor())


@pytest.mark.parametrize("kv_cache_manager_cls", [KVCacheManager, KVCacheManagerV2])
def test_create_registers_the_ranks_whose_loss_the_placement_covers(
    monkeypatch, kv_cache_manager_cls
):
    monkeypatch.setattr(peer_loss_recovery, "recovery_enabled", lambda: True)
    registered = []
    monkeypatch.setattr(
        peer_loss_recovery,
        "_register",
        lambda rank, recoverable: registered.append((rank, recoverable)) or (Mock(), frozenset()),
    )
    executor = _supported_executor(kv_cache_manager_cls)
    assert isinstance(PeerLossRecovery.create(executor), PeerLossRecovery)
    assert registered == [(0, (1, 2))]
    executor.dist.enable_peer_loss_recovery.assert_called_once()

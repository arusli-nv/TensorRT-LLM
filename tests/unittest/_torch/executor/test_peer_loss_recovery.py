# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side WideEP peer-loss recovery with fake distributed, model, and authority."""

import signal
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from tensorrt_llm._torch.distributed.communicator import HostCollectiveInterrupted, MPIDist
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm._torch.pyexecutor import peer_loss_recovery
from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    CONTROL_REQUEST_ID,
    RequestQueueItem,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmResponse
from tensorrt_llm._torch.pyexecutor.peer_loss_recovery import PeerLossRecovery, RecoveryHook
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
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
    executor.iter_counter = 0
    events = []
    dist = Mock(rank=rank, live_ranks=tuple(r for r in range(WORLD_SIZE) if r not in failed))
    dist.revoke.side_effect = lambda: events.append("revoke")
    dist.shrink.side_effect = lambda: events.append("shrink") or failed
    dist.install_survivors.side_effect = lambda: events.append("survivors")
    dist.watch_peer_failures.side_effect = lambda callback, abort_word: events.append(
        "watch"
    ) or Mock(failed=False)
    dist.broadcast.side_effect = lambda value, root: value
    dist.allgather.side_effect = lambda value, departed: [value] * WORLD_SIZE
    engine = Mock()
    engine.moe_load_balancer.masked_placement_digest.return_value = "digest"
    engine.moe_load_balancer.reconfigure_mask_only.side_effect = lambda failed: events.append(
        ("placement", frozenset(failed))
    )
    engine.moe_execution_state.health = EPGroupHealth(WORLD_SIZE)
    executor.dist = dist
    executor.model_engine = engine
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.wake.side_effect = lambda: events.append("wake")
    executor.active_requests = []
    executor.waiting_queue = FCFSWaitingQueue()
    executor.request_accumulated = []
    executor._pending_transfer_responses = []
    executor._pending_response_terminations = []
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
    recovery._watch = Mock(failed=False)
    executor.discard_interrupted_step = Mock()
    executor._peer_loss_recovery = recovery
    return recovery


def _interrupted(failed=frozenset({FAILED_RANK})):
    return HostCollectiveInterrupted("tp_allgather", failed)


def test_recovery_runs_protocol_and_installs_in_order():
    client = _FakeClient(_happy_replies(generation=1))
    recovery = _recovery(client)
    request = Mock(is_attention_dp_dummy=False)
    recovery._executor.active_requests.append(request)
    recovery._peer_failed.set()

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
        "survivors",
        ("placement", failed),
        "watch",
    ]
    assert recovery._executor.active_requests == [request]
    assert not recovery._peer_failed.is_set()
    masked_placement_digest = (
        recovery._executor.model_engine.moe_load_balancer.masked_placement_digest
    )
    masked_placement_digest.assert_called_once_with(failed)


def _queued(request_id):
    return RequestQueueItem(request_id, Mock())


def _broadcast(recovery, items, interrupted=False):
    """Broadcast ``items`` through the recovery hook, as the executor does."""
    if not interrupted:
        recovery.broadcast_requests(lambda _: None, items)
        return
    with pytest.raises(HostCollectiveInterrupted):
        recovery.broadcast_requests(Mock(side_effect=_interrupted()), items)


@pytest.mark.parametrize("rank", [0, 1])
def test_recovery_keeps_unscheduled_requests_on_rank0_only(rank):
    """Rank 0 re-broadcasts received-but-unscheduled requests after recovery."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=rank)
    executor = recovery._executor
    executor.request_accumulated = [_queued(5)]
    executor.waiting_queue.add_requests([_queued(1), _queued(2)])
    _broadcast(recovery, [_queued(3)], interrupted=True)

    recovery._recover(_interrupted())

    assert len(executor.waiting_queue) == 0
    assert recovery._unbroadcast_requests == []
    expected = [1, 2, 3, 5] if rank == 0 else [5]
    assert [item.id for item in executor.request_accumulated] == expected


def test_recovery_does_not_rebroadcast_a_request_a_survivor_started():
    """Rank 0's broadcast can complete on a survivor while rank 0 sees it interrupted."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=0)
    executor = recovery._executor
    _broadcast(recovery, [_queued(3), _queued(4)], interrupted=True)
    served = [(set(), 0), ({3}, 0), None, (set(), 0)]
    executor.dist.allgather.side_effect = lambda value, departed: served

    recovery._recover(_interrupted())

    assert [item.id for item in executor.request_accumulated] == [4]
    if recovery._request_owners is not None:
        assert recovery._request_owners[3][0] == 1


def test_survivors_resume_from_the_same_iteration():
    """A loss can stop survivors one iteration apart; they resume from the furthest one."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=1)
    executor = recovery._executor
    executor.iter_counter = 7
    executor.dist.allgather.side_effect = lambda value, departed: [
        (set(), 8),
        value,
        None,
        (set(), 7),
    ]

    recovery._recover(_interrupted())

    assert executor.iter_counter == 8


def test_recovery_reports_requests_it_already_answered_as_held():
    """A request a survivor rejected from an interrupted broadcast must not be admitted again."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=1)
    executor = recovery._executor
    rejected = Mock(spec=LlmResponse, request_id=3, result=None)
    executor._pending_transfer_responses = [(3, rejected)]
    reported = []
    executor.dist.allgather.side_effect = (
        lambda value, departed: reported.append(value) or [value] * WORLD_SIZE
    )

    recovery._recover(_interrupted())

    held, _iteration = reported[0]
    assert 3 in held


def test_recovery_rebroadcasts_the_cancellations_of_an_interrupted_broadcast():
    """Rank 0 may finish a broadcast that a survivor never received."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=0)
    executor = recovery._executor
    cancel = RequestQueueItem(7, None, is_canceled_request=True)
    _broadcast(recovery, [cancel])
    executor.waiting_queue.add_requests([_queued(1)])

    recovery._recover(_interrupted())

    assert [item.id for item in executor.request_accumulated] == [1, 7]


def test_recovery_fails_closed_on_an_interrupted_control_request():
    recovery = _recovery(_FakeClient(_happy_replies()), rank=0)
    _broadcast(recovery, [RequestQueueItem(CONTROL_REQUEST_ID)])
    with pytest.raises(RuntimeError, match="control request"):
        recovery._recover(_interrupted())


def test_gather_numbers_follow_rank0_after_a_recovery():
    """A rank ahead of rank 0 numbers its next gathers like rank 0."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=1)
    for number in range(3):
        recovery.gathering_responses([(number, Mock())])
    recovery._executor.dist.broadcast.side_effect = lambda value, root: 2
    _sent_responses(recovery)

    recovery._recover(_interrupted())

    assert recovery._gather_count == recovery._last_processed_gather == 2


def test_requests_with_occurrence_penalties_are_rejected():
    request = Mock(
        sampling_config=Mock(repetition_penalty=1.0, presence_penalty=0.5, frequency_penalty=None)
    )
    with pytest.raises(ValueError, match="penalties"):
        PeerLossRecovery.validate_request(request)
    request.sampling_config.presence_penalty = 0.0
    PeerLossRecovery.validate_request(request)


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
    ],
    ids=["wrong_failed_rank", "not_a_survivor", "install_gen", "commit_gen"],
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
            r._executor.model_engine.moe_load_balancer.reconfigure_mask_only,
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
    recovery._requeue_unscheduled_requests = Mock()
    recovery._peer_failed.set()
    inject(recovery)
    with pytest.raises(RuntimeError):
        recovery._recover(_interrupted())
    assert "watch" not in recovery._executor._events
    recovery._requeue_unscheduled_requests.assert_not_called()
    assert recovery.peer_failed


def test_run_surfaces_a_failed_recovery_after_stopping_the_watch():
    recovery = _recovery(_FakeClient([Abort("authority stopped")]))
    watch = recovery._watch
    recovery._executor.dist.watch_peer_failures.side_effect = lambda callback, abort_word: watch
    with pytest.raises(RecoveryAborted, match="authority stopped"):
        recovery.run(Mock(side_effect=_interrupted()))
    watch.stop.assert_called()


def test_recovery_discards_the_step_after_joining_its_device_work():
    recovery = _recovery(_FakeClient(_happy_replies()))
    calls = []
    batch = Mock(name="batch")
    recovery.step_prepared(batch)
    recovery._join_device_work = lambda deadline: calls.append("join")
    recovery._executor.discard_interrupted_step = lambda step: calls.append(("discard", step))

    recovery._recover(_interrupted())

    assert calls == ["join", ("discard", batch)]
    assert recovery._uncommitted_step is None


def test_run_recovers_and_resumes():
    recovery = _recovery(_FakeClient([]))
    error = _interrupted()
    event_loop = Mock(side_effect=[error, None])
    recovered = []
    recovery._recover = recovered.append

    recovery.run(event_loop)

    assert event_loop.call_count == 2
    assert recovered == [error]
    # The hang detector bounds recovery.
    hang_detector = recovery._executor.hang_detector
    hang_detector.__enter__.assert_called_once()
    hang_detector.__enter__.return_value.checkpoint.assert_called_once()
    hang_detector.__exit__.assert_called_once()


def test_only_an_admitted_serving_rank_outside_recovery_exits_alone():
    unadmitted = _recovery(_FakeClient([]), rank=0)
    unadmitted.run(Mock(return_value=None))
    assert not unadmitted.survives_own_loss

    recovery = _recovery(_FakeClient([]))
    # Not before the serving loop watches its peers.
    assert not recovery.survives_own_loss
    seen = []
    recovery.run(Mock(side_effect=lambda: seen.append(recovery.survives_own_loss)))
    assert seen == [True]

    recovery = _recovery(_FakeClient([]))
    recovery._recover = Mock()
    recovery.run(Mock(side_effect=[_interrupted(), None]))
    # A second loss is terminal, so this rank no longer exits alone.
    assert not recovery.survives_own_loss


@pytest.mark.parametrize(
    "message, killed",
    [
        ("CUDA error: an illegal memory access was encountered", True),
        ("CUDA out of memory", False),
        ("alltoall timeout", False),
        ("a bug every rank hits", False),
    ],
)
def test_crashed_loop_exits_only_this_rank_when_peers_can_recover(monkeypatch, message, killed):
    from tensorrt_llm._torch.pyexecutor import py_executor

    recovery = _recovery(_FakeClient([]))
    recovery.survives_own_loss = True
    executor = _executor()
    executor._peer_loss_recovery = recovery
    executor._run_event_loop = Mock(side_effect=RuntimeError(message))
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
    assert kills == ([signal.SIGKILL] if killed else [])


def test_run_does_not_recover_during_kv_cache_estimation():
    recovery = _recovery(_FakeClient([]))
    recovery._executor.gather_all_responses = True
    with pytest.raises(HostCollectiveInterrupted):
        recovery.run(Mock(side_effect=_interrupted()))
    recovery._executor.dist.watch_peer_failures.assert_not_called()


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


def test_commit_step_discards_a_step_whose_peer_failed_while_it_ran():
    """The flag is read after waiting for the step's device work, not before."""
    recovery = _recovery(_FakeClient([]))
    recovery.step_prepared(Mock(name="batch"))
    sample_state = Mock()
    sample_state.sampler_event.synchronize.side_effect = recovery._peer_failed.set
    with pytest.raises(HostCollectiveInterrupted):
        recovery.commit_step(sample_state)


def test_recovery_aborts_collectives_and_delivers_only_after_commit():
    client = _FakeClient(_happy_replies())
    recovery = _recovery(client)
    executor = recovery._executor
    events = executor._events
    executor.model_engine.moe_execution_state.request_abort.side_effect = lambda: events.append(
        "abort"
    )
    client_wait = client.wait
    client.wait = lambda message_type, deadline: (
        events.append(message_type.__name__) or client_wait(message_type, deadline)
    )
    executor._enqueue_responses = lambda responses: events.append("deliver")

    recovery._recover(_interrupted())

    assert events.index("abort") < events.index("shrink")
    assert events.index("Install") < events.index("survivors")
    assert events.index("Commit") < events.index("deliver")
    assert not recovery._gathers


@pytest.mark.parametrize("rank,events", [(0, ["revoke", "wake"]), (1, ["revoke"])])
def test_peer_failure_releases_host_waits(rank, events):
    recovery = _recovery(_FakeClient([]), rank=rank)
    recovery._on_peer_failure()
    assert recovery.peer_failed
    assert recovery._executor._events == events


def test_watch_aborts_the_moe_collectives_natively():
    recovery = _recovery(_FakeClient([]))
    abort = recovery._executor.model_engine.moe_execution_state.abort
    abort.data_ptr.return_value = 4096
    recovery._watch_peer_failures()
    recovery._executor.dist.watch_peer_failures.assert_called_once_with(
        recovery._on_peer_failure, 4096
    )


def test_a_failure_seen_only_by_the_watch_counts():
    recovery = _recovery(_FakeClient([]))
    recovery._watch = Mock(failed=False)
    assert not recovery.peer_failed
    recovery._watch.failed = True
    assert recovery.peer_failed


def test_discard_returns_step_resources_and_padding_request():
    executor = _executor()
    executor.kv_cache_manager = Mock()
    executor.inflight_req_ids = Mock()
    executor._pending_adp_dummy_request = None
    batch = Mock(name="batch")
    request = Mock(is_attention_dp_dummy=False)
    dummy = Mock(is_attention_dp_dummy=True, py_request_id=99)
    executor.active_requests.extend([request, dummy])

    executor.discard_interrupted_step(batch)

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

    executor.discard_interrupted_step(None)

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
        executor.publish_responses([], [finished])
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
    item = Mock(id=11, request=Mock(client_id="c11"), child_req_ids=[21])
    recovery.requests_routed({FAILED_RANK: [item]})
    assert recovery._request_owners == {11: (FAILED_RANK, item)}
    streaming = Mock(spec=LlmResponse, request_id=11, result=Mock(is_final=False))
    streaming.has_error.return_value = False
    recovery.responses_gathered([(11, streaming)])
    assert 11 in recovery._request_owners
    # The last sequence of an n > 1 request is gathered under its own id.
    final = Mock(spec=LlmResponse, request_id=11, result=Mock(is_final=True))
    final.has_error.return_value = False
    recovery.responses_gathered([(21, final)])
    assert recovery._request_owners == {} and recovery._parent_ids == {}


@pytest.mark.parametrize("sequence_id", [11, 21])
def test_an_error_of_any_sequence_ends_the_request(sequence_id):
    recovery = _recovery(_FakeClient([]), rank=0)
    recovery.requests_routed({1: [Mock(id=11, child_req_ids=[21])]})
    error = Mock(spec=LlmResponse, request_id=sequence_id, result=None)
    error.has_error.return_value = True
    recovery.responses_gathered([(sequence_id, error)])
    assert recovery._request_owners == {} and recovery._parent_ids == {}


def test_recovery_fails_requests_owned_by_departed_rank():
    """Rank 0 fails the requests whose KV cache left with the departed rank, once."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=0)
    item_12 = Mock(id=12, child_req_ids=None)
    recovery._request_owners = {
        11: (FAILED_RANK, Mock(id=11, request=Mock(client_id="c11"), child_req_ids=None)),
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
    torn, delivered = Mock(id=21, child_req_ids=None), Mock(id=22, child_req_ids=None)
    recovery._request_owners = {21: (3, torn), 22: (1, delivered)}
    held = {1: ({22}, 0), 3: (set(), 0)}
    recovery._executor.dist.allgather.side_effect = lambda value, departed: [
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


def test_recovery_fails_closed_if_an_owed_gather_is_no_longer_held():
    recovery = _recovery(_FakeClient(_happy_replies()), rank=1)
    for number in range(peer_loss_recovery._GATHER_HISTORY + 1):
        recovery.gathering_responses([(number, Mock())])
    recovery._executor.dist.broadcast.side_effect = lambda value, root: 0
    with pytest.raises(RuntimeError, match="no longer held"):
        recovery._recover(_interrupted())


def test_recovery_fails_closed_if_gather_numbers_diverged():
    """Rank 0 cannot have processed a gather this rank never made."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=1)
    recovery.gathering_responses([(5, Mock())])
    recovery._executor.dist.broadcast.side_effect = lambda value, root: 2
    with pytest.raises(RuntimeError, match="made only 1"):
        recovery._recover(_interrupted())


def test_an_unrecoverable_loss_fails_before_waiting_on_the_device():
    recovery = _recovery(_FakeClient(_happy_replies()), failed=frozenset({0}))
    with pytest.raises(RuntimeError, match="rank 0 is not recoverable"):
        recovery._recover(_interrupted(frozenset({0})))
    recovery._join_device_work.assert_not_called()


def test_joining_device_work_reports_a_device_error(monkeypatch):
    recovery = _recovery(_FakeClient([]))

    def fail(device):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", fail)
    with pytest.raises(RuntimeError, match="illegal memory access"):
        PeerLossRecovery._join_device_work(recovery, time.monotonic() + 5)


def test_joining_device_work_is_bounded(monkeypatch):
    recovery = _recovery(_FakeClient([]))
    release = threading.Event()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: release.wait())
    try:
        with pytest.raises(TimeoutError):
            PeerLossRecovery._join_device_work(recovery, time.monotonic() + 0.05)
    finally:
        release.set()


def test_recovery_delivers_buffered_error_responses_before_readmission():
    """A request that failed on its rank is retired, not re-admitted as never received."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=1)
    executor = recovery._executor
    failed_request, error = Mock(name="failed"), Mock(name="error")
    executor._pending_transfer_responses = [(9, error)]
    executor._pending_response_terminations = [failed_request]
    order = []
    executor._enqueue_responses = lambda responses: order.append(("send", list(responses)))
    executor._terminate_request = lambda request: order.append(("terminate", request))

    recovery._recover(_interrupted())

    assert order == [("send", [(9, error)]), ("terminate", failed_request)]
    assert executor._pending_transfer_responses == []
    assert executor._pending_response_terminations == []


def _supported_executor():
    load_balancer = Mock()
    load_balancer.is_static_routing.return_value = True
    load_balancer.covers_loss_of.side_effect = lambda rank: rank in (1, 2)
    kv_cache_manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    return SimpleNamespace(
        model_engine=SimpleNamespace(
            moe_load_balancer=load_balancer, moe_execution_state=Mock(), spec_config=None
        ),
        kv_cache_manager=kv_cache_manager,
        resource_manager=Mock(get_resource_manager=Mock(return_value=None)),
        dist=Mock(
            spec=MPIDist,
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
        enable_kv_pool_rebalance=False,
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
        (lambda ex: setattr(ex, "enable_kv_pool_rebalance", True), "no KV pool rebalance"),
        (lambda ex: setattr(ex.kv_cache_manager, "enable_block_reuse", True), "no KV block reuse"),
        (
            lambda ex: setattr(ex, "kv_cache_manager", Mock()),
            "KV Cache Manager V2",
        ),
        (
            lambda ex: setattr(ex.resource_manager.get_resource_manager, "return_value", Mock()),
            "no PEFT",
        ),
        (
            lambda ex: setattr(
                ex.resource_manager.get_resource_manager,
                "side_effect",
                lambda kind: Mock() if kind.name == "CROSS_KV_CACHE_MANAGER" else None,
            ),
            "no encoder-decoder cross KV cache",
        ),
        (
            lambda ex: setattr(ex.dist.mapping, "enable_lm_head_tp_in_adp", True),
            "no LM head TP",
        ),
        (lambda ex: setattr(ex.dist.mapping, "moe_tp_size", 2), "expert TP 1"),
        (lambda ex: setattr(ex, "attention_dp_enable_balance", True), "no attention DP balancing"),
        (lambda ex: setattr(ex.hang_detector, "timeout", 60), "hang detection timeout"),
        (lambda ex: setattr(ex, "dist", Mock(mapping=ex.dist.mapping)), "MPI executor"),
    ],
)
def test_create_rejects_configurations_a_discarded_step_cannot_rerun(
    monkeypatch, unsupported, requirement
):
    monkeypatch.setattr(peer_loss_recovery, "peer_loss_recovery_enabled", lambda: True)
    executor = _supported_executor()
    unsupported(executor)
    with pytest.raises(ValueError, match=requirement):
        PeerLossRecovery.create(executor)
    executor.dist.enable_peer_loss_recovery.assert_not_called()


def test_create_returns_a_pass_through_hook_when_disabled(monkeypatch):
    monkeypatch.setattr(peer_loss_recovery, "peer_loss_recovery_enabled", lambda: False)
    executor = _supported_executor()
    recovery = PeerLossRecovery.create(executor)
    assert type(recovery) is RecoveryHook
    assert not (recovery.peer_failed or recovery.exits_alone(RuntimeError("illegal memory access")))
    responses = [(1, Mock())]
    assert recovery.gathering_responses(responses) is responses
    event_loop = Mock()
    recovery.run(event_loop)
    event_loop.assert_called_once_with()
    executor.dist.enable_peer_loss_recovery.assert_not_called()


def test_create_registers_the_ranks_whose_loss_the_placement_covers(monkeypatch):
    monkeypatch.setattr(peer_loss_recovery, "peer_loss_recovery_enabled", lambda: True)
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


def test_install_resident_placement_reroutes_before_masking():
    events = []
    load_balancer = Mock()
    load_balancer.reconfigure_mask_only.side_effect = lambda dead: events.append(("routed", dead))
    health = EPGroupHealth(WORLD_SIZE)
    state = Mock(health=health)
    state.install_mask.side_effect = lambda: events.append(("mask", health.get_mask_words()))
    install = peer_loss_recovery.install_resident_placement

    install(state, load_balancer, {FAILED_RANK}, own_rank=0)

    assert health.get_failed_ranks() == frozenset({FAILED_RANK})
    # The kernels skip routes to a masked rank, so routing must exclude it first.
    assert events == [("routed", [FAILED_RANK]), ("mask", (0b1011, 0, 0, 0))]
    state.rearm.assert_called_once()

    # Invalid requests are refused before anything changes.
    events.clear()
    with pytest.raises(RuntimeError, match="already failed"):
        install(state, load_balancer, {FAILED_RANK}, own_rank=0)
    with pytest.raises(RuntimeError, match="this rank"):
        install(state, load_balancer, {0}, own_rank=0)
    assert events == [] and health.get_failed_ranks() == frozenset({FAILED_RANK})

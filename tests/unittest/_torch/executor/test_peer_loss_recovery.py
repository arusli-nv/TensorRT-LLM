# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side WideEP peer-loss recovery with fake distributed, model, and authority."""

import functools
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
from tensorrt_llm._torch.pyexecutor.peer_loss_recovery import (
    PeerLossRecovery,
    PreparedStep,
    RecoveryHook,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import DefaultADPRouter
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode as SpecMode
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
from tensorrt_llm.runtime.kv_cache_manager_v2 import GPU_LEVEL

pytestmark = pytest.mark.cpu_only

WORLD_SIZE = 4
FAILED_RANK = 2
SURVIVORS = (0, 1, 3)
ADMITTED = frozenset({frozenset({1}), frozenset({FAILED_RANK}), frozenset({3})})


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


def _happy_replies(generation=1, failed=(FAILED_RANK,), survivors=SURVIVORS):
    return [
        Propose(generation=generation, failed_ranks=failed, survivors=survivors),
        Install(generation=generation),
        Commit(generation=generation),
    ]


def _executor(rank=1, failed=frozenset({FAILED_RANK}), world_size=WORLD_SIZE):
    executor = PyExecutor.__new__(PyExecutor)
    executor.iter_counter = 0
    events = []
    dist = Mock(rank=rank, live_ranks=tuple(r for r in range(world_size) if r not in failed))
    dist.revoke.side_effect = lambda: events.append("revoke")
    dist.shrink.side_effect = lambda: events.append("shrink") or failed
    dist.install_survivors.side_effect = lambda: events.append("survivors")
    dist.watch_peer_failures.side_effect = lambda callback, abort_word: events.append(
        "watch"
    ) or Mock(failed=False)
    dist.broadcast.side_effect = lambda value, root: value
    dist.allgather.side_effect = lambda value, departed: [
        departed if r in failed else value for r in range(world_size)
    ]
    engine = Mock()
    engine.moe_load_balancer.recovery_placements.return_value = ["placement"]
    engine.moe_load_balancer.restore_backing_bytes.return_value = 0
    engine.moe_load_balancer.placement_digest.return_value = "digest"
    engine.moe_load_balancer.install_recovery_placement.side_effect = (
        lambda failed, placements: events.append(("placement", frozenset(failed)))
    )
    engine.moe_execution_state.health = EPGroupHealth(world_size)
    executor.dist = dist
    executor.model_engine = engine
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.wake.side_effect = lambda: events.append("wake")
    executor.kv_cache_manager = Mock()
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


def _recovery(
    client, rank=1, failed=frozenset({FAILED_RANK}), world_size=WORLD_SIZE, admitted=ADMITTED
):
    executor = _executor(rank, failed, world_size)
    recovery = PeerLossRecovery(executor, client, admitted)
    recovery._join_device_work = Mock()
    recovery._watch = Mock(failed=False)
    executor.discard_interrupted_steps = Mock()
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
        Stopped(rank=1, failed_ranks=(FAILED_RANK,)),
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
    load_balancer = recovery._executor.model_engine.moe_load_balancer
    # A static placement needs no agreed base.
    load_balancer.recovery_placements.assert_called_once_with(failed, None)
    # The placements installed are the ones whose digest went out with Ready.
    load_balancer.placement_digest.assert_called_once_with(failed, ["placement"])
    load_balancer.install_recovery_placement.assert_called_once_with([FAILED_RANK], ["placement"])


def test_borrowed_slots_take_kv_cache_memory_only_when_the_device_is_short(monkeypatch):
    """Borrowed slots get their memory before Ready; the KV cache gives up only what free memory lacks."""
    recovery = _recovery(_FakeClient([]))
    executor = recovery._executor
    executor.shrink_kv_cache_quota = Mock(return_value=True)
    load_balancer = Mock()
    need = 1 << 30
    load_balancer.restore_backing_bytes.return_value = need
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    monkeypatch.setattr(peer_loss_recovery, "_free_device_bytes", lambda: 4 << 30)
    recovery._back_restored_slots(load_balancer, ["placement"])
    executor.shrink_kv_cache_quota.assert_not_called()
    load_balancer.back_restored_slots.assert_called_once_with(["placement"])

    free = 512 << 20
    monkeypatch.setattr(peer_loss_recovery, "_free_device_bytes", lambda: free)
    recovery._back_restored_slots(load_balancer, ["placement"])
    executor.shrink_kv_cache_quota.assert_called_once_with(
        need + peer_loss_recovery._BORROW_MARGIN_BYTES - free
    )

    # A KV cache that cannot shrink ends the recovery before any slot changes.
    executor.shrink_kv_cache_quota.return_value = False
    load_balancer.back_restored_slots.reset_mock()
    with pytest.raises(RuntimeError, match="could not give up"):
        recovery._back_restored_slots(load_balancer, ["placement"])
    load_balancer.back_restored_slots.assert_not_called()


def test_kv_cache_quota_cut_suspends_the_active_caches_around_the_resize():
    executor = PyExecutor.__new__(PyExecutor)
    mgr = Mock()
    mgr.impl.get_quota.return_value = 10 << 30
    mgr.impl.resize.return_value = True
    mgr.is_request_active.side_effect = lambda request_id: request_id == 1
    requests = [Mock(py_request_id=1), Mock(py_request_id=2)]
    executor.kv_cache_manager = mgr
    executor.active_requests = requests
    executor.model_engine = SimpleNamespace(cuda_graph_runner=None)
    executor._read_kv_cache_capacity = Mock(return_value={})

    assert executor.shrink_kv_cache_quota(1 << 30)
    mgr.suspend_request.assert_called_once_with(requests[0])
    mgr.impl.resize.assert_called_once_with(GPU_LEVEL, 9 << 30)
    mgr.resume_request.assert_called_once_with(requests[0])
    # More than the whole quota: nothing to resize.
    assert not executor.shrink_kv_cache_quota(10 << 30)
    assert mgr.impl.resize.call_count == 1
    # A failed resize still resumes the suspended requests.
    mgr.impl.resize.side_effect = RuntimeError("resize")
    with pytest.raises(RuntimeError, match="resize"):
        executor.shrink_kv_cache_quota(1 << 30)
    assert mgr.resume_request.call_count == mgr.suspend_request.call_count == 3


def test_a_node_loss_recovers_every_rank_in_one_transaction():
    """Losing node 1 (ranks 4-7) masks all four ranks and fails the requests each one served."""
    node = frozenset({4, 5, 6, 7})
    client = _FakeClient(_happy_replies(failed=(4, 5, 6, 7), survivors=(0, 1, 2, 3)))
    recovery = _recovery(client, rank=0, failed=node, world_size=8, admitted=ADMITTED | {node})
    recovery._request_owners = {
        request_id: (owner, Mock(id=request_id, request=Mock(client_id=None), child_req_ids=None))
        for request_id, owner in ((11, 4), (12, 7), (13, 2))
    }
    recovery._executor.active_requests.append(
        Mock(is_child=False, py_request_id=13, is_attention_dp_dummy=False)
    )
    sent = _sent_responses(recovery)

    recovery._recover(_interrupted(node))

    assert client.sent[0] == Stopped(rank=0, failed_ranks=(4, 5, 6, 7))
    engine = recovery._executor.model_engine
    engine.moe_load_balancer.recovery_placements.assert_called_once_with(node, None)
    engine.moe_load_balancer.install_recovery_placement.assert_called_once_with(
        [4, 5, 6, 7], ["placement"]
    )
    assert engine.moe_execution_state.health.get_failed_ranks() == node
    assert sorted(request_id for request_id, _ in sent[-1]) == [11, 12]
    assert list(recovery._request_owners) == [13]


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
    recovery = _recovery(_FakeClient([]))
    request = Mock(
        sampling_config=Mock(repetition_penalty=1.0, presence_penalty=0.5, frequency_penalty=None),
        py_multimodal_data=None,
        py_logits_post_processors=None,
    )
    with pytest.raises(ValueError, match="penalties"):
        recovery.validate_request(request)
    request.sampling_config.presence_penalty = 0.0
    recovery.validate_request(request)
    # A logits post-processor may keep state, and a replayed step would run it twice.
    request.py_logits_post_processors = [Mock()]
    with pytest.raises(ValueError, match="post-processors"):
        recovery.validate_request(request)


@pytest.mark.parametrize("overlap", [False, True])
def test_multimodal_requests_are_rejected_only_under_the_overlap_scheduler(overlap):
    recovery = _recovery(_FakeClient([]))
    recovery._executor.disable_overlap_scheduler = not overlap
    request = Mock(
        sampling_config=Mock(repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0),
        py_multimodal_data={"image": Mock()},
        py_logits_post_processors=None,
    )
    if overlap:
        with pytest.raises(ValueError, match="Multimodal"):
            recovery.validate_request(request)
    else:
        recovery.validate_request(request)


@pytest.mark.parametrize("failed", [frozenset(), frozenset({1, 2}), frozenset({2, 3})])
def test_recovery_requires_an_admitted_failed_set(failed):
    client = _FakeClient(_happy_replies())
    recovery = _recovery(client, rank=0, failed=failed)
    with pytest.raises(RuntimeError, match="is not recoverable"):
        recovery._recover(_interrupted())
    assert client.sent == []


@pytest.mark.parametrize(
    "replies, match",
    [
        ([Propose(generation=1, failed_ranks=(3,), survivors=(0, 1, 2))], "authority proposed"),
        ([Propose(generation=1, failed_ranks=(2, 3), survivors=(0, 1))], "authority proposed"),
        ([Propose(generation=1, failed_ranks=(2,), survivors=(0, 3))], "authority proposed"),
        (_happy_replies()[:1] + [Install(generation=2)], "Expected recovery generation 1"),
        (_happy_replies()[:2] + [Commit(generation=2)], "Expected recovery generation 1"),
    ],
    ids=["wrong_failed_rank", "wrong_failed_set", "not_a_survivor", "install_gen", "commit_gen"],
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
            r._executor.model_engine.moe_load_balancer.recovery_placements,
            "side_effect",
            _fail(RuntimeError("placement")),
        ),
        lambda r: setattr(
            r._executor.dist.install_survivors, "side_effect", _fail(RuntimeError("survivors"))
        ),
        lambda r: setattr(
            r._executor.model_engine.moe_load_balancer.install_recovery_placement,
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


def test_recovery_discards_the_steps_newest_first_after_joining_their_device_work():
    recovery = _recovery(_FakeClient(_happy_replies()))
    calls = []
    older, newer = Mock(name="older"), Mock(name="newer")
    recovery.step_prepared(older)
    recovery.step_prepared(newer)
    recovery._join_device_work = lambda deadline: calls.append("join")
    recovery._executor.discard_interrupted_steps = lambda steps: calls.append(
        ("discard", [step.batch for step in steps])
    )

    recovery._recover(_interrupted())

    assert calls == ["join", ("discard", [newer, older])]
    assert not recovery._uncommitted_steps


def test_steps_commit_oldest_first_and_release_the_newest():
    recovery = _recovery(_FakeClient([]))
    older, newer = Mock(name="older"), Mock(name="newer")
    recovery.step_prepared(older)
    recovery.step_prepared(newer)
    with pytest.raises(RuntimeError, match="order"):
        recovery.commit_step(newer, Mock())
    recovery.step_released()
    recovery.commit_step(older, Mock())
    assert not recovery._uncommitted_steps


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


def test_run_reports_hang_evidence_only_when_enabled(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import hang_evidence
    from tensorrt_llm.executor.recovery import client as client_module

    started, stopped, closed = [], [], []
    monkeypatch.setattr(hang_evidence, "ProgressRowReader", lambda *args: Mock(read=lambda: None))
    monkeypatch.setattr(
        hang_evidence.HangEvidenceReporter, "start", lambda self: started.append(self)
    )
    monkeypatch.setattr(
        hang_evidence.HangEvidenceReporter, "stop", lambda self: stopped.append(self)
    )
    monkeypatch.setattr(
        client_module.RecoveryClient,
        "from_env",
        classmethod(lambda cls, rank: Mock(close=lambda: closed.append(rank))),
    )
    recovery = _recovery(_FakeClient([]))
    recovery._executor.model_engine.moe_execution_state.progress_comm.progress_row_offsets.return_value = (
        0,
        64,
        128,
    )

    monkeypatch.delenv(hang_evidence.HANG_DETECTION_ENV, raising=False)
    recovery.run(Mock(return_value=None))
    assert started == []

    monkeypatch.setenv(hang_evidence.HANG_DETECTION_ENV, "1")
    recovery.run(Mock(return_value=None))
    assert len(started) == 1 and stopped == started and closed == [1]


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
            recovery.commit_step(batch, sample_state)
    else:
        recovery.commit_step(batch, sample_state)
    sample_state.sampler_event.synchronize.assert_called_once()
    assert [step.batch for step in recovery._uncommitted_steps] == ([batch] if peer_failed else [])


def test_commit_step_discards_a_step_whose_peer_failed_while_it_ran():
    """The flag is read after waiting for the step's device work, not before."""
    recovery = _recovery(_FakeClient([]))
    batch = Mock(name="batch")
    recovery.step_prepared(batch)
    sample_state = Mock()
    sample_state.sampler_event.synchronize.side_effect = recovery._peer_failed.set
    with pytest.raises(HostCollectiveInterrupted):
        recovery.commit_step(batch, sample_state)


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
    step = PreparedStep(Mock(name="batch"), {7: 64})
    request = Mock(is_attention_dp_dummy=False)
    dummy = Mock(is_attention_dp_dummy=True, py_request_id=99)
    executor.active_requests.extend([request, dummy])

    executor.discard_interrupted_steps([step])

    executor.kv_cache_manager.revert_prepare_resources.assert_called_once_with(step.batch, {7: 64})
    executor._terminate_request.assert_called_once_with(dummy)
    assert executor.active_requests == [request]
    assert executor.previous_batch is None and request.py_batch_idx is None


def test_discard_undoes_each_step_s_request_changes_newest_first():
    """Under the overlap scheduler two steps can be uncommitted; each one's changes are undone in reverse."""
    executor = _executor()
    executor.inflight_req_ids = Mock()
    executor._pending_adp_dummy_request = None
    request = Mock(is_attention_dp_dummy=False, context_current_position=0, state="CONTEXT_INIT")
    older, newer = PreparedStep(Mock(), {}), PreparedStep(Mock(), {})
    for step, position, state in (
        (older, 64, "CONTEXT_INIT"),
        (newer, 128, "GENERATION_IN_PROGRESS"),
    ):
        executor._step_changes = step.changes
        executor._note_step_change(request)
        request.context_current_position, request.state = position, state

    executor.discard_interrupted_steps([newer, older])

    assert (request.context_current_position, request.state) == (0, "CONTEXT_INIT")
    assert executor._step_changes is None


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

    executor.discard_interrupted_steps([])

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


def test_rank0_reports_how_many_requests_the_departed_rank_was_serving():
    """The authority's status counts the requests a loss failed; rank 0 routed them all."""
    client = _FakeClient(_happy_replies())
    recovery = _recovery(client, rank=0)
    recovery._request_owners = {
        11: (FAILED_RANK, Mock(id=11, child_req_ids=None)),
        12: (FAILED_RANK, Mock(id=12, child_req_ids=None)),
        13: (1, Mock(id=13, child_req_ids=None)),
    }
    _sent_responses(recovery)

    recovery._recover(_interrupted())

    (installed,) = [message for message in client.sent if isinstance(message, Installed)]
    assert installed.failed_requests == 2


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
    with pytest.raises(RuntimeError, match=r"ranks \[0\] is not recoverable"):
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
        for thread in threading.enumerate():
            if thread.name == "peer_loss_device_join":
                thread.join(timeout=5)


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
    load_balancer.covers_loss_of.side_effect = lambda dead: tuple(dead) in ((1,), (2,), (2, 3))
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
            mapping=SimpleNamespace(enable_lm_head_tp_in_adp=False, moe_tp_size=1, gpus_per_node=2),
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
        (lambda ex: setattr(ex, "drafter", Mock()), "no speculative decoding but one-model MTP"),
        (
            lambda ex: setattr(ex.model_engine, "spec_config", _spec_config(SpecMode.MTP)),
            "no speculative decoding but one-model MTP",
        ),
        (
            lambda ex: setattr(
                ex.model_engine, "spec_config", _spec_config(SpecMode.EAGLE3_ONE_MODEL)
            ),
            "no speculative decoding but one-model MTP",
        ),
        (
            lambda ex: setattr(
                ex.model_engine,
                "spec_config",
                _spec_config(SpecMode.MTP_EAGLE_ONE_MODEL, use_dynamic_tree=True),
            ),
            "no speculative decoding but one-model MTP",
        ),
        (lambda ex: setattr(ex, "guided_decoder", Mock()), "no guided decoding"),
        (lambda ex: setattr(ex, "max_beam_width", 2), "no beam search"),
        (lambda ex: setattr(ex, "kv_connector_manager", Mock()), "no KV connector"),
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
        (lambda ex: setattr(ex.model_engine, "moe_load_balancer", None), "an EPLB load balancer"),
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


def test_failure_domains_are_every_rank_then_every_node_but_rank_0_s():
    assert peer_loss_recovery.failure_domains(8, 4) == [*((r,) for r in range(1, 8)), (4, 5, 6, 7)]
    assert peer_loss_recovery.failure_domains(6, 4)[-1] == (4, 5)
    assert peer_loss_recovery.failure_domains(4, 1) == [(1,), (2,), (3,)]


def test_create_accepts_online_eplb(monkeypatch):
    monkeypatch.setattr(peer_loss_recovery, "peer_loss_recovery_enabled", lambda: True)
    monkeypatch.setattr(
        peer_loss_recovery, "_register", lambda rank, recoverable: (Mock(), frozenset())
    )
    executor = _supported_executor()
    executor.model_engine.moe_load_balancer.is_static_routing.return_value = False
    assert isinstance(PeerLossRecovery.create(executor), PeerLossRecovery)


def test_create_accepts_the_dsa_kv_cache_manager(monkeypatch):
    from tensorrt_llm._torch.attention.backends.sparse.dsa import DSACacheManagerV2

    monkeypatch.setattr(peer_loss_recovery, "peer_loss_recovery_enabled", lambda: True)
    monkeypatch.setattr(
        peer_loss_recovery, "_register", lambda rank, recoverable: (Mock(), frozenset())
    )
    executor = _supported_executor()
    executor.kv_cache_manager = DSACacheManagerV2.__new__(DSACacheManagerV2)
    assert isinstance(PeerLossRecovery.create(executor), PeerLossRecovery)


def _spec_config(mode: "SpecMode", use_dynamic_tree: bool = False) -> SimpleNamespace:
    return SimpleNamespace(spec_dec_mode=mode, use_dynamic_tree=use_dynamic_tree)


def test_create_accepts_one_model_mtp(monkeypatch):
    monkeypatch.setattr(peer_loss_recovery, "peer_loss_recovery_enabled", lambda: True)
    monkeypatch.setattr(
        peer_loss_recovery, "_register", lambda rank, recoverable: (Mock(), frozenset())
    )
    executor = _supported_executor()
    executor.model_engine.spec_config = _spec_config(SpecMode.MTP_EAGLE_ONE_MODEL)
    assert isinstance(PeerLossRecovery.create(executor), PeerLossRecovery)


def test_create_registers_the_domains_whose_loss_the_placement_covers(monkeypatch):
    monkeypatch.setattr(peer_loss_recovery, "peer_loss_recovery_enabled", lambda: True)
    registered = []
    monkeypatch.setattr(
        peer_loss_recovery,
        "_register",
        lambda rank, recoverable: registered.append((rank, recoverable)) or (Mock(), frozenset()),
    )
    executor = _supported_executor()
    assert isinstance(PeerLossRecovery.create(executor), PeerLossRecovery)
    assert registered == [(0, ((1,), (2,), (2, 3)))]
    executor.dist.enable_peer_loss_recovery.assert_called_once()


def test_install_recovery_placement_reroutes_before_masking():
    events = []
    load_balancer = Mock()
    load_balancer.install_recovery_placement.side_effect = lambda dead, placements: events.append(
        ("routed", dead)
    )
    health = EPGroupHealth(WORLD_SIZE)
    state = Mock(health=health)
    state.install_mask.side_effect = lambda: events.append(("mask", health.get_mask_words()))
    install = functools.partial(peer_loss_recovery.install_recovery_placement, placements=[])

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


def test_placements_survive_the_protocol_encoding():
    import numpy as np

    placements = [
        np.array([[0, 1, -1], [2, 3, 0]], dtype=np.int32),
        np.array([[3, 2, 1], [0, -1, 2]]),
    ]
    text = peer_loss_recovery._encode_placements(placements)
    decoded = peer_loss_recovery._decode_placements(text, placements)
    assert [p.tolist() for p in decoded] == [p.tolist() for p in placements]
    with pytest.raises(ValueError, match="expected shape"):
        peer_loss_recovery._decode_placements(text, placements[:1])

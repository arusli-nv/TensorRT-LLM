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

import select
import threading
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import replace
from queue import SimpleQueue
from unittest.mock import Mock

import pytest

from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.recovery import (
    AdmissionClose,
    AdmissionClosed,
    CapturedExecutionIdentity,
    FrontendTaskControl,
    IdleBoundaryCommand,
    ModelAdmissionSummary,
    SchedulerIdleReceipt,
    SchedulerStopReceipt,
    ServingTaskContext,
    ServingTaskControl,
    SessionWorkerProgress,
    StopAuthority,
    TaskRetentionAck,
    TaskStarted,
    WorkerStopCommand,
    WorkerTaskIdentity,
    WorkerTaskRegistered,
    WorkerTaskStatus,
)
from tensorrt_llm.executor.worker_process_monitor import (
    LocalProcessDepartureAdapter,
    NodeDepartureAdapter,
    NodeProcessEvent,
    NodeProcessScope,
    ProcessDeparture,
    WorkerProcessIdentity,
)
from tensorrt_llm.llmapi.mpi_session import RemoteMpiCommSessionServer


@pytest.fixture
def registered_authority():
    context = ServingTaskContext(1, 0, ("tcp://authority", b"key"))
    workers = tuple(
        WorkerProcessIdentity(rank, 100 + rank, 20 + rank, "node", 9) for rank in range(3)
    )
    adapter = Mock(adapter_id="qualified-adapter")
    adapter.observe_departure.return_value = None
    authority = StopAuthority(context, 3, (1,), "original-launch", adapter)
    authority.accept(b"session", TaskStarted(1, 0, 3, "session"))
    authority.accept(b"frontend", TaskStarted(1, 0, 3, "frontend"))
    registrations = ()
    for invocation, worker in enumerate(workers):
        identity = WorkerTaskIdentity(1, 0, invocation, worker)
        route = f"worker-{invocation}".encode()
        authority.accept(route, WorkerTaskStatus(identity, "TASK_START", 10))
        registrations = authority.accept(
            route, WorkerTaskStatus(identity, "EXECUTOR_CONSTRUCTED", 10)
        )
    adapter.register.assert_called_once_with(workers)
    departure = ProcessDeparture(1, 0, workers[1], "original-launch", adapter.adapter_id, "exit")
    adapter.owns_receipt.side_effect = lambda receipt: receipt is departure
    return authority, context, workers, departure, registrations


def _begin_stop(authority, departure):
    route, command = authority.begin_process_loss(departure, "loss-1", 30.0)
    assert route == b"frontend"
    closed = AdmissionClosed(command, 7, "original-frontend-empty-ledger")
    output = authority.accept(route, closed)
    hold = output[0][1]
    commands = tuple(message for _, message in output[1:])
    return hold, commands


def _receipt(command):
    return SchedulerStopReceipt(
        command.identity,
        command.transaction_id,
        threading.get_ident(),
        3,
        (
            ("graphs", 4),
            ("registered_streams", 2),
            ("native_kv_owners", 3),
            ("pending_requests", 0),
            ("live_requests", 0),
            ("pending_responses", 0),
        ),
        0,
        2,
        True,
        True,
        True,
        "original-scheduler-local-join",
        CapturedExecutionIdentity("c" * 64, "d" * 64),
    )


def _idle_boundary(authority, transaction_id="idle-loss"):
    route, pause = authority.prepare_idle_boundary(transaction_id, 30.0)
    closed = AdmissionClosed(pause, 7, "original-frontend-empty-ledger")
    commands = tuple(message for _, message in authority.accept(route, closed))
    return commands


def _idle_receipt(command):
    summary = ModelAdmissionSummary("a" * 64, (1,), 3, 2, 4, 2, 8, 1, 8, 8, ((1, "e" * 64),))
    return SchedulerIdleReceipt(_receipt(command), 4, 9, command.frontend_request_epoch, summary)


def test_preloss_idle_does_not_substitute_for_postloss_stop(registered_authority):
    """Do all-live idle certificates permit injection but never complete post-loss STOP?"""
    authority, _, workers, departure, _ = registered_authority
    commands = _idle_boundary(authority)
    assert all(isinstance(command, IdleBoundaryCommand) for command in commands)
    for rank, command in enumerate(commands):
        authority.accept(f"worker-{rank}".encode(), _idle_receipt(command))
    assert authority.idle_boundary_complete
    assert not authority.stop_complete
    output = authority.begin_idle_process_loss(departure)
    hold = output[0][1]
    assert hold.transaction_id == "idle-loss"
    authority.accept(b"session", TaskRetentionAck(hold, (workers[0], workers[2])))
    assert not authority.stop_complete
    for route, command in output[1:]:
        authority.accept(route, _receipt(command))
    assert authority.stop_complete


def test_preloss_idle_requires_every_original_worker(registered_authority):
    """Can an incomplete all-live certificate admit a controlled kill?"""
    authority, _, _, departure, _ = registered_authority
    commands = _idle_boundary(authority)
    authority.accept(b"worker-0", _idle_receipt(commands[0]))
    assert not authority.idle_boundary_complete
    with pytest.raises(ValueError, match="every original scheduler"):
        authority.begin_idle_process_loss(departure)


def test_idle_certificate_rejects_changed_epoch_and_foreign_route(registered_authority):
    """Are certificates bound to frontend ingress and the original scheduler route?"""
    authority, _, _, _, _ = registered_authority
    commands = _idle_boundary(authority)
    receipt = _idle_receipt(commands[0])
    with pytest.raises(ValueError, match="sealed admission"):
        authority.accept(b"worker-1", receipt)
    with pytest.raises(ValueError, match="sealed admission"):
        authority.accept(b"worker-0", replace(receipt, frontend_request_epoch=8))
    authority.accept(b"worker-0", receipt)
    authority.accept(b"worker-0", receipt)
    with pytest.raises(ValueError, match="changed"):
        authority.accept(b"worker-0", replace(receipt, execution_sequence=5))


def test_idle_certificate_rejects_malformed_owner_epochs(registered_authority):
    """Can negative or boolean execution/queue epochs enter the preloss quorum?"""
    authority, _, _, _, _ = registered_authority
    commands = _idle_boundary(authority)
    receipt = _idle_receipt(commands[0])
    for invalid in (
        replace(receipt, execution_sequence=-1),
        replace(receipt, queue_ingress_epoch=True),
    ):
        with pytest.raises(ValueError, match="owner epochs"):
            authority.accept(b"worker-0", invalid)


def test_original_scheduler_publishes_idle_under_queue_seal(registered_authority):
    """Does transport certify only a pushed receipt protected by the original queue seal?"""
    authority, context, _, _, registrations = registered_authority
    command = _idle_boundary(authority)[0]
    control = ServingTaskControl(context, command.identity)
    queue = Mock()
    queue.idle_boundary_publication.return_value = nullcontext()
    control.executor_constructed(queue)
    control._accept_registration(
        next(message for route, message in registrations if route == b"worker-0")
    )
    control._accept_idle_boundary(command)
    assert control.wait_for_idle_boundary(1.0) is command
    receipt = _idle_receipt(command)
    control.publish_idle(receipt)
    queue.idle_boundary_publication.assert_called_once_with("idle-loss", 9)
    assert control._outgoing.get_nowait().kind == "EXECUTOR_CONSTRUCTED"
    assert control._outgoing.get_nowait() is receipt
    queue.validate_idle_boundary.side_effect = RuntimeError("Original queue seal changed")
    with pytest.raises(RuntimeError, match="seal changed"):
        control._accept_stop(WorkerStopCommand(command.identity, "idle-loss", 10.0))
    assert not control.stop_requested


def test_terminal_task_invalidates_completed_idle_boundary(registered_authority):
    """Does an original task ending revoke controlled-injection eligibility?"""
    authority, _, _, _, _ = registered_authority
    commands = _idle_boundary(authority)
    for rank, command in enumerate(commands):
        authority.accept(f"worker-{rank}".encode(), _idle_receipt(command))
    authority.accept(
        b"worker-0", WorkerTaskStatus(commands[0].identity, "TASK_BODY_END", 10, "RuntimeError")
    )
    with pytest.raises(RuntimeError, match="task ended"):
        _ = authority.idle_boundary_complete


def test_stop_quorum_requires_every_original_scheduler(registered_authority):
    """Can server retention alone authorize the STOP checkpoint?"""
    authority, _, workers, departure, _ = registered_authority
    hold, commands = _begin_stop(authority, departure)
    authority.accept(b"session", TaskRetentionAck(hold, (workers[0], workers[2])))
    assert not authority.stop_complete
    first = commands[0]
    authority.accept(b"worker-0", _receipt(first))
    assert not authority.stop_complete
    authority.accept(b"worker-2", _receipt(commands[1]))
    assert authority.stop_complete
    assert not hasattr(authority, "commit")
    assert not hasattr(authority, "resume")


def test_copied_departure_cannot_authorize_loss(registered_authority):
    """Does value equality substitute for an adapter-owned original pidfd receipt?"""
    authority, _, _, departure, _ = registered_authority
    with pytest.raises(ValueError, match="adapter-owned"):
        authority.begin_process_loss(replace(departure), "loss", 30.0)


def test_stop_waits_for_frontend_admission_receipt(registered_authority):
    """Can a worker report STOP before the original frontend closes admission?"""
    authority, context, workers, departure, _ = registered_authority
    authority.begin_process_loss(departure, "loss", 30.0)
    identity = WorkerTaskIdentity(context.task_id, 0, 0, workers[0])
    with pytest.raises(ValueError, match="scheduler-owned"):
        authority.accept(b"worker-0", _receipt(WorkerStopCommand(identity, "loss", 30.0)))
    assert not authority.stop_complete


def test_incomplete_local_join_cannot_complete_stop(registered_authority):
    """Does an unjoined local issuer or incomplete KV inventory block the checkpoint?"""
    authority, _, _, departure, _ = registered_authority
    _, commands = _begin_stop(authority, departure)
    for incomplete in (
        replace(_receipt(commands[0]), local_issuer_join_complete=False),
        replace(_receipt(commands[0]), native_kv_inventory_complete=False),
    ):
        with pytest.raises(ValueError, match="scheduler-owned"):
            authority.accept(b"worker-0", incomplete)
    assert not authority.stop_complete


def test_changed_stop_receipt_is_rejected(registered_authority):
    """Can a worker change its resource inventory after reporting STOP?"""
    authority, _, _, departure, _ = registered_authority
    _, commands = _begin_stop(authority, departure)
    receipt = _receipt(commands[0])
    authority.accept(b"worker-0", receipt)
    with pytest.raises(ValueError, match="changed"):
        authority.accept(b"worker-0", replace(receipt, iteration=4))


def test_worker_receipt_requires_original_route(registered_authority):
    """Can another authenticated worker impersonate an original survivor?"""
    authority, _, _, departure, _ = registered_authority
    _, commands = _begin_stop(authority, departure)
    with pytest.raises(ValueError, match="provenance"):
        authority.accept(b"worker-2", _receipt(commands[0]))


def test_stop_publication_is_pushed_by_original_scheduler(registered_authority):
    """Does control publish immutable scheduler data without traversing model objects?"""
    authority, context, _, departure, registrations = registered_authority
    _, commands = _begin_stop(authority, departure)
    command = commands[0]
    control = ServingTaskControl(context, command.identity)
    queue = Mock()
    queue.scheduler_stop_publication.return_value = nullcontext()
    control.executor_constructed(queue)
    registration = next(message for route, message in registrations if route == b"worker-0")
    assert isinstance(registration, WorkerTaskRegistered)
    control._accept_registration(registration)
    control._accept_stop(command)
    queue.enqueue_scheduler_stop.assert_called_once_with("loss-1", command.timeout_seconds)
    assert control.wait_for_scheduler_stop(1.0) is command
    receipt = _receipt(command)
    control.publish_stopped(receipt)
    assert control._outgoing.get_nowait().kind == "EXECUTOR_CONSTRUCTED"
    assert control._outgoing.get_nowait() is receipt
    with pytest.raises(ValueError, match="scheduler-owned"):
        control.publish_stopped(replace(receipt, scheduler_thread_id=-1))
    with pytest.raises(ValueError, match="original scheduler"):
        control.publish_stopped(replace(receipt, scheduler_thread_id=threading.get_ident() + 1))


def test_worker_lease_expiry_prevents_late_publication(registered_authority, monkeypatch):
    """Can a late callback publish after its authority lease expired?"""
    authority, context, _, departure, registrations = registered_authority
    _, commands = _begin_stop(authority, departure)
    control = ServingTaskControl(context, commands[0].identity)
    control.executor_constructed(Mock())
    control._accept_registration(
        next(message for route, message in registrations if route == b"worker-0")
    )
    control._accept_stop(commands[0])
    monkeypatch.setattr(
        "tensorrt_llm.executor.recovery.time.monotonic", lambda: control._authority_deadline + 1.0
    )
    with pytest.raises(TimeoutError, match="deadline"):
        control.publish_stopped(_receipt(commands[0]))


def test_task_end_preserves_failure_receipt(registered_authority):
    """Does a real task exit remain distinct from a failed MPI Future?"""
    _, context, workers, _, _ = registered_authority
    identity = WorkerTaskIdentity(1, 0, 0, workers[0])
    control = ServingTaskControl(context, identity)
    control.task_body_ended("RuntimeError")
    receipt = control._outgoing.get_nowait()
    assert receipt.kind == "TASK_BODY_END"
    assert receipt.error_type == "RuntimeError"
    with pytest.raises(RuntimeError, match="unavailable"):
        control.publish_stopped(_receipt(WorkerStopCommand(identity, "loss", 30.0)))


def test_pidfd_adapter_rejects_hidden_launcher_scope():
    """Does a container-relative cgroup root fail admission instead of implying isolation?"""
    with pytest.raises(ValueError, match="qualified launcher scope"):
        LocalProcessDepartureAdapter(1, 0, "launch", "0::/\n")


def _departure_adapter(events):
    worker = WorkerProcessIdentity(1, 123, 20, "node", 9)
    adapter = LocalProcessDepartureAdapter.__new__(LocalProcessDepartureAdapter)
    adapter._task_id = 1
    adapter._generation = 0
    adapter._reference = "launch"
    adapter.adapter_id = "original-pidfd"
    adapter._handles = {worker: (17, Mock(poll=Mock(return_value=events)))}
    adapter._receipts = {}
    return adapter, worker


@pytest.mark.parametrize(
    "events",
    [
        [(17, select.POLLERR)],
        [(17, select.POLLNVAL)],
        [(18, select.POLLIN)],
        [(17, 0)],
        [(17, select.POLLIN | select.POLLERR)],
        [(17, select.POLLOUT)],
    ],
)
def test_pidfd_errors_cannot_certify_departure(events):
    """Can an error, unknown event or another handle masquerade as original process exit?"""
    adapter, worker = _departure_adapter(events)
    with pytest.raises(RuntimeError, match="invalid departure evidence"):
        adapter.observe_departure(worker)
    assert not adapter._receipts


@pytest.mark.parametrize("flags", [select.POLLIN, select.POLLHUP, select.POLLIN | select.POLLHUP])
def test_original_pidfd_departure_receipt_is_retained(flags):
    """Does only the original readable/hung-up pidfd mint one retained receipt?"""
    adapter, worker = _departure_adapter([(17, flags)])
    receipt = adapter.observe_departure(worker)
    assert adapter.owns_receipt(receipt)
    assert adapter.observe_departure(worker) is receipt


def test_quiet_pidfd_does_not_certify_departure():
    """Does an empty poll leave the process unclassified?"""
    adapter, worker = _departure_adapter([])
    assert adapter.observe_departure(worker) is None
    assert not adapter._receipts


def test_reused_pid_birth_is_rejected_before_pidfd(monkeypatch):
    """Does PID reuse fail registration before an original handle is acquired?"""
    adapter = LocalProcessDepartureAdapter.__new__(LocalProcessDepartureAdapter)
    adapter._registered = False
    adapter._handles = {}
    adapter._receipts = {}
    worker = WorkerProcessIdentity(1, 123, 20, "node", 9)
    monkeypatch.setattr(
        "tensorrt_llm.executor.worker_process_monitor._read_process_state",
        lambda pid: ("S", 21),
    )
    monkeypatch.setattr(
        "tensorrt_llm.executor.worker_process_monitor.os.stat",
        lambda path: Mock(st_ino=worker.pid_namespace),
    )
    pidfd = Mock()
    monkeypatch.setattr("tensorrt_llm.executor.worker_process_monitor.os.pidfd_open", pidfd)
    with pytest.raises(ValueError, match="birth"):
        adapter.register((worker,))
    pidfd.assert_not_called()


@pytest.fixture
def node_departures():
    scopes = tuple(
        NodeProcessScope(f"node-{i}", f"boot-{i}", f"session-{i}", "owned-scope") for i in range(2)
    )
    workers = tuple(
        WorkerProcessIdentity(rank, 100 + rank, 20 + rank, scopes[rank // 2].hostname, 9)
        for rank in range(4)
    )
    events = {scope: deque() for scope in scopes}
    channels = {}
    for i, scope in enumerate(scopes):
        channel = Mock(use_hmac_encryption=True, hmac_key=f"node-key-{i}".encode(), socket=None)
        channel.poll.side_effect = lambda timeout, scope=scope: bool(events[scope])
        channel.get.side_effect = events[scope].popleft
        channel.put_nowait.side_effect = lambda registration, scope=scope: events[scope].append(
            dict(
                registration=registration,
                sequence=1,
                kind="REGISTERED",
                worker=None,
                evidence_id="",
            )
        )
        channels[scope] = channel
    adapter = NodeDepartureAdapter(1, 0, "launch", channels, worker_control_key=b"worker-key")
    return adapter, scopes, workers, channels, events


def test_node_departure_requires_registered_handle_event_and_retains_local_receipt(node_departures):
    """Do node exit events mint local receipts without treating serialized copies as authority?"""
    adapter, scopes, workers, channels, events = node_departures
    adapter.register(workers)
    for scope in scopes:
        registration = adapter._registrations[scope]
        assert registration.workers == tuple(
            worker for worker in workers if worker.hostname == scope.hostname
        )
    assert adapter.observe_departure(workers[2]) is None
    registration = adapter._registrations[scopes[1]]
    events[scopes[1]].append(
        NodeProcessEvent(registration, 2, "DEPARTED", workers[2], "original-exit").to_wire()
    )
    receipt = adapter.observe_departure(workers[2])
    assert adapter.owns_receipt(receipt)
    assert adapter.observe_departure(workers[2]) is receipt
    assert not adapter.owns_receipt(replace(receipt))
    adapter.close()
    adapter.close()
    assert not adapter.owns_receipt(receipt)
    for channel in channels.values():
        channel.close.assert_called_once()


@pytest.mark.parametrize("change", ["boot", "session", "scope", "generation", "sequence", "worker"])
def test_node_departure_rejects_foreign_or_stale_events(node_departures, change):
    """Can a different node epoch, logical worker or replay authorize process removal?"""
    adapter, scopes, workers, channels, events = node_departures
    adapter.register(workers)
    scope = scopes[1]
    registration = adapter._registrations[scope]
    event = NodeProcessEvent(registration, 2, "DEPARTED", workers[2], "exit")
    if change in ("boot", "session", "scope"):
        field = {"boot": "boot_id", "session": "session_id", "scope": "scope_reference"}[change]
        event = replace(
            event, registration=replace(registration, scope=replace(scope, **{field: "foreign"}))
        )
    elif change == "generation":
        event = replace(event, registration=replace(registration, generation=1))
    elif change == "sequence":
        event = replace(event, sequence=1)
    else:
        event = replace(event, worker=workers[0])
    events[scope].append(event.to_wire())
    with pytest.raises(RuntimeError):
        adapter.observe_departure(workers[2])
    assert not adapter._receipts
    with pytest.raises(RuntimeError, match="unavailable"):
        adapter.observe_departure(workers[2])
    adapter.close()


def test_node_registration_failure_releases_all_channels(node_departures):
    """Does a premature exit fail startup instead of publishing partial node membership?"""
    adapter, scopes, workers, channels, events = node_departures
    channels[scopes[1]].put_nowait.side_effect = lambda registration: events[scopes[1]].append(
        dict(
            registration=registration,
            sequence=1,
            kind="DEPARTED",
            worker=workers[2]._asdict(),
            evidence_id="early-exit",
        )
    )
    with pytest.raises(RuntimeError, match="registered original"):
        adapter.register(workers)
    assert not adapter._registered and not adapter._receipts
    for channel in channels.values():
        channel.close.assert_called_once()


@pytest.mark.parametrize("change", ["extra", "missing", "integer-alias", "typed-event"])
def test_node_wire_requires_exact_primitive_fields(node_departures, change):
    """Can extra fields, boolean aliases or an inference-package object bypass host wire checks?"""
    adapter, scopes, workers, _, events = node_departures
    adapter.register(workers)
    event = NodeProcessEvent(adapter._registrations[scopes[1]], 2, "DEPARTED", workers[2], "exit")
    payload = event.to_wire()
    if change == "extra":
        payload["unqualified"] = True
    elif change == "missing":
        del payload["worker"]
    elif change == "integer-alias":
        payload["registration"]["task_id"] = True
    else:
        payload = event
    events[scopes[1]].append(payload)
    with pytest.raises(RuntimeError):
        adapter.observe_departure(workers[2])
    assert not adapter._receipts
    adapter.close()


def test_node_registration_missing_ack_is_bounded_and_releases_channels(
    node_departures, monkeypatch
):
    """Does a missing node handle acknowledgment stop startup without granting departure?"""
    adapter, scopes, workers, channels, _ = node_departures
    channels[scopes[1]].put_nowait.side_effect = None
    now = [0.0]

    def monotonic():
        now[0] += 1.0
        return now[0]

    monkeypatch.setattr("tensorrt_llm.executor.worker_process_monitor.time.monotonic", monotonic)
    with pytest.raises(TimeoutError, match="registration timed out"):
        adapter.register(workers)
    assert not adapter._registered and not adapter._receipts
    for channel in channels.values():
        channel.close.assert_called_once()


def test_node_lease_expiry_cannot_be_revived_by_late_heartbeat(node_departures, monkeypatch):
    """Can delayed liveness revive lost control or certify worker departure?"""
    adapter, scopes, workers, channels, events = node_departures
    now = [100.0]
    monkeypatch.setattr(
        "tensorrt_llm.executor.worker_process_monitor.time.monotonic", lambda: now[0]
    )
    adapter.register(workers)
    registration = adapter._registrations[scopes[1]]
    events[scopes[1]].append(NodeProcessEvent(registration, 2, "ALIVE").to_wire())
    now[0] += 31.0
    with pytest.raises(TimeoutError, match="lease expired"):
        adapter.observe_departure(workers[2])
    assert not adapter._receipts and len(events[scopes[1]]) == 1
    adapter.close()


@pytest.mark.parametrize("failure", ["terminal", "authentication"])
def test_node_control_loss_invalidates_cached_departure(node_departures, failure):
    """Does cached process exit remain insufficient once observer control is lost?"""
    adapter, scopes, workers, channels, events = node_departures
    adapter.register(workers)
    registration = adapter._registrations[scopes[1]]
    events[scopes[1]].append(
        NodeProcessEvent(registration, 2, "DEPARTED", workers[2], "exit").to_wire()
    )
    receipt = adapter.observe_departure(workers[2])
    if failure == "terminal":
        events[scopes[1]].append(NodeProcessEvent(registration, 3, "TERMINAL").to_wire())
    else:
        channels[scopes[1]].hmac_key = b"changed"
    with pytest.raises(RuntimeError):
        adapter.observe_departure(workers[2])
    assert not adapter.owns_receipt(receipt)
    adapter.close()


def test_node_observer_credentials_must_be_independent(node_departures):
    """Can a worker-control key or shared node key impersonate a lifecycle observer?"""
    adapter, scopes, _, channels, _ = node_departures
    channels[scopes[0]].hmac_key = b"worker-key"
    with pytest.raises(ValueError, match="independently authenticated"):
        NodeDepartureAdapter(1, 0, "launch", channels, worker_control_key=b"worker-key")
    channels[scopes[0]].hmac_key = channels[scopes[1]].hmac_key
    with pytest.raises(ValueError, match="distinct"):
        NodeDepartureAdapter(1, 0, "launch", channels, worker_control_key=b"worker-key")
    adapter.close()


@pytest.fixture
def idle_proxy():
    proxy = GenerationExecutorProxy.__new__(GenerationExecutorProxy)
    context = ServingTaskContext(1, 0, ("tcp://authority", b"key"))
    proxy.shutdown = Mock()
    proxy._serving_control = Mock(context=context)
    proxy._admission_lock = threading.Lock()
    proxy._admission_closed = None
    proxy._unconsumed_results = {}
    proxy._results = {}
    proxy._request_epoch = 2
    proxy._auxiliary_requests_issued = False
    proxy.doing_shutdown = False
    proxy._engine_dead = False
    proxy._fatal_error = None
    return proxy


def test_idle_admission_retains_undelivered_terminal_result(idle_proxy):
    """Does an empty active-result map hide a queued final response?"""
    result = Mock(finished=False)
    result.queue.empty.return_value = False
    idle_proxy._unconsumed_results[3] = result
    command = AdmissionClose(1, 0, "authority", "loss", 30.0)
    with pytest.raises(ValueError, match="consumed results"):
        idle_proxy._close_idle_admission(command)
    assert idle_proxy._admission_closed is None
    result.finished = True
    result.queue.empty.return_value = True
    receipt = idle_proxy._close_idle_admission(command)
    assert receipt.command is command
    assert receipt.request_epoch == 2
    assert idle_proxy._admission_closed is receipt


def test_idle_admission_requires_both_consumption_and_empty_queue(idle_proxy):
    """Does a consumed terminal flag suffice while another response remains queued?"""
    result = Mock(finished=True)
    result.queue.empty.return_value = False
    idle_proxy._unconsumed_results[3] = result
    with pytest.raises(ValueError, match="consumed results"):
        idle_proxy._close_idle_admission(AdmissionClose(1, 0, "authority", "loss", 30.0))


def test_auxiliary_worker_request_blocks_idle_certificate(idle_proxy):
    """Can an unaccounted profile/RPC/cancellation operation be called idle?"""
    idle_proxy._auxiliary_requests_issued = True
    with pytest.raises(ValueError, match="auxiliary"):
        idle_proxy._close_idle_admission(AdmissionClose(1, 0, "authority", "loss", 30.0))


def test_admission_certificate_rejects_other_task(idle_proxy):
    """Can the original frontend close admission for a different model task?"""
    with pytest.raises(RuntimeError, match="Original frontend"):
        idle_proxy._close_idle_admission(AdmissionClose(2, 0, "authority", "loss", 30.0))


def test_submit_cannot_cross_closed_admission(idle_proxy):
    """Can a new request be issued after the genuine frontend receipt?"""
    idle_proxy._close_idle_admission(AdmissionClose(1, 0, "authority", "loss", 30.0))
    idle_proxy._submit_request = Mock()
    from tensorrt_llm.executor.utils import RequestError

    with pytest.raises(RequestError, match="admission is closed"):
        idle_proxy.submit(Mock())
    idle_proxy._submit_request.assert_not_called()


def test_frontend_receipt_must_come_from_proxy_owner(registered_authority):
    """Does the frontend transport reject a copied or unrelated closure certificate?"""
    authority, context, _, departure, registrations = registered_authority
    close = Mock()
    control = FrontendTaskControl(context, 3, close)
    registration = next(message for route, message in registrations if route == b"frontend")
    control._accept(registration)
    _, command = authority.begin_process_loss(departure, "loss", 30.0)
    close.return_value = AdmissionClosed(replace(command), 0, "copied")
    with pytest.raises(ValueError, match="original admission"):
        control._accept(command)


def test_mpi_callback_does_not_write_cross_thread_socket():
    """Does a Future callback leave all socket writes to the session owner?"""
    server = RemoteMpiCommSessionServer.__new__(RemoteMpiCommSessionServer)
    server._session_events = SimpleQueue()
    server.queue = Mock()
    future = Future()
    future.set_exception(RuntimeError("application failure"))
    server.mpi_async_error_callback(future)
    assert server._session_events.get_nowait() == ("async", future)
    server.queue.put_nowait.assert_not_called()


def test_failed_mpi_future_is_not_original_task_stop(registered_authority, monkeypatch):
    """Can a failed transport Future stand in for a worker-produced STOP?"""
    authority, _, workers, departure, _ = registered_authority
    hold, commands = _begin_stop(authority, departure)
    server = RemoteMpiCommSessionServer.__new__(RemoteMpiCommSessionServer)
    server._retention = hold
    server._invocation_workers = workers
    server._active_futures = [Future(), Future(), Future()]
    server._active_futures[0].set_exception(RuntimeError("synthetic transport result"))
    server._task_body_ended = {}
    server._worker_progress = {}
    monkeypatch.setattr(server, "_is_native_transport_failure", lambda error: True)
    assert not server._check_retained_tasks()
    server._accept_task_progress(
        SessionWorkerProgress(commands[0].identity, "loss-1", 30.0, _receipt(commands[0]))
    )
    assert not server._check_retained_tasks()
    server._accept_task_progress(
        SessionWorkerProgress(commands[1].identity, "loss-1", 30.0, _receipt(commands[1]))
    )
    assert server._check_retained_tasks()
    server._task_body_ended[0] = "real-body-ended"
    with pytest.raises(RuntimeError, match="task body ended"):
        server._check_retained_tasks()


def test_malformed_stop_inventory_never_enters_quorum(registered_authority):
    """Can missing, conflicting or impossible resource counts certify scheduler STOP?"""
    authority, _, _, departure, _ = registered_authority
    _, commands = _begin_stop(authority, departure)
    receipt = _receipt(commands[0])
    bad_counts = (
        (),
        receipt.owner_counts[:-1],
        receipt.owner_counts + (("graphs", 4),),
        tuple((key, False if key == "graphs" else count) for key, count in receipt.owner_counts),
        tuple((key, 0 if key == "graphs" else count) for key, count in receipt.owner_counts),
        tuple((key, 1 if key == "live_requests" else count) for key, count in receipt.owner_counts),
    )
    for counts in bad_counts:
        with pytest.raises(ValueError, match="scheduler-owned"):
            authority.accept(b"worker-0", replace(receipt, owner_counts=counts))
    with pytest.raises(ValueError, match="scheduler-owned"):
        authority.accept(b"worker-0", replace(receipt, iteration=-1))
    with pytest.raises(ValueError, match="scheduler-owned"):
        authority.accept(b"worker-0", replace(receipt, registered_stream_events_completed=3))
    assert not authority.stop_complete


def test_task_end_invalidates_completed_stop_quorum(registered_authority):
    """Does a task ending after STOP prevent the authority treating the hold as live?"""
    authority, _, workers, departure, _ = registered_authority
    hold, commands = _begin_stop(authority, departure)
    authority.accept(b"session", TaskRetentionAck(hold, (workers[0], workers[2])))
    for command in commands:
        authority.accept(f"worker-{command.identity.invocation_id}".encode(), _receipt(command))
    assert authority.stop_complete
    authority.accept(b"worker-0", WorkerTaskStatus(commands[0].identity, "TASK_BODY_END", 10))
    with pytest.raises(RuntimeError, match="task ended"):
        _ = authority.stop_complete


def test_shutdown_winning_publication_blocks_stop_receipt(registered_authority):
    """Can a completed GPU join authorize STOP after its request queue became terminal?"""
    authority, context, _, departure, registrations = registered_authority
    _, commands = _begin_stop(authority, departure)
    control = ServingTaskControl(context, commands[0].identity)
    queue = Mock()
    queue.scheduler_stop_publication.side_effect = RuntimeError("original queue is terminal")
    control.executor_constructed(queue)
    control._accept_registration(
        next(message for route, message in registrations if route == b"worker-0")
    )
    control._accept_stop(commands[0])
    with pytest.raises(RuntimeError, match="queue is terminal"):
        control.publish_stopped(_receipt(commands[0]))
    assert control._stopped is None
    assert control._outgoing.get_nowait().kind == "EXECUTOR_CONSTRUCTED"
    assert control._outgoing.empty()


def _recovery_proposal(authority, workers):
    from tensorrt_llm.executor.recovery import RecoveryProposal

    return RecoveryProposal(
        authority.context.task_id,
        authority.authority_id,
        "loss-1",
        0,
        1,
        (workers[0], workers[2]),
        workers[1],
        "e" * 64,
        "b" * 64,
    )


def _complete_stop(authority, workers, departure):
    for rank, command in enumerate(_idle_boundary(authority, "loss-1")):
        authority.accept(f"worker-{rank}".encode(), _idle_receipt(command))
    output = authority.begin_idle_process_loss(departure)
    hold = output[0][1]
    commands = tuple(message for _, message in output[1:])
    authority.accept(b"session", TaskRetentionAck(hold, (workers[0], workers[2])))
    for command in commands:
        authority.accept(f"worker-{command.identity.invocation_id}".encode(), _receipt(command))
    return commands


def test_preparation_requires_complete_stop_and_changes_no_generation(registered_authority):
    """Can preparation bypass real STOP, mutate generation or authorize replay?"""
    authority, context, workers, departure, _ = registered_authority
    proposal = _recovery_proposal(authority, workers)
    with pytest.raises(ValueError, match="complete scheduler STOP"):
        authority.begin_preparation(proposal)
    _complete_stop(authority, workers, departure)
    deadline = authority._deadline
    commands = authority.begin_preparation(proposal)
    assert tuple(route for route, _ in commands) == (b"worker-0", b"worker-2")
    assert all(command is proposal for _, command in commands)
    assert authority.begin_preparation(proposal) == ()
    assert authority._deadline == deadline
    assert context.expected_generation == 0
    assert authority._registration.generation == 0
    assert not hasattr(authority, "commit")
    assert not hasattr(authority, "resume")


def test_masked_placement_catalog_is_exact_and_immutable():
    """Can a missing, duplicated or mutable candidate qualify a resident rank?"""
    summary = ModelAdmissionSummary("a" * 64, (1,), 3, 2, 4, 2, 8, 1, 8, 8, ((1, "e" * 64),))
    for catalog in (
        (),
        ((2, "e" * 64),),
        ((1, "e" * 64), (1, "e" * 64)),
        [[1, "e" * 64]],
        ((True, "e" * 64),),
        ((1, "not-a-digest"),),
    ):
        with pytest.raises(ValueError, match="model admission summary"):
            replace(summary, masked_placement_digests=catalog)
    with pytest.raises(ValueError, match="model admission summary"):
        replace(
            summary,
            admitted_ranks=(1, 2),
            masked_placement_digests=((2, "f" * 64), (1, "e" * 64)),
        )
    assert replace(summary, admitted_ranks=(), masked_placement_digests=()).admitted_ranks == ()


def test_preparation_rejects_source_and_unqualified_next_placement(registered_authority):
    """Can the source placement or another candidate authorize survivor preparation?"""
    authority, _, workers, departure, _ = registered_authority
    _complete_stop(authority, workers, departure)
    proposal = _recovery_proposal(authority, workers)
    for digest in ("a" * 64, "f" * 64):
        with pytest.raises(ValueError, match="admitted masked placement"):
            authority.begin_preparation(replace(proposal, placement_digest=digest))
        assert authority._proposal is None
    assert authority.begin_preparation(proposal)


def test_preparation_rejects_stop_without_original_idle_catalog(registered_authority):
    """Can task retention and STOP substitute for model-produced placement admission?"""
    authority, _, workers, departure, _ = registered_authority
    hold, commands = _begin_stop(authority, departure)
    authority.accept(b"session", TaskRetentionAck(hold, (workers[0], workers[2])))
    for command in commands:
        authority.accept(f"worker-{command.identity.invocation_id}".encode(), _receipt(command))
    assert authority.stop_complete
    with pytest.raises(ValueError, match="original idle placement catalog"):
        authority.begin_preparation(_recovery_proposal(authority, workers))
    assert authority._proposal is None


def test_proposal_binds_original_authority_membership_and_generation(registered_authority):
    """Can a stale authority or different original birth prepare survivor state?"""
    authority, _, workers, departure, _ = registered_authority
    _complete_stop(authority, workers, departure)
    proposal = _recovery_proposal(authority, workers)
    for invalid in (
        replace(proposal, authority_id="other-authority"),
        replace(proposal, transaction_id="older-loss"),
        replace(proposal, base_generation=1, next_generation=2),
        replace(proposal, survivors=(workers[0], workers[2]._replace(start_time=999))),
    ):
        with pytest.raises(ValueError, match="original stopped membership"):
            authority.begin_preparation(invalid)
    authority.begin_preparation(proposal)
    with pytest.raises(ValueError, match="Conflicting"):
        authority.begin_preparation(replace(proposal, placement_digest="c" * 64))


def test_proposal_digest_is_canonical_and_rejects_ambiguous_layouts(registered_authority):
    """Do full original births and plan digests define one immutable proposal identity?"""
    authority, _, workers, _, _ = registered_authority
    proposal = _recovery_proposal(authority, workers)
    assert len(proposal.digest) == 64
    assert replace(proposal).digest == proposal.digest
    assert replace(proposal, backend_digest="c" * 64).digest != proposal.digest
    for fields in (
        {"next_generation": 2},
        {"survivors": (workers[2], workers[0])},
        {"survivors": (workers[0], workers[0])},
        {"placement_digest": "unqualified-plan"},
    ):
        with pytest.raises(ValueError, match="Invalid original survivor"):
            replace(proposal, **fields)


def test_preparation_requires_accepted_proposal_and_original_local_stop(registered_authority):
    """Can a copied plan or STOP receipt substitute for scheduler-owned local preparation?"""
    authority, context, workers, departure, registrations = registered_authority
    commands = _complete_stop(authority, workers, departure)
    proposal = _recovery_proposal(authority, workers)
    authority.begin_preparation(proposal)
    control = ServingTaskControl(context, commands[0].identity)
    queue = Mock()
    queue.scheduler_stop_publication.return_value = nullcontext()
    control.executor_constructed(queue)
    control._accept_registration(
        next(message for route, message in registrations if route == b"worker-0")
    )
    with pytest.raises(ValueError, match="precedes"):
        control._accept_proposal(proposal)
    with pytest.raises(ValueError, match="original accepted"):
        control.validate_preparation(None, None)
    control._accept_stop(commands[0])
    stopped = _receipt(commands[0])
    control.publish_stopped(stopped)
    control._accept_proposal(proposal)
    assert control.wait_for_recovery_proposal(1.0) is proposal
    assert control.validate_preparation(proposal, stopped) is proposal
    for copied_proposal, copied_stop in (
        (replace(proposal), stopped),
        (proposal, replace(stopped)),
    ):
        with pytest.raises(ValueError, match="original accepted"):
            control.validate_preparation(copied_proposal, copied_stop)
    assert not hasattr(control, "validate_install")
    assert not hasattr(control, "validate_commit")


def test_idle_injection_requires_unanimous_actual_model_admission(registered_authority):
    """Can a configured victim override the running model's resident-replica envelope?"""
    authority, _, _, departure, _ = registered_authority
    commands = _idle_boundary(authority)
    for rank, command in enumerate(commands):
        receipt = _idle_receipt(command)
        summary = replace(
            receipt.model_admission, admitted_ranks=(2,), masked_placement_digests=((2, "e" * 64),)
        )
        authority.accept(f"worker-{rank}".encode(), replace(receipt, model_admission=summary))
    assert authority.admitted_idle_ranks == ()
    assert not authority.idle_boundary_complete
    with pytest.raises(ValueError, match="every original scheduler"):
        authority.begin_idle_process_loss(departure)


def test_idle_model_summary_requires_matching_placement_and_real_capacity(registered_authority):
    """Can inconsistent placement or insufficient fresh-request KV budget admit injection?"""
    authority, _, _, _, _ = registered_authority
    commands = _idle_boundary(authority)
    first = _idle_receipt(commands[0])
    authority.accept(b"worker-0", first)
    other = _idle_receipt(commands[1])
    for summary in (
        replace(other.model_admission, placement_digest="b" * 64),
        replace(
            other.model_admission, admitted_ranks=(2,), masked_placement_digests=((2, "e" * 64),)
        ),
        replace(other.model_admission, masked_placement_digests=((1, "f" * 64),)),
        replace(other.model_admission, request_sequence_length=9, kv_token_capacity=9),
    ):
        with pytest.raises(ValueError, match="workers disagree"):
            authority.accept(b"worker-1", replace(other, model_admission=summary))
    with pytest.raises(ValueError, match="KV capacity"):
        authority.accept(
            b"worker-1",
            replace(other, model_admission=replace(other.model_admission, kv_token_capacity=7)),
        )


def test_postloss_stop_requires_original_graph_and_address_identity(registered_authority):
    """Can replacement graphs or moved static storage satisfy the post-loss STOP checkpoint?"""
    authority, _, _, departure, _ = registered_authority
    commands = _idle_boundary(authority)
    for rank, command in enumerate(commands):
        authority.accept(f"worker-{rank}".encode(), _idle_receipt(command))
    output = authority.begin_idle_process_loss(departure)
    stopped = _receipt(output[1][1])
    for identity in (
        None,
        replace(stopped.execution_identity, graph_digest="e" * 64),
        replace(stopped.execution_identity, address_digest="f" * 64),
    ):
        with pytest.raises(ValueError, match="captured execution changed"):
            authority.accept(b"worker-0", replace(stopped, execution_identity=identity))


def test_idle_eligibility_rejects_original_departure(registered_authority):
    """Can all idle receipts permit injection after an original worker already exited?"""
    authority, _, _, departure, _ = registered_authority
    for rank, command in enumerate(_idle_boundary(authority)):
        authority.accept(f"worker-{rank}".encode(), _idle_receipt(command))
    authority._adapter.observe_departure.side_effect = lambda worker: (
        departure if worker == departure.worker else None
    )
    with pytest.raises(RuntimeError, match="single admitted loss"):
        _ = authority.idle_boundary_complete
    authority._adapter.observe_departure.side_effect = None
    with pytest.raises(RuntimeError, match="observation is terminal"):
        _ = authority.admitted_idle_ranks


def test_idle_loss_allows_only_original_victim_departure(registered_authority):
    """Can the controlled-loss transition admit its verified victim without a bypass?"""
    authority, _, workers, departure, _ = registered_authority
    for rank, command in enumerate(_idle_boundary(authority)):
        authority.accept(f"worker-{rank}".encode(), _idle_receipt(command))
    authority._adapter.observe_departure.side_effect = lambda worker: (
        departure if worker == departure.worker else None
    )
    output = authority.begin_idle_process_loss(departure)
    assert output[0][1].failed_worker == departure.worker
    assert authority.survivors == (workers[0], workers[2])
    assert authority._adapter.observe_departure.call_args_list == [
        ((worker,),) for worker in workers
    ]


def test_second_loss_invalidates_stop_and_preparation(registered_authority):
    """Can a survivor departure be hidden by an already-complete STOP quorum?"""
    authority, _, workers, departure, _ = registered_authority
    _complete_stop(authority, workers, departure)
    assert authority.stop_complete
    second = replace(departure, worker=workers[2], evidence_id="second-exit")
    authority._adapter.owns_receipt.side_effect = lambda receipt: (
        receipt is departure or receipt is second
    )
    authority._adapter.observe_departure.side_effect = lambda worker: (
        departure if worker == departure.worker else second if worker == second.worker else None
    )
    with pytest.raises(RuntimeError, match="single admitted loss"):
        _ = authority.stop_complete
    authority._adapter.observe_departure.side_effect = None
    with pytest.raises(RuntimeError, match="observation is terminal"):
        authority.begin_preparation(_recovery_proposal(authority, workers))


def test_membership_adapter_failure_is_terminal(registered_authority):
    """Can lost departure observation turn back into idle eligibility or STOP?"""
    authority, _, _, _, _ = registered_authority
    authority._adapter.observe_departure.side_effect = OSError("node observation lost")
    with pytest.raises(OSError, match="observation lost"):
        _ = authority.idle_boundary_complete
    authority._adapter.observe_departure.side_effect = None
    with pytest.raises(RuntimeError, match="observation is terminal"):
        _ = authority.stop_complete

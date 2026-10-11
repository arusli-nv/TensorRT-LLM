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

"""Internal serving-task STOP protocol; no installation or resume permission."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import Literal, Protocol

import zmq

from .worker_process_monitor import ProcessDeparture, WorkerProcessIdentity


def _check_timeout(seconds: float, maximum: float = 60.0) -> None:
    if (
        type(seconds) not in (int, float)
        or not math.isfinite(seconds)
        or not 0 < seconds <= maximum
    ):
        raise ValueError("A finite, positive receiver-local timeout is required")


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class ServingTaskContext:
    task_id: int
    expected_generation: int
    authority_address: tuple[str, bytes]
    registration_timeout_seconds: float = 900.0

    def __post_init__(self) -> None:
        _check_timeout(self.registration_timeout_seconds, 900.0)
        if (
            type(self.task_id) is not int
            or self.task_id <= 0
            or type(self.expected_generation) is not int
            or self.expected_generation < 0
            or not all(self.authority_address)
        ):
            raise ValueError("Invalid original serving-task context")


@dataclass(frozen=True)
class WorkerTaskIdentity:
    task_id: int
    generation: int
    invocation_id: int
    worker: WorkerProcessIdentity


@dataclass(frozen=True)
class TaskStarted:
    task_id: int
    generation: int
    worker_count: int
    role: Literal["session", "frontend"]


@dataclass(frozen=True)
class WorkerTaskStatus:
    identity: WorkerTaskIdentity
    kind: Literal["TASK_START", "EXECUTOR_CONSTRUCTED", "TASK_CONTROL_ALIVE", "TASK_BODY_END"]
    owner_thread_id: int
    error_type: str = ""


@dataclass(frozen=True)
class TaskRegistration:
    task_id: int
    generation: int
    workers: tuple[WorkerProcessIdentity, ...]
    registration_reference: str
    authority_id: str
    authority_lease_seconds: float


@dataclass(frozen=True)
class WorkerTaskRegistered:
    identity: WorkerTaskIdentity
    registration: TaskRegistration


@dataclass(frozen=True)
class AuthorityLease:
    task_id: int
    generation: int
    registration_reference: str
    sequence: int
    timeout_seconds: float
    authority_id: str


class DepartureAdapter(Protocol):
    adapter_id: str

    def register(self, workers: tuple[WorkerProcessIdentity, ...]) -> None: ...

    def owns_receipt(self, receipt: ProcessDeparture) -> bool: ...

    def observe_departure(self, worker: WorkerProcessIdentity) -> ProcessDeparture | None: ...


@dataclass(frozen=True)
class AdmissionClose:
    task_id: int
    generation: int
    authority_id: str
    transaction_id: str
    timeout_seconds: float


@dataclass(frozen=True)
class AdmissionClosed:
    """Frontend-owned empty, consumed request ledger with admission held closed."""

    command: AdmissionClose
    request_epoch: int
    owner_reference: str


@dataclass(frozen=True)
class TaskRetentionRequest:
    task_id: int
    transaction_id: str
    expected_generation: int
    workers: tuple[WorkerProcessIdentity, ...]
    failed_worker: WorkerProcessIdentity
    timeout_seconds: float
    evidence_reference: str


@dataclass(frozen=True)
class TaskRetentionAck:
    """MPI task retention, never model readiness or permission to replay."""

    request: TaskRetentionRequest
    survivors: tuple[WorkerProcessIdentity, ...]


@dataclass(frozen=True)
class WorkerStopCommand:
    identity: WorkerTaskIdentity
    transaction_id: str
    timeout_seconds: float


@dataclass(frozen=True)
class CapturedExecutionIdentity:
    """Owner-produced scalar identities, not a captured-memory safety proof."""

    graph_digest: str
    address_digest: str

    def __post_init__(self) -> None:
        if not _is_digest(self.graph_digest) or not _is_digest(self.address_digest):
            raise ValueError("Invalid original captured execution identity")


@dataclass(frozen=True)
class ModelAdmissionSummary:
    """Resident-loader coverage and declared capacity for controlled idle injection."""

    placement_digest: str
    admitted_ranks: tuple[int, ...]
    logical_capacity: int
    layer_count: int
    expert_count: int
    slots_per_rank: int
    token_capacity: int
    request_capacity: int
    request_sequence_length: int
    kv_token_capacity: int
    masked_placement_digests: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        if (
            not _is_digest(self.placement_digest)
            or any(
                type(value) is not int or value <= 0
                for value in (
                    self.logical_capacity,
                    self.layer_count,
                    self.expert_count,
                    self.slots_per_rank,
                    self.token_capacity,
                    self.request_capacity,
                    self.request_sequence_length,
                )
            )
            or type(self.kv_token_capacity) is not int
            or self.kv_token_capacity < 0
            or self.expert_count > self.logical_capacity * self.slots_per_rank
            or not isinstance(self.admitted_ranks, tuple)
            or any(
                type(rank) is not int or not 0 < rank < self.logical_capacity
                for rank in self.admitted_ranks
            )
            or self.admitted_ranks != tuple(sorted(set(self.admitted_ranks)))
            or not isinstance(self.masked_placement_digests, tuple)
            or any(
                not isinstance(entry, tuple)
                or len(entry) != 2
                or type(entry[0]) is not int
                or not _is_digest(entry[1])
                for entry in self.masked_placement_digests
            )
            or tuple(rank for rank, _ in self.masked_placement_digests) != self.admitted_ranks
        ):
            raise ValueError("Invalid resident model admission summary")

    @property
    def required_kv_tokens(self) -> int:
        return self.request_capacity * self.request_sequence_length

    @property
    def shared_geometry(self) -> tuple:
        return (
            self.placement_digest,
            self.admitted_ranks,
            self.masked_placement_digests,
            self.logical_capacity,
            self.layer_count,
            self.expert_count,
            self.slots_per_rank,
            self.token_capacity,
            self.request_capacity,
            self.request_sequence_length,
        )


@dataclass(frozen=True)
class SchedulerStopReceipt:
    """Immutable result published by the original scheduler after its local joins."""

    identity: WorkerTaskIdentity
    transaction_id: str
    scheduler_thread_id: int
    iteration: int
    owner_counts: tuple[tuple[str, int], ...]
    model_generation: int
    registered_stream_events_completed: int
    ticket_retired: bool
    local_issuer_join_complete: bool
    native_kv_inventory_complete: bool
    owner_reference: str
    execution_identity: CapturedExecutionIdentity | None = None


@dataclass(frozen=True)
class IdleBoundaryCommand:
    """Close an all-live idle boundary before a controlled process-loss test."""

    identity: WorkerTaskIdentity
    admission_closed: AdmissionClosed
    timeout_seconds: float

    @property
    def transaction_id(self) -> str:
        return self.admission_closed.command.transaction_id

    @property
    def frontend_request_epoch(self) -> int:
        return self.admission_closed.request_epoch


@dataclass(frozen=True)
class SchedulerIdleReceipt:
    """Sealed local joins permit injection, never post-loss STOP or graph replay."""

    local_receipt: SchedulerStopReceipt
    execution_sequence: int
    queue_ingress_epoch: int
    frontend_request_epoch: int
    model_admission: ModelAdmissionSummary | None = None


@dataclass(frozen=True)
class RecoveryProposal:
    """One immutable survivor plan; preparation does not authorize installation or replay."""

    task_id: int
    authority_id: str
    transaction_id: str
    base_generation: int
    next_generation: int
    survivors: tuple[WorkerProcessIdentity, ...]
    failed_worker: WorkerProcessIdentity
    placement_digest: str
    backend_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.task_id) is not int
            or self.task_id <= 0
            or type(self.base_generation) is not int
            or self.base_generation < 0
            or type(self.next_generation) is not int
            or self.next_generation != self.base_generation + 1
            or not self.authority_id
            or not self.transaction_id
            or not isinstance(self.survivors, tuple)
            or not self.survivors
            or not isinstance(self.failed_worker, WorkerProcessIdentity)
            or self.failed_worker.rank == 0
            or self.failed_worker in self.survivors
            or any(not isinstance(worker, WorkerProcessIdentity) for worker in self.survivors)
            or tuple(worker.rank for worker in self.survivors)
            != tuple(sorted({worker.rank for worker in self.survivors}))
            or not all(
                _is_digest(digest) for digest in (self.placement_digest, self.backend_digest)
            )
        ):
            raise ValueError("Invalid original survivor recovery proposal")

    @property
    def digest(self) -> str:
        payload = {
            "task_id": self.task_id,
            "authority_id": self.authority_id,
            "transaction_id": self.transaction_id,
            "base_generation": self.base_generation,
            "next_generation": self.next_generation,
            "survivors": self.survivors,
            "failed_worker": self.failed_worker,
            "placement_digest": self.placement_digest,
            "backend_digest": self.backend_digest,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _validate_proposal(
    proposal: RecoveryProposal,
    registration: TaskRegistration,
    transaction_id: str,
    failed_worker: WorkerProcessIdentity,
) -> None:
    if (
        proposal.task_id != registration.task_id
        or proposal.authority_id != registration.authority_id
        or proposal.transaction_id != transaction_id
        or proposal.base_generation != registration.generation
        or proposal.failed_worker != failed_worker
        or failed_worker not in registration.workers
        or proposal.survivors
        != tuple(
            sorted(
                (worker for worker in registration.workers if worker != failed_worker),
                key=lambda worker: worker.rank,
            )
        )
    ):
        raise ValueError("Recovery proposal does not bind original stopped membership")


def _validate_idle_receipt(receipt: SchedulerIdleReceipt) -> None:
    _validate_scheduler_stop(receipt.local_receipt)
    summary = receipt.model_admission
    if (
        not isinstance(summary, ModelAdmissionSummary)
        or not isinstance(receipt.local_receipt.execution_identity, CapturedExecutionIdentity)
        or summary.kv_token_capacity < summary.required_kv_tokens
    ):
        raise ValueError("Idle boundary lacks model coverage, KV capacity or captured identity")
    if any(
        type(value) is not int or value < 0
        for value in (
            receipt.execution_sequence,
            receipt.queue_ingress_epoch,
            receipt.frontend_request_epoch,
        )
    ):
        raise ValueError("Idle boundary requires original nonnegative owner epochs")


def _validate_scheduler_stop(receipt: SchedulerStopReceipt) -> None:
    positive = {"graphs", "registered_streams", "native_kv_owners"}
    empty = {"pending_requests", "live_requests", "pending_responses"}
    rows = receipt.owner_counts
    if not isinstance(rows, tuple) or any(
        not isinstance(row, tuple) or len(row) != 2 for row in rows
    ):
        raise ValueError("Invalid scheduler STOP resource inventory")
    counts = dict(rows)
    if (
        len(counts) != len(rows)
        or set(counts) != positive | empty
        or any(type(value) is not int for value in counts.values())
        or any(counts[name] <= 0 for name in positive)
        or any(counts[name] != 0 for name in empty)
        or type(receipt.iteration) is not int
        or receipt.iteration < 0
        or type(receipt.model_generation) is not int
        or receipt.model_generation < 0
        or type(receipt.scheduler_thread_id) is not int
        or receipt.scheduler_thread_id <= 0
        or type(receipt.registered_stream_events_completed) is not int
        or receipt.registered_stream_events_completed <= 0
        or receipt.registered_stream_events_completed != counts["registered_streams"]
        or receipt.ticket_retired is not True
        or receipt.local_issuer_join_complete is not True
        or receipt.native_kv_inventory_complete is not True
        or (
            receipt.execution_identity is not None
            and not isinstance(receipt.execution_identity, CapturedExecutionIdentity)
        )
    ):
        raise ValueError("STOP receipt lacks scheduler-owned local resource joins")


@dataclass(frozen=True)
class SessionWorkerProgress:
    identity: WorkerTaskIdentity
    transaction_id: str
    timeout_seconds: float
    stopped: SchedulerStopReceipt | None = None


@dataclass(frozen=True)
class SessionTaskBodyEnded:
    status: WorkerTaskStatus


@dataclass(frozen=True)
class SessionTransportFailure:
    task_id: int
    error_type: str
    error_text: str


class StopAuthority:
    """External authority owns registration, departure admission and the STOP quorum.

    The caller owns the authenticated ROUTER socket. Returned route/message pairs
    must be sent on that socket; the authority neither joins MPI nor accesses CUDA.
    Completing STOP grants no memory reuse, installation or resume permission.
    """

    def __init__(
        self,
        context: ServingTaskContext,
        worker_count: int,
        admitted_ranks: tuple[int, ...],
        registration_reference: str,
        departure_adapter: DepartureAdapter,
        registration_timeout: float | None = None,
        lease_seconds: float = 5.0,
    ) -> None:
        registration_timeout = (
            context.registration_timeout_seconds
            if registration_timeout is None
            else registration_timeout
        )
        _check_timeout(registration_timeout, 900.0)
        _check_timeout(lease_seconds)
        if (
            worker_count < 2
            or not registration_reference
            or not departure_adapter.adapter_id
            or len(set(admitted_ranks)) != len(admitted_ranks)
            or any(rank <= 0 or rank >= worker_count for rank in admitted_ranks)
        ):
            raise ValueError("Invalid admitted original-worker set")
        self.context = context
        self.worker_count = worker_count
        self.authority_id = uuid.uuid4().hex
        self._admitted_ranks = frozenset(admitted_ranks)
        self._reference = registration_reference
        self._adapter = departure_adapter
        self._lease_seconds = lease_seconds
        self._deadline = time.monotonic() + registration_timeout
        self._roles: dict[str, bytes] = {}
        self._routes: dict[int, bytes] = {}
        self._started: dict[int, WorkerTaskIdentity] = {}
        self._constructed: set[int] = set()
        self._registration: TaskRegistration | None = None
        self._lease_sequence = 0
        self._departure: ProcessDeparture | None = None
        self._pause: AdmissionClose | None = None
        self._closed: AdmissionClosed | None = None
        self._hold: TaskRetentionRequest | None = None
        self._hold_ack: TaskRetentionAck | None = None
        self._stopped: dict[int, SchedulerStopReceipt] = {}
        self._idle: dict[int, SchedulerIdleReceipt] = {}
        self._proposal: RecoveryProposal | None = None
        self._transport_failure: SessionTransportFailure | None = None
        self._terminal_status: WorkerTaskStatus | None = None
        self._membership_error: Exception | None = None

    def check_deadline(self) -> None:
        if self._membership_error is not None:
            raise RuntimeError(
                "Original membership observation is terminal"
            ) from self._membership_error
        if self._terminal_status is not None:
            raise RuntimeError("Original serving task ended; STOP permission is unavailable")
        if time.monotonic() >= self._deadline:
            raise TimeoutError("Serving-task registration or STOP deadline expired")

    def accept(self, route: bytes, message: object) -> tuple[tuple[bytes, object], ...]:
        self.check_deadline()
        if not route:
            raise ValueError("Receipt requires an authenticated route")
        if self._registration is None:
            return self._accept_startup(route, message)
        if isinstance(message, AdmissionClosed):
            if (
                route != self._roles["frontend"]
                or message.command != self._pause
                or type(message.request_epoch) is not int
                or message.request_epoch < 0
                or not message.owner_reference
            ):
                raise ValueError("Admission receipt does not bind the original frontend pause")
            if self._closed is not None:
                if self._closed != message:
                    raise ValueError("Frontend changed its admission receipt")
                return ()
            self._closed = message
            if self._departure is None:
                remaining = self._deadline - time.monotonic()
                _check_timeout(remaining)
                return tuple(
                    (self._routes[i], IdleBoundaryCommand(identity, message, remaining))
                    for i, identity in self._started.items()
                )
            return self._request_stop()
        if isinstance(message, SchedulerIdleReceipt):
            _validate_idle_receipt(message)
            local = message.local_receipt
            identity = local.identity
            if (
                self._closed is None
                or self._departure is not None
                or self._started.get(identity.invocation_id) != identity
                or self._routes.get(identity.invocation_id) != route
                or local.transaction_id != self._pause.transaction_id
                or local.model_generation != identity.generation
                or message.frontend_request_epoch != self._closed.request_epoch
                or not local.owner_reference
                or message.model_admission.logical_capacity != self.worker_count
            ):
                raise ValueError("Idle receipt lacks the original sealed admission boundary")
            if (
                self._idle
                and message.model_admission.shared_geometry
                != next(iter(self._idle.values())).model_admission.shared_geometry
            ):
                raise ValueError("Original workers disagree on model placement or capacity")
            previous = self._idle.setdefault(identity.invocation_id, message)
            if previous != message:
                raise ValueError("Original scheduler changed its idle boundary")
            return ()
        if isinstance(message, TaskRetentionAck):
            if (
                route != self._roles["session"]
                or message.request != self._hold
                or message.survivors != self.survivors
            ):
                raise ValueError("MPI retention receipt conflicts with the admitted task")
            if self._hold_ack is not None and self._hold_ack != message:
                raise ValueError("Session changed its retention receipt")
            self._hold_ack = message
            return ()
        if isinstance(message, SessionTransportFailure):
            if route != self._roles["session"] or message.task_id != self.context.task_id:
                raise ValueError("Transport suspicion lacks original session provenance")
            self._transport_failure = message
            return ()
        if isinstance(message, (WorkerTaskStatus, SchedulerStopReceipt)):
            return self._accept_worker(route, message)
        raise ValueError("Unsupported STOP protocol message")

    def _accept_startup(self, route: bytes, message: object) -> tuple[tuple[bytes, object], ...]:
        task = (self.context.task_id, self.context.expected_generation)
        if isinstance(message, TaskStarted):
            if (
                (message.task_id, message.generation) != task
                or message.worker_count != self.worker_count
                or message.role not in ("session", "frontend")
            ):
                raise ValueError("Task role conflicts with original registration")
            previous = self._roles.setdefault(message.role, route)
            if previous != route:
                raise ValueError("Original task role changed its route")
        elif isinstance(message, WorkerTaskStatus):
            identity = message.identity
            if (
                (identity.task_id, identity.generation) != task
                or not 0 <= identity.invocation_id < self.worker_count
                or not 0 <= identity.worker.rank < self.worker_count
                or identity.worker.pid <= 0
                or identity.worker.start_time is None
                or identity.worker.pid_namespace is None
                or not identity.worker.hostname
                or message.error_type
            ):
                raise ValueError("Worker registration lacks its original process birth")
            previous_route = self._routes.setdefault(identity.invocation_id, route)
            if previous_route != route:
                raise ValueError("Original worker invocation changed its route")
            if message.kind == "TASK_START":
                previous = self._started.setdefault(identity.invocation_id, identity)
                if previous != identity:
                    raise ValueError("Original invocation changed its process birth")
            elif message.kind == "EXECUTOR_CONSTRUCTED":
                if self._started.get(identity.invocation_id) != identity:
                    raise ValueError("Construction receipt precedes original task start")
                self._constructed.add(identity.invocation_id)
            elif (
                message.kind != "TASK_CONTROL_ALIVE"
                or identity.invocation_id not in self._constructed
            ):
                raise ValueError("Original task ended before registration")
        else:
            raise ValueError("Unexpected original-task startup receipt")
        if len(self._roles) != 2 or len(self._constructed) != self.worker_count:
            return ()
        identities = tuple(self._started[i] for i in range(self.worker_count))
        workers = tuple(identity.worker for identity in identities)
        routes = tuple(self._roles.values()) + tuple(self._routes.values())
        if (
            len(set(routes)) != len(routes)
            or len(
                {
                    (worker.hostname, worker.pid_namespace, worker.pid, worker.start_time)
                    for worker in workers
                }
            )
            != self.worker_count
            or {worker.rank for worker in workers} != set(range(self.worker_count))
        ):
            raise ValueError("Original task roles or logical ranks are not distinct")
        self._adapter.register(workers)
        registration = TaskRegistration(
            *task, workers, self._reference, self.authority_id, self._lease_seconds
        )
        self._registration = registration
        self._deadline = math.inf
        return tuple((route, registration) for route in self._roles.values()) + tuple(
            (self._routes[identity.invocation_id], WorkerTaskRegistered(identity, registration))
            for identity in identities
        )

    @property
    def survivors(self) -> tuple[WorkerProcessIdentity, ...]:
        if self._registration is None or self._departure is None:
            return ()
        return tuple(
            worker for worker in self._registration.workers if worker != self._departure.worker
        )

    def begin_process_loss(
        self, departure: ProcessDeparture, transaction_id: str, timeout_seconds: float
    ) -> tuple[bytes, AdmissionClose]:
        self.check_deadline()
        _check_timeout(timeout_seconds)
        if self._pause is not None or not transaction_id:
            raise ValueError("Original transaction already closed frontend admission")
        self._bind_departure(departure)
        self._deadline = time.monotonic() + timeout_seconds
        self._pause = AdmissionClose(
            self.context.task_id,
            self.context.expected_generation,
            self.authority_id,
            transaction_id,
            timeout_seconds,
        )
        return self._roles["frontend"], self._pause

    def prepare_idle_boundary(
        self, transaction_id: str, timeout_seconds: float
    ) -> tuple[bytes, AdmissionClose]:
        """Prepare controlled idle loss; this does not authorize membership removal."""
        self.check_deadline()
        _check_timeout(timeout_seconds)
        if self._registration is None or self._pause is not None or not transaction_id:
            raise ValueError("Idle boundary requires one registered original transaction")
        self._deadline = time.monotonic() + timeout_seconds
        self._pause = AdmissionClose(
            self.context.task_id,
            self.context.expected_generation,
            self.authority_id,
            transaction_id,
            timeout_seconds,
        )
        return self._roles["frontend"], self._pause

    @property
    def idle_boundary_complete(self) -> bool:
        self.check_deadline()
        self._check_original_membership()
        return self._idle_boundary_ready()

    def _idle_boundary_ready(self) -> bool:
        return (
            self._closed is not None
            and self._departure is None
            and set(self._idle) == set(self._started)
            and bool(self._admitted_model_ranks())
        )

    def _admitted_model_ranks(self) -> tuple[int, ...]:
        if self._departure is not None:
            return ()
        if set(self._idle) != set(self._started) or not self._idle:
            return ()
        model_ranks = next(iter(self._idle.values())).model_admission.admitted_ranks
        return tuple(rank for rank in model_ranks if rank in self._admitted_ranks)

    @property
    def admitted_idle_ranks(self) -> tuple[int, ...]:
        self.check_deadline()
        self._check_original_membership()
        return self._admitted_model_ranks()

    def begin_idle_process_loss(
        self, departure: ProcessDeparture
    ) -> tuple[tuple[bytes, object], ...]:
        """Use verified death only after all original workers sealed their idle boundary."""
        self.check_deadline()
        if not self._idle_boundary_ready():
            raise ValueError("Controlled loss requires every original scheduler idle receipt")
        if (
            not isinstance(departure, ProcessDeparture)
            or departure.worker.rank not in self._admitted_model_ranks()
        ):
            raise ValueError("Worker loss is outside the actual model admission set")
        self._bind_departure(departure)
        return self._request_stop()

    def _check_original_membership(self, pending_departure: ProcessDeparture | None = None) -> None:
        if self._registration is None:
            return
        expected = self._departure if self._departure is not None else pending_departure
        try:
            for worker in self._registration.workers:
                observed = self._adapter.observe_departure(worker)
                if observed is None:
                    continue
                if (
                    not isinstance(observed, ProcessDeparture)
                    or not self._adapter.owns_receipt(observed)
                    or observed.worker != worker
                    or observed.task_id != self.context.task_id
                    or observed.generation != self.context.expected_generation
                    or observed.registration_reference != self._reference
                    or observed.adapter_id != self._adapter.adapter_id
                    or observed is not expected
                ):
                    raise RuntimeError("Original membership exceeded the single admitted loss")
        except (OSError, RuntimeError, ValueError, zmq.ZMQError) as error:
            self._membership_error = error
            raise

    def _bind_departure(self, departure: ProcessDeparture) -> None:
        if (
            self._registration is None
            or self._departure is not None
            or not isinstance(departure, ProcessDeparture)
            or not self._adapter.owns_receipt(departure)
            or departure.task_id != self.context.task_id
            or departure.generation != self.context.expected_generation
            or departure.registration_reference != self._reference
            or departure.adapter_id != self._adapter.adapter_id
            or departure.worker not in self._registration.workers
            or departure.worker.rank not in self._admitted_ranks
        ):
            raise ValueError("STOP requires adapter-owned admitted original-process departure")
        self._check_original_membership(pending_departure=departure)
        self._departure = departure

    def _request_stop(self) -> tuple[tuple[bytes, object], ...]:
        if self._closed is None or self._departure is None or self._pause is None:
            raise RuntimeError("Frontend admission has not closed")
        remaining = self._deadline - time.monotonic()
        _check_timeout(remaining)
        self._hold = TaskRetentionRequest(
            self.context.task_id,
            self._pause.transaction_id,
            self.context.expected_generation,
            self._registration.workers,
            self._departure.worker,
            remaining,
            self._departure.evidence_id,
        )
        commands: list[tuple[bytes, object]] = [(self._roles["session"], self._hold)]
        commands.extend(
            (self._routes[i], WorkerStopCommand(identity, self._pause.transaction_id, remaining))
            for i, identity in self._started.items()
            if identity.worker != self._departure.worker
        )
        return tuple(commands)

    def _accept_worker(
        self, route: bytes, message: WorkerTaskStatus | SchedulerStopReceipt
    ) -> tuple[tuple[bytes, object], ...]:
        identity = message.identity
        if (
            self._started.get(identity.invocation_id) != identity
            or self._routes.get(identity.invocation_id) != route
        ):
            raise ValueError("Worker receipt lacks original authenticated provenance")
        if isinstance(message, WorkerTaskStatus):
            if message.kind == "TASK_BODY_END":
                self._terminal_status = message
                return ((self._roles["session"], SessionTaskBodyEnded(message)),)
            if message.kind != "TASK_CONTROL_ALIVE" or message.error_type:
                raise ValueError("Unexpected original-task lifetime receipt")
            stopped = None
        else:
            _validate_scheduler_stop(message)
            if (
                self._hold is None
                or identity.worker == self._departure.worker
                or message.transaction_id != self._hold.transaction_id
                or message.model_generation != identity.generation
                or message.scheduler_thread_id <= 0
                or not message.owner_reference
                or message.registered_stream_events_completed <= 0
                or not message.ticket_retired
                or not message.local_issuer_join_complete
                or not message.native_kv_inventory_complete
            ):
                raise ValueError("STOP receipt lacks scheduler-owned local resource joins")
            idle = self._idle.get(identity.invocation_id)
            if (
                idle is not None
                and message.execution_identity != idle.local_receipt.execution_identity
            ):
                raise ValueError("Original captured execution changed after idle admission")
            previous = self._stopped.setdefault(identity.invocation_id, message)
            if previous != message:
                raise ValueError("Original scheduler changed its STOP receipt")
            stopped = message
        if self._hold is None:
            return ()
        if identity.worker == self._departure.worker:
            raise ValueError("Departed worker cannot report survivor progress")
        return (
            (
                self._roles["session"],
                SessionWorkerProgress(
                    identity,
                    self._hold.transaction_id,
                    min(self._deadline - time.monotonic(), self._lease_seconds),
                    stopped,
                ),
            ),
        )

    def heartbeat(self) -> tuple[tuple[bytes, AuthorityLease], ...]:
        self.check_deadline()
        if self._registration is None:
            return ()
        self._lease_sequence += 1
        lease = AuthorityLease(
            self.context.task_id,
            self.context.expected_generation,
            self._reference,
            self._lease_sequence,
            self._lease_seconds,
            self.authority_id,
        )
        routes = tuple(self._roles.values()) + tuple(
            route
            for i, route in self._routes.items()
            if self._departure is None or self._started[i].worker != self._departure.worker
        )
        return tuple((route, lease) for route in routes)

    @property
    def stop_complete(self) -> bool:
        self.check_deadline()
        self._check_original_membership()
        return (
            self._hold_ack is not None
            and self._closed is not None
            and {receipt.identity.worker for receipt in self._stopped.values()}
            == set(self.survivors)
        )

    def begin_preparation(
        self, proposal: RecoveryProposal
    ) -> tuple[tuple[bytes, RecoveryProposal], ...]:
        """Allow nonmutating preparation after every original survivor acknowledged STOP."""
        if not self.stop_complete:
            raise ValueError("Recovery preparation requires complete scheduler STOP")
        _validate_proposal(
            proposal, self._registration, self._hold.transaction_id, self._departure.worker
        )
        if self._proposal is not None:
            if self._proposal != proposal:
                raise ValueError("Conflicting original recovery proposal")
            return ()
        if set(self._idle) != set(self._started) or not self._idle:
            raise ValueError("Recovery preparation requires original idle placement catalog")
        admission = next(iter(self._idle.values())).model_admission
        expected_digest = dict(admission.masked_placement_digests).get(self._departure.worker.rank)
        if proposal.placement_digest != expected_digest:
            raise ValueError("Recovery proposal does not bind the admitted masked placement")
        self._proposal = proposal
        return tuple(
            (self._routes[i], proposal)
            for i, identity in self._started.items()
            if identity.worker != self._departure.worker
        )


class SchedulerStopQueue(Protocol):
    def scheduler_stop_publication(self, control_id: str) -> AbstractContextManager[None]: ...

    def enqueue_scheduler_stop(self, control_id: str, timeout_seconds: float) -> None: ...

    def enqueue_idle_boundary(self, control_id: str, timeout_seconds: float) -> None: ...

    def idle_boundary_publication(
        self, control_id: str, expected_epoch: int
    ) -> AbstractContextManager[None]: ...

    def validate_idle_boundary(self, control_id: str, expected_epoch: int) -> None: ...


class ServingTaskControl:
    """Worker socket ownership; scheduler/model receipts are pushed by their owners."""

    def __init__(self, context: ServingTaskContext, identity: WorkerTaskIdentity) -> None:
        if (
            identity.task_id != context.task_id
            or identity.generation != context.expected_generation
            or identity.worker.start_time is None
            or identity.worker.pid_namespace is None
        ):
            raise ValueError("Worker control requires the original process identity")
        self.context = context
        self.identity = identity
        self._outgoing: SimpleQueue[object] = SimpleQueue()
        self._closing = threading.Event()
        self._stop_available = threading.Event()
        self._lock = threading.Lock()
        self._stop_queue: SchedulerStopQueue | None = None
        self._stop: WorkerStopCommand | None = None
        self._stop_deadline: float | None = None
        self._idle_command: IdleBoundaryCommand | None = None
        self._idle_receipt: SchedulerIdleReceipt | None = None
        self._idle_deadline: float | None = None
        self._idle_available = threading.Event()
        self._proposal: RecoveryProposal | None = None
        self._proposal_available = threading.Event()
        self._registration: TaskRegistration | None = None
        self._authority_deadline: float | None = (
            time.monotonic() + context.registration_timeout_seconds
        )
        self._lease_sequence = 0
        self._body_ended = False
        self._stopped: SchedulerStopReceipt | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._serve, name="mpi-task-control")

    def start(self) -> None:
        self._outgoing.put(WorkerTaskStatus(self.identity, "TASK_START", threading.get_ident()))
        self._thread.start()

    def executor_constructed(self, stop_queue: SchedulerStopQueue) -> None:
        with self._lock:
            if self._error is not None or self._stop_queue is not None or self._body_ended:
                raise RuntimeError("Worker control construction is unavailable")
            self._stop_queue = stop_queue
        self._outgoing.put(
            WorkerTaskStatus(self.identity, "EXECUTOR_CONSTRUCTED", threading.get_ident())
        )

    def task_body_ended(self, error_type: str) -> None:
        with self._lock:
            self._body_ended = True
        self._outgoing.put(
            WorkerTaskStatus(self.identity, "TASK_BODY_END", threading.get_ident(), error_type)
        )

    def _accept_stop(self, command: WorkerStopCommand) -> None:
        _check_timeout(command.timeout_seconds)
        with self._lock:
            if command.identity != self.identity or not command.transaction_id or self._body_ended:
                raise ValueError("STOP does not bind this original worker task")
            if self._registration is None or self._stop_queue is None:
                raise ValueError("STOP precedes original executor registration")
            if self._stop is not None:
                if self._stop != command:
                    raise ValueError("Conflicting worker STOP")
                return
            if self._idle_command is not None:
                if (
                    self._idle_receipt is None
                    or command.transaction_id != self._idle_command.transaction_id
                ):
                    raise ValueError("STOP does not bind a sealed original idle boundary")
                self._stop_queue.validate_idle_boundary(
                    command.transaction_id, self._idle_receipt.queue_ingress_epoch
                )
            self._stop_deadline = time.monotonic() + command.timeout_seconds
            self._stop_queue.enqueue_scheduler_stop(command.transaction_id, command.timeout_seconds)
            self._stop = command
            self._stop_available.set()

    def _accept_idle_boundary(self, command: IdleBoundaryCommand) -> None:
        _check_timeout(command.timeout_seconds)
        with self._lock:
            self._check_live()
            closed = command.admission_closed
            registration = self._registration
            if (
                registration is None
                or self._stop_queue is None
                or self._stop is not None
                or command.identity != self.identity
                or closed.command.task_id != self.identity.task_id
                or closed.command.generation != self.identity.generation
                or closed.command.authority_id != registration.authority_id
                or not command.transaction_id
                or type(closed.request_epoch) is not int
                or closed.request_epoch < 0
                or not closed.owner_reference
            ):
                raise ValueError("Idle boundary lacks original frontend/worker registration")
            if self._idle_command is not None:
                if self._idle_command != command:
                    raise ValueError("Conflicting original idle boundary")
                return
            self._stop_queue.enqueue_idle_boundary(command.transaction_id, command.timeout_seconds)
            self._idle_command = command
            self._idle_deadline = time.monotonic() + command.timeout_seconds
            self._idle_available.set()

    def wait_for_idle_boundary(self, timeout_seconds: float) -> IdleBoundaryCommand:
        _check_timeout(timeout_seconds)
        if not self._idle_available.wait(timeout_seconds):
            raise TimeoutError("No authenticated idle boundary arrived")
        with self._lock:
            self._check_live()
            return self._idle_command

    @property
    def idle_boundary_deadline(self) -> float:
        with self._lock:
            self._check_live()
            if self._idle_deadline is None:
                raise RuntimeError("No original idle boundary is bound")
            return self._idle_deadline

    def publish_idle(self, receipt: SchedulerIdleReceipt) -> None:
        """Publish local joins only while the original queue's issuance seal is held."""
        _validate_idle_receipt(receipt)
        local = receipt.local_receipt
        with self._lock:
            self._check_live()
            command = self._idle_command
            if (
                command is None
                or self._stop is not None
                or local.identity != self.identity
                or local.transaction_id != command.transaction_id
                or local.model_generation != self.identity.generation
                or local.scheduler_thread_id != threading.get_ident()
                or receipt.frontend_request_epoch != command.frontend_request_epoch
                or not local.owner_reference
            ):
                raise ValueError("Idle publication does not bind the original sealed scheduler")
            if self._idle_receipt is not None:
                if self._idle_receipt != receipt:
                    raise ValueError("Original scheduler changed its idle boundary")
                return
            with self._stop_queue.idle_boundary_publication(
                command.transaction_id, receipt.queue_ingress_epoch
            ):
                self._check_live()
                self._idle_receipt = receipt
                self._outgoing.put(receipt)

    def _accept_proposal(self, proposal: RecoveryProposal) -> None:
        with self._lock:
            self._check_live()
            if self._stopped is None or self._stop is None or self._registration is None:
                raise ValueError("Recovery preparation precedes the original scheduler STOP")
            _validate_proposal(
                proposal,
                self._registration,
                self._stop.transaction_id,
                proposal.failed_worker,
            )
            if self.identity.worker not in proposal.survivors:
                raise ValueError("Departed worker cannot prepare survivor state")
            if self._proposal is not None:
                if self._proposal != proposal:
                    raise ValueError("Conflicting original recovery proposal")
                return
            self._proposal = proposal
            self._proposal_available.set()

    def wait_for_recovery_proposal(self, timeout_seconds: float) -> RecoveryProposal:
        _check_timeout(timeout_seconds)
        if not self._proposal_available.wait(timeout_seconds):
            raise TimeoutError("No original recovery proposal arrived")
        with self._lock:
            self._check_live()
            return self._proposal

    def validate_preparation(
        self, proposal: RecoveryProposal, stopped: SchedulerStopReceipt
    ) -> RecoveryProposal:
        """Verify accepted transport intent without granting any model-state publication."""
        if not isinstance(proposal, RecoveryProposal) or not isinstance(
            stopped, SchedulerStopReceipt
        ):
            raise ValueError("Preparation requires original accepted proposal and local STOP")
        with self._lock:
            self._check_live()
            if (
                self._proposal is not proposal
                or self._stopped is not stopped
                or stopped.scheduler_thread_id != threading.get_ident()
            ):
                raise ValueError("Preparation requires original accepted proposal and local STOP")
            return proposal

    def wait_for_scheduler_stop(self, timeout_seconds: float) -> WorkerStopCommand:
        _check_timeout(timeout_seconds)
        if not self._stop_available.wait(timeout_seconds):
            raise TimeoutError("No authenticated scheduler STOP arrived")
        with self._lock:
            self._check_live()
            return self._stop

    def publish_stopped(self, receipt: SchedulerStopReceipt) -> None:
        """Called by the scheduler after joining its model, copies and idle KV owners."""
        _validate_scheduler_stop(receipt)
        with self._lock:
            self._check_live()
            if (
                self._stop is None
                or receipt.identity != self.identity
                or receipt.transaction_id != self._stop.transaction_id
                or receipt.scheduler_thread_id != threading.get_ident()
            ):
                raise ValueError("STOP publication does not bind the original scheduler")
            if self._stopped is not None:
                if self._stopped != receipt:
                    raise ValueError("Original scheduler changed its STOP receipt")
                return
            with self._stop_queue.scheduler_stop_publication(receipt.transaction_id):
                self._check_live()
                self._stopped = receipt
                self._outgoing.put(receipt)

    @property
    def scheduler_stop_deadline(self) -> float:
        with self._lock:
            self._check_live()
            if self._stop is None or self._stop_deadline is None:
                raise RuntimeError("No authenticated scheduler STOP is bound")
            return self._stop_deadline

    @property
    def stop_requested(self) -> bool:
        with self._lock:
            return self._stop is not None

    def _check_live(self) -> None:
        if self._error is not None or self._body_ended:
            raise RuntimeError("Original worker control is unavailable") from self._error
        self._check_deadlines()

    def _check_deadlines(self) -> None:
        now = time.monotonic()
        if (
            (self._authority_deadline is not None and now >= self._authority_deadline)
            or (self._stop_deadline is not None and now >= self._stop_deadline)
            or (self._idle_deadline is not None and now >= self._idle_deadline)
        ):
            raise TimeoutError("Worker authority or STOP deadline expired")

    def _accept_registration(self, message: WorkerTaskRegistered) -> None:
        registration = message.registration
        _check_timeout(registration.authority_lease_seconds)
        with self._lock:
            if (
                message.identity != self.identity
                or self._registration is not None
                or registration.task_id != self.identity.task_id
                or registration.generation != self.identity.generation
                or self.identity.worker not in registration.workers
                or not registration.registration_reference
                or not registration.authority_id
            ):
                raise ValueError("Worker authority registration conflicts with original task")
            self._registration = registration
            self._authority_deadline = time.monotonic() + registration.authority_lease_seconds

    def _accept_lease(self, message: AuthorityLease) -> None:
        with self._lock:
            self._check_live()
            registration = self._registration
            if (
                registration is None
                or message.task_id != registration.task_id
                or message.generation != registration.generation
                or message.registration_reference != registration.registration_reference
                or message.authority_id != registration.authority_id
                or message.timeout_seconds != registration.authority_lease_seconds
                or type(message.sequence) is not int
                or message.sequence <= self._lease_sequence
            ):
                raise ValueError("Worker authority lease conflicts or is stale")
            self._lease_sequence = message.sequence
            self._authority_deadline = time.monotonic() + message.timeout_seconds

    def _serve(self) -> None:
        from .ipc import ZeroMqQueue

        queue = None
        last_progress = time.monotonic()
        try:
            queue = ZeroMqQueue(
                self.context.authority_address,
                is_server=False,
                socket_type=zmq.DEALER,
                use_hmac_encryption=True,
            )
            while not self._closing.is_set() or not self._outgoing.empty():
                with self._lock:
                    if not self._closing.is_set():
                        self._check_deadlines()
                        if self._idle_receipt is not None and self._stop is None:
                            self._stop_queue.validate_idle_boundary(
                                self._idle_command.transaction_id,
                                self._idle_receipt.queue_ingress_epoch,
                            )
                if not self._closing.is_set() and queue.poll(0.01):
                    message = queue.get()
                    if isinstance(message, WorkerTaskRegistered):
                        self._accept_registration(message)
                    elif isinstance(message, AuthorityLease):
                        self._accept_lease(message)
                    elif isinstance(message, WorkerStopCommand):
                        self._accept_stop(message)
                    elif isinstance(message, IdleBoundaryCommand):
                        self._accept_idle_boundary(message)
                    elif isinstance(message, RecoveryProposal):
                        self._accept_proposal(message)
                    else:
                        raise ValueError("Unsupported worker STOP message")
                now = time.monotonic()
                with self._lock:
                    if (
                        not self._body_ended
                        and self._stop_queue is not None
                        and now - last_progress >= 1.0
                    ):
                        self._outgoing.put(
                            WorkerTaskStatus(
                                self.identity, "TASK_CONTROL_ALIVE", threading.get_ident()
                            )
                        )
                        last_progress = now
                try:
                    message = self._outgoing.get_nowait()
                except Empty:
                    continue
                queue.put_nowait(message)
        except (RuntimeError, ValueError, TimeoutError, zmq.ZMQError) as error:
            with self._lock:
                self._error = error
        finally:
            if queue is not None:
                queue.socket.setsockopt(zmq.LINGER, 0)
                queue.close()

    def close(self) -> None:
        self._closing.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=1.0)
        if self._thread.is_alive():
            raise RuntimeError("Worker control owner did not terminate")
        if self._error is not None:
            raise RuntimeError("Worker control owner failed") from self._error


class FrontendTaskControl:
    """Frontend socket owner; the proxy alone owns and certifies admission closure."""

    def __init__(
        self,
        context: ServingTaskContext,
        worker_count: int,
        close_admission: Callable[[AdmissionClose], AdmissionClosed],
    ) -> None:
        self.context = context
        self._worker_count = worker_count
        self._close_admission = close_admission
        self._registration: TaskRegistration | None = None
        self._authority_deadline = time.monotonic() + context.registration_timeout_seconds
        self._stop_deadline: float | None = None
        self._lease_sequence = 0
        self._error: BaseException | None = None
        self._closing = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="frontend-task-control")

    def start(self) -> None:
        self._thread.start()

    def check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("Frontend serving control failed") from self._error
        now = time.monotonic()
        if now >= self._authority_deadline or (
            self._stop_deadline is not None and now >= self._stop_deadline
        ):
            raise TimeoutError("Frontend authority or STOP deadline expired")

    def _accept(self, message: object) -> AdmissionClosed | None:
        self.check_error()
        if isinstance(message, TaskRegistration):
            _check_timeout(message.authority_lease_seconds)
            if (
                self._registration is not None
                or message.task_id != self.context.task_id
                or message.generation != self.context.expected_generation
                or len(message.workers) != self._worker_count
                or not message.registration_reference
                or not message.authority_id
            ):
                raise ValueError("Frontend registration conflicts with original serving task")
            self._registration = message
            self._authority_deadline = time.monotonic() + message.authority_lease_seconds
            return None
        registration = self._registration
        if registration is None:
            raise ValueError("Frontend command precedes original registration")
        if isinstance(message, AuthorityLease):
            if (
                message.task_id != registration.task_id
                or message.generation != registration.generation
                or message.registration_reference != registration.registration_reference
                or message.authority_id != registration.authority_id
                or message.timeout_seconds != registration.authority_lease_seconds
                or type(message.sequence) is not int
                or message.sequence <= self._lease_sequence
            ):
                raise ValueError("Frontend authority lease conflicts or is stale")
            self._lease_sequence = message.sequence
            self._authority_deadline = time.monotonic() + message.timeout_seconds
            return None
        if not isinstance(message, AdmissionClose):
            raise ValueError("Unsupported frontend STOP protocol message")
        _check_timeout(message.timeout_seconds)
        if (
            self._stop_deadline is not None
            or message.task_id != registration.task_id
            or message.generation != registration.generation
            or message.authority_id != registration.authority_id
            or not message.transaction_id
        ):
            raise ValueError("Frontend admission closure conflicts with original task")
        self._stop_deadline = time.monotonic() + message.timeout_seconds
        receipt = self._close_admission(message)
        if (
            not isinstance(receipt, AdmissionClosed)
            or receipt.command is not message
            or not receipt.owner_reference
            or receipt.request_epoch < 0
        ):
            raise ValueError("Proxy did not produce its original admission receipt")
        self.check_error()
        return receipt

    def _serve(self) -> None:
        from .ipc import ZeroMqQueue

        queue = None
        try:
            queue = ZeroMqQueue(
                self.context.authority_address,
                is_server=False,
                socket_type=zmq.DEALER,
                use_hmac_encryption=True,
            )
            queue.put_nowait(
                TaskStarted(
                    self.context.task_id,
                    self.context.expected_generation,
                    self._worker_count,
                    "frontend",
                )
            )
            while not self._closing.is_set():
                self.check_error()
                if queue.poll(0.01):
                    receipt = self._accept(queue.get())
                    if receipt is not None:
                        queue.put_nowait(receipt)
        except (RuntimeError, ValueError, TimeoutError, zmq.ZMQError) as error:
            self._error = error
        finally:
            if queue is not None:
                queue.socket.setsockopt(zmq.LINGER, 0)
                queue.close()

    def close(self) -> None:
        self._closing.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=1.0)
        if self._thread.is_alive():
            raise RuntimeError("Frontend control owner did not terminate")

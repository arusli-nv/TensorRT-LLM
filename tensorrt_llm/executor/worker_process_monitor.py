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

"""Linux process-liveness monitoring for locally spawned executor workers."""

from __future__ import annotations

import logging
import math
import os
import select
import socket
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Literal, NamedTuple, Optional

if TYPE_CHECKING:
    from .ipc import ZeroMqQueue

logger = logging.getLogger(__name__)


class WorkerProcessIdentity(NamedTuple):
    """Stable identity for one executor worker process."""

    rank: int
    pid: int
    start_time: Optional[int]
    hostname: str
    pid_namespace: Optional[int]


def _read_process_state(pid: int) -> Optional[tuple[str, int]]:
    """Read the Linux process state and start time from ``/proc``.

    Returns:
        A ``(state, start_time)`` tuple, or ``None`` when the process does not
        exist or procfs is unavailable.
    """
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as stat_file:
            stat = stat_file.read()
    except (FileNotFoundError, ProcessLookupError):
        return None

    # The second field is parenthesized and may contain spaces or parentheses.
    command_end = stat.rfind(")")
    if command_end < 0:
        return None
    fields = stat[command_end + 2 :].split()
    if len(fields) <= 19:
        return None
    return fields[0], int(fields[19])


def _read_pid_namespace(pid: int) -> Optional[int]:
    """Return the inode identifying a Linux PID namespace."""
    try:
        return os.stat(f"/proc/{pid}/ns/pid").st_ino
    except OSError:
        return None


def capture_worker_process_identity(rank: int, pid: Optional[int] = None) -> WorkerProcessIdentity:
    """Capture the calling worker's rank, PID, and Linux start time."""
    pid = pid if pid is not None else os.getpid()
    try:
        process_state = _read_process_state(pid)
    except (OSError, ValueError):
        process_state = None
    start_time = process_state[1] if process_state is not None else None
    return WorkerProcessIdentity(
        rank=rank,
        pid=pid,
        start_time=start_time,
        hostname=socket.gethostname(),
        pid_namespace=_read_pid_namespace(pid),
    )


class WorkerProcessMonitor:
    """Monitor locally spawned workers without relying on MPI futures.

    Linux pidfds are used when Python exposes ``os.pidfd_open``.  A procfs
    identity check is retained as a fallback for older Python versions.  If
    neither mechanism is available, the worker is left to the existing MPI
    future/error-queue checks.
    """

    def __init__(self) -> None:
        self._pidfd_to_identity: Dict[int, WorkerProcessIdentity] = {}
        self._procfs_identities: List[WorkerProcessIdentity] = []
        self._dead_identity: Optional[WorkerProcessIdentity] = None
        self._poller = select.poll() if hasattr(select, "poll") else None
        self._lock = threading.Lock()
        self._local_hostname = socket.gethostname()
        self._local_pid_namespace = _read_pid_namespace(os.getpid())

    def register(self, identities: List[WorkerProcessIdentity]) -> None:
        """Register locally spawned worker identities for liveness checks."""
        with self._lock:
            self._close_unlocked()
            pidfd_open = getattr(os, "pidfd_open", None)
            for identity in identities:
                if identity.hostname != self._local_hostname:
                    continue
                if identity.pid_namespace != self._local_pid_namespace:
                    continue
                if pidfd_open is not None and self._poller is not None:
                    try:
                        pidfd = pidfd_open(identity.pid)
                    except ProcessLookupError:
                        self._dead_identity = identity
                        continue
                    except OSError as error:
                        logger.debug("pidfd_open(%d) failed: %s", identity.pid, error)
                    else:
                        if identity.start_time is not None:
                            try:
                                process_state = _read_process_state(identity.pid)
                            except (OSError, ValueError) as error:
                                logger.debug(
                                    "Failed to validate process identity for pid %d: %s",
                                    identity.pid,
                                    error,
                                )
                            else:
                                if (
                                    process_state is None
                                    or process_state[0] == "Z"
                                    or process_state[1] != identity.start_time
                                ):
                                    os.close(pidfd)
                                    self._dead_identity = identity
                                    continue
                        self._pidfd_to_identity[pidfd] = identity
                        self._poller.register(
                            pidfd, select.POLLIN | select.POLLERR | select.POLLHUP
                        )
                        continue

                if identity.start_time is not None:
                    self._procfs_identities.append(identity)

    def find_dead_worker(self) -> Optional[WorkerProcessIdentity]:
        """Return the first worker known to have exited, if any."""
        with self._lock:
            if self._dead_identity is not None:
                return self._dead_identity

            if self._poller is not None:
                for pidfd, _ in self._poller.poll(0):
                    identity = self._pidfd_to_identity.get(pidfd)
                    if identity is not None:
                        self._dead_identity = identity
                        return identity

            for identity in self._procfs_identities:
                try:
                    process_state = _read_process_state(identity.pid)
                except (OSError, ValueError) as error:
                    logger.debug(
                        "Failed to check process state for pid %d: %s", identity.pid, error
                    )
                    continue
                if (
                    process_state is None
                    or process_state[0] == "Z"
                    or process_state[1] != identity.start_time
                ):
                    self._dead_identity = identity
                    return identity
            return None

    def close(self) -> None:
        """Close all process handles and reset the monitor."""
        with self._lock:
            self._close_unlocked()

    def _close_unlocked(self) -> None:
        for pidfd in self._pidfd_to_identity:
            if self._poller is not None:
                try:
                    self._poller.unregister(pidfd)
                except (KeyError, OSError):
                    pass
            try:
                os.close(pidfd)
            except OSError:
                pass
        self._pidfd_to_identity.clear()
        self._procfs_identities.clear()
        self._dead_identity = None


@dataclass(frozen=True)
class ProcessDeparture:
    """Original-process departure; this receipt establishes no GPU drain."""

    task_id: int
    generation: int
    worker: WorkerProcessIdentity
    registration_reference: str
    adapter_id: str
    evidence_id: str


@dataclass(frozen=True)
class NodeProcessScope:
    hostname: str
    boot_id: str
    session_id: str
    scope_reference: str


@dataclass(frozen=True)
class NodeProcessRegistration:
    task_id: int
    generation: int
    registration_reference: str
    adapter_id: str
    registration_id: str
    scope: NodeProcessScope
    workers: tuple[WorkerProcessIdentity, ...]

    def to_wire(self) -> dict:
        """Use primitives so a host observer need not import the inference package."""
        return {
            "task_id": self.task_id,
            "generation": self.generation,
            "registration_reference": self.registration_reference,
            "adapter_id": self.adapter_id,
            "registration_id": self.registration_id,
            "scope": vars(self.scope).copy(),
            "workers": [worker._asdict() for worker in self.workers],
        }


@dataclass(frozen=True)
class NodeProcessEvent:
    registration: NodeProcessRegistration
    sequence: int
    kind: Literal["REGISTERED", "ALIVE", "DEPARTED", "TERMINAL"]
    worker: WorkerProcessIdentity | None = None
    evidence_id: str = ""

    def to_wire(self) -> dict:
        return {
            "registration": self.registration.to_wire(),
            "sequence": self.sequence,
            "kind": self.kind,
            "worker": self.worker._asdict() if self.worker is not None else None,
            "evidence_id": self.evidence_id,
        }


class NodeDepartureAdapter:
    """Own departure receipts from qualified, launch-owned node observers.

    The launcher supplies original boot/session scopes and independently keyed
    channels. Each observer retains original process handles before REGISTERED;
    only their exit readiness permits DEPARTED. The adapter takes ownership of
    channels, but does not launch, signal or clean up model processes.
    """

    def __init__(
        self,
        task_id: int,
        generation: int,
        registration_reference: str,
        node_channels: Mapping[NodeProcessScope, ZeroMqQueue],
        *,
        worker_control_key: bytes,
        timeout_seconds: float = 30.0,
    ) -> None:
        if (
            type(task_id) is not int
            or task_id <= 0
            or type(generation) is not int
            or generation < 0
            or not isinstance(registration_reference, str)
            or not registration_reference
            or not node_channels
            or not isinstance(worker_control_key, bytes)
            or not worker_control_key
            or type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 60
        ):
            raise ValueError("Node departure needs a bounded original task and launcher scopes")
        keys = []
        for scope, channel in node_channels.items():
            if (
                not isinstance(scope, NodeProcessScope)
                or not all(
                    isinstance(value, str) and value
                    for value in (
                        scope.hostname,
                        scope.boot_id,
                        scope.session_id,
                        scope.scope_reference,
                    )
                )
                or channel.use_hmac_encryption is not True
                or not isinstance(channel.hmac_key, bytes)
                or not channel.hmac_key
                or channel.hmac_key == worker_control_key
            ):
                raise ValueError(
                    "Node observer lacks an independently authenticated launcher scope"
                )
            keys.append(channel.hmac_key)
        if (
            len(set(keys)) != len(keys)
            or len({scope.hostname for scope in node_channels}) != len(node_channels)
            or len({id(channel) for channel in node_channels.values()}) != len(node_channels)
        ):
            raise ValueError("Node observers must have distinct hosts, channels and credentials")
        self.adapter_id = "node-pidfd:" + uuid.uuid4().hex
        self._task_id = task_id
        self._generation = generation
        self._reference = registration_reference
        self._channels = dict(node_channels)
        self._keys = dict(zip(node_channels, keys))
        self._timeout = timeout_seconds
        self._registrations: dict[NodeProcessScope, NodeProcessRegistration] = {}
        self._sequences: dict[NodeProcessScope, int] = {}
        self._deadlines: dict[NodeProcessScope, float] = {}
        self._receipts: dict[WorkerProcessIdentity, ProcessDeparture] = {}
        self._registered = False
        self._started = False
        self._terminal = False
        self._closed = False

    def _reject(self, reason: str) -> None:
        self._terminal = True
        raise RuntimeError(reason)

    def register(self, workers: tuple[WorkerProcessIdentity, ...]) -> None:
        """Bind all original workers before accepting any departure evidence."""
        if self._started or self._closed or not workers or len(set(workers)) != len(workers):
            raise ValueError("Original node registration must be one-shot and complete")
        self._started = True
        complete = False
        try:
            if any(not isinstance(worker, WorkerProcessIdentity) for worker in workers):
                raise ValueError("Node registration requires original worker identities")
            groups = {
                scope: tuple(worker for worker in workers if worker.hostname == scope.hostname)
                for scope in self._channels
            }
            if (
                len({worker.rank for worker in workers}) != len(workers)
                or len(
                    {
                        (worker.hostname, worker.pid_namespace, worker.pid, worker.start_time)
                        for worker in workers
                    }
                )
                != len(workers)
                or sum(map(len, groups.values())) != len(workers)
                or any(not group for group in groups.values())
                or any(
                    type(worker.rank) is not int
                    or worker.rank < 0
                    or any(
                        type(value) is not int or value <= 0
                        for value in (worker.pid, worker.start_time, worker.pid_namespace)
                    )
                    for worker in workers
                )
            ):
                raise ValueError("Original worker births do not match the qualified node scopes")
            deadline = time.monotonic() + self._timeout
            for scope, group in groups.items():
                registration = NodeProcessRegistration(
                    self._task_id,
                    self._generation,
                    self._reference,
                    self.adapter_id,
                    uuid.uuid4().hex,
                    scope,
                    group,
                )
                self._registrations[scope] = registration
                self._channels[scope].put_nowait(registration.to_wire())
            while len(self._sequences) != len(self._channels):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Original node handle registration timed out")
                self._poll(min(0.01, remaining / len(self._channels)))
            self._registered = complete = True
        finally:
            if not complete:
                self.close()

    @staticmethod
    def _matches_wire(actual: object, expected: object) -> bool:
        if type(actual) is not type(expected):
            return False
        if isinstance(expected, dict):
            return actual.keys() == expected.keys() and all(
                NodeDepartureAdapter._matches_wire(actual[key], value)
                for key, value in expected.items()
            )
        if isinstance(expected, list):
            return len(actual) == len(expected) and all(
                NodeDepartureAdapter._matches_wire(left, right)
                for left, right in zip(actual, expected)
            )
        return actual == expected

    def _accept(self, scope: NodeProcessScope, payload: dict) -> None:
        registration = self._registrations[scope]
        if (
            type(payload) is not dict
            or set(payload) != {"registration", "sequence", "kind", "worker", "evidence_id"}
            or not self._matches_wire(payload["registration"], registration.to_wire())
            or type(payload["sequence"]) is not int
            or payload["sequence"] != self._sequences.get(scope, 0) + 1
            or type(payload["kind"]) is not str
            or type(payload["evidence_id"]) is not str
        ):
            self._reject("Node process event changed its original registration or sequence")
        worker = None
        if payload["worker"] is not None:
            worker = next(
                (
                    worker
                    for worker in registration.workers
                    if self._matches_wire(payload["worker"], worker._asdict())
                ),
                None,
            )
            if worker is None:
                self._reject("Node event lacks an original local worker birth")
        event = NodeProcessEvent(
            registration, payload["sequence"], payload["kind"], worker, payload["evidence_id"]
        )
        if event.kind == "REGISTERED":
            if scope in self._sequences or event.worker is not None or event.evidence_id:
                self._reject("Node handle registration is conflicting or incomplete")
        elif event.kind == "ALIVE":
            if scope not in self._sequences or event.worker is not None or event.evidence_id:
                self._reject("Node liveness event precedes original handle registration")
        elif event.kind == "DEPARTED":
            if (
                not self._registered
                or not isinstance(event.worker, WorkerProcessIdentity)
                or event.worker not in event.registration.workers
                or event.worker in self._receipts
                or not isinstance(event.evidence_id, str)
                or not event.evidence_id
                or any(
                    receipt.evidence_id == event.evidence_id for receipt in self._receipts.values()
                )
            ):
                self._reject("Departure lacks one registered original node handle")
            self._receipts[event.worker] = ProcessDeparture(
                self._task_id,
                self._generation,
                event.worker,
                self._reference,
                self.adapter_id,
                event.evidence_id,
            )
        else:
            self._reject("Node observer became terminal or sent an unsupported event")
        self._sequences[scope] = event.sequence
        self._deadlines[scope] = time.monotonic() + self._timeout

    def _poll(self, timeout: float = 0.0) -> None:
        if any(time.monotonic() >= deadline for deadline in self._deadlines.values()):
            self._terminal = True
            raise TimeoutError("Node observer lease expired; process departure is unproven")
        for scope, channel in self._channels.items():
            if channel.use_hmac_encryption is not True or channel.hmac_key != self._keys[scope]:
                self._reject("Original node channel authentication changed")
            if channel.poll(timeout):
                self._accept(scope, channel.get())
        if any(time.monotonic() >= deadline for deadline in self._deadlines.values()):
            self._terminal = True
            raise TimeoutError("Node observer lease expired; process departure is unproven")

    def observe_departure(self, worker: WorkerProcessIdentity) -> ProcessDeparture | None:
        if self._terminal or self._closed or not self._registered:
            raise RuntimeError("Original node departure adapter is unavailable")
        if not any(worker in registration.workers for registration in self._registrations.values()):
            raise ValueError("Worker is not in the original node registration")
        completed = False
        try:
            self._poll()
            completed = True
            return self._receipts.get(worker)
        finally:
            if not completed:
                self._terminal = True

    def owns_receipt(self, receipt: ProcessDeparture) -> bool:
        return (
            not self._terminal
            and not self._closed
            and self._registered
            and all(time.monotonic() < deadline for deadline in self._deadlines.values())
            and all(
                channel.use_hmac_encryption is True and channel.hmac_key == self._keys[scope]
                for scope, channel in self._channels.items()
            )
            and isinstance(receipt, ProcessDeparture)
            and self._receipts.get(receipt.worker) is receipt
        )

    def close(self) -> None:
        import zmq

        if self._closed:
            return
        self._closed = True
        self._receipts.clear()
        errors = []
        for channel in self._channels.values():
            try:
                if channel.socket is not None:
                    channel.socket.setsockopt(zmq.LINGER, 0)
                channel.close()
            except zmq.ZMQError as error:
                errors.append(error)
        if errors:
            raise errors[0]


class LocalProcessDepartureAdapter:
    """Own original pidfds inside a qualified, isolated Linux launcher scope.

    Procfs birth, namespaces, UID, signal groups and cgroup must be readable from
    the authority. Unsupported or hidden namespaces fail registration closed.
    A host launcher adapter must qualify other layouts separately.
    """

    def __init__(
        self, task_id: int, generation: int, registration_reference: str, worker_cgroup: str
    ) -> None:
        if (
            task_id <= 0
            or generation < 0
            or not registration_reference
            or not worker_cgroup
            or worker_cgroup.strip() == "0::/"
            or not hasattr(os, "pidfd_open")
        ):
            raise ValueError(
                "Original-process departure needs pidfds and a qualified launcher scope"
            )
        with open("/proc/self/cgroup", encoding="utf-8") as source:
            if source.read() == worker_cgroup:
                raise ValueError("Authority and workers must have separate launcher scopes")
        self.adapter_id = "linux-pidfd:" + uuid.uuid4().hex
        self._task_id = task_id
        self._generation = generation
        self._reference = registration_reference
        self._worker_cgroup = worker_cgroup
        self._handles: dict[WorkerProcessIdentity, tuple[int, select.poll]] = {}
        self._receipts: dict[WorkerProcessIdentity, ProcessDeparture] = {}
        self._registered = False

    def _validate_worker(self, worker: WorkerProcessIdentity) -> None:
        state = _read_process_state(worker.pid)
        namespace = os.stat(f"/proc/{worker.pid}/ns/pid").st_ino
        current = WorkerProcessIdentity(
            worker.rank,
            worker.pid,
            state[1] if state is not None else None,
            socket.gethostname(),
            namespace,
        )
        if (
            state is None
            or state[0] == "Z"
            or worker != current
            or worker.hostname != socket.gethostname()
            or worker.start_time is None
            or worker.pid_namespace is None
            or worker.pid_namespace != _read_pid_namespace(os.getpid())
            or os.stat(f"/proc/{worker.pid}").st_uid != os.getuid()
            or os.getpgid(worker.pid) == os.getpgid(0)
            or os.getsid(worker.pid) == os.getsid(0)
            or os.stat(f"/proc/{worker.pid}/ns/user").st_ino != os.stat("/proc/self/ns/user").st_ino
        ):
            raise ValueError("Worker birth is outside the isolated local authority scope")
        with open(f"/proc/{worker.pid}/cgroup", encoding="utf-8") as source:
            if source.read() != self._worker_cgroup:
                raise ValueError("Worker moved outside the qualified launcher scope")

    def register(self, workers: tuple[WorkerProcessIdentity, ...]) -> None:
        """Register once, checking the same process birth before and after pidfd creation."""
        if self._registered or len(set(workers)) != len(workers):
            raise ValueError("Original-process registration must be one-shot with distinct births")
        self._registered = True
        try:
            for worker in workers:
                self._validate_worker(worker)
                handle = os.pidfd_open(worker.pid)
                self._handles[worker] = (handle, select.poll())
                poller = self._handles[worker][1]
                poller.register(handle, select.POLLIN | select.POLLERR | select.POLLHUP)
                self._validate_worker(worker)
                if poller.poll(0):
                    raise ValueError("Original worker departed before registration completed")
        except (OSError, ValueError):
            self.close()
            raise

    def observe_departure(self, worker: WorkerProcessIdentity) -> ProcessDeparture | None:
        """Return a retained original-handle receipt after the process has exited."""
        if worker not in self._handles:
            raise ValueError("No original pidfd is owned for this worker")
        if worker in self._receipts:
            return self._receipts[worker]
        handle, poller = self._handles[worker]
        events = poller.poll(0)
        if not events:
            return None
        departure_events = select.POLLIN | select.POLLHUP
        if any(
            type(descriptor) is not int
            or descriptor != handle
            or type(flags) is not int
            or not flags & departure_events
            or flags & ~departure_events
            for descriptor, flags in events
        ):
            raise RuntimeError("Original pidfd poll returned invalid departure evidence")
        receipt = ProcessDeparture(
            self._task_id,
            self._generation,
            worker,
            self._reference,
            self.adapter_id,
            uuid.uuid4().hex,
        )
        self._receipts[worker] = receipt
        return receipt

    def owns_receipt(self, receipt: ProcessDeparture) -> bool:
        return self._receipts.get(receipt.worker) is receipt

    def close(self) -> None:
        for handle, _ in self._handles.values():
            os.close(handle)
        self._handles.clear()
        self._receipts.clear()

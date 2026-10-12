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
"""Recovery authority: coordinates the survivors of one worker process loss.

Run as ``python -m tensorrt_llm.executor.recovery.authority --endpoint tcp://*:PORT
--world-size N`` with ``TLLM_FAULT_TOLERANCE_AUTHORITY_KEY`` set to the hex HMAC key
shared with the workers. Run it outside the workers' launch and termination groups:
it must outlive any one worker.
"""

import argparse
import enum
import logging
import signal
import threading
import time

import zmq

from .client import load_key
from .protocol import (
    Abort,
    Commit,
    Install,
    Installed,
    Message,
    Propose,
    ProtocolError,
    Ready,
    Register,
    Registered,
    Stopped,
    authenticate,
    decode,
    encode,
)

__all__ = ["Phase", "RecoveryAuthority", "main", "serve"]

logger = logging.getLogger(__name__)

Outbound = list[tuple[bytes, Message]]

# Upper bound on an incoming message; real messages are a few hundred bytes.
_MAX_MESSAGE_BYTES = 64 * 1024


class Phase(enum.Enum):
    """Progress of the deployment through registration and one recovery."""

    REGISTERING = enum.auto()
    SERVING = enum.auto()
    STOPPING = enum.auto()
    PREPARING = enum.auto()
    INSTALLING = enum.auto()
    RECOVERED = enum.auto()
    ABORTED = enum.auto()


class _Violation(Exception):
    """A message that the recovery protocol cannot accept; aborts recovery."""


class RecoveryAuthority:
    """State machine for recovering from the loss of one admitted worker.

    The authority performs no I/O: ``handle`` and ``expire`` return the messages to
    send as ``(routing_id, message)`` pairs. Any protocol violation, or a recovery that
    does not commit within ``recovery_timeout``, moves it to the terminal ``ABORTED``
    phase and broadcasts ``Abort``. Its attributes are read-only for callers.

    Args:
        world_size: Number of worker ranks that must register. The loss of a rank is
            recovered when every worker reports it recoverable; never rank 0.
        recovery_timeout: Seconds from the first ``Stopped`` to ``Commit``. Keep it
            below the workers' recovery timeout so that workers observe ``Abort``.
    """

    def __init__(
        self,
        world_size: int,
        recovery_timeout: float = 60.0,
    ) -> None:
        if world_size < 2:
            raise ValueError(f"world_size must be at least 2, got {world_size}")
        if recovery_timeout <= 0:
            raise ValueError(f"recovery_timeout must be positive, got {recovery_timeout}")
        self.world_size = world_size
        self.recovery_timeout = recovery_timeout
        self.admitted_ranks: frozenset[int] = frozenset()
        self.generation = 0
        self.phase = Phase.REGISTERING
        self.deadline: float | None = None
        self.failed_rank: int | None = None
        self.abort_reason: str | None = None
        self._workers: dict[bytes, Register] = {}
        self._routes: dict[int, bytes] = {}
        self._stopped: set[int] = set()
        self._ready: dict[int, str] = {}
        self._installed: set[int] = set()

    @property
    def survivors(self) -> tuple[int, ...]:
        return tuple(rank for rank in range(self.world_size) if rank != self.failed_rank)

    def handle(self, route: bytes, message: Message, now: float) -> Outbound:
        """Apply ``message`` received from ``route`` at monotonic time ``now``."""
        if self.phase is Phase.ABORTED:
            return [(route, Abort(self.abort_reason))]
        try:
            return self._dispatch(route, message, now)
        except _Violation as violation:
            return self.abort(str(violation), route)

    def expire(self, now: float) -> Outbound:
        """Abort if the recovery deadline has passed at monotonic time ``now``."""
        if self.deadline is not None and now >= self.deadline:
            return self.abort(f"recovery did not commit within {self.recovery_timeout}s")
        return []

    def abort(self, reason: str, route: bytes | None = None) -> Outbound:
        """Enter the terminal phase and tell every worker (and ``route``) why."""
        if self.phase is Phase.ABORTED:
            return [] if route is None else [(route, Abort(self.abort_reason))]
        logger.error(f"Recovery aborted in phase {self.phase.name}: {reason}")
        self.phase = Phase.ABORTED
        self.deadline = None
        self.abort_reason = reason
        routes = list(self._workers)
        if route is not None and route not in self._workers:
            routes.append(route)
        return [(peer, Abort(reason)) for peer in routes]

    def _dispatch(self, route: bytes, message: Message, now: float) -> Outbound:
        if isinstance(message, Register):
            return self._on_register(route, message)
        worker = self._workers.get(route)
        if worker is None:
            raise _Violation(f"{message} from unregistered peer")
        if self.phase is Phase.REGISTERING:
            raise _Violation(f"{message} before registration completed")
        if not isinstance(message, (Stopped, Ready, Installed)) or message.rank != worker.rank:
            raise _Violation(f"unexpected {message} from rank {worker.rank}")
        if message.rank == self.failed_rank:
            raise _Violation(f"departed rank sent {message}")
        if isinstance(message, Stopped):
            return self._on_stopped(message, now)
        if isinstance(message, Ready):
            return self._on_ready(message)
        return self._on_installed(message)

    def _on_register(self, route: bytes, message: Register) -> Outbound:
        if self.phase is not Phase.REGISTERING:
            raise _Violation(f"registration {message} after registration completed")
        if not 0 <= message.rank < self.world_size or message.rank in self._routes:
            raise _Violation(f"invalid or duplicate registration {message}")
        if route in self._workers:
            raise _Violation(f"{message} from the route of rank {self._workers[route].rank}")
        self._workers[route] = message
        self._routes[message.rank] = route
        if len(self._routes) < self.world_size:
            return []
        self._admit()
        self.phase = Phase.SERVING
        registered = Registered(tuple(sorted(self.admitted_ranks)))
        return [(peer, registered) for peer in self._workers]

    def _admit(self) -> None:
        recoverable = frozenset(range(1, self.world_size))
        for registration in self._workers.values():
            recoverable &= frozenset(registration.recoverable_ranks)
        self.admitted_ranks = recoverable
        if self.admitted_ranks:
            logger.info(
                f"Recovery authority: {self.world_size} workers registered; "
                f"recoverable ranks {sorted(self.admitted_ranks)}"
            )
        else:
            logger.warning(
                f"Recovery authority: {self.world_size} workers registered, but the expert "
                "placement covers the loss of no rank"
            )

    def _on_stopped(self, message: Stopped, now: float) -> Outbound:
        if self.phase is Phase.SERVING:
            if message.failed_rank not in self.admitted_ranks:
                raise _Violation(f"loss of rank {message.failed_rank} is not admitted")
            self.failed_rank = message.failed_rank
            self.deadline = now + self.recovery_timeout
            self._enter(Phase.STOPPING)
        elif self.phase is not Phase.STOPPING:
            raise _Violation(f"{message} in phase {self.phase.name}")
        if message.failed_rank != self.failed_rank:
            raise _Violation(f"{message} conflicts with departed rank {self.failed_rank}")
        self._record(self._stopped, message)
        if len(self._stopped) < len(self.survivors):
            return []
        self._enter(Phase.PREPARING)
        proposal = Propose(self.generation + 1, self.failed_rank, self.survivors)
        return self._to_survivors(proposal)

    def _on_ready(self, message: Ready) -> Outbound:
        self._expect(Phase.PREPARING, message)
        digests = set(self._ready.values())
        if digests and message.placement_digest not in digests:
            raise _Violation(f"{message} placement digest differs from {digests}")
        if message.rank in self._ready:
            raise _Violation(f"duplicate {message}")
        self._ready[message.rank] = message.placement_digest
        if len(self._ready) < len(self.survivors):
            return []
        self._enter(Phase.INSTALLING)
        return self._to_survivors(Install(self.generation + 1))

    def _on_installed(self, message: Installed) -> Outbound:
        self._expect(Phase.INSTALLING, message)
        self._record(self._installed, message)
        if len(self._installed) < len(self.survivors):
            return []
        self.generation += 1
        self.deadline = None
        self._enter(Phase.RECOVERED)
        logger.info(
            f"Recovery authority: committed generation {self.generation} "
            f"without rank {self.failed_rank}"
        )
        return self._to_survivors(Commit(self.generation))

    def _expect(self, phase: Phase, message: Ready | Installed) -> None:
        if self.phase is not phase:
            raise _Violation(f"{message} in phase {self.phase.name}")
        if message.generation != self.generation + 1:
            raise _Violation(f"{message} does not match generation {self.generation + 1}")

    @staticmethod
    def _record(reports: set[int], message: Stopped | Installed) -> None:
        if message.rank in reports:
            raise _Violation(f"duplicate {message}")
        reports.add(message.rank)

    def _to_survivors(self, message: Message) -> Outbound:
        return [(self._routes[rank], message) for rank in self.survivors]

    def _enter(self, phase: Phase) -> None:
        logger.info(f"Recovery authority: {self.phase.name} -> {phase.name}")
        self.phase = phase


def serve(
    authority: RecoveryAuthority,
    socket: zmq.Socket,
    key: bytes,
    stop: threading.Event,
    poll_interval: float = 0.1,
) -> None:
    """Drive ``authority`` from a bound ROUTER ``socket`` until ``stop`` is set.

    Frames that fail HMAC authentication are dropped; authenticated frames that do
    not decode abort recovery.
    """
    while not stop.is_set():
        outbound = authority.expire(time.monotonic())
        if socket.poll(int(poll_interval * 1000)):
            route, *frames = socket.recv_multipart()
            if not authenticate(frames, key):
                logger.debug("Recovery authority: dropped an unauthenticated frame")
                continue
            try:
                message = decode(frames[0])
            except ProtocolError as error:
                outbound += authority.abort(str(error), route)
            else:
                outbound += authority.handle(route, message, time.monotonic())
        for route, message in outbound:
            socket.send_multipart([route, *encode(message, key)])


def router_socket(context: zmq.Context, endpoint: str) -> zmq.Socket:
    """Return a ROUTER bound at ``endpoint`` that keeps worker identities across reconnects."""
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.ROUTER_HANDOVER, 1)
    socket.setsockopt(zmq.MAXMSGSIZE, _MAX_MESSAGE_BYTES)
    socket.bind(endpoint)
    return socket


def main(argv: list[str] | None = None) -> None:
    """Run the recovery authority until SIGINT or SIGTERM."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="ROUTER bind address, e.g. tcp://*:5555")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument(
        "--recovery-timeout",
        type=float,
        default=60.0,
        help="seconds from the first Stopped to Commit; keep below the workers' timeout",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    key = load_key()
    authority = RecoveryAuthority(args.world_size, args.recovery_timeout)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    context = zmq.Context()
    socket = router_socket(context, args.endpoint)
    try:
        serve(authority, socket, key, stop)
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()

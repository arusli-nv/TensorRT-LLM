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
"""Recovery authority: coordinates survivors through one idle worker loss.

Run as ``python -m tensorrt_llm.executor.recovery.authority --endpoint tcp://*:PORT
--world-size N --admitted-ranks 1,2`` with ``TLLM_WIDEEP_RECOVERY_KEY`` set to a hex
HMAC key shared with the workers and the frontend.
"""

import argparse
import enum
import logging
import os
import signal
import threading
import time

import zmq

from .protocol import (
    ROLE_FRONTEND,
    ROLE_WORKER,
    Abort,
    AdmissionClosed,
    CloseAdmission,
    Commit,
    Install,
    Installed,
    Message,
    Propose,
    ProtocolError,
    Ready,
    Register,
    ReopenAdmission,
    Stopped,
    authenticate,
    decode,
    encode,
)

__all__ = ["Phase", "RecoveryAuthority", "main", "serve"]

logger = logging.getLogger(__name__)

Outbound = list[tuple[bytes, Message]]


class Phase(enum.Enum):
    REGISTERING = enum.auto()
    SERVING = enum.auto()
    STOPPING = enum.auto()
    PREPARING = enum.auto()
    INSTALLING = enum.auto()
    RECOVERED = enum.auto()
    ABORTED = enum.auto()


class _Violation(Exception):
    """A report that the recovery protocol cannot accept; aborts recovery."""


class RecoveryAuthority:
    """Pure state machine for recovering from one admitted, idle worker loss.

    The authority performs no I/O: ``handle`` and ``expire`` return the messages to
    send as ``(routing_id, message)`` pairs. Any protocol violation or expired phase
    deadline moves it to the terminal ``ABORTED`` phase and broadcasts ``Abort``.

    Args:
        world_size: Number of worker ranks that must register.
        admitted_ranks: Ranks whose loss may be recovered. Rank 0 is never admitted.
        phase_timeout: Seconds allowed for each of the STOPPING, PREPARING, and
            INSTALLING phases.
        generation: Membership generation the workers start with.
    """

    def __init__(
        self,
        world_size: int,
        admitted_ranks: frozenset[int],
        phase_timeout: float = 60.0,
        generation: int = 0,
    ) -> None:
        if world_size < 2:
            raise ValueError(f"world_size must be at least 2, got {world_size}")
        if not admitted_ranks or not all(0 < rank < world_size for rank in admitted_ranks):
            raise ValueError(f"admitted ranks must lie in [1, {world_size}): {admitted_ranks}")
        self.world_size = world_size
        self.admitted_ranks = frozenset(admitted_ranks)
        self.phase_timeout = phase_timeout
        self.generation = generation
        self.phase = Phase.REGISTERING
        self.deadline: float | None = None
        self.failed_rank: int | None = None
        self.abort_reason: str | None = None
        self._peers: dict[bytes, Register] = {}
        self._workers: dict[int, bytes] = {}
        self._frontend: bytes | None = None
        self._stopped: dict[int, Stopped] = {}
        self._ready: dict[int, Ready] = {}
        self._installed: dict[int, Installed] = {}
        self._admission_closed = False

    @property
    def survivors(self) -> tuple[int, ...]:
        return tuple(rank for rank in range(self.world_size) if rank != self.failed_rank)

    def handle(self, route: bytes, message: Message, now: float) -> Outbound:
        """Apply ``message`` received from ``route`` at monotonic time ``now``."""
        if self.phase is Phase.ABORTED:
            return self.abort(self.abort_reason, route)
        try:
            return self._dispatch(route, message, now)
        except _Violation as violation:
            return self.abort(str(violation), route)

    def expire(self, now: float) -> Outbound:
        """Abort if the current phase deadline has passed at monotonic time ``now``."""
        if self.deadline is not None and now >= self.deadline:
            return self.abort(f"{self.phase.name} phase exceeded {self.phase_timeout}s")
        return []

    def abort(self, reason: str, route: bytes | None = None) -> Outbound:
        """Enter the terminal phase and tell every known peer (and ``route``) why."""
        if self.phase is Phase.ABORTED:
            return [] if route is None else [(route, Abort(self.abort_reason))]
        logger.error(f"Recovery aborted in phase {self.phase.name}: {reason}")
        self.phase = Phase.ABORTED
        self.deadline = None
        self.abort_reason = reason
        routes = list(self._peers)
        if route is not None and route not in self._peers:
            routes.append(route)
        return [(peer, Abort(reason)) for peer in routes]

    def _dispatch(self, route: bytes, message: Message, now: float) -> Outbound:
        if isinstance(message, Register):
            return self._on_register(route, message)
        peer = self._peers.get(route)
        if peer is None:
            raise _Violation(f"{message} from unregistered peer")
        if self.phase is Phase.REGISTERING:
            raise _Violation(f"{message} before registration completed")
        if isinstance(message, AdmissionClosed):
            if peer.role != ROLE_FRONTEND:
                raise _Violation(f"AdmissionClosed from worker rank {peer.rank}")
            return self._on_admission_closed(now)
        if not isinstance(message, (Stopped, Ready, Installed)):
            raise _Violation(f"unexpected {message} from {peer}")
        if peer.role != ROLE_WORKER or peer.rank != message.rank:
            raise _Violation(f"{message} sent by {peer}")
        if message.rank == self.failed_rank:
            raise _Violation(f"departed rank sent {message}")
        if isinstance(message, Stopped):
            return self._on_stopped(message, now)
        if isinstance(message, Ready):
            return self._on_ready(message, now)
        return self._on_installed(message)

    def _on_register(self, route: bytes, message: Register) -> Outbound:
        previous = self._peers.get(route)
        if previous == message:
            return []
        if previous is not None or self.phase is not Phase.REGISTERING:
            raise _Violation(f"unexpected registration {message}")
        if message.role == ROLE_WORKER:
            if not 0 <= message.rank < self.world_size or message.rank in self._workers:
                raise _Violation(f"invalid or duplicate worker registration {message}")
            self._workers[message.rank] = route
        elif message.role == ROLE_FRONTEND:
            if self._frontend is not None:
                raise _Violation(f"duplicate frontend registration {message}")
            self._frontend = route
        else:
            raise _Violation(f"unknown role in {message}")
        self._peers[route] = message
        if len(self._workers) == self.world_size and self._frontend is not None:
            self.phase = Phase.SERVING
            logger.info(f"Recovery authority: {self.world_size} workers and frontend registered")
        return []

    def _on_stopped(self, message: Stopped, now: float) -> Outbound:
        outbound: Outbound = []
        if self.phase is Phase.SERVING:
            failed = message.failed_rank
            if failed not in self.admitted_ranks:
                raise _Violation(f"loss of rank {failed} is not admitted for recovery")
            self.failed_rank = failed
            self._enter(Phase.STOPPING, now)
            outbound.append((self._frontend, CloseAdmission()))
        elif self.phase is Phase.RECOVERED:
            raise _Violation(f"second worker loss reported by {message}")
        if message.failed_rank != self.failed_rank:
            raise _Violation(f"{message} conflicts with departed rank {self.failed_rank}")
        if message.rank == self.failed_rank:
            raise _Violation(f"departed rank sent {message}")
        if not self._collect(self._stopped, message, Phase.STOPPING):
            return outbound
        return outbound + self._maybe_propose(now)

    def _on_admission_closed(self, now: float) -> Outbound:
        if self._admission_closed:
            return []
        if self.phase is not Phase.STOPPING:
            raise _Violation(f"AdmissionClosed in phase {self.phase.name}")
        self._admission_closed = True
        return self._maybe_propose(now)

    def _maybe_propose(self, now: float) -> Outbound:
        if not self._admission_closed or len(self._stopped) < len(self.survivors):
            return []
        self._enter(Phase.PREPARING, now)
        proposal = Propose(self.generation + 1, self.failed_rank, self.survivors)
        return [(self._workers[rank], proposal) for rank in self.survivors]

    def _on_ready(self, message: Ready, now: float) -> Outbound:
        if message.generation != self.generation + 1:
            raise _Violation(f"{message} does not match generation {self.generation + 1}")
        digests = {ready.placement_digest for ready in self._ready.values()}
        if digests and message.placement_digest not in digests:
            raise _Violation(f"{message} placement digest differs from {digests}")
        if not self._collect(self._ready, message, Phase.PREPARING):
            return []
        if len(self._ready) < len(self.survivors):
            return []
        self._enter(Phase.INSTALLING, now)
        return [(self._workers[rank], Install(self.generation + 1)) for rank in self.survivors]

    def _on_installed(self, message: Installed) -> Outbound:
        if message.generation != self.generation + 1:
            raise _Violation(f"{message} does not match generation {self.generation + 1}")
        if not self._collect(self._installed, message, Phase.INSTALLING):
            return []
        if len(self._installed) < len(self.survivors):
            return []
        self.generation += 1
        self.phase = Phase.RECOVERED
        self.deadline = None
        self._admission_closed = False
        logger.info(
            f"Recovery authority: committed generation {self.generation} "
            f"without rank {self.failed_rank}"
        )
        outbound: Outbound = [
            (self._workers[rank], Commit(self.generation)) for rank in self.survivors
        ]
        outbound.append((self._frontend, ReopenAdmission(self.generation)))
        return outbound

    def _collect(self, reports: dict, message: Stopped | Ready | Installed, phase: Phase) -> bool:
        """Record ``message``; return False for an identical retransmission."""
        previous = reports.get(message.rank)
        if previous == message:
            return False
        if previous is not None:
            raise _Violation(f"{message} conflicts with earlier {previous}")
        if self.phase is not phase:
            raise _Violation(f"{message} in phase {self.phase.name}")
        reports[message.rank] = message
        return True

    def _enter(self, phase: Phase, now: float) -> None:
        logger.info(f"Recovery authority: {self.phase.name} -> {phase.name}")
        self.phase = phase
        self.deadline = now + self.phase_timeout


def serve(
    authority: RecoveryAuthority,
    endpoint: str,
    key: bytes,
    stop: threading.Event,
    poll_interval: float = 0.1,
) -> None:
    """Drive ``authority`` from a ROUTER socket bound at ``endpoint`` until ``stop`` is set.

    Frames that fail HMAC authentication are dropped; authenticated frames that do not
    decode abort recovery.
    """
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(endpoint)
    try:
        while not stop.is_set():
            outbound = authority.expire(time.monotonic())
            if socket.poll(int(poll_interval * 1000)):
                route, *frames = socket.recv_multipart()
                if not authenticate(frames, key):
                    logger.warning("Recovery authority: dropped unauthenticated frame")
                    continue
                try:
                    message = decode(frames[0])
                except ProtocolError as error:
                    outbound += authority.abort(str(error), route)
                else:
                    outbound += authority.handle(route, message, time.monotonic())
            for route, message in outbound:
                socket.send_multipart([route, *encode(message, key)])
    finally:
        socket.close()
        context.term()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="ROUTER bind address, e.g. tcp://*:5555")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument(
        "--admitted-ranks", required=True, help="comma-separated ranks whose loss may be recovered"
    )
    parser.add_argument("--phase-timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    key = bytes.fromhex(os.environ["TLLM_WIDEEP_RECOVERY_KEY"])
    admitted = frozenset(int(rank) for rank in args.admitted_ranks.split(","))
    authority = RecoveryAuthority(args.world_size, admitted, args.phase_timeout)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    serve(authority, args.endpoint, key, stop)


if __name__ == "__main__":
    main()

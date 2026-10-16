# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recovery authority: coordinates the survivors of one worker process loss.

Run as ``python -m tensorrt_llm.executor.recovery.authority --world-size N`` with
``TLLM_FAULT_TOLERANCE_AUTHORITY_KEY`` set to the hex HMAC key shared with the
workers. Run it outside the workers' launch and termination groups: it must outlive
any one worker.

Experimental: traffic is authenticated but not encrypted, and a host that can reach
the port can still disrupt recovery, so the authority binds to localhost by default.
A multi-node job binds it to the job's private network interface with
``--endpoint`` and a key generated for that job.
"""

import argparse
import enum
import os
import signal
import threading
import time

import zmq

from ...logger import logger
from .client import load_key
from .protocol import (
    MAX_MESSAGE_BYTES,
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

__all__ = ["Phase", "RecoveryAuthority", "main", "router_socket", "serve"]

Outbound = list[tuple[bytes, Message]]

_CLOSE_LINGER_MS = 1000
_REJECTION_LOG_INTERVAL_S = 10.0


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
        world_size: Number of worker ranks that must register. Every worker must
            report the same recoverable ranks; the loss of rank 0 is never recovered.
        recovery_timeout: Seconds from the first ``Stopped`` to ``Commit``. Keep it
            below the workers' recovery timeout so that workers observe ``Abort``.
    """

    def __init__(
        self,
        world_size: int,
        recovery_timeout: float = 20.0,
    ) -> None:
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
        """Every rank except the departed one, in rank order."""
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
            logger.info(
                f"Recovery authority: rank {message.rank} registered "
                f"({len(self._routes)}/{self.world_size})"
            )
            return []
        self._admit()
        self.phase = Phase.SERVING
        registered = Registered(tuple(sorted(self.admitted_ranks)))
        return [(peer, registered) for peer in self._workers]

    def _admit(self) -> None:
        reported = {frozenset(r.recoverable_ranks) for r in self._workers.values()}
        if len(reported) > 1:
            raise _Violation(f"workers report different recoverable ranks: {reported}")
        self.admitted_ranks = reported.pop() & frozenset(range(1, self.world_size))
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
        if message.failed_rank == message.rank:
            raise _Violation(f"rank {message.rank} reported its own loss")
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
    not decode abort recovery, and so does stopping during a recovery.

    Args:
        authority: State machine to drive.
        socket: ROUTER socket from :func:`router_socket`.
        key: HMAC key shared with the workers.
        stop: Event that ends the loop.
        poll_interval: Seconds between deadline checks while no frame arrives.
    """
    last_rejection = -_REJECTION_LOG_INTERVAL_S
    while not stop.is_set():
        outbound = authority.expire(time.monotonic())
        if socket.poll(int(poll_interval * 1000)):
            route, *frames = socket.recv_multipart()
            if not authenticate(frames, key, route):
                if time.monotonic() - last_rejection >= _REJECTION_LOG_INTERVAL_S:
                    last_rejection = time.monotonic()
                    logger.warning(
                        f"Recovery authority: dropped a frame from {route!r} that failed "
                        "authentication; is its key the authority's?"
                    )
            else:
                try:
                    message = decode(frames[0])
                except ProtocolError as error:
                    outbound += authority.abort(str(error), route)
                else:
                    outbound += authority.handle(route, message, time.monotonic())
        for route, message in outbound:
            socket.send_multipart([route, *encode(message, key, route)])
    # A recovery in progress must not leave the survivors waiting out their deadline.
    if authority.phase in (Phase.STOPPING, Phase.PREPARING, Phase.INSTALLING):
        for route, message in authority.abort("the recovery authority stopped"):
            socket.send_multipart([route, *encode(message, key, route)])


def router_socket(context: zmq.Context, endpoint: str) -> zmq.Socket:
    """Return a ROUTER bound at ``endpoint``; a second connection with a routing id in use is refused."""
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.MAXMSGSIZE, MAX_MESSAGE_BYTES)
    socket.bind(endpoint)
    if "://*:" in endpoint or "0.0.0.0" in endpoint:
        logger.warning(
            f"Recovery authority listens on every interface ({endpoint}): any host that reaches this one can "
            "connect; bind the job's private interface and use a per-job key"
        )
    return socket


def main(argv: list[str] | None = None) -> None:
    """Run the recovery authority until SIGINT or SIGTERM."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default="tcp://127.0.0.1:5555",
        help="ROUTER bind address; bind a private interface, never a public one",
    )
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument(
        "--recovery-timeout",
        type=float,
        default=20.0,
        help="seconds from the first Stopped to Commit; keep below the workers' timeout",
    )
    args = parser.parse_args(argv)
    if "TLLM_LOG_LEVEL" not in os.environ:
        logger.set_level("info")

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
        # Let a final Abort reach the workers.
        socket.close(linger=_CLOSE_LINGER_MS)
        context.term()


if __name__ == "__main__":
    main()

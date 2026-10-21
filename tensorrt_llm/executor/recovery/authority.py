# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recovery authority: coordinates the survivors of a lost failure domain (a rank or a node).

Run as ``python -m tensorrt_llm.executor.recovery.authority --world-size N`` with
``TLLM_FAULT_TOLERANCE_AUTHORITY_KEY`` set to the hex HMAC key shared with the
workers. Run it outside the workers' launch and termination groups: it must outlive
any one worker.

Experimental: traffic is authenticated but not encrypted, and a host that can reach
the port can still disrupt recovery (for example, by taking over a worker's route; see
:func:`router_queue`), so the authority binds to localhost by default.
A multi-node job binds it to the job's private network interface with
``--endpoint`` and a key generated for that job.
"""

import argparse
import enum
import ipaddress
import os
import re
import shlex
import signal
import socket
import subprocess
import threading
import time
from collections.abc import Callable

import zmq

from ...logger import logger
from ..ipc import ZeroMqQueue
from .client import CONNECTION_OPTIONS, load_key, unauthenticated
from .config import COMMAND_FENCE_GRACE_S, CONFIG_ENV, FaultToleranceConfig, fault_tolerance_config
from .hang_monitor import HangMonitor, HungRank
from .protocol import (
    MAX_MESSAGE_BYTES,
    Abort,
    Commit,
    DeviceWait,
    Fence,
    Fenced,
    Heartbeat,
    Install,
    Installed,
    Message,
    NativeHeartbeat,
    Propose,
    Ready,
    Register,
    Registered,
    Restart,
    Sent,
    Status,
    StatusQuery,
    Stopped,
    datagram_address,
    decode_datagram,
    is_message,
)

__all__ = ["Phase", "RecoveryAuthority", "datagram_socket", "main", "router_queue", "serve"]

Outbound = list[tuple[bytes, Message]]

_CLOSE_LINGER_MS = 1000
_REJECTION_LOG_INTERVAL_S = 10.0
# Bounds on what unauthenticated peers can queue: datagrams handled per pass, and messages queued per connection.
# Connections are not bounded and the loop takes one ROUTER message per pass, so N flooding connections still cut
# the workers' share of the loop to about 1/(N+1).
_MAX_DATAGRAMS_PER_PASS = 256
_ROUTER_RCVHWM = 1024
_HOSTNAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?")


class Phase(enum.Enum):
    """Progress of the deployment through registration and its recoveries."""

    REGISTERING = enum.auto()
    SERVING = enum.auto()
    STOPPING = enum.auto()
    PREPARING = enum.auto()
    INSTALLING = enum.auto()
    RECOVERED = enum.auto()
    ABORTED = enum.auto()


def _sorted_domains(domains) -> tuple[tuple[int, ...], ...]:
    """``domains`` (rank sets) as sorted tuples, in order: the wire form of a set of domains."""
    return tuple(sorted(tuple(sorted(domain)) for domain in domains))


class _Violation(Exception):
    """A message that the recovery protocol cannot accept; aborts recovery."""


class RecoveryAuthority:
    """State machine for recovering from the loss of admitted failure domains, one recovery after another.

    A failure domain is a set of worker ranks lost together: a single rank or a whole
    node. A loss recovers only if every survivor reports the same failed set (every rank
    lost so far) and that set is a union of admitted domains; the survivors check that
    their slots still serve every expert without it. The survivors' agreement on the failed
    set may rest on a failure detector's suspicion (a stopped process is alive), so before
    proposing, the authority has rank 0 end through the MPI runtime every failed rank it has
    not fenced itself, and aborts if one cannot be ended.

    The authority performs no I/O: ``handle`` and ``expire`` return the messages to
    send as ``(routing_id, message)`` pairs. A protocol violation during registration or
    a recovery, an unrecoverable loss, or a recovery that does not commit within
    ``recovery_timeout`` moves it to the terminal ``ABORTED`` phase and broadcasts
    ``Abort``; a violation while serving, or an object that is not a message, is refused
    to its sender alone. A message that no registered member can have sent on its route
    (an unregistered route, or another rank than the route's) is dropped, and so is a
    worker's message that :meth:`receive` finds stale (a replay). Its attributes are
    read-only for callers.

    Args:
        world_size: Number of worker ranks that must register. Every worker must
            report the same recoverable domains; a domain with rank 0 is never admitted.
        recovery_timeout: Seconds from the first ``Stopped`` to ``Commit``. Keep it
            below the workers' recovery timeout so that workers observe ``Abort``.
        hang_monitor: Judges the workers' heartbeats and device waits; None ignores them. A stall of every
            rank it reports, or a hung rank whose loss is not admitted (rank 0), aborts (terminal) and has
            every worker end with ``STALLED_EXIT_CODE``.
        fence: Last resort for a recoverable rank the monitor confirms hung (rank, reason,
            host and pid): called if the rank's loss has not started a recovery
            ``COMMAND_FENCE_GRACE_S`` after the authority sent ``Fence`` to the rank's own
            evidence thread and to another rank's (which has the MPI runtime end it). It
            must end that process, whose loss then recovers as any other. None relies on
            the workers alone.
        config_digest: The ``FaultToleranceConfig.digest()`` every worker must register with
            (default: that of the default settings).
    """

    def __init__(
        self,
        world_size: int,
        recovery_timeout: float = 20.0,
        hang_monitor: HangMonitor | None = None,
        fence: Callable[[HungRank], None] | None = None,
        config_digest: str | None = None,
    ) -> None:
        self.world_size = world_size
        self.config_digest = config_digest or FaultToleranceConfig().digest()
        self.recovery_timeout = recovery_timeout
        self.hang_monitor = hang_monitor
        self._fence = fence
        self.admitted_domains: frozenset[frozenset[int]] = frozenset()
        self.generation = 0
        self.phase = Phase.REGISTERING
        self.deadline: float | None = None
        self.failed_ranks: frozenset[int] = frozenset()
        self.abort_reason: str | None = None
        self._workers: dict[bytes, Register] = {}
        self._routes: dict[int, bytes] = {}
        self._stopped: set[int] = set()
        # Times the current recovery started over because more ranks failed during it, and those ranks.
        self._replans = 0
        self._failed_during_recovery: frozenset[int] = frozenset()
        # The failed ranks of each plan proposed in the current recovery, in order.
        self._proposed: list[tuple[int, ...]] = []
        # Per plan proposed before the last: the first and last chunk its lost ranks' copies may still write.
        self._in_flight: list[tuple[int, int]] = []
        # Rank -> ring-restore staging barriers (entered, passed) it reported since the last Propose.
        self._barriers: dict[int, tuple[int, int]] = {}
        self._ready: dict[int, str] = {}
        self._installed: set[int] = set()
        # Rank -> (route of its evidence connection, host, pid), from its heartbeats.
        self._evidence: dict[int, tuple[bytes, str, int]] = {}
        # Rank -> the last native heartbeat accepted from its process.
        self._beats: dict[int, int] = {}
        # The fence command's deadline and target, while the workers' fencing is under way.
        self._command_fence: tuple[float, HungRank] | None = None
        # Failed ranks known to be ended (a fence that succeeded), and those rank 0 is fencing before Propose.
        self._fenced: set[int] = set()
        self._fencing: set[int] = set()
        # Monotonic times at which the current recovery entered STOPPING, PREPARING and INSTALLING.
        self._stage_starts: list[float] = []
        self._recovery_stage_us: tuple[int, ...] = ()
        self._failed_requests = 0
        self._last_drop_log = -_REJECTION_LOG_INTERVAL_S
        # Rank -> its process's session, from its Register; kept after the rank departs.
        self._sessions: dict[int, str] = {}
        # (rank, evidence?) -> the sequence number of the last message accepted on that stream. The
        # protocol and the evidence travel on separate connections, ordered only within each.
        self._sequences: dict[tuple[int, bool], int] = {}

    @property
    def survivors(self) -> tuple[int, ...]:
        """Every rank except the departed ones, in rank order."""
        return tuple(rank for rank in range(self.world_size) if rank not in self.failed_ranks)

    def handle(self, route: bytes, message: Message, now: float) -> Outbound:
        """Apply ``message`` received from ``route`` at monotonic time ``now``."""
        if isinstance(message, StatusQuery):
            return [(route, self.status())]
        if isinstance(message, (Heartbeat, NativeHeartbeat, DeviceWait)):
            # Evidence, not protocol: a worker sends it from its own connection or its native thread, at any time.
            if isinstance(message, Heartbeat):
                self._evidence[message.rank] = (route, message.host, message.pid)
            elif isinstance(message, NativeHeartbeat) and not self._accept(message):
                return []
            self._observe(message, now)
            return []
        if reason := self._foreign(route, message):
            return self._drop(route, message, reason, now)
        if self.phase is Phase.ABORTED:
            return [(route, Abort(self.abort_reason))]
        try:
            return self._dispatch(route, message, now)
        except _Violation as violation:
            if self.phase in (Phase.SERVING, Phase.RECOVERED) and not isinstance(message, Stopped):
                # No loss was reported: refuse the sender alone, so a stray or malformed message does not end
                # a deployment that serves. A Stopped is a real loss, and an unrecoverable one is terminal.
                logger.error(f"Recovery authority: refusing {message} from {route!r}: {violation}")
                return [(route, Abort(str(violation)))]
            # The message authenticated, so a worker (or a holder of its key) broke the protocol: the membership
            # or the plan is not one every survivor agrees on. Aborting stops every survivor now, before any
            # installs a plan, rather than at the deadline.
            return self.abort(str(violation), route)

    def receive(self, route: bytes, sent: Sent, now: float) -> Outbound:
        """Apply a worker's message received from ``route`` if it is fresh; drop it otherwise.

        Fresh: it names the rank of the envelope, comes from the session that rank registered,
        and its sequence number is above the last accepted from that rank's protocol messages
        or evidence (``Heartbeat``, ``DeviceWait``), whichever it is. A ``Register`` from
        another session is applied (a second process claiming a rank aborts as a duplicate);
        its session becomes the rank's once the registration is accepted.
        """
        message = sent.message
        session = self._sessions.get(sent.rank)
        stream = (sent.rank, isinstance(message, (Heartbeat, DeviceWait)))
        registers = isinstance(message, Register) and sent.session != session
        if getattr(message, "rank", sent.rank) != sent.rank:
            return self._drop(route, message, f"sent as rank {sent.rank}", now)
        if not registers and (
            sent.session != session or sent.seq <= self._sequences.get(stream, 0)
        ):
            return self._drop(route, message, "a stale or foreign message (a replay?)", now)
        outbound = self.handle(route, message, now)
        if registers:
            if self._workers.get(route) is not message:
                return outbound
            self._sessions[sent.rank] = sent.session
        self._sequences[stream] = sent.seq
        return outbound

    def _drop(self, route: bytes, message: Message, reason: str, now: float) -> Outbound:
        # Rate-limited: a peer that floods replays must not flood the log.
        if now - self._last_drop_log >= _REJECTION_LOG_INTERVAL_S:
            self._last_drop_log = now
            logger.warning(f"Recovery authority: dropped {message} from {route!r}: {reason}")
        return []

    def _foreign(self, route: bytes, message: Message) -> str | None:
        """Why no registered member can have sent ``message`` on ``route``, or None if one can."""
        if isinstance(message, Register) and self.phase is Phase.REGISTERING:
            return None
        worker = self._workers.get(route)
        if worker is None:
            return "unregistered route"
        rank = getattr(message, "rank", worker.rank)
        return None if rank == worker.rank else f"the route of rank {worker.rank}"

    def status(self) -> Status:
        """The recovery state, for observers; reading it changes nothing."""
        return Status(
            phase=self.phase.name,
            generation=self.generation,
            live_ranks=self.survivors,
            failed_ranks=tuple(sorted(self.failed_ranks)),
            recoveries=self.generation,
            recovery_stage_us=self._recovery_stage_us,
            failed_requests=self._failed_requests,
            abort_reason=self.abort_reason or "",
        )

    def expire(self, now: float) -> Outbound:
        """Abort if the recovery deadline passed at monotonic time ``now``; act on a hung rank."""
        if self.deadline is not None and now >= self.deadline:
            return self.abort(f"recovery did not commit within {self.recovery_timeout}s")
        if self._command_fence is not None and now >= self._command_fence[0]:
            hung = self._command_fence[1]
            self._command_fence = None
            # A recovery without the rank means the workers' fencing worked.
            if hung.rank not in self.failed_ranks and self.phase in (
                Phase.SERVING,
                Phase.RECOVERED,
            ):
                logger.error(
                    f"Recovery authority: rank {hung.rank} survived its fence; running the fence command"
                )
                # The command may fail unseen: rank 0 still fences the rank before a recovery proposes.
                self._fence(hung)
        if self.hang_monitor is not None and self.phase in (Phase.SERVING, Phase.RECOVERED):
            hung = self.hang_monitor.check(now)
            if hung is not None:
                return self._on_hung(hung, now)
            stalled = self.hang_monitor.all_stalled(now)
            if stalled is not None:
                return self._restart(f"every rank stalled ({stalled})")
        return []

    def _accept(self, message: NativeHeartbeat) -> bool:
        """Whether a native heartbeat is new and from the process its rank's evidence connection announced.

        A replayed datagram repeats an old ``beat``; one from another process has another ``pid``.
        """
        known = self._evidence.get(message.rank)
        if (
            known is None
            or known[2] != message.pid
            or message.beat <= self._beats.get(message.rank, 0)
        ):
            return False
        self._beats[message.rank] = message.beat
        return True

    def _observe(self, message: Heartbeat | NativeHeartbeat | DeviceWait, now: float) -> None:
        # During a recovery every rank waits by design; evidence counts only while serving.
        if self.hang_monitor is None or self.phase not in (Phase.SERVING, Phase.RECOVERED):
            return
        if isinstance(message, Heartbeat):
            self.hang_monitor.heartbeat(
                message.rank, message.iteration, message.waiting, now, message.host, message.pid
            )
        elif isinstance(message, NativeHeartbeat):
            self.hang_monitor.heartbeat(message.rank, message.iteration, None, now)
        else:
            self.hang_monitor.device_wait(
                message.rank, message.peer, message.waited_ms / 1000.0, now
            )

    def _on_hung(self, hung: HungRank, now: float) -> Outbound:
        """Fence a recoverable hung rank; end the deployment for one whose loss is not admitted.

        Its own evidence thread ends its process; another rank has the MPI runtime end it
        if that thread cannot (the process is stopped); the fence command if neither
        starts a recovery.
        """
        where = f"rank {hung.rank} (pid {hung.pid} on {hung.host or 'unknown host'})"
        # The monitor watches no departed rank, and rank 0's loss is never admitted.
        if frozenset({hung.rank}) not in self.admitted_domains:
            return self._restart(f"{where} is hung ({hung.reason}) and its loss is not recoverable")
        if not hung.pid:
            logger.error(f"Recovery authority: {where} is hung ({hung.reason}); not fencing it")
            return []
        logger.error(f"Recovery authority: {where} is hung ({hung.reason}); fencing it")
        fence = Fence(rank=hung.rank, pid=hung.pid)
        # Any live rank can ask the runtime: the lowest other one.
        # Two live, unsuspected ranks ask the runtime, in case one of their evidence threads cannot.
        targets = [hung.rank] + self.hang_monitor.live_helpers(hung.rank, now)
        if self._fence is not None:
            self._command_fence = (now + COMMAND_FENCE_GRACE_S, hung)
        return [(self._evidence[rank][0], fence) for rank in targets if rank in self._evidence]

    def _restart(self, why: str) -> Outbound:
        """End the deployment: terminal, and every worker exits for a restart."""
        reason = f"{why}; the deployment must restart"
        outbound = self.abort(reason)
        return outbound + [(route, Restart(reason)) for route, _, _ in self._evidence.values()]

    def reject(self, reason: str, route: bytes) -> Outbound:
        """Refuse an object that is not a protocol message to its sender alone, in every phase.

        It names no rank and no protocol step, so it changes no state; a recovery it holds
        up ends at the deadline.
        """
        return [(route, Abort(reason))]

    def abort(self, reason: str, route: bytes | None = None) -> Outbound:
        """Enter the terminal phase and tell every worker (and ``route``) why."""
        if self.phase is Phase.ABORTED:
            return [] if route is None else [(route, Abort(self.abort_reason))]
        logger.error(f"Recovery aborted in phase {self.phase.name}: {reason}")
        self._enter(Phase.ABORTED)
        self.deadline = None
        self.abort_reason = reason
        routes = list(self._workers)
        if route is not None and route not in self._workers:
            routes.append(route)
        return [(peer, Abort(reason)) for peer in routes]

    def _dispatch(self, route: bytes, message: Message, now: float) -> Outbound:
        if isinstance(message, Register):
            return self._on_register(route, message)
        worker = self._workers[route]
        if self.phase is Phase.REGISTERING:
            raise _Violation(f"{message} before registration completed")
        if not isinstance(message, (Stopped, Ready, Installed, Fenced)):
            raise _Violation(f"unexpected {message} from rank {worker.rank}")
        if message.rank in self._failed_during_recovery and isinstance(message, (Stopped, Ready)):
            # Sent before the rank failed; it was already queued behind the survivors' reports.
            return []
        if message.rank in self.failed_ranks:
            raise _Violation(f"departed rank sent {message}")
        if isinstance(message, Stopped):
            return self._on_stopped(message, now)
        if isinstance(message, Ready):
            return self._on_ready(message, now)
        if isinstance(message, Fenced):
            return self._on_fenced(message, now)
        return self._on_installed(message, now)

    def _on_register(self, route: bytes, message: Register) -> Outbound:
        if self.phase is not Phase.REGISTERING:
            raise _Violation(f"registration {message} after registration completed")
        if not 0 <= message.rank < self.world_size or message.rank in self._routes:
            raise _Violation(f"invalid or duplicate registration {message}")
        if route in self._workers:
            raise _Violation(f"{message} from the route of rank {self._workers[route].rank}")
        if message.config_digest != self.config_digest:
            raise _Violation(
                f"rank {message.rank} has fault-tolerance settings {message.config_digest}, the authority "
                f"{self.config_digest}: set the same {CONFIG_ENV} on the workers and the authority"
            )
        self._workers[route] = message
        self._routes[message.rank] = route
        if len(self._routes) < self.world_size:
            logger.info(
                f"Recovery authority: rank {message.rank} registered "
                f"({len(self._routes)}/{self.world_size})"
            )
            return []
        self._admit()
        self._enter(Phase.SERVING)
        registered = Registered(_sorted_domains(self.admitted_domains))
        return [(peer, registered) for peer in self._workers]

    def _admit(self) -> None:
        workers = self._workers.values()
        reported = {frozenset(map(frozenset, worker.recoverable_domains)) for worker in workers}
        if len(reported) > 1:
            raise _Violation(f"workers report different recoverable domains: {reported}")
        recoverable = frozenset(range(1, self.world_size))
        self.admitted_domains = frozenset(
            domain for domain in reported.pop() if domain and domain <= recoverable
        )
        if self.admitted_domains:
            logger.info(
                f"Recovery authority: {self.world_size} workers registered; "
                f"recoverable domains {list(_sorted_domains(self.admitted_domains))}"
            )
        else:
            logger.warning(
                f"Recovery authority: {self.world_size} workers registered, but the expert "
                "placement covers the loss of no rank or node"
            )

    def _on_stopped(self, message: Stopped, now: float) -> Outbound:
        failed = frozenset(message.failed_ranks)
        if message.rank in failed:
            raise _Violation(f"rank {message.rank} reported its own loss")
        if self.phase is Phase.SERVING or (
            self.phase is Phase.RECOVERED and failed > self.failed_ranks
        ):
            # A further loss after a recovery reports every rank lost so far.
            self._admissible(failed)
            if self.hang_monitor is not None:
                for rank in failed - self.failed_ranks:
                    self.hang_monitor.remove(rank)
            self.failed_ranks = failed
            self.deadline = now + self.recovery_timeout
            self._stage_starts = [now]
            self._replans = 0
            self._proposed.clear()
            self._in_flight.clear()
            self._barriers.clear()
            self._stopped.clear()
            self._ready.clear()
            self._installed.clear()
            self._enter(Phase.STOPPING)
        elif self.phase in (Phase.STOPPING, Phase.PREPARING) and failed > self.failed_ranks:
            self._replan(failed)
        elif self.phase is Phase.STOPPING and self._replans and failed < self.failed_ranks:
            # Sent before its rank saw the further loss; the rank's next Stopped follows (and counts no barrier
            # this one counted).
            self._record_barriers(message)
            return []
        elif self.phase is not Phase.STOPPING:
            raise _Violation(f"{message} in phase {self.phase.name}")
        if failed != self.failed_ranks:
            raise _Violation(f"{message} conflicts with departed ranks {sorted(self.failed_ranks)}")
        self._record(self._stopped, message)
        self._record_barriers(message)
        if len(self._stopped) < len(self.survivors):
            return []
        self._fencing = set(self.failed_ranks - self._fenced)
        if self._fencing:
            # Rank 0 is never lost. A pid only names the process in the log; the runtime ends the rank.
            return [
                (self._routes[0], Fence(rank=rank, pid=self._evidence.get(rank, (b"", "", 0))[2]))
                for rank in sorted(self._fencing)
            ]
        return self._propose(now)

    def _on_fenced(self, message: Fenced, now: float) -> Outbound:
        """Propose once rank 0 ended every failed rank not known to be dead; abort if one could not be ended."""
        if message.rank != 0 or message.target not in self.failed_ranks:
            raise _Violation(f"unexpected {message}: rank 0 fences failed ranks only")
        if message.error:
            # Its process may still run and write into the survivors' memory: recovering is not safe.
            return self.abort(
                f"departed rank {message.target} could not be fenced: {message.error}"
            )
        logger.info(
            f"Recovery authority: departed rank {message.target} fenced "
            f"{(now - self._stage_starts[0]) * 1e3:.1f} ms into the recovery"
        )
        self._fenced.add(message.target)
        if message.target not in self._fencing:
            # A second answer: a further loss restarted the round after the first request.
            return []
        self._fencing.discard(message.target)
        if self._fencing or len(self._stopped) < len(self.survivors):
            return []
        return self._propose(now)

    def _propose(self, now: float) -> Outbound:
        self._stage_starts.append(now)
        self._enter(Phase.PREPARING)
        failed = tuple(sorted(self.failed_ranks))
        if self._proposed:
            # A lost rank copied a chunk only after every rank entered the barrier before it, and finished the
            # copies of a chunk before entering its barrier.
            entered, passed = zip(*(self._barriers.get(rank, (0, 0)) for rank in self.survivors))
            self._in_flight.append((max(passed), min(entered)))
        self._barriers.clear()
        proposal = Propose(
            self.generation + 1,
            failed,
            self.survivors,
            tuple(self._proposed),
            tuple(self._in_flight),
        )
        self._proposed.append(failed)
        return self._to_survivors(proposal)

    def _record_barriers(self, message: Stopped) -> None:
        entered, passed = self._barriers.get(message.rank, (0, 0))
        self._barriers[message.rank] = (
            entered + message.barriers_entered,
            passed + message.barriers_passed,
        )

    def _admissible(self, failed: frozenset[int]) -> None:
        """Raise unless ``failed`` is a union of admitted domains; the survivors check that their slots cover it."""
        if not failed or not failed <= frozenset().union(*self.admitted_domains):
            raise _Violation(f"loss of ranks {sorted(failed)} is not a union of admitted domains")

    def _replan(self, failed: frozenset[int]) -> None:
        """Start the recovery over without ``failed``: more ranks failed before any survivor installed.

        The survivors that report it shrank again and checked that their slots cover the larger loss.
        """
        self._admissible(failed)
        logger.warning(
            f"Recovery authority: ranks {sorted(failed - self.failed_ranks)} failed during the "
            f"recovery; planning again without ranks {sorted(failed)}"
        )
        if self.hang_monitor is not None:
            for rank in failed - self.failed_ranks:
                self.hang_monitor.remove(rank)
        self._failed_during_recovery |= failed - self.failed_ranks
        self.failed_ranks = failed
        self._replans += 1
        self._stopped.clear()
        self._fencing.clear()
        self._ready.clear()
        self._stage_starts = self._stage_starts[:1]
        if self.phase is not Phase.STOPPING:
            self._enter(Phase.STOPPING)

    def _on_ready(self, message: Ready, now: float) -> Outbound:
        if self.phase is Phase.STOPPING and self._replans and message.rank not in self._stopped:
            # Sent before its rank saw the further loss; its Stopped for the new plan follows.
            return []
        self._expect(Phase.PREPARING, message)
        digests = set(self._ready.values())
        if digests and message.placement_digest not in digests:
            raise _Violation(f"{message} placement digest differs from {digests}")
        if message.rank in self._ready:
            raise _Violation(f"duplicate {message}")
        self._ready[message.rank] = message.placement_digest
        if len(self._ready) < len(self.survivors):
            return []
        self._stage_starts.append(now)
        self._enter(Phase.INSTALLING)
        return self._to_survivors(Install(self.generation + 1))

    def _on_installed(self, message: Installed, now: float) -> Outbound:
        self._expect(Phase.INSTALLING, message)
        self._record(self._installed, message)
        self._failed_requests += message.failed_requests
        if len(self._installed) < len(self.survivors):
            return []
        self.generation += 1
        self.deadline = None
        ends = [*self._stage_starts[1:], now]
        self._recovery_stage_us = tuple(
            round((end - start) * 1e6) for start, end in zip(self._stage_starts, ends)
        )
        self._enter(Phase.RECOVERED)
        if self.hang_monitor is not None:
            # Evidence from before the recovery describes waits that no longer exist.
            self.hang_monitor.reset()
        logger.info(
            f"Recovery authority: committed generation {self.generation} "
            f"without ranks {sorted(self.failed_ranks)}; stage ms "
            + ", ".join(
                f"{stage}={us / 1e3:.1f}"
                for stage, us in zip(("stop", "agree", "install"), self._recovery_stage_us)
            )
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
        # A recovery in progress, or aborted, is numbered by the generation it installs.
        recovering = (Phase.STOPPING, Phase.PREPARING, Phase.INSTALLING)
        in_recovery = phase in recovering or (phase is Phase.ABORTED and self.phase in recovering)
        generation = self.generation + 1 if in_recovery else self.generation
        logger.info(
            f"Recovery authority: {self.phase.name} -> {phase.name} gen={generation} t={time.time():.3f}"
        )
        self.phase = phase


def serve(
    authority: RecoveryAuthority,
    router: ZeroMqQueue,
    key: bytes,
    stop: threading.Event,
    poll_interval: float = 0.1,
    datagrams: socket.socket | None = None,
) -> None:
    """Drive ``authority`` from a ROUTER ``router`` until ``stop`` is set.

    Messages that fail HMAC authentication are dropped and authenticated objects that are
    not protocol messages are refused to their sender (see :meth:`RecoveryAuthority.reject`);
    neither changes the authority's state. Stopping during a recovery aborts it.

    Args:
        authority: State machine to drive.
        router: ROUTER queue from :func:`router_queue`.
        key: HMAC key shared with the workers.
        stop: Event that ends the loop.
        poll_interval: Seconds between deadline checks while no message arrives.
        datagrams: UDP socket from :func:`datagram_socket` receiving the workers' native
            heartbeats; datagrams that do not authenticate are dropped.
    """
    last_rejection = last_unsent = -_REJECTION_LOG_INTERVAL_S
    poller = zmq.Poller()
    poller.register(router.socket, zmq.POLLIN)
    if datagrams is not None:
        poller.register(datagrams.fileno(), zmq.POLLIN)
    while not stop.is_set():
        outbound = authority.expire(time.monotonic())
        ready = dict(poller.poll(int(poll_interval * 1000)))
        if datagrams is not None and datagrams.fileno() in ready:
            outbound += _receive_datagrams(authority, datagrams, key, time.monotonic())
        if router.socket in ready:
            rejection = None
            try:
                message = router.get()
            except (RuntimeError, ValueError) as error:
                if not unauthenticated(error):
                    raise
                rejection = f"dropped a message that failed authentication ({error}); is its key the authority's?"
            else:
                # ZeroMqQueue.get() keeps the sender's routing id for replies.
                route = router._last_identity
                if type(message) is Sent and is_message(message.message):
                    outbound += authority.receive(route, message, time.monotonic())
                elif type(message) is StatusQuery:
                    outbound += authority.handle(route, message, time.monotonic())
                else:
                    rejection = (
                        f"refused a {type(message).__name__} from {route!r}: not a recovery message"
                    )
                    outbound += authority.reject(rejection, route)
            # Rate-limited: a peer that floods bad messages must not flood the log.
            if rejection and time.monotonic() - last_rejection >= _REJECTION_LOG_INTERVAL_S:
                last_rejection = time.monotonic()
                logger.warning(f"Recovery authority: {rejection}")
        last_unsent = _send(router, outbound, last_unsent)
    # A recovery in progress must not leave the survivors waiting out their deadline.
    if authority.phase in (Phase.STOPPING, Phase.PREPARING, Phase.INSTALLING):
        _send(router, authority.abort("the recovery authority stopped"), last_unsent)


def _send(router: ZeroMqQueue, outbound: Outbound, last_log: float) -> float:
    """Send ``outbound`` without blocking; log (rate-limited from ``last_log``) what cannot be sent.

    A message cannot be sent to a route with no connection, or to a peer that does not read it.
    Returns the monotonic time of the last log.
    """
    for route, message in outbound:
        try:
            router.put_nowait(message, routing_id=route)
        except zmq.ZMQError as error:
            if time.monotonic() - last_log >= _REJECTION_LOG_INTERVAL_S:
                last_log = time.monotonic()
                logger.warning(
                    f"Recovery authority: could not send {message} to {route!r}: {error}"
                )
    return last_log


def _receive_datagrams(
    authority: RecoveryAuthority, datagrams: socket.socket, key: bytes, now: float
) -> Outbound:
    """Apply up to ``_MAX_DATAGRAMS_PER_PASS`` native heartbeats waiting on ``datagrams`` at time ``now``."""
    outbound = []
    for _ in range(_MAX_DATAGRAMS_PER_PASS):
        try:
            datagram = datagrams.recv(MAX_MESSAGE_BYTES)
        except BlockingIOError:
            break
        message = decode_datagram(datagram, key)
        if message is not None:
            outbound += authority.handle(b"", message, now)
    return outbound


def datagram_socket(endpoint: str) -> socket.socket | None:
    """Return a non-blocking UDP socket bound to the host and port of a ``tcp://`` ``endpoint``, else None.

    Hang detection needs it to tell a rank whose Python is starved from a stopped one, so
    a port it cannot bind is an error.
    """
    if not endpoint.startswith("tcp://"):
        return None
    host, port = datagram_address(endpoint)
    datagrams = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_DGRAM)
    try:
        datagrams.bind((host, port))
    except OSError as error:
        datagrams.close()
        raise RuntimeError(
            f"Recovery authority: cannot receive native heartbeats on UDP {host}:{port}"
        ) from error
    datagrams.setblocking(False)
    return datagrams


def router_queue(endpoint: str, key: bytes) -> ZeroMqQueue:
    """Return a ROUTER queue bound at ``endpoint``.

    A connection with a routing id in use takes the route over (a worker reconnecting after its
    connection died); the session and sequence checks of :meth:`RecoveryAuthority.receive` still
    drop a replay on it. Those checks cover only inbound messages: routing ids travel in cleartext,
    so a peer without the key that connects with a worker's id receives the authority's messages to
    that worker, and the recovery times out. A message to a route with no connection fails rather
    than vanishing.
    """
    router = ZeroMqQueue(
        (endpoint, key),
        socket_type=zmq.ROUTER,
        is_server=True,
        name="recovery-authority",
        # The linger lets a final Abort reach the workers without letting close() hang on one that is gone.
        socket_options={
            zmq.LINGER: _CLOSE_LINGER_MS,
            zmq.MAXMSGSIZE: MAX_MESSAGE_BYTES,
            zmq.RCVHWM: _ROUTER_RCVHWM,
            zmq.ROUTER_HANDOVER: 1,
            zmq.ROUTER_MANDATORY: 1,
            **CONNECTION_OPTIONS,
        },
    )
    if "://*:" in endpoint or "0.0.0.0" in endpoint:
        logger.warning(
            f"Recovery authority listens on every interface ({endpoint}): any host that reaches this one can "
            "connect; bind the job's private interface and use a per-job key"
        )
    return router


def _plain_host(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return _HOSTNAME.fullmatch(host) is not None
    return True


def command_fence(template: str) -> Callable[[HungRank], None]:
    """Fence a hung worker by running ``template`` with its ``{host}`` and ``{pid}`` filled in.

    The command runs without a shell, detached from the authority's loop; a failure is
    logged, and the rank stays hung until the executor's hang detector ends the job. The
    worker reports its host and pid, so the command runs only for a plain hostname or IP
    address and a positive pid: nothing the command could take as an option.
    """
    arguments = shlex.split(template)

    def fence(hung: HungRank) -> None:
        if not (_plain_host(hung.host) and type(hung.pid) is int and hung.pid > 0):
            logger.error(
                f"Recovery authority: not fencing rank {hung.rank} with the fence command: "
                f"host {hung.host!r} or pid {hung.pid!r} is not plain"
            )
            return
        command = [arg.format(host=hung.host, pid=hung.pid) for arg in arguments]
        try:
            subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError as error:
            logger.error(
                f"Recovery authority: fencing rank {hung.rank} with {command} failed: {error}"
            )

    return fence


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
        "--fence-command",
        default="",
        help="last-resort command that kills a hung worker the workers could not fence (its own "
        "evidence thread, then the MPI runtime), with {host} and {pid} placeholders, run without a "
        "shell, e.g. 'ssh {host} kill -KILL {pid}' or the deployment's node agent",
    )
    args = parser.parse_args(argv)
    if "TLLM_LOG_LEVEL" not in os.environ:
        logger.set_level("info")

    key = load_key()
    # Timeouts and hang detection come from the workers' configuration file (TLLM_FAULT_TOLERANCE_CONFIG).
    config = fault_tolerance_config()
    hang_monitor = None
    if config.hang_detection:
        hang_monitor = HangMonitor(
            tuple(range(args.world_size)), config.hang_confirm_s, config.heartbeat_timeout_s
        )
    fence = command_fence(args.fence_command) if args.fence_command else None
    logger.info(f"Recovery authority: fault-tolerance settings {config.digest()}: {config}")
    authority = RecoveryAuthority(
        args.world_size, config.authority_timeout_s, hang_monitor, fence, config.digest()
    )
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    datagrams = datagram_socket(args.endpoint) if hang_monitor is not None else None
    router = router_queue(args.endpoint, key)
    try:
        serve(authority, router, key, stop, datagrams=datagrams)
    finally:
        router.close()
        if datagrams is not None:
            datagrams.close()


if __name__ == "__main__":
    main()

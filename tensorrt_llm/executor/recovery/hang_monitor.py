# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decide from worker evidence that one rank hangs without dying.

The recovery authority feeds the monitor the heartbeats and wait reports of the
workers and asks it, at each deadline check, whether a rank is hung. The monitor
performs no I/O and never fences; the authority acts on its decision.
"""

from dataclasses import dataclass, field

__all__ = ["HangMonitor", "HungRank"]


@dataclass(frozen=True)
class HungRank:
    """A rank the evidence shows hung, why, and the process to fence."""

    rank: int
    reason: str
    host: str = ""
    pid: int = 0


@dataclass
class _RankEvidence:
    """What the monitor knows about one rank."""

    # Monotonic time of the last heartbeat.
    heartbeat: float
    # Executor iteration of the last heartbeat, and when it last changed.
    iteration: int = -1
    progressed: float = 0.0
    # Whether the rank last reported waiting on something outside itself: a host
    # collective, or new requests.
    waiting: bool = False
    # Peer -> monotonic time since which this rank's device waits on it.
    device_waits: dict[int, float] = field(default_factory=dict)
    # The worker process, from its heartbeats.
    host: str = ""
    pid: int = 0


class HangMonitor:
    """Confirm a hung rank once the evidence against it has held for ``confirm_s``.

    A rank ``p`` is hung when, continuously for ``confirm_s``, one of these holds:

    - at least two other ranks report device waits on ``p``, ``p`` reports none of its
      own (otherwise it waits on the head of a chain), and its iteration has not
      advanced;
    - its iteration has not advanced, it is not waiting, and every other rank is
      waiting: its host loop is stuck while its peers wait for it;
    - its heartbeat is missing (the process is stopped or its host is unreachable),
      and no device waits point at another rank: a heartbeat thread starved while
      its rank waits on a hung peer must not make that rank the suspect.

    ``confirm_s`` must exceed the slowest legitimate step plus host stalls, so a
    slow rank is never confirmed. A rank is watched from its first heartbeat on, so
    workers that send none are never suspected. Each rank is confirmed at most once.

    Args:
        ranks: The ranks that may be watched.
        confirm_s: Seconds the evidence must hold.
        heartbeat_timeout_s: Seconds without a heartbeat after which it counts
            as missing.
    """

    def __init__(
        self,
        ranks: tuple[int, ...],
        confirm_s: float,
        heartbeat_timeout_s: float,
    ) -> None:
        if confirm_s <= 0 or heartbeat_timeout_s <= 0:
            raise ValueError("confirm_s and heartbeat_timeout_s must be positive")
        self.confirm_s = confirm_s
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self._ranks = frozenset(ranks)
        self._evidence: dict[int, _RankEvidence] = {}
        # Rank -> time since which the evidence against it has held.
        self._suspected_since: dict[int, float] = {}
        self._confirmed: set[int] = set()

    def heartbeat(
        self, rank: int, iteration: int, waiting: bool, now: float, host: str = "", pid: int = 0
    ) -> None:
        """Record a heartbeat of ``rank`` (process ``pid`` on ``host``) at its executor ``iteration``."""
        if rank not in self._ranks:
            return
        evidence = self._evidence.setdefault(rank, _RankEvidence(heartbeat=now, progressed=now))
        evidence.heartbeat = now
        evidence.host, evidence.pid = host, pid
        if iteration != evidence.iteration:
            evidence.iteration = iteration
            evidence.progressed = now
        evidence.waiting = waiting

    def device_wait(self, rank: int, peer: int, waited_s: float, now: float) -> None:
        """Record that ``rank``'s device has waited ``waited_s`` on ``peer``; 0 clears it."""
        evidence = self._evidence.get(rank)
        if evidence is None or peer == rank or peer not in self._ranks:
            return
        if waited_s > 0:
            evidence.device_waits.setdefault(peer, now - waited_s)
        else:
            evidence.device_waits.pop(peer, None)

    def remove(self, rank: int) -> None:
        """Stop watching ``rank``, for example once it departed."""
        self._ranks -= {rank}
        self._evidence.pop(rank, None)
        self._suspected_since.pop(rank, None)
        for evidence in self._evidence.values():
            evidence.device_waits.pop(rank, None)

    def reset(self) -> None:
        """Forget all evidence, for example after a recovery changed what every rank waits on."""
        self._evidence.clear()
        self._suspected_since.clear()

    def check(self, now: float) -> HungRank | None:
        """Return a newly confirmed hung rank, or None."""
        for rank in sorted(self._evidence):
            reason = self._evidence_against(rank, now)
            if reason is None:
                self._suspected_since.pop(rank, None)
                continue
            since = self._suspected_since.setdefault(rank, now)
            if now - since >= self.confirm_s and rank not in self._confirmed:
                self._confirmed.add(rank)
                evidence = self._evidence[rank]
                return HungRank(
                    rank, f"{reason} for {now - since:.1f}s", evidence.host, evidence.pid
                )
        return None

    def live_helpers(self, hung: int, now: float, count: int = 2) -> list[int]:
        """Up to ``count`` ranks, lowest first, that can act for the deployment.

        Their heartbeats are fresh and no evidence points at them.
        """
        return [
            rank
            for rank, evidence in sorted(self._evidence.items())
            if rank != hung
            and rank not in self._suspected_since
            and now - evidence.heartbeat <= self.heartbeat_timeout_s
        ][:count]

    def _evidence_against(self, rank: int, now: float) -> str | None:
        evidence = self._evidence[rank]
        others = {r: e for r, e in self._evidence.items() if r != rank}
        # Evidence holds only while most peers' heartbeats arrive: when they all stop, the authority has lost
        # its view of the deployment (its host stalled or its network was cut), not the ranks their progress.
        fresh = sum(now - e.heartbeat <= self.heartbeat_timeout_s for e in others.values())
        if 2 * fresh < len(others):
            return None
        stalled = now - evidence.progressed > self.heartbeat_timeout_s
        waiters = [e for e in others.values() if rank in e.device_waits]
        if len(waiters) >= 2 and not evidence.device_waits and stalled:
            return f"{len(waiters)} ranks' devices wait on it"
        if others and not evidence.waiting and stalled and all(e.waiting for e in others.values()):
            return f"its loop stopped at iteration {evidence.iteration} while every peer waits"
        if now - evidence.heartbeat > self.heartbeat_timeout_s:
            witnessed = (
                sum(r in e.device_waits for e in self._evidence.values()) >= 2 for r in others
            )
            if not evidence.device_waits and not any(witnessed):
                return "no heartbeat"
        return None

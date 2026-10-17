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
    """A rank the evidence shows hung, and why."""

    rank: int
    reason: str


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


class HangMonitor:
    """Confirm a hung rank once the evidence against it has held for ``confirm_s``.

    A rank ``p`` is hung when, continuously for ``confirm_s``, one of these holds:

    - its heartbeat is missing: the process is stopped or its host is unreachable;
    - at least two other ranks report device waits on ``p``, and ``p`` reports none
      of its own (otherwise ``p`` waits on the head of a chain);
    - its iteration has not advanced, it is not waiting, and every other rank is
      waiting: its host loop is stuck while its peers wait for it.

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

    def heartbeat(self, rank: int, iteration: int, waiting: bool, now: float) -> None:
        """Record a heartbeat of ``rank`` at its executor ``iteration``."""
        if rank not in self._ranks:
            return
        evidence = self._evidence.setdefault(rank, _RankEvidence(heartbeat=now, progressed=now))
        evidence.heartbeat = now
        if iteration != evidence.iteration:
            evidence.iteration = iteration
            evidence.progressed = now
        evidence.waiting = waiting

    def device_wait(self, rank: int, peer: int, waited_s: float, now: float) -> None:
        """Record that ``rank``'s device has waited ``waited_s`` on ``peer``; 0 clears it."""
        evidence = self._evidence.get(rank)
        if evidence is None or peer == rank:
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
                return HungRank(rank, f"{reason} for {now - since:.1f}s")
        return None

    def _evidence_against(self, rank: int, now: float) -> str | None:
        evidence = self._evidence[rank]
        if now - evidence.heartbeat > self.heartbeat_timeout_s:
            return "no heartbeat"
        others = [e for r, e in self._evidence.items() if r != rank]
        waiters = [e for e in others if rank in e.device_waits]
        if len(waiters) >= 2 and not evidence.device_waits:
            return f"{len(waiters)} ranks' devices wait on it"
        if (
            others
            and not evidence.waiting
            and now - evidence.progressed > self.heartbeat_timeout_s
            and all(e.waiting for e in others)
        ):
            return f"its loop stopped at iteration {evidence.iteration} while every peer waits"
        return None

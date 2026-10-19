# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decide from worker evidence that one rank hangs, or that every rank does.

The recovery authority feeds the monitor the heartbeats and wait reports of the
workers and asks it, at each deadline check, whether a rank is hung or the whole
deployment stalled. The monitor performs no I/O and never fences; the authority
acts on its decision.
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
    # Highest executor iteration reported, and when it last rose.
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
    """Confirm a hung rank, or a stall of every rank, once it has held for ``confirm_s``.

    A rank ``r`` is hung when, for ``confirm_s``, every other rank with a fresh
    heartbeat waits on it and ``r`` made no progress (its iteration did not rise;
    any rise restarts the window). A peer waits on ``r`` when its device waits on
    ``r`` in an all-to-all, or when it waits on the host (a collective, or new
    requests) while ``r`` does not or has no fresh heartbeat. A rank whose own
    device waits on another rank is a victim further down a chain, not the culprit.

    The deployment is stalled when no rank made progress for ``confirm_s``, every
    rank with a fresh heartbeat has had its device wait on a peer for as long, and no
    rank is waited on by every peer: no fence can help, so it must restart. An idle
    deployment has no device waits and never matches.

    No decision is made while fewer than half the ranks' heartbeats are fresh: the
    authority has then lost its view of the deployment, not the ranks their progress.
    A rank is watched from its first heartbeat on, and confirmed hung at most once.

    Args:
        ranks: The ranks that may be watched.
        confirm_s: Seconds the evidence must hold; above the slowest legitimate step
            plus host stalls.
        heartbeat_timeout_s: Seconds without a heartbeat after which it counts as
            missing.
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
        # Rank -> time since which every peer has waited on it.
        self._suspected_since: dict[int, float] = {}
        self._confirmed: set[int] = set()

    def heartbeat(
        self,
        rank: int,
        iteration: int,
        waiting: bool | None,
        now: float,
        host: str = "",
        pid: int = 0,
    ) -> None:
        """Record a heartbeat of ``rank`` at its executor ``iteration``.

        ``waiting`` None (a native heartbeat, which does not know it) keeps the last
        reported value; empty ``host`` and zero ``pid`` keep the known process.
        """
        if rank not in self._ranks:
            return
        evidence = self._evidence.setdefault(rank, _RankEvidence(heartbeat=now, progressed=now))
        evidence.heartbeat = now
        if pid:
            evidence.host, evidence.pid = host, pid
        # Two heartbeat sources report the same counter at different times: only a rise is progress.
        if iteration > evidence.iteration:
            evidence.iteration = iteration
            evidence.progressed = now
        if waiting is not None:
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
        if not self._has_view(now):
            self._suspected_since.clear()
            return None
        for rank in sorted(self._evidence):
            if not self._waited_on_by_every_peer(rank, now):
                self._suspected_since.pop(rank, None)
                continue
            evidence = self._evidence[rank]
            since = max(self._suspected_since.setdefault(rank, now), evidence.progressed)
            if now - since >= self.confirm_s and rank not in self._confirmed:
                self._confirmed.add(rank)
                return HungRank(
                    rank,
                    f"every peer waits on it and it made no progress for {now - since:.1f}s",
                    evidence.host,
                    evidence.pid,
                )
        return None

    def all_stalled(self, now: float) -> str | None:
        """Why every rank is stuck with no rank to blame, or None."""
        if len(self._evidence) < 2 or not self._has_view(now):
            return None
        for rank, evidence in self._evidence.items():
            if now - evidence.progressed < self.confirm_s:
                return None
            if self._fresh(evidence, now) and (
                not evidence.device_waits
                or now - min(evidence.device_waits.values()) < self.confirm_s
            ):
                return None
            if self._waited_on_by_every_peer(rank, now):
                return None
        return (
            f"no rank made progress and every rank's device waited on a peer for "
            f"{self.confirm_s:.0f}s, with no single rank to blame"
        )

    def live_helpers(self, hung: int, now: float, count: int = 2) -> list[int]:
        """Up to ``count`` ranks, lowest first, that can act for the deployment.

        Their heartbeats are fresh and no evidence points at them.
        """
        return [
            rank
            for rank, evidence in sorted(self._evidence.items())
            if rank != hung and rank not in self._suspected_since and self._fresh(evidence, now)
        ][:count]

    def _fresh(self, evidence: _RankEvidence, now: float) -> bool:
        return now - evidence.heartbeat <= self.heartbeat_timeout_s

    def _has_view(self, now: float) -> bool:
        fresh = sum(self._fresh(evidence, now) for evidence in self._evidence.values())
        return 2 * fresh >= len(self._evidence)

    def _waited_on_by_every_peer(self, rank: int, now: float) -> bool:
        evidence = self._evidence[rank]
        alive = self._fresh(evidence, now)
        # The device waits of a rank without a heartbeat are stale: its process stopped.
        if alive and evidence.device_waits:
            return False
        peers = [e for r, e in self._evidence.items() if r != rank and self._fresh(e, now)]
        host_blocked = not (alive and evidence.waiting)
        return bool(peers) and all(
            rank in peer.device_waits or (peer.waiting and host_blocked) for peer in peers
        )

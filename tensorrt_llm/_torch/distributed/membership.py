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
"""Membership of a WideEP deployment with fault tolerance: the record every rank follows, and the authority that
writes it.

The authority (:class:`Authority`, no I/O) decides membership from the ranks' reports and publishes one
:class:`MembershipRecord`; a change is a two-round transaction (PROPOSE, INSTALL, then COMMITTED, or ABORT). The
authority process (:class:`AuthorityProcess`, run as ``python -m tensorrt_llm._torch.distributed.membership``) hosts
a ``TCPStore`` whose values are signed with the join secret, and a listener every rank and node supervisor keeps one
connection to: the connection carries the records and reports, and its close is the process's loss. See
``docs/source/features/wide-ep-fault-tolerance.md``.
"""

import argparse
import enum
import hashlib
import hmac
import json
import os
import pickle  # nosec B403
import select
import selectors
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

from tensorrt_llm.logger import logger

from .host_exchange_epoch import latest_epoch, next_epoch

# The live mask has room for as many ranks as the all-to-all kernels support.
MAX_RANKS = 256

# How long the end of a wait-for chain may stay without progress before it counts as hung (Authority).
PROGRESS_TIMEOUT_S = 30.0
# How often a rank's agent sends its report when nothing changes (MembershipAgent).
REPORT_INTERVAL_S = 0.5
# The authority's bound on the startup: every rank must connect within it, and a rank waited on in a startup
# collective may make no progress for this long.
RENDEZVOUS_TIMEOUT_S = 600.0

# The rendezvous address (``host:port`` of the authority's store on rank 0's node) and the hex join secret.
RENDEZVOUS_ENV = "TLLM_FAULT_TOLERANCE_RENDEZVOUS"
JOIN_SECRET_ENV = "TLLM_FAULT_TOLERANCE_JOIN_SECRET"
# The launchers' rank variables, which a process started outside the launcher's job must not inherit.
LAUNCHER_ENV_PREFIXES = ("OMPI_", "PMIX_", "PMI_", "PRTE_", "SLURM_")


class MembershipState(enum.IntEnum):
    COMMITTED = 0  # serving with the generation's live ranks
    PROPOSE = 1  # ranks stop at their next wait and reply READY
    INSTALL = 2  # ranks switch to the new membership and reply INSTALLED
    ABORT = 3  # recovery cannot succeed: every rank takes the terminal path
    SHUTDOWN = 4  # a normal exit: a closed connection is not a failure
    STARTING = 5  # the original ranks start up; the deployment serves from generation 0's COMMITTED


class Reply(enum.IntEnum):
    NONE = 0
    READY = 1
    INSTALLED = 2
    # The rank could not install the membership: the authority aborts, or, for a joining rank, gives up adding it.
    FAILED = 3


class _Message(BaseModel):
    """A value of the membership protocol, sent as JSON."""

    model_config = ConfigDict(frozen=True)


class ResumeState(_Message):
    """Where every rank continues after a membership change."""

    iteration: int = 0  # the executor's step counter (main gates some collectives on it)
    epoch: int = 0  # the next epoch every backend derives its round counters from


class MembershipRecord(_Message):
    generation: int
    state: MembershipState
    live_mask: int
    resume: ResumeState = ResumeState()
    # The process each rank currently stands for (0 = original, n = n-th replacement), assigned at its hello.
    incarnations: Tuple[int, ...] = ()
    # The live ranks whose replacement joins in this transaction: they enter the host exchange only at the iteration
    # the live ranks resume after.
    joining_mask: int = 0
    # With INSTALL: the READY payloads (RankReport.ready_payload) by rank, unread by the authority.
    ready_payloads: Tuple[Tuple[int, str], ...] = Field(default=(), repr=False)
    # With ABORT: why the deployment ends.
    reason: str = ""

    def is_live(self, rank: int) -> bool:
        return bool(self.live_mask >> rank & 1)

    def lowest_survivor(self) -> Optional[int]:
        """The lowest live rank that does not join; None if every live rank joins."""
        survivors = self.live_mask & ~self.joining_mask
        return (survivors & -survivors).bit_length() - 1 if survivors else None

    def is_member(self, rank: int, incarnation: int) -> bool:
        """Whether the process ``incarnation`` of ``rank`` is a live member."""
        current = self.incarnations[rank] if rank < len(self.incarnations) else 0
        return self.is_live(rank) and current == incarnation

    def admits(self, rank: int, incarnation: int) -> bool:
        """Whether a COMMITTED record lists the process ``incarnation`` of ``rank`` as a live member."""
        return self.state == MembershipState.COMMITTED and self.is_member(rank, incarnation)


class NodeView(_Message):
    """What the authority sends a node supervisor: the record, and per known node its host and whether its supervisor
    is connected."""

    record: MembershipRecord
    nodes: Tuple[Tuple[int, str, bool], ...] = ()


class RankReport(_Message):
    """What one rank last reported to the authority."""

    # The executor's iteration; during the startup, the startup collectives the rank completed.
    progress: int = 0
    idle: bool = False  # main's hang detector is paused, as while the executor waits for requests
    waiting_for: int = 0  # mask of the ranks the rank has waited for past a threshold
    reply_generation: int = 0
    reply: Reply = Reply.NONE
    stopped_at: ResumeState = ResumeState()  # with READY: where the rank stopped
    # With INSTALLED: the digest of the EPLB placement the rank restored, the same on every live rank.
    placement_digest: str = ""
    # With READY: what the live ranks' restore needs from this rank (None from most ranks); kept small, as the INSTALL
    # record carries it to every rank.
    ready_payload: Optional[str] = None
    incarnation: int = 0  # of the process that sent it: set by the authority, which assigned it
    # Advanced with every report: a rank whose reports stop advancing is frozen or gone.
    sequence: int = 0
    # Of the ranks in waiting_for, those an all-to-all kernel waits for, with the round flag it waits for.
    waited_rounds: Tuple[Tuple[int, int], ...] = ()
    # Of the ranks in waiting_for, those the host exchange waits for, with the epoch of the round it waits in.
    waited_epochs: Tuple[Tuple[int, int], ...] = ()
    # Advanced with every install step the rank completes: a long install is progress.
    install_steps: int = 0
    # A replacement prepared everything it can without the live ranks; until it installs the change that adds it, this
    # stands for its READY.
    joining_ready: bool = False
    started: bool = False  # an original rank's startup is done
    shutdown: bool = False  # from rank 0: the deployment ends normally


class TransactionSummary(_Message):
    """A membership transaction that ended, for the server's status: when the authority detected it (the failure or
    the join request), proposed, received the last READY (None if it did not) and ended it, in milliseconds since the
    Unix epoch on the authority's host."""

    generation: int
    outcome: Literal["COMMITTED", "ABORT"]
    removed_ranks: Tuple[int, ...]
    added_ranks: Tuple[int, ...]
    detected_unix_ms: float
    began_unix_ms: float
    ready_unix_ms: Optional[float] = None
    ended_unix_ms: float


def _bits(mask: int):
    """The positions of the set bits of ``mask``, lowest first."""
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def _chain_end(rank: int, waits: Dict[int, int]) -> bool:
    """Whether ``rank`` is an end of the wait-for graph ``waits`` (rank -> mask of the ranks it waits on): it waits on
    no rank, or only on ranks that wait back on it. Such a cycle forms while a stopped process's last report, which
    names a peer that now waits on it, has not expired."""
    if any(not waits.get(q, 0) for q in _bits(waits.get(rank, 0))):
        return False  # it waits on a rank that waits on none

    def reach(r: int) -> int:
        reached = frontier = waits.get(r, 0)
        while frontier:
            beyond = 0
            for q in _bits(frontier):
                beyond |= waits.get(q, 0)
            frontier = beyond & ~reached
            reached |= frontier
        return reached

    return all(reach(q) >> rank & 1 for q in _bits(reach(rank)))


class Authority:
    """Decides membership from the ranks' reports; :meth:`tick` returns a record to publish when it changes.

    A rank fails when its connection closes, or when it is hung: the end of a wait-for chain (waited on, waiting on no
    live rank but those that wait back on it) that makes no progress for the timeout. A rank's waits count only while
    its reports advance. During a change, the authority waits on every live rank that has not replied in the phase.
    Progress is the iteration, the rounds peers wait for the rank in, and its install steps; rank 0 waiting for
    requests while COMMITTED is idle, not hung.

    While STARTING, any failure (with ``rendezvous_timeout_s`` as the hang timeout), or a rank that does not connect
    within ``rendezvous_timeout_s``, aborts. A replacement (:meth:`request_join`) is proposed once it reports
    ``joining_ready``, together with the other replacements of ranks lost in the same transaction whose reports
    advance; one that fails to join is left out, never ending the deployment.
    """

    def __init__(
        self,
        world_size: int,
        progress_timeout_s: float = PROGRESS_TIMEOUT_S,
        rendezvous_timeout_s: float = RENDEZVOUS_TIMEOUT_S,
    ):
        if not 0 < world_size <= MAX_RANKS:
            raise ValueError(f"world_size must be in (0, {MAX_RANKS}], got {world_size}")
        self.world_size = world_size
        self.progress_timeout_s = progress_timeout_s
        self.rendezvous_timeout_s = rendezvous_timeout_s
        self.record = MembershipRecord(
            generation=0, state=MembershipState.STARTING, live_mask=(1 << world_size) - 1
        )
        self._committed_mask = self.record.live_mask
        self._rendezvous_began_s: Optional[float] = None  # the first tick
        self._connected_once = 0  # mask of the ranks that connected while STARTING
        # rank -> ((iteration, install steps, rounds waited for it in), time it last advanced)
        self._progress: Dict[int, tuple] = {}
        # rank -> since when it has been the end of a wait-for chain (waited on, not exempt)
        self._stalled_since: Dict[int, float] = {}
        # rank -> ((incarnation, report sequence), time it last advanced)
        self._report_sequence: Dict[int, tuple] = {}
        self._requested: Set[int] = set()  # ranks whose replacement asked to join, not proposed yet
        # rank -> ((incarnation, startup progress) of its replacement, time it last advanced)
        self._startup_progress: Dict[int, tuple] = {}
        self._joining: Set[int] = set()  # ranks whose replacement the current transaction adds
        self._lost_in: Dict[
            int, int
        ] = {}  # lost rank -> generation of the transaction that removed it
        self._detected_s: Optional[float] = None  # when the next or current transaction became due
        self._transaction_began_s: Optional[float] = None  # restarts keep it
        self._ready_s: Optional[float] = None  # when the current attempt's last READY arrived
        self.last_transaction: Optional[TransactionSummary] = None

    def request_join(self, rank: int) -> None:
        """A replacement for ``rank`` asks to become live; if ``rank`` is still live, the replacement takes it over."""
        self._requested.add(rank)

    def tick(
        self, now_s: float, reports: Dict[int, RankReport], connected: int
    ) -> Optional[MembershipRecord]:
        """``now_s``: ``time.monotonic()``; ``connected``: mask of the ranks whose current process is connected."""
        before, committed = self.record.state, self._committed_mask
        record = self._decide(now_s, reports, connected)
        if record is None:
            return None
        if before == MembershipState.COMMITTED:
            self._transaction_began_s = now_s
        if before != MembershipState.STARTING and record.state in (
            MembershipState.COMMITTED,
            MembershipState.ABORT,
        ):
            now_unix_ms = time.time() * 1e3

            def unix_ms(s: Optional[float]) -> Optional[float]:
                return None if s is None else now_unix_ms + (s - now_s) * 1e3

            began_s = self._transaction_began_s
            self.last_transaction = TransactionSummary(
                generation=record.generation,
                outcome=record.state.name,
                removed_ranks=tuple(self._ranks(committed & ~record.live_mask)),
                added_ranks=tuple(self._ranks(record.live_mask & ~committed)),
                detected_unix_ms=unix_ms(began_s if self._detected_s is None else self._detected_s),
                began_unix_ms=unix_ms(began_s),
                ready_unix_ms=unix_ms(self._ready_s),
                ended_unix_ms=now_unix_ms,
            )
            self._detected_s = None
        return record

    def _decide(
        self, now_s: float, reports: Dict[int, RankReport], connected: int
    ) -> Optional[MembershipRecord]:
        state = self.record.state
        if state in (MembershipState.ABORT, MembershipState.SHUTDOWN):
            return None
        if 0 in reports and reports[0].shutdown:
            return self._set(MembershipState.SHUTDOWN, self.record.live_mask)
        if state == MembershipState.STARTING:
            return self._form(now_s, reports, connected)
        failed = self._failed(now_s, reports, connected, self.progress_timeout_s)
        gone = set(self._ranks(failed | ~connected))  # a replacement that died joining is dropped
        self._requested -= gone
        self._joining -= gone
        if state == MembershipState.COMMITTED:
            ready = {r for r in self._requested if r in reports and reports[r].joining_ready}
            starting = {r for r in self._requested if self._starting(r, reports, now_s)}
            if not failed and not ready:
                # Nothing is due, also after a ready join whose replacement died while it waited for others.
                self._detected_s = None
                return None
            if self._detected_s is None:
                self._detected_s = now_s
            if not failed and self._awaits_more_joins(now_s, reports, ready, starting):
                return None
            self._requested -= ready
            self._joining |= ready
            if ready:
                logger.info(f"Membership: adding joining ranks {sorted(ready)}")
            return self._propose((self.record.live_mask & ~failed) | self._mask(ready))
        if failed:  # a failure during the transaction: restart it without those ranks too
            return self._propose(self.record.live_mask & ~failed)
        live = self._ranks(self.record.live_mask)
        expected = Reply.READY if state == MembershipState.PROPOSE else Reply.INSTALLED

        def replied(r: int, reply: Reply) -> bool:
            report = reports.get(r)
            # A joining rank's readiness, reported before the PROPOSE, stands for its READY.
            ready_to_join = r in self._joining and report is not None and report.joining_ready
            return (reply == Reply.READY and ready_to_join) or self._replied(report, reply)

        failed_replies = {r for r in live if replied(r, Reply.FAILED)}
        if failed_replies - self._joining:
            return self.abort(
                f"in {state.name}, ranks {sorted(failed_replies - self._joining)} replied FAILED"
            )
        if failed_replies:
            # The change goes on without them; a replacement that failed to join is not added.
            self._joining -= failed_replies
            return self._propose(self.record.live_mask & ~self._mask(failed_replies))
        if not all(replied(r, expected) for r in live):
            return None
        if state == MembershipState.PROPOSE:
            survivors = [
                r for r in live if self._committed_mask >> r & 1 and r not in self._joining
            ]
            # Every live rank resumes past the furthest iteration a survivor stopped at, and one past the latest epoch a
            # survivor stopped in (epochs wrap, so not the largest).
            resume = ResumeState(
                iteration=max(
                    [self.record.resume.iteration]
                    + [reports[r].stopped_at.iteration for r in survivors]
                ),
                epoch=next_epoch(latest_epoch([reports[r].stopped_at.epoch for r in survivors])),
            )
            self._ready_s = now_s
            payloads = tuple(
                (r, reports[r].ready_payload) for r in live if reports[r].ready_payload is not None
            )
            return self._set(
                MembershipState.INSTALL, self.record.live_mask, resume, ready_payloads=payloads
            )
        # Survivors whose placements differ cannot serve; a joining rank whose placement differs is left out.
        joining = self.record.joining_mask
        digests = {reports[r].placement_digest for r in live if not joining >> r & 1}
        if len(digests) > 1:
            return self.abort("the surviving ranks restored different expert placements")
        mismatched = {r for r in self._ranks(joining) if reports[r].placement_digest not in digests}
        if mismatched:
            logger.warning(
                f"Membership: joining ranks {sorted(mismatched)} restored another expert placement than the "
                "surviving ranks: they are not added"
            )
            self._joining -= mismatched
            return self._propose(self.record.live_mask & ~self._mask(mismatched))
        self._joining -= set(live)
        for r in self._ranks(self._committed_mask & ~self.record.live_mask):
            self._lost_in[r] = self.record.generation
        for r in live:
            self._lost_in.pop(r, None)
        self._committed_mask = self.record.live_mask
        # No rank made progress during the transaction, and a replacement's clock does not start from the process it
        # replaces: every clock restarts.
        self._progress.clear()
        self._stalled_since.clear()
        return self._set(MembershipState.COMMITTED, self.record.live_mask)

    def _form(
        self, now_s: float, reports: Dict[int, RankReport], connected: int
    ) -> Optional[MembershipRecord]:
        """While STARTING: generation 0's COMMITTED once every rank has started; any failure aborts."""
        if self._rendezvous_began_s is None:
            self._rendezvous_began_s = now_s
        everyone = self.record.live_mask
        self._connected_once |= connected
        failed = self._failed(now_s, reports, connected, self.rendezvous_timeout_s)
        if failed & self._connected_once:
            return self.abort(
                f"ranks {self._ranks(failed & self._connected_once)} failed during the startup"
            )
        absent = everyone & ~self._connected_once
        if absent and now_s - self._rendezvous_began_s > self.rendezvous_timeout_s:
            return self.abort(
                f"ranks {self._ranks(absent)} did not reach the rendezvous within {self.rendezvous_timeout_s:.0f} s"
            )
        if absent or not all(reports.get(r, RankReport()).started for r in self._ranks(everyone)):
            return None
        logger.info("Membership: every rank started")
        # The startup's progress is not the executor's: every clock restarts.
        self._progress.clear()
        self._stalled_since.clear()
        return self._set(MembershipState.COMMITTED, everyone)

    def _awaits_more_joins(
        self, now_s: float, reports: Dict[int, RankReport], ready: Set[int], starting: Set[int]
    ) -> bool:
        """Whether the replacements ``ready`` to join wait for another one of a rank lost in the same transaction, which
        asked to join, whose reports advance and whose startup progresses (``starting``)."""
        lost_with = {self._lost_in[r] for r in ready if r in self._lost_in}
        return any(
            self._lost_in.get(r) in lost_with and self._reports_advance(r, reports, now_s)
            for r in starting - ready
        )

    def _starting(self, r: int, reports: Dict[int, RankReport], now_s: float) -> bool:
        """Whether the startup of ``r``'s replacement progressed within ``rendezvous_timeout_s``, the startup's bound
        on a rank's progress."""
        report = reports.get(r, RankReport())
        progress = (report.incarnation, report.progress)
        if self._startup_progress.get(r, (None,))[0] != progress:
            self._startup_progress[r] = (progress, now_s)
        return now_s - self._startup_progress[r][1] <= self.rendezvous_timeout_s

    def _reports_advance(self, r: int, reports: Dict[int, RankReport], now_s: float) -> bool:
        """Whether the reports of ``r``'s current process advanced within half of ``progress_timeout_s``."""
        report = reports.get(r, RankReport())
        sequence = (report.incarnation, report.sequence)
        if self._report_sequence.get(r, (None,))[0] != sequence:
            self._report_sequence[r] = (sequence, now_s)
        return now_s - self._report_sequence[r][1] <= self.progress_timeout_s / 2

    def _propose(self, live_mask: int) -> MembershipRecord:
        self._ready_s = None
        if live_mask & 1 == 0:
            logger.error("Membership: the deployment aborts: rank 0 failed")
            return self._set(MembershipState.ABORT, live_mask, reason="rank 0 failed")
        return self._set(MembershipState.PROPOSE, live_mask, generation=self.record.generation + 1)

    def abort(self, reason: str) -> MembershipRecord:
        logger.error(f"Membership: the deployment aborts: {reason}")
        return self._set(MembershipState.ABORT, self.record.live_mask, reason=reason)

    def _set(
        self,
        state: MembershipState,
        live_mask: int,
        resume: Optional[ResumeState] = None,
        generation: Optional[int] = None,
        ready_payloads: Tuple[Tuple[int, str], ...] = (),
        reason: str = "",
    ) -> MembershipRecord:
        self.record = MembershipRecord(
            generation=self.record.generation if generation is None else generation,
            state=state,
            live_mask=live_mask,
            resume=self.record.resume if resume is None else resume,
            joining_mask=self._mask(self._joining) & live_mask,
            ready_payloads=ready_payloads,
            reason=reason,
        )
        return self.record

    def _replied(self, report: Optional[RankReport], reply: Reply) -> bool:
        """Whether ``report`` gives ``reply`` to the current generation."""
        return (
            report is not None
            and report.reply_generation == self.record.generation
            and report.reply == reply
        )

    def _failed(
        self, now_s: float, reports: Dict[int, RankReport], connected: int, timeout_s: float
    ) -> int:
        """Mask of the live ranks that are dead, or hung for ``timeout_s``."""
        state, live_mask = self.record.state, self.record.live_mask
        live = self._ranks(live_mask)

        def advance(seen: Dict[int, tuple], r: int, value) -> None:
            if seen.get(r, (None,))[0] != value:
                seen[r] = (value, now_s)

        # During a change, the authority waits on every live rank that has not given the phase's reply.
        expected = {MembershipState.PROPOSE: Reply.READY, MembershipState.INSTALL: Reply.INSTALLED}
        waited_on = 0
        if state in expected:
            waited_on = self._mask(
                r for r in live if not self._replied(reports.get(r), expected[state])
            )
        # rank -> mask of the live ranks it waits on, while its reports advance
        waits: Dict[int, int] = {}
        rounds: Dict[int, Set[tuple]] = {}  # rank -> the rounds live ranks wait for it in
        for r in live:
            report = reports.get(r, RankReport())
            if self._reports_advance(r, reports, now_s):
                waits[r] = report.waiting_for & live_mask & ~(1 << r)
                waited_on |= waits[r]
            for peer, round_flag in report.waited_rounds:
                rounds.setdefault(peer, set()).add(("all-to-all", round_flag))
            for peer, epoch in report.waited_epochs:
                rounds.setdefault(peer, set()).add(("host exchange", epoch))
        failed = live_mask & ~connected
        for r in live:
            report = reports.get(r, RankReport())
            advance(
                self._progress,
                r,
                (report.progress, report.install_steps, frozenset(rounds.get(r, ()))),
            )
            idle = r == 0 and state == MembershipState.COMMITTED and r in waits and report.idle
            if waited_on >> r & 1 and not idle and _chain_end(r, waits):
                since_s = max(self._stalled_since.setdefault(r, now_s), self._progress[r][1])
                if now_s - since_s > timeout_s:
                    failed |= 1 << r
                    logger.warning(
                        f"Membership: rank {r} is hung: no progress for {now_s - since_s:.1f} s while live ranks "
                        f"wait for it. Last reports: {self._describe(live, reports, waits, now_s)}"
                    )
            else:
                self._stalled_since.pop(r, None)
        return failed

    def _describe(
        self, live: List[int], reports: Dict[int, RankReport], waits: Dict[int, int], now_s: float
    ) -> str:
        """Per live rank, what its last report says it waits on, for the log of a hang."""
        described = []
        for r in live:
            report = reports.get(r, RankReport())
            described.append(
                f"rank {r}: iteration {report.progress}, idle {report.idle}, waits for "
                f"{self._ranks(waits.get(r, 0))}, all-to-all rounds {list(report.waited_rounds)}, host-exchange "
                f"epochs {list(report.waited_epochs)}, reported {now_s - self._report_sequence[r][1]:.1f} s ago"
            )
        return "; ".join(described)

    def _ranks(self, mask: int) -> List[int]:
        return [r for r in range(self.world_size) if mask >> r & 1]

    @staticmethod
    def _mask(ranks) -> int:
        return sum(1 << r for r in ranks)


class SignedStore:
    """A torch ``TCPStore`` whose values carry an HMAC-SHA256 tag under the join secret; a value without a valid tag
    counts as absent."""

    def __init__(self, store, secret: bytes):
        self.store = store
        self._secret = secret

    def set(self, key: str, value: bytes) -> None:
        self.store.set(key, self.tag(key.encode(), value) + value)

    def get(self, key: str) -> bytes:
        """The value of ``key``; waits until it is set, up to the store's timeout.

        Raises:
            PermissionError: The value is not signed with the join secret.
        """
        value = self._verified(key, self.store.get(key))
        if value is None:
            raise PermissionError(f"rendezvous value {key!r} is not signed with the join secret")
        return value

    def get_if_set(self, key: str) -> Optional[bytes]:
        """The value of ``key``; None while it is not set, or if it is not signed."""
        return self._verified(key, self.store.get(key)) if self.store.check([key]) else None

    def multi_get(self, keys: List[str]) -> List[Optional[bytes]]:
        """The values of ``keys``, which must all be set; None for a value not signed."""
        return [
            self._verified(key, raw)
            for key, raw in zip(keys, self.store.multi_get(keys) if keys else [])
        ]

    def tag(self, key: bytes, value: bytes) -> bytes:
        message = len(key).to_bytes(4, "little") + key + value
        return hmac.new(self._secret, message, hashlib.sha256).digest()

    def _verified(self, key: str, raw: bytes) -> Optional[bytes]:
        tag, value = raw[:32], raw[32:]
        return value if hmac.compare_digest(tag, self.tag(key.encode(), value)) else None


# Store keys.
LIVENESS_KEY = "liveness"  # host:port of the authority's liveness listener
RECORD_KEY = "record"
TRANSACTION_KEY = "last_transaction"  # the authority's last TransactionSummary
# Per rank, the newest incarnation whose liveness connection closed, as did those of every older one (-1 if none): a
# process's connection closes when it ends. A JSON list, unset while no process has left.
DISCONNECTED_KEY = "disconnected"
# The name prefix of every node's shared-memory segments of host weights, which the node supervisors unlink at the end.
HOST_WEIGHTS_PREFIX_KEY = "host_weights_prefix"


# Ranks per store counter of an arrival mask: the store's counters are signed 64-bit integers.
_RANKS_PER_COUNTER = 63


def mark_arrived(store: SignedStore, prefix: str, rank: int) -> None:
    """Record that ``rank`` set its value of the gather under ``prefix``: one bit of a counter of 63 ranks. The
    counters only say whose value to wait for, so they are not signed."""
    store.store.add(
        f"{prefix}/arrived/{rank // _RANKS_PER_COUNTER}", 1 << rank % _RANKS_PER_COUNTER
    )


def _arrival_counters(prefix: str, world_size: int) -> List[str]:
    return [
        f"{prefix}/arrived/{counter}" for counter in range(-(-world_size // _RANKS_PER_COUNTER))
    ]


def arrived_ranks(store: SignedStore, prefix: str, world_size: int) -> int:
    """Mask of the ranks that marked their arrival under ``prefix`` (:func:`mark_arrived`)."""
    return sum(
        store.store.add(key, 0) << counter * _RANKS_PER_COUNTER
        for counter, key in enumerate(_arrival_counters(prefix, world_size))
    )


def authority_ports(address: str) -> Tuple[str, int, int]:
    """The host, store port and liveness port (the next one) of the authority at the rendezvous ``address``."""
    host, port = address.rsplit(":", 1)
    return host, int(port), int(port) + 1


def open_store(address: str, secret: bytes, timeout_s: float) -> SignedStore:
    """A client of the store the authority hosts at ``address`` (``host:port``)."""
    from datetime import timedelta

    from torch.distributed import TCPStore

    host, port = address.rsplit(":", 1)
    store = TCPStore(
        host,
        int(port),
        is_master=False,
        timeout=timedelta(seconds=timeout_s),
        wait_for_workers=False,
    )
    return SignedStore(store, secret)


def rendezvous_store(timeout_s: float = 60.0) -> SignedStore:
    """A client of the deployment's store, from this process's environment."""
    return open_store(
        os.environ[RENDEZVOUS_ENV], bytes.fromhex(os.environ[JOIN_SECRET_ENV]), timeout_s
    )


# Ranks per store counter of an arrival mask: the store's counters are signed 64-bit integers.
_RANKS_PER_COUNTER = 63
# The first poll of a wait for values in the store (wait_for_keys), and the longest poll of a group's wait.
_FIRST_POLL_S = 0.001
STORE_POLL_S = 0.1


def wait_for_keys(
    store: SignedStore,
    keys: List[str],
    poll_s: float,
    check: Callable[[Set[str]], None],
) -> List[bytes]:
    """The values of ``keys`` once every one is set and signed. Polls with a backoff up to ``poll_s`` (a blocking
    store wait that times out logs a warning), calling ``check`` after every poll with the keys set unsigned;
    ``check`` raises to stop waiting."""
    backoff_s = _FIRST_POLL_S
    while True:
        unsigned: Set[str] = set()
        if store.store.check(keys):
            values = store.multi_get(keys)
            if None not in values:
                return values
            unsigned = {key for key, value in zip(keys, values) if value is None}
        check(unsigned)
        time.sleep(backoff_s)
        backoff_s = min(2 * backoff_s, poll_s)


class StoreGroup:
    """A communicator of the global ranks ``ranks`` (communicator rank ``i`` is ``ranks[i]``) over the rendezvous
    store, on the pattern of main's ``StoreDwdpGroup``: call ``n`` of every member sets its value under
    ``<name>/<n>/<rank>`` and waits for the other members'. Only the members in ``live`` (every member by default)
    take part: the others' values are None, and a group whose only live member is this rank exchanges nothing.

    While a call waits, ``on_wait`` gets after every poll the mask (by global rank) of the live members whose values
    are missing, read from their arrival counters once per report interval, and may raise to stop waiting; it gets 0
    once the call ends. A call deletes this rank's value of the call before (every member has read it): only the
    last call's values stay in the store. The splits by node take the node of rank ``r`` as ``r // gpus_per_node``, as
    ``Mapping.get_node_rank`` does. ``exchange_failed`` records that an exchange raised.

    Given ``gather``, the group exchanges through it instead of the store, as an MNNVL remap does through the
    exchange of a membership change: ``gather(value)`` returns the values by global rank, None for an absent one.
    """

    def __init__(
        self,
        store: Optional[SignedStore],
        name: str,
        rank: int,
        ranks: Sequence[int],
        on_wait: Callable[[int], None] = lambda waiting: None,
        live: Optional[Sequence[int]] = None,
        gpus_per_node: int = 1,
        gather: Optional[Callable[[Any], Sequence[Any]]] = None,
    ):
        self.membership = tuple(ranks)
        self.rank = rank
        self.exchange_failed = False
        self._store = store
        self._name = name
        self._on_wait = on_wait
        self._live = tuple(r for r in self.membership if live is None or r in live)
        self._gpus_per_node = gpus_per_node
        self._gather = gather
        self._calls = 0
        self._subgroups = 0

    def Get_rank(self) -> int:
        return self.membership.index(self.rank)

    def Get_size(self) -> int:
        return len(self.membership)

    def allgather(self, value: Any) -> List[Any]:
        """Every member's value, by communicator rank; None for a member that is not live.

        Raises:
            RuntimeError: The store, which the authority's process hosts, is lost.
        """
        try:
            by_rank = self._exchange(value)
        except BaseException:
            self.exchange_failed = True
            raise
        return [by_rank.get(r) for r in self.membership]

    def barrier(self) -> None:
        self.allgather(None)

    Barrier = barrier

    def bcast(self, obj: Any, root: int = 0) -> Any:
        """The ``obj`` of communicator rank ``root`` (None if it is not live)."""
        return self.allgather(obj)[root]

    def allreduce(self, obj: Any, op: Any = None) -> Any:
        """The live members' ``obj`` reduced by ``op``: ``MPI.SUM`` (the default), ``MPI.MIN`` or ``MPI.MAX``."""
        from mpi4py import MPI

        op = MPI.SUM if op is None else op
        values = [value for value in self.allgather(obj) if value is not None]
        for mpi_op, reduce in ((MPI.SUM, sum), (MPI.MIN, min), (MPI.MAX, max)):
            if op == mpi_op:
                return reduce(values)
        raise NotImplementedError(f"StoreGroup reduces by MPI.SUM, MPI.MIN or MPI.MAX, not {op}")

    def Free(self) -> None:
        """A group holds nothing to free: its exchanges share this process's store client."""

    def subgroup(self, ranks: Sequence[int]) -> "StoreGroup":
        """The group of ``ranks``, members of this group with this rank among them, in that order."""
        ranks = tuple(ranks)
        digest = hashlib.sha256(repr(ranks).encode()).hexdigest()[:16]
        self._subgroups += 1
        return StoreGroup(
            self._store,
            f"{self._name}/{self._subgroups}-{digest}",
            self.rank,
            ranks,
            self._on_wait,
            self._live,
            self._gpus_per_node,
        )

    def Split(self, color: int = 0, key: int = 0) -> "StoreGroup":
        """The members that give the same ``color``, ordered by ``key``, then by their communicator rank; a
        replacement takes the original ranks' colors from the startup record."""
        from .startup_record import startup_record

        splits = startup_record.agree("startup_group_split", lambda: self.allgather((color, key)))
        ranks = sorted(
            (split[1], index, rank)
            for index, (rank, split) in enumerate(zip(self.membership, splits))
            if split[0] == color
        )
        return self.subgroup(rank for _, _, rank in ranks)

    def Split_type(self, split_type: int, key: int = 0, info: Any = None) -> "StoreGroup":
        """The members on this rank's node, for the node split types (``MPI.COMM_TYPE_SHARED``, Open MPI's
        ``COMM_TYPE_HOST``)."""
        from mpi4py import MPI

        from tensorrt_llm._utils import OMPI_COMM_TYPE_HOST

        if split_type not in (MPI.COMM_TYPE_SHARED, OMPI_COMM_TYPE_HOST):
            raise NotImplementedError(
                f"StoreGroup splits by node only, not by split type {split_type}"
            )
        node = self.rank // self._gpus_per_node
        return self.subgroup(r for r in self.membership if r // self._gpus_per_node == node)

    def _exchange(self, value: Any) -> Dict[int, Any]:
        if self._gather is not None:
            values = self._gather(value)
            return {r: values[r] for r in self.membership}
        if self._live == (self.rank,):
            self._on_wait(0)  # a call that ends: a replacement's startup progress
            return {self.rank: value}
        from torch.distributed import DistError

        prefix = f"{self._name}/{self._calls}"
        self._calls += 1
        try:
            self._store.set(f"{prefix}/{self.rank}", pickle.dumps(value))
            self._store.store.add(
                self._counter(prefix, self.rank), 1 << self.rank % _RANKS_PER_COUNTER
            )
            try:
                values = self._wait(prefix)
            finally:
                self._on_wait(0)
            if self._calls > 1:
                previous = f"{self._name}/{self._calls - 2}"
                self._store.store.delete_key(f"{previous}/{self.rank}")
                if self.rank == min(self._live):
                    for key in self._counters(previous):
                        self._store.store.delete_key(key)
        except DistError as error:
            raise RuntimeError(
                f"Rank {self.rank} lost the rendezvous store, which the membership authority hosts"
            ) from error
        # Only processes with the join secret write the store's values (SignedStore).
        return {r: pickle.loads(v) for r, v in zip(self._live, values)}  # nosec B301

    def _wait(self, prefix: str) -> List[bytes]:
        keys = [f"{prefix}/{r}" for r in self._live]
        live_mask = sum(1 << r for r in self._live)
        read_s, missing = None, 0

        def check(unsigned: Set[str]) -> None:
            nonlocal read_s, missing
            if read_s is None or time.monotonic() - read_s >= REPORT_INTERVAL_S:
                read_s = time.monotonic()
                arrived = sum(
                    self._store.store.add(key, 0) << int(key.rsplit("/", 1)[1]) * _RANKS_PER_COUNTER
                    for key in self._counters(prefix)
                )
                missing = live_mask & ~arrived
            self._on_wait(
                missing | sum(1 << r for r, key in zip(self._live, keys) if key in unsigned)
            )

        return wait_for_keys(self._store, keys, STORE_POLL_S, check)

    def _counters(self, prefix: str) -> List[str]:
        return sorted({self._counter(prefix, r) for r in self._live})

    @staticmethod
    def _counter(prefix: str, rank: int) -> str:
        # The counters only say whose value to wait for, so they are not signed.
        return f"{prefix}/arrived/{rank // _RANKS_PER_COUNTER}"


def join_secret_digest() -> str:
    """A name of this deployment that every process of a node may list: a digest of the join secret."""
    return hashlib.sha256(bytes.fromhex(os.environ[JOIN_SECRET_ENV])).hexdigest()[:32]


def node_supervisor_address() -> bytes:
    """The abstract Unix socket that keeps a second node supervisor of the deployment off the node."""
    return f"\0tllm-wide-ep-ft-{join_secret_digest()}".encode()


# The exit status of a replacement whose startup failure another replacement would repeat: its node supervisor then
# starts no other one.
REPLACEMENT_CANNOT_START = 78


class ReplacementCannotStart(SystemExit):
    """Ends a replacement rank with :data:`REPLACEMENT_CANNOT_START`; ``str()`` gives the reason."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.code = REPLACEMENT_CANNOT_START


# A liveness connection opens with the authority's nonce line and the hello line, tagged under the join secret and that
# nonce, so a recorded hello opens no other connection. Then each side sends JSON lines: records one way, reports the
# other.
HELLO_TIMEOUT_S = 10.0  # the authority closes a connection without a valid hello by then
_NONCE_BYTES = 16
_MAX_LINE_BYTES = 1 << 20
_MAX_UNSENT_BYTES = 1 << 22  # a process that has not read this much of its records is gone
# The selector data of the descriptor whose end of file tells the authority that rank 0's process ended.
_PARENT = object()


def _line(message: _Message) -> bytes:
    return message.model_dump_json().encode() + b"\n"


def _decided_at_once(report: RankReport) -> tuple:
    """The fields of ``report`` the authority decides on as soon as they change."""
    return (
        report.reply_generation,
        report.reply,
        report.joining_ready,
        report.started,
        report.shutdown,
    )


class _Lines:
    """The lines one side of a liveness connection receives after the hello."""

    def __init__(self):
        self._buffer = b""

    def decode(self, data: bytes) -> List[bytes]:
        """The lines that ``data`` completes.

        Raises:
            ValueError: A line is too long: the connection must close.
        """
        *lines, self._buffer = (self._buffer + data).split(b"\n")
        if len(self._buffer) > _MAX_LINE_BYTES:
            raise ValueError("a liveness line is too long")
        return lines


def _read_line(sock: socket.socket) -> bytes:
    line = b""
    while not line.endswith(b"\n"):
        chunk = sock.recv(256)
        if not chunk:
            raise ConnectionError("the membership authority closed the liveness connection")
        line += chunk
    return line.strip()


@dataclass
class LivenessConnection:
    """A process's liveness connection to the authority; closed at the end of a ``with`` block."""

    sock: socket.socket
    lines: _Lines

    def __enter__(self) -> "LivenessConnection":
        return self

    def __exit__(self, *exc_info) -> None:
        self.sock.close()


def _open_connection(store: SignedStore, announcement: str, timeout_s: float) -> LivenessConnection:
    """A connection to the authority whose hello announces ``announcement``."""
    host, port = store.get(LIVENESS_KEY).decode().rsplit(":", 1)
    sock = socket.create_connection((host, int(port)), timeout=timeout_s)
    authority_nonce = bytes.fromhex(_read_line(sock).decode())
    hello = announcement.encode()
    sock.sendall(store.tag(b"hello", authority_nonce + hello).hex().encode() + b" " + hello + b"\n")
    sock.settimeout(None)
    return LivenessConnection(sock, _Lines())


def open_liveness_connection(
    store: SignedStore, rank: int, replacement: bool = False, timeout_s: float = HELLO_TIMEOUT_S
) -> LivenessConnection:
    """This process's liveness connection to the authority for ``rank``; a ``replacement`` asks to join. The first
    record the authority sends over it gives the process its incarnation (``MembershipRecord.incarnations``)."""
    connection = _open_connection(store, f"{rank} {int(replacement)}", timeout_s)
    # A forked child must not keep the connection open.
    os.register_at_fork(after_in_child=connection.sock.close)
    return connection


def open_node_supervisor_connection(
    store: SignedStore, node: int, ranks: List[int], host: str, timeout_s: float = HELLO_TIMEOUT_S
) -> LivenessConnection:
    """The connection of the supervisor of ``node`` (on ``host``, running ``ranks``) to the authority, which sends
    :class:`NodeView` lines over it and takes ``ranks`` for failed once it closes."""
    return _open_connection(store, f"node {node} {','.join(map(str, ranks))} {host}", timeout_s)


class MembershipAgent:
    """A rank's side of membership: a thread that keeps the liveness connection, calls ``on_record`` with every
    record and sends the rank's report every ``REPORT_INTERVAL_S`` (after ``on_tick``) and at once when
    :meth:`report` changes it. It stops once a record no longer lists this process after one did, or on an error,
    which closes the connection and is passed to ``on_error``. A ``replacement`` asks to join; :attr:`incarnation`
    is the one the authority assigned this process."""

    def __init__(
        self,
        rank: int,
        store: SignedStore,
        on_record: Callable[[MembershipRecord], None],
        replacement: bool = False,
        on_tick: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        self.rank = rank
        self._store = store
        self._on_record = on_record
        self._on_tick = on_tick
        self._on_error = on_error
        self._replacement = replacement
        self._report = RankReport()
        self.record: Optional[MembershipRecord] = None
        self.incarnation: Optional[int] = None  # from the first record, once started
        # Whether the liveness connection is open: the authority closes it only when it ends or fails this rank.
        self.connected = False
        self._lock = threading.Lock()
        # Held while a record is delivered and while the callbacks change.
        self._delivery = threading.Lock()
        self._stop = threading.Event()
        self._wake_read, self._wake_write = os.pipe()
        os.set_blocking(self._wake_read, False)
        os.set_blocking(self._wake_write, False)
        self._thread = threading.Thread(
            target=self._run, name=f"MembershipAgent{rank}", daemon=True
        )
        self._sock: Optional[socket.socket] = None
        self._lines: Optional[_Lines] = None
        self._received: List[bytes] = []  # the lines start() read, which the thread delivers first

    def start(self) -> None:
        """Connect, take this process's incarnation from the authority's first record and start the thread.

        Raises:
            ConnectionError: The authority refused this process.
        """
        connection = open_liveness_connection(self._store, self.rank, self._replacement)
        self._sock, self._lines = connection.sock, connection.lines
        self._sock.settimeout(HELLO_TIMEOUT_S)
        while not self._received:
            data = self._sock.recv(1 << 16)
            if not data:
                self._sock.close()
                raise ConnectionError(
                    f"the membership authority refused this process of rank {self.rank}"
                )
            self._received = self._lines.decode(data)
        self._sock.settimeout(None)
        self.incarnation = MembershipRecord.model_validate_json(self._received[0]).incarnations[
            self.rank
        ]
        self.connected = True
        self._send_report()  # the authority has it as soon as it admits the connection
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake()
        self._thread.join()
        self._sock.close()
        # A report() racing this stop must not write to a closed or reused descriptor.
        with self._lock:
            os.close(self._wake_read)
            os.close(self._wake_write)
            self._wake_write = None

    def bind(
        self,
        on_record: Callable[[MembershipRecord], None],
        on_tick: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        """Call these from now on in place of the callbacks given so far; ``on_record`` first receives the newest
        record, if one arrived already."""
        with self._delivery:
            self._on_record, self._on_tick, self._on_error = on_record, on_tick, on_error
            if self.record is not None:
                on_record(self.record)

    def report(self, **changes) -> None:
        """Update fields of the report, which the agent then sends."""
        with self._lock:
            self._report = self._report.model_copy(update=changes)
        self._wake()

    def _wake(self) -> None:
        with self._lock:
            if self._wake_write is None:
                return  # stopped
            try:
                os.write(self._wake_write, b"\0")
            except BlockingIOError:
                pass  # already woken

    def _send_report(self) -> None:
        with self._lock:
            self._report = self._report.model_copy(update={"sequence": self._report.sequence + 1})
            report = self._report
        self._sock.sendall(_line(report))

    def _run(self) -> None:
        listed = False  # a record listed this process as a live member
        tick_s = time.monotonic()  # when the next periodic report is due
        lines, self._received = self._received, []
        try:
            while not self._stop.is_set():
                for line in lines:
                    record = MembershipRecord.model_validate_json(line)
                    with self._delivery:
                        self.record = record
                        self._on_record(record)
                    member = record.is_member(self.rank, self.incarnation)
                    if listed and not member:
                        return  # excluded, left out or replaced
                    listed = member
                readable, _, _ = select.select(
                    [self._sock, self._wake_read], [], [], max(0.0, tick_s - time.monotonic())
                )
                lines = []
                if self._sock in readable:
                    data = self._sock.recv(1 << 16)
                    if not data:
                        raise ConnectionError(
                            "the membership authority closed the liveness connection"
                        )
                    lines = self._lines.decode(data)
                due = time.monotonic() >= tick_s
                if due:
                    tick_s = time.monotonic() + REPORT_INTERVAL_S
                    if self._on_tick is not None:
                        self._on_tick()
                if due or self._wake_read in readable:
                    try:
                        os.read(self._wake_read, 1 << 12)
                    except BlockingIOError:
                        pass
                    self._send_report()
        except Exception as error:  # noqa: BLE001 - any failure here must end this rank's membership visibly
            logger.error(f"Membership agent of rank {self.rank} failed: {error!r}")
            self.connected = False
            self._sock.close()
            if self._on_error is not None:
                self._on_error(error)


@dataclass
class _Connection:
    """A liveness connection the authority accepted, and once its hello is valid, the process it stands for."""

    nonce: bytes
    accepted_s: float
    received: bytes = b""
    member: Optional[Tuple[int, int]] = None  # (rank, incarnation), once a rank's hello is valid
    node: Optional[int] = None  # once a node supervisor's hello is valid
    node_ranks: Tuple[int, ...] = ()
    lines: Optional[_Lines] = None  # once its hello is valid
    report: Optional[RankReport] = None
    unsent: bytes = b""


class AuthorityProcess:
    """The authority as a process: publishes every record in the store and over every connection, and decides with
    :class:`Authority` from the reports.

    A replacement joins by connecting as one, and gets the rank's next incarnation: never for rank 0, before the
    deployment has started, or while the rank's current process is connected, so only the authority's verdict removes
    a live process. Once ``parent`` (readable only at its end of file: rank 0's process ended) ends, the deployment
    aborts. A node supervisor's connection receives :class:`NodeView` lines; once it closes while the deployment
    serves, the connections of its ranks are closed too.
    """

    def __init__(
        self,
        world_size: int,
        store: SignedStore,
        authority: Authority,
        parent: Optional[int] = None,
    ):
        self.world_size = world_size
        self._store = store
        self._authority = authority
        self._parent = parent
        self._parent_ended_s: Optional[float] = None
        # Something the authority decides on at once changed since its last decision (see _serve).
        self._changed = False
        self._incarnation = [0] * world_size
        self._disconnected = [-1] * world_size  # as last written to DISCONNECTED_KEY
        self._connections: Dict[socket.socket, _Connection] = {}
        self._selector = selectors.DefaultSelector()
        self._record: Optional[MembershipRecord] = None  # the last one published
        # node -> host, of every node whose supervisor connected once
        self._hosts: Dict[int, str] = {}

    def run(self, host: str = "", port: int = 0) -> MembershipRecord:
        """Serve until the record is ABORT or SHUTDOWN, the liveness listener bound to ``host`` and ``port`` (all
        addresses, a free port by default); returns the last record."""
        try:
            with socket.create_server((host, port)) as listener:
                listener.setblocking(False)
                self._selector.register(listener, selectors.EVENT_READ)
                if self._parent is not None:
                    self._selector.register(self._parent, selectors.EVENT_READ, _PARENT)
                self._publish(self._authority.record)
                self._store.set(
                    LIVENESS_KEY,
                    f"{host or socket.gethostname()}:{listener.getsockname()[1]}".encode(),
                )
                return self._serve()
        finally:
            for sock in list(self._connections):
                sock.close()
            self._selector.close()

    def _serve(self) -> MembershipRecord:
        """Decide every ``REPORT_INTERVAL_S``, and at once when a connection opens or closes, rank 0's process ends or
        a report changes a field of :func:`_decided_at_once`."""
        decide_s = time.monotonic()
        while True:
            self._poll(max(0.0, decide_s - time.monotonic()))
            if not self._changed and time.monotonic() < decide_s:
                continue
            self._changed = False
            decide_s = time.monotonic() + REPORT_INTERVAL_S
            summary = self._authority.last_transaction
            if self._parent_ended_s is not None:
                record = self._authority.abort("rank 0's process ended")
            else:
                record = self._authority.tick(time.monotonic(), self._reports(), self._connected())
            if record is None:
                continue
            logger.info(f"Membership: {record!r}")
            transaction = self._authority.last_transaction
            if transaction is not summary:
                logger.info(f"Membership transaction: {transaction!r}")
                self._store.set(TRANSACTION_KEY, transaction.model_dump_json().encode())
            record = self._publish(record)
            if record.state in (MembershipState.ABORT, MembershipState.SHUTDOWN):
                # The store lives in this process: keep it until every live rank has read the record and left, or,
                # once rank 0's process ended, a rank that has not left within the hang timeout counts as hung.
                while self._connected() & record.live_mask and (
                    self._parent_ended_s is None
                    or time.monotonic() - self._parent_ended_s < self._authority.progress_timeout_s
                ):
                    self._poll(REPORT_INTERVAL_S)
                return record

    def _publish(self, record: MembershipRecord) -> MembershipRecord:
        record = record.model_copy(update={"incarnations": tuple(self._incarnation)})
        self._store.set(RECORD_KEY, record.model_dump_json().encode())
        self._record = record
        self._send_record()
        return record

    def _send_record(self, ranks: bool = True) -> None:
        """Send the record to every rank's process (with ``ranks``), and the :class:`NodeView` to every node
        supervisor."""
        record = _line(self._record)
        connected = {c.node for c in self._connections.values() if c.node is not None}
        view = _line(
            NodeView(
                record=self._record,
                nodes=tuple(
                    (node, host, node in connected) for node, host in sorted(self._hosts.items())
                ),
            )
        )
        for sock, connection in list(self._connections.items()):
            # A connection an earlier send in this loop dropped is gone.
            if sock in self._connections and connection.lines is not None:
                if connection.node is not None or ranks:
                    connection.unsent += record if connection.node is None else view
                    self._flush(sock, connection)

    def _flush(self, sock: socket.socket, connection: _Connection) -> None:
        try:
            sent = sock.send(connection.unsent)
        except BlockingIOError:
            sent = 0
        except OSError:
            self._drop(sock)
            return
        connection.unsent = connection.unsent[sent:]
        if len(connection.unsent) > _MAX_UNSENT_BYTES:
            sender = connection.member if connection.node is None else f"node {connection.node}"
            logger.warning(f"Membership: closing the connection of {sender}, which reads nothing")
            self._drop(sock)
            return
        write = selectors.EVENT_WRITE if connection.unsent else 0
        self._selector.modify(sock, selectors.EVENT_READ | write, connection)

    def _poll(self, timeout_s: float) -> None:
        for key, events in self._selector.select(timeout_s):
            sock, connection = key.fileobj, key.data
            if connection is None:  # the listener
                self._accept(sock)
                continue
            if connection is _PARENT:  # readable only at its end of file
                self._selector.unregister(sock)
                self._parent_ended_s = time.monotonic()
                self._changed = True
                continue
            if sock not in self._connections:  # closed earlier in this batch
                continue
            if events & selectors.EVENT_WRITE:
                self._flush(sock, connection)
            if events & selectors.EVENT_READ and sock in self._connections:
                self._receive(sock, connection)
        now_s = time.monotonic()
        for sock, connection in list(self._connections.items()):
            if (
                sock in self._connections
                and connection.lines is None
                and now_s - connection.accepted_s > HELLO_TIMEOUT_S
            ):
                logger.warning("Membership: closing a liveness connection without a valid hello")
                self._drop(sock)

    def _accept(self, listener: socket.socket) -> None:
        """Accept a connection; at most ``MAX_RANKS`` wait for their hello, the oldest dropped first."""
        waiting = [sock for sock, c in self._connections.items() if c.lines is None]
        if len(waiting) >= MAX_RANKS:
            logger.warning(
                "Membership: closing the oldest liveness connection waiting for its hello"
            )
            self._drop(waiting.pop(0))
        try:
            sock, _ = listener.accept()
        except BlockingIOError:
            return
        except OSError as error:  # out of descriptors (EMFILE, ENFILE) or memory
            logger.warning(f"Membership: cannot accept a liveness connection: {error!r}")
            if waiting:
                self._drop(waiting[0])
            return
        sock.setblocking(False)
        nonce = os.urandom(_NONCE_BYTES)
        try:
            sock.send(nonce.hex().encode() + b"\n")  # fits the empty send buffer
        except OSError:
            sock.close()
            return
        self._connections[sock] = _Connection(nonce, time.monotonic())
        self._selector.register(sock, selectors.EVENT_READ, self._connections[sock])

    def _receive(self, sock: socket.socket, connection: _Connection) -> None:
        try:
            data = sock.recv(1 << 16)
        except BlockingIOError:
            return
        except OSError:
            data = b""
        if not data:
            self._drop(sock)
            return
        try:
            if connection.lines is None:
                data = self._read_hello(sock, connection, data)
            for line in connection.lines.decode(data) if data else ():
                if connection.node is not None:
                    raise ValueError(f"the supervisor of node {connection.node} sent a report")
                report = RankReport.model_validate_json(line).model_copy(
                    update={"incarnation": connection.member[1]}
                )
                if connection.report is None or _decided_at_once(
                    connection.report
                ) != _decided_at_once(report):
                    self._changed = True
                connection.report = report
        except (ValueError, TypeError, KeyError) as error:
            logger.warning(f"Membership: closing a liveness connection: {error}")
            self._drop(sock)

    def _read_hello(self, sock: socket.socket, connection: _Connection, data: bytes) -> bytes:
        """Admit the connection once its hello line is complete and valid; returns what follows the line.

        Raises:
            ValueError: The hello is not valid, or opens no connection (:meth:`_refusal`).
        """
        connection.received += data
        if b"\n" not in connection.received:
            if len(connection.received) > 256:
                raise ValueError("no hello")
            return b""
        line, rest = connection.received.split(b"\n", 1)
        tag_hex, hello = line.split(b" ", 1)
        if not hmac.compare_digest(
            bytes.fromhex(tag_hex.decode()), self._store.tag(b"hello", connection.nonce + hello)
        ):
            raise ValueError("a hello not signed with the join secret")
        announcement = hello.decode().split()
        if announcement[:1] == ["node"]:
            _, node, ranks, host = announcement
            node, ranks = int(node), tuple(int(rank) for rank in ranks.split(","))
            refusal = self._node_refusal(node, ranks)
            if refusal is not None:
                raise ValueError(refusal)
            connection.node, connection.node_ranks, connection.lines = node, ranks, _Lines()
            self._hosts[node] = host
            logger.info(
                f"Membership: the supervisor of node {node} ({host}, ranks {list(ranks)}) connected"
            )
            self._send_record(ranks=False)
            return rest
        rank, replacement = (int(field) for field in announcement)
        refusal = self._refusal(rank, replacement)
        if refusal is not None:
            raise ValueError(refusal)
        if replacement:
            self._incarnation[rank] += 1
            self._authority.request_join(rank)
        connection.member, connection.lines = (rank, self._incarnation[rank]), _Lines()
        self._changed = True
        # The first record a process receives gives it its incarnation.
        connection.unsent += _line(
            self._record.model_copy(update={"incarnations": tuple(self._incarnation)})
        )
        self._flush(sock, connection)
        return rest

    def _node_refusal(self, node: int, ranks: Tuple[int, ...]) -> Optional[str]:
        """Why a valid hello of the supervisor of ``node`` running ``ranks`` opens no connection; None if it does."""
        if not ranks or not all(0 <= rank < self.world_size for rank in ranks):
            return f"node {node}'s ranks {list(ranks)} are not all in the deployment"
        for c in self._connections.values():
            if c.node == node or set(c.node_ranks) & set(ranks):
                return f"node {node} or one of its ranks {list(ranks)} has a supervisor already"
        return None

    def _refusal(self, rank: int, replacement: int) -> Optional[str]:
        """Why a valid hello for ``rank``, of a ``replacement`` or of the rank's first process, opens no connection;
        None if it does."""
        if not 0 <= rank < self.world_size:
            return f"rank {rank} is not in the deployment"
        if not replacement:
            if self._incarnation[rank]:
                return f"rank {rank}'s first process was replaced"
            if any(c.member == (rank, 0) for c in self._connections.values()):
                return f"rank {rank}'s first process is connected already"
            return None
        if rank == 0:
            return "rank 0 is never replaced"
        if self._authority.record.state == MembershipState.STARTING:
            return "no replacement joins before the deployment has started"
        if self._authority.record.is_live(rank) and self._connected() >> rank & 1:
            return f"rank {rank} is live: no replacement joins for it"
        return None

    def _drop(self, sock: socket.socket) -> None:
        self._selector.unregister(sock)
        connection = self._connections.pop(sock)
        sock.close()
        self._changed = True
        if connection.member is not None:
            self._publish_disconnected(connection.member[0])
        if connection.node is None:
            return
        if self._record.state not in (MembershipState.ABORT, MembershipState.SHUTDOWN):
            logger.warning(
                f"Membership: the supervisor of node {connection.node} left: its ranks "
                f"{list(connection.node_ranks)} count as failed"
            )
            for other, peer in list(self._connections.items()):
                if peer.member is not None and peer.member[0] in connection.node_ranks:
                    self._drop(other)
        self._send_record(ranks=False)

    def _publish_disconnected(self, rank: int) -> None:
        """Write :data:`DISCONNECTED_KEY` once a connection of ``rank``'s processes closed, if that advanced it."""
        connected = [
            c.member[1] for c in self._connections.values() if c.member and c.member[0] == rank
        ]
        newest = min(connected) - 1 if connected else self._incarnation[rank]
        if newest > self._disconnected[rank]:
            self._disconnected[rank] = newest
            self._store.set(DISCONNECTED_KEY, json.dumps(self._disconnected).encode())

    def _connected(self) -> int:
        return sum(
            1 << rank
            for rank, incarnation in (c.member for c in self._connections.values() if c.member)
            if incarnation == self._incarnation[rank]
        )

    def _reports(self) -> Dict[int, RankReport]:
        """The last report of every rank's current process."""
        return {
            c.member[0]: c.report
            for c in self._connections.values()
            if c.member is not None
            and c.report is not None
            and c.member[1] == self._incarnation[c.member[0]]
        }


def main(argv: Optional[List[str]] = None) -> int:
    """The authority process: hosts the store at the rendezvous address, writes it to ``--address-fd`` once it
    listens, and serves until SHUTDOWN (exit status 0) or ABORT (1). Its standard input is a pipe whose writing end
    the starting process (rank 0's) holds: at its end of file the deployment aborts."""
    parser = argparse.ArgumentParser(description="WideEP fault-tolerance membership authority")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--address-fd", type=int, default=1)
    parser.add_argument("--rendezvous-timeout-s", type=float, default=RENDEZVOUS_TIMEOUT_S)
    args = parser.parse_args(argv)

    from torch.distributed import TCPStore

    address = os.environ[RENDEZVOUS_ENV]
    host, store_port, liveness_port = authority_ports(address)
    tcp_store = TCPStore(host, store_port, is_master=True, wait_for_workers=False)
    store = SignedStore(tcp_store, bytes.fromhex(os.environ[JOIN_SECRET_ENV]))
    os.write(args.address_fd, f"{address}\n".encode())
    if args.address_fd != 1:
        os.close(args.address_fd)
    authority = Authority(args.world_size, rendezvous_timeout_s=args.rendezvous_timeout_s)
    # A signal for the deployment (Ctrl-C in its terminal) is its ranks' to act on.
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, signal.SIG_IGN)
    process = AuthorityProcess(args.world_size, store, authority, parent=sys.stdin.fileno())
    record = process.run(host, liveness_port)
    return 0 if record.state == MembershipState.SHUTDOWN else 1


if __name__ == "__main__":
    sys.exit(main())

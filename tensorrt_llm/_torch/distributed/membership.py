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

The authority is the one process that decides membership. It reads every rank's report (its progress, whether it
is idle, the peers it waits for, and its replies) and writes one :class:`MembershipRecord` that every rank follows:
the generation, the transaction state, the live ranks and the state every rank resumes from. A change of membership
is a transaction of two rounds: ``PROPOSE`` (ranks stop and reply ``READY`` with where they stopped and what their
restore needs from them, which the authority passes on unread), ``INSTALL`` (ranks switch to the new membership,
restore the lost experts and reply ``INSTALLED``, or ``FAILED``), then ``COMMITTED``. ``ABORT`` ends the deployment
when recovery cannot succeed, with the reason in the record. Before the deployment serves, the record is
``STARTING``: every rank joins at the entry of its worker, and the loss of any rank during the startup aborts.

The decision logic (:class:`Authority`) does no I/O, so it is tested by itself. The authority process
(:class:`AuthorityProcess`) needs no GPU: it hosts a torch ``TCPStore`` whose values are signed with the join secret
(:class:`SignedStore`), where the record and the ranks' rendezvous values are, and a listener every rank keeps one TCP
connection to, which carries the record to the rank as it changes and the rank's report to the authority. The
connection is the rank's liveness: the kernel closes it when the process ends, however it ends, while a rank whose
Python thread is starved keeps it open (a rank that becomes unreachable stops reporting, and the hang rule finds it).
It is the authority's liveness for the
rank as well, and its hello, signed with the join secret, authenticates it. Each node supervisor
(``tensorrt_llm/llmapi/_llmapi_node_supervisor.py``) keeps a connection too, over which it follows the record
(:class:`NodeView`); once that connection closes, the authority takes the node's ranks for failed.

Run as ``python -m tensorrt_llm._torch.distributed.membership`` (:func:`main`), the module is the authority process.
"""

import argparse
import enum
import hashlib
import hmac
import itertools
import os
import pickle  # nosec B403
import secrets
import select
import selectors
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field, computed_field

from tensorrt_llm.logger import logger

from .host_exchange_epoch import latest_epoch, next_epoch

# The live mask has room for as many ranks as the all-to-all kernels support.
MAX_RANKS = 256

# How long the end of a wait-for chain may stay without progress before it counts as hung (Authority).
PROGRESS_TIMEOUT_S = 30.0
# How often a rank's agent sends its report when nothing changes (MembershipAgent).
REPORT_INTERVAL_S = 0.5
# How long the original ranks have to reach the rendezvous, from the entry of their workers before any model loads:
# every rank's startup waits share this deadline, and the authority waits this long for every rank to connect. During
# the startup, a rank waited on in a startup collective may make no progress for this long (Authority).
RENDEZVOUS_TIMEOUT_S = 600.0

# ``host:port`` on rank 0's node where the authority hosts its store (the rendezvous address; its liveness listener
# takes the next port, authority_ports), and the join secret (hex) that signs its values. Every process of the
# deployment takes both from its environment.
RENDEZVOUS_ENV = "TLLM_FAULT_TOLERANCE_RENDEZVOUS"
JOIN_SECRET_ENV = "TLLM_FAULT_TOLERANCE_JOIN_SECRET"
# The environment variables through which MPI and Slurm launchers give a process its rank in their job: a process
# started outside that job (the authority, a rank a node supervisor starts) runs without them.
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
    """A value of the membership protocol, sent as JSON (``model_dump_json``, ``model_validate_json``)."""

    model_config = ConfigDict(frozen=True)


class ResumeState(_Message):
    """Where every rank continues after a membership change; the authority sets it from the READY replies."""

    iteration: int = 0  # the executor's step counter (main gates some collectives on it)
    epoch: int = 0  # the next epoch every backend derives its round counters from

    def at_least(self, other: "ResumeState") -> "ResumeState":
        return ResumeState(
            iteration=max(self.iteration, other.iteration), epoch=max(self.epoch, other.epoch)
        )


class MembershipRecord(_Message):
    generation: int
    state: MembershipState
    live_mask: int
    resume: ResumeState = ResumeState()
    incarnations: Tuple[
        int, ...
    ] = ()  # the process each rank currently stands for (0 = original, n = n-th replacement)
    # Mask of the live ranks whose replacement joins in this transaction: they enter the host exchange only at the
    # iteration the live ranks resume after.
    joining_mask: int = 0
    # Advanced by the authority with every record it publishes: a reader ignores a record older than one it has.
    sequence: int = 0
    # With INSTALL: per live rank that sent one, the payload of its READY (RankReport.ready_payload), unread by the
    # authority; every live rank restores from these. Left out of the record's log line.
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
        """Whether the process ``incarnation`` of ``rank`` is admitted: a COMMITTED record lists it as a live
        member. A process stays admitted until a record no longer lists it."""
        return self.state == MembershipState.COMMITTED and self.is_member(rank, incarnation)


class NodeView(_Message):
    """What the authority sends a node supervisor: the record, and per node whose supervisor connected once, the node,
    its host and whether its supervisor's connection is open."""

    record: MembershipRecord
    nodes: Tuple[Tuple[int, str, bool], ...] = ()


class RankReport(_Message):
    """What one rank last reported to the authority."""

    # The executor's iteration (a replacement continues from the resume state's); during the startup, the startup
    # collectives the rank completed.
    progress: int = 0
    idle: bool = False  # main's hang detector is paused, as while the executor waits for requests
    waiting_for: int = (
        0  # mask of the ranks the rank has waited for past a threshold, 0 when not waiting
    )
    reply_generation: int = 0
    reply: Reply = Reply.NONE
    stopped_at: ResumeState = ResumeState()  # with READY: where the rank stopped
    # With INSTALLED: the digest of the EPLB placement the rank restored, the same on every live rank.
    placement_digest: str = ""
    # With READY: what the live ranks' restore needs from this rank, if anything; the INSTALL record carries it to
    # every live rank, so it is kept small (None from most ranks).
    ready_payload: Optional[str] = None
    incarnation: int = 0  # 0 for the original process of the rank, n for its n-th replacement
    # Advanced with every report the rank sends: a rank whose reports stop advancing is frozen or gone.
    sequence: int = 0
    # Of the ranks in waiting_for, those a MoE all-to-all kernel waits for, each with the round flag it waits for: a
    # rank that delivers the round and is then waited for in a later one makes progress.
    waited_rounds: Tuple[Tuple[int, int], ...] = ()
    # Of the ranks in waiting_for, those the host exchange waits for, each with the epoch of the round it waits in.
    waited_epochs: Tuple[Tuple[int, int], ...] = ()
    # Advanced with every step of installing a membership change the rank completes: a long install is progress.
    install_steps: int = 0
    # A replacement: it prepared everything it can without the live ranks, its READY payload included, so the
    # authority may add it. Until it installs the change that adds it, this report stands for its READY.
    joining_ready: bool = False
    # An original rank: its startup is done (its executor is ready to serve).
    started: bool = False
    # From rank 0: the deployment ends normally, so the authority publishes SHUTDOWN.
    shutdown: bool = False


class TransactionSummary(_Message):
    """A membership transaction that ended, from its first PROPOSE to its COMMIT or ABORT, for the server's status.

    When each phase ended, in milliseconds since the Unix epoch on the authority's host, to compare with the ranks'
    logs: the authority saw the failure (a closed connection or a hang verdict) or the join request, sent the first
    PROPOSE, received the last READY of the attempt that went on (restarts included; None if none did), and ended the
    transaction with the COMMIT, which follows the last INSTALLED at once, or the ABORT. The durations of the phases
    derive from these times.
    """

    generation: int
    state: MembershipState  # COMMITTED or ABORT
    removed_ranks: Tuple[int, ...]  # live before the transaction and not after it
    added_ranks: Tuple[int, ...]  # live after the transaction and not before it
    detected_unix_ms: float
    began_unix_ms: float
    ready_unix_ms: Optional[float] = None
    ended_unix_ms: float

    @computed_field
    @property
    def duration_s(self) -> float:
        return (self.ended_unix_ms - self.began_unix_ms) / 1e3

    @computed_field
    @property
    def detect_to_propose_ms(self) -> float:
        return round(self.began_unix_ms - self.detected_unix_ms, 1)

    @computed_field
    @property
    def propose_to_ready_ms(self) -> Optional[float]:
        ready = self.ready_unix_ms
        return None if ready is None else round(ready - self.began_unix_ms, 1)

    @computed_field
    @property
    def ready_to_installed_ms(self) -> Optional[float]:
        if self.state != MembershipState.COMMITTED:
            return None
        return round(self.ended_unix_ms - self.ready_unix_ms, 1)

    def times_unix_ms(self) -> Dict[str, Optional[int]]:
        """When each phase ended (see the class), by name; None for a phase the transaction did not reach."""
        committed = self.ended_unix_ms if self.state == MembershipState.COMMITTED else None
        times = {
            "detected": self.detected_unix_ms,
            "proposed": self.began_unix_ms,
            "ready": self.ready_unix_ms,
            "committed": committed,
        }
        return {phase: None if t is None else round(t) for phase, t in times.items()}


def _bits(mask: int):
    """The positions of the set bits of ``mask``, lowest first."""
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def _chain_end(rank: int, waits: Dict[int, int]) -> bool:
    """Whether ``rank`` is an end of the wait-for graph ``waits`` (rank -> mask of the ranks it waits on): it waits on
    no rank, or only on ranks that wait back on it, as the members of a cycle that waits on no other rank. Such a cycle
    forms while a stopped process's last report, which names a peer that now waits on it, has not expired: the
    stopped process is an end from its stop on, so it is found hung one hang timeout later."""
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

    A rank is dead when its connection to the authority is gone, and hung when it is the end of a wait-for chain that
    does not move for ``progress_timeout_s``: it is waited on, makes no progress, and waits on no live rank, or only on
    ranks that wait back on it (a cycle, whose members are one end), all along that time. A rank waits on the ranks
    its report names, but only while its reports advance (within half of ``progress_timeout_s``): a stopped process
    stops its agent thread too, so its last report names no rank and exempts it from nothing. While a membership
    change is in progress, the authority also waits on every live rank that has not replied in the current phase
    (PROPOSE or INSTALL): a rank that hangs during the change counts as failed like one that crashes, which restarts
    the change with the larger set of failed ranks. A rank's progress is
    its iteration, the rounds its peers wait for it in (all-to-all round flags, host-exchange epochs) and the steps of
    an install it completed: a rank that delivers the rounds its peers wait for, or goes on installing, advances. The
    one wait that may last is for client requests: rank 0 is exempt while its reports advance and say it is idle, and
    the membership is committed. Rank 0's report also ends the deployment normally (``shutdown``: SHUTDOWN).

    A membership without rank 0 is aborted, never served. The INSTALL record carries the payloads of the READY replies
    unread (``ready_payloads``): every live rank plans the expert restore from them alike, and that plan is the one
    rule for whether the live ranks keep every expert. A surviving rank that replies FAILED (a plan that cannot keep
    every expert fails on every live rank), or surviving ranks whose INSTALLED replies carry different placement
    digests, abort the transaction.

    A replacement asks to join (:meth:`request_join`) and is added once its report says it is ready to join
    (``joining_ready``: it prepared everything it can without the live ranks), so the live ranks stop for it only
    then; it is marked in the record's joining mask until the transaction commits. Its readiness stands for its READY:
    the live ranks never wait for a joining rank's reply before INSTALL. The hang rule bounds the wait for it as for
    any live rank. The ranks lost in one transaction (a node's) are added in one: a ready join waits for every other
    replacement of a rank lost in the same transaction that asked to join and whose reports advance, and the log of
    the proposal says how long after the first one each was ready. A joining rank that replies FAILED, fails or
    restores another placement than the survivors is not added: the transaction restarts without it, so a replacement
    never ends a deployment that serves without it. :attr:`last_transaction` summarizes the last transaction that
    ended.

    The record is STARTING while the original ranks start up, from the entry of their workers, and becomes generation
    0's COMMITTED once every rank reports that its startup is done (``started``). A deployment that loses a rank
    before then cannot serve: the authority aborts as soon as a rank's connection closes, a connected rank's reports
    stop advancing for ``progress_timeout_s``, ``rendezvous_timeout_s`` passes before every rank has connected, or a
    rank waited on in a startup collective is the end of a wait-for chain that does not move for
    ``rendezvous_timeout_s`` (the hang rule, with the startup's bound).
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
        self._joining: Set[int] = set()  # ranks whose replacement the current transaction adds
        # rank -> when its replacement, not proposed yet, was first seen ready to join
        self._ready_to_join_s: Dict[int, float] = {}
        self._lost_in: Dict[
            int, int
        ] = {}  # lost rank -> generation of the transaction that removed it
        self._detected_s: Optional[float] = None  # when the next or current transaction became due
        self._transaction_began_s: Optional[float] = None  # restarts keep it
        self._ready_s: Optional[float] = None  # when the current attempt's last READY arrived
        self.last_transaction: Optional[TransactionSummary] = None

    def request_join(self, rank: int) -> None:
        """A replacement for ``rank`` asks to become live; it is proposed once it reports that it is ready to join. If
        ``rank`` is still live (its old process was not yet found failed), the replacement takes it over."""
        self._requested.add(rank)

    def abort(self, reason: str) -> MembershipRecord:
        return self._abort(reason)

    def tick(
        self, now_s: float, reports: Dict[int, RankReport], connected: int
    ) -> Optional[MembershipRecord]:
        """``now_s``: the current time on a clock that does not jump (``time.monotonic()``); ``connected``: mask of the
        ranks whose current process holds its connection to the authority."""
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
                state=record.state,
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
            self._ready_to_join_s = {r: self._ready_to_join_s.get(r, now_s) for r in ready}
            if not failed and not ready:
                # Nothing is due, also after a ready join whose replacement died while it waited for others.
                self._detected_s = None
                return None
            if self._detected_s is None:
                self._detected_s = now_s
            if not failed and self._awaits_more_joins(now_s, reports, ready):
                return None
            self._requested -= ready
            self._joining |= ready
            if ready:
                first_s = min(self._ready_to_join_s.values())
                after_first = ", ".join(
                    f"{self._ready_to_join_s[r] - first_s:.2f}" for r in sorted(ready)
                )
                logger.info(
                    f"Membership: adding joining ranks {sorted(ready)}, ready to join {after_first} s after "
                    f"the first one; proposed {now_s - first_s:.2f} s after it"
                )
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
            return self._abort(
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
            # The survivors decide where every live rank resumes, a joining rank included: at the furthest iteration
            # a survivor stopped at, and one past the latest epoch a survivor stopped in (epochs wrap, so not the
            # largest).
            resume = self.record.resume
            for r in survivors:
                resume = resume.at_least(reports[r].stopped_at)
            last = latest_epoch([reports[r].stopped_at.epoch for r in survivors])
            resume = resume.model_copy(update={"epoch": next_epoch(last)})
            self._ready_s = now_s
            payloads = tuple(
                (r, reports[r].ready_payload) for r in live if reports[r].ready_payload is not None
            )
            return self._set(
                MembershipState.INSTALL, self.record.live_mask, resume, ready_payloads=payloads
            )
        # Every live rank restored the placement it planned from the INSTALL record's: the plans must agree. Survivors
        # that disagree hold no placement the deployment can serve with; a joining rank that disagrees is left out.
        joining = self.record.joining_mask
        digests = {reports[r].placement_digest for r in live if not joining >> r & 1}
        if len(digests) > 1:
            return self._abort("the surviving ranks restored different expert placements")
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
        """While STARTING: generation 0's COMMITTED once every rank has started, or ABORT (see the class
        docstring)."""
        if self._rendezvous_began_s is None:
            self._rendezvous_began_s = now_s
        everyone = self.record.live_mask
        ended = self._connected_once & ~connected
        self._connected_once |= connected
        if ended:
            return self._abort(f"ranks {self._ranks(ended)} ended during the startup")
        absent = everyone & ~self._connected_once
        if absent and now_s - self._rendezvous_began_s > self.rendezvous_timeout_s:
            return self._abort(
                f"ranks {self._ranks(absent)} did not reach the rendezvous within {self.rendezvous_timeout_s:.0f} s"
            )
        silent = []
        for r in self._ranks(connected):
            self._reports_advance(r, reports, now_s)
            if now_s - self._report_sequence[r][1] > self.progress_timeout_s:
                silent.append(r)
        if silent:
            return self._abort(
                f"ranks {silent} stopped reporting for {self.progress_timeout_s:.0f} s during the startup"
            )
        hung = self._failed(now_s, reports, connected, self.rendezvous_timeout_s) & connected
        if hung:
            return self._abort(
                f"ranks {self._ranks(hung)} made no progress for {self.rendezvous_timeout_s:.0f} s while other "
                "ranks waited for them in a startup collective"
            )
        if absent or not all(reports.get(r, RankReport()).started for r in self._ranks(everyone)):
            return None
        logger.info("Membership: every rank started")
        # The startup's progress is not the executor's: every clock restarts.
        self._progress.clear()
        self._stalled_since.clear()
        return self._set(MembershipState.COMMITTED, everyone)

    def _awaits_more_joins(
        self, now_s: float, reports: Dict[int, RankReport], ready: Set[int]
    ) -> bool:
        """Whether the replacements ``ready`` to join wait for another replacement of a rank lost in the same
        transaction as one of them, which asked to join and whose reports advance, so that one transaction adds them
        all. One whose process ends or stops reporting is not waited for."""
        lost_with = {self._lost_in[r] for r in ready if r in self._lost_in}
        return any(
            self._lost_in.get(r) in lost_with and self._reports_advance(r, reports, now_s)
            for r in self._requested - ready
        )

    def _reports_advance(self, r: int, reports: Dict[int, RankReport], now_s: float) -> bool:
        """Whether the reports of ``r``'s current process advanced within half of ``progress_timeout_s``: a stopped
        or unreachable process's do not."""
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

    def _abort(self, reason: str) -> MembershipRecord:
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
        """Mask of the live ranks that are dead, or hung for ``timeout_s``, at ``now_s`` (see the class docstring)."""
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
    """A torch ``TCPStore`` whose values carry an HMAC-SHA256 tag under the join secret: only processes started with
    the secret can publish membership values, and a value without a valid tag counts as absent."""

    def __init__(self, store, secret: bytes):
        self.store = store
        self._secret = secret

    def set(self, key: str, value: bytes) -> None:
        self.store.set(key, self.tag(key.encode(), value) + value)

    def get(self, key: str) -> bytes:
        """The value of ``key``; waits until it is set.

        Raises:
            PermissionError: The value is not signed with the join secret.
        """
        value = self._verified(key, self.store.get(key))
        if value is None:
            raise PermissionError(f"rendezvous value {key!r} is not signed with the join secret")
        return value

    def get_if_set(self, key: str) -> Optional[bytes]:
        """The value of ``key``; None while it is not set, or if it is not signed with the join secret."""
        return self._verified(key, self.store.get(key)) if self.store.check([key]) else None

    def multi_get(self, keys: List[str]) -> List[Optional[bytes]]:
        """The values of ``keys``, which must all be set; None for a value not signed with the join secret."""
        return [
            self._verified(key, raw)
            for key, raw in zip(keys, self.store.multi_get(keys) if keys else [])
        ]

    def wait(self, keys: List[str], timeout_s: float) -> bool:
        """Whether every key of ``keys`` is set within ``timeout_s``."""
        from datetime import timedelta

        from torch.distributed import DistStoreError

        try:
            self.store.wait(keys, timedelta(seconds=timeout_s))
        except DistStoreError:  # not all set in time
            return False
        return True

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
# How the names of the shared-memory segments of every node's host weights start (rank 0 sets it with EPLB): the
# node supervisors unlink them once the deployment ended.
HOST_WEIGHTS_PREFIX_KEY = "host_weights_prefix"


def incarnation_key(rank: int) -> str:
    """The last incarnation claimed for ``rank`` (:func:`claim_incarnation`): ``<incarnation> <nonce>``, unset before
    the rank's first replacement."""
    return f"incarnation/{rank}"


def _last_claim(store: SignedStore, rank: int) -> bytes:
    key = incarnation_key(rank)
    return store.store.get(key) if store.store.check([key]) else b""


def claimed_incarnation(store: SignedStore, rank: int) -> int:
    """The last incarnation claimed for ``rank``; 0 before its first replacement."""
    claim = _last_claim(store, rank)
    return int(claim.split()[0]) if claim else 0


def claim_incarnation(store: SignedStore, rank: int, failed: int) -> Optional[int]:
    """Claim the incarnation of a replacement for the process ``failed`` of ``rank``: ``failed + 1``, if the rank's
    last claim is ``failed``, by one compare-and-set in the store. Of the claims that replace one process, one wins;
    None for the others."""
    last = _last_claim(store, rank)
    if (int(last.split()[0]) if last else 0) != failed:
        return None
    # The nonce tells this claim from another of the same incarnation.
    claim = f"{failed + 1} {secrets.token_hex(8)}"
    won = store.store.compare_set(incarnation_key(rank), last.decode(), claim) == claim.encode()
    return failed + 1 if won else None


def replacement_key(rank: int) -> str:
    """Where the node supervisor of ``rank`` publishes how replacing it goes (:class:`ReplacementState`)."""
    return f"replacement/{rank}"


class ReplacementState(_Message):
    """How the node supervisor's replacement of a rank goes, for the server's status: the last replacement it
    started is ``starting`` (until a record admits it), or ended before its admission and the next one starts after
    a backoff (``retrying``), or no replacement can start (``cannot_start``: one exited with
    :data:`REPLACEMENT_CANNOT_START`, or another claim replaced the process)."""

    incarnation: int
    state: Literal["starting", "retrying", "cannot_start"]
    exit_status: Optional[int] = None  # of the last replacement that ended before its admission


# Ranks per store counter of an arrival mask: the store's counters are signed 64-bit integers.
_RANKS_PER_COUNTER = 63


def mark_arrived(store: SignedStore, prefix: str, rank: int) -> None:
    """Record that ``rank`` set its value of the gather under ``prefix``: one bit of a counter of 63 ranks, so that
    :func:`arrived_ranks` tells who did with one store call per 63 ranks rather than one per rank. The counters carry
    no value, only whose value to wait for, so they are not signed."""
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
    """The host and the two ports of the authority at the rendezvous address ``address`` (``host:port``): its store
    listens at ``port`` and its liveness listener, bound to ``host``, at ``port + 1``. A firewall or network policy
    opens these two ports of rank 0's node to the deployment's nodes."""
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
    """A client of the deployment's store, at the rendezvous address and with the join secret of this process's
    environment."""
    return open_store(
        os.environ[RENDEZVOUS_ENV], bytes.fromhex(os.environ[JOIN_SECRET_ENV]), timeout_s
    )


# The first poll of a wait for values in the store (wait_for_keys).
_FIRST_POLL_S = 0.001


def wait_for_keys(
    store: SignedStore,
    keys: List[str],
    poll_s: float,
    check: Callable[[Set[str]], None],
) -> List[bytes]:
    """The values of ``keys`` once every one is set and signed with the join secret (:class:`SignedStore`). Polls
    with a backoff up to ``poll_s`` (torch logs a warning for every blocking wait of the store that times out), and
    calls ``check`` after every poll with the keys set without a valid signature (yet); ``check`` raises to stop
    waiting."""
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


def store_gather(
    store: SignedStore,
    name: str,
    rank: int,
    ranks: Sequence[int],
    size: int,
    wait: Callable[[List[str], List[int], Callable[[], int]], List[bytes]],
) -> Callable[[object], List[object]]:
    """An allgather among ``ranks`` (``rank`` among them) through ``store``, under keys of ``name`` and the call:
    every call returns the values of ranks ``0 .. size - 1``, None for a rank outside ``ranks``.
    ``wait(keys, ranks, missing)`` returns the values of ``keys``, which ``ranks`` set (:func:`wait_for_keys`);
    ``missing()`` is the mask of those ranks that have not set theirs yet (:func:`arrived_ranks`). Only processes with
    the join secret can write the values. The keys of a call are deleted once every rank has completed the next one,
    so that only the last call's stay."""
    ranks = list(ranks)
    mask = sum(1 << r for r in ranks)
    calls = itertools.count()

    def gather(value: object) -> List[object]:
        call = next(calls)
        prefix = f"{name}/{call}"
        store.set(f"{prefix}/{rank}", pickle.dumps(value))
        mark_arrived(store, prefix, rank)
        values = wait(
            [f"{prefix}/{r}" for r in ranks],
            ranks,
            lambda: mask & ~arrived_ranks(store, prefix, size),
        )
        if call:
            # Every rank set its value of this call, so each has read the values of the call before.
            previous = f"{name}/{call - 1}"
            store.store.delete_key(f"{previous}/{rank}")
            if rank == min(ranks):
                for counter in _arrival_counters(previous, size):
                    store.store.delete_key(counter)
        by_rank = dict(zip(ranks, values))
        return [
            pickle.loads(by_rank[r]) if r in by_rank else None  # nosec B301
            for r in range(size)
        ]

    return gather


def startup_gather(
    store: SignedStore,
    name: str,
    rank: int,
    ranks: Sequence[int],
    size: int,
    on_wait: Callable[[int], None],
) -> Callable[[object], List[object]]:
    """A :func:`store_gather` among the ``ranks`` that start the deployment together, under keys of
    ``startup/<name>``. A wait for peers' values blocks: the rank's membership agent ends the process once the
    deployment ends, and the loss of the store, which the authority's process hosts, ends the wait with an error.
    ``on_wait`` is called with the mask of the peers the wait is for before it blocks, and with 0 once it returns.

    Raises:
        PermissionError: A value is not signed with the join secret.
    """

    def wait(keys: List[str], _ranks: List[int], missing: Callable[[], int]) -> List[bytes]:
        from torch.distributed import DistNetworkError

        try:
            waiting_for = missing()
            if waiting_for:
                on_wait(waiting_for)
            while not store.wait(keys, RENDEZVOUS_TIMEOUT_S):
                pass  # the authority decides how long a startup wait may last
            values = store.multi_get(keys)
        except DistNetworkError as error:
            raise RuntimeError(
                f"Rank {rank} lost the rendezvous store, which the membership authority hosts, while it waited for "
                "ranks to start"
            ) from error
        on_wait(0)
        if None in values:
            raise PermissionError(f"a startup value of {keys} is not signed with the join secret")
        return values

    return store_gather(store, f"startup/{name}", rank, ranks, size, wait)


def join_secret_digest() -> str:
    """Names this deployment where every process of a node can list the name: a digest of the join secret of this
    process's environment, not the secret."""
    return hashlib.sha256(bytes.fromhex(os.environ[JOIN_SECRET_ENV])).hexdigest()[:32]


def node_supervisor_address() -> bytes:
    """The abstract Unix socket the deployment's node supervisor binds, which keeps a second node supervisor of the
    deployment off the node."""
    return f"\0tllm-wide-ep-ft-{join_secret_digest()}".encode()


# The exit status of a replacement rank whose startup fails in a way that starting another one would repeat: its
# node supervisor then starts no other one.
REPLACEMENT_CANNOT_START = 78


class ReplacementCannotStart(SystemExit):
    """Ends a replacement rank whose startup fails in a way that starting another one would repeat, with exit status
    :data:`REPLACEMENT_CANNOT_START`; ``str()`` gives the reason."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.code = REPLACEMENT_CANNOT_START


# A liveness connection opens with the authority's nonce line and the hello line, which announces the process and is
# tagged under the join secret and the authority's nonce: only a process started with the secret opens a connection,
# and a recorded hello opens no other. The connection is then authenticated, and each side sends JSON lines over it:
# the authority the record, the rank its report.
HELLO_TIMEOUT_S = 10.0  # the authority closes a connection without a valid hello by then
_NONCE_BYTES = 16
_MAX_LINE_BYTES = 1 << 20
_MAX_UNSENT_BYTES = 1 << 22  # a process that has not read this much of its records is gone
# The selector data of the descriptor whose end of file tells the authority that rank 0's process ended.
_PARENT = object()


def _line(message: _Message) -> bytes:
    return message.model_dump_json().encode() + b"\n"


def _decided_at_once(report: RankReport) -> tuple:
    """The fields of ``report`` the authority decides on as soon as they change (AuthorityProcess._serve)."""
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
    """A process's liveness connection to the authority, and the lines it receives after the hello; closed at the end
    of a ``with`` block."""

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
    store: SignedStore, rank: int, incarnation: int, timeout_s: float = HELLO_TIMEOUT_S
) -> LivenessConnection:
    """This process's liveness connection to the authority, announced with a hello for ``rank`` and
    ``incarnation``: a replacement, whose incarnation is above the rank's current one, thereby asks to join."""
    connection = _open_connection(store, f"{rank} {incarnation}", timeout_s)
    # A forked child must not keep the connection open.
    os.register_at_fork(after_in_child=connection.sock.close)
    return connection


def open_node_supervisor_connection(
    store: SignedStore, node: int, ranks: List[int], host: str, timeout_s: float = HELLO_TIMEOUT_S
) -> LivenessConnection:
    """The connection of the supervisor of ``node`` (on ``host``, running ``ranks``) to the authority, which sends a
    :class:`NodeView` over it, and takes ``ranks`` for failed once it closes. The authority refuses it (closes it at
    once) while another node supervisor holds ``node`` or one of ``ranks``."""
    return _open_connection(store, f"node {node} {','.join(map(str, ranks))} {host}", timeout_s)


class MembershipAgent:
    """A rank's side of membership: a thread that keeps the liveness connection, follows the records the authority
    sends over it and sends the rank's report: every ``interval_s``, after ``on_tick`` (called on this thread) has
    updated it, and at once when :meth:`report` changes it. ``on_record`` is called on this thread with every newer
    record. Once a record no longer lists this process as a live member after one did (excluded, left out of the
    change that would add it, or replaced by a newer incarnation), the thread stops; so does an error, which also
    closes the connection, so the authority treats the rank as failed rather than silent, and then calls ``on_error``
    with it. :meth:`bind` replaces the callbacks, as
    once the rank's executor takes over the membership the rank joined at the entry of its worker."""

    def __init__(
        self,
        rank: int,
        store: SignedStore,
        on_record: Callable[[MembershipRecord], None],
        interval_s: float = REPORT_INTERVAL_S,
        incarnation: int = 0,
        on_tick: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        self.rank = rank
        self._store = store
        self._on_record = on_record
        self._on_tick = on_tick
        self._on_error = on_error
        self._interval_s = interval_s
        self._report = RankReport(incarnation=incarnation)
        self.record: Optional[MembershipRecord] = None
        # Whether the liveness connection is open: the authority closes it only when it ends or fails this rank.
        self.connected = False
        self._lock = threading.Lock()
        self._delivery = (
            threading.Lock()
        )  # held while a record is delivered and while the callbacks change
        self._stop = threading.Event()
        self._wake_read, self._wake_write = os.pipe()
        os.set_blocking(self._wake_read, False)
        os.set_blocking(self._wake_write, False)
        self._thread = threading.Thread(
            target=self._run, name=f"MembershipAgent{rank}", daemon=True
        )
        self._sock: Optional[socket.socket] = None
        self._lines: Optional[_Lines] = None

    def start(self) -> None:
        connection = open_liveness_connection(self._store, self.rank, self._report.incarnation)
        self._sock, self._lines = connection.sock, connection.lines
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
        """Update fields of the report (progress, idle, waiting_for, replies, placement_digest...), which the
        agent then sends."""
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
        incarnation = self._report.incarnation
        listed = False  # a record listed this process as a live member
        tick_s = time.monotonic()  # when the next periodic report is due
        try:
            while not self._stop.is_set():
                readable, _, _ = select.select(
                    [self._sock, self._wake_read], [], [], max(0.0, tick_s - time.monotonic())
                )
                if self._sock in readable:
                    data = self._sock.recv(1 << 16)
                    if not data:
                        raise ConnectionError(
                            "the membership authority closed the liveness connection"
                        )
                    for line in self._lines.decode(data):
                        record = MembershipRecord.model_validate_json(line)
                        with self._delivery:
                            self.record = record
                            self._on_record(record)
                        member = record.is_member(self.rank, incarnation)
                        if listed and not member:
                            return  # excluded, left out or replaced
                        listed = member
                due = time.monotonic() >= tick_s
                if due:
                    tick_s = time.monotonic() + self._interval_s
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
    """A liveness connection the authority accepted: its nonce and, until the hello, what it sent; then the process
    it stands for (a rank's process or a node supervisor), the lines it sends and the rank's last report."""

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
    """The authority as a process: publishes the record in the store and sends it over every liveness connection,
    reads the ranks' reports from their connections, and decides with :class:`Authority`.

    A replacement for rank r joins by connecting with a newer incarnation of r (:func:`open_liveness_connection`):
    never for rank 0, and not while r is live and its current process holds its connection, so that only the
    authority's own verdict removes a live process. From then on only that incarnation counts for r; the record lists
    every rank's current incarnation, so a replaced process learns it is no longer a member. A process has one
    connection; one without a valid hello within ``hello_timeout_s`` is closed, as is one that sends a line that is
    not a report. No replacement joins while the original ranks start (STARTING). Rank 0's report stops the authority
    (``shutdown``). Once ``parent``, a descriptor that becomes readable only at its end of file, ends (the process that
    started the authority, rank 0's, ended), the deployment aborts, and the authority waits for the live ranks to leave
    for at most the hang timeout. If the authority itself fails, it ends without a verdict: the ranks find their
    connections closed and serve on without protection (no lease).

    A node supervisor connects with a hello for its node, its host and the ranks it runs
    (:func:`open_node_supervisor_connection`); one node, and each rank, has at most one. It receives a
    :class:`NodeView` instead of the record, also whenever a node supervisor connects or leaves. Its ranks die with it,
    so once its connection closes while the deployment serves, the authority closes theirs too: they count as failed
    at once, a process of theirs that outlives its supervisor included.
    """

    def __init__(
        self,
        world_size: int,
        store: SignedStore,
        authority: Authority,
        interval_s: float = REPORT_INTERVAL_S,
        hello_timeout_s: float = HELLO_TIMEOUT_S,
        parent: Optional[int] = None,
    ):
        self.world_size = world_size
        self._store = store
        self._authority = authority
        self._interval_s = interval_s
        self._hello_timeout_s = hello_timeout_s
        self._parent = parent
        self._parent_ended_s: Optional[float] = None
        # Something the authority decides on at once changed since its last decision (see _serve).
        self._changed = False
        self._incarnation = [0] * world_size
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
        """Decide every ``interval_s``, and at once when a connection opens or closes, rank 0's process ends or a
        report changes a reply or what the rank says of its startup, readiness to join or shutdown; a report that only
        advances the rank's progress or waits waits for the next interval."""
        decide_s = time.monotonic()
        while True:
            self._poll(max(0.0, decide_s - time.monotonic()))
            if not self._changed and time.monotonic() < decide_s:
                continue
            self._changed = False
            decide_s = time.monotonic() + self._interval_s
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
                    self._poll(self._interval_s)
                return record

    def _publish(self, record: MembershipRecord) -> MembershipRecord:
        record = record.model_copy(
            update={
                "incarnations": tuple(self._incarnation),
                "sequence": 0 if self._record is None else self._record.sequence + 1,
            }
        )
        self._store.set(RECORD_KEY, record.model_dump_json().encode())
        self._record = record
        self._send_record()
        return record

    def _send_record(self, ranks: bool = True) -> None:
        """Send every process that announced itself the record (with ``ranks``; the replaced and excluded ones
        included), and every node supervisor its :class:`NodeView`."""
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
                and now_s - connection.accepted_s > self._hello_timeout_s
            ):
                logger.warning("Membership: closing a liveness connection without a valid hello")
                self._drop(sock)

    def _accept(self, listener: socket.socket) -> None:
        """Accept a connection; at most ``MAX_RANKS`` wait for their hello: the oldest goes first, as when the process
        is out of descriptors."""
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
                report = RankReport.model_validate_json(line)
                if connection.report is None or _decided_at_once(
                    connection.report
                ) != _decided_at_once(report):
                    self._changed = True
                connection.report = report
        except (ValueError, TypeError, KeyError) as error:
            logger.warning(f"Membership: closing a liveness connection: {error}")
            self._drop(sock)

    def _read_hello(self, sock: socket.socket, connection: _Connection, data: bytes) -> bytes:
        """Admit the connection once its hello line is complete and valid; returns what follows the line (nothing
        before).

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
        rank, incarnation = (int(field) for field in announcement)
        refusal = self._refusal(rank, incarnation)
        if refusal is not None:
            raise ValueError(refusal)
        if incarnation > self._incarnation[rank]:  # a replacement asks to join
            self._incarnation[rank] = incarnation
            self._authority.request_join(rank)
        connection.member, connection.lines = (rank, incarnation), _Lines()
        self._changed = True
        connection.unsent += _line(self._record)
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

    def _refusal(self, rank: int, incarnation: int) -> Optional[str]:
        """Why a valid hello of the process ``incarnation`` of ``rank`` opens no connection; None if it does."""
        if not 0 <= rank < self.world_size:
            return f"rank {rank} is not in the deployment"
        if incarnation < self._incarnation[rank]:
            return f"rank {rank}'s incarnation {incarnation} was replaced"
        if any(c.member == (rank, incarnation) for c in self._connections.values()):
            return f"rank {rank}'s incarnation {incarnation} is connected already"
        if incarnation > self._incarnation[rank]:
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
    """The authority process: hosts the store at the rendezvous address (``RENDEZVOUS_ENV``, which must name this
    node) and its liveness listener at the next port (:func:`authority_ports`), writes that ``host:port`` to
    ``--address-fd`` (default stdout) once it listens, then serves until the record is SHUTDOWN (exit status 0) or
    ABORT (1). The join secret comes from ``JOIN_SECRET_ENV``. Once the process that started it (rank 0's) has ended,
    the deployment aborts and the process ends: that process holds the writing end of a pipe that is this process's
    standard input, which nothing writes to.
    """
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
    # A signal for the deployment (Ctrl-C in its terminal) is its ranks' to act on: the authority ends with rank 0's
    # process, or once the deployment shut down.
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, signal.SIG_IGN)
    process = AuthorityProcess(args.world_size, store, authority, parent=sys.stdin.fileno())
    record = process.run(host, liveness_port)
    return 0 if record.state == MembershipState.SHUTDOWN else 1


if __name__ == "__main__":
    sys.exit(main())

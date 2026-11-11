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
"""Membership of a Wide-EP deployment with fault tolerance.

The authority process decides membership from the ranks' reports and publishes it as a :class:`MembershipRecord`.
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

# The active mask has room for as many ranks as the all-to-all kernels support.
MAX_RANKS = 256

HANG_TIMEOUT_S = 30.0
REPORT_INTERVAL_S = 0.5
RENDEZVOUS_TIMEOUT_S = 600.0

RENDEZVOUS_ENV = "TLLM_FAULT_TOLERANCE_RENDEZVOUS"
SHARED_SECRET_ENV = "TLLM_FAULT_TOLERANCE_SHARED_SECRET"
# The launchers' rank variables, which a process started outside the launcher's job must not inherit.
LAUNCHER_ENV_PREFIXES = ("OMPI_", "PMIX_", "PMI_", "PRTE_", "SLURM_")


class MembershipState(enum.IntEnum):
    COMMITTED = 0
    PROPOSE = 1
    INSTALL = 2
    ABORTED = 3
    SHUTDOWN = 4  # a normal exit, in which a closed connection is not a failure
    STARTING = 5


class Reply(enum.IntEnum):
    NONE = 0
    READY = 1
    INSTALLED = 2
    FAILED = 3


class _Message(BaseModel):
    """A value of the membership protocol, sent as JSON."""

    model_config = ConfigDict(frozen=True)


class ResumeState(_Message):
    """Where every rank continues after a membership change."""

    iteration: int = 0
    epoch: int = 0  # every backend derives its round counters from this epoch


class MembershipRecord(_Message):
    generation: int
    state: MembershipState
    active_mask: int
    resume: ResumeState = ResumeState()
    # The process each rank currently stands for: 0 is the original, n the n-th replacement.
    incarnations: Tuple[int, ...] = ()
    joining_mask: int = 0
    ready_payloads: Tuple[Tuple[int, str], ...] = Field(default=(), repr=False)
    reason: str = ""

    def is_active(self, rank: int) -> bool:
        return bool(self.active_mask >> rank & 1)

    def lowest_survivor(self) -> Optional[int]:
        """Return the lowest active rank that does not join, or None if every active rank joins."""
        survivors = self.active_mask & ~self.joining_mask
        return (survivors & -survivors).bit_length() - 1 if survivors else None

    def is_member(self, rank: int, incarnation: int) -> bool:
        current = self.incarnations[rank] if rank < len(self.incarnations) else 0
        return self.is_active(rank) and current == incarnation

    def admits(self, rank: int, incarnation: int) -> bool:
        return self.state == MembershipState.COMMITTED and self.is_member(rank, incarnation)


class NodeView(_Message):
    """The record and, per known node, its host and whether its supervisor is connected."""

    record: MembershipRecord
    nodes: Tuple[Tuple[int, str, bool], ...] = ()


class RankReport(_Message):
    """What one rank last reported to the authority."""

    # The executor's iteration, or during the startup the number of startup collectives the rank completed.
    progress: int = 0
    idle: bool = False
    waiting_for: int = 0
    reply_generation: int = 0
    reply: Reply = Reply.NONE
    paused_at: ResumeState = ResumeState()
    placement_digest: str = ""
    # The INSTALL record carries every READY payload to every rank, so keep it small.
    ready_payload: Optional[str] = None
    incarnation: int = 0
    sequence: int = 0
    waited_rounds: Tuple[Tuple[int, int], ...] = ()
    waited_epochs: Tuple[Tuple[int, int], ...] = ()
    install_steps: int = 0
    # A joining replacement's readiness stands for its READY to the change that adds it.
    joining_ready: bool = False
    started: bool = False
    shutdown: bool = False


class TransactionSummary(_Message):
    """A membership transaction that ended, with its milestones in Unix milliseconds on the authority's host."""

    generation: int
    outcome: Literal["COMMITTED", "ABORTED"]
    removed_ranks: Tuple[int, ...]
    added_ranks: Tuple[int, ...]
    detected_unix_ms: float
    began_unix_ms: float
    ready_unix_ms: Optional[float] = None
    ended_unix_ms: float


def _bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def _chain_end(rank: int, waits: Dict[int, int]) -> bool:
    """Whether ``rank`` waits on no rank, or only on ranks that wait back on it, in the wait-for graph ``waits``.

    Such a cycle forms while a hung process's last report names a peer that now waits on it.
    """
    if any(not waits.get(q, 0) for q in _bits(waits.get(rank, 0))):
        return False

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
    """Decide membership from the ranks' reports.

    A rank fails when its connection closes, or when it is the end of a wait-for chain that makes no progress for
    the timeout. Any failure while STARTING aborts. A replacement that fails to join is left out.
    """

    def __init__(
        self,
        world_size: int,
        hang_timeout_s: float = HANG_TIMEOUT_S,
        rendezvous_timeout_s: float = RENDEZVOUS_TIMEOUT_S,
    ):
        if not 0 < world_size <= MAX_RANKS:
            raise ValueError(f"world_size must be in (0, {MAX_RANKS}], got {world_size}")
        self.world_size = world_size
        self.hang_timeout_s = hang_timeout_s
        self.rendezvous_timeout_s = rendezvous_timeout_s
        self.record = MembershipRecord(
            generation=0, state=MembershipState.STARTING, active_mask=(1 << world_size) - 1
        )
        self._committed_mask = self.record.active_mask
        self._rendezvous_began_s: Optional[float] = None
        self._connected_once = 0
        # rank -> (value, time it last advanced), for each of the progress, report and startup clocks
        self._progress: Dict[int, tuple] = {}
        self._hung_since: Dict[int, float] = {}
        self._report_sequence: Dict[int, tuple] = {}
        self._requested: Set[int] = set()
        self._startup_progress: Dict[int, tuple] = {}
        self._joining: Set[int] = set()
        self._failed_in: Dict[int, int] = {}
        # The resume epoch of the INSTALL that a newer change replaced before its commit, or 0.
        self._replaced_install_epoch = 0
        self._detected_s: Optional[float] = None
        self._transaction_began_s: Optional[float] = None
        self._ready_s: Optional[float] = None
        self.last_transaction: Optional[TransactionSummary] = None

    def request_join(self, rank: int) -> None:
        """Request that a replacement of ``rank`` join, taking ``rank`` over even if it is still active."""
        self._requested.add(rank)

    def tick(
        self, now_s: float, reports: Dict[int, RankReport], connected: int
    ) -> Optional[MembershipRecord]:
        """Return a new record to publish, or None if membership did not change."""
        before, committed = self.record.state, self._committed_mask
        record = self._decide(now_s, reports, connected)
        if record is None:
            return None
        if before == MembershipState.COMMITTED:
            self._transaction_began_s = now_s
        if before != MembershipState.STARTING and record.state in (
            MembershipState.COMMITTED,
            MembershipState.ABORTED,
        ):
            now_unix_ms = time.time() * 1e3

            def unix_ms(s: Optional[float]) -> Optional[float]:
                return None if s is None else now_unix_ms + (s - now_s) * 1e3

            began_s = self._transaction_began_s
            self.last_transaction = TransactionSummary(
                generation=record.generation,
                outcome=record.state.name,
                removed_ranks=tuple(self._ranks(committed & ~record.active_mask)),
                added_ranks=tuple(self._ranks(record.active_mask & ~committed)),
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
        if state in (MembershipState.ABORTED, MembershipState.SHUTDOWN):
            return None
        if 0 in reports and reports[0].shutdown:
            return self._set(MembershipState.SHUTDOWN, self.record.active_mask)
        if state == MembershipState.STARTING:
            return self._form(now_s, reports, connected)
        failed = self._failed(now_s, reports, connected, self.hang_timeout_s)
        gone = set(self._ranks(failed | ~connected))
        self._requested -= gone
        self._joining -= gone
        if state == MembershipState.COMMITTED:
            ready = {r for r in self._requested if r in reports and reports[r].joining_ready}
            starting = {r for r in self._requested if self._starting(r, reports, now_s)}
            if not failed and not ready:
                self._detected_s = None
                return None
            if self._detected_s is None:
                self._detected_s = now_s
            if not failed and self._awaits_more_joins(now_s, reports, ready, starting):
                return None
            self._requested -= ready
            self._joining |= ready
            if ready:
                logger.info(f"Adding joining ranks {sorted(ready)} to the membership")
            return self._propose((self.record.active_mask & ~failed) | self._mask(ready))
        if failed:
            return self._propose(self.record.active_mask & ~failed)
        active = self._ranks(self.record.active_mask)
        expected = Reply.READY if state == MembershipState.PROPOSE else Reply.INSTALLED

        def replied(r: int, reply: Reply) -> bool:
            report = reports.get(r)
            # A joining rank reports its readiness before the PROPOSE, which stands for its READY.
            ready_to_join = r in self._joining and report is not None and report.joining_ready
            return (reply == Reply.READY and ready_to_join) or self._replied(report, reply)

        failed_replies = {r for r in active if replied(r, Reply.FAILED)}
        if failed_replies - self._joining:
            return self.abort(
                f"in {state.name}, ranks {sorted(failed_replies - self._joining)} replied FAILED"
            )
        if failed_replies:
            self._joining -= failed_replies
            return self._propose(
                self.record.active_mask & ~self._mask(failed_replies),
                reason="a replacement could not install the change that adds it",
            )
        if not all(replied(r, expected) for r in active):
            return None
        if state == MembershipState.PROPOSE:
            survivors = [
                r for r in active if self._committed_mask >> r & 1 and r not in self._joining
            ]
            epochs = [reports[r].paused_at.epoch for r in survivors]
            replaced = self._replaced_install_epoch
            # Every round waits for every active rank. If a survivor stopped before the epoch of a replaced INSTALL, no
            # rank ran a round from that epoch, and the ranks that installed it install the same epoch again.
            if replaced and any(latest_epoch([replaced, epoch]) != epoch for epoch in epochs):
                epoch = replaced
            else:
                # Epochs wrap, so the resume epoch follows the latest one, not the largest.
                epoch = next_epoch(latest_epoch(epochs))
            resume = ResumeState(
                iteration=max(
                    [self.record.resume.iteration]
                    + [reports[r].paused_at.iteration for r in survivors]
                ),
                epoch=epoch,
            )
            self._ready_s = now_s
            payloads = tuple(
                (r, reports[r].ready_payload)
                for r in active
                if reports[r].ready_payload is not None
            )
            return self._set(
                MembershipState.INSTALL, self.record.active_mask, resume, ready_payloads=payloads
            )
        joining = self.record.joining_mask
        digests = {reports[r].placement_digest for r in active if not joining >> r & 1}
        if len(digests) > 1:
            return self.abort("the surviving ranks restored different expert placements")
        mismatched = {r for r in self._ranks(joining) if reports[r].placement_digest not in digests}
        if mismatched:
            logger.warning(
                f"Joining ranks {sorted(mismatched)} restored a different expert placement than the "
                "surviving ranks, so they are not added"
            )
            self._joining -= mismatched
            return self._propose(
                self.record.active_mask & ~self._mask(mismatched),
                reason="a replacement restored a different expert placement than the surviving ranks",
            )
        self._joining -= set(active)
        self._replaced_install_epoch = 0
        for r in self._ranks(self._committed_mask & ~self.record.active_mask):
            self._failed_in[r] = self.record.generation
        for r in active:
            self._failed_in.pop(r, None)
        self._committed_mask = self.record.active_mask
        # A replacement's progress clock must not continue from the process it replaces.
        self._progress.clear()
        self._hung_since.clear()
        return self._set(MembershipState.COMMITTED, self.record.active_mask)

    def _form(
        self, now_s: float, reports: Dict[int, RankReport], connected: int
    ) -> Optional[MembershipRecord]:
        if self._rendezvous_began_s is None:
            self._rendezvous_began_s = now_s
        everyone = self.record.active_mask
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
        logger.info("Every rank of the deployment started")
        self._progress.clear()
        self._hung_since.clear()
        return self._set(MembershipState.COMMITTED, everyone)

    def _awaits_more_joins(
        self, now_s: float, reports: Dict[int, RankReport], ready: Set[int], starting: Set[int]
    ) -> bool:
        """Whether ``ready`` should wait for a ``starting`` replacement of a rank that failed in the same change."""
        failed_with = {self._failed_in[r] for r in ready if r in self._failed_in}
        return any(
            self._failed_in.get(r) in failed_with and self._reports_advance(r, reports, now_s)
            for r in starting - ready
        )

    def _starting(self, r: int, reports: Dict[int, RankReport], now_s: float) -> bool:
        report = reports.get(r, RankReport())
        progress = (report.incarnation, report.progress)
        if self._startup_progress.get(r, (None,))[0] != progress:
            self._startup_progress[r] = (progress, now_s)
        return now_s - self._startup_progress[r][1] <= self.rendezvous_timeout_s

    def _reports_advance(self, r: int, reports: Dict[int, RankReport], now_s: float) -> bool:
        report = reports.get(r, RankReport())
        sequence = (report.incarnation, report.sequence)
        if self._report_sequence.get(r, (None,))[0] != sequence:
            self._report_sequence[r] = (sequence, now_s)
        return now_s - self._report_sequence[r][1] <= self.hang_timeout_s / 2

    def _propose(self, active_mask: int, reason: str = "") -> MembershipRecord:
        self._ready_s = None
        if active_mask & 1 == 0:
            logger.error("Deployment aborted: rank 0 failed")
            return self._set(MembershipState.ABORTED, active_mask, reason="rank 0 failed")
        if self.record.state == MembershipState.INSTALL:
            self._replaced_install_epoch = self.record.resume.epoch
        return self._set(
            MembershipState.PROPOSE,
            active_mask,
            generation=self.record.generation + 1,
            reason=reason,
        )

    def abort(self, reason: str) -> MembershipRecord:
        logger.error(f"Deployment aborted: {reason}")
        return self._set(MembershipState.ABORTED, self.record.active_mask, reason=reason)

    def _set(
        self,
        state: MembershipState,
        active_mask: int,
        resume: Optional[ResumeState] = None,
        generation: Optional[int] = None,
        ready_payloads: Tuple[Tuple[int, str], ...] = (),
        reason: str = "",
    ) -> MembershipRecord:
        self.record = MembershipRecord(
            generation=self.record.generation if generation is None else generation,
            state=state,
            active_mask=active_mask,
            resume=self.record.resume if resume is None else resume,
            joining_mask=self._mask(self._joining) & active_mask,
            ready_payloads=ready_payloads,
            reason=reason,
        )
        return self.record

    def _replied(self, report: Optional[RankReport], reply: Reply) -> bool:
        return (
            report is not None
            and report.reply_generation == self.record.generation
            and report.reply == reply
        )

    def _failed(
        self, now_s: float, reports: Dict[int, RankReport], connected: int, timeout_s: float
    ) -> int:
        """Mask of the active ranks that are dead, or hung for ``timeout_s``."""
        state, active_mask = self.record.state, self.record.active_mask
        active = self._ranks(active_mask)

        def advance(seen: Dict[int, tuple], r: int, value) -> None:
            if seen.get(r, (None,))[0] != value:
                seen[r] = (value, now_s)

        expected = {MembershipState.PROPOSE: Reply.READY, MembershipState.INSTALL: Reply.INSTALLED}
        waited_on = 0
        if state in expected:
            waited_on = self._mask(
                r for r in active if not self._replied(reports.get(r), expected[state])
            )
        # A rank's waits count only while its reports advance.
        waits: Dict[int, int] = {}
        rounds: Dict[int, Set[tuple]] = {}
        for r in active:
            report = reports.get(r, RankReport())
            if self._reports_advance(r, reports, now_s):
                waits[r] = report.waiting_for & active_mask & ~(1 << r)
                waited_on |= waits[r]
            for peer, round_flag in report.waited_rounds:
                rounds.setdefault(peer, set()).add(("all-to-all", round_flag))
            for peer, epoch in report.waited_epochs:
                rounds.setdefault(peer, set()).add(("host exchange", epoch))
        failed = active_mask & ~connected
        for r in active:
            report = reports.get(r, RankReport())
            advance(
                self._progress,
                r,
                (report.progress, report.install_steps, frozenset(rounds.get(r, ()))),
            )
            idle = r == 0 and state == MembershipState.COMMITTED and r in waits and report.idle
            if waited_on >> r & 1 and not idle and _chain_end(r, waits):
                since_s = max(self._hung_since.setdefault(r, now_s), self._progress[r][1])
                if now_s - since_s > timeout_s:
                    failed |= 1 << r
                    logger.warning(
                        f"Rank {r} is hung. It made no progress for {now_s - since_s:.1f} s while active ranks "
                        f"wait for it. Last reports: {self._describe(active, reports, waits, now_s)}"
                    )
            else:
                self._hung_since.pop(r, None)
        return failed

    def _describe(
        self, active: List[int], reports: Dict[int, RankReport], waits: Dict[int, int], now_s: float
    ) -> str:
        described = []
        for r in active:
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


class RendezvousStore:
    """A torch ``TCPStore`` whose values carry an HMAC-SHA256 tag under the shared secret."""

    def __init__(self, store, secret: bytes):
        self.store = store
        self._secret = secret

    def set(self, key: str, value: bytes) -> None:
        self.store.set(key, self.tag(key.encode(), value) + value)

    def get(self, key: str) -> bytes:
        """Return the value of ``key`` once it is set, or raise PermissionError if it is not signed."""
        value = self._verified(key, self.store.get(key))
        if value is None:
            raise PermissionError(f"rendezvous value {key!r} is not signed with the shared secret")
        return value

    def get_if_set(self, key: str) -> Optional[bytes]:
        """Return the value of ``key``, or None if it is not set or not signed."""
        return self._verified(key, self.store.get(key)) if self.store.check([key]) else None

    def multi_get(self, keys: List[str]) -> List[Optional[bytes]]:
        """Return the values of ``keys``, which must all be set, with None for an unsigned value."""
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


LIVENESS_KEY = "liveness"
RECORD_KEY = "record"
TRANSACTION_KEY = "last_transaction"
# Per rank, the newest incarnation whose liveness connection closed after those of every older one.
DISCONNECTED_KEY = "disconnected"
HOST_WEIGHTS_PREFIX_KEY = "host_weights_prefix"


# Ranks per store counter of an arrival mask. The store's counters are signed 64-bit integers.
_RANKS_PER_COUNTER = 63


def authority_ports(address: str) -> Tuple[str, int, int]:
    """Return the host, store port and liveness port of the authority at ``address``."""
    host, port = address.rsplit(":", 1)
    return host, int(port), int(port) + 1


def open_store(address: str, secret: bytes, timeout_s: float) -> RendezvousStore:
    """Open a client of the store the authority hosts at ``host:port``."""
    from datetime import timedelta

    from torch.distributed import TCPStore

    host, port, _ = authority_ports(address)
    store = TCPStore(
        host,
        port,
        is_master=False,
        timeout=timedelta(seconds=timeout_s),
        wait_for_workers=False,
    )
    return RendezvousStore(store, secret)


def rendezvous_store(timeout_s: float = 60.0) -> RendezvousStore:
    """Open a client of the rendezvous store from this process's environment."""
    return open_store(
        os.environ[RENDEZVOUS_ENV], bytes.fromhex(os.environ[SHARED_SECRET_ENV]), timeout_s
    )


_FIRST_POLL_S = 0.001
STORE_POLL_S = 0.1


def wait_for_keys(
    store: RendezvousStore,
    keys: List[str],
    poll_s: float,
    check: Callable[[Set[str]], None],
) -> List[bytes]:
    """Return the values of ``keys`` once all are set and signed, calling ``check`` after every poll.

    A blocking store wait would log a warning at every timeout, so this polls with a backoff up to ``poll_s``.
    """
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
    """A communicator of the global ranks ``ranks`` over the rendezvous store, on the pattern of ``StoreDwdpGroup``.

    Only the members in ``active`` take part, and the others' values are None. While a call waits, ``on_wait`` gets the
    mask of the active members whose values are missing, and may raise to stop waiting. Given ``gather``, the group
    exchanges through it instead of the store. The exchanges of a ``startup`` group are startup collectives, which
    rank 0 records and a replacement replays.
    """

    def __init__(
        self,
        store: Optional[RendezvousStore],
        name: str,
        rank: int,
        ranks: Sequence[int],
        on_wait: Callable[[int], None] = lambda waiting: None,
        active: Optional[Sequence[int]] = None,
        gpus_per_node: int = 1,
        gather: Optional[Callable[[Any], Sequence[Any]]] = None,
        startup: bool = False,
    ):
        self.membership = tuple(ranks)
        self.rank = rank
        self.exchange_failed = False
        self._store = store
        self._name = name
        self._on_wait = on_wait
        self._active = tuple(r for r in self.membership if active is None or r in active)
        self._gpus_per_node = gpus_per_node
        self._gather = gather
        self._startup = startup
        self._calls = 0
        self._subgroups = 0

    def Get_rank(self) -> int:
        return self.membership.index(self.rank)

    def Get_size(self) -> int:
        return len(self.membership)

    def allgather(self, value: Any, bound: Optional[Callable[[Any, Any], Any]] = None) -> List[Any]:
        """Return every member's value by communicator rank, with None for a member that is not active.

        ``bound`` is ``min`` or ``max`` for a reduction by it, else None.
        """
        try:
            return self._exchange(value, bound)
        except BaseException:
            self.exchange_failed = True
            raise

    def barrier(self) -> None:
        self.allgather(None)

    Barrier = barrier

    def bcast(self, obj: Any, root: int = 0) -> Any:
        """Return the ``obj`` of communicator rank ``root``, or None if it is not active."""
        return self.allgather(obj)[root]

    def allreduce(self, obj: Any, op: Any = None) -> Any:
        """Reduce the active members' ``obj`` by ``MPI.SUM`` (the default), ``MPI.MIN`` or ``MPI.MAX``."""
        from mpi4py import MPI

        op = MPI.SUM if op is None else op
        for mpi_op, reduce in ((MPI.SUM, sum), (MPI.MIN, min), (MPI.MAX, max)):
            if op == mpi_op:
                values = self.allgather(obj, None if reduce is sum else reduce)
                return reduce(value for value in values if value is not None)
        raise NotImplementedError(f"StoreGroup reduces by MPI.SUM, MPI.MIN or MPI.MAX, not {op}")

    def Free(self) -> None:
        """Do nothing, since the group's exchanges share this process's store client."""

    def subgroup(self, ranks: Sequence[int]) -> "StoreGroup":
        """Return the group of ``ranks``, members of this group that include this rank."""
        ranks = tuple(ranks)
        digest = hashlib.sha256(repr(ranks).encode()).hexdigest()[:16]
        self._subgroups += 1
        return StoreGroup(
            self._store,
            f"{self._name}/{self._subgroups}-{digest}",
            self.rank,
            ranks,
            self._on_wait,
            self._active,
            self._gpus_per_node,
        )

    def Split(self, color: int = 0, key: int = 0) -> "StoreGroup":
        """Split as ``MPI.Comm.Split`` does, with a replacement taking the colors from the startup record."""
        from .startup_record import startup_record

        splits = startup_record.agree("startup_group_split", lambda: self.allgather((color, key)))
        ranks = sorted(
            (split[1], index, rank)
            for index, (rank, split) in enumerate(zip(self.membership, splits))
            if split[0] == color
        )
        return self.subgroup(rank for _, _, rank in ranks)

    def Split_type(self, split_type: int, key: int = 0, info: Any = None) -> "StoreGroup":
        """Return the members on this rank's node, supporting only the node split types."""
        from mpi4py import MPI

        from tensorrt_llm._utils import OMPI_COMM_TYPE_HOST

        if split_type not in (MPI.COMM_TYPE_SHARED, OMPI_COMM_TYPE_HOST):
            raise NotImplementedError(
                f"StoreGroup splits by node only, not by split type {split_type}"
            )
        node = self.rank // self._gpus_per_node
        return self.subgroup(r for r in self.membership if r // self._gpus_per_node == node)

    def _exchange(self, value: Any, bound: Optional[Callable[[Any, Any], Any]]) -> List[Any]:
        if self._gather is not None:
            values = self._gather(value)
            return [values[r] for r in self.membership]
        from .startup_record import startup_record

        if self._active == (self.rank,):
            self._on_wait(0)
        payload = pickle.dumps(value)
        if self._startup and startup_record.active:
            payloads = startup_record.exchange(
                self.Get_rank(), payload, None, lambda: self._store_exchange(payload), bound
            )
        else:
            payloads = self._store_exchange(payload)
        # Only processes with the shared secret write the store's values (RendezvousStore).
        return [None if p is None else pickle.loads(p) for p in payloads]  # nosec B301

    def _store_exchange(self, payload: bytes) -> List[Optional[bytes]]:
        if self._active == (self.rank,):
            return [payload if r == self.rank else None for r in self.membership]
        from torch.distributed import DistError

        prefix = f"{self._name}/{self._calls}"
        self._calls += 1
        try:
            self._store.set(f"{prefix}/{self.rank}", payload)
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
                if self.rank == min(self._active):
                    for key in self._counters(previous):
                        self._store.store.delete_key(key)
        except DistError as error:
            raise RuntimeError(
                f"Rank {self.rank} lost the rendezvous store, which the membership authority hosts"
            ) from error
        by_rank = dict(zip(self._active, values))
        return [by_rank.get(r) for r in self.membership]

    def _wait(self, prefix: str) -> List[bytes]:
        keys = [f"{prefix}/{r}" for r in self._active]
        active_mask = sum(1 << r for r in self._active)
        read_s, missing = None, 0

        def check(unsigned: Set[str]) -> None:
            nonlocal read_s, missing
            if read_s is None or time.monotonic() - read_s >= REPORT_INTERVAL_S:
                read_s = time.monotonic()
                arrived = sum(
                    self._store.store.add(key, 0) << int(key.rsplit("/", 1)[1]) * _RANKS_PER_COUNTER
                    for key in self._counters(prefix)
                )
                missing = active_mask & ~arrived
            self._on_wait(
                missing | sum(1 << r for r, key in zip(self._active, keys) if key in unsigned)
            )

        return wait_for_keys(self._store, keys, STORE_POLL_S, check)

    def _counters(self, prefix: str) -> List[str]:
        return sorted({self._counter(prefix, r) for r in self._active})

    @staticmethod
    def _counter(prefix: str, rank: int) -> str:
        return f"{prefix}/arrived/{rank // _RANKS_PER_COUNTER}"


def node_supervisor_address() -> bytes:
    """Return the abstract Unix socket that keeps a second node supervisor of the deployment off the node."""
    digest = hashlib.sha256(bytes.fromhex(os.environ[SHARED_SECRET_ENV])).hexdigest()[:32]
    return f"\0tllm-wide-ep-ft-{digest}".encode()


# The exit status of a replacement whose startup failure any other replacement would repeat.
REPLACEMENT_CANNOT_START = 78


class ReplacementCannotStart(SystemExit):
    """A ``SystemExit`` that ends a replacement rank with :data:`REPLACEMENT_CANNOT_START` and a reason."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.code = REPLACEMENT_CANNOT_START


# The hello is tagged under the shared secret and the authority's nonce, so a recorded hello opens no other connection.
HELLO_TIMEOUT_S = 10.0
_NONCE_BYTES = 16
_MAX_LINE_BYTES = 1 << 20
_MAX_UNSENT_BYTES = 1 << 22
# The selector data of the descriptor whose end of file tells the authority that rank 0's process ended.
_PARENT = object()


def _line(message: _Message) -> bytes:
    return message.model_dump_json().encode() + b"\n"


def _decided_at_once(report: RankReport) -> tuple:
    """Return the fields of ``report`` the authority decides on as soon as they change."""
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
    """A process's liveness connection to the authority, closed at the end of a ``with`` block."""

    sock: socket.socket
    lines: _Lines

    def __enter__(self) -> "LivenessConnection":
        return self

    def __exit__(self, *exc_info) -> None:
        self.sock.close()


def _open_connection(
    store: RendezvousStore, announcement: str, timeout_s: float
) -> LivenessConnection:
    host, port = store.get(LIVENESS_KEY).decode().rsplit(":", 1)
    sock = socket.create_connection((host, int(port)), timeout=timeout_s)
    authority_nonce = bytes.fromhex(_read_line(sock).decode())
    hello = announcement.encode()
    sock.sendall(store.tag(b"hello", authority_nonce + hello).hex().encode() + b" " + hello + b"\n")
    sock.settimeout(None)
    return LivenessConnection(sock, _Lines())


def open_liveness_connection(
    store: RendezvousStore, rank: int, replacement: bool = False, timeout_s: float = HELLO_TIMEOUT_S
) -> LivenessConnection:
    """Open this process's liveness connection to the authority for ``rank``."""
    connection = _open_connection(store, f"{rank} {int(replacement)}", timeout_s)
    # A forked child must not keep the connection open.
    os.register_at_fork(after_in_child=connection.sock.close)
    return connection


def open_node_supervisor_connection(
    store: RendezvousStore,
    node: int,
    ranks: List[int],
    host: str,
    timeout_s: float = HELLO_TIMEOUT_S,
) -> LivenessConnection:
    """Open the connection of the supervisor of ``node``, whose close fails ``ranks``."""
    return _open_connection(store, f"node {node} {','.join(map(str, ranks))} {host}", timeout_s)


class MembershipClient:
    """A rank's thread that keeps its liveness connection, delivers every record and sends the rank's reports."""

    def __init__(
        self,
        rank: int,
        store: RendezvousStore,
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
        self.incarnation: Optional[int] = None
        self.connected = False
        self._lock = threading.Lock()
        self._delivery = threading.Lock()
        self._stop = threading.Event()
        self._wake_read, self._wake_write = os.pipe()
        os.set_blocking(self._wake_read, False)
        os.set_blocking(self._wake_write, False)
        self._thread = threading.Thread(
            target=self._run, name=f"MembershipClient{rank}", daemon=True
        )
        self._sock: Optional[socket.socket] = None
        self._lines: Optional[_Lines] = None
        self._received: List[bytes] = []

    def start(self) -> None:
        """Connect, take this process's incarnation from the authority's first record and start the thread."""
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
        self._send_report()
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
        """Replace the callbacks, delivering the newest record to the new ``on_record`` at once."""
        with self._delivery:
            self._on_record, self._on_tick, self._on_error = on_record, on_tick, on_error
            if self.record is not None:
                on_record(self.record)

    def report(self, **changes) -> None:
        """Update fields of the report and wake the thread to send it."""
        with self._lock:
            self._report = self._report.model_copy(update=changes)
        self._wake()

    def _wake(self) -> None:
        with self._lock:
            if self._wake_write is None:
                return
            try:
                os.write(self._wake_write, b"\0")
            except BlockingIOError:
                pass

    def _send_report(self) -> None:
        with self._lock:
            self._report = self._report.model_copy(update={"sequence": self._report.sequence + 1})
            report = self._report
        self._sock.sendall(_line(report))

    def _run(self) -> None:
        listed = False
        tick_s = time.monotonic()
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
                        return
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
        except Exception as error:
            logger.error(f"Membership client of rank {self.rank} failed: {error!r}")
            self.connected = False
            self._sock.close()
            if self._on_error is not None:
                self._on_error(error)


@dataclass
class _Connection:
    """A liveness connection the authority accepted."""

    nonce: bytes
    accepted_s: float
    received: bytes = b""
    member: Optional[Tuple[int, int]] = None  # (rank, incarnation)
    node: Optional[int] = None
    node_ranks: Tuple[int, ...] = ()
    lines: Optional[_Lines] = None
    report: Optional[RankReport] = None
    unsent: bytes = b""


class AuthorityProcess:
    """Run :class:`Authority` and publish every record in the store and over every liveness connection.

    A replacement is admitted only while the rank's current process is not connected, so only the authority's
    verdict removes an active process. The deployment aborts once ``parent``, held by rank 0's process, ends.
    """

    def __init__(
        self,
        world_size: int,
        store: RendezvousStore,
        authority: Authority,
        parent: Optional[int] = None,
    ):
        self.world_size = world_size
        self._store = store
        self._authority = authority
        self._parent = parent
        self._parent_ended_s: Optional[float] = None
        self._changed = False
        self._incarnation = [0] * world_size
        self._disconnected = [-1] * world_size
        self._connections: Dict[socket.socket, _Connection] = {}
        self._selector = selectors.DefaultSelector()
        self._record: Optional[MembershipRecord] = None
        self._hosts: Dict[int, str] = {}

    def run(self, host: str = "", port: int = 0) -> MembershipRecord:
        """Serve until the record is ABORT or SHUTDOWN and return the last record."""
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
            if record.state in (MembershipState.ABORTED, MembershipState.SHUTDOWN):
                # The store lives in this process, so keep it until every active rank has read the record and left.
                while self._connected() & record.active_mask and (
                    self._parent_ended_s is None
                    or time.monotonic() - self._parent_ended_s < self._authority.hang_timeout_s
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
            logger.warning(f"Closing the connection of {sender}, which reads nothing")
            self._drop(sock)
            return
        write = selectors.EVENT_WRITE if connection.unsent else 0
        self._selector.modify(sock, selectors.EVENT_READ | write, connection)

    def _poll(self, timeout_s: float) -> None:
        for key, events in self._selector.select(timeout_s):
            sock, connection = key.fileobj, key.data
            if connection is None:
                self._accept(sock)
                continue
            if connection is _PARENT:
                self._selector.unregister(sock)
                self._parent_ended_s = time.monotonic()
                self._changed = True
                continue
            if sock not in self._connections:
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
                logger.warning("Closing a liveness connection without a valid hello")
                self._drop(sock)

    def _accept(self, listener: socket.socket) -> None:
        """Accept a connection, dropping the oldest of ``MAX_RANKS`` that wait for their hello."""
        waiting = [sock for sock, c in self._connections.items() if c.lines is None]
        if len(waiting) >= MAX_RANKS:
            logger.warning("Closing the oldest liveness connection waiting for its hello")
            self._drop(waiting.pop(0))
        try:
            sock, _ = listener.accept()
        except BlockingIOError:
            return
        except OSError as error:  # out of descriptors (EMFILE, ENFILE) or memory
            logger.warning(f"Cannot accept a liveness connection: {error!r}")
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
            logger.warning(f"Closing a liveness connection: {error}")
            self._drop(sock)

    def _read_hello(self, sock: socket.socket, connection: _Connection, data: bytes) -> bytes:
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
            raise ValueError("a hello not signed with the shared secret")
        announcement = hello.decode().split()
        if announcement[:1] == ["node"]:
            _, node, ranks, host = announcement
            node, ranks = int(node), tuple(int(rank) for rank in ranks.split(","))
            refusal = self._node_refusal(node, ranks)
            if refusal is not None:
                raise ValueError(refusal)
            connection.node, connection.node_ranks, connection.lines = node, ranks, _Lines()
            self._hosts[node] = host
            logger.info(f"The supervisor of node {node} ({host}, ranks {list(ranks)}) connected")
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
        connection.unsent += _line(
            self._record.model_copy(update={"incarnations": tuple(self._incarnation)})
        )
        self._flush(sock, connection)
        return rest

    def _node_refusal(self, node: int, ranks: Tuple[int, ...]) -> Optional[str]:
        if not ranks or not all(0 <= rank < self.world_size for rank in ranks):
            return f"node {node}'s ranks {list(ranks)} are not all in the deployment"
        for c in self._connections.values():
            if c.node == node or set(c.node_ranks) & set(ranks):
                return f"node {node} or one of its ranks {list(ranks)} has a supervisor already"
        return None

    def _refusal(self, rank: int, replacement: int) -> Optional[str]:
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
        if self._authority.record.is_active(rank) and self._connected() >> rank & 1:
            return f"rank {rank} is active: no replacement joins for it"
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
        if self._record.state not in (MembershipState.ABORTED, MembershipState.SHUTDOWN):
            logger.warning(
                f"The supervisor of node {connection.node} left, so its ranks "
                f"{list(connection.node_ranks)} count as failed"
            )
            for other, peer in list(self._connections.items()):
                if peer.member is not None and peer.member[0] in connection.node_ranks:
                    self._drop(other)
        self._send_record(ranks=False)

    def _publish_disconnected(self, rank: int) -> None:
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
        return {
            c.member[0]: c.report
            for c in self._connections.values()
            if c.member is not None
            and c.report is not None
            and c.member[1] == self._incarnation[c.member[0]]
        }


def main(argv: Optional[List[str]] = None) -> int:
    """Run the authority process, which aborts the deployment once its standard input, held by rank 0, closes."""
    parser = argparse.ArgumentParser(description="Wide-EP fault-tolerance membership authority")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--address-fd", type=int, default=1)
    parser.add_argument("--rendezvous-timeout-s", type=float, default=RENDEZVOUS_TIMEOUT_S)
    args = parser.parse_args(argv)

    from torch.distributed import TCPStore

    address = os.environ[RENDEZVOUS_ENV]
    host, store_port, liveness_port = authority_ports(address)
    tcp_store = TCPStore(host, store_port, is_master=True, wait_for_workers=False)
    store = RendezvousStore(tcp_store, bytes.fromhex(os.environ[SHARED_SECRET_ENV]))
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

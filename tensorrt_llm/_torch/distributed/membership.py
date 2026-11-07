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

The authority is the one process that decides membership. It reads every rank's report (a heartbeat, the step the
rank is on, the peers it waits for, and its replies), and writes one :class:`MembershipRecord` that every rank reads:
the generation, the transaction state, the live ranks and the state every rank resumes from. A change of membership
is a transaction of two rounds: ``PROPOSE`` (ranks stop and reply ``READY`` with where they stopped), ``INSTALL``
(ranks switch to the new membership and reply ``INSTALLED``), then ``COMMITTED``. ``ABORT`` ends the deployment when
recovery cannot succeed.

The decision logic (:class:`Authority`) reads and writes no memory, so it is tested by itself. The transport below
carries reports and records through GPU memory shared over NVLink: every rank owns a control row the authority
writes the record into, and the authority owns an inbox every rank writes its report into, each record and report
behind a seqlock (a version word that is odd while a write is in progress). A rendezvous store hosted by the
authority, its values signed with the join secret, exchanges the memory handles.
"""

import enum
import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import numpy as np

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger

# Words of the live mask; the same layout as EPGroupHealth.get_mask_words() and the all-to-all kernels' rank mask.
MASK_WORDS = 4
MAX_RANKS = 64 * MASK_WORDS


class MembershipState(enum.IntEnum):
    COMMITTED = 0  # serving with the generation's live ranks
    PROPOSE = 1  # ranks stop at their next wait and reply READY
    INSTALL = 2  # ranks switch to the new membership and reply INSTALLED
    ABORT = 3  # recovery cannot succeed: every rank takes the terminal path
    SHUTDOWN = 4  # a normal exit: a missing heartbeat is not a failure


class Reply(enum.IntEnum):
    NONE = 0
    READY = 1
    INSTALLED = 2


@dataclass(frozen=True)
class ResumeState:
    """Where every rank continues after a membership change; the authority sets it from the READY replies."""

    iteration: int = 0  # the executor's step counter (main gates some collectives on it)
    epoch: int = 0  # the next host exchange epoch
    alltoall_flag: int = 0  # the next all-to-all round flag value
    request_broadcast: int = 0  # the last request broadcast every rank must have applied
    eplb_round: int = 0  # the EPLB statistics round

    def at_least(self, other: "ResumeState") -> "ResumeState":
        return ResumeState(*(max(a, b) for a, b in zip(self.as_tuple(), other.as_tuple())))

    def as_tuple(self) -> tuple:
        return (
            self.iteration,
            self.epoch,
            self.alltoall_flag,
            self.request_broadcast,
            self.eplb_round,
        )


@dataclass(frozen=True)
class MembershipRecord:
    generation: int
    state: MembershipState
    live_mask: int
    resume: ResumeState = field(default_factory=ResumeState)

    WORDS = 2 + MASK_WORDS + 5

    def is_live(self, rank: int) -> bool:
        return bool(self.live_mask >> rank & 1)

    def to_words(self) -> np.ndarray:
        mask_words = [(self.live_mask >> (64 * i)) & (2**64 - 1) for i in range(MASK_WORDS)]
        return np.array(
            [self.generation, int(self.state), *mask_words, *self.resume.as_tuple()],
            dtype=np.uint64,
        )

    @classmethod
    def from_words(cls, words: np.ndarray) -> "MembershipRecord":
        words = [int(w) for w in words[: cls.WORDS]]
        mask = sum(w << (64 * i) for i, w in enumerate(words[2 : 2 + MASK_WORDS]))
        return cls(words[0], MembershipState(words[1]), mask, ResumeState(*words[2 + MASK_WORDS :]))


@dataclass(frozen=True)
class RankReport:
    """What one rank last wrote for the authority."""

    heartbeat: int = (
        0  # bumped by the rank's failure-detector thread while the process and its GPU work
    )
    step: int = 0  # the executor step the rank is on
    waiting_for: int = 0  # mask of the ranks whose message the rank waits for, 0 when not waiting
    reply_generation: int = 0
    reply: Reply = Reply.NONE
    stopped_at: ResumeState = field(
        default_factory=ResumeState
    )  # with READY: where the rank stopped
    incarnation: int = 0  # 0 for the original process of the rank, n for its n-th replacement

    WORDS = 2 + MASK_WORDS + 2 + 5 + 1

    def to_words(self) -> np.ndarray:
        waiting = [(self.waiting_for >> (64 * i)) & (2**64 - 1) for i in range(MASK_WORDS)]
        return np.array(
            [
                self.heartbeat,
                self.step,
                *waiting,
                self.reply_generation,
                int(self.reply),
                *self.stopped_at.as_tuple(),
                self.incarnation,
            ],
            dtype=np.uint64,
        )

    @classmethod
    def from_words(cls, words: np.ndarray) -> "RankReport":
        words = [int(w) for w in words[: cls.WORDS]]
        waiting = sum(w << (64 * i) for i, w in enumerate(words[2 : 2 + MASK_WORDS]))
        rest = words[2 + MASK_WORDS :]
        return cls(
            words[0], words[1], waiting, rest[0], Reply(rest[1]), ResumeState(*rest[2:7]), rest[7]
        )


@dataclass
class _Liveness:
    heartbeat: int = 0
    heartbeat_s: float = 0.0
    step: int = 0
    step_s: float = 0.0


class Authority:
    """Decides membership from the ranks' reports; :meth:`tick` returns a record to publish when it changes.

    A rank is dead when its heartbeat stops for ``liveness_timeout_s``, and hung when it is alive but its step does not
    advance for ``progress_timeout_s`` while a live peer waits for it. ``covers(live_mask)`` tells whether the ranks in
    the mask still hold every expert; a membership that does not is aborted, never served. A transaction that has not
    committed ``deadline_s`` after it started is aborted. A failure during a transaction restarts it with the larger
    set of failed ranks; a replacement is added once it asks to join (:meth:`request_join`).
    """

    def __init__(
        self,
        world_size: int,
        covers: Callable[[int], bool],
        liveness_timeout_s: float = 1.0,
        progress_timeout_s: float = 30.0,
        deadline_s: float = 60.0,
    ):
        if not 0 < world_size <= MAX_RANKS:
            raise ValueError(f"world_size must be in (0, {MAX_RANKS}], got {world_size}")
        self.world_size = world_size
        self._covers = covers
        self.liveness_timeout_s = liveness_timeout_s
        self.progress_timeout_s = progress_timeout_s
        self.deadline_s = deadline_s
        self.record = MembershipRecord(0, MembershipState.COMMITTED, (1 << world_size) - 1)
        self._liveness: Dict[int, _Liveness] = {}
        self._joining: Set[int] = set()
        self._started_s: Optional[float] = None

    def request_join(self, rank: int) -> None:
        """A replacement for ``rank`` mapped the memory and asks to become live. If ``rank`` is still live (its old
        process was not yet found failed), the replacement takes it over in the next transaction."""
        self._joining.add(rank)

    def shutdown(self) -> MembershipRecord:
        self.record = MembershipRecord(
            self.record.generation, MembershipState.SHUTDOWN, self.record.live_mask
        )
        return self.record

    def tick(self, now_s: float, reports: Dict[int, RankReport]) -> Optional[MembershipRecord]:
        state = self.record.state
        if state in (MembershipState.ABORT, MembershipState.SHUTDOWN):
            return None
        failed = self._failed(now_s, reports)
        self._joining -= set(
            self._ranks(failed)
        )  # a replacement that died while joining is dropped
        if state == MembershipState.COMMITTED:
            if not failed and not self._joining:
                return None
            return self._propose(
                now_s, (self.record.live_mask & ~failed) | self._mask(self._joining)
            )
        if failed:  # a failure during the transaction: restart it without those ranks too
            return self._propose(now_s, self.record.live_mask & ~failed)
        if now_s - self._started_s > self.deadline_s:
            return self._set(MembershipState.ABORT, self.record.live_mask)
        expected = Reply.READY if state == MembershipState.PROPOSE else Reply.INSTALLED
        live = self._ranks(self.record.live_mask)
        if not all(
            r in reports
            and reports[r].reply_generation == self.record.generation
            and reports[r].reply == expected
            for r in live
        ):
            return None
        if state == MembershipState.PROPOSE:
            resume = self.record.resume
            for r in live:
                resume = resume.at_least(reports[r].stopped_at)
            # Ranks continue one past anything a survivor already used.
            resume = ResumeState(
                resume.iteration,
                resume.epoch + 1,
                resume.alltoall_flag + 1,
                resume.request_broadcast,
                resume.eplb_round,
            )
            return self._set(MembershipState.INSTALL, self.record.live_mask, resume)
        self._joining -= set(live)
        self._started_s = None
        return self._set(MembershipState.COMMITTED, self.record.live_mask, self.record.resume)

    def _propose(self, now_s: float, live_mask: int) -> MembershipRecord:
        self._started_s = now_s
        if live_mask & 1 == 0 or not self._covers(live_mask):
            return self._set(MembershipState.ABORT, live_mask)
        for (
            r
        ) in self._joining:  # a replacement's clocks start now, not from the process it replaces
            self._liveness[r] = _Liveness(heartbeat_s=now_s, step_s=now_s)
        return self._set(MembershipState.PROPOSE, live_mask, generation=self.record.generation + 1)

    def _set(
        self,
        state: MembershipState,
        live_mask: int,
        resume: Optional[ResumeState] = None,
        generation: Optional[int] = None,
    ) -> MembershipRecord:
        self.record = MembershipRecord(
            self.record.generation if generation is None else generation,
            state,
            live_mask,
            self.record.resume if resume is None else resume,
        )
        return self.record

    def _failed(self, now_s: float, reports: Dict[int, RankReport]) -> int:
        """Mask of live ranks that are dead or hung at ``now_s``."""
        live = self._ranks(self.record.live_mask)
        failed = 0
        waited_on = 0
        for r in live:
            report = reports.get(r, RankReport())
            seen = self._liveness.setdefault(
                r, _Liveness(report.heartbeat, now_s, report.step, now_s)
            )
            if report.heartbeat != seen.heartbeat:
                seen.heartbeat, seen.heartbeat_s = report.heartbeat, now_s
            if report.step != seen.step:
                seen.step, seen.step_s = report.step, now_s
            if now_s - seen.heartbeat_s > self.liveness_timeout_s:
                failed |= 1 << r
            waited_on |= report.waiting_for
        for r in live:
            if waited_on >> r & 1 and now_s - self._liveness[r].step_s > self.progress_timeout_s:
                failed |= 1 << r
        return failed

    def _ranks(self, mask: int) -> list:
        return [r for r in range(self.world_size) if mask >> r & 1]

    @staticmethod
    def _mask(ranks) -> int:
        return sum(1 << r for r in ranks)


# A seqlock slot: the version word, then the payload words; 64-byte aligned.
SLOT_BYTES = 64 * -(-8 * (1 + max(MembershipRecord.WORDS, RankReport.WORDS)) // 64)


class SignedStore:
    """A torch ``TCPStore`` whose values carry an HMAC-SHA256 tag under the join secret; a value without a valid tag
    is rejected, so only processes started with the secret can publish or read handles."""

    def __init__(self, store, secret: bytes):
        self._store = store
        self._secret = secret

    def set(self, key: str, value: bytes) -> None:
        self._store.set(key, self._tag(key, value) + value)

    def get(self, key: str) -> bytes:
        """The value of ``key``; waits until it is set."""
        raw = self._store.get(key)
        tag, value = raw[:32], raw[32:]
        if not hmac.compare_digest(tag, self._tag(key, value)):
            raise PermissionError(f"rendezvous value {key!r} is not signed with the join secret")
        return value

    def get_if_set(self, key: str) -> Optional[bytes]:
        """The value of ``key``, or None while it is not set."""
        return self.get(key) if self._store.check([key]) else None

    def _tag(self, key: str, value: bytes) -> bytes:
        message = len(key).to_bytes(4, "little") + key.encode() + value
        return hmac.new(self._secret, message, hashlib.sha256).digest()


class SharedRegion:
    """GPU memory of this process that other processes map over NVLink (CUDA fabric handles), or such memory of
    another process mapped into this one."""

    def __init__(self, nbytes: int, device: int, handle_bytes: Optional[bytes] = None):
        from .mnnvl_memory import MnnvlMemory, _check_cu_result, cuda

        self._cuda = cuda
        prop = MnnvlMemory.get_allocation_prop(device)
        granularity = MnnvlMemory.get_allocation_granularity(device)
        self.nbytes = granularity * -(-nbytes // granularity)
        if handle_bytes is None:
            self._handle = _check_cu_result(cuda.cuMemCreate(self.nbytes, prop, 0))
            exported = _check_cu_result(
                cuda.cuMemExportToShareableHandle(self._handle, prop.requestedHandleTypes, 0)
            )
            self.handle_bytes = bytes(exported.data)
        else:
            self._handle = _check_cu_result(
                cuda.cuMemImportFromShareableHandle(handle_bytes, prop.requestedHandleTypes)
            )
            self.handle_bytes = handle_bytes
        self.ptr = int(_check_cu_result(cuda.cuMemAddressReserve(self.nbytes, granularity, 0, 0)))
        _check_cu_result(cuda.cuMemMap(self.ptr, self.nbytes, 0, self._handle, 0))
        access = cuda.CUmemAccessDesc()
        access.location = prop.location
        access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _check_cu_result(cuda.cuMemSetAccess(self.ptr, self.nbytes, [access], 1))
        if handle_bytes is None:
            _check_cu_result(cuda.cuMemsetD8(self.ptr, 0, self.nbytes))

    def close(self) -> None:
        """Unmap the memory (an owner also releases it once every importer did)."""
        from .mnnvl_memory import _check_cu_result

        cuda = self._cuda
        _check_cu_result(cuda.cuMemUnmap(self.ptr, self.nbytes))
        _check_cu_result(cuda.cuMemAddressFree(self.ptr, self.nbytes))
        _check_cu_result(cuda.cuMemRelease(self._handle))


class SeqlockSlots:
    """Writes and reads of fixed-size seqlock slots through the copy engines, on one stream.

    A writer sets the slot's version to an odd value, copies the payload, then sets the next even value, each step
    ordered after the previous one by ``cuStreamWriteValue64``. A reader copies the version, the payload and the version
    again, and keeps the payload only when both versions are equal, even and not 0 (never written).
    """

    def __init__(self):
        import torch

        from .mnnvl_memory import _check_cu_result, cuda

        self._cuda = cuda
        self._check = _check_cu_result
        self._stream = torch.cuda.Stream()
        self._versions: Dict[int, int] = {}
        self._host = (
            torch.empty(SLOT_BYTES // 8 + 2, dtype=torch.int64, pin_memory=prefer_pinned())
            .numpy()
            .view(np.uint64)
        )

    def write(self, address: int, words: np.ndarray) -> None:
        cuda, stream = self._cuda, self._stream.cuda_stream
        if (
            address not in self._versions
        ):  # continue after an earlier writer, such as a replaced process
            self._check(cuda.cuMemcpyDtoHAsync(self._host[0:1].ctypes.data, address, 8, stream))
            self._stream.synchronize()
            self._versions[address] = int(self._host[0]) + (int(self._host[0]) & 1)
        version = self._versions[address]
        flags = cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT
        self._check(cuda.cuStreamWriteValue64(stream, address, version + 1, flags))
        payload = self._host[2 : 2 + len(words)]
        payload[:] = words
        self._check(
            cuda.cuMemcpyHtoDAsync(address + 8, payload.ctypes.data, payload.nbytes, stream)
        )
        self._check(cuda.cuStreamWriteValue64(stream, address, version + 2, flags))
        self._versions[address] = version + 2
        self._stream.synchronize()

    def read(self, address: int, nwords: int) -> Optional[np.ndarray]:
        cuda, stream = self._cuda, self._stream.cuda_stream
        host = self._host
        self._check(cuda.cuMemcpyDtoHAsync(host[0:1].ctypes.data, address, 8, stream))
        self._check(cuda.cuMemcpyDtoHAsync(host[2:].ctypes.data, address + 8, 8 * nwords, stream))
        self._check(cuda.cuMemcpyDtoHAsync(host[1:2].ctypes.data, address, 8, stream))
        self._stream.synchronize()
        if host[0] != host[1] or host[0] == 0 or host[0] & 1:
            return None
        return host[2 : 2 + nwords].copy()


class MembershipAgent:
    """A rank's side of membership: a thread that writes the rank's report into the authority's inbox every
    ``interval_s`` and follows the record the authority writes into the rank's control row.

    The report's heartbeat advances only while this thread runs, so it stops when the process or its GPU is gone. The
    executor sets the rest of the report (:meth:`report`); ``on_record`` is called on this thread with every new record.
    """

    def __init__(
        self,
        rank: int,
        device: int,
        control_row: SharedRegion,
        inbox: SharedRegion,
        on_record: Callable[[MembershipRecord], None],
        interval_s: float = 0.05,
        incarnation: int = 0,
    ):
        import threading

        self.rank = rank
        self._device = device
        self._row = control_row
        self._inbox = inbox
        self._on_record = on_record
        self._interval_s = interval_s
        self._report = RankReport(incarnation=incarnation)
        self.record: Optional[MembershipRecord] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"MembershipAgent{rank}", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def report(self, **changes) -> None:
        """Update the fields of the next report (step, waiting_for, reply_generation, reply, stopped_at)."""
        with self._lock:
            self._report = RankReport(**{**self._report.__dict__, **changes})

    def _run(self) -> None:
        import torch

        torch.cuda.set_device(self._device)  # a new thread starts on device 0
        slots = SeqlockSlots()
        while not self._stop.is_set():
            with self._lock:
                self._report = RankReport(
                    **{**self._report.__dict__, "heartbeat": self._report.heartbeat + 1}
                )
                report = self._report
            slots.write(self._inbox.ptr + self.rank * SLOT_BYTES, report.to_words())
            words = slots.read(self._row.ptr, MembershipRecord.WORDS)
            if words is not None:
                record = MembershipRecord.from_words(words)
                if record != self.record:
                    admitted = self.record is not None and self.record.is_live(self.rank)
                    self.record = record
                    self._on_record(record)
                    if admitted and not record.is_live(self.rank):
                        return  # excluded: the inbox slot now belongs to this rank's replacement
            self._stop.wait(self._interval_s)


class AuthorityProcess:
    """The authority as a process: reads the ranks' reports from its inbox, decides with :class:`Authority`, and
    writes every new record into every rank's control row.

    Rank r publishes ``row/<r>`` (its control row's handle) to the store; the authority publishes ``inbox``. The n-th
    replacement for rank r (n from 1) publishes its control row's handle as ``join/<r>/<n>``. Rank 0 sets
    ``shutdown`` to stop the authority. Values not signed with the join secret are ignored.
    """

    def __init__(
        self,
        world_size: int,
        store: SignedStore,
        device: int,
        authority: Authority,
        interval_s: float = 0.05,
    ):
        self.world_size = world_size
        self._store = store
        self._device = device
        self._authority = authority
        self._interval_s = interval_s
        self._rows: Dict[int, SharedRegion] = {}
        self._rejected: Set[str] = set()

    def run(self) -> MembershipRecord:
        """Serve until the record is ABORT or SHUTDOWN; returns the last record."""
        import time

        inbox = SharedRegion(self.world_size * SLOT_BYTES, self._device)
        self._store.set("inbox", inbox.handle_bytes)
        for r in range(self.world_size):
            self._rows[r] = SharedRegion(SLOT_BYTES, self._device, self._store.get(f"row/{r}"))
        slots = SeqlockSlots()
        self._publish(slots, self._authority.record)
        reports: Dict[int, RankReport] = {}
        joined = [0] * self.world_size  # how many replacements of each rank asked to join
        while True:
            for r in range(self.world_size):
                words = slots.read(inbox.ptr + r * SLOT_BYTES, RankReport.WORDS)
                if words is not None:
                    try:
                        report = RankReport.from_words(words)
                    except ValueError:  # not a report this version writes
                        report = None
                    if (
                        report is not None and report.incarnation == joined[r]
                    ):  # a replaced process never counts
                        reports[r] = report
                handle = self._get_if_signed(f"join/{r}/{joined[r] + 1}")
                if handle is not None:
                    joined[r] += 1
                    old = self._authority.record
                    excluded = MembershipRecord(
                        old.generation, old.state, old.live_mask & ~(1 << r), old.resume
                    )
                    slots.write(
                        self._rows[r].ptr, excluded.to_words()
                    )  # an old process that still runs stops
                    self._rows[r].close()
                    self._rows[r] = SharedRegion(SLOT_BYTES, self._device, handle)
                    reports.pop(r, None)
                    self._authority.request_join(r)
            if self._get_if_signed("shutdown") is not None:
                record = self._authority.shutdown()
            else:
                record = self._authority.tick(time.monotonic(), reports)
            if record is not None:
                self._publish(slots, record)
                if record.state in (MembershipState.ABORT, MembershipState.SHUTDOWN):
                    return record
            time.sleep(self._interval_s)

    def _get_if_signed(self, key: str) -> Optional[bytes]:
        try:
            return self._store.get_if_set(key)
        except PermissionError:
            if key not in self._rejected:
                self._rejected.add(key)
                logger.warning(
                    f"Ignoring rendezvous value {key!r}: not signed with the join secret"
                )
            return None

    def _publish(self, slots: SeqlockSlots, record: MembershipRecord) -> None:
        """Write ``record`` into every rank's control row; an excluded rank that still runs reads its exclusion."""
        words = record.to_words()
        for row in self._rows.values():
            slots.write(row.ptr, words)


def expert_coverage(holders: List[int]) -> Callable[[int], bool]:
    """``covers`` for :class:`Authority`: every expert keeps a live holder; ``holders[e]`` is the mask of the ranks
    that hold expert e or can restore it into a spare slot."""
    return lambda live_mask: all(h & live_mask for h in holders)

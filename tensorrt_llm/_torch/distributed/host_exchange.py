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
"""Host collectives over MNNVL memory for Wide-EP fault tolerance.

``MnnvlDist`` implements the attention-DP host collectives of ``Distributed`` without MPI, so they
keep working after a rank fails or a replacement joins. Each collective is one or more rounds of
``TPHostExchange``. In a round, each rank writes its slot into every active rank's row and polls
its own row. Each rank owns one row with two slot sets and one slot per sender::

    row:  | slot set 0: slot 0 | slot 1 | ... | slot set 1: slot 0 | slot 1 | ... |
    slot: | header | payload packet 1 | payload packet 2 | ... |

A slot is a run of 8-byte packets. The lower 32 bits hold the payload length in the header and 4
payload bytes in a payload packet. The upper 32 bits hold the round's epoch, so a packet is new when
it carries the current epoch and a slot needs no ready flag.
"""

from __future__ import annotations

import functools
import pickle  # nosec B403
from typing import Callable, Iterator, Optional, Sequence

import numpy as np
import torch
from numpy.typing import ArrayLike

from tensorrt_llm._utils import mpi_disabled, prefer_pinned
from tensorrt_llm._wide_ep_fault_tolerance import check_fault_tolerance_requirements
from tensorrt_llm.bindings.internal.runtime import (
    unpack_payloads,
    unpack_payloads_into,
    wait_for_slots,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.math_utils import ceil_div

from . import mnnvl_memory
from .communicator import Distributed, ReduceOp, _check_reduce_op
from .membership import StoreGroup, all_ranks, has_rank, next_epoch, rank_mask
from .membership_mnnvl_memory import MembershipMnnvlMemory, comm_members
from .mnnvl_memory import (
    MnnvlMemory,
    TpMnnvlMemory,
    _check_cu_result,
    _MnnvlAllocationRecord,
    _MnnvlAllocationState,
)

# Rows sent by one copy as (destination, destination pitch, source, source pitch, width in bytes,
# height); a source pitch of 0 sends one source row to every destination row.
SlotCopy = tuple[int, int, int, int, int, int]


class HostExchangeMnnvlMemory(MembershipMnnvlMemory):
    """MNNVL memory of one row per rank, every row mapped in every process at ``ptr + r * stride``.

    The rows sit back to back, so one strided copy reaches every row. The row of an inactive rank
    maps local scratch memory. The rows a former member may still write to are remapped onto fresh
    pages, so its late writes land in pages that no rank maps.
    """

    absent_rows_on_scratch = True

    @classmethod
    def uses_fabric_handles(cls, dev_id: int) -> bool:
        """Return whether device ``dev_id`` shares memory through fabric handles."""
        handle_type = cls.get_allocation_prop(dev_id).requestedHandleTypes
        return handle_type == mnnvl_memory.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC

    def __init__(self, comm: StoreGroup, nbytes: int):
        """Reserve a row of ``nbytes`` per member of ``comm`` and map it among the active members.

        Raises:
            MnnvlPeerHandleError: An active rank could not import a peer's handle.
            MnnvlMappingError: Mapping or zeroing failed on some active ranks.
            RuntimeError: The device does not share memory through fabric handles.
        """
        cls = type(self)
        dev_id = cls._init_device()
        if not cls.uses_fabric_handles(dev_id):
            raise RuntimeError(
                "HostExchangeMnnvlMemory requires fabric memory handles, which this device lacks"
            )
        granularity = cls.get_allocation_granularity(dev_id)
        stride = granularity * ceil_div(nbytes, granularity)
        nranks = comm.Get_size()
        ptr = int(
            _check_cu_result(
                mnnvl_memory.cuda.cuMemAddressReserve(nranks * stride, granularity, 0, 0)
            )
        )
        cls._register_allocation(
            ptr,
            _MnnvlAllocationRecord(
                comm=None,
                comm_size=nranks,
                comm_rank=comm.Get_rank(),
                comm_membership=comm_members(comm),
                aligned_size=stride,
                mem_handles=[None] * nranks,
                start_address=ptr,
                rank_stride=stride,
                address_offset=0,
                state=_MnnvlAllocationState.UNMAPPED,
            ),
        )
        self.segment_size = nbytes
        self.ptr, self.rank_stride = ptr, stride
        try:
            self.remap(comm, frozenset(comm_members(comm)))
        except BaseException:
            self.close()
            raise

    def _reset_own_row(self) -> None:
        _check_cu_result(
            mnnvl_memory.cuda.cuMemsetD8(
                self.row(type(self).allocated_map[self.ptr].comm_rank), 0, self.stride
            )
        )
        torch.cuda.synchronize()

    @property
    def stride(self) -> int:
        return self.rank_stride

    def row(self, rank: int) -> int:
        return self.ptr + rank * self.rank_stride

    def close(self) -> None:
        """Unmap every row and free the address range."""
        ptr = self.ptr
        del self.ptr
        type(self).close_mnnvl_memory(ptr)


class TPHostExchange:
    """All-to-all of byte strings between the host processes of a TP group, over MNNVL memory.

    The exchange runs the all-to-all kernels' LL protocol on the copy engines, so it takes no SMs.
    Consecutive rounds alternate between two slot sets, so a rank never overwrites a slot that a
    slower rank still reads. Every active rank gets the same result across a membership change. If
    a rank fails in a round that a survivor already finished, the other survivors complete it from
    the packets in the survivors' rows. Otherwise the survivors run the round again without it.
    """

    # Sized so that a step's request broadcast and response gather usually take one round.
    DEFAULT_SLOT_PAYLOAD_BYTES = 64 * 1024
    # The payload packets (1 KiB of payload) that the header poll copies with each header. A payload
    # that fits goes to every row, so it arrives with the header poll and needs no second poll.
    _HEADER_POLL_PACKETS = 256
    # A wait polls back to back for _SPIN_S, then sleeps between polls. An idle wait may last until
    # the next request arrives, so it sleeps after _IDLE_SPIN_S.
    _SPIN_S = 10e-3
    _IDLE_SPIN_S = 20e-6

    def __init__(
        self, rank: int, nranks: int, rows: HostExchangeMnnvlMemory, slot_payload_bytes: int
    ):
        self._cuda = mnnvl_memory.cuda
        self.rank = rank
        self.nranks = nranks
        self.rows = rows
        self._words = slot_payload_bytes // 4
        self._slot_bytes = 8 * (1 + self._words)
        # A round uses slot set epoch & 1, which starts at (epoch & 1) * _slot_set_bytes.
        self._slot_set_bytes = nranks * self._slot_bytes
        # The round counter. It wraps and skips 0, because the rows start zeroed.
        self._epoch = 0
        self.active_mask = all_ranks(nranks)
        # interrupted() reports a membership change that needs this rank. The exchange checks it
        # before each round and, during a wait, after each wake(). When it holds, the exchange sets
        # stopped_epoch to the current or next round's epoch and calls on_interrupt, which runs the
        # change and calls install(). The exchange then takes the round that install() completed
        # from the peers' rows, or runs the round again.
        self.interrupted: Callable[[], bool] = lambda: False
        self.on_interrupt: Optional[Callable[[], None]] = None
        self.stopped_epoch = 0
        # Ranks the running wait still lacks packets from, and the wake word that wake() sets
        # from another thread to end the wait.
        self._waiting = np.zeros(nranks, dtype=bool)
        self._wake = np.zeros(1, dtype=np.int32)
        # Active ranks still missing from a round that install() completed. The round's result
        # treats them as inactive.
        self._missing = 0
        # Readers whose payload this collective lacks. Their payloads come back as None.
        self._failed = 0
        # This collective's readers. Completing a round needs their whole slots and only the
        # headers from other ranks.
        self._round_readers = 0
        # Packets that install() completed from the peers' rows. The round takes them instead of
        # polling.
        self._finished_round: Optional[np.ndarray] = None
        # The epoch to continue from once the round takes _finished_round.
        self._epoch_after_round: Optional[int] = None
        self._stream = torch.cuda.Stream()
        # The outgoing slot. Each send copies it to every row in a range with a source pitch of 0.
        self._send = self._pinned((1, 1 + self._words))
        self._recv = self._pinned((nranks, 1 + self._words))
        self._send_halves = self._send.view(np.uint32)[0]
        self._send_ptr = self._send.ctypes.data
        # Cached _send_runs results, keyed by recipient mask.
        self._send_runs_by_recipients: dict[int, tuple[tuple[int, int, bool], ...]] = {}
        self._send_runs_to_all = ((0, nranks, True),)
        # Cached _ranks_of results, keyed by mask.
        self._ranks_by_mask: dict[int, tuple[np.ndarray, list[int]]] = {}

    @staticmethod
    def _pinned(shape: tuple[int, int]) -> np.ndarray:
        """Return a host array of uint64 packets, pinned where pinning helps."""
        return (
            torch.empty(shape, dtype=torch.int64, pin_memory=prefer_pinned())
            .numpy()
            .view(np.uint64)
        )

    @staticmethod
    def buffer_bytes(nranks: int, slot_payload_bytes: int) -> int:
        return 2 * nranks * 8 * (1 + slot_payload_bytes // 4)

    @classmethod
    def create(cls, mapping) -> "TPHostExchange":
        """Create the TP group's exchange, collectively over its communicator's ranks."""
        comm = TpMnnvlMemory.get_comm(mapping)
        active_mask = rank_mask(
            (rank for rank, active in enumerate(comm.allgather(True)) if active), mapping.tp_size
        )
        rows = HostExchangeMnnvlMemory(
            comm, cls.buffer_bytes(mapping.tp_size, cls.DEFAULT_SLOT_PAYLOAD_BYTES)
        )
        exchange = cls(mapping.tp_rank, mapping.tp_size, rows, cls.DEFAULT_SLOT_PAYLOAD_BYTES)
        exchange.active_mask = active_mask
        return exchange

    @property
    def epoch(self) -> int:
        return self._epoch

    def admit(self, active_mask: int, epoch: int) -> None:
        """Take part with the ranks in ``active_mask``, starting with the round of ``epoch``."""
        self.active_mask = active_mask
        # Each round advances the epoch before it sends.
        self._epoch = epoch - 1

    def install(
        self,
        comm: StoreGroup,
        active_mask: int,
        epoch: int,
        fresh_rows: frozenset[int],
    ) -> None:
        """Switch to the ranks in ``active_mask`` while stopped, collectively over ``comm``.

        The rows in ``fresh_rows`` get fresh pages. ``epoch`` is the first round after the change.
        On an active rank it shows how far this rank stopped behind the latest active rank. At
        ``stopped_epoch`` or the epoch after it, this rank runs the round again. Two epochs after
        it, another rank finished this rank's round, and this rank completes it from the peers'
        rows.

        Raises:
            RuntimeError: This rank stopped more than one round before the latest active rank.
        """
        if has_rank(active_mask, self.rank):
            if epoch in (self.stopped_epoch, next_epoch(self.stopped_epoch)):
                # An earlier attempt of this change may have finished the round with a since-failed
                # rank's packets.
                self._finished_round = self._epoch_after_round = None
            elif self._finished_round is None:
                if (
                    next_epoch(next_epoch(self.stopped_epoch)) != epoch
                    or self.stopped_epoch != self._epoch
                ):
                    raise RuntimeError(
                        f"host exchange stopped at epoch {self.stopped_epoch}, more than one round "
                        f"before the latest active rank (epoch {epoch} is next)"
                    )
                # Read before the remap, as the remap may release the peers' rows.
                self._finished_round = self._finish_from_peers(active_mask)
        round_epoch = self._epoch
        self.rows.remap(comm, fresh_rows)
        self.admit(active_mask, epoch)
        if self._finished_round is not None:
            # Take the completed round at its own epoch, then continue from the installed epoch.
            self._epoch_after_round, self._epoch = self._epoch, round_epoch

    def _finish_from_peers(self, active_mask: int) -> np.ndarray:
        """Complete this rank's current round with the slots it lacks, copied from peer rows.

        A reader's slot counts once it is whole. Another rank's slot counts once its header is in.
        """
        parity_offset = (self._epoch & 1) * self._slot_set_bytes
        rows = self._pinned((self.nranks, 1 + self._words))
        self._copy_to_host(self.rows.row(self.rank) + parity_offset, rows)
        readers = self._ranks_of(self._round_readers)[0]
        no_readers = np.zeros(self.nranks, dtype=bool)
        lacks_header = self._lacks_packets(rows, no_readers)
        lacks_slot = self._lacks_packets(rows, readers) & self._ranks_of(self.active_mask)[0]
        peers = [p for p in range(self.nranks) if p != self.rank and has_rank(active_mask, p)]
        for sender in np.flatnonzero(lacks_slot).tolist():
            reads = readers[sender : sender + 1]
            for peer in peers:
                slot = rows[sender : sender + 1].copy()
                self._copy_to_host(
                    self.rows.row(peer) + parity_offset + sender * self._slot_bytes, slot
                )
                if reads[0] and not self._lacks_packets(slot, reads)[0]:
                    rows[sender] = slot[0]
                    break
                if lacks_header[sender] and not self._lacks_packets(slot, no_readers[:1])[0]:
                    rows[sender] = slot[0]
                    if not reads[0]:
                        break
        return rows

    def _lacks_packets(self, slots: np.ndarray, readers: np.ndarray) -> np.ndarray:
        """Return which of ``slots`` lack this epoch's header, or for ``readers`` the whole slot."""
        lacking = np.zeros(len(slots), dtype=bool)
        wait_for_slots(
            self._stream.cuda_stream,
            [],
            None,
            slots,
            np.ones(len(slots), dtype=bool),
            readers,
            self._epoch,
            slots.shape[1],
            lacking,
            self._wake,
            0.0,
        )
        return lacking

    def _copy_to_host(self, src: int, dst: np.ndarray) -> None:
        cuda = self._cuda
        copy = cuda.CUDA_MEMCPY2D()
        copy.srcMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_DEVICE
        copy.srcDevice, copy.srcPitch = src, self._slot_bytes
        copy.dstMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_HOST
        copy.dstHost, copy.dstPitch = dst.ctypes.data, self._slot_bytes
        copy.WidthInBytes, copy.Height = self._slot_bytes, len(dst)
        _check_cu_result(cuda.cuMemcpy2DAsync(copy, self._stream.cuda_stream))
        self._stream.synchronize()

    def _send_runs(self, recipients: int) -> tuple[tuple[int, int, bool], ...]:
        """Return ``(first rank, count, with payload)`` per run of ranks in or out of ``recipients``."""
        runs = self._send_runs_by_recipients.get(recipients)
        if runs is None:
            full = [has_rank(recipients, r) for r in range(self.nranks)]
            runs, first = [], 0
            for r in range(1, self.nranks + 1):
                if r == self.nranks or full[r] != full[first]:
                    runs.append((first, r - first, full[first]))
                    first = r
            runs = self._send_runs_by_recipients[recipients] = tuple(runs)
        return runs

    def _send_round(
        self, parity_offset: int, words: np.ndarray, length: int, recipients: Optional[int]
    ) -> list[SlotCopy]:
        """Return copies that send the header to every row and the payload to ``recipients``' rows.

        ``wait_for_slots`` enqueues the copies. ``recipients`` None means every rank.
        """
        packets = 1 + words.size
        halves = self._send_halves[: 2 * packets]
        halves[0] = length
        halves[1::2] = self._epoch
        halves[2::2] = words
        if recipients is None or words.size <= self._HEADER_POLL_PACKETS:
            # A payload that a header poll copies whole goes to every row.
            runs = self._send_runs_to_all
        else:
            runs = self._send_runs(recipients)
        own_slot = parity_offset + self.rank * self._slot_bytes
        return [
            (
                self.rows.row(first) + own_slot,
                self.rows.stride,
                self._send_ptr,
                0,
                8 * (packets if payload else 1),
                count,
            )
            for first, count, payload in runs
        ]

    def exchange(
        self,
        payload: bytes,
        readers: Optional[Sequence[int]] = None,
        recipients: Optional[Sequence[int]] = None,
        idle: bool = False,
    ) -> list[Optional[bytes]]:
        """Send ``payload`` to every rank and return every rank's payload, in rank order.

        Args:
            payload: The bytes this rank sends.
            readers: The ranks whose payloads this rank needs (default all). The others come back as
                None, as do ranks outside ``active_mask`` and ranks whose payload a membership
                change left incomplete.
            recipients: The ranks that need this rank's payload (default all). It must include every
                rank that names this rank among its ``readers``.
            idle: Sleep between polls sooner, for waits that may last until the next request.
        """
        reader_mask = self.active_mask
        if readers is not None:
            reader_mask &= rank_mask(readers, self.nranks)
        if recipients is not None:
            recipients = rank_mask(recipients, self.nranks)
        ranks = self._ranks_of(reader_mask)[1]
        rounds = self._rounds(payload, reader_mask, recipients, idle)
        lengths, rows = next(rounds)
        received = self._chunks(rows, lengths, ranks)
        # A payload longer than a slot takes several rounds. Join each rank's chunks in order.
        remaining_chunks = [self._chunks(rows, lengths, ranks) for lengths, rows in rounds]
        if remaining_chunks:
            received = [b"".join(parts) for parts in zip(received, *remaining_chunks)]
        payloads = [None] * self.nranks
        for r, received_payload in zip(ranks, received):
            if not has_rank(self._failed, r):
                payloads[r] = received_payload
        return payloads

    def allgather_array(self, row: np.ndarray, active_only: bool = False) -> np.ndarray:
        """Return every rank's ``row`` as ``[nranks, row.size]``, with zero rows for inactive ranks.

        With ``active_only``, return only the rows that this exchange received.

        Raises:
            ValueError: The ranks passed rows of different sizes.
        """
        row = np.ascontiguousarray(row).reshape(-1)
        data = row.view(np.uint8)
        readers = self.active_mask
        ranks = self._ranks_of(readers)[1]
        out = np.zeros((self.nranks, data.size), dtype=np.uint8)
        step = 4 * self._words
        for k, (lengths, rows) in enumerate(self._rounds(data.tobytes(), readers)):
            if k == 0 and (lengths[self._ranks_of(readers & ~self._failed)[0]] != data.size).any():
                raise ValueError(
                    f"allgather_array rows differ in size across ranks: {lengths.tolist()}"
                )
            if rows is not None:
                unpack_payloads_into(rows, lengths, ranks, out, step * k)
        kept = readers & ~self._failed
        if active_only:
            out = out[self._ranks_of(kept)[0]]
        elif kept != all_ranks(self.nranks):
            out[~self._ranks_of(kept)[0]] = 0
        return out.view(row.dtype)

    def _ranks_of(self, mask: int) -> tuple[np.ndarray, list[int]]:
        """Return ``mask`` as a bool array and as a sorted rank list, cached per mask."""
        ranks = self._ranks_by_mask.get(mask)
        if ranks is None:
            in_mask = np.array([has_rank(mask, r) for r in range(self.nranks)], dtype=bool)
            ranks = self._ranks_by_mask[mask] = (in_mask, np.flatnonzero(in_mask).tolist())
        return ranks

    def _mask_of(self, ranks: np.ndarray) -> int:
        return rank_mask((int(r) for r in np.flatnonzero(ranks)), self.nranks)

    def _chunks(
        self, rows: Optional[np.ndarray], lengths: np.ndarray, ranks: list[int]
    ) -> list[bytes]:
        """Return the payload bytes of ``ranks`` in one round's packets."""
        if rows is None:
            return [b""] * len(ranks)
        return unpack_payloads(rows, lengths, ranks)

    def _rounds(
        self, data: bytes, readers: int, recipients: Optional[int] = None, idle: bool = False
    ) -> Iterator[tuple[np.ndarray, Optional[np.ndarray]]]:
        """Yield header lengths and packets per round, sending one slot of ``data`` per round."""
        self._round_readers = readers
        self._failed = 0
        step = 4 * self._words
        lengths, rows = self._round(data[:step], len(data), readers, recipients, idle)
        yield lengths, rows
        for k in range(1, ceil_div(int(lengths.max()), step)):
            chunk = data[step * k : step * (k + 1)]
            yield self._round(chunk, len(chunk), readers, recipients, idle)

    def _round(
        self,
        chunk: bytes,
        length: int,
        readers: int,
        recipients: Optional[int] = None,
        idle: bool = False,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Exchange up to one slot per rank, running the round again after a membership change."""
        result = None
        while result is None:
            result = self._try_round(chunk, length, readers, recipients, idle)
        return result

    def _try_round(
        self, chunk: bytes, length: int, readers: int, recipients: Optional[int], idle: bool
    ) -> Optional[tuple[np.ndarray, Optional[np.ndarray]]]:
        """Return every rank's header length and the packets received, or None to run it again.

        The packets are None if no reader sent a payload.
        """
        if self.interrupted():
            self._stop(next_epoch(self._epoch))
        self._epoch = next_epoch(self._epoch)
        self._missing = 0
        parity_offset = (self._epoch & 1) * self._slot_set_bytes
        words = np.frombuffer(chunk + b"\0" * (-len(chunk) % 4), dtype=np.uint32)
        sends = self._send_round(parity_offset, words, length, recipients)
        round_mask = self.active_mask
        header_packets = 1 + min(self._words, self._HEADER_POLL_PACKETS) if readers else 1
        rows = self._wait(
            sends, parity_offset, round_mask, readers & round_mask, header_packets, idle
        )
        if rows is None:
            return None
        active = round_mask & ~self._missing
        lengths = self._header_lengths(rows, active)
        payload_bytes = int(lengths[self._ranks_of(readers & active)[0]].max(initial=0))
        packets = 1 + min(self._words, ceil_div(payload_bytes, 4))
        self._failed |= readers & (~round_mask | self._missing)
        if self._epoch_after_round is not None:
            self._epoch, self._epoch_after_round = self._epoch_after_round, None
            self._finished_round = None
        return lengths, (rows[:, :packets] if packets > 1 else None)

    def _header_lengths(self, headers: np.ndarray, active: int) -> np.ndarray:
        """Return the payload length in each rank's header, or 0 for a rank outside ``active``."""
        return headers.view(np.uint32)[:, 0] * self._ranks_of(active)[0]

    @property
    def waiting_for(self) -> int:
        """Return the mask of ranks the current wait still lacks packets from. Thread-safe."""
        # The running wait rewrites _waiting without the GIL. np.flatnonzero counts the set flags,
        # then fills their indices, so it needs a copy that cannot change in between.
        return self._mask_of(self._waiting.copy())

    def wake(self) -> None:
        """Make a running wait return and check ``interrupted``. Safe from any thread."""
        self._wake[0] = 1

    def _stop(self, epoch: int) -> None:
        """Record ``epoch`` as ``stopped_epoch`` and call ``on_interrupt``."""
        self._waiting[:] = False
        self.stopped_epoch = epoch
        if self.on_interrupt is not None:
            self.on_interrupt()

    def _wait_for_slots(
        self,
        sends: list[SlotCopy],
        slots: Optional[int],
        received: np.ndarray,
        active: int,
        readers: int,
        header_packets: int,
        spin_s: float,
    ) -> bool:
        return wait_for_slots(
            self._stream.cuda_stream,
            sends,
            slots,
            received,
            self._ranks_of(active)[0],
            self._ranks_of(readers)[0],
            self._epoch,
            header_packets,
            self._waiting,
            self._wake,
            spin_s,
        )

    def _take_finished_round(self, sends: list[SlotCopy], active: int, readers: int) -> np.ndarray:
        """Send ``sends`` and return the round that install() finished, noting missing ranks."""
        rows = self._finished_round
        if not self._wait_for_slots(sends, None, rows, active, readers, rows.shape[1], 0.0):
            self._missing |= self._mask_of(self._waiting)
            self._waiting[:] = False
        return rows

    def _wait(
        self,
        sends: list[SlotCopy],
        parity_offset: int,
        active: int,
        readers: int,
        header_packets: int,
        idle: bool,
    ) -> Optional[np.ndarray]:
        """Send ``sends`` and wait for the ``active`` ranks' headers and the ``readers``' payloads.

        Return the received slots, or None if a membership change restarts the round.
        """
        if self._finished_round is not None:
            return self._take_finished_round(sends, active, readers)
        slots = self.rows.row(self.rank) + parity_offset
        spin_s = self._IDLE_SPIN_S if idle else self._SPIN_S
        while not self._wait_for_slots(
            sends, slots, self._recv, active, readers, header_packets, spin_s
        ):
            sends = []
            # Clear the word before the check, so that a wake after the check ends the next wait.
            self._wake[0] = 0
            if self.interrupted():
                self._stop(self._epoch)
                if self._finished_round is not None:
                    return self._take_finished_round([], active, readers)
                return None
        return self._recv


class MnnvlDist(Distributed):
    """``Distributed`` whose host collectives go through MNNVL memory, over the active ranks.

    Object gathers return None for inactive ranks, reductions skip them, and ``tp_allgather_int64``
    gives them zero rows. A gather with ``active_only`` returns the active ranks' values only. On a
    replacement rank the startup collectives return the original ranks' results from the startup
    record. It has no pipeline-parallel, context-parallel or point-to-point communication.
    """

    _REDUCE = {
        ReduceOp.SUM: lambda a, b: a + b,
        ReduceOp.PRODUCT: lambda a, b: a * b,
        ReduceOp.MIN: min,
        ReduceOp.MAX: max,
        ReduceOp.BAND: lambda a, b: a & b,
        ReduceOp.BOR: lambda a, b: a | b,
        ReduceOp.BXOR: lambda a, b: a ^ b,
    }

    # broadcast_or_none is a single host collective, so the request broadcaster skips its
    # request-count probe.
    supports_single_round_broadcast = True

    # The first MnnvlDist that Distributed.get created in this process.
    _first: Optional[MnnvlDist] = None

    def __init__(self, mapping: Mapping):
        super().__init__(mapping)
        self._host_exchange = None
        # The MnnvlDist whose host exchange this one uses.
        self._host_exchange_owner = self

    @staticmethod
    def check_requirements(mapping: Mapping) -> None:
        """Raise ValueError naming every unmet ``MnnvlDist`` requirement."""
        checks = [
            (
                not mpi_disabled(),
                "the MPI orchestrator (TLLM_DISABLE_MPI unset), because a replacement rank cannot "
                "join a torch.distributed process group",
            ),
            (
                mapping.enable_attention_dp,
                "attention DP (enable_attention_dp=True), so that a failed rank takes only its "
                "own requests with it",
            ),
            (
                mapping.pp_size == 1,
                "pipeline_parallel_size=1, because MnnvlDist has no pipeline-parallel group",
            ),
            (
                mapping.cp_size == 1,
                "context_parallel_size=1, because MnnvlDist has no context-parallel group",
            ),
            (
                mapping.tp_size == mapping.world_size,
                "tensor_parallel_size equal to the world size, because the host collectives run "
                "over a TP group that holds every rank",
            ),
        ]
        if all(met for met, _ in checks):
            if not MnnvlMemory.supports_mnnvl():
                checks.append(
                    (
                        False,
                        "GPUs in one multi-node NVLink (MNNVL) domain, such as a GB200 NVL72, "
                        "because fault tolerance fences off a failed rank by remapping its memory",
                    )
                )
            elif not HostExchangeMnnvlMemory.uses_fabric_handles(torch.cuda.current_device()):
                checks.append(
                    (
                        False,
                        "MNNVL memory shared through fabric handles (GB200-class aarch64 nodes "
                        "with IMEX running and an IMEX channel in the container)",
                    )
                )
        check_fault_tolerance_requirements(checks)

    @staticmethod
    def create(mapping: Mapping) -> MnnvlDist:
        """Return a ``MnnvlDist`` for ``mapping`` after checking its requirements.

        It shares the process's first host exchange, so a membership change reaches all of them.
        """
        MnnvlDist.check_requirements(mapping)
        dist = MnnvlDist(mapping)
        if Distributed.get.cache_info().currsize:
            dist._host_exchange_owner = MnnvlDist._first
        else:
            MnnvlDist._first = dist
        return dist

    @property
    def local_world_size(self):
        """Return the number of ranks on this node that share its free memory as this rank starts.

        An original rank counts all ranks on the node. A replacement rank counts only the node's
        ranks outside the membership, because the active ranks already hold their memory.
        ``local_mpi_size()`` still counts all ranks on the node.
        """
        mapping = self.mapping
        node_ranks = [
            rank
            for rank in range(self.world_size)
            if mapping.get_node_rank(rank) == mapping.node_rank
        ]
        startup_record = Distributed.startup_record
        if startup_record is not None and startup_record.replacement:
            from .membership.startup import membership_client

            record = membership_client().record
            node_ranks = [rank for rank in node_ranks if not record.is_active(rank)]
        return len(node_ranks)

    @property
    def host_exchange(self):
        """Return the ``TPHostExchange``, created collectively on first use.

        Raises:
            RuntimeError: The exchange this communicator shares serves another TP group.
        """
        if self._host_exchange is None:
            owner = self._host_exchange_owner
            exchange = TPHostExchange.create(self.mapping) if owner is self else owner.host_exchange
            if (exchange.rank, exchange.nranks) != (self.mapping.tp_rank, self.tp_size):
                raise RuntimeError(
                    f"Host exchange for TP rank {exchange.rank}/{exchange.nranks} cannot serve "
                    f"TP rank {self.mapping.tp_rank}/{self.tp_size}"
                )
            self._host_exchange = exchange
        return self._host_exchange

    def _exchange(
        self, obj, readers=None, recipients=None, idle=False, reduce_op=None, active_only=False
    ) -> list:
        """Exchange ``obj`` as ``TPHostExchange.exchange`` does, and unpickle the payloads.

        ``reduce_op`` (``min`` or ``max``) lets a replacement's startup replay check its own value
        against the original ranks' reduced result. ``active_only`` drops inactive ranks.
        """
        host_exchange = self.host_exchange
        payload = pickle.dumps(obj)

        def run():
            return host_exchange.exchange(payload, readers, recipients, idle)

        record = Distributed.startup_record
        if record is not None and record.active:
            payloads = record.exchange(self.mapping.tp_rank, payload, readers, run, reduce_op)
        else:
            payloads = run()
        return [
            None if payload is None else pickle.loads(payload)  # nosec B301
            for payload in payloads
            if payload is not None or not active_only
        ]

    def broadcast(self, obj, root=0, chunk_size: int = 4 * 1024 * 1024, prefer_cpu=False):
        # Only the idle request fetch passes prefer_cpu=True. Its wait may last until the next
        # request arrives, so it runs as an idle wait.
        if self.mapping.tp_rank == root:
            self._exchange(obj, readers=(), idle=prefer_cpu)
            return obj
        return self._exchange(None, readers=(root,), recipients=(), idle=prefer_cpu)[root]

    def broadcast_or_none(self, obj, root=0):
        return self.broadcast(obj, root)

    def allgather(self, obj):
        return self._exchange(obj)

    def barrier(self):
        self._exchange(None, readers=(), recipients=())

    def tp_barrier(self):
        self.barrier()

    def tp_allgather(
        self,
        obj,
        chunk_size: int = 4 * 1024 * 1024,
        *,
        small_payload: bool = False,
        active_only: bool = False,
    ):
        return self._exchange(obj, active_only=active_only)

    def tp_gather(self, obj, root=0, chunk_size: int = 4 * 1024 * 1024):
        if self.mapping.tp_rank != root:
            self._exchange(obj, readers=(), recipients=(root,))
            return None
        return self._exchange(obj, recipients=(root,))

    def tp_broadcast(self, obj, root=0, chunk_size: int = 4 * 1024 * 1024, **kwargs):
        return self.broadcast(obj, root)

    def cp_broadcast(self, obj, root=0, **kwargs):
        raise NotImplementedError("MnnvlDist has no context-parallel group")

    def cp_allgather(self, obj, *, small_payload: bool = False):
        raise NotImplementedError("MnnvlDist has no context-parallel group")

    def allreduce(self, obj, op: ReduceOp = ReduceOp.SUM):
        _check_reduce_op(op)
        reduce_op = self._REDUCE[op] if op in (ReduceOp.MIN, ReduceOp.MAX) else None
        values = self._exchange(obj, reduce_op=reduce_op, active_only=True)
        return functools.reduce(self._REDUCE[op], values)

    def tp_allreduce(self, obj, op: ReduceOp = ReduceOp.SUM):
        return self.allreduce(obj, op)

    def tp_allgather_int64(self, values: ArrayLike, *, active_only: bool = False) -> np.ndarray:
        host_exchange = self.host_exchange
        values = np.asarray(values, dtype=np.int64)
        record = Distributed.startup_record
        if record is None or not record.active:
            return host_exchange.allgather_array(values, active_only)
        # The original ranks were all active at their startup.
        rows = record.exchange(
            self.mapping.tp_rank,
            values.tobytes(),
            None,
            lambda: [row.tobytes() for row in host_exchange.allgather_array(values)],
            decode=lambda row: np.frombuffer(row, dtype=np.int64).tolist(),
        )
        return np.stack([np.frombuffer(row, dtype=np.int64) for row in rows])

    def broadcast_int64(
        self, values: ArrayLike, root: int = 0, *, prefer_cpu: bool = False
    ) -> np.ndarray:
        row = np.ascontiguousarray(np.asarray(values, dtype=np.int64).reshape(-1))
        return self.broadcast(row, root, prefer_cpu=prefer_cpu)

    def tp_cp_broadcast_int64(self, values: ArrayLike, root: int = 0) -> np.ndarray:
        return self.broadcast_int64(values, root)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host collectives over MNNVL memory, for WideEP fault tolerance.

``MnnvlRows`` maps one row of MNNVL memory per rank, ``TPHostExchange`` exchanges byte strings between the
host processes of a TP group over those rows, and ``MnnvlDist`` runs the host collectives of ``Distributed``
over that exchange.
"""

from __future__ import annotations

import pickle  # nosec B403
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from numpy.typing import ArrayLike

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.mapping import Mapping

from . import mnnvl_memory
from .communicator import MPIDist, ReduceOp, _check_reduce_op
from .host_exchange_epoch import next_epoch
from .mnnvl_memory import (
    MnnvlMemory,
    TpMnnvlMemory,
    _check_cu_result,
    _MnnvlAllocationRecord,
    _MnnvlAllocationState,
)
from .startup_record import startup_record


class MnnvlRows(MnnvlMemory):
    """MNNVL memory of one row per rank, every row mapped in every process at ``ptr + r * stride``.

    Unlike other :class:`MnnvlMemory` allocations, whose rows lie a fabric page apart with unmapped address space
    between them, the rows sit back to back, so one strided copy reaches every row. The row of an absent rank maps
    local scratch memory. A membership change maps the rows at the same addresses, on fresh pages if a process that
    is no longer a member may still write into the current ones (:meth:`needs_fresh_pages`): it can then only write
    into pages nobody maps anymore. :meth:`remap` zeroes this rank's row, so on return every live rank may write
    into any row. Per-class state is separate from MnnvlMemory's.
    """

    absent_rows_on_scratch = True

    @classmethod
    def uses_fabric_handles(cls, dev_id: int) -> bool:
        """Whether device ``dev_id`` shares this memory through fabric handles, which the rows require."""
        handle_type = cls.get_allocation_prop(dev_id).requestedHandleTypes
        return handle_type == mnnvl_memory.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC

    def __init__(self, rank: int, nranks: int, nbytes: int, gather: Callable[[Any], Sequence[Any]]):
        """Reserve the rows' addresses and map them: collective over the live ranks, which ``gather`` reaches (see
        :meth:`remap`). On any failure nothing stays mapped, allocated or reserved.

        Raises:
            MnnvlPeerHandleError: Some live rank could not import a live rank's handle, as when that rank exited
                after exporting it; ``ranks`` are the ranks whose handles failed.
            MnnvlMappingError: Mapping or zeroing failed on some live ranks for another reason; ``ranks`` are those
                ranks.
            RuntimeError: The device does not share memory through fabric handles.
        """
        cls = type(self)
        dev_id = cls._init_device()
        if not cls.uses_fabric_handles(dev_id):
            raise RuntimeError(
                "MnnvlRows requires fabric memory handles, which this device does not use"
            )
        granularity = cls.get_allocation_granularity(dev_id)
        stride = granularity * -(-nbytes // granularity)
        ptr = int(
            _check_cu_result(
                mnnvl_memory.cuda.cuMemAddressReserve(nranks * stride, granularity, 0, 0)
            )
        )
        cls.allocated_map[ptr] = _MnnvlAllocationRecord(
            comm=None,
            comm_size=nranks,
            comm_rank=rank,
            comm_membership=tuple(range(nranks)),
            aligned_size=stride,
            mem_handles=[None] * nranks,
            start_address=ptr,
            rank_stride=stride,
            address_offset=0,
            state=_MnnvlAllocationState.UNMAPPED,
        )
        cls.address_refcnt[ptr] = 1
        self.segment_size = nbytes
        self.ptr, self.rank_stride = ptr, stride
        try:
            self.remap(gather)
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
        """Unmap every row and free the address range; not collective. Closes at most once, even if it fails."""
        ptr = self.ptr
        del self.ptr
        type(self).close_mnnvl_memory(ptr)


class _RoundRestarted(Exception):
    """The membership changed while this rank was in the last round any rank reached: the round runs again."""


class TPHostExchange:
    """All-to-all of byte strings between the host processes of a TP group, over one row of MNNVL memory per rank.

    The same LL protocol as the kernels above, driven by the host through the GPU's copy engines: a payload travels
    as 8-byte packets ``(epoch << 32) | word`` that the sender copies into its receive slot in the buffers of the
    ranks that read it; a receiver copies the slots it reads from its own buffer to pinned host memory until every
    packet it waits for carries the current epoch. Each exchange advances the epoch and uses one of two slot sets
    chosen by its parity, so, as for the kernels, a rank never overwrites a slot a slower rank still reads: every
    exchange waits for every rank.

    A sender's first packet, its header, holds its payload's length and goes into every rank's buffer. Payloads
    longer than one slot take several rounds; the first round's lengths tell every rank how many rounds follow. Every
    live rank calls :meth:`exchange` in step, like any collective. No kernel runs, so the exchange never competes with
    the model for SMs.

    Membership is a mask: an exchange waits for and returns only the ranks in ``live_mask``; the others come back as
    None. A membership change stops the exchange, so a rank waiting for a failed peer can take part in it: the start
    of every round, and every poll of a wait, stop once ``interrupted`` returns True. Where it stopped,
    the exchange calls ``on_interrupt`` (set together with ``interrupted``), which runs the change, calling
    :meth:`install` on every live rank, and returns once the change is committed. The exchange then continues, and
    every live rank gets the same result from every exchange: a round that some rank finished before the change is
    finished by the others with the packets that rank received; any other round runs again on the new rows without
    the ranks that left. No rank is more than one round ahead of another, as a round only ends for a rank once every
    live rank has sent into it.
    """

    # Bytes of payload per slot and round; larger payloads take more rounds. Sized so that a step's request broadcast
    # and response gather usually take one round; a rank's row holds about 4 * nranks bytes per slot payload byte.
    DEFAULT_SLOT_PAYLOAD_BYTES = 64 * 1024
    # Payload packets per rank that a header poll also copies: payloads up to 1 KB arrive with their headers.
    _POLL_PACKETS = 256
    # A wait polls without sleeping, but in an idle exchange (a peer may park until the next request) it sleeps
    # between polls, with a doubling backoff up to this long.
    _MAX_SLEEP_S = 1e-3

    def __init__(self, rank: int, nranks: int, rows: MnnvlRows, slot_payload_bytes: int):
        from .mnnvl_memory import _check_cu_result, cuda

        self._cuda = cuda
        self._check_cu = _check_cu_result
        self.rank = rank
        self.nranks = nranks
        self.rows = rows
        self._words = slot_payload_bytes // 4
        self._slot_bytes = 8 * (1 + self._words)
        self._parity_bytes = nranks * self._slot_bytes
        self._epoch = 0  # epoch 0 is never sent: the buffers start zeroed
        self.live_mask = (1 << nranks) - 1
        self.interrupted: Callable[[], bool] = lambda: False
        self.on_interrupt: Optional[Callable[[], None]] = None
        # The epoch of the round this rank stopped in, or of the round it was about to start.
        self.stopped_epoch = 0
        # The packets the current wait has not received yet, by rank (see waiting_for); None while not waiting.
        self._stale_packets: Optional[np.ndarray] = None
        # Ranks missing from the current round, and ranks whose payload the current exchange lost to a change.
        self._missing = 0
        self._lost = 0
        # Mask of the live ranks whose payloads the current exchange reads.
        self._round_readers = 0
        # The current round as finished from the peers' rows during a change, and the epoch to continue from after it.
        self._finished_round: Optional[np.ndarray] = None
        self._epoch_after_round: Optional[int] = None
        self._stream = torch.cuda.Stream()
        # One copy of the outgoing slot per peer: a 2D copy needs a source pitch of at least its width.
        self._send = self._pinned((nranks, 1 + self._words))
        self._recv = self._pinned((nranks, 1 + self._words))
        self._recv_epochs = self._epochs(self._recv)
        # The outgoing packets as (word, epoch) pairs of little-endian uint32, filled without temporaries.
        self._send_halves = self._send.view(np.uint32)
        # The copies of a round over consecutive slots, by (direction, slot set offset in a row, first slot, slots):
        # this rank's slot into those ranks' rows, or those slots of this rank's row to the host. Built once, as the
        # rows keep their addresses; each use sets its width.
        self._copies: Dict[Tuple[bool, int, int, int], object] = {}
        # Per mask of the ranks that read this rank's payload, the copies that send a round (see _send_runs); the one
        # copy that sends it into every row.
        self._runs: Dict[int, Tuple[Tuple[int, int, bool], ...]] = {}
        self._runs_to_all = ((0, nranks, True),)
        # Per mask of ranks, whether each rank is in it and the ranks in it (see _ranks_of).
        self._masks: Dict[int, Tuple[np.ndarray, List[int]]] = {}
        # The offset of each packet of a slot in its payload: the header (-1) precedes every payload.
        self._packet_offsets = 4 * np.arange(1 + self._words) - 4
        self._packet_offsets[0] = -1
        self._rank_index = np.arange(nranks)

    @staticmethod
    def _pinned(shape: Tuple[int, int]) -> np.ndarray:
        """A host array of uint64 packets, pinned unless pinning does not help (Confidential Compute)."""
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
        """Collective over the TP group, which must have MNNVL memory. The live ranks of the exchange are those the
        group's communicator reaches: a replacement rank's (isolated startup) reaches only itself until its admission
        rebuilds the exchange with its peers."""
        comm = TpMnnvlMemory.get_comm(mapping)
        live_mask = sum(1 << rank for rank, live in enumerate(comm.allgather(True)) if live)
        rows = MnnvlRows(
            mapping.tp_rank,
            mapping.tp_size,
            cls.buffer_bytes(mapping.tp_size, cls.DEFAULT_SLOT_PAYLOAD_BYTES),
            comm.allgather,
        )
        exchange = cls(mapping.tp_rank, mapping.tp_size, rows, cls.DEFAULT_SLOT_PAYLOAD_BYTES)
        exchange.live_mask = live_mask
        return exchange

    @property
    def epoch(self) -> int:
        """The epoch of the last exchange round."""
        return self._epoch

    def rebuild(self, gather: Callable[[Any], Sequence[Any]], live_mask: int, epoch: int) -> None:
        """Continue with the ranks in ``live_mask`` on the rows of the ranks ``gather`` reaches, at the same addresses,
        from ``epoch`` (above every epoch any rank used): on fresh rows if a process that maps the current ones may not
        be among those ranks (:meth:`MnnvlMemory.needs_fresh_pages`), else on the current rows, mapping those of the
        ranks that join.

        Collective over the ranks ``gather`` reaches, whose rows are mapped: a superset of ``live_mask`` maps the rows
        of ranks that enter later (:meth:`admit`). No rank may exchange until every one has rebuilt.
        """
        if self.rows.needs_fresh_pages(gather):
            self.rows.checkpoint_prepare()
            self.rows.remap(gather)
        else:
            self.rows.admit(gather)
        self.admit(live_mask, epoch)

    def admit(self, live_mask: int, epoch: int) -> None:
        """Continue with the ranks in ``live_mask``, whose rows the last :meth:`rebuild` mapped, from ``epoch``
        (above every epoch any rank used). Called by every rank in ``live_mask`` between the same two rounds."""
        self.live_mask = live_mask
        self._epoch = epoch - 1

    def install(self, gather: Callable[[Any], Sequence[Any]], live_mask: int, epoch: int) -> None:
        """Switch to the ranks in ``live_mask`` while stopped; called by ``on_interrupt`` on every live rank.

        ``epoch`` is the epoch after the latest :attr:`stopped_epoch` among the live ranks; the round stopped in last
        runs again from it. A rank stopped in the round before finishes that round with the packets its live peers
        received (a peer that went on to the next round received them all). Then :meth:`rebuild`, collective over the
        ranks ``gather`` reaches.

        Raises:
            RuntimeError: This rank stopped more than one round before the latest live rank.
        """
        if next_epoch(self.stopped_epoch) == epoch:
            # Every live rank runs this round again, also after an earlier attempt of the change finished it here
            # with the packets of a rank that has failed since.
            self._finished_round = self._epoch_after_round = None
        elif self._finished_round is None:
            if (
                next_epoch(next_epoch(self.stopped_epoch)) != epoch
                or self.stopped_epoch != self._epoch
            ):
                raise RuntimeError(
                    f"host exchange stopped at epoch {self.stopped_epoch}, more than one round before the "
                    f"latest live rank (epoch {epoch} is next)"
                )
            # Read before rebuild: the peers' rows stay mapped here until this rank's checkpoint_prepare.
            self._finished_round = self._finish_from_peers(live_mask)
        round_epoch = self._epoch
        self.rebuild(gather, live_mask, epoch)
        if self._finished_round is not None:
            self._epoch_after_round, self._epoch = self._epoch, round_epoch

    def _finish_from_peers(self, live_mask: int) -> np.ndarray:
        """This rank's current round: every sender's slot from this rank's row or, where that lacks the sender's
        header or a payload this rank reads, from the first row of a peer in ``live_mask`` that holds it, else one
        that holds the header (every peer that finished the round received every header; a payload went only into
        the rows of its readers, and only they waited for it)."""
        parity_offset = (self._epoch & 1) * self._parity_bytes
        rows = self._pinned((self.nranks, 1 + self._words))
        self._copy_to_host(self.rows.row(self.rank) + parity_offset, rows)
        epochs = self._epochs(rows)
        peers = [p for p in range(self.nranks) if p != self.rank and live_mask >> p & 1]
        for sender in range(self.nranks):
            reads = self._round_readers >> sender & 1
            if not self.live_mask >> sender & 1 or (
                self._slot_complete(rows[sender]) if reads else epochs[sender, 0] == self._epoch
            ):
                continue
            for peer in peers:
                slot = rows[sender : sender + 1].copy()
                self._copy_to_host(
                    self.rows.row(peer) + parity_offset + sender * self._slot_bytes, slot
                )
                if reads and self._slot_complete(slot[0]):
                    rows[sender] = slot[0]
                    break
                if self._epochs(slot)[0, 0] == self._epoch and epochs[sender, 0] != self._epoch:
                    rows[sender] = slot[0]  # its header, until a peer with the whole slot is found
                    if not reads:
                        break
        return rows

    def _slot_complete(self, slot: np.ndarray) -> bool:
        """Whether ``slot`` holds a header and every payload packet of the current epoch."""
        length, epoch = slot[:1].view(np.uint32).tolist()
        if epoch != self._epoch:
            return False
        packets = 1 + min(self._words, -(-length // 4))
        return bool((self._epochs(slot[:packets]) == self._epoch).all())

    def _slot_copy(self, send: bool, host: int, device: int, count: int, device_pitch: int):
        """A copy of ``count`` slots, consecutive at host address ``host`` and ``device_pitch`` bytes apart from
        device address ``device``: with ``send`` to the device, else to the host. Each use may set its width (one
        slot by default)."""
        cuda = self._cuda
        on_host, on_device = (
            cuda.CUmemorytype.CU_MEMORYTYPE_HOST,
            cuda.CUmemorytype.CU_MEMORYTYPE_DEVICE,
        )
        copy = cuda.CUDA_MEMCPY2D()
        if send:
            copy.srcMemoryType, copy.srcHost, copy.srcPitch = on_host, host, self._slot_bytes
            copy.dstMemoryType, copy.dstDevice, copy.dstPitch = on_device, device, device_pitch
        else:
            copy.srcMemoryType, copy.srcDevice, copy.srcPitch = on_device, device, device_pitch
            copy.dstMemoryType, copy.dstHost, copy.dstPitch = on_host, host, self._slot_bytes
        copy.WidthInBytes, copy.Height = self._slot_bytes, count
        return copy

    def _copy_to_host(self, src: int, dst: np.ndarray) -> None:
        """Copy ``len(dst)`` consecutive slots from device address ``src`` into ``dst``."""
        copy = self._slot_copy(False, dst.ctypes.data, src, len(dst), self._slot_bytes)
        self._check_cu(self._cuda.cuMemcpy2DAsync(copy, self._stream.cuda_stream))
        self._stream.synchronize()

    def _round_copy(self, send: bool, parity_offset: int, first: int, count: int):
        """The copy of ranks ``first`` to ``first + count - 1`` in the slot set at ``parity_offset`` (see ``_copies``):
        with ``send``, this rank's outgoing slot into those ranks' rows, else those ranks' slots of this rank's row to
        the host."""
        key = (send, parity_offset, first, count)
        copy = self._copies.get(key)
        if copy is None:
            if send:
                copy = self._slot_copy(
                    True,
                    self._send.ctypes.data + first * self._slot_bytes,
                    self.rows.row(first) + parity_offset + self.rank * self._slot_bytes,
                    count,
                    self.rows.stride,
                )
            else:
                copy = self._slot_copy(
                    False,
                    self._recv.ctypes.data + first * self._slot_bytes,
                    self.rows.row(self.rank) + parity_offset + first * self._slot_bytes,
                    count,
                    self._slot_bytes,
                )
            self._copies[key] = copy
        return copy

    def _send_runs(self, recipients: int) -> Tuple[Tuple[int, int, bool], ...]:
        """The copies that send a round, as (first rank, ranks, with the payload), one per run of consecutive ranks
        alike: the payload goes into the rows of the ranks in the mask ``recipients``, the header alone into the
        others."""
        runs = self._runs.get(recipients)
        if runs is None:
            full = [bool(recipients >> r & 1) for r in range(self.nranks)]
            runs, first = [], 0
            for r in range(1, self.nranks + 1):
                if r == self.nranks or full[r] != full[first]:
                    runs.append((first, r - first, full[first]))
                    first = r
            runs = self._runs[recipients] = tuple(runs)
        return runs

    def _send_round(
        self, parity_offset: int, words: np.ndarray, length: int, recipients: Optional[int]
    ) -> None:
        """Send this rank's packets of the current round, in the slot set at ``parity_offset``: the header
        (``length``) into its slot in every row, the payload ``words`` too into the rows of the ranks in the mask
        ``recipients`` (default all). A payload that a header poll copies whole goes into every row with the header,
        filled in place in every outgoing slot; a longer one is filled once, then copied into the outgoing slots that
        carry it."""
        packets = 1 + words.size
        if words.size <= self._POLL_PACKETS:
            runs = self._runs_to_all
            # Packet i is the uint32 pair (word, epoch); the header's word is the length.
            halves = self._send_halves[:, : 2 * packets]
            halves[:, 0] = length
            halves[:, 1::2] = self._epoch
            halves[:, 2::2] = words
        else:
            runs = self._runs_to_all if recipients is None else self._send_runs(recipients)
            filled = None  # the outgoing slot that holds the round's packets
            for first, count, payload in runs:
                if not payload:
                    header = self._send_halves[first : first + count]
                    header[:, 0] = length
                    header[:, 1] = self._epoch
                    continue
                if filled is None:
                    halves = self._send_halves[first, : 2 * packets]
                    halves[0] = length
                    halves[1::2] = self._epoch
                    halves[2::2] = words
                    filled, first, count = first, first + 1, count - 1
                if count:
                    self._send[first : first + count, :packets] = self._send[filled, :packets]
        for first, count, payload in runs:
            copy = self._round_copy(True, parity_offset, first, count)
            copy.WidthInBytes = 8 * (packets if payload else 1)
            self._check_cu(self._cuda.cuMemcpy2DAsync(copy, self._stream.cuda_stream))

    @staticmethod
    def _epochs(packets: np.ndarray) -> np.ndarray:
        """The epoch halves of ``packets`` (a view; the last axis must be contiguous)."""
        return packets.view(np.uint32)[..., 1::2]

    def exchange(
        self,
        payload: bytes,
        readers: Optional[Sequence[int]] = None,
        recipients: Optional[Sequence[int]] = None,
        idle: bool = False,
    ) -> list[Optional[bytes]]:
        """Send ``payload`` to every rank and return every rank's payload, in rank order.

        ``readers``: the ranks whose payloads this rank needs (default all); the others, every rank outside
        ``live_mask``, and every rank whose payload a membership change cut short come back as None, and their data
        is not copied to the host. ``recipients``: the ranks that need this rank's payload (default all), which must
        include every rank that names this rank among its ``readers``; only their rows get the payload, every row its
        length. ``idle``: this rank may wait long here, as when the executor has no work and a peer parks until the
        next request: the wait sleeps between polls (see ``_MAX_SLEEP_S``).
        """
        reader_mask = self.live_mask
        if readers is not None:
            reader_mask &= sum(1 << r for r in readers)
        if recipients is not None:
            recipients = sum(1 << r for r in recipients)
        ranks = self._ranks_of(reader_mask)[1]
        rounds = self._rounds(payload, reader_mask, recipients, idle)
        lengths, rows = next(rounds)
        received = self._chunks(rows, lengths, ranks)
        later = [self._chunks(rows, lengths, ranks) for lengths, rows in rounds]
        if later:
            received = [b"".join(parts) for parts in zip(received, *later)]
        payloads = [None] * self.nranks
        for r, received_payload in zip(ranks, received):
            if not self._lost >> r & 1:
                payloads[r] = received_payload
        return payloads

    def allgather_array(self, row: np.ndarray) -> np.ndarray:
        """Every rank's ``row`` as ``[nranks, row.size]``, sent as raw bytes and decoded for all ranks at once.

        ``row`` has the same dtype and size on every rank. Rows of the ranks outside ``live_mask`` and of the ranks
        whose payload a membership change cut short are zero.

        Raises:
            ValueError: The ranks passed rows of different sizes.
        """
        row = np.ascontiguousarray(row).reshape(-1)
        data = row.view(np.uint8)
        readers = self.live_mask
        out = np.zeros((self.nranks, -(-data.size // 4)), dtype=np.uint32)
        step = 4 * self._words
        for k, (lengths, rows) in enumerate(self._rounds(data.tobytes(), readers)):
            if k == 0 and (lengths[self._ranks_of(readers & ~self._lost)[0]] != data.size).any():
                raise ValueError(
                    f"allgather_array rows differ in size across ranks: {lengths.tolist()}"
                )
            if rows is not None:
                words = min(-(-min(step, data.size - step * k) // 4), rows.shape[1] - 1)
                out[:, self._words * k : self._words * k + words] = rows.view(np.uint32)[
                    :, 2 : 2 + 2 * words : 2
                ]
        kept = readers & ~self._lost
        if kept != (1 << self.nranks) - 1:
            out[~self._ranks_of(kept)[0]] = 0
        return out.view(np.uint8)[:, : data.size].view(row.dtype)

    def _ranks_of(self, mask: int) -> Tuple[np.ndarray, List[int]]:
        """Whether each rank is in ``mask`` (a bool array) and the ranks in it in increasing order; computed once per
        mask (live masks and the readers of a few collectives)."""
        ranks = self._masks.get(mask)
        if ranks is None:
            in_mask = np.array([mask >> r & 1 for r in range(self.nranks)], dtype=bool)
            ranks = self._masks[mask] = (in_mask, np.flatnonzero(in_mask).tolist())
        return ranks

    @staticmethod
    def _mask_of(ranks: np.ndarray) -> int:
        """The mask of the ranks a bool array marks."""
        return sum(1 << int(r) for r in np.flatnonzero(ranks))

    def _chunks(
        self, rows: Optional[np.ndarray], lengths: np.ndarray, ranks: List[int]
    ) -> List[bytes]:
        """For each of ``ranks``, the first ``lengths[r]`` bytes (at most one slot) of its payload in a round's
        packets."""
        if rows is None or not ranks:
            return [b""] * len(ranks)
        first = ranks[0]
        row_bytes = 4 * (rows.shape[1] - 1)
        data = np.ascontiguousarray(rows[first : ranks[-1] + 1].view(np.uint32)[:, 2::2]).tobytes()
        ends = (np.minimum(lengths, row_bytes) + row_bytes * (self._rank_index - first)).tolist()
        return [data[row_bytes * (r - first) : ends[r]] for r in ranks]

    def _rounds(
        self, data: bytes, readers: int, recipients: Optional[int] = None, idle: bool = False
    ) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """The rounds of an exchange of ``data``, one slot of it per round: each round's header lengths and packets
        (see :meth:`_round`). The first round's headers hold every rank's total length, whose largest sets how many
        rounds follow; a later round's headers hold the lengths of its chunks."""
        self._round_readers = readers
        self._lost = 0
        step = 4 * self._words
        lengths, rows = self._round(data[:step], len(data), readers, recipients, idle)
        yield lengths, rows
        for k in range(1, -(-int(lengths.max()) // step)):
            chunk = data[step * k : step * (k + 1)]
            yield self._round(chunk, len(chunk), readers, recipients, idle)

    def _round(
        self,
        chunk: bytes,
        length: int,
        readers: int,
        recipients: Optional[int] = None,
        idle: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """One exchange of up to one slot per rank: returns every rank's header length and the round's packets.
        ``readers``: mask of the ranks whose payloads this rank reads; ``recipients``: mask of the ranks whose rows
        get the payload (default all)."""
        while True:
            try:
                return self._try_round(chunk, length, readers, recipients, idle)
            except _RoundRestarted:
                pass

    def _try_round(
        self, chunk: bytes, length: int, readers: int, recipients: Optional[int], idle: bool
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Every rank's header length (0 for a rank that sent nothing) and the packets received, ``[nranks, width]``
        (None when no reader sent a payload; only the readers' rows are valid)."""
        # A membership change is pending: stop before the next round.
        if self.interrupted():
            self._stop(next_epoch(self._epoch))
        self._epoch = next_epoch(self._epoch)
        self._missing = 0
        parity_offset = (self._epoch & 1) * self._parity_bytes
        words = np.frombuffer(chunk + b"\0" * (-len(chunk) % 4), dtype=np.uint32)
        self._send_round(parity_offset, words, length, recipients)
        # The ranks this round started with, also when a change finishes it from the peers' rows.
        round_mask = self.live_mask
        round_live, live_ranks = self._ranks_of(round_mask)
        headers = self._wait(
            parity_offset,
            round_live[:, None],
            live_ranks,
            1 + min(self._words, self._POLL_PACKETS) if readers else 1,
            idle,
        )
        # A rank whose header a change cut short sends nothing in this round.
        live = round_mask & ~self._missing
        lengths = headers.view(np.uint32)[:, 0] * self._ranks_of(live)[0]
        # A header holds the sender's total length in the first round and its chunk's length in later ones.
        reader_live, reader_ranks = self._ranks_of(readers & live)
        # Signed: -1 marks a rank this rank does not read (in uint32 it would be the largest length).
        payload_bytes = np.where(reader_live, lengths.astype(np.int64), -1)
        packets = 1 + min(self._words, -(-int(payload_bytes.max()) // 4))
        rows = None
        if packets > 1:
            # Packet i of a reader's slot holds its payload bytes from 4 * (i - 1) on.
            wanted = self._packet_offsets[:packets] < payload_bytes[:, None]
            rows = (
                headers
                if self._arrived(headers, wanted)
                else self._wait(parity_offset, wanted, reader_ranks, 1, idle)
            )
        self._lost |= readers & (~round_mask | self._missing)
        if self._epoch_after_round is not None:  # this round was finished from the peers' rows
            self._epoch, self._epoch_after_round = self._epoch_after_round, None
            self._finished_round = None
        return lengths, rows

    @property
    def waiting_for(self) -> int:
        """Mask of the ranks the current wait waits for; 0 while not waiting. Read from any thread."""
        stale = self._stale_packets
        return 0 if stale is None else self._mask_of(stale.any(axis=1))

    def _stop(self, epoch: int) -> None:
        """Stop at round ``epoch`` for a membership change; returns once the change is committed."""
        self._stale_packets = None
        self.stopped_epoch = epoch
        self.on_interrupt()

    def _stale(self, epochs: np.ndarray, wanted: np.ndarray) -> np.ndarray:
        """The ``wanted`` packets (``[nranks, packets]``) whose epoch halves (``epochs``) are not the current epoch."""
        return (epochs[:, : wanted.shape[1]] != self._epoch) & wanted

    def _take_finished_round(self, wanted: np.ndarray) -> np.ndarray:
        """The round finished from the peers' rows; the ranks whose wanted packets it lacks become missing."""
        rows = self._finished_round[:, : wanted.shape[1]]
        self._missing |= self._mask_of(self._stale(self._epochs(rows), wanted).any(axis=1))
        return rows

    def _arrived(self, rows: np.ndarray, wanted: np.ndarray) -> bool:
        """Whether ``rows`` holds every ``wanted`` packet with the current epoch."""
        return (
            rows.shape[1] >= wanted.shape[1] and not self._stale(self._epochs(rows), wanted).any()
        )

    def _wait(
        self,
        parity_offset: int,
        wanted: np.ndarray,
        waited: List[int],
        width: int,
        idle: bool,
    ) -> np.ndarray:
        """Copy the slots of the ``waited`` ranks (those with packets to wait for, in increasing order) in the slot set
        at ``parity_offset`` to the host until every ``wanted`` packet (``[nranks, packets]``, each rank's first ones)
        carries the current epoch. Each poll copies at least ``width`` packets per rank, of the ranks from the first
        to the last waited one, and stops for a pending membership change. ``idle``: sleep between polls."""
        if self._finished_round is not None:
            return self._take_finished_round(wanted)
        cuda = self._cuda
        first = waited[0]
        width = max(width, wanted.shape[1])
        poll = self._round_copy(False, parity_offset, first, waited[-1] + 1 - first)
        poll.WidthInBytes = 8 * width
        rows = self._recv[:, :width]
        epochs = self._recv_epochs[:, : wanted.shape[1]]
        sleep_s = 1e-6
        while True:
            self._check_cu(cuda.cuMemcpy2DAsync(poll, self._stream.cuda_stream))
            self._stream.synchronize()
            stale = self._stale(epochs, wanted)
            if not stale.any():
                self._stale_packets = None
                return rows
            self._stale_packets = stale
            # Checked right after a poll that found the round incomplete.
            if self.interrupted():
                self._stop(self._epoch)
                if self._finished_round is not None:
                    return self._take_finished_round(wanted)
                raise _RoundRestarted
            if idle:
                time.sleep(sleep_s)
                sleep_s = min(2 * sleep_s, self._MAX_SLEEP_S)


class MnnvlDist(MPIDist):
    """``MPIDist`` whose host collectives go through MNNVL memory instead of MPI.

    Used with WideEP fault tolerance on pure attention-DP layouts, where the TP
    group spans the world. The collectives run as rounds of a
    ``TPHostExchange``: every rank copies its payload into the receive buffers
    of the ranks that read it and reads its own, with no MPI call. The host
    exchange buffers are allocated on first use, which every rank reaches in
    step. Ranks outside the exchange's live mask are absent: object gathers
    return None in their place, reductions skip them, and
    ``tp_allgather_int64`` gives them zero rows, so rows stay indexed by rank
    and counts summed over them stay true; votes take ``tp_live_rows``. Every
    live rank gets the same result from a collective that a membership change
    interrupted. During the startup, rank 0 records the collectives, and a
    replacement rank, whose exchange reaches only itself, takes their results
    from the record (``startup_record``). Methods not overridden here, such as
    point-to-point calls and the PP/CP groups, stay on MPI.
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

    # broadcast_or_none is one exchange: the root's pickled None stands for
    # "nothing".
    supports_single_round_broadcast = True

    def __init__(self, mapping: Mapping):
        super().__init__(mapping)
        self._host_exchange = None
        # (live mask, rows of the live ranks) of the last tp_live_rows call.
        self._live_rows_index = (None, None)

    @staticmethod
    def unmet_requirements(mapping: Mapping) -> List[str]:
        """What the layout lacks of pure attention DP (one TP group over the
        world), or else what the GPUs lack of MNNVL memory shared through
        fabric handles (MnnvlRows); empty if both are met."""
        unmet = [
            requirement
            for supported, requirement in (
                (mapping.enable_attention_dp, "attention DP (enable_attention_dp=True)"),
                (mapping.pp_size == 1, "pipeline_parallel_size=1"),
                (mapping.cp_size == 1, "context_parallel_size=1"),
                (
                    mapping.tp_size == mapping.world_size,
                    "tensor_parallel_size equal to the world size",
                ),
            )
            if not supported
        ]
        if unmet:
            return unmet
        if not MnnvlMemory.supports_mnnvl():
            return ["GPUs in one multi-node NVLink (MNNVL) domain, such as a GB200 NVL72"]
        if not MnnvlRows.uses_fabric_handles(torch.cuda.current_device()):
            return [
                "MNNVL memory shared through fabric handles (GB200-class "
                "aarch64 nodes with IMEX running and an IMEX channel in the "
                "container)"
            ]
        return []

    @staticmethod
    def supports(mapping: Mapping) -> bool:
        """Whether the layout and the GPUs meet ``unmet_requirements``."""
        return not MnnvlDist.unmet_requirements(mapping)

    @property
    def local_world_size(self):
        """The ranks on this rank's node, from the mapping: a replacement
        rank's MPI world is only itself."""
        mapping = self.mapping
        return sum(
            mapping.get_node_rank(rank) == mapping.node_rank for rank in range(self.world_size)
        )

    @property
    def host_exchange(self):
        """The ``TPHostExchange`` under the collectives; collective on first
        use."""
        if self._host_exchange is None:
            self._host_exchange = TPHostExchange.create(self.mapping)
        return self._host_exchange

    def _exchange(self, obj, readers=None, recipients=None, idle=False, bound=None) -> list:
        """Every rank's pickled ``obj`` (only live ``readers``' are unpickled);
        the other arguments are ``TPHostExchange.exchange``'s, and ``bound``
        ``StartupRecord.exchange``'s."""
        host_exchange = self.host_exchange
        payload = pickle.dumps(obj)
        if not startup_record.active:
            payloads = host_exchange.exchange(payload, readers, recipients, idle)
        else:

            def run():
                return host_exchange.exchange(payload, readers, recipients, idle)

            payloads = startup_record.exchange(self.mapping.tp_rank, payload, readers, run, bound)
        return [
            None if payload is None else pickle.loads(payload)  # nosec B301
            for payload in payloads
        ]

    def broadcast(self, obj, root=0, chunk_size: int = 4 * 1024 * 1024, prefer_cpu=False):
        # prefer_cpu marks the idle path, where the ranks may wait long for
        # each other: a long wait there sleeps between polls; others spin.
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

    def tp_allgather(self, obj, chunk_size: int = 4 * 1024 * 1024, *, small_payload: bool = False):
        return self._exchange(obj)

    def tp_gather(self, obj, root=0, chunk_size: int = 4 * 1024 * 1024):
        if self.mapping.tp_rank != root:
            self._exchange(obj, readers=(), recipients=(root,))
            return None
        return self._exchange(obj, recipients=(root,))

    def tp_broadcast(self, obj, root=0, chunk_size: int = 4 * 1024 * 1024, **kwargs):
        return self.broadcast(obj, root)

    def allreduce(self, obj, op: ReduceOp = ReduceOp.SUM):
        _check_reduce_op(op)
        bound = self._REDUCE[op] if op in (ReduceOp.MIN, ReduceOp.MAX) else None
        values = [value for value in self._exchange(obj, bound=bound) if value is not None]
        result = values[0]
        for value in values[1:]:
            result = self._REDUCE[op](result, value)
        return result

    def tp_allreduce(self, obj, op: ReduceOp = ReduceOp.SUM):
        return self.allreduce(obj, op)

    def tp_allgather_int64(self, values: ArrayLike) -> np.ndarray:
        host_exchange = self.host_exchange
        values = np.asarray(values, dtype=np.int64)
        if not startup_record.active:
            return host_exchange.allgather_array(values)

        def run():
            return [row.tobytes() for row in host_exchange.allgather_array(values)]

        rows = startup_record.exchange(self.mapping.tp_rank, values.tobytes(), None, run)
        return np.stack([np.frombuffer(row, dtype=np.int64) for row in rows])

    def tp_live_rows(self, rows: np.ndarray) -> np.ndarray:
        live_mask = self.host_exchange.live_mask
        if live_mask == (1 << self.tp_size) - 1 or startup_record.replaying:
            return rows  # every rank takes part
        if self._live_rows_index[0] != live_mask:
            self._live_rows_index = (
                live_mask,
                np.array([r for r in range(self.tp_size) if live_mask >> r & 1], dtype=np.intp),
            )
        return rows[self._live_rows_index[1]]

    def broadcast_int64(
        self, values: ArrayLike, root: int = 0, *, prefer_cpu: bool = False
    ) -> np.ndarray:
        row = np.ascontiguousarray(np.asarray(values, dtype=np.int64).reshape(-1))
        return self.broadcast(row, root, prefer_cpu=prefer_cpu)

    def tp_cp_broadcast_int64(self, values: ArrayLike, root: int = 0) -> np.ndarray:
        return self.broadcast_int64(values, root)

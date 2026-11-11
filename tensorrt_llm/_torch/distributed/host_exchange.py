# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host collectives over MNNVL memory for Wide-EP fault tolerance."""

from __future__ import annotations

import pickle  # nosec B403
import time
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from numpy.typing import ArrayLike

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.math_utils import ceil_div

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


def _all_ranks(nranks: int) -> int:
    """Return the rank mask of ranks ``0`` to ``nranks - 1``."""
    return (1 << nranks) - 1


def _has_rank(mask: int, rank: int) -> bool:
    return bool(mask >> rank & 1)


def _rank_mask(ranks: Iterable[int]) -> int:
    return sum(1 << rank for rank in ranks)


class HostExchangeMnnvlMemory(MnnvlMemory):
    """MNNVL memory of one row per rank, every row mapped in every process at ``ptr + r * stride``.

    The rows sit back to back, so one strided copy reaches every row. The row of an absent rank maps local scratch
    memory. The rows a former member may still write to are remapped onto fresh pages, so its late writes land in
    pages that no rank maps.
    """

    absent_rows_on_scratch = True

    @classmethod
    def uses_fabric_handles(cls, dev_id: int) -> bool:
        """Return whether device ``dev_id`` shares memory through fabric handles."""
        handle_type = cls.get_allocation_prop(dev_id).requestedHandleTypes
        return handle_type == mnnvl_memory.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC

    def __init__(self, rank: int, nranks: int, nbytes: int, gather: Callable[[Any], Sequence[Any]]):
        """Reserve and map the rows, collectively over the ranks ``gather`` reaches.

        Raises:
            MnnvlPeerHandleError: An active rank could not import a peer's handle.
            MnnvlMappingError: Mapping or zeroing failed on some active ranks.
            RuntimeError: The device does not share memory through fabric handles.
        """
        cls = type(self)
        dev_id = cls._init_device()
        if not cls.uses_fabric_handles(dev_id):
            raise RuntimeError(
                "HostExchangeMnnvlMemory requires fabric memory handles, which this device does not use"
            )
        granularity = cls.get_allocation_granularity(dev_id)
        stride = granularity * ceil_div(nbytes, granularity)
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
            self.remap(gather, frozenset(range(nranks)))
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


class _RoundRestarted(Exception):
    """Raised when a membership change requires the current round to run again."""


class TPHostExchange:
    """All-to-all of byte strings between the host processes of a TP group, over one row of MNNVL memory per rank.

    The exchange uses the LL protocol of the all-to-all kernels through the copy engines, so it takes no SMs. Each
    exchange alternates between two slot sets, so a rank never overwrites a slot that a slower rank still reads.
    Every active rank gets the same result across a membership change. The ranks finish a round that some rank
    finished from that rank's packets, and run any other round again.
    """

    # Sized so that a step's request broadcast and response gather usually take one round.
    DEFAULT_SLOT_PAYLOAD_BYTES = 64 * 1024
    _POLL_PACKETS = 256
    # An idle wait spins for _IDLE_SPIN_S and sleeps _IDLE_SLEEP_S between later polls, about the cost of one poll.
    _IDLE_SPIN_S = 20e-6
    _IDLE_SLEEP_S = 20e-6

    def __init__(
        self, rank: int, nranks: int, rows: HostExchangeMnnvlMemory, slot_payload_bytes: int
    ):
        from .mnnvl_memory import _check_cu_result, cuda

        self._cuda = cuda
        self._check_cu = _check_cu_result
        self.rank = rank
        self.nranks = nranks
        self.rows = rows
        self._words = slot_payload_bytes // 4
        self._slot_bytes = 8 * (1 + self._words)
        self._parity_bytes = nranks * self._slot_bytes
        # Epoch 0 is never sent because the buffers start zeroed.
        self._epoch = 0
        self.active_mask = _all_ranks(nranks)
        self.interrupted: Callable[[], bool] = lambda: False
        self.on_interrupt: Optional[Callable[[], None]] = None
        self.stopped_epoch = 0
        self._stale_packets: Optional[np.ndarray] = None
        self._missing = 0
        self._failed = 0
        self._round_readers = 0
        self._finished_round: Optional[np.ndarray] = None
        self._epoch_after_round: Optional[int] = None
        self._stream = torch.cuda.Stream()
        # One outgoing slot per peer, because a 2D copy needs a source pitch of at least its width.
        self._send = self._pinned((nranks, 1 + self._words))
        self._recv = self._pinned((nranks, 1 + self._words))
        self._recv_epochs = self._epochs(self._recv)
        self._send_halves = self._send.view(np.uint32)
        self._copies: Dict[Tuple[bool, int, int, int], object] = {}
        self._runs: Dict[int, Tuple[Tuple[int, int, bool], ...]] = {}
        self._runs_to_all = ((0, nranks, True),)
        self._masks: Dict[int, Tuple[np.ndarray, List[int]]] = {}
        self._packet_offsets = 4 * np.arange(1 + self._words) - 4
        self._packet_offsets[0] = -1
        self._rank_index = np.arange(nranks)

    @staticmethod
    def _pinned(shape: Tuple[int, int]) -> np.ndarray:
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
        """Create the exchange of the TP group, collectively over the ranks its communicator reaches."""
        comm = TpMnnvlMemory.get_comm(mapping)
        active_mask = _rank_mask(rank for rank, active in enumerate(comm.allgather(True)) if active)
        rows = HostExchangeMnnvlMemory(
            mapping.tp_rank,
            mapping.tp_size,
            cls.buffer_bytes(mapping.tp_size, cls.DEFAULT_SLOT_PAYLOAD_BYTES),
            comm.allgather,
        )
        exchange = cls(mapping.tp_rank, mapping.tp_size, rows, cls.DEFAULT_SLOT_PAYLOAD_BYTES)
        exchange.active_mask = active_mask
        return exchange

    @property
    def epoch(self) -> int:
        return self._epoch

    def admit(self, active_mask: int, epoch: int) -> None:
        """Continue with the ranks in ``active_mask`` from ``epoch``, in step on every rank in ``active_mask``."""
        self.active_mask = active_mask
        self._epoch = epoch - 1

    def install(
        self,
        gather: Callable[[Any], Sequence[Any]],
        active_mask: int,
        epoch: int,
        fresh_rows: frozenset[int],
    ) -> None:
        """Switch to the ranks in ``active_mask`` while stopped, collectively over the ranks ``gather`` reaches.

        ``epoch`` follows the latest ``stopped_epoch`` among the active ranks. A rank that stopped one round behind
        finishes that round from its peers' rows. The rows in ``fresh_rows`` get fresh pages.

        Raises:
            RuntimeError: This rank stopped more than one round before the latest active rank.
        """
        if not _has_rank(active_mask, self.rank):
            pass
        elif next_epoch(self.stopped_epoch) == epoch:
            # An earlier attempt of this change may have finished the round with a since-failed rank's packets.
            self._finished_round = self._epoch_after_round = None
        elif self._finished_round is None:
            if (
                next_epoch(next_epoch(self.stopped_epoch)) != epoch
                or self.stopped_epoch != self._epoch
            ):
                raise RuntimeError(
                    f"host exchange stopped at epoch {self.stopped_epoch}, more than one round before the "
                    f"latest active rank (epoch {epoch} is next)"
                )
            # Read before the remap, which may release the peers' rows.
            self._finished_round = self._finish_from_peers(active_mask)
        round_epoch = self._epoch
        self.rows.remap(gather, fresh_rows)
        self.admit(active_mask, epoch)
        if self._finished_round is not None:
            self._epoch_after_round, self._epoch = self._epoch, round_epoch

    def _finish_from_peers(self, active_mask: int) -> np.ndarray:
        """Return this rank's current round, taking each slot it lacks from a peer row that holds it."""
        parity_offset = (self._epoch & 1) * self._parity_bytes
        rows = self._pinned((self.nranks, 1 + self._words))
        self._copy_to_host(self.rows.row(self.rank) + parity_offset, rows)
        epochs = self._epochs(rows)
        peers = [p for p in range(self.nranks) if p != self.rank and _has_rank(active_mask, p)]
        for sender in range(self.nranks):
            reads = _has_rank(self._round_readers, sender)
            if not _has_rank(self.active_mask, sender) or (
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
                    rows[sender] = slot[0]
                    if not reads:
                        break
        return rows

    def _slot_complete(self, slot: np.ndarray) -> bool:
        """Return whether ``slot`` holds its header and every payload packet of the current epoch."""
        length, epoch = slot[:1].view(np.uint32).tolist()
        if epoch != self._epoch:
            return False
        packets = 1 + min(self._words, ceil_div(length, 4))
        return bool((self._epochs(slot[:packets]) == self._epoch).all())

    def _slot_copy(self, send: bool, host: int, device: int, count: int, device_pitch: int):
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
        copy = self._slot_copy(False, dst.ctypes.data, src, len(dst), self._slot_bytes)
        self._check_cu(self._cuda.cuMemcpy2DAsync(copy, self._stream.cuda_stream))
        self._stream.synchronize()

    def _round_copy(self, send: bool, parity_offset: int, first: int, count: int):
        """Return the cached copy of ``count`` ranks' slots, into their rows if ``send``, else to the host."""
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
        """Return ``(first rank, ranks, with payload)`` per run of consecutive ranks alike in ``recipients``."""
        runs = self._runs.get(recipients)
        if runs is None:
            full = [_has_rank(recipients, r) for r in range(self.nranks)]
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
        """Send the header to every row and the payload to the rows of ``recipients`` (default all)."""
        packets = 1 + words.size
        if words.size <= self._POLL_PACKETS:
            # A payload that a header poll copies whole goes to every row.
            runs = self._runs_to_all
            halves = self._send_halves[:, : 2 * packets]
            halves[:, 0] = length
            halves[:, 1::2] = self._epoch
            halves[:, 2::2] = words
        else:
            runs = self._runs_to_all if recipients is None else self._send_runs(recipients)
            filled = None
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
        """Return a view of the epoch halves of ``packets``."""
        return packets.view(np.uint32)[..., 1::2]

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
            readers: The ranks whose payloads this rank needs (default all). The others come back as None, as do
                ranks outside ``active_mask`` and ranks whose payload a membership change left incomplete.
            recipients: The ranks that need this rank's payload (default all). It must include every rank that
                names this rank among its ``readers``.
            idle: Sleep between polls, for waits that may last until the next request.
        """
        reader_mask = self.active_mask
        if readers is not None:
            reader_mask &= _rank_mask(readers)
        if recipients is not None:
            recipients = _rank_mask(recipients)
        ranks = self._ranks_of(reader_mask)[1]
        rounds = self._rounds(payload, reader_mask, recipients, idle)
        lengths, rows = next(rounds)
        received = self._chunks(rows, lengths, ranks)
        later = [self._chunks(rows, lengths, ranks) for lengths, rows in rounds]
        if later:
            received = [b"".join(parts) for parts in zip(received, *later)]
        payloads = [None] * self.nranks
        for r, received_payload in zip(ranks, received):
            if not _has_rank(self._failed, r):
                payloads[r] = received_payload
        return payloads

    def allgather_array(self, row: np.ndarray) -> np.ndarray:
        """Return every rank's ``row`` as ``[nranks, row.size]``, with zero rows for absent ranks.

        Raises:
            ValueError: The ranks passed rows of different sizes.
        """
        row = np.ascontiguousarray(row).reshape(-1)
        data = row.view(np.uint8)
        readers = self.active_mask
        out = np.zeros((self.nranks, ceil_div(data.size, 4)), dtype=np.uint32)
        step = 4 * self._words
        for k, (lengths, rows) in enumerate(self._rounds(data.tobytes(), readers)):
            if k == 0 and (lengths[self._ranks_of(readers & ~self._failed)[0]] != data.size).any():
                raise ValueError(
                    f"allgather_array rows differ in size across ranks: {lengths.tolist()}"
                )
            if rows is not None:
                words = min(ceil_div(min(step, data.size - step * k), 4), rows.shape[1] - 1)
                out[:, self._words * k : self._words * k + words] = rows.view(np.uint32)[
                    :, 2 : 2 + 2 * words : 2
                ]
        kept = readers & ~self._failed
        if kept != _all_ranks(self.nranks):
            out[~self._ranks_of(kept)[0]] = 0
        return out.view(np.uint8)[:, : data.size].view(row.dtype)

    def _ranks_of(self, mask: int) -> Tuple[np.ndarray, List[int]]:
        """Return a bool array of the ranks in ``mask`` and their sorted list, cached per mask."""
        ranks = self._masks.get(mask)
        if ranks is None:
            in_mask = np.array([_has_rank(mask, r) for r in range(self.nranks)], dtype=bool)
            ranks = self._masks[mask] = (in_mask, np.flatnonzero(in_mask).tolist())
        return ranks

    @staticmethod
    def _mask_of(ranks: np.ndarray) -> int:
        return _rank_mask(int(r) for r in np.flatnonzero(ranks))

    def _chunks(
        self, rows: Optional[np.ndarray], lengths: np.ndarray, ranks: List[int]
    ) -> List[bytes]:
        """Return the payload bytes of ``ranks`` in one round's packets."""
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
        """Yield the header lengths and packets of each round, one slot of ``data`` per round."""
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
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Exchange up to one slot per rank, running the round again if a membership change restarts it."""
        while True:
            try:
                return self._try_round(chunk, length, readers, recipients, idle)
            except _RoundRestarted:
                pass

    def _try_round(
        self, chunk: bytes, length: int, readers: int, recipients: Optional[int], idle: bool
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return every rank's header length and the packets received, or None if no reader sent a payload."""
        if self.interrupted():
            self._stop(next_epoch(self._epoch))
        self._epoch = next_epoch(self._epoch)
        self._missing = 0
        parity_offset = (self._epoch & 1) * self._parity_bytes
        words = np.frombuffer(chunk + b"\0" * (-len(chunk) % 4), dtype=np.uint32)
        self._send_round(parity_offset, words, length, recipients)
        round_mask = self.active_mask
        round_active, active_ranks = self._ranks_of(round_mask)
        headers = self._wait(
            parity_offset,
            round_active[:, None],
            active_ranks,
            1 + min(self._words, self._POLL_PACKETS) if readers else 1,
            idle,
        )
        active = round_mask & ~self._missing
        lengths = headers.view(np.uint32)[:, 0] * self._ranks_of(active)[0]
        reader_active, reader_ranks = self._ranks_of(readers & active)
        # Signed, because -1 marks a rank this rank does not read.
        payload_bytes = np.where(reader_active, lengths.astype(np.int64), -1)
        packets = 1 + min(self._words, ceil_div(int(payload_bytes.max()), 4))
        rows = None
        if packets > 1:
            wanted = self._packet_offsets[:packets] < payload_bytes[:, None]
            rows = (
                headers
                if self._arrived(headers, wanted)
                else self._wait(parity_offset, wanted, reader_ranks, 1, idle)
            )
        self._failed |= readers & (~round_mask | self._missing)
        if self._epoch_after_round is not None:
            self._epoch, self._epoch_after_round = self._epoch_after_round, None
            self._finished_round = None
        return lengths, rows

    @property
    def waiting_for(self) -> int:
        """Return the mask of the ranks the current wait waits for. Any thread may read it."""
        stale = self._stale_packets
        return 0 if stale is None else self._mask_of(stale.any(axis=1))

    def _stop(self, epoch: int) -> None:
        self._stale_packets = None
        self.stopped_epoch = epoch
        self.on_interrupt()

    def _stale(self, epochs: np.ndarray, wanted: np.ndarray) -> np.ndarray:
        return (epochs[:, : wanted.shape[1]] != self._epoch) & wanted

    def _take_finished_round(self, wanted: np.ndarray) -> np.ndarray:
        rows = self._finished_round[:, : wanted.shape[1]]
        self._missing |= self._mask_of(self._stale(self._epochs(rows), wanted).any(axis=1))
        return rows

    def _arrived(self, rows: np.ndarray, wanted: np.ndarray) -> bool:
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
        """Poll the slots of the ``waited`` ranks until every ``wanted`` packet carries the current epoch."""
        if self._finished_round is not None:
            return self._take_finished_round(wanted)
        cuda = self._cuda
        first = waited[0]
        width = max(width, wanted.shape[1])
        poll = self._round_copy(False, parity_offset, first, waited[-1] + 1 - first)
        poll.WidthInBytes = 8 * width
        rows = self._recv[:, :width]
        epochs = self._recv_epochs[:, : wanted.shape[1]]
        spin_end = time.monotonic() + self._IDLE_SPIN_S if idle else 0.0
        while True:
            self._check_cu(cuda.cuMemcpy2DAsync(poll, self._stream.cuda_stream))
            self._stream.synchronize()
            stale = self._stale(epochs, wanted)
            if not stale.any():
                self._stale_packets = None
                return rows
            self._stale_packets = stale
            if self.interrupted():
                self._stop(self._epoch)
                if self._finished_round is not None:
                    return self._take_finished_round(wanted)
                raise _RoundRestarted
            if idle and time.monotonic() >= spin_end:
                time.sleep(self._IDLE_SLEEP_S)


class MnnvlDist(MPIDist):
    """``MPIDist`` whose host collectives go through MNNVL memory instead of MPI.

    Object gathers return None for absent ranks, reductions skip them, and
    ``tp_allgather_int64`` gives them zero rows.
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

    supports_single_round_broadcast = True

    def __init__(self, mapping: Mapping):
        super().__init__(mapping)
        self._host_exchange = None
        self._active_rows_index = (None, None)

    @staticmethod
    def unmet_requirements(mapping: Mapping) -> List[str]:
        """Return the unmet layout or MNNVL requirements, or an empty list if all are met."""
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
        if not HostExchangeMnnvlMemory.uses_fabric_handles(torch.cuda.current_device()):
            return [
                "MNNVL memory shared through fabric handles (GB200-class "
                "aarch64 nodes with IMEX running and an IMEX channel in the "
                "container)"
            ]
        return []

    @staticmethod
    def supports(mapping: Mapping) -> bool:
        return not MnnvlDist.unmet_requirements(mapping)

    @property
    def local_world_size(self):
        # From the mapping, because a replacement rank's MPI world is only itself.
        mapping = self.mapping
        return sum(
            mapping.get_node_rank(rank) == mapping.node_rank for rank in range(self.world_size)
        )

    @property
    def host_exchange(self):
        """Return the ``TPHostExchange``, created collectively on first use."""
        if self._host_exchange is None:
            self._host_exchange = TPHostExchange.create(self.mapping)
        return self._host_exchange

    def _exchange(self, obj, readers=None, recipients=None, idle=False, bound=None) -> list:
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
        # prefer_cpu marks the idle path, where a long wait sleeps between polls.
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

    def tp_active_rows(self, rows: np.ndarray) -> np.ndarray:
        active_mask = self.host_exchange.active_mask
        if active_mask == _all_ranks(self.tp_size) or startup_record.replaying:
            return rows
        if self._active_rows_index[0] != active_mask:
            self._active_rows_index = (
                active_mask,
                np.array(
                    [r for r in range(self.tp_size) if _has_rank(active_mask, r)], dtype=np.intp
                ),
            )
        return rows[self._active_rows_index[1]]

    def broadcast_int64(
        self, values: ArrayLike, root: int = 0, *, prefer_cpu: bool = False
    ) -> np.ndarray:
        row = np.ascontiguousarray(np.asarray(values, dtype=np.int64).reshape(-1))
        return self.broadcast(row, root, prefer_cpu=prefer_cpu)

    def tp_cp_broadcast_int64(self, values: ArrayLike, root: int = 0) -> np.ndarray:
        return self.broadcast_int64(values, root)

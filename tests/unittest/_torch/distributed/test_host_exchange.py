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
"""Tests for the host collectives over MNNVL memory, ``TPHostExchange`` and ``MnnvlDist``.

The tests run the exchange over rows of host memory or over fake exchange rounds.
"""

import ctypes
import pickle
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.distributed import host_exchange, mnnvl_memory
from tensorrt_llm._torch.distributed.host_exchange import (
    HostExchangeMnnvlMemory,
    MnnvlDist,
    TPHostExchange,
)
from tensorrt_llm._torch.distributed.host_exchange_epoch import latest_epoch, next_epoch
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

WORLD = 4
SURVIVORS = 0b1011


def _ranks_in(mask, nranks=WORLD):
    """Return the ranks below ``nranks`` whose bit is set in ``mask``."""
    return [r for r in range(nranks) if mask & (1 << r)]


def _mnnvl_dist(exchange, **mapping):
    """Return an ``MnnvlDist`` that runs its collectives over ``exchange``."""
    dist = MnnvlDist(SimpleNamespace(**mapping))
    dist._host_exchange = exchange
    return dist


def test_int64_gather_zeroes_absent_ranks(monkeypatch):
    exchange = _host_exchange_group(monkeypatch, WORLD, _SLOT_BYTES)[0]
    exchange.active_mask = SURVIVORS

    def one_round(chunk, length, readers, recipients=None, idle=False):
        """Return a round in which every active rank r sent [r, 10 * r] and the absent rank's slot is stale."""
        readers = _ranks_in(readers)
        assert readers == _ranks_in(SURVIVORS)
        packets = np.full((WORLD, 5), 0xDEAD, dtype=np.uint64)
        for r in readers:
            packets[r, 0] = length
            packets[r, 1:] = np.array([r, 10 * r], dtype=np.int64).view(np.uint32)
        return np.array([length if r in readers else 0 for r in range(WORLD)], np.uint32), packets

    monkeypatch.setattr(exchange, "_round", one_round)
    dist = _mnnvl_dist(exchange, tp_size=WORLD)
    gathered = dist.tp_allgather_int64([0, 0])
    assert gathered.tolist() == [[0, 0], [1, 10], [0, 0], [3, 30]]
    assert dist.tp_active_rows(gathered).tolist() == [[0, 0], [1, 10], [3, 30]]
    monkeypatch.setattr(dist, "_exchange", lambda obj, readers=None, bound=None: [1, 2, None, 4])
    assert dist.allreduce(1) == 7


@pytest.mark.parametrize("fabric", [True, False])
def test_mnnvl_dist_needs_fabric_handles(monkeypatch, fabric):
    """Check that MnnvlDist is supported only where rows are shared through fabric handles."""
    from tensorrt_llm._torch.distributed import mnnvl_memory

    handle_types = mnnvl_memory.cuda.CUmemAllocationHandleType
    handle_type = (
        handle_types.CU_MEM_HANDLE_TYPE_FABRIC
        if fabric
        else handle_types.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    )
    monkeypatch.setattr(mnnvl_memory.MnnvlMemory, "supports_mnnvl", staticmethod(lambda: True))
    monkeypatch.setattr(
        HostExchangeMnnvlMemory,
        "get_allocation_prop",
        classmethod(lambda cls, dev_id: SimpleNamespace(requestedHandleTypes=handle_type)),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    mapping = SimpleNamespace(
        enable_attention_dp=True, pp_size=1, cp_size=1, tp_size=WORLD, world_size=WORLD
    )
    assert MnnvlDist.supports(mapping) == fabric


# The host exchange under a membership change, two ranks of one process over rows of host memory.


class _HostCopies:
    """Run the CUDA driver copies of TPHostExchange between host addresses.

    Each copy completes at once and whole, as the GPU writes a packet whole.
    """

    CUresult = SimpleNamespace(CUDA_SUCCESS=0)
    CUmemorytype = SimpleNamespace(CU_MEMORYTYPE_DEVICE="device", CU_MEMORYTYPE_HOST="host")

    def __init__(self):
        self._lock = threading.Lock()
        # Each entry is (source, destination, width, rows), with source and destination as (memory type, address).
        self.log = []

    @staticmethod
    def CUDA_MEMCPY2D():
        return SimpleNamespace()

    def cuMemcpy2DAsync(self, copy, stream):
        src = copy.srcHost if copy.srcMemoryType == "host" else copy.srcDevice
        dst = copy.dstHost if copy.dstMemoryType == "host" else copy.dstDevice
        with self._lock:
            self.log.append(
                (
                    (copy.srcMemoryType, src),
                    (copy.dstMemoryType, dst),
                    copy.WidthInBytes,
                    copy.Height,
                )
            )
            for row in range(copy.Height):
                ctypes.memmove(
                    dst + row * copy.dstPitch, src + row * copy.srcPitch, copy.WidthInBytes
                )
        return (0,)


class _HostRows:
    """Map HostExchangeMnnvlMemory rows in host memory as rank ``rank`` sees them.

    Rank r's row starts at ``ptr + r * stride``. Every rank's view shares ``memory``.
    """

    def __init__(self, memory, stride, rank):
        self._memory, self.stride, self.rank = memory, stride, rank
        self.ptr = memory.ctypes.data

    def row(self, rank):
        return self.ptr + rank * self.stride

    def remap(self, gather, fresh_rows):
        """Zero this rank's row if it gets fresh pages. The memory is shared, so ``gather`` is unused."""
        if self.rank in fresh_rows:
            self._memory[self.rank * self.stride : (self.rank + 1) * self.stride] = 0

    def close(self):
        pass


_SLOT_BYTES = 64
# Slots longer than a header poll copies (TPHostExchange._POLL_PACKETS payload packets).
_LONG_SLOT_BYTES = 4 * 1024


def _host_exchange_group(monkeypatch, nranks, slot_bytes):
    """Return a TPHostExchange for each of ``nranks`` ranks over shared host rows."""
    monkeypatch.setattr(mnnvl_memory, "cuda", _HostCopies())
    monkeypatch.setattr(host_exchange, "prefer_pinned", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "Stream", lambda: SimpleNamespace(cuda_stream=0, synchronize=lambda: None)
    )
    stride = TPHostExchange.buffer_bytes(nranks, slot_bytes)
    memory = np.zeros(nranks * stride, dtype=np.uint8)
    return [
        TPHostExchange(rank, nranks, _HostRows(memory, stride, rank), slot_bytes)
        for rank in range(nranks)
    ]


@pytest.fixture
def host_exchanges(monkeypatch):
    return _host_exchange_group(monkeypatch, 2, _SLOT_BYTES)


def _run_ranks(exchanges, calls):
    """Run ``calls[rank](exchanges[rank])`` for every rank in its own thread and return the results by rank."""
    results, errors = {}, []

    def run(rank):
        try:
            results[rank] = calls[rank](exchanges[rank])
        except BaseException as error:  # surface thread failures in the test
            errors.append(error)

    ranks = [threading.Thread(target=run, args=(rank,), daemon=True) for rank in range(len(calls))]
    for thread in ranks:
        thread.start()
    for thread in ranks:
        thread.join(30)
    assert not errors, errors
    assert not any(thread.is_alive() for thread in ranks)
    return results


@pytest.mark.timeout(60)
@pytest.mark.parametrize("idle", [False, True])
def test_only_an_idle_wait_sleeps(host_exchanges, monkeypatch, idle):
    """Check that only an idle wait sleeps between polls.

    A busy wait for a late peer spins, and it still reports the rank it waits for.
    """
    sleeps = []
    monkeypatch.setattr(
        host_exchange,
        "time",
        SimpleNamespace(
            monotonic=time.monotonic, sleep=lambda s: sleeps.append(s) or time.sleep(s)
        ),
    )
    waiter = host_exchanges[1]
    reported = []

    def late(exchange):
        # Rank 0 sends well after rank 1 reports that it waits for rank 0.
        deadline = time.monotonic() + 10
        while waiter.waiting_for != 1 << 0 and time.monotonic() < deadline:
            time.sleep(1e-3)
        reported.append(waiter.waiting_for)
        time.sleep(0.02)
        return exchange.exchange(b"r0", idle=idle)

    results = _run_ranks(
        host_exchanges, [late, lambda exchange: exchange.exchange(b"r1", idle=idle)]
    )
    assert results == {0: [b"r0", b"r1"], 1: [b"r0", b"r1"]}
    assert reported == [1 << 0]
    assert bool(sleeps) == idle, sleeps


@pytest.mark.timeout(60)
def test_idle_wait_sleeps_briefly(host_exchanges, monkeypatch):
    """Check that an idle wait spins for ``_IDLE_SPIN_S`` and then sleeps ``_IDLE_SLEEP_S`` between polls.

    The sleep stays short however long the peer takes, so a new request ends the wait about one poll later.
    """
    clock = [0.0]
    sleeps = []  # (clock at the sleep, sleep length)
    waited_long = threading.Event()

    def monotonic():
        clock[0] += 1e-6  # each poll takes time
        return clock[0]

    def sleep(seconds):
        sleeps.append((clock[0], seconds))
        clock[0] += seconds
        if len(sleeps) == 1000:
            waited_long.set()

    monkeypatch.setattr(host_exchange, "time", SimpleNamespace(monotonic=monotonic, sleep=sleep))

    def late(exchange):
        # Rank 0 sends once rank 1 has slept 1000 times.
        waited_long.wait(30)
        return exchange.exchange(b"r0", idle=True)

    results = _run_ranks(
        host_exchanges, [late, lambda exchange: exchange.exchange(b"r1", idle=True)]
    )
    assert results == {0: [b"r0", b"r1"], 1: [b"r0", b"r1"]}
    assert sleeps[0][0] >= TPHostExchange._IDLE_SPIN_S
    assert max(seconds for _, seconds in sleeps) == TPHostExchange._IDLE_SLEEP_S


def test_only_request_probe_is_idle():
    """Check that only the executor's idle path (``prefer_cpu``) runs an idle host exchange.

    A broadcast's non-roots and a gather's senders name the ranks that read their payloads.
    """
    calls = []

    def exchange(payload, readers, recipients, idle):
        calls.append((readers, recipients, idle))
        return [pickle.dumps(np.array([5])), None, None, None]

    dist = _mnnvl_dist(SimpleNamespace(exchange=exchange), tp_rank=1)
    assert dist.broadcast_int64([0], root=0, prefer_cpu=True).tolist() == [5]
    dist.broadcast_or_none(None)
    assert dist.tp_gather([], root=0) is None
    dist.barrier()
    assert calls == [((0,), (), True), ((0,), (), False), ((), (0,), False), ((), (), False)]


def test_request_probe_root_is_idle():
    """Check that the broadcast's root, which waits for its readers, also takes the idle path."""
    calls = []

    def exchange(payload, readers, recipients, idle):
        calls.append((readers, recipients, idle))
        return [None] * 4

    dist = _mnnvl_dist(SimpleNamespace(exchange=exchange), tp_rank=0)
    assert dist.broadcast_int64([5], root=0, prefer_cpu=True).tolist() == [5]
    dist.broadcast_or_none(None)
    assert calls == [((), None, True), ((), None, False)]


@pytest.mark.timeout(60)
def test_long_payload_only_in_reader_rows(monkeypatch):
    """Check where a gather to rank 0 writes payloads longer than a header poll.

    Every row gets every sender's header, and only rank 0's row gets the payloads. A rank that finishes the round
    from the rows, as after a stop, reads only its own row because it reads no payload.
    """
    exchanges = _host_exchange_group(monkeypatch, 3, _LONG_SLOT_BYTES)
    payloads = [bytes([1 + rank]) * (3000 + rank) for rank in range(3)]
    results = _run_ranks(
        exchanges,
        [lambda exchange: exchange.exchange(payloads[0], recipients=(0,))]
        + [lambda exchange: exchange.exchange(payloads[exchange.rank], readers=(), recipients=(0,))]
        * 2,
    )
    assert results == {0: payloads, 1: [None] * 3, 2: [None] * 3}
    exchange = exchanges[1]
    # The gather's slots, indexed by [row, sender, packet].
    slots = np.frombuffer(
        ctypes.string_at(exchange.rows.ptr, 3 * exchange.rows.stride), dtype=np.uint64
    ).reshape(3, 2, 3, -1)[:, exchange.epoch & 1]
    epochs = slots >> np.uint64(32)
    assert (epochs[:, :, 0] == exchange.epoch).all()
    assert (epochs[0, :, 1:751] == exchange.epoch).all()
    assert not slots[1:, :, 1:].any()
    log = mnnvl_memory.cuda.log
    copies = len(log)
    finished = exchange._finish_from_peers(0b111)
    assert len(log) == copies + 1  # its own row only
    assert (exchange._epochs(finished)[:, 0] == exchange.epoch).all()


@pytest.mark.timeout(60)
def test_broadcast_receiver_polls_root_payload(monkeypatch):
    """Check that readers of a long broadcast poll every header and then only the root's slot."""
    exchanges = _host_exchange_group(monkeypatch, 3, _LONG_SLOT_BYTES)
    payload = bytes(range(256)) * 12
    results = _run_ranks(
        exchanges,
        [lambda exchange: exchange.exchange(payload, readers=())]
        + [lambda exchange: exchange.exchange(b"", readers=(0,), recipients=())] * 2,
    )
    assert [results[rank][0] for rank in (1, 2)] == [payload, payload]
    head = 8 * (1 + TPHostExchange._POLL_PACKETS)
    for exchange in exchanges[1:]:
        recv = exchange._recv.ctypes.data
        polls = [
            (src, width, count)
            for (_, src), (memory, dst), width, count in mnnvl_memory.cuda.log
            if memory == "host" and recv <= dst < recv + exchange._recv.nbytes
        ]
        own_slots = exchange.rows.row(exchange.rank) + (exchange.epoch & 1) * exchange._parity_bytes
        assert polls[0] == (own_slots, head, 3)
        assert {poll for poll in polls if poll[1] > head} == {
            (own_slots, 8 * (1 + len(payload) // 4), 1)
        }


@pytest.mark.timeout(60)
def test_change_without_failure_keeps_results(host_exchanges):
    """Check that a change which only adds a replacement gives both ranks the same results.

    The change reaches rank 1 while it waits for rank 0, and reaches rank 0 only in the next round. Rank 1 stops in
    the round that rank 0 finished, and the change finishes it with the packets rank 0 received. Rank 0 stops in the
    next round, which runs again on the new rows.
    """
    seen = [False, False]  # Whether each rank has received the PROPOSE.
    stopped = {}
    transaction = threading.Barrier(2, timeout=20)

    def change(rank, exchange):
        # Run the membership change from READY to the commit.
        stopped[rank] = exchange.stopped_epoch
        transaction.wait()
        # Every process that maps the rows is active, so all of them keep their pages.
        exchange.install(None, 0b11, next_epoch(latest_epoch(list(stopped.values()))), frozenset())
        seen[rank] = False
        transaction.wait()

    for rank, exchange in enumerate(host_exchanges):
        exchange.interrupted = lambda rank=rank: seen[rank]
        exchange.on_interrupt = lambda rank=rank, exchange=exchange: change(rank, exchange)
    results, errors = {0: [], 1: []}, []

    def run(rank):
        exchange = host_exchanges[rank]
        try:
            if rank == 0:
                # Rank 0 sees the PROPOSE only once it waits in the round after the one it is late to.
                threading.Timer(0.35, seen.__setitem__, (0, True)).start()
                # Rank 1 waits for rank 0's packets and sees the PROPOSE meanwhile.
                time.sleep(0.2)
                seen[1] = True
                time.sleep(0.05)
            results[rank].append(exchange.exchange(b"r%d" % rank))
            results[rank].append(exchange.exchange(b"s%d" % rank))
        except BaseException as error:  # surface thread failures in the test
            errors.append(error)
            transaction.abort()

    ranks = [threading.Thread(target=run, args=(rank,), daemon=True) for rank in range(2)]
    for thread in ranks:
        thread.start()
    for thread in ranks:
        thread.join(30)
    assert not errors, errors
    assert not any(thread.is_alive() for thread in ranks)
    assert results[0] == [[b"r0", b"r1"], [b"s0", b"s1"]]
    assert results[1] == results[0]
    # Rank 0 stops one round ahead of rank 1.
    assert stopped == {0: 2, 1: 1}


@pytest.mark.timeout(60)
def test_wait_for_failed_rank_stops(monkeypatch):
    """Check that ranks waiting for a failed rank stop for the change that drops it.

    Rank 2 fails before it sends. Ranks 0 and 1 spin until the change reaches them, install it, and run the round
    again without rank 2.
    """
    exchanges = _host_exchange_group(monkeypatch, 3, _SLOT_BYTES)
    survivors = 0b011
    proposed = threading.Event()
    stopped = {}
    transaction = threading.Barrier(2, timeout=20)

    def change(exchange):
        # Run the membership change from READY to the commit.
        stopped[exchange.rank] = exchange.stopped_epoch
        transaction.wait()
        exchange.install(
            None, survivors, next_epoch(latest_epoch(list(stopped.values()))), frozenset(range(3))
        )
        exchange.interrupted = lambda: False
        transaction.wait()

    for exchange in exchanges[:2]:
        exchange.interrupted = proposed.is_set
        exchange.on_interrupt = lambda exchange=exchange: change(exchange)

    def survivor(exchange):
        return exchange.exchange(b"r%d" % exchange.rank), exchange.exchange(b"s%d" % exchange.rank)

    threading.Timer(0.2, proposed.set).start()
    results = _run_ranks(exchanges[:2], [survivor, survivor])
    assert results[0] == ([b"r0", b"r1", None], [b"s0", b"s1", None])
    assert results[1] == results[0]
    # Both ranks stop in the round that rank 2 never sent into.
    assert stopped == {0: 1, 1: 1}


def test_joining_rank_installs_rows(monkeypatch):
    """Check that a joining rank maps the rows at INSTALL without entering the exchange.

    It continues from the record's epoch and enters the exchange only when admitted.
    """
    joining = _host_exchange_group(monkeypatch, 3, _SLOT_BYTES)[2]
    joining.rows._memory[:] = 1
    joining.install(None, 0b011, 7, frozenset(range(3)))
    assert joining.active_mask == 0b011
    assert joining.epoch == 6
    row = joining.rows._memory[2 * joining.rows.stride : 3 * joining.rows.stride]
    assert not row.any()


@pytest.mark.timeout(60)
@pytest.mark.parametrize("absent", [1, 3])
def test_collectives_skip_absent_rank(monkeypatch, absent):
    """Check MnnvlDist's collectives with an absent rank in the middle or at the end of the group.

    Object gathers give None in its place, reductions skip it, and int64 gathers give it a zero row. The payloads
    either fit a header poll or take several rounds. Afterwards the rank is admitted again.
    """
    nranks = 4
    exchanges = _host_exchange_group(monkeypatch, nranks, _LONG_SLOT_BYTES)
    active = ((1 << nranks) - 1) & ~(1 << absent)
    active_ranks = _ranks_in(active, nranks)
    for exchange in exchanges:
        if exchange.rank != absent:
            exchange.admit(active, next_epoch(exchange.epoch))
    payloads = (
        lambda r: {"rank": r},
        lambda r: bytes([r]) * (2 * _LONG_SLOT_BYTES + 100 * r),
    )

    def dist_of(exchange):
        return _mnnvl_dist(exchange, tp_rank=exchange.rank, tp_size=nranks)

    def collectives(exchange):
        dist, rank = dist_of(exchange), exchange.rank
        for payload in payloads:
            expected = [payload(r) if r != absent else None for r in range(nranks)]
            assert dist.allgather(payload(rank)) == expected
            assert dist.tp_gather(payload(rank), root=0) == (expected if rank == 0 else None)
            for root in active_ranks:
                assert dist.broadcast(payload(rank), root=root) == payload(root)
            assert dist.broadcast_or_none(payload(rank) if rank == 0 else None) == payload(0)
            assert dist.broadcast_or_none(None) is None
        assert dist.allreduce(rank + 1) == sum(r + 1 for r in active_ranks)
        for n in (3, _LONG_SLOT_BYTES // 8 + 5):
            gathered = dist.tp_allgather_int64(np.full(n, rank + 1))
            assert gathered.tolist() == [[r + 1 if r != absent else 0] * n for r in range(nranks)]
            assert dist.tp_active_rows(gathered).tolist() == [[r + 1] * n for r in active_ranks]
        assert dist.broadcast_int64([7, rank], root=0).tolist() == [7, 0]
        dist.barrier()

    _run_ranks([exchanges[r] for r in active_ranks], [collectives] * len(active_ranks))
    epoch = next_epoch(max(exchange.epoch for exchange in exchanges))
    for exchange in exchanges:
        exchange.admit((1 << nranks) - 1, epoch)
    results = _run_ranks(
        exchanges, [lambda exchange: dist_of(exchange).allgather(exchange.rank)] * nranks
    )
    assert results == {r: list(range(nranks)) for r in range(nranks)}


@pytest.mark.parametrize("rank, local", [(0, 4), (3, 4), (5, 2)])
def test_node_ranks_follow_gpus_per_node(rank, local):
    dist = MnnvlDist(Mapping(world_size=6, rank=rank, gpus_per_node=4, tp_size=6))
    assert dist.local_world_size == local

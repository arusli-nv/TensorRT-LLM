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
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.distributed import host_exchange, mnnvl_memory
from tensorrt_llm._torch.distributed.host_exchange import MnnvlDist, TPHostExchange
from tensorrt_llm._torch.distributed.host_exchange_epoch import latest_epoch, next_epoch
from tensorrt_llm._torch.distributed.membership import Authority, RankReport, Reply, ResumeState
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

WORLD = 4


def _ranks_in(mask, nranks=WORLD):
    """Return the ranks below ``nranks`` whose bit is set in ``mask``."""
    return [r for r in range(nranks) if mask & (1 << r)]


def _mnnvl_dist(exchange, **mapping):
    """Return an ``MnnvlDist`` that runs its collectives over ``exchange``."""
    dist = MnnvlDist(SimpleNamespace(**mapping))
    dist._host_exchange = exchange
    return dist


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


class _Replaced(Exception):
    """Raised where a newer membership change ends a rank's wait."""


@pytest.mark.timeout(60)
def test_replaced_install_resumes_same_round(monkeypatch):
    """Check that an install that a newer change replaces on some ranks leaves every rank the same results.

    Rank 3 fails, then rank 2 fails while ranks 0 to 2 remap the rows for the change that drops rank 3. Rank 1
    completes the remap. Rank 0 stops waiting in the remap's last gather, because the newer change has reached it.
    The authority resumes the newer change from the replaced install's epoch, which rank 1 installs again.
    """
    exchanges = _host_exchange_group(monkeypatch, WORLD, _SLOT_BYTES)
    authority = Authority(WORLD)
    authority.tick(0.0, {r: RankReport(started=True) for r in range(WORLD)}, 0b1111)
    connected = {1: 0b0111, 2: 0b0011}
    stopped = {1: {}, 2: {}}
    epochs = {}

    def decide(generation):
        ranks = _ranks_in(connected[generation])
        authority.tick(generation, {r: RankReport() for r in ranks}, connected[generation])
        ready = {
            r: RankReport(
                reply=Reply.READY,
                reply_generation=generation,
                paused_at=ResumeState(epoch=stopped[generation][r]),
            )
            for r in ranks
        }
        epochs[generation] = authority.tick(generation, ready, connected[generation]).resume.epoch

    ready = {1: threading.Barrier(3, action=lambda: decide(1), timeout=20)}
    ready[2] = threading.Barrier(2, action=lambda: decide(2), timeout=20)
    committed = threading.Barrier(2, timeout=20)
    generations = {0: [1, 2], 1: [1, 2], 2: [1]}
    remaps = []

    def remap_of_rank_0(gather, fresh_rows):
        remaps.append(fresh_rows)
        if len(remaps) == 1:
            raise _Replaced
        _HostRows.remap(exchanges[0].rows, gather, fresh_rows)

    exchanges[0].rows.remap = remap_of_rank_0

    def change(exchange):
        # Run the membership changes this rank takes part in, from READY to the commit.
        while True:
            generation = generations[exchange.rank].pop(0)
            stopped[generation][exchange.rank] = exchange.stopped_epoch
            ready[generation].wait()
            if exchange.rank == 2:
                raise _Replaced  # Its process ends.
            try:
                exchange.install(
                    None, connected[generation], epochs[generation], frozenset(range(WORLD))
                )
            except _Replaced:
                continue
            if generation == 2:
                committed.wait()
                exchange.interrupted = lambda: False
            return

    proposed = threading.Event()
    for exchange in exchanges[:3]:
        exchange.interrupted = proposed.is_set
        exchange.on_interrupt = lambda exchange=exchange: change(exchange)

    def survivor(exchange):
        return exchange.exchange(b"r%d" % exchange.rank), exchange.exchange(b"s%d" % exchange.rank)

    def fails(exchange):
        with pytest.raises(_Replaced):
            survivor(exchange)

    threading.Timer(0.2, proposed.set).start()
    results = _run_ranks(exchanges[:3], [survivor, survivor, fails])
    assert epochs == {1: 2, 2: 2}
    assert results[0] == ([b"r0", b"r1", None, None], [b"s0", b"s1", None, None])
    assert results[1] == results[0]


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

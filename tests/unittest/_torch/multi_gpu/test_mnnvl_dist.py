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
"""MnnvlDist's host collectives (over TPHostExchange) against MPIDist's on the same inputs, including payloads
that take several exchange rounds and absent ranks, and a stress loop of random payload sizes checked by content."""

import pickle
import sys
import threading
import time
import traceback
from types import SimpleNamespace

import cloudpickle
import numpy as np
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed import host_exchange
from tensorrt_llm._torch.distributed.communicator import MPIDist, ReduceOp
from tensorrt_llm._torch.distributed.host_exchange import MnnvlDist, MnnvlRows, TPHostExchange
from tensorrt_llm._torch.distributed.host_exchange_epoch import latest_epoch, next_epoch
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = pytest.mark.threadleak(enabled=False)


def _dists(tp):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(
        world_size=tp, tp_size=tp, moe_ep_size=tp, rank=rank, enable_attention_dp=True
    )
    MPI.COMM_WORLD.barrier()
    return MnnvlDist(mapping), MPIDist(mapping), rank


def _collectives(tp):
    mnnvl, mpi, rank = _dists(tp)
    assert MnnvlDist.supports(mnnvl.mapping)
    assert mnnvl.supports_single_round_broadcast
    small = {"rank": rank, "load": [rank, 2 * rank]}
    # Larger than one exchange slot, so it takes several rounds; ranks differ in length.
    large = bytes(range(256)) * (3000 + 500 * rank)
    for obj in (small, large, None, b""):
        assert mnnvl.allgather(obj) == mpi.allgather(obj)
        assert mnnvl.tp_allgather(obj) == mpi.tp_allgather(obj)
        assert mnnvl.tp_gather(obj, root=1) == mpi.tp_gather(obj, root=1)
        assert mnnvl.broadcast(obj, root=tp - 1) == mpi.broadcast(obj, root=tp - 1)
        assert mnnvl.broadcast_or_none(obj if rank == 0 else None) == mpi.broadcast_or_none(
            obj if rank == 0 else None
        )
    for op in (ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN):
        assert mnnvl.allreduce(rank + 1, op) == mpi.allreduce(rank + 1, op)
    assert mnnvl.tp_allreduce([rank], ReduceOp.SUM) == mpi.tp_allreduce([rank], ReduceOp.SUM)
    values = [rank, -rank, 2**40 + rank]
    assert np.array_equal(mnnvl.tp_allgather_int64(values), mpi.tp_allgather_int64(values))
    # Rows that arrive with their headers, that need a second poll, and that take several rounds.
    slot_values = TPHostExchange.DEFAULT_SLOT_PAYLOAD_BYTES // 8
    for n in (0, 1, TPHostExchange._POLL_PACKETS, slot_values, 2 * slot_values + 3):
        row = np.arange(n, dtype=np.int64) * (rank + 1) - 2**40
        assert np.array_equal(mnnvl.tp_allgather_int64(row), mpi.tp_allgather_int64(row)), n
    assert np.array_equal(
        mnnvl.broadcast_int64(values, root=1), mpi.broadcast_int64(values, root=1)
    )
    gathered = mnnvl.tp_allgather_int64(values)
    # With every rank live, the gather itself (no copy).
    assert mnnvl.tp_live_rows(gathered) is gathered
    mnnvl.barrier()

    # A round whose payloads fit the header poll's copy waits once.
    exchange = mnnvl.host_exchange
    waits = []
    wait = exchange._wait
    exchange._wait = lambda *args: waits.append(args) or wait(*args)
    try:
        mnnvl.tp_allgather_int64(values)
        mnnvl.allgather(small)
        assert mnnvl.broadcast(small, root=0) == mpi.broadcast(small, root=0)
    finally:
        del exchange._wait
    assert len(waits) == 3, len(waits)

    # The rounds reuse their copy descriptors: none is built once both slot sets have been used.
    cuda = exchange._cuda
    built = []
    exchange._cuda = SimpleNamespace(
        CUDA_MEMCPY2D=lambda: built.append(None) or cuda.CUDA_MEMCPY2D(),
        CUmemorytype=cuda.CUmemorytype,
        cuMemcpy2DAsync=cuda.cuMemcpy2DAsync,
    )
    try:
        for _ in range(4):
            assert mnnvl.allgather(small) == mpi.allgather(small)
            assert np.array_equal(mnnvl.tp_allgather_int64(values), mpi.tp_allgather_int64(values))
    finally:
        exchange._cuda = cuda
    assert not built, len(built)


def _absent(tp):
    """Every collective with the last rank absent: object gathers return None in its place, reductions skip it,
    int64 gathers give it a zero row that ``tp_live_rows`` drops. MPI stands in for the membership transaction."""
    mnnvl, mpi, rank = _dists(tp)
    leaver = tp - 1
    exchange = mnnvl.host_exchange
    mnnvl.barrier()

    def gather_handles(handle):
        return mpi.allgather(handle)

    survivors = ((1 << tp) - 1) & ~(1 << leaver)
    epoch = max(mpi.allgather(exchange.epoch)) + 1
    ptr, copies = exchange.rows.ptr, dict(exchange._copies)
    if rank == leaver:
        _skip_rebuild(gather_handles)
    else:
        _rebuild(exchange, gather_handles, survivors, epoch)
    mpi.barrier()
    if rank != leaver:
        # Fresh pages at the same addresses: the copies built for them are kept.
        assert exchange.rows.ptr == ptr
        assert all(exchange._copies[key] is copy for key, copy in copies.items())
        slot = TPHostExchange.DEFAULT_SLOT_PAYLOAD_BYTES
        payloads = (
            lambda r: {"rank": r},
            lambda r: bytes([r]) * (100 + r),
            lambda r: bytes([r]) * (slot + 100 * r),  # several rounds
        )
        for obj_of in payloads:
            expected = [obj_of(r) if r != leaver else None for r in range(tp)]
            assert mnnvl.allgather(obj_of(rank)) == expected
            assert mnnvl.tp_allgather(obj_of(rank)) == expected
            assert mnnvl.tp_gather(obj_of(rank), root=0) == (expected if rank == 0 else None)
            assert mnnvl.broadcast(obj_of(rank), root=0) == expected[0]
            assert mnnvl.broadcast_or_none(obj_of(rank) if rank == 0 else None) == expected[0]
            assert mnnvl.broadcast_or_none(None) is None
        assert mnnvl.allreduce(rank + 1, ReduceOp.SUM) == sum(r + 1 for r in range(tp - 1))
        assert mnnvl.tp_allreduce(rank + 1, ReduceOp.MAX) == tp - 1
        for n in (3, TPHostExchange.DEFAULT_SLOT_PAYLOAD_BYTES // 8 + 5):
            gathered = mnnvl.tp_allgather_int64(np.full(n, rank + 1))
            assert np.array_equal(
                gathered, [np.full(n, r + 1 if r != leaver else 0) for r in range(tp)]
            ), n
            assert np.array_equal(
                mnnvl.tp_live_rows(gathered), [np.full(n, r + 1) for r in range(tp - 1)]
            )
        assert np.array_equal(mnnvl.broadcast_int64([7, rank], root=0), [7, 0])
        mnnvl.barrier()

    epoch = max(mpi.allgather(exchange.epoch)) + 1
    _rebuild(exchange, gather_handles, (1 << tp) - 1, epoch)
    mpi.barrier()
    gathered = mnnvl.tp_allgather_int64([rank + 1])
    assert np.array_equal(mnnvl.tp_live_rows(gathered), [[r + 1] for r in range(tp)])


def _stress(tp):
    mnnvl, _, rank = _dists(tp)
    rng = np.random.default_rng(1234)
    slot = TPHostExchange.DEFAULT_SLOT_PAYLOAD_BYTES
    for step in range(2000):
        sizes = rng.integers(0, 3 * slot, size=tp)  # the same draw on every rank
        mine = np.random.default_rng(step * tp + rank).bytes(int(sizes[rank]))
        gathered = mnnvl.allgather(mine)
        for r, payload in enumerate(gathered):
            assert payload == np.random.default_rng(step * tp + r).bytes(int(sizes[r])), (step, r)
        # int64 rows of one size per step, up to three rounds.
        n = int(sizes[0]) // 8
        rows = mnnvl.tp_allgather_int64(
            np.random.default_rng(step * tp + rank).integers(-(2**62), 2**62, n)
        )
        for r in range(tp):
            assert np.array_equal(
                rows[r], np.random.default_rng(step * tp + r).integers(-(2**62), 2**62, n)
            ), (step, r)


def _epoch_wrap(tp):
    mnnvl, _, rank = _dists(tp)
    exchange = mnnvl.host_exchange
    exchange._epoch = 0xFFFFFFFE - 4  # a few exchanges before the epoch wraps to 1
    epochs = []
    for _ in range(8):  # single-round exchanges: consecutive rounds use alternate slot sets
        mnnvl.allgather(rank)
        epochs.append(exchange._epoch)
    assert 1 in epochs and all((a ^ b) & 1 for a, b in zip(epochs, epochs[1:])), epochs
    exchange._epoch = 0xFFFFFFFE - 4
    half_slot = TPHostExchange.DEFAULT_SLOT_PAYLOAD_BYTES // 2
    for step in range(16):
        large = bytes([step, rank]) * (half_slot + 100 * rank)  # two rounds each
        assert mnnvl.allgather(large) == [
            bytes([step, r]) * (half_slot + 100 * r) for r in range(tp)
        ]


def _membership(tp):
    """The last rank leaves: the others' wait is interrupted, they finish the exchange on fresh rows without it, then
    it is admitted again. MPI stands in for the membership transaction that tells every rank the new mask and epoch."""
    mnnvl, mpi, rank = _dists(tp)
    leaver = tp - 1
    survivors = ((1 << tp) - 1) & ~(1 << leaver)
    exchange = mnnvl.host_exchange
    mnnvl.barrier()

    # Every process takes part once per rebuild; the leaver contributes None.
    def gather_handles(handle):
        return mpi.allgather(handle)

    if rank == leaver:
        mpi.allgather(None)  # the epochs the survivors stopped at
        _skip_rebuild(gather_handles)
        mpi.barrier()  # the commit
    else:
        stop_at = time.monotonic() + 0.5
        exchange.interrupted = lambda: time.monotonic() > stop_at

        def change():
            stopped = [e for e in mpi.allgather(exchange.stopped_epoch) if e is not None]
            exchange.install(
                gather_handles,
                survivors,
                next_epoch(latest_epoch(stopped)),
                exchange.rows.stale_rows(gather_handles),
            )
            exchange.interrupted = lambda: False
            mpi.barrier()  # the commit

        exchange.on_interrupt = change
        # The leaver never sends: the round runs again without it.
        assert mnnvl.allgather(rank) == [r if survivors >> r & 1 else None for r in range(tp)]
        assert mnnvl.allreduce(rank + 1, ReduceOp.SUM) == sum(
            r + 1 for r in range(tp) if r != leaver
        )

    # The admission: every live rank maps fresh rows at INSTALL (the rank that left still maps the rows it mapped
    # before), the joining rank's included, which the others keep writing into until it enters the exchange where
    # they resume.
    all_ranks = (1 << tp) - 1
    epoch = next_epoch(max(mpi.allgather(exchange.epoch)))
    _rebuild(exchange, gather_handles, survivors, epoch)
    if rank != leaver:
        assert mnnvl.allgather(rank) == [r if survivors >> r & 1 else None for r in range(tp)]
    exchange.admit(all_ranks, next_epoch(mpi.broadcast(exchange.epoch, root=0)))
    assert mnnvl.allgather(rank) == list(range(tp))


def _rebuild(exchange, gather, live_mask, epoch):
    """Map the host exchange's rows among the ranks ``gather`` reaches, as an INSTALL does, then continue with the
    ranks in ``live_mask`` from ``epoch``, outside any round."""
    exchange.rows.remap(gather, exchange.rows.stale_rows(gather))
    exchange.admit(live_mask, epoch)


def _skip_rebuild(gather_handles):
    """Take part, as a rank that left, in the survivors' rebuild of the host exchange: their choice of fresh rows,
    their new rows' handles, and their mapping outcomes."""
    for _ in range(3):
        gather_handles(None)


def _admission(tp):
    """The last rank leaves, then its replacement joins: a new exchange of that rank whose rows map only its own, as at
    an isolated startup. No process that maps the survivors' rows is gone, so every rank keeps the pages it maps and
    imports only the rows it does not map yet; then every rank exchanges with every other. MPI stands in for the
    membership transaction."""
    from tensorrt_llm._torch.distributed.membership import StoreGroup

    mnnvl, mpi, rank = _dists(tp)
    leaver = tp - 1
    survivors = ((1 << tp) - 1) & ~(1 << leaver)
    exchange = mnnvl.host_exchange
    mnnvl.barrier()
    epoch = next_epoch(max(mpi.allgather(exchange.epoch)))
    if rank == leaver:
        _skip_rebuild(mpi.allgather)
        exchange.rows.close()
        slot = TPHostExchange.DEFAULT_SLOT_PAYLOAD_BYTES
        isolated = StoreGroup(None, "rows", rank, range(tp), live=[rank])
        rows = MnnvlRows(rank, tp, TPHostExchange.buffer_bytes(tp, slot), isolated.allgather)
        exchange = mnnvl._host_exchange = TPHostExchange(rank, tp, rows, slot)
    else:
        _rebuild(exchange, mpi.allgather, survivors, epoch)
        assert mnnvl.allgather(rank) == [r if survivors >> r & 1 else None for r in range(tp)]
    mpi.barrier()

    own = np.zeros((2 * tp, 1 + exchange._words), dtype=np.uint64)
    exchange._copy_to_host(exchange.rows.row(rank), own)
    epoch = next_epoch(max(mpi.allgather(exchange.epoch)))
    _rebuild(exchange, mpi.allgather, (1 << tp) - 1, epoch)
    kept = np.zeros_like(own)
    exchange._copy_to_host(exchange.rows.row(rank), kept)
    mpi.barrier()  # no rank writes into a row before its owner has read it
    # The admission kept every row's pages: the packets of the survivors' last exchanges are still there.
    assert np.array_equal(kept, own) and (rank == leaver or own.any())
    assert exchange.rows.stale_rows(mpi.allgather) == frozenset()
    assert mnnvl.allgather(rank) == list(range(tp))
    assert mnnvl.allreduce(rank + 1, ReduceOp.SUM) == sum(r + 1 for r in range(tp))


def _arrived(tp):
    """A collective whose payloads no rank reads (the barrier before warmup) on a rank that every peer has already
    sent into: its first poll finds every header, and it returns without waiting for any payload."""
    mnnvl, mpi, rank = _dists(tp)
    exchange = mnnvl.host_exchange
    mnnvl.barrier()
    waits = []
    wait = exchange._wait
    exchange._wait = lambda *args: waits.append(args) or wait(*args)
    try:
        if rank == 0:
            time.sleep(0.2)  # the others' headers arrive meanwhile
        mnnvl.barrier()
    finally:
        del exchange._wait
    assert len(waits) == 1, len(waits)
    assert mnnvl.allgather(rank) == list(range(tp))


def _send_round_to(exchange, payload, dest):
    """Send this rank's packets of the next exchange round into rank ``dest``'s row only, as a rank that fails while
    it sends would."""
    from tensorrt_llm._torch.distributed.mnnvl_memory import _check_cu_result, cuda

    epoch = next_epoch(exchange.epoch)
    tag = np.uint64(epoch) << np.uint64(32)
    words = np.frombuffer(payload + b"\0" * (-len(payload) % 4), dtype=np.uint32)
    packets = np.empty(1 + words.size, dtype=np.uint64)
    packets[0] = tag | np.uint64(len(payload))
    packets[1:] = tag | words.astype(np.uint64)
    slot = (epoch & 1) * exchange._parity_bytes + exchange.rank * exchange._slot_bytes
    _check_cu_result(
        cuda.cuMemcpyHtoD(exchange.rows.row(dest) + slot, packets.ctypes.data, packets.nbytes)
    )
    torch.cuda.synchronize()


def _membership_change(tp):
    """The last rank fails after sending its packets of a round to rank 0 only. Rank 0 finishes that round and stops
    in the next; the others stop in that round. The change finishes the round for them with the packets rank 0
    received and runs the next round again without the failed rank, so every survivor gets the same results. The
    failed round is the last before the epoch wraps. MPI stands in for the membership transaction."""
    mnnvl, mpi, rank = _dists(tp)
    failed = tp - 1
    survivors = ((1 << tp) - 1) & ~(1 << failed)
    exchange = mnnvl.host_exchange
    exchange._epoch = 0xFFFFFFFE - 2
    mnnvl.barrier()

    def gather_handles(handle):
        return mpi.allgather(handle)

    if rank == failed:
        _send_round_to(exchange, pickle.dumps(rank), dest=0)
        mpi.allgather(None)  # the epochs the survivors stopped at
        _skip_rebuild(gather_handles)
        mpi.barrier()  # the commit
        return

    stop_at = time.monotonic() + 1.0
    exchange.interrupted = lambda: time.monotonic() > stop_at
    changes = []

    def change():
        stopped = [epoch for epoch in mpi.allgather(exchange.stopped_epoch) if epoch is not None]
        epoch = next_epoch(latest_epoch(stopped))  # the INSTALL record's epoch
        exchange.install(gather_handles, survivors, epoch, exchange.rows.stale_rows(gather_handles))
        exchange.interrupted = lambda: False
        changes.append((exchange.stopped_epoch, epoch))
        mpi.barrier()

    exchange.on_interrupt = change
    assert mnnvl.allgather(rank) == list(range(tp))  # rank 0 received the failed rank's packets
    assert mnnvl.allgather(10 + rank) == [10 + r if survivors >> r & 1 else None for r in range(tp)]
    assert mnnvl.allreduce(1, ReduceOp.SUM) == tp - 1
    ((stopped, epoch),) = changes
    # Rank 0 stopped after the wrap, in the round that runs again.
    assert (next_epoch(stopped) == epoch) == (rank == 0), (rank, stopped, epoch)


def _change_without_loss(tp):
    """A change that loses no process (it only adds a replacement) reaches the other ranks while they wait for rank 0
    in a round, and rank 0 only once it finished that round and waits in the next. Each stops where the change
    reaches it, as for a failure: the others in the round rank 0 finished, which the change finishes for them with
    the packets rank 0 sent, rank 0 in the next round, which runs again. Every rank gets the same results. MPI stands
    in for the membership transaction."""
    mnnvl, mpi, rank = _dists(tp)
    every = (1 << tp) - 1
    exchange = mnnvl.host_exchange
    mnnvl.barrier()
    seen = [False]  # whether this rank has the change
    exchange.interrupted = lambda: seen[0]
    stops = []

    def change():
        stopped = mpi.allgather(exchange.stopped_epoch)
        stops.append(stopped)
        exchange.install(
            mpi.allgather,
            every,
            next_epoch(latest_epoch(stopped)),
            exchange.rows.stale_rows(mpi.allgather),
        )
        exchange.interrupted = lambda: False
        mpi.barrier()  # the commit

    exchange.on_interrupt = change
    if rank == 0:
        # Rank 0 has the change only once it waits in the round after the one it is late to.
        threading.Timer(0.4, seen.__setitem__, (0, True)).start()
        time.sleep(0.3)  # the others wait for rank 0, and have the change meanwhile
    else:
        threading.Timer(0.1, seen.__setitem__, (0, True)).start()
    assert mnnvl.allgather(rank) == list(range(tp))
    assert mnnvl.allgather(10 + rank) == [10 + r for r in range(tp)]
    (stopped,) = stops
    # Rank 0 stopped one round after the others, in the round that runs again.
    assert stopped[1:] == [stopped[1]] * (tp - 1) and stopped[0] == next_epoch(stopped[1]), stopped


def _backoff(tp):
    """A wait never sleeps while the executor has work, however late the peer; on the idle path (the request probe
    with prefer_cpu) it sleeps between polls."""
    mnnvl, _, rank = _dists(tp)
    real_time = host_exchange.time
    sleeps = []
    host_exchange.time = SimpleNamespace(
        monotonic=real_time.monotonic,
        sleep=lambda s: sleeps.append(s) or real_time.sleep(s),
    )
    try:
        for idle in (False, True):
            mnnvl.barrier()
            MPI.COMM_WORLD.barrier()
            sleeps.clear()
            if rank == 0:
                real_time.sleep(0.1)
            if idle:
                assert mnnvl.broadcast_int64([7], root=0, prefer_cpu=True).tolist() == [7]
            else:
                assert mnnvl.allgather(rank) == list(range(tp))
            if rank != 0:
                assert bool(sleeps) == idle, (idle, len(sleeps))
    finally:
        host_exchange.time = real_time


def _payload_rows(tp):
    """A payload longer than a header poll copies goes only into the rows of the ranks that read it, which poll only
    the slots holding payloads for them; every row gets every header."""
    mnnvl, mpi, rank = _dists(tp)
    payload = bytes([1 + rank]) * (3000 + rank)
    # The first exchange: the rows hold nothing else.
    assert mnnvl.tp_gather(payload, root=0) == mpi.tp_gather(payload, root=0)
    exchange = mnnvl.host_exchange
    row = np.zeros((tp, 1 + exchange._words), dtype=np.uint64)
    exchange._copy_to_host(
        exchange.rows.row(rank) + (exchange.epoch & 1) * exchange._parity_bytes, row
    )
    epochs = row >> np.uint64(32)
    assert (epochs[:, 0] == exchange.epoch).all()
    if rank == 0:
        assert (epochs[:, 1:751] == exchange.epoch).all()
    else:
        assert not row[:, 1:].any()

    cuda = exchange._cuda
    polls = []

    def copy(descriptor, stream):
        if descriptor.dstMemoryType == cuda.CUmemorytype.CU_MEMORYTYPE_HOST:
            polls.append((int(descriptor.srcDevice), descriptor.WidthInBytes, descriptor.Height))
        return cuda.cuMemcpy2DAsync(descriptor, stream)

    exchange._cuda = SimpleNamespace(
        CUDA_MEMCPY2D=cuda.CUDA_MEMCPY2D, CUmemorytype=cuda.CUmemorytype, cuMemcpy2DAsync=copy
    )
    large = bytes(range(256)) * 12
    try:
        assert mnnvl.broadcast(large if rank == 0 else None, root=0) == large
    finally:
        exchange._cuda = cuda
    if rank != 0:
        own_slots = exchange.rows.row(rank) + (exchange.epoch & 1) * exchange._parity_bytes
        packets = 1 + -(-len(pickle.dumps(large)) // 4)
        head = 8 * (1 + TPHostExchange._POLL_PACKETS)
        assert {poll for poll in polls if poll[1] > head} == {(own_slots, 8 * packets, 1)}


def _worker(fn, tp):
    try:
        result = fn(tp)
    except Exception:
        traceback.print_exc()
        raise
    return True if result is None else result


CASES = {
    "collectives": _collectives,
    "absent": _absent,
    "stress": _stress,
    "epoch_wrap": _epoch_wrap,
    "backoff": _backoff,
    "payload_rows": _payload_rows,
    "membership": _membership,
    "membership_change": _membership_change,
    "change_without_loss": _change_without_loss,
    "admission": _admission,
    "arrived": _arrived,
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_mnnvl_dist(case, mpi_pool_executor):
    tp = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    if case == "membership_change" and tp < 3:
        pytest.skip("needs a rank that finishes the round, one that does not, and a failed one")
    results = list(mpi_pool_executor.map(_worker, [CASES[case]] * tp, [tp] * tp))
    assert all(results)

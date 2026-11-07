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
that take several exchange rounds, and a stress loop of random payload sizes checked by content."""

import pickle
import sys
import traceback

import cloudpickle
import numpy as np
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed.communicator import MnnvlDist, MPIDist, ReduceOp
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
    assert np.array_equal(
        mnnvl.broadcast_int64(values, root=1), mpi.broadcast_int64(values, root=1)
    )
    mnnvl.barrier()


def _stress(tp):
    mnnvl, _, rank = _dists(tp)
    rng = np.random.default_rng(1234)
    for step in range(2000):
        sizes = rng.integers(0, 3 * 8192, size=tp)  # the same draw on every rank
        mine = np.random.default_rng(step * tp + rank).bytes(int(sizes[rank]))
        gathered = mnnvl.allgather(mine)
        for r, payload in enumerate(gathered):
            assert payload == np.random.default_rng(step * tp + r).bytes(int(sizes[r])), (step, r)


def _epoch_wrap(tp):
    mnnvl, _, rank = _dists(tp)
    mnnvl.barrier()  # allocates the exchange
    exchange = mnnvl._host_exchange
    exchange._epoch = 0xFFFFFFFE - 4  # a few exchanges before the epoch wraps to 1
    epochs = []
    for _ in range(8):  # single-round exchanges: consecutive rounds use alternate slot sets
        mnnvl.allgather(rank)
        epochs.append(exchange._epoch)
    assert 1 in epochs and all((a ^ b) & 1 for a, b in zip(epochs, epochs[1:])), epochs
    exchange._epoch = 0xFFFFFFFE - 4
    for step in range(16):
        large = bytes([step, rank]) * (8192 + 100 * rank)  # two rounds each
        assert mnnvl.allgather(large) == [bytes([step, r]) * (8192 + 100 * r) for r in range(tp)]


def _worker(fn, tp):
    try:
        fn(tp)
    except Exception:
        traceback.print_exc()
        raise
    return True


CASES = {"collectives": _collectives, "stress": _stress, "epoch_wrap": _epoch_wrap}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_mnnvl_dist(case, mpi_pool_executor):
    tp = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    results = mpi_pool_executor.map(_worker, [CASES[case]] * tp, [tp] * tp)
    assert all(results)

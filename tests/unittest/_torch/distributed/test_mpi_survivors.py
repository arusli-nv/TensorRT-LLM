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
"""MPIDist peer-failure interruption and survivor communicators.

ULFM behavior is modeled with fake communicators, so these tests run on a
single CPU process without an MPI launcher.
"""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_llm._torch.distributed import communicator
from tensorrt_llm._torch.distributed.communicator import HostCollectiveInterrupted, MPIDist

pytestmark = pytest.mark.cpu_only

ERR_PROC_FAILED = 75
ERR_REVOKED = 77
ERR_OTHER = 16


class FakeMPIException(Exception):
    def __init__(self, error_class: int) -> None:
        super().__init__(f"MPI error class {error_class}")
        self.error_class = error_class

    def Get_error_class(self) -> int:
        return self.error_class


class FakeGroup:
    """A group whose members are given by their ranks in the world group."""

    def __init__(self, world_ranks) -> None:
        self.world_ranks = list(world_ranks)

    def Get_size(self) -> int:
        return len(self.world_ranks)

    def Translate_ranks(self, ranks, group):
        return [self.world_ranks[rank] for rank in ranks]

    def Free(self) -> None:
        pass


class FakeComm:
    """Communicator over world ranks ``members`` seen from world rank ``me``.

    Peer contributions are synthesized from their world rank so results can
    be checked for logical placement.
    """

    def __init__(self, members, me, failed=()) -> None:
        self.members = tuple(members)
        self.me = me
        self.failed = tuple(failed)
        self.error = None
        self.revoke_calls = 0
        self.bcast_roots = []
        self.barriers = 0
        self.shrink_members = None

    def _check(self) -> None:
        if self.error is not None:
            raise self.error

    def Get_rank(self) -> int:
        return self.members.index(self.me)

    def Get_size(self) -> int:
        return len(self.members)

    def Get_group(self) -> FakeGroup:
        return FakeGroup(self.members)

    def Get_failed(self) -> FakeGroup:
        return FakeGroup(self.failed)

    def Is_revoked(self) -> bool:
        return self.revoke_calls > 0

    def Revoke(self) -> None:
        self.revoke_calls += 1

    def Shrink(self) -> "FakeComm":
        members = self.shrink_members
        if members is None:
            members = [m for m in self.members if m not in self.failed]
        return FakeComm(members, self.me)

    def allgather(self, obj):
        self._check()
        peer = (lambda m: [f"rank{m}"]) if isinstance(obj, list) else (lambda m: f"rank{m}")
        return [obj if m == self.me else peer(m) for m in self.members]

    def Allgather(self, sendspec, recvspec) -> None:
        self._check()
        sendbuf, recvbuf = sendspec[0], recvspec[0]
        n = sendbuf.size
        for i, m in enumerate(self.members):
            recvbuf[i * n : (i + 1) * n] = sendbuf + 100 * m

    def Bcast(self, bufspec, root) -> None:
        self._check()
        self.bcast_roots.append(root)

    def Barrier(self) -> None:
        self._check()
        self.barriers += 1

    def allreduce(self, obj, op):
        self._check()
        return obj * len(self.members)


WORLD = 4
ME = 1
FAILED = 2


def _mapping(**overrides):
    fields = dict(
        rank=ME,
        world_size=WORLD,
        tp_size=WORLD,
        tp_rank=ME,
        moe_ep_size=WORLD,
        pp_size=1,
        cp_size=1,
        enable_attention_dp=True,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def fake_mpi(monkeypatch):
    fake = SimpleNamespace(
        Exception=FakeMPIException,
        ERR_PROC_FAILED=ERR_PROC_FAILED,
        ERR_PROC_FAILED_PENDING=76,
        ERR_REVOKED=ERR_REVOKED,
        ERR_UNKNOWN=14,
        ERR_OTHER=ERR_OTHER,
        ERR_INTERN=17,
        ERR_LASTCODE=92,
        UNDEFINED=-32766,
        INT64_T="int64",
        BYTE="byte",
    )
    monkeypatch.setattr(communicator, "MPI", fake)
    communicator._peer_failure_error_classes.cache_clear()
    yield fake
    communicator._peer_failure_error_classes.cache_clear()


@pytest.fixture
def world(fake_mpi, monkeypatch):
    world = FakeComm(range(WORLD), ME, failed=(FAILED,))
    monkeypatch.setattr(communicator, "mpi_comm", lambda: world)
    monkeypatch.setattr(communicator, "mpi_barrier", lambda: world.Barrier())
    monkeypatch.setattr(communicator, "mpi_allgather", lambda obj: world.allgather(obj))
    return world


@pytest.fixture
def dist(world):
    dist = MPIDist(_mapping())
    dist._tp_comm = FakeComm(range(WORLD), ME, failed=(FAILED,))
    return dist


def test_healthy_paths_return_compact_results(dist, world):
    assert dist.live_ranks == (0, 1, 2, 3)
    assert dist.tp_allgather("me", small_payload=True) == ["rank0", "me", "rank2", "rank3"]
    rows = dist.tp_allgather_int64([5, 6])
    np.testing.assert_array_equal(rows[:, 0], [5, 105, 205, 305])
    dist.broadcast_int64([1], root=3)
    assert world.bcast_roots == [3] and dist.allreduce(2) == 8


@pytest.mark.parametrize("error_class", [ERR_PROC_FAILED, ERR_REVOKED])
@pytest.mark.parametrize(
    "operation, call",
    [
        ("tp_allgather", lambda d: d.tp_allgather(0, small_payload=True)),
        ("tp_allgather_int64", lambda d: d.tp_allgather_int64([0])),
        ("broadcast_int64", lambda d: d.broadcast_int64([0])),
        ("allreduce", lambda d: d.allreduce(1)),
        ("allgather", lambda d: d.allgather(1)),
        ("barrier", lambda d: d.barrier()),
        ("tp_barrier", lambda d: d.tp_barrier()),
    ],
)
def test_peer_failure_raises_host_collective_interrupted(dist, world, operation, call, error_class):
    error = FakeMPIException(error_class)
    world.error = dist._tp_comm.error = error
    with pytest.raises(HostCollectiveInterrupted) as info:
        call(dist)
    assert info.value.operation == operation
    assert info.value.failed_ranks == frozenset({FAILED})
    assert info.value.__cause__ is error


def test_other_mpi_errors_propagate_unchanged(dist):
    dist._tp_comm.error = FakeMPIException(ERR_OTHER)
    with pytest.raises(FakeMPIException):
        dist.tp_allgather_int64([0])


def test_failed_ranks_empty_without_ulfm_query(dist, world, monkeypatch):
    monkeypatch.delattr(FakeComm, "Get_failed")
    world.error = FakeMPIException(ERR_PROC_FAILED)
    with pytest.raises(HostCollectiveInterrupted) as info:
        dist.broadcast_int64([0])
    assert info.value.failed_ranks == frozenset()


def test_host_collective_interrupted_pickles():
    exc = pickle.loads(pickle.dumps(HostCollectiveInterrupted("barrier", frozenset({3}))))
    assert exc.operation == "barrier" and exc.failed_ranks == frozenset({3})


def test_revoke_is_idempotent(dist, world):
    dist.revoke()
    dist.revoke()
    assert world.revoke_calls == 1 and dist._tp_comm.revoke_calls == 1


def _install(dist):
    failed = dist.shrink()
    dist.install_survivors(failed)
    return failed


def test_shrink_agrees_on_failed_ranks_before_install(dist):
    dist.revoke()
    assert dist.shrink() == frozenset({FAILED})
    assert dist.live_ranks == (0, 1, 2, 3)
    assert dist._survivor_comm is None


def test_install_survivors_keeps_logical_ranks(dist, world):
    dist.revoke()
    assert _install(dist) == frozenset({FAILED})

    assert dist.live_ranks == (0, 1, 3)
    assert (dist.tp_rank, dist.tp_size) == (ME, WORLD)
    assert dist.tp_allgather("me", small_payload=True) == ["rank0", "me", None, "rank3"]
    assert dist.allgather("me") == ["rank0", "me", None, "rank3"]
    assert dist.tp_cp_allgather("me", small_payload=True) == ["rank0", "me", None, "rank3"]

    rows = dist.tp_allgather_int64([5, 6])
    np.testing.assert_array_equal(rows, [[5, 6], [105, 106], [0, 0], [305, 306]])
    rows = dist.tp_allgather_int64([5, 6], fill_value=[-1, 1])
    np.testing.assert_array_equal(rows[FAILED], [-1, 1])
    rows = dist.tp_cp_allgather_int64([5], fill_value=-1)
    np.testing.assert_array_equal(rows[:, 0], [5, 105, -1, 305])

    survivor = dist.tp_comm
    assert survivor is dist._survivor_comm and survivor.members == (0, 1, 3)
    dist.broadcast_int64([1], root=3)
    dist.tp_cp_broadcast_int64([1], root=0)
    assert survivor.bcast_roots == [2, 0]
    assert dist.allreduce(1) == 3
    dist.barrier()
    assert survivor.barriers == 1 and world.barriers == 0
    with pytest.raises(ValueError):
        dist.broadcast_int64([1], root=FAILED)


def test_install_survivors_tp_gather_returns_logical_list(dist, monkeypatch):
    _install(dist)
    roots = []

    def fake_safe_gather(comm, obj, root, chunk_size):
        roots.append(root)
        return [f"rank{m}" for m in comm.members]

    monkeypatch.setattr(communicator, "safe_gather", fake_safe_gather)
    assert dist.tp_gather("x", root=0) == ["rank0", "rank1", None, "rank3"]
    assert roots == [0]


def test_interruption_after_install_reports_logical_ranks(dist):
    _install(dist)
    survivor = dist._survivor_comm
    survivor.failed = (3,)
    survivor.error = FakeMPIException(ERR_PROC_FAILED)
    with pytest.raises(HostCollectiveInterrupted) as info:
        dist.tp_allgather_int64([0])
    assert info.value.failed_ranks == frozenset({3})


@pytest.mark.parametrize(
    "overrides",
    [
        {"pp_size": 2},
        {"cp_size": 2},
        {"enable_attention_dp": False},
        {"moe_ep_size": 2},
    ],
)
def test_shrink_rejects_unsupported_layouts(world, overrides):
    dist = MPIDist(_mapping(**overrides))
    dist._tp_comm = FakeComm(range(WORLD), ME, failed=(FAILED,))
    with pytest.raises(ValueError):
        dist.shrink()


def test_install_survivors_requires_shrink(dist):
    with pytest.raises(RuntimeError):
        dist.install_survivors(frozenset({FAILED}))


def test_install_survivors_rejects_disagreeing_failed_set(dist):
    dist.shrink()
    with pytest.raises(RuntimeError):
        dist.install_survivors(frozenset({FAILED, 3}))
    assert dist.live_ranks == (0, 1, 2, 3)


def test_shrink_only_once(dist):
    _install(dist)
    with pytest.raises(RuntimeError):
        dist.shrink()

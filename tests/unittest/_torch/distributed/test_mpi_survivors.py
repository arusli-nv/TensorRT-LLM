# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MPIDist peer-failure interruption and survivor communicators.

ULFM behavior is modeled with fake communicators, so these tests run on a
single CPU process without an MPI launcher.
"""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_llm import _utils
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
        self.free_count = 0
        self.errhandler = None

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

    def Set_errhandler(self, errhandler) -> None:
        self.errhandler = errhandler

    def Is_revoked(self) -> bool:
        return self.revoke_calls > 0

    def Revoke(self) -> None:
        self.revoke_calls += 1

    def Shrink(self) -> "FakeComm":
        return FakeComm([m for m in self.members if m not in self.failed], self.me)

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

    def Dup(self) -> "FakeComm":
        return self

    def Free(self) -> None:
        self.free_count += 1

    def py2f(self) -> int:
        return 7


class FakeNativeWatch:
    """Stands in for the native watch thread; the test fires its failure."""

    def __init__(self, fortran_comm, abort_word, on_failure, poll_interval_s) -> None:
        self.args = (fortran_comm, abort_word, poll_interval_s)
        self.on_failure = on_failure
        self.failed = False
        self.stops = 0
        self.on_watch_thread = False

    def fire(self, error_class) -> None:
        self.failed = True
        self.on_watch_thread = True
        try:
            self.on_failure(error_class)
        finally:
            self.on_watch_thread = False

    def stop(self) -> bool:
        # As the native watch: from its own callback it cannot join itself and releases nothing.
        if self.on_watch_thread:
            return False
        self.stops += 1
        return True


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
        ANY_SOURCE=-1,
        INT64_T="int64",
        BYTE="byte",
        THREAD_MULTIPLE=3,
        ERRORS_RETURN="errors_return",
        Query_thread=lambda: 3,
        Comm=SimpleNamespace(Revoke=None, Shrink=None, Get_failed=None),
    )
    monkeypatch.setattr(communicator, "MPI", fake)
    monkeypatch.setenv("OMPI_MCA_mpi_ft_enable", "1")
    monkeypatch.setattr(
        communicator,
        "is_mpi_peer_failure",
        lambda error: isinstance(error, FakeMPIException)
        and error.error_class in (ERR_PROC_FAILED, 76, ERR_REVOKED),
    )
    return fake


@pytest.fixture
def world(fake_mpi, monkeypatch):
    world = FakeComm(range(WORLD), ME)
    monkeypatch.setattr(communicator, "mpi_comm", lambda: world)
    monkeypatch.setattr(_utils, "mpi_comm", lambda: world)
    monkeypatch.setattr(communicator, "mpi_barrier", lambda: world.Barrier())
    monkeypatch.setattr(communicator, "mpi_allgather", lambda obj: world.allgather(obj))
    return world


@pytest.fixture
def dist(world):
    dist = MPIDist(_mapping())
    dist._tp_comm = FakeComm(range(WORLD), ME, failed=(FAILED,))
    dist.enable_peer_loss_recovery()
    return dist


def test_recovery_revokes_a_private_world_not_the_session_world(world, monkeypatch):
    """The MPI session keeps using the world communicator after a peer loss."""
    private = FakeComm(range(WORLD), ME)
    monkeypatch.setattr(world, "Dup", lambda: private, raising=False)
    dist = MPIDist(_mapping())
    dist._tp_comm = FakeComm(range(WORLD), ME, failed=(FAILED,))
    with pytest.raises(RuntimeError):
        dist.revoke()
    dist.enable_peer_loss_recovery()
    dist.broadcast_int64([1], root=0)
    dist.revoke()
    assert private.bcast_roots == [0] and world.bcast_roots == []
    assert private.revoke_calls == 1 and world.revoke_calls == 0


def test_recovery_refuses_an_mpi_library_without_ulfm(world, monkeypatch):
    def get_failed():
        raise NotImplementedError

    monkeypatch.setattr(world, "Get_failed", get_failed)
    with pytest.raises(RuntimeError, match="ULFM calls"):
        MPIDist(_mapping()).enable_peer_loss_recovery()


def test_recovery_communicators_return_errors(dist, world, native_watch, monkeypatch):
    """A peer failure must raise in each communicator recovery uses, not abort the job."""

    def fresh_dup():
        return FakeComm(range(WORLD), ME)

    monkeypatch.setattr(world, "Dup", fresh_dup)
    comms = [dist._host_world, dist._tp_comm, dist.watch_peer_failures(lambda: None)._comm]
    dist.revoke()
    dist.shrink()
    monkeypatch.setattr(dist._shrunk[0], "Dup", fresh_dup)
    comms += [dist._shrunk[0], dist.watch_survivor_failures(lambda: None)._comm]
    assert [comm.errhandler for comm in comms] == ["errors_return"] * len(comms)


def test_healthy_paths_return_compact_results(dist):
    assert dist.tp_allgather("me", small_payload=True) == ["rank0", "me", "rank2", "rank3"]
    rows = dist.tp_allgather_int64([5, 6])
    np.testing.assert_array_equal(rows[:, 0], [5, 105, 205, 305])
    assert dist.allreduce(2) == 8


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
def test_peer_failure_raises_host_collective_interrupted(dist, world, operation, call):
    error = FakeMPIException(ERR_PROC_FAILED)
    world.error = dist._tp_comm.error = error
    world.failed = (FAILED,)
    with pytest.raises(HostCollectiveInterrupted) as info:
        call(dist)
    assert info.value.operation == operation
    assert info.value.failed_ranks == frozenset({FAILED})
    assert info.value.__cause__ is error


def test_other_mpi_errors_propagate_unchanged(dist, monkeypatch):
    monkeypatch.setattr(communicator, "_PEER_FAILURE_EVIDENCE_S", 0.0)
    dist._tp_comm.error = FakeMPIException(ERR_OTHER)
    with pytest.raises(FakeMPIException):
        dist.tp_allgather_int64([0])


def test_an_error_on_a_revoked_communicator_is_an_interruption(dist, world, monkeypatch):
    """Recovery revoked the communicator: no need to wait for ULFM to name a failed rank."""
    monkeypatch.setattr(communicator, "_PEER_FAILURE_EVIDENCE_S", 60.0)
    dist._tp_comm.error = FakeMPIException(ERR_OTHER)
    world.Revoke()
    with pytest.raises(HostCollectiveInterrupted) as info:
        dist.tp_allgather_int64([0])
    assert info.value.failed_ranks == frozenset()


def test_without_recovery_errors_and_results_pass_through(world):
    dist = MPIDist(_mapping())
    dist._tp_comm = FakeComm(range(WORLD), ME)
    world.failed = (FAILED,)
    assert dist.allgather("me")[ME] == "me"
    dist._tp_comm.error = FakeMPIException(ERR_PROC_FAILED)
    with pytest.raises(FakeMPIException):
        dist.tp_barrier()


def test_corrupted_result_after_a_peer_failure_is_an_interruption(dist, world):
    """ULFM can complete a collective a peer died in with undefined data."""
    dist._tp_comm.error = pickle.UnpicklingError("unpickling stack underflow")
    world.failed = (FAILED,)
    with pytest.raises(HostCollectiveInterrupted) as info:
        dist.tp_allgather(0, small_payload=True)
    assert info.value.failed_ranks == frozenset({FAILED})
    assert isinstance(info.value.__cause__, pickle.UnpicklingError)


def test_a_local_error_is_never_relabelled_as_a_peer_loss(dist, world, monkeypatch):
    """A bug in this rank raises as is, without the evidence wait, even if a peer has failed."""
    monkeypatch.setattr(communicator, "_PEER_FAILURE_EVIDENCE_S", 60.0)
    dist._tp_comm.error = ValueError("bad argument")
    world.failed = (FAILED,)
    with pytest.raises(ValueError, match="bad argument"):
        dist.tp_allgather(0, small_payload=True)


def test_result_completed_after_a_peer_failure_is_an_interruption(dist, world):
    world.failed = (FAILED,)
    with pytest.raises(HostCollectiveInterrupted) as info:
        dist.tp_allgather_int64([0])
    assert info.value.failed_ranks == frozenset({FAILED})


def test_peer_failure_classes_exclude_generic_aliases(monkeypatch):
    """An MPI library without ULFM may alias the ULFM classes to a generic one."""
    fake = SimpleNamespace(
        Exception=FakeMPIException,
        ERR_PROC_FAILED=ERR_PROC_FAILED,
        ERR_PROC_FAILED_PENDING=14,
        ERR_REVOKED=ERR_REVOKED,
        ERR_UNKNOWN=14,
    )
    monkeypatch.setattr(_utils, "MPI", fake)
    _utils._peer_failure_error_classes.cache_clear()
    try:
        assert _utils._peer_failure_error_classes() == {ERR_PROC_FAILED, ERR_REVOKED}
        assert _utils.is_mpi_peer_failure(FakeMPIException(ERR_REVOKED))
        assert not _utils.is_mpi_peer_failure(FakeMPIException(14))
    finally:
        _utils._peer_failure_error_classes.cache_clear()


def test_host_collective_interrupted_pickles():
    exc = pickle.loads(pickle.dumps(HostCollectiveInterrupted("barrier", frozenset({3}))))
    assert exc.operation == "barrier" and exc.failed_ranks == frozenset({3})


def test_revoke_is_idempotent(dist, world):
    dist.revoke()
    dist.revoke()
    assert world.revoke_calls == 1 and dist._tp_comm.revoke_calls == 1


def _install(dist):
    failed = dist.shrink()
    dist.install_survivors()
    return failed


def test_install_survivors_keeps_logical_ranks(dist, world):
    assert dist.live_ranks == tuple(range(WORLD)) and dist.live_ranks is dist.live_ranks
    dist.revoke()
    assert _install(dist) == frozenset({FAILED})

    assert dist.live_ranks == (0, 1, 3)
    assert (dist.tp_rank, dist.tp_size) == (ME, WORLD)
    assert dist.tp_allgather("me", small_payload=True, departed=None) == [
        "rank0",
        "me",
        None,
        "rank3",
    ]
    assert dist.allgather("me", departed=0) == ["rank0", "me", 0, "rank3"]

    # A caller that does not say what a departed rank contributes fails stop.
    for gather in (
        lambda: dist.tp_allgather("me", small_payload=True),
        lambda: dist.allgather("me"),
        lambda: dist.tp_cp_allgather("me", small_payload=True),
    ):
        with pytest.raises(RuntimeError, match="departed="):
            gather()
    with pytest.raises(RuntimeError, match="departed"):
        dist.tp_allgather_int64([5, 6])
    rows = dist.tp_allgather_int64([5, 6], departed=[-1, 1])
    np.testing.assert_array_equal(rows, [[5, 6], [105, 106], [-1, 1], [305, 306]])

    survivor = dist.tp_comm
    assert survivor is dist._survivor_comm and survivor.members == (0, 1, 3)
    dist.broadcast_int64([1])
    dist.tp_cp_broadcast_int64([1])
    dist.broadcast_int64([1], root=3)
    assert survivor.bcast_roots == [0, 0, 2]
    assert dist.allreduce(1) == 3
    dist.barrier()
    assert survivor.barriers == 1 and world.barriers == 0


def test_install_survivors_tp_gather_returns_logical_list(dist, monkeypatch):
    _install(dist)
    roots = []

    def fake_safe_gather(comm, obj, root, chunk_size):
        roots.append(root)
        return [f"rank{m}" for m in comm.members]

    monkeypatch.setattr(communicator, "safe_gather", fake_safe_gather)
    assert dist.tp_gather("x", root=0, departed=None) == ["rank0", "rank1", None, "rank3"]
    with pytest.raises(RuntimeError, match="departed="):
        dist.tp_gather("x", root=0)
    dist.tp_gather("x", root=3, departed=None)
    assert roots == [0, 0, 2]


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
def test_recovery_rejects_unsupported_layouts(world, overrides):
    with pytest.raises(ValueError):
        MPIDist(_mapping(**overrides)).enable_peer_loss_recovery()


def test_install_survivors_requires_shrink(dist):
    with pytest.raises(RuntimeError):
        dist.install_survivors()


def test_a_loss_after_a_recovery_shrinks_the_installed_survivors(dist):
    dist.revoke()
    _install(dist)
    installed = dist._survivor_comm
    installed.failed = (3,)
    # Revoking releases the survivors waiting on the installed communicator.
    dist.revoke()
    assert installed.Is_revoked()
    assert _install(dist) == frozenset({FAILED, 3})
    assert dist.live_ranks == (0, 1)
    assert dist.allgather("me", departed=None) == ["rank0", "me", None, None]


def test_shrink_again_before_install_drops_a_peer_lost_during_recovery(dist):
    with pytest.raises(RuntimeError, match="prior shrink"):
        dist.watch_survivor_failures(lambda: None)
    dist.revoke()
    assert dist.shrink() == frozenset({FAILED})
    survivors = dist._shrunk[0]
    survivors.failed = (3,)
    # Revoking again releases survivors waiting on the communicator the first shrink built.
    dist.revoke()
    assert survivors.Is_revoked()
    assert dist.shrink() == frozenset({FAILED, 3})
    dist.install_survivors()
    assert dist.live_ranks == (0, 1)


@pytest.fixture
def native_watch(monkeypatch):
    watches = []
    monkeypatch.setattr(
        communicator,
        "_native_peer_failure_watch",
        lambda *args: watches.append(FakeNativeWatch(*args)) or watches[-1],
    )
    return watches


@pytest.mark.parametrize("error_class", [ERR_PROC_FAILED, ERR_OTHER])
def test_watch_peer_failures_reports_peer_failures_and_detection_errors(
    dist, world, native_watch, error_class
):
    calls = []
    watch = dist.watch_peer_failures(lambda: calls.append(1), abort_word=1234)
    (native,) = native_watch
    assert native.args == (7, 1234, communicator._PEER_FAILURE_POLL_INTERVAL_S)
    assert not watch.failed
    native.fire(error_class)
    assert calls == [1] and watch.failed


def test_stopped_watch_releases_the_native_thread_and_communicator_once(dist, world, native_watch):
    watch = dist.watch_peer_failures(lambda: None)
    watch.stop()
    watch.stop()
    assert native_watch[0].stops == 1 and world.free_count == 1


def test_a_watch_stopped_from_its_own_callback_is_released_by_the_next_stop(
    dist, world, native_watch
):
    holder = []
    watch = dist.watch_peer_failures(lambda: holder[0].stop())
    holder.append(watch)
    native_watch[0].fire(ERR_PROC_FAILED)
    assert native_watch[0].stops == 0 and world.free_count == 0
    watch.stop()
    assert native_watch[0].stops == 1 and world.free_count == 1


def test_peer_failures_stay_mpi_errors_without_recovery(world):
    dist = MPIDist(_mapping())
    dist._tp_comm = FakeComm(range(WORLD), ME, failed=(FAILED,))
    world.error = FakeMPIException(ERR_PROC_FAILED)
    with pytest.raises(FakeMPIException):
        dist.broadcast_int64([0])


def test_recovery_requires_mpi4py_ulfm_launch_and_thread_multiple(world, fake_mpi, monkeypatch):
    dist = MPIDist(_mapping())
    monkeypatch.setattr(fake_mpi, "Query_thread", lambda: 1)
    with pytest.raises(RuntimeError, match="THREAD_MULTIPLE"):
        dist.enable_peer_loss_recovery()
    monkeypatch.delenv("OMPI_MCA_mpi_ft_enable")
    with pytest.raises(RuntimeError, match="--with-ft ulfm"):
        dist.enable_peer_loss_recovery()
    monkeypatch.setattr(fake_mpi, "Comm", SimpleNamespace())
    with pytest.raises(RuntimeError, match="mpi4py>=4"):
        dist.enable_peer_loss_recovery()

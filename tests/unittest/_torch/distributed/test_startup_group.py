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
"""WideEP fault tolerance starts every rank as its own process: the ranks that start together meet in the rendezvous
store (startup_gather, StartupGroup), the loss of the store ends their waits, and a replacement's groups answer
alone. Threads stand for ranks; the store is a real TCPStore."""

import socket
import subprocess  # nosec B404
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from unittest.mock import Mock

import pytest
from mpi4py import MPI
from torch.distributed import TCPStore

from tensorrt_llm import _utils
from tensorrt_llm._torch.distributed.membership import (
    MembershipRecord,
    MembershipState,
    SignedStore,
    startup_gather,
)
from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory, TpMnnvlMemory
from tensorrt_llm._torch.distributed.startup_record import StartupRecord, startup_record
from tensorrt_llm._torch.pyexecutor import rank_failure_recovery as recovery_module
from tensorrt_llm._torch.pyexecutor.rank_failure_recovery import StartupGroup, join_startup_group
from tensorrt_llm._utils import FAULT_TOLERANCE_RANK_ENV, OMPI_COMM_TYPE_HOST, RankIdentity
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

SECRET = b"startup-group-secret"
WORLD = 4


def _client(port):
    return SignedStore(
        TCPStore("127.0.0.1", port, is_master=False, timeout=timedelta(seconds=30)), SECRET
    )


@pytest.fixture
def store(monkeypatch):
    """The rendezvous store, as the authority hosts it; each thread (a rank) gets its own client. The ranks share one
    membership agent, which takes their startup reports."""
    server = TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    clients = threading.local()

    def client():
        if not hasattr(clients, "store"):
            clients.store = _client(server.port)
        return clients.store

    monkeypatch.setattr(recovery_module, "_startup_store", client)
    monkeypatch.setattr(recovery_module, "_agent", Mock())
    _replay(monkeypatch, None)
    yield server.port


def _replay(monkeypatch, values):
    """A fresh startup record of this process, which replays ``values`` (startup values by name) unless None."""
    for name, value in vars(StartupRecord()).items():
        monkeypatch.setattr(startup_record, name, value)
    if values is not None:
        startup_record.replay({"values": values, "exchanges": []})


def _on_ranks(ranks, fn):
    """``fn(rank)`` on one thread per rank; the results (or raised errors) by rank."""

    def run(rank):
        try:
            return fn(rank)
        except Exception as error:  # noqa: BLE001 - returned to the test
            return error

    with ThreadPoolExecutor(len(ranks)) as pool:
        return dict(zip(ranks, pool.map(run, ranks)))


def _ignore_waits(waiting_for):
    """A startup gather's ``on_wait`` for tests that do not look at the waits."""


def test_the_ranks_that_start_together_gather_through_the_store(store):
    def rank(r):
        gather = startup_gather(_client(store), "g", r, range(WORLD), WORLD, _ignore_waits)
        pairs = (
            startup_gather(_client(store), "pair", r, [1, 3], WORLD, _ignore_waits)
            if r in (1, 3)
            else None
        )
        return gather(10 * r), gather(f"x{r}"), pairs and pairs(r)

    results = _on_ranks(range(WORLD), rank)
    for r in range(WORLD):
        assert results[r][:2] == ([0, 10, 20, 30], ["x0", "x1", "x2", "x3"])
    assert results[1][2] == results[3][2] == [None, 1, None, 3]


def test_a_startup_collective_reports_the_peers_it_waits_for_and_then_a_step_done(store):
    """The authority's startup hang rule reads these reports. Fails without the change: startup_gather took no
    ``on_wait``, and a rank reported nothing while it waited."""
    waits = []

    def late_peer():
        time.sleep(0.3)
        startup_gather(_client(store), "g", 1, range(2), 2, _ignore_waits)(1)

    peer = threading.Thread(target=late_peer)
    peer.start()
    try:
        assert startup_gather(_client(store), "g", 0, range(2), 2, waits.append)(0) == [0, 1]
    finally:
        peer.join(30)
    assert waits == [0b10, 0]


def test_a_gather_keeps_only_its_last_call_s_keys(store):
    """Repeated calls do not grow the store. Fails without the change: every call left its values and its arrival
    counter behind, 3 keys per call."""
    counter = TCPStore("127.0.0.1", store, is_master=False, timeout=timedelta(seconds=30))
    before = counter.num_keys()

    def rank(r):
        gather = startup_gather(_client(store), "g", r, range(2), 2, _ignore_waits)
        return [gather(call) for call in range(20)]

    results = _on_ranks(range(2), rank)
    assert results[0] == results[1] == [[call, call] for call in range(20)]
    # The last call's two values and its arrival counter.
    assert counter.num_keys() - before <= 3


def test_the_startup_ends_when_the_authority_is_lost(monkeypatch):
    """The store lives in the authority's process: once that process is gone, a wait for a peer ends at once."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    authority = subprocess.Popen(  # nosec B603
        [
            sys.executable,
            "-c",
            "import time; from torch.distributed import TCPStore; "
            f"s = TCPStore('127.0.0.1', {port}, is_master=True, wait_for_workers=False); "
            "print('listening', flush=True); time.sleep(600)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert authority.stdout.readline().strip() == "listening"
    try:
        gather = startup_gather(_client(port), "g", 0, range(2), 2, _ignore_waits)
        threading.Timer(0.5, authority.kill).start()
        with pytest.raises(RuntimeError, match="lost the rendezvous store"):
            gather(0)
    finally:
        authority.kill()
        authority.wait()


def test_a_replacement_starts_alone(monkeypatch):
    """Its groups have the deployment's layout with only itself live, and never reach the store; each exchange is a
    step of startup progress it reports, which the authority bounds a joiner by. Fails without the change: a
    replacement reported no startup progress."""
    monkeypatch.setattr(
        recovery_module, "_startup_store", lambda: pytest.fail("a replacement used the store")
    )
    monkeypatch.setattr(
        recovery_module, "_mapping", Mapping(world_size=8, rank=5, tp_size=8, gpus_per_node=4)
    )
    agent = Mock()
    monkeypatch.setattr(recovery_module, "_agent", agent)
    monkeypatch.setattr(recovery_module, "_startup_steps", 0)
    world = StartupGroup(RankIdentity(5, 8, 4, 1), range(8))
    assert (world.Get_rank(), world.Get_size()) == (5, 8)
    assert world.allgather("v") == [None] * 5 + ["v", None, None]
    world.Barrier()
    node = world.Split_type(MPI.COMM_TYPE_SHARED)
    assert (node.membership, node.Get_rank(), node.Get_size()) == ((4, 5, 6, 7), 1, 4)
    assert node.allgather(True) == [None, True, None, None]
    assert node.allreduce(3, op=MPI.MIN) == 3
    assert [call.kwargs["progress"] for call in agent.report.call_args_list] == [1, 2, 3, 4]


def test_original_ranks_split_and_reduce_through_the_store(store, monkeypatch):
    monkeypatch.setattr(
        recovery_module, "_mapping", Mapping(world_size=WORLD, tp_size=WORLD, gpus_per_node=2)
    )

    def rank(r):
        world = StartupGroup(RankIdentity(r, WORLD, 2, 0), range(WORLD))
        node = world.Split_type(OMPI_COMM_TYPE_HOST)
        reversed_group = world.subgroup([3, 2, 1, 0])
        world.Barrier()
        return (
            node.membership,
            node.allreduce(r, op=MPI.MIN),
            world.allreduce(r),
            world.allreduce(r, op=MPI.MAX),
            reversed_group.Get_rank(),
            reversed_group.allgather(r),
        )

    results = _on_ranks(range(WORLD), rank)
    for r in range(WORLD):
        node = (0, 1) if r < 2 else (2, 3)
        assert results[r] == (node, node[0], 6, 3, 3 - r, [3, 2, 1, 0])


# Per rank, its split color and key: the even ranks and the odd ones, each in reverse rank order.
_SPLITS = [[0, 3], [1, 2], [0, 1], [1, 0]]


def test_a_split_groups_the_members_of_a_color_in_key_order(store, monkeypatch):
    """As ``MPI_Comm_split``; a replacement takes every member's color and key from the agreed startup and never
    reaches the store. Fails without the change: StartupGroup had no Split."""

    def rank(r):
        world = StartupGroup(RankIdentity(r, WORLD, WORLD, 0), range(WORLD))
        color, key = _SPLITS[r]
        split = world.Split(color, key)
        return split.membership, split.Get_rank(), split.allgather(r)

    results = _on_ranks(range(WORLD), rank)
    assert results[0] == ((2, 0), 1, [2, 0])
    assert results[2] == ((2, 0), 0, [2, 0])
    assert results[1] == ((3, 1), 1, [3, 1])
    assert results[3] == ((3, 1), 0, [3, 1])

    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", True)
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, "2,4,4,1")
    _replay(monkeypatch, {"startup_group_split": [_SPLITS]})
    monkeypatch.setattr(
        recovery_module, "_startup_store", lambda: pytest.fail("a replacement used the store")
    )
    split = StartupGroup(RankIdentity(2, WORLD, WORLD, 1), range(WORLD)).Split(*_SPLITS[2])
    assert (split.membership, split.Get_rank()) == ((2, 0), 0)


def _mirror(store_port, rank, steps):
    """A peer of rank 1's process in test_an_original_rank_starts_without_mpi: the same collectives on its groups."""
    world = StartupGroup(RankIdentity(rank, WORLD, WORLD, 0), range(WORLD))
    node = world.Split_type(OMPI_COMM_TYPE_HOST)  # local_mpi_comm()
    tp = world.subgroup(range(WORLD))  # TpMnnvlMemory.get_comm()
    for step in steps:
        {
            "barrier": world.barrier,
            "allgather": lambda: world.allgather(rank),
            "node_barrier": node.barrier,
            "node_min": lambda: node.allreduce(rank, op=MPI.MIN),
            "tp_allgather": lambda: tp.allgather(rank),
        }[step]()


class _RecordingComm:
    """Records every attribute an MPI communicator is asked for."""

    def __init__(self, comm, calls):
        self._comm, self._calls = comm, calls

    def __getattr__(self, name):
        self._calls.append(name)
        return getattr(self._comm, name)


def test_an_original_rank_starts_without_mpi(store, monkeypatch):
    """Rank 1 of 4, its own MPI world: after join_startup_group, main's startup collectives reach its peers through
    the store, and none goes to mpi4py."""
    calls = []
    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", True)
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, "1,4,4,0")
    monkeypatch.setenv("TLLM_RANK", "-")
    monkeypatch.setattr(_utils, "comm", _RecordingComm(_utils.comm, calls))
    monkeypatch.setattr(_utils, "_local_comm", None)
    monkeypatch.setattr(MPI, "COMM_WORLD", _RecordingComm(MPI.COMM_WORLD, calls))
    monkeypatch.setattr(TpMnnvlMemory, "comm", None)
    monkeypatch.setattr(TpMnnvlMemory, "comm_signature", None)
    # Every rank on one node; set before the peers split by node, as rank 1 sets it in join_startup_group.
    mapping = Mapping(
        world_size=WORLD, rank=1, tp_size=WORLD, moe_ep_size=WORLD, enable_attention_dp=True
    )
    monkeypatch.setattr(recovery_module, "_mapping", mapping)
    steps = ["barrier", "allgather", "node_barrier", "node_min", "tp_allgather"]
    peers = ThreadPoolExecutor(3)
    mirrored = [peers.submit(_mirror, store, rank, steps) for rank in (0, 2, 3)]

    join_startup_group(mapping)
    _utils.mpi_barrier()
    gathered = _utils.mpi_comm().allgather(1)
    _utils.local_mpi_barrier()
    least = _utils.local_mpi_comm().allreduce(1, op=MPI.MIN)
    tp_values = TpMnnvlMemory.get_comm(mapping).allgather(1)
    for peer in mirrored:
        peer.result(timeout=60)
    peers.shutdown()

    assert calls == []
    assert gathered == tp_values == [0, 1, 2, 3]
    assert least == 0
    assert (_utils.mpi_rank(), _utils.mpi_world_size()) == (1, 4)
    assert (_utils.global_mpi_rank(), _utils.global_mpi_size()) == (1, 4)
    assert _utils.local_mpi_size() == 4
    assert recovery_module.os.environ["TLLM_RANK"] == "1"


@pytest.mark.parametrize("incarnation", [0, 1], ids=["original", "replacement"])
def test_the_node_of_every_rank_comes_from_the_mapping(monkeypatch, incarnation):
    """Rank 2 of two nodes of two GPUs, original or replacement: a split by node gives ranks 2 and 3, and joining
    the startup group exchanges and records nothing. Fails without the change: the ranks agreed on their hosts,
    gathered through the store, when they joined."""
    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", True)
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, f"2,{WORLD},2,{incarnation}")
    monkeypatch.setenv("TLLM_RANK", "-")
    monkeypatch.setattr(_utils, "comm", _utils.comm)
    monkeypatch.setattr(_utils, "_local_comm", None)
    monkeypatch.setattr(recovery_module, "_mapping", None)
    _replay(monkeypatch, None)
    store = Mock()
    monkeypatch.setattr(recovery_module, "_startup_store", lambda: store)
    join_startup_group(Mapping(world_size=WORLD, rank=2, tp_size=WORLD, gpus_per_node=2))
    node = _utils.local_mpi_comm()
    assert isinstance(node, StartupGroup)
    assert (node.membership, node.Get_rank()) == ((2, 3), 0)
    assert store.method_calls == []
    assert startup_record.values == {}


@pytest.mark.parametrize(
    "supervisor_gpus, mapping_gpus", [(2, 4), (4, 2), (1, 8)], ids=["fewer", "more", "one-per-node"]
)
def test_a_rank_whose_mapping_places_the_ranks_on_other_nodes_does_not_start(
    monkeypatch, supervisor_gpus, mapping_gpus
):
    """The node supervisor runs ``supervisor_gpus`` ranks per node, the mapping says ``mapping_gpus`` (one node of
    every rank under both is accepted, as in test_an_original_rank_starts_without_mpi). Fails without the change: the
    rank joined, and its splits by node grouped ranks of different nodes."""
    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", True)
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, f"2,{WORLD},{supervisor_gpus},0")
    monkeypatch.setenv("TLLM_RANK", "-")
    monkeypatch.setattr(_utils, "comm", _utils.comm)
    monkeypatch.setattr(recovery_module, "_mapping", None)
    mapping = Mapping(world_size=WORLD, rank=2, tp_size=WORLD, gpus_per_node=mapping_gpus)
    with pytest.raises(
        ValueError, match=f"gpus_per_node {supervisor_gpus}, .* not {mapping_gpus}; turn it off"
    ):
        join_startup_group(mapping)
    assert recovery_module._mapping is None
    assert recovery_module.os.environ["TLLM_RANK"] == "-"


def test_an_original_rank_waits_for_every_rank_to_start(monkeypatch):
    """It returns once a record past STARTING arrives, on the agent's thread. Fails without the change: the wait
    polled the agent's record, which this agent never updates."""
    starting = MembershipRecord(generation=0, state=MembershipState.STARTING, live_mask=0b1111)
    agent = Mock(record=starting)
    monkeypatch.setattr(recovery_module, "_agent", agent)
    monkeypatch.setattr(recovery_module, "_every_rank_started", threading.Event())

    def records():
        time.sleep(0.2)
        recovery_module._on_record_at_startup(starting)
        time.sleep(0.2)
        recovery_module._on_record_at_startup(
            starting.model_copy(update={"state": MembershipState.COMMITTED})
        )

    threading.Thread(target=records, daemon=True).start()
    began = time.monotonic()
    waiter = threading.Thread(target=recovery_module.wait_for_every_rank, daemon=True)
    waiter.start()
    waiter.join(10)
    assert not waiter.is_alive()
    assert time.monotonic() - began >= 0.4
    agent.report.assert_called_once_with(started=True)


def test_a_rank_without_its_identity_does_not_start(monkeypatch):
    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", True)
    monkeypatch.delenv(FAULT_TOLERANCE_RANK_ENV, raising=False)
    with pytest.raises(ValueError, match=FAULT_TOLERANCE_RANK_ENV):
        join_startup_group()


@pytest.mark.parametrize(
    "memory_cls, kwargs, rank, ranks",
    [
        (MnnvlMemory, dict(tp_size=4, moe_ep_size=4), 2, [0, 1, 2, 3]),
        (MnnvlMemory, dict(tp_size=4, moe_tp_size=2, moe_ep_size=2), 2, [2, 3]),
        (TpMnnvlMemory, dict(tp_size=2, pp_size=2), 3, [2, 3]),
    ],
)
def test_an_mnnvl_group_has_the_ranks_split_would_give(
    memory_cls, kwargs, rank, ranks, monkeypatch
):
    """Under fault tolerance the group comes from the mapping, as a subgroup of the startup group."""
    mapping = Mapping(world_size=4, rank=rank, **kwargs)
    assert memory_cls.comm_split_ranks(mapping) == ranks
    world = StartupGroup(RankIdentity(rank, 4, 4, 1), range(4))
    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", True)
    monkeypatch.setattr("tensorrt_llm._torch.distributed.mnnvl_memory.mpi_comm", lambda: world)
    monkeypatch.setattr(memory_cls, "comm", None)
    monkeypatch.setattr(memory_cls, "comm_signature", None)
    comm = memory_cls.get_comm(mapping)
    assert (comm.membership, comm.Get_rank()) == (tuple(ranks), ranks.index(rank))

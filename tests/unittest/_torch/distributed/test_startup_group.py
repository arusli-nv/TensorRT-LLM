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
"""Tests for the startup groups of Wide-EP fault tolerance, where every rank starts as its own process.

The ranks that start together meet in the rendezvous store through ``StoreGroup``. Losing the store ends their
waits. A replacement's groups answer alone. Threads stand in for ranks, and the store is a real TCPStore.
"""

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

from tensorrt_llm import _utils, _wide_ep_fault_tolerance
from tensorrt_llm._torch.distributed.membership import (
    MembershipRecord,
    MembershipState,
    RendezvousStore,
    StoreGroup,
)
from tensorrt_llm._torch.distributed.mnnvl_memory import TpMnnvlMemory
from tensorrt_llm._torch.distributed.startup_record import StartupRecord, startup_record
from tensorrt_llm._torch.pyexecutor import rank_failure_recovery as recovery_module
from tensorrt_llm._torch.pyexecutor.rank_failure_recovery import join_startup_group
from tensorrt_llm._utils import OMPI_COMM_TYPE_HOST
from tensorrt_llm._wide_ep_fault_tolerance import FAULT_TOLERANCE_RANK_ENV
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

SECRET = b"startup-group-secret"
WORLD = 4


def _client(port):
    return RendezvousStore(
        TCPStore("127.0.0.1", port, is_master=False, timeout=timedelta(seconds=30)), SECRET
    )


@pytest.fixture
def store(monkeypatch):
    """Host the rendezvous store as the authority does and return its port.

    Each thread gets its own store client. The threads share one mock membership client for their startup reports.
    """
    server = TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    clients = threading.local()

    def client():
        if not hasattr(clients, "store"):
            clients.store = _client(server.port)
        return clients.store

    monkeypatch.setattr(recovery_module, "_startup_store", client)
    monkeypatch.setattr(recovery_module, "_client", Mock())
    _reset_record(monkeypatch)
    yield server.port


def _reset_record(monkeypatch):
    """Reset this process's startup record."""
    for name, value in vars(StartupRecord()).items():
        monkeypatch.setattr(startup_record, name, value)


def _on_ranks(ranks, fn):
    """Run ``fn(rank)`` on one thread per rank and return the results or raised errors by rank."""

    def run(rank):
        try:
            return fn(rank)
        except Exception as error:  # surface thread failures in the test
            return error

    with ThreadPoolExecutor(len(ranks)) as pool:
        return dict(zip(ranks, pool.map(run, ranks)))


def test_starting_ranks_gather_through_store(store):
    def rank(r):
        world = StoreGroup(_client(store), "w", r, range(WORLD), gpus_per_node=2)
        pairs = StoreGroup(_client(store), "pair", r, range(WORLD), active=[1, 3])
        node = world.Split_type(OMPI_COMM_TYPE_HOST)
        reversed_group = world.subgroup([3, 2, 1, 0])
        world.Barrier()
        return (
            world.allgather(f"x{r}"),
            r in (1, 3) and pairs.allgather(r),
            node.membership,
            node.allreduce(r, op=MPI.MIN),
            (world.allreduce(r), world.allreduce(r, op=MPI.MAX)),
            (reversed_group.Get_rank(), reversed_group.allgather(r)),
        )

    results = _on_ranks(range(WORLD), rank)
    for r in range(WORLD):
        node = (0, 1) if r < 2 else (2, 3)
        assert results[r][0] == ["x0", "x1", "x2", "x3"]
        assert results[r][2:] == (node, node[0], (6, 3), (3 - r, [3, 2, 1, 0]))
    assert results[1][1] == [None, 1, None, 3]
    assert results[3][1] == results[1][1]


def test_startup_collective_reports_waits_and_steps(store, monkeypatch):
    """Check that a startup collective reports each change of the peers it waits for, then a finished step.

    The authority's startup hang rule reads these reports. A wait that keeps polling reports nothing new.
    """
    client = recovery_module._client
    monkeypatch.setattr(recovery_module, "_startup_steps", 0)
    monkeypatch.setattr(recovery_module, "_startup_waiting_for", 0)

    def late_peer():
        time.sleep(0.3)
        StoreGroup(_client(store), "g", 1, range(2)).allgather(1)

    peer = threading.Thread(target=late_peer)
    peer.start()
    try:
        group = StoreGroup(_client(store), "g", 0, range(2), recovery_module._report_startup_wait)
        assert group.allgather(0) == [0, 1]
    finally:
        peer.join(30)
    assert [call.kwargs for call in client.report.call_args_list] == [
        dict(waiting_for=0b10, progress=0),
        dict(waiting_for=0, progress=1),
    ]


def test_startup_ends_when_authority_fails(monkeypatch):
    """Check that a wait for a peer ends at once when the authority process, which hosts the store, is gone."""
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
        group = StoreGroup(_client(port), "g", 0, range(2))
        threading.Timer(0.5, authority.kill).start()
        with pytest.raises(RuntimeError, match="lost the rendezvous store"):
            group.allgather(0)
        assert group.exchange_failed
    finally:
        authority.kill()
        authority.wait()


def test_a_replacement_starts_alone(monkeypatch):
    """Check that a replacement's groups have the deployment's layout with only the replacement active.

    The groups never reach the store. Each exchange reports a step of startup progress, which the authority uses to
    bound a joining rank.
    """
    client = Mock()
    monkeypatch.setattr(recovery_module, "_client", client)
    monkeypatch.setattr(recovery_module, "_startup_steps", 0)
    world = StoreGroup(
        None, "w", 5, range(8), recovery_module._report_startup_wait, active=[5], gpus_per_node=4
    )
    assert (world.Get_rank(), world.Get_size()) == (5, 8)
    assert world.allgather("v") == [None] * 5 + ["v", None, None]
    world.Barrier()
    node = world.Split_type(MPI.COMM_TYPE_SHARED)
    assert (node.membership, node.Get_rank(), node.Get_size()) == ((4, 5, 6, 7), 1, 4)
    assert node.allgather(True) == [None, True, None, None]
    assert node.allreduce(3, op=MPI.MIN) == 3
    assert [call.kwargs["progress"] for call in client.report.call_args_list] == [1, 2, 3, 4]


def test_replacement_bcast_ignores_own_value(store, monkeypatch):
    """Check that a replacement's value to a broadcast from rank 0 need not match its predecessor's."""
    startup_record.record()  # in rank 0's process only

    def bcast(group, value):
        # One call site, since a replacement makes the original ranks' collectives from the same callers.
        return group.bcast(value)

    def rank(r):
        return bcast(StoreGroup(_client(store), "w", r, range(2), startup=r == 0), f"rank {r}")

    assert _on_ranks(range(2), rank) == {0: "rank 0", 1: "rank 0"}
    published = startup_record.published()
    _reset_record(monkeypatch)
    startup_record.replay(published)
    replacement = StoreGroup(None, "w", 1, range(2), active=[1], startup=True)
    assert bcast(replacement, "replacement") == "rank 0"


def _mirror(store_port, rank, steps):
    """Run the collectives of rank 1's peers in test_an_original_rank_starts_without_mpi."""
    world = StoreGroup(
        recovery_module._startup_store(), "startup/world", rank, range(WORLD), gpus_per_node=WORLD
    )
    # The groups of local_mpi_comm() and TpMnnvlMemory.get_comm().
    node = world.Split_type(OMPI_COMM_TYPE_HOST)
    tp = world.subgroup(range(WORLD))
    for step in steps:
        {
            "barrier": world.barrier,
            "allgather": lambda: world.allgather(rank),
            "node_barrier": node.barrier,
            "node_min": lambda: node.allreduce(rank, op=MPI.MIN),
            "tp_allgather": lambda: tp.allgather(rank),
        }[step]()


class _RecordingComm:
    """Record every attribute that an MPI communicator is asked for."""

    def __init__(self, comm, calls):
        self._comm, self._calls = comm, calls

    def __getattr__(self, name):
        self._calls.append(name)
        return getattr(self._comm, name)


def test_an_original_rank_starts_without_mpi(store, monkeypatch):
    """Check that rank 1 of 4, alone in its MPI world, reaches its peers through the store.

    After join_startup_group, no startup collective goes to mpi4py.
    """
    calls = []
    monkeypatch.setattr(_wide_ep_fault_tolerance, "_enabled", True)
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, "1,4,4,0")
    monkeypatch.setenv("TLLM_RANK", "-")
    monkeypatch.setattr(_utils, "comm", _RecordingComm(_utils.comm, calls))
    monkeypatch.setattr(_utils, "world_comm", _utils.world_comm)
    monkeypatch.setattr(_utils, "_local_comm", None)
    monkeypatch.setattr(MPI, "COMM_WORLD", _RecordingComm(MPI.COMM_WORLD, calls))
    monkeypatch.setattr(TpMnnvlMemory, "comm", None)
    monkeypatch.setattr(TpMnnvlMemory, "comm_signature", None)
    # Every rank on one node.
    mapping = Mapping(
        world_size=WORLD, rank=1, tp_size=WORLD, moe_ep_size=WORLD, enable_attention_dp=True
    )
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
    assert gathered == [0, 1, 2, 3]
    assert tp_values == gathered
    assert least == 0
    assert (_utils.mpi_rank(), _utils.mpi_world_size()) == (1, 4)
    assert (_utils.global_mpi_rank(), _utils.global_mpi_size()) == (1, 4)
    assert _utils.local_mpi_size() == 4
    assert recovery_module.os.environ["TLLM_RANK"] == "1"


def test_original_rank_waits_for_every_rank(monkeypatch):
    """Check that the wait returns once a record past STARTING arrives on the client's thread.

    The client's own ``record`` attribute never changes, so a wait that polls it would not return.
    """
    starting = MembershipRecord(generation=0, state=MembershipState.STARTING, active_mask=0b1111)
    client = Mock(record=starting, rank=1)
    monkeypatch.setattr(recovery_module, "_client", client)
    monkeypatch.setattr(recovery_module, "_every_rank_started", threading.Event())

    def records():
        time.sleep(0.2)
        recovery_module._on_record_at_startup(starting)
        time.sleep(0.2)
        recovery_module._on_record_at_startup(
            starting.model_copy(update={"state": MembershipState.COMMITTED})
        )

    delivery = threading.Thread(target=records, daemon=True)
    delivery.start()
    began = time.monotonic()
    waiter = threading.Thread(target=recovery_module.wait_for_every_rank, daemon=True)
    waiter.start()
    waiter.join(10)
    delivery.join(10)
    assert not waiter.is_alive()
    assert not delivery.is_alive()
    assert time.monotonic() - began >= 0.4
    client.report.assert_called_once_with(started=True)

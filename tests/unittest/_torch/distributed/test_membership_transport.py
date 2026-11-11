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
"""Tests for the authority process and the membership clients in separate CPU processes.

They run over the rendezvous store and the liveness connections.
"""

import multiprocessing as mp
import os
import queue
import signal
import socket
import time
from datetime import timedelta

import pytest
from torch.distributed import TCPStore

from tensorrt_llm._torch.distributed.membership import (
    LIVENESS_KEY,
    RECORD_KEY,
    Authority,
    AuthorityProcess,
    MembershipClient,
    MembershipState,
    NodeView,
    RendezvousStore,
    Reply,
    open_liveness_connection,
    open_node_supervisor_connection,
)

pytestmark = pytest.mark.cpu_only

WORLD = 4
SECRET = b"test-shared-secret"
STARTUP_S = 180  # each spawned process imports tensorrt_llm


def _store(port, secret=SECRET):
    return RendezvousStore(
        TCPStore("127.0.0.1", port, is_master=False, timeout=timedelta(seconds=60)), secret
    )


class _ReportingAuthorityProcess(AuthorityProcess):
    """Authority process that puts ``("refused", rank, replacement, reason)`` on ``results`` for each refusal."""

    def __init__(self, results, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._results = results

    def _refusal(self, rank, replacement):
        reason = super()._refusal(rank, replacement)
        if reason is not None:
            self._results.put(("refused", rank, replacement, reason))
        return reason


def _authority(port, results):
    process = _ReportingAuthorityProcess(results, WORLD, _store(port), Authority(WORLD))
    record = process.run(host="127.0.0.1")
    results.put(("authority", record.state.name))


# The test sets this key to shut the cluster down, and rank 0 then reports a shutdown as its executor does.
_STOP_KEY = "test_stop"


def _rank(port, rank, replacement, results):
    """Answer the records as an executor does until the deployment shuts down or the rank leaves."""
    client = None

    def on_record(record):
        if record.state == MembershipState.PROPOSE:
            client.report(
                reply=Reply.READY, reply_generation=record.generation, ready_payload=f"rank {rank}"
            )
        elif record.state == MembershipState.INSTALL:
            client.report(reply=Reply.INSTALLED, reply_generation=record.generation)
        results.put(
            (
                rank,
                client.incarnation,
                record.generation,
                record.state.name,
                record.active_mask,
                time.time(),
            )
        )

    store = _store(port)
    client = MembershipClient(rank, store, on_record, replacement=replacement)
    client.start()
    if replacement:
        client.report(joining_ready=True, ready_payload=f"rank {rank}")
    else:
        client.report(started=True)
    progress = 0
    while client._thread.is_alive():
        progress += 1
        client.report(
            progress=progress, shutdown=rank == 0 and store.get_if_set(_STOP_KEY) is not None
        )
        if client.record is not None and client.record.state in (
            MembershipState.SHUTDOWN,
            MembershipState.ABORTED,
        ):
            break
        time.sleep(0.01)
    results.close()
    results.join_thread()  # deliver what was reported before exiting


def _until(results, predicate, timeout_s=30):
    seen = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            item = results.get(timeout=1)
        except queue.Empty:
            continue
        seen.append(item)
        if predicate(item):
            return seen
    raise AssertionError(f"timed out; seen {seen}")


def _until_all(results, *predicates, timeout_s=30):
    pending = list(predicates)
    deadline = time.monotonic() + timeout_s
    while pending and time.monotonic() < deadline:
        try:
            item = results.get(timeout=1)
        except queue.Empty:
            continue
        pending = [p for p in pending if not p(item)]
    assert not pending, f"{len(pending)} condition(s) not met"


@pytest.fixture
def cluster():
    ctx = mp.get_context("spawn")
    master = TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    results = ctx.Queue()
    procs = {}

    def start(name, target, *args):
        proc = ctx.Process(target=target, args=(master.port, *args, results))
        proc.start()
        procs[name] = proc
        return proc

    def start_cluster():
        start("authority", _authority)
        ranks = [start(r, _rank, r, False) for r in range(WORLD)]
        _until_all(
            results,
            *(lambda item, r=r: item[0] == r and item[3] == "COMMITTED" for r in range(WORLD)),
            timeout_s=STARTUP_S,
        )
        return ranks

    yield RendezvousStore(master, SECRET), results, start, start_cluster
    RendezvousStore(master, SECRET).set(_STOP_KEY, b"")
    _until(
        results, lambda item: item == ("authority", "SHUTDOWN")
    )  # the authority survived the test
    for proc in procs.values():
        proc.join(timeout=30)
        if proc.is_alive():
            proc.kill()


def _committed(rank, generation, mask):
    return lambda item: item[0] == rank and item[2:5] == (generation, "COMMITTED", mask)


def _kill_and_expect_removal(results, proc, rank, generation, mask):
    killed_at = time.time()
    os.kill(proc.pid, signal.SIGKILL)
    # Items are (rank, incarnation, generation, state, active mask, time).
    seen = _until(results, _committed(0, generation, mask))
    assert seen[-1][5] - killed_at < 5.0, f"removed {seen[-1][5] - killed_at:.2f} s after the kill"


def test_killed_rank_removed_replacement_joins(cluster):
    _, results, start, start_cluster = cluster
    ranks = start_cluster()
    _kill_and_expect_removal(results, ranks[3], 3, 1, 0b0111)
    start("replacement-3", _rank, 3, True)
    _until(results, _committed(0, 2, 0b1111), timeout_s=STARTUP_S)


def test_hung_rank_keeps_connection(cluster):
    _, results, _, start_cluster = cluster
    ranks = start_cluster()
    # No Python runs in the stopped process, as in a starved interpreter. Nobody waits on it here.
    os.kill(ranks[2].pid, signal.SIGSTOP)
    time.sleep(4.0)
    os.kill(ranks[2].pid, signal.SIGCONT)
    with pytest.raises(AssertionError):
        _until(results, lambda item: item[3] == "PROPOSE", timeout_s=2)
    # The authority still handles failures.
    _kill_and_expect_removal(results, ranks[1], 1, 1, 0b1101)


def test_replacement_waits_for_old_process(cluster):
    """Refuse a replacement for rank 2 while its process serves, and admit one after the process is killed."""
    _, results, start, start_cluster = cluster
    ranks = start_cluster()
    start("refused-replacement-2", _rank, 2, True)
    with pytest.raises(AssertionError):
        _until(results, lambda item: item[3] == "PROPOSE", timeout_s=STARTUP_S / 6)
    _kill_and_expect_removal(results, ranks[2], 2, 1, 0b1011)
    start("replacement-2", _rank, 2, True)
    _until(results, _committed(0, 2, 0b1111), timeout_s=STARTUP_S)


def _refused(store, rank, replacement):
    """Return whether the authority closes a liveness connection for ``rank`` unread."""
    with open_liveness_connection(store, rank, replacement) as connection:
        connection.sock.settimeout(10)
        return connection.sock.recv(1) == b""


def test_join_for_active_rank_refused(cluster):
    """Refuse a replacement for rank 2 while its process is connected, and keep the membership."""
    store, results, _, start_cluster = cluster
    start_cluster()
    assert _refused(store, 2, True)
    _until(
        results,
        lambda item: item == ("refused", 2, 1, "rank 2 is active: no replacement joins for it"),
    )
    with pytest.raises(AssertionError):
        _until(results, lambda item: item[3] == "PROPOSE", timeout_s=2)


def test_second_connection_is_refused(cluster):
    """Refuse a second connection of rank 1's first process, and keep watching the first one."""
    store, results, _, start_cluster = cluster
    ranks = start_cluster()
    assert _refused(store, 1, False)
    _until(
        results,
        lambda item: item == ("refused", 1, 0, "rank 1's first process is connected already"),
    )
    with pytest.raises(AssertionError):
        _until(results, lambda item: item[3] == "PROPOSE", timeout_s=2)
    _kill_and_expect_removal(results, ranks[1], 1, 1, 0b1101)


def test_unsigned_input_is_ignored(cluster):
    store, results, _, start_cluster = cluster
    ranks = start_cluster()
    forged = RendezvousStore(store.store, b"another-secret")
    host, port = store.get(LIVENESS_KEY).decode().rsplit(":", 1)
    with socket.create_connection((host, int(port))) as garbage:
        garbage.sendall(b"zz not-hex\n")
    with socket.create_connection((host, int(port))) as hello:
        nonce = bytes.fromhex(hello.recv(64).decode().strip())
        line = b"3 9"
        hello.sendall(forged.tag(b"hello", nonce + line).hex().encode() + b" " + line + b"\n")
        hello.settimeout(10)
        assert hello.recv(1) == b""  # a forged replacement for rank 3 is closed
    forged.set(RECORD_KEY, b'{"generation": 9, "state": 3, "active_mask": 0}')  # a forged ABORTED
    with pytest.raises(AssertionError):
        _until(results, lambda item: item[3] in ("PROPOSE", "ABORTED"), timeout_s=2)
    assert store.get_if_set(RECORD_KEY) is None  # counts as absent for the watchers and the server
    forged.set("probe", b"value")
    with pytest.raises(PermissionError):
        store.get("probe")
    # A signed ("row/1", "2x") does not verify as ("row/12", "x").
    store.set("row/1", b"2x")
    raw = store.store.get("row/1")
    store.store.set("row/12", raw[:32] + raw[33:])
    with pytest.raises(PermissionError):
        store.get("row/12")
    # The authority still handles failures.
    _kill_and_expect_removal(results, ranks[2], 2, 1, 0b1011)


class _NodeSupervisor:
    """Node supervisor connection that reads the views the authority sends."""

    def __init__(self, store, node, ranks):
        self.connection = open_node_supervisor_connection(store, node, ranks, f"host-{node}")
        self.connection.sock.settimeout(10)

    def next_view(self) -> NodeView:
        """Return the next view, or None once the authority closed the connection."""
        while True:
            data = self.connection.sock.recv(1 << 16)
            if not data:
                return None
            lines = self.connection.lines.decode(data)
            if lines:
                return NodeView.model_validate_json(lines[-1])

    def until(self, predicate) -> NodeView:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            view = self.next_view()
            if view is None or predicate(view):
                return view
        raise AssertionError("no such view")


def test_node_supervisor_loss_fails_its_ranks(cluster):
    """Remove a lost supervisor's ranks at once and tell the other supervisors that its node left.

    The ranks' processes still run, because the supervisor's loss stands for the loss of its node.
    """
    store, results, _, start_cluster = cluster
    start_cluster()
    node_0 = _NodeSupervisor(store, 0, [0, 1])
    node_1 = _NodeSupervisor(store, 1, [2, 3])
    view = node_0.until(lambda view: len(view.nodes) == 2)
    assert view.nodes == ((0, "host-0", True), (1, "host-1", True))
    assert view.record.state == MembershipState.COMMITTED
    left_at = time.time()
    node_1.connection.sock.close()
    seen = _until(results, _committed(0, 1, 0b0011))
    assert seen[-1][5] - left_at < 5.0
    view = node_0.until(lambda view: view.record.active_mask == 0b0011)
    assert view.nodes == ((0, "host-0", True), (1, "host-1", False))
    # Shut down while node 0's supervisor stays connected, because its loss would abort the deployment.
    store.set(_STOP_KEY, b"")
    node_0.until(lambda view: view.record.state == MembershipState.SHUTDOWN)


def test_second_node_supervisor_is_refused(cluster):
    """Refuse a second supervisor of a node or of one of its ranks without changing the membership."""
    store, results, _, start_cluster = cluster
    start_cluster()
    first = _NodeSupervisor(store, 1, [2, 3])
    assert first.next_view() is not None
    assert _NodeSupervisor(store, 1, [2, 3]).next_view() is None
    assert _NodeSupervisor(store, 2, [3]).next_view() is None
    with pytest.raises(AssertionError):
        _until(results, lambda item: item[3] == "PROPOSE", timeout_s=2)
    # Every supervisor's view lists a supervisor that connects.
    node_0 = _NodeSupervisor(store, 0, [0, 1])
    expected = ((0, "host-0", True), (1, "host-1", True))
    assert first.until(lambda view: view.nodes == expected).record.active_mask == 0b1111
    # Shut down while node 0's supervisor stays connected, because its loss would abort the deployment.
    store.set(_STOP_KEY, b"")
    node_0.until(lambda view: view.record.state == MembershipState.SHUTDOWN)

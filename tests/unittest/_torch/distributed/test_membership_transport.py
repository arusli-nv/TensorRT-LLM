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
"""The authority process and the ranks' agents over the signed store and liveness connections (CPU processes): a
killed rank is removed within seconds, a frozen one is kept, a replacement joins once the process it replaces is
gone, and connections or values not signed with the join secret are ignored without disturbing the authority."""

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
    MembershipAgent,
    MembershipState,
    NodeView,
    Reply,
    SignedStore,
    open_liveness_connection,
    open_node_supervisor_connection,
)

pytestmark = pytest.mark.cpu_only

WORLD = 4
SECRET = b"test-join-secret"
STARTUP_S = 180  # each spawned process imports tensorrt_llm


def _store(port, secret=SECRET):
    return SignedStore(
        TCPStore("127.0.0.1", port, is_master=False, timeout=timedelta(seconds=60)), secret
    )


class _ReportingAuthorityProcess(AuthorityProcess):
    """The authority process, which also puts ("refused", rank, replacement, reason) on ``results`` for each valid
    hello it refuses."""

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


# The test sets it once the cluster is to shut down; rank 0 then reports so, as its executor does when it stops.
_STOP_KEY = "test_stop"


def _rank(port, rank, replacement, results):
    """Follow the records and answer them like an executor would, until shut down or no longer a member. A
    ``replacement`` asks to join by connecting, and then reports that it is ready to join."""
    agent = None

    def on_record(record):
        if record.state == MembershipState.PROPOSE:
            agent.report(
                reply=Reply.READY, reply_generation=record.generation, ready_payload=f"rank {rank}"
            )
        elif record.state == MembershipState.INSTALL:
            agent.report(reply=Reply.INSTALLED, reply_generation=record.generation)
        results.put(
            (
                rank,
                agent.incarnation,
                record.generation,
                record.state.name,
                record.live_mask,
                time.time(),
            )
        )

    store = _store(port)
    agent = MembershipAgent(rank, store, on_record, replacement=replacement)
    agent.start()
    if replacement:
        agent.report(joining_ready=True, ready_payload=f"rank {rank}")
    else:
        agent.report(started=True)
    progress = 0
    while agent._thread.is_alive():
        progress += 1
        agent.report(
            progress=progress, shutdown=rank == 0 and store.get_if_set(_STOP_KEY) is not None
        )
        if agent.record is not None and agent.record.state in (
            MembershipState.SHUTDOWN,
            MembershipState.ABORT,
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

    yield SignedStore(master, SECRET), results, start, start_cluster
    SignedStore(master, SECRET).set(_STOP_KEY, b"")
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
    # Items are (rank, incarnation, generation, state, live mask, time).
    seen = _until(results, _committed(0, generation, mask))
    assert seen[-1][5] - killed_at < 5.0, f"removed {seen[-1][5] - killed_at:.2f} s after the kill"


def test_a_killed_rank_is_removed_quickly_and_its_replacement_joins(cluster):
    _, results, start, start_cluster = cluster
    ranks = start_cluster()
    _kill_and_expect_removal(results, ranks[3], 3, 1, 0b0111)
    start("replacement-3", _rank, 3, True)
    _until(results, _committed(0, 2, 0b1111), timeout_s=STARTUP_S)


def test_a_frozen_rank_keeps_its_connection_and_stays_live(cluster):
    _, results, _, start_cluster = cluster
    ranks = start_cluster()
    os.kill(
        ranks[2].pid, signal.SIGSTOP
    )  # no Python runs, like a starved interpreter; nobody waits on it here
    time.sleep(4.0)
    os.kill(ranks[2].pid, signal.SIGCONT)
    with pytest.raises(AssertionError):
        _until(results, lambda item: item[3] == "PROPOSE", timeout_s=2)
    _kill_and_expect_removal(
        results, ranks[1], 1, 1, 0b1101
    )  # the authority still handles failures


def test_a_replacement_joins_once_the_process_it_replaces_is_gone(cluster):
    """While rank 2's process serves, a replacement for it is refused: only the authority's verdict removes a live
    process."""
    _, results, start, start_cluster = cluster
    ranks = start_cluster()
    start("refused-replacement-2", _rank, 2, True)
    with pytest.raises(AssertionError):
        _until(results, lambda item: item[3] == "PROPOSE", timeout_s=STARTUP_S / 6)
    _kill_and_expect_removal(results, ranks[2], 2, 1, 0b1011)
    start("replacement-2", _rank, 2, True)
    _until(results, _committed(0, 2, 0b1111), timeout_s=STARTUP_S)


def _refused(store, rank, replacement):
    """Open a liveness connection for ``rank``, of a ``replacement`` or of its first process; returns whether the
    authority closed it unread."""
    with open_liveness_connection(store, rank, replacement) as connection:
        connection.sock.settimeout(10)
        return connection.sock.recv(1) == b""


def test_a_join_for_a_live_rank_is_refused(cluster):
    """A replacement's hello for rank 2 while rank 2's process holds its connection is refused, with the reason; the
    membership does not change."""
    store, results, _, start_cluster = cluster
    start_cluster()
    assert _refused(store, 2, True)
    _until(
        results,
        lambda item: item == ("refused", 2, 1, "rank 2 is live: no replacement joins for it"),
    )
    with pytest.raises(AssertionError):
        _until(results, lambda item: item[3] == "PROPOSE", timeout_s=2)


def test_a_second_connection_of_a_first_process_is_refused(cluster):
    """A second hello of rank 1's connected first process is refused, with the reason; the process's own connection
    still tells when it ends."""
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


def test_input_not_signed_with_the_join_secret_is_ignored(cluster):
    store, results, _, start_cluster = cluster
    ranks = start_cluster()
    forged = SignedStore(store.store, b"another-secret")
    host, port = store.get(LIVENESS_KEY).decode().rsplit(":", 1)
    with socket.create_connection((host, int(port))) as garbage:
        garbage.sendall(b"zz not-hex\n")
    with socket.create_connection((host, int(port))) as hello:
        nonce = bytes.fromhex(hello.recv(64).decode().strip())
        line = b"3 9"
        hello.sendall(forged.tag(b"hello", nonce + line).hex().encode() + b" " + line + b"\n")
        hello.settimeout(10)
        assert hello.recv(1) == b""  # a forged replacement for rank 3 is closed
    forged.set(RECORD_KEY, b'{"generation": 9, "state": 3, "live_mask": 0}')  # a forged ABORT
    with pytest.raises(AssertionError):
        _until(results, lambda item: item[3] in ("PROPOSE", "ABORT"), timeout_s=2)
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
    _kill_and_expect_removal(
        results, ranks[2], 2, 1, 0b1011
    )  # the authority still handles failures


class _NodeSupervisor:
    """The authority's side of a node supervisor: its connection, and the views the authority sends over it."""

    def __init__(self, store, node, ranks):
        self.connection = open_node_supervisor_connection(store, node, ranks, f"host-{node}")
        self.connection.sock.settimeout(10)

    def next_view(self) -> NodeView:
        """The next view that arrives, None once the authority closed the connection."""
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


def test_a_node_supervisor_s_loss_fails_its_ranks_at_once(cluster):
    """The supervisor's death is its node's loss: the authority removes its ranks, whose processes still run, and
    tells the other node supervisors that the node's supervisor left (node 0's relaunches it)."""
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
    view = node_0.until(lambda view: view.record.live_mask == 0b0011)
    assert view.nodes == ((0, "host-0", True), (1, "host-1", False))
    # The deployment shuts down while node 0's supervisor stays: its leaving would be node 0's loss, which aborts.
    store.set(_STOP_KEY, b"")
    node_0.until(lambda view: view.record.state == MembershipState.SHUTDOWN)


def test_a_second_supervisor_of_a_node_or_of_its_ranks_is_refused(cluster):
    """At most one node supervisor holds a node and each of its ranks; a refused one changes no membership and no
    view (the authority sends none for it), and the first one keeps its node."""
    store, results, _, start_cluster = cluster
    start_cluster()
    first = _NodeSupervisor(store, 1, [2, 3])
    assert first.next_view() is not None
    assert _NodeSupervisor(store, 1, [2, 3]).next_view() is None
    assert _NodeSupervisor(store, 2, [3]).next_view() is None
    with pytest.raises(AssertionError):
        _until(results, lambda item: item[3] == "PROPOSE", timeout_s=2)
    node_0 = _NodeSupervisor(
        store, 0, [0, 1]
    )  # a supervisor that connects: every supervisor's view lists it
    expected = ((0, "host-0", True), (1, "host-1", True))
    assert first.until(lambda view: view.nodes == expected).record.live_mask == 0b1111
    # The deployment shuts down while node 0's supervisor stays: its leaving would be node 0's loss, which aborts.
    store.set(_STOP_KEY, b"")
    node_0.until(lambda view: view.record.state == MembershipState.SHUTDOWN)

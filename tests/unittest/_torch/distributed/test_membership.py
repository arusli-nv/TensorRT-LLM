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
"""The Wide-EP fault-tolerance authority's decisions, driven by simulated ranks (no GPU, no I/O)."""

import contextlib
import json
import os
import random
import socket
import threading
import time
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.distributed import membership
from tensorrt_llm._torch.distributed.membership import (
    RECORD_KEY,
    Authority,
    AuthorityProcess,
    MembershipClient,
    MembershipRecord,
    MembershipState,
    NodeView,
    RankReport,
    RendezvousStore,
    Reply,
    ResumeState,
    open_liveness_connection,
    open_node_supervisor_connection,
)

pytestmark = pytest.mark.cpu_only

WORLD = 4
ALL = (1 << WORLD) - 1


def _without(*ranks):
    """Return the mask of every rank except ``ranks``."""
    return ALL & ~sum(1 << rank for rank in ranks)


@pytest.fixture(autouse=True)
def _report_interval(monkeypatch):
    """Shorten the report and decision interval for the tests."""
    monkeypatch.setattr(membership, "REPORT_INTERVAL_S", 0.05)


def _formed(authority):
    """Start every rank and return ``authority`` with generation 0 committed."""
    world = authority.world_size
    reports = {r: RankReport(started=True) for r in range(world)}
    assert authority.tick(0.0, reports, (1 << world) - 1).state == MembershipState.COMMITTED
    return authority


class Cluster:
    """Simulate ranks that report their progress and reply to the authority's records."""

    def __init__(self, world=WORLD, **timeouts):
        self.authority = _formed(Authority(world, **timeouts))
        self.now = 0.0
        self.alive = set(range(world))  # processes that hold their connection
        self.hung = set()  # connected but making no progress
        self.stopped = set()  # stopped by SIGSTOP, still connected, but no longer reporting
        self.idle = set()  # main's hang detector paused, as while waiting for requests
        self.shutdown = set()  # ranks whose report says that the deployment ends normally
        self.waiting_for = {}  # rank -> mask it waits for
        self.waited_rounds = {}  # rank -> ((peer, round flag its kernels wait for), ...)
        self.waited_epochs = {}  # rank -> ((peer, epoch of the host-exchange round it waits in), ...)
        self.installing = set()  # completing a step of an install at every report
        self.unprepared = set()  # replacements that have not reported they are ready to join
        self.install_reply = {r: Reply.INSTALLED for r in range(world)}
        self.digest = {r: "placement-a" for r in range(world)}  # of the placement a rank restored
        self.reports = {r: RankReport() for r in range(world)}
        self.paused_at = {r: ResumeState(iteration=10 + r, epoch=100 + r) for r in range(world)}
        self.published = []
        self.published_at = []

    def step(self, seconds=0.05):
        self.now += seconds
        record = self.authority.record
        for r in self.alive:
            if r in self.stopped:
                continue
            old = self.reports[r]
            reply, generation = old.reply, old.reply_generation
            if r not in self.hung and record.is_active(r):
                if record.state == MembershipState.PROPOSE:
                    reply, generation = Reply.READY, record.generation
                elif record.state == MembershipState.INSTALL:
                    reply, generation = self.install_reply[r], record.generation
            busy = r not in self.hung and r not in self.idle
            self.reports[r] = RankReport(
                sequence=old.sequence + 1,
                progress=old.progress + (1 if busy else 0),
                idle=r in self.idle,
                waiting_for=self.waiting_for.get(r, 0),
                waited_rounds=self.waited_rounds.get(r, ()),
                waited_epochs=self.waited_epochs.get(r, ()),
                install_steps=old.install_steps + (1 if r in self.installing else 0),
                reply_generation=generation,
                reply=reply,
                paused_at=self.paused_at[r],
                placement_digest=self.digest[r],
                ready_payload=f"payload of rank {r}",
                joining_ready=r not in self.unprepared,
                shutdown=r in self.shutdown,
            )
        connected = sum(1 << r for r in self.alive)
        new = self.authority.tick(self.now, self.reports, connected)
        if new is not None:
            self.published.append(new)
            self.published_at.append(self.now)
        return new

    def run(self, seconds):
        for _ in range(round(seconds / 0.05)):
            self.step()
        return self.authority.record


def states(cluster):
    return [r.state for r in cluster.published]


TRANSACTION = [MembershipState.PROPOSE, MembershipState.INSTALL, MembershipState.COMMITTED]


def test_healthy_ranks_change_nothing():
    cluster = Cluster()
    cluster.run(5.0)
    assert cluster.published == []
    assert cluster.authority.record == MembershipRecord(
        generation=0, state=MembershipState.COMMITTED, active_mask=ALL
    )


@pytest.mark.parametrize(
    "epochs, resume_epoch", [({}, 104), ({0: 0xFFFFFFFE, 1: 1, 3: 1}, 2)], ids=["plain", "wrapped"]
)
def test_closed_connection_removes_the_rank(epochs, resume_epoch):
    cluster = Cluster()
    for r, epoch in epochs.items():  # ranks 1 and 3 went on past the wrap
        cluster.paused_at[r] = ResumeState(iteration=10 + r, epoch=epoch)
    cluster.alive.discard(2)
    record = cluster.run(1.0)
    assert states(cluster) == TRANSACTION
    assert record.generation == 1
    assert record.active_mask == _without(2)
    # Resume from the furthest survivor, with the next epoch.
    assert record.resume.iteration == 13
    assert record.resume.epoch == resume_epoch


def test_install_carries_ready_payloads():
    """Pass each active rank's READY payload unchanged in the INSTALL record only, and keep it out of the log."""
    cluster = Cluster()
    cluster.alive.discard(2)
    cluster.run(1.0)
    _join(cluster, 2)
    cluster.published.clear()
    cluster.run(1.0)
    install = next(r for r in cluster.published if r.state == MembershipState.INSTALL)
    assert install.ready_payloads == tuple((r, f"payload of rank {r}") for r in range(WORLD))
    assert all(r.ready_payloads == () for r in cluster.published if r is not install)
    assert "payload" not in repr(install)
    assert MembershipRecord.model_validate_json(install.model_dump_json()) == install
    lowest = MembershipRecord(
        generation=2, state=MembershipState.PROPOSE, active_mask=ALL, joining_mask=0b0101
    )
    assert lowest.lowest_survivor() == 1
    assert lowest.model_copy(update=dict(active_mask=0b0101)).lowest_survivor() is None


def test_slow_rank_kept_hung_rank_removed():
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.hung.add(1)
    cluster.run(3.0)  # nobody waits for rank 1, so it is slow, not hung
    assert cluster.authority.record.active_mask == ALL
    cluster.waiting_for[0] = 1 << 1
    record = cluster.run(3.0)
    assert record.state == MembershipState.COMMITTED
    assert not record.is_active(1)


@pytest.mark.parametrize("waits_in", ["all-to-all", "host exchange"])
def test_delivered_rounds_are_not_a_hang(waits_in):
    """Keep a rank whose iteration stalls while every round its peers wait for still arrives."""
    cluster = Cluster(hang_timeout_s=1.0)
    rounds = cluster.waited_rounds if waits_in == "all-to-all" else cluster.waited_epochs
    cluster.hung.add(3)  # its iteration does not advance
    cluster.waiting_for.update({r: 1 << 3 for r in range(3)})
    for round_ in range(1, 13):  # a new round every 0.25 s for 3 s
        rounds.update({r: ((3, round_),) for r in range(3)})
        cluster.run(0.25)
    assert cluster.published == []
    record = cluster.run(1.5)  # the same round for longer than the hang timeout
    assert record.state == MembershipState.COMMITTED
    assert not record.is_active(3)


# The hang rule. Only the end of a wait-for chain that does not move for the hang timeout is hung.


def _first_change(cluster, seconds):
    """Return the first record the authority publishes within ``seconds``, or None."""
    for _ in range(round(seconds / 0.05)):
        if cluster.step() is not None:
            return cluster.published[-1]
    return None


def _release(cluster):
    """Resume the active ranks after the authority removed the hung ones."""
    cluster.hung.clear()
    cluster.waiting_for.clear()
    cluster.waited_rounds.clear()
    cluster.waited_epochs.clear()


@pytest.mark.parametrize("stopped", [False, True])
def test_only_wait_chain_end_is_hung(stopped):
    """Remove only rank 2 when rank 3 waits on it and ranks 0 and 1 wait on both.

    Rank 2's process stops (SIGSTOP), or only its main thread hangs.
    """
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.run(0.5)
    (cluster.stopped if stopped else cluster.hung).add(2)
    cluster.hung.update({0, 1, 3})
    cluster.waiting_for.update({0: 0b1100, 1: 0b1100, 3: 1 << 2})
    cluster.waited_rounds[3] = ((2, 7),)
    hung_s = cluster.now
    record = _first_change(cluster, 2.0)
    assert record.state == MembershipState.PROPOSE
    assert record.active_mask == _without(2)
    assert 1.0 < cluster.now - hung_s <= 1.2
    _release(cluster)
    assert cluster.run(1.0).active_mask == _without(2)
    assert states(cluster) == TRANSACTION


@pytest.mark.parametrize("stopped_in", ["all-to-all", "host exchange", "paused wait"])
def test_stopped_process_is_hung_anywhere(stopped_in):
    """Remove a stopped rank one hang timeout after it stops, and keep the peer its last report names.

    Rank 2 stops while it waits on a peer, which then waits on rank 2.
    """
    cluster = Cluster(hang_timeout_s=1.0)
    peer = 0 if stopped_in == "paused wait" else 3
    cluster.waiting_for[2] = 1 << peer
    if stopped_in == "all-to-all":
        cluster.waited_rounds[2] = ((peer, 7),)
    elif stopped_in == "host exchange":
        cluster.waited_epochs[2] = ((peer, 40),)
    else:
        cluster.idle.add(2)
    cluster.run(0.5)
    cluster.stopped.add(2)
    cluster.hung.update({0, 1, 3})  # the others end up waiting on rank 2
    cluster.waiting_for.update({r: 1 << 2 for r in (0, 1, 3)})
    stopped_s = cluster.now
    record = _first_change(cluster, 2.0)
    assert record.state == MembershipState.PROPOSE
    assert record.active_mask == _without(2)
    assert 1.0 < cluster.now - stopped_s <= 1.2
    _release(cluster)
    assert cluster.run(1.0).state == MembershipState.COMMITTED
    assert MembershipState.ABORTED not in states(cluster)


def _idle_deployment():
    """Return a cluster whose rank 0 waits for requests while the others wait on rank 0."""
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.idle.update(range(WORLD))
    cluster.waiting_for.update({r: 1 for r in range(1, WORLD)})
    cluster.run(5.0)
    return cluster


def test_late_rank_after_idle_not_hung():
    cluster = _idle_deployment()
    cluster.idle.clear()  # a request arrives, and rank 2 is late to the next collective
    cluster.hung.update(range(WORLD))
    cluster.waiting_for.update({r: 1 << 2 for r in (0, 1, 3)})
    cluster.run(0.3)
    _release(cluster)
    cluster.run(2.0)
    assert cluster.published == []


@pytest.mark.parametrize("how", ["hung", "stopped"])
@pytest.mark.parametrize("phase", [MembershipState.PROPOSE, MembershipState.INSTALL])
def test_hang_during_transaction_restarts_it(phase, how):
    """Fail a rank that hangs in a phase like a crashed one, after the hang timeout."""
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.alive.discard(1)  # a crash starts the transaction
    while cluster.authority.record.state != phase:
        cluster.step()
    getattr(cluster, how).add(3)
    phase_s = cluster.now
    record = _first_change(cluster, 2.0)
    assert record.state == MembershipState.PROPOSE
    assert record.active_mask == 0b0101
    assert 1.0 < cluster.now - phase_s <= 1.2
    record = cluster.run(1.0)
    assert record.state == MembershipState.COMMITTED
    assert record.active_mask == 0b0101
    assert MembershipState.ABORTED not in states(cluster)


def test_failure_during_transaction_restarts_it():
    cluster = Cluster()
    cluster.alive.discard(3)
    cluster.hung.add(2)  # 2 cannot reply, so the first transaction waits ...
    cluster.run(0.3)
    assert cluster.authority.record.state == MembershipState.PROPOSE
    cluster.alive.discard(2)  # ... and then 2 dies too
    record = cluster.run(2.0)
    assert record.state == MembershipState.COMMITTED
    assert record.active_mask == 0b0011
    assert record.generation == 2


@pytest.mark.parametrize("cause", ["placements", "install", "rank 0"])
def test_transaction_aborts(cause):
    """Abort when the survivors restored different placements, a survivor fails the install, or rank 0 is lost."""
    cluster = Cluster()
    if cause == "placements":
        cluster.digest[1] = "placement-b"
    elif cause == "install":
        cluster.install_reply[1] = Reply.FAILED
    cluster.alive.discard(0 if cause == "rank 0" else 3)
    assert cluster.run(1.0).state == MembershipState.ABORTED
    assert MembershipState.COMMITTED not in states(cluster)
    assert cluster.authority.last_transaction.outcome == "ABORTED"


def test_replacement_with_other_placement_excluded():
    """Commit without a replacement whose placement differs from the survivors' placement."""
    cluster = Cluster()
    cluster.alive.discard(2)
    cluster.run(1.0)
    cluster.published.clear()
    cluster.digest[2] = "placement-b"
    _join(cluster, 2)
    record = cluster.run(1.0)
    assert record.state == MembershipState.COMMITTED
    assert record.active_mask == _without(2)
    assert record.joining_mask == 0
    assert MembershipState.ABORTED not in states(cluster)
    assert states(cluster) == [MembershipState.PROPOSE, MembershipState.INSTALL] + TRANSACTION
    # The node supervisor reads why the change left the replacement out.
    assert "different expert placement" in cluster.published[2].reason


def test_survivor_failure_during_admission_commits():
    """Commit when rank 1 fails while rank 2's replacement is admitted after ranks 2 and 3 failed.

    The authority leaves the spare slot count to the active ranks' restore plan.
    """
    cluster = Cluster()
    cluster.alive -= {2, 3}
    assert cluster.run(1.0).state == MembershipState.COMMITTED
    _join(cluster, 2)  # rank 3's replacement did not ask to join, so rank 2's does not wait for it
    while cluster.authority.record.state != MembershipState.PROPOSE:
        cluster.step()
    cluster.alive.discard(1)
    record = cluster.run(1.0)
    assert record.state == MembershipState.COMMITTED
    assert record.active_mask == 0b0101
    assert MembershipState.ABORTED not in states(cluster)


def test_commit_needs_every_install():
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.alive.discard(3)
    cluster.run(1.0)  # in generation 1, every survivor's last reply is INSTALLED
    cluster.alive.discard(2)
    while cluster.authority.record.state != MembershipState.INSTALL:
        cluster.step()
    cluster.hung.add(1)  # replied READY to generation 2, never INSTALLED
    record = cluster.run(3.0)
    assert record.state == MembershipState.COMMITTED
    assert record.active_mask == 0b0001
    committed = [r.generation for r in cluster.published if r.state == MembershipState.COMMITTED]
    assert committed == [1, 3]  # never generation 2, which rank 1 did not install


def test_hang_clock_restarts_at_commit():
    """Give a rank waited on during the install a full hang timeout after the commit."""
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.alive.discard(3)
    while cluster.authority.record.state != MembershipState.INSTALL:
        cluster.step()
    cluster.hung.add(1)
    cluster.waiting_for[2] = 1 << 1
    assert cluster.run(0.8).state == MembershipState.INSTALL
    # It replies without progress (main's hang detector stays paused through the change), and the change commits.
    cluster.hung.discard(1)
    cluster.idle.add(1)
    assert _first_change(cluster, 0.2).state == MembershipState.COMMITTED
    cluster.idle.discard(1)
    cluster.hung.add(1)  # its first step after the commit
    cluster.run(0.5)
    _release(cluster)
    record = cluster.run(1.0)
    assert record.state == MembershipState.COMMITTED
    assert record.active_mask == 0b0111
    assert states(cluster) == TRANSACTION


def test_progressing_install_is_not_hung():
    """Count each completed install step as progress, so only a stalled install is a hang."""
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.alive.discard(1)
    while cluster.authority.record.state != MembershipState.INSTALL:
        cluster.step()
    cluster.hung.add(3)  # replies nothing and its iteration does not advance ...
    cluster.installing.add(3)  # ... but it completes install steps
    cluster.run(3.0)
    assert cluster.authority.record.state == MembershipState.INSTALL
    cluster.installing.discard(3)
    stalled_s = cluster.now
    record = _first_change(cluster, 2.0)
    assert record.state == MembershipState.PROPOSE
    assert record.active_mask == 0b0101
    assert 1.0 < cluster.now - stalled_s <= 1.2


def test_replacement_joins_dead_joiner_dropped():
    cluster = Cluster()
    cluster.alive.discard(2)
    cluster.run(1.0)
    cluster.alive.add(2)  # a new process for rank 2, with fresh reports
    cluster.reports[2] = RankReport()
    cluster.authority.request_join(2)
    record = cluster.run(1.0)
    assert record.state == MembershipState.COMMITTED
    assert record.active_mask == ALL
    assert record.generation == 2

    cluster.alive.discard(3)
    cluster.run(1.0)
    cluster.reports[3] = RankReport()
    cluster.authority.request_join(3)  # the replacement never connects
    record = cluster.run(1.0)
    assert record.state == MembershipState.COMMITTED
    assert not record.is_active(3)


def _join(cluster, rank):
    cluster.alive.add(rank)  # a new process for the rank, with fresh reports
    cluster.reports[rank] = RankReport()
    cluster.authority.request_join(rank)


def test_replacements_added_in_one_transaction():
    """Add the replacements of a node's ranks together, although they become ready a little apart."""
    cluster = Cluster()
    cluster.alive -= {2, 3}
    cluster.run(1.0)
    cluster.published.clear()
    _join(cluster, 2)
    _join(cluster, 3)
    cluster.unprepared.add(3)
    cluster.run(2.0)
    assert cluster.published == []  # waits for rank 3's replacement
    cluster.unprepared.discard(3)
    record = cluster.run(1.0)
    assert states(cluster) == TRANSACTION
    assert record.active_mask == ALL
    assert cluster.authority.last_transaction.added_ranks == (2, 3)
    # The transaction is detected at rank 2's readiness and includes the wait for rank 3's.
    summary = cluster.authority.last_transaction
    assert summary.began_unix_ms - summary.detected_unix_ms >= 2000


@pytest.mark.parametrize("failure", ["failed_reply", "hang"])
@pytest.mark.parametrize("phase", [MembershipState.PROPOSE, MembershipState.INSTALL])
def test_failed_replacement_left_out(phase, failure):
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.alive.discard(2)
    cluster.run(1.0)
    cluster.alive.add(2)
    cluster.reports[2] = RankReport()
    cluster.authority.request_join(2)
    while cluster.authority.record.state != phase:
        cluster.step()
    if failure == "failed_reply" and phase == MembershipState.INSTALL:
        cluster.install_reply[2] = Reply.FAILED
    else:
        cluster.hung.add(2)  # never replies again
        if failure == "failed_reply":
            cluster.reports[2] = cluster.reports[2].model_copy(
                update=dict(
                    reply=Reply.FAILED, reply_generation=cluster.authority.record.generation
                )
            )
    record = cluster.run(3.0)
    assert record.state == MembershipState.COMMITTED
    assert record.active_mask == _without(2)
    assert record.joining_mask == 0
    assert MembershipState.ABORTED not in states(cluster)


@pytest.mark.parametrize("phase", [MembershipState.PROPOSE, MembershipState.INSTALL])
@pytest.mark.parametrize("victim", [1, 2, 3])
def test_second_failure_in_each_phase_commits(phase, victim):
    cluster = Cluster()
    first = 1 if victim != 1 else 2
    cluster.alive.discard(first)
    for _ in range(100):
        cluster.step()
        if cluster.authority.record.state == phase:
            break
    assert cluster.authority.record.state == phase
    cluster.alive.discard(victim)
    record = cluster.run(3.0)
    assert record.state == MembershipState.COMMITTED
    assert not record.is_active(first)
    assert not record.is_active(victim)


def test_rank_0_shutdown_report_ends_deployment():
    """End the deployment normally on rank 0's shutdown report only, so the ranks that leave are not failures."""
    cluster = Cluster()
    cluster.shutdown.add(1)
    cluster.run(1.0)
    assert cluster.published == []
    cluster.shutdown.add(0)
    cluster.step()
    assert cluster.authority.record.state == MembershipState.SHUTDOWN
    cluster.alive.clear()
    cluster.run(2.0)
    assert [record.state for record in cluster.published] == [MembershipState.SHUTDOWN]


# The startup stays STARTING until every rank has started. Any loss before then aborts at once.


def _starting_reports(started=(), sequence=1):
    return {r: RankReport(started=r in started, sequence=sequence) for r in range(WORLD)}


@pytest.mark.parametrize("failed_rank", [0, 2])
def test_startup_rank_failure_aborts_at_once(failed_rank):
    """Abort without waiting for the rendezvous timeout, and give the reason in the record."""
    authority = Authority(WORLD)
    authority.tick(0.0, _starting_reports(), ALL)
    record = authority.tick(0.05, _starting_reports(started={1}), _without(failed_rank))
    assert record.state == MembershipState.ABORTED
    assert record.reason == f"ranks [{failed_rank}] failed during the startup"


def test_stopped_reports_abort_the_startup():
    """Abort when peers wait in a startup collective on a stopped rank for the rendezvous timeout."""
    authority = Authority(WORLD, rendezvous_timeout_s=60.0)
    for second in range(0, 91):
        reports = {
            r: RankReport(sequence=second, progress=3, waiting_for=1 << 1) for r in range(WORLD)
        }
        reports[1] = RankReport(sequence=min(second, 10), progress=3)  # stopped at 10 s
        record = authority.tick(float(second), reports, ALL)
        if second <= 60:
            assert record is None
    assert authority.record.state == MembershipState.ABORTED
    assert authority.record.reason == "ranks [1] failed during the startup"


@pytest.mark.parametrize("peer", ["hangs", "progresses"])
def test_startup_collective_hang_aborts(peer):
    """Abort when a rank others wait on completes no startup collective for the rendezvous timeout.

    A rank whose startup collectives go on is not hung, however long the startup takes.
    """
    authority = Authority(WORLD, rendezvous_timeout_s=600.0)
    for second in range(0, 1300, 10):
        reports = {
            r: RankReport(sequence=second, progress=3, waiting_for=1 << 2) for r in range(WORLD)
        }
        reports[2] = RankReport(
            sequence=second, progress=3 + (second if peer == "progresses" else 0)
        )
        record = authority.tick(float(second), reports, ALL)
        if second <= 600 or peer == "progresses":
            assert record is None
    if peer == "hangs":
        assert authority.record.state == MembershipState.ABORTED
        assert authority.record.reason == "ranks [2] failed during the startup"


class DictStore:
    """In-memory store with the TCPStore calls that the authority and its ranks use."""

    def __init__(self):
        self.values = {}
        self._lock = threading.Lock()

    def set(self, key, value):
        with self._lock:
            self.values[key] = bytes(value)

    def get(self, key):
        return self.values[key]

    def check(self, keys):
        return all(key in self.values for key in keys)

    def multi_get(self, keys):
        return [self.values[key] for key in keys]

    def compare_set(self, key, expected, desired):
        """Set ``desired`` if the value is ``expected`` and return the value after, as TCPStore does.

        An unset value matches an empty ``expected``.
        """
        with self._lock:
            current = self.values.get(key)
            if current == expected.encode() or (current is None and not expected):
                self.values[key] = desired.encode()
            return self.values.get(key, expected.encode())


@contextlib.contextmanager
def _deployment(world, **authority_options):
    """Run an authority process on a thread with one client per rank, and yield the store and the clients.

    On exit, rank 0 reports ``shutdown`` and the authority returns the SHUTDOWN record, as in a normal exit.
    """
    store = RendezvousStore(DictStore(), b"shared-secret")
    process = AuthorityProcess(world, store, Authority(world, **authority_options))
    returned = []
    thread = threading.Thread(
        target=lambda: returned.append(process.run(host="127.0.0.1")), daemon=True
    )
    thread.start()
    clients = []
    try:
        _until(lambda: store.store.check([membership.LIVENESS_KEY]))
        clients += [_member(store, r, world) for r in range(world)]
        _until(
            lambda: all(
                client.record is not None and client.record.state == MembershipState.COMMITTED
                for client in clients
            )
        )
        yield store, clients
        clients[0].report(shutdown=True)
        _until(
            lambda: (
                MembershipRecord.model_validate_json(store.get(RECORD_KEY)).state
                == MembershipState.SHUTDOWN
            )
        )
        # The store lives in the authority's process, which serves until every active rank has read the record.
        thread.join(0.2)
        assert thread.is_alive()
    finally:
        _leave(clients)
        thread.join(10)
    assert [record.state for record in returned] == [MembershipState.SHUTDOWN]


def _until(condition, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def _member(store, rank, world, replacement=False, on_error=None):
    """Start a rank's client that replies to every record as an executor does."""
    client = None

    def on_record(record):
        if record.state == MembershipState.PROPOSE:
            client.report(
                reply=Reply.READY, reply_generation=record.generation, ready_payload=f"rank {rank}"
            )
        elif record.state == MembershipState.INSTALL:
            client.report(reply=Reply.INSTALLED, reply_generation=record.generation)

    client = MembershipClient(rank, store, on_record, replacement=replacement, on_error=on_error)
    client.start()
    if replacement:
        client.report(joining_ready=True, ready_payload=f"rank {rank}")
    else:
        client.report(started=True)
    return client


def _refused(store, rank, replacement):
    """Return whether the authority closes a liveness connection for ``rank``."""
    with open_liveness_connection(store, rank, replacement) as connection:
        connection.sock.settimeout(10)
        return connection.sock.recv(1 << 16) == b""


def _leave(clients):
    for client in clients:
        if client._thread.is_alive():
            client.stop()


def test_ended_process_is_removed_at_once():
    with _deployment(3) as (_, clients):
        clients[2]._sock.shutdown(socket.SHUT_RDWR)  # as when its process ends
        ended_s = time.monotonic()
        _until(
            lambda: all(
                client.record.state == MembershipState.COMMITTED
                and client.record.active_mask == 0b011
                for client in clients[:2]
            )
        )
        assert time.monotonic() - ended_s < 1.0


def test_shutdown_record_precedes_close():
    """Check that a rank still connected when the authority ends after rank 0 has read the SHUTDOWN record."""
    store = RendezvousStore(DictStore(), b"shared-secret")
    parent_read, parent_write = os.pipe()
    process = AuthorityProcess(2, store, Authority(2, hang_timeout_s=2.0), parent=parent_read)
    thread = threading.Thread(target=process.run, kwargs={"host": "127.0.0.1"}, daemon=True)
    thread.start()
    states_at_close = []
    clients = []
    try:
        _until(lambda: store.store.check([membership.LIVENESS_KEY]))
        clients.append(_member(store, 0, 2))
        clients.append(
            _member(
                store, 1, 2, on_error=lambda error: states_at_close.append(clients[1].record.state)
            )
        )
        _until(
            lambda: clients[0].record is not None
            and clients[0].record.state == MembershipState.COMMITTED
        )
        clients[0].report(shutdown=True)
        _until(lambda: clients[0].record.state == MembershipState.SHUTDOWN)
        clients[0].stop()
        os.close(parent_write)  # as when rank 0's process ends
        thread.join(10)
        assert not thread.is_alive()
        _until(lambda: states_at_close)
    finally:
        _leave(clients)
        os.close(parent_read)
    assert states_at_close == [MembershipState.SHUTDOWN]
def test_liveness_connections_keep_alive(monkeypatch):
    """Enable TCP keepalive at both ends of the connections of a rank and of a node supervisor."""
    keep_alive, opened = membership._keep_alive, []
    monkeypatch.setattr(
        membership, "_keep_alive", lambda sock: keep_alive(sock) or opened.append(sock)
    )
    with _deployment(1) as (store, clients):
        _NodeSupervisor(store, 0, [0]).until(lambda view: view.nodes)
        assert len(opened) == 4
        assert clients[0]._sock in opened
        for sock in opened:
            assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
            idle_s = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE)
            interval_s = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL)
            probes = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT)
            assert idle_s + interval_s * probes == membership.DEAD_TIMEOUT_S
            user_timeout_ms = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT)
            assert user_timeout_ms == membership.DEAD_TIMEOUT_S * 1000


def test_deployment_at_rest_is_quiet():
    """Send no record and write nothing to the store while the membership does not change."""
    with _deployment(2) as (store, clients):
        clients[1].stop()
        _until(lambda: clients[0].record.active_mask == 0b01)
        _until(lambda: clients[0].record.state == MembershipState.COMMITTED)
        writes = Mock(wraps=store.store.set)
        store.store.set = writes
        # No transaction starts for a replacement that has not reported that it is ready to join.
        with open_liveness_connection(store, 1, replacement=True) as connection:
            connection.sock.settimeout(1.5)
            received = b""
            with contextlib.suppress(socket.timeout):
                while True:
                    data = connection.sock.recv(1 << 16)
                    assert data, "the authority closed the connection"
                    received += data
            # Only the record that gives it its incarnation arrives.
            assert len(connection.lines.decode(received)) == 1
            writes.assert_not_called()  # until the connection closes


def test_unsigned_store_values_refused():
    store = RendezvousStore(DictStore(), b"shared-secret")
    RendezvousStore(store.store, b"another-secret").set(RECORD_KEY, b'{"state": 3}')
    assert store.get_if_set(RECORD_KEY) is None
    with pytest.raises(PermissionError):
        store.get(RECORD_KEY)
    # A signed ("row/1", "2x") does not verify as ("row/12", "x").
    store.set("row/1", b"2x")
    raw = store.store.get("row/1")
    store.store.set("row/12", raw[:32] + raw[33:])
    with pytest.raises(PermissionError):
        store.get("row/12")


class _NodeSupervisor:
    """Node supervisor connection that reads the views the authority sends."""

    def __init__(self, store, node, ranks):
        self.connection = open_node_supervisor_connection(store, node, ranks, f"host-{node}")
        self.connection.sock.settimeout(10)

    def until(self, predicate) -> NodeView:
        """Return the first view that satisfies ``predicate``."""
        while True:
            data = self.connection.sock.recv(1 << 16)
            assert data, "the authority closed the connection"
            views = [
                NodeView.model_validate_json(line) for line in self.connection.lines.decode(data)
            ]
            if views and predicate(views[-1]):
                return views[-1]


def test_node_supervisor_loss_fails_its_ranks():
    """Remove a lost supervisor's ranks at once and tell the other supervisors that its node left.

    The supervisor's loss stands for the loss of its node, even while the processes of its ranks still run.
    """
    with _deployment(4) as (store, clients):
        # Node 0's supervisor stays connected until the shutdown, because its loss would abort the deployment.
        node_0 = _NodeSupervisor(store, 0, [0, 1])
        node_1 = _NodeSupervisor(store, 1, [2, 3])
        view = node_0.until(lambda view: len(view.nodes) == 2)
        assert view.nodes == ((0, "host-0", True), (1, "host-1", True))
        node_1.connection.sock.close()
        _until(
            lambda: (
                clients[0].record.state == MembershipState.COMMITTED
                and clients[0].record.active_mask == 0b0011
            )
        )
        view = node_0.until(lambda view: view.record.active_mask == 0b0011)
        assert view.nodes == ((0, "host-0", True), (1, "host-1", False))


def test_replacement_joins_after_process_gone():
    """Refuse a replacement for rank 0 or for a rank whose process still serves.

    The replacement gets the rank's next incarnation, and the process it replaced is no longer admitted.
    """
    with _deployment(3) as (store, clients):
        assert _refused(store, 2, True)
        assert _refused(store, 0, True)
        assert clients[0].record.active_mask == 0b111
        assert clients[2]._thread.is_alive()
        clients[2]._sock.shutdown(socket.SHUT_RDWR)
        _until(lambda: clients[0].record.active_mask == 0b011)
        clients.append(_member(store, 2, 3, replacement=True))
        assert clients[-1].incarnation == 1
        _until(
            lambda: (
                clients[0].record.active_mask == 0b111
                and clients[0].record.state == MembershipState.COMMITTED
            )
        )
        assert clients[0].record.incarnations == (0, 0, 1)
        assert _refused(store, 2, False)


def test_authority_publishes_closed_incarnations():
    """Publish per rank the newest incarnation whose connection closed after every older one."""
    with _deployment(2) as (store, clients):
        assert store.get_if_set(membership.DISCONNECTED_KEY) is None
        clients[1].stop()
        _until(lambda: clients[0].record.active_mask == 0b01)
        _until(lambda: clients[0].record.state == MembershipState.COMMITTED)
        assert json.loads(store.get(membership.DISCONNECTED_KEY)) == [-1, 0]
        held = open_liveness_connection(store, 1, replacement=True)  # incarnation 1, held open
        clients[1] = _member(store, 1, 2, replacement=True)  # incarnation 2
        assert clients[1].incarnation == 2
        _until(lambda: clients[0].record.active_mask == 0b11)
        clients[1].stop()
        _until(lambda: clients[0].record.active_mask == 0b01)
        assert json.loads(store.get(membership.DISCONNECTED_KEY)) == [-1, 0]
        held.sock.close()
        _until(lambda: json.loads(store.get(membership.DISCONNECTED_KEY)) == [-1, 2])


def test_transaction_summary_spans_propose_to_commit():
    cluster = Cluster()
    cluster.run(1.0)
    assert cluster.authority.last_transaction is None
    cluster.alive.discard(2)
    for _ in range(100):
        if cluster.step() is not None and cluster.authority.record.state == MembershipState.INSTALL:
            break
    assert cluster.authority.record.state == MembershipState.INSTALL
    cluster.alive.discard(3)  # restarts the transaction, which keeps its start
    record = cluster.run(1.0)
    assert record.state == MembershipState.COMMITTED
    began = cluster.published_at[0]
    ended = cluster.published_at[-1]
    summary = cluster.authority.last_transaction
    assert (summary.generation, summary.outcome) == (record.generation, "COMMITTED")
    assert (summary.removed_ranks, summary.added_ranks) == ((2, 3), ())
    assert summary.ended_unix_ms - summary.began_unix_ms == pytest.approx((ended - began) * 1e3)
    assert ended - began > 0
    # A closed connection is proposed in the tick that sees it, and the simulated ranks install in one step.
    assert summary.detected_unix_ms == summary.began_unix_ms
    assert summary.ended_unix_ms - summary.ready_unix_ms == pytest.approx(50.0)


def _interleave(cluster, rng):
    """Apply one random crash, hang, join, join readiness or install outcome, and delay random reports."""
    record, r = cluster.authority.record, rng.randrange(1, cluster.authority.world_size)
    event = rng.randrange(5)
    if event == 0:
        cluster.alive.discard(r)
    elif event == 1 and r not in cluster.alive:
        _join(cluster, r)
        cluster.unprepared.add(r)
        cluster.hung.discard(r)
        cluster.digest[r], cluster.install_reply[r] = "placement-a", Reply.INSTALLED
    elif event == 2:
        cluster.unprepared.discard(r)
    elif event == 3 and rng.random() < 0.1:
        cluster.hung.add(r)
    elif event == 4 and (record.is_joining(r) or rng.random() < 0.01):
        if rng.random() < 0.5:
            cluster.digest[r] = "placement-b"
        else:
            cluster.install_reply[r] = Reply.FAILED
    cluster.stopped = {q for q in cluster.alive if rng.random() < 0.2}


@pytest.mark.parametrize("seed", range(10))
def test_random_interleavings_keep_invariants(seed):
    """Check the protocol's invariants after every tick of random failures, joins and late replies."""
    rng, world = random.Random(seed), 4 + seed % 5
    cluster = Cluster(world, hang_timeout_s=1.0)
    expected = {MembershipState.PROPOSE: Reply.READY, MembershipState.INSTALL: Reply.INSTALLED}
    record = cluster.authority.record
    committed, joining = record.active_mask, 0
    for step in range(600):
        if step < 400:
            _interleave(cluster, rng)
        else:
            cluster.stopped, cluster.unprepared = set(), set()
        previous, new = record, cluster.step()
        record = cluster.authority.record
        assert record.generation >= previous.generation
        if record.state == MembershipState.PROPOSE and new is not None:
            assert record.generation > previous.generation
        if record.state == MembershipState.ABORTED:
            break
        assert record.active_mask & ~sum(1 << r for r in cluster.alive) == 0
        active = [r for r in range(world) if record.is_active(r)]
        replied = [
            cluster.reports[r].reply_generation == record.generation
            and cluster.reports[r].reply == expected.get(previous.state)
            for r in active
        ]
        if record.state in expected:
            joining = record.joining_mask
            assert record.active_mask & ~committed & ~joining == 0
            assert new is not None or not all(replied)
        elif new is not None:
            assert previous.state == MembershipState.INSTALL
            assert all(replied)
            assert record.active_mask & ~committed & ~joining == 0
            committed = record.active_mask
    assert record.state in (MembershipState.COMMITTED, MembershipState.ABORTED)

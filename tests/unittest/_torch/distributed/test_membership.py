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
import itertools
import json
import os
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
    RankReport,
    RendezvousStore,
    Reply,
    ResumeState,
    TransactionSummary,
    open_liveness_connection,
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
        self.report_every = {}  # rank -> steps between its reports (a client thread starved of the GIL)
        self.waiting_for = {}  # rank -> mask it waits for
        self.waited_rounds = {}  # rank -> ((peer, round flag its kernels wait for), ...)
        self.waited_epochs = {}  # rank -> ((peer, epoch of the host-exchange round it waits in), ...)
        self.not_replying = set()  # making progress, but not at the point where it replies yet
        self.installing = set()  # completing a step of an install at every report
        self.unprepared = set()  # replacements that have not reported they are ready to join
        self.steps = 0
        self.install_reply = {r: Reply.INSTALLED for r in range(world)}
        self.digest = {r: "placement-a" for r in range(world)}  # of the placement a rank restored
        self.reports = {r: RankReport() for r in range(world)}
        self.paused_at = {r: ResumeState(iteration=10 + r, epoch=100 + r) for r in range(world)}
        self.published = []
        self.published_at = []

    def step(self, seconds=0.05):
        self.now += seconds
        self.steps += 1
        record = self.authority.record
        for r in self.alive:
            if r in self.stopped or self.steps % self.report_every.get(r, 1):
                continue
            old = self.reports[r]
            reply, generation = old.reply, old.reply_generation
            if r not in self.hung | self.not_replying and record.is_active(r):
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


def test_closed_connection_removes_the_rank():
    cluster = Cluster()
    cluster.alive.discard(2)
    record = cluster.run(1.0)
    assert states(cluster) == TRANSACTION
    assert record.generation == 1
    assert record.active_mask == _without(2)
    # Resume from the furthest survivor, with the next epoch.
    assert record.resume.iteration == 13
    assert record.resume.epoch == 104


def test_next_epoch_follows_across_epoch_wrap():
    cluster = Cluster()
    for r, epoch in {0: 0xFFFFFFFE, 1: 1, 3: 1}.items():  # ranks 1 and 3 went on past the wrap
        cluster.paused_at[r] = ResumeState(epoch=epoch)
    cluster.alive.discard(2)
    assert cluster.run(1.0).resume.epoch == 2


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


def test_idle_rank_waited_on_not_hung():
    cluster = Cluster(hang_timeout_s=1.0)
    # An idle server. Rank 0 waits for requests and the others wait for rank 0.
    cluster.idle.update(range(WORLD))
    cluster.waiting_for.update({r: 1 for r in range(1, WORLD)})
    cluster.run(5.0)
    assert cluster.published == []


def test_rank_hung_while_idle():
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.idle.update(range(WORLD))
    cluster.waiting_for.update({r: 1 for r in range(1, WORLD)})
    cluster.run(0.5)
    cluster.stopped.add(0)  # rank 0's process stops in its idle wait and stops reporting
    record = cluster.run(3.0)
    assert record.state == MembershipState.ABORTED  # rank 0 failed


def test_hung_rank_log_lists_waits(monkeypatch):
    warnings = []
    monkeypatch.setattr(membership.logger, "warning", warnings.append)
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.hung.add(3)
    cluster.waiting_for.update({r: 1 << 3 for r in range(3)})
    cluster.waited_rounds.update({r: ((3, 7),) for r in range(3)})
    cluster.run(3.0)
    [hung] = [w for w in warnings if "is hung" in w]
    assert hung.startswith("Rank 3 is hung")
    assert "rank 0: iteration" in hung
    assert "waits for [3], all-to-all rounds [(3, 7)]" in hung
    assert "rank 3: iteration" in hung
    assert "waits for []" in hung


def test_idle_rank_behind_peers_is_hung():
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.run(0.5)
    cluster.idle.add(3)  # rank 3 claims to be idle, but the others moved on and wait for it
    cluster.waiting_for.update({r: 1 << 3 for r in range(3)})
    record = cluster.run(3.0)
    assert record.state == MembershipState.COMMITTED
    assert not record.is_active(3)


def test_hung_rank_report_keeps_its_peers():
    """Remove only rank 2 when it stops while waiting for rank 1, which then waits for rank 2."""
    authority = _formed(Authority(WORLD, hang_timeout_s=30.0))
    frozen = RankReport(progress=5, waiting_for=1 << 1, sequence=1)
    for tick in range(1, 1000):  # 0.05 s apart
        reports = {
            r: RankReport(progress=5 if r == 1 else tick, sequence=tick) for r in range(WORLD)
        }
        reports[1] = reports[1].model_copy(update=dict(waiting_for=1 << 2))
        reports[2] = frozen
        record = authority.tick(tick * 0.05, reports, ALL)
        if record is not None:
            break
    assert record.state == MembershipState.PROPOSE
    assert record.active_mask == _without(2)


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


@pytest.mark.parametrize("peers_paused", [False, True])
def test_main_thread_hang_in_paused_wait(peers_paused):
    """Remove rank 2 when its main thread hangs in the request broadcast while it reports idle.

    Only rank 0, which waits for client requests, may stay idle while others wait on it.
    """
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.run(0.5)
    cluster.idle.add(2)
    (cluster.idle if peers_paused else cluster.hung).update({0, 1, 3})
    cluster.waiting_for.update({r: 1 << 2 for r in (0, 1, 3)})
    record = _first_change(cluster, 2.0)
    assert record.state == MembershipState.PROPOSE
    assert record.active_mask == _without(2)


def test_gil_starved_rank_is_not_hung():
    """Keep rank 2 when it reports and advances only every 0.4 s while the others wait on it."""
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.report_every[2] = 8
    cluster.hung.update({0, 1, 3})
    cluster.waiting_for.update({r: 1 << 2 for r in (0, 1, 3)})
    cluster.run(5.0)
    assert cluster.published == []


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


def test_stopped_rank_in_idle_deployment():
    """Remove a stopped rank one hang timeout after the first request makes the others wait on it."""
    cluster = _idle_deployment()
    cluster.stopped.add(2)
    cluster.run(5.0)
    assert cluster.published == []
    cluster.idle.clear()
    cluster.hung.update({0, 1, 3})
    cluster.waiting_for.update({r: 1 << 2 for r in (0, 1, 3)})
    request_s = cluster.now
    record = _first_change(cluster, 2.0)
    assert record.state == MembershipState.PROPOSE
    assert record.active_mask == _without(2)
    assert 1.0 < cluster.now - request_s <= 1.2


def test_mutually_waiting_ranks_are_one_end():
    """Remove both ranks 2 and 3 when they wait on each other and the others wait on rank 2.

    No report singles one of them out, so serving goes on without both.
    """
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.run(0.5)
    cluster.hung.update(range(WORLD))
    cluster.waiting_for.update({0: 1 << 2, 1: 1 << 2, 2: 1 << 3, 3: 1 << 2})
    record = _first_change(cluster, 2.0)
    assert record.state == MembershipState.PROPOSE
    assert record.active_mask == 0b0011
    _release(cluster)
    assert cluster.run(1.0).state == MembershipState.COMMITTED
    assert MembershipState.ABORTED not in states(cluster)


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


@pytest.mark.parametrize("phase", [MembershipState.PROPOSE, MembershipState.INSTALL])
def test_idle_rank_0_hang_in_transaction(phase):
    """Abort when rank 0 reports idle but does not reply in a phase for the hang timeout.

    Only a committed membership exempts an idle rank 0.
    """
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.alive.discard(1)  # a crash starts the transaction
    while cluster.authority.record.state != phase:
        cluster.step()
    cluster.idle.add(0)
    cluster.not_replying.add(0)
    record = _first_change(cluster, 2.0)
    assert record is not None
    assert record.state == MembershipState.ABORTED


def test_wait_on_inactive_rank_exempts_none():
    """Remove rank 3 when it hangs while its reports still name rank 1, which is no longer active."""
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.waiting_for[3] = 1 << 1
    cluster.waited_rounds[3] = ((1, 7),)
    cluster.alive.discard(1)
    while cluster.authority.record.state != MembershipState.PROPOSE:
        cluster.step()
    cluster.hung.add(3)
    record = _first_change(cluster, 2.0)
    assert record is not None
    assert record.state == MembershipState.PROPOSE
    assert record.active_mask == 0b0101
    cluster.waiting_for.clear()
    cluster.waited_rounds.clear()
    record = cluster.run(1.0)
    assert record.state == MembershipState.COMMITTED
    assert record.active_mask == 0b0101


def test_phase_waits_while_rank_advances():
    """Wait for a READY reply past any timeout while the rank's progress advances."""
    cluster = Cluster()
    cluster.alive.discard(1)
    cluster.not_replying.add(3)
    cluster.run(70.0)
    assert cluster.authority.record.state == MembershipState.PROPOSE
    assert cluster.authority.record.is_active(3)
    cluster.not_replying.clear()
    record = cluster.run(1.0)
    assert record.state == MembershipState.COMMITTED
    assert record.active_mask == _without(1)


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


def test_different_restored_placements_abort():
    cluster = Cluster()
    cluster.digest[1] = "placement-b"
    cluster.alive.discard(3)
    assert cluster.run(1.0).state == MembershipState.ABORTED
    assert MembershipState.COMMITTED not in states(cluster)


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


@pytest.mark.parametrize("slots", ["spare", "none spare"])
def test_failed_install_aborts(slots):
    """Abort also when every active rank fails the install because an expert has no spare slot."""
    cluster = Cluster()
    if slots == "none spare":  # rank 3 alone held an expert
        cluster.install_reply = {r: Reply.FAILED for r in range(WORLD)}
    else:
        cluster.install_reply[1] = Reply.FAILED
    cluster.alive.discard(3)
    assert cluster.run(1.0).state == MembershipState.ABORTED
    assert states(cluster) == [
        MembershipState.PROPOSE,
        MembershipState.INSTALL,
        MembershipState.ABORTED,
    ]


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


def test_no_progress_in_transaction_not_hung():
    authority = _formed(Authority(WORLD, hang_timeout_s=1.0))
    calls = itertools.count(1)

    def reports(reply=Reply.NONE, generation=0, waiting_for=0):
        # Every rank stays at iteration 5 from the start, and the survivors' replies follow the record.
        sequence = next(calls)
        return {
            r: RankReport(
                sequence=sequence,
                progress=5,
                waiting_for=waiting_for if r == 0 else 0,
                reply=reply,
                reply_generation=generation,
                placement=((WORLD, tuple((q, -1) for q in range(WORLD))),),
            )
            for r in range(3)
        } | {3: RankReport(progress=5)}

    authority.tick(0.0, reports(waiting_for=1 << 3), ALL)
    assert authority.tick(1.5, reports(waiting_for=1 << 3), ALL).state == MembershipState.PROPOSE
    assert authority.tick(1.6, reports(Reply.READY, 1), ALL).state == MembershipState.INSTALL
    assert authority.tick(1.7, reports(Reply.INSTALLED, 1), ALL).state == MembershipState.COMMITTED
    # Right after the commit, rank 0 waits for rank 1, which has not moved past iteration 5 since time 0.
    assert authority.tick(1.8, reports(Reply.INSTALLED, 1, waiting_for=1 << 1), ALL) is None
    assert authority.record.is_active(1)


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


def test_losing_rank_0_aborts():
    cluster = Cluster()
    cluster.alive.discard(0)
    assert cluster.run(1.0).state == MembershipState.ABORTED


@pytest.mark.parametrize("silent_rank, survives", [(1, True), (0, False)])
def test_silent_rank_in_phase_counts_failed(silent_rank, survives):
    """Restart the transaction without a silent rank, or abort if it is rank 0."""
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.alive.discard(3)
    cluster.hung.add(silent_rank)  # connected but never replies, and only the authority waits on it
    record = cluster.run(3.0)
    if survives:
        assert record.state == MembershipState.COMMITTED
        assert record.active_mask == 0b0101
        assert MembershipState.ABORTED not in states(cluster)
    else:
        assert record.state == MembershipState.ABORTED


def test_transaction_fails_only_wait_chain_end():
    """Fail only rank 1 when rank 2 cannot reply INSTALLED because it waits for rank 1."""
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.alive.discard(3)
    while cluster.authority.record.state != MembershipState.INSTALL:
        cluster.step()
    cluster.hung.update({1, 2})
    cluster.waiting_for[2] = 1 << 1
    while cluster.authority.record.state == MembershipState.INSTALL:
        cluster.step()
    assert cluster.authority.record.state == MembershipState.PROPOSE
    assert cluster.authority.record.active_mask == 0b0101


def test_each_phase_has_own_hang_timeout():
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.alive.discard(3)
    cluster.hung.add(1)  # replies late in each phase
    while cluster.authority.record.state != MembershipState.PROPOSE:
        cluster.step()
    cluster.run(0.8)
    cluster.hung.discard(1)
    while cluster.authority.record.state != MembershipState.INSTALL:
        cluster.step()
    cluster.hung.add(1)
    cluster.run(0.8)  # 1.6 s into the transaction, 0.8 s into INSTALL
    assert cluster.authority.record.state == MembershipState.INSTALL
    cluster.hung.discard(1)
    assert cluster.run(0.5).state == MembershipState.COMMITTED


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


def test_admission_proposal_logs_joining_ranks(monkeypatch):
    """Log one line per admission and none for the failure."""
    infos = []
    monkeypatch.setattr(membership.logger, "info", infos.append)
    cluster = Cluster()
    cluster.alive -= {2, 3}
    cluster.run(1.0)
    _join(cluster, 3)
    _join(cluster, 2)
    cluster.unprepared.add(2)
    cluster.run(1.0)  # waits for rank 2's replacement
    cluster.unprepared.discard(2)
    cluster.run(1.0)
    [line] = [i for i in infos if "joining ranks" in i]
    assert line == "Adding joining ranks [2, 3] to the membership"


def test_transaction_phase_times_are_ordered():
    """Record the detection, proposal, READY and commit times in milliseconds of wall-clock time."""
    cluster = Cluster()
    cluster.alive -= {2, 3}
    cluster.run(1.0)
    cluster.published.clear()
    cluster.published_at.clear()
    _join(cluster, 2)
    _join(cluster, 3)
    cluster.unprepared.add(3)
    detected = cluster.now + 0.05  # the next tick sees the ready join
    cluster.run(2.0)
    cluster.unprepared.discard(3)
    before_s = time.time()
    record = cluster.run(1.0)
    after_s = time.time()
    assert states(cluster) == TRANSACTION
    assert record.active_mask == ALL
    summary = cluster.authority.last_transaction
    times = {
        "detected": summary.detected_unix_ms,
        "proposed": summary.began_unix_ms,
        "ready": summary.ready_unix_ms,
        "committed": summary.ended_unix_ms,
    }
    assert times["detected"] < times["proposed"] < times["ready"] < times["committed"]
    # The authority's clock at the detection and at each record it published, in milliseconds.
    ticks_ms = [t * 1e3 for t in (detected, *cluster.published_at)]
    phases_ms = [later - earlier for earlier, later in itertools.pairwise(times.values())]
    assert phases_ms == pytest.approx([b - a for a, b in itertools.pairwise(ticks_ms)], abs=1)
    # The committing tick ran between the two wall-clock reads.
    assert before_s * 1e3 - 1 <= times["committed"] <= after_s * 1e3 + 1


@pytest.mark.parametrize("how", ["stops", "ends"])
def test_ready_join_waits_while_peer_advances(how):
    """Wait for the replacement failed with the same rank only while its reports advance.

    Rank 2's replacement is added alone once rank 3's stops reporting or its process ends.
    """
    cluster = Cluster(hang_timeout_s=2.0)
    cluster.alive -= {2, 3}
    cluster.run(1.0)
    cluster.published.clear()
    _join(cluster, 2)
    _join(cluster, 3)
    cluster.unprepared.add(3)
    cluster.run(8.0)
    assert cluster.published == []
    if how == "stops":
        cluster.stopped.add(3)
        assert cluster.run(0.9).state == MembershipState.COMMITTED
        assert cluster.published == []
    else:
        cluster.alive.discard(3)
    record = cluster.run(0.5)
    assert states(cluster) == TRANSACTION
    assert record.active_mask == _without(3)


def test_ready_join_stops_on_stalled_startup():
    """Add rank 2's replacement alone once rank 3's startup makes no progress for the rendezvous timeout."""
    cluster = Cluster(rendezvous_timeout_s=5.0)
    cluster.alive -= {2, 3}
    cluster.run(1.0)
    cluster.published.clear()
    _join(cluster, 2)
    _join(cluster, 3)
    cluster.unprepared.add(3)
    cluster.run(8.0)  # rank 3's startup progresses, so rank 2's waits past the timeout
    assert cluster.published == []
    cluster.hung.add(3)  # its startup stalls while its client reports on
    assert cluster.run(4.5).state == MembershipState.COMMITTED
    assert cluster.published == []
    record = cluster.run(1.0)
    assert states(cluster) == TRANSACTION
    assert record.active_mask == _without(3)


def test_ended_join_leaves_no_detection_time():
    """Detect a later failure when it happens, not at a ready join whose replacement died before its proposal."""
    cluster = Cluster(world=8)
    cluster.alive -= {2, 3}
    cluster.run(1.0)
    _join(cluster, 2)
    _join(cluster, 3)
    cluster.unprepared.add(3)
    cluster.run(1.0)
    assert cluster.authority.record.generation == 1  # it waits for rank 3's replacement
    cluster.alive.discard(2)  # the replacement dies
    cluster.run(2.0)
    cluster.published.clear()
    cluster.alive.discard(5)
    record = cluster.run(1.0)
    assert states(cluster) == TRANSACTION
    assert not record.is_active(5)
    summary = cluster.authority.last_transaction
    assert summary.began_unix_ms == summary.detected_unix_ms


def test_ready_join_waits_for_failed_peers():
    """Wait for the replacement of a rank that failed together with this one, then add both at once."""
    cluster = Cluster()
    cluster.alive -= {2, 3}
    cluster.run(1.0)
    cluster.published.clear()
    _join(cluster, 2)
    _join(cluster, 3)
    cluster.unprepared.add(3)
    cluster.run(2.0)
    assert cluster.published == []
    cluster.unprepared.discard(3)
    record = cluster.run(1.0)
    assert states(cluster) == TRANSACTION
    assert record.active_mask == ALL
    assert cluster.authority.last_transaction.added_ranks == (2, 3)


def test_ready_join_skips_unrelated_failures():
    """Propose rank 2's replacement at once while rank 5's, from an earlier failure, is still starting."""
    cluster = Cluster(world=8)
    cluster.alive -= {5, 6}
    cluster.run(1.0)
    cluster.alive.discard(2)
    cluster.run(1.0)
    cluster.published.clear()
    _join(cluster, 5)
    cluster.unprepared.add(5)
    _join(cluster, 2)
    cluster.run(1.0)
    assert cluster.published[0].state == MembershipState.PROPOSE
    assert cluster.published[0].joining_mask == 1 << 2


def test_replacement_proposed_once_ready():
    """Keep serving without a replacement that asked to join and then went silent."""
    cluster = Cluster()
    cluster.alive.discard(2)
    cluster.run(1.0)
    cluster.published.clear()
    _join(cluster, 2)
    cluster.unprepared.add(2)
    cluster.run(5.0)
    assert cluster.published == []
    cluster.unprepared.discard(2)
    record = cluster.run(1.0)
    assert states(cluster) == TRANSACTION
    assert record.active_mask == ALL


def test_silent_joining_rank_left_out():
    """Commit without a joining rank that goes silent after it reported ready, after the hang timeout.

    Its readiness stands for its READY reply, so INSTALL follows the survivors' READY at once.
    """
    cluster = Cluster(hang_timeout_s=1.0)
    cluster.alive.discard(2)
    cluster.run(1.0)
    cluster.published.clear()
    cluster.published_at.clear()
    _join(cluster, 2)
    cluster.hung.add(2)  # ready, then replies neither READY nor INSTALLED
    record = cluster.run(3.0)
    assert record.state == MembershipState.COMMITTED
    assert record.active_mask == _without(2)
    assert states(cluster) == [MembershipState.PROPOSE, MembershipState.INSTALL] + TRANSACTION
    proposed, installed, restarted = cluster.published_at[:3]
    assert installed - proposed == pytest.approx(0.05)
    assert 1.0 < restarted - proposed <= 1.2


def test_admission_marks_replacement_joining():
    cluster = Cluster()
    cluster.alive.discard(2)
    cluster.run(1.0)
    assert all(record.joining_mask == 0 for record in cluster.published)
    cluster.published.clear()
    cluster.alive.add(2)
    cluster.reports[2] = RankReport()
    cluster.authority.request_join(2)
    cluster.run(1.0)
    assert states(cluster) == TRANSACTION
    assert [record.joining_mask for record in cluster.published] == [1 << 2, 1 << 2, 0]


def test_replacement_resumes_with_survivors():
    """Resume at the survivors' iteration and epoch, whatever the joining rank reports."""
    cluster = Cluster()
    cluster.alive.discard(2)
    cluster.run(1.0)
    cluster.paused_at[2] = ResumeState(iteration=50, epoch=150)
    _join(cluster, 2)
    record = cluster.run(1.0)
    assert record.active_mask == ALL
    assert record.generation == 2
    assert record.resume == ResumeState(iteration=13, epoch=104)


def test_replacement_takes_over_active_rank():
    cluster = Cluster()
    cluster.reports[1] = RankReport()  # the replacement's reports replace the old process's
    cluster.authority.request_join(1)
    record = cluster.run(1.0)
    assert states(cluster) == TRANSACTION
    assert record.active_mask == ALL
    assert record.generation == 1


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


def test_record_and_report_serialization_round_trip():
    record = MembershipRecord(
        generation=7,
        state=MembershipState.INSTALL,
        active_mask=(1 << 200) | 0b1011,
        resume=ResumeState(iteration=1, epoch=2),
        incarnations=(0, 2, 0, 1),
        joining_mask=0b0010,
        ready_payloads=((0, "[[4, [[0, -1]]]]"), (1, "[[3, 1]]")),
    )
    assert MembershipRecord.model_validate_json(record.model_dump_json()) == record
    aborted = record.model_copy(update=dict(state=MembershipState.ABORTED, reason="rank 0 failed"))
    assert MembershipRecord.model_validate_json(aborted.model_dump_json()) == aborted
    assert record.is_member(1, 2)
    assert not record.is_member(1, 0)
    assert not record.is_member(2, 0)
    report = RankReport(
        progress=9,
        idle=True,
        waiting_for=1 << 255,
        reply_generation=7,
        reply=Reply.INSTALLED,
        paused_at=ResumeState(iteration=1, epoch=2),
        placement_digest="digest",
        incarnation=2,
        shutdown=True,
    )
    ready = RankReport(reply=Reply.READY, ready_payload="[[4, [[0, -1], [3, 1]]]]")
    for sent in (report, ready):
        assert RankReport.model_validate_json(sent.model_dump_json()) == sent
    # A line that is not a valid report raises a ValueError, which closes the connection.
    for line in (b"not json", b'{"progress": "nine"}', b'{"reply": 9}'):
        with pytest.raises(ValueError):
            RankReport.model_validate_json(line)


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


def test_deployment_serves_once_all_started():
    authority = Authority(WORLD)
    assert authority.record.state == MembershipState.STARTING
    assert authority.tick(0.0, _starting_reports(), ALL) is None
    assert authority.tick(1.0, _starting_reports(started={0, 1, 2}, sequence=2), ALL) is None
    record = authority.tick(2.0, _starting_reports(started=range(WORLD), sequence=3), ALL)
    assert (record.generation, record.state, record.active_mask) == (
        0,
        MembershipState.COMMITTED,
        ALL,
    )
    assert authority.last_transaction is None  # the startup is not a membership transaction


@pytest.mark.parametrize("failed_rank", [0, 2])
def test_startup_rank_failure_aborts_at_once(failed_rank):
    """Abort without waiting for the rendezvous timeout, and give the reason in the record."""
    authority = Authority(WORLD)
    authority.tick(0.0, _starting_reports(), ALL)
    record = authority.tick(0.05, _starting_reports(started={1}), _without(failed_rank))
    assert record.state == MembershipState.ABORTED
    assert record.reason == f"ranks [{failed_rank}] failed during the startup"


def test_startup_waits_until_rendezvous_timeout():
    """Abort the startup only if a rank does not connect within the rendezvous timeout.

    A long model load does not count, because every rank connects at the start of its worker.
    """
    authority = Authority(WORLD, rendezvous_timeout_s=600.0)
    late = _without(3)
    for second in range(0, 600, 10):
        assert authority.tick(float(second), _starting_reports(sequence=second), late) is None
    # The last rank connects at 590 s and starts 3000 s later, and the deployment serves.
    for second in range(590, 3600, 10):
        assert authority.tick(float(second), _starting_reports(sequence=second), ALL) is None
    record = authority.tick(3600.0, _starting_reports(started=range(WORLD), sequence=3600), ALL)
    assert record.state == MembershipState.COMMITTED

    never = Authority(WORLD, rendezvous_timeout_s=600.0)
    never.tick(0.0, _starting_reports(), late)
    assert never.tick(599.0, _starting_reports(sequence=2), late) is None
    assert never.tick(601.0, _starting_reports(sequence=3), late).state == MembershipState.ABORTED


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


def test_no_join_before_deployment_started():
    process = AuthorityProcess(
        WORLD, RendezvousStore(DictStore(), b"shared-secret"), Authority(WORLD)
    )
    assert "has started" in process._refusal(2, 1)
    _formed(process._authority)
    assert process._refusal(2, 1) is None  # rank 2's process is not connected


def test_liveness_port_follows_rendezvous_port():
    """Bind the liveness listener to the rendezvous address at the port after the store's."""
    assert membership.authority_ports("10.0.0.7:29600") == ("10.0.0.7", 29600, 29601)
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    store = RendezvousStore(DictStore(), b"shared-secret")
    process = AuthorityProcess(1, store, Authority(1))
    thread = threading.Thread(target=lambda: process.run("127.0.0.1", port), daemon=True)
    thread.start()
    _until(lambda: store.store.check([membership.LIVENESS_KEY]))
    assert store.get(membership.LIVENESS_KEY) == f"127.0.0.1:{port}".encode()
    with open_liveness_connection(store, 0) as rank_0:
        rank_0.sock.sendall(membership._line(RankReport(shutdown=True)))
        rank_0.sock.settimeout(10)
        membership._read_line(rank_0.sock)  # the record, at the hello
        assert b'"state":4' in membership._read_line(rank_0.sock)  # SHUTDOWN
    thread.join(10)
    assert not thread.is_alive()


def test_authority_aborts_when_parent_ends():
    """Abort once rank 0's end of the ``parent`` pipe closes.

    The authority waits at most the hang timeout for an active rank to leave.
    """
    store = RendezvousStore(DictStore(), b"shared-secret")
    read_fd, write_fd = os.pipe()
    process = AuthorityProcess(1, store, Authority(1, hang_timeout_s=2.0), parent=read_fd)
    returned = []
    thread = threading.Thread(
        target=lambda: returned.append(process.run(host="127.0.0.1")), daemon=True
    )
    try:
        thread.start()
        _until(lambda: store.store.check([membership.LIVENESS_KEY]))
        with open_liveness_connection(store, 0):  # a rank that never leaves
            thread.join(0.3)
            assert thread.is_alive()
            os.close(write_fd)
            closed_s = time.monotonic()
            thread.join(10)
            assert 2.0 <= time.monotonic() - closed_s < 6.0
        assert [record.state for record in returned] == [MembershipState.ABORTED]
        assert (
            MembershipRecord.model_validate_json(store.get(RECORD_KEY)).state
            == MembershipState.ABORTED
        )
    finally:
        os.close(read_fd)


def test_bind_hands_newest_record():
    """Hand the executor the membership its process joined at the start of its worker."""
    client = MembershipClient(1, RendezvousStore(None, b"secret"), lambda record: None)
    client.record = MembershipRecord(generation=0, state=MembershipState.COMMITTED, active_mask=ALL)
    records = []
    client.bind(records.append)
    assert records == [client.record]
    assert client._on_record == records.append


def _socket_pair():
    with socket.create_server(("127.0.0.1", 0)) as listener:
        client = socket.create_connection(listener.getsockname())
        server, _ = listener.accept()
    return client, server


def test_lines_decoded_whole_and_bounded():
    lines = membership._Lines()
    assert lines.decode(b'{"progress"') == []
    assert lines.decode(b': 1}\n{"progress": 2}\n{') == [b'{"progress": 1}', b'{"progress": 2}']
    with pytest.raises(ValueError):
        lines.decode(b"x" * (membership._MAX_LINE_BYTES + 1))


def test_client_closes_ended_connection():
    """Close the connection and report the error when the authority is gone."""
    errors = []
    client = MembershipClient(
        1, RendezvousStore(None, b"secret"), lambda record: None, on_error=errors.append
    )
    client._sock, authority_side = _socket_pair()
    client._lines = membership._Lines()
    client.connected = True
    authority_side.close()
    client._run()
    assert client._sock.fileno() == -1
    assert len(errors) == 1
    assert isinstance(errors[0], ConnectionError)
    assert not client.connected


def test_client_follows_records_and_reports(monkeypatch):
    """Exchange records and reports as JSON lines over the authenticated connection."""
    monkeypatch.setattr(membership, "REPORT_INTERVAL_S", 60.0)
    records = []
    client = MembershipClient(1, RendezvousStore(None, b"secret"), records.append)
    client._sock, authority_side = _socket_pair()
    client._lines = membership._Lines()
    client._thread.start()
    try:
        current = MembershipRecord(generation=1, state=MembershipState.COMMITTED, active_mask=ALL)
        newer = current.model_copy(update=dict(generation=2))
        authority_side.sendall(b"".join(membership._line(record) for record in (current, newer)))
        _until(lambda: len(records) == 2)
        client.report(reply=Reply.READY, reply_generation=2)
        received = membership._Lines()
        reports = []
        authority_side.settimeout(10)
        while not any(report.reply == Reply.READY for report in reports):
            reports += [
                RankReport.model_validate_json(line)
                for line in received.decode(authority_side.recv(4096))
            ]
        assert records == [current, newer]
    finally:
        client.stop()
        authority_side.close()


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


def _member(store, rank, world, replacement=False):
    """Start a rank's client that replies to every record as an executor does."""
    client = None

    def on_record(record):
        if record.state == MembershipState.PROPOSE:
            client.report(
                reply=Reply.READY, reply_generation=record.generation, ready_payload=f"rank {rank}"
            )
        elif record.state == MembershipState.INSTALL:
            client.report(reply=Reply.INSTALLED, reply_generation=record.generation)

    client = MembershipClient(rank, store, on_record, replacement=replacement)
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


def test_stopped_client_releases_file_descriptors():
    """Keep the open file descriptors constant while clients of one rank start and stop again and again."""
    with _deployment(2) as (store, clients):
        clients[1].stop()
        open_fds = len(os.listdir("/proc/self/fd"))
        for incarnation in range(1, 21):
            client = _member(store, 1, 2, replacement=True)
            assert client.incarnation == incarnation
            _until(lambda: client.record is not None and client.record.is_member(1, incarnation))
            client.stop()
        # The count above may include the authority's end of clients[1]'s connection, which it closes later.
        _until(lambda: len(os.listdir("/proc/self/fd")) <= open_fds)
        clients[1] = _member(store, 1, 2, replacement=True)
        _until(lambda: clients[1].record is not None and clients[1].record.is_member(1, 21))


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


def _committed_without_rank_1(client):
    return client.record.state == MembershipState.COMMITTED and client.record.active_mask == 0b01


def test_only_valid_hello_opens_connection(monkeypatch):
    """Accept a replacement's hello for rank 1 only when it is signed for this connection.

    The authority closes an unsigned hello, a replayed one, a second connection of a connected process, and a
    connection that sends no hello.
    """
    monkeypatch.setattr(membership, "HELLO_TIMEOUT_S", 0.5)
    with _deployment(2) as (store, clients):
        clients[1]._sock.shutdown(socket.SHUT_RDWR)
        _until(lambda: _committed_without_rank_1(clients[0]))
        host, port = store.get(membership.LIVENESS_KEY).decode().rsplit(":", 1)
        hello = b"1 1"
        for tag in (b"", store.tag(b"hello", bytes(16) + hello).hex().encode() + b" "):
            with socket.create_connection((host, int(port)), timeout=10) as forged:
                membership._read_line(forged)
                forged.sendall(tag + hello + b"\n")
                assert forged.recv(1) == b""
        assert _refused(store, 0, False)
        with socket.create_connection((host, int(port)), timeout=10) as silent:
            membership._read_line(silent)
            assert silent.recv(1) == b""  # after HELLO_TIMEOUT_S
        assert clients[0].record.active_mask == 0b01
        # The same hello, signed for its own connection.
        clients[1] = _member(store, 1, 2, replacement=True)
        _until(lambda: clients[0].record.active_mask == 0b11)


def test_invalid_report_line_closes_connection():
    with _deployment(2) as (_, clients):
        clients[1]._sock.sendall(b'{"progress": "nine"}\n')
        _until(lambda: clients[0].record.active_mask == 0b01)


def test_deployment_at_rest_is_quiet():
    """Send no record and write nothing to the store while the membership does not change."""
    with _deployment(2) as (store, clients):
        clients[1].stop()
        _until(lambda: _committed_without_rank_1(clients[0]))
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
        _until(lambda: _committed_without_rank_1(clients[0]))
        assert json.loads(store.get(membership.DISCONNECTED_KEY)) == [-1, 0]
        held = open_liveness_connection(store, 1, replacement=True)  # incarnation 1, held open
        clients[1] = _member(store, 1, 2, replacement=True)  # incarnation 2
        assert clients[1].incarnation == 2
        _until(lambda: clients[0].record.active_mask == 0b11)
        clients[1].stop()
        _until(lambda: _committed_without_rank_1(clients[0]))
        assert json.loads(store.get(membership.DISCONNECTED_KEY)) == [-1, 0]
        held.sock.close()
        _until(lambda: json.loads(store.get(membership.DISCONNECTED_KEY)) == [-1, 2])


class FailingAuthority(Authority):
    """An authority whose decision raises once ``fail`` is set."""

    def __init__(self, world_size):
        super().__init__(world_size)
        self.fail = threading.Event()

    def tick(self, now_s, reports, connected):
        if self.fail.is_set():
            raise RuntimeError("a fault in the authority")
        return super().tick(now_s, reports, connected)


def test_failed_authority_ends_without_verdict():
    """Close the connections of a failed authority and let the ranks serve on unprotected."""
    store = RendezvousStore(DictStore(), b"shared-secret")
    authority = FailingAuthority(2)
    raised = []

    def run():
        try:
            AuthorityProcess(2, store, authority).run(host="127.0.0.1")
        except RuntimeError as error:
            raised.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    _until(lambda: store.store.check([membership.LIVENESS_KEY]))
    errors = []
    clients = [
        MembershipClient(r, store, lambda record: None, on_error=errors.append) for r in range(2)
    ]
    for client in clients:
        client.start()
        client.report(started=True)
    try:
        _until(
            lambda: all(
                client.record is not None and client.record.state == MembershipState.COMMITTED
                for client in clients
            )
        )
        authority.fail.set()
        thread.join(10)
        assert len(raised) == 1
        _until(lambda: len(errors) == 2)
        assert all(client.record.state == MembershipState.COMMITTED for client in clients)
        assert (
            MembershipRecord.model_validate_json(store.get(RECORD_KEY)).state
            == MembershipState.COMMITTED
        )
    finally:
        _leave(clients)


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


def test_abort_and_rejoin_are_summarized():
    cluster = Cluster()
    cluster.alive.discard(2)
    cluster.run(1.0)
    cluster.alive.add(2)
    cluster.reports[2] = RankReport()
    cluster.authority.request_join(2)
    record = cluster.run(1.0)
    assert cluster.authority.last_transaction.generation == record.generation
    assert cluster.authority.last_transaction.added_ranks == (2,)
    assert cluster.authority.last_transaction.removed_ranks == ()

    cluster.alive.discard(0)
    assert cluster.run(1.0).state == MembershipState.ABORTED
    summary = cluster.authority.last_transaction
    assert (summary.generation, summary.outcome) == (record.generation, "ABORTED")
    assert (summary.removed_ranks, summary.added_ranks) == ((0,), ())
    # An ABORT ends the transaction before any READY, in the tick that detected it.
    assert summary.ready_unix_ms is None
    assert summary.detected_unix_ms == summary.began_unix_ms == summary.ended_unix_ms


def test_transaction_summary_round_trip():
    summary = TransactionSummary(
        generation=3,
        outcome="COMMITTED",
        removed_ranks=(0, 5),
        added_ranks=(2,),
        detected_unix_ms=1_762_000_000_122.0,
        began_unix_ms=1_762_000_000_123.0,
        ready_unix_ms=1_762_000_000_243.0,
        ended_unix_ms=1_762_000_001_373.0,
    )
    assert TransactionSummary.model_validate_json(summary.model_dump_json()) == summary


def test_client_stops_once_no_longer_listed(monkeypatch):
    """Stop a replacement's client when the restarted change leaves it out, even before any commit."""
    monkeypatch.setattr(membership, "REPORT_INTERVAL_S", 60.0)
    records = []
    client = MembershipClient(1, RendezvousStore(None, b"secret"), records.append, replacement=True)
    client.incarnation = 1
    client._sock, authority_side = _socket_pair()
    client._lines = membership._Lines()
    client._thread.start()
    try:
        unlisted = MembershipRecord(
            generation=1, state=MembershipState.COMMITTED, active_mask=0b01, incarnations=(0, 1)
        )
        adding = MembershipRecord(
            generation=2,
            state=MembershipState.PROPOSE,
            active_mask=0b11,
            incarnations=(0, 1),
            joining_mask=0b10,
        )
        left_out = MembershipRecord(
            generation=3, state=MembershipState.PROPOSE, active_mask=0b01, incarnations=(0, 1)
        )
        authority_side.sendall(membership._line(unlisted) + membership._line(adding))
        _until(lambda: len(records) == 2)
        assert client._thread.is_alive()
        authority_side.sendall(membership._line(left_out))
        _until(lambda: not client._thread.is_alive())
        assert records == [unlisted, adding, left_out]
    finally:
        client.stop()
        authority_side.close()


class CountingAuthority(Authority):
    """An authority that counts its decisions."""

    def __init__(self, world_size):
        super().__init__(world_size)
        self.decisions = 0

    def tick(self, now_s, reports, connected):
        self.decisions += 1
        return super().tick(now_s, reports, connected)


def test_authority_decides_per_interval_or_reply(monkeypatch):
    """Decide on a progress-only report at the next interval, and on the startup and shutdown at once."""
    monkeypatch.setattr(membership, "REPORT_INTERVAL_S", 1.0)
    store = RendezvousStore(DictStore(), b"shared-secret")
    authority = CountingAuthority(1)
    process = AuthorityProcess(1, store, authority)
    thread = threading.Thread(target=lambda: process.run(host="127.0.0.1"), daemon=True)
    thread.start()
    _until(lambda: store.store.check([membership.LIVENESS_KEY]))
    client = MembershipClient(0, store, lambda record: None)
    try:
        client.start()
        reported_s = time.monotonic()
        client.report(started=True)
        _until(
            lambda: client.record is not None and client.record.state == MembershipState.COMMITTED
        )
        assert time.monotonic() - reported_s < 0.5
        decisions = authority.decisions
        for progress in range(100):  # 100 reports in about 1 s
            client.report(progress=progress)
            time.sleep(0.01)
        assert authority.decisions - decisions <= 4
        reported_s = time.monotonic()
        client.report(shutdown=True)
        _until(lambda: client.record.state == MembershipState.SHUTDOWN)
        assert time.monotonic() - reported_s < 0.5
    finally:
        _leave([client])
        thread.join(10)


def test_connections_without_hello_are_capped(monkeypatch):
    """Close the oldest connection that waits for its hello when a new one arrives over the cap."""
    store = RendezvousStore(DictStore(), b"shared-secret")
    process = AuthorityProcess(1, store, Authority(1))
    monkeypatch.setattr(membership, "MAX_RANKS", 2)
    thread = threading.Thread(target=lambda: process.run(host="127.0.0.1"), daemon=True)
    thread.start()
    _until(lambda: store.store.check([membership.LIVENESS_KEY]))
    host, port = store.get(membership.LIVENESS_KEY).decode().rsplit(":", 1)
    silent = []
    try:
        for _ in range(3):
            silent.append(socket.create_connection((host, int(port)), timeout=10))
            membership._read_line(silent[-1])  # accepted
        assert silent[0].recv(1) == b""  # closed for the third
        silent[1].settimeout(0.5)
        with pytest.raises(socket.timeout):
            silent[1].recv(1)
        with open_liveness_connection(store, 0) as rank_0:  # a valid hello still opens one
            rank_0.sock.sendall(membership._line(RankReport(shutdown=True)))
            rank_0.sock.settimeout(10)
            assert membership._read_line(rank_0.sock)  # the record, at the hello
            assert b'"state":4' in membership._read_line(rank_0.sock)  # SHUTDOWN
        thread.join(10)
        assert not thread.is_alive()
    finally:
        for sock in silent:
            sock.close()

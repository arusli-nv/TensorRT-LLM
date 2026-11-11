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
"""The WideEP fault-tolerance authority's decisions, driven by simulated ranks (no GPU, no I/O)."""

import contextlib
import itertools
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
    MembershipAgent,
    MembershipRecord,
    MembershipState,
    RankReport,
    Reply,
    ResumeState,
    SignedStore,
    TransactionSummary,
    claim_incarnation,
    claimed_incarnation,
    open_liveness_connection,
)

pytestmark = pytest.mark.cpu_only

WORLD = 4
ALL = (1 << WORLD) - 1


@pytest.fixture(autouse=True)
def _report_interval(monkeypatch):
    """The agents' report interval and the authority process's decision interval, short for the tests."""
    monkeypatch.setattr(membership, "REPORT_INTERVAL_S", 0.05)


def _formed(authority):
    """``authority`` once every rank has started: generation 0 is COMMITTED."""
    world = authority.world_size
    reports = {r: RankReport(started=True) for r in range(world)}
    assert authority.tick(0.0, reports, (1 << world) - 1).state == MembershipState.COMMITTED
    return authority


class Cluster:
    """Simulated ranks: each connected process advances its progress and answers the authority's record."""

    def __init__(self, world=WORLD, **timeouts):
        self.authority = _formed(Authority(world, **timeouts))
        self.now = 0.0
        self.alive = set(range(world))  # processes that hold their connection
        self.hung = set()  # connected but making no progress
        # Stopped processes (as by SIGSTOP): connected, but their agent thread stops too, so their reports do.
        self.stopped = set()
        self.idle = set()  # main's hang detector paused, as while waiting for requests
        self.shutdown = set()  # ranks whose report says that the deployment ends normally
        self.report_every = {}  # rank -> steps between its reports (an agent thread starved of the GIL)
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
        self.stopped_at = {r: ResumeState(iteration=10 + r, epoch=100 + r) for r in range(world)}
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
            if r not in self.hung | self.not_replying and record.is_live(r):
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
                stopped_at=self.stopped_at[r],
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
        generation=0, state=MembershipState.COMMITTED, live_mask=ALL
    )


def test_a_closed_connection_removes_the_rank_through_one_transaction():
    cluster = Cluster()
    cluster.alive.discard(2)
    record = cluster.run(1.0)
    assert states(cluster) == TRANSACTION
    assert record.generation == 1 and record.live_mask == ALL & ~(1 << 2)
    # Resume from the furthest survivor, with the next epoch.
    assert record.resume.iteration == 13 and record.resume.epoch == 104


def test_the_next_epoch_follows_the_latest_survivor_across_the_epoch_wrap():
    cluster = Cluster()
    for r, epoch in {0: 0xFFFFFFFE, 1: 1, 3: 1}.items():  # ranks 1 and 3 went on past the wrap
        cluster.stopped_at[r] = ResumeState(epoch=epoch)
    cluster.alive.discard(2)
    assert cluster.run(1.0).resume.epoch == 2


def test_install_carries_every_live_rank_s_ready_payload_unread():
    """The restore's input goes from the READY replies to every live rank in the INSTALL record, as each rank sent
    it; a joining rank's comes with its readiness. No other record carries any, and the record's log line leaves
    them out."""
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
        generation=2, state=MembershipState.PROPOSE, live_mask=ALL, joining_mask=0b0101
    )
    assert lowest.lowest_survivor() == 1
    assert lowest.model_copy(update=dict(live_mask=0b0101)).lowest_survivor() is None


def test_a_slow_rank_is_kept_and_a_hung_one_is_removed():
    cluster = Cluster(progress_timeout_s=1.0)
    cluster.hung.add(1)
    cluster.run(3.0)  # nobody waits for rank 1: slow, not hung
    assert cluster.authority.record.live_mask == ALL
    cluster.waiting_for[0] = 1 << 1
    record = cluster.run(3.0)
    assert record.state == MembershipState.COMMITTED and not record.is_live(1)


@pytest.mark.parametrize("waits_in", ["all-to-all", "host exchange"])
def test_a_rank_that_delivers_the_rounds_its_peers_wait_for_is_not_hung(waits_in):
    """In a step longer than the progress timeout, rank 3's iteration does not advance, but every round its peers
    wait for it in (an all-to-all kernel's round flag, a host-exchange epoch) arrives: only a round that does not is a
    hang."""
    cluster = Cluster(progress_timeout_s=1.0)
    rounds = cluster.waited_rounds if waits_in == "all-to-all" else cluster.waited_epochs
    cluster.hung.add(3)  # its iteration does not advance
    cluster.waiting_for.update({r: 1 << 3 for r in range(3)})
    for round_ in range(1, 13):  # a new round every 0.25 s for 3 s
        rounds.update({r: ((3, round_),) for r in range(3)})
        cluster.run(0.25)
    assert cluster.published == []
    record = cluster.run(1.5)  # the same round for longer than the progress timeout
    assert record.state == MembershipState.COMMITTED and not record.is_live(3)


def test_an_idle_rank_others_wait_for_is_not_hung():
    cluster = Cluster(progress_timeout_s=1.0)
    cluster.idle.update(
        range(WORLD)
    )  # an idle server: rank 0 waits for requests, the others wait for rank 0
    cluster.waiting_for.update({r: 1 for r in range(1, WORLD)})
    cluster.run(5.0)
    assert cluster.published == []


def test_a_rank_that_hangs_while_idle_is_hung():
    cluster = Cluster(progress_timeout_s=1.0)
    cluster.idle.update(range(WORLD))
    cluster.waiting_for.update({r: 1 for r in range(1, WORLD)})
    cluster.run(0.5)
    cluster.stopped.add(0)  # rank 0's process stops in its idle wait: its reports stop
    record = cluster.run(3.0)
    assert record.state == MembershipState.ABORT  # rank 0 is lost


def test_a_hung_rank_is_logged_with_what_every_live_rank_waits_on(monkeypatch):
    warnings = []
    monkeypatch.setattr(membership.logger, "warning", warnings.append)
    cluster = Cluster(progress_timeout_s=1.0)
    cluster.hung.add(3)
    cluster.waiting_for.update({r: 1 << 3 for r in range(3)})
    cluster.waited_rounds.update({r: ((3, 7),) for r in range(3)})
    cluster.run(3.0)
    [hung] = [w for w in warnings if "is hung" in w]
    assert hung.startswith("Membership: rank 3 is hung")
    assert "rank 0: iteration" in hung and "waits for [3], all-to-all rounds [(3, 7)]" in hung
    assert "rank 3: iteration" in hung and "waits for []" in hung


def test_an_idle_rank_behind_its_peers_is_hung():
    cluster = Cluster(progress_timeout_s=1.0)
    cluster.run(0.5)
    cluster.idle.add(3)  # rank 3 claims to be idle, but the others moved on and wait for it
    cluster.waiting_for.update({r: 1 << 3 for r in range(3)})
    record = cluster.run(3.0)
    assert record.state == MembershipState.COMMITTED and not record.is_live(3)


def test_a_frozen_rank_s_last_report_does_not_remove_the_peers_it_waited_for():
    """Rank 2's process stops while it briefly waits for rank 1, which then waits for rank 2: only rank 2 is hung."""
    authority = _formed(Authority(WORLD, progress_timeout_s=30.0))
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
    assert record.state == MembershipState.PROPOSE and record.live_mask == ALL & ~(1 << 2)


# The hang rule, case by case: only the end of a wait-for chain that does not move for the progress timeout is hung.


def _first_change(cluster, seconds):
    """The first record the authority publishes within ``seconds``, or None."""
    for _ in range(round(seconds / 0.05)):
        if cluster.step() is not None:
            return cluster.published[-1]
    return None


def _release(cluster):
    """The live ranks resume once the authority removed the hung ranks: no rank waits or hangs any longer."""
    cluster.hung.clear()
    cluster.waiting_for.clear()
    cluster.waited_rounds.clear()
    cluster.waited_epochs.clear()


@pytest.mark.parametrize("stopped", [False, True])
def test_only_the_end_of_a_wait_for_chain_is_hung(stopped):
    """Rank 2 hangs: its process stops (SIGSTOP) or only its main thread does. Rank 3's host is blocked on its GPU,
    whose all-to-all waits on rank 2, and ranks 0 and 1 wait in the host exchange for both: rank 3 is waited on and
    makes no progress too, but it waits on rank 2, so only rank 2 is hung."""
    cluster = Cluster(progress_timeout_s=1.0)
    cluster.run(0.5)
    (cluster.stopped if stopped else cluster.hung).add(2)
    cluster.hung.update({0, 1, 3})
    cluster.waiting_for.update({0: 0b1100, 1: 0b1100, 3: 1 << 2})
    cluster.waited_rounds[3] = ((2, 7),)
    hung_s = cluster.now
    record = _first_change(cluster, 2.0)
    assert record.state == MembershipState.PROPOSE and record.live_mask == ALL & ~(1 << 2)
    assert 1.0 < cluster.now - hung_s <= 1.2
    _release(cluster)
    assert cluster.run(1.0).live_mask == ALL & ~(1 << 2)
    assert states(cluster) == TRANSACTION


@pytest.mark.parametrize("stopped_in", ["all-to-all", "host exchange", "paused wait"])
def test_a_stopped_process_is_hung_wherever_it_stopped(stopped_in):
    """Rank 2's process stops (SIGSTOP), and its agent thread with it, while it waits on a peer: in an all-to-all
    kernel or the host exchange on rank 3, or on rank 0 in the request broadcast, which pauses main's hang detector.
    Its last report keeps naming that peer, which now waits on rank 2 (a cycle, one end): rank 2 is hung one hang
    timeout after its stop, not half a hang timeout later once its report expires, and the peer is kept."""
    cluster = Cluster(progress_timeout_s=1.0)
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
    assert record.state == MembershipState.PROPOSE and record.live_mask == ALL & ~(1 << 2)
    assert 1.0 < cluster.now - stopped_s <= 1.2
    _release(cluster)
    assert cluster.run(1.0).state == MembershipState.COMMITTED
    assert MembershipState.ABORT not in states(cluster)


@pytest.mark.parametrize("peers_paused", [False, True])
def test_a_rank_whose_main_thread_hangs_in_a_paused_wait_is_hung(peers_paused):
    """Rank 2's main thread hangs inside a wait that pauses main's hang detector (the request broadcast), while its
    agent thread reports it idle; the others wait on it, in that broadcast too or not. Only rank 0 waiting for client
    requests may stay idle while others wait on it."""
    cluster = Cluster(progress_timeout_s=1.0)
    cluster.run(0.5)
    cluster.idle.add(2)
    (cluster.idle if peers_paused else cluster.hung).update({0, 1, 3})
    cluster.waiting_for.update({r: 1 << 2 for r in (0, 1, 3)})
    record = _first_change(cluster, 2.0)
    assert record.state == MembershipState.PROPOSE and record.live_mask == ALL & ~(1 << 2)


def test_a_rank_starved_of_the_gil_is_not_hung():
    """Rank 2's agent thread reports only every 0.4 s and its iteration advances as slowly, while the others wait on
    it: its reports stay fresh and it advances within the progress timeout."""
    cluster = Cluster(progress_timeout_s=1.0)
    cluster.report_every[2] = 8
    cluster.hung.update({0, 1, 3})
    cluster.waiting_for.update({r: 1 << 2 for r in (0, 1, 3)})
    cluster.run(5.0)
    assert cluster.published == []


def _idle_deployment():
    """Rank 0 waits for client requests; the others wait on it in the request broadcast."""
    cluster = Cluster(progress_timeout_s=1.0)
    cluster.idle.update(range(WORLD))
    cluster.waiting_for.update({r: 1 for r in range(1, WORLD)})
    cluster.run(5.0)
    return cluster


def test_a_rank_late_once_after_a_long_idle_wait_is_not_hung():
    cluster = _idle_deployment()
    cluster.idle.clear()  # a request arrives; rank 2 is late to the next collective
    cluster.hung.update(range(WORLD))
    cluster.waiting_for.update({r: 1 << 2 for r in (0, 1, 3)})
    cluster.run(0.3)
    _release(cluster)
    cluster.run(2.0)
    assert cluster.published == []


def test_a_rank_stopped_in_an_idle_deployment_is_hung_once_waited_on_for_the_progress_timeout():
    """Nobody waits on rank 2 while the deployment is idle; the first request makes the others wait on it."""
    cluster = _idle_deployment()
    cluster.stopped.add(2)
    cluster.run(5.0)
    assert cluster.published == []
    cluster.idle.clear()
    cluster.hung.update({0, 1, 3})
    cluster.waiting_for.update({r: 1 << 2 for r in (0, 1, 3)})
    request_s = cluster.now
    record = _first_change(cluster, 2.0)
    assert record.state == MembershipState.PROPOSE and record.live_mask == ALL & ~(1 << 2)
    assert 1.0 < cluster.now - request_s <= 1.2


def test_ranks_that_wait_only_on_each_other_are_one_end():
    """Ranks 2 and 3 wait on each other with their agents running, and the others wait on rank 2. Collectives that
    send before they wait form no such cycle among live processes (a stopped one's report expires first); a fault
    beyond process failures may. No report singles one out: both are hung, and serving goes on without them."""
    cluster = Cluster(progress_timeout_s=1.0)
    cluster.run(0.5)
    cluster.hung.update(range(WORLD))
    cluster.waiting_for.update({0: 1 << 2, 1: 1 << 2, 2: 1 << 3, 3: 1 << 2})
    record = _first_change(cluster, 2.0)
    assert record.state == MembershipState.PROPOSE and record.live_mask == 0b0011
    _release(cluster)
    assert cluster.run(1.0).state == MembershipState.COMMITTED
    assert MembershipState.ABORT not in states(cluster)


@pytest.mark.parametrize("how", ["hung", "stopped"])
@pytest.mark.parametrize("phase", [MembershipState.PROPOSE, MembershipState.INSTALL])
def test_a_rank_that_hangs_during_a_transaction_restarts_it_after_the_progress_timeout(phase, how):
    """The authority waits on every live rank for the phase's reply: a rank that hangs in the phase is failed like
    one that crashes, once it has not moved for the progress timeout."""
    cluster = Cluster(progress_timeout_s=1.0)
    cluster.alive.discard(1)  # a crash starts the transaction
    while cluster.authority.record.state != phase:
        cluster.step()
    getattr(cluster, how).add(3)
    phase_s = cluster.now
    record = _first_change(cluster, 2.0)
    assert record.state == MembershipState.PROPOSE and record.live_mask == 0b0101
    assert 1.0 < cluster.now - phase_s <= 1.2
    record = cluster.run(1.0)
    assert record.state == MembershipState.COMMITTED and record.live_mask == 0b0101
    assert MembershipState.ABORT not in states(cluster)


@pytest.mark.parametrize("phase", [MembershipState.PROPOSE, MembershipState.INSTALL])
def test_an_idle_rank_0_that_hangs_during_a_transaction_is_hung(phase):
    """Main's hang detector stays paused while a rank takes part in a membership change, so a rank 0 whose main thread
    hangs there reports idle ticks that advance. Only a committed membership exempts an idle rank 0: in a phase the
    authority waits on it for the reply, and the deployment aborts once it has not moved for the progress timeout."""
    cluster = Cluster(progress_timeout_s=1.0)
    cluster.alive.discard(1)  # a crash starts the transaction
    while cluster.authority.record.state != phase:
        cluster.step()
    cluster.idle.add(0)
    cluster.not_replying.add(0)
    record = _first_change(cluster, 2.0)
    assert record is not None and record.state == MembershipState.ABORT


def test_a_wait_on_a_rank_that_is_no_longer_live_exempts_no_rank():
    """Rank 3's all-to-all waited on rank 1 when rank 1's process ended, and rank 3 then hangs in the change, its
    reports still naming rank 1: a wait on a rank outside the live membership does not keep it from being hung."""
    cluster = Cluster(progress_timeout_s=1.0)
    cluster.waiting_for[3] = 1 << 1
    cluster.waited_rounds[3] = ((1, 7),)
    cluster.alive.discard(1)
    while cluster.authority.record.state != MembershipState.PROPOSE:
        cluster.step()
    cluster.hung.add(3)
    record = _first_change(cluster, 2.0)
    assert record is not None and record.state == MembershipState.PROPOSE
    assert record.live_mask == 0b0101
    cluster.waiting_for.clear()
    cluster.waited_rounds.clear()
    record = cluster.run(1.0)
    assert record.state == MembershipState.COMMITTED and record.live_mask == 0b0101


def test_a_phase_waits_for_a_rank_as_long_as_it_advances():
    """No fixed deadline: rank 3 takes longer than any timeout to reply READY, finishing a step whose progress
    shows, and the transaction goes on with it once it replies."""
    cluster = Cluster()
    cluster.alive.discard(1)
    cluster.not_replying.add(3)
    cluster.run(70.0)
    assert cluster.authority.record.state == MembershipState.PROPOSE
    assert cluster.authority.record.is_live(3)
    cluster.not_replying.clear()
    record = cluster.run(1.0)
    assert record.state == MembershipState.COMMITTED and record.live_mask == ALL & ~(1 << 1)


def test_a_failure_during_the_transaction_restarts_it():
    cluster = Cluster()
    cluster.alive.discard(3)
    cluster.hung.add(2)  # 2 cannot reply, so the first transaction waits ...
    cluster.run(0.3)
    assert cluster.authority.record.state == MembershipState.PROPOSE
    cluster.alive.discard(2)  # ... and then 2 dies too
    record = cluster.run(2.0)
    assert record.state == MembershipState.COMMITTED
    assert record.live_mask == 0b0011 and record.generation == 2


def test_installed_replies_that_restored_different_placements_abort():
    cluster = Cluster()
    cluster.digest[1] = "placement-b"
    cluster.alive.discard(3)
    assert cluster.run(1.0).state == MembershipState.ABORT
    assert MembershipState.COMMITTED not in states(cluster)


def test_a_replacement_that_restored_another_placement_than_the_survivors_is_not_added():
    """The survivors agree with each other: they commit without the replacement, which never ends the deployment."""
    cluster = Cluster()
    cluster.alive.discard(2)
    cluster.run(1.0)
    cluster.published.clear()
    cluster.digest[2] = "placement-b"
    _join(cluster, 2)
    record = cluster.run(1.0)
    assert record.state == MembershipState.COMMITTED and record.live_mask == ALL & ~(1 << 2)
    assert record.joining_mask == 0 and MembershipState.ABORT not in states(cluster)
    assert states(cluster) == [MembershipState.PROPOSE, MembershipState.INSTALL] + TRANSACTION


@pytest.mark.parametrize("slots", ["spare", "none spare"])
def test_a_rank_that_fails_to_install_aborts(slots):
    """Also when no rank holds an expert the failed rank alone held: the live ranks' plan of the restore is the one
    rule for whether they keep every expert, and it fails the install on every live rank."""
    cluster = Cluster()
    if slots == "none spare":  # rank 3 alone held an expert
        cluster.install_reply = {r: Reply.FAILED for r in range(WORLD)}
    else:
        cluster.install_reply[1] = Reply.FAILED
    cluster.alive.discard(3)
    assert cluster.run(1.0).state == MembershipState.ABORT
    assert states(cluster) == [
        MembershipState.PROPOSE,
        MembershipState.INSTALL,
        MembershipState.ABORT,
    ]


def test_a_survivor_lost_during_an_admission_leaves_the_others_serving():
    """Ranks 2 and 3 were lost and their experts restored into the spare slots of ranks 0 and 1; rank 2's
    replacement joins, and rank 1 fails before the admission installs. Only one spare slot of the live ranks is
    empty for the two lost experts, but the joining rank holds expert 2 again, so the plan empties rank 0's spare
    slot too: the authority leaves that to the live ranks' plan instead of counting empty slots, and the change
    commits."""
    cluster = Cluster()
    cluster.alive -= {2, 3}
    assert cluster.run(1.0).state == MembershipState.COMMITTED
    _join(cluster, 2)  # rank 3's replacement did not ask to join: rank 2's does not wait for it
    while cluster.authority.record.state != MembershipState.PROPOSE:
        cluster.step()
    cluster.alive.discard(1)
    record = cluster.run(1.0)
    assert record.state == MembershipState.COMMITTED and record.live_mask == 0b0101
    assert MembershipState.ABORT not in states(cluster)


def test_a_generation_commits_only_once_every_live_rank_installed_it():
    cluster = Cluster(progress_timeout_s=1.0)
    cluster.alive.discard(3)
    cluster.run(1.0)  # generation 1: every survivor's last reply is INSTALLED
    cluster.alive.discard(2)
    while cluster.authority.record.state != MembershipState.INSTALL:
        cluster.step()
    cluster.hung.add(1)  # replied READY to generation 2, never INSTALLED
    record = cluster.run(3.0)
    assert record.state == MembershipState.COMMITTED and record.live_mask == 0b0001
    committed = [r.generation for r in cluster.published if r.state == MembershipState.COMMITTED]
    assert committed == [1, 3]  # never generation 2, which rank 1 did not install


def test_survivors_that_made_no_progress_during_the_transaction_are_not_hung_at_its_commit():
    authority = _formed(Authority(WORLD, progress_timeout_s=1.0))
    calls = itertools.count(1)

    def reports(reply=Reply.NONE, generation=0, waiting_for=0):
        # Every rank is stuck at iteration 5 from the start; the survivors' replies follow the record.
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
    assert authority.record.is_live(1)


def test_a_rank_waited_on_through_a_transaction_has_a_full_progress_timeout_after_its_commit():
    """Rank 2 waits on rank 1 during the install (as in a gather through the store), and rank 1 replies late in the
    phase, then takes half the progress timeout for its first step after the commit while rank 2 still waits on it.
    Its wait during the transaction does not count: the hang clocks restart at the commit."""
    cluster = Cluster(progress_timeout_s=1.0)
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
    assert record.state == MembershipState.COMMITTED and record.live_mask == 0b0111
    assert states(cluster) == TRANSACTION


def test_losing_rank_0_aborts():
    cluster = Cluster()
    cluster.alive.discard(0)
    assert cluster.run(1.0).state == MembershipState.ABORT


@pytest.mark.parametrize("silent_rank, survives", [(1, True), (0, False)])
def test_a_rank_that_does_not_reply_in_a_phase_for_the_progress_timeout_counts_as_failed(
    silent_rank, survives
):
    """The transaction restarts without it, unless it is rank 0: the deployment then aborts."""
    cluster = Cluster(progress_timeout_s=1.0)
    cluster.alive.discard(3)
    cluster.hung.add(silent_rank)  # connected but never replies; only the authority waits on it
    record = cluster.run(3.0)
    if survives:
        assert record.state == MembershipState.COMMITTED and record.live_mask == 0b0101
        assert MembershipState.ABORT not in states(cluster)
    else:
        assert record.state == MembershipState.ABORT


def test_during_a_transaction_only_the_end_of_a_wait_for_chain_counts_as_failed():
    """Rank 2 cannot reply INSTALLED while it waits for rank 1 (a gather of the install): only rank 1 failed."""
    cluster = Cluster(progress_timeout_s=1.0)
    cluster.alive.discard(3)
    while cluster.authority.record.state != MembershipState.INSTALL:
        cluster.step()
    cluster.hung.update({1, 2})
    cluster.waiting_for[2] = 1 << 1
    while cluster.authority.record.state == MembershipState.INSTALL:
        cluster.step()
    assert cluster.authority.record.state == MembershipState.PROPOSE
    assert cluster.authority.record.live_mask == 0b0101


def test_each_phase_has_its_own_progress_timeout():
    cluster = Cluster(progress_timeout_s=1.0)
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


def test_an_install_that_keeps_completing_steps_is_not_hung():
    """Installing can take long (a large expert restore, the memory of many peers to map): every step a rank
    completes is progress, so only an install that stops advancing for the hang timeout is a hang."""
    cluster = Cluster(progress_timeout_s=1.0)
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
    assert record.state == MembershipState.PROPOSE and record.live_mask == 0b0101
    assert 1.0 < cluster.now - stalled_s <= 1.2


def test_a_replacement_joins_and_one_that_dies_while_joining_is_dropped():
    cluster = Cluster()
    cluster.alive.discard(2)
    cluster.run(1.0)
    cluster.alive.add(2)  # a new process for rank 2, with fresh reports
    cluster.reports[2] = RankReport()
    cluster.authority.request_join(2)
    record = cluster.run(1.0)
    assert (
        record.state == MembershipState.COMMITTED
        and record.live_mask == ALL
        and record.generation == 2
    )

    cluster.alive.discard(3)
    cluster.run(1.0)
    cluster.reports[3] = RankReport()
    cluster.authority.request_join(3)  # the replacement never connects
    record = cluster.run(1.0)
    assert record.state == MembershipState.COMMITTED and not record.is_live(3)


def _join(cluster, rank):
    cluster.alive.add(rank)  # a new process for the rank, with fresh reports
    cluster.reports[rank] = RankReport()
    cluster.authority.request_join(rank)


def test_the_replacements_of_ranks_lost_together_are_added_in_one_transaction():
    """A node's ranks are lost in one transaction; their replacements are ready to join a little apart."""
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
    assert states(cluster) == TRANSACTION and record.live_mask == ALL
    assert cluster.authority.last_transaction.added_ranks == (2, 3)
    # Detected at rank 2's readiness: the wait for rank 3's is part of the transaction.
    summary = cluster.authority.last_transaction
    assert summary.began_unix_ms - summary.detected_unix_ms >= 2000


def test_the_proposal_that_adds_replacements_logs_them(monkeypatch):
    """One line per admission, none for the failure."""
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
    assert line == "Membership: adding joining ranks [2, 3]"


def test_a_transaction_s_phases_end_at_ordered_wall_clock_times_in_milliseconds():
    """Detected at rank 2's readiness, proposed once rank 3's replacement is ready too, then READY and INSTALLED."""
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
    assert states(cluster) == TRANSACTION and record.live_mask == ALL
    summary = cluster.authority.last_transaction
    times = {
        "detected": summary.detected_unix_ms,
        "proposed": summary.began_unix_ms,
        "ready": summary.ready_unix_ms,
        "committed": summary.ended_unix_ms,
    }
    assert times["detected"] < times["proposed"] < times["ready"] < times["committed"]
    # The authority's clock at the detection and at each record it published (PROPOSE, then INSTALL once every READY
    # was in, then COMMITTED once every INSTALLED was in), in milliseconds.
    ticks_ms = [t * 1e3 for t in (detected, *cluster.published_at)]
    phases_ms = [later - earlier for earlier, later in itertools.pairwise(times.values())]
    assert phases_ms == pytest.approx([b - a for a, b in itertools.pairwise(ticks_ms)], abs=1)
    # Milliseconds since the Unix epoch: the tick that committed ran between the two wall-clock reads.
    assert before_s * 1e3 - 1 <= times["committed"] <= after_s * 1e3 + 1


@pytest.mark.parametrize("how", ["stops", "ends"])
def test_a_ready_join_waits_for_a_replacement_lost_with_its_rank_only_while_its_reports_advance(
    how,
):
    """Rank 3's replacement asked to join and is still starting: rank 2's, ready, waits for it as long as it takes,
    but no longer once its reports stop advancing (half the hang timeout after a SIGSTOP), nor once its process
    ended. Rank 2's is then added alone."""
    cluster = Cluster(progress_timeout_s=2.0)
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
        assert cluster.run(0.9).state == MembershipState.COMMITTED and cluster.published == []
    else:
        cluster.alive.discard(3)
    record = cluster.run(0.5)
    assert states(cluster) == TRANSACTION and record.live_mask == ALL & ~(1 << 3)


def test_a_ready_join_stops_waiting_for_a_replacement_whose_startup_makes_no_progress():
    """Rank 3's replacement reports on but its startup stalls: rank 2's, ready, waits for it only until its startup
    made no progress for the rendezvous timeout, then is added alone. Fails without the change: it waited forever."""
    cluster = Cluster(rendezvous_timeout_s=5.0)
    cluster.alive -= {2, 3}
    cluster.run(1.0)
    cluster.published.clear()
    _join(cluster, 2)
    _join(cluster, 3)
    cluster.unprepared.add(3)
    cluster.run(8.0)  # its startup progresses: rank 2's waits past the bound
    assert cluster.published == []
    cluster.hung.add(3)  # its startup stalls while its agent reports on
    assert cluster.run(4.5).state == MembershipState.COMMITTED and cluster.published == []
    record = cluster.run(1.0)
    assert states(cluster) == TRANSACTION and record.live_mask == ALL & ~(1 << 3)


def test_a_join_that_ends_before_it_is_proposed_leaves_no_detection_time_behind():
    """Rank 2's replacement is ready and waits for rank 3's, lost with it, then dies: nothing is due any longer. A
    later failure is detected when it happens, not at that join."""
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
    assert states(cluster) == TRANSACTION and not record.is_live(5)
    summary = cluster.authority.last_transaction
    assert summary.began_unix_ms == summary.detected_unix_ms


def test_a_ready_join_waits_for_the_replacements_lost_with_its_rank_to_be_ready_too():
    """Rank 3's replacement asked to join but is still starting: rank 2's, ready, waits for it, and both are added in
    one transaction."""
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
    assert states(cluster) == TRANSACTION and record.live_mask == ALL
    assert cluster.authority.last_transaction.added_ranks == (2, 3)


def test_a_ready_join_does_not_wait_for_the_ranks_lost_with_one_that_is_not_ready():
    """Ranks 5 and 6 were lost together, rank 2 later and alone. Rank 5's replacement asked to join but is still
    starting: rank 2's, ready, is proposed at once, without waiting for rank 6's."""
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


def test_a_replacement_is_proposed_only_once_it_is_ready_to_join():
    """One that asked to join and then went silent never stops the live ranks: they serve on without it."""
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
    assert states(cluster) == TRANSACTION and record.live_mask == ALL


def test_the_active_ranks_wait_for_a_silent_joining_rank_for_the_hang_timeout():
    """Its readiness stands for its READY: INSTALL follows the survivors' READY at once. A joining rank that goes
    silent once it reported ready is left out by the hang rule, as any live rank: the survivors commit without it,
    and its supervisor starts another replacement."""
    cluster = Cluster(progress_timeout_s=1.0)
    cluster.alive.discard(2)
    cluster.run(1.0)
    cluster.published.clear()
    cluster.published_at.clear()
    _join(cluster, 2)
    cluster.hung.add(2)  # ready, then silent: replies neither READY nor INSTALLED
    record = cluster.run(3.0)
    assert record.state == MembershipState.COMMITTED and record.live_mask == ALL & ~(1 << 2)
    assert states(cluster) == [MembershipState.PROPOSE, MembershipState.INSTALL] + TRANSACTION
    proposed, installed, restarted = cluster.published_at[:3]
    assert installed - proposed == pytest.approx(0.05)
    assert 1.0 < restarted - proposed <= 1.2


def test_the_transaction_that_adds_a_replacement_marks_it_joining():
    """The live ranks keep the host exchange to themselves until they resume; the record says who joins."""
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


def test_a_replacement_resumes_where_the_survivors_do():
    """Where a joining rank stopped moves neither the iteration nor the epoch every live rank resumes at."""
    cluster = Cluster()
    cluster.alive.discard(2)
    cluster.run(1.0)
    cluster.stopped_at[2] = ResumeState(iteration=50, epoch=150)
    _join(cluster, 2)
    record = cluster.run(1.0)
    assert record.live_mask == ALL and record.generation == 2
    assert record.resume == ResumeState(iteration=13, epoch=104)


def test_a_replacement_for_a_rank_still_live_takes_it_over():
    cluster = Cluster()
    cluster.reports[1] = RankReport()  # the replacement's reports replace the old process's
    cluster.authority.request_join(1)
    record = cluster.run(1.0)
    assert states(cluster) == TRANSACTION
    assert record.live_mask == ALL and record.generation == 1


@pytest.mark.parametrize("failure", ["failed_reply", "hang"])
@pytest.mark.parametrize("phase", [MembershipState.PROPOSE, MembershipState.INSTALL])
def test_a_replacement_that_cannot_join_is_left_out_and_the_others_commit(phase, failure):
    cluster = Cluster(progress_timeout_s=1.0)
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
    assert record.live_mask == ALL & ~(1 << 2) and record.joining_mask == 0
    assert MembershipState.ABORT not in states(cluster)


@pytest.mark.parametrize("phase", [MembershipState.PROPOSE, MembershipState.INSTALL])
@pytest.mark.parametrize("victim", [1, 2, 3])
def test_a_second_failure_at_each_phase_still_commits_without_both(phase, victim):
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
    assert not record.is_live(first) and not record.is_live(victim)


def test_record_and_report_serialization_round_trip():
    record = MembershipRecord(
        generation=7,
        state=MembershipState.INSTALL,
        live_mask=(1 << 200) | 0b1011,
        resume=ResumeState(iteration=1, epoch=2),
        incarnations=(0, 2, 0, 1),
        joining_mask=0b0010,
        ready_payloads=((0, "[[4, [[0, -1]]]]"), (1, "[[3, 1]]")),
    )
    assert MembershipRecord.model_validate_json(record.model_dump_json()) == record
    aborted = record.model_copy(update=dict(state=MembershipState.ABORT, reason="rank 0 failed"))
    assert MembershipRecord.model_validate_json(aborted.model_dump_json()) == aborted
    assert record.is_member(1, 2) and not record.is_member(1, 0) and not record.is_member(2, 0)
    report = RankReport(
        progress=9,
        idle=True,
        waiting_for=1 << 255,
        reply_generation=7,
        reply=Reply.INSTALLED,
        stopped_at=ResumeState(iteration=1, epoch=2),
        placement_digest="digest",
        incarnation=2,
        shutdown=True,
    )
    ready = RankReport(reply=Reply.READY, ready_payload="[[4, [[0, -1], [3, 1]]]]")
    for sent in (report, ready):
        assert RankReport.model_validate_json(sent.model_dump_json()) == sent
    # A line that is not a report, or whose field has another type, raises a ValueError: the connection closes.
    for line in (b"not json", b'{"progress": "nine"}', b'{"reply": 9}'):
        with pytest.raises(ValueError):
            RankReport.model_validate_json(line)


def test_rank_0_s_report_ends_the_deployment_normally():
    """Only rank 0's: then the ranks that leave are not failures."""
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


# The startup: STARTING until every rank has started; any loss before then aborts at once.


def _starting_reports(started=(), sequence=1):
    return {r: RankReport(started=r in started, sequence=sequence) for r in range(WORLD)}


def test_the_deployment_serves_once_every_rank_has_started():
    authority = Authority(WORLD)
    assert authority.record.state == MembershipState.STARTING
    assert authority.tick(0.0, _starting_reports(), ALL) is None
    assert authority.tick(1.0, _starting_reports(started={0, 1, 2}, sequence=2), ALL) is None
    record = authority.tick(2.0, _starting_reports(started=range(WORLD), sequence=3), ALL)
    assert (record.generation, record.state, record.live_mask) == (
        0,
        MembershipState.COMMITTED,
        ALL,
    )
    assert authority.last_transaction is None  # the startup is not a membership transaction


@pytest.mark.parametrize("lost", [0, 2])
def test_a_rank_lost_during_the_startup_aborts_at_once(lost):
    """Not at the rendezvous timeout: a deployment that lost a rank before it served cannot serve. The record says
    why, for the startup error."""
    authority = Authority(WORLD)
    authority.tick(0.0, _starting_reports(), ALL)
    record = authority.tick(0.05, _starting_reports(started={1}), ALL & ~(1 << lost))
    assert record.state == MembershipState.ABORT
    assert record.reason == f"ranks [{lost}] failed during the startup"


def test_the_startup_waits_for_the_ranks_not_connected_yet_until_the_rendezvous_timeout():
    """A rank that never connects aborts the startup at the rendezvous timeout; a model load does not count, as every
    rank connects at the start of its worker."""
    authority = Authority(WORLD, rendezvous_timeout_s=600.0)
    late = ALL & ~(1 << 3)
    for second in range(0, 600, 10):
        assert authority.tick(float(second), _starting_reports(sequence=second), late) is None
    # Connected at 590 s, the last rank starts 3000 s later: the deployment serves.
    for second in range(590, 3600, 10):
        assert authority.tick(float(second), _starting_reports(sequence=second), ALL) is None
    record = authority.tick(3600.0, _starting_reports(started=range(WORLD), sequence=3600), ALL)
    assert record.state == MembershipState.COMMITTED

    never = Authority(WORLD, rendezvous_timeout_s=600.0)
    never.tick(0.0, _starting_reports(), late)
    assert never.tick(599.0, _starting_reports(sequence=2), late) is None
    assert never.tick(601.0, _starting_reports(sequence=3), late).state == MembershipState.ABORT


def test_a_rank_whose_reports_stop_during_the_startup_aborts_it_once_waited_on_for_the_rendezvous_timeout():
    """A stopped process (SIGSTOP) keeps its connection open but its agent stops reporting too: once its peers wait
    for it in a startup collective, it is the end of their wait-for chain."""
    authority = Authority(WORLD, rendezvous_timeout_s=60.0)
    for second in range(0, 91):
        reports = {
            r: RankReport(sequence=second, progress=3, waiting_for=1 << 1) for r in range(WORLD)
        }
        reports[1] = RankReport(sequence=min(second, 10), progress=3)  # stopped at 10 s
        record = authority.tick(float(second), reports, ALL)
        if second <= 60:
            assert record is None
    assert authority.record.state == MembershipState.ABORT
    assert authority.record.reason == "ranks [1] failed during the startup"


@pytest.mark.parametrize("peer", ["hangs", "progresses"])
def test_a_rank_others_wait_for_in_a_startup_collective_is_hung_after_the_rendezvous_timeout(peer):
    """Its main thread hangs while its agent reports on: the ranks waiting for it in a startup collective name it, and
    the startup aborts once it made no progress (startup collectives completed) for the rendezvous timeout. A rank
    whose startup collectives go on is not hung however long the startup takes."""
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
        assert authority.record.state == MembershipState.ABORT
        assert authority.record.reason == "ranks [2] failed during the startup"


def test_no_replacement_joins_before_the_deployment_has_started():
    process = AuthorityProcess(WORLD, SignedStore(DictStore(), b"join-secret"), Authority(WORLD))
    assert "has started" in process._refusal(2, 1)
    _formed(process._authority)
    assert process._refusal(2, 1) is None  # rank 2's process is not connected


def test_the_authority_listens_at_the_port_after_the_rendezvous_port():
    """Two ports a firewall opens: the store's, and the next one for the liveness listener, bound to the rendezvous
    address. Fails without the change: the listener took an ephemeral port on every address."""
    assert membership.authority_ports("10.0.0.7:29600") == ("10.0.0.7", 29600, 29601)
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    store = SignedStore(DictStore(), b"join-secret")
    process = AuthorityProcess(1, store, Authority(1))
    thread = threading.Thread(target=lambda: process.run("127.0.0.1", port), daemon=True)
    thread.start()
    _until(lambda: store.store.check([membership.LIVENESS_KEY]))
    assert store.get(membership.LIVENESS_KEY) == f"127.0.0.1:{port}".encode()
    with open_liveness_connection(store, 0, 0) as rank_0:
        rank_0.sock.sendall(membership._line(RankReport(shutdown=True)))
        rank_0.sock.settimeout(10)
        membership._read_line(rank_0.sock)  # the record, at the hello
        assert b'"state":4' in membership._read_line(rank_0.sock)  # SHUTDOWN
    thread.join(10)
    assert not thread.is_alive()


def test_the_authority_aborts_once_the_process_that_started_it_has_ended():
    """Rank 0's process started it and holds the writing end of the pipe whose reading end is ``parent``: once that
    end closes, the deployment aborts, starting or not, and the authority waits for a live rank to leave for at most
    the hang timeout."""
    store = SignedStore(DictStore(), b"join-secret")
    read_fd, write_fd = os.pipe()
    process = AuthorityProcess(1, store, Authority(1, progress_timeout_s=2.0), parent=read_fd)
    returned = []
    thread = threading.Thread(
        target=lambda: returned.append(process.run(host="127.0.0.1")), daemon=True
    )
    try:
        thread.start()
        _until(lambda: store.store.check([membership.LIVENESS_KEY]))
        with open_liveness_connection(store, 0, 0):  # a rank that never leaves
            thread.join(0.3)
            assert thread.is_alive()
            os.close(write_fd)
            closed_s = time.monotonic()
            thread.join(10)
            assert 2.0 <= time.monotonic() - closed_s < 6.0
        assert [record.state for record in returned] == [MembershipState.ABORT]
        assert (
            MembershipRecord.model_validate_json(store.get(RECORD_KEY)).state
            == MembershipState.ABORT
        )
    finally:
        os.close(read_fd)


def test_an_agent_hands_the_newest_record_to_the_callbacks_it_is_bound_to():
    """The executor takes over the membership its process joined at the start of its worker."""
    agent = MembershipAgent(1, SignedStore(None, b"secret"), lambda record: None)
    agent.record = MembershipRecord(generation=0, state=MembershipState.COMMITTED, live_mask=ALL)
    records = []
    agent.bind(records.append)
    assert records == [agent.record]
    assert agent._on_record == records.append


def _socket_pair():
    with socket.create_server(("127.0.0.1", 0)) as listener:
        client = socket.create_connection(listener.getsockname())
        server, _ = listener.accept()
    return client, server


def test_lines_are_taken_whole_across_reads_and_a_line_too_long_closes_the_connection():
    lines = membership._Lines()
    assert lines.decode(b'{"progress"') == []
    assert lines.decode(b': 1}\n{"progress": 2}\n{') == [b'{"progress": 1}', b'{"progress": 2}']
    with pytest.raises(ValueError):
        lines.decode(b"x" * (membership._MAX_LINE_BYTES + 1))


def test_an_agent_whose_connection_ends_closes_it_and_reports_the_error():
    """The authority is gone: the rank serves unprotected, and its connection no longer counts as open."""
    errors = []
    agent = MembershipAgent(
        1, SignedStore(None, b"secret"), lambda record: None, on_error=errors.append
    )
    agent._sock, authority_side = _socket_pair()
    agent._lines = membership._Lines()
    agent.connected = True
    authority_side.close()
    agent._run()
    assert agent._sock.fileno() == -1
    assert len(errors) == 1 and isinstance(errors[0], ConnectionError)
    assert not agent.connected


def test_an_agent_follows_every_record_and_reports_at_once(monkeypatch):
    """The records and reports are JSON lines over the connection its hello authenticated."""
    monkeypatch.setattr(membership, "REPORT_INTERVAL_S", 60.0)
    records = []
    agent = MembershipAgent(1, SignedStore(None, b"secret"), records.append)
    agent._sock, authority_side = _socket_pair()
    agent._lines = membership._Lines()
    agent._thread.start()
    try:
        current = MembershipRecord(generation=1, state=MembershipState.COMMITTED, live_mask=ALL)
        newer = current.model_copy(update=dict(generation=2))
        authority_side.sendall(b"".join(membership._line(record) for record in (current, newer)))
        _until(lambda: len(records) == 2)
        agent.report(reply=Reply.READY, reply_generation=2)
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
        agent.stop()
        authority_side.close()


class DictStore:
    """The TCPStore calls of the authority and its ranks, in memory."""

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
        """As TCPStore's: sets ``desired`` if the value is ``expected`` (or unset and ``expected`` is empty), and
        returns the value after."""
        with self._lock:
            current = self.values.get(key)
            if current == expected.encode() or (current is None and not expected):
                self.values[key] = desired.encode()
            return self.values.get(key, expected.encode())


@contextlib.contextmanager
def _deployment(world, **authority_options):
    """An authority process's loop on a thread and, per rank, an agent that answers every record (``_member``);
    yields the store and the agents. Ends as a deployment's normal exit does: rank 0 reports ``shutdown``, the agents
    stop once the authority has published the SHUTDOWN record, and the authority then returns that record."""
    store = SignedStore(DictStore(), b"join-secret")
    process = AuthorityProcess(world, store, Authority(world, **authority_options))
    returned = []
    thread = threading.Thread(
        target=lambda: returned.append(process.run(host="127.0.0.1")), daemon=True
    )
    thread.start()
    agents = []
    try:
        _until(lambda: store.store.check([membership.LIVENESS_KEY]))
        agents += [_member(store, r, world) for r in range(world)]
        _until(
            lambda: all(
                agent.record is not None and agent.record.state == MembershipState.COMMITTED
                for agent in agents
            )
        )
        yield store, agents
        agents[0].report(shutdown=True)
        _until(
            lambda: MembershipRecord.model_validate_json(store.get(RECORD_KEY)).state
            == MembershipState.SHUTDOWN
        )
        # The store lives in the authority's process: it serves on until every live rank has read the record and
        # left.
        thread.join(0.2)
        assert thread.is_alive()
    finally:
        _leave(agents)
        thread.join(10)
    assert [record.state for record in returned] == [MembershipState.SHUTDOWN]


def _until(condition, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def _member(store, rank, world, incarnation=0):
    """A rank's agent that replies to every record like an executor. A replacement reports that it is ready to join
    once it started, as its executor does."""
    agent = None

    def on_record(record):
        if record.state == MembershipState.PROPOSE:
            agent.report(
                reply=Reply.READY, reply_generation=record.generation, ready_payload=f"rank {rank}"
            )
        elif record.state == MembershipState.INSTALL:
            agent.report(reply=Reply.INSTALLED, reply_generation=record.generation)

    agent = MembershipAgent(rank, store, on_record, incarnation=incarnation)
    agent.start()
    if not incarnation:
        agent.report(started=True)
    if incarnation:
        agent.report(joining_ready=True, ready_payload=f"rank {rank}")
    return agent


def _refused(store, rank, incarnation):
    """Whether the authority closes the connection a hello for ``rank`` and ``incarnation`` opens."""
    with open_liveness_connection(store, rank, incarnation) as connection:
        connection.sock.settimeout(10)
        return connection.sock.recv(1 << 16) == b""


def _leave(agents):
    for agent in agents:
        if agent._thread.is_alive():
            agent.stop()


def test_a_stopped_agent_releases_every_file_descriptor_it_opened():
    """A process that starts and stops agents again and again (as replacements of one rank do) keeps its open file
    descriptors constant."""
    with _deployment(2) as (store, agents):
        agents[1].stop()
        open_fds = len(os.listdir("/proc/self/fd"))
        for incarnation in range(1, 21):
            agent = _member(store, 1, 2, incarnation=incarnation)
            _until(lambda: agent.record is not None and agent.record.is_member(1, incarnation))
            agent.stop()
        # The authority's thread closes its end of an agent's connection once it sees the connection end, and the
        # count above may include the end of agents[1]'s that it had not closed yet.
        _until(lambda: len(os.listdir("/proc/self/fd")) <= open_fds)
        agents[1] = _member(store, 1, 2, incarnation=21)
        _until(lambda: agents[1].record is not None and agents[1].record.is_member(1, 21))


def test_a_rank_whose_process_ends_is_removed_and_every_rank_hears_it_at_once():
    with _deployment(3) as (_, agents):
        agents[2]._sock.shutdown(socket.SHUT_RDWR)  # as when its process ends
        ended_s = time.monotonic()
        _until(
            lambda: all(
                agent.record.state == MembershipState.COMMITTED and agent.record.live_mask == 0b011
                for agent in agents[:2]
            )
        )
        assert time.monotonic() - ended_s < 1.0


def _committed_without_rank_1(agent):
    return agent.record.state == MembershipState.COMMITTED and agent.record.live_mask == 0b01


def test_only_a_valid_hello_of_a_process_not_connected_yet_opens_a_connection(monkeypatch):
    """Rank 1's process is gone, so a replacement's hello for it opens a connection, but only signed with the join
    secret for this connection: an unsigned hello, one tagged for another connection (a recorded one), a second
    connection of a connected process, or none at all are closed."""
    monkeypatch.setattr(membership, "HELLO_TIMEOUT_S", 0.5)
    with _deployment(2) as (store, agents):
        agents[1]._sock.shutdown(socket.SHUT_RDWR)
        _until(lambda: _committed_without_rank_1(agents[0]))
        host, port = store.get(membership.LIVENESS_KEY).decode().rsplit(":", 1)
        hello = b"1 1"
        for tag in (b"", store.tag(b"hello", bytes(16) + hello).hex().encode() + b" "):
            with socket.create_connection((host, int(port)), timeout=10) as forged:
                membership._read_line(forged)
                forged.sendall(tag + hello + b"\n")
                assert forged.recv(1) == b""
        assert _refused(store, 0, 0)
        with socket.create_connection((host, int(port)), timeout=10) as silent:
            membership._read_line(silent)
            assert silent.recv(1) == b""  # after HELLO_TIMEOUT_S
        assert agents[0].record.live_mask == 0b01
        agents[1] = _member(store, 1, 2, incarnation=1)  # the same hello, signed for its connection
        _until(lambda: agents[0].record.live_mask == 0b11)


def test_a_line_that_is_not_a_report_closes_the_connection():
    with _deployment(2) as (_, agents):
        agents[1]._sock.sendall(b'{"progress": "nine"}\n')
        _until(lambda: agents[0].record.live_mask == 0b01)


def test_a_deployment_at_rest_sends_and_writes_nothing():
    """While nothing changes, the authority sends no record and writes nothing to the store: the open connections
    are the liveness of the ranks and of the authority alike."""
    with _deployment(2) as (store, agents):
        agents[1].stop()
        _until(lambda: _committed_without_rank_1(agents[0]))
        writes = Mock(wraps=store.store.set)
        store.store.set = writes
        # A replacement that has not reported that it is ready to join: no transaction starts for it.
        with open_liveness_connection(store, 1, 1) as connection:
            connection.sock.settimeout(1.5)
            received = b""
            with contextlib.suppress(socket.timeout):
                while True:
                    data = connection.sock.recv(1 << 16)
                    assert data, "the authority closed the connection"
                    received += data
        assert len(connection.lines.decode(received)) == 1  # the record, with the hello's answer
        writes.assert_not_called()


def test_a_replacement_joins_only_once_the_process_it_replaces_is_gone():
    """Never for rank 0; a join for a rank whose process still serves would evict it."""
    with _deployment(3) as (store, agents):
        assert _refused(store, 2, 1)
        assert _refused(store, 0, 1)
        assert agents[0].record.live_mask == 0b111 and agents[2]._thread.is_alive()
        agents[2]._sock.shutdown(socket.SHUT_RDWR)
        _until(lambda: agents[0].record.live_mask == 0b011)
        agents.append(_member(store, 2, 3, incarnation=1))
        _until(
            lambda: agents[0].record.live_mask == 0b111
            and agents[0].record.state == MembershipState.COMMITTED
        )
        assert agents[0].record.incarnations == (0, 0, 1)


class FailingAuthority(Authority):
    """An authority whose decision raises once ``fail`` is set."""

    def __init__(self, world_size):
        super().__init__(world_size)
        self.fail = threading.Event()

    def tick(self, now_s, reports, connected):
        if self.fail.is_set():
            raise RuntimeError("a fault in the authority")
        return super().tick(now_s, reports, connected)


def test_an_authority_that_fails_ends_without_a_verdict():
    """No lease: its connections close and the ranks serve on unprotected; nothing tells them to abort."""
    store = SignedStore(DictStore(), b"join-secret")
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
    agents = [
        MembershipAgent(r, store, lambda record: None, on_error=errors.append) for r in range(2)
    ]
    for agent in agents:
        agent.start()
        agent.report(started=True)
    try:
        _until(
            lambda: all(
                agent.record is not None and agent.record.state == MembershipState.COMMITTED
                for agent in agents
            )
        )
        authority.fail.set()
        thread.join(10)
        assert len(raised) == 1
        _until(lambda: len(errors) == 2)
        assert all(agent.record.state == MembershipState.COMMITTED for agent in agents)
        assert (
            MembershipRecord.model_validate_json(store.get(RECORD_KEY)).state
            == MembershipState.COMMITTED
        )
    finally:
        _leave(agents)


def test_a_transaction_is_summarized_from_its_first_propose_to_its_commit():
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
    # A closed connection is proposed in the tick that sees it; the simulated ranks install in one step.
    assert summary.detected_unix_ms == summary.began_unix_ms
    assert summary.ended_unix_ms - summary.ready_unix_ms == pytest.approx(50.0)


def test_an_abort_and_a_rejoin_are_summarized():
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
    assert cluster.run(1.0).state == MembershipState.ABORT
    summary = cluster.authority.last_transaction
    assert (summary.generation, summary.outcome) == (record.generation, "ABORT")
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


def test_of_the_claims_that_replace_one_process_one_wins():
    """Two supervisors that saw incarnation 1 of rank 2 fail cannot both start a replacement for it; a claim from an
    incarnation that is no longer the last claimed one loses, and each rank has its own claims."""
    store = SignedStore(DictStore(), b"join-secret")
    assert claimed_incarnation(store, 2) == 0
    assert claim_incarnation(store, 2, 0) == 1
    assert claim_incarnation(store, 2, 0) is None
    assert [claim_incarnation(store, 2, 1), claim_incarnation(store, 2, 1)] == [2, None]
    assert claimed_incarnation(store, 2) == 2 and claimed_incarnation(store, 3) == 0
    assert claim_incarnation(store, 3, 0) == 1


def test_an_agent_stops_once_a_record_no_longer_lists_its_process_after_one_did(monkeypatch):
    """A replacement is listed by the PROPOSE that adds it; the change that restarts without it ends its membership,
    before any COMMITTED record admitted it."""
    monkeypatch.setattr(membership, "REPORT_INTERVAL_S", 60.0)
    records = []
    agent = MembershipAgent(1, SignedStore(None, b"secret"), records.append, incarnation=1)
    agent._sock, authority_side = _socket_pair()
    agent._lines = membership._Lines()
    agent._thread.start()
    try:
        unlisted = MembershipRecord(
            generation=1, state=MembershipState.COMMITTED, live_mask=0b01, incarnations=(0, 1)
        )
        adding = MembershipRecord(
            generation=2,
            state=MembershipState.PROPOSE,
            live_mask=0b11,
            incarnations=(0, 1),
            joining_mask=0b10,
        )
        left_out = MembershipRecord(
            generation=3, state=MembershipState.PROPOSE, live_mask=0b01, incarnations=(0, 1)
        )
        authority_side.sendall(membership._line(unlisted) + membership._line(adding))
        _until(lambda: len(records) == 2)
        assert agent._thread.is_alive()
        authority_side.sendall(membership._line(left_out))
        _until(lambda: not agent._thread.is_alive())
        assert records == [unlisted, adding, left_out]
    finally:
        agent.stop()
        authority_side.close()


class CountingAuthority(Authority):
    """An authority that counts its decisions."""

    def __init__(self, world_size):
        super().__init__(world_size)
        self.decisions = 0

    def tick(self, now_s, reports, connected):
        self.decisions += 1
        return super().tick(now_s, reports, connected)


def test_the_authority_decides_once_per_interval_and_at_once_on_a_reply(monkeypatch):
    """A report that only advances progress waits for the next decision; the start of the deployment and its shutdown
    are decided at once."""
    monkeypatch.setattr(membership, "REPORT_INTERVAL_S", 1.0)
    store = SignedStore(DictStore(), b"join-secret")
    authority = CountingAuthority(1)
    process = AuthorityProcess(1, store, authority)
    thread = threading.Thread(target=lambda: process.run(host="127.0.0.1"), daemon=True)
    thread.start()
    _until(lambda: store.store.check([membership.LIVENESS_KEY]))
    agent = MembershipAgent(0, store, lambda record: None)
    try:
        agent.start()
        reported_s = time.monotonic()
        agent.report(started=True)
        _until(lambda: agent.record is not None and agent.record.state == MembershipState.COMMITTED)
        assert time.monotonic() - reported_s < 0.5
        decisions = authority.decisions
        for progress in range(100):  # 100 reports in about 1 s
            agent.report(progress=progress)
            time.sleep(0.01)
        assert authority.decisions - decisions <= 4
        reported_s = time.monotonic()
        agent.report(shutdown=True)
        _until(lambda: agent.record.state == MembershipState.SHUTDOWN)
        assert time.monotonic() - reported_s < 0.5
    finally:
        _leave([agent])
        thread.join(10)


def test_connections_that_send_no_hello_are_capped(monkeypatch):
    """The oldest of the connections that wait for their hello is closed for a new one, so that connections that
    never send one cannot take every descriptor of the authority."""
    store = SignedStore(DictStore(), b"join-secret")
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
        with open_liveness_connection(store, 0, 0) as rank_0:  # a valid hello still opens one
            rank_0.sock.sendall(membership._line(RankReport(shutdown=True)))
            rank_0.sock.settimeout(10)
            assert membership._read_line(rank_0.sock)  # the record, at the hello
            assert b'"state":4' in membership._read_line(rank_0.sock)  # SHUTDOWN
        thread.join(10)
        assert not thread.is_alive()
    finally:
        for sock in silent:
            sock.close()

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
"""The WideEP fault-tolerance authority's decisions, driven by simulated rank reports (no GPU)."""

import pytest

from tensorrt_llm._torch.distributed.membership import (
    Authority,
    MembershipRecord,
    MembershipState,
    RankReport,
    Reply,
    ResumeState,
)

pytestmark = pytest.mark.cpu_only

WORLD = 4
ALL = (1 << WORLD) - 1


class Cluster:
    """Simulated ranks: each live process bumps its heartbeat and answers the authority's record."""

    def __init__(self, world=WORLD, covers=lambda mask: True, liveness_timeout_s=0.2, **timeouts):
        self.authority = Authority(world, covers, liveness_timeout_s=liveness_timeout_s, **timeouts)
        self.now = 0.0
        self.alive = set(range(world))  # processes that still run
        self.hung = set()  # alive but making no progress
        self.waiting_for = {}  # rank -> mask it waits for
        self.reports = {r: RankReport() for r in range(world)}
        self.stopped_at = {r: ResumeState(iteration=10 + r, epoch=100 + r) for r in range(world)}
        self.published = []

    def step(self, seconds=0.05):
        self.now += seconds
        record = self.authority.record
        for r in self.alive:
            old = self.reports[r]
            reply, generation = old.reply, old.reply_generation
            if r not in self.hung and record.is_live(r):
                if record.state == MembershipState.PROPOSE:
                    reply, generation = Reply.READY, record.generation
                elif record.state == MembershipState.INSTALL:
                    reply, generation = Reply.INSTALLED, record.generation
            self.reports[r] = RankReport(
                heartbeat=old.heartbeat + 1,
                step=old.step + (0 if r in self.hung else 1),
                waiting_for=self.waiting_for.get(r, 0),
                reply_generation=generation,
                reply=reply,
                stopped_at=self.stopped_at[r],
            )
        new = self.authority.tick(self.now, self.reports)
        if new is not None:
            self.published.append(new)
        return new

    def run(self, seconds):
        for _ in range(round(seconds / 0.05)):
            self.step()
        return self.authority.record


def test_healthy_ranks_change_nothing():
    cluster = Cluster()
    cluster.run(5.0)
    assert cluster.published == []
    assert cluster.authority.record == MembershipRecord(0, MembershipState.COMMITTED, ALL)


def test_a_dead_rank_is_removed_through_propose_install_commit():
    cluster = Cluster()
    cluster.alive.discard(2)
    record = cluster.run(2.0)
    states = [r.state for r in cluster.published]
    assert states == [MembershipState.PROPOSE, MembershipState.INSTALL, MembershipState.COMMITTED]
    assert record.generation == 1 and record.live_mask == ALL & ~(1 << 2)
    # Resume from one past the furthest survivor.
    assert record.resume.iteration == 13 and record.resume.epoch == 104


def test_a_slow_rank_is_kept_and_a_hung_one_is_removed():
    cluster = Cluster(progress_timeout_s=1.0)
    cluster.hung.add(1)
    cluster.run(3.0)  # nobody waits for rank 1: slow, not hung
    assert cluster.authority.record.live_mask == ALL
    cluster.waiting_for[0] = 1 << 1
    record = cluster.run(3.0)
    assert record.state == MembershipState.COMMITTED and not record.is_live(1)


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


def test_losing_an_expert_aborts():
    cluster = Cluster(covers=lambda mask: mask & 0b0100 != 0)  # only rank 2 holds some expert
    cluster.alive.discard(2)
    assert cluster.run(1.0).state == MembershipState.ABORT


def test_losing_rank_0_aborts():
    cluster = Cluster()
    cluster.alive.discard(0)
    assert cluster.run(1.0).state == MembershipState.ABORT


def test_a_transaction_past_its_deadline_aborts():
    cluster = Cluster(deadline_s=1.0, progress_timeout_s=100.0)
    cluster.alive.discard(3)
    cluster.hung.add(1)  # alive but never replies, and nobody waits on it
    assert cluster.run(3.0).state == MembershipState.ABORT


def test_a_replacement_joins_and_one_that_dies_while_joining_is_dropped():
    cluster = Cluster()
    cluster.alive.discard(2)
    cluster.run(2.0)
    cluster.alive.add(2)  # a new process for rank 2, with fresh reports
    cluster.reports[2] = RankReport()
    cluster.authority.request_join(2)
    record = cluster.run(2.0)
    assert (
        record.state == MembershipState.COMMITTED
        and record.live_mask == ALL
        and record.generation == 2
    )

    cluster.alive.discard(3)
    cluster.run(2.0)
    cluster.reports[3] = RankReport()
    cluster.authority.request_join(3)  # the replacement never starts its heartbeat
    record = cluster.run(2.0)
    assert record.state == MembershipState.COMMITTED and not record.is_live(3)


def test_a_replacement_for_a_rank_still_live_takes_it_over():
    cluster = Cluster()
    cluster.reports[1] = RankReport()  # the replacement's reports replace the old process's
    cluster.authority.request_join(1)
    record = cluster.run(1.0)
    assert [r.state for r in cluster.published] == [
        MembershipState.PROPOSE,
        MembershipState.INSTALL,
        MembershipState.COMMITTED,
    ]
    assert record.live_mask == ALL and record.generation == 1


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


def test_record_words_round_trip():
    record = MembershipRecord(
        7, MembershipState.INSTALL, (1 << 200) | 0b1011, ResumeState(1, 2, 3, 4, 5)
    )
    assert MembershipRecord.from_words(record.to_words()) == record
    report = RankReport(
        9, 8, 1 << 255, 7, Reply.INSTALLED, ResumeState(1, 2, 3, 4, 5), incarnation=2
    )
    assert RankReport.from_words(report.to_words()) == report


def test_shutdown_stops_failure_handling():
    cluster = Cluster()
    cluster.authority.shutdown()
    cluster.alive.clear()
    cluster.run(2.0)
    assert cluster.authority.record.state == MembershipState.SHUTDOWN

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
"""The authority's hang rule over every small deployment's failures (no GPU, no I/O).

A rank blocked in a collective waits on a peer, so only the end of a wait-for chain is hung. Each
case gives every rank one role and derives the waits that its reports carry:

- waiting: running and blocked, waits on the blockers, or on the waiting ranks that wait on them;
- slow: running and making progress, waits on nobody;
- idle: rank 0 waiting for requests with its hang detector paused;
- crashed: its connection closes and it reports nothing more;
- hung: its main thread stops outside any wait, so its fresh reports wait on nobody;
- stopped: its reports stop, and the last one waits on nobody or on the ranks that now wait on it.

The blockers are any nonempty set of non-waiting ranks. Ranks 1 and above are interchangeable to
the rule, so their roles, the blockers and the chained ranks are enumerated up to swapping ranks of
one role. The round evidence (``waited_flags``, ``waited_epochs``) and the install steps stay
empty, so a rank's progress is its iteration.
"""

import itertools

import pytest

from tensorrt_llm._torch.distributed.membership import (
    Authority,
    MembershipState,
    RankReport,
    Reply,
    authority,
)

pytestmark = pytest.mark.cpu_only

HANG_TIMEOUT_S = 1.0
STEP_S = HANG_TIMEOUT_S / 4
# Every case settles within four hang timeouts. The cases run one hang timeout longer.
SETTLE_S = 4 * HANG_TIMEOUT_S
TICKS = round((SETTLE_S + HANG_TIMEOUT_S) / STEP_S)

WAITING, SLOW, IDLE = "waiting", "slow", "idle"
CRASHED, HUNG, STOPPED = "crashed", "hung", "stopped"
FAULTY = (CRASHED, HUNG, STOPPED)
RANK_ROLES = (WAITING, SLOW, CRASHED, HUNG, STOPPED)

_reports = {}


def _report(**fields):
    """Return a cached report with ``fields``, since the cases send the same reports many times."""
    key = tuple(sorted(fields.items()))
    if key not in _reports:
        _reports[key] = RankReport(**fields)
    return _reports[key]


def _mask(ranks):
    return sum(1 << r for r in ranks)


def _groups(roles):
    """Return the ranks grouped by role, rank 0 on its own, since the other ranks are alike."""
    groups = [[0]]
    for r in range(1, len(roles)):
        if r > 1 and roles[r] == roles[r - 1]:
            groups[-1].append(r)
        else:
            groups.append([r])
    return groups


def _subsets(groups):
    """Yield one mask per subset of the ranks in ``groups``, up to swapping alike ranks."""
    for sizes in itertools.product(*(range(len(group) + 1) for group in groups)):
        yield _mask(r for group, size in zip(groups, sizes) for r in group[:size])


def _cases(world):
    """Yield ``(roles, waits, stale)`` for every role assignment and every wait they imply."""
    for rank0_role in RANK_ROLES + (IDLE,):
        for rest in itertools.combinations_with_replacement(RANK_ROLES, world - 1):
            roles = (rank0_role,) + rest
            groups = _groups(roles)
            waiting = [g for g in groups if roles[g[0]] == WAITING]
            others = [g for g in groups if roles[g[0]] != WAITING]
            if not waiting:
                yield from _variants(roles, {})
                continue
            # Running ranks never wait on each other in a cycle, because collectives send before
            # they wait.
            for blockers in filter(None, _subsets(others)):
                # The chained ranks wait on the other waiting ranks, not on the blockers.
                for chained in _subsets(waiting):
                    first_level = _mask(r for g in waiting for r in g) & ~chained
                    if first_level:
                        waits = {
                            r: first_level if chained >> r & 1 else blockers
                            for g in waiting
                            for r in g
                        }
                        yield from _variants(roles, waits)


def _variants(roles, waits):
    stopped = [r for r, role in enumerate(roles) if role == STOPPED]
    yield roles, waits, {}
    if stopped:
        # A stopped rank's last report waits on the ranks that now wait on it.
        yield roles, waits, {s: _mask(r for r in waits if waits[r] >> s & 1) for s in stopped}


def _run(world, roles, waits, stale):
    """Drive the authority through the case and return ``(time, record)`` for each record."""
    auth = Authority(world, hang_timeout_s=HANG_TIMEOUT_S)
    everyone = (1 << world) - 1
    started = {r: RankReport(started=True) for r in range(world)}
    assert auth.tick(0.0, started, everyone).state == MembershipState.COMMITTED
    connected = everyone & ~_mask(r for r in range(world) if roles[r] == CRASHED)
    published = []
    for tick in range(1, TICKS + 1):
        record = auth.record
        reports = {}
        for r, role in enumerate(roles):
            if role == CRASHED:
                continue
            if role == STOPPED:
                reports[r] = _report(received_s=STEP_S, waiting_for=stale.get(r, 0))
            elif role == HUNG:
                reports[r] = _report(received_s=tick * STEP_S)
            else:
                reports[r] = _running_report(record, role, waits.get(r, 0), tick)
        now_s = tick * STEP_S
        new = auth.tick(now_s, reports, connected)
        if new is not None:
            published.append((now_s, new))
        if new is not None and new.state == MembershipState.ABORTED:
            break
        # Once the running ranks commit without the failed ranks, they only make progress.
        settled = record.generation and record.state == MembershipState.COMMITTED
        if settled and now_s - published[-1][0] > HANG_TIMEOUT_S:
            break
    return published


def _running_report(record, role, waiting_for, tick):
    """Return a running rank's report, which replies to each change and resumes after the commit."""
    if record.generation:
        resumed = record.state == MembershipState.COMMITTED
        reply = {
            MembershipState.PROPOSE: Reply.READY,
            MembershipState.INSTALL: Reply.INSTALLED,
        }.get(record.state, Reply.NONE)
        return _report(
            received_s=tick * STEP_S,
            progress=tick if resumed else 0,
            reply=reply,
            reply_generation=record.generation,
        )
    if role == SLOW:
        return _report(received_s=tick * STEP_S, progress=tick)
    return _report(received_s=tick * STEP_S, idle=role == IDLE, waiting_for=waiting_for)


def _mismatch(world, roles, waits, stale, published):
    """Return why ``published`` breaks the hang rule for the case, or None."""
    everyone = (1 << world) - 1
    running = _mask(r for r in range(world) if roles[r] not in FAULTY)
    crashed = _mask(r for r in range(world) if roles[r] == CRASHED)
    waited_on = 0
    for mask in waits.values():
        waited_on |= mask
    # The ends of the wait-for chains: the hung or stopped ranks that a waiting rank waits on.
    roots = waited_on & everyone & ~running & ~crashed
    if any(record.active_mask & running != running for _, record in published):
        return "removed a running rank"
    if not crashed and not roots:
        return "changed membership without a failure" if published else None
    if not published:
        return "missed the failure"
    first_s, first = published[0]
    removed = everyone & ~first.active_mask
    expected = crashed or roots
    if removed != expected:
        return f"first removed {removed:#b}, expected {expected:#b}"
    # The failures start with the first reports, one step in.
    if crashed:
        in_time = first_s == STEP_S
    else:
        in_time = HANG_TIMEOUT_S < first_s - STEP_S <= HANG_TIMEOUT_S + 2 * STEP_S
    if not in_time:
        return f"first change at {first_s} s"
    last_s, last = published[-1]
    if last_s > SETTLE_S:
        return f"still changing at {last_s} s"
    if running & 1:
        if last.state != MembershipState.COMMITTED or last.active_mask != running:
            return f"settled on {last}"
    elif last.state != MembershipState.ABORTED:
        return f"did not abort without rank 0: {last}"
    return None


@pytest.mark.parametrize("world", [2, 3, 4, 5])
def test_only_chain_ends_are_hung(world):
    """Remove exactly the crashed ranks or the ends of the wait-for chains, then every failed rank.

    A crash starts a change at once. Without one, the authority removes the hung or stopped ranks
    that waiting ranks wait on, one hang timeout after they stop. The change then waits on every
    other failed rank, which never replies, so the deployment settles on the running ranks, or
    aborts without rank 0. A failed rank that no running rank waits on is slow, not hung.
    """
    mismatches = []
    count = 0
    for roles, waits, stale in _cases(world):
        count += 1
        problem = _mismatch(world, roles, waits, stale, _run(world, roles, waits, stale))
        if problem is not None:
            mismatches.append(f"roles {roles}, waits {waits}, stale {stale}: {problem}")
    assert count > 0
    assert not mismatches, f"{len(mismatches)} of {count} cases: " + "\n".join(mismatches[:20])


@pytest.mark.parametrize("world", [2, 3, 4])
def test_chain_end_matches_reachability(world):
    """Find the chain ends of every wait-for graph, cycles included.

    A rank is a chain end when every rank it reaches through the waits reaches it back.
    """
    pairs = [(r, q) for r in range(world) for q in range(world) if q != r]
    for edges in itertools.product((False, True), repeat=len(pairs)):
        waits = {r: 0 for r in range(world)}
        for (r, q), edge in zip(pairs, edges):
            waits[r] |= edge << q
        reach = dict(waits)
        for _ in range(world):
            for r in range(world):
                for q in range(world):
                    if reach[r] >> q & 1:
                        reach[r] |= reach[q]
        for r in range(world):
            expected = all(reach[q] >> r & 1 for q in range(world) if reach[r] >> q & 1)
            assert authority._chain_end(r, waits) == expected, (r, waits)


def test_hang_timer_restarts_after_stall_ends():
    """Time a hang from the start of the latest stall, not from an earlier stall that ended."""
    world, hung = 4, 2
    auth = Authority(world, hang_timeout_s=HANG_TIMEOUT_S)
    everyone = (1 << world) - 1
    started = {r: RankReport(started=True) for r in range(world)}
    assert auth.tick(0.0, started, everyone).state == MembershipState.COMMITTED
    # Rank 2 makes no progress, and the others wait on it twice, first for less than the timeout.
    first_stall, second_stall = range(1, 4), 6
    for tick in range(1, TICKS + 1):
        waiting_for = 1 << hung if tick in first_stall or tick >= second_stall else 0
        reports = {
            r: _report(received_s=tick * STEP_S, waiting_for=waiting_for) for r in range(world)
        }
        reports[hung] = _report(received_s=tick * STEP_S)
        record = auth.tick(tick * STEP_S, reports, everyone)
        if record is not None:
            break
    assert record is not None
    assert record.active_mask == everyone & ~(1 << hung)
    stalled_s = tick * STEP_S - second_stall * STEP_S
    assert HANG_TIMEOUT_S < stalled_s <= HANG_TIMEOUT_S + 2 * STEP_S

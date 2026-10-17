# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hang confirmation from worker heartbeats and device waits, and the authority acting on it."""

import pytest

from tensorrt_llm.executor.recovery import authority as authority_module
from tensorrt_llm.executor.recovery import protocol
from tensorrt_llm.executor.recovery.authority import Phase, RecoveryAuthority
from tensorrt_llm.executor.recovery.hang_monitor import HangMonitor, HungRank
from tensorrt_llm.executor.recovery.protocol import DeviceWait, Fence, Heartbeat, Register, Stopped

pytestmark = pytest.mark.cpu_only

RANKS = (0, 1, 2, 3)
CONFIRM, HB_TIMEOUT = 10.0, 3.0


def _monitor() -> HangMonitor:
    return HangMonitor(RANKS, confirm_s=CONFIRM, heartbeat_timeout_s=HB_TIMEOUT)


def _beat(monitor, now, iterations=None, waiting=()):
    for rank in RANKS:
        iteration = (iterations or {}).get(rank, int(now))
        monitor.heartbeat(rank, iteration, rank in waiting, now)


def test_healthy_progress_and_idle_waiting_are_never_hung():
    monitor = _monitor()
    for now in range(0, 60):
        _beat(monitor, float(now))
        assert monitor.check(float(now)) is None
    # Idle: no iteration advances, every rank waits (rank 0 for requests, the others in a collective).
    for now in range(60, 120):
        _beat(monitor, float(now), iterations={r: 59 for r in RANKS}, waiting=RANKS)
        assert monitor.check(float(now)) is None


def test_losing_every_heartbeat_fences_nobody():
    """The authority cut off from the workers (or stalled itself): every heartbeat stops at once."""
    monitor = _monitor()
    _beat(monitor, 0.0)
    for now in range(1, 60):
        assert monitor.check(float(now)) is None


def test_ranks_without_heartbeats_are_not_watched():
    monitor = _monitor()
    assert monitor.check(1000.0) is None


def test_missing_heartbeat_is_confirmed_after_the_confirm_window():
    monitor = _monitor()
    _beat(monitor, 0.0)
    hung = None
    now = 0.0
    while hung is None and now < 100:
        now += 1.0
        for rank in (0, 1, 3):
            monitor.heartbeat(rank, int(now), False, now)
        hung = monitor.check(now)
    assert hung == HungRank(2, hung.reason) and "no heartbeat" in hung.reason
    # Suspected once the heartbeat timeout passes, confirmed CONFIRM later.
    assert now == pytest.approx(HB_TIMEOUT + 1 + CONFIRM)
    assert monitor.check(now + 1) is None  # confirmed once


def test_a_slow_rank_that_catches_up_is_not_hung():
    monitor = _monitor()
    for now in range(0, 40):
        t = float(now)
        # Rank 2 stalls for CONFIRM - 1 seconds while its peers wait in a collective, then resumes.
        stalled = 10 <= now < 10 + CONFIRM - 1
        iterations = {r: (10 if r == 2 and stalled else now) for r in RANKS}
        _beat(monitor, t, iterations=iterations, waiting=(0, 1, 3) if stalled else ())
        assert monitor.check(t) is None, now


def test_stuck_host_loop_while_peers_wait_is_hung():
    monitor = _monitor()
    hung = None
    for now in range(0, 40):
        iterations = {r: 5 for r in RANKS}
        _beat(monitor, float(now), iterations=iterations, waiting=(0, 1, 3))
        hung = hung or monitor.check(float(now))
    assert hung is not None and hung.rank == 2 and "loop stopped" in hung.reason


def test_device_waits_need_two_witnesses_and_point_at_the_chain_head():
    monitor = _monitor()
    _beat(monitor, 0.0)
    # Only rank 1 waits on rank 2: not enough evidence.
    monitor.device_wait(1, 2, 1.0, 0.0)
    for now in range(1, 30):
        _beat(monitor, float(now))
        assert monitor.check(float(now)) is None
    # Ranks 0 and 1 wait on rank 3, which itself waits on rank 2; rank 3 is a victim, not the head.
    monitor = _monitor()
    _beat(monitor, 0.0)
    for waiter in (0, 1):
        monitor.device_wait(waiter, 3, 1.0, 0.0)
    monitor.device_wait(3, 2, 1.0, 0.0)
    monitor.device_wait(1, 2, 1.0, 0.0)
    hung = None
    for now in range(1, 30):
        # Ranks 2 and 3 are stuck in a step; their iterations stop.
        _beat(monitor, float(now), iterations={2: 0, 3: 0})
        hung = hung or monitor.check(float(now))
    assert hung is not None and hung.rank == 2


def test_device_waits_on_a_rank_that_keeps_stepping_are_not_a_hang():
    monitor = _monitor()
    _beat(monitor, 0.0)
    for waiter in (0, 1):
        monitor.device_wait(waiter, 2, 1.0, 0.0)
    for now in range(1, 40):
        _beat(monitor, float(now))  # rank 2's iteration still advances: a long step, not a hang
        assert monitor.check(float(now)) is None


def test_a_starved_heartbeat_of_a_waiting_victim_points_at_the_hung_rank():
    monitor = _monitor()
    _beat(monitor, 0.0, iterations={r: 0 for r in RANKS})
    for waiter in (0, 1):
        monitor.device_wait(waiter, 3, 1.0, 0.0)
    hung = None
    for now in range(1, 40):
        # Rank 3's device is stuck; rank 1's heartbeat thread is starved and goes silent.
        for rank in (0, 2, 3):
            monitor.heartbeat(rank, 0, False, float(now))
        hung = hung or monitor.check(float(now))
    assert hung is not None and hung.rank == 3


def test_waits_on_unknown_peers_are_ignored():
    monitor = _monitor()
    _beat(monitor, 0.0)
    monitor.device_wait(0, 7, 1.0, 0.0)
    monitor.device_wait(1, 7, 1.0, 0.0)
    for now in range(1, 40):
        _beat(monitor, float(now))
        assert monitor.check(float(now)) is None


def test_cleared_device_waits_drop_the_evidence():
    monitor = _monitor()
    _beat(monitor, 0.0)
    for waiter in (0, 1):
        monitor.device_wait(waiter, 2, 1.0, 0.0)
    for now in range(1, 5):
        _beat(monitor, float(now))
        assert monitor.check(float(now)) is None
    for waiter in (0, 1):
        monitor.device_wait(waiter, 2, 0.0, 5.0)
    for now in range(5, 40):
        _beat(monitor, float(now))
        assert monitor.check(float(now)) is None


def _serving_authority(fenced):
    authority = RecoveryAuthority(
        len(RANKS),
        hang_monitor=_monitor(),
        fence=lambda hung: fenced.append((hung.rank, hung.reason, hung.host, hung.pid)),
    )
    for rank in RANKS:
        authority.handle(f"rank-{rank}".encode(), Register(rank, ((1,), (2,), (3,))), 0.0)
    assert authority.phase is Phase.SERVING
    return authority


def _hang_rank_2(authority, until, stop_at=None):
    """Ranks 0, 1, 3 beat and progress; rank 2 beats once, then stops. Returns the outbound Fences by time."""
    fences = {}
    for now in range(1, until):
        for rank in (0, 1, 3):
            # Heartbeats come from each worker's own evidence connection, not its protocol route.
            assert (
                authority.handle(
                    f"evidence-{rank}".encode(),
                    Heartbeat(rank, now, False, "node", 100 + rank),
                    float(now),
                )
                == []
            )
        if now == 1:
            authority.handle(b"evidence-2", Heartbeat(2, 0, False, "node", 102), 1.0)
        if now == stop_at:
            authority.handle(b"rank-0", Stopped(0, (2,)), float(now))
        outbound = authority.expire(float(now))
        if outbound:
            fences[now] = outbound
    return fences


def test_authority_fences_a_hung_rank_through_itself_and_a_node_mate_then_the_command():
    fenced = []
    authority = _serving_authority(fenced)
    fences = _hang_rank_2(authority, 60)
    # Once, at confirmation: to the hung rank's evidence thread and to its lowest node-mate.
    ((confirmed, outbound),) = fences.items()
    assert outbound == [(b"evidence-2", Fence(2, 102)), (b"evidence-0", Fence(2, 102))]
    # The workers could not end it (no recovery started): the fence command, once, after the grace.
    assert [(rank, host, pid) for rank, _, host, pid in fenced] == [(2, "node", 102)]


def test_the_fence_command_is_skipped_once_the_hung_rank_s_loss_starts_a_recovery():
    fenced = []
    authority = _serving_authority(fenced)
    confirmed = next(iter(_hang_rank_2(_serving_authority([]), 60)))
    _hang_rank_2(authority, confirmed + 1)
    authority.handle(b"rank-0", Stopped(0, (2,)), float(confirmed + 1))
    for now in range(confirmed + 1, confirmed + 10):
        authority.expire(float(now))
    assert fenced == []


def test_command_fence_fills_host_and_pid_and_runs_without_a_shell():
    launched = []
    original = authority_module.subprocess.Popen
    authority_module.subprocess.Popen = lambda command, **kwargs: launched.append(command)
    try:
        fence = authority_module.command_fence("ssh {host} kill -KILL {pid}")
        fence(HungRank(2, "no heartbeat for 30.0s", "node-7", 4242))
    finally:
        authority_module.subprocess.Popen = original
    assert launched == [["ssh", "node-7", "kill", "-KILL", "4242"]]


def test_authority_does_not_fence_rank_zero_and_ignores_evidence_during_recovery():
    fenced = []
    authority = _serving_authority(fenced)
    authority.handle(b"heartbeat", Heartbeat(0, 0, False, "node", 100), 0.0)
    for now in range(1, 40):
        for rank in (1, 2, 3):
            authority.handle(
                b"heartbeat", Heartbeat(rank, now, False, "node", 100 + rank), float(now)
            )
        authority.expire(float(now))
    assert fenced == []  # rank 0's loss is never recovered: reported only

    fenced = []
    authority = _serving_authority(fenced)
    authority.handle(b"rank-0", Stopped(0, (2,)), 1.0)
    assert authority.phase is Phase.STOPPING
    for now in range(1, 40):
        authority.handle(b"heartbeat", DeviceWait(1, 3, 5000), float(now))
        authority.expire(min(float(now), 10.0))
    assert fenced == []


def test_after_a_recovery_a_hung_rank_is_only_reported():
    fenced = []
    authority = _serving_authority(fenced)
    authority.phase = Phase.RECOVERED  # one loss per job: a second one cannot be recovered
    authority.handle(b"heartbeat", Heartbeat(2, 0, False, "node", 102), 0.0)
    for now in range(1, 40):
        for rank in (0, 1, 3):
            authority.handle(
                b"heartbeat", Heartbeat(rank, now, False, "node", 100 + rank), float(now)
            )
        authority.expire(float(now))
    assert fenced == []


def test_evidence_messages_round_trip():
    key, route = b"k" * 32, b"rank-3-heartbeat"
    for message in (Heartbeat(3, 1234, True, "node-3", 4321), DeviceWait(1, 2, 2500)):
        frames = protocol.encode(message, key, route)
        assert protocol.authenticate(frames, key, route)
        assert protocol.decode(frames[0]) == message

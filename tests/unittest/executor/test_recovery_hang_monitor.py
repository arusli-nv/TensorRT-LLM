# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hang and stall confirmation from worker heartbeats and device waits, and the authority acting on it."""

import hashlib
import hmac
import pickle  # nosec B403
import socket

import pytest

from tensorrt_llm.executor.recovery import authority_server, protocol
from tensorrt_llm.executor.recovery.authority import Phase, RecoveryAuthority
from tensorrt_llm.executor.recovery.config import FaultToleranceConfig
from tensorrt_llm.executor.recovery.hang_monitor import HangMonitor, HungRank
from tensorrt_llm.executor.recovery.protocol import (
    Abort,
    DeviceWait,
    Fence,
    Fenced,
    Heartbeat,
    NativeHeartbeat,
    Propose,
    Register,
    Restart,
    Sent,
    Stopped,
)

pytestmark = pytest.mark.cpu_only

RANKS = (0, 1, 2, 3)
PEERS = (0, 1, 3)
DIGEST = FaultToleranceConfig().digest()
CONFIRM, HB_TIMEOUT = 10.0, 3.0
KEY = b"k" * 32


def _monitor() -> HangMonitor:
    return HangMonitor(RANKS, confirm_s=CONFIRM, heartbeat_timeout_s=HB_TIMEOUT)


def _beat(monitor, now, iterations=None, waiting=(), ranks=RANKS):
    for rank in ranks:
        iteration = (iterations or {}).get(rank, int(now))
        monitor.heartbeat(rank, iteration, rank in waiting, now)


def _wait_on(monitor, waiters, peer, now):
    for waiter in waiters:
        monitor.device_wait(waiter, peer, 1.0, now)


def _first_confirmation(monitor, step, until=100):
    """Call ``step(monitor, now)`` each second; return (now, hung) at the first confirmation, else (until, None)."""
    for now in range(1, until):
        step(monitor, float(now))
        assert monitor.all_stalled(float(now)) is None, now
        hung = monitor.check(float(now))
        if hung is not None:
            return float(now), hung
    return float(until), None


def test_healthy_progress_and_idle_waiting_are_never_hung():
    monitor = _monitor()
    for now in range(0, 60):
        _beat(monitor, float(now))
        assert monitor.check(float(now)) is None and monitor.all_stalled(float(now)) is None
    # Idle: no iteration advances, every rank waits (rank 0 for requests, the others in a collective).
    for now in range(60, 120):
        _beat(monitor, float(now), iterations={r: 59 for r in RANKS}, waiting=RANKS)
        assert monitor.check(float(now)) is None and monitor.all_stalled(float(now)) is None


def test_losing_every_heartbeat_fences_nobody():
    """The authority cut off from the workers (or stalled itself): every heartbeat stops at once."""
    monitor = _monitor()
    _beat(monitor, 0.0)
    _wait_on(monitor, PEERS, 2, 0.0)
    for now in range(1, 60):
        assert monitor.check(float(now)) is None and monitor.all_stalled(float(now)) is None


def test_ranks_without_heartbeats_are_not_watched():
    monitor = _monitor()
    assert monitor.check(1000.0) is None


def test_every_peer_waiting_on_a_rank_without_progress_is_fenced_after_the_window():
    """Rank 2's device is stuck: its heartbeats keep coming, its iteration does not move, every peer waits on it."""
    monitor = _monitor()
    _beat(monitor, 0.0)
    _wait_on(monitor, PEERS, 2, 0.0)
    now, hung = _first_confirmation(
        monitor, lambda m, t: _beat(m, t, iterations={r: 0 for r in RANKS})
    )
    assert hung == HungRank(2, hung.reason) and "every peer waits on it" in hung.reason
    assert now == CONFIRM + 1  # suspected at the first check, confirmed CONFIRM later
    assert monitor.check(now + 1) is None  # confirmed once


def test_any_progress_restarts_the_window():
    """Rank 2 finishes a step just before each window would close; every peer waits on it the whole time."""
    monitor = _monitor()
    _beat(monitor, 0.0)
    _wait_on(monitor, PEERS, 2, 0.0)
    last_step = 0

    def step(m, now):
        nonlocal last_step
        if now < 50 and now - last_step == CONFIRM - 1:
            last_step = int(now)
        _beat(m, now, iterations={0: 0, 1: 0, 2: last_step, 3: 0})

    now, hung = _first_confirmation(monitor, step)
    assert hung is not None and hung.rank == 2
    # Confirmed only CONFIRM after its last step.
    assert now == last_step + CONFIRM


def test_not_every_peer_waiting_is_no_fence():
    monitor = _monitor()
    _beat(monitor, 0.0)
    # Ranks 0 and 1 wait on rank 2's device; rank 3 still steps and waits on nothing.
    _wait_on(monitor, (0, 1), 2, 0.0)
    _, hung = _first_confirmation(
        monitor, lambda m, t: _beat(m, t, iterations={0: 0, 1: 0, 2: 0}), until=60
    )
    assert hung is None


def test_peers_waiting_in_a_collective_on_a_stopped_rank_fence_it():
    """Rank 2 is stopped between steps: its peers wait in a host collective, its heartbeats stop."""
    monitor = _monitor()
    # Its last heartbeat said it waited too: it is not trusted once its heartbeat is missing.
    _beat(monitor, 0.0, waiting=RANKS)
    now, hung = _first_confirmation(
        monitor,
        lambda m, t: _beat(m, t, iterations={r: 0 for r in RANKS}, waiting=RANKS, ranks=PEERS),
    )
    assert hung is not None and hung.rank == 2
    # Missing after HB_TIMEOUT, confirmed CONFIRM later.
    assert now == pytest.approx(HB_TIMEOUT + 1 + CONFIRM)


def test_a_stuck_host_loop_while_every_peer_waits_on_the_host_is_fenced():
    monitor = _monitor()
    now, hung = _first_confirmation(
        monitor, lambda m, t: _beat(m, t, iterations={r: 5 for r in RANKS}, waiting=PEERS)
    )
    assert hung is not None and hung.rank == 2 and now == CONFIRM + 1


def test_a_waiting_rank_is_not_blamed_for_its_peers_host_waits():
    """Every rank waits on the host and none steps (an idle deployment, or a host collective deadlock)."""
    monitor = _monitor()
    _, hung = _first_confirmation(
        monitor, lambda m, t: _beat(m, t, iterations={r: 5 for r in RANKS}, waiting=RANKS), until=60
    )
    assert hung is None


def test_the_head_of_a_chain_is_fenced_not_its_victims():
    monitor = _monitor()
    _beat(monitor, 0.0)
    # Rank 3's device waits on rank 2; ranks 0 and 1 wait on both.
    _wait_on(monitor, (0, 1), 3, 0.0)
    _wait_on(monitor, (0, 1, 3), 2, 0.0)
    _, hung = _first_confirmation(
        monitor, lambda m, t: _beat(m, t, iterations={r: 0 for r in RANKS})
    )
    assert hung is not None and hung.rank == 2


def test_a_rank_that_keeps_stepping_is_never_fenced():
    monitor = _monitor()
    _beat(monitor, 0.0)
    _wait_on(monitor, PEERS, 2, 0.0)
    _, hung = _first_confirmation(
        monitor, lambda m, t: _beat(m, t, iterations={0: 0, 1: 0, 3: 0}), until=60
    )
    assert hung is None


def test_native_heartbeats_keep_a_starved_rank_alive_while_it_steps():
    """Rank 2's Python threads are starved: only its native heartbeat arrives, with the iteration it reached."""
    monitor = _monitor()
    _beat(monitor, 0.0, waiting=PEERS)

    def starved(m, now):
        _beat(m, now, iterations={r: 0 for r in RANKS}, waiting=PEERS, ranks=PEERS)
        m.heartbeat(2, int(now) // 5, None, now)  # a step every 5 s

    _, hung = _first_confirmation(monitor, starved, until=60)
    assert hung is None


def test_a_starved_rank_that_completes_no_step_for_the_window_is_fenced():
    monitor = _monitor()
    _beat(monitor, 0.0)

    def starved(m, now):
        _beat(m, now, iterations={r: 0 for r in RANKS}, waiting=PEERS, ranks=PEERS)
        m.heartbeat(2, 0, None, now)

    now, hung = _first_confirmation(monitor, starved)
    assert hung is not None and hung.rank == 2 and now == CONFIRM + 1


def test_every_rank_stalled_on_the_device_is_dead_and_fences_no_rank():
    """A deadlock or desync every rank shares: each device waits on the next rank, no iteration advances."""
    monitor = _monitor()
    _beat(monitor, 0.0)
    for rank in RANKS:
        monitor.device_wait(rank, (rank + 1) % len(RANKS), 1.0, 0.0)
    for now in range(1, int(CONFIRM) - 1):
        _beat(monitor, float(now), iterations={r: 0 for r in RANKS})
        assert monitor.all_stalled(float(now)) is None and monitor.check(float(now)) is None
    now = CONFIRM + 1
    _beat(monitor, now, iterations={r: 0 for r in RANKS})
    assert monitor.check(now) is None
    assert "no single rank to blame" in monitor.all_stalled(now)
    # A rank whose device stopped waiting (it got through) means the deployment still moves.
    monitor.device_wait(2, 3, 0.0, now + 1)
    assert monitor.all_stalled(now + 1) is None


def test_progress_or_a_single_culprit_is_not_a_stall():
    monitor = _monitor()
    _beat(monitor, 0.0)
    for rank in RANKS:
        monitor.device_wait(rank, (rank + 1) % len(RANKS), 1.0, 0.0)
    _beat(monitor, CONFIRM + 1, iterations={0: 0, 1: 0, 3: 0})  # rank 2 stepped
    assert monitor.all_stalled(CONFIRM + 1) is None
    # Every peer waits on stopped rank 2: fence it rather than restart the deployment.
    monitor = _monitor()
    _beat(monitor, 0.0)
    _wait_on(monitor, PEERS, 2, 0.0)
    for now in range(1, 30):
        _beat(monitor, float(now), iterations={r: 0 for r in RANKS}, ranks=PEERS)
        assert monitor.all_stalled(float(now)) is None


def test_an_all_rank_host_stall_is_neither_a_hang_nor_dead():
    """Every serving loop sleeps at once: no progress, no waits on the host or the device."""
    monitor = _monitor()
    _, hung = _first_confirmation(
        monitor, lambda m, t: _beat(m, t, iterations={r: 0 for r in RANKS}), until=60
    )
    assert hung is None


def test_waits_on_unknown_peers_are_ignored():
    monitor = _monitor()
    _beat(monitor, 0.0)
    _wait_on(monitor, (0, 1, 2, 3), 7, 0.0)
    _, hung = _first_confirmation(monitor, lambda m, t: _beat(m, t, iterations={}), until=40)
    assert hung is None


def test_cleared_device_waits_drop_the_evidence():
    monitor = _monitor()
    _beat(monitor, 0.0)
    _wait_on(monitor, PEERS, 2, 0.0)
    for now in range(1, 5):
        _beat(monitor, float(now), iterations={r: 0 for r in RANKS})
        assert monitor.check(float(now)) is None
    for waiter in PEERS:
        monitor.device_wait(waiter, 2, 0.0, 5.0)
    for now in range(5, 40):
        _beat(monitor, float(now), iterations={r: 0 for r in RANKS})
        assert monitor.check(float(now)) is None


def test_a_rank_that_stops_before_its_first_heartbeat_after_a_reset_is_fenced():
    """A recovery resets the evidence; a rank that stops right after it is still watched, with its process."""
    monitor = _monitor()
    for rank in RANKS:
        monitor.heartbeat(rank, 0, False, -10.0, "node", 100 + rank)
    monitor.reset(0.0)
    # Rank 2 sends nothing more; its peers wait for it in a collective.
    now, hung = _first_confirmation(
        monitor,
        lambda m, t: _beat(m, t, iterations={r: 5 for r in RANKS}, waiting=PEERS, ranks=PEERS),
    )
    assert hung == HungRank(2, hung.reason, "node", 102)
    assert now == CONFIRM + 1


def _serving_authority(fenced):
    authority = RecoveryAuthority(
        len(RANKS),
        hang_monitor=_monitor(),
        fence=lambda hung: fenced.append((hung.rank, hung.reason, hung.host, hung.pid)),
    )
    for rank in RANKS:
        authority.handle(f"rank-{rank}".encode(), Register(rank, ((1,), (2,), (3,)), DIGEST), 0.0)
    assert authority.phase is Phase.SERVING
    return authority


def _hang_rank_2(authority, until, stop_at=None):
    """Rank 2 beats once, then stops; its peers wait for it in a collective. Returns the outbound by time."""
    sent = {}
    for now in range(1, until):
        for rank in PEERS:
            # Heartbeats come from each worker's own evidence connection, not its protocol route.
            assert (
                authority.handle(
                    f"evidence-{rank}".encode(),
                    Heartbeat(rank, 0, True, "node", 100 + rank),
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
            sent[now] = outbound
    return sent


def test_authority_fences_a_hung_rank_through_itself_and_another_rank_then_the_command():
    fenced = []
    authority = _serving_authority(fenced)
    sent = _hang_rank_2(authority, 60)
    # Once, at confirmation: to the hung rank's evidence thread and to the two lowest live, unsuspected ranks
    # (either asks the runtime).
    ((_, outbound),) = sent.items()
    assert outbound == [
        (b"evidence-2", Fence(2, 102)),
        (b"evidence-0", Fence(2, 102)),
        (b"evidence-1", Fence(2, 102)),
    ]
    # The workers could not end it (no recovery started): the fence command, once, after the grace.
    assert [(rank, host, pid) for rank, _, host, pid in fenced] == [(2, "node", 102)]
    assert authority.phase is Phase.SERVING


def test_a_rank_the_fence_command_ran_for_is_fenced_by_rank_0_before_the_proposal():
    """The command's outcome is not observed, so it may have failed: rank 0 ends the rank before any plan."""
    fenced = []
    authority = _serving_authority(fenced)
    _hang_rank_2(authority, 60)
    assert [rank for rank, *_ in fenced] == [2]
    for rank in PEERS:
        out = authority.handle(f"rank-{rank}".encode(), Stopped(rank, (2,)), 60.0)
    assert out == [(b"rank-0", Fence(2, 102))]
    out = authority.handle(b"rank-0", Fenced(0, 2, ""), 60.5)
    assert out == [(f"rank-{rank}".encode(), Propose(1, (2,), PEERS)) for rank in PEERS]


def test_the_fence_command_is_skipped_once_the_hung_rank_s_loss_starts_a_recovery():
    fenced = []
    authority = _serving_authority(fenced)
    confirmed = next(iter(_hang_rank_2(_serving_authority([]), 60)))
    _hang_rank_2(authority, confirmed + 1)
    authority.handle(b"rank-0", Stopped(0, (2,)), float(confirmed + 1))
    for now in range(confirmed + 1, confirmed + 10):
        authority.expire(float(now))
    assert fenced == []


def test_authority_reports_a_stall_of_every_rank_as_terminal_and_restarts_every_worker():
    fenced = []
    authority = _serving_authority(fenced)
    sent = {}
    for now in range(0, 40):
        for rank in RANKS:
            route = f"evidence-{rank}".encode()
            authority.handle(route, Heartbeat(rank, 7, False, "node", 100 + rank), float(now))
            if now == 0:
                authority.handle(route, DeviceWait(rank, (rank + 1) % len(RANKS), 2000), 0.0)
        outbound = authority.expire(float(now))
        if outbound:
            sent[now] = outbound
    ((_, outbound),) = sent.items()
    assert authority.phase is Phase.ABORTED and "every rank stalled" in authority.abort_reason
    aborts = [route for route, message in outbound if isinstance(message, Abort)]
    restarts = [route for route, message in outbound if isinstance(message, Restart)]
    assert sorted(aborts) == [f"rank-{rank}".encode() for rank in RANKS]
    assert sorted(restarts) == [f"evidence-{rank}".encode() for rank in RANKS]
    assert fenced == [] and authority.status().phase == "ABORTED"


def test_command_fence_fills_host_and_pid_and_runs_without_a_shell():
    launched = []
    original = authority_server.subprocess.Popen
    authority_server.subprocess.Popen = lambda command, **kwargs: launched.append(command)
    try:
        fence = authority_server.command_fence("ssh {host} kill -KILL {pid}")
        fence(HungRank(2, "every peer waits on it", "node-7", 4242))
    finally:
        authority_server.subprocess.Popen = original
    assert launched == [["ssh", "node-7", "kill", "-KILL", "4242"]]


@pytest.mark.parametrize(
    "host, pid",
    [("-oProxyCommand=x", 4242), ("node 7", 4242), ("node;7", 4242), ("", 4242), ("node-7", "-1")],
)
def test_command_fence_refuses_a_host_or_pid_that_is_not_plain(host, pid, monkeypatch):
    launched = []
    monkeypatch.setattr(
        authority_server.subprocess, "Popen", lambda command, **kwargs: launched.append(command)
    )
    authority_server.command_fence("ssh {host} kill -KILL {pid}")(HungRank(2, "hung", host, pid))
    assert launched == []
    authority_server.command_fence("ssh {host} kill -KILL {pid}")(HungRank(2, "hung", "fd00::7", 9))
    assert launched == [["ssh", "fd00::7", "kill", "-KILL", "9"]]


def test_authority_ends_the_deployment_when_rank_zero_hangs():
    """Rank 0's loss is never recovered: its confirmed hang is terminal and every worker exits for a restart."""
    fenced = []
    authority = _serving_authority(fenced)
    authority.handle(b"heartbeat-0", Heartbeat(0, 0, False, "node", 100), 0.0)
    sent = {}
    for now in range(1, 40):
        for rank in (1, 2, 3):
            authority.handle(
                f"heartbeat-{rank}".encode(),
                Heartbeat(rank, 0, True, "node", 100 + rank),
                float(now),
            )
        outbound = authority.expire(float(now))
        if outbound:
            sent[now] = outbound
    ((_, outbound),) = sent.items()
    assert authority.phase is Phase.ABORTED and "rank 0" in authority.abort_reason
    assert authority.status().phase == "ABORTED"
    aborts = [route for route, message in outbound if isinstance(message, Abort)]
    restarts = [route for route, message in outbound if isinstance(message, Restart)]
    assert sorted(aborts) == [f"rank-{rank}".encode() for rank in RANKS]
    assert sorted(restarts) == [f"heartbeat-{rank}".encode() for rank in RANKS]
    assert fenced == []


def test_authority_ignores_evidence_during_recovery():
    fenced = []
    authority = _serving_authority(fenced)
    authority.handle(b"rank-0", Stopped(0, (2,)), 1.0)
    assert authority.phase is Phase.STOPPING
    for now in range(1, 40):
        authority.handle(b"heartbeat", DeviceWait(1, 3, 5000), float(now))
        authority.expire(min(float(now), 10.0))
    assert fenced == []


def test_after_a_recovery_a_hung_rank_that_has_not_departed_is_fenced():
    """A further loss recovers too, so a hung admitted rank is fenced after a recovery as before it."""
    fenced = []
    authority = _serving_authority(fenced)
    authority.phase = Phase.RECOVERED
    fences = _hang_rank_2(authority, 60)
    assert [message for _, message in next(iter(fences.values()))] == [Fence(2, 102)] * 3
    assert [rank for rank, *_ in fenced] == [2]


def test_the_fence_command_is_skipped_once_the_hung_rank_departed():
    fenced = []
    authority = _serving_authority(fenced)
    confirmed = next(iter(_hang_rank_2(_serving_authority([]), 60)))
    _hang_rank_2(authority, confirmed + 1)
    # The hung rank's loss recovered within the grace: the authority serves again, without it.
    authority.phase, authority.failed_ranks = Phase.RECOVERED, frozenset({2})
    for now in range(confirmed + 1, confirmed + 10):
        authority.expire(float(now))
    assert fenced == []


def _datagrams():
    receiver = authority_server.datagram_socket("tcp://127.0.0.1:0")
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender.connect(receiver.getsockname())
    return receiver, sender


def _drain(authority, receiver, now):
    # Datagrams on the loopback are queued at once.
    return authority_server._receive_datagrams(authority, receiver, KEY, now)


def _send_forgeries(sender, now):
    """Datagrams reporting progress that must not count: a replay, another process, another key."""
    sender.send(protocol.encode_datagram(NativeHeartbeat(2, 102, 1, 99 + now), KEY))
    sender.send(protocol.encode_datagram(NativeHeartbeat(2, 7, 10_000 + now, 99 + now), KEY))
    sender.send(
        protocol.encode_datagram(NativeHeartbeat(2, 102, 10_000 + now, 99 + now), b"x" * 32)
    )


def test_native_heartbeats_keep_a_rank_whose_python_is_silent_from_being_fenced():
    """Rank 2's Python evidence thread goes silent; its native datagrams still report each step it finishes."""
    fenced = []
    authority = _serving_authority(fenced)
    receiver, sender = _datagrams()
    try:
        authority.handle(b"evidence-2", Heartbeat(2, 0, False, "node", 102), 0.0)
        for now in range(1, 60):
            for rank in PEERS:
                route = f"evidence-{rank}".encode()
                authority.handle(route, Heartbeat(rank, 0, True, "node", 100 + rank), float(now))
            # A step every 5 s, within the window.
            sender.send(protocol.encode_datagram(NativeHeartbeat(2, 102, now, now // 5), KEY))
            assert _drain(authority, receiver, float(now)) == []
            assert authority.expire(float(now)) == [], now
        # Its native heartbeat stops reporting steps after one at 60 s: every peer waits on it, so it is fenced
        # CONFIRM later, whatever the forged datagrams claim.
        sent = {}
        for now in range(60, 90):
            for rank in PEERS:
                route = f"evidence-{rank}".encode()
                authority.handle(route, Heartbeat(rank, 0, True, "node", 100 + rank), float(now))
            sender.send(protocol.encode_datagram(NativeHeartbeat(2, 102, now, 12), KEY))
            _send_forgeries(sender, now)
            _drain(authority, receiver, float(now))
            if outbound := authority.expire(float(now)):
                sent[now] = outbound
        assert list(sent) == [60 + int(CONFIRM)]
        assert sent[60 + int(CONFIRM)][0] == (b"evidence-2", Fence(2, 102))
    finally:
        receiver.close()
        sender.close()


def test_an_authority_that_cannot_bind_its_datagram_port_fails():
    taken = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        taken.bind(("127.0.0.1", 0))
        with pytest.raises(RuntimeError, match="native heartbeats"):
            authority_server.datagram_socket(f"tcp://127.0.0.1:{taken.getsockname()[1]}")
    finally:
        taken.close()


def test_a_datagram_flood_yields_to_the_loop(monkeypatch):
    monkeypatch.setattr(authority_server, "_MAX_DATAGRAMS_PER_PASS", 2)
    authority = _serving_authority([])
    receiver, sender = _datagrams()
    try:
        for _ in range(3):
            sender.send(b"forged")
        assert _drain(authority, receiver, 0.0) == []
        # The third waits for the next pass.
        assert receiver.recv(16) == b"forged"
    finally:
        receiver.close()
        sender.close()


def test_the_native_heartbeat_wire_format():
    """The datagram ``HeartbeatSender`` (C++) builds: its JSON payload, then HMAC-SHA256 over a zero byte and it."""
    payload = b'{"beat": 3, "iteration": 41, "pid": 4321, "rank": 2, "type": "NativeHeartbeat"}'
    tag = hmac.new(KEY, b"\x00" + payload, hashlib.sha256).digest()
    assert protocol.decode_datagram(payload + tag, KEY) == NativeHeartbeat(2, 4321, 3, 41)
    assert protocol.encode_datagram(NativeHeartbeat(2, 4321, 3, 41), KEY) == payload + tag
    assert protocol.decode_datagram(payload + tag, b"y" * 32) is None
    assert protocol.decode_datagram(tag, KEY) is None
    # A connection's message (a pickle, then its HMAC-SHA256 tag) does not authenticate as a datagram, nor a
    # datagram as a connection's message; nor does any other message decode as a datagram.
    data = pickle.dumps(NativeHeartbeat(2, 4321, 3, 41))
    signed = data + hmac.new(KEY, data, hashlib.sha256).digest()
    assert protocol.decode_datagram(signed, KEY) is None
    assert hmac.new(KEY, payload, hashlib.sha256).digest() != tag
    other = protocol.encode_datagram(Heartbeat(2, 1, False, "node", 4321), KEY)
    assert protocol.decode_datagram(other, KEY) is None
    assert protocol.datagram_address("tcp://10.1.2.3:5555") == ("10.1.2.3", 5555)
    assert protocol.datagram_address("tcp://*:5555") == ("0.0.0.0", 5555)
    assert protocol.datagram_address("tcp://[::1]:5555") == ("::1", 5555)


def test_evidence_messages_round_trip():
    for message in (
        Heartbeat(3, 1234, True, "node-3", 4321),
        DeviceWait(1, 2, 2500),
        Restart("every rank stalled"),
    ):
        assert protocol.is_message(message)
        assert pickle.loads(pickle.dumps(message)) == message


@pytest.mark.parametrize("replayed", [True, False], ids=["replayed", "sent_anew"])
def test_a_replayed_device_wait_is_not_evidence(replayed):
    """Every peer's device waits on rank 2 once, then clears it; rank 2 is healthy.

    Replaying the captured waits from another connection must not fence rank 2. The control
    sends the same waits anew, as fresh messages of the peers' sessions: those fence it.
    """
    fenced = []
    authority = RecoveryAuthority(
        len(RANKS),
        hang_monitor=_monitor(),
        fence=lambda hung: fenced.append(hung.rank),
    )
    seq = dict.fromkeys(RANKS, 1)
    for rank in RANKS:
        register = Register(rank, ((1,), (2,), (3,)), DIGEST)
        authority.receive(f"rank-{rank}".encode(), Sent(rank, f"s{rank}", 1, register), 0.0)
    assert authority.phase is Phase.SERVING

    def send(rank, message, now):
        seq[rank] += 1
        sent = Sent(rank, f"s{rank}", seq[rank], message)
        authority.receive(f"evidence-{rank}".encode(), sent, now)
        return sent

    outbound = []
    for now in map(float, range(1, 60)):
        for rank in RANKS:
            send(rank, Heartbeat(rank, 0, False, "node", 100 + rank), now)
        if now == 1:
            captured = [send(rank, DeviceWait(rank, 2, 1000), now) for rank in PEERS]
        elif now == 2:
            for rank in PEERS:
                send(rank, DeviceWait(rank, 2, 0), now)
        elif replayed:
            for sent in captured:
                assert authority.receive(b"another-connection", sent, now) == []
        else:
            for rank in PEERS:
                send(rank, DeviceWait(rank, 2, 1000), now)
        outbound += authority.expire(now)
    if replayed:
        assert (outbound, fenced) == ([], [])
    else:
        assert (b"evidence-2", Fence(2, 102)) in outbound

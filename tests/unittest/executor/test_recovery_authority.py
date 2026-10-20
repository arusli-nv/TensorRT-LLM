# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recovery authority state machine, wire protocol, and client."""

import hmac
import pickle  # nosec B403
import threading
import time

import pytest
import zmq

from tensorrt_llm._utils import peer_loss_recovery_enabled
from tensorrt_llm.executor.ipc import ZeroMqQueue
from tensorrt_llm.executor.recovery import authority as authority_module
from tensorrt_llm.executor.recovery import client as recovery_client
from tensorrt_llm.executor.recovery import protocol
from tensorrt_llm.executor.recovery.client import (
    RecoveryAborted,
    RecoveryClient,
    RecoveryError,
    RecoveryInterrupted,
    RecoveryTimeout,
)
from tensorrt_llm.executor.recovery.config import CONFIG_ENV, FaultToleranceConfig
from tensorrt_llm.executor.recovery.protocol import (
    Abort,
    Commit,
    Fence,
    Fenced,
    Heartbeat,
    Install,
    Installed,
    Propose,
    Ready,
    Register,
    Registered,
    Sent,
    Status,
    StatusQuery,
    Stopped,
)

pytestmark = pytest.mark.cpu_only

Phase = authority_module.Phase
RecoveryAuthority = authority_module.RecoveryAuthority

WORLD = 4
FAILED = 2
SURVIVORS = (0, 1, 3)
RECOVERABLE = ((1,), (2,), (3,))
# With two GPUs per node, ranks 2 and 3 form node 1.
NODE = (2, 3)
KEY = b"k" * 32
DIGEST = FaultToleranceConfig().digest()


def _route(rank: int) -> bytes:
    return f"rank-{rank}".encode()


def _register_all(authority, recoverable=lambda rank: RECOVERABLE):
    out = []
    for rank in range(WORLD):
        assert authority.phase is Phase.REGISTERING
        out = authority.handle(_route(rank), Register(rank, recoverable(rank), DIGEST), 0.0)
    return out


def _registered(timeout: float = 10.0) -> RecoveryAuthority:
    authority = RecoveryAuthority(WORLD, recovery_timeout=timeout)
    out = _register_all(authority, lambda rank: ((1,), NODE, (FAILED,)))
    assert authority.phase is Phase.SERVING
    admitted = Registered(((1,), (FAILED,), NODE))
    assert out == [(_route(rank), admitted) for rank in range(WORLD)]
    return authority


def _report(authority, message_type, ranks=SURVIVORS, now=1.0, **fields):
    out = []
    for rank in ranks:
        out = authority.handle(_route(rank), message_type(rank, **fields), now)
    return _fence_departed(authority, out, now)


def _fence_departed(authority, out, now):
    """Rank 0 answers the authority's request to fence departed ranks: the runtime ended each."""
    fences = [message for _, message in out if isinstance(message, Fence)]
    if not fences:
        return out
    assert [route for route, _ in out] == [_route(0)] * len(fences)
    for fence in fences:
        out = authority.handle(_route(0), Fenced(0, fence.rank, ""), now)
    return out


def _assert_aborted(out, authority: RecoveryAuthority) -> None:
    assert authority.phase is Phase.ABORTED
    assert authority.deadline is None
    assert out and all(isinstance(message, Abort) for _, message in out)


def test_admission_is_the_reported_domains_without_rank0():
    authority = RecoveryAuthority(WORLD)
    out = _register_all(authority, lambda rank: ((0,), (3,), (WORLD,), (0, 1), NODE, (2,), ()))
    assert authority.admitted_domains == {frozenset({2}), frozenset({3}), frozenset(NODE)}
    assert all(message == Registered(((2,), NODE, (3,))) for _, message in out)


@pytest.mark.parametrize("other", [((2,), (3,)), ((1,), (2,), (3,), NODE)])
def test_workers_reporting_different_recoverable_domains_abort(other):
    authority = RecoveryAuthority(WORLD)
    out = _register_all(authority, lambda rank: RECOVERABLE if rank != 3 else other)
    _assert_aborted(out, authority)
    assert {route for route, _ in out} == {_route(rank) for rank in range(WORLD)}


def test_idle_loss_recovers():
    authority = _registered()
    authority.handle(_route(0), Stopped(0, (FAILED,)), 1.0)
    assert authority.phase is Phase.STOPPING
    assert authority.deadline == pytest.approx(11.0)

    out = _report(authority, Stopped, SURVIVORS[1:], failed_ranks=(FAILED,))
    assert out == [(_route(rank), Propose(1, (FAILED,), SURVIVORS)) for rank in SURVIVORS]
    out = _report(authority, Ready, generation=1, placement_digest="digest")
    assert out == [(_route(rank), Install(1)) for rank in SURVIVORS]
    out = _report(authority, Installed, generation=1)
    assert out == [(_route(rank), Commit(1)) for rank in SURVIVORS]
    assert authority.phase is Phase.RECOVERED and authority.generation == 1
    assert authority.deadline is None and authority.expire(1e9) == []


def test_node_loss_recovers_in_one_transaction():
    authority = _registered()
    out = _report(authority, Stopped, (0, 1), failed_ranks=NODE)
    assert out == [(_route(rank), Propose(1, NODE, (0, 1))) for rank in (0, 1)]
    _report(authority, Ready, (0, 1), generation=1, placement_digest="digest")
    out = _report(authority, Installed, (0, 1), generation=1)
    assert out == [(_route(rank), Commit(1)) for rank in (0, 1)]
    assert authority.failed_ranks == frozenset(NODE) and authority.phase is Phase.RECOVERED


def _status(authority) -> Status:
    ((route, status),) = authority.handle(b"observer", StatusQuery(), 0.0)
    assert route == b"observer"
    return status


def test_status_reports_the_recovery_and_its_stage_durations():
    authority = _registered()
    assert _status(authority) == Status("SERVING", 0, (0, 1, 2, 3), (), 0, (), 0, "")
    authority.handle(_route(0), Stopped(0, NODE), 1.0)
    assert _status(authority).phase == "STOPPING"
    _report(authority, Stopped, (1,), now=1.5, failed_ranks=NODE)
    _report(authority, Ready, (0, 1), now=1.75, generation=1, placement_digest="digest")
    # Rank 0 routes every request, so it reports how many the departed ranks were serving.
    authority.handle(_route(0), Installed(0, 1, failed_requests=3), 2.0)
    _report(authority, Installed, (1,), now=2.0, generation=1)
    assert _status(authority) == Status(
        "RECOVERED", 1, (0, 1), NODE, 1, (500_000, 250_000, 250_000), 3, ""
    )


def test_status_is_answered_after_an_abort_and_changes_nothing():
    authority = _registered()
    authority.handle(_route(0), Stopped(0, (FAILED,)), 1.0)
    # An observer polling mid-recovery neither registers nor disturbs the recovery.
    assert _status(authority).phase == "STOPPING" and authority.phase is Phase.STOPPING
    authority.expire(1e9)
    status = _status(authority)
    assert status.phase == "ABORTED" and "did not commit" in status.abort_reason


@pytest.mark.parametrize("failed", [(0,), (0, 1), ()])
def test_unadmitted_loss_aborts_every_worker(failed):
    authority = _registered()
    out = authority.handle(_route(1), Stopped(1, failed), 1.0)
    _assert_aborted(out, authority)
    assert {route for route, _ in out} == {_route(rank) for rank in range(WORLD)}


def test_a_stray_message_while_serving_is_refused_without_ending_the_deployment():
    authority = _registered()
    out = authority.handle(_route(1), Ready(1, 1, "digest"), 1.0)
    assert [(route, type(message)) for route, message in out] == [(_route(1), Abort)]
    assert authority.phase is Phase.SERVING
    # A frame that does not decode (e.g. version skew) likewise.
    assert [route for route, _ in authority.reject("bad frame", _route(1))] == [_route(1)]
    assert authority.phase is Phase.SERVING
    # A loss report is real: one the deployment cannot recover from is terminal.
    _assert_aborted(authority.handle(_route(1), Stopped(1, (0,)), 1.0), authority)


def _in_phase(phase: Phase) -> RecoveryAuthority:
    """An authority in ``phase``, recovering from the loss of rank ``FAILED`` once registered."""
    if phase is Phase.REGISTERING:
        authority = RecoveryAuthority(WORLD, recovery_timeout=10.0)
        authority.handle(_route(0), Register(0, RECOVERABLE, DIGEST), 0.0)
        return authority
    authority = _registered()
    steps = [
        lambda: authority.handle(_route(0), Stopped(0, (FAILED,)), 1.0),
        lambda: _report(authority, Stopped, SURVIVORS[1:], failed_ranks=(FAILED,)),
        lambda: _report(authority, Ready, generation=1, placement_digest="d"),
        lambda: _report(authority, Installed, generation=1),
    ]
    # Each step enters the next phase: STOPPING, PREPARING, INSTALLING, RECOVERED.
    for step in steps:
        if authority.phase is phase:
            break
        step()
    assert authority.phase is phase
    return authority


@pytest.mark.parametrize(
    "phase",
    [
        Phase.REGISTERING,
        Phase.SERVING,
        Phase.STOPPING,
        Phase.PREPARING,
        Phase.INSTALLING,
        Phase.RECOVERED,
    ],
    ids=lambda phase: phase.name.lower(),
)
def test_a_frame_that_does_not_decode_is_refused_to_its_sender_in_every_phase(phase):
    """It names no rank and no protocol step: only its sender hears of it, and the authority's state stays."""
    authority = _in_phase(phase)
    deadline, failed = authority.deadline, authority.failed_ranks
    for route in (_route(1), b"stranger"):
        assert authority.reject("bad frame", route) == [(route, Abort("bad frame"))]
    assert authority.phase is phase
    assert (authority.deadline, authority.failed_ranks) == (deadline, failed)
    assert authority.abort_reason is None


def test_a_recovery_completes_after_a_frame_that_does_not_decode():
    authority = _in_phase(Phase.STOPPING)
    authority.reject("bad frame", _route(1))
    _report(authority, Stopped, SURVIVORS[1:], failed_ranks=(FAILED,))
    _report(authority, Ready, generation=1, placement_digest="digest")
    out = _report(authority, Installed, generation=1)
    assert authority.phase is Phase.RECOVERED
    assert out == [(_route(rank), Commit(1)) for rank in SURVIVORS]


@pytest.mark.parametrize(
    "route, message",
    [
        (_route(3), Stopped(3, (1,))),  # second distinct failure
        (_route(0), Stopped(0, (1, 3))),  # a different failed set (a larger one is a re-plan)
        (_route(FAILED), Stopped(FAILED, (FAILED,))),  # departed rank still talking
        (_route(1), Stopped(1, (FAILED,))),  # duplicate report
        (_route(1), Ready(1, 1, "digest")),  # Ready before Propose
        (_route(1), Commit(1)),  # authority-only message from a worker
        (_route(1), Register(1, RECOVERABLE, DIGEST)),  # registration after it completed
    ],
)
def test_protocol_violations_abort(route, message):
    authority = _registered()
    authority.handle(_route(1), Stopped(1, (FAILED,)), 1.0)
    out = authority.handle(route, message, 1.0)
    _assert_aborted(out, authority)
    assert (route, Abort(authority.abort_reason)) in out


@pytest.mark.parametrize(
    "phase",
    [Phase.SERVING, Phase.STOPPING, Phase.PREPARING, Phase.INSTALLING, Phase.RECOVERED],
    ids=lambda phase: phase.name.lower(),
)
@pytest.mark.parametrize(
    "route, message",
    [
        (b"stranger", Stopped(1, (FAILED,))),  # a member's report replayed on another connection
        (b"stranger", Stopped(FAILED, (FAILED,))),
        (b"stranger", Ready(1, 1, "d")),
        (b"stranger", Installed(1, 1)),
        (b"stranger", Register(1, RECOVERABLE, DIGEST)),
        (b"stranger", Commit(1)),
        (_route(1), Stopped(3, (FAILED,))),  # another rank's report on rank 1's route
        (_route(1), Ready(3, 1, "d")),
        (_route(1), Register(3, RECOVERABLE, DIGEST)),
    ],
    ids=lambda value: type(value).__name__ if not isinstance(value, bytes) else value.decode(),
)
def test_messages_no_member_can_send_on_their_route_are_dropped(phase, route, message):
    """Messages are not bound to their connection, so these may be replays: they change nothing."""
    authority = _in_phase(phase)
    state = (authority.phase, authority.deadline, authority.failed_ranks, authority.generation)
    assert authority.handle(route, message, 1.0) == []
    assert (
        authority.phase,
        authority.deadline,
        authority.failed_ranks,
        authority.generation,
    ) == state
    if phase is Phase.STOPPING:
        # The recovery still commits.
        _report(authority, Stopped, SURVIVORS[1:], failed_ranks=(FAILED,))
        _report(authority, Ready, generation=1, placement_digest="d")
        assert _report(authority, Installed, generation=1) == [
            (_route(rank), Commit(1)) for rank in SURVIVORS
        ]


def test_rank_reporting_its_own_loss_aborts():
    authority = _registered()
    _assert_aborted(authority.handle(_route(1), Stopped(1, (1,)), 1.0), authority)


def test_placement_digests_must_agree():
    authority = _registered()
    _report(authority, Stopped, failed_ranks=(FAILED,))
    authority.handle(_route(0), Ready(0, 1, "a"), 2.0)
    _assert_aborted(authority.handle(_route(1), Ready(1, 1, "b"), 2.0), authority)


@pytest.mark.parametrize("message_type", [Ready, Installed])
def test_reports_for_another_generation_abort(message_type):
    authority = _registered()
    _report(authority, Stopped, failed_ranks=(FAILED,))
    if message_type is Installed:
        _report(authority, Ready, generation=1, placement_digest="digest")
    fields = {"placement_digest": "digest"} if message_type is Ready else {}
    out = authority.handle(_route(0), message_type(0, generation=2, **fields), 2.0)
    _assert_aborted(out, authority)


@pytest.mark.parametrize(
    "route, message",
    [
        (_route(1), Register(WORLD, RECOVERABLE, DIGEST)),  # rank above the world
        (_route(1), Register(-1, RECOVERABLE, DIGEST)),  # negative rank
        (_route(1), Register(0, RECOVERABLE, DIGEST)),  # duplicate rank
        (_route(1), Register(1, RECOVERABLE, "other")),  # different fault-tolerance settings
        (_route(0), Stopped(0, (FAILED,))),  # message before registration completed
    ],
)
def test_registration_violations_abort(route, message):
    authority = RecoveryAuthority(WORLD)
    authority.handle(_route(0), Register(0, RECOVERABLE, DIGEST), 0.0)
    _assert_aborted(authority.handle(route, message, 0.0), authority)


def test_a_message_from_an_unregistered_route_during_registration_is_dropped():
    authority = RecoveryAuthority(WORLD)
    authority.handle(_route(0), Register(0, RECOVERABLE, DIGEST), 0.0)
    assert authority.handle(b"stranger", Stopped(1, (FAILED,)), 0.0) == []
    assert authority.phase is Phase.REGISTERING


def test_registration_with_other_settings_is_refused():
    other = FaultToleranceConfig(hang_detection=True).digest()
    authority = RecoveryAuthority(WORLD)
    out = authority.handle(_route(0), Register(0, RECOVERABLE, other), 0.0)
    _assert_aborted(out, authority)
    assert other in authority.abort_reason and CONFIG_ENV in authority.abort_reason


@pytest.mark.parametrize("phase_reports", [1, 2, 3], ids=["stopping", "preparing", "installing"])
def test_recovery_deadline_aborts_and_aborted_is_terminal(phase_reports):
    authority = _registered(timeout=5.0)
    authority.handle(_route(0), Stopped(0, (FAILED,)), 1.0)
    if phase_reports >= 2:
        _report(authority, Stopped, SURVIVORS[1:], failed_ranks=(FAILED,))
    if phase_reports >= 3:
        _report(authority, Ready, generation=1, placement_digest="digest")
    assert authority.expire(5.9) == []
    _assert_aborted(authority.expire(6.0), authority)
    reply = authority.handle(_route(1), Stopped(1, (FAILED,)), 7.0)
    assert reply == [(_route(1), Abort(authority.abort_reason))]
    assert authority.expire(1e9) == []


def test_a_loss_during_the_recovery_plans_again_with_every_lost_rank():
    """Rank 3 fails while the survivors prepare to drop rank 2: the plan restarts without both."""
    authority = _registered()
    authority.handle(_route(1), Stopped(1, (FAILED,)), 1.0)
    out = _report(authority, Stopped, (0, 3), failed_ranks=(FAILED,))
    assert out == [(_route(rank), Propose(1, (FAILED,), SURVIVORS)) for rank in SURVIVORS]
    assert authority.handle(_route(0), Ready(0, 1, "digest"), 1.0) == []

    lost = (FAILED, 3)
    assert authority.handle(_route(0), Stopped(0, lost), 2.0) == []
    assert authority.phase is Phase.STOPPING and authority.failed_ranks == frozenset(lost)
    # Every round shares the first round's deadline.
    assert authority.deadline == pytest.approx(11.0)
    # Rank 3 sent this before it failed.
    assert authority.handle(_route(3), Ready(3, 1, "digest"), 2.0) == []
    # Rank 1 sent these before it saw rank 3 fail; its Stopped for the new plan follows.
    assert authority.handle(_route(1), Ready(1, 1, "digest"), 2.0) == []
    assert authority.handle(_route(1), Stopped(1, (FAILED,)), 2.0) == []
    out = authority.handle(_route(1), Stopped(1, lost), 2.0)
    # Rank 2 was fenced in the first round; only rank 3 is asked for.
    assert out == [(_route(0), Fence(3, 0))]
    out = _fence_departed(authority, out, 2.0)
    assert out == [(_route(rank), Propose(1, lost, (0, 1))) for rank in (0, 1)]
    _report(authority, Ready, (0, 1), generation=1, placement_digest="digest")
    out = _report(authority, Installed, (0, 1), generation=1)
    assert out == [(_route(rank), Commit(1)) for rank in (0, 1)]
    assert authority.phase is Phase.RECOVERED and authority.survivors == (0, 1)


def test_a_departed_rank_not_known_dead_is_fenced_before_the_proposal():
    """The survivors may agree on a failure detector's suspicion: rank 0 ends the rank before any plan."""
    authority = _registered()
    authority.handle(b"evidence-2", Heartbeat(FAILED, 0, False, "node", 4242), 0.5)
    authority.handle(_route(1), Stopped(1, (FAILED,)), 1.0)
    out = authority.handle(_route(3), Stopped(3, (FAILED,)), 1.0)
    assert out == []
    out = authority.handle(_route(0), Stopped(0, (FAILED,)), 1.0)
    assert out == [(_route(0), Fence(FAILED, 4242))]
    assert authority.phase is Phase.STOPPING
    out = authority.handle(_route(0), Fenced(0, FAILED, ""), 1.25)
    assert out == [(_route(rank), Propose(1, (FAILED,), SURVIVORS)) for rank in SURVIVORS]
    assert authority.phase is Phase.PREPARING


def test_a_departed_rank_that_cannot_be_fenced_aborts_the_recovery():
    authority = _registered()
    for rank in SURVIVORS:
        authority.handle(_route(rank), Stopped(rank, (FAILED,)), 1.0)
    out = authority.handle(_route(0), Fenced(0, FAILED, "PMIx refused to kill rank 2"), 1.25)
    _assert_aborted(out, authority)
    assert "could not be fenced" in authority.abort_reason
    assert not any(isinstance(message, Propose) for _, message in out)


@pytest.mark.parametrize(
    "message",
    [Fenced(1, FAILED, ""), Fenced(0, 3, "")],
    ids=["not_rank0", "rank_that_did_not_fail"],
)
def test_a_fence_answer_from_another_rank_or_for_a_live_rank_aborts(message):
    authority = _registered()
    for rank in SURVIVORS:
        authority.handle(_route(rank), Stopped(rank, (FAILED,)), 1.0)
    _assert_aborted(authority.handle(_route(message.rank), message, 1.25), authority)


def test_a_loss_during_the_recovery_that_takes_rank0_aborts():
    authority = _registered()
    _report(authority, Stopped, (1, 3), failed_ranks=(FAILED,))
    _assert_aborted(authority.handle(_route(1), Stopped(1, (0, FAILED)), 2.0), authority)


def test_a_loss_after_install_aborts():
    authority = _registered()
    _report(authority, Stopped, failed_ranks=(FAILED,))
    _report(authority, Ready, generation=1, placement_digest="digest")
    assert authority.phase is Phase.INSTALLING
    _assert_aborted(authority.handle(_route(0), Stopped(0, (FAILED, 3)), 2.0), authority)


def _recovered_without_failed() -> RecoveryAuthority:
    authority = _registered()
    _report(authority, Stopped, failed_ranks=(FAILED,))
    _report(authority, Ready, generation=1, placement_digest="digest")
    _report(authority, Installed, generation=1)
    assert authority.phase is Phase.RECOVERED
    return authority


def test_a_further_loss_after_a_recovery_recovers_again():
    """Rank 1 is lost after the recovery without rank 2: a second recovery commits generation 2.

    The survivors report both ranks; the recovery has its own deadline and stage durations.
    """
    authority = _recovered_without_failed()
    lost = (1, FAILED)
    authority.handle(_route(0), Stopped(0, lost), 50.0)
    assert authority.phase is Phase.STOPPING and authority.deadline == pytest.approx(60.0)
    out = authority.handle(_route(3), Stopped(3, lost), 50.5)
    # Rank 2 was fenced in the first recovery; only rank 1 is asked for.
    assert out == [(_route(0), Fence(1, 0))]
    out = _fence_departed(authority, out, 50.5)
    assert out == [(_route(rank), Propose(2, lost, (0, 3))) for rank in (0, 3)]
    out = _report(authority, Ready, (0, 3), now=51.0, generation=2, placement_digest="digest2")
    assert out == [(_route(rank), Install(2)) for rank in (0, 3)]
    out = _report(authority, Installed, (0, 3), now=51.25, generation=2, failed_requests=1)
    assert out == [(_route(rank), Commit(2)) for rank in (0, 3)]
    assert authority.phase is Phase.RECOVERED and authority.deadline is None
    assert authority.status() == Status(
        "RECOVERED", 2, (0, 3), lost, 2, (500_000, 500_000, 250_000), 2, ""
    )


@pytest.mark.parametrize(
    "failed", [(1,), (FAILED,), (0, FAILED)], ids=["omits_departed", "no_new_rank", "rank0"]
)
def test_a_further_loss_report_that_is_not_a_superset_or_takes_rank0_aborts(failed):
    authority = _recovered_without_failed()
    _assert_aborted(authority.handle(_route(1), Stopped(1, failed), 5.0), authority)


def _dealer(endpoint: str, key: bytes = KEY, options: dict | None = None) -> ZeroMqQueue:
    return ZeroMqQueue(
        (endpoint, key),
        socket_type=zmq.DEALER,
        is_server=False,
        socket_options={zmq.LINGER: 0, **(options or {})},
    )


def _signed(obj, key: bytes = KEY) -> bytes:
    """``obj`` as ``ZeroMqQueue`` sends it: its pickle, then the pickle's HMAC-SHA256 tag."""
    data = pickle.dumps(obj)
    return data + hmac.new(key, data, "sha256").digest()


def test_wire_roundtrip_and_authentication(tmp_path):
    endpoint = f"ipc://{tmp_path}/authority"
    router = authority_module.router_queue(endpoint, KEY)
    dealer = _dealer(endpoint)
    try:
        # Unlike ZeroMqQueue's default, a connection cannot queue without bound.
        assert router.socket.getsockopt(zmq.RCVHWM) == authority_module._ROUTER_RCVHWM
        for message in (
            Register(1, ((1,), NODE), DIGEST),
            Registered(((1,), NODE)),
            Stopped(1, NODE),
            Propose(3, NODE, (0, 1)),
            Fence(2, 0),
            Fenced(0, 2, "PMIx refused"),
            Abort("lost"),
            StatusQuery(),
            Status("RECOVERED", 1, (0, 1), NODE, 1, (500_000, 250_000, 250_000), 2, ""),
            Status("SERVING", 0, (0, 1), (), 0, (), 0, ""),
        ):
            dealer.put(message)
            assert router.get(timeout=5.0) == message
            router.put(message, routing_id=router._last_identity)
            assert dealer.get(timeout=5.0) == message
        stranger = _dealer(endpoint, b"x" * 32)
        try:
            stranger.put(Install(1))
            with pytest.raises(RuntimeError, match="HMAC"):
                router.get(timeout=5.0)
        finally:
            stranger.close()
        # A tampered message does not authenticate either.
        signed = _signed(Install(1))
        dealer.socket.send(signed[:-33] + bytes([signed[-33] ^ 1]) + signed[-32:])
        with pytest.raises(RuntimeError, match="HMAC"):
            router.get(timeout=5.0)
    finally:
        dealer.close()
        router.close()


@pytest.mark.parametrize(
    "obj",
    [
        "not a message",
        {"type": "Install", "generation": 1},
        Install,
        type("Install", (Install,), {})(1),
        None,
    ],
)
def test_objects_that_are_not_messages_are_recognized(obj):
    assert not protocol.is_message(obj)
    assert protocol.is_message(Install(1))


@pytest.mark.parametrize(
    "payload",
    [
        b"not json",
        b"[]",
        b'{"type": "Nope"}',
        b'{"type": "NativeHeartbeat"}',
        b'{"beat": 3, "iteration": 41, "pid": 4321, "rank": "2", "type": "NativeHeartbeat"}',
        b'{"beat": 3, "iteration": 41, "pid": 4321, "rank": true, "type": "NativeHeartbeat"}',
        b'{"beat": 3, "extra": 0, "iteration": 41, "pid": 4321, "rank": 2, "type": "NativeHeartbeat"}',
        b'{"beat": 3, "iteration": [41], "pid": 4321, "rank": 2, "type": "NativeHeartbeat"}',
        b'{"beat": 3, "iteration": 41, "pid": 4321, "rank": 2, "type": "Install"}',
        b'{"beat": '
        + b"9" * 5000
        + b', "iteration": 41, "pid": 4321, "rank": 2, "type": "NativeHeartbeat"}',
        b"[" * 100000 + b"]" * 100000,
    ],
)
def test_malformed_datagrams_are_rejected(payload):
    tag = hmac.new(KEY, b"\x00" + payload, "sha256").digest()
    assert protocol.decode_datagram(payload + tag, KEY) is None


def test_load_key_validates_the_environment(monkeypatch):
    env = recovery_client.AUTHORITY_KEY_ENV
    monkeypatch.delenv(env, raising=False)
    with pytest.raises(ValueError, match="not set"):
        recovery_client.load_key()
    monkeypatch.setenv(env, "zz")
    with pytest.raises(ValueError, match="hex"):
        recovery_client.load_key()
    monkeypatch.setenv(env, "00" * 8)
    with pytest.raises(ValueError, match="16 bytes"):
        recovery_client.load_key()
    monkeypatch.setenv(env, KEY.hex())
    assert recovery_client.load_key() == KEY


def test_client_from_env(monkeypatch):
    monkeypatch.setenv("TLLM_FAULT_TOLERANCE_MODE", "1")
    monkeypatch.delenv("TLLM_FAULT_TOLERANCE_AUTHORITY", raising=False)
    peer_loss_recovery_enabled.cache_clear()
    assert RecoveryClient.from_env(0) is None  # detection-only mode
    monkeypatch.setenv("TLLM_FAULT_TOLERANCE_AUTHORITY", "tcp://127.0.0.1:1")
    monkeypatch.delenv("TLLM_FAULT_TOLERANCE_AUTHORITY_KEY", raising=False)
    peer_loss_recovery_enabled.cache_clear()
    with pytest.raises(ValueError, match="KEY"):
        RecoveryClient.from_env(0)
    monkeypatch.delenv("TLLM_FAULT_TOLERANCE_MODE")
    peer_loss_recovery_enabled.cache_clear()
    with pytest.raises(ValueError, match="TLLM_FAULT_TOLERANCE_MODE=1"):
        RecoveryClient.from_env(0)
    peer_loss_recovery_enabled.cache_clear()


# End to end over ZeroMQ.


@pytest.fixture
def served(tmp_path):
    authority = RecoveryAuthority(WORLD, recovery_timeout=10.0)
    endpoint = f"ipc://{tmp_path}/authority"
    router = authority_module.router_queue(endpoint, KEY)
    stop = threading.Event()
    thread = threading.Thread(
        target=authority_module.serve, args=(authority, router, KEY, stop, 0.01), daemon=True
    )
    thread.start()
    yield authority, endpoint
    stop.set()
    thread.join(timeout=5)
    router.close()


@pytest.fixture
def make_client():
    clients = []

    def make(endpoint, rank, key=KEY):
        clients.append(RecoveryClient(endpoint, key, rank))
        return clients[-1]

    yield make
    for client in clients:
        client.close()


@pytest.fixture
def clients(served, make_client):
    return {rank: make_client(served[1], rank) for rank in range(WORLD)}


def _register_clients(clients):
    admitted = {}

    def register(rank):
        admitted[rank] = clients[rank].register(((FAILED,),), time.monotonic() + 10.0)

    threads = [threading.Thread(target=register, args=(rank,)) for rank in clients]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return admitted


def _wait_for_proposal(client, rank, deadline) -> Propose:
    """Wait for the plan; rank 0 first answers each request to fence a departed rank."""
    proposal = client.wait((Propose, Fence), deadline)
    while isinstance(proposal, Fence):
        client.send(Fenced(rank, proposal.rank, ""))
        proposal = client.wait((Propose, Fence), deadline)
    return proposal


def _worker_recovery(client, rank, results) -> None:
    deadline = time.monotonic() + 10.0
    try:
        client.send(Stopped(rank, (FAILED,)))
        proposal = _wait_for_proposal(client, rank, deadline)
        client.send(Ready(rank, proposal.generation, "digest"))
        client.wait(Install, deadline)
        client.send(Installed(rank, proposal.generation))
        results[rank] = client.wait(Commit, deadline)
    except RecoveryError as error:
        results[rank] = error


def test_end_to_end_over_zmq(served, clients):
    authority = served[0]
    assert _register_clients(clients) == {rank: {frozenset({FAILED})} for rank in range(WORLD)}

    results = {}
    workers = [
        threading.Thread(target=_worker_recovery, args=(clients[rank], rank, results))
        for rank in SURVIVORS
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert results == {rank: Commit(1) for rank in SURVIVORS}
    assert authority.phase is Phase.RECOVERED


def test_abort_reaches_every_worker(clients):
    _register_clients(clients)
    clients[1].send(Stopped(1, (3,)))  # rank 3 is not admitted
    for rank in (0, 3):
        with pytest.raises(RecoveryAborted):
            clients[rank].wait(Propose, time.monotonic() + 5.0)
    with pytest.raises(RecoveryTimeout):
        clients[0].wait(Propose, time.monotonic() + 0.05)


def test_an_interrupted_wait_takes_a_message_that_already_arrived(clients):
    """E.g. an Install the other survivors act on: dropping it for a later loss would leave them waiting."""
    _register_clients(clients)
    further_loss = threading.Event()
    further_loss.set()
    clients[1].send(Stopped(1, (3,)))
    assert clients[0]._queue.poll(5)
    with pytest.raises(RecoveryAborted):
        clients[0].wait(Propose, time.monotonic() + 5.0, interrupt=further_loss)
    with pytest.raises(RecoveryInterrupted):
        clients[0].wait(Propose, time.monotonic() + 5.0, interrupt=further_loss)


def test_stopping_during_a_recovery_aborts_it(tmp_path, make_client):
    authority = RecoveryAuthority(WORLD, recovery_timeout=10.0)
    endpoint = f"ipc://{tmp_path}/authority"
    router = authority_module.router_queue(endpoint, KEY)
    stop = threading.Event()
    thread = threading.Thread(
        target=authority_module.serve, args=(authority, router, KEY, stop, 0.01)
    )
    thread.start()
    try:
        clients = {rank: make_client(endpoint, rank) for rank in range(WORLD)}
        _register_clients(clients)
        clients[0].send(Stopped(0, (FAILED,)))
        deadline = time.monotonic() + 5.0
        while authority.phase is not Phase.STOPPING:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        stop.set()
        thread.join(timeout=5)
        with pytest.raises(RecoveryAborted):
            clients[1].wait(Propose, time.monotonic() + 5.0)
    finally:
        stop.set()
        thread.join(timeout=5)
        router.close()


def _captured(client) -> list:
    """Record every object ``client`` puts on its connection: what a network observer captures."""
    captured, put = [], client._queue.put
    client._queue.put = lambda obj: (captured.append(obj), put(obj))[1]
    return captured


def _wait_for(authority, phase):
    deadline = time.monotonic() + 5.0
    while authority.phase is not phase:
        assert time.monotonic() < deadline
        time.sleep(0.01)


def _replay(endpoint, captured, routing_id=None) -> Status:
    """Send ``captured`` again from a new connection; return the status the authority answers next."""
    options = {} if routing_id is None else {zmq.ROUTING_ID: routing_id}
    # A routing id frees once the authority sees its old connection close: retry until it is accepted.
    for _ in range(50):
        replayer = _dealer(endpoint, options=options)
        try:
            replayer.put(StatusQuery())
            if not replayer.poll(1):
                continue
            replayer.get()
            for obj in captured:
                replayer.socket.send(_signed(obj))
            replayer.put(StatusQuery())
            status = replayer.get(timeout=5.0)
            assert isinstance(status, Status), status
            return status
        finally:
            replayer.close()
    raise AssertionError("the authority never accepted the replaying connection")


def _finish_recovery(clients) -> dict:
    """Rank 0 already reported; every survivor completes the recovery from rank ``FAILED``."""
    results = {}

    def rank0():
        try:
            proposal = _wait_for_proposal(clients[0], 0, time.monotonic() + 10.0)
            clients[0].send(Ready(0, proposal.generation, "digest"))
            clients[0].wait(Install, time.monotonic() + 10.0)
            clients[0].send(Installed(0, proposal.generation))
            results[0] = clients[0].wait(Commit, time.monotonic() + 10.0)
        except RecoveryError as error:
            results[0] = error

    workers = [threading.Thread(target=rank0)] + [
        threading.Thread(target=_worker_recovery, args=(clients[rank], rank, results))
        for rank in SURVIVORS[1:]
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    return results


def test_a_replayed_report_does_not_abort_a_recovery(served, clients):
    """Rank 0's captured, authenticated messages sent again from another connection: the recovery commits."""
    authority, endpoint = served
    captured = _captured(clients[0])
    _register_clients(clients)
    clients[0].send(Stopped(0, (FAILED,)))
    _wait_for(authority, Phase.STOPPING)
    assert _replay(endpoint, captured).phase == "STOPPING"
    assert _finish_recovery(clients) == {rank: Commit(1) for rank in SURVIVORS}
    assert authority.phase is Phase.RECOVERED


def test_a_departed_ranks_replayed_messages_are_dropped_on_its_own_route(served, clients):
    """A rank departs; its captured messages come again on a connection with its routing id."""
    authority, endpoint = served
    captured = _captured(clients[FAILED])
    _register_clients(clients)
    routing_id = clients[FAILED]._queue.socket.getsockopt(zmq.ROUTING_ID)
    clients[FAILED].close()
    clients[0].send(Stopped(0, (FAILED,)))
    _wait_for(authority, Phase.STOPPING)
    # Accepted, its Register would abort: registration after it completed, from a departed rank.
    assert _replay(endpoint, captured, routing_id).phase == "STOPPING"
    assert _finish_recovery(clients) == {rank: Commit(1) for rank in SURVIVORS}
    assert authority.phase is Phase.RECOVERED


def test_only_fresh_messages_of_a_ranks_session_are_applied():
    authority = RecoveryAuthority(WORLD, recovery_timeout=10.0)
    domains = ((1,), NODE, (FAILED,))
    for rank in range(WORLD):
        authority.receive(
            _route(rank), Sent(rank, f"s{rank}", 1, Register(rank, domains, DIGEST)), 0.0
        )
    assert authority.phase is Phase.SERVING
    assert authority.receive(_route(1), Sent(1, "s1", 5, Stopped(1, (FAILED,))), 1.0) == []
    assert authority.phase is Phase.STOPPING
    # Applied, a Ready before Propose would abort: each of these is dropped instead.
    for sent in (
        Sent(1, "s1", 4, Ready(1, 1, "d")),  # out of order
        Sent(1, "s1", 5, Ready(1, 1, "d")),  # a repeated sequence number
        Sent(1, "other", 9, Ready(1, 1, "d")),  # another session
        Sent(3, "s3", 9, Ready(1, 1, "d")),  # another rank's envelope
        Sent(1, "s1", 1, Register(1, domains, DIGEST)),  # the rank's own Register, replayed
    ):
        assert authority.receive(_route(1), sent, 1.0) == [], sent
        assert authority.phase is Phase.STOPPING
    # Evidence is a stream of its own (another connection): a lower sequence number there is fresh.
    authority.receive(b"evidence-1", Sent(1, "s1", 2, Heartbeat(1, 0, True, "node", 101)), 1.0)
    assert authority._sequences[(1, True)] == 2
    _assert_aborted(
        authority.receive(_route(1), Sent(1, "s1", 6, Ready(1, 1, "d")), 1.0), authority
    )


def test_a_fence_answer_is_applied_only_fresh_from_rank_0_s_protocol_stream():
    authority = RecoveryAuthority(WORLD, recovery_timeout=10.0)
    domains = ((1,), NODE, (FAILED,))
    for rank in range(WORLD):
        authority.receive(
            _route(rank), Sent(rank, f"s{rank}", 1, Register(rank, domains, DIGEST)), 0.0
        )
    for rank in SURVIVORS:
        out = authority.receive(
            _route(rank), Sent(rank, f"s{rank}", 2, Stopped(rank, (FAILED,))), 1.0
        )
    assert out == [(_route(0), Fence(FAILED, 0))]
    # Applied, a successful answer would propose (and an error abort): each of these is dropped instead.
    for route, sent in (
        (_route(0), Sent(0, "s0", 2, Fenced(0, FAILED, ""))),  # a repeated sequence number
        (_route(0), Sent(0, "other", 9, Fenced(0, FAILED, ""))),  # another session
        (_route(1), Sent(1, "s1", 9, Fenced(0, FAILED, ""))),  # rank 1's envelope
        (
            b"evidence-0",
            Sent(0, "s0", 1, Fenced(0, FAILED, "refused")),
        ),  # stale on the protocol stream
    ):
        assert authority.receive(route, sent, 1.0) == [], sent
        assert authority.phase is Phase.STOPPING
    out = authority.receive(_route(0), Sent(0, "s0", 3, Fenced(0, FAILED, "")), 1.25)
    assert out == [(_route(rank), Propose(1, (FAILED,), SURVIVORS)) for rank in SURVIVORS]


def test_a_second_session_registering_a_rank_aborts():
    """Not a replay: two processes claim the rank."""
    authority = RecoveryAuthority(WORLD)
    authority.receive(_route(0), Sent(0, "s0", 1, Register(0, RECOVERABLE, DIGEST)), 0.0)
    out = authority.receive(_route(1), Sent(0, "other", 1, Register(0, RECOVERABLE, DIGEST)), 0.0)
    _assert_aborted(out, authority)


@pytest.mark.parametrize(
    "error", [RuntimeError("not an HMAC failure"), ValueError("not a frame count")]
)
def test_serve_raises_errors_other_than_authentication_failures(tmp_path, monkeypatch, error):
    endpoint = f"ipc://{tmp_path}/authority"
    router = authority_module.router_queue(endpoint, KEY)
    dealer = _dealer(endpoint)
    try:
        dealer.put(StatusQuery())

        def get(timeout=None):
            raise error

        monkeypatch.setattr(router, "get", get)
        raised, stop = [], threading.Event()

        def run():
            try:
                authority_module.serve(RecoveryAuthority(WORLD), router, KEY, stop, 0.01)
            except (RuntimeError, ValueError) as caught:
                raised.append(caught)

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout=5)
        stop.set()
        thread.join(timeout=5)
        assert raised == [error]
    finally:
        dealer.close()
        router.close()


def test_frames_signed_with_another_key_are_dropped(served, make_client):
    authority, endpoint = served
    make_client(endpoint, WORLD, key=b"x" * 32).send(Register(0, RECOVERABLE, DIGEST))
    clients = {rank: make_client(endpoint, rank) for rank in range(WORLD)}
    assert _register_clients(clients) == {rank: {frozenset({FAILED})} for rank in range(WORLD)}
    assert authority.phase is Phase.SERVING


def test_authenticated_objects_that_are_not_messages_are_refused_without_ending_registration(
    served, make_client
):
    authority, endpoint = served
    sender = _dealer(endpoint)
    try:
        sender.put({"type": "Nope"})
        assert isinstance(sender.get(timeout=5.0), Abort), "the sender was not refused"
        # A message of more than one frame is dropped without stopping the authority.
        sender.socket.send_multipart([_signed(StatusQuery()), b"extra"])
        sender.put(StatusQuery())
        assert sender.get(timeout=5.0).phase == "REGISTERING"
    finally:
        sender.close()
    assert authority.phase is Phase.REGISTERING
    clients = {rank: make_client(endpoint, rank) for rank in range(WORLD)}
    assert _register_clients(clients) == {rank: {frozenset({FAILED})} for rank in range(WORLD)}
    assert authority.phase is Phase.SERVING


@pytest.mark.parametrize(
    "reply, error",
    [(_signed(Commit(1), b"x" * 32), "unauthenticated"), (_signed("Commit"), "malformed")],
    ids=["another_key", "not_a_message"],
)
def test_client_rejects_unauthenticated_and_malformed_messages(tmp_path, reply, error):
    endpoint = f"ipc://{tmp_path}/fake"
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.setsockopt(zmq.LINGER, 0)
    router.bind(endpoint)
    client = RecoveryClient(endpoint, KEY, 0)
    try:
        client.send(Stopped(0, (FAILED,)))
        route, _ = router.recv_multipart()
        router.send_multipart([route, reply])
        with pytest.raises(RecoveryError, match=error):
            client.wait(Commit, time.monotonic() + 5.0)
    finally:
        client.close()
        router.close()
        context.term()


def test_server_probe_reads_the_status_and_reports_unprotected_without_an_answer(served):
    from tensorrt_llm.serve.fault_tolerance_status import RecoveryStatusProbe, health_state

    authority, endpoint = served
    probe = RecoveryStatusProbe(endpoint, KEY, timeout_s=2.0, max_age_s=0.0)
    try:
        status = probe.status()
        assert status is not None and status.phase == "REGISTERING"
        assert health_state(status) == "starting"
        # A probe with the wrong key gets no answer, so it reports nothing rather than a stale state.
        stranger = RecoveryStatusProbe(endpoint, b"x" * 32, timeout_s=0.2, max_age_s=0.0)
        try:
            assert stranger.status() is None
            assert health_state(stranger.status()) == "unprotected"
        finally:
            stranger.close()
    finally:
        probe.close()
    assert authority.phase is Phase.REGISTERING


def test_server_probe_answers_within_its_timeout_while_the_authority_is_down(tmp_path):
    from tensorrt_llm.serve.fault_tolerance_status import RecoveryStatusProbe

    probe = RecoveryStatusProbe(f"ipc://{tmp_path}/down", KEY, timeout_s=0.5, max_age_s=0.0)
    try:
        start = time.monotonic()
        assert probe.status() is None
        assert time.monotonic() - start < 2.0
        # The first query still waits to be sent: later ones fail at once rather than queue behind it.
        for _ in range(3):
            start = time.monotonic()
            assert probe.status() is None
            assert time.monotonic() - start < 0.25
    finally:
        probe.close()


@pytest.mark.parametrize(
    "phase,state",
    [
        ("REGISTERING", "starting"),
        ("SERVING", "healthy"),
        ("STOPPING", "recovering"),
        ("PREPARING", "recovering"),
        ("INSTALLING", "recovering"),
        ("RECOVERED", "degraded"),
        ("ABORTED", "terminal"),
        ("A_PHASE_THIS_SERVER_DOES_NOT_KNOW", "unknown"),
    ],
)
def test_health_states_follow_the_authority_phase(phase, state):
    from tensorrt_llm.serve.fault_tolerance_status import SERVING_STATES, health_state

    assert health_state(Status(phase, 0, (0, 1), (), 0, (), 0, "")) == state
    assert (state in SERVING_STATES) == (state in ("healthy", "degraded"))


def test_status_body_reports_live_ranks_stages_and_failed_requests():
    from tensorrt_llm.serve.fault_tolerance_status import status_body

    status = Status("RECOVERED", 1, (0, 1), NODE, 1, (40_000, 6_000, 18_000), 256, "")
    assert status_body(status) == {
        "state": "degraded",
        "generation": 1,
        "live_ranks": [0, 1],
        "failed_ranks": [2, 3],
        "recoveries": 1,
        "last_recovery_stage_ms": {"stop": 40.0, "agree": 6.0, "install": 18.0},
        "failed_requests": 256,
    }
    assert status_body(None) == {"state": "unprotected"}
    aborted = Status("ABORTED", 1, (0, 1), NODE, 1, (), 0, "rank 2 (pid 7 on node-a) is hung")
    assert status_body(aborted)["state"] == "terminal" and "abort_reason" not in status_body(
        aborted
    )


@pytest.mark.parametrize("mode,body", [("1", b'{"state":"unprotected"}'), (None, b"")])
def test_health_reports_unprotected_when_only_detection_is_on(monkeypatch, mode, body):
    import asyncio

    from tensorrt_llm.serve.openai_server import OpenAIServer

    if mode is None:
        monkeypatch.delenv("TLLM_FAULT_TOLERANCE_MODE", raising=False)
    else:
        monkeypatch.setenv("TLLM_FAULT_TOLERANCE_MODE", mode)
    server = object.__new__(OpenAIServer)
    server._check_health = lambda: True
    server._recovery_status = None

    response = asyncio.run(server.health())

    assert response.status_code == 200 and response.body == body

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recovery authority state machine, wire protocol, and client."""

import hmac
import threading
import time

import pytest
import zmq

from tensorrt_llm._utils import peer_loss_recovery_enabled
from tensorrt_llm.executor.recovery import authority as authority_module
from tensorrt_llm.executor.recovery import client as recovery_client
from tensorrt_llm.executor.recovery import protocol
from tensorrt_llm.executor.recovery.client import (
    RecoveryAborted,
    RecoveryClient,
    RecoveryError,
    RecoveryTimeout,
)
from tensorrt_llm.executor.recovery.protocol import (
    Abort,
    Commit,
    Install,
    Installed,
    Propose,
    Ready,
    Register,
    Registered,
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


def _route(rank: int) -> bytes:
    return f"rank-{rank}".encode()


def _register_all(authority, recoverable=lambda rank: RECOVERABLE):
    out = []
    for rank in range(WORLD):
        assert authority.phase is Phase.REGISTERING
        out = authority.handle(_route(rank), Register(rank, recoverable(rank)), 0.0)
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


def test_propose_relays_rank_0s_placement():
    authority = _registered()
    authority.handle(_route(1), Stopped(1, (FAILED,)), 1.0)
    authority.handle(_route(0), Stopped(0, (FAILED,), placement="rank0"), 1.0)
    out = authority.handle(_route(3), Stopped(3, (FAILED,)), 1.0)
    assert out == [(_route(rank), Propose(1, (FAILED,), SURVIVORS, "rank0")) for rank in SURVIVORS]


def test_a_placement_from_another_rank_aborts():
    authority = _registered()
    _assert_aborted(
        authority.handle(_route(1), Stopped(1, (FAILED,), placement="rank1"), 1.0), authority
    )


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


@pytest.mark.parametrize("failed", [(0,), (3,), (1, FAILED), (0, 1), (1, 2, 3), ()])
def test_unadmitted_loss_aborts_every_worker(failed):
    authority = _registered()
    out = authority.handle(_route(1), Stopped(1, failed), 1.0)
    _assert_aborted(out, authority)
    assert {route for route, _ in out} == {_route(rank) for rank in range(WORLD)}


@pytest.mark.parametrize(
    "route, message",
    [
        (_route(3), Stopped(3, (1,))),  # second distinct failure
        (_route(0), Stopped(0, NODE)),  # a different failed set
        (_route(FAILED), Stopped(FAILED, (FAILED,))),  # departed rank still talking
        (_route(1), Stopped(3, (FAILED,))),  # rank spoofing
        (_route(1), Stopped(1, (FAILED,))),  # duplicate report
        (_route(1), Ready(1, 1, "digest")),  # Ready before Propose
        (_route(1), Commit(1)),  # authority-only message from a worker
        (_route(1), Register(1, RECOVERABLE)),  # registration after it completed
        (b"stranger", Stopped(1, (FAILED,))),  # unregistered peer
    ],
)
def test_protocol_violations_abort(route, message):
    authority = _registered()
    authority.handle(_route(1), Stopped(1, (FAILED,)), 1.0)
    out = authority.handle(route, message, 1.0)
    _assert_aborted(out, authority)
    assert (route, Abort(authority.abort_reason)) in out


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
        (_route(1), Register(WORLD, RECOVERABLE)),  # rank above the world
        (_route(1), Register(-1, RECOVERABLE)),  # negative rank
        (_route(1), Register(0, RECOVERABLE)),  # duplicate rank
        (_route(0), Stopped(0, (FAILED,))),  # message before registration completed
    ],
)
def test_registration_violations_abort(route, message):
    authority = RecoveryAuthority(WORLD)
    authority.handle(_route(0), Register(0, RECOVERABLE), 0.0)
    _assert_aborted(authority.handle(route, message, 0.0), authority)


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


def test_second_loss_after_recovery_aborts():
    authority = _registered()
    _report(authority, Stopped, failed_ranks=(FAILED,))
    _report(authority, Ready, generation=1, placement_digest="digest")
    _report(authority, Installed, generation=1)
    _assert_aborted(authority.handle(_route(0), Stopped(0, (1,)), 5.0), authority)


def test_wire_roundtrip_and_authentication():
    route = _route(1)
    for message in (
        Register(1, ((1,), NODE)),
        Registered(((1,), NODE)),
        Stopped(1, NODE),
        Stopped(0, NODE, "AAEAAg=="),
        Propose(3, NODE, (0, 1)),
        Propose(3, NODE, (0, 1), "AAEAAg=="),
        Abort("lost"),
        StatusQuery(),
        Status("RECOVERED", 1, (0, 1), NODE, 1, (500_000, 250_000, 250_000), 2, ""),
        Status("SERVING", 0, (0, 1), (), 0, (), 0, ""),
    ):
        frames = protocol.encode(message, KEY, route)
        assert protocol.authenticate(frames, KEY, route)
        assert protocol.decode(frames[0]) == message
    frames = protocol.encode(Install(1), KEY, route)
    assert not protocol.authenticate(frames, b"x" * 32, route)
    assert not protocol.authenticate([frames[0] + b" ", frames[1]], KEY, route)
    # A frame replayed on another connection does not authenticate.
    assert not protocol.authenticate(frames, KEY, _route(2))


@pytest.mark.parametrize(
    "payload",
    [
        b"not json",
        b"[]",
        b'{"type": "Nope"}',
        b'{"type": "Install"}',
        b'{"type": "Install", "generation": "1"}',
        b'{"type": "Install", "generation": true}',
        b'{"type": "Install", "generation": 1, "extra": 0}',
        b'{"type": "Propose", "generation": 1, "failed_ranks": [2], "survivors": [0, "1"]}',
        b'{"type": "Propose", "generation": 1, "failed_ranks": 2, "survivors": [0, 1]}',
        b'{"type": "Register", "rank": 1, "recoverable_domains": [1, 2]}',
        b'{"type": "Register", "rank": 1, "recoverable_domains": [[1], [2, "3"]]}',
        b'{"type": "Install", "generation": ' + b"9" * 5000 + b"}",
        b"[" * 100000 + b"]" * 100000,
    ],
)
def test_malformed_payloads_are_rejected(payload):
    with pytest.raises(protocol.ProtocolError):
        protocol.decode(payload)


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
    context = zmq.Context()
    socket = authority_module.router_socket(context, endpoint)
    stop = threading.Event()
    thread = threading.Thread(
        target=authority_module.serve, args=(authority, socket, KEY, stop, 0.01), daemon=True
    )
    thread.start()
    yield authority, endpoint
    stop.set()
    thread.join(timeout=5)
    socket.close()
    context.term()


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


def _worker_recovery(client, rank, results) -> None:
    deadline = time.monotonic() + 10.0
    try:
        client.send(Stopped(rank, (FAILED,)))
        proposal = client.wait(Propose, deadline)
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


def test_stopping_during_a_recovery_aborts_it(tmp_path, make_client):
    authority = RecoveryAuthority(WORLD, recovery_timeout=10.0)
    endpoint = f"ipc://{tmp_path}/authority"
    context = zmq.Context()
    socket = authority_module.router_socket(context, endpoint)
    stop = threading.Event()
    thread = threading.Thread(
        target=authority_module.serve, args=(authority, socket, KEY, stop, 0.01)
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
        socket.close()
        context.term()


def test_frames_signed_with_another_key_are_dropped(served, make_client):
    authority, endpoint = served
    make_client(endpoint, WORLD, key=b"x" * 32).send(Register(0, RECOVERABLE))
    clients = {rank: make_client(endpoint, rank) for rank in range(WORLD)}
    assert _register_clients(clients) == {rank: {frozenset({FAILED})} for rank in range(WORLD)}
    assert authority.phase is Phase.SERVING


def test_authenticated_malformed_frames_abort(served):
    authority, endpoint = served
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.LINGER, 0)
    route = b"malformed"
    socket.setsockopt(zmq.ROUTING_ID, route)
    socket.connect(endpoint)
    payload = b'{"type": "Nope"}'
    tag = hmac.new(KEY, bytes([len(route)]) + route + payload, "sha256").digest()
    socket.send_multipart([payload, tag])
    deadline = time.monotonic() + 5.0
    while authority.phase is not Phase.ABORTED:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    socket.close()
    context.term()


def test_client_rejects_unauthenticated_frames(tmp_path):
    endpoint = f"ipc://{tmp_path}/fake"
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.bind(endpoint)
    client = RecoveryClient(endpoint, KEY, 0)
    client.send(Stopped(0, (FAILED,)))
    route, *_ = router.recv_multipart()
    router.send_multipart([route, *protocol.encode(Commit(1), b"x" * 32, route)])
    with pytest.raises(RecoveryError, match="unauthenticated"):
        client.wait(Commit, time.monotonic() + 5.0)
    client.close()
    router.close()
    context.term()


def test_server_probe_reads_the_status_and_reports_unavailable_without_an_answer(served):
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
            assert health_state(stranger.status()) == "unavailable"
        finally:
            stranger.close()
    finally:
        probe.close()
    assert authority.phase is Phase.REGISTERING


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
    assert status_body(None) == {"state": "unavailable"}

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
"""Recovery authority state machine, wire protocol, and client."""

import hmac
import threading
import time

import pytest
import zmq

from tensorrt_llm.executor import recovery
from tensorrt_llm.executor.recovery import (
    Abort,
    Commit,
    Install,
    Installed,
    Propose,
    Ready,
    Register,
    Registered,
    Stopped,
    protocol,
)
from tensorrt_llm.executor.recovery import authority as authority_module

pytestmark = pytest.mark.cpu_only

Phase = authority_module.Phase
RecoveryAuthority = authority_module.RecoveryAuthority

WORLD = 4
FAILED = 2
SURVIVORS = (0, 1, 3)
RECOVERABLE = (1, 2, 3)
KEY = b"k" * 32


def _route(rank: int) -> bytes:
    return f"rank-{rank}".encode()


def _register_all(authority, recoverable=lambda rank: RECOVERABLE):
    out = []
    for rank in range(WORLD):
        assert authority.phase is Phase.REGISTERING
        out = authority.handle(_route(rank), Register(rank, 100 + rank, recoverable(rank)), 0.0)
    return out


def _registered(timeout: float = 10.0) -> RecoveryAuthority:
    authority = RecoveryAuthority(WORLD, frozenset({1, FAILED}), recovery_timeout=timeout)
    out = _register_all(authority)
    assert authority.phase is Phase.SERVING
    assert out == [(_route(rank), Registered((1, FAILED))) for rank in range(WORLD)]
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


def test_registration_completes_only_when_every_rank_registered():
    authority = RecoveryAuthority(WORLD)
    for rank in range(WORLD - 1):
        assert authority.handle(_route(rank), Register(rank, rank, RECOVERABLE), 0.0) == []
    assert authority.phase is Phase.REGISTERING


def test_admission_is_what_every_worker_can_recover_and_never_rank0():
    authority = RecoveryAuthority(WORLD)
    out = _register_all(authority, lambda rank: (0, 1, 2, 3) if rank != 3 else (0, 2, 3))
    assert authority.admitted_ranks == frozenset({2, 3})
    assert all(message == Registered((2, 3)) for _, message in out)


def test_admitting_an_uncovered_rank_aborts_registration():
    authority = RecoveryAuthority(WORLD, frozenset({1, FAILED}))
    out = _register_all(authority, lambda rank: (FAILED, 3))
    _assert_aborted(out, authority)
    assert "[1]" in authority.abort_reason


def test_idle_loss_recovers():
    authority = _registered()
    authority.handle(_route(0), Stopped(0, FAILED), 1.0)
    assert authority.phase is Phase.STOPPING
    assert authority.deadline == pytest.approx(11.0)

    out = _report(authority, Stopped, SURVIVORS[1:], failed_rank=FAILED)
    assert out == [(_route(rank), Propose(1, FAILED, SURVIVORS)) for rank in SURVIVORS]
    out = _report(authority, Ready, generation=1, placement_digest="digest")
    assert out == [(_route(rank), Install(1)) for rank in SURVIVORS]
    out = _report(authority, Installed, generation=1)
    assert out == [(_route(rank), Commit(1)) for rank in SURVIVORS]
    assert authority.phase is Phase.RECOVERED and authority.generation == 1
    assert authority.deadline is None and authority.expire(1e9) == []


@pytest.mark.parametrize("failed", [0, 3])
def test_unadmitted_loss_aborts_every_worker(failed):
    authority = _registered()
    out = authority.handle(_route(1), Stopped(1, failed), 1.0)
    _assert_aborted(out, authority)
    assert {route for route, _ in out} == {_route(rank) for rank in range(WORLD)}


@pytest.mark.parametrize(
    "route, message",
    [
        (_route(3), Stopped(3, 1)),  # second distinct failure
        (_route(FAILED), Stopped(FAILED, FAILED)),  # departed rank still talking
        (_route(1), Stopped(3, FAILED)),  # rank spoofing
        (_route(1), Stopped(1, FAILED)),  # duplicate report
        (_route(1), Ready(1, 1, "digest")),  # Ready before Propose
        (_route(1), Commit(1)),  # authority-only message from a worker
        (_route(1), Register(1, 101, RECOVERABLE)),  # registration after it completed
        (b"stranger", Stopped(1, FAILED)),  # unregistered peer
    ],
)
def test_protocol_violations_abort(route, message):
    authority = _registered()
    authority.handle(_route(1), Stopped(1, FAILED), 1.0)
    out = authority.handle(route, message, 1.0)
    _assert_aborted(out, authority)
    assert (route, Abort(authority.abort_reason)) in out


def test_placement_digests_must_agree():
    authority = _registered()
    _report(authority, Stopped, failed_rank=FAILED)
    authority.handle(_route(0), Ready(0, 1, "a"), 2.0)
    _assert_aborted(authority.handle(_route(1), Ready(1, 1, "b"), 2.0), authority)


@pytest.mark.parametrize("message_type", [Ready, Installed])
def test_reports_for_another_generation_abort(message_type):
    authority = _registered()
    _report(authority, Stopped, failed_rank=FAILED)
    if message_type is Installed:
        _report(authority, Ready, generation=1, placement_digest="digest")
    fields = {"placement_digest": "digest"} if message_type is Ready else {}
    out = authority.handle(_route(0), message_type(0, generation=2, **fields), 2.0)
    _assert_aborted(out, authority)


@pytest.mark.parametrize(
    "registration",
    [Register(WORLD, 1, RECOVERABLE), Register(-1, 1, RECOVERABLE)],
    ids=["above_world", "negative"],
)
def test_invalid_registrations_abort(registration):
    authority = RecoveryAuthority(WORLD)
    _assert_aborted(authority.handle(_route(0), registration, 0.0), authority)


def test_duplicate_registrations_abort():
    authority = RecoveryAuthority(WORLD)
    authority.handle(_route(0), Register(0, 1, RECOVERABLE), 0.0)
    _assert_aborted(authority.handle(_route(1), Register(0, 2, RECOVERABLE), 0.0), authority)


def test_messages_before_registration_completes_abort():
    authority = RecoveryAuthority(WORLD, frozenset({FAILED}))
    authority.handle(_route(0), Register(0, 1, RECOVERABLE), 0.0)
    _assert_aborted(authority.handle(_route(0), Stopped(0, FAILED), 0.0), authority)


@pytest.mark.parametrize("phase_reports", [1, 2, 3], ids=["stopping", "preparing", "installing"])
def test_recovery_deadline_aborts_and_aborted_is_terminal(phase_reports):
    authority = _registered(timeout=5.0)
    authority.handle(_route(0), Stopped(0, FAILED), 1.0)
    if phase_reports >= 2:
        _report(authority, Stopped, SURVIVORS[1:], failed_rank=FAILED)
    if phase_reports >= 3:
        _report(authority, Ready, generation=1, placement_digest="digest")
    assert authority.expire(5.9) == []
    _assert_aborted(authority.expire(6.0), authority)
    reply = authority.handle(_route(1), Stopped(1, FAILED), 7.0)
    assert reply == [(_route(1), Abort(authority.abort_reason))]
    assert authority.expire(1e9) == []


def test_second_loss_after_recovery_aborts():
    authority = _registered()
    _report(authority, Stopped, failed_rank=FAILED)
    _report(authority, Ready, generation=1, placement_digest="digest")
    _report(authority, Installed, generation=1)
    _assert_aborted(authority.handle(_route(0), Stopped(0, 1), 5.0), authority)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(world_size=1),
        dict(world_size=WORLD, admitted_ranks=frozenset({0})),
        dict(world_size=WORLD, admitted_ranks=frozenset({WORLD})),
        dict(world_size=WORLD, recovery_timeout=0),
    ],
)
def test_invalid_construction(kwargs):
    with pytest.raises(ValueError):
        RecoveryAuthority(**kwargs)


def test_wire_roundtrip_and_authentication():
    for message in (Propose(3, 2, (0, 1, 3)), Registered((1, 2)), Abort("lost")):
        frames = protocol.encode(message, KEY)
        assert protocol.authenticate(frames, KEY)
        assert protocol.decode(frames[0]) == message
    frames = protocol.encode(Install(1), KEY)
    assert not protocol.authenticate(frames, b"x" * 32)
    assert not protocol.authenticate([frames[0] + b" ", frames[1]], KEY)


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
        b'{"type": "Propose", "generation": 1, "failed_rank": 2, "survivors": [0, "1"]}',
    ],
)
def test_malformed_payloads_are_rejected(payload):
    with pytest.raises(protocol.ProtocolError):
        protocol.decode(payload)


def test_load_key_validates_the_environment(monkeypatch):
    env = recovery.client.AUTHORITY_KEY_ENV
    monkeypatch.delenv(env, raising=False)
    with pytest.raises(ValueError, match="not set"):
        recovery.client.load_key()
    monkeypatch.setenv(env, "zz")
    with pytest.raises(ValueError, match="hex"):
        recovery.client.load_key()
    monkeypatch.setenv(env, "00" * 8)
    with pytest.raises(ValueError, match="16 bytes"):
        recovery.client.load_key()
    monkeypatch.setenv(env, KEY.hex())
    assert recovery.client.load_key() == KEY


def test_client_from_env(monkeypatch):
    monkeypatch.setenv("TLLM_FAULT_TOLERANCE_MODE", "1")
    monkeypatch.delenv("TLLM_FAULT_TOLERANCE_AUTHORITY", raising=False)
    assert recovery.RecoveryClient.from_env(0) is None  # detection-only mode
    monkeypatch.setenv("TLLM_FAULT_TOLERANCE_AUTHORITY", "tcp://127.0.0.1:1")
    monkeypatch.delenv("TLLM_FAULT_TOLERANCE_AUTHORITY_KEY", raising=False)
    with pytest.raises(ValueError, match="KEY"):
        recovery.RecoveryClient.from_env(0)
    monkeypatch.delenv("TLLM_FAULT_TOLERANCE_MODE")
    with pytest.raises(ValueError, match="TLLM_FAULT_TOLERANCE_MODE=1"):
        recovery.RecoveryClient.from_env(0)


# End to end over ZeroMQ.


@pytest.fixture
def served(tmp_path):
    authority = RecoveryAuthority(WORLD, frozenset({FAILED}), recovery_timeout=10.0)
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


def _clients(endpoint):
    return {rank: recovery.RecoveryClient(endpoint, KEY, rank) for rank in range(WORLD)}


def _register_clients(clients):
    admitted = {}

    def register(rank):
        admitted[rank] = clients[rank].register(RECOVERABLE, time.monotonic() + 10.0)

    threads = [threading.Thread(target=register, args=(rank,)) for rank in clients]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return admitted


def _worker_recovery(client, rank, results) -> None:
    deadline = time.monotonic() + 10.0
    try:
        client.send(Stopped(rank, FAILED))
        proposal = client.wait(Propose, deadline)
        client.send(Ready(rank, proposal.generation, "digest"))
        client.wait(Install, deadline)
        client.send(Installed(rank, proposal.generation))
        results[rank] = client.wait(Commit, deadline)
    except recovery.RecoveryError as error:
        results[rank] = error


def test_end_to_end_over_zmq(served):
    authority, endpoint = served
    clients = _clients(endpoint)
    assert _register_clients(clients) == {rank: frozenset({FAILED}) for rank in range(WORLD)}

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
    for client in clients.values():
        client.close()


def test_abort_reaches_every_worker(served):
    authority, endpoint = served
    clients = _clients(endpoint)
    _register_clients(clients)
    clients[1].send(Stopped(1, 3))  # rank 3 is not admitted
    for rank in (0, 3):
        with pytest.raises(recovery.RecoveryAborted):
            clients[rank].wait(Propose, time.monotonic() + 5.0)
    with pytest.raises(recovery.RecoveryTimeout):
        clients[0].wait(Propose, time.monotonic() + 0.05)
    for client in clients.values():
        client.close()


def test_frames_signed_with_another_key_are_dropped(served):
    authority, endpoint = served
    intruder = recovery.RecoveryClient(endpoint, b"x" * 32, 0)
    intruder.send(Register(0, 1, RECOVERABLE))
    time.sleep(0.2)
    assert authority.phase is Phase.REGISTERING and not authority.admitted_ranks
    intruder.close()


def test_authenticated_malformed_frames_abort(served):
    authority, endpoint = served
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(endpoint)
    payload = b'{"type": "Nope"}'
    socket.send_multipart([payload, protocol.encode(Install(1), KEY)[1]])
    socket.send_multipart([payload, hmac.new(KEY, payload, "sha256").digest()])
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
    client = recovery.RecoveryClient(endpoint, KEY, 0)
    client.send(Stopped(0, FAILED))
    route, *_ = router.recv_multipart()
    router.send_multipart([route, *protocol.encode(Commit(1), b"x" * 32)])
    with pytest.raises(recovery.RecoveryError, match="unauthenticated"):
        client.wait(Commit, time.monotonic() + 5.0)
    client.close()
    router.close()
    context.term()

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
"""Recovery authority state machine, wire protocol, client, and frontend admission."""

# The recovery package has no torch dependency; load it from source so these tests do
# not import the heavy ``tensorrt_llm`` package.
import importlib
import importlib.util
import pathlib
import socket
import sys
import threading
import time

import pytest

pytestmark = pytest.mark.cpu_only

_PACKAGE_DIR = (
    pathlib.Path(__file__).resolve().parents[3] / "tensorrt_llm" / "executor" / "recovery"
)
_spec = importlib.util.spec_from_file_location(
    "_recovery_under_test",
    _PACKAGE_DIR / "__init__.py",
    submodule_search_locations=[str(_PACKAGE_DIR)],
)
recovery = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = recovery
_spec.loader.exec_module(recovery)
authority_module = importlib.import_module(f"{_spec.name}.authority")
protocol = importlib.import_module(f"{_spec.name}.protocol")

Phase = authority_module.Phase
RecoveryAuthority = authority_module.RecoveryAuthority
Abort = recovery.Abort
AdmissionClosed = recovery.AdmissionClosed
CloseAdmission = recovery.CloseAdmission
Commit = recovery.Commit
Install = recovery.Install
Installed = recovery.Installed
Propose = recovery.Propose
Ready = recovery.Ready
Register = recovery.Register
ReopenAdmission = recovery.ReopenAdmission
Stopped = recovery.Stopped

WORLD = 4
FAILED = 2
SURVIVORS = (0, 1, 3)
FRONTEND = b"frontend"
KEY = b"k" * 32


def _route(rank: int) -> bytes:
    return f"worker-{rank}".encode()


def _registered(timeout: float = 10.0) -> RecoveryAuthority:
    authority = RecoveryAuthority(WORLD, frozenset({1, FAILED}), phase_timeout=timeout)
    for rank in range(WORLD):
        assert authority.handle(_route(rank), Register("worker", rank, 100 + rank), 0.0) == []
    assert authority.phase is Phase.REGISTERING
    authority.handle(FRONTEND, Register("frontend", 0, 99), 0.0)
    assert authority.phase is Phase.SERVING
    return authority


def _stopped(authority: RecoveryAuthority, now: float = 1.0) -> None:
    out = authority.handle(_route(0), Stopped(0, FAILED), now)
    assert out == [(FRONTEND, CloseAdmission())]
    for rank in SURVIVORS[1:]:
        assert authority.handle(_route(rank), Stopped(rank, FAILED), now) == []
    assert authority.phase is Phase.STOPPING


def _assert_aborted(out, authority: RecoveryAuthority) -> None:
    assert authority.phase is Phase.ABORTED
    assert authority.deadline is None
    assert out and all(isinstance(message, Abort) for _, message in out)


def test_idle_loss_recovers_and_reopens_admission():
    authority = _registered()
    _stopped(authority)
    assert authority.deadline == pytest.approx(11.0)

    out = authority.handle(FRONTEND, AdmissionClosed(), 2.0)
    assert out == [(_route(rank), Propose(1, FAILED, SURVIVORS)) for rank in SURVIVORS]
    assert authority.phase is Phase.PREPARING
    assert authority.deadline == pytest.approx(12.0)

    for rank in SURVIVORS:
        out = authority.handle(_route(rank), Ready(rank, 1, "digest"), 3.0)
    assert out == [(_route(rank), Install(1)) for rank in SURVIVORS]
    assert authority.phase is Phase.INSTALLING

    for rank in SURVIVORS:
        out = authority.handle(_route(rank), Installed(rank, 1), 4.0)
    assert out == [(_route(rank), Commit(1)) for rank in SURVIVORS] + [
        (FRONTEND, ReopenAdmission(1))
    ]
    assert authority.phase is Phase.RECOVERED
    assert authority.generation == 1
    # Deadlines are cleared once recovery commits.
    assert authority.deadline is None
    assert authority.expire(1e9) == []


def test_identical_retransmissions_are_ignored():
    authority = _registered()
    _stopped(authority)
    assert authority.handle(_route(1), Stopped(1, FAILED), 1.5) == []
    authority.handle(FRONTEND, AdmissionClosed(), 2.0)
    assert authority.handle(FRONTEND, AdmissionClosed(), 2.0) == []
    authority.handle(_route(0), Ready(0, 1, "digest"), 3.0)
    assert authority.handle(_route(0), Ready(0, 1, "digest"), 3.0) == []
    assert authority.phase is Phase.PREPARING


@pytest.mark.parametrize("failed", [0, 3])
def test_unadmitted_loss_aborts_every_peer(failed):
    authority = _registered()
    out = authority.handle(_route(1), Stopped(1, failed), 1.0)
    _assert_aborted(out, authority)
    assert {route for route, _ in out} == {FRONTEND, *(_route(rank) for rank in range(WORLD))}


@pytest.mark.parametrize(
    "route, message",
    [
        (_route(3), Stopped(3, 1)),  # second distinct failure
        (_route(FAILED), Stopped(FAILED, FAILED)),  # departed rank still talking
        (_route(FAILED), Ready(FAILED, 1, "digest")),  # departed rank still talking
        (_route(1), Stopped(3, FAILED)),  # rank spoofing
        (_route(1), Ready(1, 1, "digest")),  # Ready before Propose
        (_route(1), AdmissionClosed()),  # AdmissionClosed from a worker
        (FRONTEND, Commit(1)),  # authority-only message from a peer
        (b"stranger", Stopped(1, FAILED)),  # unregistered peer
    ],
)
def test_protocol_violations_abort(route, message):
    authority = _registered()
    authority.handle(_route(0), Stopped(0, FAILED), 1.0)
    out = authority.handle(route, message, 1.0)
    _assert_aborted(out, authority)
    assert (route, Abort(authority.abort_reason)) in out


def test_conflicting_duplicate_and_digest_mismatch_abort():
    authority = _registered()
    _stopped(authority)
    authority.handle(FRONTEND, AdmissionClosed(), 2.0)
    authority.handle(_route(0), Ready(0, 1, "a"), 3.0)
    _assert_aborted(authority.handle(_route(1), Ready(1, 1, "b"), 3.0), authority)

    authority = _registered()
    _stopped(authority)
    authority.handle(FRONTEND, AdmissionClosed(), 2.0)
    _assert_aborted(authority.handle(_route(0), Ready(0, 2, "a"), 3.0), authority)


def test_messages_before_registration_completes_abort():
    authority = RecoveryAuthority(WORLD, frozenset({FAILED}))
    authority.handle(_route(0), Register("worker", 0, 1), 0.0)
    _assert_aborted(authority.handle(_route(0), Stopped(0, FAILED), 0.0), authority)

    authority = RecoveryAuthority(WORLD, frozenset({FAILED}))
    authority.handle(_route(0), Register("worker", 0, 1), 0.0)
    _assert_aborted(authority.handle(_route(1), Register("worker", 0, 2), 0.0), authority)


def test_phase_deadline_aborts_and_aborted_is_terminal():
    authority = _registered(timeout=5.0)
    _stopped(authority, now=1.0)
    assert authority.expire(5.9) == []
    _assert_aborted(authority.expire(6.0), authority)
    reply = authority.handle(FRONTEND, AdmissionClosed(), 7.0)
    assert reply == [(FRONTEND, Abort(authority.abort_reason))]
    assert authority.expire(1e9) == []


def test_second_loss_after_recovery_aborts():
    authority = _registered()
    _stopped(authority)
    authority.handle(FRONTEND, AdmissionClosed(), 2.0)
    for rank in SURVIVORS:
        authority.handle(_route(rank), Ready(rank, 1, "digest"), 3.0)
    for rank in SURVIVORS:
        authority.handle(_route(rank), Installed(rank, 1), 4.0)
    _assert_aborted(authority.handle(_route(0), Stopped(0, 1), 5.0), authority)


def test_invalid_construction():
    with pytest.raises(ValueError):
        RecoveryAuthority(WORLD, frozenset({0}))
    with pytest.raises(ValueError):
        RecoveryAuthority(WORLD, frozenset({WORLD}))


def test_wire_roundtrip_and_authentication():
    message = Propose(3, 2, (0, 1, 3))
    frames = protocol.encode(message, KEY)
    assert protocol.authenticate(frames, KEY)
    assert protocol.decode(frames[0]) == message
    assert protocol.decode(protocol.encode(CloseAdmission(), KEY)[0]) == CloseAdmission()
    assert not protocol.authenticate(frames, b"x" * 32)
    assert not protocol.authenticate([frames[0] + b" ", frames[1]], KEY)
    for payload in (
        b"not json",
        b'{"type": "Nope"}',
        b'{"type": "Install"}',
        b'{"type": "Install", "generation": "1"}',
        b'{"type": "Install", "generation": true}',
        b'{"type": "Install", "generation": 1, "extra": 0}',
        b"[]",
    ):
        with pytest.raises(protocol.ProtocolError):
            protocol.decode(payload)


def test_client_from_env(monkeypatch):
    monkeypatch.delenv("TLLM_WIDEEP_RECOVERY", raising=False)
    assert recovery.RecoveryClient.from_env("worker", 0) is None
    monkeypatch.setenv("TLLM_WIDEEP_RECOVERY", "tcp://127.0.0.1:1")
    monkeypatch.delenv("TLLM_WIDEEP_RECOVERY_KEY", raising=False)
    with pytest.raises(ValueError):
        recovery.RecoveryClient.from_env("worker", 0)


@pytest.fixture
def served():
    """Run ``authority.serve`` on a loopback port; yield ``(authority, endpoint)``."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    authority = RecoveryAuthority(WORLD, frozenset({FAILED}), phase_timeout=10.0)
    stop = threading.Event()
    thread = threading.Thread(
        target=authority_module.serve,
        args=(authority, f"tcp://127.0.0.1:{port}", KEY, stop, 0.01),
    )
    thread.start()
    yield authority, f"tcp://127.0.0.1:{port}"
    stop.set()
    thread.join()


def _worker_recovery(client, rank: int, results: dict) -> None:
    """The sequence a surviving scheduler runs after observing the loss."""
    deadline = time.monotonic() + 10.0
    try:
        client.send(Stopped(rank, FAILED))
        proposal = client.wait(Propose, deadline)
        client.send(Ready(rank, proposal.generation, "digest"))
        client.wait(Install, deadline)
        client.send(Installed(rank, proposal.generation))
        results[rank] = client.wait(Commit, deadline)
    except (recovery.RecoveryError, TimeoutError) as error:
        results[rank] = error


def test_end_to_end_over_zmq(served):
    authority, endpoint = served
    clients = {
        rank: recovery.RecoveryClient(endpoint, KEY, "worker", rank) for rank in range(WORLD)
    }
    in_flight = threading.Event()
    in_flight.set()
    failures = []
    admission = recovery.FrontendAdmission(
        recovery.RecoveryClient(endpoint, KEY, "frontend", 0),
        is_idle=lambda: not in_flight.is_set(),
        on_failure=failures.append,
        poll_interval=0.01,
    )
    _wait_for(lambda: authority.phase is Phase.SERVING)

    results = {}
    workers = [
        threading.Thread(target=_worker_recovery, args=(clients[rank], rank, results))
        for rank in SURVIVORS
    ]
    for worker in workers:
        worker.start()
    _wait_for(lambda: admission.closed)
    # The frontend withholds AdmissionClosed while a request is in flight.
    time.sleep(0.2)
    assert authority.phase is Phase.STOPPING
    in_flight.clear()
    for worker in workers:
        worker.join()

    assert results == {rank: Commit(1) for rank in SURVIVORS}
    _wait_for(lambda: not admission.closed)
    assert authority.phase is Phase.RECOVERED
    assert failures == []
    admission.close()
    for client in clients.values():
        client.close()


def test_abort_reaches_workers_and_frontend(served):
    authority, endpoint = served
    clients = {
        rank: recovery.RecoveryClient(endpoint, KEY, "worker", rank) for rank in range(WORLD)
    }
    failures = []
    admission = recovery.FrontendAdmission(
        recovery.RecoveryClient(endpoint, KEY, "frontend", 0),
        is_idle=lambda: True,
        on_failure=failures.append,
        poll_interval=0.01,
    )
    _wait_for(lambda: authority.phase is Phase.SERVING)

    clients[1].send(Stopped(1, 3))  # rank 3 is not admitted
    with pytest.raises(recovery.RecoveryAborted):
        clients[0].wait(Propose, time.monotonic() + 5.0)
    _wait_for(lambda: failures)
    assert isinstance(failures[0], recovery.RecoveryAborted)
    with pytest.raises(TimeoutError):
        clients[0].wait(Propose, time.monotonic() + 0.05)
    admission.close()
    for client in clients.values():
        client.close()


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)

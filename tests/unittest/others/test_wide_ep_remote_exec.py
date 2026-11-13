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
"""Tests for the request authentication of the Wide-EP fault tolerance example's remote_exec.py."""

import importlib.util
import json
import os
import secrets
import socket
import threading
from pathlib import Path

import pytest

_REMOTE_EXEC = (
    Path(__file__).resolve().parents[3] / "examples/wide_ep/fault_tolerance/remote_exec.py"
)
_spec = importlib.util.spec_from_file_location("remote_exec", _REMOTE_EXEC)
remote_exec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(remote_exec)


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("REMOTE_EXEC_KEY", secrets.token_hex(32))
    with remote_exec._Server(("127.0.0.1", 0), remote_exec._Handler) as server:
        monkeypatch.setenv("REMOTE_EXEC_PORT", str(server.server_address[1]))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield server
        server.shutdown()


def _send(request: bytes) -> tuple[bytes, bytes]:
    """Send ``request`` on a new connection, and return the nonce and everything the server sent."""
    with socket.create_connection(
        ("127.0.0.1", int(os.environ["REMOTE_EXEC_PORT"])), timeout=10
    ) as sock:
        reply = sock.makefile("rb")
        nonce = reply.readline()
        sock.sendall(request)
        return nonce, reply.read()


def test_signed_request_runs(server, tmp_path, capfd):
    marker = tmp_path / "ran"
    assert remote_exec.run("127.0.0.1", f"echo ok > {marker}; echo out; exit 3") == 3
    assert marker.read_text() == "ok\n"
    assert capfd.readouterr().out == "out\n"


def test_replayed_request_is_rejected(server, tmp_path):
    marker = tmp_path / "runs"
    payload = json.dumps(f"echo run >> {marker}").encode()
    with socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=10) as sock:
        reply = sock.makefile("rb")
        nonce = reply.readline().rstrip(b"\n")
        request = remote_exec._sign(nonce, payload) + b" " + payload + b"\n"
        sock.sendall(request)
        reply.read()
    assert marker.read_text() == "run\n"

    replay_nonce, output = _send(request)
    assert replay_nonce.rstrip(b"\n") != nonce
    assert output == b""
    assert marker.read_text() == "run\n"


def test_request_with_wrong_key_is_rejected(server, tmp_path):
    marker = tmp_path / "ran"
    payload = json.dumps(f"touch {marker}").encode()
    _, output = _send(b"0" * 64 + b" " + payload + b"\n")
    assert output == b""
    assert not marker.exists()


def test_serve_refuses_loopback(monkeypatch):
    monkeypatch.setenv("REMOTE_EXEC_PORT", "0")
    with pytest.raises(SystemExit, match="loopback"):
        remote_exec.serve("localhost")

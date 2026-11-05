# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``/health`` of a server with WideEP fault tolerance."""

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from tensorrt_llm.executor.recovery.protocol import Status
from tensorrt_llm.serve import openai_server
from tensorrt_llm.serve.openai_server import OpenAIServer

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("mode,body", [("1", b'{"state":"unprotected"}'), (None, b"")])
def test_health_reports_unprotected_when_only_detection_is_on(monkeypatch, mode, body):
    if mode is None:
        monkeypatch.delenv("TLLM_FAULT_TOLERANCE_MODE", raising=False)
    else:
        monkeypatch.setenv("TLLM_FAULT_TOLERANCE_MODE", mode)
    server = object.__new__(OpenAIServer)
    server._check_health = lambda: True
    server._recovery_status = None

    response = asyncio.run(server.health())

    assert response.status_code == 200 and response.body == body


@pytest.mark.parametrize(
    "phase,code",
    [
        ("SERVING", 200),
        ("DEGRADED", 200),
        ("STOPPING", 503),
        ("ABORTED", 503),
        ("A_NEWER_PHASE", 200),
    ],
)
def test_health_returns_503_only_while_recovering_or_terminal(phase, code):
    """A phase from an authority of another version reads ``unknown``: the server still serves."""
    server = object.__new__(OpenAIServer)
    server._check_health = lambda: True
    answer = Future()
    answer.set_result(Status(phase, 1, (0, 1), (2,), 1, (), 0, ""))
    server._recovery_status = SimpleNamespace(status_future=lambda: answer)

    response = asyncio.run(server.health())

    assert response.status_code == code


def test_lifespan_teardown_closes_the_recovery_status_probe(monkeypatch):
    probe = Mock()
    monkeypatch.setattr(openai_server.RecoveryStatusProbe, "from_env", lambda: probe)
    monkeypatch.setattr(OpenAIServer, "_init_llm", lambda self, chat_template=None: None)
    monkeypatch.setattr(OpenAIServer, "register_routes", lambda self: None)
    generator = Mock(args=SimpleNamespace())
    server = OpenAIServer(
        generator, "model", tool_parser=None, server_role=None, metadata_server_cfg=None
    )

    with TestClient(server.app):
        probe.close.assert_not_called()

    probe.close.assert_called_once_with()
    generator.shutdown.assert_called_once_with()

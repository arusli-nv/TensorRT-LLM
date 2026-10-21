# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``/health`` of a server with WideEP fault tolerance."""

import asyncio
from types import SimpleNamespace

import pytest

from tensorrt_llm.executor.recovery.protocol import Status
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
        ("RECOVERED", 200),
        ("STOPPING", 503),
        ("ABORTED", 503),
        ("A_NEWER_PHASE", 200),
    ],
)
def test_health_returns_503_only_while_recovering_or_terminal(phase, code):
    """A phase from an authority of another version reads ``unknown``: the server still serves."""
    server = object.__new__(OpenAIServer)
    server._check_health = lambda: True
    server._recovery_status = SimpleNamespace(
        status=lambda: Status(phase, 1, (0, 1), (2,), 1, (), 0, "")
    )

    response = asyncio.run(server.health())

    assert response.status_code == code

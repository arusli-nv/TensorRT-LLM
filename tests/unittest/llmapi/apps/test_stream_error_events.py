# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A streamed request that fails after its first chunk ends with an SSE error event."""

import asyncio
import json
from http import HTTPStatus

import pytest

from tensorrt_llm.executor.utils import (
    PEER_LOSS_RETRYABLE_CODE,
    RequestError,
    peer_loss_retryable_message,
)
from tensorrt_llm.serve.openai_server import (
    OpenAIServer,
    _end_stream_on_error,
    _stream_error_events,
)

pytestmark = pytest.mark.cpu_only


def test_failed_stream_ends_with_an_error_event_then_done():
    events = _stream_error_events(RuntimeError("Attention DP rank 2 failed; retry it"))

    assert events[-1] == "data: [DONE]\n\n"
    assert events[0].startswith("data: ") and events[0].endswith("\n\n")
    # OpenAI clients raise on an event whose data carries an "error" object.
    error = json.loads(events[0][len("data: ") :])["error"]
    assert error["message"] == "Attention DP rank 2 failed; retry it"
    assert error["type"] == "server_error"
    assert error["code"] is None


def _drain(events):
    async def collect():
        return [event async for event in _end_stream_on_error(events)]

    return asyncio.run(collect())


def test_a_stream_failing_mid_way_ends_with_one_error_event_and_one_done():
    """Chat, harmony chat and Responses streams: the events sent so far, then the error and [DONE]."""

    async def failing():
        yield "data: first\n\n"
        raise RuntimeError("Attention DP rank 2 failed; retry it")

    events = _drain(failing())

    assert events[0] == "data: first\n\n"
    assert events[1:] == _stream_error_events(RuntimeError("Attention DP rank 2 failed; retry it"))
    assert events.count("data: [DONE]\n\n") == 1


def test_a_stream_that_completes_passes_through_unchanged():
    async def completing():
        yield "data: first\n\n"
        yield "data: [DONE]\n\n"

    assert _drain(completing()) == ["data: first\n\n", "data: [DONE]\n\n"]


def test_a_peer_loss_failure_streams_a_retryable_code(monkeypatch):
    monkeypatch.setenv("TLLM_FAULT_TOLERANCE_MODE", "1")
    message = peer_loss_retryable_message("Attention DP rank 2 failed; retry it")

    (event, _) = _stream_error_events(RequestError(message))

    error = json.loads(event[len("data: ") :])["error"]
    assert error["code"] == PEER_LOSS_RETRYABLE_CODE and error["message"] == message


def test_a_peer_loss_failure_is_a_503_with_retry_after(monkeypatch):
    monkeypatch.setenv("TLLM_FAULT_TOLERANCE_MODE", "1")
    message = peer_loss_retryable_message("Attention DP rank 2 is not serving")

    response = OpenAIServer.create_error_response(f"Request failed: {message}")

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.headers["Retry-After"] == "1"
    body = json.loads(response.body)
    assert body["code"] == PEER_LOSS_RETRYABLE_CODE
    assert body["type"] == "ServiceUnavailableError"


def test_without_fault_tolerance_a_tagged_message_is_an_ordinary_error(monkeypatch):
    """Only fault tolerance emits the tag; without it, a message that carries one is answered as on main."""
    monkeypatch.delenv("TLLM_FAULT_TOLERANCE_MODE", raising=False)
    message = peer_loss_retryable_message("user text")

    response = OpenAIServer.create_error_response(message)
    (event, _) = _stream_error_events(RequestError(message))

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "Retry-After" not in response.headers
    assert json.loads(response.body)["code"] == 400
    assert json.loads(event[len("data: ") :])["error"]["code"] is None


def test_other_errors_stay_400_without_retry_after():
    response = OpenAIServer.create_error_response("Attention DP rank 2 failed; retry it")

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "Retry-After" not in response.headers
    assert json.loads(response.body)["code"] == 400

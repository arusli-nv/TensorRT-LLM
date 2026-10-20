# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A streamed request that fails after its first chunk ends with an SSE error event."""

import asyncio
import json

from tensorrt_llm.serve.openai_server import _end_stream_on_error, _stream_error_events


def test_failed_stream_ends_with_an_error_event_then_done():
    events = _stream_error_events(RuntimeError("Attention DP rank 2 failed; retry it"))

    assert events[-1] == "data: [DONE]\n\n"
    assert events[0].startswith("data: ") and events[0].endswith("\n\n")
    # OpenAI clients raise on an event whose data carries an "error" object.
    error = json.loads(events[0][len("data: ") :])["error"]
    assert error["message"] == "Attention DP rank 2 failed; retry it"
    assert error["type"] == "server_error"


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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A streamed request that fails after its first chunk ends with an SSE error event."""

import json

from tensorrt_llm.serve.openai_server import _stream_error_events


def test_failed_stream_ends_with_an_error_event_then_done():
    events = _stream_error_events(RuntimeError("Attention DP rank 2 failed; retry it"))

    assert events[-1] == "data: [DONE]\n\n"
    assert events[0].startswith("data: ") and events[0].endswith("\n\n")
    # OpenAI clients raise on an event whose data carries an "error" object.
    error = json.loads(events[0][len("data: ") :])["error"]
    assert error["message"] == "Attention DP rank 2 failed; retry it"
    assert error["type"] == "server_error"

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
"""Execution-descriptor checks of the fence MoE All-to-All ops.

Run in a singleton MPI process with CUDA available. Multi-rank cases place every logical
peer in one allocation and only rank 0 issues ops, so they check the native layout,
masking and polling, not peer transport.
"""

import os
import subprocess
import sys
import time
from collections.abc import Callable

import pytest
import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm.bindings import internal

_CAPACITY, _HIDDEN = 8, 128


def _offset(meta: torch.Tensor, name: str) -> int:
    return int(meta[int(getattr(internal.thop, f"MOE_A2A_{name}_OFFSET_INDEX"))])


def _word(workspace: torch.Tensor, offset: int) -> torch.Tensor:
    return workspace[0, offset : offset + 4].view(torch.int32)


def _descriptor(mask: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a native descriptor (mask words, then the abort record) and its abort view."""
    descriptor = torch.tensor([mask, 0, 0, 0, 0], dtype=torch.int64, device="cuda")
    return descriptor, descriptor[4:].view(torch.int32)


def _setup(ep_size: int, tokens: int = _CAPACITY, **init) -> tuple[torch.Tensor, ...]:
    workspace = torch.zeros((ep_size, 1 << 20), dtype=torch.uint8, device="cuda")
    meta = torch.ops.trtllm.moe_a2a_initialize(workspace, 0, ep_size, _CAPACITY, **init)
    experts = torch.zeros((tokens, 1), dtype=torch.int32, device="cuda")
    payload = torch.arange(tokens * _HIDDEN, dtype=torch.bfloat16, device="cuda")
    return workspace, meta, experts, payload.reshape(tokens, _HIDDEN)


def _controls(descriptor: torch.Tensor | None) -> dict:
    if descriptor is None:
        return {}
    return {
        "enable_rank_mask": True,
        "active_rank_mask": torch.tensor([2**64 - 1] * 4, dtype=torch.uint64),
        "execution_descriptor": descriptor,
    }


def _dispatch(workspace, meta, experts, payloads, descriptor, tokens=_CAPACITY, stats=None):
    ep_size = workspace.size(0)
    args = (experts, payloads, workspace, meta, tokens, 0, ep_size, 1, ep_size, stats)
    return torch.ops.trtllm.moe_a2a_dispatch(
        *args, expert_id_payload_index=1, invalid_token_expert_id=-1, **_controls(descriptor)
    )


def _combine(workspace, meta, payload, offset, descriptor, tokens=_CAPACITY, in_workspace=False):
    ep_size = workspace.size(0)
    args = (payload, tokens, workspace, meta, tokens, 0, ep_size, 1, offset, in_workspace)
    return torch.ops.trtllm.moe_a2a_combine(*args, **_controls(descriptor))


def _round(workspace, meta, experts, payload, descriptor):
    received, offset, _ = _dispatch(workspace, meta, experts, [payload, experts], descriptor)
    return received[1], _combine(workspace, meta, received[0], offset, descriptor)


def test_cancel_and_exclusion_on_replay() -> None:
    """Does one captured graph observe the abort record and the mask written before each replay?"""
    workspace, meta, experts, payload = _setup(2)
    descriptor, abort = _descriptor(3)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ids, output = _round(workspace, meta, experts, payload, descriptor)

    # Peer 1 never signals: a pending request releases both waits instead of a timeout.
    abort[0] = 1
    graph.replay()
    torch.cuda.synchronize()
    assert abort.tolist() == [1, 1]
    assert torch.all(ids == -1) and torch.count_nonzero(output) == 0

    # Rearmed with peer 1 excluded, the same graph completes without it.
    workspace[0].zero_()
    workspace[0, _offset(meta, "RECV_COUNTERS") :][: 2 * 2 * 4].fill_(0xFF)
    abort.zero_()
    descriptor[0] = 1
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, payload, rtol=0, atol=0)
        assert torch.all(ids[0] == 0) and torch.all(ids[1] == -1)
    assert abort.tolist() == [0, 0]


@pytest.mark.parametrize("row_bytes", [16, 3])
def test_registered_source_slots_preserve_excluded_state(row_bytes: int) -> None:
    """Do graphs of different batch sizes share fixed source slots and leave excluded ones untouched?"""
    ep_size = 2
    workspace, meta, experts, payload = _setup(ep_size, eplb_stats_num_experts=ep_size)
    byte_payload = torch.arange(_CAPACITY * row_bytes, dtype=torch.uint8, device="cuda")
    byte_payload = byte_payload.reshape(_CAPACITY, row_bytes)
    descriptor, abort = _descriptor(1)

    def padded(size: int) -> int:
        return (size + 127) // 128 * 128

    start = _offset(meta, "PAYLOAD_DATA")
    source_stride = sum(padded(_CAPACITY * width) for width in (_HIDDEN * 2, 4, row_bytes))
    tail = (workspace.size(1) - ep_size * _CAPACITY * _HIDDEN * 2) // 128 * 128
    assert start + ep_size * source_stride <= tail
    fixed = torch.ops.trtllm.moe_a2a_get_combine_payload_tensor(
        workspace, 0, ep_size, _CAPACITY, tail, payload.dtype, _HIDDEN
    ).view(-1, _HIDDEN)

    # Poison everything source rank 1 owns in rank 0's workspace.
    counters = _offset(meta, "RECV_COUNTERS")
    tombstones = [workspace[0, start + source_stride : start + 2 * source_stride]]
    tombstones.extend(_word(workspace, counters + 4 * (parity * ep_size + 1)) for parity in (0, 1))
    for name in ("DISPATCH_COMPLETION_FLAGS", "COMBINE_COMPLETION_FLAGS"):
        tombstones.append(_word(workspace, _offset(meta, name) + 4))
    stats = _offset(meta, "EPLB_GATHERED_STATS") + ep_size * 4
    tombstones.append(workspace[0, stats : stats + ep_size * 4].view(torch.int32))
    tombstones[0].fill_(0xA5)
    for value in tombstones[1:]:
        value.fill_(2**31 - 1)

    local_stats = {
        bucket: torch.tensor([bucket, 0], dtype=torch.int32, device="cuda") for bucket in (4, 8)
    }

    def transfer(bucket: int) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        ids = experts[:bucket]
        payloads = [payload[:bucket], ids, byte_payload[:bucket]]
        received, offset, gathered = _dispatch(
            workspace, meta, ids, payloads, descriptor, bucket, local_stats[bucket]
        )
        assert offset == start + ep_size * source_stride
        expert_output = fixed[: ep_size * bucket]
        expert_output.copy_(received[0].view(-1, _HIDDEN))
        expert_output = expert_output.view(ep_size, bucket, _HIDDEN)
        output = _combine(workspace, meta, expert_output, tail, descriptor, bucket, True)
        return received, gathered, output

    captures = {}
    for bucket in (4, 8):
        transfer(bucket)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = transfer(bucket)
        captures[bucket] = graph, *outputs

    for bucket in (8, 4, 8, 4):
        graph, received, gathered, output = captures[bucket]
        graph.replay()
        torch.cuda.synchronize()
        expected = payload[:bucket]
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        order = received[0][0][:, 0].argsort()
        torch.testing.assert_close(received[0][0][order], expected, rtol=0, atol=0)
        torch.testing.assert_close(received[2][0][order], byte_payload[:bucket], rtol=0, atol=0)
        assert torch.all(received[1][0] == 0) and torch.all(received[1][1] == -1)
        assert gathered.tolist() == [[bucket, 0], [0, 0]]
        assert torch.all(tombstones[0] == 0xA5)
        assert all(torch.all(value == 2**31 - 1) for value in tombstones[1:])
        assert abort.tolist() == [0, 0]


@pytest.mark.parametrize("count,valid", [(3, 3), (_CAPACITY + 1, 0), (-5, 0)])
def test_gather_bounds_received_count(count: int, valid: int) -> None:
    """Does the gather only trust a peer's received count within [0, runtime tokens]?"""
    workspace, meta, experts, payload = _setup(2)
    descriptor, _ = _descriptor(3)
    # Stand in for peer 1: signal the next dispatch round and publish its count.
    expected_flag = _word(workspace, _offset(meta, "FLAG_VAL")).item() + 1
    parity = ((expected_flag - 1) >> 1) & 1
    _word(workspace, _offset(meta, "DISPATCH_COMPLETION_FLAGS") + 4).fill_(expected_flag)
    _word(workspace, _offset(meta, "RECV_COUNTERS") + 4 * (parity * 2 + 1)).fill_(count)
    received, _, _ = _dispatch(workspace, meta, experts, [payload, experts], descriptor)
    torch.cuda.synchronize()
    ids = received[1].view(2, _CAPACITY)
    assert torch.all(ids[0] == 0)
    assert torch.all(ids[1, :valid] == 0) and torch.all(ids[1, valid:] == -1)


def test_source_slots_reject_runtime_over_capacity() -> None:
    workspace, meta, experts, payload = _setup(1, _CAPACITY + 1)
    descriptor, _ = _descriptor(1)
    with pytest.raises(RuntimeError, match="static workspace capacity"):
        _dispatch(workspace, meta, experts, [experts], descriptor, _CAPACITY + 1)
    with pytest.raises(RuntimeError, match="static workspace capacity"):
        _combine(workspace, meta, payload.view(1, -1, _HIDDEN), 0, descriptor, _CAPACITY + 1)


@pytest.mark.parametrize(
    "make,controls,message",
    [
        (lambda: torch.zeros(5, dtype=torch.int64), {}, "workspace CUDA device"),
        (lambda: torch.zeros(5, dtype=torch.int32, device="cuda"), {}, "int64 tensor of 5"),
        (lambda: torch.zeros(1, dtype=torch.int64, device="cuda"), {}, "int64 tensor of 5"),
        (lambda: torch.zeros(10, dtype=torch.int64, device="cuda")[::2], {}, "int64 tensor"),
        (lambda: _descriptor(1)[0], {"use_cft_counted_writes": True}, "fence path"),
        (lambda: _descriptor(1)[0], {"enable_rank_mask": False}, "enable_rank_mask=True"),
    ],
)
def test_descriptor_validation(
    make: Callable[[], torch.Tensor], controls: dict, message: str
) -> None:
    workspace, meta, experts, payload = _setup(1)
    kwargs = {**_controls(make()), **controls}
    if not kwargs["enable_rank_mask"]:
        del kwargs["active_rank_mask"]
    with pytest.raises(RuntimeError, match=message):
        torch.ops.trtllm.moe_a2a_dispatch(
            experts, [payload], workspace, meta, _CAPACITY, 0, 1, 1, 1, **kwargs
        )


def test_descriptor_schema_and_fake_signatures() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    for op in (torch.ops.trtllm.moe_a2a_dispatch, torch.ops.trtllm.moe_a2a_combine):
        names = [argument.name for argument in op.default._schema.arguments]
        assert names[-1] == "execution_descriptor" and "abort_state" not in names
    with FakeTensorMode():
        workspace = torch.empty((1, 1 << 20), dtype=torch.uint8, device="cuda")
        meta = torch.empty(15, dtype=torch.int64)
        experts = torch.empty((_CAPACITY, 1), dtype=torch.int32, device="cuda")
        payload = torch.empty((_CAPACITY, _HIDDEN), dtype=torch.bfloat16, device="cuda")
        descriptor = torch.empty(5, dtype=torch.int64, device="cuda")
        for value in (None, descriptor):
            ids, output = _round(workspace, meta, experts, payload, value)
            assert ids.shape == (1, _CAPACITY, 1)
            assert output.shape == (_CAPACITY, _HIDDEN) and output.dtype == torch.bfloat16


def _pending_wait(phase: str, tokens: int) -> None:
    """Does a fence wait on a silent peer return once the abort record is requested?"""
    question = f"{phase}/{tokens} tokens"
    workspace, meta, experts, payload = _setup(2, tokens)
    descriptor, abort = _descriptor(1 if phase == "combine" else 3)
    if phase == "combine":
        # Dispatch without the silent peer, then wait for it in combine.
        received, offset, _ = _dispatch(workspace, meta, experts, [payload, experts], descriptor)
        descriptor[0] = 3
    torch.cuda.synchronize()
    execution, control = torch.cuda.Stream(), torch.cuda.Stream()
    done = torch.cuda.Event()
    with torch.cuda.stream(execution):
        if phase == "dispatch":
            ids, output = _round(workspace, meta, experts, payload, descriptor)
        else:
            output = _combine(workspace, meta, received[0], offset, descriptor, tokens)
        done.record(execution)
    time.sleep(0.5)
    assert not done.query(), f"{question}: completed before the silent peer signaled"
    with torch.cuda.stream(control):
        abort[:1].fill_(1)
    deadline = time.monotonic() + 10
    while not done.query():
        assert time.monotonic() < deadline, f"{question}: wait ignored the abort request"
        time.sleep(0.01)
    torch.cuda.synchronize()
    assert abort.tolist() == [1, 1], question
    assert torch.count_nonzero(output).item() == 0, question
    if phase == "dispatch":
        assert torch.all(ids == -1).item(), question


def _fence_timeout() -> None:
    """Does a fence wait on a silent peer still fail stop at the timeout?"""
    workspace, meta, experts, payload = _setup(2)
    descriptor, _ = _descriptor(3)
    _dispatch(workspace, meta, experts, [payload, experts], descriptor)
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        print("fence timeout trapped", flush=True)
        os._exit(0)
    raise AssertionError("dispatch completed without its peer")


@pytest.mark.parametrize("mode,timeout_s", [("cancel", 60), ("timeout", 1)])
def test_pending_fence_waits(mode: str, timeout_s: int) -> None:
    # The native timeout budget is cached on first use, so each mode needs a fresh interpreter.
    env = {
        **os.environ,
        "TRTLLM_MOE_A2A_TIMEOUT_SEC": str(timeout_s),
        "TRTLLM_MOE_A2A_WARMUP_TIMEOUT_SEC": str(timeout_s),
    }
    try:
        result = subprocess.run(
            [sys.executable, __file__, mode], env=env, capture_output=True, text=True, timeout=300
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"Pending fence subprocess deadline: {error.stdout!r} {error.stderr!r}")
    assert result.returncode == 0, result.stdout + result.stderr
    if mode == "timeout":
        assert "fence timeout trapped" in result.stdout, result.stdout + result.stderr


if __name__ == "__main__":
    if sys.argv[1] == "timeout":
        _fence_timeout()
    for phase in ("dispatch", "combine"):
        for tokens in (0, _CAPACITY):
            print(f"Checking {phase}/{tokens} tokens", flush=True)
            _pending_wait(phase, tokens)

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
    """Return a native descriptor (mask words, then the abort record's address) and the record."""
    abort = torch.zeros(2, dtype=torch.int32).pin_memory()
    descriptor = torch.tensor([mask, 0, 0, 0, abort.data_ptr()], dtype=torch.int64, device="cuda")
    return descriptor, abort


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
    # Rearm rank 0's workspace like moe_a2a_initialize, which starts the receive counters at -1.
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
    # Rank 0 itself resets its receive counters for source 1 each round and never reads them while
    # source 1 is masked, so they are not part of the quarantined state.
    tombstones = [workspace[0, start + source_stride : start + 2 * source_stride]]
    for name in ("DISPATCH_COMPLETION_FLAGS", "COMBINE_COMPLETION_FLAGS"):
        tombstones.append(_word(workspace, _offset(meta, name) + 4))
    stats = _offset(meta, "EPLB_GATHERED_STATS") + ep_size * 4
    tombstones.append(workspace[0, stats : stats + ep_size * 4].view(torch.int32))
    tombstones[0].fill_(0xA5)
    for value in tombstones[1:]:
        value.fill_(2**31 - 1)
    tombstone_values = [torch.full_like(value, 2**31 - 1) for value in tombstones[1:]]

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
        assert [value.tolist() for value in tombstones[1:]] == [
            value.tolist() for value in tombstone_values
        ]
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


@pytest.mark.parametrize(
    "make,controls,message",
    [
        (lambda: torch.zeros(5, dtype=torch.int64), {}, "workspace CUDA device"),
        (lambda: torch.zeros(1, dtype=torch.int64, device="cuda"), {}, "int64 tensor of 5"),
        (lambda: _descriptor(1)[0], {"use_cft_counted_writes": True}, "fence path"),
        (
            lambda: _descriptor(1)[0],
            {"enable_rank_mask": False, "active_rank_mask": None},
            "enable_rank_mask=True",
        ),
    ],
)
def test_descriptor_validation(
    make: Callable[[], torch.Tensor], controls: dict, message: str
) -> None:
    workspace, meta, experts, payload = _setup(1)
    kwargs = {
        **_controls(make()),
        "expert_id_payload_index": 1,
        "invalid_token_expert_id": -1,
        **controls,
    }
    with pytest.raises(RuntimeError, match=message):
        torch.ops.trtllm.moe_a2a_dispatch(
            experts, [payload, experts], workspace, meta, _CAPACITY, 0, 1, 1, 1, **kwargs
        )


def test_descriptor_schema_and_fake_signatures() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

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


def test_fence_timeout_still_traps() -> None:
    # The native timeout budget is cached on first use, so the test needs a fresh interpreter.
    # The child traps on purpose; keep it off the enclosing launcher's process-management interface.
    launcher_prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_")
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(launcher_prefixes)},
        "TRTLLM_MOE_A2A_TIMEOUT_SEC": "1",
        "TRTLLM_MOE_A2A_WARMUP_TIMEOUT_SEC": "1",
    }
    try:
        result = subprocess.run(
            [sys.executable, __file__], env=env, capture_output=True, text=True, timeout=300
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"Pending fence subprocess deadline: {error.stdout!r} {error.stderr!r}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fence timeout trapped" in result.stdout, result.stdout + result.stderr


if __name__ == "__main__":
    _fence_timeout()

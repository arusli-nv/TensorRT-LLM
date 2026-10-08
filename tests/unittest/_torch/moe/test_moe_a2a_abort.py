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
"""Registered fence-op checks. Run in a singleton MPI process with CUDA available."""

import os
import subprocess
import sys
import time
from collections.abc import Callable

import pytest
import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm.bindings import internal

_A2ATensors = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@pytest.fixture
def a2a() -> _A2ATensors:
    workspace = torch.empty((1, 1 << 20), dtype=torch.uint8, device="cuda")
    meta = torch.ops.trtllm.moe_a2a_initialize(workspace, 0, 1, 8)
    state = torch.zeros(2, dtype=torch.int32, device="cuda")
    experts = torch.zeros((8, 1), dtype=torch.int32, device="cuda")
    payload = torch.arange(8 * 128, dtype=torch.bfloat16, device="cuda").reshape(8, 128)
    return workspace, meta, state, experts, payload


def _controls(
    state: torch.Tensor | None, execution_descriptor: torch.Tensor | None = None
) -> dict[str, bool | torch.Tensor]:
    if state is None and execution_descriptor is None:
        return {}
    controls = {
        "enable_rank_mask": True,
        "active_rank_mask": torch.tensor([2**64 - 1] * 4, dtype=torch.uint64),
    }
    if state is not None:
        controls["abort_state"] = state
    if execution_descriptor is not None:
        controls["execution_descriptor"] = execution_descriptor
    return controls


def _dispatch(
    a2a: _A2ATensors,
    state: torch.Tensor | None,
    execution_descriptor: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], int, torch.Tensor]:
    workspace, meta, _, experts, payload = a2a
    ep_size = workspace.size(0)
    return torch.ops.trtllm.moe_a2a_dispatch(
        experts,
        [payload, experts],
        workspace,
        meta,
        8,
        0,
        ep_size,
        1,
        ep_size,
        **_controls(state, execution_descriptor),
    )


def _round(
    a2a: _A2ATensors, state: torch.Tensor | None, execution_descriptor: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    workspace, meta, _, experts, _ = a2a
    received, offset, _ = _dispatch(a2a, state, execution_descriptor)
    torch.ops.trtllm.moe_a2a_sanitize_expert_ids(
        received[1],
        workspace,
        meta,
        0,
        -1,
        abort_state=state,
        execution_descriptor=execution_descriptor,
    )
    output = torch.ops.trtllm.moe_a2a_combine(
        received[0],
        experts.size(0),
        workspace,
        meta,
        8,
        0,
        workspace.size(0),
        1,
        offset,
        False,
        **_controls(state, execution_descriptor),
    )
    return received[1], output


def _flag(a2a: _A2ATensors) -> torch.Tensor:
    workspace, meta, *_ = a2a
    index = int(internal.thop.MOE_A2A_FLAG_VAL_OFFSET_INDEX)
    offset = int(meta[index])
    return workspace[0, offset : offset + 4].view(torch.int32)


def test_registered_fence_abort_capture(a2a: _A2ATensors) -> None:
    _, _, state, _, payload = a2a
    _, normal = _round(a2a, None)
    _, optional = _round(a2a, state)
    torch.testing.assert_close(normal, payload, rtol=0, atol=0)
    torch.testing.assert_close(optional, normal, rtol=0, atol=0)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ids1, output1 = _round(a2a, state)
        ids2, output2 = _round(a2a, state)
    graph.replay()
    torch.testing.assert_close(output1, normal, rtol=0, atol=0)
    torch.testing.assert_close(output2, normal, rtol=0, atol=0)
    flag = _flag(a2a).clone()

    # Cancellation must suppress both preparation nodes on the same graph.
    state[0] = 1
    graph.replay()
    assert state.cpu().tolist() == [1, 1]
    assert torch.equal(_flag(a2a), flag)
    assert torch.all(ids1 == -1) and torch.all(ids2 == -1)
    assert torch.count_nonzero(output1) == 0 and torch.count_nonzero(output2) == 0

    # Clearing the request must not clear the failure or advance the workspace round.
    state[0] = 0
    graph.replay()
    assert state.cpu().tolist() == [0, 1]
    assert torch.equal(_flag(a2a), flag)
    assert torch.count_nonzero(output1) == 0 and torch.count_nonzero(output2) == 0
    assert torch.equal(torch.ones(8, device="cuda") + 1, torch.full((8,), 2, device="cuda"))


def test_registered_combine_preserves_first_status(a2a: _A2ATensors) -> None:
    workspace, meta, state, *_ = a2a
    received, offset, _ = _dispatch(a2a, state)
    state[1] = 2
    state[0] = 1
    flag = _flag(a2a).clone()
    output = torch.ops.trtllm.moe_a2a_combine(
        received[0], 8, workspace, meta, 8, 0, 1, 1, offset, False, **_controls(state)
    )
    assert state.cpu().tolist() == [1, 2]
    assert torch.equal(_flag(a2a), flag)
    assert torch.count_nonzero(output) == 0


def test_registered_descriptor_snapshot_and_bank_switch(a2a: _A2ATensors) -> None:
    workspace, meta, state, _, payload = a2a
    bank1 = torch.empty_like(workspace)
    assert torch.equal(torch.ops.trtllm.moe_a2a_initialize(bank1, 0, 1, 8), meta)
    state1 = torch.zeros_like(state)
    descriptors = [
        torch.tensor(
            [bank.data_ptr(), 1, 0, 0, 0, status.data_ptr()], dtype=torch.int64, device="cuda"
        )
        for bank, status in ((workspace, state), (bank1, state1))
    ]
    published = torch.tensor([descriptors[0].data_ptr()], dtype=torch.int64, device="cuda")
    slot = torch.empty_like(published)
    torch.ops.trtllm.moe_a2a_fence_latch_descriptor(published, slot)
    published.fill_(descriptors[1].data_ptr())
    _, output = _round(a2a, None, slot)
    torch.testing.assert_close(output, payload, rtol=0, atol=0)
    assert slot.item() == descriptors[0].data_ptr()
    assert _flag((bank1, meta, state1, a2a[3], payload)).item() == 0

    published.fill_(descriptors[0].data_ptr())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops.trtllm.moe_a2a_fence_latch_descriptor(published, slot)
        ids1, output1 = _round(a2a, None, slot)
        ids2, output2 = _round(a2a, None, slot)
    graph.replay()
    for value in (output1, output2):
        torch.testing.assert_close(value, payload, rtol=0, atol=0)
    flag = _flag(a2a).clone()
    state[0] = 1
    graph.replay()
    assert state.cpu().tolist() == [1, 1] and torch.equal(_flag(a2a), flag)
    assert torch.all(ids1 == -1) and torch.all(ids2 == -1)
    assert torch.count_nonzero(output1) == 0 and torch.count_nonzero(output2) == 0
    torch.cuda.synchronize()
    old_bank = workspace.clone()
    addresses = [value.data_ptr() for value in (payload, slot, ids1, ids2, output1, output2)]

    # The local test has no remote issuer; retain both banks through the final graph join.
    # Distinct input contents make a stale bank-A payload read observable.
    payload.add_(1)
    published.fill_(descriptors[1].data_ptr())
    graph.replay()
    for value in (output1, output2):
        torch.testing.assert_close(value, payload, rtol=0, atol=0)
    assert addresses == [
        value.data_ptr() for value in (payload, slot, ids1, ids2, output1, output2)
    ]
    assert state1.cpu().tolist() == [0, 0] and state.cpu().tolist() == [1, 1]
    assert torch.equal(workspace, old_bank)

    _, offset, _ = _dispatch(a2a, None, slot)
    alias = workspace[0, offset : offset + payload.numel() * payload.element_size()]
    alias = alias.view(payload.dtype).reshape(1, *payload.shape)
    for shortcut, message in ((False, "must not alias workspace"), (True, "private combine input")):
        with pytest.raises(RuntimeError, match=message):
            torch.ops.trtllm.moe_a2a_combine(
                alias, 8, workspace, meta, 8, 0, 1, 1, offset, shortcut, **_controls(None, slot)
            )
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    "slot",
    [
        lambda: torch.empty(1, dtype=torch.int64),
        lambda: torch.empty(1, dtype=torch.int32, device="cuda"),
        lambda: torch.empty(2, dtype=torch.int64, device="cuda"),
    ],
)
def test_registered_descriptor_slot_validation(
    a2a: _A2ATensors, slot: Callable[[], torch.Tensor]
) -> None:
    with pytest.raises(RuntimeError, match="execution_descriptor"):
        _dispatch(a2a, None, slot())


@pytest.mark.parametrize(
    "cft,rank_mask,separate_state",
    [(True, True, False), (False, False, False), (False, True, True)],
)
def test_registered_descriptor_admission(
    a2a: _A2ATensors, cft: bool, rank_mask: bool, separate_state: bool
) -> None:
    workspace, meta, state, experts, payload = a2a
    controls = _controls(None, torch.empty(1, dtype=torch.int64, device="cuda"))
    controls.update(use_cft_counted_writes=cft, enable_rank_mask=rank_mask)
    if separate_state:
        controls["abort_state"] = state
    with pytest.raises(RuntimeError, match="fence rank-mask mode and descriptor-owned abort state"):
        torch.ops.trtllm.moe_a2a_dispatch(
            experts, [payload, experts], workspace, meta, 8, 0, 1, 1, 1, **controls
        )


@pytest.mark.parametrize(
    "state,message",
    [
        (lambda: torch.zeros(2, dtype=torch.int32), "workspace CUDA device"),
        (lambda: torch.zeros(2, dtype=torch.int64, device="cuda"), "dtype int32"),
        (lambda: torch.zeros(3, dtype=torch.int32, device="cuda"), "shape"),
        (lambda: torch.zeros(4, dtype=torch.int32, device="cuda")[::2], "contiguous"),
    ],
)
def test_registered_abort_state_validation(
    a2a: _A2ATensors, state: Callable[[], torch.Tensor], message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _dispatch(a2a, state())


def test_registered_abort_rejects_cft(a2a: _A2ATensors) -> None:
    workspace, meta, state, experts, payload = a2a
    with pytest.raises(RuntimeError, match="fence path"):
        torch.ops.trtllm.moe_a2a_dispatch(
            experts,
            [payload],
            workspace,
            meta,
            8,
            0,
            1,
            1,
            1,
            use_cft_counted_writes=True,
            **_controls(state),
        )


def test_registered_abort_requires_rank_mask_mode(a2a: _A2ATensors) -> None:
    workspace, meta, state, experts, payload = a2a
    with pytest.raises(RuntimeError, match="enable_rank_mask=True"):
        torch.ops.trtllm.moe_a2a_dispatch(
            experts, [payload], workspace, meta, 8, 0, 1, 1, 1, abort_state=state
        )


def test_registered_abort_fake_signatures() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        workspace = torch.empty((1, 1 << 20), dtype=torch.uint8, device="cuda")
        meta = torch.empty(15, dtype=torch.int64)
        state = torch.empty(2, dtype=torch.int32, device="cuda")
        experts = torch.empty((8, 1), dtype=torch.int32, device="cuda")
        payload = torch.empty((8, 128), dtype=torch.bfloat16, device="cuda")
        a2a = workspace, meta, state, experts, payload
        slot = torch.empty(1, dtype=torch.int64, device="cuda")
        torch.ops.trtllm.moe_a2a_fence_latch_descriptor(torch.empty_like(slot), slot)
        for control, descriptor in ((None, None), (state, None), (None, slot)):
            ids, output = _round(a2a, control, descriptor)
            assert ids.shape == (1, 8, 1)
            assert output.shape == (8, 128)
            assert output.dtype == torch.bfloat16


def _pending_fence_wait(phase: str, action: str, tokens: int) -> None:
    """Does a published fence wait terminate without the synthetic peer signaling?"""
    question = f"{phase}/{action}/{tokens} tokens"
    # Both logical peers occupy one GPU allocation. This checks polling, not peer transport.
    workspace = torch.zeros((2, 1 << 20), dtype=torch.uint8, device="cuda")
    meta = torch.ops.trtllm.moe_a2a_initialize(workspace, 0, 2, 8)
    state = torch.zeros(2, dtype=torch.int32, device="cuda")
    experts = torch.zeros((tokens, 1), dtype=torch.int32, device="cuda")
    payload = torch.ones((tokens, 128), dtype=torch.bfloat16, device="cuda")
    a2a = workspace, meta, state, experts, payload
    execution, observer, control = (torch.cuda.Stream() for _ in range(3))

    def wait(event: torch.cuda.Event) -> None:
        deadline = time.monotonic() + 10
        while not event.query():
            assert time.monotonic() < deadline, f"{question}: CUDA event did not complete"

    observer_complete = torch.cuda.Event()
    terminal = torch.cuda.Event()

    def read(tensor: torch.Tensor, host: torch.Tensor) -> torch.Tensor:
        with torch.cuda.stream(observer):
            host.copy_(tensor, non_blocking=True)
            observer_complete.record(observer)
        wait(observer_complete)
        return host

    if phase == "combine":
        # Prepare valid routing with only logical peer zero active; peer one never issues an op.
        controls = {
            **_controls(state),
            "active_rank_mask": torch.tensor([1, 0, 0, 0], dtype=torch.uint64),
        }
        received, offset, _ = torch.ops.trtllm.moe_a2a_dispatch(
            experts, [payload, experts], workspace, meta, 8, 0, 2, 1, 2, **controls
        )
    torch.cuda.synchronize()
    before = _flag(a2a).item()
    flags_index = getattr(internal.thop, f"MOE_A2A_{phase.upper()}_COMPLETION_FLAGS_OFFSET_INDEX")
    flags_offset = int(meta[int(flags_index)])
    flags = workspace[0, flags_offset : flags_offset + 8].view(torch.int32)
    flags_host = torch.empty_like(flags, device="cpu", pin_memory=True)
    state_host = torch.empty_like(state, device="cpu", pin_memory=True)
    # Events are created lazily on their first recording; initialize them before the pending launch.
    observer_complete.record(observer)
    terminal.record(execution)
    wait(observer_complete)
    wait(terminal)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=execution):
        if phase == "dispatch":
            ids, output = _round(a2a, state)
        else:
            output = torch.ops.trtllm.moe_a2a_combine(
                received[0], tokens, workspace, meta, 8, 0, 2, 1, offset, False, **_controls(state)
            )
    with torch.cuda.stream(execution):
        graph.replay()
        terminal.record(execution)
    deadline = time.monotonic() + 10
    while read(flags, flags_host).tolist() != [before + 1, 0]:
        assert time.monotonic() < deadline, f"{question}: own fence flag was not published"
    assert not terminal.query(), f"{question}: graph completed before the absent peer signaled"
    assert read(state, state_host).tolist() == [0, 0], (
        f"{question}: failure before wait observation"
    )
    if action == "cancel":
        with torch.cuda.stream(control):
            state[:1].fill_(1)
    wait(terminal)
    expected = 1 if action == "cancel" else 2
    assert state.cpu().tolist() == [int(action == "cancel"), expected], question
    assert torch.count_nonzero(output).item() == 0, question
    if phase == "dispatch":
        assert torch.all(ids == -1).item(), question
    assert _flag(a2a).item() == before + 1, question
    assert (torch.ones(1, device="cuda") + 1).item() == 2, question


def test_registered_pending_fence_waits() -> None:
    # The native timeout budget is cached on its first use, so use a fresh interpreter.
    env = {
        **os.environ,
        "TRTLLM_MOE_A2A_TIMEOUT_SEC": "1",
        "TRTLLM_MOE_A2A_WARMUP_TIMEOUT_SEC": "1",
    }
    try:
        result = subprocess.run(
            [sys.executable, __file__], env=env, capture_output=True, text=True, timeout=180
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"Pending fence subprocess deadline: {error.stdout!r} {error.stderr!r}")
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    for phase in ("dispatch", "combine"):
        for action in ("cancel", "timeout"):
            for tokens in (0, 8):
                print(f"Checking {phase}/{action}/{tokens} tokens", flush=True)
                _pending_fence_wait(phase, action, tokens)

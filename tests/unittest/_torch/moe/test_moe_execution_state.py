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
"""MoeExecutionState checks. Run in a singleton MPI process with CUDA available."""

import threading
import time
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import MoeExecutionState
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm.executor.recovery import client as recovery_client


class _FakeMemory:
    def __init__(self) -> None:
        self.pins = 0

    def pin(self) -> None:
        self.pins += 1

    def unpin(self) -> None:
        self.pins -= 1


def _frontend(health: EPGroupHealth, *, cft: bool = False) -> NVLinkOneSided:
    """Build only the state that execution binding reads; no workspace is mapped."""
    comm = NVLinkOneSided.__new__(NVLinkOneSided)
    comm.can_use_cft_counted_writes = cft
    comm.ep_group_health = health
    comm._execution_state = None
    comm.mnnvl_mem = _FakeMemory()
    return comm


_EP_SIZE, _TOKENS, _HIDDEN = 2, 8, 128


def _collective_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Both logical peers live in one allocation; logical rank 1 never issues an op."""
    workspace = torch.zeros((_EP_SIZE, 1 << 20), dtype=torch.uint8, device="cuda")
    meta = torch.ops.trtllm.moe_a2a_initialize(workspace, 0, _EP_SIZE, _TOKENS)
    experts = torch.zeros((_TOKENS, 1), dtype=torch.int32, device="cuda")
    payload = torch.arange(_TOKENS * _HIDDEN, dtype=torch.bfloat16, device="cuda")
    return workspace, meta, experts, payload.reshape(_TOKENS, _HIDDEN)


def _collective_round(
    state: MoeExecutionState,
    workspace: torch.Tensor,
    meta: torch.Tensor,
    experts: torch.Tensor,
    payload: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run dispatch and combine as rank 0 under ``state``; return the expert ids and output."""
    controls = {
        "enable_rank_mask": True,
        "active_rank_mask": torch.tensor([3, 0, 0, 0], dtype=torch.uint64),
    }
    received, offset, _ = torch.ops.trtllm.moe_a2a_dispatch(
        experts,
        [payload, experts],
        workspace,
        meta,
        _TOKENS,
        0,
        _EP_SIZE,
        1,
        _EP_SIZE,
        expert_id_payload_index=1,
        invalid_token_expert_id=-1,
        **controls,
        execution_descriptor=state.descriptor,
    )
    output = torch.ops.trtllm.moe_a2a_combine(
        received[0],
        _TOKENS,
        workspace,
        meta,
        _TOKENS,
        0,
        _EP_SIZE,
        1,
        offset,
        False,
        **controls,
        execution_descriptor=state.descriptor,
    )
    return received[1], output


def test_descriptor_layout_and_install_mask() -> None:
    health = EPGroupHealth(2)
    state = MoeExecutionState(health, torch.device("cuda"))
    # Mask words followed by the address of the (requested, status) abort record in host memory.
    assert state.descriptor.cpu().tolist() == [3, 0, 0, 0, state.abort.data_ptr()]
    assert state.abort.is_pinned() and state.abort.tolist() == [0, 0]

    health.mark_failed(1)
    state.install_mask()
    assert state.descriptor.cpu().tolist() == [1, 0, 0, 0, state.abort.data_ptr()]


def test_abort_rearm_and_exclusion_resume_after_peer_lost_mid_collective() -> None:
    """Can a dispatch stuck on a lost peer be aborted from another thread, and the next round
    run on the same workspace after rearming and excluding that peer?"""
    # The dispatch waits for logical rank 1's completion flag until the abort arrives.
    workspace, meta, experts, payload = _collective_inputs()
    health = EPGroupHealth(_EP_SIZE)
    state = MoeExecutionState(health, workspace.device)
    resets = []

    NVLinkOneSided._init_constants()
    row_owner = SimpleNamespace(
        workspace=workspace,
        ep_rank=0,
        ep_size=_EP_SIZE,
        moe_a2a_metainfo=meta,
        RECV_COUNTERS_OFFSET_INDEX=NVLinkOneSided.RECV_COUNTERS_OFFSET_INDEX,
    )

    def reset_workspace() -> None:
        resets.append(1)
        NVLinkOneSided._reset_local_workspace(row_owner)

    state.retain(_FakeMemory(), reset_workspace)

    ids, output = _collective_round(state, workspace, meta, experts, payload)
    done = torch.cuda.Event()
    done.record()
    time.sleep(0.2)
    assert not done.query(), "collective completed without its peer"

    requester = threading.Thread(target=state.request_abort)
    requester.start()
    requester.join(timeout=10)
    assert not requester.is_alive(), "abort request queued behind the spinning kernel"
    deadline = time.monotonic() + 10
    while not done.query():
        assert time.monotonic() < deadline, "collective kept waiting after the abort request"
        time.sleep(0.01)
    assert state.abort.cpu().tolist() == [1, 1]
    assert torch.all(ids == -1) and torch.count_nonzero(output) == 0

    health.mark_failed(1)
    with pytest.raises(RuntimeError, match="abort record"):
        state.install_mask()
    assert state.descriptor.cpu().tolist()[:4] == [3, 0, 0, 0]
    state.rearm()
    assert resets == [1] and state.abort.cpu().tolist() == [0, 0]
    state.install_mask()
    for _ in range(2):
        _, output = _collective_round(state, workspace, meta, experts, payload)
        torch.testing.assert_close(output, payload, rtol=0, atol=0)
    assert state.abort.cpu().tolist() == [0, 0]


def test_bind_execution_state_pins_workspace_once() -> None:
    health = EPGroupHealth(2)
    state = MoeExecutionState(health, torch.device("cuda"))
    comms = [_frontend(health), _frontend(health)]
    comms[1].mnnvl_mem = comms[0].mnnvl_mem
    for comm in comms:
        comm.bind_execution_state(state)
    assert comms[0].mnnvl_mem.pins == 1

    with pytest.raises(RuntimeError, match="already bound"):
        comms[0].bind_execution_state(state)
    with pytest.raises(ValueError, match="CFT"):
        _frontend(health, cft=True).bind_execution_state(state)

    state.close()
    assert comms[0].mnnvl_mem.pins == 0


def test_for_model_follows_recovery_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    model = torch.nn.Linear(1, 1)
    monkeypatch.delenv("TLLM_FAULT_TOLERANCE_AUTHORITY", raising=False)
    recovery_client.recovery_requested.cache_clear()
    try:
        assert MoeExecutionState.for_model(model) is None
        monkeypatch.setenv("TLLM_FAULT_TOLERANCE_MODE", "1")
        monkeypatch.setenv("TLLM_FAULT_TOLERANCE_AUTHORITY", "tcp://127.0.0.1:1")
        recovery_client.recovery_requested.cache_clear()
        with pytest.raises(RuntimeError, match="NVLinkOneSided"):
            MoeExecutionState.for_model(model)
    finally:
        monkeypatch.undo()
        recovery_client.recovery_requested.cache_clear()

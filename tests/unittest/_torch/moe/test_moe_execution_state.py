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

import pytest
import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.moe.fused_moe import wide_ep_ft
from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import MoeExecutionState
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth


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
    comm._dispatch_state = {"phase": "idle"}
    comm.mnnvl_mem = _FakeMemory()
    return comm


def test_descriptor_layout_and_install_mask() -> None:
    health = EPGroupHealth(2)
    state = MoeExecutionState(health, torch.device("cuda"))
    assert state.descriptor.cpu().tolist() == [3, 0, 0, 0, state.abort.data_ptr()]
    assert state.published.item() == state.descriptor.data_ptr()
    state.latch()
    assert state.slot.item() == state.descriptor.data_ptr()
    assert state.dispatch_kwargs() == state.combine_kwargs()
    assert state.dispatch_kwargs()["execution_descriptor"] is state.slot

    # Only committed membership may be installed.
    with pytest.raises(ValueError, match="committed"):
        state.install_mask((1, 0, 0, 0))
    health.commit_failed(1, expected_generation=0)
    with pytest.raises(ValueError, match="committed"):
        state.install_mask((1, 0, 0))
    state.install_mask(health.get_mask_words())
    assert state.descriptor.cpu().tolist() == [1, 0, 0, 0, state.abort.data_ptr()]
    assert state.installed_generation == 1


def test_install_mask_rejects_aborted_record() -> None:
    health = EPGroupHealth(2)
    state = MoeExecutionState(health, torch.device("cuda"))
    state.abort[1] = 2
    health.commit_failed(1, expected_generation=0)
    with pytest.raises(RuntimeError, match="abort record"):
        state.install_mask(health.get_mask_words())
    assert state.descriptor.cpu().tolist()[:4] == [3, 0, 0, 0]


def test_construction_requires_full_membership() -> None:
    health = EPGroupHealth(2)
    health.mark_failed(1)
    with pytest.raises(ValueError, match="full initial"):
        MoeExecutionState(health, torch.device("cuda"))


def test_graph_captured_before_install_runs_after_peer_exclusion() -> None:
    """Does a graph captured with full membership complete once the absent peer is excluded?"""
    # Both logical peers live in one allocation; logical rank 1 never issues an op.
    ep_size, tokens, hidden = 2, 8, 128
    workspace = torch.zeros((ep_size, 1 << 20), dtype=torch.uint8, device="cuda")
    meta = torch.ops.trtllm.moe_a2a_initialize(workspace, 0, ep_size, tokens)
    experts = torch.zeros((tokens, 1), dtype=torch.int32, device="cuda")
    payload = torch.arange(tokens * hidden, dtype=torch.bfloat16, device="cuda")
    payload = payload.reshape(tokens, hidden)
    health = EPGroupHealth(ep_size)
    state = MoeExecutionState(health, workspace.device)
    controls = {
        "enable_rank_mask": True,
        "active_rank_mask": torch.tensor([3, 0, 0, 0], dtype=torch.uint64),
    }

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state.latch()
        received, offset, _ = torch.ops.trtllm.moe_a2a_dispatch(
            experts,
            [payload, experts],
            workspace,
            meta,
            tokens,
            0,
            ep_size,
            1,
            ep_size,
            **controls,
            **state.dispatch_kwargs(),
        )
        torch.ops.trtllm.moe_a2a_sanitize_expert_ids(
            received[1], workspace, meta, 0, -1, **state.dispatch_kwargs()
        )
        output = torch.ops.trtllm.moe_a2a_combine(
            received[0],
            tokens,
            workspace,
            meta,
            tokens,
            0,
            ep_size,
            1,
            offset,
            False,
            **controls,
            **state.combine_kwargs(),
        )

    health.commit_failed(1, expected_generation=0)
    state.install_mask(health.get_mask_words())
    for _ in range(2):
        graph.replay()
        torch.testing.assert_close(output, payload, rtol=0, atol=0)
        assert torch.all(received[1][1] == -1)
    assert state.abort.cpu().tolist() == [0, 0]


def test_bind_execution_state_pins_workspace_once() -> None:
    health = EPGroupHealth(2)
    state = MoeExecutionState(health, torch.device("cuda"))
    comms = [_frontend(health), _frontend(health)]
    comms[1].mnnvl_mem = comms[0].mnnvl_mem
    for comm in comms:
        comm.bind_execution_state(state)
        comm.bind_execution_state(state)
    assert comms[0].mnnvl_mem.pins == 1

    with pytest.raises(RuntimeError, match="already bound"):
        comms[0].bind_execution_state(MoeExecutionState(health, torch.device("cuda")))
    with pytest.raises(ValueError, match="CFT"):
        _frontend(health, cft=True).bind_execution_state(state)
    with pytest.raises(ValueError, match="different EP membership"):
        _frontend(EPGroupHealth(2)).bind_execution_state(state)
    busy = _frontend(health)
    busy._dispatch_state = {"phase": "dispatched"}
    with pytest.raises(RuntimeError, match="dispatch/combine pair"):
        busy.bind_execution_state(state)

    state.close()
    assert comms[0].mnnvl_mem.pins == 0


def test_for_model_follows_recovery_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    model = torch.nn.Linear(1, 1)
    monkeypatch.delenv("TLLM_WIDEEP_RECOVERY", raising=False)
    wide_ep_ft.recovery_enabled.cache_clear()
    try:
        assert MoeExecutionState.for_model(model) is None
        monkeypatch.setenv("TLLM_WIDEEP_RECOVERY", "tcp://127.0.0.1:1")
        wide_ep_ft.recovery_enabled.cache_clear()
        with pytest.raises(RuntimeError, match="NVLinkOneSided"):
            MoeExecutionState.for_model(model)
    finally:
        monkeypatch.undo()
        wide_ep_ft.recovery_enabled.cache_clear()

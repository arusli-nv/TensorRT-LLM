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
"""MoeExecutionState, and the NVLinkOneSided layers bound to it at model load."""

import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.moe.fused_moe import ConfigurableMoE, TritonFusedMoE, VanillaMoE
from tensorrt_llm._torch.moe.fused_moe.communication import AllGatherReduceScatter
from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import MoeExecutionState
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm._torch.pyexecutor import rank_failure_model_load
from tensorrt_llm._torch.pyexecutor.rank_failure_model_load import moe_comms, prepare_model
from tensorrt_llm.bindings import internal
from tensorrt_llm.mapping import Mapping

_EP_SIZE, _TOKENS, _HIDDEN = 2, 8, 128


def test_mask_holds_the_active_ranks() -> None:
    """Rank 70 of 72 is bit 6 of the second mask word."""
    health = EPGroupHealth(72)
    health.mark_failed(70)
    state = MoeExecutionState(health, 0, torch.device("cuda"))
    assert state.active_rank_mask.is_cuda
    assert state.active_rank_mask.cpu().tolist() == [(1 << 64) - 1, 0xBF, 0, 0]


def test_install_mask_rewrites_the_mask_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    """The device mask changes only with the generation of the EP group health, at one address."""
    health = EPGroupHealth(4)
    state = MoeExecutionState(health, 0, torch.device("cuda"))
    address = state.active_rank_mask.data_ptr()

    def install() -> list[int]:
        state.install_mask()
        return state.active_rank_mask.cpu().tolist()

    # An unchanged generation writes nothing, so it does not wait for the GPU either.
    with monkeypatch.context() as patch:
        synchronize = Mock()
        patch.setattr(torch.cuda, "synchronize", synchronize)
        assert install() == [0b1111, 0, 0, 0]
        synchronize.assert_not_called()
    health.mark_failed(3)
    assert install() == [0b0111, 0, 0, 0]
    # The kernels treat this rank as active, so a mask without it is refused.
    health.mark_failed(0)
    with pytest.raises(RuntimeError, match="must mark the local ep_rank"):
        install()
    assert state.active_rank_mask.data_ptr() == address
    assert state.active_rank_mask.cpu().tolist() == [0b0111, 0, 0, 0]


def test_state_requires_active_local_rank() -> None:
    excluded = EPGroupHealth(2)
    excluded.mark_failed(0)
    with pytest.raises(RuntimeError, match="must mark the local ep_rank"):
        MoeExecutionState(excluded, 0, torch.device("cuda"))


def test_graph_replay_follows_installed_mask() -> None:
    """A CUDA graph captured with every rank active replays without the rank masked out later.

    Both EP ranks' workspace rows are on this GPU, and only rank 0 runs. Its tokens all go to
    itself, so its collectives complete once rank 1 is masked out.
    """
    meta = NVLinkOneSided._make_workspace_layout(
        _EP_SIZE, 1, _TOKENS, _HIDDEN, torch.bfloat16, None, 0
    )
    size = int(meta[int(internal.thop.MOE_A2A_WORKSPACE_SIZE_INDEX)])
    workspace = torch.zeros((_EP_SIZE, size), dtype=torch.uint8, device="cuda")
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, 0, _EP_SIZE)
    experts = torch.zeros((_TOKENS, 1), dtype=torch.int32, device="cuda")
    payload = torch.arange(_TOKENS * _HIDDEN, dtype=torch.bfloat16, device="cuda")
    payload = payload.reshape(_TOKENS, _HIDDEN)
    health = EPGroupHealth(_EP_SIZE)
    state = MoeExecutionState(health, 0, workspace.device)
    controls = {"enable_rank_mask": True, "active_rank_mask": state.active_rank_mask}

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        received, offset, _ = torch.ops.trtllm.moe_a2a_dispatch(
            experts,
            [payload],
            workspace,
            meta,
            _TOKENS,
            0,
            _EP_SIZE,
            1,
            _EP_SIZE,
            None,
            **controls,
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
        )

    health.mark_failed(1)
    state.install_mask()
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, payload, rtol=0, atol=0)


def _frontend(*, fault_tolerance: bool = True, ep_rank: int = 0) -> NVLinkOneSided:
    """Build an NVLinkOneSided with only the state that dispatch and combine read."""
    comm = NVLinkOneSided.__new__(NVLinkOneSided)
    comm._rank_mask_enabled = fault_tolerance
    comm.execution_state = None
    comm._dispatch_state = {"phase": "idle"}
    comm._workspace_lifecycle = Mock()
    comm.mnnvl_mem = SimpleNamespace(mapped=True)
    comm._sentinel_row = None
    comm.enable_eplb = False
    comm.workspace = torch.zeros(1, dtype=torch.uint8)
    comm.ep_rank, comm.ep_size, comm.top_k, comm.num_experts = ep_rank, _EP_SIZE, 1, _EP_SIZE
    comm.invalid_token_expert_id = -1
    comm.payload_in_workspace = False
    comm.use_low_precision_combine = False
    return comm


@pytest.mark.cpu_only
def test_bind_refuses_unsupported_layers() -> None:
    state = SimpleNamespace(ep_rank=0)
    comm = _frontend()
    comm.bind_execution_state(state)
    with pytest.raises(RuntimeError, match="already bound"):
        comm.bind_execution_state(state)
    with pytest.raises(ValueError, match="built with enable_wide_ep_fault_tolerance"):
        _frontend(fault_tolerance=False).bind_execution_state(state)
    with pytest.raises(ValueError, match="cannot run EP rank 1"):
        _frontend(ep_rank=1).bind_execution_state(state)


@pytest.mark.cpu_only
def test_bound_layer_passes_the_device_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass the bound state's device mask to the all-to-all ops, also during CUDA graph capture."""
    calls = []
    payload = torch.zeros(_EP_SIZE, _TOKENS, _HIDDEN)
    received = [payload, torch.zeros(_EP_SIZE, _TOKENS, 1, dtype=torch.int32)]

    def dispatch_op(*args, **kwargs):
        calls.append(kwargs)
        return received, 0, torch.empty(0)

    def combine_op(*args, **kwargs):
        calls.append(kwargs)
        return torch.zeros(_TOKENS, _HIDDEN)

    monkeypatch.setattr(torch.ops.trtllm, "moe_a2a_dispatch", dispatch_op, raising=False)
    monkeypatch.setattr(torch.ops.trtllm, "moe_a2a_combine", combine_op, raising=False)
    monkeypatch.setattr(torch.ops.trtllm, "moe_a2a_sanitize_expert_ids", Mock(), raising=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    state = SimpleNamespace(ep_rank=0, active_rank_mask=torch.zeros(4, dtype=torch.uint64))
    hidden, experts = torch.zeros(_TOKENS, _HIDDEN), torch.zeros(_TOKENS, 1, dtype=torch.int32)
    inputs = (hidden, None, experts, None, [_TOKENS] * _EP_SIZE)
    without_mask = _frontend(fault_tolerance=False)
    bound = _frontend()
    with pytest.raises(RuntimeError, match="dispatches only once bound"):
        bound.dispatch(*inputs)
    bound.bind_execution_state(state)
    for comm in (without_mask, bound):
        comm.dispatch(*inputs)
        comm.combine(payload)

    # Without fault tolerance the ops get no rank mask.
    fence = {"enable_rank_mask": False, "active_rank_mask": None, "use_sentinel": False}
    assert calls[:2] == [
        {"expert_id_payload_index": None, "invalid_token_expert_id": None, **fence},
        {"use_low_precision": False, **fence},
    ]
    for kwargs in calls[2:]:
        assert kwargs["enable_rank_mask"] is True and kwargs["use_sentinel"] is False
        assert kwargs["active_rank_mask"] is state.active_rank_mask


def _moe_layer(cls: type, **attributes) -> torch.nn.Module:
    """Build a MoE layer of class ``cls`` without its ``__init__`` and set ``attributes`` on it."""
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    for name, value in attributes.items():
        setattr(layer, name, value)
    return layer


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "kind, uses",
    [
        ("all_gather", "(ConfigurableMoE) uses AllGatherReduceScatter"),
        ("no_communication", "(ConfigurableMoE) uses no communication strategy"),
        ("triton", "(TritonFusedMoE) uses no communication strategy"),
        ("vanilla", "(VanillaMoE) uses no communication strategy"),
    ],
)
def test_unbindable_moe_layer_is_refused(kind: str, uses: str) -> None:
    """Every MoE layer must bind, whatever its class."""
    layer = {
        "all_gather": lambda: _moe_layer(ConfigurableMoE, comm=AllGatherReduceScatter(Mapping())),
        "no_communication": lambda: _moe_layer(ConfigurableMoE, comm=None),
        "triton": lambda: _moe_layer(TritonFusedMoE),
        "vanilla": lambda: _moe_layer(VanillaMoE),
    }[kind]()
    bound = _moe_layer(ConfigurableMoE, comm=_frontend())
    with pytest.raises(ValueError, match=re.escape(f"MoE layer 1 {uses}")) as error:
        moe_comms(torch.nn.ModuleList([bound, layer]))
    assert "enable_wide_ep_fault_tolerance=False" in str(error.value)


@pytest.mark.cpu_only
def test_model_without_moe_layers_refused() -> None:
    with pytest.raises(ValueError, match="requires a model with MoE layers"):
        moe_comms(torch.nn.Linear(1, 1))


@pytest.mark.cpu_only
def test_layers_bound_to_one_execution_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """prepare_model binds every MoE layer to one execution state over every EP rank."""
    monkeypatch.setattr(
        rank_failure_model_load,
        "MoeExecutionState",
        lambda health, ep_rank, device: SimpleNamespace(health=health, ep_rank=ep_rank),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    comms = [Mock(spec=["bind_execution_state"]) for _ in range(2)]
    model = torch.nn.ModuleList([_moe_layer(ConfigurableMoE, comm=comm) for comm in comms])
    model.model_config = SimpleNamespace(mapping=SimpleNamespace(moe_ep_size=4, moe_ep_rank=1))
    prepare_model(model)
    (state,), (other,) = (comm.bind_execution_state.call_args.args for comm in comms)
    assert state is other
    assert state.ep_rank == 1
    assert state.health.get_mask() == 0b1111

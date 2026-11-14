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
"""MoeExecutionState checks."""

import re
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_moe_a2a_abort import _combine, _dispatch, _setup

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.alltoall_watchdog import AlltoAllWatchdogCoordinator
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.distributed.startup_record import StartupRecord
from tensorrt_llm._torch.moe.fused_moe import ConfigurableMoE, TritonFusedMoE, VanillaMoE
from tensorrt_llm._torch.moe.fused_moe.communication import AllGatherReduceScatter, execution_state
from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import MoeExecutionState
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm._torch.pyexecutor.rank_failure_recovery import moe_comms, prepare_model
from tensorrt_llm._wide_ep_fault_tolerance import TURN_FAULT_TOLERANCE_OFF
from tensorrt_llm.bindings import internal
from tensorrt_llm.mapping import Mapping


def _frontend(
    health: EPGroupHealth | None = None, *, fault_tolerance: bool = True
) -> NVLinkOneSided:
    """Build an NVLinkOneSided with only the state that execution binding reads."""
    comm = NVLinkOneSided.__new__(NVLinkOneSided)
    comm.can_use_cft_counted_writes = False
    comm.ep_group_health = health
    comm._rank_mask_enabled = health is not None or fault_tolerance
    comm._enable_wide_ep_fault_tolerance = fault_tolerance
    comm.execution_state = None
    comm.ep_rank = 0
    comm._dispatch_state = {"phase": "idle"}
    comm.mnnvl_mem = SimpleNamespace(mapped=False)
    comm._workspace_lifecycle = Mock()
    comm.workspace = torch.zeros(2, 1, dtype=torch.uint8)
    return comm


_EP_SIZE, _TOKENS, _HIDDEN = 2, 8, 128
# Word of the abort state holding the wait evidence on peer 0.
_EVIDENCE = int(internal.thop.MOE_A2A_ABORT_WAITED_FLAG_INDEX)
_JOIN_TIMEOUT_S = 10.0


def _collective_round(state: MoeExecutionState, workspace, meta, experts, payload) -> torch.Tensor:
    """Run dispatch and combine as rank 0 under ``state``, and return the combined output."""
    controls = {
        "enable_rank_mask": True,
        "active_rank_mask": state.active_rank_mask,
        "abort_state": state.abort,
    }
    received, offset, _ = _dispatch(workspace, meta, experts, controls, payload)
    return _combine(workspace, meta, received[0], offset, controls)


def _wait_for(condition, state: MoeExecutionState, failure: str, timeout_s: float = 10) -> None:
    """Wait until ``condition()`` holds. At the deadline, abort the collectives and fail."""
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            state.request_abort()
            torch.cuda.synchronize()
            pytest.fail(failure)
        time.sleep(0.01)


def _queued_work_done():
    done = torch.cuda.Event()
    done.record()
    return done.query


def test_abort_and_rearm_after_peer_failure() -> None:
    """A dispatch stuck on a failed peer aborts from another thread, and runs again after rearm."""
    workspace, meta, experts, payload = _setup(_EP_SIZE)
    health = EPGroupHealth(_EP_SIZE)
    state = MoeExecutionState(health, 0, workspace.device)
    resets = []

    def reset_workspace() -> None:
        resets.append(1)
        torch.ops.trtllm.moe_a2a_initialize(workspace, meta, 0, _EP_SIZE, barrier=False)

    state.register_workspace(object(), reset_workspace, workspace)

    _collective_round(state, workspace, meta, experts, payload)
    done = _queued_work_done()
    # A long wait names the flag value it expects from rank 1. It is not an abort.
    _wait_for(state.waited_flags, state, "the dispatch published no wait evidence")
    assert state.waited_flags() == {1: 1}
    assert not state.aborted()
    requester = threading.Thread(target=state.request_abort)
    requester.start()
    requester.join(_JOIN_TIMEOUT_S)
    assert not requester.is_alive(), "abort request queued behind the spinning kernel"
    _wait_for(done, state, "collective kept waiting after the abort request")
    assert state.aborted()
    assert state.waited_flags() == {}

    health.mark_failed(1)
    with pytest.raises(RuntimeError, match="abort state"):
        state.install_mask()
    assert state.active_rank_mask.cpu().tolist() == [3, 0, 0, 0]
    state.rearm()
    assert resets == [1]
    assert not state.aborted()
    state.install_mask()
    assert state.active_rank_mask.cpu().tolist() == [1, 0, 0, 0]
    for _ in range(2):
        output = _collective_round(state, workspace, meta, experts, payload)
        _wait_for(_queued_work_done(), state, "the round without rank 1 waited for it")
        torch.testing.assert_close(output, payload, rtol=0, atol=0)
    assert not state.aborted()


def test_graph_replay_follows_written_mask() -> None:
    """A CUDA graph captured with every rank active replays without the rank excluded afterwards."""
    workspace, meta, experts, payload = _setup(_EP_SIZE)
    health = EPGroupHealth(_EP_SIZE)
    state = MoeExecutionState(health, 0, workspace.device)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = _collective_round(state, workspace, meta, experts, payload)

    health.mark_failed(1)
    state.write_rank_mask()
    for _ in range(2):
        graph.replay()
        _wait_for(_queued_work_done(), state, "the replay waited for the excluded rank")
        torch.testing.assert_close(output, payload, rtol=0, atol=0)
    assert not state.aborted()


def _workspace(device_bytes: int, fresh_rows: frozenset[int]) -> Mock:
    """Return a mock membership workspace."""
    return Mock(
        local_device_bytes=device_bytes, **{"rows_needing_fresh_pages.return_value": fresh_rows}
    )


def test_bound_layer_remaps_its_memory() -> None:
    """Binding a layer registers its MNNVL memory as the workspace that a change remaps."""
    state = MoeExecutionState(EPGroupHealth(2), 0, torch.device("cuda"))
    comm = _frontend()
    comm.mnnvl_mem = _workspace(6 << 20, frozenset({1}))
    comm.bind_execution_state(state)
    store_group = Mock()

    assert state.workspace_bytes() == 6 << 20
    assert state.workspace_rows_needing_fresh_pages(store_group) == frozenset({1})
    state.remap_workspaces(store_group, frozenset({1}))
    comm.mnnvl_mem.remap.assert_called_once_with(store_group, frozenset({1}))


def test_shared_workspace_counted_once() -> None:
    """Layers that share a workspace remap, count and reset it once. Only rearm resets them."""
    state = MoeExecutionState(EPGroupHealth(4), 0, torch.device("cuda"))
    shared = _workspace(6 << 20, frozenset())
    other = _workspace(2 << 20, frozenset({0, 3}))
    resets = []
    rows = {
        shared: torch.empty(4, 6, dtype=torch.uint8),
        other: torch.empty(4, 2, dtype=torch.uint8),
    }
    for workspace, name in ((shared, "shared"), (shared, "shared again"), (other, "other")):
        state.register_workspace(workspace, lambda name=name: resets.append(name), rows[workspace])
    comm = Mock()

    # Restores stage expert weights through the largest workspace.
    assert state.staging_workspace() is rows[shared]
    assert state.workspace_bytes() == 8 << 20
    assert state.workspace_rows_needing_fresh_pages(comm) == frozenset({0, 3})
    state.remap_workspaces(comm, frozenset({1}))
    shared.remap.assert_called_once_with(comm, frozenset({1}))
    other.remap.assert_called_once_with(comm, frozenset({1}))
    comm.allgather.assert_not_called()
    assert resets == []
    state.rearm()
    assert resets == ["shared", "other"]


def test_bind_refuses_unsupported_layers() -> None:
    health = EPGroupHealth(2)
    state = MoeExecutionState(health, 0, torch.device("cuda"))
    comm = _frontend()
    comm.bind_execution_state(state)
    with pytest.raises(RuntimeError, match="already bound"):
        comm.bind_execution_state(state)
    # Only a workspace allocated for fault tolerance can be remapped after a membership change.
    with pytest.raises(ValueError, match="built with enable_wide_ep_fault_tolerance"):
        _frontend(EPGroupHealth(2), fault_tolerance=False).bind_execution_state(state)
    other_rank = _frontend()
    other_rank.ep_rank = 1
    with pytest.raises(ValueError, match="cannot run EP rank 1"):
        other_rank.bind_execution_state(state)
    # The remaps of a membership change leave out the workspace of a layer bound afterwards.
    health.mark_failed(1)
    with pytest.raises(RuntimeError, match="only before the EP membership changes"):
        _frontend().bind_execution_state(state)


def test_waited_flags_map_peer_to_flag() -> None:
    """waited_flags() maps each peer a kernel waits for to the round flag value it waits for."""
    state = MoeExecutionState(EPGroupHealth(4), 0, torch.device("cuda"))
    assert state.waited_flags() == {}
    state.abort[_EVIDENCE + 1] = 7  # the dispatch of round 3
    state.abort[_EVIDENCE + 3] = 8  # the combine of round 3
    state.abort[_EVIDENCE + 2] = -(1 << 31)  # uint32 flag 1 << 31, past the int32 range
    assert state.waited_flags() == {1: 7, 2: 1 << 31, 3: 8}


def test_rearm_resets_dispatched_layer() -> None:
    """Rearm resets a layer that an abort left between dispatch and combine."""
    state = MoeExecutionState(EPGroupHealth(2), 0, torch.device("cuda"))
    comm = _frontend()
    comm._reset_local_workspace = lambda: None
    comm.bind_execution_state(state)
    comm._dispatch_state = {"phase": "dispatched"}
    state.rearm()
    assert comm._dispatch_state == {"phase": "idle"}


def test_write_rank_mask_without_gpu_sync() -> None:
    """Writing a new mask queues its copy behind the GPU's work instead of waiting for it."""
    health = EPGroupHealth(4)
    state = MoeExecutionState(health, 0, torch.device("cuda"))
    for failed, words in ((1, [0b1101, 0, 0, 0]), (2, [0b1001, 0, 0, 0])):
        torch.cuda._sleep(1 << 31)  # about a second of GPU work ahead of the copy
        busy = torch.cuda.Event()
        busy.record()
        health.mark_failed(failed)
        state.write_rank_mask()
        assert not busy.query(), "write_rank_mask waited for the GPU"
        assert state.active_rank_mask.cpu().tolist() == words


def test_mask_rewritten_only_on_new_generation(monkeypatch) -> None:
    """The device mask changes only with the committed generation."""
    health = EPGroupHealth(4)
    state = MoeExecutionState(health, 0, torch.device("cuda"))
    address = state.active_rank_mask.data_ptr()

    def write() -> list[int]:
        state.write_rank_mask()
        return state.active_rank_mask.cpu().tolist()

    assert write() == [0b1111, 0, 0, 0]
    # An unchanged generation never reads the mask of the health.
    with monkeypatch.context() as patch:
        patch.setattr(health, "snapshot", Mock(side_effect=AssertionError("read the health")))
        assert write() == [0b1111, 0, 0, 0]
    health.mark_failed(3)
    assert write() == [0b0111, 0, 0, 0]
    # The kernels treat this rank's bit as set, so a mask that clears it is refused.
    health.mark_failed(0)
    with pytest.raises(RuntimeError, match="must mark the local ep_rank"):
        write()
    assert state.active_rank_mask.data_ptr() == address
    assert state.active_rank_mask.cpu().tolist() == [0b0111, 0, 0, 0]


def test_rank_mask_mode_mask_words_cover_a_second_word() -> None:
    """Rank 70 of 72 is bit 6 of the second mask word."""
    health = EPGroupHealth(72)
    health.mark_failed(70)
    state = MoeExecutionState(health, 0, torch.device("cuda"))
    assert state.rank_mask_words == ((1 << 64) - 1, 0xBF, 0, 0)
    assert state.active_rank_mask.cpu().tolist() == [(1 << 64) - 1, 0xBF, 0, 0]


def test_state_requires_active_local_rank() -> None:
    """The kernels assume that the local rank is active."""
    excluded = EPGroupHealth(2)
    excluded.mark_failed(0)
    with pytest.raises(RuntimeError, match="must mark the local ep_rank"):
        MoeExecutionState(excluded, 0, torch.device("cuda"))


def _collective_frontend(fault_tolerance: bool) -> NVLinkOneSided:
    """Build a ``_frontend`` with the state its dispatch and combine read, for the caller's ops."""
    comm = _frontend(fault_tolerance=fault_tolerance)
    comm._workspace_lifecycle = Mock(
        coordinator=AlltoAllWatchdogCoordinator(
            workspace_state={},
            workspace=comm.workspace,
            metainfo=torch.zeros(1, dtype=torch.int64),
            metainfo_index={},
            ep_rank=0,
        ),
        **{"watchdog_for.return_value": None},
    )
    comm.mnnvl_mem.mapped = True
    comm._force_cft = None
    comm.cft_max_batch_for_dispatch = comm.cft_max_batch_for_combine = None
    comm.enable_eplb = False
    comm.workspace = torch.zeros(1, dtype=torch.uint8)
    comm.ep_size, comm.top_k, comm.num_experts = _EP_SIZE, 1, _EP_SIZE
    comm.invalid_token_expert_id = -1
    comm.payload_in_workspace = False
    comm.use_low_precision_combine = False
    return comm


@pytest.mark.cpu_only
def test_bound_layers_pass_state_records(monkeypatch) -> None:
    """Pass the device mask and the abort state of the execution state to the all-to-all ops."""
    calls = []
    payload = torch.zeros(_EP_SIZE, _TOKENS, _HIDDEN)
    received = [payload, torch.zeros(_EP_SIZE, _TOKENS, 1, dtype=torch.int32)]

    def dispatch_op(*args, **kwargs):
        calls.append(args[-3:])
        return received, 0, torch.empty(0)

    def combine_op(*args, **kwargs):
        calls.append(args[-3:])
        return torch.zeros(_TOKENS, _HIDDEN)

    monkeypatch.setattr(torch.ops.trtllm, "moe_a2a_dispatch", dispatch_op, raising=False)
    monkeypatch.setattr(torch.ops.trtllm, "moe_a2a_combine", combine_op, raising=False)
    monkeypatch.setattr(torch.ops.trtllm, "moe_a2a_sanitize_expert_ids", Mock(), raising=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    state = SimpleNamespace(
        ep_rank=0,
        abort=torch.zeros(4, dtype=torch.int32),
        active_rank_mask=torch.zeros(4, dtype=torch.uint64),
        register_layer=Mock(),
        register_workspace=Mock(),
    )
    hidden, experts = torch.zeros(_TOKENS, _HIDDEN), torch.zeros(_TOKENS, 1, dtype=torch.int32)
    inputs = (hidden, None, experts, None, [_TOKENS] * _EP_SIZE)
    without_mask = _collective_frontend(fault_tolerance=False)
    bound = _collective_frontend(fault_tolerance=True)
    # A layer built for fault tolerance takes its membership only from an execution state.
    with pytest.raises(RuntimeError, match="dispatches only once bound"):
        bound.dispatch(*inputs)
    bound.bind_execution_state(state)
    for comm in (without_mask, bound):
        comm.dispatch(*inputs)
        comm.combine(payload)

    assert calls[:2] == [(False, None, None)] * 2
    for enabled, mask, abort in calls[2:]:
        assert enabled is True and mask is state.active_rank_mask and abort is state.abort
    caller_mask = torch.tensor([0b11, 0, 0, 0], dtype=torch.uint64)
    with pytest.raises(ValueError, match="requires committed EP group health"):
        without_mask.dispatch(*inputs, active_rank_mask=caller_mask)
    with pytest.raises(ValueError, match="takes no active_rank_mask"):
        bound.dispatch(*inputs, active_rank_mask=caller_mask)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    bound.dispatch(*inputs)
    bound.combine(payload)


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
def test_unbindable_moe_layer_is_refused(kind, uses) -> None:
    """Every MoE layer must bind, whatever its class."""
    layer = {
        "all_gather": lambda: _moe_layer(ConfigurableMoE, comm=AllGatherReduceScatter(Mapping())),
        "no_communication": lambda: _moe_layer(ConfigurableMoE, comm=None),
        "triton": lambda: _moe_layer(TritonFusedMoE),
        "vanilla": lambda: _moe_layer(VanillaMoE),
    }[kind]()
    # Bound layers hand every backend packed rows, so a layer with an unfinalized combine binds.
    bound = _moe_layer(ConfigurableMoE, comm=_frontend(), unfinalized_combine_fn=lambda **kw: None)
    with pytest.raises(ValueError, match=re.escape(f"MoE layer 1 {uses}")) as error:
        moe_comms(torch.nn.ModuleList([bound, layer]))
    assert TURN_FAULT_TOLERANCE_OFF in str(error.value)


@pytest.mark.cpu_only
def test_model_without_moe_layers_refused() -> None:
    with pytest.raises(ValueError, match="requires a model with MoE layers"):
        moe_comms(torch.nn.Linear(1, 1))


@pytest.mark.cpu_only
@pytest.mark.parametrize("replacement", [False, True], ids=["original", "replacement"])
def test_layers_bound_to_one_membership(monkeypatch, replacement) -> None:
    """prepare_model binds every MoE layer to one execution state over a new membership."""
    monkeypatch.setattr(
        execution_state,
        "MoeExecutionState",
        lambda health, ep_rank, device: SimpleNamespace(health=health, ep_rank=ep_rank),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(Distributed, "startup_record", StartupRecord())
    monkeypatch.setattr(StartupRecord, "replacement", replacement)
    comms = [Mock(spec=["bind_execution_state"]) for _ in range(2)]
    model = torch.nn.ModuleList([_moe_layer(ConfigurableMoE, comm=comm) for comm in comms])
    model.model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(architectures=["DeepseekV3ForCausalLM"]),
        mapping=SimpleNamespace(moe_ep_size=4, moe_ep_rank=1),
    )
    prepare_model(model)
    (state,), (other,) = (comm.bind_execution_state.call_args.args for comm in comms)
    assert state is other
    assert state.ep_rank == 1
    assert state.health.get_mask() == (0b0010 if replacement else 0b1111)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoeExecutionState checks. Run in a singleton MPI process with CUDA available."""

import re
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.alltoall_watchdog import ActiveRankMaskSnapshot
from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import MoeExecutionState
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm.bindings import internal


def _frontend(health: EPGroupHealth | None, *, cft: bool = False) -> NVLinkOneSided:
    """Build only the state that execution binding reads; no workspace is mapped."""
    comm = NVLinkOneSided.__new__(NVLinkOneSided)
    comm.can_use_cft_counted_writes = cft
    comm.ep_group_health = health
    comm._rank_mask_enabled = health is not None
    comm._execution_state = None
    comm._bound = False
    comm._abort_state_kwargs = {}
    comm.ep_rank = 0
    comm._dispatch_state = {"phase": "idle"}
    comm.mnnvl_mem = SimpleNamespace(mapped=False)
    comm._workspace_lifecycle = Mock()
    return comm


_EP_SIZE, _TOKENS, _HIDDEN = 2, 8, 128
# Word of the abort record holding the wait evidence on peer 0.
_EVIDENCE = int(internal.thop.MOE_A2A_ABORT_WAIT_EVIDENCE_INDEX)


def _collective_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Both logical peers live in one allocation; logical rank 1 never issues an op."""
    meta = NVLinkOneSided._make_workspace_layout(
        _EP_SIZE, 1, _TOKENS, _HIDDEN, torch.bfloat16, None, 32, False, False, rank_mask=True
    )
    size = int(meta[int(internal.thop.MOE_A2A_WORKSPACE_SIZE_INDEX)])
    workspace = torch.zeros((_EP_SIZE, size), dtype=torch.uint8, device="cuda")
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, 0, _EP_SIZE)
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
    """Run dispatch and combine as rank 0 under ``state``; return the received hidden states, [ep_size, tokens,
    hidden] (source s's rows at [s]), and the output."""
    controls = {
        "enable_rank_mask": True,
        "active_rank_mask": state.active_rank_mask,
        "abort_state": state.abort,
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
    return received[0], output


def _workspace_word(
    workspace: torch.Tensor, meta: torch.Tensor, row: int, region: str, word: int
) -> int:
    """The ``word``-th int32 of a workspace region of ``row``."""
    offset = int(meta[int(getattr(internal.thop, f"MOE_A2A_{region}_OFFSET_INDEX"))]) + 4 * word
    return int(workspace[row, offset : offset + 4].view(torch.int32).item())


def test_abort_rearm_and_exclusion_resume_after_rank_failure_mid_collective() -> None:
    """Can a dispatch stuck on a failed peer be aborted from another thread, and the next round
    run on the same workspace after rearming and excluding that peer?"""
    # The dispatch waits for logical rank 1's completion flag until the abort arrives.
    workspace, meta, experts, payload = _collective_inputs()
    health = EPGroupHealth(_EP_SIZE)
    state = MoeExecutionState(health, workspace.device)
    resets = []

    row_owner = SimpleNamespace(
        workspace=workspace,
        moe_a2a_metainfo=meta,
        ep_rank=0,
        ep_size=_EP_SIZE,
    )

    def reset_workspace() -> None:
        resets.append(1)
        NVLinkOneSided._reset_local_workspace(row_owner)

    state.retain(object(), reset_workspace)

    received, output = _collective_round(state, workspace, meta, experts, payload)
    done = torch.cuda.Event()
    done.record()
    time.sleep(0.2)
    assert not done.query(), "collective completed without its peer"
    # The dispatch of round 0 has been waiting on rank 1 for far longer than a healthy wait: the record
    # names the flag value it expects from rank 1, and nothing from rank 0. Wait evidence alone is no abort.
    waiting_evidence = state.abort[_EVIDENCE : _EVIDENCE + _EP_SIZE].tolist()
    aborted_while_waiting = state.aborted()

    requester = threading.Thread(target=state.request_abort)
    requester.start()
    requester.join(timeout=10)
    assert not requester.is_alive(), "abort request queued behind the spinning kernel"
    deadline = time.monotonic() + 10
    while not done.query():
        assert time.monotonic() < deadline, "collective kept waiting after the abort request"
        time.sleep(0.01)
    assert waiting_evidence == [0, 1]
    assert not aborted_while_waiting
    assert state.abort[:2].tolist() == [1, 1]
    assert state.aborted()
    assert torch.count_nonzero(state.abort[_EVIDENCE:]) == 0
    # This rank's own rows arrived before it waited for the failed peer: every token goes to expert 0 on
    # rank 0, so they are the payload's rows in an order the dispatch assigns.
    own_rows = received[0].float()
    torch.testing.assert_close(own_rows[own_rows[:, 0].argsort()], payload.float(), rtol=0, atol=0)
    # The combine of the cancelled round takes this rank's own rows only, never a peer's.
    torch.testing.assert_close(output, payload, rtol=0, atol=0)
    # The aborted dispatch marks the round incomplete, so the combine published no readiness: a peer that
    # did not abort cannot combine outputs computed without the failed peer's rows.
    assert _workspace_word(workspace, meta, 0, "FLAG_VAL", 1) == 1
    assert _workspace_word(workspace, meta, 1, "COMBINE_COMPLETION_FLAGS", 0) == 0

    health.mark_failed(1)
    with pytest.raises(RuntimeError, match="abort record"):
        state.install_mask()
    assert state.active_rank_mask.cpu().tolist() == [3, 0, 0, 0]
    state.rearm()
    assert resets == [1] and state.abort[:2].tolist() == [0, 0]
    assert _workspace_word(workspace, meta, 0, "FLAG_VAL", 1) == 0
    state.install_mask()
    for _ in range(2):
        _, output = _collective_round(state, workspace, meta, experts, payload)
        torch.testing.assert_close(output, payload, rtol=0, atol=0)
    assert state.abort[:2].tolist() == [0, 0]


def test_bind_execution_state_registers_a_shared_workspace_once() -> None:
    health = EPGroupHealth(2)
    state = MoeExecutionState(health, torch.device("cuda"))
    comms = [_frontend(health), _frontend(health)]
    comms[1]._workspace_lifecycle = comms[0]._workspace_lifecycle
    for comm in comms:
        comm._reset_local_workspace = Mock()
        comm.bind_execution_state(state)
    state.rearm()
    comms[0]._reset_local_workspace.assert_called_once_with()
    comms[1]._reset_local_workspace.assert_not_called()

    with pytest.raises(RuntimeError, match="already bound"):
        comms[0].bind_execution_state(state)
    with pytest.raises(ValueError, match="TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT=0"):
        _frontend(health, cft=True).bind_execution_state(state)
    with pytest.raises(ValueError, match="rank-mask mode"):
        _frontend(None).bind_execution_state(state)


def test_remap_workspaces_remaps_each_workspace_in_binding_order_without_a_reset() -> None:
    """Every rank remaps the workspaces in the same order; their reset is :meth:`MoeExecutionState.rearm`'s."""
    health = EPGroupHealth(2)
    state = MoeExecutionState(health, torch.device("cuda"))
    calls = Mock()
    comms = [_frontend(health), _frontend(health)]
    for index, comm in enumerate(comms):
        comm._workspace_lifecycle = getattr(calls, f"workspace{index}")
        comm._reset_local_workspace = getattr(calls, f"reset{index}")
        comm.bind_execution_state(state)
    gather = Mock()

    state.remap_workspaces(gather)

    assert [name for name, _, _ in calls.mock_calls] == ["workspace0.remap", "workspace1.remap"]
    for comm in comms:
        comm._workspace_lifecycle.remap.assert_called_once_with(gather)
    gather.assert_not_called()


def test_workspace_bytes_count_each_bound_workspace_once() -> None:
    """The device memory a remap allocates: layers sharing one workspace count it once."""
    health = EPGroupHealth(2)
    state = MoeExecutionState(health, torch.device("cuda"))
    shared = Mock(device_bytes=6 << 20)
    comms = [_frontend(health), _frontend(health), _frontend(health)]
    for comm, workspace in zip(comms, (shared, shared, Mock(device_bytes=2 << 20))):
        comm._workspace_lifecycle = workspace
        comm._reset_local_workspace = Mock()
        comm.bind_execution_state(state)
    assert state.workspace_bytes() == 8 << 20


@pytest.mark.parametrize("needs", [(False, False), (False, True)])
def test_the_workspaces_need_fresh_pages_when_one_of_them_does(needs) -> None:
    """Each workspace answers alike on every rank of the change (``MnnvlMemory.needs_fresh_pages``)."""
    health = EPGroupHealth(4)
    state = MoeExecutionState(health, torch.device("cuda"))
    comms = [_frontend(health), _frontend(health)]
    for comm, need in zip(comms, needs):
        comm._workspace_lifecycle = Mock(needs_fresh_pages=Mock(return_value=need))
        comm._reset_local_workspace = Mock()
        comm.bind_execution_state(state)
    gather = Mock()
    assert state.workspaces_need_fresh_pages(gather) is any(needs)
    comms[0]._workspace_lifecycle.needs_fresh_pages.assert_called_once_with(gather)


def test_an_admission_maps_the_joining_rows_of_every_workspace_without_releasing_any() -> None:
    health = EPGroupHealth(2)
    state = MoeExecutionState(health, torch.device("cuda"))
    calls = Mock()
    comms = [_frontend(health), _frontend(health)]
    for index, comm in enumerate(comms):
        comm._workspace_lifecycle = getattr(calls, f"workspace{index}")
        comm._reset_local_workspace = getattr(calls, f"reset{index}")
        comm.bind_execution_state(state)
    gather = Mock()
    state.admit_to_workspaces(gather)
    assert [name for name, _, _ in calls.mock_calls] == ["workspace0.admit", "workspace1.admit"]
    for comm in comms:
        comm._workspace_lifecycle.admit.assert_called_once_with(gather)
    gather.assert_not_called()


def test_waited_rounds_name_each_peer_waited_for_with_its_round() -> None:
    """The membership counts a peer that delivers the round it is waited for in as making progress."""
    state = MoeExecutionState(EPGroupHealth(4), torch.device("cuda"))
    assert state.waited_rounds() == {}
    state.abort[_EVIDENCE + 1] = 7  # the dispatch of round 3
    state.abort[_EVIDENCE + 3] = 8  # the combine of round 3
    assert state.waited_rounds() == {1: 7, 3: 8}


def test_rearm_resets_bound_layers_and_install_refuses_a_membership_without_this_rank() -> None:
    health = EPGroupHealth(2)
    state = MoeExecutionState(health, torch.device("cuda"))
    comm = _frontend(health)
    comm._reset_local_workspace = lambda: None
    comm.bind_execution_state(state)
    # An abort can leave an eager layer between dispatch and combine.
    comm._dispatch_state = {"phase": "dispatched"}
    state.rearm()
    assert comm._dispatch_state == {"phase": "idle"}

    health.mark_failed(1)
    state.install_mask()
    assert state.active_rank_mask.cpu().tolist() == [1, 0, 0, 0]
    # Nor may a layer of a rank the installed membership excludes bind.
    excluded = _frontend(health)
    excluded.ep_rank = 1
    with pytest.raises(RuntimeError, match="excludes this rank"):
        excluded.bind_execution_state(state)
    assert excluded._execution_state is None

    # A rank whose own bit the committed membership clears must not run masked collectives.
    health = EPGroupHealth(2)
    state = MoeExecutionState(health, torch.device("cuda"))
    comm = _frontend(health)
    comm.ep_rank = 1
    comm.bind_execution_state(state)
    health.mark_failed(1)
    with pytest.raises(RuntimeError, match="excludes this rank"):
        state.install_mask()
    assert state.active_rank_mask.cpu().tolist() == [3, 0, 0, 0]


def test_layers_of_one_membership_share_one_mask_record() -> None:
    """Unbound rank-mask layers read the device record of their membership's shared state, which has no abort
    record: they never pass one, so the state binds no layer."""
    health = EPGroupHealth(2)
    state = MoeExecutionState.for_health(health, torch.device("cuda"))
    assert MoeExecutionState.for_health(health, torch.device("cuda")) is state
    assert MoeExecutionState.for_health(EPGroupHealth(2), torch.device("cuda")) is not state
    assert state.abort is None
    with pytest.raises(RuntimeError, match="without an abort record binds no layer"):
        _frontend(health).bind_execution_state(state)


def test_a_published_mask_is_copied_without_waiting_for_the_gpu() -> None:
    """Publishing a new mask before a dispatch queues its copy behind the GPU's work instead of waiting for it."""
    state = MoeExecutionState(EPGroupHealth(4), torch.device("cuda"))
    for words, generation in (([3, 0, 0, 0], 1), ([1, 0, 0, 0], 2)):
        torch.cuda._sleep(1 << 31)  # about a second of GPU work ahead of the copy
        busy = torch.cuda.Event()
        busy.record()
        mask = torch.tensor(words, dtype=torch.uint64)
        state.publish_rank_mask(ActiveRankMaskSnapshot(mask, generation), ep_rank=0)
        assert not busy.query(), "publish_rank_mask waited for the GPU"
        assert state.active_rank_mask.cpu().tolist() == words


def test_published_mask_is_rewritten_only_for_a_new_generation() -> None:
    """The device record of the rank mask is rewritten only when the committed generation (or, without one, the
    caller's mask) changes."""
    state = MoeExecutionState(EPGroupHealth(4), torch.device("cuda"))
    record = state.active_rank_mask.data_ptr()

    def publish(words: list[int], generation: int | None) -> list[int]:
        mask = torch.tensor(words, dtype=torch.uint64)
        state.publish_rank_mask(ActiveRankMaskSnapshot(mask, generation), ep_rank=0)
        return state.active_rank_mask.cpu().tolist()

    assert publish([3, 0, 0, 0], 0) == [3, 0, 0, 0]
    # The generation alone decides: an unchanged generation never reads the mask.
    assert publish([1, 0, 0, 0], 0) == [3, 0, 0, 0]
    assert publish([1, 0, 0, 0], 1) == [1, 0, 0, 0]
    # A caller's mask carries no generation and is compared by its words.
    assert publish([3, 0, 0, 0], None) == [3, 0, 0, 0]
    assert publish([1, 0, 0, 0], 1) == [1, 0, 0, 0]
    # The kernels do not check the local rank's bit; the record keeps its address and last mask.
    with pytest.raises(RuntimeError, match="excludes this rank"):
        publish([2, 0, 0, 0], 2)
    assert state.active_rank_mask.data_ptr() == record
    assert state.active_rank_mask.cpu().tolist() == [1, 0, 0, 0]


def test_an_unbound_rank_mask_dispatch_refuses_cuda_graph_capture(monkeypatch) -> None:
    """A layer that reads the shared state but is not bound to one: nothing could abort a replayed collective."""
    comm = _frontend(EPGroupHealth(2))
    comm._execution_state = MoeExecutionState.for_health(comm.ep_group_health, torch.device("cuda"))
    comm.mnnvl_mem.mapped = True
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="only once bound to an execution state"):
        comm.dispatch(torch.empty(0), None, torch.empty(0), None, [1, 1])


def _collective_frontend(health: EPGroupHealth | None) -> NVLinkOneSided:
    """``_frontend`` with the state its dispatch and combine read; the ops are the caller's to replace."""
    comm = _frontend(health)
    comm.mnnvl_mem.mapped = True
    comm._force_cft = None
    comm.cft_max_batch_for_dispatch = comm.cft_max_batch_for_combine = None
    comm.enable_eplb = False
    comm.workspace = torch.zeros(1, dtype=torch.uint8)
    comm.ep_size, comm.top_k, comm.num_experts = _EP_SIZE, 1, _EP_SIZE
    comm.invalid_token_expert_id = -1
    comm.payload_in_workspace = False
    comm.use_low_precision_combine = False
    coordinator = comm._workspace_lifecycle.coordinator
    coordinator.capture_active_rank_mask.return_value = ActiveRankMaskSnapshot(None, None)
    coordinator.active_rank_mask_for_combine.return_value = None
    return comm


@pytest.mark.cpu_only
def test_only_a_bound_layer_hands_the_ops_an_abort_record(monkeypatch) -> None:
    """Without rank-mask mode, dispatch and combine call the all-to-all ops with their positional
    arguments only; a bound layer adds its state's abort record, settled when it binds."""
    calls = []
    received = [
        torch.zeros(_EP_SIZE, _TOKENS, _HIDDEN),
        torch.zeros(_EP_SIZE, _TOKENS, 1, dtype=torch.int32),
    ]

    def dispatch_op(*args, **kwargs):
        calls.append((args, kwargs))
        return received, 0, torch.empty(0)

    def combine_op(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.zeros(_TOKENS, _HIDDEN)

    monkeypatch.setattr(torch.ops.trtllm, "moe_a2a_dispatch", dispatch_op, raising=False)
    monkeypatch.setattr(torch.ops.trtllm, "moe_a2a_combine", combine_op, raising=False)
    monkeypatch.setattr(torch.ops.trtllm, "moe_a2a_sanitize_expert_ids", Mock(), raising=False)
    state = SimpleNamespace(
        abort=torch.zeros(4, dtype=torch.int32),
        active_rank_mask=torch.zeros(4, dtype=torch.uint64),
        track_layer=Mock(),
        retain=Mock(),
    )
    unbound = _collective_frontend(None)
    bound = _collective_frontend(EPGroupHealth(_EP_SIZE))
    bound.bind_execution_state(state)
    for comm in (unbound, bound):
        experts = torch.zeros(_TOKENS, 1, dtype=torch.int32)
        comm.dispatch(torch.zeros(_TOKENS, _HIDDEN), None, experts, None, [_TOKENS] * _EP_SIZE)
        comm.combine(torch.zeros(_EP_SIZE, _TOKENS, _HIDDEN))

    # Dispatch takes 15 positional arguments and combine 14; the last two are the rank-mask switch and mask.
    (dispatch, dispatch_kwargs), (combine, combine_kwargs) = calls[:2]
    assert (len(dispatch), dispatch[-2:], dispatch_kwargs) == (15, (False, None), {})
    assert (len(combine), combine[-2:], combine_kwargs) == (14, (False, None), {})
    for args, kwargs in calls[2:]:
        assert args[-2] is True and args[-1] is state.active_rank_mask
        assert kwargs.keys() == {"abort_state"} and kwargs["abort_state"] is state.abort
    assert len(calls) == 4


def _moe_layer(cls: type, **attributes) -> torch.nn.Module:
    """A MoE layer of class ``cls``, built without ``__init__``, with ``attributes`` set."""
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
def test_a_moe_layer_whose_communication_cannot_bind_is_refused(kind, uses) -> None:
    """Every MoE layer must take part, whatever its class, next to one that binds."""
    from tensorrt_llm._torch.moe.fused_moe import ConfigurableMoE, TritonFusedMoE, VanillaMoE
    from tensorrt_llm._torch.moe.fused_moe.communication import AllGatherReduceScatter

    layer = {
        "all_gather": lambda: _moe_layer(
            ConfigurableMoE, comm=AllGatherReduceScatter.__new__(AllGatherReduceScatter)
        ),
        "no_communication": lambda: _moe_layer(ConfigurableMoE, comm=None),
        "triton": lambda: _moe_layer(TritonFusedMoE),
        "vanilla": lambda: _moe_layer(VanillaMoE),
    }[kind]()
    bound = _moe_layer(ConfigurableMoE, comm=_frontend(EPGroupHealth(2)))
    with pytest.raises(ValueError, match=re.escape(f"MoE layer 1 {uses}")) as error:
        MoeExecutionState.model_comms(torch.nn.ModuleList([bound, layer]))
    assert "enable_wide_ep_fault_tolerance=False" in str(error.value)


@pytest.mark.cpu_only
def test_a_model_without_moe_layers_is_refused() -> None:
    with pytest.raises(ValueError, match="requires a model with MoE layers"):
        MoeExecutionState.model_comms(torch.nn.Linear(1, 1))


@pytest.mark.cpu_only
@pytest.mark.parametrize("separate", [True, False], ids=["separate_healths", "no_rank_mask"])
def test_the_moe_layers_must_share_one_ep_group_health(separate) -> None:
    """Bound layers run in rank-mask mode on the one membership (``bind_execution_state``)."""
    from tensorrt_llm._torch.moe.fused_moe import ConfigurableMoE

    healths = (EPGroupHealth(2), EPGroupHealth(2)) if separate else (None, None)
    layers = [_moe_layer(ConfigurableMoE, comm=_frontend(health)) for health in healths]
    with pytest.raises(ValueError, match="rank-mask mode on one shared EPGroupHealth") as error:
        MoeExecutionState.model_comms(torch.nn.ModuleList(layers))
    assert "communication factory from the model's ModelConfig" in str(error.value)
    assert "enable_wide_ep_fault_tolerance=False" in str(error.value)


def test_for_model_binds_a_layer_with_an_unfinalized_combine(monkeypatch) -> None:
    """Bound layers hand every backend packed rows, so a registered unfinalized combine binds like any layer."""
    from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE

    moe = ConfigurableMoE.__new__(ConfigurableMoE)
    torch.nn.Module.__init__(moe)
    moe.comm = _frontend(EPGroupHealth(2))
    moe.unfinalized_combine_fn = lambda **kwargs: None
    moe.comm.workspace = torch.empty(0)
    bind = Mock()
    monkeypatch.setattr(NVLinkOneSided, "bind_execution_state", bind)
    state = MoeExecutionState.for_model(moe)
    bind.assert_called_once_with(state)

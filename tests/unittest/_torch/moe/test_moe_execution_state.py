# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoeExecutionState checks. Run in a singleton MPI process with CUDA available."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.alltoall_watchdog import ActiveRankMaskSnapshot
from tensorrt_llm._torch.moe.fused_moe import moe_scheduler
from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import MoeExecutionState
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm._torch.moe.fused_moe.moe_scheduler import ExternalCommMoEScheduler
from tensorrt_llm.bindings import internal


class _FakeMemory:
    def __init__(self) -> None:
        self.pins = 0

    def pin(self) -> None:
        self.pins += 1

    def unpin(self) -> None:
        self.pins -= 1


def _frontend(health: EPGroupHealth | None, *, cft: bool = False) -> NVLinkOneSided:
    """Build only the state that execution binding reads; no workspace is mapped."""
    comm = NVLinkOneSided.__new__(NVLinkOneSided)
    comm.can_use_cft_counted_writes = cft
    comm.ep_group_health = health
    comm._rank_mask_enabled = health is not None
    comm._execution_state = None
    comm.ep_rank = 0
    comm._dispatch_state = {"phase": "idle"}
    comm.mnnvl_mem = _FakeMemory()
    comm._workspace_lifecycle = Mock()
    return comm


_EP_SIZE, _TOKENS, _HIDDEN = 2, 8, 128


def _collective_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Both logical peers live in one allocation; logical rank 1 never issues an op."""
    meta = NVLinkOneSided._make_workspace_layout(
        _EP_SIZE, 1, _TOKENS, _HIDDEN, torch.bfloat16, None, 32, False, False, rank_mask=True
    )
    size = int(meta[int(internal.thop.MOE_A2A_WORKSPACE_SIZE_INDEX)])
    workspace = torch.zeros((_EP_SIZE, size), dtype=torch.uint8, device="cuda")
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, 0, _EP_SIZE, barrier=False)
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


def test_records_and_install_mask() -> None:
    health = EPGroupHealth(2)
    state = MoeExecutionState(health, torch.device("cuda"))
    assert state.active_rank_mask.is_cuda
    assert state.active_rank_mask.cpu().tolist() == [3, 0, 0, 0]
    # The (requested, status) abort record lives in host memory the kernels poll.
    assert state.abort.is_pinned() and state.abort.tolist() == [0, 0]

    health.mark_failed(1)
    state.install_mask()
    assert state.active_rank_mask.cpu().tolist() == [1, 0, 0, 0]


def test_abort_rearm_and_exclusion_resume_after_peer_failure_mid_collective() -> None:
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

    state.retain(_FakeMemory(), reset_workspace, Mock())

    received, output = _collective_round(state, workspace, meta, experts, payload)
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
    assert state.aborted()
    # This rank's own rows arrived before it waited for the failed peer: every token goes to expert 0 on
    # rank 0, so they are the payload's rows in an order the dispatch assigns.
    own_rows = received[0].float()
    torch.testing.assert_close(own_rows[own_rows[:, 0].argsort()], payload.float(), rtol=0, atol=0)
    assert torch.count_nonzero(output) == 0
    # The aborted dispatch marks the round incomplete, so the combine published no readiness: a peer that
    # did not abort cannot combine outputs computed without the failed peer's rows.
    assert _workspace_word(workspace, meta, 0, "FLAG_VAL", 1) == 1
    assert _workspace_word(workspace, meta, 1, "COMBINE_COMPLETION_FLAGS", 0) == 0

    health.mark_failed(1)
    with pytest.raises(RuntimeError, match="abort record"):
        state.install_mask()
    assert state.active_rank_mask.cpu().tolist() == [3, 0, 0, 0]
    state.rearm()
    assert resets == [1] and state.abort.cpu().tolist() == [0, 0]
    assert _workspace_word(workspace, meta, 0, "FLAG_VAL", 1) == 0
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
    with pytest.raises(ValueError, match="TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT=0"):
        _frontend(health, cft=True).bind_execution_state(state)
    with pytest.raises(ValueError, match="rank-mask mode"):
        _frontend(None).bind_execution_state(state)

    state.close()
    assert comms[0].mnnvl_mem.pins == 0


def test_workspaces_release_and_remap_in_binding_order_then_reset() -> None:
    """Every retained workspace releases its backing, once the failed ranks are off the device mask, then
    remaps through the survivors' gather and resets."""
    state = MoeExecutionState(EPGroupHealth(66), torch.device("cuda"))
    calls = []
    for name, size in (("first", 1 << 20), ("second", 2 << 20)):
        lifecycle = Mock()
        lifecycle.release_backing.return_value = size
        lifecycle.remap.side_effect = lambda gather, reset, name=name: (
            calls.append(("remap", name, gather)),
            reset(),
        )
        state.retain(
            _FakeMemory(),
            lambda name=name: calls.append(("reset", name)),
            lifecycle,
        )
    assert state.release_workspaces([1, 65]) == 3 << 20
    words = [word % (1 << 64) for word in state.active_rank_mask.cpu().tolist()]
    assert words[:2] == [((1 << 64) - 1) & ~2, 1] and words[2:] == [0, 0]

    def gather(value):
        return [value, None]

    state.remap_workspaces(gather)
    assert calls == [
        ("remap", "first", gather),
        ("reset", "first"),
        ("remap", "second", gather),
        ("reset", "second"),
    ]
    state.close()


def test_a_rank_does_not_release_its_own_workspaces_as_failed() -> None:
    state = MoeExecutionState(EPGroupHealth(4), torch.device("cuda"))
    state.track_layer(2, lambda: None)
    try:
        with pytest.raises(RuntimeError, match="own workspaces"):
            state.release_workspaces([1, 2])
    finally:
        state.close()


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


def test_detection_only_publish_rewrites_the_mask_only_on_a_new_generation() -> None:
    comm = _frontend(EPGroupHealth(2))
    comm.workspace = torch.empty(0)
    comm._rank_mask_record = comm._rank_mask_words = comm._rank_mask_generation = None

    def publish(words: list[int], generation: int | None) -> list[int]:
        mask = torch.tensor(words, dtype=torch.uint64)
        comm._publish_rank_mask(ActiveRankMaskSnapshot(mask, generation))
        return comm._rank_mask_record.tolist()

    assert publish([3, 0, 0, 0], 0) == [3, 0, 0, 0]
    # The generation alone decides: an unchanged generation never reads the mask.
    assert publish([1, 0, 0, 0], 0) == [3, 0, 0, 0]
    assert publish([1, 0, 0, 0], 1) == [1, 0, 0, 0]
    # A caller's mask carries no generation and is compared by its words.
    assert publish([3, 0, 0, 0], None) == [3, 0, 0, 0]
    assert publish([1, 0, 0, 0], 1) == [1, 0, 0, 0]


def test_an_unbound_rank_mask_dispatch_refuses_cuda_graph_capture(monkeypatch) -> None:
    """Detection-only fault tolerance binds no execution state: nothing could abort a replayed collective."""
    comm = _frontend(EPGroupHealth(2))
    comm.mnnvl_mem.mapped = True
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="only once bound to an execution state"):
        comm.dispatch(torch.empty(0), None, torch.empty(0), None, [1, 1])


def test_for_model_refuses_models_it_cannot_bind() -> None:
    with pytest.raises(RuntimeError, match="NVLinkOneSided"):
        MoeExecutionState.for_model(torch.nn.Linear(1, 1))


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
    assert state.progress_comm is moe.comm


@pytest.mark.parametrize("recovery", [True, False])
def test_scheduler_records_peer_loss_recovery_and_finds_the_one_slot_table(
    monkeypatch, recovery: bool
) -> None:
    """The scheduler records whether peer-loss recovery is on; its slot table is the one-slot table either way."""
    monkeypatch.setattr(moe_scheduler, "peer_loss_recovery_enabled", lambda: recovery)
    table = torch.zeros(2, dtype=torch.int32)
    comm = _frontend(EPGroupHealth(2))
    scheduler = ExternalCommMoEScheduler(
        SimpleNamespace(
            comm=comm,
            enable_alltoall=True,
            layer_load_balancer=Mock(one_slot_table=Mock(return_value=table)),
        )
    )
    assert scheduler._peer_loss_recovery is recovery
    assert scheduler._dispatch_slot_table() is table


class _Dispatched(Exception):
    """Stops the scheduler at the dispatch, carrying the dispatch's keyword arguments."""


@pytest.mark.parametrize("recovery", [True, False])
def test_only_peer_loss_recovery_dispatches_by_the_expert_to_slot_table(
    monkeypatch, recovery: bool
) -> None:
    """Without peer-loss recovery the scheduler routes the slots itself and never passes a slot table, so the
    dispatch never takes the kernel's expert-to-slot (USE_EXPERT_TO_SLOT) path."""
    monkeypatch.setattr(moe_scheduler, "peer_loss_recovery_enabled", lambda: recovery)
    table = torch.zeros(4, dtype=torch.int32)
    experts = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    comm = Mock(spec=NVLinkOneSided)
    comm.supports_post_quant_dispatch.return_value = True

    def dispatch(**kwargs):
        raise _Dispatched(kwargs)

    comm.dispatch.side_effect = dispatch
    moe = MagicMock(comm=comm, apply_router_weight_on_input=False, enable_dummy_allreduce=False)
    moe.routing_method.experts_per_token = 2
    moe.routing_method.apply.return_value = (experts, torch.ones(2, 2))
    moe._maybe_get_perfect_router_expert_ids.side_effect = lambda ids: ids
    moe.backend.input_requirement.routing_scales_dtype = None
    moe.backend.input_requirement.requires_sanitized_expert_ids = False
    moe.backend.quantize_input.side_effect = lambda x: (x, None)
    moe.layer_load_balancer.one_slot_table.return_value = table
    scheduler = ExternalCommMoEScheduler(moe)

    with pytest.raises(_Dispatched) as dispatched:
        scheduler._forward_chunk_impl(
            torch.zeros(2, 8),
            torch.zeros(2, 4),
            torch.bfloat16,
            [2, 2],
            use_dp_padding=False,
            is_first_call=True,
            is_last_call=False,
            row_offset=None,
        )

    (kwargs,) = dispatched.value.args
    if recovery:
        assert kwargs["expert_to_slot"] is table
        assert kwargs["token_selected_slots"] is experts
    else:
        assert "expert_to_slot" not in kwargs
        moe.layer_load_balancer.one_slot_table.assert_not_called()
        assert kwargs["token_selected_slots"] is moe._load_balancer_route.return_value

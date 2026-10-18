# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
from tensorrt_llm._torch.moe.fused_moe.impl_contract import DispatchSourceSlots
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
    return comm


_EP_SIZE, _TOKENS, _HIDDEN = 2, 8, 128


def _collective_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Both logical peers live in one allocation; logical rank 1 never issues an op."""
    meta = NVLinkOneSided._make_workspace_layout(
        _EP_SIZE, 1, _TOKENS, _HIDDEN, torch.bfloat16, None, 32, False, False
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
    """Run dispatch and combine as rank 0 under ``state``; return each source's received rows and the output."""
    controls = {
        "enable_rank_mask": True,
        "active_rank_mask": state.active_rank_mask,
        "abort_state": state.abort,
    }
    valid_counts = torch.zeros(_EP_SIZE, dtype=torch.int32, device="cuda")
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
        source_valid_counts=valid_counts,
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
    return valid_counts, output


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


def test_abort_rearm_and_exclusion_resume_after_peer_lost_mid_collective() -> None:
    """Can a dispatch stuck on a lost peer be aborted from another thread, and the next round
    run on the same workspace after rearming and excluding that peer?"""
    # The dispatch waits for logical rank 1's completion flag until the abort arrives.
    workspace, meta, experts, payload = _collective_inputs()
    health = EPGroupHealth(_EP_SIZE)
    state = MoeExecutionState(health, workspace.device)
    resets = []

    row_owner = SimpleNamespace(
        workspace=workspace, moe_a2a_metainfo=meta, ep_rank=0, ep_size=_EP_SIZE
    )

    def reset_workspace() -> None:
        resets.append(1)
        NVLinkOneSided._reset_local_workspace(row_owner)

    state.retain(_FakeMemory(), reset_workspace)

    counts, output = _collective_round(state, workspace, meta, experts, payload)
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
    # Only the lost peer's rows are dropped; this rank's own rows arrived.
    assert counts.tolist() == [_TOKENS, 0]
    assert torch.count_nonzero(output) == 0

    health.mark_failed(1)
    with pytest.raises(RuntimeError, match="abort record"):
        state.install_mask()
    assert state.active_rank_mask.cpu().tolist() == [3, 0, 0, 0]
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
    with pytest.raises(ValueError, match="rank-mask mode"):
        _frontend(None).bind_execution_state(state)

    state.close()
    assert comms[0].mnnvl_mem.pins == 0


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


def test_for_model_refuses_models_it_cannot_bind() -> None:
    with pytest.raises(RuntimeError, match="NVLinkOneSided"):
        MoeExecutionState.for_model(torch.nn.Linear(1, 1))


@pytest.mark.parametrize("reads_slots", [True, False])
def test_backends_get_the_slots_or_a_compact_copy(reads_slots: bool) -> None:
    comm = _frontend(EPGroupHealth(2))
    slots = DispatchSourceSlots(torch.zeros(2, dtype=torch.int32), source_tokens=2, slot_tokens=3)
    comm._dispatch_state = {"source_slots": slots}
    comm.supports_post_quant_dispatch = lambda: True
    scheduler = ExternalCommMoEScheduler.__new__(ExternalCommMoEScheduler)
    backend = SimpleNamespace(supports_dispatch_receive_slots=lambda: reads_slots)
    scheduler.moe = SimpleNamespace(comm=comm, enable_alltoall=True, backend=backend)
    plan = scheduler._build_comm_plan(
        all_rank_num_tokens=[2, 2], output_dtype=None, do_finalize=False
    )
    # Source s's token i is slot row 3 * s + i; row 2 of each slot holds no token of this forward.
    x = torch.arange(6 * 4).view(6, 4)
    x_sf = torch.arange(6).view(6, 1)
    compact_x, compact_sf = scheduler._compact_receive_slots(x, x_sf)
    if reads_slots:
        assert plan.source_slots is slots
        assert compact_x is x and compact_sf is x_sf
    else:
        assert plan.source_slots is None
        assert compact_x.tolist() == x[[0, 1, 3, 4]].tolist()
        assert compact_sf.view(-1).tolist() == [0, 1, 3, 4]


def test_compact_copy_is_contiguous_with_one_token_per_source() -> None:
    # A draft step sends one token per source; reshaping [sources, 1, hidden] alone would keep the slot stride.
    slots = DispatchSourceSlots(torch.zeros(4, dtype=torch.int32), source_tokens=1, slot_tokens=3)
    x = torch.arange(12 * 5).view(12, 5)
    compact = slots.compact(x)
    assert compact.is_contiguous()
    assert compact.tolist() == x[[0, 3, 6, 9]].tolist()

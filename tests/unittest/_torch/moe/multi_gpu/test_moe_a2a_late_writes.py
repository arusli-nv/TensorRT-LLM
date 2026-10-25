# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A lost EP peer's late writes land only on pages the survivors no longer map.

Two processes share a real MNNVL MoE All-to-All workspace under bound execution. The victim, rank 1,
keeps pushing dispatches into the survivor's workspace row. Its token count differs from the survivor's,
so its rows of the packed layout (source s's rows at [s * T, (s + 1) * T) for runtime token count T)
fall on the survivor's own rows. The survivor, rank 0, aborts the dispatch that waits on the victim,
releases the workspace's backing and remaps fresh pages at the same addresses shared only with itself,
as peer-loss recovery does, then rearms and installs a membership without the victim. Its CUDA graph
captured before the loss then returns exact results while the victim, killed or still writing, reaches
none of its bytes.

``no-remap`` is the negative control: without the remap the victim's writes reach the survivor's live
rows.
"""

import itertools
import os
import subprocess
import sys
import time
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from _mnnvl_recovery_utils import StoreComm, mnnvl_skip_reason, workspace_offset
from torch.distributed import TCPStore

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import MoeExecutionState
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm.bindings import internal
from tensorrt_llm.mapping import Mapping

_EP_SIZE, _SURVIVOR, _VICTIM = 2, 0, 1
_CAPACITY, _HIDDEN = 1024, 7168
# The victim's token count: its hidden rows [_VICTIM_TOKENS, 2 * _VICTIM_TOKENS) are the survivor's own rows.
_VICTIM_TOKENS = _CAPACITY // 2
_GRACE_S = 2.0
# The negative control polls the live rows this often until the victim's writes show, at most this long.
_POLL_S, _CONTROL_DEADLINE_S = 0.05, 10.0
_TIMEOUT = timedelta(seconds=300)


def _rows(source: int) -> torch.Tensor:
    """[_CAPACITY, _HIDDEN] bf16 rows, each distinct: columns 0-2 encode the token and the source, exactly in
    bf16 (integers up to 256), so a misplaced or mixed-up row reads as wrong values."""
    token = torch.arange(_CAPACITY, device="cuda")[:, None]
    values = (token + torch.arange(_HIDDEN, device="cuda")) % 128
    values[:, 0:1] = token % 128
    values[:, 1:2] = token // 128
    values[:, 2] = source
    return values.bfloat16()


def _controls(mask: torch.Tensor, abort: torch.Tensor) -> dict:
    return {"enable_rank_mask": True, "active_rank_mask": mask, "abort_state": abort}


def _dispatch(workspace, meta, rank, experts, payloads, tokens, mask, abort):
    """Dispatch [*payloads, experts] of ``tokens`` tokens per rank."""
    args = (experts, [*payloads, experts], workspace, meta, tokens, rank, _EP_SIZE, 1, _EP_SIZE)
    return torch.ops.trtllm.moe_a2a_dispatch(*args, **_controls(mask, abort))


def _bound_state(
    memory: MnnvlMemory, workspace: torch.Tensor, meta: torch.Tensor
) -> MoeExecutionState:
    """The survivor's MoeExecutionState, bound by an NVLinkOneSided layer on ``workspace`` as a model binds it.

    The layer carries only the fields binding, the workspace reset and the remap read; no MoE layer is built.
    """
    health = EPGroupHealth(_EP_SIZE)
    state = MoeExecutionState(health, workspace.device)
    comm = NVLinkOneSided.__new__(NVLinkOneSided)
    comm.can_use_cft_counted_writes = False
    comm.ep_group_health = health
    comm._rank_mask_enabled = True
    comm._execution_state = None
    comm._dispatch_state = {"phase": "idle"}
    comm.mnnvl_mem, comm.workspace = memory, workspace
    # The workspace lifecycle supplies the metainfo and the remap of the real allocation; this layer runs no
    # watchdog.
    comm._workspace_lifecycle = SimpleNamespace(
        metainfo=meta,
        watchdog_for=lambda layer: None,
        release_backing=memory.release_backing,
        remap=lambda gather, reset: (memory.remap(gather), reset()),
    )
    comm.ep_rank, comm.ep_size = _SURVIVOR, _EP_SIZE
    comm.bind_execution_state(state)
    return state


def _survivors_gather(value: Any) -> list[Any]:
    """The remap's exchange among the survivors: the survivor alone, None for the victim."""
    return [value if rank == _SURVIVOR else None for rank in range(_EP_SIZE)]


def _victim_steps(store: TCPStore) -> int:
    return int(store.add("steps", 0))


def _victim(store: TCPStore, workspace: torch.Tensor, meta: torch.Tensor) -> None:
    # A pending abort request lets every dispatch push all its rows and then return instead of
    # waiting for the silent survivor, so new NVLink writes are always in flight.
    mask = torch.tensor([3, 0, 0, 0], dtype=torch.uint64, device="cuda")
    abort = torch.tensor([1, 0], dtype=torch.int32).pin_memory()
    experts = torch.zeros((_VICTIM_TOKENS, 1), dtype=torch.int32, device="cuda")  # expert 0: rank 0
    raw = torch.empty((_VICTIM_TOKENS, 2 * _HIDDEN), dtype=torch.uint8, device="cuda")
    payload = raw.view(torch.bfloat16)
    store.wait(["go"])
    for step in itertools.count(1):
        # Alternating bytes, so each write changes every byte it reaches, not only the first after a reset.
        raw.fill_(0xC3 if step % 2 else 0x3C)
        _dispatch(workspace, meta, _VICTIM, experts, [payload], _VICTIM_TOKENS, mask, abort)
        if step % 16 == 0:
            torch.cuda.synchronize()
            store.add("steps", 1)
        if step == 64:
            store.set("running", "1")


def _survivor(
    store: TCPStore, memory: MnnvlMemory, workspace: torch.Tensor, meta: torch.Tensor, mode: str
) -> None:
    row = workspace[_SURVIVOR]
    state = _bound_state(memory, workspace, meta)
    mask, abort = state.active_rank_mask, state.abort
    experts = torch.zeros((_CAPACITY, 1), dtype=torch.int32, device="cuda")
    payload = torch.empty((_CAPACITY, _HIDDEN), dtype=torch.bfloat16, device="cuda")

    # Captured with full membership; replayed only after the victim is lost.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        received, offset, _ = _dispatch(
            workspace, meta, _SURVIVOR, experts, [payload], _CAPACITY, mask, abort
        )
        args = (received[0], _CAPACITY, workspace, meta, _CAPACITY, _SURVIVOR, _EP_SIZE, 1)
        output = torch.ops.trtllm.moe_a2a_combine(*args, offset, False, **_controls(mask, abort))
    # moeA2ADispatchOp returns each payload's receive buffer as [ep_size, runtime tokens, ...] from
    # DISPATCH_PAYLOAD_OFFSET: source s's rows are [s, :] (moeA2ADispatchKernel writes source s's token at row
    # s * max_tokens_per_rank + target_index).
    assert offset == workspace_offset(meta, "COMBINE_INPUT")
    assert [tuple(r.shape) for r in received] == [
        (_EP_SIZE, _CAPACITY, _HIDDEN),
        (_EP_SIZE, _CAPACITY, 1),
    ]
    assert received[0].data_ptr() == row.data_ptr() + workspace_offset(meta, "DISPATCH_PAYLOAD")

    torch.cuda.synchronize()
    store.set("go", "1")
    store.wait(["proceed"])

    # The victim's first dispatch published its round-1 flag, with its complete rows, before its pending abort
    # released it, so the survivor's next dispatch completes on that flag. The combine after it waits for the
    # lost victim, which publishes no further flag, until recovery aborts it.
    done = torch.cuda.Event()
    eager, eager_offset, _ = _dispatch(
        workspace, meta, _SURVIVOR, experts, [payload], _CAPACITY, mask, abort
    )
    torch.cuda.synchronize()
    assert abort.tolist() == [0, 0]
    eager_args = (eager[0], _CAPACITY, workspace, meta, _CAPACITY, _SURVIVOR, _EP_SIZE, 1)
    torch.ops.trtllm.moe_a2a_combine(*eager_args, eager_offset, False, **_controls(mask, abort))
    done.record()
    time.sleep(0.2)
    assert not done.query(), "combine completed without the lost victim"
    state.request_abort()
    deadline = time.monotonic() + 10
    while not done.query():
        assert time.monotonic() < deadline, "combine kept waiting after the abort request"
        time.sleep(0.01)
    assert abort.tolist() == [1, 1]

    # Recovery: fresh pages at the same addresses among the survivors, then the new membership.
    state.health.mark_failed(_VICTIM)
    if mode != "no-remap":
        assert state.release_workspaces({_VICTIM}) > 0
        state.remap_workspaces(_survivors_gather)
    state.rearm()
    state.install_mask()
    assert mask.cpu().tolist() == [1 << _SURVIVOR, 0, 0, 0]

    torch.cuda.synchronize()
    before = row.clone()
    steps = _victim_steps(store)
    # The victim alternates two patterns, so a single later look can find a byte it rewrote unchanged: look
    # every _POLL_S.
    if mode == "no-remap":
        live = received[0][_SURVIVOR]
        start = live.data_ptr() - row.data_ptr()
        end = start + live.numel() * live.element_size()
        deadline = time.monotonic() + _CONTROL_DEADLINE_S
        while not bool((row[start:end] != before[start:end]).any()):
            assert time.monotonic() < deadline, (
                "the victim's writes missed the survivor's live rows without the remap"
            )
            time.sleep(_POLL_S)
        state.close()
        return
    deadline = time.monotonic() + _GRACE_S
    while True:
        changed = (row != before).nonzero().flatten()
        assert changed.numel() == 0, f"bytes of the remapped row changed: {changed[:16]}"
        if time.monotonic() >= deadline:
            break
        time.sleep(_POLL_S)
    if mode != "killed":
        assert _victim_steps(store) > steps, "the victim stopped writing"

    values = _rows(_SURVIVOR)
    steps = _victim_steps(store)
    for step in range(4):
        payload.copy_(values + step)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, payload, rtol=0, atol=0)
        # The survivor's rows hold each of its tokens once, in the order of route_dispatch_token's
        # atomicAdd on send_counters (target_index), not by token: read each row's token from its columns 0-1.
        rows = received[0][_SURVIVOR]
        tokens = ((rows[:, 0].float() - step) + 128 * (rows[:, 1].float() - step)).long()
        assert torch.equal(tokens.sort().values, torch.arange(_CAPACITY, device="cuda"))
        assert torch.equal(rows, payload[tokens]), "a received row differs from its token's row"
        assert torch.all(received[1][_SURVIVOR] == 0)
        # Nothing writes the lost victim's rows: they keep the reset's zeros.
        assert torch.count_nonzero(received[0][_VICTIM]) == 0
        assert torch.count_nonzero(received[1][_VICTIM]) == 0
        assert abort.tolist() == [0, 0]
    if mode == "writing":
        assert _victim_steps(store) > steps, "the victim stopped writing during the replays"
    state.close()


def _worker(rank: int, port: int, mode: str) -> None:
    store = TCPStore("127.0.0.1", port, is_master=False, timeout=_TIMEOUT)
    torch.cuda.set_device(rank)
    mapping = Mapping(_EP_SIZE, rank, _EP_SIZE, tp_size=_EP_SIZE, moe_ep_size=_EP_SIZE)
    MnnvlMemory.initialize()
    comm = StoreComm(store, rank, _EP_SIZE)
    MnnvlMemory.comm = comm
    MnnvlMemory.comm_signature = MnnvlMemory.comm_layout_signature(mapping)
    meta = NVLinkOneSided._make_workspace_layout(
        _EP_SIZE, 1, _CAPACITY, _HIDDEN, torch.bfloat16, None, 0, False, False
    )
    memory = MnnvlMemory(mapping, int(meta[int(internal.thop.MOE_A2A_WORKSPACE_SIZE_INDEX)]))
    workspace = memory.as_torch_strided_tensor(torch.uint8)
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, rank, _EP_SIZE, barrier=False)
    # No peer may write into a row before its owner has initialized it.
    comm.barrier()
    if rank == _SURVIVOR:
        _survivor(store, memory, workspace, meta, mode)
    else:
        _victim(store, workspace, meta)


@pytest.mark.parametrize(
    "mode, kill_delay_s",
    [
        ("killed", 0.0),
        ("killed", 0.003),
        ("killed", 0.05),
        ("writing", None),
        ("no-remap", None),
    ],
)
def test_lost_peer_late_writes_land_only_on_orphaned_pages(
    mode: str, kill_delay_s: float | None
) -> None:
    """After the remap the survivor's results stay exact and none of its bytes change, whether the victim was
    killed mid-dispatch (``killed``) or keeps writing to the end (``writing``); ``no-remap`` is the negative
    control, whose victim writes must reach the survivor's live rows."""
    reason = mnnvl_skip_reason(_EP_SIZE)
    if reason:
        pytest.skip(reason)
    store = TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False, timeout=_TIMEOUT)
    # Keep the workers off any enclosing launcher's process-management interface: one is killed.
    launcher_prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(launcher_prefixes)}
    workers = [
        subprocess.Popen([sys.executable, __file__, str(rank), str(store.port), mode], env=env)
        for rank in range(_EP_SIZE)
    ]
    survivor, victim = workers[_SURVIVOR], workers[_VICTIM]
    try:
        deadline = time.monotonic() + _TIMEOUT.total_seconds()
        while not store.check(["running"]):
            assert all(worker.poll() is None for worker in workers), "worker exited during setup"
            assert time.monotonic() < deadline, "victim never started dispatching"
            time.sleep(0.01)
        if kill_delay_s is not None:
            time.sleep(kill_delay_s)
            victim.kill()
            victim.wait()
        store.set("proceed", "1")
        assert survivor.wait(timeout=_TIMEOUT.total_seconds()) == 0
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait()


if __name__ == "__main__":
    _worker(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])

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
"""A failed EP peer's late writes land only on pages the survivors no longer map.

Four processes share a real MNNVL MoE All-to-All workspace. The victim, the last rank, keeps dispatching into the
survivors' rows, where its smaller token count lands on source 0's rows. The survivors remap the workspace with
fresh pages at the same addresses, rearm, and replay CUDA graphs captured before the failure. ``no-remap`` is the
negative control, in which the victim's writes reach the survivors' active rows.
"""

import itertools
import os
import pickle
import subprocess
import sys
import time
from datetime import timedelta
from typing import Any

import pytest
import torch
from torch.distributed import TCPStore

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.distributed.membership_mnnvl_memory import MembershipMnnvlMemory
from tensorrt_llm._torch.mnnvl_alltoall_workspace import _MnnvlAlltoAllWorkspaceLifecycle
from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import MoeExecutionState
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm._wide_ep_fault_tolerance import set_wide_ep_fault_tolerance_enabled
from tensorrt_llm.bindings import internal
from tensorrt_llm.mapping import Mapping

_EP_SIZE, _CAPACITY, _HIDDEN = 4, 1024, 7168
_VICTIM = _EP_SIZE - 1
_GRACE_S = 2.0
# The negative control polls the active rows this often until the victim's writes show, at most this long.
_POLL_S, _CONTROL_DEADLINE_S = 0.05, 10.0
_TIMEOUT = timedelta(seconds=300)


class _StoreComm:
    """Implement the communicator subset MembershipMnnvlMemory uses over a TCPStore, so a peer can be killed.

    Calling it gathers a value from the ranks in ``members``, with None for the others.
    """

    def __init__(self, store: TCPStore, rank: int, name: str, members: range) -> None:
        self.store, self.rank, self.name, self.members, self.calls = store, rank, name, members, 0

    def Get_rank(self) -> int:
        return self.rank

    def Get_size(self) -> int:
        return _EP_SIZE

    def __call__(self, value: Any) -> list[Any]:
        self.calls += 1
        self.store.set(f"{self.name}/{self.calls}/{self.rank}", pickle.dumps(value))
        return [
            pickle.loads(self.store.get(f"{self.name}/{self.calls}/{rank}"))
            if rank in self.members
            else None
            for rank in range(_EP_SIZE)
        ]

    allgather = __call__

    def barrier(self) -> None:
        self(None)


def _rows(source: int) -> torch.Tensor:
    """Return distinct [_CAPACITY, _HIDDEN] bf16 rows of ``source``.

    Columns 0-2 encode the token and the source exactly, so a misplaced row reads as wrong values.
    """
    token = torch.arange(_CAPACITY, device="cuda")[:, None]
    values = (token + torch.arange(_HIDDEN, device="cuda")) % 128
    values[:, 0:1] = token % 128
    values[:, 1:2] = token // 128
    values[:, 2] = source
    return values.bfloat16()


def _decode_tokens(rows: torch.Tensor, step: int) -> torch.Tensor:
    """Return the tokens of received ``_rows(source) + step`` rows, read from their columns 0-1."""
    return ((rows[:, 0].float() - step) + 128 * (rows[:, 1].float() - step)).long()


def _mask_words(ranks) -> list[int]:
    """Return the rank mask of ``ranks`` as the kernels' four uint64 words, for fewer than 64 ranks."""
    return [sum(1 << rank for rank in ranks), 0, 0, 0]


def _controls(mask: torch.Tensor, abort: torch.Tensor) -> dict:
    return {"enable_rank_mask": True, "active_rank_mask": mask, "abort_state": abort}


def _dispatch(workspace, meta, rank, experts, payloads, tokens, mask, abort):
    """Dispatch [*payloads, experts] of ``tokens`` tokens per rank. Expert e is on rank e."""
    args = (experts, [*payloads, experts], workspace, meta, tokens, rank, _EP_SIZE, 1, _EP_SIZE)
    return torch.ops.trtllm.moe_a2a_dispatch(*args, **_controls(mask, abort))


def _combine(received, offset, workspace, meta, rank, mask, abort):
    args = (received, _CAPACITY, workspace, meta, _CAPACITY, rank, _EP_SIZE, 1)
    return torch.ops.trtllm.moe_a2a_combine(*args, offset, False, **_controls(mask, abort))


def _bound_state(
    memory: MembershipMnnvlMemory, workspace: torch.Tensor, meta: torch.Tensor, rank: int
) -> MoeExecutionState:
    """Return a survivor's MoeExecutionState, bound by an NVLinkOneSided layer on ``workspace``.

    The layer carries only the fields that binding and the workspace reset read, with the workspace's real
    lifecycle. No MoE layer is built and no watchdog runs.
    """
    NVLinkOneSided._init_constants()
    health = EPGroupHealth(_EP_SIZE)
    state = MoeExecutionState(health, workspace.device)
    comm = NVLinkOneSided.__new__(NVLinkOneSided)
    comm.can_use_cft_counted_writes = False
    comm.ep_group_health = health
    comm._rank_mask_enabled = True
    comm.execution_state = None
    comm._bound = False
    comm._dispatch_state = {"phase": "idle"}
    comm.workspace = workspace
    comm.ep_rank, comm.ep_size = rank, _EP_SIZE
    comm._workspace_lifecycle = _MnnvlAlltoAllWorkspaceLifecycle(
        workspace_state={},
        memory=memory,
        workspace=workspace,
        metainfo=meta,
        metainfo_index={
            "FLAG_VAL_OFFSET_INDEX": NVLinkOneSided.FLAG_VAL_OFFSET_INDEX,
            "DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX": NVLinkOneSided.DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX,
            "COMBINE_COMPLETION_FLAGS_OFFSET_INDEX": NVLinkOneSided.COMBINE_COMPLETION_FLAGS_OFFSET_INDEX,
        },
        ep_rank=rank,
        ep_size=_EP_SIZE,
        health=health,
    )
    comm.bind_execution_state(state)
    return state


def _victim_steps(store: TCPStore) -> int:
    return int(store.add("steps", 0))


def _victim(store: TCPStore, workspace: torch.Tensor, meta: torch.Tensor) -> None:
    # A pending abort request lets every dispatch push all its rows and return without waiting for the silent
    # survivors, so new NVLink writes are always in flight. A dispatch that returns early leaves its round
    # incomplete, and an incomplete round routes only to this rank. Resetting the row before every dispatch
    # keeps the writes going into the survivors' rows.
    mask = torch.tensor(_mask_words(range(_EP_SIZE)), dtype=torch.uint64, device="cuda")
    abort = torch.zeros(
        int(internal.thop.MOE_A2A_ABORT_STATE_WORDS), dtype=torch.int32
    ).pin_memory()
    abort[0] = 1
    tokens = _CAPACITY // _EP_SIZE
    # Its tokens go to every survivor in turn.
    experts = (torch.arange(tokens, device="cuda") % _VICTIM).int()[:, None]
    raw = torch.empty((tokens, 2 * _HIDDEN), dtype=torch.uint8, device="cuda")
    payload = raw.view(torch.bfloat16)
    store.wait([f"captured/{survivor}" for survivor in range(_VICTIM)])
    for step in itertools.count(1):
        torch.ops.trtllm.moe_a2a_initialize(workspace, meta, _VICTIM, _EP_SIZE, barrier=False)
        # Alternating bytes, so each write changes every byte it reaches, not only the first after a reset.
        raw.fill_(0xC3 if step % 2 else 0x3C)
        _dispatch(workspace, meta, _VICTIM, experts, [payload], tokens, mask, abort)
        if step % 16 == 0:
            torch.cuda.synchronize()
            store.add("steps", 1)
        if step == 64:
            store.set("running", "1")


def _check_late_writes(store: TCPStore, row: torch.Tensor, active: torch.Tensor, mode: str) -> None:
    """Check that no byte of this survivor's ``row`` changes for _GRACE_S while the victim keeps writing.

    In ``no-remap`` mode, check instead that the victim's writes reach ``active``, source 0's rows of ``row``.
    """
    torch.cuda.synchronize()
    before = row.clone()
    steps = _victim_steps(store)
    # The victim alternates two patterns, so one late look could miss a rewritten byte. Look every _POLL_S.
    if mode == "no-remap":
        start = active.data_ptr() - row.data_ptr()
        end = start + active.numel() * active.element_size()
        deadline = time.monotonic() + _CONTROL_DEADLINE_S
        while not bool((row[start:end] != before[start:end]).any()):
            assert time.monotonic() < deadline, (
                "the victim's writes missed the survivor's active rows without the remap"
            )
            time.sleep(_POLL_S)
        return
    deadline = time.monotonic() + _GRACE_S
    while True:
        changed = (row != before).nonzero().flatten()
        assert changed.numel() == 0, f"bytes of the remapped row changed: {changed[:16]}"
        if time.monotonic() >= deadline:
            break
        time.sleep(_POLL_S)
    if mode == "writing":
        assert _victim_steps(store) > steps, "the victim stopped writing"


def _survivor(
    store: TCPStore,
    memory: MembershipMnnvlMemory,
    workspace: torch.Tensor,
    meta: torch.Tensor,
    rank: int,
    mode: str,
) -> None:
    """Remap with the other survivors and replay a graph whose tokens go to every survivor."""
    row = workspace[rank]
    state = _bound_state(memory, workspace, meta, rank)
    mask, abort = state.active_rank_mask, state.abort
    token = torch.arange(_CAPACITY, device="cuda")
    # Every source sends token t to survivor t % _VICTIM.
    experts = (token % _VICTIM).int()[:, None]
    payload = torch.empty((_CAPACITY, _HIDDEN), dtype=torch.bfloat16, device="cuda")

    # Capture with full membership and replay only after the victim failed.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        received, offset, _ = _dispatch(
            workspace, meta, rank, experts, [payload], _CAPACITY, mask, abort
        )
        output = _combine(received[0], offset, workspace, meta, rank, mask, abort)
    torch.cuda.synchronize()
    store.set(f"captured/{rank}", "1")
    store.wait(["proceed"])

    gather = _StoreComm(store, rank, "survivors", range(_VICTIM))
    state.health.mark_failed(_VICTIM)
    if mode != "no-remap":
        state.remap_workspaces(gather, frozenset(range(_EP_SIZE)))
        assert memory.mapped
    state.rearm()
    state.install_mask()
    assert mask.cpu().tolist() == _mask_words(range(_VICTIM))
    # No survivor writes into a peer's row before that peer has rearmed or while it checks its row.
    gather(None)
    _check_late_writes(store, row, received[0][0], mode)
    gather(None)
    if mode == "no-remap":
        return

    mine = token[token % _VICTIM == rank]
    steps = _victim_steps(store)
    for step in range(4):
        payload.copy_(_rows(rank) + step)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, payload, rtol=0, atol=0)
        # Every survivor's tokens for this rank arrive once each in this rank's fresh pages, in dispatch order.
        for source in range(_VICTIM):
            rows = received[0][source][: mine.numel()]
            tokens = _decode_tokens(rows, step)
            assert torch.equal(tokens.sort().values, mine), f"tokens from source {source}"
            assert torch.equal(rows, (_rows(source) + step)[tokens]), f"rows from source {source}"
        # The failed victim's rows keep the zeros of the reset.
        assert torch.count_nonzero(received[0][_VICTIM]) == 0
        assert abort[:2].tolist() == [0, 0]
        # No survivor dispatches the next round into this rank's rows while it reads them.
        gather(None)
    if mode == "writing":
        assert _victim_steps(store) > steps, "the victim stopped writing during the replays"


def _worker(rank: int, port: int, mode: str) -> None:
    store = TCPStore("127.0.0.1", port, is_master=False, timeout=_TIMEOUT)
    # Every rank of a deployment that survives a failed rank runs with Wide-EP fault tolerance on.
    set_wide_ep_fault_tolerance_enabled(True)
    torch.cuda.set_device(rank)
    mapping = Mapping(_EP_SIZE, rank, _EP_SIZE, tp_size=_EP_SIZE, moe_ep_size=_EP_SIZE)
    MembershipMnnvlMemory.initialize()
    comm = _StoreComm(store, rank, "world", range(_EP_SIZE))
    MembershipMnnvlMemory.comm = comm
    MembershipMnnvlMemory.comm_signature = MembershipMnnvlMemory.comm_layout_signature(mapping)
    meta = NVLinkOneSided._make_workspace_layout(
        _EP_SIZE, 1, _CAPACITY, _HIDDEN, torch.bfloat16, None, 0, False, False, rank_mask=True
    )
    memory = MembershipMnnvlMemory(
        mapping, int(meta[int(internal.thop.MOE_A2A_WORKSPACE_SIZE_INDEX)])
    )
    workspace = memory.as_torch_strided_tensor(torch.uint8)
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, rank, _EP_SIZE, barrier=False)
    # No peer may write into a row before its owner has initialized it.
    comm.barrier()
    if rank == _VICTIM:
        _victim(store, workspace, meta)
    else:
        _survivor(store, memory, workspace, meta, rank, mode)


@pytest.mark.parametrize(
    "mode, kill_delay_s", [("killed", 0.003), ("writing", None), ("no-remap", None)]
)
def test_late_writes_land_on_orphaned_pages(mode: str, kill_delay_s: float | None) -> None:
    """Check that the survivors' results stay exact and none of their bytes change after the remap.

    The victim is killed mid-dispatch (``killed``) or keeps writing (``writing``). In the ``no-remap`` control,
    the victim's writes must reach the survivors' active rows.
    """
    if torch.cuda.device_count() < _EP_SIZE:
        pytest.skip(f"Requires {_EP_SIZE} local GPUs")
    MembershipMnnvlMemory.initialize()
    if not MembershipMnnvlMemory.supports_mnnvl():
        pytest.skip("Requires an NVLink fabric supported by MNNVL")
    store = TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False, timeout=_TIMEOUT)
    # One worker is killed, so keep the workers off any enclosing launcher's process management.
    launcher_prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(launcher_prefixes)}
    workers = [
        subprocess.Popen([sys.executable, __file__, str(rank), str(store.port), mode], env=env)
        for rank in range(_EP_SIZE)
    ]
    *survivors, victim = workers
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
        for survivor in survivors:
            assert survivor.wait(timeout=_TIMEOUT.total_seconds()) == 0
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait()


if __name__ == "__main__":
    _worker(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])

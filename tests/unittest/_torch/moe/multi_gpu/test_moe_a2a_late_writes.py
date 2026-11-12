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

The processes share a real MNNVL MoE All-to-All workspace under bound execution. The victim, the last
rank, keeps pushing dispatches into the survivors' workspace rows. Its token count, _CAPACITY // ep_size,
differs from the survivors', so its rows of the packed layout fall on rows of source 0, a survivor. The
survivors remap the workspace with fresh pages at the same addresses, shared only among themselves through
``MoeExecutionState.remap_workspaces`` over a TCPStore. They rearm and install a membership without the
victim. CUDA graphs captured before the failure return exact results, and the victim reaches none of the
survivors' bytes, whether killed or still writing.

With two ranks, the survivor first aborts the combine that waits on the victim. With four, three survivors
import each other's fresh pages and exchange tokens among themselves after the remap. In ``admitted`` mode
the victim's replacement joins and keeps the survivors' pages, and every rank exchanges tokens with every
other.

``no-remap`` is the negative control. Without the remap, the victim's writes reach the survivors' active
rows.
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
from tensorrt_llm._torch.distributed.membership import StoreGroup
from tensorrt_llm._torch.distributed.membership_mnnvl_memory import MembershipMnnvlMemory
from tensorrt_llm._torch.mnnvl_alltoall_workspace import _MnnvlAlltoAllWorkspaceLifecycle
from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import MoeExecutionState
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm._wide_ep_fault_tolerance import set_wide_ep_fault_tolerance_enabled
from tensorrt_llm.bindings import internal
from tensorrt_llm.mapping import Mapping

_CAPACITY, _HIDDEN = 1024, 7168
_GRACE_S = 2.0
# The negative control polls the active rows this often until the victim's writes show, at most this long.
_POLL_S, _CONTROL_DEADLINE_S = 0.05, 10.0
_TIMEOUT = timedelta(seconds=300)


class _StoreComm:
    """Implement the communicator subset MembershipMnnvlMemory uses over a TCPStore, so a peer can be killed."""

    def __init__(self, store: TCPStore, rank: int, size: int) -> None:
        self.store, self.rank, self.size, self.calls = store, rank, size, 0

    def Get_rank(self) -> int:
        return self.rank

    def Get_size(self) -> int:
        return self.size

    def allgather(self, value: Any) -> list[Any]:
        self.calls += 1
        self.store.set(f"gather/{self.calls}/{self.rank}", pickle.dumps(value))
        return [
            pickle.loads(self.store.get(f"gather/{self.calls}/{rank}")) for rank in range(self.size)
        ]

    def barrier(self) -> None:
        self.allgather(None)


class _StoreGather:
    """Exchange values among the EP ranks in ``active`` over a TCPStore, under keys of ``name``.

    The result has one entry per EP rank, with None for a rank outside ``active``.
    """

    def __init__(self, store: TCPStore, rank: int, ep_size: int, name: str, active: range) -> None:
        self.store, self.rank, self.ep_size, self.calls = store, rank, ep_size, 0
        self.name, self.active = name, active

    def __call__(self, value: Any) -> list[Any]:
        self.calls += 1
        self.store.set(f"{self.name}/{self.calls}/{self.rank}", pickle.dumps(value))
        return [
            pickle.loads(self.store.get(f"{self.name}/{self.calls}/{rank}"))
            if rank in self.active
            else None
            for rank in range(self.ep_size)
        ]


def _survivors_gather(store: TCPStore, rank: int, ep_size: int) -> _StoreGather:
    """Return the remap's exchange among the survivors, which are every rank but the last."""
    return _StoreGather(store, rank, ep_size, "survivors", range(ep_size - 1))


def _admission_gather(store: TCPStore, rank: int, ep_size: int) -> _StoreGather:
    """Return the admission's exchange among the survivors and the victim's replacement."""
    return _StoreGather(store, rank, ep_size, "admission", range(ep_size))


def _offset(meta: torch.Tensor, name: str) -> int:
    """Return the offset of region ``name`` in a workspace row of layout ``meta``."""
    return int(meta[int(getattr(internal.thop, f"MOE_A2A_{name}_OFFSET_INDEX"))])


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


def _dispatch(workspace, meta, rank, ep_size, experts, payloads, tokens, mask, abort):
    """Dispatch [*payloads, experts] of ``tokens`` tokens per rank. Expert e is on rank e."""
    args = (experts, [*payloads, experts], workspace, meta, tokens, rank, ep_size, 1, ep_size)
    return torch.ops.trtllm.moe_a2a_dispatch(*args, **_controls(mask, abort))


def _combine(received, offset, workspace, meta, rank, ep_size, mask, abort):
    args = (received, _CAPACITY, workspace, meta, _CAPACITY, rank, ep_size, 1)
    return torch.ops.trtllm.moe_a2a_combine(*args, offset, False, **_controls(mask, abort))


def _bound_state(
    memory: MembershipMnnvlMemory,
    workspace: torch.Tensor,
    meta: torch.Tensor,
    rank: int,
    ep_size: int,
) -> MoeExecutionState:
    """Return a survivor's MoeExecutionState, bound by an NVLinkOneSided layer on ``workspace``.

    The layer carries only the fields that binding and the workspace reset read, with the workspace's real
    lifecycle. No MoE layer is built and no watchdog runs.
    """
    NVLinkOneSided._init_constants()
    health = EPGroupHealth(ep_size)
    state = MoeExecutionState(health, workspace.device)
    comm = NVLinkOneSided.__new__(NVLinkOneSided)
    comm.can_use_cft_counted_writes = False
    comm.ep_group_health = health
    comm._rank_mask_enabled = True
    comm.execution_state = None
    comm._bound = False
    comm._dispatch_state = {"phase": "idle"}
    comm.workspace = workspace
    comm.ep_rank, comm.ep_size = rank, ep_size
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
        ep_size=ep_size,
        health=health,
    )
    comm.bind_execution_state(state)
    return state


def _victim_steps(store: TCPStore) -> int:
    return int(store.add("steps", 0))


def _victim(store: TCPStore, workspace: torch.Tensor, meta: torch.Tensor, ep_size: int) -> None:
    rank = ep_size - 1
    # A pending abort request lets every dispatch push all its rows and return without waiting for the silent
    # survivors, so new NVLink writes are always in flight. A dispatch that returns early leaves its round
    # incomplete, and an incomplete round routes only to this rank. Resetting the row before every dispatch
    # keeps the writes going into the survivors' rows.
    mask = torch.tensor(_mask_words(range(ep_size)), dtype=torch.uint64, device="cuda")
    abort = torch.zeros(
        int(internal.thop.MOE_A2A_ABORT_STATE_WORDS), dtype=torch.int32
    ).pin_memory()
    abort[0] = 1
    tokens = _CAPACITY // ep_size
    # Its tokens go to every survivor in turn.
    experts = (torch.arange(tokens, device="cuda") % rank).int()[:, None]
    raw = torch.empty((tokens, 2 * _HIDDEN), dtype=torch.uint8, device="cuda")
    payload = raw.view(torch.bfloat16)
    store.wait([f"captured/{survivor}" for survivor in range(rank)])
    for step in itertools.count(1):
        torch.ops.trtllm.moe_a2a_initialize(workspace, meta, rank, ep_size)
        # Alternating bytes, so each write changes every byte it reaches, not only the first after a reset.
        raw.fill_(0xC3 if step % 2 else 0x3C)
        _dispatch(workspace, meta, rank, ep_size, experts, [payload], tokens, mask, abort)
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
    mode: str,
) -> None:
    """Run rank 0 of two. Abort the combine waiting on the victim, remap alone and replay the graph."""
    rank, ep_size, victim = 0, 2, 1
    row = workspace[rank]
    state = _bound_state(memory, workspace, meta, rank, ep_size)
    mask, abort = state.active_rank_mask, state.abort
    experts = torch.zeros((_CAPACITY, 1), dtype=torch.int32, device="cuda")
    payload = torch.empty((_CAPACITY, _HIDDEN), dtype=torch.bfloat16, device="cuda")

    # Capture with full membership and replay only after the victim failed.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        received, offset, _ = _dispatch(
            workspace, meta, rank, ep_size, experts, [payload], _CAPACITY, mask, abort
        )
        output = _combine(received[0], offset, workspace, meta, rank, ep_size, mask, abort)
    # Each payload's receive buffer is [ep_size, runtime tokens, ...] from DISPATCH_PAYLOAD.
    assert offset == _offset(meta, "COMBINE_INPUT")
    assert [tuple(r.shape) for r in received] == [
        (ep_size, _CAPACITY, _HIDDEN),
        (ep_size, _CAPACITY, 1),
    ]
    assert received[0].data_ptr() == row.data_ptr() + _offset(meta, "DISPATCH_PAYLOAD")

    torch.cuda.synchronize()
    store.set(f"captured/{rank}", "1")
    store.wait(["proceed"])

    # The victim's first dispatch published its round flag and complete rows before its abort left the round
    # incomplete, so the survivor's next dispatch completes on that flag. The combine after it waits for the
    # victim until the survivor aborts it.
    done = torch.cuda.Event()
    eager, eager_offset, _ = _dispatch(
        workspace, meta, rank, ep_size, experts, [payload], _CAPACITY, mask, abort
    )
    torch.cuda.synchronize()
    assert abort[:2].tolist() == [0, 0]
    _combine(eager[0], eager_offset, workspace, meta, rank, ep_size, mask, abort)
    done.record()
    time.sleep(0.2)
    assert not done.query(), "combine completed without the failed victim"
    state.request_abort()
    deadline = time.monotonic() + 10
    while not done.query():
        assert time.monotonic() < deadline, "combine kept waiting after the abort request"
        time.sleep(0.01)
    assert abort[:2].tolist() == [1, 1]

    # Map fresh pages at the same addresses among the survivors and install the new membership.
    state.health.mark_failed(victim)
    if mode != "no-remap":
        state.remap_workspaces(_survivors_gather(store, rank, ep_size), frozenset(range(ep_size)))
        assert memory.mapped
    state.rearm()
    state.install_mask()
    assert mask.cpu().tolist() == _mask_words([rank])

    _check_late_writes(store, row, received[0][0], mode)
    if mode == "no-remap":
        return

    values = _rows(rank)
    steps = _victim_steps(store)
    for step in range(4):
        payload.copy_(values + step)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, payload, rtol=0, atol=0)
        # The survivor's rows hold each of its tokens once, in dispatch order.
        rows = received[0][rank]
        tokens = _decode_tokens(rows, step)
        assert torch.equal(tokens.sort().values, torch.arange(_CAPACITY, device="cuda"))
        assert torch.equal(rows, payload[tokens]), "a received row differs from its token's row"
        assert torch.all(received[1][rank] == 0)
        # The failed victim's rows keep the zeros of the reset.
        assert torch.count_nonzero(received[0][victim]) == 0
        assert torch.count_nonzero(received[1][victim]) == 0
        assert abort[:2].tolist() == [0, 0]
    if mode == "writing":
        assert _victim_steps(store) > steps, "the victim stopped writing during the replays"


def _survivor_with_peers(
    store: TCPStore,
    memory: MembershipMnnvlMemory,
    workspace: torch.Tensor,
    meta: torch.Tensor,
    rank: int,
    ep_size: int,
    mode: str,
) -> None:
    """Run one of several survivors. Remap with the others and replay a graph whose tokens go to every survivor.

    In ``admitted`` mode, also admit the victim's replacement and replay a graph whose tokens go to every rank.
    """
    victim = ep_size - 1
    row = workspace[rank]
    state = _bound_state(memory, workspace, meta, rank, ep_size)
    mask, abort = state.active_rank_mask, state.abort
    token = torch.arange(_CAPACITY, device="cuda")
    # Every source sends token t to survivor t % victim.
    experts = (token % victim).int()[:, None]
    payload = torch.empty((_CAPACITY, _HIDDEN), dtype=torch.bfloat16, device="cuda")

    # Capture with full membership and replay only after the victim failed.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        received, offset, _ = _dispatch(
            workspace, meta, rank, ep_size, experts, [payload], _CAPACITY, mask, abort
        )
        output = _combine(received[0], offset, workspace, meta, rank, ep_size, mask, abort)
    if mode == "admitted":
        # Capture with full membership and replay only after the replacement is admitted.
        every_rank = torch.cuda.CUDAGraph()
        with torch.cuda.graph(every_rank):
            received_all, offset_all, _ = _dispatch(
                workspace,
                meta,
                rank,
                ep_size,
                _to_every_rank(ep_size),
                [payload],
                _CAPACITY,
                mask,
                abort,
            )
            output_all = _combine(
                received_all[0], offset_all, workspace, meta, rank, ep_size, mask, abort
            )
    torch.cuda.synchronize()
    store.set(f"captured/{rank}", "1")
    store.wait(["proceed"])

    gather = _survivors_gather(store, rank, ep_size)
    state.health.mark_failed(victim)
    if mode != "no-remap":
        state.remap_workspaces(gather, frozenset(range(ep_size)))
        # The remap exchanges the handles and the mapping outcome of every survivor.
        assert memory.mapped
        assert gather.calls == 2
    state.rearm()
    state.install_mask()
    assert mask.cpu().tolist() == _mask_words(range(victim))
    # No survivor writes into a peer's row before that peer has rearmed or while it checks its row.
    gather(None)
    _check_late_writes(store, row, received[0][0], mode)
    gather(None)
    if mode == "no-remap":
        return

    mine = token[token % victim == rank]
    steps = _victim_steps(store)
    for step in range(4):
        payload.copy_(_rows(rank) + step)
        graph.replay()
        torch.cuda.synchronize()
        # Each token comes back from the survivor it was sent to.
        torch.testing.assert_close(output, payload, rtol=0, atol=0)
        # Every survivor's tokens for this rank arrive once each in this rank's fresh pages, in dispatch order.
        for source in range(victim):
            rows = received[0][source][: mine.numel()]
            tokens = _decode_tokens(rows, step)
            assert torch.equal(tokens.sort().values, mine), f"tokens from source {source}"
            assert torch.equal(rows, (_rows(source) + step)[tokens]), f"rows from source {source}"
            assert torch.all(received[1][source][: mine.numel()] == rank)
        # The failed victim's rows keep the zeros of the reset.
        assert torch.count_nonzero(received[0][victim]) == 0
        assert torch.count_nonzero(received[1][victim]) == 0
        assert abort[:2].tolist() == [0, 0]
        # No survivor dispatches the next round into this rank's rows while it reads them.
        gather(None)
    if mode == "writing":
        assert _victim_steps(store) > steps, "the victim stopped writing during the replays"
    if mode == "admitted":
        admission = _admit(store, state, memory, rank, ep_size)

        def replay():
            every_rank.replay()
            return received_all, output_all

        _exchange_with_every_rank(replay, payload, rank, ep_size, admission)


def _to_every_rank(ep_size: int) -> torch.Tensor:
    """Return the experts that send token t of every source to rank t % ep_size."""
    return (torch.arange(_CAPACITY, device="cuda") % ep_size).int()[:, None]


def _admit(
    store: TCPStore,
    state: MoeExecutionState,
    memory: MembershipMnnvlMemory,
    rank: int,
    ep_size: int,
) -> _StoreGather:
    """Admit the victim's replacement on any rank and return the admission's exchange.

    Every rank keeps the rows it maps with the same handles and maps only the others, with no fresh pages.
    Every rank rearms and installs the full membership afterwards.
    """
    gather = _admission_gather(store, rank, ep_size)
    record = MembershipMnnvlMemory.allocated_map[memory.ptr]
    kept = list(record.mem_handles)
    for peer in range(ep_size):
        state.health.mark_active(peer)
    assert state.stale_workspace_rows(gather) == frozenset()
    state.remap_workspaces(gather, frozenset())
    assert all(handle is None or record.mem_handles[r] is handle for r, handle in enumerate(kept))
    assert all(handle is not None for handle in record.mem_handles)
    state.rearm()
    state.install_mask()
    assert state.active_rank_mask.cpu().tolist() == _mask_words(range(ep_size))
    # No rank writes into a peer's row before that peer has rearmed.
    gather(None)
    return gather


def _exchange_with_every_rank(run, payload: torch.Tensor, rank: int, ep_size: int, gather) -> None:
    """Exchange this rank's rows through ``run`` and check that every rank's tokens arrive exactly.

    ``run`` dispatches token t to rank t % ep_size and returns the received payloads and the combined output.
    """
    token = torch.arange(_CAPACITY, device="cuda")
    mine = token[token % ep_size == rank]
    for step in range(4):
        payload.copy_(_rows(rank) + step)
        received, output = run()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, payload, rtol=0, atol=0)
        for source in range(ep_size):
            rows = received[0][source][: mine.numel()]
            tokens = _decode_tokens(rows, step)
            assert torch.equal(tokens.sort().values, mine), f"tokens from source {source}"
            assert torch.equal(rows, (_rows(source) + step)[tokens]), f"rows from source {source}"
            assert torch.all(received[1][source][: mine.numel()] == rank)
        # No rank dispatches the next round into this rank's rows while it reads them.
        gather(None)


def _workspace_layout(ep_size: int) -> torch.Tensor:
    return NVLinkOneSided._make_workspace_layout(
        ep_size, 1, _CAPACITY, _HIDDEN, torch.bfloat16, None, 0, False, False, rank_mask=True
    )


def _replacement(store: TCPStore, rank: int, ep_size: int) -> None:
    """Run the victim's replacement and exchange tokens with every rank once admitted.

    The replacement opens the workspace with only its own row mapped, as at an isolated startup.
    """
    torch.cuda.set_device(rank)
    mapping = Mapping(ep_size, rank, ep_size, tp_size=ep_size, moe_ep_size=ep_size)
    MembershipMnnvlMemory.initialize()
    # An isolated startup uses the deployment's layout with only this rank active.
    MembershipMnnvlMemory.comm = StoreGroup(None, "workspace", rank, range(ep_size), active=[rank])
    MembershipMnnvlMemory.comm_signature = MembershipMnnvlMemory.comm_layout_signature(mapping)
    meta = _workspace_layout(ep_size)
    memory = MembershipMnnvlMemory(
        mapping, int(meta[int(internal.thop.MOE_A2A_WORKSPACE_SIZE_INDEX)])
    )
    workspace = memory.as_torch_strided_tensor(torch.uint8)
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, rank, ep_size)
    handles = MembershipMnnvlMemory.allocated_map[memory.ptr].mem_handles
    assert [handle is not None for handle in handles] == [r == rank for r in range(ep_size)]
    state = _bound_state(memory, workspace, meta, rank, ep_size)
    admission = _admit(store, state, memory, rank, ep_size)
    experts = _to_every_rank(ep_size)
    payload = torch.empty((_CAPACITY, _HIDDEN), dtype=torch.bfloat16, device="cuda")
    mask, abort = state.active_rank_mask, state.abort

    def dispatch_and_combine():
        received, offset, _ = _dispatch(
            workspace, meta, rank, ep_size, experts, [payload], _CAPACITY, mask, abort
        )
        return received, _combine(received[0], offset, workspace, meta, rank, ep_size, mask, abort)

    _exchange_with_every_rank(dispatch_and_combine, payload, rank, ep_size, admission)


def _worker(rank: int, port: int, mode: str, ep_size: int) -> None:
    store = TCPStore("127.0.0.1", port, is_master=False, timeout=_TIMEOUT)
    # Every rank of a deployment that survives a failed rank runs with Wide-EP fault tolerance on.
    set_wide_ep_fault_tolerance_enabled(True)
    if mode == "replacement":
        _replacement(store, rank, ep_size)
        return
    torch.cuda.set_device(rank)
    mapping = Mapping(ep_size, rank, ep_size, tp_size=ep_size, moe_ep_size=ep_size)
    MembershipMnnvlMemory.initialize()
    comm = _StoreComm(store, rank, ep_size)
    MembershipMnnvlMemory.comm = comm
    MembershipMnnvlMemory.comm_signature = MembershipMnnvlMemory.comm_layout_signature(mapping)
    meta = _workspace_layout(ep_size)
    memory = MembershipMnnvlMemory(
        mapping, int(meta[int(internal.thop.MOE_A2A_WORKSPACE_SIZE_INDEX)])
    )
    workspace = memory.as_torch_strided_tensor(torch.uint8)
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, rank, ep_size)
    # No peer may write into a row before its owner has initialized it.
    comm.barrier()
    if rank == ep_size - 1:
        _victim(store, workspace, meta, ep_size)
    elif ep_size == 2:
        _survivor(store, memory, workspace, meta, mode)
    else:
        _survivor_with_peers(store, memory, workspace, meta, rank, ep_size, mode)


def _run(ep_size: int, mode: str, kill_delay_s: float | None) -> None:
    """Run ``ep_size`` workers with the last one as the victim, and kill it after ``kill_delay_s`` if set."""
    if torch.cuda.device_count() < ep_size:
        pytest.skip(f"Requires {ep_size} local GPUs")
    MembershipMnnvlMemory.initialize()
    if not MembershipMnnvlMemory.supports_mnnvl():
        pytest.skip("Requires an NVLink fabric supported by MNNVL")
    store = TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False, timeout=_TIMEOUT)
    # One worker is killed, so keep the workers off any enclosing launcher's process management.
    launcher_prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(launcher_prefixes)}
    args = [str(store.port), mode, str(ep_size)]
    workers = [
        subprocess.Popen([sys.executable, __file__, str(rank), *args], env=env)
        for rank in range(ep_size)
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
        if mode == "admitted":
            # The victim's replacement, on its GPU.
            replacement = [str(ep_size - 1), str(store.port), "replacement", str(ep_size)]
            workers.append(subprocess.Popen([sys.executable, __file__, *replacement], env=env))
            survivors.append(workers[-1])
        for survivor in survivors:
            assert survivor.wait(timeout=_TIMEOUT.total_seconds()) == 0
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait()


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
def test_late_writes_land_on_orphaned_pages(mode: str, kill_delay_s: float | None) -> None:
    """After the remap, the survivor's results stay exact and none of its bytes change.

    The victim is killed mid-dispatch (``killed``) or keeps writing (``writing``). In the ``no-remap`` control,
    the victim's writes must reach the survivor's active rows.
    """
    _run(2, mode, kill_delay_s)


@pytest.mark.parametrize(
    "mode, kill_delay_s", [("killed", 0.003), ("writing", None), ("no-remap", None)]
)
def test_survivors_remap_and_exchange_exactly(mode: str, kill_delay_s: float | None) -> None:
    """Three survivors remap together and exchange exact results while the victim writes.

    Each survivor maps the others' fresh pages at the same addresses. ``no-remap`` is the negative control.
    """
    _run(4, mode, kill_delay_s)


def test_replacement_admitted_to_survivor_pages() -> None:
    """A replacement joins the survivors' remapped workspace and exchanges exact results with all ranks.

    Each survivor keeps its pages and maps only the replacement's row. The survivors replay a graph captured
    before the failure.
    """
    _run(4, "admitted", 0.003)


if __name__ == "__main__":
    _worker(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))

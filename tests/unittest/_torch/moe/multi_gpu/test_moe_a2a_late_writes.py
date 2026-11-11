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
differs from the survivors', so its rows of the packed layout (source s's rows at [s * T, (s + 1) * T) for
runtime token count T) fall on rows of source 0, a survivor. The survivors remap the workspace with fresh
pages at the same addresses shared only among themselves (``MoeExecutionState.remap_workspaces``, its
handle and outcome exchanges over a TCPStore), then rearm and install a membership without the victim.
Their CUDA graphs captured before the failure then return exact results while the victim, killed or still
writing, reaches none of their bytes.

With two ranks, the survivor first aborts the combine that waits on the victim. With four, three survivors
import each other's fresh pages and exchange tokens among themselves after the remap; ``admitted`` then
admits the victim's replacement, which keeps the survivors' pages (``MoeExecutionState.admit_to_workspaces``),
and every rank exchanges tokens with every other.

``no-remap`` is the negative control: without the remap the victim's writes reach the survivors' live
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
from tensorrt_llm._torch.distributed.mnnvl_live_group import LiveGroupComm, isolated_gather
from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
from tensorrt_llm._torch.mnnvl_alltoall_workspace import _MnnvlAlltoAllWorkspaceLifecycle
from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import MoeExecutionState
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm.bindings import internal
from tensorrt_llm.mapping import Mapping

_CAPACITY, _HIDDEN = 1024, 7168
_GRACE_S = 2.0
# The negative control polls the live rows this often until the victim's writes show, at most this long.
_POLL_S, _CONTROL_DEADLINE_S = 0.05, 10.0
_TIMEOUT = timedelta(seconds=300)


class _StoreComm:
    """The communicator subset MnnvlMemory uses, over a TCPStore, so that a peer can be killed."""

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
    """A membership change's exchange among the EP ranks in ``live`` over a TCPStore, under keys of ``name``: one
    entry per EP rank, None for a rank not in ``live``."""

    def __init__(self, store: TCPStore, rank: int, ep_size: int, name: str, live: range) -> None:
        self.store, self.rank, self.ep_size, self.calls = store, rank, ep_size, 0
        self.name, self.live = name, live

    def __call__(self, value: Any) -> list[Any]:
        self.calls += 1
        self.store.set(f"{self.name}/{self.calls}/{self.rank}", pickle.dumps(value))
        return [
            pickle.loads(self.store.get(f"{self.name}/{self.calls}/{rank}"))
            if rank in self.live
            else None
            for rank in range(self.ep_size)
        ]


def _survivors_gather(store: TCPStore, rank: int, ep_size: int) -> _StoreGather:
    """The remap's exchange among the survivors, every rank but the last, the victim."""
    return _StoreGather(store, rank, ep_size, "survivors", range(ep_size - 1))


def _admission_gather(store: TCPStore, rank: int, ep_size: int) -> _StoreGather:
    """The admission's exchange among the survivors and the victim's replacement."""
    return _StoreGather(store, rank, ep_size, "admission", range(ep_size))


def _offset(meta: torch.Tensor, name: str) -> int:
    """The offset of region ``name`` (``MOE_A2A_<name>_OFFSET_INDEX``) in a workspace row of layout ``meta``."""
    return int(meta[int(getattr(internal.thop, f"MOE_A2A_{name}_OFFSET_INDEX"))])


def _rows(source: int) -> torch.Tensor:
    """[_CAPACITY, _HIDDEN] bf16 rows, each distinct: columns 0-2 encode the token and the source, exactly in
    bf16 (integers up to 256), so a misplaced or mixed-up row reads as wrong values."""
    token = torch.arange(_CAPACITY, device="cuda")[:, None]
    values = (token + torch.arange(_HIDDEN, device="cuda")) % 128
    values[:, 0:1] = token % 128
    values[:, 1:2] = token // 128
    values[:, 2] = source
    return values.bfloat16()


def _decode_tokens(rows: torch.Tensor, step: int) -> torch.Tensor:
    """The tokens of received ``_rows(source) + step`` rows, read from their columns 0-1."""
    return ((rows[:, 0].float() - step) + 128 * (rows[:, 1].float() - step)).long()


def _controls(mask: torch.Tensor, abort: torch.Tensor) -> dict:
    return {"enable_rank_mask": True, "active_rank_mask": mask, "abort_state": abort}


def _dispatch(workspace, meta, rank, ep_size, experts, payloads, tokens, mask, abort):
    """Dispatch [*payloads, experts] of ``tokens`` tokens per rank; expert e is on rank e."""
    args = (experts, [*payloads, experts], workspace, meta, tokens, rank, ep_size, 1, ep_size)
    return torch.ops.trtllm.moe_a2a_dispatch(*args, **_controls(mask, abort))


def _combine(received, offset, workspace, meta, rank, ep_size, mask, abort):
    args = (received, _CAPACITY, workspace, meta, _CAPACITY, rank, ep_size, 1)
    return torch.ops.trtllm.moe_a2a_combine(*args, offset, False, **_controls(mask, abort))


def _bound_state(
    memory: MnnvlMemory, workspace: torch.Tensor, meta: torch.Tensor, rank: int, ep_size: int
) -> MoeExecutionState:
    """A survivor's MoeExecutionState, bound by an NVLinkOneSided layer on ``workspace`` as a model binds it.

    The layer carries only the fields binding and the workspace reset read, with the workspace's real lifecycle;
    no MoE layer is built, and no watchdog runs.
    """
    NVLinkOneSided._init_constants()
    health = EPGroupHealth(ep_size)
    state = MoeExecutionState(health, workspace.device)
    comm = NVLinkOneSided.__new__(NVLinkOneSided)
    comm.can_use_cft_counted_writes = False
    comm.ep_group_health = health
    comm._rank_mask_enabled = True
    comm._execution_state = None
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
    # A pending abort request lets every dispatch push all its rows and then return instead of
    # waiting for the silent survivors, so new NVLink writes are always in flight. A dispatch that
    # returns early leaves its round incomplete, after which this rank's collectives route to itself
    # only until its workspace row is reset: the row is reset before every dispatch, as a survivor's
    # rearm resets its own, so that every dispatch writes into the survivors' rows.
    mask = torch.tensor([(1 << ep_size) - 1, 0, 0, 0], dtype=torch.uint64, device="cuda")
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


def _check_late_writes(store: TCPStore, row: torch.Tensor, live: torch.Tensor, mode: str) -> None:
    """With the remap, no byte of this survivor's ``row`` changes for _GRACE_S while the victim, unless killed,
    keeps writing. Without it (``no-remap``), the victim's writes reach ``live``, source 0's rows of ``row``."""
    torch.cuda.synchronize()
    before = row.clone()
    steps = _victim_steps(store)
    # The victim alternates two patterns, so a single later look can find a byte it rewrote unchanged: look
    # every _POLL_S.
    if mode == "no-remap":
        start = live.data_ptr() - row.data_ptr()
        end = start + live.numel() * live.element_size()
        deadline = time.monotonic() + _CONTROL_DEADLINE_S
        while not bool((row[start:end] != before[start:end]).any()):
            assert time.monotonic() < deadline, (
                "the victim's writes missed the survivor's live rows without the remap"
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
    store: TCPStore, memory: MnnvlMemory, workspace: torch.Tensor, meta: torch.Tensor, mode: str
) -> None:
    """Rank 0 of two: abort the combine waiting on the victim, remap alone, then replay its graph."""
    rank, ep_size, victim = 0, 2, 1
    row = workspace[rank]
    state = _bound_state(memory, workspace, meta, rank, ep_size)
    mask, abort = state.active_rank_mask, state.abort
    experts = torch.zeros((_CAPACITY, 1), dtype=torch.int32, device="cuda")
    payload = torch.empty((_CAPACITY, _HIDDEN), dtype=torch.bfloat16, device="cuda")

    # Captured with full membership; replayed only after the victim failed.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        received, offset, _ = _dispatch(
            workspace, meta, rank, ep_size, experts, [payload], _CAPACITY, mask, abort
        )
        output = _combine(received[0], offset, workspace, meta, rank, ep_size, mask, abort)
    # Each payload's receive buffer is [ep_size, runtime tokens, ...] from DISPATCH_PAYLOAD: source s's rows
    # are [s, :].
    assert offset == _offset(meta, "COMBINE_INPUT")
    assert [tuple(r.shape) for r in received] == [
        (ep_size, _CAPACITY, _HIDDEN),
        (ep_size, _CAPACITY, 1),
    ]
    assert received[0].data_ptr() == row.data_ptr() + _offset(meta, "DISPATCH_PAYLOAD")

    torch.cuda.synchronize()
    store.set(f"captured/{rank}", "1")
    store.wait(["proceed"])

    # The victim's first dispatch published its round flag, with its complete rows, before its pending abort
    # left its round incomplete, so the survivor's next dispatch completes on that flag. The combine after it
    # waits for the failed victim, which publishes no further flag to it, until the survivor aborts it.
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

    # Recovery: fresh pages at the same addresses among the survivors, then the new membership.
    state.health.mark_failed(victim)
    if mode != "no-remap":
        state.remap_workspaces(_survivors_gather(store, rank, ep_size))
        assert memory.mapped
    state.rearm()
    state.install_mask()
    assert mask.cpu().tolist() == [1 << rank, 0, 0, 0]

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
        # The survivor's rows hold each of its tokens once, in the order the dispatch assigns them, not by
        # token.
        rows = received[0][rank]
        tokens = _decode_tokens(rows, step)
        assert torch.equal(tokens.sort().values, torch.arange(_CAPACITY, device="cuda"))
        assert torch.equal(rows, payload[tokens]), "a received row differs from its token's row"
        assert torch.all(received[1][rank] == 0)
        # Nothing writes the failed victim's rows: they keep the reset's zeros.
        assert torch.count_nonzero(received[0][victim]) == 0
        assert torch.count_nonzero(received[1][victim]) == 0
        assert abort[:2].tolist() == [0, 0]
    if mode == "writing":
        assert _victim_steps(store) > steps, "the victim stopped writing during the replays"


def _survivor_with_peers(
    store: TCPStore,
    memory: MnnvlMemory,
    workspace: torch.Tensor,
    meta: torch.Tensor,
    rank: int,
    ep_size: int,
    mode: str,
) -> None:
    """One of several survivors: remap with the others, then replay a graph whose tokens go to every survivor;
    ``admitted``: then admit the victim's replacement and replay a graph whose tokens go to every rank."""
    victim = ep_size - 1
    row = workspace[rank]
    state = _bound_state(memory, workspace, meta, rank, ep_size)
    mask, abort = state.active_rank_mask, state.abort
    token = torch.arange(_CAPACITY, device="cuda")
    # Every source sends token t to survivor t % victim.
    experts = (token % victim).int()[:, None]
    payload = torch.empty((_CAPACITY, _HIDDEN), dtype=torch.bfloat16, device="cuda")

    # Captured with full membership; replayed only after the victim failed.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        received, offset, _ = _dispatch(
            workspace, meta, rank, ep_size, experts, [payload], _CAPACITY, mask, abort
        )
        output = _combine(received[0], offset, workspace, meta, rank, ep_size, mask, abort)
    if mode == "admitted":
        # Captured with full membership; replayed only after the victim's replacement was admitted.
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
        state.remap_workspaces(gather)
        # The handle exchange and the exchange of every survivor's mapping outcome; each survivor resets its row
        # when it rearms.
        assert memory.mapped and gather.calls == 2
    state.rearm()
    state.install_mask()
    assert mask.cpu().tolist() == [(1 << victim) - 1, 0, 0, 0]
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
        # Every survivor's tokens for this rank, written into this rank's fresh pages through its peers'
        # imports of them, each once, in the order the dispatch assigns them.
        for source in range(victim):
            rows = received[0][source][: mine.numel()]
            tokens = _decode_tokens(rows, step)
            assert torch.equal(tokens.sort().values, mine), f"tokens from source {source}"
            assert torch.equal(rows, (_rows(source) + step)[tokens]), f"rows from source {source}"
            assert torch.all(received[1][source][: mine.numel()] == rank)
        # Nothing writes the failed victim's rows: they keep the reset's zeros.
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
    """The experts of _CAPACITY tokens that every source sends token t to rank t % ep_size with."""
    return (torch.arange(_CAPACITY, device="cuda") % ep_size).int()[:, None]


def _admit(
    store: TCPStore, state: MoeExecutionState, memory: MnnvlMemory, rank: int, ep_size: int
) -> _StoreGather:
    """Admit the victim's replacement, on any rank: every rank keeps the rows it maps with the same handles and
    maps only the others, with no fresh pages; then every rank rearms and installs the full membership. Returns
    the admission's exchange."""
    gather = _admission_gather(store, rank, ep_size)
    record = MnnvlMemory.allocated_map[memory.ptr]
    kept = list(record.mem_handles)
    for peer in range(ep_size):
        state.health.mark_active(peer)
    assert not state.workspaces_need_fresh_pages(gather)
    state.admit_to_workspaces(gather)
    assert all(handle is None or record.mem_handles[r] is handle for r, handle in enumerate(kept))
    assert memory.holders == set(range(ep_size))
    state.rearm()
    state.install_mask()
    assert state.active_rank_mask.cpu().tolist() == [(1 << ep_size) - 1, 0, 0, 0]
    # No rank writes into a peer's row before that peer has rearmed.
    gather(None)
    return gather


def _exchange_with_every_rank(run, payload: torch.Tensor, rank: int, ep_size: int, gather) -> None:
    """Dispatch and combine (``run`` returns the received payloads and the combine's output) this rank's rows,
    each token t to rank t % ep_size, and check that every rank's tokens for this rank arrive exactly."""
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
    """The victim's replacement: open the workspace with only its own row mapped, as at its isolated startup, be
    admitted, then dispatch to and combine from every rank."""
    torch.cuda.set_device(rank)
    mapping = Mapping(ep_size, rank, ep_size, tp_size=ep_size, moe_ep_size=ep_size)
    MnnvlMemory.initialize()
    # The class's group at an isolated startup: the deployment's layout with only this rank live.
    MnnvlMemory.comm = LiveGroupComm(tuple(range(ep_size)), rank, isolated_gather(rank, ep_size))
    MnnvlMemory.comm_signature = MnnvlMemory.comm_layout_signature(mapping)
    meta = _workspace_layout(ep_size)
    memory = MnnvlMemory(mapping, int(meta[int(internal.thop.MOE_A2A_WORKSPACE_SIZE_INDEX)]))
    workspace = memory.as_torch_strided_tensor(torch.uint8)
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, rank, ep_size)
    assert memory.holders == {rank}
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
    if mode == "replacement":
        _replacement(store, rank, ep_size)
        return
    torch.cuda.set_device(rank)
    mapping = Mapping(ep_size, rank, ep_size, tp_size=ep_size, moe_ep_size=ep_size)
    MnnvlMemory.initialize()
    comm = _StoreComm(store, rank, ep_size)
    MnnvlMemory.comm = comm
    MnnvlMemory.comm_signature = MnnvlMemory.comm_layout_signature(mapping)
    meta = _workspace_layout(ep_size)
    memory = MnnvlMemory(mapping, int(meta[int(internal.thop.MOE_A2A_WORKSPACE_SIZE_INDEX)]))
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
    """Run ``ep_size`` workers, the last one the victim; kill it after ``kill_delay_s`` unless None."""
    if torch.cuda.device_count() < ep_size:
        pytest.skip(f"Requires {ep_size} local GPUs")
    MnnvlMemory.initialize()
    if not MnnvlMemory.supports_mnnvl():
        pytest.skip("Requires an NVLink fabric supported by MNNVL")
    store = TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False, timeout=_TIMEOUT)
    # Keep the workers off any enclosing launcher's process-management interface: one is killed.
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
def test_failed_peer_late_writes_land_only_on_orphaned_pages(
    mode: str, kill_delay_s: float | None
) -> None:
    """After the remap the survivor's results stay exact and none of its bytes change, whether the victim was
    killed mid-dispatch (``killed``) or keeps writing to the end (``writing``); ``no-remap`` is the negative
    control, whose victim writes must reach the survivor's live rows."""
    _run(2, mode, kill_delay_s)


@pytest.mark.parametrize(
    "mode, kill_delay_s", [("killed", 0.003), ("writing", None), ("no-remap", None)]
)
def test_survivors_remap_together_and_exchange_exact_results(
    mode: str, kill_delay_s: float | None
) -> None:
    """Three survivors and a victim: after the survivors remap together, each maps the others' fresh pages at
    the same addresses, none of their bytes change while the victim writes, and a dispatch and combine among
    them is exact; ``no-remap`` is the negative control."""
    _run(4, mode, kill_delay_s)


def test_a_replacement_is_admitted_to_the_survivors_pages_and_exchanges_exact_results() -> None:
    """After the survivors remapped without the killed victim, its replacement opens the workspace with only its
    own row mapped and is admitted: each survivor keeps its pages and imports and maps only the replacement's
    row, the replacement maps the survivors' rows, and a dispatch and combine among all four ranks (the
    survivors' from a graph captured before the failure) is exact."""
    _run(4, "admitted", 0.003)


if __name__ == "__main__":
    _worker(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))

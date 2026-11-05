# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recovery of the MNNVL MoE All-to-All among survivors, on four GPUs sharing a real fabric workspace.

``test_remap_orphans_late_writer``: a failed rank whose kernels keep writing the survivors' rows reaches only the
released pages; MoeExecutionState.remap_workspaces backs the survivors' rows with fresh pages at the same
addresses, and their masked round trips never touch the failed rank's unmapped row. The survivors run
RankFailureRecovery._remap_workspaces, which takes room from a stand-in KV cache, then install_recovery_placement's
sequence; RankFailureRecovery._return_remap_room returns the room once the victim's death frees its pages.

``test_second_failure_orphans_importer_of_remapped_pages``: a rank that imported a remap's fresh pages and fails
next keeps writing them; the second remap, without both failed ranks, keeps its writes off the survivors' live
rows, and their masked round trips stay exact.

``test_aborted_round_with_runahead_rank``: on packed rows, a rank whose round an abort left incomplete runs
rounds ahead and makes its peers' round abort on request, neither trapping, hanging nor completing, and a rearm
restores exact round trips. Its ``graph-mask`` case installs a mask between two replays of a captured graph; the
next replay must read it.

Workers are separate processes over a TCPStore, as in test_moe_a2a_late_writes.py, since one may be SIGKILLed.
Each layer is a fake NVLinkOneSided on the real workspace lifecycle, with its watchdog, bound to a
MoeExecutionState; dispatch and combine are the fence ops.
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
from tensorrt_llm._torch.alltoall_watchdog import DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S
from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
from tensorrt_llm._torch.mnnvl_alltoall_workspace import _MnnvlAlltoAllWorkspaceLifecycle
from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import MoeExecutionState
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm._torch.pyexecutor import rank_failure_recovery
from tensorrt_llm._torch.pyexecutor.rank_failure_recovery import RankFailureRecovery
from tensorrt_llm.bindings import internal
from tensorrt_llm.mapping import Mapping

_EP_SIZE = 4
_EVERYONE = list(range(_EP_SIZE))
_CAPACITY, _HIDDEN = 256, 7168
# The failed rank of the remap test, the rank whose incomplete round runs ahead, and the rank that fails between
# graph replays.
_VICTIM, _RUNAHEAD, _FAILED = 2, 1, 3
# The two ranks that fail in turn in the re-plan test: the second imports the first remap's pages.
_FIRST_FAILED, _SECOND_FAILED = 3, 2
_AHEAD_ROUNDS = 4
_PATTERN = 0xA5
_WRITE_S = 1.0
# How often the re-plan test looks at the survivors' rows.
_POLL_S = 0.05
# A kernel wait that is never released traps long before the test times out.
_KERNEL_TIMEOUT_S = 60
_TIMEOUT = timedelta(seconds=300)


class _Survivors:
    """The survivor collectives of Distributed after shrink(), over the TCPStore."""

    def __init__(self, comm: StoreComm, live: list[int]) -> None:
        self.comm, self.live = comm, live

    def allgather_among_survivors(self, obj: Any) -> list[Any]:
        """One entry per logical rank, None for a failed one, as Distributed.allgather_among_survivors."""
        return self.comm.exchange(obj, self.live)

    def barrier_among_survivors(self) -> None:
        self.comm.exchange(None, self.live)


def _tokens(seed: int, source: int, live: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Source's [_CAPACITY, _HIDDEN] bf16 rows and [_CAPACITY, 1] target ranks, on the host; every rank rebuilds
    every source's. Columns 0-3 encode the token, the source and the seed exactly in bf16 (integers below 128,
    so doubled ones stay exact); each token goes to one live rank, the seed shifting which."""
    token = torch.arange(_CAPACITY)[:, None]
    x = (token + torch.arange(_HIDDEN) + seed) % 128
    x[:, 0:1] = token % 128
    x[:, 1:2] = token // 128
    x[:, 2] = source
    x[:, 3] = seed % 128
    targets = torch.tensor(live, dtype=torch.int32)[(token + source + seed) % len(live)]
    return x.bfloat16(), targets


def _check_round(
    rank: int, seed: int, live: list[int], received: list[torch.Tensor], combined: torch.Tensor
) -> None:
    """Check by value the rows each live source sent here, packed at the front of its rows in arrival order,
    and this rank's combined tokens: the experts double each row, top_k is 1."""
    rows, experts = (t.cpu() for t in received)
    for source in live:
        x, targets = _tokens(seed, source, live)
        sent = targets[:, 0] == rank
        count = int(sent.sum())
        got = rows[source, :count]
        order = (got[:, 0].int() + 128 * got[:, 1].int()).argsort()
        where = f"rank {rank}, seed {seed}, source {source}"
        assert torch.equal(got[order], x[sent]), (
            f"{where}: rows wrong, seeds {got[:, 3].unique().tolist()}"
        )
        assert bool((experts[source, :count] == rank).all()), f"{where}: expert ids wrong"
    x, _ = _tokens(seed, rank, live)
    assert torch.equal(combined.cpu(), x * 2), f"rank {rank}, seed {seed}: combined wrong"


def _await(
    event: torch.cuda.Event, state: MoeExecutionState, what: str, seconds: float = 10
) -> None:
    """Wait for ``event``; past ``seconds``, release the kernels so the failure is reported instead of hung."""
    deadline = time.monotonic() + seconds
    while not event.query():
        if time.monotonic() > deadline:
            state.request_abort()
            raise AssertionError(what)
        time.sleep(0.01)


def _reset_row(ctx: "_Rank") -> torch.Tensor:
    """A workspace row as moe_a2a_initialize leaves it: zeros, with both parities' receive counters at -1."""
    reset = torch.zeros(ctx.size, dtype=torch.uint8)
    recv = workspace_offset(ctx.meta, "RECV_COUNTERS")
    reset[recv : recv + 2 * _EP_SIZE * 4] = 0xFF
    return reset


def _held_bytes() -> int:
    """Device bytes in use outside PyTorch's caching allocator, as MNNVL pages are."""
    free, total = torch.cuda.mem_get_info()
    return total - free - torch.cuda.memory_reserved()


class _Rank:
    """One EP rank: its real MNNVL workspace, bound to a MoeExecutionState as a model binds its layers."""

    def __init__(self, store: TCPStore, rank: int) -> None:
        self.store, self.rank = store, rank
        torch.cuda.set_device(rank)
        mapping = Mapping(_EP_SIZE, rank, _EP_SIZE, tp_size=_EP_SIZE, moe_ep_size=_EP_SIZE)
        MnnvlMemory.initialize()
        self.comm = MnnvlMemory.comm = StoreComm(store, rank, _EP_SIZE)
        MnnvlMemory.comm_signature = MnnvlMemory.comm_layout_signature(mapping)
        self.meta = NVLinkOneSided._make_workspace_layout(
            _EP_SIZE, 1, _CAPACITY, _HIDDEN, torch.bfloat16, None, 0, False, False, rank_mask=True
        )
        self.size = int(self.meta[int(internal.thop.MOE_A2A_WORKSPACE_SIZE_INDEX)])
        self.memory = MnnvlMemory(mapping, self.size)
        self.workspace = self.memory.as_torch_strided_tensor(torch.uint8)
        torch.ops.trtllm.moe_a2a_initialize(
            self.workspace, self.meta, rank, _EP_SIZE, barrier=False
        )
        torch.ops.trtllm.moe_a2a_set_timeout(_KERNEL_TIMEOUT_S)
        self.state = self._bind()
        # No peer may write into a row before its owner has initialized it.
        self.comm.barrier()

    def _bind(self) -> MoeExecutionState:
        """Bind a fake NVLinkOneSided layer on the workspace; no MoE layer is built.

        The layer carries only the fields binding and the workspace reset read. It registers with the real
        workspace lifecycle and its watchdog, as NVLinkOneSided.__init__ does under ep_group_health, so the
        release stops the watchdog and the remap restarts it. The ops below are called directly and queue no
        phase, so the watchdog never times out.
        """
        health = EPGroupHealth(_EP_SIZE)
        state = MoeExecutionState(health, self.workspace.device)
        layer = self.layer = NVLinkOneSided.__new__(NVLinkOneSided)
        NVLinkOneSided._init_constants()
        layer.can_use_cft_counted_writes = False
        layer.ep_group_health = health
        layer._rank_mask_enabled = True
        layer._execution_state = None
        layer._dispatch_state = {"phase": "idle"}
        layer.mnnvl_mem, layer.workspace = self.memory, self.workspace
        layer.ep_rank, layer.ep_size = self.rank, _EP_SIZE
        self.lifecycle = layer._workspace_lifecycle = _MnnvlAlltoAllWorkspaceLifecycle(
            workspace_state={},
            memory=self.memory,
            workspace=self.workspace,
            metainfo=self.meta,
            metainfo_index={
                name: getattr(NVLinkOneSided, name)
                for name in (
                    "FLAG_VAL_OFFSET_INDEX",
                    "DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX",
                    "COMBINE_COMPLETION_FLAGS_OFFSET_INDEX",
                )
            },
            ep_rank=self.rank,
            ep_size=_EP_SIZE,
            health=health,
        )
        self.lifecycle.register(
            layer,
            watchdog_timeout_s=_TIMEOUT.total_seconds(),
            watchdog_poll_interval_s=DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S,
            watchdog_on_timeout=None,
        )
        layer.bind_execution_state(state)
        return state

    def watchdog_running(self) -> bool:
        return self.lifecycle.watchdog_for(self.layer) is not None

    def launch(
        self, x: torch.Tensor, targets: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Dispatch ``x`` and its expert ids to ``targets`` (expert e on rank e), double every received row as the
        experts, and combine. Nothing waits for the kernels."""
        controls = {
            "enable_rank_mask": True,
            "active_rank_mask": self.state.active_rank_mask,
            "abort_state": self.state.abort,
        }
        layout = (self.workspace, self.meta, _CAPACITY, self.rank, _EP_SIZE, 1)
        received, offset, _ = torch.ops.trtllm.moe_a2a_dispatch(
            targets, [x, targets], *layout, _EP_SIZE, **controls
        )
        # The receive rows are views of this rank's workspace row, which a peer's next dispatch overwrites once
        # this combine completes; the experts read them before it, and so do the checks, from these copies.
        received = [t.clone() for t in received]
        combined = torch.ops.trtllm.moe_a2a_combine(
            received[0] * 2, _CAPACITY, *layout, offset, False, **controls
        )
        return received, combined

    def round_trip(self, seed: int, live: list[int]) -> None:
        x, targets = _tokens(seed, self.rank, live)
        received, combined = self.launch(x.cuda(), targets.cuda())
        torch.cuda.synchronize()
        _check_round(self.rank, seed, live, received, combined)

    def round_flags(self) -> list[int]:
        """This rank's round counter and incomplete-round word."""
        row = self.workspace[self.rank]
        return [
            int(row[offset : offset + 4].view(torch.int32).item())
            for offset in (
                workspace_offset(self.meta, "FLAG_VAL"),
                workspace_offset(self.meta, "INCOMPLETE_ROUND"),
            )
        ]


# ---- test_remap_orphans_late_writer ----


def _late_writer(ctx: _Rank, targets: list[int]) -> None:
    """Keep filling the rows of ``targets`` through this rank's mappings until told to stop, or killed."""
    rows = [ctx.workspace[r] for r in targets]
    for step in itertools.count():
        for row in rows:
            row.fill_(_PATTERN)
        if step % 8 == 0:
            torch.cuda.synchronize()
            ctx.store.set("progress", str(step))
            if ctx.store.check(["stop"]):
                # Every write was to pages the survivors released while they still lived.
                ctx.store.set("stopped", "1")
                return


def _survivor(ctx: _Rank, mode: str) -> None:
    live = [r for r in _EVERYONE if r != _VICTIM]
    state, row = ctx.state, ctx.workspace[ctx.rank]
    ctx.store.wait(["progress"])  # the victim's writes are in flight

    # RankFailureRecovery._remap_workspaces among the survivors, then install_recovery_placement's mask commit.
    survivors = _Survivors(ctx.comm, live)
    live_mask = [sum(1 << r for r in live), 0, 0, 0]
    held = _held_bytes()
    need = MnnvlMemory.allocated_map[ctx.memory.ptr].aligned_size
    recovery, quota = _remap_recovery(ctx, survivors, need, live_mask)
    recovery._remap_workspaces(frozenset({_VICTIM}), time.monotonic() + 60)
    assert ctx.watchdog_running(), "the remap did not restart the watchdog"
    # The remap took what the device lacked from the KV cache.
    lent = recovery._remap_lent
    assert quota == [-lent] and abs(lent - need // 2) <= need // 4
    state.health.mark_failed(_VICTIM)
    state.rearm()
    state.install_mask()
    assert state.active_rank_mask.cpu().tolist() == live_mask
    handles = MnnvlMemory.allocated_map[ctx.memory.ptr].mem_handles
    assert [handle is None for handle in handles] == [r == _VICTIM for r in _EVERYONE], (
        "only the victim's row may stay unmapped"
    )
    # The victim still maps the released pages, so they stay allocated beside the fresh ones.
    slack = need // 4
    remapped = _held_bytes()
    assert remapped - held >= need - slack, f"held {held} -> {remapped} bytes, remapped {need}"
    progress = int(ctx.store.get("progress"))
    ctx.store.set(f"remapped/{ctx.rank}", "1")

    # The launcher lets the victim write for _WRITE_S (then kills it, in the kill case).
    ctx.store.wait(["written"])
    assert int(ctx.store.get("progress")) > progress, "the victim stopped writing before the check"
    reset = _reset_row(ctx)
    host_row = row.cpu()
    contaminated = (host_row != reset).nonzero().flatten()[:16]
    assert contaminated.numel() == 0, (
        f"bytes changed after the reset at {contaminated.tolist()}: {host_row[contaminated].tolist()}"
    )
    # A peer's round trip writes this row: none starts before every survivor checked its own.
    survivors.barrier_among_survivors()

    # Masked round trips touch no byte of the unmapped row: an access would fault this context.
    for seed in (1, 2):
        ctx.round_trip(seed, live)
    assert state.abort.tolist() == [0, 0]

    if mode == "kill":
        # The victim's death drops the last mappings of the released pages. Recovery returns the KV cache share
        # it lent to the remap only once cudaMemGetInfo shows them free (RankFailureRecovery._return_remap_room).
        deadline = time.monotonic() + 60
        while _held_bytes() > remapped - need + slack:
            assert time.monotonic() < deadline, (
                f"cudaMemGetInfo still counts the {need} released bytes 60 s after the victim's SIGKILL: "
                "the driver keeps a killed importer's pages, so recovery would keep the KV cache share it lent"
            )
            time.sleep(0.1)
        recovery._return_remap_room()
        assert quota[0] == -lent and quota[1:] and sum(quota[1:]) >= lent - slack
        assert recovery._remap_lent <= slack
    else:
        assert _held_bytes() - held >= need - slack, "pages freed while the victim maps them"
        # The victim still maps the released pages: nothing is returned yet.
        recovery._return_remap_room()
        assert quota == [-lent] and recovery._remap_lent == lent
        ctx.store.set(f"checked/{ctx.rank}", "1")
        # Exit only after the victim's last write, so every write it made found its exporter alive.
        ctx.store.wait(["stopped"])
    state.close()


def _remap_recovery(
    ctx: _Rank, survivors: _Survivors, need: int, live_mask: list[int]
) -> tuple[RankFailureRecovery, list[int]]:
    """A RankFailureRecovery that runs only _remap_workspaces and _return_remap_room for this survivor; returns it
    and the KV cache quota changes it makes.

    The device reports, outside PyTorch's caching allocator, ``need // 2`` bytes less than the remap needs
    beyond the margin, so the remap takes them from the KV cache, as on a device the KV cache fills.
    """
    recovery = RankFailureRecovery.__new__(RankFailureRecovery)
    quota, barriers = [], []

    def resize_kv_cache_quota(delta: int) -> bool:
        quota.append(delta)
        return True

    def barrier_among_survivors() -> None:
        barriers.append(None)
        if len(barriers) == 1:
            # The first barrier follows the release: the failed rank left the device mask with its row,
            # before the committed membership does, and the watchdog stopped reading the unmapped rows.
            assert ctx.state.active_rank_mask.cpu().tolist() == live_mask
            assert not ctx.watchdog_running(), (
                "the release left the watchdog reading the unmapped rows"
            )
        survivors.barrier_among_survivors()

    recovery._executor = SimpleNamespace(
        dist=SimpleNamespace(
            rank=ctx.rank,
            barrier_among_survivors=barrier_among_survivors,
            allgather_among_survivors=survivors.allgather_among_survivors,
        ),
        model_engine=SimpleNamespace(moe_execution_state=ctx.state),
        resize_kv_cache_quota=resize_kv_cache_quota,
    )
    recovery._row_reader = None
    recovery._deferred_logs = None
    recovery._remap_lent = 0

    def untracked_free() -> int:
        return torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved()

    reported = need + rank_failure_recovery._UNBACKED_MARGIN_BYTES - need // 2
    hidden = untracked_free() - reported
    rank_failure_recovery._free_device_bytes = lambda: untracked_free() - hidden
    return recovery, quota


def _remap_worker(ctx: _Rank, mode: str) -> None:
    ctx.round_trip(0, _EVERYONE)
    ctx.comm.barrier()
    if ctx.rank == _VICTIM:
        _late_writer(ctx, [r for r in _EVERYONE if r != _VICTIM])
    else:
        _survivor(ctx, mode)


# ---- test_second_failure_orphans_importer_of_remapped_pages ----


def _replan(ctx: _Rank, failed: list[int]) -> list[int]:
    """Recovery without ``failed`` among the other ranks: RankFailureRecovery._remap_workspaces, then
    install_recovery_placement's mask commit. Returns the live ranks."""
    live = [r for r in _EVERYONE if r not in failed]
    live_mask = [sum(1 << r for r in live), 0, 0, 0]
    need = MnnvlMemory.allocated_map[ctx.memory.ptr].aligned_size
    recovery, _ = _remap_recovery(ctx, _Survivors(ctx.comm, live), need, live_mask)
    recovery._remap_workspaces(frozenset(failed), time.monotonic() + 60)
    assert ctx.watchdog_running(), "the remap did not restart the watchdog"
    for rank in failed:
        ctx.state.health.mark_failed(rank)
    ctx.state.rearm()
    ctx.state.install_mask()
    assert ctx.state.active_rank_mask.cpu().tolist() == live_mask
    handles = MnnvlMemory.allocated_map[ctx.memory.ptr].mem_handles
    assert [handle is None for handle in handles] == [r in failed for r in _EVERYONE], (
        "only the failed ranks' rows may stay unmapped"
    )
    return live


def _replan_worker(ctx: _Rank) -> None:
    ctx.round_trip(0, _EVERYONE)
    ctx.comm.barrier()
    if ctx.rank == _FIRST_FAILED:
        # Failed but alive: it keeps the original pages mapped and runs nothing more.
        ctx.store.wait(["stopped"])
        return

    # Round 1: the others remap without the first failed rank; the second imports the fresh pages.
    live = _replan(ctx, [_FIRST_FAILED])
    for seed in (1, 2):
        ctx.round_trip(seed, live)
    # Every round trip has read its rows before the second failed rank writes them.
    _Survivors(ctx.comm, live).barrier_among_survivors()
    survivors = [r for r in live if r != _SECOND_FAILED]
    if ctx.rank == _SECOND_FAILED:
        _late_writer(ctx, survivors)
        return

    # Its writes reach this rank's live rows through round 1's pages. One fill covers the row, so it shows
    # whole; polled, since the store reports progress only every few fills.
    row = ctx.workspace[ctx.rank]
    ctx.store.wait(["progress"])
    deadline = time.monotonic() + 10
    while not bool((row == _PATTERN).all()):
        assert time.monotonic() < deadline, (
            "the second failed rank's writes never reached round 1's pages"
        )
        time.sleep(_POLL_S)

    # Round 2: the survivors remap without both failed ranks while the second keeps writing.
    assert _replan(ctx, [_FIRST_FAILED, _SECOND_FAILED]) == survivors
    reset = _reset_row(ctx).cuda()
    progress = int(ctx.store.get("progress"))
    deadline = time.monotonic() + _WRITE_S
    while True:
        changed = (row != reset).nonzero().flatten()[:16]
        assert changed.numel() == 0, f"bytes of the remapped row changed at {changed.tolist()}"
        if time.monotonic() >= deadline:
            break
        time.sleep(_POLL_S)
    assert int(ctx.store.get("progress")) > progress, "the second failed rank stopped writing"
    # A peer's round trip writes this row: none starts before every survivor checked its own.
    survivor_group = _Survivors(ctx.comm, survivors)
    survivor_group.barrier_among_survivors()
    for seed in (3, 4):
        ctx.round_trip(seed, survivors)
    assert ctx.state.abort.tolist() == [0, 0]
    survivor_group.barrier_among_survivors()
    if ctx.rank == survivors[0]:
        ctx.store.set("stop", "1")
    # Exit only after the writer's last write, so every write it made found its exporter alive.
    ctx.store.wait(["stopped"])
    ctx.state.close()


# ---- test_aborted_round_with_runahead_rank ----


def _run_ahead(ctx: _Rank) -> None:
    """With its rounds left incomplete by its own abort request, run _AHEAD_ROUNDS rounds the peers have not
    started."""
    ctx.state.request_abort()
    for k in range(_AHEAD_ROUNDS):
        # Each round routes differently, so its receive counts differ from those of the peers' round.
        x, targets = _tokens(100 + k, ctx.rank, _EVERYONE)
        ctx.launch(x.cuda(), targets.cuda())
    done = torch.cuda.Event()
    done.record()
    _await(done, ctx.state, "a rank with an incomplete round kept waiting for its peers")
    assert ctx.state.abort.tolist() == [1, 1]
    counter, incomplete = ctx.round_flags()
    assert incomplete == 1, "the run-ahead rank does not record its incomplete round"
    ctx.store.set("ahead", str(counter))


def _fall_behind(ctx: _Rank) -> None:
    """Run the round the run-ahead rank already left: wait for it, then return aborted on request.

    The dispatch may complete: the run-ahead rank published this round's dispatch flag before its wait was
    released. The combine waits, since a rank with an incomplete round publishes no flag until its workspace is
    reset.
    """
    ahead = int(ctx.store.get("ahead"))
    x, targets = _tokens(1, ctx.rank, _EVERYONE)
    ctx.launch(x.cuda(), targets.cuda())
    done = torch.cuda.Event()
    done.record()
    time.sleep(0.3)
    assert not done.query(), "the round completed without the run-ahead rank's flags"
    ctx.state.request_abort()
    _await(done, ctx.state, "the round kept waiting after the abort request")
    assert ctx.state.abort.tolist() == [1, 1], "the round did not report itself aborted"
    # The dispatch and combine prepare kernels advance flag_val[0] on every launch, aborted or not.
    assert ctx.round_flags()[0] < ahead, "the run-ahead rank's round counter is not ahead"


def _replay_across_mask(ctx: _Rank) -> None:
    """Replay a graph captured with every rank; between two replays the failed rank leaves and the others
    install a mask without it. The next replays must read it: waiting for the failed rank never ends."""
    x = torch.empty((_CAPACITY, _HIDDEN), dtype=torch.bfloat16, device="cuda")
    targets = torch.empty((_CAPACITY, 1), dtype=torch.int32, device="cuda")

    def load(seed: int, live: list[int]) -> None:
        host_x, host_targets = _tokens(seed, ctx.rank, live)
        x.copy_(host_x)
        targets.copy_(host_targets)

    load(4, _EVERYONE)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        received, combined = ctx.launch(x, targets)
    graph.replay()
    torch.cuda.synchronize()
    _check_round(ctx.rank, 4, _EVERYONE, received, combined)
    ctx.comm.barrier()
    if ctx.rank == _FAILED:
        ctx.store.wait(["done"])
        return

    live = [r for r in _EVERYONE if r != _FAILED]
    # No remap: the failed rank stays mapped, so only the mask keeps the replays off it.
    ctx.state.health.mark_failed(_FAILED)
    ctx.state.install_mask()
    for seed in (5, 6):
        load(seed, live)
        graph.replay()
        done = torch.cuda.Event()
        done.record()
        _await(done, ctx.state, "a replay after install_mask waited for the failed rank")
        _check_round(ctx.rank, seed, live, received, combined)
        assert ctx.state.abort.tolist() == [0, 0]
    _Survivors(ctx.comm, live).barrier_among_survivors()
    if ctx.rank == live[0]:
        ctx.store.set("done", "1")


def _runahead_worker(ctx: _Rank, mode: str) -> None:
    ctx.round_trip(0, _EVERYONE)
    ctx.comm.barrier()
    if mode == "graph-mask":
        _replay_across_mask(ctx)
    else:
        (_run_ahead if ctx.rank == _RUNAHEAD else _fall_behind)(ctx)
        # Every rank is quiescent before any resets, and has reset before any runs the next round.
        ctx.comm.barrier()
        ctx.state.rearm()
        ctx.comm.barrier()
        assert ctx.state.abort.tolist() == [0, 0] and ctx.round_flags() == [0, 0]
        for seed in (2, 3):
            ctx.round_trip(seed, _EVERYONE)
        assert ctx.state.abort.tolist() == [0, 0]
    ctx.state.close()


def _worker(test: str, rank: int, port: int, mode: str) -> None:
    store = TCPStore("127.0.0.1", port, is_master=False, timeout=_TIMEOUT)
    ctx = _Rank(store, rank)
    if test == "replan":
        _replan_worker(ctx)
    else:
        (_remap_worker if test == "remap" else _runahead_worker)(ctx, mode)


# ---- launcher ----


def _launch(test: str, mode: str) -> tuple[TCPStore, list[subprocess.Popen]]:
    store = TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False, timeout=_TIMEOUT)
    # Keep the workers off any enclosing launcher's process-management interface: one may be killed.
    launcher_prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(launcher_prefixes)}
    workers = [
        subprocess.Popen(
            [sys.executable, __file__, test, str(rank), str(store.port), mode], env=env
        )
        for rank in range(_EP_SIZE)
    ]
    return store, workers


def _wait_for(store: TCPStore, keys: list[str], workers: list[subprocess.Popen]) -> None:
    deadline = time.monotonic() + _TIMEOUT.total_seconds()
    while not store.check(keys):
        assert all(worker.poll() in (None, 0) for worker in workers), "a worker failed"
        assert time.monotonic() < deadline, f"timed out waiting for {keys}"
        time.sleep(0.01)


def _join(workers: list[subprocess.Popen]) -> None:
    for worker in workers:
        assert worker.wait(timeout=_TIMEOUT.total_seconds()) == 0


def _reap(workers: list[subprocess.Popen]) -> None:
    for worker in workers:
        if worker.poll() is None:
            worker.kill()
            worker.wait()


@pytest.mark.parametrize("mode", ["stop", "kill"])
def test_remap_orphans_late_writer(mode: str) -> None:
    """Rank 2 keeps writing a pattern into rows 0, 1 and 3 while the survivors release and remap the workspace
    among themselves. Their rows then hold exactly the reset state for _WRITE_S, their masked round trips are
    exact, and the released pages stay allocated while rank 2 lives (``stop``) and are freed once it is
    SIGKILLed (``kill``)."""
    reason = mnnvl_skip_reason(_EP_SIZE)
    if reason:
        pytest.skip(reason)
    store, workers = _launch("remap", mode)
    survivors = [r for r in _EVERYONE if r != _VICTIM]
    victim = workers[_VICTIM]
    try:
        _wait_for(store, [f"remapped/{r}" for r in survivors], workers)
        time.sleep(_WRITE_S)
        if mode == "kill":
            victim.kill()
            victim.wait()
        store.set("written", "1")
        if mode == "stop":
            _wait_for(store, [f"checked/{r}" for r in survivors], workers)
            store.set("stop", "1")
            # Its writes to the pages it kept mapped never faulted.
            _join([victim])
        _join([workers[r] for r in survivors])
    finally:
        _reap(workers)


def test_second_failure_orphans_importer_of_remapped_pages() -> None:
    """Rank 3 fails and ranks 0-2 remap without it. Rank 2, which imported those fresh pages, then keeps
    writing a pattern into rows 0 and 1, which their polls see; ranks 0 and 1 remap without ranks 2 and 3. Their
    rows then hold exactly the reset state, polled over _WRITE_S of rank 2's writes, and their masked round trips
    are exact."""
    reason = mnnvl_skip_reason(_EP_SIZE)
    if reason:
        pytest.skip(reason)
    _store, workers = _launch("replan", "")  # the store serves the workers until they exit
    try:
        _join(workers)
    finally:
        _reap(workers)


@pytest.mark.parametrize("mode", ["runahead", "graph-mask"])
def test_aborted_round_with_runahead_rank(mode: str) -> None:
    """``runahead``: rank 1, its round left incomplete by an abort, runs _AHEAD_ROUNDS rounds; its peers' next round
    waits, returns aborted on request without trapping, and after a rearm every round trip is exact.
    ``graph-mask``: a graph captured with every rank replays exactly among ranks 0-2 after they install a mask
    without rank 3."""
    reason = mnnvl_skip_reason(_EP_SIZE)
    if reason:
        pytest.skip(reason)
    _store, workers = _launch("runahead", mode)  # the store serves the workers until they exit
    try:
        _join(workers)
    finally:
        _reap(workers)


if __name__ == "__main__":
    _worker(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])

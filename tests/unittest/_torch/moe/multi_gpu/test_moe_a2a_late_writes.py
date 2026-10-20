# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Writes a killed EP peer issued before its death stay inside the storage it owns.

Two processes share a real MNNVL MoE All-to-All workspace under bound (fixed source
slot) execution. The victim, rank 1, keeps pushing capacity-sized
dispatches into the survivor's workspace and is SIGKILLed while they are in flight.
The survivor, rank 0, then checks that every byte rank 1 does not own still holds
its sentinel. A dispatch it then issues waits on the dead rank 1 until its MoeExecutionState
aborts it; after the state rearms and installs a membership without rank 1, its captured
dispatch/combine graph returns exact results and nothing from rank 1.
"""

import itertools
import os
import pickle
import subprocess
import sys
import time
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest
import torch
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
_GRACE_S = 2.0
_TIMEOUT = timedelta(seconds=300)


class _StoreComm:
    """The communicator subset MnnvlMemory uses, over a TCPStore."""

    def __init__(self, store: TCPStore, rank: int) -> None:
        self.store, self.rank, self.calls = store, rank, 0

    def Get_rank(self) -> int:
        return self.rank

    def Get_size(self) -> int:
        return _EP_SIZE

    def allgather(self, value: Any) -> list[Any]:
        self.calls += 1
        self.store.set(f"allgather/{self.calls}/{self.rank}", pickle.dumps(value))
        return [
            pickle.loads(self.store.get(f"allgather/{self.calls}/{r}")) for r in range(_EP_SIZE)
        ]

    def barrier(self) -> None:
        self.allgather(None)


def _offset(meta: torch.Tensor, name: str) -> int:
    return int(meta[int(getattr(internal.thop, f"MOE_A2A_{name}_OFFSET_INDEX"))])


_WIDTHS = {"hidden_slot": _HIDDEN * 2, "expert_id_slot": 4}  # payload bytes per token


def _regions(meta: torch.Tensor) -> dict[str, int]:
    """Start of each payload's receive region: a fixed slot of the capacity per source, cacheline padded."""
    starts, offset = {}, _offset(meta, "DISPATCH_PAYLOAD")
    for name, width in _WIDTHS.items():
        starts[name] = offset
        offset += (_EP_SIZE * _CAPACITY * width + 127) // 128 * 128
    starts["end"] = offset
    return starts


def _victim_owned(meta: torch.Tensor) -> dict[str, tuple[int, int]]:
    """Byte ranges of the survivor's row that the victim writes under bound execution."""
    regions = _regions(meta)
    recv = _offset(meta, "RECV_COUNTERS")
    ranges = {
        name: (
            regions[name] + _VICTIM * _CAPACITY * w,
            regions[name] + (_VICTIM + 1) * _CAPACITY * w,
        )
        for name, w in _WIDTHS.items()
    }
    for parity in range(2):
        counter = recv + 4 * (parity * _EP_SIZE + _VICTIM)
        ranges[f"recv_counter_{parity}"] = (counter, counter + 4)
    for name in ("DISPATCH_COMPLETION_FLAGS", "COMBINE_COMPLETION_FLAGS"):
        flag = _offset(meta, name) + 4 * _VICTIM
        ranges[name.lower()] = (flag, flag + 4)
    return ranges


def _controls(mask: torch.Tensor, abort: torch.Tensor) -> dict:
    return {"enable_rank_mask": True, "active_rank_mask": mask, "abort_state": abort}


def _records(requested: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A device rank mask of both peers and a pinned (requested, status) abort record."""
    mask = torch.tensor([3, 0, 0, 0], dtype=torch.uint64, device="cuda")
    abort = torch.tensor([requested, 0], dtype=torch.int32).pin_memory()
    return mask, abort


def _dispatch(workspace, meta, rank, experts, payload, mask, abort, valid_counts):
    """Dispatch [payload, experts]; ``valid_counts`` receives the rows of each source's slot to read."""
    args = (experts, [payload, experts], workspace, meta, _CAPACITY, rank, _EP_SIZE, 1, _EP_SIZE)
    return torch.ops.trtllm.moe_a2a_dispatch(
        *args, **_controls(mask, abort), source_valid_counts=valid_counts
    )


def _bound_state(
    memory: MnnvlMemory, workspace: torch.Tensor, meta: torch.Tensor
) -> MoeExecutionState:
    """The survivor's MoeExecutionState, bound by an NVLinkOneSided layer on ``workspace`` as a model binds it.

    The layer carries only the fields binding and the workspace reset read; no MoE layer is built.
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
    # The workspace lifecycle supplies the metainfo; this layer runs no watchdog.
    comm._workspace_lifecycle = SimpleNamespace(metainfo=meta, watchdog_for=lambda layer: None)
    comm.ep_rank, comm.ep_size = _SURVIVOR, _EP_SIZE
    comm.bind_execution_state(state)
    return state


def _victim(store: TCPStore, workspace: torch.Tensor, meta: torch.Tensor) -> None:
    # A pending abort request lets every dispatch push all its rows and then return instead of
    # waiting for the silent survivor, so new NVLink writes are always in flight.
    mask, abort = _records(requested=1)
    experts = torch.zeros((_CAPACITY, 1), dtype=torch.int32, device="cuda")  # expert 0: rank 0
    payload = torch.full((_CAPACITY, _HIDDEN * 2), 0xC3, dtype=torch.uint8, device="cuda")
    payload = payload.view(torch.bfloat16)
    valid_counts = torch.zeros(_EP_SIZE, dtype=torch.int32, device="cuda")
    store.wait(["go"])
    for step in itertools.count():
        _dispatch(workspace, meta, _VICTIM, experts, payload, mask, abort, valid_counts)
        if step == 64:
            torch.cuda.synchronize()
            store.set("running", "1")


def _survivor(
    store: TCPStore, memory: MnnvlMemory, workspace: torch.Tensor, meta: torch.Tensor
) -> None:
    row = workspace[_SURVIVOR]
    state = _bound_state(memory, workspace, meta)
    mask, abort = state.active_rank_mask, state.abort
    experts = torch.zeros((_CAPACITY, 1), dtype=torch.int32, device="cuda")
    payload = torch.empty((_CAPACITY, _HIDDEN), dtype=torch.bfloat16, device="cuda")

    # Captured with full membership; replayed only after the victim is gone.
    valid_counts = torch.zeros(_EP_SIZE, dtype=torch.int32, device="cuda")  # the graph keeps it
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        received, offset, _ = _dispatch(
            workspace, meta, _SURVIVOR, experts, payload, mask, abort, valid_counts
        )
        args = (received[0], _CAPACITY, workspace, meta, _CAPACITY, _SURVIVOR, _EP_SIZE, 1)
        output = torch.ops.trtllm.moe_a2a_combine(*args, offset, False, **_controls(mask, abort))
    # Every payload is read from its slots in place.
    assert offset == _offset(meta, "COMBINE_INPUT")
    assert received[0].data_ptr() == row.data_ptr() + _regions(meta)["hidden_slot"]
    assert received[1].data_ptr() == row.data_ptr() + _regions(meta)["expert_id_slot"]

    sentinel = torch.randint(0, 256, row.shape, dtype=torch.uint8, device="cuda")
    row.copy_(sentinel)
    torch.cuda.synchronize()
    store.set("go", "1")
    store.wait(["dead"])
    time.sleep(_GRACE_S)

    owned = _victim_owned(meta)
    live = torch.ones_like(row, dtype=torch.bool)
    for start, end in owned.values():
        live[start:end] = False
    contaminated = ((row != sentinel) & live).nonzero().flatten()
    assert contaminated.numel() == 0, f"bytes outside the victim's storage: {contaminated[:16]}"
    # Ring restores stage around what source_write_ranges reports for a lost rank: it covers those bytes.
    staged = torch.ones_like(row, dtype=torch.bool)
    for start, end in NVLinkOneSided.source_write_ranges(meta, _VICTIM):
        staged[start:end] = False
    assert not bool((~live & staged).any()), "source_write_ranges misses storage the victim owns"

    # The next dispatch waits for the dead victim's flags (they ran past its round) until recovery aborts it.
    done = torch.cuda.Event()
    _dispatch(
        workspace, meta, _SURVIVOR, experts, payload, mask, abort, torch.zeros_like(valid_counts)
    )
    done.record()
    time.sleep(0.2)
    assert not done.query(), "dispatch completed without the dead victim"
    state.request_abort()
    deadline = time.monotonic() + 10
    while not done.query():
        assert time.monotonic() < deadline, "dispatch kept waiting after the abort request"
        time.sleep(0.01)
    assert abort.tolist() == [1, 1]

    state.health.mark_failed(_VICTIM)
    with pytest.raises(RuntimeError, match="abort record"):
        state.install_mask()
    state.rearm()
    state.install_mask()
    assert mask.cpu().tolist() == [1 << _SURVIVOR, 0, 0, 0]
    values = torch.arange(_CAPACITY * _HIDDEN, device="cuda").view(_CAPACITY, _HIDDEN) % 128
    for step in range(4):
        payload.copy_(values + step)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, payload, rtol=0, atol=0)
        # No row of the victim's slots counts as received.
        assert valid_counts[_SURVIVOR].item() == _CAPACITY and valid_counts[_VICTIM].item() == 0
        assert torch.all(received[1][_SURVIVOR] == 0)
        assert abort.tolist() == [0, 0]
    # The survivor's own rounds rewrite its receive counters for every source.
    for name, (start, end) in owned.items():
        if not name.startswith("recv_counter"):
            assert torch.count_nonzero(row[start:end]) == 0, f"late write into {name}"
    state.close()


def _worker(rank: int, port: int) -> None:
    store = TCPStore("127.0.0.1", port, is_master=False, timeout=_TIMEOUT)
    torch.cuda.set_device(rank)
    mapping = Mapping(_EP_SIZE, rank, _EP_SIZE, tp_size=_EP_SIZE, moe_ep_size=_EP_SIZE)
    MnnvlMemory.initialize()
    comm = _StoreComm(store, rank)
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
        _survivor(store, memory, workspace, meta)
    else:
        _victim(store, workspace, meta)


def _mnnvl_skip_reason() -> str | None:
    if torch.cuda.device_count() < _EP_SIZE:
        return f"Requires {_EP_SIZE} local GPUs"
    MnnvlMemory.initialize()
    if not MnnvlMemory.supports_mnnvl():
        return "Requires an NVLink fabric supported by MNNVL"
    return None


@pytest.mark.parametrize("kill_delay_s", [0.0, 0.003, 0.05])
def test_killed_peer_writes_stay_in_its_storage(kill_delay_s: float) -> None:
    reason = _mnnvl_skip_reason()
    if reason:
        pytest.skip(reason)
    store = TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False, timeout=_TIMEOUT)
    # Keep the workers off any enclosing launcher's process-management interface: one is killed.
    launcher_prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(launcher_prefixes)}
    workers = [
        subprocess.Popen([sys.executable, __file__, str(rank), str(store.port)], env=env)
        for rank in range(_EP_SIZE)
    ]
    survivor, victim = workers[_SURVIVOR], workers[_VICTIM]
    try:
        deadline = time.monotonic() + _TIMEOUT.total_seconds()
        while not store.check(["running"]):
            assert all(worker.poll() is None for worker in workers), "worker exited during setup"
            assert time.monotonic() < deadline, "victim never started dispatching"
            time.sleep(0.01)
        time.sleep(kill_delay_s)
        victim.kill()
        victim.wait()
        store.set("dead", "1")
        assert survivor.wait(timeout=_TIMEOUT.total_seconds()) == 0
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait()


if __name__ == "__main__":
    _worker(int(sys.argv[1]), int(sys.argv[2]))

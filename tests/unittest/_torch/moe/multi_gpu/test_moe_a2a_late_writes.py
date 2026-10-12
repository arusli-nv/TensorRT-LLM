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
"""Writes a killed EP peer issued before its death stay inside the storage it owns.

Two processes share a real MNNVL MoE All-to-All workspace under execution-descriptor
(fixed source slot) execution. The victim, rank 1, keeps pushing capacity-sized
dispatches into the survivor's workspace and is SIGKILLed while they are in flight.
The survivor, rank 0, then checks that every byte rank 1 does not own still holds
its sentinel, and that, with rank 1 masked and the workspace rearmed, its captured
dispatch/combine graph returns exact results and nothing from rank 1.
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
from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm.bindings import internal
from tensorrt_llm.mapping import Mapping

_EP_SIZE, _SURVIVOR, _VICTIM = 2, 0, 1
_CAPACITY, _HIDDEN = 1024, 7168
_GRACE_S = 2.0
_TIMEOUT = timedelta(seconds=300)
_FULL_MASK = torch.tensor([2**64 - 1] * 4, dtype=torch.uint64)


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


def _source_stride() -> int:
    """Bytes of one source slot: the hidden-state and expert-id payloads, each cacheline padded."""
    return sum((_CAPACITY * width + 127) // 128 * 128 for width in (_HIDDEN * 2, 4))


def _victim_owned(meta: torch.Tensor) -> dict[str, tuple[int, int]]:
    """Byte ranges of the survivor's row that the victim writes under descriptor execution."""
    slot = _offset(meta, "PAYLOAD_DATA") + _VICTIM * _source_stride()
    recv = _offset(meta, "RECV_COUNTERS")
    stats = _offset(meta, "EPLB_GATHERED_STATS") + _VICTIM * _EP_SIZE * 4
    ranges = {
        "source_slot": (slot, slot + _source_stride()),
        "eplb_stats": (stats, stats + _EP_SIZE * 4),
    }
    for parity in range(2):
        counter = recv + 4 * (parity * _EP_SIZE + _VICTIM)
        ranges[f"recv_counter_{parity}"] = (counter, counter + 4)
    for name in ("DISPATCH_COMPLETION_FLAGS", "COMBINE_COMPLETION_FLAGS"):
        flag = _offset(meta, name) + 4 * _VICTIM
        ranges[name.lower()] = (flag, flag + 4)
    return ranges


def _controls(descriptor: torch.Tensor) -> dict:
    return {
        "enable_rank_mask": True,
        "active_rank_mask": _FULL_MASK,
        "execution_descriptor": descriptor,
    }


def _dispatch(workspace, meta, rank, experts, payload, descriptor, stats):
    args = (experts, [payload, experts], workspace, meta, _CAPACITY, rank, _EP_SIZE, 1, _EP_SIZE)
    return torch.ops.trtllm.moe_a2a_dispatch(
        *args, stats, expert_id_payload_index=1, invalid_token_expert_id=-1, **_controls(descriptor)
    )


def _victim(store: TCPStore, workspace: torch.Tensor, meta: torch.Tensor) -> None:
    # A pending abort request lets every dispatch push all its rows and then return instead of
    # waiting for the silent survivor, so new NVLink writes are always in flight.
    descriptor = torch.tensor([3, 0, 0, 0, 1], dtype=torch.int64, device="cuda")
    experts = torch.zeros((_CAPACITY, 1), dtype=torch.int32, device="cuda")  # expert 0: rank 0
    payload = torch.full((_CAPACITY, _HIDDEN * 2), 0xC3, dtype=torch.uint8, device="cuda")
    payload = payload.view(torch.bfloat16)
    stats = torch.full((_EP_SIZE,), 7, dtype=torch.int32, device="cuda")
    store.wait(["go"])
    for step in itertools.count():
        _dispatch(workspace, meta, _VICTIM, experts, payload, descriptor, stats)
        if step == 64:
            torch.cuda.synchronize()
            store.set("running", "1")


def _survivor(store: TCPStore, workspace: torch.Tensor, meta: torch.Tensor) -> None:
    row = workspace[_SURVIVOR]
    descriptor = torch.tensor([3, 0, 0, 0, 0], dtype=torch.int64, device="cuda")
    abort = descriptor[4:].view(torch.int32)
    experts = torch.zeros((_CAPACITY, 1), dtype=torch.int32, device="cuda")
    payload = torch.empty((_CAPACITY, _HIDDEN), dtype=torch.bfloat16, device="cuda")
    stats = torch.tensor([_CAPACITY, 0], dtype=torch.int32, device="cuda")

    # Captured with full membership; replayed only after the victim is gone.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        received, offset, gathered = _dispatch(
            workspace, meta, _SURVIVOR, experts, payload, descriptor, stats
        )
        args = (received[0], _CAPACITY, workspace, meta, _CAPACITY, _SURVIVOR, _EP_SIZE, 1)
        output = torch.ops.trtllm.moe_a2a_combine(*args, offset, False, **_controls(descriptor))
    assert offset == _offset(meta, "PAYLOAD_DATA") + _EP_SIZE * _source_stride()

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

    # Exclude the victim and rearm as MoeExecutionState.rearm does for NVLinkOneSided.
    descriptor[0] = 1 << _SURVIVOR
    row.zero_()
    recv = _offset(meta, "RECV_COUNTERS")
    row[recv : recv + 2 * _EP_SIZE * 4].fill_(0xFF)
    abort.zero_()
    values = torch.arange(_CAPACITY * _HIDDEN, device="cuda").view(_CAPACITY, _HIDDEN) % 128
    for step in range(4):
        payload.copy_(values + step)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, payload, rtol=0, atol=0)
        assert torch.all(received[1][_SURVIVOR] == 0) and torch.all(received[1][_VICTIM] == -1)
        assert torch.count_nonzero(received[0][_VICTIM]) == 0
        assert gathered.tolist() == [[_CAPACITY, 0], [0, 0]]
        assert abort.tolist() == [0, 0]
    # The survivor's own rounds rewrite its receive counters for every source.
    for name, (start, end) in owned.items():
        if not name.startswith("recv_counter"):
            assert torch.count_nonzero(row[start:end]) == 0, f"late write into {name}"


def _worker(rank: int, port: int) -> None:
    store = TCPStore("127.0.0.1", port, is_master=False, timeout=_TIMEOUT)
    torch.cuda.set_device(rank)
    mapping = Mapping(_EP_SIZE, rank, _EP_SIZE, tp_size=_EP_SIZE, moe_ep_size=_EP_SIZE)
    MnnvlMemory.initialize()
    comm = _StoreComm(store, rank)
    MnnvlMemory.comm = comm
    MnnvlMemory.comm_signature = MnnvlMemory.comm_layout_signature(mapping)
    size = NVLinkOneSided.calculate_required_workspace_size(
        _EP_SIZE,
        1,
        _CAPACITY,
        _HIDDEN,
        torch.bfloat16,
        eplb_stats_num_experts=_EP_SIZE,
        fixed_source_slots=True,
    )
    memory = MnnvlMemory(mapping, size)
    workspace = memory.as_torch_strided_tensor(torch.uint8)
    meta = torch.ops.trtllm.moe_a2a_initialize(workspace, rank, _EP_SIZE, _CAPACITY, _EP_SIZE)
    # No peer may write into a row before its owner has initialized it.
    comm.barrier()
    (_survivor if rank == _SURVIVOR else _victim)(store, workspace, meta)


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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ring host copies restore through real MNNVL NVLinkOneSided workspaces, bit-exactly.

Four processes, one GPU each, map a real MoE All-to-All workspace. The ring
treats ranks 0 and 2 as one node and ranks 1 and 3 as the other; ranks 1 and 3
are lost. The survivors copy the lost experts from host memory into the
target's workspace (their own or the peer's, over NVLink) and from there into
the target's slots, in chunks of half of what the all-to-alls leave unwritten of the workspace, as a WideEP
restore does: the bytes the lost ranks' all-to-alls may write still hold what they held before. In the
``lost_writing`` case the lost ranks keep dispatching into every rank's workspace while the survivors stage.
"""

import functools
import os
import pickle
import subprocess
import sys
import time
from datetime import timedelta
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch.distributed import TCPStore

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.moe.fused_moe.eplb_recovery import restore_lost_experts, ring_copy_nodes
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    MoeLoadBalancer,
    SingleLayerMoeLoadBalancer,
)
from tensorrt_llm.bindings import internal
from tensorrt_llm.mapping import Mapping

_EP_SIZE = 4
_HOSTS = ["a", "b", "a", "b"]
_DEAD = [1, 3]
_SURVIVORS = [0, 2]
_EXPERTS = 8
# Two slots per rank hold one expert each; two ghost slots per rank take restores.
_PLACEMENT = [[0, 1, -1, -1], [2, 3, -1, -1], [4, 5, -1, -1], [6, 7, -1, -1]]
# Per slot: w1 3 MiB of bytes, w2 1.5 MiB of bfloat16 (sizes of real expert weights' order of magnitude).
_WEIGHTS = {"w1": (3 << 20, torch.uint8), "w2": (768 << 10, torch.bfloat16)}
# The workspace of a bound layer with 128 tokens per rank: halves of about 3.5 MiB, so weights continue across
# chunks and each target reads from both halves.
_TOKENS, _HIDDEN = 128, 7168
_TIMEOUT = timedelta(seconds=300)


class _StoreComm:
    """The communicator subset MnnvlMemory uses, and a barrier of the survivors, over a TCPStore."""

    def __init__(self, store: TCPStore, rank: int) -> None:
        self.store, self.rank, self.calls, self.survivor_calls = store, rank, 0, 0

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

    def survivor_barrier(self) -> None:
        self.survivor_calls += 1
        self.store.set(f"survivors/{self.survivor_calls}/{self.rank}", b"")
        self.store.wait([f"survivors/{self.survivor_calls}/{r}" for r in _SURVIVORS])


def _host_copy(layer_idx: int, expert: int, name: str) -> torch.Tensor:
    count, dtype = _WEIGHTS[name]
    generator = torch.Generator().manual_seed(layer_idx * 1000 + expert * 10 + len(name))
    data = torch.randint(0, 256, (count * dtype.itemsize,), dtype=torch.uint8, generator=generator)
    return data.view(dtype)


def _load_balancer(rank: int) -> MoeLoadBalancer:
    """One layer over _PLACEMENT, with this rank's ring host copies and slots."""
    copy_nodes = ring_copy_nodes(_HOSTS)
    layer = SingleLayerMoeLoadBalancer.__new__(SingleLayerMoeLoadBalancer)
    layer.single_layer_load_balancer_impl = Mock(
        get_layer_id=Mock(return_value=0), get_old_rank_expert_ids=Mock(return_value=_PLACEMENT)
    )
    layer.home_ranks = np.repeat(np.arange(_EP_SIZE), 2)
    layer.weight_slots = {
        (slot, name): torch.zeros(count, dtype=dtype, device="cuda")
        for slot in range(len(_PLACEMENT[0]))
        for name, (count, dtype) in _WEIGHTS.items()
    }
    layer.ring_host_weights = {
        (expert, name): _host_copy(0, expert, name)
        for expert in range(_EXPERTS)
        if rank in copy_nodes[layer.home_ranks[expert]]
        for name in _WEIGHTS
    }
    load_balancer = MoeLoadBalancer.__new__(MoeLoadBalancer)
    load_balancer.ep_rank, load_balancer.ep_size = rank, _EP_SIZE
    load_balancer.copy_nodes = copy_nodes
    load_balancer.single_layer_load_balancers = [layer]
    return load_balancer


def _late_writer(comm: _StoreComm, workspace: torch.Tensor, meta: torch.Tensor) -> None:
    """Dispatch rows to every rank until both survivors have staged.

    A pending abort request lets every dispatch push all its rows and then return instead of
    waiting for its peers, so new NVLink writes are in flight throughout the staging.
    """
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, comm.rank, _EP_SIZE, barrier=False)
    controls = {
        "enable_rank_mask": True,
        "active_rank_mask": torch.tensor(
            [(1 << _EP_SIZE) - 1, 0, 0, 0], dtype=torch.uint64, device="cuda"
        ),
        "abort_state": torch.tensor([1, 0], dtype=torch.int32).pin_memory(),
    }
    # Expert e lives on rank e.
    experts = torch.arange(_TOKENS, dtype=torch.int32, device="cuda").view(_TOKENS, 1) % _EP_SIZE
    payload = torch.full((_TOKENS, _HIDDEN * 2), 0xC3, dtype=torch.uint8, device="cuda")
    payload = payload.view(torch.bfloat16)
    valid_counts = torch.zeros(_EP_SIZE, dtype=torch.int32, device="cuda")
    args = (experts, [payload, experts], workspace, meta, _TOKENS, comm.rank, _EP_SIZE, 1, _EP_SIZE)
    while True:
        for _ in range(16):
            torch.ops.trtllm.moe_a2a_dispatch(*args, **controls, source_valid_counts=valid_counts)
        torch.cuda.synchronize()
        comm.store.set(f"writing/{comm.rank}", b"")
        if comm.store.check([f"staged/{rank}" for rank in _SURVIVORS]):
            return


def _survivor(
    comm: _StoreComm, workspace: torch.Tensor, meta: torch.Tensor, lost_writing: bool
) -> None:
    load_balancer = _load_balancer(comm.rank)
    placement = np.array(_PLACEMENT, dtype=np.int32)
    placement[_DEAD] = -1
    assert restore_lost_experts(placement, [2, 3, 6, 7], _DEAD) == []
    write_ranges = functools.partial(NVLinkOneSided.source_write_ranges, meta)
    plan = load_balancer.restore_copy_plan(_DEAD, [placement], workspace.size(1), write_ranges)
    # Both survivors copy into the other's workspace as well as their own.
    assert {(source, target) for source, target in plan[:, [6, 2]].tolist()} == {
        (0, 0),
        (0, 2),
        (2, 0),
        (2, 2),
    }
    assert plan[-1, 0] >= 2
    if lost_writing:
        comm.store.wait([f"writing/{rank}" for rank in _DEAD])
    workspace[comm.rank].fill_(0xA5)
    torch.cuda.synchronize()
    comm.survivor_barrier()
    load_balancer.stage_restored_slots(plan, workspace, comm.survivor_barrier)
    comm.store.set(f"staged/{comm.rank}", b"")
    (layer,) = load_balancer.single_layer_load_balancers
    for slot, expert in enumerate(placement[comm.rank].tolist()):
        if _PLACEMENT[comm.rank][slot] == expert:
            continue
        for name in _WEIGHTS:
            loaded = layer.weight_slots[(slot, name)].cpu().view(torch.uint8)
            expected = _host_copy(0, expert, name).view(torch.uint8)
            assert torch.equal(loaded, expected), f"rank {comm.rank} slot {slot} {name}"
    row = workspace[comm.rank]
    lost_bytes = [row[begin:end] for rank in _DEAD for begin, end in write_ranges(rank)]
    if lost_writing:
        # The lost ranks' writes landed in this row after the fill, while it staged.
        assert any(bool((data != 0xA5).any()) for data in lost_bytes), f"rank {comm.rank}"
    else:
        for data in lost_bytes:
            assert bool((data == 0xA5).all()), f"rank {comm.rank} staged into a lost rank's range"


def _worker(rank: int, port: int, lost_writing: bool) -> None:
    store = TCPStore("127.0.0.1", port, is_master=False, timeout=_TIMEOUT)
    torch.cuda.set_device(rank)
    mapping = Mapping(_EP_SIZE, rank, _EP_SIZE, tp_size=_EP_SIZE, moe_ep_size=_EP_SIZE)
    MnnvlMemory.initialize()
    comm = _StoreComm(store, rank)
    MnnvlMemory.comm = comm
    MnnvlMemory.comm_signature = MnnvlMemory.comm_layout_signature(mapping)
    meta = NVLinkOneSided._make_workspace_layout(
        _EP_SIZE, 1, _TOKENS, _HIDDEN, torch.bfloat16, None, 0, False, False
    )
    memory = MnnvlMemory(mapping, int(meta[int(internal.thop.MOE_A2A_WORKSPACE_SIZE_INDEX)]))
    workspace = memory.as_torch_strided_tensor(torch.uint8)
    if rank in _SURVIVORS:
        _survivor(comm, workspace, meta, lost_writing)
    elif lost_writing:
        _late_writer(comm, workspace, meta)
    # The lost ranks keep their workspaces mapped until the survivors are done.
    comm.barrier()


def _mnnvl_skip_reason() -> str | None:
    if torch.cuda.device_count() < _EP_SIZE:
        return f"Requires {_EP_SIZE} local GPUs"
    MnnvlMemory.initialize()
    if not MnnvlMemory.supports_mnnvl():
        return "Requires an NVLink fabric supported by MNNVL"
    return None


@pytest.mark.parametrize("lost_writing", [False, True], ids=["lost_idle", "lost_writing"])
def test_ring_restore_stages_through_peer_workspaces_bit_exactly(lost_writing: bool) -> None:
    reason = _mnnvl_skip_reason()
    if reason:
        pytest.skip(reason)
    store = TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False, timeout=_TIMEOUT)
    launcher_prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(launcher_prefixes)}
    workers = [
        subprocess.Popen(
            [sys.executable, __file__, str(rank), str(store.port), str(int(lost_writing))], env=env
        )
        for rank in range(_EP_SIZE)
    ]
    try:
        deadline = time.monotonic() + _TIMEOUT.total_seconds()
        for worker in workers:
            assert worker.wait(timeout=max(1.0, deadline - time.monotonic())) == 0
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait()


if __name__ == "__main__":
    _worker(int(sys.argv[1]), int(sys.argv[2]), bool(int(sys.argv[3])))

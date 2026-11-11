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
"""With WideEP fault tolerance every rank is its own process (a one-process MPI world). Four such processes, started
independently on four GPUs, meet in the rendezvous store: they build MnnvlDist's host collectives and an
NVLinkOneSided workspace and round-trip a dispatch and combine. A replacement rank alone builds both with only its
own rank live. Run directly, this file is one such rank."""

import json
import os
import secrets
import socket
import subprocess  # nosec B404
import sys
from unittest.mock import Mock

import pytest
import torch

WORLD = 4
_LAUNCHER_ENV_PREFIXES = ("OMPI_", "PMIX_", "PMI_", "PRTE_", "SLURM_")


def _rank_main(rank: int, world_size: int, incarnation: int) -> dict:
    from mpi4py import MPI

    from tensorrt_llm._torch.distributed.communicator import MnnvlDist
    from tensorrt_llm._torch.distributed.startup_record import startup_record
    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
    from tensorrt_llm._torch.pyexecutor import rank_failure_recovery as recovery
    from tensorrt_llm._utils import mpi_comm, set_fault_tolerance_mode
    from tensorrt_llm.mapping import Mapping

    errors = []

    def check(ok: bool, what: str) -> None:
        if not ok:
            errors.append(f"rank {rank}: {what}")

    check(MPI.COMM_WORLD.Get_size() == 1, "the rank is not its own MPI world")
    set_fault_tolerance_mode(True)
    torch.cuda.set_device(rank)
    # Stands for the membership agent the worker's entry starts, which takes the startup collectives' reports.
    recovery._agent = Mock()
    if incarnation:
        # The startup record rank 0 published: every rank on this host.
        startup_record.replay(
            {"values": {"rank_hosts": [[socket.gethostname()] * world_size]}, "exchanges": []}
        )
    recovery.join_startup_group()
    # Past the startup record (a replacement's host collectives reach only itself).
    startup_record.close()
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        tp_size=world_size,
        moe_ep_size=world_size,
        enable_attention_dp=True,
        gpus_per_node=WORLD,
    )
    live = [rank] if incarnation else list(range(world_size))
    dist = MnnvlDist(mapping)
    check(
        dist.tp_allgather(rank) == [r if r in live else None for r in range(world_size)],
        "host collective",
    )
    check(dist.allreduce(rank + 1) == sum(r + 1 for r in live), "host reduction")

    hidden, top_k, capacity, slots_per_rank = 256, 2, 8, 2
    comm = NVLinkOneSided(
        mapping=mapping,
        num_slots=world_size * slots_per_rank,
        top_k=top_k,
        max_num_tokens_per_rank=capacity,
        hidden_size=hidden,
        dtype=torch.bfloat16,
    )
    check(comm.workspace.shape[0] == world_size, "workspace layout")
    if not incarnation:
        counts = [capacity - r % 2 for r in range(world_size)]
        generator = torch.Generator().manual_seed(rank)
        token = torch.arange(counts[rank])[:, None]
        # Each token goes to one rank, both of its slots there, so each combined row is that rank's output.
        slots = ((token + rank) % world_size * slots_per_rank + torch.arange(top_k)).int().cuda()
        weights = torch.rand(counts[rank], top_k, generator=generator).cuda()
        x = torch.randn(counts[rank], hidden, generator=generator).bfloat16().cuda()
        mpi_comm().barrier()
        recv = comm.dispatch(x, None, slots, weights, counts)
        unrouted = (recv[2] < 0).all(dim=1, keepdim=True)
        output = torch.where(unrouted, float("nan"), recv[0] * 2)
        combined = comm.combine(output, all_rank_max_num_tokens=max(counts))
        torch.cuda.synchronize()
        check(torch.equal(combined, x * 2), "dispatch and combine")
        mpi_comm().barrier()
    comm.destroy()
    return {"rank": rank, "errors": errors}


def _spawn(rank: int, identity: str, port: int, secret: str) -> subprocess.Popen:
    env = {k: v for k, v in os.environ.items() if not k.startswith(_LAUNCHER_ENV_PREFIXES)}
    env.update(
        TLLM_FAULT_TOLERANCE_RANK=identity,
        TLLM_FAULT_TOLERANCE_RENDEZVOUS=f"127.0.0.1:{port}",
        TLLM_FAULT_TOLERANCE_JOIN_SECRET=secret,
    )
    return subprocess.Popen(  # nosec B603
        [sys.executable, __file__, identity], env=env, stdout=subprocess.PIPE, text=True
    )


def _results(processes, timeout_s=900):
    results = []
    for process in processes:
        out, _ = process.communicate(timeout=timeout_s)
        assert process.returncode == 0, f"a rank exited with {process.returncode}"
        results.append(json.loads(out.strip().splitlines()[-1]))
    return results


@pytest.fixture
def rendezvous():
    """The rendezvous store, as the authority hosts it: its port and the join secret."""
    from torch.distributed import TCPStore

    store = TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    yield store.port, secrets.token_hex(32)


def _require_mnnvl(gpus: int) -> None:
    from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory

    if torch.cuda.device_count() < gpus or not MnnvlMemory.supports_mnnvl():
        pytest.skip(f"needs {gpus} GPUs with MNNVL")


def test_independent_ranks_build_host_collectives_and_the_all_to_all(rendezvous):
    _require_mnnvl(WORLD)
    port, secret = rendezvous
    processes = [_spawn(rank, f"{rank},{WORLD},0", port, secret) for rank in range(WORLD)]
    for result in _results(processes):
        assert result["errors"] == []


def test_a_replacement_builds_its_groups_alone(rendezvous):
    _require_mnnvl(WORLD)
    port, secret = rendezvous
    (result,) = _results([_spawn(2, f"2,{WORLD},1", port, secret)])
    assert result["errors"] == []


if __name__ == "__main__":
    rank, world_size, incarnation = (int(field) for field in sys.argv[1].split(","))
    print(json.dumps(_rank_main(rank, world_size, incarnation)), flush=True)
    os._exit(0)  # no MPI_Finalize: nothing ran over MPI

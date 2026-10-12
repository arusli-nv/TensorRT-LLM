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
"""MPIDist survivor recovery against a real ULFM MPI library.

The test launches this file under ``mpirun --with-ft ulfm``; one rank exits
and the others detect, interrupt, shrink and keep running host collectives.
"""

import importlib.metadata
import os
import shutil
import subprocess
import sys
import threading

import pytest

pytestmark = pytest.mark.cpu_only

WORLD_SIZE = 4
VICTIM = 2
_OK = "ULFM_SURVIVOR_OK"


def _ulfm_available() -> bool:
    """Whether Open MPI has the ULFM extension and mpi4py binds it (mpi4py>=4)."""
    if shutil.which("ompi_info") is None or shutil.which("mpirun") is None:
        return False
    if int(importlib.metadata.version("mpi4py").split(".")[0]) < 4:
        return False
    info = subprocess.run(["ompi_info"], capture_output=True, text=True, timeout=60)
    return "ftmpi" in info.stdout


@pytest.mark.skipif(not _ulfm_available(), reason="requires Open MPI with ULFM and mpi4py>=4")
def test_survivors_recover_from_a_departed_rank():
    command = [
        "mpirun",
        "--with-ft",
        "ulfm",
        "--mca",
        "pml",
        "ob1",
        "--mca",
        "btl",
        "self,tcp",
        "--mca",
        "coll",
        "^hcoll,ucc",
        "--allow-run-as-root",
        "--oversubscribe",
        "-np",
        str(WORLD_SIZE),
        sys.executable,
        __file__,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr
    survivors = [rank for rank in range(WORLD_SIZE) if rank != VICTIM]
    for rank in survivors:
        assert f"{_OK} {rank}" in output, output


def _run_rank() -> None:
    from mpi4py import MPI

    from tensorrt_llm._torch.distributed.communicator import HostCollectiveInterrupted, MPIDist
    from tensorrt_llm.mapping import Mapping

    rank = MPI.COMM_WORLD.Get_rank()
    mapping = Mapping(
        world_size=WORLD_SIZE,
        rank=rank,
        tp_size=WORLD_SIZE,
        moe_ep_size=WORLD_SIZE,
        enable_attention_dp=True,
    )
    dist = MPIDist(mapping)
    dist.enable_peer_loss_recovery()
    assert dist.tp_allgather(rank) == list(range(WORLD_SIZE))

    failure_seen = threading.Event()
    watch = dist.watch_peer_failures(failure_seen.set)
    dist.barrier()
    if rank == VICTIM:
        os._exit(0)

    assert failure_seen.wait(60), "the watch did not observe the departed rank"
    with pytest.raises(HostCollectiveInterrupted):
        dist.tp_allgather(rank)
    watch.stop()

    dist.revoke()
    failed = dist.shrink()
    assert failed == frozenset({VICTIM}), failed
    dist.install_survivors(failed)

    survivors = [r for r in range(WORLD_SIZE) if r != VICTIM]
    assert dist.live_ranks == tuple(survivors)
    expected = [None if r == VICTIM else r for r in range(WORLD_SIZE)]
    assert dist.tp_allgather(rank) == expected
    assert dist.allgather(rank) == expected
    gathered = dist.tp_allgather_int64([rank], fill_value=-1)[:, 0].tolist()
    assert gathered == [-1 if r == VICTIM else r for r in range(WORLD_SIZE)]
    assert dist.broadcast(rank * 10, root=survivors[-1]) == survivors[-1] * 10

    # Survivors can watch again after recovery; a clean stop never fires.
    fired = threading.Event()
    watch = dist.watch_peer_failures(fired.set)
    dist.barrier()
    watch.stop()
    assert not fired.is_set()
    print(f"{_OK} {rank}", flush=True)


if __name__ == "__main__":
    _run_rank()

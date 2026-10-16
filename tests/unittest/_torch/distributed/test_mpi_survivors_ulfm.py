# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
    import numpy as np
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
    abort = np.zeros(1, dtype=np.int32)
    watch = dist.watch_peer_failures(failure_seen.set, abort.ctypes.data)
    dist.barrier()
    if rank == VICTIM:
        os._exit(0)

    assert failure_seen.wait(60), "the watch did not observe the departed rank"
    assert watch.failed and abort[0] == 1
    with pytest.raises(HostCollectiveInterrupted):
        dist.tp_allgather(rank)
    watch.stop()

    dist.revoke()
    failed = dist.shrink()
    assert failed == frozenset({VICTIM}), failed
    dist.install_survivors()

    survivors = [r for r in range(WORLD_SIZE) if r != VICTIM]
    assert dist.live_ranks == tuple(survivors)
    expected = [None if r == VICTIM else r for r in range(WORLD_SIZE)]
    assert dist.tp_allgather(rank, departed=None) == expected
    assert dist.allgather(rank, departed=None) == expected
    gathered = dist.tp_allgather_int64([rank], departed=-1)[:, 0].tolist()
    assert gathered == [-1 if r == VICTIM else r for r in range(WORLD_SIZE)]
    assert dist.broadcast(rank * 10) == 0

    # Survivors can watch again after recovery; a clean stop never fires.
    fired = threading.Event()
    watch = dist.watch_peer_failures(fired.set)
    dist.barrier()
    watch.stop()
    assert not fired.is_set()
    print(f"{_OK} {rank}", flush=True)


if __name__ == "__main__":
    _run_rank()

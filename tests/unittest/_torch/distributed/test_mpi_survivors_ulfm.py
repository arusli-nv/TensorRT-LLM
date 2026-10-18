# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MPIDist survivor recovery against a real ULFM MPI library.

The test launches this file under ``mpirun --with-ft ulfm``; one rank exits (or
freezes and is fenced through the runtime) and the others detect, interrupt,
shrink and keep running host collectives.
"""

import importlib.metadata
import os
import shutil
import signal
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
@pytest.mark.parametrize(
    "second_victim, frozen",
    [(None, False), (3, False), (None, True)],
    ids=["exit", "replan", "fenced"],
)
def test_survivors_recover_from_departed_ranks(second_victim, frozen):
    """One rank exits; with ``second_victim``, another exits during the recovery, before install.

    ``frozen``: the victim stops itself (SIGSTOP) instead, and rank 0 fences it through the runtime
    (PMIx job control): PRRTE ends it alone, and the survivors see it fail.
    """
    command = [
        "mpirun",
        "--with-ft",
        "ulfm",
        "-x",
        f"ULFM_SECOND_VICTIM={-1 if second_victim is None else second_victim}",
        "-x",
        f"ULFM_FROZEN_VICTIM={int(frozen)}",
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
    survivors = [rank for rank in range(WORLD_SIZE) if rank not in (VICTIM, second_victim)]
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
        if os.environ.get("ULFM_FROZEN_VICTIM") == "1":
            os.kill(os.getpid(), signal.SIGSTOP)
        os._exit(0)
    if rank == 0 and os.environ.get("ULFM_FROZEN_VICTIM") == "1":
        from tensorrt_llm._torch.distributed.pmix_fence import kill_rank

        kill_rank(VICTIM)

    assert failure_seen.wait(60), "the watch did not observe the departed rank"
    assert watch.failed and abort[0] == 1
    with pytest.raises(HostCollectiveInterrupted):
        dist.tp_allgather(rank)
    watch.stop()

    dist.revoke()
    failed = dist.shrink()
    assert failed == frozenset({VICTIM}), failed
    departed = {VICTIM}
    second = int(os.environ.get("ULFM_SECOND_VICTIM", "-1"))
    if second >= 0:
        # Another rank fails during the recovery: the survivors shrink again.
        replan = threading.Event()
        watch = dist.watch_survivor_failures(replan.set)
        if rank == second:
            os._exit(0)
        assert replan.wait(60), "the survivor watch did not observe the second departed rank"
        watch.stop()
        dist.revoke()
        failed = dist.shrink()
        departed.add(second)
        assert failed == frozenset(departed), failed
    dist.install_survivors()

    survivors = [r for r in range(WORLD_SIZE) if r not in departed]
    assert dist.live_ranks == tuple(survivors)
    expected = [None if r in departed else r for r in range(WORLD_SIZE)]
    assert dist.tp_allgather(rank, departed=None) == expected
    assert dist.allgather(rank, departed=None) == expected
    gathered = dist.tp_allgather_int64([rank], departed=-1)[:, 0].tolist()
    assert gathered == [-1 if r in departed else r for r in range(WORLD_SIZE)]
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

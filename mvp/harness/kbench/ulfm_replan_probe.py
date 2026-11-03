# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Why did survivors abort ~1 s after a re-planned recovery ("An error occurred in PMIx Event Notification ... on a
NULL communicator ... MPI_ERRORS_ARE_FATAL")? The pure-MPI test of the re-plan passes, but its victims exit cleanly.

Run under ``mpirun --with-ft ulfm -np 4`` (as test_mpi_survivors_ulfm.py). Rank 2 dies, the survivors shrink; with
SECOND=3, rank 3 dies after the survivor watch is armed and the survivors shrink again. Victims die by MODE (exit:
os._exit(0); kill: SIGKILL, as in production). Survivors then run collectives for HOLD seconds and print OK.
Usage: ulfm_replan_probe.py MODE SECOND HOLD
"""

import os
import signal
import sys
import threading
import time

from mpi4py import MPI

from tensorrt_llm._torch.distributed.communicator import HostCollectiveInterrupted, MPIDist
from tensorrt_llm.mapping import Mapping

mode, second, hold = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
world, victim = 4, 2
rank = MPI.COMM_WORLD.Get_rank()


def die() -> None:
    print(f"PROBE rank {rank} dies ({mode})", flush=True)
    if mode == "kill":
        os.kill(os.getpid(), signal.SIGKILL)
    os._exit(0)


dist = MPIDist(Mapping(world_size=world, rank=rank, tp_size=world, moe_ep_size=world, enable_attention_dp=True))
dist.enable_peer_loss_recovery()
dist.tp_allgather(rank)
seen = threading.Event()
watch = dist.watch_peer_failures(seen.set)
dist.barrier()
if rank == victim:
    die()
assert seen.wait(60)
try:
    dist.tp_allgather(rank)
except HostCollectiveInterrupted:
    pass
watch.stop()
dist.revoke()
failed = dist.shrink()
print(f"PROBE rank {rank} first shrink {sorted(failed)}", flush=True)
if second >= 0:
    replan = threading.Event()
    watch = dist.watch_survivor_failures(replan.set)
    if rank == second:
        die()
    assert replan.wait(60)
    watch.stop()
    dist.revoke()
    failed = dist.shrink()
    print(f"PROBE rank {rank} second shrink {sorted(failed)}", flush=True)
dist.install_survivors()
end = time.monotonic() + hold
rounds = 0
while time.monotonic() < end:
    dist.tp_allgather(rank, departed=None)
    rounds += 1
    time.sleep(0.05)
print(f"PROBE_OK rank {rank} mode={mode} second={second} rounds={rounds}", flush=True)

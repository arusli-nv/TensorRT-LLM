"""Why are host collectives slower after a loss? Latency on the same ranks before and after a shrink.

Under ULFM (mpirun with --with-ft ulfm), on the FT transport. Per collective and communicator: 1000 calls after
100 warmup calls; rank 0 prints the median and p99 microseconds.
  world     : a duplicate of MPI_COMM_WORLD, all ranks, before any failure
  split     : the same ranks minus the victim, made by Comm_split before any failure (31 ranks, healthy)
  then the victim exits (SIGKILL itself), the survivors revoke `world` and shrink it:
  shrunk    : the communicator Shrink returns (what recovery serves on)
  split-after: `split` again, after the failure (same members as `shrunk`, built before the failure)
Each communicator also times Get_failed() plus Get_size() of the group, the dead-peer check the executor runs
after every host collective.
Usage (under mpirun): python3 hostcoll_shrink.py LABEL [VICTIM]
"""

import os
import signal
import statistics
import sys
import time

from mpi4py import MPI

CALLS, WARMUP = 1000, 100


def measure(comm, fn) -> tuple:
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(CALLS):
        comm.Barrier()
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e6)
    samples.sort()
    return statistics.median(samples), samples[int(len(samples) * 0.99)]


def failed_check(comm) -> int:
    group = comm.Get_failed()
    size = group.Get_size()
    group.Free()
    return size


def bench(label: str, name: str, comm) -> None:
    rank = comm.Get_rank()
    small = {"rank": rank, "num_tokens": 8, "can_queue": True}
    medium = [{"id": rank * 1000 + i, "tokens": list(range(16))} for i in range(8)]  # ~1-2 KB
    cases = {
        "allgather_small": lambda: comm.allgather(small),
        "allgather_1kb": lambda: comm.allgather(medium),
        "bcast_1kb": lambda: comm.bcast(medium if rank == 0 else None, root=0),
        "gather_1kb": lambda: comm.gather(medium, root=0),
        "allreduce_int": lambda: comm.allreduce(rank, op=MPI.SUM),
        "get_failed": lambda: failed_check(comm),
    }
    for op, fn in cases.items():
        median, p99 = measure(comm, fn)
        if rank == 0:
            print(f"HOSTCOLL label={label} comm={name} ranks={comm.Get_size()} op={op} "
                  f"median_us={median:.1f} p99_us={p99:.1f}", flush=True)


def main(label: str, victim: int) -> None:
    world = MPI.COMM_WORLD.Dup()
    rank = world.Get_rank()
    split = world.Split(0 if rank != victim else MPI.UNDEFINED, rank)
    bench(label, "world", world)
    if split != MPI.COMM_NULL:
        bench(label, "split", split)
    world.Barrier()
    if rank == victim:
        print(f"HOSTCOLL_VICTIM rank={rank} exiting", flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
    # Detect the loss the way recovery does: a collective on `world` fails, then revoke and shrink.
    try:
        while True:
            world.Barrier()
            time.sleep(0.01)
    except MPI.Exception:
        pass
    world.Revoke()
    start = time.perf_counter()
    shrunk = world.Shrink()
    if rank == 0:
        print(f"HOSTCOLL label={label} op=shrink ms={(time.perf_counter() - start) * 1e3:.1f}", flush=True)
    bench(label, "shrunk", shrunk)
    bench(label, "split-after", split)
    shrunk.Barrier()
    if rank == 0:
        print("HOSTCOLL_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2)

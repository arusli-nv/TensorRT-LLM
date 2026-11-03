"""Latency of the executor's host collectives (pickled mpi4py objects) on MPI_COMM_WORLD.

Per collective: 2000 calls after 200 warmup calls; prints the median and p99 microseconds on rank 0.
Usage (under mpirun): python3 hostcoll_bench.py LABEL
"""

import statistics
import sys
import time

from mpi4py import MPI

CALLS, WARMUP = 2000, 200


def measure(fn) -> tuple:
    comm = MPI.COMM_WORLD
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


def main(label: str) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    small = {"rank": rank, "num_tokens": 8, "can_queue": True}
    medium = [{"id": rank * 1000 + i, "tokens": list(range(16))} for i in range(8)]  # ~1-2 KB
    cases = {
        "allgather_small": lambda: comm.allgather(small),
        "allgather_1kb": lambda: comm.allgather(medium),
        "bcast_1kb": lambda: comm.bcast(medium if rank == 0 else None, root=0),
        "gather_1kb": lambda: comm.gather(medium, root=0),
        "allreduce_int": lambda: comm.allreduce(rank, op=MPI.SUM),
    }
    for name, fn in cases.items():
        median, p99 = measure(fn)
        if rank == 0:
            print(f"HOSTCOLL label={label} ranks={comm.Get_size()} op={name} median_us={median:.1f} p99_us={p99:.1f}",
                  flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

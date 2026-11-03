# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark: the cost of the post-success ULFM failure check that
``_interruptible`` (tensorrt_llm/_torch/distributed/communicator.py) adds to
every host collective with peer-loss recovery on.

Run in the serving container, same launch flags as the FT-on job, e.g.:

    mpirun --with-ft ulfm -np 32 [--map-by ...] python3 -I bench_get_failed.py
    mpirun --with-ft ulfm -np 32 python3 -I bench_get_failed.py --native-watch

Prints, on rank 0, median/p90 microseconds of:
  get_failed       comm.Get_failed() + Get_size() + Free()   (raw mpi4py)
  mpi_failed_ranks tensorrt_llm._utils.mpi_failed_ranks(comm) (if importable)
  watch_flag       PeerFailureWatch.failed property read       (--native-watch)
  allgather_i64    1-int64 Allgather on the ERRORS_RETURN dup (what tp_allgather_int64 does)
  allgather_i64+gf the same followed by mpi_failed_ranks
  step_N           N small collectives without / with the check (N = --per-step, default 10)
--native-watch also runs the real native watch thread (5 ms MPI_Test poll) in the
background so the collective timings include its progress-lock contention.
"""
import argparse
import statistics
import time

import mpi4py

mpi4py.rc.thread_level = "multiple"
from mpi4py import MPI  # noqa: E402
import numpy as np  # noqa: E402


def raw_failed(comm):
    group = comm.Get_failed()
    size = group.Get_size()
    group.Free()
    return size


def timeit(fn, iters, comm=None):
    samples = []
    for _ in range(iters):
        if comm is not None:
            comm.Barrier()
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1e3)
    samples.sort()
    return statistics.median(samples), samples[int(0.9 * (len(samples) - 1))]


def report(comm, name, stats):
    rows = comm.gather(stats, root=0)
    if comm.Get_rank() == 0:
        med = max(r[0] for r in rows)
        p90 = max(r[1] for r in rows)
        print(f"{name:22s} max-over-ranks median {med:8.2f} us   p90 {p90:8.2f} us", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--per-step", type=int, default=10)
    parser.add_argument("--native-watch", action="store_true")
    args = parser.parse_args()

    world = MPI.COMM_WORLD
    if MPI.Query_thread() != MPI.THREAD_MULTIPLE and world.Get_rank() == 0:
        print("WARNING: MPI is not THREAD_MULTIPLE; FT runs require it", flush=True)
    comm = world.Dup()
    comm.Set_errhandler(MPI.ERRORS_RETURN)

    try:
        from tensorrt_llm._utils import mpi_failed_ranks
    except ImportError:
        mpi_failed_ranks = None

    watch = None
    if args.native_watch:
        from tensorrt_llm.bindings.internal.runtime import PeerFailureWatch as Native
        watch_comm = comm.Dup()
        watch_comm.Set_errhandler(MPI.ERRORS_RETURN)
        watch = Native(watch_comm.py2f(), 0, lambda error_class: None, 0.005)

    report(comm, "get_failed", timeit(lambda: raw_failed(comm), args.iters))
    if mpi_failed_ranks is not None:
        report(comm, "mpi_failed_ranks", timeit(lambda: mpi_failed_ranks(comm), args.iters))
    if watch is not None:
        report(comm, "watch_flag", timeit(lambda: watch.failed, args.iters))

    send = np.zeros(1, dtype=np.int64)
    recv = np.empty(comm.Get_size(), dtype=np.int64)

    def allgather():
        comm.Allgather([send, MPI.INT64_T], [recv, MPI.INT64_T])

    check = (lambda: mpi_failed_ranks(comm)) if mpi_failed_ranks else (lambda: raw_failed(comm))

    def allgather_checked():
        allgather()
        check()

    coll_iters = max(1, args.iters // 10)
    report(comm, "allgather_i64", timeit(allgather, coll_iters, comm))
    report(comm, "allgather_i64+gf", timeit(allgather_checked, coll_iters, comm))

    def step(checked):
        def run():
            for _ in range(args.per_step):
                allgather_checked() if checked else allgather()
        return run

    step_iters = max(1, coll_iters // args.per_step)
    report(comm, f"step_{args.per_step}_plain", timeit(step(False), step_iters, comm))
    report(comm, f"step_{args.per_step}_checked", timeit(step(True), step_iters, comm))

    if watch is not None:
        watch.stop()
        watch_comm.Free()
    comm.Free()


if __name__ == "__main__":
    main()

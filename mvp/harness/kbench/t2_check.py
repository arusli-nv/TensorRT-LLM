"""T2 transport check, one process per rank under mpirun: the executor's host-collective pattern on a worker thread
(MPI_THREAD_MULTIPLE, main thread waiting), while a second thread polls a never-matched receive as the
peer-failure watch does (MPI_Test every 5 ms on a duplicate of COMM_WORLD). Each round: rank 0 broadcasts a
payload as communicator.safe_broadcast does (an int64 header, then raw-byte Bcasts of at most 4 MB), every rank
joins an int64 allgather on a split communicator, then a pkl5 object allgather of responses, as PyExecutor does
each step. Payloads grow from 256 B (eager) to 32 MB (rendezvous, chunked). T2_CHECK_WATCH=0 drops the watch
thread. Prints CHECK_OK with the slowest round per size, or hangs (the launcher's timeout ends it).
"""

import os
import threading
import time

import numpy as np
from mpi4py import MPI
from mpi4py.util import pkl5

SIZES = [256, 16 << 10, 256 << 10, 4 << 20, 32 << 20]
ROUNDS = 50
CHUNK = 4 << 20


def safe_broadcast(comm, size: int) -> None:
    header = np.array([1, size, -(-size // CHUNK)], dtype=np.int64)
    comm.Bcast([header, MPI.INT64_T], root=0)
    buf = np.zeros(int(header[1]), dtype=np.uint8)
    for start in range(0, buf.size, CHUNK):
        comm.Bcast([buf[start:start + CHUNK], MPI.BYTE], root=0)


def watch(comm, stop: threading.Event) -> None:
    request = comm.Irecv([bytearray(1), MPI.BYTE], source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
    while not stop.is_set():
        assert not request.Test(), "the watch receive completed"
        time.sleep(0.005)
    request.Cancel()
    request.Wait()


def run(result: dict) -> None:
    world = MPI.COMM_WORLD
    rank = world.Get_rank()
    tp = world.Split(0, rank)
    objects = pkl5.Intracomm(world)
    for size in SIZES:
        slowest = 0.0
        for _ in range(ROUNDS):
            start = time.perf_counter()
            safe_broadcast(world, size)
            rows = np.empty((tp.Get_size(), 4), dtype=np.int64)
            tp.Allgather(np.full(4, rank, dtype=np.int64), rows)
            assert rows[:, 0].tolist() == list(range(tp.Get_size()))
            responses = [{"id": rank * 1000 + i, "tokens": list(range(64))} for i in range(max(1, size >> 14))]
            assert len(objects.allgather(responses)) == world.Get_size()
            slowest = max(slowest, time.perf_counter() - start)
        result[size] = slowest


def main() -> None:
    result: dict = {}
    stop = threading.Event()
    watcher = None
    if os.environ.get("T2_CHECK_WATCH", "1") == "1":
        watcher = threading.Thread(target=watch, args=(MPI.COMM_WORLD.Dup(), stop))
        watcher.start()
    worker = threading.Thread(target=run, args=(result,))
    worker.start()
    worker.join()
    stop.set()
    if watcher is not None:
        watcher.join()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("CHECK_OK", MPI.Query_thread() == MPI.THREAD_MULTIPLE, f"watch={watcher is not None}",
              " ".join(f"{size}:{seconds * 1e3:.1f}ms" for size, seconds in result.items()), flush=True)


if __name__ == "__main__":
    main()

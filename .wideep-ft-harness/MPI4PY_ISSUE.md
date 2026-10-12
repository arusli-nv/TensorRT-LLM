# Draft: mpi4py.futures and ULFM process failures (not posted)

**Title:** `MPICommExecutor` cannot shut down (or collect results) after a worker process fails under ULFM

**Environment:** mpi4py 4.0.0, Open MPI 5.0.x with ULFM (`mpirun --with-ft ulfm`), Linux aarch64.

**Scenario:** An application uses `MPICommExecutor(MPI.COMM_WORLD)` to dispatch tasks to all
ranks and tolerates the loss of a worker process using ULFM (`Comm.Revoke`/`Comm.Shrink` on its
own communicators). The surviving ranks keep working, but the futures executor cannot:

1. **Results stop arriving.** `client_exec` receives results with `iprobe(ANY_SOURCE)` /
   `recv(ANY_SOURCE)`. After a process failure, ULFM stops matching any-source receives on the
   executor's intercommunicator until the failure is acknowledged (`MPIX_Comm_ack_failed`). The
   surviving workers block in `server_exec`'s `issend` of their result, and the client's futures
   never complete.
2. **Shutdown blocks.** `client_stop` sends `None` to every worker (including the failed one)
   and waits for all sends; `server_stop` then calls the collective `MPI_Comm_disconnect`, which
   cannot complete with a failed member.

**Expected:** With ULFM available, the executor (a) acknowledges failures on its communicator so
results from live workers keep flowing, (b) fails the futures of tasks assigned to failed workers
(e.g. with `BrokenExecutor` or the MPI error), and (c) stops only live workers and releases the
communicator without a collective that includes failed processes.

**Possible approach:** in `client_exec`, on `ERR_PROC_FAILED_PENDING` from an any-source
operation, call `comm.Ack_failed()`, fail the pending futures of `comm.Get_failed()` ranks, drop
those ranks from the worker set, and continue; in `client_stop`/`server_stop`, skip failed ranks
and use `Free` instead of `Disconnect` when the communicator has failed processes.

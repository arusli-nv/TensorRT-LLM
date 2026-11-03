## PR 10: ULFM survivor communicators (`64eac94`)

**1. major: the new CPU test file is not registered in any test list.**
- `tests/unittest/_torch/distributed/test_mpi_survivors.py` is not in `l0_cpu.yml` or any other file under `tests/integration/test_lists/test-db`.
- Only `test_mpi_survivors_ulfm.py` is listed (`l0_gb200_multi_gpus.yml:25`). The commit message says nothing about this.
- Failure scenario: the fake-communicator tests for `_interruptible`, `_to_logical`, `shrink` and `install_survivors` never run in CI, so regressions in this logic go unnoticed.
- Fix: add it to `l0_cpu.yml`, as PR 11 did for its launch test.

**2. major: no test covers the native `PeerFailureWatch`.**
- `peerFailureWatch.cpp` (216 lines) has the highest-risk logic in the PR:
  - a destructor that detaches the thread;
  - `stop()` returning false when called from the callback;
  - the GIL handling in `report()`;
  - `call_once` joining the thread and cancelling the request.
- I could not find a test of it. The Python tests fake `_native_peer_failure_watch`, as the PR-10 file list shows.
- Failure scenario: a deadlock, a use-after-free on the detach path, or a double `MPI_Cancel` would only appear on a real ULFM job.
- Fix: add a test that drives the native class with a mocked comm, or at least a multi-process test that kills a rank. The commit message does not address this.

**3. major: the extra ULFM query on every host collective is undocumented.**
- `communicator.py:142` runs `mpi_failed_ranks(self._world_comm())` after every successful wrapped collective. That is an `MPI_Comm_get_failed` plus a group alloc and free per `tp_allgather` and `allgather`, which are on the per-iteration attention-DP path.
- It only applies when recovery is enabled. The commit message states the off-by-default behaviour but not this cost.
- Fix: measure it and state the overhead, or rely on the revoke and watch path instead of polling on every call.

**4. minor: `in_host_collective` can be wrong.**
- `communicator.py:~118` sets `in_host_collective = True`, and the `finally` unconditionally sets it to `False`.
- If a wrapped method is ever reached from another wrapped method, or from a second thread, the inner exit clears the outer's flag. Hang detection reads this flag.
- Fix: use a depth counter, or document that wrapped methods are not reentrant and not thread-safe.

**5. minor: exception classification is time-dependent and blocks.**
- `_failure_evidence` busy-polls with `time.sleep(0.001)` for up to 1 s on any non-ULFM exception before re-raising.
- A genuine unpickle or logic error therefore stalls the executor loop for 1 s when recovery is on. The 1 s window is also a heuristic. I do not see a test for the boundary.
- The comment justifies the intent: "a garbled payload can fail to unpickle with any exception type". The stall itself is not mentioned.

**6. minor: communicators leak by design.**
- `install_survivors` and repeated `shrink()` never free the replaced communicators. The docstring says "freeing one with failed members is not guaranteed to be local".
- This is justified in the docstring and bounded by the number of recoveries. It is acceptable.

**7. minor: unsupported layouts have no coverage.**
- `_interruptible` wraps `tp_*`, the world-level methods and `broadcast_int64`.
- `cp_*` and `pp_*` are not wrapped, and `cp_allgather_int64` and `tp_cp_broadcast_int64` use `cp_comm` with a raw `root`.
- `enable_peer_loss_recovery` restricts to pure attention DP, so this is justified. Nothing stops a caller from using `cp_comm` after it is enabled.

**8. minor: the abstract API is wider than what `Distributed` can honour.**
- `departed=` was added to the abstract `allgather`, `tp_allgather` and `tp_gather`. `TorchDist` accepts and ignores it.
- This is justified in the code comments ("Peer-loss recovery runs only on MPIDist").
- Any out-of-tree `Distributed` subclass that does not accept `departed=` will raise `TypeError` once PRs later in this stack pass `departed=` in `py_executor.py` and `cuda_graph_runner.py`. For a non-FT user that is an API break for such subclasses.
- Fix: note it in the release notes, or have callers go through a helper.

**Non-FT behaviour for PR 10**
- `_interruptible` takes the `_host_world is None` early return, so the method runs unchanged.
- The `Distributed.__init__` attributes are additive.
- `mpi_barrier` and `mpi_allgather` are still used when `_host_world` is None.
- No default-path change found. The commit message's claims hold.

**Verdict PR 10: approve with changes.** Blocking items are 1 and 2. Item 3 needs a measured number or a note.

---

## PR 11: keep the MPI session serving and exit cleanly (`699181…`)

**1. major: `_finalize_and_exit` only stops the startup watch before `MPI_Finalize`.**
- `mpi_session.py:~125` calls `recovery.stop_startup_watch()`. The comment says "no MPI call may run while it does".
- The runtime watch that `peer_loss_recovery.py` creates (`self._watch`, around lines 548 and 635) is a native thread polling `MPI_Test`. It is only stopped in the executor's shutdown (line 579) and at lines 954 and 1060.
- The `task_wrapper` degraded-world path (`_leave_degraded_world` → `_finalize_and_exit`) can run while that watch is alive. `MPI_Test` then runs concurrently with `MPI_Finalize`.
- Failure scenario: a survivor's task ends after a loss while the executor watch is still polling. That is undefined behaviour in MPI, and it can crash or hang inside the 30 s grace period.
- Fix: stop every live `PeerFailureWatch` here. Keep a module-level registry rather than just the startup watch.

**2. major: `_mpi_finalize` depends on an assumption about mpi4py.**
- It uses `ctypes.CDLL(mpi4py.MPI.__file__).MPI_Finalize`. That only works if the extension's dependency chain exposes `MPI_Finalize` to `dlsym`.
- The `AttributeError` fallback calls `mpi4py.MPI.Finalize()`, which holds the GIL. This is the case the Timer thread exists to handle. In the fallback the Timer cannot run, so only the native SIGALRM backstop (61 s) is left.
- The commit message states this ("the GIL-free backstop still bounds it"). It is only tested with a mocked `_mpi_finalize`, never the real ctypes path.
- Fix: add a test (or an ULFM integration test) that exercises the real symbol lookup. Alternatively, log loudly when falling back.

**3. major: `_leave_degraded_world(False)` calls `COMM_WORLD.Abort(1)` from `task_wrapper`.**
- `mpi_session.py:~1340` calls it from `task_wrapper`'s `finally` when `_num_failed_ranks()` is non-zero and the task raised.
- Any survivor whose task fails after a loss aborts the whole job. The commit message says "ends the world with MPI_Abort" only in the docstring.
- Failure scenario: a survivor hits a transient error during recovery. This kills every other survivor that was meant to keep serving, which defeats the feature.
- Note that `RuntimeError` is unconditionally `Abort` rather than SIGKILL, so it is inconsistent with `run_worker_rank`, which kills only the failed rank.
- Fix: use the same SIGKILL path as `_exit`, or justify the Abort explicitly.

**4. minor: `BrokenExecutor` blocks the server loop.**
- `_is_departed_peer_failure` busy-waits with `time.sleep(0.01)` for up to 1 s inside `serve()`.
- That loop handles control messages and the stop deadline. A non-departure `BrokenExecutor` stalls it for 1 s per call. This only happens with recovery on.

**5. minor: name shadowing and duplicated logic.**
- `RemoteMpiCommSessionServer._leave_degraded_world` (method) shares its name with the module-level `_leave_degraded_world` it calls. This works but is confusing.
- Failed-rank counting is duplicated across `_num_failed_ranks`, `_is_departed_peer_failure` and the `communicator.py` helpers. There are now two different "1.0 s evidence" constants, `_FAILED_RANK_EVIDENCE_S` here and `_PEER_FAILURE_EVIDENCE_S` in PR 10, with the same wait-loop logic written twice.
- Fix: share one helper.

**6. minor: the shell and Python enable checks can disagree.**
- The bash `peer_loss_recovery_enabled` returns false silently when the authority is set without `TLLM_FAULT_TOLERANCE_MODE=1`.
- The Python function raises `ValueError` in that case (`_utils.py:526`). The same misconfiguration is therefore silent in the launcher and fatal in the worker.
- Fix: validate in the launcher too, or document it.

**7. minor: the `l0_cpu.yml` entry is out of alphabetical order.**
- `test_trtllm_llmapi_launch_exit.py` sits after `test_trtllm_serve_config_schemas.py`.

**8. minor: `mgmn_worker_node.py` now parses `argparse` at module import.**
- It does this before the MPI imports, so any `-h` or unknown flag is silently swallowed by `parse_known_args`.
- This is justified: the launcher passes these flags, and the lock fd must be made non-inheritable before any import can fork.

**Non-FT behaviour for PR 11**
- `run_worker_rank` returns after `serve()` unless recovery is on.
- The launcher `exec` and the OMPI defaults are gated on `TLLM_FAULT_TOLERANCE_MODE=1` plus the authority.
- `exit_fault_tolerant_process` is a no-op.
- The `trtllm-llmapi-launch` tests cover recovery on and off, including the temporary-workspace cleanup.
- The commit message's claims hold, and I found no default-path regression.
- Also: `exec` skips the launcher's EXIT trap, but guards are only created in the rank-0 branch, so nothing leaks (`trtllm-llmapi-launch:364`).

**Verdict PR 11: request changes.** Item 1 (MPI call racing `MPI_Finalize`) and item 3 (job-wide Abort on a survivor failure) need a fix or an explicit justification. Item 2 needs a real-library test.

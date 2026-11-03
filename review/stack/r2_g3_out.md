# Round 2 review: PR 04 (`10.diff`) and PR 05 (`11.diff`)

I checked the current diffs against the tree at `/tmp/claude-161238/wt-stack`.

## PR 04: ULFM survivor communicators (commit 10)

| # | Earlier finding | Status |
|---|---|---|
| 1 | `test_mpi_survivors.py` not in any CI list | **STILL A PROBLEM (major)** |
| 2 | No test of the native `PeerFailureWatch` | **JUSTIFIED**, with a caveat |
| 3 | Per-collective `MPI_Comm_get_failed` cost undocumented | **JUSTIFIED** |
| 4 | `in_host_collective` is not reentrant | Not fixed, minor, not re-raised |
| 5 | Up to 1 s busy-poll on non-ULFM errors | Not fixed, minor, not re-raised |
| 6 | Replaced communicators are never freed | **JUSTIFIED** |
| 7 | `cp_*` and `pp_*` are not wrapped | **JUSTIFIED** |
| 8 | `departed=` widens the abstract API | **JUSTIFIED** |

**1. STILL A PROBLEM (major): the CPU test file is still not registered.**
- Grepping `tests/integration/test_lists` for `test_mpi_survivors` matches only `l0_gb200_multi_gpus.yml:25`, which lists `test_mpi_survivors_ulfm.py`.
- `l0_cpu.yml` has no `_torch/distributed` entry at all.
- The decision was to add it, and the PR text now cites `test_mpi_survivors.py` as test coverage, but the fix is not in the tree.
- Failure scenario: the fake-communicator tests for `_interruptible`, `_to_logical`, `shrink` and `install_survivors` never run in CI.
- Smallest fix: add `unittest/_torch/distributed/test_mpi_survivors.py` to `l0_cpu.yml`.

**2. JUSTIFIED, with a caveat.**
- `test_mpi_survivors_ulfm.py:112` and `:174` drive the real native watch, with a SIGSTOP victim, an abort word and a fired callback. The test is in the GB200 multi-GPU list.
- Those cases cover detection and the abort word.
- The detach-on-destroy path and `stop()` called from the callback are not directly exercised there.
- Not blocking. If you have time, add a small case where the callback calls `stop()`.

**3. JUSTIFIED.** The PR text now states the cost: "mean decode step +0.35% to +0.87%, client throughput within noise". The cost applies only with recovery on, and the numbers are given.

**New issues in PR 04:** none at blocker or major level. I re-read the native watch for lock order and GIL handling and found no new defect:
- `stop()` releases the GIL through `call_guard` before it takes `mMutex` and joins.
- `watch()` unlocks before `report()`.
- The detached-thread path touches no member after `onFailure` returns.

**Verdict PR 04: approve with changes.** The only blocker is registering `test_mpi_survivors.py` in `l0_cpu.yml`.

---

## PR 05: MPI session serving and clean exit (commit 11)

| # | Earlier finding | Status |
|---|---|---|
| 1 | Only the startup watch is stopped before `MPI_Finalize` | **STILL A PROBLEM**, downgraded to minor |
| 2 | Real ctypes `MPI_Finalize` path untested | **STILL A PROBLEM**, minor |
| 3 | `Abort(1)` after a survivor's task fails | **JUSTIFIED** |
| 4 | `BrokenExecutor` blocks the serve loop for up to 1 s | Not fixed, minor, not re-raised |
| 5a | Shadowed method name | **FIXED** |
| 5b | Duplicated failure-evidence logic | Not fixed, minor, not re-raised |
| 6 | Shell and Python enable checks disagree | Not addressed, minor, not re-raised |
| 7 | `l0_cpu.yml` entry out of order | Not fixed, minor, not re-raised |
| 8 | `argparse` at import in `mgmn_worker_node` | **JUSTIFIED** |

**1. STILL A PROBLEM, downgraded from major to minor: the claim and the code disagree.**
- The commit and PR text say "Every live peer-failure watch is stopped before `MPI_Finalize`".
- `mpi_session.py:167-170` stops only `stop_startup_watch()`. There is no registry of watches.
- The only test, `test_finalize_and_exit_stops_the_startup_watch_first`, checks just that watch.
- The race I raised is mostly closed by ordering. The executor's `finally` (`peer_loss_recovery.py:579`) stops the runtime watch before the task returns. `_stop` (`:954`) and the survivor watch (`:1060`) stop theirs on the recovery path. The clean `_finalize_and_exit` path is reached only after the serving loop returned.
- Remaining gap: nothing enforces that ordering for another caller of `_finalize_and_exit`.
- Smallest fix, either of:
  - Reword the description to "the startup watch; the serving loop stops its own watches before it returns".
  - Add a module-level weak registry of live `PeerFailureWatch` objects and stop them all here.

**2. STILL A PROBLEM, minor: the real `ctypes.CDLL(mpi4py.MPI.__file__).MPI_Finalize` lookup has no test.**
- Every test monkeypatches `_mpi_finalize`.
- The `AttributeError` fallback calls `mpi4py.MPI.Finalize()`, which holds the GIL. That is exactly the case the Timer thread cannot cover, so only the 61 s SIGALRM remains.
- The commit message says the backstop still bounds it, and that is true.
- Smallest fix: log a warning on the fallback, and confirm the lookup in the ULFM integration run (PR 19).

**3. JUSTIFIED.**
- The PR text says: "A survivor whose task fails after a loss ends the world with `MPI_Abort`. Errors that leave a single rank unusable are handled earlier: that rank ends itself (PR 16), so the survivors recover from its loss."
- That holds. A task exception that reaches `task_wrapper` means recovery has already given up or the rank is not recoverable.
- The `SIGKILL` path in `run_worker_rank` and the `Abort` here cover different states.

**Non-FT behaviour:**
- All new paths are gated on `peer_loss_recovery_enabled()`.
- The launcher `exec` and the OMPI defaults are gated on mode plus authority.
- `_num_failed_ranks` returns 0 immediately when recovery is off.
- The launcher tests cover recovery on and off.

**New issue in PR 05 (minor):**
- `mgmn_worker_node.py` now imports `tensorrt_llm.llmapi.mpi_session` at module load for every user, not only with FT on. Before, the worker node imported only `mpi4py`.
- That pulls the heavy `tensorrt_llm` import onto the worker's startup path and changes import order, though the worker imports it later anyway.
- I did not find a bug. Note it in the PR text, or import lazily inside `serve()`.

**Verdict PR 05: approve with changes.**
- I no longer treat item 1 as a blocker, because the ordering covers it.
- Before approving, fix the description so it matches the code, or add the registry.
- The two kept trade-offs (`Abort`, argparse at import) are soundly justified.

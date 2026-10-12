[None][feat] keep the MPI session serving after a peer loss

## Description

`trtllm-llmapi-launch` runs a `RemoteMpiCommSessionServer` that dispatches worker tasks through mpi4py.futures on `COMM_WORLD`. When one worker dies, that server forwards the failure to the client as fatal, and its task-end barrier and future waits never finish. Recovery needs the session to stay out of the way while survivors recover, and to end in bounded time later. With this PR, process death does not kill healthy workers, and teardown is bounded. It is PR 12 of the WideEP peer-loss recovery series and depends on PR 6 for `recovery_requested()` and on PR 7 for `is_mpi_peer_failure` and `mpi_failed_ranks`. The degraded mode does nothing until PR 14 recovers the executor.

With recovery enabled, a ULFM peer-failure error, or a `BrokenExecutor` once ULFM reports a failed rank, counts as a departed peer:

- The async task callback logs it and leaves recovery to the survivors instead of failing the client.
- The task-end barrier tolerates it.
- The server stops waiting for task futures once a rank has failed. ULFM stops matching mpi4py.futures' any-source receives, so no further result arrives.
- Later tasks fail fast. A sync task returns the error to the client.
- Shutdown flushes the logs and calls `MPI_Abort(0)`, because mpi4py.futures cannot release a world that lost a member.

Every rank exits after the abort. On multiple nodes, `mpirun` itself can outlive the ranks, so the launcher must stop it once the server exits. The docs in PR 14 state this.

mpi4py.futures threads print `MPI_ERR_PROC_FAILED` tracebacks at the loss and at shutdown. They are harmless. Fixing the futures gap cleanly, upstream in mpi4py or with a TRT-LLM dispatcher, is follow-up work.

FT-off impact: none. Without recovery, every new branch is skipped and the failed-rank query returns 0 without calling MPI. The recovery check reads the environment once per process.

## Test Coverage

New tests in `tests/unittest/llmapi/test_mpi_session.py` with a fake ULFM MPI. `l0_cpu` runs this file with `ISOLATION`.

- `test_server_async_callback_departed_peer`, with and without recovery; a `BrokenExecutor` counts as a departure only once a rank has failed.
- `test_task_end_barrier_tolerates_departed_peer_with_recovery`, with and without recovery.
- `test_server_stops_waiting_for_tasks_after_a_rank_departed`
- `test_degraded_server_fails_sync_tasks_and_ends_the_world_on_shutdown`

The existing tests in the file cover the FT-off session.

## PR Checklist

Please review the following before submitting your PR:
- PR description clearly explains what and why. If using CodeRabbit's summary, please make sure it makes sense.
- PR Follows [TRT-LLM CODING GUIDELINES](https://github.com/NVIDIA/TensorRT-LLM/blob/main/CODING_GUIDELINES.md) to the best of your knowledge.
- Test cases are provided for new code paths (see [test instructions](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tests#1-how-does-the-ci-work))
- If PR introduces API changes, an appropriate PR label is added - either `api-compatible` or `api-breaking`. For `api-breaking`, include `BREAKING` in the PR title.
- Any new dependencies have been scanned for license and vulnerabilities
- [CODEOWNERS](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/CODEOWNERS) updated if ownership changes
- Documentation updated as needed
- Update [tava architecture diagram](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/tava_architecture_diagram.md) if there is a significant design change in PR.
- The reviewers assigned automatically/manually are appropriate for the PR.


- [ ] Please check this after reviewing the above items as appropriate for this PR.

## GitHub Bot Help

To see a list of available CI bot commands, please comment `/bot help`.

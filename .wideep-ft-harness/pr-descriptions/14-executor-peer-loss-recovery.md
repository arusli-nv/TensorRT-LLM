[None][feat] recover the executor from a WideEP peer loss

## Description

When one worker process of a WideEP deployment dies, idle or mid-step, the other ranks now keep serving on the CUDA graphs captured at startup. Nothing reloads, recompiles, recaptures or retunes. Requests on surviving ranks continue with the same tokens, and requests of the dead rank's attention-DP shard fail with a retryable error. This PR ties the series together and implements the executor side of v7 items 1c.4, 1c.4b and 1c.4c, plus 1d.4. It is PR 9 of the series and depends on PRs 1 to 8.

`peer_loss_recovery.py` adds `PeerLossRecovery`, and `py_executor.py` calls it. On each survivor:

1. Detect. A host collective raises `HostCollectiveInterrupted`, or the peer-failure watch fires. The watch sets the step-abort flag, requests a device abort and revokes the host communicators.
2. Discard. A commit gate before `_update_request_states` syncs the step and raises if the flag is set. `revert_prepare_resources` then undoes the step. KV manager V1 releases first-chunk sequences without storing their blocks for reuse and rewinds generation tokens. V2 undoes the scheduler's allocations, and a discarded V2 step never commits blocks for reuse. The ADP padding request retires, and unscheduled requests stay in rank 0's queue.
3. Agree and install. The survivor runs `Shrink`, reports `Stopped` to the authority, answers `Propose` with its placement digest, and on `Install` switches to the survivor communicator and installs the resident placement and the mask.
4. Commit. Survivors resend the response gathers rank 0 did not process, fail the dead rank's requests, restart the watch and resume.

Recovery runs under the executor's `HangDetector`. Workers wait at most 120 s for the authority, set by `TLLM_FAULT_TOLERANCE_RECOVERY_TIMEOUT_S`, and the authority's deadline is 60 s. Any failure takes today's terminal path. `PeerLossRecovery.create()` rejects unsupported configurations at startup; `MPIDist.enable_peer_loss_recovery()` from PR 3 adds the pure attention-DP layout check (PP and CP of 1). It requires the MPI executor, expert TP 1, `DefaultADPRouter`, static EPLB, the non-overlap scheduler, KV manager V1 or V2, and a hang-detection timeout above the recovery timeout. It rejects speculative or guided decoding, beam search, PEFT, a KV connector or transceiver, attention-DP balancing, a conversation manager, sleep/wakeup, LM-head TP in ADP, hybrid or Mamba cache managers. Losing rank 0, a second loss and a rank without replicas fail closed. If a rank's own event loop crashes while the survivors could recover without it, that rank SIGKILLs itself instead of tearing down the job, and the others recover as from any other loss.

The PR adds `docs/source/features/wide-ep-fault-tolerance.md` with an index entry, marked experimental. It documents three limits. The server keeps reporting "healthy" while it recovers and after it lost a rank. On multiple nodes `mpirun` can outlive the ranks after `MPI_Abort`, so the launcher must stop it. Penalties, seeded sampling and multi-token stop words are not rolled back with a discarded step. An earlier version also paused frontend admission during recovery. v7 does not ask for it, so it is dropped, and new requests wait in rank 0's queue.

Evidence on DeepSeek-R1 NVFP4, EP32 on eight 4-GPU GB200 nodes, attention DP with static EPLB spare slots:

- Survivors' requests completed 0.8 s after an idle kill, 1.4 s after a kill mid-decode, and 2.6 s after a kill during MoE dispatch and combine, in which all 31 survivors aborted their in-flight collectives. Tokens matched a same-routing oracle exactly. The oracle's EPLB excludes the victim from the start, because greedy tokens depend on which rank serves each expert. No CUDA graph was captured after the kill.
- With block reuse on, the first token arrived 0.16 s after the kill with correct answers. Exact comparison does not apply there, because the pre-loss cache was filled under the original routing. A unit test checks that a discarded step never becomes a reusable prefix.
- Stress on the final series: five repeated kills during MoE were all exact (2.6 s); victims on rank 1 and on rank 30 of another node were exact; a kill under full chunked-prefill load, including one at the victim's eager prefill forward, completed all 248 survivor requests; EP16 recovered exactly after idle and mid-MoE kills; losing a rank without replicas failed closed with an explicit error on every request.
- An nsys trace shows every decode all-to-all running inside the captured graphs.
- All-active throughput with FT on was 14,488 tok/s against 14,689 tok/s with FT off, 1.4% lower over three trials each.

FT-off impact: `create()` returns `None`, every hook checks for it, `_publish_responses` keeps the enqueue-then-terminate order, and the idle broadcast timeout is unchanged. The FT-off kernels are opcode-identical to main, as PR 4 shows.

## Test Coverage

- `tests/unittest/_torch/executor/test_peer_loss_recovery.py`, new, runs in `l0_cpu` through `unittest/_torch/executor`. It covers the protocol order, failure at every transition, the deadline, the commit gate, discarding a step, deferred terminations, request ownership, response resend and the `create()` checks.
- `tests/unittest/_torch/executor/test_kv_cache_revert.py`, new, round-trips V1 and V2 rollback on real managers and checks that a reverted context is not offered for reuse. It needs a GPU; the GPU pre-merge stages that collect `unittest/_torch/executor` (`l0_h100`, `l0_b300`) run it.
- `tests/integration/defs/wide_ep_ft/test_peer_loss_recovery.py::test_peer_loss_recovery[idle]` and `[inflight]` are added to `l0_gb200_multi_gpus` post-merge. They serve DeepSeek-V3-Lite at EP4 on one GB200 node and SIGKILL rank 2. They need the mpi4py 4.1.2 image from PR 1 and skip without it. `tests/integration/defs/wide_ep_ft/README.md` describes how to run them.

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

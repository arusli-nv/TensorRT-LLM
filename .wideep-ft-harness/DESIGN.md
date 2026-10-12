# WideEP Phase 1 MVP — worker-loss recovery (implementation spec)

Scope: DeepSeek-R1 NVFP4, EP32 on GB200, MPI (Open MPI ULFM via `mpirun --with-ft ulfm`,
mpi4py 4.0), attention DP, expert TP1, static EPLB with resident replicas, NVLinkOneSided
fence path (non-CFT), decode CUDA graphs, non-overlap scheduler, KV cache manager V1.
One recoverable non-rank-0 worker **process** dies — idle or mid-step; the frontend and the
31 survivors keep serving on their **original** CUDA graphs. Survivor-owned requests continue
(the interrupted step is discarded and re-run); requests owned by the dead rank fail with a
retryable error. Everything else (rank-0 loss, a second loss, an uncovered rank, link/device
loss, CFT, KV manager V2, overlap scheduler, spec decode) fails closed (today's terminal path).

Feature gate: `TLLM_FAULT_TOLERANCE_MODE=1` plus `TLLM_FAULT_TOLERANCE_AUTHORITY=<endpoint>`
and `TLLM_FAULT_TOLERANCE_AUTHORITY_KEY=<hex HMAC key>`. Mode alone keeps today's
detection-only behavior; the authority without the mode is rejected. No LlmArgs field for the
MVP (a public config field is the 1d.1 follow-up).

## Invariants
- Logical EP size N, logical rank ids, graphs, slot geometry and tensor addresses never
  change. Only host membership is compacted (a ULFM survivor communicator).
- A failed rank's write lanes (its per-source dispatch slots, flags and counters in every
  survivor's workspace) are never reused or read again (quarantine).
- Routing must exclude a rank before the rank mask does: the kernels silently skip routes to a
  masked rank. Install order: EPLB placement → device routing check → re-arm → mask.
- Membership changes only through the authority's Commit. Local failure evidence only ever
  discards the current step.
- Every recovery wait is bounded; any failure ⇒ existing terminal path.

## Recovery sequence (per survivor)
1. Detect: a host collective raises `HostCollectiveInterrupted` (ULFM `ERR_PROC_FAILED` /
   `ERR_REVOKED`), or the peer-failure watcher thread (unmatched `Irecv(ANY_SOURCE)` on a
   private communicator) sets the step-abort flag and requests a cooperative MoE abort, which
   reaches kernels already spinning on the lost peer through a side stream.
2. Discard: the commit gate (before `_update_request_states`) raises if the flag was set; the
   uncommitted step is rolled back (KV reservations returned via
   `KVCacheManager.revert_prepare_resources`, ADP padding request retired). Requests that were
   received but never scheduled are kept on rank 0 and re-broadcast.
3. Agree: revoke the private host communicator, `Shrink` → every survivor gets the same failed
   set; `Stopped` → authority.
4. Prepare: `Propose` → validate the masked EPLB placement covers every expert →
   `Ready(placement digest)`; the authority requires identical digests.
5. Install: `Install` → survivor communicator, masked EPLB placement, live device-routing
   check, re-arm (each survivor resets its own workspace row and abort record), rank-mask
   install → `Installed`.
6. Commit: `Commit` → resend responses rank 0 did not process, fail the dead rank's requests,
   re-arm the watcher, resume the event loop with the same active requests.

## Components
- MoE data path: cooperative abort + graph-latched descriptor (mask + abort record) + fixed
  per-source dispatch slots (`moeAlltoAllKernels.cu`, `moeAlltoAllOp.cpp`);
  `MoeExecutionState` (`latch`, `request_abort`, `rearm`, `install_mask`);
  `NVLinkOneSided.bind_execution_state` / `reset_local_workspace`.
- MPI adapter (`communicator.py`): `enable_peer_loss_recovery` (private world communicator,
  so revoking it leaves the MPI session's `COMM_WORLD` usable), `revoke`, `shrink`,
  `install_survivors`, `watch_peer_failures`, logical-length gathers.
- EPLB (`moe_load_balancer.py`): `masked_placement_digest`, `covers_loss_of`,
  `reconfigure_mask_only_with_timeout`, `check_routing_excludes`.
- Authority (`executor/recovery/`, torch-free): one CPU process outside worker termination
  groups; Register(recoverable ranks) → admission = ranks every worker can recover from (or
  an explicit list, rejected at startup if uncovered); Stopped → Propose → Ready → Install →
  Installed → Commit; Abort is terminal.
- Executor (`py_executor.py`): recovery loop, commit gate, rollback, request disposition
  (rank 0 tracks request → ADP owner), response redelivery.
- MPI session (`mpi_session.py`): peer-failure errors of a departed worker do not tear down the
  session.

## Known limitations
- Process loss only. Link/fabric/device loss can poison survivor CUDA contexts (no graph
  reuse possible) and must stay terminal until the hardware containment story is known.
- In-flight re-run is idempotent only for state written at fixed positions (paged KV/MLA,
  greedy sampling). Recurrent state (linear attention, Mamba/SSM), sampler RNG/penalties and
  speculative decoding need explicit rollback; such configurations are rejected.
- The dead rank's own requests fail (retryable). Continuing them on a survivor needs a
  re-prefill — follow-up.
- Communicators split from `COMM_WORLD` outside the executor's host collectives (EPLB shared
  memory comm) still contain the departed rank; teardown must not block on them.
- The ULFM launch is a harness script; a supported launcher mode is follow-up (1d.1).
- Draft PRs: see `DRAFT_PRS.md` (#19610 rebase, #15895 abort-ABI alignment).

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WideEP failure behavior

Baseline: DeepSeek-R1-0528 NVFP4, EP32/expert TP1/attention DP/PP1, 288 static slots,
GB200/R580/CUDA 13.4, NVLinkOneSided fence, decode graphs `[1,2,4,8]`.
CFT, rank-mask FT, AlltoAll watchdog, overlap and autotuning were disabled.
Python `cb03225c13`, native build `764bde9d6d`, identical native sources. Five modules
were restored to exclude branch fixes; historical Ray results include them.

## Failure and restart

| Injection | Observed MPI behavior |
|---|---|
| Worker SIGKILL, between requests or after nonfinal streamed output | `srun --mpi=pmix`: Slurm PMIx cancels the step, exit 137, despite `--kill-on-bad-exit=0`. Client receives MPI `RequestError` or is killed first. |
| Fence completion round +2 | Completion-flag timeout → trap/CUDA719 → executor failure/crash escalation → exit 137. Baseline client receives `RequestError`. |
| Synthetic host `MPI_ERR_OTHER` | Raised at `MPIDist._allgather_int64_comm → comm.Allgather`, from `PyExecutor._can_queue`, with rank 2 alive, before native transport. Executor failure → 10-second crash escalation → `MPI_Abort`/exit 137, about 50 seconds after receipt. Client is killed without an error receipt. |

Explicit same-GPU restart follows healthy startup: new processes/CUDA contexts →
weight loading and communication setup → both warmup/capture passes → inference readiness.
Communicators, KV and graphs are new; weights, warmup and capture reload/rerun.
No model run showed automatic restart, survivor agreement or N−1 serving.
SIGKILL victims cannot acknowledge. Healthy shutdown retains UCX disconnect diagnostics.
Cleanup requires no owned steps/processes or GPU compute processes and memory within
64 MiB per GPU of baseline. This proves resource release after teardown, not survivor
quiescence, failed-issuer drain or safe buffer reuse.

Historical Ray kills report `EngineDeadError`; fence faults reach CUDA719. Its local
PyTorch `WORLD` destruction tested Gloo teardown, not physical NCCL failure.

## MPI survivor controls

`mpirun --with-ft ulfm` with PRTE/Slurm daemon steps using `--mpi=none` avoids Slurm's PMIx failure handler.
The tested image supplies Open MPI 5.0.10rc2; controls used ob1/TCP, not UCX-PML.

- A four-rank/two-node CPU control passed 5/5: failed-peer receive errors → revoke releases
  healthy-peer waits → shrink/agreement → successful survivor collectives on logical `[0,1,3]`.
- An eight-rank/two-node `MPI_THREAD_MULTIPLE` control completed two control exchanges while
  seven native `Allreduce` callers remained unfinished. Rank 2 SIGKILL produced seven MPI
  failure errors; survivor agreement/collectives completed on `[0,1,3,4,5,6,7]`.
  Native entry markers precede the underlying MPI call; they do not locate its internal wait.
  The image's mpi4py 3.1.5 lacks revoke/shrink/agree methods; this control used a test-only C bridge.
  Separately, control exchanges and rank-0 broadcast progressed during seven GPU fence waits.
  Releasing the live peer restored exact outputs. Both controls exited and cleaned up normally.
- Killing that held peer in a separate run left seven GPU operations incomplete while
  survivor control exchanges and rank-0 broadcast completed. The fixture then exited;
  it did not drain/replay GPU work or resume inference.
- In the CPU task control, `mpi4py.futures`' manager thread exited on peer failure without
  failing its three pending task futures. A 30-second timeout and cleanup intervention followed.
- Real EP32 under ULFM reached healthy graph inference. Rank 2 SIGKILL reached the client as
  `RequestError(MPI_ERR_PROC_FAILED)` in 0.111 seconds. `RequestBroadcaster`'s native `Bcast`
  failed; TRT-LLM's crash escalation called `MPI_Abort` about 10 seconds later, exit 137.
  Explicit same-GPU restart reached readiness in 211.0 seconds and exited cleanly.
  Its direct-srun death-log assertion failed despite independent death proof; this remains an
  observation, not qualification. Cleanup preceded the extra harness intervention.
- A separate idle-kill run directly verified CUDA operations on all 31 surviving workers
  and their original GPUs. Receipts arrived within 0.607 seconds of the probe request,
  before native abort was observed. The run reported a client error and ended in native
  abort and unforced cleanup.

These controls do not qualify recovery. ULFM agreement is not a membership commit.
Arbitrary network-failure recovery is outside the resident-replica MVP. See [supported ULFM launch/components](https://github.com/open-mpi/ompi/blob/v5.0.x/docs/features/ulfm.rst).

## Optional AlltoAll watchdog

This is the host completion-flag poller, distinct from `RankCrashKillWatchdog`'s fatal escalation.
`TLLM_FAULT_TOLERANCE_MODE=1` supplies shared process-local `EPGroupHealth`, 5-second
detection and 0.1-second polling. Enabled tests force non-CFT `NVLINK_ONE_SIDED` and eager
execution: rank-mask mode currently rejects CUDA graphs.
Default `on_timeout=None` logs detection; it neither publishes membership nor releases GPU waits.

- Holding a live peer before dispatch produced seven callbacks identifying rank 2 in
  5.007–5.013 seconds. After release, all eight completed transfers and CUDA probes.
  Processes/membership were unchanged. This tests silence, not death.
- Fence round +2, with the watchdog enabled, reported downstream stalled **combine** flags on
  every rank at about 5 seconds. Its dispatch poll accepts advanced counters; the native fence
  requires the matching round. These suspects do not identify the injected rank.
- Eight-rank ON/OFF controls reached timeout/CUDA719 at about 291 seconds. ON also enables
  full rank-mask health, so this is not an isolated poller toggle. Enabled finalization hit
  pinned-allocator/CUDA-event errors. Stock and recorder runs failed; cleanup passed.
- Unmodified EP32 FT startup failed when `ConfigurableMoE` deep-copied `EPGroupHealth`'s lock.
  A test-only identity-preserving copy shim enabled healthy eager inference and 32 callbacks.
  Timeout/trap/abort still followed without a client-error receipt; that run failed and
  did not restart. This is not unmodified-model qualification.

Watchdog detection supplies suspicion; it does not implement GPU escape, quiescence or resume.

## Isolated fence proof

An extracted BF16 combine candidate passed 30 cooperative-abort trials with a live held peer,
producing 90 survivor receipts, plus 24 healthy checks. Dependent outputs were suppressed;
same-context CUDA probes and teardown passed.
It covers small eager grids with PDL on/off, not dispatch, graphs or failed-issuer drain.
Ten live-writer tests corrupted prematurely reused mapped storage: terminal survivor work
alone cannot authorize reuse. Prior scratch abort-status and teardown failures are retained.
A separate NVLink atomic-writer SIGKILL left three original survivor contexts usable with
receiver-owned backing retained. A quiet 192-ms observation window does not prove drain.

## Baseline measurements and remaining work

Five baseline batches passed four scenarios: 20 pairs, 40 launches.
Initial readiness was 197.0–284.9 seconds; explicit restart 197.3–255.8 seconds.
Readiness includes imports/container startup through two greedy, fixed-seed results on one
parent monotonic clock. Per-rank loading took 46.9–125.6 seconds initially, 46.9–90.9 on restart.
Initial-engine warmup took about 86–90 seconds, final-engine warmup 6–10; capture 1.78–2.71 per pass.
Capture is inside warmup; intervals overlap.
Communicator init is untimed. Queue time was 13–2691 seconds, reported separately.
Caches were uncontrolled; autotuning was off.
For those batches, retained bounds/maxima/margins in seconds: startup 720/284.9/435.1, fence client
420/292.3/127.7, error-to-step-exit 180/48.7/131.3, cleanup 60/4.8/55.2.
SIGKILL trigger-to-exit maximum was 40.9 seconds; reporting clients 0.113 seconds.
Otherwise the parent enforces the joint 210-second client/teardown deadline.

Targets 16/17 own sole experts and are outside resident-replica recovery.
Actual loaded placement preserves expert coverage after loss of ranks 1, 2, 29, 30 or 31
across all 58 MoE layers.
Job `7775889` verified loader/backend/native placement and all 1,856 resident replica
pairs, including transformed weights and quantization scales. Coverage alone is not admission.
Under fixed logical EP32 and at most 256 inputs per live source, dispatch deduplicates
destinations: 31 sources need at most 7,936 receiver rows within the allocated 8,192.
This source-derived bound does not qualify degraded execution, KV capacity or memory safety;
healthy workloads and idle free-HBM measurements do not prove recovery headroom. Next, review:

| Blocker | Required change or proof |
|---|---|
| Host control | Make broadcasts/cached groups, routing and MPI sessions survivor-aware. Replace world barriers, fixed worker counts and abort policy. Bound pending-request failure and translate compact positions to logical ranks. |
| Fence escape/memory | Host masks cannot release running waits. CUDA719 requires a fresh context. Escape before trap while preserving contexts/barriers and suppressing failed outputs. Prove old peer accesses/writes drained or contained before reuse; `trap` → `return` is insufficient. |
| Commit/graphs | One authority owns admission, coherent installation and resume. Supply replay-visible state and audit captured communication before removing graph guards. |

Baseline batches: `7764417`, `7764675`, `7765082`, `7765083`, `7765084`.
Cleanup hardening passed 74 CPU checks, with three physical skips; GPU recheck `7769273`
passed healthy/restart, streaming kill/restart and controller interruption cleanup in 22.4 seconds.
Controls: model ULFM `7770276`, CUDA probes `7770613`, host error `7769981`, collective
`7770517`, futures `7770162`; watchdog results: `watchdog-control/final-assessment.json`.
Independent control `7777184`, GPU-blocked peer death `7777305`; extracted combine `7777162`.
Atomic-writer death `7777361`. Earlier failed control `7776519` lacks a final original-node
resource receipt; those nodes became unavailable, so that attempt remains unqualified.
Resident coverage and conditional capacity: `prerequisite-proofs/admission/{findings,capacity-assessment}.json`.
All attempts, including failures and source manifests, stay uncommitted under
`.wideep-ft-runs/mpi-characterization/`. CFT and physical device/link loss remain unqualified.
See [README.md](README.md) to run.

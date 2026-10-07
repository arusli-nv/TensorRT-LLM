<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WideEP failure behavior

DeepSeek-R1-0528 NVFP4, EP32/expert TP1/attention DP/PP1, 288 static slots,
GB200/R580/CUDA 13.4, NVLinkOneSided fence, decode graphs `[1,2,4,8]`.
CFT, rank-mask FT, AlltoAll watchdog, overlap and autotuning were disabled.
MPI used Python `cb03225c13`; native build `764bde9d6d` has identical native sources.
Five changed Python modules were restored and hash-checked, excluding branch fixes.
Historical Ray results include branch propagation/actor-death fixes.

| Injection | MPI baseline | Historical Ray | Avoiding teardown would require |
|---|---|---|---|
| Middle worker SIGKILL between requests | PMIx cancels the whole step, exit 137; client receives MPI `RequestError` or dies first. | `EngineDeadError`. | Survivor-preserving launch/control, expert coverage and safe peer-memory lifetime. |
| Middle worker SIGKILL after nonfinal streamed output | Same whole-step cancellation interrupts the pending request; both client outcomes occurred. | `EngineDeadError`. | Idle-kill requirements plus interrupted-transfer and request/KV disposition. |
| Fence round +2 | Completion-flag timeout → trap/CUDA719 → fatal executor error → `RequestError` and rank-crash escalation → exit 137. | Same timeout/trap → `EngineDeadError`. | Escape before the trap, then prove quiescence. Rebuilding communicators cannot repair a CUDA719 context. |
| Local PyTorch `WORLD` destruction | Not ported; MPI uses a different control path. | Gloo failure/rank-crash escalation → `EngineDeadError`. | Qualified communication-resource rebuilding; this diagnostic is not physical NCCL failure. |

`RankCrashKillWatchdog` escalates fatal executor errors; the disabled AlltoAll watchdog
only detects completion-flag failures. Kernel timeouts remain active. Slurm's installed
PMIx error handler kills the step independently of `--kill-on-bad-exit=0`.
A two-task control survived peer death without MPI, but died with MPI/PMIx.
TRT-LLM crash and session-shutdown paths also invoke MPI abort. No model run showed
automatic restart, membership agreement or shrink. A SIGKILL victim cannot acknowledge.
Client triggers do not prove GPU quiescence. Historical Ray CUDA probes passed on
31 peers after kills; after the fence fault, 31 failed and one was missing.

Cleanup verifies no owned steps/processes or GPU compute processes remain and memory
returns within 64 MiB per GPU of baseline. Explicit same-GPU restart follows healthy
startup: new processes/CUDA contexts → weight loading and communication setup
→ both warmup/capture passes → inference readiness. Communicators, KV and graph
executables are new; weights reload, warmup and capture rerun. Static EPLB creates
no shared host expert-weight segments. Healthy shutdown exits 0 and releases resources,
but logs UCX disconnect errors; those diagnostics are retained.

Five consecutive frozen-source batches passed each of the four scenarios: 20 pairs,
40 fresh launches; CPU checks passed 68 tests with three opt-in physical skips.
Readiness measures model-step launch through two greedy, fixed-seed results on one
parent monotonic clock, including imports/container startup. Seconds across all runs:

| Interval | Initial | Restart |
|---|---:|---:|
| Readiness | 197.0 to 284.9 | 197.3 to 255.8 |
| Model loading, per rank | 46.9 to 125.6 | 46.9 to 90.9 |
| Initial-engine warmup | 86.1 to 88.3 | 86.1 to 89.7 |
| Final-engine warmup | 6.3 to 9.2 | 6.3 to 10.0 |
| Initial-engine graph capture | 1.80 to 2.09 | 1.78 to 2.09 |
| Final-engine graph capture | 1.80 to 2.71 | 1.83 to 2.04 |

Native intervals overlap; capture is inside warmup. Communicator init is untimed.
Allocation queue time was 13 to 2691 s, reported separately. Caches are uncontrolled;
autotuning was off. This measures process startup, not cold caches or degraded correctness.
SIGKILL trigger-to-step-exit took at most 40.9 s; fence client error took at most 292.3 s.
Retained bounds/maxima/margins in seconds: startup 720/284.9/435.1, fence client
420/292.3/127.7, error-to-step-exit 180/48.7/131.3, normal cleanup 60/4.8/55.2.
Kill clients that reported errors took at most 0.113 s within their 30 s bound.
Without an error receipt, the parent enforces the joint 210 s post-readiness deadline.

Targets 16/17 own sole expert copies and must fail closed under this placement.
Configured expert coverage remains complete after loss of non-root ranks
1, 2, 29, 30 or 31 across all 58 MoE layers; resident-weight/capacity admission is unqualified.
A separate four-rank/two-node CPU control passed 5/5, without retries, using
unmodified image MPI/UCX packages on compatible native Arm64 nodes. Open MPI
5.0.10rc2 used `mpirun --with-ft ulfm`, PRTE/Slurm `--mpi=none` daemon launch and
ob1/TCP. After rank 2 SIGKILL, a failed-peer receive returned an error; revoke
released healthy-peer receives. Shrink, agreement, all-gather and all-reduce
succeeded on logical ranks `[0,1,3]`; rank processes and launcher steps disappeared.
This qualifies that single-threaded CPU control, not TRT-LLM, CUDA, UCX-PML or
in-flight collective failure. ULFM's boolean agreement is not a recovery authority/commit protocol.
See [ULFM's supported launch/components](https://github.com/open-mpi/ompi/blob/v5.0.x/docs/features/ulfm.rst).

| Remaining recovery blocker | Required change or proof |
|---|---|
| `RequestBroadcaster` / `MPIDist` / `ADPRouter` | Replace fixed-world broadcasts and cached TP control groups with survivor membership; translate compact collective positions to fixed logical IDs. Expert TP1 does not make the ADP host group single-rank. |
| MPI session / crash watchdog | Replace pre/post-task world barriers, fixed worker counts and abort escalation with bounded survivor-safe control. Disabling aborts alone permits hangs. |
| Fence kernels | Launch masks are copied by value and checked before waits. Updating a host mask cannot release running waits. Cooperative escape must preserve barrier progress, contexts and suppress partial outputs; replacing `trap` with `return` alone is unsafe. |
| Peer memory | Prove failed-peer writes/accesses are drained or safely contained before reusing payloads/counters. Timeout or membership removal is insufficient. |
| Graphs | Rank-mask capture is currently rejected. Audit captured communication resources and provide replay-visible state before claiming unchanged-graph recovery. |

Next, review the survivor-control and fence-abort contracts before implementing recovery.
CFT drain, late-write safety and physical device/link loss remain unqualified.

Physical batches qualified the harness at `0420dc9213`. Later cleanup hardening passed
74 CPU checks; three physical tests skipped, GPU rerun pending.
Earlier failures and before/after tests are retained; final-source interruption cleanup
passed in 26.5 s. Final batches: `7764417`, `7764675`,
`7765082`, `7765083`, `7765084`; CPU ULFM pilot `7767258`, qualification `7767277`.
No hidden retries. Raw attempts stay uncommitted under
`.wideep-ft-runs/mpi-characterization/`; historical Ray evidence is under
`.wideep-ft-runs/lean-characterization/`. See [README.md](README.md) to run.

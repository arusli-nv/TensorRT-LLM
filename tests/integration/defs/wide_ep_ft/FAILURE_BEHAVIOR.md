<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WideEP failure behavior

DeepSeek-R1-0528 NVFP4, EP32/expert TP1/attention DP/PP1, 288 static slots,
GB200/R580/CUDA 13.4, NVLinkOneSided fence, decode graphs `[1,2,4,8]`.
CFT, rank-mask FT, the opt-in AlltoAll completion-flag watchdog, overlap and
autotuning were disabled. Target rank 16 owns sole expert copies; its loss is outside the resident-replica MVP envelope.
These tests characterize failure and explicit restart, without N−1 recovery.

**MPI baseline:** Python `cb03225c13`; native build `764bde9d6d` has identical
native sources. Five changed Python modules were restored and hash-checked.
Test-only instrumentation recorded workers and issued faults; branch fixes were absent.
Launch: Slurm/PMIx → `trtllm-llmapi-launch` → LLM API + MPI workers.

| Injection | Failure path | Fresh same-GPU restart readiness |
|---|---|---|
| SIGKILL between requests | Verified rank-16 death → PMIx error `-61` → whole step canceled, exit 137; client terminated before reporting an error. | 166.3 s |
| SIGKILL after nonfinal streamed output | Verified rank-16 death → same whole-step cancellation; no client-error receipt. | 165.6 s |
| Fence round +2 | Dispatch flag timeout → device trap/CUDA failure 719 → `RequestError` at 291.5 s → `RankCrashKillWatchdog` escalation → step exit 137. | 189.2 s |

`--kill-on-bad-exit=0` did not preserve the MPI step. An isolated two-task control
confirmed the surviving task completed without MPI, but was terminated with MPI/PMIx.
This establishes behavior of the tested launcher stack, not every MPI implementation.
All three cases released GPU compute contexts; independent host scans found no
processes from the failed phases. Restarts used 32 new process identities on the same
hostname/GPU UUID mapping. One streaming attempt stopped on a probe self-counting bug;
it was diagnosed and retained, not counted as a complete pass. GPU memory returned
to 0 MiB after both phases of the completed streaming and fence pairs. These are
single characterization passes, not repeated qualification.

**Startup/restart:** workers → weight loading/communication → warmup/capture →
healthy inference. Explicit restart follows the same construction path: new CUDA
contexts, communicators, KV and graph executables; weights reload and both warmup/
capture passes rerun. No automatic restart, communicator shrink, survivor agreement
or graph reuse occurred. A SIGKILL victim cannot acknowledge its death.

MPI initial / clean-restart readiness: 185.5 / 156.9 s; online EPLB also passed (311.3 s).
Rank-0 phase seconds for the idle-kill deployment / its explicit restart:

| Marker | Initial | Restart |
|---|---:|---:|
| Model loading | 69.0 | 58.2 |
| Initial/final warmup | 87.8 / 7.5 | 86.7 / 6.7 |
| Initial/final graph capture | 1.92 / 1.98 | 2.01 / 2.00 |

Readiness here measures LLM construction through two greedy, fixed-seed inference
checks; it excludes imports, container startup and allocation queue time (4 s
for the healthy MPI job, from Slurm). It uses one client monotonic clock. Phase timers are rank-local and overlap; capture is
included in warmup. Communicator init has no separate timer. Filesystem/compiler
caches were uncontrolled; the checks are not an exact-token correctness reference.

**Historical Ray runs:** `764bde9d6d` included branch fatal-error propagation and
actor-death monitoring. Kill between requests / after streamed output delivered
`EngineDeadError` in 0.164 / 0.156 s. Local WORLD destruction reached the rank-crash
watchdog, peer Gloo failure and `EngineDeadError` (10.3 s). Fence +2 reached kernel
completion-flag timeouts, CUDA failure and `EngineDeadError` (291.5 s).
All required explicit fresh restart. CUDA probes passed on 31 peers after kills;
after the fence fault, 31 failed and one was missing. This did not prove safe drain.

Raw evidence is uncommitted under `.wideep-ft-runs/mpi-characterization/`
(jobs `7759972`, `7760555`, `7760788`, `7760904`, `7761065`) and
`.wideep-ft-runs/lean-characterization/physical-{7738662,7739144,7740233}/`.
MPI packaging is in progress; the committed runner still uses Ray. No physical
link/device loss or survivor-preserving transfer recovery has been qualified.

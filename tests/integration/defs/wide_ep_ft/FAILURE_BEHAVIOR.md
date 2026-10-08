<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WideEP failure behavior

DeepSeek-R1-0528 NVFP4, GB200/R580, EP32/expert-TP1/attention-DP/PP1, 288 static
slots, decode graphs `[1,2,4,8]`, NVLinkOneSided fence. CFT, rank-mask FT, optional
AlltoAll watchdog, overlap and autotuning are off. Open MPI 5.0.10rc2, mpi4py 4.0.0.
Requests use the local LLM API, not HTTP. Build/source hashes are in the raw evidence.

## Scenarios and outcomes

Every scenario first completes two healthy requests. Faults target rank 2.

| Scenario | Injection | Observed failure path |
|---|---|---|
| Healthy | No fault; close the model, verify cleanup, start it again | Both launches serve successfully. |
| Idle kill | SIGKILL between requests; then submit a new request | MPI peer failure → client `RequestError` → rank-crash escalation → `MPI_Abort`. |
| Streaming kill | SIGKILL after the first nonfinal output of a 256-token stream | Same fatal path; the interrupted stream is not resumed. This targets a client boundary, not an exact GPU phase. |
| Fence mismatch | Between requests, increment the local fence round by two; then submit a request | Completion wait timeout → trap/CUDA719 → executor failure → whole-job termination and client `RequestError`. This is synthetic, not device/link loss. |

PRTE daemons use `srun --mpi=none` to avoid Slurm's PMIx cancellation path; ULFM
reports MPI peer failure. Ordinary PMIx launch can cancel the step before the client
reports. Faulted steps exit 137. `RankCrashKillWatchdog` enforces shutdown; the AlltoAll flag
watchdog is separate. Dead workers cannot acknowledge. There is no committed survivor
agreement or automatic restart.

## What the times measure

After verified cleanup, the harness starts fresh workers on the same GPU UUIDs.
Weights reload; communicator setup, warmup and capture rerun. CUDA contexts, KV and
graphs are recreated. These measure fresh restart, not N−1 recovery.

Baseline `f3bb2c78c1` passed five consecutive trials per scenario, 40 model launches.
Values below are observed minima and maxima across those five trials.

| Healthy launch | Launch → readiness |
|---|---:|
| Initial process start | 255.2 to 287.5 s |
| New process start after clean shutdown | 222.5 to 257.1 s |

| Fault | Client fault request → error | Injection receipt → cleanup | Injection receipt → first restarted result |
|---|---:|---:|---:|
| Idle kill | 0.064 to 0.116 s | 44.7 to 65.8 s | 276.2 to 305.9 s |
| Streaming kill | 0.057 to 0.066 s | 45.3 to 80.0 s | 268.5 to 311.9 s |
| Fence mismatch | 291.7 to 293.1 s | 335.6 to 343.6 s | 569.8 to 589.3 s |

Readiness means two complete greedy, fixed-seed requests with checked answers.
The first result is a complete generation, not a first streamed token. Client error
times use the client clock. Cleanup/restart times use one controller's file-receipt
clock. Kill receipts contain intent written before SIGKILL; fence receipts confirm
completed mutation. Polling and independent probes add delay, so these are not exact
failure/detection timestamps. Queue time is separate. Filesystem caches are uncontrolled.

| Healthy startup phase, rank-local | Initial launch | Clean restart |
|---|---:|---:|
| Weight loading | 62.1 to 69.7 s | 48.6 to 64.6 s |
| Initial-engine warmup | 97.7 to 101.8 s | 93.5 to 99.4 s |
| Final-engine warmup | 6.8 to 7.6 s | 6.9 to 7.9 s |

Graph capture totals 1.9 to 2.3 s per engine, inside warmup. Do not sum overlapping
intervals. Process/import/communicator phases lack separate trustworthy markers.
Earlier infrastructure failures remain FAIL. Their artifacts and calibrated test
deadlines are retained; they are not included as passing qualification trials.

## What remains for recovery

- Idle kill left 31 survivor contexts usable before abort. ULFM survivor collectives
  and independent rank-0 control passed isolated tests; neither proves GPU drain/replay.
- Resident copies cover every expert after losing ranks `[1,2,29,30,31]`. Coverage
  alone is not capacity/safety admission.
- Same-graph recovery needs non-trapping escape, proven quiescence, valid mappings,
  replay-visible membership/generation, repaired host collectives and in-place EPLB.
  Streaming also needs safe request/KV disposition. Rank-mask graphs are rejected.
- The optional watchdog did not prevent CUDA719, which requires fresh contexts.
  Live-peer combine escape passed isolated controls; dispatch/dead-issuer safety remains open.

The separate RPC propagation/socket-cleanup fixes do not prevent MPI abort. MPI KV
compatibility, CFT and device/link loss remain unqualified. No recovery is implemented.

Run commands: [README.md](README.md). Raw attempts, including failures, stay outside
Git under `.wideep-ft-runs/mpi-characterization/`: `prerequisite-proofs/evidence-index.json`
and `mpi4py4-qualification/{qualification-summary,calibrated-bounds,evidence-index}.json`.

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

Graph capture totals 1.9 to 2.3 s per engine, inside warmup. An EP32 trace confirmed
four profiling captures, then four serving captures per rank. Each active variant
contains 58 dispatch/combine pairs through one shared workspace/coordinator.
Do not sum overlapping intervals. Process/import/communicator phases lack separate trustworthy markers.
Earlier infrastructure failures remain FAIL. Their artifacts and calibrated test
deadlines are retained; they are not included as passing qualification trials.
After cleanup changes, `6f00fb52c7` passed one further trial per scenario. The current
native/Python fixes passed another four-case check, including eight fresh launches.
A separate CPU diagnostic step failed; it remains recorded as FAIL.

## Progress toward recovery

No N−1 serving is implemented. The isolated controls establish the following.

| Area | Proven scope and remaining gap |
|---|---|
| Process survival | Idle kill left 31 usable contexts. Rank-0 control and ULFM survivor collectives passed; TRT-LLM still enforces fatal teardown. |
| Fence escape | Native abort escaped dispatch/combine waits with zero-token calls and PDL on/off. Sticky status blocked later preparation. Each rank's interrupted EP32 replay and auxiliary streams completed with usable CUDA; failed outputs were withheld before sampling. All workers were alive. |
| Detection | The shared AlltoAll coordinator armed a fixed two-call graph across healthy replays, detected the held rank and triggered native abort. Pinned H2D cancellation progressed against a helper occupying every SM's resident thread slots. Four CPU callback-close race cases passed. Serving-loop and DMA-completion integration remain unfinished. |
| Peer memory | A two-second SM-read operation over retained exporter memory remained pending after verified process death, finished with exact data, and left CUDA usable. This was one GB200 trial, separate from native combine. Active-writer observations establish neither drain nor safe reuse. |
| Buffer isolation | EP4 NVFP4 reused the original workspace, graphs and addresses after rank-2 loss, then idle rank-1 loss, in both PDL modes; 60 exact survivor replay receipts. Lost-source payload/counter/flag/stats bytes stayed unchanged. The first kill followed completed dispatch; the second was GPU-idle. Routing exercised templates 0/1, not full-model coverage. Native combine during death remains unqualified. |
| Resident experts | Ranks `[1,2,29,30,31]` preserve coverage; survivor HBM/token capacity and placement publication remain unqualified. |
| Host communication | After real death, compact survivor votes and fixed logical token metadata passed; zero-filled votes failed. Installed futures can remain pending. A CPU cleanup prototype settled three futures once and finalized survivors normally. Send/receive error paths and serving continuation remain unqualified. |

Healthy native EP32 answer checks passed, but greedy tokens differed from historical
runs. Full-model exact parity and post-failure inference remain unqualified.
Rank-mask serving graphs remain rejected; CUDA719 requires fresh contexts.

The current candidate retains the original allocation and permanently retires lost
source slots. Allocation layout, logical N and imported mappings stay fixed. Stop
issuance and require every live issuer's full CUDA join and acknowledgment before
reusing live slots. A source-only simplification
removes bank rebasing and shrinks the descriptor from 48 to 40 bytes. Its build and
GPU qualification remain separate from the passed six-word component controls.
Matched component runs saved 4.55 to 4.79 µs with a 13-net-line gather-copy change.
The workspace stayed 30 MiB/GPU, versus 60 MiB for two banks; two captured buckets
used about 194 KiB of private gather storage. These are component costs, not serving performance.

To resume, integrate isolated or provably quiescent storage, a replay-visible
descriptor, agreed generation installation, survivor host control, quiesced EPLB,
and request/KV disposition. RPC propagation/socket-cleanup fixes do not prevent
MPI abort. A separate fix skips absent sampling results after handled execution
errors. MPI KV compatibility, CFT and device/link loss remain unqualified.

Run commands: [README.md](README.md). Raw attempts, including failures, stay outside
Git under `.wideep-ft-runs/mpi-characterization/`: `prerequisite-proofs/evidence-index.json`
and `mpi4py4-qualification/{qualification-summary,calibrated-bounds,evidence-index}.json`.
Graph/backing controls and failed attempts are indexed in
`.wideep-ft-runs/backend-safety/assessment.json`. The latest verified append archive is
`.wideep-ft-runs/backend-sequential-followup-20261009.tar.gz`; earlier archives
and all failed attempts remain intact. Every archived file matched its frozen hash.

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

Correct N−1 full-model serving is not yet proven. Components and healthy runs establish:

| Area | Proven scope and remaining gap |
|---|---|
| Process survival | Idle kill left 31 usable contexts. Rank-0 control and ULFM survivor collectives passed; TRT-LLM still enforces fatal teardown. |
| Fence escape | Cooperative abort escaped native waits and interrupted EP32 replays, retained usable contexts and withheld failed outputs. Workers stayed alive; real in-flight death is unproven. |
| Detection | The AlltoAll flag watchdog detected a held peer and triggered abort in graph controls. Cancellation progressed with occupied SMs. Job `7853768` separately passed real abort publication, same-graph cancellation and watchdog thread/copy retirement on one GPU. Serving-loop integration remains unqualified. |
| Peer memory | Retained exporter reads completed after real process death. This does not prove failed-issuer drain or safe reuse. |
| Buffer isolation | EP4 exporter-loss controls retained mappings and graphs. Job `7861416` then passed exact survivor outputs during live remote writes into quarantined source lanes, with original graphs. Stale combine routing still read a masked peer's output. Component evidence only; no drain or full-model recovery proof. |
| Resident experts | All 32 placement snapshots agree across 58 layers; ranks `[1,2,29,30,31]` preserve coverage. Recovery admission remains empty until capacity, GPU safety and placement installation are qualified. Static EPLB skips its iteration guard, so that guard cannot establish GPU quiescence. |
| Host communication | Real death released an actual TRT-LLM allgather with peer-failure errors. Survivor broadcasts preserved logical ranks. Persistent tasks survived a separate kill, but the stock pool manager died with futures pending. Actual serving-process retention is unproven. |
| Healthy full model | Normal-owner EP32 job `7857880` passed matched ordinary/descriptor batches `[1,2,4,8]`, token parity, graph/address stability and cleanup. Earlier job `7849756` also matched paired logits; its 128 serving graphs passed the dependency inventory with no captured conventional NCCL collective. Neither run proves recovery. |

The fused comparison `7849213` failed token parity; ordinary repeats also had logit
variation. The cause is unresolved. Unfused finalization is the current qualification
choice, with about 952 MiB more reserved memory per rank in these runs. That comparison
used different allocations; serving overhead is not measured. Rank-mask graphs in the
stock path remain rejected, and CUDA719 requires fresh contexts.

The candidate keeps logical N, allocation layout, mappings and expert-output addresses
fixed. It uses private contiguous gather and a 40-byte mask/abort descriptor, permanently
retiring lost-source slots. All live issuers must stop and join before live slots are
reused. The 48 native unit cases passed across both PDL modes. EP4 component overhead
was 13.2–15.3 µs, or 15.5–19.0%, with the same 30 MiB workspace and about 194 KiB extra
Torch allocation. These are component costs, not serving costs.

Next is the actual idle-kill test with V1 KV, no prefix reuse and one fresh request.
Job `7851671` passed all-peer access across 32 GPUs and ran two healthy requests.
A resource probe failed on unreadable launcher-process metadata before injection;
strict cleanup passed. No worker kill or survivor request has passed this test.

Production integration must put session loss handling, consumed requests and model
state under their existing owners. An external CPU authority grants resume only after
all survivor readiness and installation acknowledgments. Retained mappings also need
a lifecycle guard against explicit checkpoint/unmap operations; strong references alone
do not prevent them. Job `7853102` passed 80 focused ownership tests through real ARM
imports, with strict cleanup. Model-owner and recovery-control integration remain
unqualified. Three normal-owner attempts exposed config copying, startup/per-forward
buffer selection, and source-slot return-layout errors; their failures remain recorded.
The corrected run `7857880` passed. It also exposed duplicate stream wrappers causing
redundant join events; the stream-identity correction is not yet qualified. Job `7858396`
passed actual native V1 sequence inventory, ordinary release and transfer-stream joins
with a separately built binding. Full-model scheduler STOP remains unproven.
Experimental file polling, frame inspection and import hooks are not
production interfaces. In-flight death, failed-issuer safety, CFT and device/link
loss remain unqualified. RPC propagation/cleanup fixes do not prevent MPI abort.

Run commands: [README.md](README.md). Raw attempts, including failures, stay outside
Git under `.wideep-ft-runs/mpi-characterization/`: `prerequisite-proofs/evidence-index.json`
and `mpi4py4-qualification/{qualification-summary,calibrated-bounds,evidence-index}.json`.
Graph/backing controls and failed attempts are indexed in
`.wideep-ft-runs/backend-safety/assessment.json`. The latest verified append archive is
`.wideep-ft-runs/backend-component-checkpoint-20261009.tar.gz`; earlier archives
and all failed attempts remain intact. Every archived file matched its frozen hash.

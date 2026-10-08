<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WideEP failure behavior

DeepSeek-R1-0528 NVFP4, GB200/R580, EP32/TP1/attention-DP/PP1, 288 static slots,
NVLinkOneSided fence, decode graphs `[1,2,4,8]`. CFT, rank-mask FT, optional AlltoAll
watchdog, overlap and autotuning are off. Native build `ef1b238cff`; Python refreshed
to main `72688933d5` plus branch fixes. No intervening native changes. CUDA 13.4,
Open MPI 5.0.10rc2, mpi4py 4.0.0; MPI uses native IPC.

## Failure and restart paths

| Rank-2 injection | What happens |
|---|---|
| Idle SIGKILL | ULFM reports peer failure → client `RequestError` → rank-crash escalation invokes `MPI_Abort`. Ordinary PMIx launch instead lets Slurm cancel the step, sometimes killing the client before reporting. |
| SIGKILL after nonfinal streamed output | Same fatal host/process path; GPU work may also wait on the dead peer. Interrupted inference is not resumed. |
| Fence round incremented by two | Round disagreement → completion wait timeout → device trap/CUDA719 → executor failure/crash escalation → whole-job termination; client `RequestError`. Synthetic kernel-state fault, not device/link loss. |

Faulted steps exit 137. The dead worker cannot acknowledge. There is no committed
survivor agreement or automatic serving restart. `RankCrashKillWatchdog` enforces
fatal shutdown; it is separate from the optional AlltoAll completion-flag watchdog.

Explicit restart follows the healthy path on the same GPU UUIDs: new workers/CUDA
contexts → reload weights/setup communication → initial/final warmup and capture →
serve. Communicators, KV and graphs are recreated; warmup reruns. Faster startup may
reflect uncontrolled filesystem caches, not graph/state reuse. Before restart,
cleanup requires no owned steps/processes/zombies or GPU compute processes and
memory within 64 MiB/GPU of baseline. This does not prove in-place quiescence.

## Measurements

Five consecutive passes per scenario, including healthy restart, total 40 launches.
Earlier infrastructure failures remain FAIL; affected scenarios restarted
qualification after investigation. All attempts are retained, with no hidden retries.

| Scenario | Client fault request → error | Injection receipt → cleanup | → fresh first response | → fresh readiness |
|---|---:|---:|---:|---:|
| Idle kill | 0.064–0.116 s | 44.7–65.8 s | 276.2–305.9 s | 276.8–306.6 s |
| Streaming kill | 0.057–0.066 s | 45.3–80.0 s | 268.5–311.9 s | 269.3–312.5 s |
| Fence mismatch | 291.7–293.1 s | 335.6–343.6 s | 569.8–589.3 s | 570.4–590.0 s |

Healthy initial/restart readiness: 255.2–287.5/222.5–257.1 s. Readiness requires two
complete greedy, fixed-seed results. Client intervals use a client-local monotonic
clock; other intervals use one controller's receipt clock, including probe overhead.
These are not exact GPU failure/detection times. Queue time is separate.

Healthy rank-local initial/restart markers: loading 62.1–69.7/48.6–64.6 s;
initial-engine warmup 97.7–101.8/93.5–99.4 s; final warmup 6.8–7.6/6.9–7.9 s.
Capture takes 1.9–2.3 s per pass inside warmup. Do not sum overlapping intervals.
Process/import/communicator phases lack qualified markers. Result-wait deadlines
are 30 s for kills and 420 s for fence mismatch, after a separate 15-s injection
receipt wait. Client intervals above include that wait. Bounds/maxima/margins,
including valid earlier phases: startup 720/302/418 s; error-to-exit 180/11/169 s;
clean shutdown-to-exit 180/48.1/131.9 s; exit-to-cleanup 180/81.2/98.8 s.
Cleanup includes Pyxis deletion.

## Evidence and remaining MVP gates

- Two-node ULFM revoke/shrink/agree and survivor collectives passed. mpi4py 4.0 exposes
  them directly. Healthy futures and CUDA-buffer communication passed; full MPI KV
  transceiver remains unqualified. Death still leaves three futures pending.
- Idle kill left 31 survivor CUDA contexts usable before abort. Independent rank-0
  control progressed during MPI/GPU waits. Neither proves drain or graph replay.
- Loaded replicas cover all 58 MoE layers after loss of ranks `[1,2,29,30,31]`.
  Rank 16/17 loss removes experts. Coverage alone is not capacity/safety admission.
- Optional AlltoAll watchdog detected eager held-peer stalls near 5 s; fence mismatch
  produced downstream stalls without culprit attribution or preventing the trap.
  Rank-mask graphs remain rejected; EP32 FT startup also hit a deepcopy-lock bug.
- Live-peer combine-abort controls passed; dispatch, graph replay and dead-issuer
  safety remain open. Live writers contaminated prematurely reused storage.
  A synthetic live-worker host `Allgather` error also caused fatal shutdown.

Same-graph recovery remains a target. Idle loss needs valid contexts/peer mappings,
fixed logical ranks/addresses/shapes, repaired host control, replay-visible membership
and generation, and in-place EPLB routing. Streaming also needs non-trapping escape,
proven quiescence and safe request/KV handling. Captured masks and affected collectives
need changes; Python checks do not run during replay. CUDA719 requires fresh contexts.

Two independent PR-ready fixes fail pending RPC requests/preserve fatal causes and
bound socket close with zero linger. Close may discard undelivered sends. Neither
repairs MPI communication or suppresses `MPI_Abort`. No recovery is implemented.

Run commands: [README.md](README.md). Raw evidence stays uncommitted under
`.wideep-ft-runs/mpi-characterization/`: `prerequisite-proofs/evidence-index.json` and
`mpi4py4-qualification/{qualification-summary,calibrated-bounds,evidence-index}.json`.
Runtime/source manifests retain exact hashes. CFT and physical device/link loss are unqualified.

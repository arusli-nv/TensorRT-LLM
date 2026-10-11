<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WideEP recovery handoff · 2026-10-11

Development stopped at the user's request. **No post-loss serving or current EP32 STOP pass.**
Branch `wideep-ft`; HEAD `1a8a89c6fb53d1cb1fa52f52310a970c5bf06f4c`.
No new commit, rebase or push. The composed implementation remains intentionally
uncommitted until actual STOP and installation qualify. Sources are safely archived,
not promoted as a passing MVP.

Paths below use `B=.wideep-ft-runs/backend-safety` and
`P=$B/admission-readiness/normal-state-integration`, relative to repository root.

## Code and canonical package

Modified production files, relative to repository root:

```text
cpp/tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.{cu,h}
cpp/tensorrt_llm/thop/moe/communication/moeAlltoAllOp.cpp
cpp/tensorrt_llm/nanobind/batch_manager/kvCacheManager.cpp
tensorrt_llm/_mnnvl_utils.py
tensorrt_llm/_torch/{alltoall_watchdog.py,mnnvl_alltoall_workspace.py}
tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py
tensorrt_llm/_torch/distributed/communicator.py
tensorrt_llm/_torch/moe/fused_moe/{configurable_moe.py,ep_group_health.py,wide_ep_ft.py}
tensorrt_llm/_torch/moe/fused_moe/communication/{communication_factory.py,nvlink_one_sided.py}
tensorrt_llm/_torch/pyexecutor/{cuda_graph_runner.py,executor_request_queue.py,model_engine.py,py_executor.py,request_utils.py}
tensorrt_llm/_torch/pyexecutor/scheduler/adp_router.py
tensorrt_llm/executor/{base_worker.py,proxy.py,worker_process_monitor.py}
tensorrt_llm/llmapi/mpi_session.py
```

Untracked production: `tensorrt_llm/executor/recovery.py` and
`tensorrt_llm/_torch/moe/fused_moe/communication/{execution_state.py,execution_ticket.py}`.
Associated tests and documents also remain uncommitted. Exact statuses/hashes for
all 45 checkpointed files: `B/handoff-20261011/files.json`; preserved source archive
`source-checkpoint.tar.gz` and tracked diff `tracked-changes.patch` in that directory.
Archive SHA256 `cbdfb1b912c1f1981414e61f58644bafe04a409372bec8e6d559373f90ce061b`.
This is a backup, not another import overlay. This handoff and final role-proof documentation were written afterward.

**Next experiment:** `P/canonical-stop-driver-retention`, manifest
`92c58c3505da8bf0d70c16106350d893783c2a81831ed807258dee35985dc552`.
It freezes corrected capture identity and invocation-order retention, but excludes
newer placement/catalog and host-preparation edits. The working tree is newer than
this package; do not substitute it silently. Its manifest, source freeze and actual
role proof govern execution. Copied `launch-inputs.json` contains historical status.

Superseded, preserved: `canonical-stop-driver` (`9b5c1331`),
`canonical-stop-driver-current` (`ea258931`), `canonical-idle-preflight` (`483c8271`),
and earlier frozen fixture overlays. Reuse only explicitly pinned build/bootstrap
artifacts. Do not restart obsolete wrappers or delete failed attempts.

## Architecture and resume gate

One external CPU authority owns the transaction. MPI session owns task/transport
retention; original scheduler owns request state and STOP; ModelEngine owns placement;
ExecutionState owns descriptors, original graphs, streams and mapped storage.
Original logical N=32 and nine slots/rank stay fixed; host membership becomes compact 31.
Retain imported backing, quarantine failed-source writable lanes permanently, and
exclude those lanes before reads. FC2 output stays in the original local tail;
private gather prevents failed-row reads. No bank alternation or reclamation.

Before resume: verified original departure; closed frontend admission; 31 actual
scheduler STOP receipts after graph/auxiliary/sampler/native-KV/watchdog joins;
complete captured write/read confinement and retained mapping lifetime; per-layer
resident coverage/capacity; survivor-safe host membership; one matching next-placement
proposal; all READY and INSTALLED acknowledgments; coherent descriptor/generation
publication; authorized original-graph replay with fresh routing; frontend opens last.
Server HOLD, timeout, mask, process exit or matching replies alone cannot prove safety.
A second loss, rank 0/shard loss, partial install or unqualified device/fabric fault is terminal.

## Evidence

| Result | Artifact under B | Exact limit |
|---|---|---|
| Healthy EP32 exact greedy parity at batches 1/2/4/8; all 32 fabric RW; original 128 graphs | `admission-readiness/normal-state-source-slot-parity/transaction-7857880.json` | Earlier source; no death, current STOP or N−1 serving. |
| Native descriptor/quarantine component build | `native-port/source-slot-quarantine/immutable-workspace-candidate/build/attempt-build-7829359/build-receipt.json` | Component evidence, not full-model recovery. |
| Actual V1 allocate/refresh/free binding | `request-disposition-integration/owner-migration/v1-idle-composition-mounted/attempt-7858396/` | Native ownership query only. |
| Stale-routing negative and continuous remote late writer with exact survivor output | `native-port/source-slot-quarantine/quarantine-negative-tests/attempt-7861416/assessment.json` | EP4 live writer; no death/drain/reclamation. Stale combine routing remains hazardous. |
| Real-package imports and 112 focused cases | `P/canonical-current-preflight/attempt-7875544/assessment.json` | Overall FAIL: special CPU import mode circular dependency. Later source edits not covered. |
| Blocked native Bcast released after SIGKILL, original survivors retained, no Revoke; 52.1 ms from injection intent | `host-control-next-step/idle-survivor-integration/idle-broadcast-release/RESULTS.md` | Job 7875893: three CPU workers, safe-broadcast header; not actual request-count ingress/EP32 STOP. |
| Live CPU authority ordinary imports, MPI singleton, original birth/cgroup, four original GPUs zero MiB/noapps | `P/canonical-authority-owned-role-preflight/attempt-7876691/qualified-role-proof.json` | One-node authority isolation, not model task retention. Cleanup/departure/accounting PASS. |
| Geometry/capture-generation, invocation permutation and post-collective deadline fixes | `P/canonical-stop-driver-current/identity-consumer-source-review.json`; `host-control-next-step/invocation-order-receipt.json`; `host-control-next-step/idle-survivor-integration/idle-broadcast-release/host-preparation-review.json` | Focused source checks; native/full-model successors unexecuted. |

Detailed evidence/index: `B/assessment.json`; physical safety contract:
[RECOVERY_CONTRACT.md](RECOVERY_CONTRACT.md). Failed role attempts
7875544/7875756/7875890/7876277 and MPI packaging7875886 remain unchanged.

## Stopped resources and unfinished work

No active owned Slurm allocations, jobs, controllers or agent execution cells.
Final `squeue --user=arusli` returned empty. Jobs 7876691 and 7875893 have strict
process/GPU cleanup and departure receipts. Nothing requires cancellation.
Previously approved 67-case preflight and old `97a7` MPI/pidfd fixture are superseded;
do not execute them. No required approval is outstanding for checkpoint preparation.
Any future run must respect the user's current stop instruction.

Preserved packaging failure: `P/canonical-stop-driver-retention/metadata-failure-b534/`.
A path-variable error wrote metadata through a dependency symlink. The exact original
`deep_ep.py` was restored against its wheel RECORD hash `702ec636…`; corrected manifest
now verifies 122 source pins, 232 links and 49 modes. No job used the invalid package.

First runtime blocker: corrected canonical EP32 healthy parity/admission and actual
worker STOP have not executed. Full captured argument/KV/output bounds and failed-lane
read exclusion still block physical admission to resume. Storage retention is not drain.

Checkpointed newer source: host-only `PreparedPlacement` and actual idle NEXT-digest
catalog; 3 focused placement wrappers pass, 62 isolated protocol cases pass with seven
native proxy/session cases excluded. These successors have no actual package/EP32 proof.
Their source hashes and tests are in `files.json`. No READY/install/commit/resume exists.
Remaining integration: bounded survivor communicator creation/ownership, operation-specific
host calls, device placement install, authority commit, scheduler return, request disposition,
watchdog rearm and degraded cleanup. Never compact fixed logical GPU token rows.

## Next test and runtime reproduction

Run only after work is resumed. This tests matched healthy output, actual rank 2 idle
admission/SIGKILL, frontend/original 31 retention and worker-produced STOP. It grants no resume.

```bash
B=.wideep-ft-runs/backend-safety
P=$B/admission-readiness/normal-state-integration
python3 "$P/canonical-stop-driver-retention/submit-and-run.py" \
  --manifest-sha256 92c58c3505da8bf0d70c16106350d893783c2a81831ed807258dee35985dc552 \
  --role-proof "$P/canonical-authority-owned-role-preflight/attempt-7876691/qualified-role-proof.json"
```

The wrapper owns one 8-node/32-GPU segment-8 allocation, export/import/all-peer RW before
weights, controller and strict cleanup. Monitor `submission-<job>.json`,
`external-controller-<job>.log`, `external-benchmark-<job>/assessment.json`, then
`transaction-<job>.json` in the candidate directory. On interruption, let its exact-owned
cleanup finish; cancel only its recorded allocation if containment fails.

Reuse ARM image `.wideep-ft-runs/devel-1.4.0rc0.dev202610020600-arm64.sqsh`,
SHA256 `5c7112457d8ade87d048e6db5452fc3094a437df0312f089bf3ce74b33b67a48`.
Use native core 7829359 plus V1 binding 7858396; candidate `package-preparation.json`
and `manifest.json` pin exact paths/mounts: core `libtensorrt_llm` SHA `3b1d3af7…`,
`libth_common` `398e4301…`, V1 binding `e7b5cc9f…`. No rebuild needed for this test.
Container Python is `/run_artifacts/exact-build-dabc1180e65b0f934e588db2b9484b4413d4b05f/venv-3.12/bin/python3`.
Qualified runtime uses mpi4py 4.0.0 and local ULFM OpenMPI/PRTE, `mpirun --with-ft ulfm`,
`ob1`/TCP, PRTE daemons under `srun --mpi=none`; preserve pinned prefixes/environment.
GB200 non-CFT fence, PDL 1, CuteDSL unfused-finalize, attention-DP/expert-TP1/PP1, static 288 slots,
V1/no reuse/no overlap/autotuning; graph keys 1/2/4/8. Profile is candidate `config.yaml`.
Set `LLM_MODELS_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/common`;
weights there: `DeepSeek-R1-0528-FP4-V2`. The wrapper records and enforces these settings.

Takeover: run the corrected STOP checkpoint first, fix its first genuine failure,
close captured-memory admission, then integrate one authorized placement/commit/resume
transaction. Existing component successes do not establish N−1 serving.

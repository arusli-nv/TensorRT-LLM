<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# WideEP process-fault regression

This opt-in QA suite characterizes one real EP32 deployment using the native
Ray orchestrator. It injects one verified process failure and asserts native
client behavior and bounded teardown. It does not implement recovery.
[PROCESS_LOSS.md](PROCESS_LOSS.md) retains the physical findings and source analysis.

The qualified deployment is aggregated DeepSeek-R1-0528 NVFP4 on eight four-GPU
GB200 nodes: EP32, expert TP1, attention DP with local TP1, PP1, CUTEDSL,
NVLinkOneSided, static 288-slot placement, decode graph buckets `[1, 2, 4, 8]`,
overlap disabled and autotuning disabled. The current R580 host driver uses the
fence path: CFT and rank-mask FT are disabled. This is a characterization
configuration, not an admitted recovery envelope. Rank 16 is a fixed middle
rank for reproducibility; its expert coverage does not affect these tests.

## Why Ray

This suite uses Ray because the tested MPI/Slurm stack cancelled the model step
on worker death, including with `--kill-on-bad-exit=0`; the plain-process control
survived. Ray exposes actor death independently of model communication. This
finding concerns the tested launcher stack. PROCESS_LOSS.md retains the MPI
controls and archive access. No second runtime is maintained here.

## Scenario contracts

Every attempt first verifies healthy inference, worker identities, resolved
arguments, all 1,856 rank/layer placement and communication records, and final
graph capture. Injection is released only after that evidence is complete.

| Scenario | Question | Required behavior |
|---|---|---|
| `idle_kill` | Does death of middle rank 16 between completed requests reach the next API request? | One matching SIGKILL intent and Ray death; native `EngineDeadError` within the client bound; native shutdown and independent resource cleanup. |
| `stream_kill` | Does the same death interrupt an unfinished stream with a terminal native error? | Kill after a copied nonempty, nonfinal frontend output; preserve its delivered prefix; bounded `EngineDeadError` and complete cleanup. |
| `rank0_kill` | Does loss of the RPC-root rank reach the API client without timing out? | Bounded native `EngineDeadError` and complete cleanup; this remains outside the recovery envelope. Historical timeout XFAILs remain recorded. |
| `idle_term` | Does graceful process termination still become a bounded terminal client error? | One verified SIGTERM and matching Ray death; native `EngineDeadError`, complete teardown. |
| `frontend_kill` | Does an independent HTTP client fail promptly when the real frontend is lost? | Healthy native OpenAI completion first; pinned frontend SIGKILL; explicit connection loss/refusal and complete external cleanup. Native shutdown is unavailable for the killed process. |
| `idle_stop` | Does a verified OS-stopped worker leave the client waiting? | Matching SIGSTOP intent and independent pinned OS state `T`; current client reaches its observation deadline; native shutdown and resource cleanup still required. This asserts the stall gap, not successful detection or recovery. |
| `healthy` | Does a fresh deployment become ready and cleanly shut down? | Same qualified configuration, healthy outputs and cleanup; reconciled cold-start report and separate allocation timing. |

PROCESS_LOSS.md records qualification results, frozen revisions and all earlier
failures. A harness timeout cannot satisfy the native-error contract.

A frontend stream marker does not identify a GPU dispatch/combine phase.
CUDA probe replies are diagnostic evidence; process liveness, CUDA usability
and survivor agreement are distinct. The killed process cannot acknowledge.
Default native crash policy stays enabled; no hard-kill grace escape hatch.
The frontend uses the native OpenAIServer over the same Ray LLM and an
independent HTTP client in the allocation controller; it adds no test-only
serving endpoint.

Qualified test bounds are 720 s from Ray service launch to healthy
readiness, 30 s from injection RPC issuance to client result, and 180 s for
external cleanup. Native shutdown has its own external watchdog; cancellation
of the API driver is a test failure. A timed-out client may exceed its budget
by at most one second for scheduling/recording, independently checked. Native
error delivery must satisfy the actual client bound.
[PROCESS_LOSS.md](PROCESS_LOSS.md#startup-measurements-and-interpretation) records
the observed maxima and explicit margins after qualification.

Cleanup cancels only owned numeric Slurm steps, verifies accounting and local
`srun` exit, then independently checks original process identities, owned step
cgroups, absence of new NVML compute PIDs and memory returning to the pre-launch baseline.
The memory allowance is 64 MiB per GPU; all 40 accepted qualification runs
measured zero delta and no new compute contexts. Missing or inaccessible
evidence fails the test. These
checks do not prove transfer quiescence or late-write containment. After these
checks succeed, cleanup archives Ray session logs and removes only the pinned
node-local temporary tree; the checker verifies archive hashes and all eight
removal records.

## Prepare and run

Use an ARM64 runtime built and verified from the intended native source, with
a SHA256-bound build manifest. Follow the [source-build guide](../../../../docs/source/installation/build-from-source.md)
and use `--cuda_architectures 100-real`. All eight nodes need one healthy
NVLink fabric domain and working IMEX access. The launcher mounts IMEX/GDR
devices and a **shared complete node-local `/tmp`** for Ray services and client
containers, including native RPC sockets. Explicit native Ray placement keeps
four consecutive model ranks on each node.

Install Ray in an ARM64 virtual environment using the native image's system
packages; this deployment qualified Ray 2.55.1. The dependency directory must
contain `venv/bin/python3` and `venv/bin/ray`, with container paths resolving
under `/raydeps`. Verify dependency compatibility and CUDA before qualification.
Host Python needs PyYAML; submitting Python needs pytest. This is a local/QA
suite, not an automatically scheduled multi-node CI test.

From the repository root, set absolute shared paths:

```bash
export LLM_MODELS_ROOT=/path/to/checkpoints
export WIDEEP_MODEL_PATH="$LLM_MODELS_ROOT/DeepSeek-R1-0528-FP4-V2"
export WIDEEP_CONTAINER_IMAGE=/path/to/qualified-runtime.sqsh
export WIDEEP_BUILD_MANIFEST=/path/to/bound-build-manifest.json
export WIDEEP_RAY_DEPENDENCIES=/path/to/qualified-ray-dependencies
export WIDEEP_RUN_ROOT=/path/to/evidence
export WIDEEP_SLURM_ACCOUNT=your-account
# On sites requiring --segment, set it to the requested node count.
# export WIDEEP_SLURM_SEGMENT=8
mkdir -p "$WIDEEP_RUN_ROOT"
python3 tests/integration/defs/wide_ep_ft/deployment_profile.py \
  --config tests/integration/defs/wide_ep_ft/deepseek_r1_ep32.yaml \
  --model-config "$WIDEEP_MODEL_PATH/config.json" \
  --output-dir "$WIDEEP_RUN_ROOT/profile"
export WIDEEP_CONFIG_PATH="$WIDEEP_RUN_ROOT/profile/config.yaml"
```

The generator defaults to 288 slots; 512 slots remain an optional comparison.
Bind the manifest with `deployment_profile.py --image ... --build-manifest ...
--output ...`. `WIDEEP_HOST_PYTHONPATH` may supply shared host dependencies;
keep the native source checkout out of it.

A healthy smoke deployment uses exactly the same runner without injection:

```bash
WIDEEP_SCENARIO=healthy sbatch --account="$WIDEEP_SLURM_ACCOUNT" \
  --partition=batch --qos=short \
  --output="$WIDEEP_RUN_ROOT/regression-%j.log" \
  tests/integration/defs/wide_ep_ft/launch_process_loss.slurm
```

For manual submission on a site requiring segmentation, also pass `--segment=8`
for this eight-node profile. Choose partition and QoS for that site.

Routine regressions run one fresh deployment per scenario:

```bash
WIDEEP_FT_RUN=1 python3 -m pytest -c /dev/null \
  tests/integration/defs/wide_ep_ft/test_process_loss_integration.py \
  --confcutdir=tests/integration/defs/wide_ep_ft -v -r a \
  --junitxml="$WIDEEP_RUN_ROOT/routine.xml"
```

For qualification or changed expectations, set `WIDEEP_QUALIFICATION_RUNS=5`.
This selects five attempts per scenario and ten for streaming; routine uses one.
Add `-x` to the pytest command for qualification so its first failure stops
the batch. No retries: a failed batch stops for diagnosis; every attempt and its original
PASS/XFAIL/XPASS/FAIL/ERROR disposition remains retained. Historical root XFAILs
are findings; corrected root tests require PASS.

Allocation waiting has a separate 7200 s default `WIDEEP_ALLOCATION_TIMEOUT`.
Account/partition/QoS are configurable; `WIDEEP_SLURM_SEGMENT` adds `--segment`
only when required by the site. The fixture derives nodes, GPUs and the middle
rank from native configuration. Manual `sbatch` must override EP32 defaults for
another shape. Structural checks accept uniformly partitioned EP sizes, slots
and graph buckets; only recorded physical runs qualify them. Hardware must
match `driver_report` in the image-bound manifest.

The Slurm batch wrapper supervises the controller. An interrupted fixture sends
USR1 to request cleanup before cancelling the allocation. Controller death
invokes independent owned-step cleanup and fresh `rescue_cleanup` GPU/process
and `rescue_temporary_cleanup` storage records. Original evidence is immutable.
Missing cleanup proof fails explicitly. Forced allocation/node loss can prevent
these probes; allocation disappearance alone never establishes clean resources.

If using the earlier exact native image with the committed Python fixes,
`WIDEEP_PYTHON_OVERLAY=1` freezes five modules (four executor modules and the
LLM entry point) into each writable container before import. Each service/client verifies source and
installed hashes. The build manifest still identifies the base image;
`python_overlay.json` identifies this explicit difference. Do not claim a new
full binary build. Default operation uses the supplied image directly.

## Evidence and startup measurement

Each immutable `process-loss-JOB_ID` directory contains launch commands, source
hashes, image/build provenance, resolved actor metadata, raw launcher/client
logs, normalized per-rank model logs, healthy output, named trigger, one durable
injection intent, observed death, client outcome, diagnostic CUDA replies,
native shutdown, accounting, and independent per-node resource checks.
Records identify their producer host, `CLOCK_MONOTONIC`, and informational UTC;
compare event ordering only within a producer clock. Controller readiness
uses its own receipt time. Actor RPC death timing is evidence receipt latency,
not the precise OS death instant.
JUnit cases carry the `wideep_evidence` property linking their artifact directory.
Diagnostic snapshots retain host thread stacks and local engine error state
before attempting the CUDA operation; absent replies remain unavailable.

Cold startup uses fresh Ray services, API process and actors. The controller
measures service launch to receipt of the healthy-completion record for both
fixed prompts. Filesystem/page/compiler cache state is uncontrolled;
allocation queue time is separate.

`cold_start.json` reconciles controller readiness, client-clock API construction
and native rank-local metrics with their original records.
`ExecutorWorker.get_startup_metrics()` supplies the trees retained in
`initialized.json` for rank zero and `workers.json` for every rank. Intervals
may overlap and **must not be summed**; host elapsed time does not establish GPU
completion. Missing markers, including isolated communicator/JIT/tuning costs,
remain unavailable. Select `-k cold_start`; use five qualification attempts or
one routine attempt.

| Interval | Native marker and interpretation |
|---|---|
| API construction | `initialized.json:construction_seconds`, measured on the client clock around `LLM(...)`. Includes distributed actor initialization. |
| Configuration/distributed initialization | Native `[startup]` marker `configuration_and_distributed_init`; a broad initialization prefix, not an isolated communicator timer. |
| Model loading | `model_loader.total_model_loading_seconds`; checkpoint preparation and weight population are nested intervals, not additional costs. |
| Profiling/final warmup | `initial_model_engine` and `final_model_engine` metrics; retain the two passes separately. |
| Decode graph capture | `gen_cuda_graph_capture_seconds` for each pass. `gen_cuda_graph_warmup_seconds` is separate; the rounded log phase `cuda_graph_capture` includes both. |
| Autotuning | Profile and resolved arguments identify it as disabled. A small `autotuner_warmup_seconds` value is the disabled path's host overhead, not evidence that JIT/warmup disappeared. |

Clean restart-to-readiness is a follow-up; this milestone measures cold startup.

CPU safety and evidence checks run without Ray, GPUs or model weights:

```bash
python3 -m pytest -c /dev/null tests/integration/defs/wide_ep_ft \
  --confcutdir=tests/integration/defs/wide_ep_ft -q
```

The profile checker supports contiguous DeepSeek-V3-family and Qwen3 MoE
checkpoint schemas; it rejects mixed/sparse layouts rather than guessing their
placement. Only recorded physical runs qualify a model/backend configuration.

The launcher requires Slurm accounting, Pyxis/Enroot container options, Linux
PID/cgroup visibility and NVIDIA GPU accounting. It has no fixed cluster name,
account or checkpoint path. Another site must qualify these launch and cleanup
requirements and its GPU fabric. Another scheduler requires a different launcher
and resource probe; the scenario expectations can remain the same.

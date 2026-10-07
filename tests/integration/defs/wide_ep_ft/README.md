<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WideEP fault injection

Run healthy inference, inject one fault, verify teardown, then start new workers
on the same GPUs. Findings are in [FAILURE_BEHAVIOR.md](FAILURE_BEHAVIOR.md).
There is no survivor recovery.

Use dedicated full-GPU hosts in a time-limited Slurm allocation matching the profile's
GPU count. Supply matching TensorRT-LLM binaries and model weights. Model tasks
and CPU probes must see the same source/evidence paths, hostnames and host PID
namespace. CPU probes must own the workers in the host UID view; container UID
remapping is allowed.
The API client and rank-0 worker must share `/tmp` on their host. Allocation and
image setup stay outside this test.

```bash
export LLM_MODELS_ROOT=/path/to/models
export TLLM_FAULT_TOLERANCE_MODE=0
export TRTLLM_FORCE_COMM_METHOD=NVLINK_ONE_SIDED TRTLLM_MOE_A2A_FORCE_CFT=0
unset TLLM_DISABLE_MPI
# Add your model container/mount options to this native srun prefix.
export WIDEEP_FT_MPI_LAUNCHER='srun --mpi=pmix --kill-on-bad-exit=0'
export WIDEEP_FT_PROBE_LAUNCHER="srun --mpi=none --overlap --ntasks=$SLURM_NNODES --ntasks-per-node=1"
python tests/integration/defs/wide_ep_ft/fault_injection.py \
  --launcher "$WIDEEP_FT_MPI_LAUNCHER" --probe-launcher "$WIDEEP_FT_PROBE_LAUNCHER" \
  --model "$LLM_MODELS_ROOT/DeepSeek-R1-0528-FP4-V2" \
  --config tests/integration/defs/wide_ep_ft/deepseek_r1_ep32.yaml \
  --scenario worker_sigkill_streaming --output-dir /shared/evidence/attempt-001
```

Prefixes contain `srun` options only; no shell evaluation. The test owns unique
step names, per-rank logs, bounded cancellation and verified process cleanup.
Forward its `PYTHONPATH` into containers to enable the opt-in MPI worker hook.
CPU probes run once per GPU host and never open CUDA.

Scenarios: `worker_sigkill_idle`, `worker_sigkill_streaming`, `fence_round_mismatch`.
`healthy` measures startup and clean restart. Default target: the middle nonzero
rank; override with `--target-rank`. Triggers identify client boundaries, not GPU
phases. The fence fault changes one local round counter by +2; it is not device loss.
Native kernel timeouts remain unchanged. The tested PMIx stack exits 137 after
faults. A kill either delivers an MPI `RequestError` or terminates the client first;
the fence returns `RequestError`. Client timeouts, unrelated errors and successful
post-fault inference fail the regression.

Each attempt records the prepared config, source hashes, worker identities, GPU/OS probes,
logs, triggers, injection, client outcome, timing and restart. New output directory
required; no implicit retries. Cleanup checks both environment-tagged processes
and identities retained from live host probes, including unreaped zombies.
Unknown unreadable same-UID processes fail preflight; identified host daemons and
positively foreign Slurm jobs are excluded. Forced cleanup records its final step
and resource proof. Intervention remains a failure even if cleanup succeeds.
Failed evidence writes do not block cancellation. Unavailable probe storage prevents
resource verification; the test retries owned-step cancellation and remains failed.
Static cyclic placement supports DeepSeek-V3 and Qwen3 MoE; other layouts require
native `initial_global_assignments`. Every worker must report the requested EP
geometry, non-CFT NVLinkOneSided, and captured graphs when requested.

Readiness: model-step launch to receipt of two greedy, fixed-seed inference results,
on one parent monotonic clock. Allocation queue time is excluded; obtain it
separately from Slurm accounting. Imports, container initialization and construction
are included. Startup phase timers are rank-local and overlap. Autotuning/cache
state is recorded; caches are uncontrolled.
Qualified bounds: startup 720 s, client 30 s (420 s for fence), shutdown 180 s,
cleanup 60 s. Observed maxima and margins are in [FAILURE_BEHAVIOR.md](FAILURE_BEHAVIOR.md).
After readiness, the parent allows client + shutdown time; a client-error receipt
starts the shutdown deadline. These bounds are not service guarantees.

CPU checks are listed in `l0_cpu.yml`. Physical runs are opt-in QA: set the two
launcher variables above plus `WIDEEP_FT_MODEL`, `WIDEEP_FT_CONFIG` and
`WIDEEP_FT_OUTPUT_DIR`, then run `pytest tests/integration/defs/wide_ep_ft/test_fault_injection.py`.
Historical Ray code/results remain in Git history; MPI now follows the native
WideEP/online-EPLB launch path. Identity, trigger and evidence checks stay independent
of recovery policy.

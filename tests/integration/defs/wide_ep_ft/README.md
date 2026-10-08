<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WideEP fault injection

Run healthy inference, inject one fault, verify teardown, and restart on the same
GPUs. Read [FAILURE_BEHAVIOR.md](FAILURE_BEHAVIOR.md) for results and recovery limits.

Supply a dedicated, time-limited Slurm allocation, model weights and matching
TensorRT-LLM binaries. Allocation and image provisioning stay outside the test.
Tasks and probes need shared source/evidence paths, matching hostnames and the host
PID namespace. Probes must own workers in the host UID view; container UID remapping
is allowed. The API client and rank-0 worker must share `/tmp`.

```bash
export LLM_MODELS_ROOT=/path/to/models
export TLLM_FAULT_TOLERANCE_MODE=0
export TRTLLM_FORCE_COMM_METHOD=NVLINK_ONE_SIDED TRTLLM_MOE_A2A_FORCE_CFT=0
unset TLLM_DISABLE_MPI
# Add your container/mount options to the model launch prefix.
export WIDEEP_FT_MPI_LAUNCHER='srun --mpi=pmix --kill-on-bad-exit=0'
export WIDEEP_FT_PROBE_LAUNCHER="srun --mpi=none --overlap --ntasks=$SLURM_NNODES --ntasks-per-node=1"
python tests/integration/defs/wide_ep_ft/fault_injection.py \
  --launcher "$WIDEEP_FT_MPI_LAUNCHER" --probe-launcher "$WIDEEP_FT_PROBE_LAUNCHER" \
  --model "$LLM_MODELS_ROOT/DeepSeek-R1-0528-FP4-V2" \
  --config tests/integration/defs/wide_ep_ft/deepseek_r1_ep32.yaml \
  --scenario worker_sigkill_streaming --output-dir /shared/evidence/attempt-001
```

For ULFM, add `--launcher-mode ulfm` and supply an executable wrapper as
`--launcher`. It must accept `--job-name=`, `--jobid=` and `--output=.../rank-%t.log`,
preserve the run environment and name all PRTE Slurm steps as supplied. The tested
configuration uses `mpirun --with-ft ulfm`, ob1/TCP and daemon `srun --mpi=none`.
Preserve Slurm allocation/configuration variables, including `SLURM_JOBID`, when
launching from outside the allocation. The test hashes the wrapper; retain its
companion scripts. Forward `PYTHONPATH` into containers for the opt-in worker hook.
Probes use direct `srun`, run once per GPU host and never open CUDA.

Scenarios are `worker_sigkill_idle`, `worker_sigkill_streaming` and
`fence_round_mismatch`. `healthy` measures startup and clean restart. Default target
is the middle nonzero rank; use `--target-rank 2` for this profile's covered replica
placement. Triggers name client boundaries, not GPU phases. The fence injection
increments one local round counter by two; it does not simulate device loss.
PMIx kills permit a native MPI `RequestError` or client termination. ULFM kills
require the error, independent target-death evidence and native whole-job abort.
Successful post-fault inference, timeouts and unrelated errors fail the regression.

Each new output directory retains config/source hashes, identities, GPU/OS probes,
per-rank logs, injection, client outcome, timing and restart. There are no retries.
Cleanup checks owned steps, RUN_ID-tagged processes, pinned identities, zombies,
GPU compute processes and memory. Unknown unreadable same-UID processes fail
preflight. Partial live probes cannot qualify cleanup. Forced cleanup is recorded
and leaves the attempt failed. Static placement supports DeepSeek-V3 and Qwen3 MoE;
other layouts need native `initial_global_assignments`. Workers must confirm the
requested EP geometry, non-CFT NVLinkOneSided and requested graph captures.

Timing uses one controller monotonic clock: launch to first complete greedy result,
readiness after two results, and injection receipt through cleanup to fresh serving.
It includes observer overhead. Queue time is separate in Slurm accounting.
Rank-local phase intervals overlap; missing markers remain unmeasured. Autotuning
state is recorded and filesystem caches are uncontrolled. Bounds/maxima are in
[FAILURE_BEHAVIOR.md](FAILURE_BEHAVIOR.md).

CPU checks run in `l0_cpu.yml`. Physical tests are opt-in QA. Set the launcher
variables above, `WIDEEP_FT_MODEL`, `WIDEEP_FT_CONFIG`, `WIDEEP_FT_OUTPUT_DIR` and,
for ULFM, `WIDEEP_FT_MPI_LAUNCHER_MODE=ulfm`; then run
`pytest tests/integration/defs/wide_ep_ft/test_fault_injection.py`.
Historical Ray results remain in Git history. MPI follows the native WideEP/EPLB
path; this test adds no recovery policy.

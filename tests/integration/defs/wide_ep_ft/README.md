<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WideEP fault injection

Inject one fault, verify teardown, then start fresh workers on the same GPUs.
[FAILURE_BEHAVIOR.md](FAILURE_BEHAVIOR.md) defines the scenarios, measured times and recovery limits.

Provide a dedicated, time-limited Slurm allocation, model weights and matching
binaries. Image/allocation provisioning stays outside the test. Workers and probes
need shared source/evidence paths and the host PID namespace; probes must own the
workers in the host UID view. Client and rank-0 worker must share `/tmp`.

```bash
export LLM_MODELS_ROOT=/path/to/models
export TLLM_FAULT_TOLERANCE_MODE=0
export TRTLLM_FORCE_COMM_METHOD=NVLINK_ONE_SIDED TRTLLM_MOE_A2A_FORCE_CFT=0
unset TLLM_DISABLE_MPI
# Include the deployment's container/mount options in this prefix.
export WIDEEP_FT_MPI_LAUNCHER='srun --mpi=pmix --kill-on-bad-exit=0'
export WIDEEP_FT_PROBE_LAUNCHER="srun --mpi=none --overlap --ntasks=$SLURM_NNODES --ntasks-per-node=1"
python tests/integration/defs/wide_ep_ft/fault_injection.py \
  --launcher "$WIDEEP_FT_MPI_LAUNCHER" --probe-launcher "$WIDEEP_FT_PROBE_LAUNCHER" \
  --model "$LLM_MODELS_ROOT/DeepSeek-R1-0528-FP4-V2" \
  --config tests/integration/defs/wide_ep_ft/deepseek_r1_ep32.yaml \
  --scenario worker_sigkill_streaming --target-rank 2 --output-dir /shared/evidence/attempt-001
```

Scenarios: `healthy`, `worker_sigkill_idle`, `worker_sigkill_streaming`,
`fence_round_mismatch`. Default target is the middle nonzero rank; rank 2 preserves
expert coverage for this profile. Every run checks actual EP geometry, fence backend
and requested graph captures. EPLB placement is held static.

For ULFM, use `--launcher-mode ulfm --launcher /path/to/wrapper`. The external wrapper
must accept `--job-name=`, `--jobid=` and `--output=.../rank-%t.log`, preserve the run
environment, and use the supplied name for every PRTE Slurm step. Tested setup:
`mpirun --with-ft ulfm`, ob1/TCP, daemon `srun --mpi=none`. Forward `PYTHONPATH` into
containers for the worker hook. Preserve Slurm configuration/allocation variables,
including `SLURM_JOBID`. Retain the hashed wrapper and companion scripts with evidence.
CPU probes use direct `srun`, one per GPU host, and never open CUDA.

Each fresh output directory retains hashes, identities, per-rank logs, injection,
client outcome, timing and cleanup. No retries. Cleanup requires no owned steps,
processes/zombies or GPU compute processes and memory within 64 MiB/GPU of baseline.
Unreadable unaccounted-for same-UID processes fail preflight. Partial live observations
cannot qualify cleanup; forced cleanup leaves the attempt failed. PMIx kills may
terminate the client; ULFM kills require a native MPI error, target-death evidence and
whole-job abort. Successful post-fault inference and unrelated errors fail this baseline.

Default deadlines: startup 720 s, result wait 30 s or 420 s for fence mismatch,
shutdown and cleanup 180 s each. Result waiting follows a separate 15-s injection
receipt wait. Static assignments can be generated for DeepSeek-V3/Qwen3 MoE; other
layouts must supply native `initial_global_assignments`.

CPU tests are listed in `l0_cpu.yml`; GPU regressions are opt-in QA. Set launcher
variables above plus `WIDEEP_FT_MODEL`, `WIDEEP_FT_CONFIG`, `WIDEEP_FT_OUTPUT_DIR` and
optionally `WIDEEP_FT_MPI_LAUNCHER_MODE=ulfm`; run
`pytest tests/integration/defs/wide_ep_ft/test_fault_injection.py`.
MPI follows the native WideEP/EPLB path.

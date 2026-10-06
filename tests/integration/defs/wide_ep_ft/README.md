<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WideEP fault injection

Inject one fault, check teardown and restart with new workers on the same GPUs.
The test does not implement survivor recovery. MPI baseline and historical Ray
results are in [FAILURE_BEHAVIOR.md](FAILURE_BEHAVIOR.md). The runner below uses Ray;
MPI launcher/probe packaging is still in progress.

Provide a dedicated, idle Ray cluster with exactly the profile's GPU count and
matching TensorRT-LLM binaries on every node. Run the driver on a GPU node.
Rank 0 uses local IPC, so containers on that node must share `/tmp`.
Supply the allocation, image and Ray services. The test owns model jobs and CPU probes.

Set these variables before starting Ray:

```bash
export TLLM_DISABLE_MPI=1 TLLM_FAULT_TOLERANCE_MODE=0
export TRTLLM_FORCE_COMM_METHOD=NVLINK_ONE_SIDED TRTLLM_MOE_A2A_FORCE_CFT=0
```

```bash
export LLM_MODELS_ROOT=/path/to/models
python tests/integration/defs/wide_ep_ft/fault_injection.py \
  --address "$RAY_ADDRESS" --model "$LLM_MODELS_ROOT/DeepSeek-R1-0528-FP4-V2" \
  --config tests/integration/defs/wide_ep_ft/deepseek_r1_ep32.yaml \
  --scenario worker_sigkill_streaming --output-dir /shared/evidence/attempt-001
```

Choose `worker_sigkill_idle`, `worker_sigkill_streaming`, `process_group_destroy`
or `fence_round_mismatch`. Use `healthy` for startup and clean restart.
Communication faults destroy local WORLD groups, including registered CPU/CUDA
groups, or corrupt a fence round. Physical hardware loss remains untested.
Workers must report non-CFT NVLinkOneSided and matching EP size/rank before inference.
Static cyclic EPLB supports DeepSeek-V3 and Qwen3 MoE layouts. For other layouts,
supply native `initial_global_assignments`.

Keep native kernel timeouts unchanged. For `fence_round_mismatch`, add
`--client-timeout-s 420`. The nominal 300-second fence timeout uses GPU clock
cycles, so elapsed time depends on the SM clock. Triggers mark request/client
boundaries, not exact GPU phases.

Evidence includes identities, logs, startup intervals, injection, client errors,
CUDA probes, cleanup and restart. Use a new output directory for each attempt.
Failed attempts remain on disk. The test never retries implicitly.
It records forced cleanup as an intervention.

Timing starts at model-driver launch after Ray is ready. Ray startup is excluded.
Allocation queue time is unavailable. End-to-end timing uses one parent clock.
Producer clocks are host-local. Startup intervals overlap.
Autotuning and cache state are recorded.

CPU checks use `pytest tests/integration/defs/wide_ep_ft/test_fault_injection.py`.
Physical pytest runs require `WIDEEP_FT_RAY_ADDRESS`, `WIDEEP_FT_MODEL`,
`WIDEEP_FT_CONFIG` and a shared `WIDEEP_FT_OUTPUT_DIR`.
Startup/client/shutdown/cleanup defaults are 720/30/180/60 seconds, with 420 for
the fence client. Measured maxima plus margins are 340.4 + 379.6 = 720 s for
startup, 10.3 + 19.7 = 30 s for the non-fence client and 291.5 + 128.5 = 420 s
for the fence client. Shutdown/cleanup retain initial bounds. All bounds are provisional.
Recalibrate from repeated runs before treating them as service guarantees.

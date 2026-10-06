# WideEP fault injection

[FAILURE_BEHAVIOR.md](FAILURE_BEHAVIOR.md) is the concise findings report.
This test characterizes one failure and an **explicit fresh restart**; it does
not implement survivor recovery.

Supply a dedicated, idle Ray cluster with exactly the profile’s GPU count and
matching TensorRT-LLM binaries on every node. Run the driver on one of those
GPU nodes; rank 0 uses local IPC. Containers on that node must share `/tmp`.
Allocation, image installation and Ray-service lifetime remain outside the test.
The test owns only its model jobs and CPU resource probes.

Before starting Ray, configure the qualified fence path:

```bash
export TLLM_DISABLE_MPI=1 TLLM_FAULT_TOLERANCE_MODE=0
export TRTLLM_FORCE_COMM_METHOD=NVLINK_ONE_SIDED TRTLLM_MOE_A2A_FORCE_CFT=0
```

Keep native kernel timeouts unchanged. For `fence_round_mismatch`, pass
`--client-timeout-s 420`: the steady fence timeout is nominally 300 seconds.
Captured warmup budgets are not precise phase triggers.

```bash
export LLM_MODELS_ROOT=/path/to/models
python tests/integration/defs/wide_ep_ft/fault_injection.py \
  --address "$RAY_ADDRESS" --model "$LLM_MODELS_ROOT/DeepSeek-R1-0528-FP4-V2" \
  --config tests/integration/defs/wide_ep_ft/deepseek_r1_ep32.yaml \
  --scenario worker_sigkill_streaming --output-dir /shared/evidence/attempt-001
```

Scenarios: `worker_sigkill_idle`, `worker_sigkill_streaming`,
`host_collective_abort`, `fence_round_mismatch`; `healthy` measures startup
and clean restart without a fault. Communication scenarios are synthetic:
Gloo backend abort and protocol-round mismatch, not hardware loss.
Static cyclic EPLB preparation supports the documented DeepSeek/Qwen MoE
layouts; other layouts supply native `initial_global_assignments` explicitly.

Evidence contains identities, logs, native startup intervals, the verified
injection, client error, CUDA-probe replies, cleanup checks and restart result.
Producer clocks are host-local; end-to-end timing uses one parent clock.
Autotuning and cache state are labeled; phase intervals can overlap.
Timing starts at model-driver launch after Ray is ready; cluster startup is
excluded. Allocation queue time is unavailable to this externally provisioned test.
All attempts get new directories; failed attempts are retained, never retried
implicitly. A forced cleanup is evidence of harness intervention.

CPU checks: `pytest tests/integration/defs/wide_ep_ft/test_fault_injection.py`.
Physical cases are opt-in with `WIDEEP_FT_RAY_ADDRESS`, `WIDEEP_FT_MODEL`,
`WIDEEP_FT_CONFIG` and shared `WIDEEP_FT_OUTPUT_DIR`.
Initial startup/client/shutdown/cleanup bounds are 720/30/180/60 seconds
(420 seconds for the fence client);
recalibration requires measured maxima plus a documented margin.

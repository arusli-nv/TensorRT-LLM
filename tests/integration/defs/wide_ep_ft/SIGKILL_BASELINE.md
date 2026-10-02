<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# One-rank SIGKILL experiments

These opt-in experiments measure one non-rank-0 model process death. They build on
`fault_injector.py` and `process_loss_harness.py`; they do not implement membership changes,
transfer quiescence, communicator reconstruction, or inference recovery. Run each case in an
isolated allocation with a whole-job deadline and save launcher, model-worker, and client logs
beside the structured output. Do not commit machine-specific logs or model paths.

## Ray model observation

Start a Ray cluster on the allocated nodes using the same Python environment, repository checkout,
and shared output directory on every node. Make the repository and this directory importable to Ray
workers before `ray start`:

```bash
export PYTHONPATH="$PWD:$PWD/tests/integration/defs/wide_ep_ft${PYTHONPATH:+:$PYTHONPATH}"
```

`ray_process_loss_probe.py` attaches to that cluster; it does not launch Ray or Slurm. Supply the
model path and a JSON object of `LLM(...)` keyword arguments. For a two-node, eight-GPU TP control
using a suitable model:

```json
{
  "tensor_parallel_size": 8,
  "gpus_per_node": 4,
  "enable_autotuner": false,
  "cuda_graph_config": null,
  "max_batch_size": 4,
  "max_num_tokens": 512,
  "max_seq_len": 2048,
  "kv_cache_config": {"free_gpu_memory_fraction": 0.5}
}
```

Save that object as `llm_kwargs.json`, then run from the same environment:

```bash
timeout --kill-after=10s 15m python3 tests/integration/defs/wide_ep_ft/ray_process_loss_probe.py \
  --model "$LLM_MODELS_ROOT/Qwen3/Qwen3-8B" \
  --llm-kwargs llm_kwargs.json --ray-address "$RAY_ADDRESS" \
  --expected-ranks 8 --target-rank 4 --output-dir "$RUN_DIR/ray_sigkill"
```

The controller waits for Ray resources, completes a healthy model request, obtains each rank's
identity from the model actor, and schedules the existing one-shot injector on the victim's node.
It then sends a *diagnostic* proposal over independent Ray actor RPCs. Matching replies show that
those actor processes can receive and validate the same message; they are not readiness to resume,
distributed membership authority, or a committed generation. The next request's response, error,
or timeout is recorded rather than prescribed. The controller uses Ray's worker handles exposed by
TRT-LLM's Ray executor, so a change to that internal test hook requires requalification.

To observe replies after TRT-LLM's ordinary ten-second rank-crash kill window, launch **all** Ray
nodes with `TLLM_RANK_CRASH_HARD_KILL_GRACE=-1` and add `--probe-delay-s 15`. This is a diagnostic
escape hatch only: it leaves a failed executor loop in place and must not be used as a recovery or
serving policy. The run configuration records the setting. Without it, use the default immediate
probe and retain the normal fail-closed behavior. Add `--require-all-survivors` only for a
qualified control-path regression run; it makes missing or inconsistent replies fail the scenario
after still recording the client result.

The output directory is exclusive to one run. It contains `configuration.json`,
`healthy_request.json`, `rank_map.json`, `injection/injection_intent.json`,
`injection/injection_result.json`, `diagnostic_proposal.json`, `survivor_probe.json`,
`post_failure_request.json`, and `run_summary.json`. `observation_complete` means the observations
were saved; `injection_uncertain` means the intent/result and process logs require manual review.
The controller does not bound model startup or shutdown; the external whole-job deadline remains
mandatory.

## MPI controls

The existing HTTP `process_loss_harness.py` can observe a TRT-LLM job launched with MPI; its README
describes the rank/PID map and external launcher requirements. Separately,
`mpi_ulfm_sigkill_control.c` tests an MPI *runtime primitive*, not a TRT-LLM model:

```bash
mpicc -O2 -o /tmp/mpi_ulfm_sigkill_control \
  tests/integration/defs/wide_ep_ft/mpi_ulfm_sigkill_control.c
timeout --kill-after=5s 60s mpirun --with-ft ulfm -np 3 /tmp/mpi_ulfm_sigkill_control
```

Use an Open MPI build with ULFM support and `mpirun` inside an allocation. Rank 1 self-SIGKILLs
after a healthy collective; ranks 0 and 2 must report `SURVIVOR_AGREE` with count 2 and a
successful collective on the shrunken communicator. A launcher may return a nonzero status for
the killed rank even when both survivors succeed; check both survivor records and the deadline.
This control does not show that TRT-LLM's current launcher or model collectives can use ULFM.

## Observed behavior and limits

- In a tested two-node MPI TRT-LLM TP8 run, a healthy request preceded rank 1 SIGKILL; Slurm's
  PMIx error handler cancelled the step. A separate MPI pool control also lost a second worker
  after one worker died. These are launcher/runtime observations, not a universal MPI rule.
- In a tested two-node Ray TRT-LLM TP8 run, the fixed-world Gloo request-count broadcast failed
  after rank 4 died. Ray actor RPCs initially reached the surviving ranks; TRT-LLM's executor-loop
  crash timer later killed rank 0. With that timer disabled *for diagnosis only*, all seven
  survivors replied to the same test-only proposal after 15 seconds, while the next model request
  still failed. Actor reachability and matching replies are not inference continuation.
- A one-node, three-rank Open MPI ULFM control completed shrink, agreement, and a survivor-only
  collective after one death when MPI processes were launcher-managed directly. A shell-child
  launch arrangement did not show the same progress. The exact cause of rank-0 exit in the tested
  TRT-LLM ULFM model arrangement remains unproven.

These configurations had no admitted expert-replica recovery, no demonstrated CFT failed-issuer
quiescence, and no survivor CUDA-graph reuse proof. `SIGKILL` models process death, not device or
fabric failure. NCCL-communicator and transfer-level faults require separate injection cases.

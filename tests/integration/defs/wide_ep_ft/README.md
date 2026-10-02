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

# One-rank process failure injector

This first slice sends one `SIGKILL` to a manually identified local, nonzero model world rank.
It does not launch TRT-LLM, coordinate survivors, or test recovery.

Run the local CPU targeting tests with
`python3 tests/integration/defs/wide_ep_ft/test_fault_injector.py` and
`python3 tests/integration/defs/wide_ep_ft/test_process_loss_harness.py`. Both are selected in the
CPU pre-merge test list; the physical experiment is opt-in.
Use an isolated job with a separate log directory.

1. Start a model job and complete a healthy request. Identify the **model process** for
   world rank 1, not only its Slurm task or launcher process.
2. On the same node and in a PID namespace where that process is visible, run
   `python3 tests/integration/defs/wide_ep_ft/fault_injector.py inspect --world-rank 1 --pid <pid>`.
   Save its JSON output as `ranks.json` in the run directory. Check that its rank, host, and PID
   match the worker logs. The `start_time_ticks` field distinguishes this process from a later
   process that reuses its PID.
3. With the job still running and between requests, run
   `python3 tests/integration/defs/wide_ep_ft/fault_injector.py kill --rank-map <run_dir>/ranks.json --target-rank 1 --output-dir <run_dir>/injection`.
4. Save the next client result, per-rank logs, launcher output, and Slurm output. Use an external
   job deadline so a hung collective cannot hold the experiment indefinitely.

`injection_intent.json` is written before signaling. `injection_result.json` records whether the
signal request succeeded. A missing result after a recorded intent means the outcome is uncertain;
inspect the process and launcher logs. Reusing an output directory is refused to prevent a second
injection from the same run. A successful signal proves delivery was requested for the chosen
process; it does not prove TRT-LLM detected the loss or that survivors recovered.

`process_loss_harness.py` automates the between-request observation after an external launcher
starts the model job. The launcher must put the verified target's `ranks.json` on the victim node,
save its model configuration and per-rank logs, and run the controller there so the target PID is
visible. With `RANK0_HOST`, `PORT`, and `RUN_DIR` set by the launcher, run from the victim node
within an isolated allocation:

```bash
timeout --kill-after=10s 15m python3 tests/integration/defs/wide_ep_ft/process_loss_harness.py \
  --base-url "http://${RANK0_HOST}:${PORT}" --rank-map "${RUN_DIR}/ranks.json" \
  --target-rank 1 --output-dir "${RUN_DIR}/observation"
```

The launcher also needs its own Slurm time limit and cleanup trap: the controller's socket timeouts
and readiness deadline do not bound the entire model job. The controller writes
`healthy_request.json` before injection, then the injector's intent/result records, followed by
`post_failure_request.json` and `run_summary.json`. A post-failure disconnect, explicit error,
timeout, or response is an observation; `observation_complete` means evidence was recorded, not
that inference recovered. `injection_uncertain` requires checking the injector's intent/result
files and process logs before drawing a conclusion. No signal is sent unless the healthy request
produced a nonempty completion. That gate checks the serving path, not model-answer correctness.

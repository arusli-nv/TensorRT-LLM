<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# WideEP deployment evidence

`deployment_profile.py` prepares static expert placement in native LLM arguments
and binds a build manifest to the runtime-image SHA256. It checks requested and
resolved configuration, complete per-rank/layer placement and communication logs,
and graph capture. Structural validation does not qualify recovery or hardware.
`process_probe.py` publishes exclusive run records and independently checks
process identities, NVML resources and pinned temporary storage.

From the repository root, prepare a native profile without GPUs:

```bash
python3 tests/integration/defs/wide_ep_ft/deployment_profile.py \
  --config tests/integration/defs/wide_ep_ft/deepseek_r1_ep32.yaml \
  --model-config /path/to/checkpoint/config.json --output-dir /path/to/profile
```

The default is DeepSeek-R1 EP32/288 slots with attention DP, expert TP1, static
EPLB and decode graphs. The contiguous DeepSeek-V3-family and Qwen3 MoE schemas
share the checker; mixed/sparse layouts are rejected. Rank/node/GPU counts are
derived from configuration under uniform whole-node placement. The coverage
report concerns routed experts only, not full-model recovery admission.

CPU validation:

```bash
python3 -m pytest -c /dev/null tests/integration/defs/wide_ep_ft/test_deployment_profile.py \
  --confcutdir=tests/integration/defs/wide_ep_ft
```

A physical run requires a separately qualified runtime, hardware/driver,
launcher, backend and checkpoint combination. CFT, survivor agreement,
unchanged-graph recovery and correct post-failure inference remain unproved.

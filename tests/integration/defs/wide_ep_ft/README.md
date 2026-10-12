<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WideEP peer-loss recovery tests

Tests for WideEP peer-loss recovery. The feature, its environment variables and its limits are
documented in `docs/source/features/wide-ep-fault-tolerance.md`.

## Single-node test

`test_peer_loss_recovery.py` runs DeepSeek-V3-Lite BF16 on one 4-GPU GB200 node: EP4 with
attention DP, static EPLB holding two replicas of every expert, decode CUDA graphs and the
NVLinkOneSided all-to-all. It starts the authority, launches
`mpirun --with-ft ulfm -np 4 trtllm-llmapi-launch python3 peer_loss_driver.py` and SIGKILLs
the worker of rank 2:

- `idle`: between two rounds of requests;
- `inflight`: while one request per attention-DP rank is decoding.

It checks that the survivors' requests return the expected greedy answers before, during
and after the loss, that the request of rank 2 fails with the retry error, that every
survivor logs `resumed serving without rank 2`, that no CUDA graph is captured after the
kill, and that `mpirun` exits within a bound once the driver finishes. Any `mpirun` exit
status is accepted: ending a world that lost a rank calls `MPI_Abort`.

```bash
LLM_MODELS_ROOT=/path/to/llm-models \
    pytest tests/integration/defs/wide_ep_ft/test_peer_loss_recovery.py -v
```

The test skips without 4 GB200 GPUs, without an Open MPI that reports the `ftmpi`
extension in `ompi_info`, or with mpi4py older than 4. Per-run logs (`mpirun.log`,
`authority.log`, `result.json`) are in the pytest `tmp_path`.

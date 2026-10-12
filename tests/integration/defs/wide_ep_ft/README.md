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

## Multi-node qualification (EP32)

The target deployment is DeepSeek-R1 NVFP4 on EP32 across eight 4-GPU GB200 nodes. It is not
CI-portable; qualify it as follows.

1. Write an LLM API config with `enable_attention_dp: true`,
   `disable_overlap_scheduler: true`, `enable_lm_head_tp_in_adp: false`, KV cache
   decode
   `cuda_graph_config`, and `moe_config.load_balancer` with `layer_updates_per_iter: 0` and
   `initial_global_assignments` for every MoE layer. Give each rank one or more spare slots
   so that every expert of each rank that may be lost has a replica on another rank.
2. On the head node, generate a key and start the authority with `--world-size 32`.
   Export the three recovery variables to every rank.
3. Launch one `mpirun --with-ft ulfm` across the nodes with `--mca pml ob1`,
   `--mca btl self,tcp` and `--mca coll ^hcoll,ucc`, running
   `trtllm-llmapi-launch python3 <driver>`, where the driver is an LLM API program like
   `peer_loss_driver.py` that pins requests to attention-DP ranks with
   `SchedulingParams(attention_dp_rank=..., attention_dp_relax=False)`.
4. SIGKILL the `tensorrt_llm.llmapi.mgmn_worker_node` process of a covered rank, on its node,
   idle and mid-decode. For token-exact checks, compare against a healthy run whose EPLB
   routing already excludes the victim.
5. Pass when survivor-owned requests complete (token-exact where checked), the victim's
   requests fail with the retry error, the authority logs
   `committed generation 1 without rank N`, and the job ends after the driver shuts down. If
   the launcher outlives every rank, stop it after a fixed grace period.

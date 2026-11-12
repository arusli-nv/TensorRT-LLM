<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Wide-EP Fault Tolerance (Prototype)

```{warning}
Wide-EP fault tolerance is a prototype and is subject to change.
```

A wide expert-parallel (Wide-EP) deployment spreads the experts of every MoE layer over all its ranks, so without
fault tolerance one failed rank ends the whole deployment. With `enable_wide_ep_fault_tolerance`, the surviving ranks
restore the failed rank's experts from host weights into spare EPLB slots and keep serving on the same CUDA graphs.
A replacement process then starts on the same GPU and joins, and the deployment is back at full capacity.

```text
healthy (N ranks) --rank fails--> degraded (N-K ranks, same CUDA graphs) --replacement joins--> healthy (N ranks)
```

A rank fails when its process exits (a crash, `kill -9`, or the loss of every process of the deployment on its node)
or when it makes no progress for 30 s while other ranks wait for it. Serving pauses for a few seconds after a crash,
and for the hang timeout plus a few seconds after a hang.

## Enabling it

```yaml
enable_wide_ep_fault_tolerance: true
enable_attention_dp: true
kv_cache_config:
  use_kv_cache_manager_v2: true
moe_config:
  load_balancer:
    num_slots: 256
```

The LLM API takes the same option, `LLM(..., enable_wide_ep_fault_tolerance=True)`. On one node,
`trtllm-serve <model> --tp_size 4 --ep_size 4 --config config.yaml` needs nothing more: the server starts every rank
as its own process and replaces a failed one.

## Launching on several nodes

1. Export the same shared secret (at least 32 bytes, hex) and rendezvous address (rank 0's node and a free port, with
   the next port free too) on every node. Pass them in the environment, never on the command line.
2. Give `/dev/shm` on every node room for the host weights of every expert (DeepSeek-R1 FP4: about 364 GB). Use
   `docker run --shm-size`, or on Kubernetes an `emptyDir` volume with `medium: Memory` mounted at `/dev/shm`.
3. Start `trtllm-llmapi-launch` once per node under a launcher that keeps the other processes running when one
   exits. It starts the node's ranks and a replacement on the same GPU for a rank that fails.

On Slurm, start the node supervisors with `mpirun --enable-recovery` inside a `--no-requeue` allocation. `srun` is
rejected because Slurm cancels the whole job step when one task exits.

```bash
#SBATCH --no-requeue
export TLLM_FAULT_TOLERANCE_SHARED_SECRET=$(openssl rand -hex 32)
export TLLM_FAULT_TOLERANCE_RENDEZVOUS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1):29600
mpirun --enable-recovery -np 8 --map-by ppr:1:node \
    -x TLLM_FAULT_TOLERANCE_SHARED_SECRET -x TLLM_FAULT_TOLERANCE_RENDEZVOUS \
    trtllm-llmapi-launch trtllm-serve <model> --tp_size 32 --ep_size 32 --config config.yaml
```

On Kubernetes, run one pod per node in a group that is not recreated when one pod restarts (for a LeaderWorkerSet,
no `RecreateGroupOnPodRestart`). Set `PMI_RANK` to the node rank, `PMI_SIZE` to the node count, and the rendezvous to
`LWS_LEADER_ADDRESS`.

## Requests and health

- Requests in flight on the surviving ranks continue by recompute, and tokens already streamed are kept.
- A request of the failed rank runs again on a surviving rank if it has not streamed output yet. Otherwise it fails
  with HTTP 503, or an `error` event when streaming, with the error `code` `rank_failure`. Send it again.
- `GET /health` answers 200 while the deployment serves, also with failed ranks. It answers 503 once the deployment
  cannot recover, and the server then shuts down.
- `GET /fault_tolerance_status` reports the active and failed ranks (`active_ranks`, `failed_ranks`), the membership
  generation, each rank's incarnation, the last membership change and the `state`:

| State | Meaning |
| --- | --- |
| `healthy` | Every rank serves |
| `degraded` | Some ranks failed and the rest serve |
| `recovering` | A membership change is in progress and serving pauses |
| `unprotected` | The authority does not answer, so a further failure is not recovered |
| `failed` | The deployment cannot recover |

## Limitations

- Supported: one NVLink domain (such as GB200 NVL72), attention DP and expert parallelism over all ranks,
  `NVLinkOneSided` MoE communication, the Cutlass MoE backend, or for NVFP4 an explicit `backend: CUTEDSL` (not with
  `enable_locality_domains` or on SM107), standard EPLB, KV cache manager V2, the overlap scheduler, the default
  attention-DP router, one serving frontend and the MPI executor. `modeling_v2: auto` builds the built-in model.
- Required: `max_beam_width: 1`, `context_parallel_size: 1`, `gpus_per_node` equal to the ranks per node,
  `enable_userbuffers: false` in `torch_compile_config`, `TRTLLM_MOE_COMM_TIMEOUT_SEC` and the executor hang detector
  timeout of at least 60 s, and CDMM mode with GDRCopy for online EPLB (else `layer_updates_per_iter: 0`).
- Rejected: pipeline parallelism, TRTLLM-Gen, DeepGEMM (FP8 DeepSeek-R1) and other MoE backends (not supported yet),
  two-model speculation (such as Eagle3 or draft-target), NGram, user-provided and DSpark drafters (one-model MTP
  works), disaggregated serving, KV cache connectors and events, sleep mode, `enable_lm_head_tp_in_adp`,
  `modeling_v2: require`, hybrid linear-attention models, and requests with `n > 1`, `best_of > 1` or beam search.
- At a failure, survivors lend part of their KV cache to the restored experts until the replacement joins. Startup
  fails if they cannot lend enough for a node (add ranks, use fewer EPLB slots or raise `free_gpu_memory_fraction`).
- Hardware failures and the loss of rank 0 or its node are not covered.
- The authority listens on the rendezvous port and the next one on rank 0's node. Run on a trusted network.

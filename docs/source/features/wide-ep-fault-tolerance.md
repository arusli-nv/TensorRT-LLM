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

A rank fails when its process exits (a crash, `kill -9`, or the loss of every process of the deployment on its node),
when its node is cut off from the network for 60 s, or when it makes no progress for 60 s while other ranks wait for it.

## Configuration

A DeepSeek-R1 NVFP4 deployment on 8 GB200 nodes of 4 GPUs (32 ranks):

```yaml
enable_wide_ep_fault_tolerance: true
enable_attention_dp: true
cuda_graph_config:
  enable_padding: true
  max_batch_size: 128
kv_cache_config:
  dtype: fp8
  free_gpu_memory_fraction: 0.5
  use_kv_cache_manager_v2: true
speculative_config:
  decoding_type: MTP
  num_nextn_predict_layers: 1
moe_config:
  backend: CUTEDSL
  load_balancer:
    num_slots: 256
    layer_updates_per_iter: 0
```

- Static EPLB (`layer_updates_per_iter: 0`) keeps the initial expert placement and needs nothing more. Online EPLB
  (`layer_updates_per_iter` above 0) needs CDMM mode and GDRCopy, with `/dev/gdrdrv` mapped into the container
  (see the [Wide-EP example](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep)).
- The survivors keep the CUDA graphs they captured at startup, so capture the batch sizes they serve when degraded.
  With `torch_compile_config`, set `enable_userbuffers: false` in it.
- The LLM API takes the same option. On one node, `trtllm-serve <model> --tp_size 4 --ep_size 4` needs no launcher.

## Launching on several nodes

1. Export the same shared secret (at least 32 bytes, hex) and rendezvous address on every node, in the environment
   and never on the command line. The rendezvous is node rank 0's host and a free port, with the next port free too.
2. Give `/dev/shm` on every node room for the host weights of every routed expert: the experts times the MoE layers
   times the bytes of one expert (DeepSeek-R1 NVFP4: about 364 GB per node).
3. Start `trtllm-llmapi-launch` once per node under a launcher that keeps the other processes running when one
   exits. Its node supervisor starts the node's ranks, and a replacement on the same GPU for a rank that fails.

On Slurm, use `mpirun --enable-recovery` of Open MPI 5 in a `--no-requeue` allocation. With several nodes, startup
fails unless `prun` and `pterm` are on `PATH`, since node 0 relaunches a lost node and ends the job through them.
`srun` is rejected because Slurm cancels the whole job step when one task exits. With containers, run one task per
node that keeps its container up with a way to run commands in it (such as an sshd), and start `mpirun` in the first
node's container with PRRTE's ssh launcher pointed at a command with the `ssh HOST COMMAND` interface.
Node rank 0 is the first host of `--host`. `--allow-run-as-root` is for root only.

```bash
export TLLM_FAULT_TOLERANCE_SHARED_SECRET=$(openssl rand -hex 32)
export TLLM_FAULT_TOLERANCE_RENDEZVOUS=$(hostname):29600
export TRTLLM_EPLB_SHM_NAME=trtllm_eplb_${SLURM_JOB_ID}
HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed 's/$/:4/' | paste -sd, -)
mpirun --enable-recovery --allow-run-as-root \
    --prtemca plm ssh --prtemca plm_ssh_agent "<agent>" --prtemca plm_ssh_no_tree_spawn 1 \
    --host "$HOSTS" -np 8 --map-by ppr:1:node --bind-to none \
    -x TLLM_FAULT_TOLERANCE_SHARED_SECRET -x TLLM_FAULT_TOLERANCE_RENDEZVOUS -x TRTLLM_EPLB_SHM_NAME \
    -x PATH -x LD_LIBRARY_PATH \
    trtllm-llmapi-launch trtllm-serve <model> --host 0.0.0.0 --tp_size 32 --ep_size 32 --gpus_per_node 4 \
    --config config.yaml
```

Kubernetes is not tested yet. Run one pod per node in a group that is not recreated when one pod restarts, and let
the pod's container restart, which restarts a lost node's supervisor. Set `PMI_RANK` to the node rank, `PMI_SIZE` to
the node count and the rendezvous to the leader pod's address. Only the leader pod (rank 0) serves HTTP, so point
the Service and HTTP probes at it only. A `/dev/shm` `emptyDir` with `medium: Memory` counts against the pod's memory
limit. Open the rendezvous port and the next one on the leader pod to the other pods, and multi-node NVLink between
the pods.

The host weights are named `<TRTLLM_EPLB_SHM_NAME>_<deployment id>_l<layer>_lr<local rank>_all` in `/dev/shm`
(`moe_shared_...` when the variable is unset). The node supervisors remove them when the deployment ends, but they
stay after a cancelled or timed-out job. Prefer a `/dev/shm` per job (Slurm's `job_container/tmpfs`, or a container's
own `--shm-size` without `--ipc=host`). With the host's `/dev/shm`, give each job its own `TRTLLM_EPLB_SHM_NAME` and
remove its segments in a Slurm epilog: `rm -f /dev/shm/trtllm_eplb_${SLURM_JOB_ID}_*`.

## Requests, health and monitoring

- Requests in flight on the surviving ranks continue by recompute, and tokens already streamed are kept.
- A request of the failed rank runs again on a surviving rank if it has not streamed output yet. Otherwise it fails
  with HTTP 503 and the error `code` `rank_failure`, or when streaming with a final event
  `data: {"error": {"message": ..., "type": "server_error", "code": "rank_failure", "param": null}}` followed by
  `data: [DONE]`. Send it again.
- `GET /health` answers 200 while the deployment serves, degraded or not, and 503 once it cannot recover, after which
  the server shuts down. Neither 503 carries a `Retry-After` header. Use `/health` for readiness and liveness, not
  `/health_generate`, which can fail during a pause.
- While degraded, the survivors also hold the failed ranks' experts in device memory taken from their KV cache, so
  fewer requests fit. Startup fails if they cannot lend enough for the largest recoverable failure (add ranks, use
  fewer EPLB slots or raise `free_gpu_memory_fraction`).
- `GET /fault_tolerance_status` returns `state`, `authority_alive`, `generation` (the membership generation),
  `active_ranks`, `failed_ranks`, `incarnations` (per rank, 0 for its first process and one more per replacement),
  `replacements` (`{rank: "pending" | "given_up"}` for the failed ranks) and `last_transaction` (`outcome`,
  `removed_ranks`, `added_ranks` and Unix-ms timestamps). A load balancer can weight a replica by its active ranks.

| State | Meaning | Duration or alert |
| --- | --- | --- |
| `healthy` | Every rank serves | |
| `recovering` | A membership change is in progress and serving pauses | A few seconds after a crash, up to 65 s after a hang, about 1-2 s when a replacement joins |
| `degraded` | Some ranks failed and the rest serve | Until the replacement has loaded (about 2 min for DeepSeek-V3-Lite, 10 min for DeepSeek-R1); alert when longer or on `given_up` |
| `unprotected` | The rendezvous store does not answer, so a further failure is not recovered | Alert when it lasts |
| `failed` | The deployment cannot recover, also once the authority (which runs with rank 0) ended | Alert |

With `return_perf_metrics`, `/prometheus/metrics` reports the last status read, prefixed `trtllm_fault_tolerance_`:
`state` (1 for the current state), `active_ranks`, `failed_ranks`, `membership_changes_total` by `kind` (`failover`,
`admission`, `abort`) and `last_change_pause_seconds`.

The launcher's output has these log lines:

| Log line | Meaning |
| --- | --- |
| `Rank <r> (...) ended with status <s>` | A rank's process ended |
| `Rank <r> is hung. It made no progress for ...` | A hung rank is removed |
| `The supervisor of node <n> left, so its ranks [...] count as failed` | A node is lost |
| `Starting a replacement of rank <r>` / `Node <n> (<host>) failed. Relaunching its supervisor.` | Recovery started |
| `Rank <r> resumed serving with membership generation <g>, ranks [...]` | Serving resumed after a change |
| `<n> replacements of rank <r> failed in a row because ...` / `The replacement of rank <r> cannot start, nor could another one` | No replacement starts again (`given_up`) |
| `Deployment aborted: <reason>` | The deployment ends |

## Limitations

- Supported: one NVLink domain (such as GB200 NVL72), attention DP and expert parallelism over all ranks,
  `NVLinkOneSided` MoE communication, the Cutlass MoE backend, or for NVFP4 an explicit `backend: CUTEDSL` (not with
  `enable_locality_domains` or on SM107), standard EPLB, KV cache manager V2, the overlap scheduler, the default
  attention-DP router, one serving frontend and the MPI executor. `modeling_v2: auto` builds the built-in model.
- Required: `max_beam_width: 1`, `context_parallel_size: 1`, `gpus_per_node` equal to the ranks per node, and
  `TRTLLM_MOE_COMM_TIMEOUT_SEC`, if set, of at least 120 s.
- Rejected: pipeline parallelism, TRTLLM-Gen, DeepGEMM (FP8 DeepSeek-R1) and other MoE backends (not supported yet),
  two-model speculation (such as Eagle3 or draft-target), NGram, user-provided and DSpark drafters (one-model MTP
  works), disaggregated serving, KV cache connectors and events, sleep mode, `enable_lm_head_tp_in_adp`, userbuffers,
  `modeling_v2: require`, hybrid linear-attention models, and requests with `n > 1`, `best_of > 1` or beam search.
- One failure at a time (one node, or one rank on rank 0's node). A second one before the replacement joins ends the
  deployment.
- Hardware failures are not covered. The loss of rank 0, its node or the authority ends the deployment, with a
  `Deployment aborted` or `Rank 0 ended` log line and `/health` failing. Restart it from outside (on Slurm, submit
  the job again). If a GPU fails, its replacements fail too, and the node supervisor stops starting them after 3
  failed replacements in a row (`given_up`). Drain or repair the node and restart the deployment.
- The authority listens on the rendezvous port and the next one on rank 0's node. Run on a trusted network.

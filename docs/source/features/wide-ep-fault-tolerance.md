<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Wide-EP Fault Tolerance (Prototype)

```{warning}
Wide-EP fault tolerance is a prototype and is subject to change.
```

A wide expert-parallel (Wide-EP) deployment spreads the experts of every MoE layer over all its ranks, so without
fault tolerance one failed rank ends the whole deployment. With `enable_wide_ep_fault_tolerance`, a rank failure goes
through these phases:

1. Failover: the surviving ranks remove the failed rank, and restore its experts from host weights into spare EPLB
   slots.
2. Degraded: the surviving ranks serve at reduced capacity on the same CUDA graphs.
3. Replacement: a new process of the failed rank starts on the same GPU.
4. Rejoin: the replacement is added back to the deployment.
5. Recovered: the deployment serves at full capacity again.

Recovery is the whole path. On this page, N is the number of ranks of a deployment and K the number of failed ranks,
so a degraded deployment serves on N-K ranks. Rank 0 starts the membership authority, the process that decides which
ranks serve and runs each membership change.

A rank fails when its process exits (a crash, `kill -9`, or the loss of every process of the deployment on its node),
when its node is cut off from the network for 60 s, or when it makes no progress for 60 s while other ranks wait for it.
This 60 s hang timeout is fixed. A hung rank is removed about 60.5 s (the timeout plus one 0.5 s report interval)
after it stopped, or after other ranks started waiting for it if that came later. A slow rank that still makes
progress has not failed. To remove it, end its process.

## Prerequisites

- GB200 NVL72 nodes (aarch64), with every GPU of the deployment in one NVLink domain, also called a multi-node NVLink
  (MNNVL) domain. On Slurm, allocate the nodes in one domain, for example with `--segment=<nodes>` where the cluster
  supports it.
- IMEX running on the nodes, and an IMEX channel in the container. Check that `/dev/nvidia-caps-imex-channels` exists
  in the container, and mount it when launching the container if it does not (see the
  [Wide-EP example](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep)). On Slurm, the cluster's IMEX
  setup gives each job its channel. If the path is missing there, ask your cluster admin. Without a channel, startup
  fails asking for MNNVL memory shared through fabric handles.
- The TensorRT-LLM release container (`nvcr.io/nvidia/tensorrt-llm/release`) of a release that includes this feature.
  On several nodes, it also provides the Open MPI 5 `mpirun` that the launch needs (see
  [Launching on several nodes](#launching-on-several-nodes)).
- `tensor_parallel_size` and `moe_expert_parallel_size` (`--tp_size` and `--ep_size`) equal to the world size, with
  attention DP and KV cache manager V2 (see [Configuration](#configuration)).

On one node, `trtllm-serve` and the LLM API need no launcher and no environment variable. On several nodes, every node
needs the same shared secret and rendezvous address in its environment (step 1 of
[Launching on several nodes](#launching-on-several-nodes)).

## Scope

The prototype covers a narrow first target. What it supports is tested on GB200 NVL72 nodes, or pending a test:

| Supported | Tested |
| --- | --- |
| DeepSeek-V3-Lite with the Cutlass MoE backend, or NVFP4 with `backend: CUTEDSL` | 4 ranks on one node: a rank crash, then a rank hang, each followed by its replacement's rejoin |
| DeepSeek-V3-Lite NVFP4 with the Cutlass MoE backend (`backend: CUTLASS`, which `AUTO` selects for NVFP4) | Pending: 4 ranks on one node, a rank crash followed by its replacement's rejoin |
| DeepSeek-R1 NVFP4 with `backend: CUTEDSL` and static EPLB | Pending: 32 ranks on 8 nodes |
| The loss of every process of a node (serving goes on at reduced capacity, and is back at full capacity only on the next restart) | Pending: DeepSeek-V3-Lite on 8 ranks on 2 nodes |
| Online EPLB (`layer_updates_per_iter` above 0) | DeepSeek-V3-Lite on 4 ranks only, not at DeepSeek-R1 scale |
| Serving at reduced capacity on the same CUDA graphs after a rank failure, then back to full capacity once a replacement rejoins on the same GPU | Serving resumes after a crash, after a hang once the hang timeout has passed, and after a replacement rejoins |
| Requests in flight continue by recompute and keep their streamed tokens. A failed rank's request runs again if it streamed nothing, and otherwise fails with `rank_failure` | No accuracy loss in any phase on GSM8K |
| `/health`, `/fault_tolerance_status` and the `trtllm_fault_tolerance_` metrics | `/health` and `/fault_tolerance_status` in every test above, and `trtllm_fault_tolerance_membership_changes_total` after each rejoin. The other metrics are covered by unit tests |
| `mpirun --enable-recovery` on several nodes; `trtllm-serve` or the LLM API on one node | Both |

Fault tolerance rejects at startup, with the reason, what it does not support:

| Not supported | Why |
| --- | --- |
| Without attention DP, with pipeline parallelism, or with tensor or expert parallelism not over every rank | A failed rank must take only its own requests and hold whole experts, and the host collectives use the tensor-parallel group |
| `max_beam_width` above 1 | Continuing an interrupted request by recompute does not restore its beams yet |
| `disable_overlap_scheduler: true` | Only the overlap scheduler's loop accepts or discards each step yet |
| Without `enable_chunked_prefill`, a `max_seq_len` that is unset or above `max_num_tokens` | An interrupted request continues by recompute as one context of up to `max_seq_len` tokens, and without chunked prefill the scheduler never schedules a context longer than `max_num_tokens` |
| No EPLB, or per-layer dynamic EPLB; online EPLB without GDRCopy | Spare slots take a failed rank's experts; online EPLB writes them from the CPU through GDRCopy |
| KV-balanced admission, KV-aware routing or conversation affinity in `attention_dp_config` | Not supported yet: they keep per-rank state that a replacement does not share |
| KV cache events, KV cache connectors, KV cache compression, KV cache manager V1 | Not supported yet: removal events for a failed rank's blocks, saving only accepted steps, and resizing compressed pools are missing; only V2 holds reuse commits until a step is accepted |
| NGram, user-provided and DSpark drafters | Not supported yet: their drafting state is not rolled back when a failure discards a step |
| Sleep mode | Not supported yet: sleep releases and restores the weights' memory, whose spare slots fault tolerance backs and unbacks itself |
| Userbuffers, `enable_lm_head_tp_in_adp`, disaggregated serving | Userbuffers exchange handles over an MPI world of every rank; a failed rank would take its vocabulary share; KV transfers do not follow a membership change |
| TRTLLM-Gen and DeepGEMM (so FP8 DeepSeek-R1, which runs TRTLLM-Gen by default on GB200), CuteDSL with locality domains or on SM107, and other MoE backends without spare-slot tuning | Their tuner profiles the unbacked spare slots |
| DeepEP, MegaMoE and other MoE communication besides `NVLinkOneSided`; `TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT=1` | They lack the active rank mask, the abort state, the fence path (the all-to-all synchronization mode that honors both) or the late-write remap, which remaps a failed rank's peer memory at the same addresses so that its late writes stay out of the survivors' memory |
| `TLLM_FAULT_TOLERANCE_MODE=1` | It gives the MoE layers an EP group health of their own, which the option replaces |
| Architectures other than `DeepseekV3ForCausalLM`; `modeling_v2: require` | Only DeepSeek-V3 and DeepSeek-R1 are tested, and no `modeling_v2` model supports fault tolerance |
| `TLLM_DISABLE_MPI`, the Ray and RPC executors, no MNNVL or no fabric memory handles, a `torch.distributed` process group created at startup, the server started on every rank of an MPI job | The membership comes with the MPI executor; host collectives and the late-write remap use MNNVL memory; every rank is its own process, so a replacement could not rejoin such a group |
| Node supervisors started by `srun` | Slurm's PMIx plugin cancels the job step when one of them exits |
| On several nodes, `mpirun` on another host than node 0's, or without `--enable-recovery` on its command line | Without recovery the first rank exit ends the job, and node 0 ends a failed job through its `mpirun` |
| An all-to-all timeout under 120 s, too little `/dev/shm` for the host weights, a KV cache that cannot lend the largest failure's restore | The all-to-all must outlast the hang timeout twice; the resources a recovery needs are checked up front |

Requests with `n > 1`, `best_of > 1` or `use_beam_search` are rejected when submitted, with HTTP 400, for the same
reason as `max_beam_width` above 1.

Kubernetes is not supported. Launchers other than `mpirun` are not tested yet, though startup does not reject them.
Fault tolerance can lower throughput while no rank fails, and no bound on that cost is guaranteed yet. After
the loss of every process of a node, the deployment fails over and serves degraded, and recovers only on the next
restart. See [Limitations](#limitations) for this and the other limits.

Planned next steps, each an extension of an existing interface:

| Next step | Extends |
| --- | --- |
| Returning a lost node to full capacity, by the deployment's own path (Slurm, or the NVIDIA Dynamo serving framework on Kubernetes) | The node supervisor |
| Kubernetes, and restarting a rank on another GPU or node | The node supervisor and the rendezvous |
| Beam search, `n > 1` and `best_of > 1` | Recompute of interrupted requests |
| The non-overlap executor loop, and NGram, user-provided and DSpark drafters | Step acceptance, which discards a failed step and rolls back its state |
| KV-balanced admission, KV-aware routing and conversation affinity | The attention-DP router |
| KV cache events, connectors and compression | KV cache manager V2's deferred reuse commit and pool resize |
| Sleep mode | The spare-slot storage that fault tolerance backs and unbacks |
| More MoE backends (TRTLLM-Gen and DeepGEMM for FP8 DeepSeek-R1, CuteDSL locality domains) | Spare-slot tuning (`supports_spare_slots`) |
| More MoE communication (DeepEP, MegaMoE) | The execution-state binding: the active rank mask, the abort state and the late-write remap |
| A measured healthy cost within 1%, and the fixes it calls for | The host-collective implementation behind `Distributed` |
| Online EPLB with host weights on two nodes per expert instead of every node | The EPLB host-weight sharer |
| A restartable authority | The membership authority and its clients |
| More than one NVLink domain | An RDMA all-to-all that keeps late writes out by revoking memory keys, and a network host-collective transport |

## Configuration

A DeepSeek-V3-Lite deployment on one node of 4 GPUs (4 ranks):

```yaml
enable_wide_ep_fault_tolerance: true
enable_attention_dp: true
max_batch_size: 32
max_seq_len: 4096  # at most max_num_tokens (8192 by default), or enable_chunked_prefill: true
cuda_graph_config:
  enable_padding: true
  max_batch_size: 32
kv_cache_config:
  free_gpu_memory_fraction: 0.5
  use_kv_cache_manager_v2: true
speculative_config:
  decoding_type: MTP
  num_nextn_predict_layers: 1
moe_config:
  load_balancer:
    num_slots: 64  # one slot per routed expert; fault tolerance adds the spare slots
    layer_updates_per_iter: 0
```

```bash
trtllm-serve <model> --tp_size 4 --ep_size 4 --config config.yaml
```

- Static EPLB (`layer_updates_per_iter: 0`) keeps the initial expert placement and needs nothing more. Online EPLB
  (`layer_updates_per_iter` above 0) needs CDMM mode (Coherent Driver-Based Memory Management, where the driver
  rather than the OS manages GPU memory, a driver setting of the node) and GDRCopy, with `/dev/gdrdrv` mapped into
  the container (see the [Wide-EP example](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep)).
  Unlike online EPLB without fault tolerance, it does not fall back to managed memory or to GPU memory exposed as
  NUMA nodes when CDMM is off or GDRCopy is missing. Startup fails then.
- The survivors keep the CUDA graphs they captured at startup. After the loss of K of N ranks, each survivor's batch
  grows by about N/(N-K), so size `max_batch_size` with that headroom and set `cuda_graph_config.max_batch_size` to
  the same value. A batch larger than the largest captured graph runs eager. With `torch_compile_config` or
  `prefill_cuda_graph_backend: piecewise`, set `enable_userbuffers: false` in `torch_compile_config`.
- On one node, `trtllm-serve` and the LLM API start the ranks themselves and need no launcher and no environment
  variable. They generate the shared secret and pick a rendezvous port. The LLM API takes the same option.
- The option alone turns the feature on. Leave `TLLM_FAULT_TOLERANCE_MODE` unset. Startup fails if it is set to `1`
  with this option.
- `num_slots` counts the active EPLB slots. Fault tolerance adds unbacked spare slots to every rank, enough to take
  the experts of the largest recoverable failure. On M nodes of G GPUs (N = M x G ranks) that is the loss of a node,
  and each rank gets ceil((num_slots / M) / ((M - 1) x G)) spare slots per MoE layer. On one node (N = G ranks) it is
  the loss of one rank, and each rank gets ceil((num_slots / N) / (N - 1)). Rank 0 logs the count at startup:
  `Wide-EP fault tolerance adds up to <k> unbacked spare slots per rank`.

## Launching on several nodes

1. Give every node the same shared secret and rendezvous address: export them on node 0 and pass them to every node
   with `mpirun -x`, as in the example below. The membership authority accepts only processes that hold the shared
   secret, and the secret also names the deployment's host weights. Set `TLLM_FAULT_TOLERANCE_SHARED_SECRET` to at
   least 32 bytes as hex (64 hex digits), such as the output of `openssl rand -hex 32`, in the environment and never
   on the command line, where other users of the host can read it (`ps`, `/proc`). Generate a new secret for every
   deployment and never reuse one across deployments. Set
   `TLLM_FAULT_TOLERANCE_RENDEZVOUS` to `host:port`: a host name or IP address of node rank 0 that every node reaches,
   and a free port. A host that resolves to a loopback address, such as a host name that `/etc/hosts` maps to
   `127.0.1.1`, fails at startup. Deployments that run at the same time on the same nodes need different secrets
   and different rendezvous ports. The authority of an ended deployment keeps its port for up to 60 s. A deployment
   started in that time with the same secret and rendezvous address fails with `Use a new shared secret`, and one
   with a new secret on the same port fails because the port is taken. Use a new secret and a new port, or wait.
2. Size `/dev/shm` on every node. With static EPLB, each node keeps the host weights of the experts placed on the
   next node (node i keeps those of node i+1, and the last node those of node 0), so that the experts of a lost node
   still have a host copy. A node then needs the experts of one node times the MoE layers times the bytes of one
   expert, about 46 GB for DeepSeek-R1 NVFP4 on 8 nodes. With online EPLB, every node keeps every routed expert,
   about 364 GB for DeepSeek-R1 NVFP4. In a container with its own `/dev/shm`, set `--shm-size` to match. With
   Slurm, a per-job tmpfs (`job_container/tmpfs`) also gives each job its own `/dev/shm`.
3. Start `trtllm-llmapi-launch` once per node with `mpirun --enable-recovery` of Open MPI 5, which keeps the other
   processes running when one exits. The exported secret makes it start a node supervisor, which starts one rank per
   GPU that it sees (those in `CUDA_VISIBLE_DEVICES`, or every GPU of the node when it is unset), and a replacement
   on the same GPU for a rank that fails. Set `--gpus_per_node` to that count, or startup fails. Without
   `enable_wide_ep_fault_tolerance`, those ranks refuse to start. The TensorRT-LLM release container provides Open
   MPI 5 in `/usr/local/mpi/bin`, from the HPC-X of its PyTorch base image. Elsewhere, install Open MPI 5 (Ubuntu's
   `openmpi-bin` is Open MPI 4, which has no `--enable-recovery`), and check that `ompi_info --version` reports v5.

Node rank 0 is the first host of `--host`. Node 0's supervisor ends a failed deployment's job through `mpirun`, which
then exits with a failure status. On several nodes:

- Run `mpirun` on node 0's host, with `--enable-recovery` on its command line even if PRRTE's parameter files also
  set recovery. Without it, the first rank exit ends the job. Node 0's supervisor fails at startup when `mpirun`
  runs on another host or lacks the flag.
- On Slurm, run `mpirun` in a `--no-requeue` allocation. Node supervisors started by `srun` are rejected, since Slurm
  cancels the whole job step when one of its tasks exits.
- Other launchers, such as Kubernetes, are not supported or tested yet.

The example below runs DeepSeek-R1 NVFP4 on 8 GB200 nodes of 4 GPUs (32 ranks), with this `config.yaml`. HTTP is
served on node 0.

```yaml
enable_wide_ep_fault_tolerance: true
enable_attention_dp: true
max_batch_size: 128
max_seq_len: 2304  # at most max_num_tokens (8192 by default), or enable_chunked_prefill: true
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
  use_low_precision_moe_combine: true
  load_balancer:
    num_slots: 256  # one slot per routed expert; fault tolerance adds the spare slots
    layer_updates_per_iter: 0
```

```bash
# On node0
export TLLM_FAULT_TOLERANCE_SHARED_SECRET=$(openssl rand -hex 32)
export TLLM_FAULT_TOLERANCE_RENDEZVOUS=node0:29600  # node 0's name or address that every node reaches
mpirun --enable-recovery \
    --host node0,node1,node2,node3,node4,node5,node6,node7 -np 8 --map-by ppr:1:node --bind-to none \
    -x TLLM_FAULT_TOLERANCE_SHARED_SECRET -x TLLM_FAULT_TOLERANCE_RENDEZVOUS \
    trtllm-llmapi-launch trtllm-serve <model> --host 0.0.0.0 --tp_size 32 --ep_size 32 --gpus_per_node 4 \
    --config config.yaml
```

Choose the rendezvous port outside the hosts' ephemeral port range (`net.ipv4.ip_local_port_range`, often
32768-60999). Rank 0's membership authority binds it only when it starts, and an outgoing connection can take a port
in that range first. The server warns at startup when the port is in that range.

When the processes run as root outside the TensorRT-LLM containers, add `--allow-run-as-root`. The containers allow
it already, through `OMPI_ALLOW_RUN_AS_ROOT` and `PRTE_ALLOW_RUN_AS_ROOT`. With containers, start `mpirun` in the
first node's container and let PRRTE start its daemons in the other nodes' containers. Either run `sshd` in every
container for PRRTE's default ssh launcher, or, on Slurm, use PRRTE's own Slurm launcher (`--prtemca plm slurm`),
which starts the daemons with `srun`. Neither setup is tested with fault tolerance yet.

The rendezvous store assumes a trusted job network. Every value in the store is signed with the shared secret, and a
process checks the signature before it uses the value. The store's arrival counters are not signed. The store
accepts any process that reaches its port, which can read, delete or write back values and so disrupt the
deployment. It does not encrypt its traffic.

To stop the deployment, send SIGINT to the `trtllm-serve` process on node 0. The launcher's output names its PID in
the line `Rank0 task guard PID: <guard PID>; workload PGID: <server PID>` (not the `Rank0 server guard PID` line of
rank 0's worker):

```bash
# On node0
kill -INT <server PID>
```

Without the launcher's output, the same process is the oldest of the user's processes whose command line starts with
`trtllm-serve`, or with the Python interpreter that runs it. The pattern below matches neither `mpirun`, the node
supervisor nor `trtllm-llmapi-launch`, whose command lines also hold `trtllm-serve`. With several serving frontends,
`-o` picks frontend 0, which started first:

```bash
# On node0
pkill -INT -o -u "$(id -u)" -f '^([^ ]*python[^ ]* )?([^ ]*/)?trtllm-serve( |$)'
```

Rank 0 then drains the requests in flight, with every rank taking part, and shuts the deployment down, and `mpirun`
ends on its own. Send the signal to that process only: a rank that ends before the shutdown counts as a failure, and
a SIGTERM to `mpirun` does not drain (see [Limitations](#limitations)). Whether Ctrl-C in `mpirun`'s terminal drains
is untested. A node supervisor that receives SIGINT or SIGTERM ends the ranks of its node that still run 10 s after
the signal with their process groups, within 20 s of the signal. Node 0's supervisor first forwards the signal to
rank 0, which drains and shuts the deployment down within that time. On another node, the signal is the loss of that node, and the
deployment serves on without its ranks.
Whether Slurm's own signals at a cancellation or a time limit also reach the ranks is untested.

The host weights are named `<TRTLLM_EPLB_SHM_NAME>_<hash>_l<layer>_lr<local rank>_all` in `/dev/shm`
(`moe_shared_...` when the variable is unset), where `<hash>` stands for a hash of the shared secret. The node
supervisors remove them when the deployment shuts down, but they stay after a cancelled, killed or timed-out job. Prefer a `/dev/shm` per job (Slurm's `job_container/tmpfs`, or a container's own `--shm-size` without
`--ipc=host`). With the host's `/dev/shm`, give each job its own `TRTLLM_EPLB_SHM_NAME` (such as
`trtllm_eplb_${SLURM_JOB_ID}`, passed with `-x`) and remove its host weights in a Slurm epilog:
`rm -f /dev/shm/trtllm_eplb_${SLURM_JOB_ID}_*`.

## Requests, health and monitoring

- Requests in flight on the surviving ranks continue by recompute, and tokens already streamed are kept.
- Requests sent during a pause wait in the queue and are served when it ends.
- A request of the failed rank runs again once on a surviving rank if it has not streamed output yet. Otherwise it fails
  with HTTP 503 and the error `code` `rank_failure`. When streaming from `/v1/chat/completions` or `/v1/completions`,
  it ends with a final event
  `data: {"error": {"message": ..., "type": "server_error", "code": "rank_failure", "param": null}}` followed by
  `data: [DONE]`. Discard the partial output and send the request again. A `/v1/responses` stream ends with an
  `error` event of `code` `rank_failure`, then `response.failed`. The Anthropic `/v1/messages` endpoint reports the
  same failure, with the rank failure's message, as HTTP 503 with an error of type `api_error`, or as a stream
  `error` event of type `api_error`.
- A request pinned to a failed rank (`attention_dp_rank` without `attention_dp_relax`) runs again once, like the
  failed rank's other requests. If the rank still does not serve then, the request fails with HTTP 503 and
  `rank_failure` while a replacement of the rank may still rejoin, so send it again later. Once the rank's
  replacements are given up (`given_up`), it fails with HTTP 400, without the `rank_failure` code, instead. Send it without `attention_dp_rank` or with
  `attention_dp_relax`.
- A request that crashes its rank before streaming output is re-run once on a survivor. If it crashes that rank too,
  that is a second failure before the replacement rejoins (see Limitations), which usually ends the deployment.
- `GET /health` answers 200 while the deployment serves, degraded or not, and 503 once the server shuts down or rank 0
  ended, which rank 0 does when the deployment cannot recover or shuts down. A failed deployment's server starts
  shutting itself down within about 1 s, once its status read finds the `failed` state, and then exits, whether `trtllm-llmapi-launch` or plain
  `trtllm-serve` started the ranks, so an orchestrator sees the process exit. The 503 carries no `Retry-After`
  header. Use `/health` for liveness, not `/health_generate`, which can fail during a pause.
- A hung rank stalls every request for about 60 s (the hang timeout) before it is removed, while `/health` answers
  200. The state reads `stalled` once the stall lasts 30 s, half the hang timeout, so that one long step, such as a
  long prefill, does not read `stalled`. Set client and load-balancer timeouts above 70 s. With several replicas,
  take a replica out of rotation while `/fault_tolerance_status` reports `recovering` or `stalled`, or a
  `stalled_seconds` above a few seconds.
- While degraded, the survivors also hold the failed ranks' experts in device memory taken from their KV cache, so
  fewer requests fit. Startup fails if they cannot lend enough for the largest recoverable failure, the loss of a
  node other than rank 0's or of one rank (add ranks, use fewer EPLB slots or raise `free_gpu_memory_fraction`).
  Each survivor lends at most its spare slots times the MoE layers times the bytes of one expert. For DeepSeek-R1
  NVFP4 (58 MoE layers, about 26 MB per expert) with `num_slots: 256`, that is 32 spare slots and about 49 GB per GPU
  on 2 nodes of 4 GPUs, and 2 spare slots and about 3 GB per GPU on 8 nodes.
- `GET /fault_tolerance_status` returns `state`, `authority_alive`, `stalled_seconds` (how long the active ranks have
  waited on a rank that makes no progress, 0 when none), `generation` (the membership generation), `active_ranks`,
  `failed_ranks`, `incarnations` (per rank, 0 for its first process and one more per replacement), `replacements`
  (`{rank: "pending" | "given_up"}` for the failed ranks) and `last_transaction`, the last ended membership change:
  `generation`, `outcome` (`COMMITTED` or `ABORTED`), `removed_ranks`, `causes` (`{rank: "disconnected" | "hung"}`),
  `added_ranks`, the Unix-ms timestamps `detected_unix_ms` (when the first removed rank stalled or disconnected, or
  when the first replacement was ready), `began_unix_ms`, `ready_unix_ms` (null when no replacement was ready) and
  `ended_unix_ms`, `changes` (the number of ended changes by kind, `failover`, `admission` (rejoin) and `abort`, this
  one included) and the abort `reason`. Rank keys are JSON strings. A load balancer can weight a replica by its active
  ranks.
  This endpoint and the metrics answer from the status that the server reads every second, so a slow rendezvous
  store never delays them. With several serving frontends, only the launcher (frontend 0) reads the status,
  and the other frontends forward this endpoint and `/prometheus/metrics` to it. They answer `/health` themselves and
  exit about 15 s after the launcher's `/health` fails.

| State | Meaning | Duration or alert |
| --- | --- | --- |
| `healthy` | Every rank serves | |
| `recovering` | A membership change is in progress and serving pauses | A few seconds after a crash or a detected hang (a hang reads `stalled` first), about 1-2 s when a replacement rejoins |
| `degraded` | Some ranks failed and the rest serve | Until the replacement has loaded (about 2-3 min for DeepSeek-V3-Lite; longer for larger models); alert when longer or on `trtllm_fault_tolerance_replacements{state="given_up"}` |
| `stalled` | The active ranks have waited 30 s (half the hang timeout) or longer on a rank that makes no progress, and serving pauses | Until the rank is removed at the hang timeout (60 s), then `recovering`; alert |
| `unprotected` | No membership was read yet, or the rendezvous store answered no read for 30 s (when the ranks take the authority as hung), so a further failure is not recovered | Until the store first answers after the server starts; alert otherwise |
| `failed` | The deployment cannot recover, also once the authority (which runs with rank 0) ended without a shutdown | Alert |
| `stopped` | The deployment shut down without a failure | Until the server exits |

`/prometheus/metrics` reports the last status read, prefixed `trtllm_fault_tolerance_` and labeled `model_name` and
`engine_type`, also without `return_perf_metrics`: `state` (1 for the current state), `stalled_seconds`,
`active_ranks`, `failed_ranks`, `replacements` by `state` (`pending`, `given_up`), `rank_restarts_total` (the sum of
the incarnations), `membership_changes_total` by `kind` (`failover`, `admission`, `abort`) and
`last_change_pause_seconds` (from the detection to the end of the last change, so a hang's stall included).
`/fault_tolerance_status` and the `trtllm_fault_tolerance_*` metrics can lag a membership change by up to about 1 s.

The launcher's output has these log lines:

| Log line | Meaning |
| --- | --- |
| `Rank <r> (...) ended with status <s>` | A rank's process ended, with its exit status, or minus the signal that ended it (such as `-9` for SIGKILL) |
| `Rank <r> is hung. It made no progress for ...` | A hung rank is removed |
| `The supervisor of node <n> left, so its ranks [...] count as failed` | A node's supervisor and ranks are gone |
| `Killing rank <r> ...` / `Killing the replacement of rank <r> ...` | A rank left out of the membership, or a stalled replacement, is ended |
| `Starting a replacement of rank <r>` | A replacement starts |
| `Rank <r> resumed serving with membership generation <g>, ranks [...]` | Serving resumed after a change |
| `<k> replacements of rank <r> failed in a row, ...` / `The replacement of rank <r> cannot start, nor could another one` | No replacement starts again (`given_up`) |
| `Deployment aborted: <reason>` / `Rank 0 ended, ...` / `Rank <r> failed before serving ...` / `Ending the job: <reason>` | The deployment ends |
| `Cannot end the job: <error>` | The job keeps its allocation with no deployment. Cancel it |

## Limitations

- Supported: one NVLink domain (such as GB200 NVL72, with IMEX running and an IMEX channel in the container, see
  [Prerequisites](#prerequisites)), attention DP and expert parallelism over all ranks, `NVLinkOneSided` MoE
  communication (`TRTLLM_FORCE_COMM_METHOD=NVLINK_ONE_SIDED` selects it), the Cutlass MoE backend (which `AUTO`
  selects for BF16 and NVFP4), or for NVFP4 also
  `backend: CUTEDSL` (not with `enable_locality_domains` or on SM107), standard EPLB, KV cache manager V2, the overlap scheduler, the default attention-DP router, several serving frontends and the MPI executor.
  `modeling_v2: auto` builds the built-in model.
  - Models: DeepSeek-V3 and DeepSeek-R1 (`DeepseekV3ForCausalLM`). Other architectures are rejected when the model
    loads. DeepSeek-R1 FP8 is not supported, because its MoE backends (TRTLLM-Gen by default on GB200, or DeepGEMM)
    tune the unbacked spare slots. Use the NVFP4 checkpoint with the CuteDSL or Cutlass MoE backend.
- Required: `tensor_parallel_size` and `moe_expert_parallel_size` equal to the world size, `max_beam_width: 1`,
  `context_parallel_size: 1`, `gpus_per_node` equal to the GPUs that each node supervisor sees (one rank per GPU),
  and, if set, `TRTLLM_MOE_COMM_TIMEOUT_SEC` of at least 120 s.
- Rejected at startup: pipeline parallelism, TRTLLM-Gen, DeepGEMM (the FP8 DeepSeek-R1 backends) and other MoE
  backends (not supported yet), two-model speculation (such as Eagle3 or draft-target), NGram, user-provided and
  DSpark drafters (one-model MTP works), disaggregated serving, KV cache connectors, events and compression
  (`kv_cache_compression_config`), sleep mode, `enable_lm_head_tp_in_adp`, userbuffers,
  `TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT=1` (fault tolerance runs on the fence path) and `modeling_v2: require`.
- Rejected when submitted: requests with `n > 1`, `best_of > 1` or `use_beam_search`.
- A failure before the previous replacement rejoined is recovered only if the remaining spare slots still hold every
  expert and the survivors' KV cache can lend the memory. Otherwise the deployment ends.
- With static EPLB, each expert has host weights on one other node only, the node before its own in node order. If
  that node and a rank holding the expert both fail before the first failure's replacement rejoins, the expert is on
  no surviving host, and the deployment ends.
- A replacement loads its checkpoint with `checkpoint_io_policy: native` and logs it if another policy is set,
  because it loads alone and rank-striped read-ahead splits the checkpoint reads among the ranks of a node.
- Kubernetes is not supported. A replacement restarts on the same GPU of the same job; moving a rank to another GPU
  or node is not supported.
- Hardware failures are not covered. A node that crashes, reboots, loses power or fails in Slurm (`NODE_FAIL`) ends
  the deployment, because the launcher's own processes on it die too. Only the loss of the deployment's processes on
  a node is covered.
- After the loss of every process of a node, the deployment fails over and serves without the K ranks of that node
  (N-K of N ranks), and is back at N ranks only on the next restart. Until then, `/fault_tolerance_status` reports the node's ranks as
  `given_up`, the node's host weights stay in its `/dev/shm`, and with static EPLB a later loss of a rank on the
  next node may end the deployment.
- The loss of rank 0, its node or the authority ends the deployment, as does an authority that sends the ranks
  nothing for 30 s (such as a 30 s network or CPU stall on node 0), with a `Deployment aborted` or `Rank 0 ended` log
  line and `/health` failing. Restart it from outside (on Slurm, submit the job again).
- A failure before the deployment serves ends it, as does a rank that during startup does not reach the rendezvous,
  or keeps the others waiting, for 10 min.
- A replacement that makes no progress for 20 s while it rejoins is left out, and one that makes no startup
  progress for 10 min is killed. A rank's replacements stop (`given_up`) after 3 in a row fail for any reason (each
  ended before it served 10 min). A failed GPU leads there. Drain or repair the node and restart the deployment.
- A SIGTERM to `mpirun` ends the deployment without draining, and the requests in flight fail. A SIGTERM to node 0's
  supervisor drains for at most 10 s, after which it ends node 0's ranks. A SIGTERM to another node's supervisor is
  the loss of that node, and the deployment serves on without its ranks. To stop the deployment with a drain, send SIGINT to the `trtllm-serve` process on node 0 (see
  [Launching on several nodes](#launching-on-several-nodes)).
- The authority listens on the rendezvous port and on a port the system assigns on rank 0's node. It accepts only
  processes that hold the shared secret. Every value in its store is signed with that secret and checked before use,
  but the arrival counters are not, and the store accepts any process that reaches its port. Make the rendezvous
  port reachable only on a trusted network.

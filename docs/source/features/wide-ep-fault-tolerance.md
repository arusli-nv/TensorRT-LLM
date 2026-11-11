<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WideEP Fault Tolerance (Prototype)

```{warning}
WideEP fault tolerance is a prototype and is subject to change.
```

A wide expert-parallel (WideEP) deployment spreads the experts of every MoE layer over all its ranks, so without
fault tolerance one failed rank ends the whole deployment. With `enable_wide_ep_fault_tolerance`, a deployment
that loses a rank keeps serving on the remaining ranks:

- A rank fails when its process exits (a crash or `kill -9`) or hangs: other ranks wait for it, and for 30 s it makes
  no progress while it waits for no other rank (rank 0 waiting for client requests is not hung). A hang during the
  recovery from a failure counts the same. Hardware failures and losing rank 0 are not covered.
- The surviving ranks stop at their next collective, drop the failed rank, restore its experts from host weights
  into spare EPLB slots, and continue on the same CUDA graphs at reduced capacity. A crash pauses serving for a
  few seconds; a hang for the hang timeout plus a few seconds.
- Requests in flight on the surviving ranks continue; tokens already streamed are kept. A request of the failed
  rank runs again on a survivor if it has not streamed any output yet; otherwise it fails as retryable: HTTP 503
  (no `Retry-After` header) with the error `code` `rank_failure`, or, for a streamed request that has
  received output, an `error` event with that `code`. Send it again. A request pinned to the failed rank's
  attention-DP rank (`attention_dp_rank` without `attention_dp_relax`) fails.
- A replacement process for the failed rank starts on the same GPU, loads the model on its own and then joins; the
  survivors pause only while it joins (their requests in flight continue by recompute, as at a failure), and the
  deployment is back at full capacity. A replacement that cannot join
  is left out, the survivors keep serving, and another replacement starts after a backoff. This holds where the
  server starts the ranks itself (`trtllm-serve` or the LLM API on one node) and under `trtllm-llmapi-launch` (see
  [Launching](#launching)). Under `trtllm-llmapi-launch`, when a node loses every process of the deployment (its
  ranks and its node supervisor) while the launcher's own per-node processes keep running (for example `prted` or
  the kubelet), its node supervisor and ranks start again (see Launching); losing the launcher's per-node processes
  too is not supported yet.

## Enabling it

In the `trtllm-serve` configuration file:

```yaml
enable_wide_ep_fault_tolerance: true
enable_attention_dp: true
kv_cache_config:
  use_kv_cache_manager_v2: true
moe_config:
  load_balancer:
    num_slots: 256  # EPLB; fault tolerance adds the spare slots that receive a failed rank's experts
```

The LLM API takes the same option, `LLM(..., enable_wide_ep_fault_tolerance=True)`. Launch the ranks as described
in [Launching](#launching).

## Supported configurations

The configuration is checked when the LLM arguments are built, and what depends on the model or the ranks when
they start; anything unsupported is rejected with the reason.

| Requirement | Why |
|---|---|
| One NVLink domain (for example GB200 NVL72); attention DP over all ranks (no attention TP); no pipeline parallelism | Collectives run over NVLink memory that can drop a rank |
| `NVLinkOneSided` MoE communication | The all-to-all that can skip a failed rank |
| Standard EPLB (`moe_config.load_balancer`, static or online, not `per_layer_dynamic`; `num_slots` at least the number of experts); online EPLB also needs GDRCopy for its slots (GPU memory not a NUMA node) | Fault tolerance adds the spare slots that receive a failed rank's experts; they use no GPU memory until a failure. Online EPLB writes the slots with CPU copies, as without fault tolerance |
| Room in `/dev/shm` on every node for the host weights of every expert (DeepSeek-R1 FP4: about 364 GB) | A failed rank's experts are restored from them |
| KV cache manager V2, overlap scheduler | Interrupted requests continue by recompute |
| The default attention-DP routing and admission, with or without `enable_balance` (no `enable_kv_balanced_admission`, `enable_kv_cache_aware_routing` or `kv_cache_routing_conversation_affinity`) | Their per-rank state would differ on a replacement |
| One serving frontend (`num_serve_frontends: 1`, the default with fault tolerance) | The server follows the membership |
| `TRTLLM_MOE_COMM_TIMEOUT_SEC`, if set, at least 60 s (twice the hang timeout) | The all-to-all waits on a hung rank must last until it is dropped |
| Not supported: disaggregated serving, KV cache connectors, KV cache events (`event_buffer_max_size`), sleep mode, `enable_lm_head_tp_in_adp`, `modeling_v2` targets, `torch_compile_config`, NGram and user-provided drafters and DSpark (MTP is supported), hybrid linear-attention models, the Ray and RPC executors (`orchestrator_type`) | |
| Requests: `n > 1`, `best_of > 1` and beam search are rejected | |

## Launching

On one node, `trtllm-serve <model> --tp_size 4 --ep_size 4 --config config.yaml` (or the LLM API) needs nothing
more: the server starts the ranks, chooses the secret and the address below, and restarts a failed rank. Every
rank runs as its own process, so the LLM API takes no MPI session of your own.

The deployment starts once every rank has started; it aborts at once if a rank fails or stops reporting before
then, and if a rank does not reach the rendezvous, at the start of its worker, within 600 s of rank 0.

Otherwise, the launcher must keep the other ranks running when one exits. The startup check rejects ranks started
as Slurm tasks; what the ranks cannot see (`mpirun` without `--enable-recovery`, a LeaderWorkerSet with
`RecreateGroupOnPodRestart`) is stated below.

Every rank needs the same join secret, which signs the membership messages, and the rendezvous address: rank 0's
node (an IP address works where host names do not resolve, as in many Kubernetes clusters) and a free port, where
the rendezvous store listens; the next port must be free too. Where the server does not start the ranks, pass both in
the environment (the secret never on the command line):

```bash
export TLLM_FAULT_TOLERANCE_JOIN_SECRET=$(openssl rand -hex 32)
export TLLM_FAULT_TOLERANCE_RENDEZVOUS=<rank 0's host>:29600
```

With the join secret set, `trtllm-llmapi-launch` is launched once per node: it runs the node supervisor, which
starts the node's ranks (one per GPU, rank = node rank x GPUs per node + GPU index) as its children and starts a
replacement on the same GPU for a rank that fails. A rank's process ends with its node supervisor, so the
supervisor's end is the loss of its node. Under `mpirun`, node 0's supervisor relaunches a lost node's supervisor
through the job's own PRRTE (`prun`); elsewhere, start the node's launch again with the same command on the same
node: a node supervisor started while the deployment serves starts replacements for the node's lost ranks. At most
one node supervisor of a deployment runs on a node. Once the deployment ended, each node supervisor removes its
node's host copies of the expert weights.

`mpirun` (Open MPI) with `--enable-recovery`:

```bash
mpirun --enable-recovery -np 1 --map-by ppr:1:node \
    -x TLLM_FAULT_TOLERANCE_JOIN_SECRET -x TLLM_FAULT_TOLERANCE_RENDEZVOUS \
    trtllm-llmapi-launch trtllm-serve <model> \
    --tp_size 4 --ep_size 4 --config config.yaml
```

Slurm: the ranks must not be Slurm tasks (`srun --mpi pmix`, rejected at startup): when one of them exits, Slurm's
PMIx plugin cancels the whole job step. Start them with `mpirun --enable-recovery` inside the allocation, and submit with
`--no-requeue`:

```bash
#SBATCH --no-requeue
export TLLM_FAULT_TOLERANCE_JOIN_SECRET=$(openssl rand -hex 32)
export TLLM_FAULT_TOLERANCE_RENDEZVOUS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1):29600
mpirun --enable-recovery -np 8 --map-by ppr:1:node \
    -x TLLM_FAULT_TOLERANCE_JOIN_SECRET -x TLLM_FAULT_TOLERANCE_RENDEZVOUS \
    trtllm-llmapi-launch trtllm-serve <model> \
    --tp_size 32 --ep_size 32 --config config.yaml
```

Kubernetes: run all ranks in one pod group that is not recreated when one pod restarts (for a LeaderWorkerSet,
do not use `RecreateGroupOnPodRestart`), one pod per node (the host copies of the expert weights are shared per
pod), with each pod's container running `trtllm-llmapi-launch` once, so that restarting the container starts the
node again; give each pod its node rank in `PMI_RANK` and the node count in `PMI_SIZE`, and set the rendezvous address to the leader pod's
address (`LWS_LEADER_ADDRESS`).

Every replacement starts through a node supervisor: the supervisor of a lost rank's node starts it, and a lost node
comes back by starting its launch again (under `mpirun`, node 0's supervisor does so; on Kubernetes, restart the
node's pod), whose supervisor then replaces each of its ranks. Rank 0 is never replaced.

## Costs and limits

- Host memory: every node keeps the host weights of every expert in `/dev/shm` to restore a failed rank's experts
  (DeepSeek-R1 FP4: about 364 GB per node). Containers default `/dev/shm` to 64 MB: give it room with
  `docker run --shm-size`, or on Kubernetes mount an `emptyDir` volume with `medium: Memory` and a `sizeLimit` at
  `/dev/shm`. The startup check rejects a node whose `/dev/shm` has too little free space, before the weights
  load.
- GPU memory: at a failure, each survivor gives up the part of its KV cache that the restored experts and the
  remapped all-to-all memory need beyond the GPU memory freed since startup; it gets the KV cache back once the
  replacement has joined. A configuration whose largest failure would need more than a survivor's whole KV cache
  is rejected at startup.
- Recompute: the interrupted requests run their prefill again; with KV cache block reuse (the default) their
  prompt blocks are reused.
- Rank 0 and its node are not covered: rank 0 runs the membership authority, a CPU-only process that decides the
  membership. It listens on the rendezvous port (on every address, as main's TCP store does) and on the next port
  (on the rendezvous address only); open these two ports of rank 0's node to the deployment's nodes, and no others,
  in a firewall or network policy. The deployment must run on a trusted network, as with main's TCP store and MPI.
  Without the authority, the deployment serves on unprotected. Once rank 0 is gone, node 0's supervisor ends the job
  under `mpirun`; elsewhere, main's hang detector ends the other ranks.
- Capacity: until the replacement joins, the deployment serves with the remaining ranks only.

## Monitoring

`GET /health` answers 200 while the deployment is alive, with a JSON body whose `state` is `healthy` (every rank
serves), `degraded` (some ranks are lost), `recovering` (a membership change is in progress; serving pauses for
it) or `unprotected` (serving, but the membership authority does not answer, so a further failure would not be
recovered). It answers 503 with the same JSON body once the deployment cannot recover (`stopped`); the server then
shuts down, so an orchestrator restarts the instance. A probe that needs the deployment to serve requests now
should read `state`.

`GET /fault_tolerance_status` reports the membership, read from the membership store at each request:

```json
{
  "state": "degraded",
  "authority_alive": true,
  "generation": 1,
  "membership_state": "COMMITTED",
  "live_ranks": [0, 1, 3],
  "lost_ranks": [2],
  "incarnations": [0, 0, 0, 0],
  "last_transaction": {
    "generation": 1,
    "outcome": "COMMITTED",
    "removed_ranks": [2],
    "added_ranks": [],
    "duration_s": 1.8,
    "phases_ms": {
      "detect_to_propose": 12.0,
      "propose_to_ready": 420.0,
      "ready_to_installed": 1380.0
    },
    "times_unix_ms": {
      "detected": 1762790399988,
      "proposed": 1762790400000,
      "ready": 1762790400420,
      "committed": 1762790401800
    }
  }
}
```

`generation` counts membership changes. `incarnations` counts each rank's replacements. `last_transaction` is the
last membership change that ended (`COMMITTED` or `ABORT`), from its start to its end. `phases_ms` splits it: from
detecting the failure (or the join request) to the first proposal, from there to the last READY, and from there to
the last INSTALLED, on which the authority commits (null for a phase it did not reach). `times_unix_ms` gives when
each phase ended, in milliseconds since the Unix epoch on the authority's host, to line up with the ranks' logs.
The membership authority also logs one line per membership change.

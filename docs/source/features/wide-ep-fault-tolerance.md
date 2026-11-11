<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WideEP Fault Tolerance (Prototype)

```{warning}
WideEP fault tolerance is a prototype and is subject to change.
```

A wide expert-parallel (WideEP) deployment spreads the experts of every MoE layer over all its ranks, so without
fault tolerance one failed rank ends the whole deployment. With `enable_wide_ep_fault_tolerance`, it keeps serving:

```text
healthy (N ranks) --rank fails--> degraded (N-K ranks, same CUDA graphs) --replacement joins--> healthy (N ranks)
```

- A rank fails when its process exits (a crash, `kill -9`, or every process of the deployment on a node) or hangs:
  other ranks wait for it and for 30 s it makes no progress. Hardware failures and losing rank 0 or its node are not
  covered.
- The surviving ranks drop the failed ranks, restore their experts from host weights into spare EPLB slots and
  continue on the same CUDA graphs at reduced capacity. Serving pauses for a few seconds after a crash, and for the
  hang timeout plus a few seconds after a hang.
- A replacement process starts on the same GPU, loads the model on its own and joins. The survivors pause only while
  it joins; their requests in flight continue. Then the deployment is back at full capacity.

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

The LLM arguments are checked when they are built, and the model and the ranks when they start; anything
unsupported is rejected with the reason.

| Requirement | Why |
|---|---|
| One NVLink domain (for example GB200 NVL72); attention DP and expert parallelism over all ranks; no pipeline parallelism | Collectives run over NVLink memory that can drop a rank |
| `NVLinkOneSided` MoE communication on its fence path | The all-to-all that can skip a failed rank |
| Standard EPLB (`moe_config.load_balancer`, static or online, not `per_layer_dynamic`); online EPLB also needs GDRCopy for its slots | Fault tolerance adds spare slots that receive a failed rank's experts; they use no GPU memory until a failure |
| Room in `/dev/shm` on every node for the host weights of every expert (DeepSeek-R1 FP4: about 364 GB) | A failed rank's experts are restored from them |
| KV cache manager V2, the overlap scheduler | Interrupted requests continue by recompute |
| The default attention-DP routing and admission, with or without `enable_balance` | Other policies keep per-rank state a replacement would not share |
| One serving frontend (`num_serve_frontends: 1`); the MPI executor (no `orchestrator_type`) | The server follows the membership |
| `TRTLLM_MOE_COMM_TIMEOUT_SEC`, if set, at least 60 s | An all-to-all waiting on a hung rank must last until the rank is dropped |
| Not supported: disaggregated serving, KV cache connectors, KV cache events, sleep mode, `enable_lm_head_tp_in_adp`, `modeling_v2` targets, `torch_compile_config`, NGram and user-provided drafters and DSpark (MTP is supported), hybrid linear-attention models | |
| Requests: `n > 1`, `best_of > 1` and beam search are rejected | |

## Requests during a failure

- Requests in flight on the surviving ranks continue by recompute; tokens already streamed are kept.
- A request of a failed rank runs again on a survivor if it has not streamed any output yet.
- Otherwise it fails as retryable: HTTP 503 (no `Retry-After` header) with the error `code` `rank_failure`, or, for
  a streamed request that has received output, an `error` event with that `code`. Send it again.
- A request pinned to the failed attention-DP rank (`attention_dp_rank` without `attention_dp_relax`) fails.

## Launching

```text
node supervisor (trtllm-llmapi-launch, one per node) --starts and replaces--> the node's ranks (one per GPU)
rank 0 --starts--> membership authority (CPU only, hosts the rendezvous store)
```

- On one node, `trtllm-serve <model> --tp_size 4 --ep_size 4 --config config.yaml` (or the LLM API) needs nothing
  more: the server starts every rank as its own process and replaces a failed one.
- On several nodes, launch `trtllm-llmapi-launch` once per node under a launcher that keeps the other processes
  running when one exits. Its node supervisor starts the node's ranks (rank = node rank x GPUs per node + GPU index)
  and starts a replacement on the same GPU for a rank that fails. A replacement that cannot join is left out and
  another one starts after a backoff.
- A rank ends with its node supervisor, so a supervisor's end is the loss of its node. Under `mpirun`, node 0's
  supervisor starts a lost node's supervisor again through the job's own PRRTE (`prun`); elsewhere, start the node's
  launch again with the same command on the same node. This requires the launcher's own per-node processes (for
  example `prted` or the kubelet) to keep running.
- The deployment starts once every rank has started; it aborts if a rank's process ends before then, if a rank does
  not reach the rendezvous within 600 s of rank 0, or if a rank makes no progress for 600 s while other ranks wait
  for it.

Every rank needs the same join secret (at least 32 bytes, hex), which signs the membership messages, and the
rendezvous address: rank 0's node and a free port; the next port must be free too. Where the server does not start
the ranks, pass both in the environment, never on the command line:

```bash
export TLLM_FAULT_TOLERANCE_JOIN_SECRET=$(openssl rand -hex 32)
export TLLM_FAULT_TOLERANCE_RENDEZVOUS=<rank 0's host>:29600
```

Slurm: start the node supervisors with `mpirun --enable-recovery` inside the allocation, not as Slurm tasks (`srun`
is rejected at startup: Slurm's PMIx plugin cancels the whole job step when one task exits), and submit with
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

Without `--enable-recovery`, `mpirun` ends the job when one process exits; the ranks cannot detect this, so check
the command.

Kubernetes: run all ranks in one pod group that is not recreated when one pod restarts (for a LeaderWorkerSet, do
not use `RecreateGroupOnPodRestart`), one pod per node, each pod's container running `trtllm-llmapi-launch` once.
Give each pod its node rank in `PMI_RANK` and the node count in `PMI_SIZE`, and set the rendezvous address to the
leader pod's address (`LWS_LEADER_ADDRESS`). A restarted pod starts its node again.

## Costs and limits

- Host memory: every node keeps the host weights of every expert in `/dev/shm` (DeepSeek-R1 FP4: about 364 GB per
  node). Containers default `/dev/shm` to 64 MB: use `docker run --shm-size`, or on Kubernetes an `emptyDir` volume
  with `medium: Memory` and a `sizeLimit` mounted at `/dev/shm`. The startup check rejects a node with too little
  free space there.
- GPU memory: at a failure, each survivor gives up the part of its KV cache that the restored experts and the
  remapped all-to-all memory need, and gets it back once the replacement has joined. A configuration whose largest
  failure would need more than a survivor's whole KV cache is rejected at startup.
- Recompute: interrupted requests run their prefill again; with KV cache block reuse (the default) their prompt
  blocks are reused.
- Capacity: until a replacement joins, the deployment serves with the remaining ranks only.
- Network: the authority listens on the rendezvous port and the next one on rank 0's node; open only these two ports
  to the deployment's nodes. Run on a trusted network, as with main's TCP store and MPI. If the authority stops
  answering, the deployment keeps serving, unprotected.

## Monitoring

- `GET /health` answers 200 while the deployment serves, with a JSON body whose `state` is `healthy` (every rank
  serves), `degraded` (some ranks are lost), `recovering` (a membership change is in progress; serving pauses) or
  `unprotected` (the authority does not answer, so a further failure is not recovered).
- It answers 503 with `state` `stopped` once the deployment cannot recover; the server then shuts down, so an
  orchestrator restarts the instance.
- `GET /fault_tolerance_status` reports the membership:

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
    "detected_unix_ms": 1762790399988.0,
    "began_unix_ms": 1762790400000.0,
    "ready_unix_ms": 1762790400420.0,
    "ended_unix_ms": 1762790401800.0
  }
}
```

- `generation` counts membership changes and `incarnations` each rank's replacements. `last_transaction` is the last
  change that ended (`COMMITTED` or `ABORT`): when the authority detected the failure (or the join request), proposed
  the change, received the last READY (null if it did not) and ended it, in milliseconds since the Unix epoch on the
  authority's host, to line up with the ranks' logs. The authority also logs one line per membership change.

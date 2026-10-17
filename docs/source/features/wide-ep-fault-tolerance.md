# Wide-EP Fault Tolerance (experimental)

Wide expert parallelism spreads a model's experts over many GPUs, so without fault tolerance the loss of
one GPU's process ends the whole job. With fault tolerance on, the surviving ranks detect the loss, stop
the step in flight, agree on who survived, route the lost rank's experts to their copies on other GPUs and
keep serving on the same CUDA graphs. Requests that the lost rank was serving fail with a retryable error;
every other request continues.

## Requirements

- Attention data parallelism with expert parallelism over the whole world (DEP), PyTorch backend,
  KV Cache Manager V2 without KV block reuse (`kv_cache_config.enable_block_reuse: false`), the default
  attention-DP router.
- The CuteDSL NVFP4 MoE backend on Blackwell (`moe_config.backend: CUTEDSL`).
- Static expert placement with resident copies: `moe_config.load_balancer` with `num_slots` above the
  number of experts and `layer_updates_per_iter: 0`. A rank is recoverable only if every expert it hosts
  has a copy on another GPU; rank 0 is never recoverable.
- Open MPI built with ULFM fault tolerance (`ompi_info | grep ftmpi` lists it) and mpi4py 4.1 or later.

## Running

1. Start the recovery authority on a host every rank can reach (it holds no GPU), bound to the job's
   private network interface, with a key generated for this job:
   ```bash
   export TLLM_FAULT_TOLERANCE_AUTHORITY_KEY=$(python3 -c "import os; print(os.urandom(32).hex())")
   python3 -m tensorrt_llm.executor.recovery.authority --endpoint "tcp://<private-ip>:5555" --world-size 32
   ```
   The authority's messages are authenticated but not encrypted, and any host that can reach its port can
   disrupt recovery: never bind it to a public interface or reuse a key across jobs. Without `--endpoint` it
   binds to localhost (single-node jobs).
2. Launch the server under `mpirun --with-ft ulfm` with the ob1 transport and the same key:
   ```bash
   export TLLM_FAULT_TOLERANCE_MODE=1
   export TLLM_FAULT_TOLERANCE_AUTHORITY=tcp://<authority-host>:5555
   mpirun --with-ft ulfm --mca pml ob1 --mca btl self,tcp --mca coll ^hcoll,ucc \
       -x TLLM_FAULT_TOLERANCE_MODE -x TLLM_FAULT_TOLERANCE_AUTHORITY -x TLLM_FAULT_TOLERANCE_AUTHORITY_KEY \
       ... trtllm-llmapi-launch trtllm-serve <model> --config <config.yaml>
   ```
   ULFM needs Open MPI's ob1 messaging layer; use TCP within nodes too (`btl self,tcp`). On hosts with
   several network interfaces, add `--mca btl_tcp_if_include <interface>`.

| Variable | Meaning |
|---|---|
| `TLLM_FAULT_TOLERANCE_MODE=1` | Enables fault tolerance |
| `TLLM_FAULT_TOLERANCE_AUTHORITY` | The authority's address; required to recover |
| `TLLM_FAULT_TOLERANCE_AUTHORITY_KEY` | Shared HMAC key (64 hex characters) for authority messages |
| `TLLM_FAULT_TOLERANCE_RECOVERY_TIMEOUT_S` | How long a recovery may take before the job stops (default 30) |
| `TLLM_FAULT_TOLERANCE_EXTRA_SLOTS` | EPLB slots beyond one per expert: `replicas` (default) serve replicas; `empty` are not routed to while healthy and take a lost rank's experts |

At startup rank 0 logs which ranks' and nodes' loss the job can recover from. A configuration that recovery does
not support is refused at startup with the requirement it misses.

## What is covered

One loss per job, with the NVLink fabric healthy, of either:

- a recoverable rank's process (killed, for example by the out-of-memory killer, or failed with an error
  that corrupts its CUDA context, such as an illegal memory access, after which it ends itself), or
- every rank of one node other than node 0 (ranks `n * gpus_per_node` to `(n + 1) * gpus_per_node - 1`).
  The survivors must be able to hold every expert: with `s` EPLB slots per rank, `N` ranks and `K` GPUs
  per node, `s * (N - K)` must be at least the number of experts. For DeepSeek-R1 (256 experts) at DEP32
  with 4 GPUs per node, that is 320 slots.

Not covered: rank 0, any other set of ranks (for example two ranks, or part of a node), a second loss,
including one during a recovery, a rank that hangs without dying, the authority itself, and other errors
in the serving loop, which end the job as without fault tolerance. A job that lost a rank ends with a
non-zero exit status at shutdown.

## Not supported together with fault tolerance

Refused at startup, with the requirement that is missing: speculative decoding (MTP),
attention-DP balancing or a non-default attention-DP router, expert tensor parallelism, beam search, guided
decoding, LM-head tensor parallelism in attention DP, KV cache transceivers (disaggregated serving) and
connectors, the conversation manager, SWA scratch reuse, KV pool rebalancing, PEFT/LoRA, encoder-decoder
models, sleep/wake-up, a non-MPI executor (Ray, RPC), and a hang-detection timeout that is not above twice
`TLLM_FAULT_TOLERANCE_RECOVERY_TIMEOUT_S`.

Rejected per request: repetition, presence and frequency penalties (their token counts cannot be rolled back
when a step is discarded), and multimodal requests under the overlap scheduler.

## What happens during a recovery

- Survivors detect the loss within milliseconds and resume serving well under a second later. A step
  that was running finishes first, so under heavy prefill a recovery takes up to the length of one step
  (`max_num_tokens` bounds it).
- Requests that the lost ranks were serving end with a retryable error. Non-streaming requests get an error
  response. Streaming `/v1/completions` and `/v1/chat/completions` responses end with an SSE `error` event
  followed by `[DONE]`, so OpenAI clients raise an `APIError`. Clients should retry those requests.
- All other requests continue with their full output. Greedy outputs after a recovery match a server that
  never used the lost rank's experts.
- No CUDA graphs are recaptured, so serving resumes at once without the lost GPUs.
- Expected in the logs: on rank 0, an `mpi4py.futures` manager thread exits with `MPI_ERR_PROC_FAILED`,
  followed by "a rank departed; only shutdown is accepted from now on". This is the MPI session noticing the
  loss; serving is not affected.

## Cost

Fault tolerance lowers healthy throughput, most at low load. Most of the cost is the MPI transport that ULFM
requires (Open MPI's ob1 over TCP instead of UCX) for the executor's host collectives; the mechanism itself
(the rank mask, the abort polling and the fixed receive slots) costs less.

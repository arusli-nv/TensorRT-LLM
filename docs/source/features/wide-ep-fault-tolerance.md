# Wide-EP Fault Tolerance (experimental)

Wide expert parallelism spreads a model's experts over many GPUs, so without fault tolerance the loss of
one GPU's process ends the whole job. With fault tolerance on, the surviving ranks detect the loss, stop
the step in flight, agree on who survived, route the lost rank's experts to their copies on other GPUs and
keep serving on the same CUDA graphs. Requests that the lost rank was serving fail with a retryable error;
every other request continues.

## Requirements

- Attention data parallelism with expert parallelism over the whole world (DEP), PyTorch backend,
  KV Cache Manager V2 (KV block reuse is supported: a step's context blocks become reusable only once the step
  is accepted, so an interrupted step publishes none), the default attention-DP router.
- NVLinkOneSided all-to-all. The CuteDSL NVFP4 MoE backend on Blackwell (`moe_config.backend: CUTEDSL`)
  reads the receive slots in place; layers on other backends (for example a BF16 MTP layer) get a compact
  copy of their hidden states.
- An EPLB load balancer (`moe_config.load_balancer` with `num_slots` above the number of experts), static
  (`layer_updates_per_iter: 0`) or online. Its per-node host copies of every expert restore the experts of a
  lost rank into extra slots; rank 0 is never recoverable. With online EPLB the survivors adopt rank 0's
  placement during the recovery and restart the update rotation together.
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
| `TLLM_FAULT_TOLERANCE_CONFIG` | Optional YAML file with the settings below; set the same file on the workers and the authority, which refuses a worker's registration with different settings (it logs its settings' digest at startup) |

| Setting | Meaning |
|---|---|
| `failure_domain` | `node` (default): one rank's loss, or every rank of one host but rank 0's; `rank`: one rank's loss only, for placements with fewer spare slots |
| `extra_slots` | With a static EPLB placement, the slots beyond one per expert: `empty` (default) are not routed to while healthy and take a lost rank's experts; `borrowed` are empty slots that hold no device memory until a restore maps it (taken from the KV cache when the device lacks it). Online EPLB keeps serving replicas in them; setting `extra_slots` with online EPLB is an error |
| `recovery_timeout_s` | How long a worker's recovery may take before the job stops (default 30) |
| `authority_timeout_s` | How long the authority waits from the first report to the commit (default 20; below `recovery_timeout_s`) |
| `hang_detection` | Workers report heartbeats and long all-to-all waits; the authority confirms a hung rank and fences it (default off) |
| `hang_confirm_s` | How long the evidence of a hang must hold before it is confirmed (default 30; above the slowest legitimate step) |
| `hang_suspect_s`, `heartbeat_s`, `heartbeat_timeout_s` | Device wait reported after (default 2 s); heartbeat interval (default 1 s, at most 2 s); a worker without a heartbeat this long counts as missing (default 5 s) |

A confirmed hung rank is fenced by its own evidence thread (its process ends itself), else, 1 s later, by
another rank asking the MPI runtime to kill it (PMIx job control of that one process; under `mpirun` PRRTE
ends its process group and the job keeps running), else by the authority's `--fence-command` (a
`{host}`/`{pid}` template, run without a shell, 5 s later); its loss then recovers as any other.

Fault tolerance ends a worker process in exactly these places, always with SIGKILL so that no handler,
atexit code or MPI finalization of a broken process runs:
- a rank whose serving loop fails with an error that leaves it unusable (for example a sticky CUDA error)
  kills itself, so that the survivors recover from its loss (`PyExecutor` event loop);
- a rank the authority confirmed hung kills itself from its evidence thread (`hang_evidence`);
- another rank has the MPI runtime kill a hung rank that could not (`pmix_fence`, PMIx job control);
- the authority's `--fence-command`, if configured.

At startup rank 0 logs which ranks' and nodes' loss the job can recover from. A configuration that recovery does
not support is refused at startup with the requirement it misses.

## What is covered

The loss, with the NVLink fabric healthy, of either:

- a recoverable rank's process (killed, for example by the out-of-memory killer, or failed with an error
  that corrupts its CUDA context, such as an illegal memory access, after which it ends itself), or
- with `failure_domain: node` (the default), every rank on one host other than rank 0's (the ranks that report the same
  host name). The survivors must be able to hold every expert: with `s` EPLB slots per rank, `N` ranks and
  `K` ranks per host, `s * (N - K)` must be at least the number of experts, which startup checks. For
  DeepSeek-R1 (256 experts) at DEP32 with 4 GPUs per node, that is 320 slots.

Several such ranks lost together, or one more during a recovery (before the survivors install the new
communicator), are recovered as one loss if the survivors' slots still hold every expert.

Not covered: rank 0, a loss after a recovery or after the survivors installed, a rank that hangs without
dying unless hang detection is on, the authority itself, and other errors in the serving loop, which end
the job as without fault tolerance. A job that lost a rank ends with a
non-zero exit status at shutdown.

Failure model: a failing rank stops (it dies, or it hangs and is fenced). A rank that keeps running but
computes wrong results (for example silent memory corruption) is not detected: the survivors keep its
results until it fails, and any tokens they streamed meanwhile stay delivered.

Timeouts (see the settings above):

| What | How long |
|---|---|
| A dead process | detected when its TCP connections close, in milliseconds |
| A hung rank | `hang_suspect_s` (2 s) until a device wait is reported, plus `hang_confirm_s` (30 s) until the authority confirms it, plus 1 s before another rank has the runtime end it (5 s before the fence command). Measured from a SIGSTOP to serving again at DEP32: 10.6 / 15.9 / 36.7 s with `hang_confirm_s` 3 / 10 / 30 |
| A recovery | `recovery_timeout_s` (30 s) at the workers, `authority_timeout_s` (20 s) at the authority; a recovery takes milliseconds after detection |

`hang_confirm_s` must exceed the slowest legitimate step plus host stalls (page-cache or file-system stalls of
up to a minute were observed under load): a shorter value fences a slow but healthy rank.

## Not supported together with fault tolerance

Refused at startup, with the requirement that is missing: speculative decoding other than one-model MTP
(MTP with one predict layer and a linear draft is supported),
attention-DP balancing or a non-default attention-DP router, expert tensor parallelism, beam search, guided
decoding, LM-head tensor parallelism in attention DP, KV cache transceivers (disaggregated serving) and
connectors, the conversation manager, SWA scratch reuse, KV pool rebalancing, PEFT/LoRA, encoder-decoder
models, sleep/wake-up, a non-MPI executor (Ray, RPC), and a hang-detection timeout that is not above twice
`recovery_timeout_s`.

Rejected per request: repetition, presence and frequency penalties (their token counts cannot be rolled back
when a step is discarded), and multimodal requests under the overlap scheduler.

## What happens during a recovery

- Survivors detect the loss within milliseconds and resume serving well under a second later. A step
  that was running finishes first, so under heavy prefill a recovery takes up to the length of one step
  (`max_num_tokens` bounds it).
- After a node loss, the requests its ranks were serving that had delivered nothing yet (every non-streaming
  request, and a streaming one before its first token) run again from their prompt on a survivor. After the loss
  of a single rank they do not: the request may be what crashed the rank, and the survivors cannot recover from
  a second loss. Those, and requests that had streamed tokens, end with a retryable error. Non-streaming requests get an error
  response. Streaming `/v1/completions` and `/v1/chat/completions` responses end with an SSE `error` event
  followed by `[DONE]`, so OpenAI clients raise an `APIError`. Clients should retry those requests.
- All other requests continue with their full output. Greedy outputs after a recovery match a server that
  never used the lost rank's experts.
- No CUDA graphs are recaptured, so serving resumes at once without the lost GPUs.
- Expected in the logs: on rank 0, an `mpi4py.futures` manager thread exits with `MPI_ERR_PROC_FAILED`,
  followed by "a rank departed; only shutdown is accepted from now on". This is the MPI session noticing the
  loss; serving is not affected.

## Monitoring

`trtllm-serve` asks the recovery authority for its state. `GET /health` returns 200 with the state while the
server serves: `healthy` (no loss), `degraded` (serving after a recovery) or `unprotected` (the authority did not
answer, so a loss would end the job: alert, but keep serving). It returns 503 with the state while `starting`,
`recovering`, `terminal` (a recovery aborted: restart the deployment) or `unknown` (the authority reported a
state this server does not know, for example from a different version). `GET /fault_tolerance` returns that state with the membership generation, live and failed ranks
(sorted), the number of recoveries, the last recovery's stop / agree / install durations in milliseconds,
and the number of requests the loss failed.

## Cost

Fault tolerance lowers healthy throughput, most at low load. Most of the cost is the MPI transport that ULFM
requires (Open MPI's ob1 over TCP instead of UCX) for the executor's host collectives; the mechanism itself
(the rank mask, the abort polling and the fixed receive slots) costs less.

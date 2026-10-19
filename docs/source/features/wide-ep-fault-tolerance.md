# Wide-EP Fault Tolerance (experimental)

Wide expert parallelism spreads a model's experts over many GPUs, so without fault tolerance the loss of
one GPU's process ends the whole job. With fault tolerance on, the surviving ranks detect the loss, stop
the step in flight, agree on who survived, route the lost rank's experts to their copies on other GPUs and
keep serving on the same CUDA graphs. Requests that the lost rank was serving fail with a retryable error;
every other request continues.

## Requirements

- Attention data parallelism with expert parallelism over the whole world (DEP), PyTorch backend,
  KV Cache Manager V2 or its DSA sparse-attention variant (KV block reuse is supported: a step's context blocks become reusable only once the step
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
   binds to localhost (single-node jobs). With hang detection on, it also receives the workers' native
   heartbeats as UDP datagrams on the same port number: let UDP reach it.
2. Launch the server under `mpirun --with-ft ulfm` with the ob1 transport and the same key:
   ```bash
   export TLLM_FAULT_TOLERANCE_MODE=1
   export TLLM_FAULT_TOLERANCE_AUTHORITY=tcp://<authority-host>:5555
   mpirun --with-ft ulfm --mca pml ob1 --mca btl self,tcp --mca coll ^hcoll,ucc \
       --mca mpi_ft_detector true --mca mpi_ft_detector_thread true \
       --mca mpi_ft_detector_period 1 --mca mpi_ft_detector_timeout 5 \
       -x TLLM_FAULT_TOLERANCE_MODE -x TLLM_FAULT_TOLERANCE_AUTHORITY -x TLLM_FAULT_TOLERANCE_AUTHORITY_KEY \
       ... trtllm-llmapi-launch trtllm-serve <model> --config <config.yaml>
   ```
   ULFM needs Open MPI's ob1 messaging layer; use TCP within nodes too (`btl self,tcp`). On hosts with
   several network interfaces, add `--mca btl_tcp_if_include <interface>`. The `mpi_ft_detector` settings turn
   on Open MPI's in-band failure detector on its own thread. It notices a dead or frozen process whose
   connections stay open (a stopped process, a frozen host) within about the timeout; without it, only closed
   connections and hang detection reveal one. It showed no detectable healthy cost at DEP32 (resolved to about 3% at 128 requests per GPU). Open MPI marks it
   experimental. Its unthreaded form can declare a slow-starting rank dead.

| Variable | Meaning |
|---|---|
| `TLLM_FAULT_TOLERANCE_MODE=1` | Enables fault tolerance |
| `TLLM_FAULT_TOLERANCE_AUTHORITY` | The authority's address; required to recover |
| `TLLM_FAULT_TOLERANCE_AUTHORITY_KEY` | Shared HMAC key (64 hex characters) for authority messages |
| `TLLM_FAULT_TOLERANCE_CONFIG` | Optional YAML file with the settings below; set the same file on the workers and the authority, which refuses a worker's registration with different settings (it logs its settings' digest at startup) |

| Setting | Meaning |
|---|---|
| `failure_domain` | `node` (default): one rank's loss, or every rank of one host but rank 0's; `rank`: one rank's loss only, for placements with fewer spare slots |
| `extra_slots` | With a static EPLB placement, the slots beyond one per expert: `ghost` (default) are empty slots that hold no device memory until a restore maps it (taken from the KV cache when the device lacks it: that rank's KV cache stays smaller until the deployment restarts, and if it cannot shrink, the recovery fails and the job ends). `empty` slots hold their memory and are not routed to while healthy (no mapping at recovery: a faster resume). Online EPLB keeps serving replicas in them; setting `extra_slots` with online EPLB is an error |
| `recovery_timeout_s` | How long a worker's recovery may take before the job stops (default 30) |
| `authority_timeout_s` | How long the authority waits from the first report to the commit (default 20; below `recovery_timeout_s`) |
| `hang_detection` | Workers report heartbeats and long all-to-all waits; the authority confirms a hung rank and fences it, or reports a stall of every rank as terminal (default off) |
| `hang_confirm_s` | How long the evidence of a hang or a stall must hold before it is confirmed (default 30; above the slowest legitimate step) |
| `hang_suspect_s`, `heartbeat_s`, `heartbeat_timeout_s` | Device wait reported after (default 2 s); heartbeat interval (default 1 s, at most 2 s); a worker without a heartbeat this long counts as missing (default 5 s) |

A confirmed hung rank is fenced by its own evidence thread (its process ends itself), else, 1 s later, by
another rank asking the MPI runtime to kill it (PMIx job control of that one process; under `mpirun` PRRTE
ends its process group and the job keeps running), else by the authority's `--fence-command` (a
`{host}`/`{pid}` template, run without a shell, 5 s later); its loss then recovers as any other.

The survivors' agreement on the failed ranks can rest on Open MPI's in-band failure detector, which
declares a stopped process failed after its timeout while the process still holds its GPU and its mappings
of the survivors' memory, and the agreement does not complete until that process ends. So each survivor
first has the MPI runtime kill the ranks its own MPI view reports failed, and before it proposes a plan, the
authority has rank 0 kill every failed rank it has not fenced itself (a no-op for a process that already
ended); it aborts the recovery if the runtime refuses or does not answer within 2 s.

Fault tolerance ends a worker process in exactly these places, with SIGKILL so that no handler, atexit
code or MPI finalization of a broken process runs, except the last, which exits with code 75:
- a rank whose serving loop fails with an error that leaves it unusable (for example a sticky CUDA error)
  kills itself, so that the survivors recover from its loss (`PyExecutor` event loop);
- a rank the authority confirmed hung kills itself from its evidence thread (`hang_evidence`);
- another rank has the MPI runtime kill a hung rank that could not (`pmix_fence`, PMIx job control);
- each survivor has the MPI runtime kill the ranks its MPI view reports failed before agreeing on them, and
  rank 0 each failed rank before a recovery plan (`peer_loss_recovery`);
- the authority's `--fence-command`, if configured;
- every rank, when the authority found every rank stalled with no rank to blame (`hang_evidence`).

At startup rank 0 logs which ranks' and nodes' loss the job can recover from. A configuration that recovery does
not support is refused at startup with the requirement it misses.

## What is covered

The loss, with the NVLink fabric healthy, of either:

- a recoverable rank's process (killed, for example by the out-of-memory killer, or failed with an error
  that corrupts its CUDA context, such as an illegal memory access, after which it ends itself), or
- with `failure_domain: node` (the default), every rank on one host other than rank 0's (the ranks that report the same
  host name). The survivors must be able to hold every expert: with `s` EPLB slots per rank, `N` ranks and
  `K` ranks per host, `s * (N - K)` must be at least the number of experts, which startup checks. For
  DeepSeek-R1 (256 experts) at DEP32 with 4 GPUs per node, that is 320 slots (320 = 256 active + 64 ghost).

Several such ranks lost together, or one more during a recovery (before the survivors install the new
communicator), are recovered as one loss if the survivors' slots still hold every expert. A further loss after a
recovery is recovered the same way: the survivors recover again as long as their slots still hold every expert
without every rank lost so far (startup checks only one failure domain, so how many further losses a deployment
survives depends on its spare slots).

Not covered: rank 0, a loss after the survivors installed a recovery and before it committed, a further loss
the remaining slots cannot cover, a rank that hangs without
dying unless hang detection is on, a stall that every rank shares (a deadlock or desync with no single
culprit: with hang detection on it is reported, see below, else the executor's own hang detection,
`hang_detection_timeout`, ends it), losses beyond one failure
domain (for example a fault that takes down several hosts or a whole NVLink domain), the authority itself,
and other errors in the serving loop, which end the job as without fault tolerance. A job that recovered
from a loss still stops cleanly: stopping `trtllm-serve` finalizes MPI on the survivors and `mpirun` exits with
status 0 (a survivor that does not finalize within 120 s ends the job with a non-zero status).

After a recovery the deployment serves `degraded` and a lost rank never rejoins: each loss lowers capacity
(and may take more of the KV cache for restored experts). A request that was running on a lost rank and had
delivered nothing runs again once on a survivor, even if it was pinned to the lost rank with
`attention_dp_relax=False` (a new request pinned to a lost rank fails with "Attention DP rank N is not
serving"). Because it may be what crashed the rank, a request that loses its rank a second time is not run again: it fails with a retryable error. If a re-run or retried request that crashes
every rank it runs on lands on rank 0, or on a rank whose loss the remaining slots cannot cover, its crash ends the
deployment; the server does not recognize a client's retry of a failed request.

Failure model: a failing rank stops (it dies, or it hangs and is fenced). A rank that keeps running but
computes wrong results (for example silent memory corruption) is not detected: the survivors keep its
results until it fails, and any tokens they streamed meanwhile stay delivered.

Timeouts (see the settings above):

| What | How long |
|---|---|
| A dead process | detected when its TCP connections close, in milliseconds |
| A hung rank | `hang_suspect_s` (2 s) until a device wait is reported, or `heartbeat_timeout_s` (5 s) until a stopped rank's heartbeat counts as missing, plus `hang_confirm_s` (30 s) until the authority confirms it, plus 1 s before another rank has the runtime end it (5 s before the fence command). Measured from a SIGSTOP to serving again at DEP32 under an earlier rule: 10.6 / 15.9 / 36.7 s with `hang_confirm_s` 3 / 10 / 30 |
| A stall of every rank | `hang_confirm_s` (30 s) from its start until the deployment is reported terminal and every worker exits with code 75 |
| A recovery | `recovery_timeout_s` (30 s) at the workers from the survivors' agreement on the failed ranks, `authority_timeout_s` (20 s) at the authority; a recovery takes milliseconds after detection. The agreement waits for every survivor: up to the hang's resolution time (with hang detection) plus `recovery_timeout_s`, since a survivor can detect a stopped peer before the others, which stay blocked on it until it is fenced |

How a hang is judged. Each worker sends heartbeats twice: from a native thread, which runs without the Python
interpreter lock and reports the iteration of the last finished step, and from a Python thread, which also
reports whether the rank waits on the host (a collective, or new requests) and which peers its device has
waited on in an all-to-all for `hang_suspect_s`. A rank is fenced only when, for `hang_confirm_s`, every peer
whose heartbeat arrives waits on it (its device waits on that rank, or it waits on the host while that rank
does not or sends no heartbeat) and the rank finished no step; any finished step restarts the window. A rank
whose own device waits on another rank is not blamed. When no rank finished a step and every rank's device
waited on a peer for `hang_confirm_s`, with no single rank waited on by all, no fence can help: the
authority reports the deployment terminal (`/health` 503 `terminal`, with the reason) and
every worker exits with code 75 so that the orchestrator restarts it. An idle deployment, or one whose ranks
all stall on the host between steps, reports no device waits and matches neither rule. Nothing is decided
while fewer than half the ranks' heartbeats arrive.

`hang_confirm_s` must exceed the slowest legitimate step plus host stalls (page-cache or file-system stalls of
up to a minute were observed under load): a shorter value fences a slow but healthy rank. That includes a rank
whose serving loop is starved of the Python interpreter lock (for example by a CPU-bound thread in the same
process): its native heartbeat keeps it known to be alive, but if it finishes no step for `hang_confirm_s` while
its peers wait for it, it is fenced like a hung rank. Without the native heartbeat (the UDP datagrams cannot
reach the authority) the Python heartbeat alone is used, and a starved rank whose heartbeat stops looks stopped.

Two thresholds govern a silent or stuck rank. A process frozen as a whole (no thread runs, for example a
stopped process or a host that stops scheduling it) is declared failed by Open MPI's in-band detector after
`mpi_ft_detector_timeout` (5 s above); the recovery then fences it and continues without it. A process that is
alive but makes no progress (a device hang, a deadlock, a host or interpreter-lock stall) keeps its native threads
running, so the detector does not fire and only `hang_confirm_s` applies. At DEP32, a 10 s whole-process freeze
cost the rank, while 45 s host stalls and interpreter-lock stalls cost nothing. Raise `mpi_ft_detector_timeout`
to tolerate longer whole-process freezes, at the cost of detecting a dead host as much later.

## Not supported together with fault tolerance

Speculative decoding: every method is accepted. A recovery discards the interrupted steps' draft tokens
and draft KV, and target verification keeps the output exact. Only one-model MTP with a linear draft is
tested; other methods log a warning at startup, because per-slot draft state kept inside the CUDA graphs
(the relaxed-acceptance delta, the suffix automaton, dynamic-tree state) is not reset: acceptance can drop
after a recovery, and relaxed acceptance, lossy by design, can accept different tokens.

Refused at startup, with the requirement that is missing:
attention-DP balancing or a non-default attention-DP router, expert tensor parallelism, beam search, guided
decoding, LM-head tensor parallelism in attention DP, KV cache transceivers (disaggregated serving) and
connectors, the conversation manager, SWA scratch reuse, KV pool rebalancing, PEFT/LoRA, encoder-decoder
models, sleep/wake-up, a non-MPI executor (Ray, RPC), and a hang-detection timeout that is not above three times
`recovery_timeout_s`, plus the hang's resolution time when hang detection is on (see the timeouts above).

Rejected per request: repetition, presence and frequency penalties (their token counts cannot be rolled back
when a step is discarded), and multimodal requests under the overlap scheduler.

## What happens during a recovery

- Survivors detect the loss within milliseconds and resume serving well under a second later. A step
  that was running finishes first, so under heavy prefill a recovery takes up to the length of one step
  (`max_num_tokens` bounds it).
- The requests the lost ranks were serving that had delivered nothing yet (every non-streaming request, and a
  streaming one before its first token) run again from their prompt on a survivor, once. A request that already
  ran again and loses its rank a second time may be what crashes them: it is not run again. Those, and requests
  that had streamed tokens, end with a retryable error. Non-streaming requests get an error
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
state this server does not know, for example from a different version). While the authority answers, the body
also carries the membership generation, live and failed ranks (sorted), the number of recoveries, the last
recovery's stop / agree / install durations in milliseconds, the number of requests the losses failed and, once
terminal, the reason.

What an orchestrator should do with these signals:
- `degraded`: the deployment serves with fewer GPUs and survives a further loss only while its remaining slots
  hold every expert without it. Restart it (or replace it with a fresh replica) when convenient to get the
  capacity and the full protection back.
- `terminal`, a `/health` that stops answering, or a server process that exits: the loss was not recoverable
  (rank 0, more than one failure domain at once, a further loss the remaining slots cannot cover, a failed
  recovery), or every rank stalled (the workers then exit with code 75). Move traffic to another replica,
  have clients retry their failed requests there, and restart the deployment.
- `unprotected`: keep serving, alert, and check the authority.

`/health` only reports. There is no control action: no drain, no retry of a lost rank on the same
GPUs and no scale-down request; drain a deployment by stopping traffic to it at the load balancer.

## How well each part is tested

The whole feature is experimental. The loss of a single rank's process (with replica or empty extra slots, with
or without the overlap scheduler) and the authority's recovery protocol are tested most, including repeated
kills under load at DEP32.
These are tested only in single runs or at small scale so far, so treat them as less mature: the loss of a whole
host, `ghost` extra slots, hang detection and fencing, online EPLB, KV block reuse, MTP, and the `/health`
states. A further loss after a recovery, and re-running a request once after the loss of a single rank, are
tested only by unit tests so far.

## Cost

Fault tolerance lowers healthy throughput, most at low load. The cost has two sources: the MPI transport that
ULFM requires (Open MPI's ob1 over TCP instead of UCX) for the executor's host collectives, and the mechanism
itself (the rank mask, the abort polling and the fixed receive slots). Their shares have not been measured
separately yet.

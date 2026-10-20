# Wide-EP Fault Tolerance (experimental)

Wide expert parallelism spreads a model's experts over many GPUs, so without fault tolerance the loss of
one GPU's process ends the whole job. With fault tolerance on, the surviving ranks detect the loss, stop
the step in flight, agree on who survived, route the lost rank's experts to their copies on other GPUs and
keep serving on the same CUDA graphs. Requests the lost rank was serving run again on a survivor if they
had delivered nothing yet, else end with a retryable error; every other request continues. A request that a
second loss interrupts ends with a retryable error, whichever ranks were lost.

## Requirements

- Attention data parallelism with expert parallelism over the whole world (DEP), PyTorch backend,
  KV Cache Manager V2 or its DSA sparse-attention variant (KV block reuse is supported: a step's context blocks become reusable only once the step
  is accepted, so an interrupted step publishes none), the default attention-DP router.
- NVLinkOneSided all-to-all. The CuteDSL NVFP4 MoE backend on Blackwell (`moe_config.backend: CUTEDSL`)
  and the Cutlass backend for unquantized layers without LoRA (for example a BF16 MTP layer) read the receive
  slots in place; layers on other backends get a compact copy of their hidden states.
- An EPLB load balancer (`moe_config.load_balancer` with `num_slots` above the number of experts), static
  (`layer_updates_per_iter: 0`) or online. Host copies of the experts in `/dev/shm` restore the experts of a
  lost rank into extra slots; rank 0 is never recoverable. Online EPLB keeps every expert's copy on every node; the
  survivors adopt rank 0's placement during the recovery and restart the update rotation together. A static
  placement keeps on each node only the home experts of the next node (nodes in order of their first rank, the
  last node keeping the first's; a single node keeps its own), about `1/N` of the copies with `N` nodes. A restore
  copies each expert from a GPU of that node into the restoring rank's NVLinkOneSided workspace and from there
  into its slot, in chunks of half the workspace; losing a node together with the node that keeps its copies is
  not recoverable. Size `/dev/shm` (host memory) for these copies: with online EPLB, every node needs the expert
  weights of all MoE layers; with a static placement, about `1/N` of them. Startup fails with the sizes if a
  rank's copies do not fit.
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
   connections stay open (a stopped process, a frozen host) within about the timeout: a process silent at the
   OS level for more than 5 s is treated as failed. Without it, only closed connections and hang detection
   reveal one. It showed no detectable healthy cost at DEP32 (resolved to about 3% at 128 requests per GPU). Open MPI marks it
   experimental. Its unthreaded form can declare a slow-starting rank dead.

| Variable | Meaning |
|---|---|
| `TLLM_FAULT_TOLERANCE_MODE=1` | Enables fault tolerance; alone it turns on the rank mask (and its cost) but only detects a loss: rank 0 warns at startup and `/health` reports `unprotected` |
| `TLLM_FAULT_TOLERANCE_AUTHORITY` | The authority's address; required to recover |
| `TLLM_FAULT_TOLERANCE_AUTHORITY_KEY` | Shared HMAC key (64 hex characters) for authority messages |
| `TLLM_FAULT_TOLERANCE_CONFIG` | Optional YAML file with the settings below; set the same file on the workers and the authority, which refuses a worker's registration with different settings (it logs its settings' digest at startup) |

| Setting | Meaning |
|---|---|
| `failure_domain` | `node` (default): one rank's loss, or every rank of one host but rank 0's; `rank`: one rank's loss only, for placements with fewer spare slots |
| `extra_slots` | With a static EPLB placement, the slots beyond one per expert: `ghost` (default) are empty slots that hold no device memory while healthy (the KV cache keeps it) and get memory only when a recovery restores a lost rank's experts into them (taken from the KV cache when the device lacks it: that rank's KV cache stays smaller until the deployment restarts, and if it cannot shrink, the recovery fails and the job ends). `empty` slots hold their memory and are not routed to while healthy (no mapping at recovery: a faster resume). With online EPLB, its slots keep serving replicas; where they cannot hold every expert after the loss of a failure domain, each rank gets the missing slots of this kind after EPLB's own, and EPLB's updates use one once a restore filled it |
| `recovery_timeout_s` | How long a worker's recovery may take, including finishing a long step first, before the job stops (default 30 s) |
| `authority_timeout_s` | How long the authority waits from the first report to the commit (default 20 s; must be below `recovery_timeout_s`) |
| `hang_detection` | Workers report heartbeats and long all-to-all waits; the authority confirms a hung rank and fences it, or reports a hung rank 0 or a stall of every rank as terminal (default `false`) |
| `hang_suspect_s` | How long a worker's device waits on a peer in an all-to-all before it reports the wait (default 2 s) |
| `hang_confirm_s` | An alive but stuck rank is fenced only after every peer has waited on it, with the rank finishing no step, for this long; also how long a stall of every rank must last to be reported terminal (default 30 s; above the slowest legitimate step plus host stalls) |
| `heartbeat_s` | Worker heartbeat interval (default 1 s, at most 2 s, below `heartbeat_timeout_s`) |
| `heartbeat_timeout_s` | A worker without a heartbeat this long counts as missing (default 5 s) |

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
- every rank, when the authority found every rank stalled with no rank to blame, or rank 0 hung (`hang_evidence`).

At startup the workers register with the authority (it must be running), check that the survivors of each failure
domain's loss can still serve every expert, and rank 0 logs which ranks' and nodes' loss the job can recover from.
Startup fails closed: a configuration that recovery does not support, or slots that do not cover a failure domain,
is refused with the requirement it misses, and the loss of a rank before serving starts (model loading, warmup, CUDA
graph capture, KV cache estimation) ends the job (`MPI_Abort`).

## What is covered

The loss, with the NVLink fabric healthy, of either:

- a recoverable rank's process (killed, for example by the out-of-memory killer, or failed with an error
  that corrupts its CUDA context, such as an illegal memory access, after which it ends itself), or
- with `failure_domain: node` (the default), every rank on one host other than rank 0's (the ranks that report the same
  host name). The survivors must be able to hold every expert: with `s` EPLB slots per rank, `N` ranks and
  `K` ranks per host, `s * (N - K)` must be at least the number of experts, which startup checks. For
  DeepSeek-R1 (256 experts) at DEP32 with 4 GPUs per node, that is 320 slots (320 = 256 active + 64 ghost,
  10 per rank): the survivors of one node's loss hold 280 slots. The 64 ghost slots cover one node, and up to
  6 lost GPUs in total over repeated losses (26 survivors hold 260 slots).
  With online EPLB (`num_slots: 288`), fault tolerance adds one ghost slot per rank after EPLB's own: 320 in all.

Several such ranks lost together, or one more during a recovery (before the survivors install the new
communicator), are recovered as one loss if the survivors' slots still hold every expert. A further loss after a
recovery is recovered the same way: the survivors recover again as long as their slots still hold every expert
without every rank lost so far (startup checks only one failure domain, so how many further losses a deployment
survives depends on its spare slots).

Not covered (these fail closed: the job ends as without fault tolerance):
- rank 0, which runs the server's coordination;
- the death of a host itself (power loss, kernel panic, or its Open MPI daemon ending): Open MPI's launcher ends
  the job when it loses a daemon. `failure_domain: node` covers the loss of every rank process of a host that
  stays up;
- a fault that takes down a GPU device, the NVLink fabric, several hosts or a rack;
- a further loss after the survivors installed a recovery and before it committed (a loss before the install
  re-plans the recovery, see above), and a further loss the remaining slots cannot cover;
- a rank that hangs without dying unless hang detection is on, and a stall that every rank shares (a deadlock
  or desync with no single culprit: with hang detection on it is reported, see below, else the executor's own
  hang detection, `hang_detection_timeout`, ends it);
- other errors in the serving loop.

The authority is not a single point of failure for serving: if it stops, the deployment keeps serving but
unprotected (`/health` reports `unprotected`), and a rank loss then ends the job. A job that recovered
from a loss still stops cleanly: stopping `trtllm-serve` finalizes MPI on the survivors and `mpirun` exits with
status 0 (a survivor that does not finalize within 120 s ends the job with a non-zero status).

After a recovery the deployment serves `degraded` and a lost rank never rejoins: each loss lowers capacity
(and may take more of the KV cache for restored experts). A request re-run after its rank's loss (see below) runs
on any survivor, even if it was pinned to the lost rank with `attention_dp_relax=False`; a new request pinned to a
lost rank fails with "Attention DP rank N is not serving" (HTTP 503, see below). If a re-run or retried request
that crashes every rank it runs on lands on rank 0, or on a rank whose loss the remaining slots cannot cover, its
crash ends the deployment; the server does not recognize a client's retry of a failed request.

Failure model: a failing rank stops (it dies, or it hangs and is fenced). A rank that keeps running but
computes wrong results (for example silent memory corruption) is not detected: the survivors keep its
results until it fails, and any tokens they streamed meanwhile stay delivered.

Timeouts (see the settings above):

| What | How long |
|---|---|
| A dead process | detected when its TCP connections close, in milliseconds |
| A hung rank | `hang_suspect_s` (2 s) until a device wait is reported, or `heartbeat_timeout_s` (5 s) until a stopped rank's heartbeat counts as missing, plus `hang_confirm_s` (30 s) until the authority confirms it, plus 1 s before another rank has the runtime end it (5 s before the fence command). Measured from a SIGSTOP to serving again at DEP32 under an earlier rule: 10.6 / 15.9 / 36.7 s with `hang_confirm_s` 3 / 10 / 30 |
| A stall of every rank, or a hung rank 0 | `hang_confirm_s` (30 s) from its start until the deployment is reported terminal and every worker exits with code 75 |
| A recovery | `recovery_timeout_s` (30 s) at the workers from the survivors' agreement on the failed ranks, `authority_timeout_s` (20 s) at the authority; a recovery takes milliseconds after detection. The agreement waits for every survivor: up to the hang's resolution time (with hang detection) plus `recovery_timeout_s`, since a survivor can detect a stopped peer before the others, which stay blocked on it until it is fenced |

How a hang is judged. Each worker sends heartbeats twice: from a native thread, which runs without the Python
interpreter lock and reports the iteration of the last finished step, and from a Python thread, which also
reports whether the rank waits on the host (a collective, or new requests) and which peers its device has
waited on in an all-to-all for `hang_suspect_s`. A rank is fenced only when, for `hang_confirm_s`, every peer
whose heartbeat arrives waits on it (its device waits on that rank, or it waits on the host while that rank
does not or sends no heartbeat) and the rank finished no step; any finished step restarts the window. A rank
whose own device waits on another rank is not blamed. When no rank finished a step and every rank's device
waited on a peer for `hang_confirm_s`, with no single rank waited on by all, no fence can help: the
authority reports the deployment terminal (`/health` 503 `terminal`; the reason is in its log) and
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

## Known limits

- A hang outside the MoE all-to-alls, or any hang with `hang_detection` off, ends the job only through the
  executor's hang detection (`hang_detection_timeout`, 300 s by default) or the MoE communication timeout.
- Two ranks hung at once are not confirmed (no single rank is waited on by every peer); they end the job
  through the same timeouts.
- A fenced process stuck in the GPU driver may not exit when killed, which can stall the job's shutdown.
- Whether and how fast the death of a host or its Open MPI (PRRTE) daemon ends the job depends on PRRTE.
- Nothing restarts the authority, and the workers register only at startup: an authority started again
  reports `starting` (503) until the deployment restarts. Restart the authority only together with the job,
  with a new key, never under a running job.
- Stopping the job (`scancel`, deleting its pods) signals every rank; a rank signaled a moment after another may
  log a recovery attempt before it exits. The job ends anyway, with the status of the signal.
- With ring host copies (online EPLB off), losing a node (or only its copy node) leaves the next node's
  experts without host copies, so a later loss there is unrecoverable and ends the job.

## Not supported together with fault tolerance

Speculative decoding: every method is accepted. A recovery discards the interrupted steps' draft tokens
and draft KV, and target verification keeps the output exact. Only one-model MTP with a linear draft is
tested; other methods log a warning at startup, because per-slot draft state kept inside the CUDA graphs
(the relaxed-acceptance delta, the suffix automaton, dynamic-tree state) is not reset: acceptance can drop
after a recovery, and relaxed acceptance, lossy by design, can accept different tokens.

Refused at startup, with the requirement that is missing:
a non-default attention-DP router (attention-DP balancing is supported), expert tensor parallelism, beam
search, guided decoding, LM-head tensor parallelism in attention DP, KV cache transceivers (disaggregated serving)
and connectors, the conversation manager, SWA scratch reuse, KV pool rebalancing, PEFT/LoRA, encoder-decoder
models, sleep/wake-up, a non-MPI executor (Ray, RPC), and a hang-detection timeout that is not above three times
`recovery_timeout_s`, plus the hang's resolution time when hang detection is on (see the timeouts above).

Rejected per request: repetition, presence and frequency penalties (their token counts cannot be rolled back
when a step is discarded), logits post-processors (they may keep state and would run again on a re-run step),
and multimodal requests under the overlap scheduler.

## What happens during a recovery

- Survivors detect the loss within milliseconds and resume serving well under a second later. A step
  that was running finishes first, so under heavy prefill a recovery takes up to the length of one step
  (`max_num_tokens` bounds it).
- The requests the lost ranks were serving that had delivered nothing yet (every non-streaming request, and a
  streaming one before its first token) run again from their prompt on a survivor. A request is interrupted
  once at most: one that an earlier loss interrupted (it ran again, or a survivor was running it) may be what
  crashes the ranks, so it is quarantined, not continued. Quarantined requests, and streaming requests of the
  lost ranks that had already delivered tokens (they are not continued on a survivor), end with a retryable
  error. Non-streaming requests get HTTP 503 with `Retry-After: 1` and the error code `peer_loss_retryable`.
  Streaming `/v1/completions`, `/v1/chat/completions` and `/v1/responses` responses end with an SSE `error`
  event with that code followed by `[DONE]`, so OpenAI clients raise an `APIError`. Clients should retry those
  requests.
- A request of the lost ranks whose cancellation they had not answered yet ends cancelled.
- All other requests continue with their full output. Greedy outputs after a recovery match a server that
  never used the lost rank's experts.
- No CUDA graphs are recaptured, so serving resumes at once without the lost GPUs.
- Each rank logs one line when it resumes: "resumed serving without ranks [...] at membership generation N",
  the time since the interruption, the discarded steps, the remaining ranks and the KV cache bytes given to
  restored experts; on rank 0 also how many requests ran again and how many failed.
- Expected in the logs: on rank 0, an `mpi4py.futures` manager thread exits with `MPI_ERR_PROC_FAILED`,
  followed by "a rank departed; only shutdown is accepted from now on". This is the MPI session noticing the
  loss; serving is not affected.

## Monitoring

`trtllm-serve` asks the recovery authority for its state. `GET /health` returns 200 with the state while the
server serves: `healthy` (no loss), `degraded` (serving after a recovery) or `unprotected` (the authority did not
answer, so a loss would end the job: alert, but keep serving). It returns 503 with the state while `starting`,
`recovering`, `terminal` (a recovery aborted: restart the deployment) or `unknown` (the authority reported a
state this server does not know, for example from a different version). The JSON body always has `state`. While
the authority answers it also has `generation` (the membership generation), `live_ranks` and `failed_ranks`
(sorted), `recoveries`, `last_recovery_stage_ms` (the last recovery's `stop`, `agree` and `install` durations),
`failed_requests` (requests the losses failed). The reason a recovery aborted is only in the authority's log:
`/health` is not authenticated. An `unprotected` body has only `state`. If the engine itself failed, `/health` returns 503 as without fault tolerance.

What an orchestrator should do with these signals:
- `degraded`: the deployment serves with fewer GPUs and survives a further loss only while its remaining slots
  hold every expert without it. Restart it (or replace it with a fresh replica) when convenient to get the
  capacity and the full protection back.
- `terminal`, a `/health` that stops answering, or a server process that exits: the loss was not recoverable
  (rank 0, more than one failure domain at once, a further loss the remaining slots cannot cover, a failed
  recovery), or every rank stalled (the workers then exit with code 75). Move traffic to another replica,
  have clients retry their failed requests there, and restart the deployment.
- `unprotected`: keep serving, alert, and check the authority.

Health probes: a recovery returns 503 (`recovering`) for about a second, at most `authority_timeout_s`. Let a
readiness probe take the replica out of rotation after two failures 5 s apart, and give a liveness probe a
tolerance above `recovery_timeout_s` (for example 6 failures 10 s apart), so that a recovery never restarts the
deployment while `terminal` still does. Cover model loading (`starting`) with a startup probe.

Exit status of `mpirun` (and of the workers):

| Status | Meaning |
|---|---|
| 0 | Stopped cleanly, also after recoveries |
| 75 | Every rank stalled, or rank 0 hung: restart the deployment |
| Any other | The loss or error was not recoverable: restart the deployment |

`/health` only reports. There is no control action: no drain, no retry of a lost rank on the same
GPUs and no scale-down request; drain a deployment by stopping traffic to it at the load balancer.

## How well each part is tested

The whole feature is experimental. The loss of a single rank's process (with replica or empty extra slots, with
or without the overlap scheduler) and the authority's recovery protocol are tested most, including repeated
kills under load at DEP32.
These are tested only in single runs or at small scale so far, so treat them as less mature: the loss of a whole
host, `ghost` extra slots, hang detection and fencing, online EPLB, KV block reuse, MTP, and the `/health`
states. A further loss after a recovery, re-running a request after the loss of a single rank, and quarantining
a request a second loss interrupts are tested only by unit tests so far.

## Cost

Fault tolerance lowers healthy throughput, most at low load. The cost has two sources: the MPI transport that
ULFM requires (Open MPI's ob1 over TCP instead of UCX) for the executor's host collectives, and the mechanism
itself (the rank mask, the abort polling and the fixed receive slots). Their shares have not been measured
separately yet.

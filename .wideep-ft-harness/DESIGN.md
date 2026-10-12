# WideEP worker-loss recovery, Phase 1 MVP design

This guide is for TRT-LLM engineers who review or extend the work. It implements the MPI, non-CFT
MVP of the v7 document "WideEP Fault Tolerance in TensorRT-LLM", items 1a.8, 1b.x, 1c.3a to 1c.4c,
1d.1 and 1d.4. The code lives on branch `wideep-ft`, and `PR_PLAN.md` splits it into PRs. The
qualification scripts live in this directory.

## 1. Goal

When one worker process of a WideEP deployment dies, idle or mid-step, the other ranks keep
serving on the CUDA graphs they captured at startup. Nothing reloads, recompiles, recaptures or
retunes. Requests on surviving ranks continue with the exact same tokens. Requests that the dead
rank's attention-DP shard served fail with a retryable error.

We qualified it on DeepSeek-R1 NVFP4 at EP32 on 8×4 GB200. The configuration uses attention DP,
expert TP 1, static EPLB with spare slots, the NVLinkOneSided fence all-to-all without CFT,
decode graphs for batch sizes 1, 2, 4 and 8, the non-overlap scheduler, KV cache manager V1 or V2
with block reuse on or off, and greedy sampling.

Everything else fails closed and takes today's terminal path. That covers the loss of rank 0, a
second loss, a rank whose experts have no replica, link, fabric or device loss, CFT, online
EPLB, the overlap scheduler, speculative or guided decoding, beam search, PEFT, a KV connector or
transceiver, PP or CP, and LM-head TP in ADP. `PeerLossRecovery.create()` rejects these
configurations at startup.

## 2. Background

**WideEP data path.** Attention runs data-parallel, so each GPU serves its own requests. The MoE
experts spread across GPUs. At EP32, R1's 256 experts sit 8 per GPU, plus one spare slot per GPU.
Every MoE layer runs two NVLinkOneSided kernels around the local expert GEMMs. Dispatch pushes
each token's payloads into the workspaces of the GPUs that own its top-k experts. The payloads
are the hidden state, its scales, the expert ids and the routing weights. Combine pulls the
expert outputs back and sums them per token.

The workspaces live in MNNVL symmetric memory. Every peer maps each GPU's buffer, so a kernel can
load from and store to another GPU directly. Each kernel signals its peers with completion flags
and spins until it sees theirs. A dead peer never signals. Survivors spin until the device
timeout traps, and the trap kills their CUDA context.

**CUDA graphs.** Decode replays graphs captured at startup. A graph bakes in kernel arguments and
tensor addresses. Any change after a loss must therefore live in device memory that the kernels
read at run time.

**EPLB.** Expert placement maps logical experts to physical slots. Static placement is fixed at
startup through `initial_global_assignments` with `layer_updates_per_iter: 0`. Online EPLB moves
weights while serving. A spare slot that holds a replica of another rank's expert lets routing
avoid a dead rank without moving any weights.

**ULFM.** User-Level Failure Mitigation is the MPI fault-tolerance extension in Open MPI 5. You
enable it with `mpirun --with-ft ulfm`, and mpi4py 4 and later expose it. Without ULFM, one dead
process aborts the whole job. With it, three things change.
- An MPI call that involves a dead process fails with `ERR_PROC_FAILED` instead of hanging.
- `Comm.Revoke()` marks a communicator unusable on every member. Ranks blocked in it, or entering
  it later, get `ERR_REVOKED`. We use it to wake everyone.
- `Comm.Shrink()` runs a fault-tolerant agreement among the live members and returns a new
  communicator without the dead ones. Every survivor gets the same membership, even if they
  noticed the failure at different times.

**mpi4py.futures.** TRT-LLM's MPI session dispatches worker tasks through `MPICommExecutor` on
`COMM_WORLD`. It has no ULFM support. With a dead member it can neither run tasks nor shut down
cleanly.

## 3. What survivors must do

1. Unblock the host collectives and MoE kernels that wait on the dead rank.
2. Discard the interrupted step, whose results may be garbage, and undo its reservations so it
   can run again.
3. Agree on the new membership. All survivors install it, or none resumes.
4. Route around the dead rank without changing anything a captured graph depends on.

## 4. Invariants

- Logical EP size, rank ids, graph shapes, slot layout and tensor addresses never change. Only
  the host membership shrinks, through the ULFM survivor communicator.
- Nothing reads or reuses memory that the dead rank could still write into a survivor's
  workspace. We call this quarantine.
- Routing excludes a rank before the mask does, because masked kernels drop routes silently.
  The install order is EPLB placement, device routing check, rearm, then mask.
- Only the authority's Commit changes membership. Local evidence only discards a step.
- Every wait in recovery has a bound. Any failure takes the terminal path.
- With FT off, behavior and performance stay the same. The FT-off kernels compile to the same
  opcodes as main.

## 5. Recovery sequence on each survivor

1. **Detect.** Either a host collective raises `HostCollectiveInterrupted`, or the watcher fires.
   The watcher thread posts an `Irecv(ANY_SOURCE)` that no rank matches, on a private
   communicator. ULFM completes it with an error when any member dies. The watcher then sets the
   step-abort flag, requests a device abort, and revokes the host communicators so every blocked
   survivor wakes. The abort record lives in pinned host memory that kernels read through its
   mapped address, so the request is a plain CPU store. It needs no CUDA work to make progress,
   so it reaches kernels that occupy the whole GPU, eager or graph-replayed.
2. **Discard.** The commit gate sits before `_update_request_states`. It syncs the step and raises
   if the flag is set. Rollback then runs `revert_prepare_resources`. V1 releases first-chunk
   sequences without storing their blocks for reuse and rewinds generation tokens. V2 undoes the
   scheduler's allocations. V2 allocates during scheduling, so the executor records the step right
   after scheduling, and a discarded V2 step never commits blocks for reuse. Rollback also
   retires the ADP padding request. Unscheduled requests stay in rank 0's queue.
3. **Agree.** Revoke, call `Shrink`, and send `Stopped` with the failed rank to the authority.
4. **Prepare.** On `Propose`, each survivor computes the masked EPLB placement, checks that every
   expert still has a slot, and replies `Ready` with a placement digest. The authority requires
   identical digests.
5. **Install.** On `Install`, each survivor switches to the survivor communicator, installs the
   masked placement, checks device routing, rearms its own workspace row and abort record, and
   installs the mask. It replies `Installed`.
6. **Commit.** On `Commit`, survivors resend the responses rank 0 did not process, fail the dead
   rank's requests, restart the watcher, and resume the same active requests.

A rank whose own event loop crashes while recovery could replace it exits alone with SIGKILL, so
the survivors recover from its loss as from any other.

Recovery runs under the executor's `HangDetector`, which hard-kills the job if a step stalls.
Workers wait at most 120 s for the authority, whose own deadline is 60 s. At EP32, survivors'
requests complete 0.8 s after an idle kill and 1.4 s after an in-flight kill.

## 6. Components

**MoE data path.** Files: `moeAlltoAllKernels.cu`, `moeAlltoAllOp.cpp`, `execution_state.py`,
`nvlink_one_sided.py`. With FT on, dispatch and combine read one execution descriptor at a fixed
address inside the graphs. It holds the active-rank mask words and an abort record with
`requested` and `status`.
- Waits and loads skip masked ranks.
- Spin loops exit when `requested` is set. A real timeout still traps. An aborted combine
  withholds its readiness flag, so no peer pulls its garbage outputs.
- Each source rank gets a fixed slot. Main places dispatch rows by this step's max tokens per
  rank R, as `[payload][source][R rows]`. Addresses move whenever R changes, so a dead rank's late
  writes can land on live rows. FT lays out the same workspace as `[source][payload][C rows]`,
  where C is the static `max_num_tokens`. A source's footprint never moves, and the layout needs
  no extra workspace.
- An FT-only gather kernel, launched with PDL, copies the valid rows into the dense `[ep, R, row]`
  tensors the expert kernels expect. Main hands out these tensors as zero-copy workspace views.
  Rows from masked sources, rows past a source's count, and every row of an aborted step get the
  invalid expert id and a zero payload.
- Dispatch writes into peers' memory, and combine only reads from it. So only dispatch needs
  write containment.

`MoeExecutionState` owns the descriptor and provides `request_abort`, `rearm` and `install_mask`.

**MPI adapter.** `MPIDist` in `communicator.py`. `enable_peer_loss_recovery` requires mpi4py 4 or
later, ULFM enabled at launch, and `MPI_THREAD_MULTIPLE`. It checks `OMPI_MCA_mpi_ft_enable`,
because the library exposes the ULFM calls even when the launch did not enable them. It then moves
host collectives to a private duplicate of `COMM_WORLD`, so revoking that communicator leaves the
MPI session usable. The adapter also provides `revoke`, `shrink`, `install_survivors` and
`PeerFailureWatch`. After a loss, gathers keep each result at its logical rank and fill the dead
rank's position with a neutral value.

**EPLB.** Files: `moe_load_balancer.py`, `model_engine.py`. `covers_loss_of(rank)` decides which
ranks can be recovered. `install_resident_placement` points routing at the resident replicas.
No weights move.

**Authority.** Package `executor/recovery/`, with no torch dependency. It is one CPU process
outside the workers' launch and termination group. v7 requires it, because agreement among
survivors over MPI does not prove they can make progress. It speaks ZeroMQ ROUTER/DEALER, signed
with HMAC, and assumes a trusted network with no replay protection. Workers send `Register` with
the ranks they can recover from. The authority replies `Registered` with the admitted ranks: the
intersection of those sets, never rank 0. A recovery then runs `Stopped`, `Propose`, `Ready`,
`Install`, `Installed`, `Commit`. `Abort` is terminal, and losing the authority fails closed.

**Executor.** `peer_loss_recovery.py` plus hooks in `py_executor.py`. It owns the commit gate,
step recording and release, rollback, the configuration checks, and rank 0's map from request to
owning rank, which it uses to fail the dead rank's requests. It keeps numbered response gathers
so it can resend an interrupted one, and it defers request terminations until delivery.

**MPI session.** `mpi_session.py`. After a loss, new tasks fail fast. Shutdown flushes the logs
and ends with `MPI_Abort(0)`, which bounds teardown as v7 1d.1 asks. Every rank exits, but on
multiple nodes `mpirun` can outlive them after the abort, so the launcher must stop it once the
server exits. mpi4py's threads print `MPI_ERR_PROC_FAILED` tracebacks, which are harmless.

**Launch.** Run `mpirun --with-ft ulfm --mca pml ob1 --mca btl self,tcp --mca coll ^hcoll,ucc`.
ULFM needs these components. MPI carries only host control traffic, and MoE data uses NVLink.
Under Slurm, run mpirun inside the allocation, because PMIx kills the whole job step when one
task dies.

## 7. Decisions and rejected alternatives

- We recover only from process loss. Link or device loss can leave sticky CUDA errors, after
  which no graph can run. A rank that is alive but unreachable must also be fenced first. v7 item
  1d.4a covers this later.
- We discard and re-run the step instead of salvaging it. That is correct for paged KV and greedy
  sampling. We reject features that carry state across steps: recurrent state, penalties, seeds
  and speculative decoding.
- We chose fixed slots and a gather. Pull-based dispatch would add a round trip to every MoE layer
  and rewrite main's hot kernel. Strided expert inputs would change every MoE backend's input
  contract. The cost is one on-GPU copy per MoE layer, with FT on only.
- The kernels read a device descriptor at run time instead of recapturing graphs, so graphs stay
  valid by construction.
- The external authority follows v7. `Shrink` only rebuilds the communicator.
- We dropped the frontend admission pause, which v7 does not ask for. New requests wait in rank
  0's queue.
- mpi4py 4.1.2 has the ULFM bindings and matches the security-scanning lock file. The bounded
  teardown covers the mpi4py.futures gap. Fixing it cleanly, upstream or with a TRT-LLM
  dispatcher, is a Phase 1 decision.
- Environment variables gate the feature while it is experimental: `TLLM_FAULT_TOLERANCE_MODE`
  plus the authority variables. A public `LlmArgs` field is follow-up work.

## 8. Cost

FT off costs nothing. FT on adds the gather per MoE layer, one sync per step at the commit gate,
10 ms watcher polling, and `ob1` over TCP for host control traffic. The perf gate in
`REVIEW_FINDINGS.md` records the measured overhead. The target is 1 to 2% at most.

## 9. Validation

- Unit tests cover the authority and protocol, the ULFM adapter with fakes and with a real 4-rank
  `mpirun --with-ft ulfm` run, session teardown, and rollback for V1 and V2. One rollback test
  checks that a discarded step never becomes a reusable prefix. Kernel tests cover abort inside
  captured graphs, slot quarantine, count bounds and the timeout trap.
- The CI integration test `tests/integration/defs/wide_ep_ft/` runs post-merge on GB200 4-GPU.
  It serves DeepSeek-V3-Lite at EP4 and SIGKILLs rank 2 while idle and mid-decode.
- EP32 qualification uses this directory's scripts on R1. It compares tokens exactly with a
  same-routing oracle. Greedy tokens depend on which rank serves each expert, so the oracle's EPLB
  excludes the victim from the start. It runs idle and in-flight kills and checks that nothing
  recaptures, teardown is bounded, and performance holds. A Kimi-K2 run checks robustness.

## 10. Limits and next phases

The MVP survives one loss. Rank 0 must survive, recoverable ranks need replicas, and only
aggregated serving is supported. The dead rank's requests fail instead of migrating. Online EPLB
needs the quiesced EPLB integration of v7 item 1b.3.

Phase 2 restores the job to N ranks. It fences the dead process, rearms its slot and readmits a
replacement, with the same layout and descriptor. Later work covers the CFT path, Ray,
disaggregated serving, repeated failures, and removing the gather.

## 11. v7 draft PRs

#19610, the NVLinkOneSided overhaul, is the main conflict. We rebase after it merges. #15895
proposes a competing abort ABI that does not work with graphs, so we align with its owner.
#15785, #15789 and #15788 do not touch this path. `DRAFT_PRS.md` has the details.

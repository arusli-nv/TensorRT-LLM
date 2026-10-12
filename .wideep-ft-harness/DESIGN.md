# WideEP worker-loss recovery — Phase 1 MVP design

Status 2026-10-11. Implements the MPI / non-CFT MVP of "WideEP Fault Tolerance in TensorRT-LLM"
(v7; items 1a.8, 1b.x, 1c.3a–1c.4c, 1d.1, 1d.4). Code: branch `wideep-ft` (split plan in
`PR_PLAN.md`); qualification harness in this directory.

## 1. Goal and scope
A WideEP deployment keeps serving after **one worker process** dies, idle or mid-step, and
continues on the **CUDA graphs captured at startup** (no reload, recompile, recapture, retune).
Survivor-owned requests continue exactly (the interrupted step is discarded and re-run);
requests the dead attention-DP rank served fail with a retryable error.

Qualified deployment: DeepSeek-R1 NVFP4, EP32 on 8×4 GB200, attention DP, expert TP 1, static
EPLB with replica slots, NVLinkOneSided fence (non-CFT) all-to-all, decode graphs [1,2,4,8],
non-overlap scheduler, KV cache manager V1 or V2 without block reuse, greedy sampling.

Fail closed (today's terminal path): rank-0 loss, a second loss, a rank whose experts lack a
replica, link/fabric/device loss, CFT, overlap scheduler, speculative/guided decoding, beam
search, block reuse, PEFT, KV connector/transceiver, PP/CP, LM-head TP in ADP. `create()`
rejects unsupported configurations at startup.

## 2. Failure model and the four problems
A SIGKILLed rank stops answering MPI and NVLink. Survivors must:
1. **Unblock** host MPI collectives and device MoE kernels that wait on the dead rank.
2. **Discard safely**: the interrupted step's results may be garbage; its resource reservations
   must be undone so it can re-run.
3. **Agree**: every survivor installs the same membership, or none resumes.
4. **Route around** the dead rank's experts, host collectives and NVLink peers without changing
   anything a captured graph depends on.

## 3. Invariants
- Logical EP size, logical rank ids, graph shapes, slot geometry and tensor addresses never
  change. Only host membership is compacted (a ULFM survivor communicator).
- A dead rank's NVLink write targets (its per-source slot, flags, counters in each survivor's
  workspace) are never read or reused again (quarantine).
- Routing excludes a rank before the mask does (kernels silently skip masked ranks):
  install order is EPLB placement → device routing check → rearm → mask.
- Membership changes only through the authority's Commit; local evidence only discards a step.
- Every recovery wait is bounded; any failure ⇒ terminal path.
- FT off ⇒ no behavior or performance change (FT-off kernels opcode-identical to main).

## 4. Recovery sequence (each survivor)
1. **Detect.** Either a host collective raises `HostCollectiveInterrupted` (ULFM
   `ERR_PROC_FAILED`/`ERR_REVOKED`), or the watcher thread fires: it holds an
   `Irecv(ANY_SOURCE)` that nobody matches on a private communicator, which ULFM completes with
   an error when any member dies. The watcher sets the step-abort flag, requests a device abort
   (pinned host→device copy on a side stream: needs no SM, so it reaches kernels occupying the
   GPU) and revokes the host communicators so every blocked survivor wakes.
2. **Discard.** The commit gate (before `_update_request_states`) syncs the step and raises if
   the flag is set. Rollback: `revert_prepare_resources` (V1: free first-chunk sequences, rewind
   generation tokens; V2: undo scheduler allocations — V2 allocates at scheduling, so the step is
   recorded right after scheduling), retire the ADP padding request. Unscheduled requests stay
   queued on rank 0 and are re-broadcast.
3. **Agree.** Revoke, ULFM `Shrink` (a fault-tolerant agreement: every survivor gets the same
   failed set), `Stopped(failed)` → authority.
4. **Prepare.** `Propose` → each survivor computes the masked EPLB placement and checks every
   expert stays covered → `Ready(placement digest)`; the authority requires identical digests.
5. **Install.** `Install` → survivor communicator, masked placement, device routing check, rearm
   (reset own workspace row + abort record), install mask → `Installed`.
6. **Commit.** `Commit` → redeliver responses rank 0 did not process, fail the dead rank's
   requests, re-arm the watcher, resume the same active requests.

All of it runs under the executor's `HangDetector` (hard kill on lapse); authority waits use a
120 s worker deadline above the authority's 60 s recovery deadline. Measured on EP32: survivors'
requests complete 0.8 s (idle) / 1.4 s (in-flight) after the kill.

## 5. Components
**MoE data path** (`moeAlltoAllKernels.cu`, `moeAlltoAllOp.cpp`, `execution_state.py`,
`nvlink_one_sided.py`). Dispatch *pushes* tokens into receivers' workspaces (the sender knows the
routing; remote writes are one-way). Combine *pulls* expert outputs and reduces in registers (the
token's owner knows where results are and does the sum). Only dispatch lets a peer write into a
survivor's memory, so only dispatch needs write containment. FT adds one fixed-address
**execution descriptor** `[active-rank mask words…, {requested, status}]`, read by the kernels
inside captured graphs:
- mask: waits and reads skip masked ranks;
- abort: spin loops exit when `requested` is set (status = cancelled); timeouts still trap; an
  aborted combine withholds its readiness so peers never pull its garbage;
- **fixed per-source slots**: main places dispatch payloads by the step's runtime token count R
  (`[payload][source][R rows]`), so addresses move between steps and a dead rank's late writes
  could hit live rows. FT lays out the same workspace reservation as `[source][payload][C rows]`
  with static capacity C, so each source's footprint never moves;
- **gather** (FT only, PDL-launched): copies valid rows into the dense `[ep, R, row]` tensors the
  expert kernels expect (main gets these zero-copy as workspace views); rows from masked
  sources, beyond a source's count, or of an aborted step get the invalid expert id and zeros.

`MoeExecutionState` owns the descriptor (`request_abort`, `rearm`, `install_mask`).

**MPI adapter** (`communicator.py`). `enable_peer_loss_recovery` requires mpi4py ≥ 4, ULFM
enabled at launch (`OMPI_MCA_mpi_ft_enable`; the library exposes ULFM calls either way) and
`MPI_THREAD_MULTIPLE`, then moves host collectives to a private duplicate of the world (revoking
it leaves the MPI session's `COMM_WORLD` usable). `revoke`, `shrink`, `install_survivors`
(logical-rank gathers with neutral fill values for the dead rank), `PeerFailureWatch`.

**EPLB** (`moe_load_balancer.py`, `model_engine.py`). Static placement with spare slots holding
replicas; `covers_loss_of(rank)` defines recoverable ranks; `masked_placement_digest`;
`install_resident_placement` reprograms routing to resident replicas (no weight movement).

**Authority** (`executor/recovery/`, torch-free). One CPU process outside the workers'
termination group (v7: agreement among survivors over MPI does not prove post-death progress).
ZeroMQ ROUTER/DEALER + HMAC (trusted network, no replay protection).
`Register(recoverable)` → `Registered(admitted = intersection, never rank 0)`;
`Stopped → Propose → Ready → Install → Installed → Commit`; `Abort` is terminal. Authority loss
fails closed.

**Executor** (`peer_loss_recovery.py`, hooks in `py_executor.py`). Commit gate, step record /
release, rollback, rank-0 request→owner map (fails the dead rank's requests), numbered response
gathers kept for redelivery, deferred terminations. `create()` validates the configuration.

**MPI session** (`mpi_session.py`). mpi4py.futures cannot serve or shut down a world with a dead
member: after a loss new tasks fail fast and shutdown ends with `MPI_Abort(0)` after flushing logs
(bounded teardown, v7 1d.1). Its threads print `MPI_ERR_PROC_FAILED` tracebacks (cosmetic).

**Launch.** `mpirun --with-ft ulfm --mca pml ob1 --mca btl self,tcp --mca coll ^hcoll,ucc` (ULFM
needs FT-capable components; only host control traffic uses MPI). Under Slurm, launch mpirun
inside the allocation (PMIx kills the whole step when a task dies).

## 6. Key decisions and rejected alternatives
- **Process loss only.** Link/device loss can leave sticky CUDA errors (no graph reuse possible),
  and a live-but-unreachable rank must be fenced first (v7 1d.4a, later).
- **Discard and re-run** the interrupted step instead of salvaging it: correct for paged KV and
  greedy sampling; stateful features (recurrent state, penalties, seeds, spec decode) rejected.
- **Fixed slots + gather** instead of pull-based dispatch (adds a round trip to every layer and
  rewrites main's hot kernel) or strided expert inputs (changes every MoE backend's contract).
  Cost: one local copy per MoE layer, FT on only.
- **Device descriptor read at run time** instead of recapture: graphs stay valid by construction.
- **External authority**, as v7 requires; Shrink only rebuilds the communicator.
- **No frontend admission pause** (not in v7): new requests wait in rank 0's queue.
- **mpi4py 4.1.2**: ULFM bindings, matches the security-scanning lock. The futures/ULFM gap is
  bounded; the clean fix (upstream mpi4py or an own dispatcher) is a Phase 1 decision.
- **Env-var gate** (`TLLM_FAULT_TOLERANCE_MODE` + authority variables) for the experimental MVP; a
  public LlmArgs field is follow-up.

## 7. Costs
FT off: none (kernel opcodes identical, host paths gated). FT on: the gather per MoE layer, a
per-step sync at the commit gate, 10 ms watcher polling, `ob1`/TCP for host control traffic.
Measured overhead: perf gate in `REVIEW_FINDINGS.md` (target ≲ 1–2%).

## 8. Validation
- Unit: authority/protocol; ULFM adapter (fakes + a real 4-rank `mpirun --with-ft ulfm` test);
  session teardown; executor recovery and rollback (V1/V2 GPU round trip); kernels (abort inside
  captured graphs, slot quarantine, count bounds, timeout trap).
- Integration (CI, GB200 4-GPU post-merge): `tests/integration/defs/wide_ep_ft/` — DeepSeek-V3-Lite
  EP4, SIGKILL rank 2 idle and mid-decode.
- Qualification (harness here, EP32 R1): exact tokens vs a same-routing oracle (greedy tokens
  depend on which rank serves each expert, so the oracle's EPLB excludes the victim from the
  start), idle and in-flight kills, no recapture, bounded teardown, perf A/B; Kimi-K2 robustness.

## 9. Limits and next phases
One loss; rank 0 must survive; coverage requires replicas; aggregated serving only; the dead
rank's requests fail rather than migrate. Phase 2 (restore N): fence the dead process, rearm its
slot, re-admit — same layout and descriptor. Later: CFT path, Ray runtime, disaggregated serving,
repeated failures, removing the gather copy.

## 10. Draft PRs in v7
#19610 (NVLinkOneSided overhaul) is the biggest conflict: rebase after it merges. #15895 (abort
ABI) competes and is not graph-compatible: align with its owner. #15785/#15789/#15788 are not on
this path. Details in `DRAFT_PRS.md`.

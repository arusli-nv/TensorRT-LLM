# Round 4 design review: is the WideEP FT MVP a sound, lean foundation?

Scope: `fold` (branch `wideep-ft-folded`, `git diff origin/main..HEAD`), DESIGN.md, MVP_ACCEPTANCE.md, v7 text,
probe-abort/ outputs, ep32/ configs. Paths are relative to `fold/` unless noted.

**Verdict.** The core architecture is right and I would keep it. It has a device descriptor at a fixed address,
fixed per-source slots, a sticky abort record that is only rearmed while every survivor is quiescent, ULFM
Revoke/Shrink on a private duplicate communicator, logical-rank-preserving gathers, and an external authority gating
Install and Commit. Phase 2, 1c.6, CFT and online EPLB all fit on top without a redesign of the data path. Three
things stand between this and a production-credible MVP:
1. The abort request's delivery mechanism is not robust.
2. Rank-local fatal errors still kill the whole world.
3. Health is not reported truthfully, which v7 makes a release gate.

The perf gate also needs re-measuring on a production-shaped config.

## Ranked recommendations

1. **Replace the side-stream H2D abort copy with a host-mapped abort word written by the CPU** (section 4). It is
   small, it removes an undocumented GPU-scheduling dependency and an unbounded wait, and it covers eager prefill and
   eager decode above the captured batch sizes, which today is unproven or failing.
2. **With FT on, a rank-local fatal error on an admitted rank should exit only that process, not the world.**
   `_event_loop_wrapper` → `hard_kill_on_rank_crash` (`py_executor.py:1489`, `hang_detector.py:253`) still ends in
   `propagate_hard_kill`, which is MPI_Abort of everyone. Real faults usually surface as a CUDA error or an exception
   in the victim (Xid, ECC, illegal address), not as SIGKILL. So the envelope you qualified (SIGKILL) is not the one
   production hits. Fix: when recovery is enabled and this rank is admitted, flush logs and SIGKILL self, and let ULFM
   turn it into a process loss. This is still "local fatal stays fatal" in v7's sense, and it is a few lines.
3. **Truthful health and metrics (v7 1d.2/1d.3, a listed MVP release gate: "truthful health/metrics").** The doc
   says the opposite (`docs/source/features/wide-ep-fault-tolerance.md:89`, "keeps reporting healthy").
   - Minimum: `/health` reports `degraded` (and `recovering` during the window).
   - A gauge of live EP ranks, and a flag `ft_armed` that goes false after the one allowed loss and when the
     authority is unreachable.
   - A recovery counter and a per-stage duration log line or histogram (detect, abort, shrink, ready, install,
     commit, first token).
   - Add a 1d.2/1d.3 row to MVP_ACCEPTANCE; it is currently missing.
4. **Re-run the perf gate on a production-shaped config.**
   - Both measured configs are `disable_overlap_scheduler: true`, `max_batch_size: 8`, `max_num_tokens: 256` (`ep32/config.yaml:23-29`).
   - The 1.4% is therefore FT-on vs a non-default baseline, at a size where the gather is cheapest.
   - The cost users actually pay is overlap-scheduler loss plus the commit-gate sync plus the gather at large decode
     batches (K-S6 is still open: about 1 ms/step at EP64).
   - Measure FT-on against the FT-off production default (overlap on) from `examples/configs/database`.
5. **Close the open 1a.8a, 1a.11 and 1c.4b evidence.**
   - Record the 2-process late-write run (`tests/unittest/_torch/moe/multi_gpu/test_moe_a2a_late_writes.py` exists
     and is listed, but MVP_ACCEPTANCE still says "being written").
   - Finish the captured-collective audit.
   - Run physical fault injection at Propose, Install and Commit: kill the authority, and kill a second rank.
   - Add one physical mid-MoE kill during an eager prefill.
6. **Cheap now, avoids churn later:**
   - Widen the protocol to `failed_ranks: tuple[int, ...]` (`protocol.py:72,97`).
   - Have `Registered` carry the authority's recovery timeout, so workers check `authority < worker < hang detector`.
   - Allgather the FT flag at startup. The env gate is read per rank; a rank missing the env silently uses the main
     slot layout and the job sits 600 s in registration.
7. **Decouple logical rank from COMM_WORLD rank** (`communicator.py:999-1003` `_world_ranks` translates to
   `mpi_comm()`). Move `_to_logical`, `_compact_root` and `live_ranks` into `Distributed`. This is the one structural
   change that Phase 2 and Ray both need (section 2).
8. **Lower the device timeout when FT is on**, for example 60 s instead of 300 s (`moeAlltoAllKernels.cu:79`). Also
   consider `%globaltimer` instead of `clock64` at an assumed 2 GHz (`.cu:114,250`). A hung-but-alive peer is out of
   envelope and fails closed only by trap. Today that is a 5-minute outage, and longer under SM clock throttling,
   where the hang detector races the trap.

## 1. v7 MVP conformance

| v7 item | v7 acceptance | Assessment |
|---|---|---|
| 1a.8 abort | "bounded abort without context poisoning or escaped failed outputs" | **Partial.** Met under graph replay: EP32 mid-MoE 2.6 s, exact. Outputs are contained by the combine zeroing, the gather invalidating rows and the commit gate (`peer_loss_recovery.py:258-276`). Delivery of the abort itself is not proven for eager launches, and the probe shows it failing (section 4). |
| 1a.8a late writes | "issued-before-death writes cannot contaminate reused storage; record supported software/hardware" | **Met by construction** (fixed `[source][payload][C]` slots; masked flag, counter and stats rows). Physical evidence: the test exists, but no recorded result. The software/hardware envelope is not recorded. |
| 1a.8b integration | "real fault-injected abort/quiesce/reset/replay preserves contexts and correct results" | **Met for decode graphs** (EP32 R1 idle 0.8 s, in-flight 1.4 s, mid-MoE 2.6 s, exact). Not shown for eager prefill. |
| 1a.11 graphs | "same graph executable/static addresses, zero recovery capture/compile/tuning, all-active perf measured"; scope includes "per-execution generation latching, replay-time watchdog/error wiring and captured-collective audit" | **Partial.** The descriptor and zero-recapture claims hold. The audit is open. Latching and the watchdog are deliberate deviations (below). |
| 1b.2a admission | "reject insufficient coverage ... admit concrete resident-replica configurations" | **Met.** `covers_loss_of`, the authority's intersection, and a re-check at Prepare. Overflow is not an issue because A2A capacity is per source. |
| 1b.4a mapping | "sparse membership and compact control results map consistently to logical slots" | **Met.** But identity is derived from COMM_WORLD (rec. 7). |
| 1c.3a / 1c.4a | "no dead-rank acknowledgment, conflicting proposals rejected, control loss fails closed"; "remove a middle rank without poisoned-world calls" | **Met** (`authority.py` `_dispatch` rejects the departed rank and conflicting Stopped; real 4-rank ULFM test). |
| 1c.4b coordinator | "fault injection at every transition prevents mixed state or early resume; coordinator loss fails closed" | **Met in unit tests, not physically** (rec. 5). There is no authority liveness while serving, so loss is only noticed at the next failure (fails closed, but silently disarmed). |
| 1c.4/1c.4c | "no partial logits, duplicate streamed tokens or uncertain KV reuse; explicit error" | **Met for the greedy profile.** The sampling restrictions (penalties, seeds, stop words) are docs only (`peer_loss_recovery.py:127-131`). That is acceptable: a re-run never duplicates streamed tokens, it only resamples. |
| 1d.1 lifecycle | "process death does not terminate healthy workers; unsupported fallback rejected"; bounded teardown; unified deadlines | **Partial.** SIGKILL deaths are fine. A *local error* on a worker still kills everyone (rec. 2). Teardown is MPI_Abort, and multi-node mpirun can outlive it (documented). Deadlines are not cross-validated with the authority. `create()` runs after model load (slow failure). |
| 1d.2/1d.3 health | gate: "truthful health/metrics"; "stale telemetry is unavailable, never apparently healthy" | **Not done**, and the doc states the opposite (rec. 3). |
| 1d.4 physical | "bounded resume, no late-write contamination, correct requests, agreed state, zero ... eager serving" | **Met on EP32 R1** for idle, in-flight and mid-MoE (graph). Kimi-K2 is pending, and so is a mid-prefill kill. |
| 1d.5 perf | "all-active overhead, degraded goodput/tail latency, every recovery stage" | **Measured, but on an unrepresentative config, with no tail latency** (rec. 4). |

### Deliberate deviations: are they sound?

- **No per-execution generation latching.** Sound under the current invariant, which is enforced:
  - The mask is written only after `_join_device_work` (bounded) and a device-wide sync in `rearm`.
  - `install_mask` refuses while the abort record is non-zero (`execution_state.py:150-152`).
  - No survivor resumes before `Commit`, which requires every `Installed`.
  - So no execution on any survivor can straddle a mask change, and latching would be redundant.
  - The invariant breaks only if someone later wants to mask a peer *live*, without quiescing (for example
    evidence-driven exclusion of a hung peer). Write the invariant into `MoeA2AFenceDescriptor`'s comment as a
    contract and keep it.
- **No replay-time host watchdog** (`nvlink_one_sided.py:973`, "relies on the device timeout"). This is sound for the
  process-loss envelope, because ULFM is a better evidence source than a kernel watchdog. It leaves hung-alive peers
  to the 300 s trap. It is acceptable for the MVP only if documented as out of envelope and the timeout is shortened (rec. 8).
- **Env-var gate instead of an `LlmArgs` field.** Fine while experimental. The real risk is cross-rank inconsistency
  (rec. 6), not the lack of a public API. v7 1d.1 "public admission policy" needs the field before GA.
- **External authority.** This is v7-conformant, not a deviation. Its marginal value over Shrink in the MVP is
  modest: digest agreement, an Install barrier and fail-closed. It is the correct home for Phase 2 and Phase 3
  ("reuse this authority for Phase 2 Snapshot"). It is a deployment burden: one more process, a key and an endpoint,
  with no launcher integration.
- **Discard and re-run instead of salvage.** Sound for paged KV plus greedy. It is the reason for the long reject
  list in `create()`.

## 2. Foundation for later phases

Nothing in the MVP *forces* a redesign. The data-plane choices (descriptor, fixed slots, sticky abort, rearm only
while quiescent) are exactly what Phase 2 and 1c.6 need. The friction is in the host control plane.

**Phase 2 (fence, rearm the slot, readmit).**
- *Fits:*
  - The descriptor mask can set the bit back. `EPGroupHealth.mark_active` exists.
  - Fixed slots make "the dead rank's slot stays quarantined until fenced" explicit.
  - `rearm` resets the whole local row, so readmission is one more quiesce, install and commit cycle on all ranks.
  - The descriptor also enables a *solo warmup* of the replacement: a mask with only itself set lets it run
    autotune, warmup and capture without peers. Main's warmup issues EP collectives the survivors would not join, so
    this matters.
- *Must change:*
  1. **The host plane cannot admit a process.**
     - `_world_ranks` and `_known_failed_ranks` translate everything to COMM_WORLD (`communicator.py:999`).
     - The MPI session is mpi4py.futures on COMM_WORLD, in degraded "shutdown only" mode after a loss (`mpi_session.py` `_live_worker_futures`).
     - A replacement needs MPI dynamic processes (PRRTE + `Comm_connect`/`Intercomm_merge`) or a non-MPI host group
       rendezvoused through the authority.
     - This is the biggest Phase 2 risk. Prototype the readmission handshake early, and make logical rank an
       explicit attribute of each communicator member (rec. 7).
  2. **MNNVL remap at a fixed VA.** Survivors must unmap the dead rank's chunk and map the replacement's handle at
     the same address, so captured kernel arguments stay valid. `MnnvlMemory.pin` is whole-allocation
     (`_mnnvl_utils.py:329`), so it needs a per-rank remap. The imported dead allocation also keeps HBM pinned on the
     dead GPU until survivors unmap it.
  3. **The fence.** Evidence that the dead context is torn down before its slot is reused. Candidates: NVML on that
     node shows no process, or EXCLUSIVE_PROCESS compute mode so that the replacement's context creation implies
     teardown. If the driver team confirms that "ULFM reported exit ⇒ channels torn down", the same proof could
     retire the gather and its K-S6 cost.
  4. **The authority.** Its world and registration are closed after REGISTERING (`authority.py:165`), and RECOVERED
     is terminal. It needs a readmit transaction.
  5. **EPLB.** `reconfigure_mask_only` needs an unmask counterpart.

**Repeated or concurrent failures (1c.6).** These are local changes, and none is blocked:
- the `shrink` one-shot guard (`communicator.py:938`);
- `len(failed) != 1` (`peer_loss_recovery.py:339`);
- protocol `failed_rank: int` (rec. 6);
- the authority's state machine needs RECOVERED→SERVING, and cancel-and-re-propose with a higher generation when a
  Stopped arrives carrying a larger failed set;
- admission must cover failure *combinations*, so compute coverage at Prepare (already done there) and make the
  registration admitted set advisory;
- the `_gathers` deque (maxlen 2) assumes one interruption, so re-check it for nested ones.

Partial installs are safe because `rearm` and `install_mask` are idempotent and resume is gated on Commit.

**Online EPLB (1b.3).**
1. `create()` requires static routing (`peer_loss_recovery.py:180`). Coverage becomes per placement version, so
   registration-time admission must become advisory. The Prepare-time `masked_placement_digest` check becomes the
   authority and must be computed against the *committed* placement version.
2. The planner needs a coverage invariant (at least one replica outside each failure domain), or online EPLB
   silently disarms FT.
3. EPLB's `wait_gpu_stage` spin kernels and the weight-copy pipeline need abort integration, and an in-flight weight
   update needs v7's "reversible assignments". The descriptor's abort word can be shared.
4. `_shared_barrier` already tolerates loss.

**CFT.** The kernels hard-reject descriptor+CFT (`TLLM_CHECK` in both launchers) and `bind_execution_state` rejects
CFT. The descriptor struct is mode-neutral, so CFT only needs its own abort polling, completion and teardown proof.
The gather is fence-only. No API change.

**Ray.**
- `Distributed` already has the adapter surface: `revoke`, `shrink`, `install_survivors`, `watch_peer_failures`, `live_ranks`.
- The logical/compact translation and fill semantics live in `MPIDist`; move them to the base (rec. 7).
- Ray needs a survivor process group rebuilt through a store (Gloo has no revoke), and actor-death watching
  (`ActorDiedError`).
- `create()` hard-requires the MPI executor, and EPLB's host weight sharing uses an MPI shared communicator (v7 calls this out).

## 3. Over-design and under-design

**Heavier than needed (simplify when convenient; none is urgent):**
- **Departed-rank conventions in gathers: five idioms.** `None` entries, scalar `fill_value`, a per-column
  `fill_value` with the magic `_ADP_BALANCE_DEPARTED_RANK_COUNTS`, filtering in `_gather_rank_status`, and
  `live_ranks` slicing in `cuda_graph_runner.py`. Pick one: logical-indexed results with `None`, and callers iterate
  `dist.live_ranks`. This removes the `fill_value` API from `Distributed`.
- **Three generation counters:** the authority's, `EPGroupHealth`'s, and the `expected_generation` loop in
  `install_resident_placement`. Make `EPGroupHealth.generation` *equal* to the authority generation and assert it.
- **Three round trips** (Stopped→Propose→Ready→Install→Installed→Commit). After Shrink every survivor knows the
  failed set, so Stopped can carry the digest and Propose can go (r2 S-A1). Latency is not a problem (0.14 s
  kill-to-commit); this is about code and test surface. Optional.
- **Torn request-broadcast handling is three mechanisms:** `_unbroadcast_requests`,
  `_requeue_unscheduled_requests` and `_readmit_undelivered_requests`. They could be one post-Commit reconciliation
  driven by rank 0's `_request_owners`. Medium-low value.

**Missing, roughly in the order production hits it:**
1. A local fatal error kills the world (rec. 2).
2. Health and metrics, and visibility of "FT disarmed" after the single allowed loss or when the authority is
   unreachable (rec. 3). There is no authority heartbeat while serving.
3. Operator surface:
   - authority status (phase, generation, admitted ranks, failed rank) as a log or a status request;
   - a structured recovery event with stage timings, which today come from the harness, not the code.
4. Launcher burden:
   - mpirun with `--with-ft ulfm` only, not srun or PMIx steps;
   - mpirun can outlive MPI_Abort on multi-node;
   - the authority is started by hand.

   Most fleets need a wrapper. At least ship one script that starts the authority, mpirun and a reaper.
5. Timeout tuning (rec. 6, rec. 8): 600 s registration, 120 s recovery, 60 s authority, 300 s hang detector and 300 s
   device timeout. Only `hang > recovery` is checked (`peer_loss_recovery.py:177`).
6. Faults out of envelope with no fast fail:
   - A hung-alive peer.
   - A process stuck in D-state during GPU teardown, which never "dies" for ULFM.

   Both end in a 300 s trap and a world kill. Document them, and shorten the timeout.

## 4. Robustness

**The abort request: replace it.** Today `request_abort` does a 4-byte pinned H2D `copy_` on a private torch stream,
then `stream.synchronize()` (`execution_state.py:124-133`). Kernels poll the record in device memory with
`ld.relaxed.sys`.

The evidence:
- `probe-abort/out.txt` and `out2.txt`: with the round eager on a non-default stream, the copy is "STUCK" with PDL on
  or off and with `CUDA_DEVICE_MAX_CONNECTIONS` unset, 8 or 1.
- `out3.txt`: a raw `cudaStreamNonBlocking` copy, and a side-stream kernel, are also stuck.
- `out4.txt`: a main-path spin (no descriptor) lets a copy to a *different* address complete.
- The caller's probe: graph replay completes.

The root cause is unknown. Candidates are copy-engine or front-end scheduling while a descriptor-mode kernel spins,
or the copy target being the very line under `.sys` polling. Whatever it is, progress of the abort depends on
undocumented driver and hardware behaviour that varies between eager and graph launches.

Production runs eager for prefill and for any decode batch above the largest captured size. Qualification hid this
with `max_batch_size: 8` and graph sizes 1, 2, 4 and 8.

Failure mode when the copy sticks:
- The watcher thread blocks in `synchronize()` *before* `revoke()`, which sits in `finally` at `peer_loss_recovery.py:243-246`.
- `_recover` calls `request_abort()` again on the main thread (`:333`).
- The kernel spins until the 300 s trap, which poisons the survivor's context, and then the hang detector kills the job.

The result is fail-closed, but it is a 5-minute total outage instead of a recovery. The
`CUDA_DEVICE_MAX_CONNECTIONS > 1` check in `create()` (`:173`) guards against one hypothesis that the probe
already disproves.

**Replacement: a host-mapped abort word.**
- Allocate `MoeA2AAbortState::requested` in pinned, mapped host memory (`cudaHostAlloc(...Mapped)`, or a pinned
  tensor through UVA) and store its device pointer in the descriptor, or pass it next to the descriptor. Its address
  is fixed, so it is graph-safe.
- `request_abort` becomes a CPU store plus a fence. It needs no stream, no copy engine and no synchronize, works the
  same eager or under graphs, and cannot block the watcher.
- The kernels already use `ld.relaxed.sys`, the right scope for host-written memory. On GB200 the read goes over
  coherent C2C. On PCIe hosts it is an uncached read, which is the standard pattern for host-to-device control flags.
- Keep the mask in device memory, because it is read on the hot path.
- Keep `status` in device memory and write it with a plain store, not a system atomic to host memory, which PCIe
  hosts may lack.
- Poll the host word only on the wait path, every N spins, so peer-arrival latency is unaffected.
- The gather should read dispatch's device-local outcome instead of the host word per block.
- `rearm` clears the word after the device-wide sync.
- Also move `revoke()` before the abort in `_on_peer_failure`. Revoke is non-blocking, and this order means a slow
  abort can never delay waking host collectives.

**Other single points of failure, unbounded waits and races (all fail closed today):**
- **Single points of failure:**
  - Rank 0: ingress, broadcaster and `_request_owners`. Out of envelope, by v7.
  - The authority: fail closed, with no liveness check.
  - The mpirun/PRRTE HNP node: losing its node loses everything.
- **Unbounded but backstopped by the hang detector (300 s):**
  - `Shrink`;
  - `commit_step`'s sync (`:270-273`);
  - the device-wide syncs in `rearm` and `install_mask`.
- **Unbounded with no backstop:** the watcher's `request_abort` sync, which this fix removes.
- **Timing:**
  - The device timeout is `clock64` cycles at an assumed 2 GHz. Under throttling the trap can come after the hang
    detector, which is harmless but makes the ordering inconsistent.
  - ULFM detection of process exit is sub-second relative to 300 s, so there is no tight coupling. The real
    dependency is that the process *exits*, which D-state teardown can prevent.
- **Races checked and OK:**
  - The flag is set before the abort, so a step whose kernels saw the abort always fails the commit gate.
  - Survivors may commit or discard the same step differently. That is safe under attention DP and is handled by
    numbered gathers.
  - Watcher stop and restart are collective over survivors and run on both sides of the protocol.
  - Concurrent revoke was checked in r3.

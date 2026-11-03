
## 2026-10-21 16:40 PT: statistics, fencing, finals sizing (user decisions via the monitor)
- Fence always (user, Decision 2 option A): the deployed authority fences every failed rank before each plan, with or
  without hang detection (v5 fa884734ad). Healthy cost none; recovery adds one PMIx kill call before Propose (measured
  in the C5/v5 point test from the authority stage times). Doc late-write section rewritten: payload rows of a source
  are not fixed (source_write_ranges); safety rests on the fence plus drained writes before resume (not measured).
- Healthy statistics (user, Decision 1 = B): each cell reported as FT - off with a paired 95% CI per node set; bar =
  CI upper bound < 1% or explained. Repeats are guides, not rules: start n=4 ABBA per cell on the mode's node set;
  add repeats / a second node set where a cell is noisy or its CI straddles 1% (no-EPLB 32 and 8/GPU expected);
  fewer where settled. Each adjustment is logged here with its reason. Rounds in one run dir are repeats, not groups.

## 2026-10-21 18:00 PT: d (FT shutdown) fix review (standing rule: senior design + red team)
- Senior design: a02f262be7 (3 s MPI_Finalize bound under the ULFM detector) is a band-aid. By Open MPI v5 design
  (comm_ft_detector.c), ULFM finalize runs a shrink agreement + barrier, then each rank sends a quit heartbeat and waits
  for its watched neighbour's quit or a timeout (ours: thread mode, period 1 s, timeout 5 s, launch/task.sh:55), so it
  should end in <~5 s; the 43.5 vs 14.2 s gap (CLAIMS 152) equals the old 30 s bound firing, i.e. finalize likely never
  completes (a quit lost or the timeout check not running). Stopping the detector at runtime is not possible (READONLY
  MCA var); an app barrier adds nothing (ULFM finalize already has one); skipping Finalize loses the PMIx fence/finalize.
- Red team: same verdict (change): 3 s is below the detector's own 5 s timeout, cuts a degraded-world agreement short,
  is gated on an exact env string, logs a false alarm per rank, and the mpirun exit code / leftovers under the cut are
  unverified. FT-off path untouched.
- Chosen: do not ship a02f262be7 as the fix. Diagnose first: one detector-on clean shutdown with
  OMPI_MCA_mpi_ft_verbose=40 (per-rank "quit message" / "suspected dead" / "DONE WITH FINALIZE" lines). If a quit is lost
  in thread mode, it is an Open MPI bug: report upstream and, only then, a bound derived from the configured detector
  timeout + slack (read via MPI_T, not an env string). PT-D (8138133) still measures the current bound's exit codes.
- (monitor) a02f262be7 leaves v5 unless the diagnosis ends at an Open MPI bug and the bound is re-derived from the
  detector timeout; an Open MPI bug is documented with its repro (never filed: the user owns upstream contact).
- 18:20 (monitor, to the user): a graceful stop exits like FT off's (143 under SIGTERM) whether or not a recovery
  happened; degraded state is reported by /health and logs, not exit codes. The clean shutdown fix must make survivors
  finalize normally and the launcher return the stop's status (PT-D 8138133 ft-loss returned the lost rank's 137).
  Verdict expectation 143 for ft-clean and ft-loss (as job.sh already sets).

## 2026-10-21 18:30 PT: working mode (user via monitor) + transport finding
- Reason first (swarms settle mechanism/design from code); GPU jobs only confirm a predicted outcome with a pre-written
  pass rule. Dev owns execution and job design; the monitor comes in at decision points. Milestone reports <= 5 bullets.
  Jobs: container names (reuse a failed group's nodes), no fragile in-serving diagnostics (NCU under graphs failed,
  ATTR32 G8), 2 ready jobs queued.
- Transport (swarm C, from earlier runs): FT's ob1 is self,tcp only, so node-mates use TCP loopback. ob1 + sm "Shrink
  hang" root cause (HOSTCOLL-8028647): trtllm-llmapi-launch runs the worker as a bash child without exec (line ~312), so
  PRTE/PMIx does not deliver the failure event to the dead rank's node-mates over sm; without the wrapper Shrink over sm
  takes 44.8 ms (HOSTCOLL-8028330); with the threaded detector sm recovers in ~3.1 s (SM-DIAG2-8028320). Microbench
  (HOSTCOLL-7990026, 32 ranks): sm+tcp matches UCX (allgather 335 vs 329 us; tcp-only 520). Candidate: btl self,sm,tcp
  + smsc cma, plus exec'ing the worker in the launcher (a one-word, general launcher fix). Caveat: transport alone vs
  UCX was -0.25% at 128/GPU (CLAIMS 98, n=2), so the transport may explain prefill/stalls but not the decode a2a cost.
- 18:35 swarm B (minimal a2a, verified spot checks): FT a2a = main's kernels + mask/abort by pointer (graph-safe) +
  expert_to_slot; fixed per-source slots + receive_source CTAs + compact routing copy exist only for late-write
  containment and can go (~400-500 lines removed) if survivors wait for runtime-confirmed termination of every failed
  rank before rearm (the C23 guarantee). Expected healthy saving ~1 us/layer (matches the isolated microbench). B's
  "4-5 us hidden-state compaction" does NOT apply to our serving config: CuteDSL NVFP4 reads the receive slots in place
  (fused_moe_cute_dsl.py:2190); only backends without slot support (e.g. quantized Cutlass) compact. So the serving-only
  ~5 us/layer remains unexplained by kernel work; skew/transport (SKEW8) is the lead. Design goes to senior + red-team
  review after the inventory, late-write and stall-path reports.
- 18:40 swarm late-write (agrees with B): epochs protect flags/counts only (already per-source words); in main's packed
  layout payload row addresses depend on runtime_max and payload widths, so a stale write can land on a live source's
  rows: the mask cannot make it safe. Only "wait until the runtime reports the failed process terminated" (PMIx
  PMIX_EVENT_PROC_TERMINATED after kill_rank, bounded by RUNTIME_FENCE_CALL_S, fail-closed if unconfirmed: partition,
  lost prted, MPS) makes the packed layout safe; ~0 healthy cost, + victim teardown time per recovery. It also closes
  C23 for today's design. GPU test: canary late-write (victim loops remote stores; SIGSTOP + SIGKILL variants; negative
  control without the wait must detect corruption).
- 18:45 swarm stall path (code verified): routing after a fetch adds no delay (deterministic router, same step on every
  rank); the ADP gate's waiting counter free-runs (ticks at burst+60k), so each straggler costs one slow step and stays
  out of phase for many waves; FT only needs to seed more first-time stragglers. Most likely seed: FT's rank-0 Python per
  step (gathering_responses loop, responses_gathered, requests_routed, bigger gather pickles) holding the GIL and
  slicing rank 0's request-enqueue thread so one wave's refills span two timeout-0 fetches; second: TCP collective
  jitter. Fix candidates (main's gate untouched): near-zero rank-0 FT work on steps without final responses (owner map
  off the hot path), sm intra-node transport. SKEW8 discriminates (predecessor completion step per straggler, client
  send vs rank-0 fetch time, collective p99 on fetch-split steps). Added to the minimal-FT design as item 6.
- 18:50 red team of the minimal-FT design (key points verified against code paths cited):
  (1) a whole-node or partitioned-node loss can never be confirmed terminated (prted dies with the node; MNNVL fabric may
  stay up), so the packed layout would fail closed on the most common real failure or treat "daemon lost" as terminated
  (unsafe); (2) PROC_TERMINATED does not prove GPU stores stopped (fd held by a forked child, MPS, undocumented drain);
  (3) recovery staging into the a2a workspace relies on fixed-slot write ranges (execution_state.py:122-128);
  (4) exec in trtllm-llmapi-launch drops its exit handling (SIGKILL conversion, cleanup trap, lock fd) and rank 0 cannot
  exec; (5) the exec fix for the sm Shrink hang is a hypothesis; (6) replacing Get_failed with the watch flag widens the
  window for consuming a collective result with a dead peer's part undefined; (7) RUNTIME_FENCE_CALL_S=2 s is tuned;
  (10) the a2a part saves ~1 us/layer, while the serving gap is not kernel work: trades proven containment for an
  unproven runtime guarantee.
- Provisional convergence (final after the senior review): KEEP the fixed slots (proven containment, node loss
  recoverable); drop the a2a removal and the watch-flag trim. Pursue the host side, where the measured cost is
  (CLAIMS 154: ~0.7-0.9 ms/step slower TCP collectives): sm intra-node transport (needs its own recovery proof:
  4 ranks/node shrink, node kill, at 32 and beyond) without changing the launcher's exit handling, and near-zero rank-0
  FT work on steps without final responses. Confirm with SKEW8's skew/stall data first.
- 18:55 senior review (adopt with 6 changes) vs red team (keep slots): reconciled.
  Verified by both: fixed slots never contained payload late writes (source_write_ranges counts the whole payload
  region as writable by any source, nvlink_one_sided.py:784); safety already rests on the fence alone. So the
  termination guarantee (C23) is needed in EITHER design, and node/power loss fails closed in either (prted gone).
  Removing the slots therefore loses no guarantee; it gains ~1 us/layer and ~400-500 lines, at the cost of a kernel +
  recovery rewrite (staging write ranges, clamping released sources, CFT question) before the freeze.
  Dev recommendation (to the monitor/user): (1) now, freeze-critical and measured: C23 termination confirmation
  (PMIx PROC_TERMINATED event handler, before Fenced; fail closed), host transport sm intra-node + launcher exec on the
  worker branch with cleanup/SIGKILL-on-fatal moved into Python (senior's order: launcher+transport first, then the sm
  repro + microbench), rank-0 FT work trimmed on steps without final responses, Get_failed once per step at commit_step
  instead of per collective; (2) the a2a slot removal after SKEW8 shows how much of the 32/GPU cost is kernel vs skew,
  as its own reviewed change (it is mostly a PR-quality win).
- 18:50 late-write swarm (user challenge: a dead process does not stop writes already issued):
  * Writes (verified): FT path uses plain SM stores only (16-B payload stores into fixed per-source slots,
    moeAlltoAllKernels.cu:444-463, 790-795; per-source count words :1056; fence.release.sys + st.relaxed.sys flag
    words); CFT/TMA is excluded under FT; combine pulls (its only remote writes are per-source flag words).
  * Fabric (knowledge, labelled): an issued store lands within ~us; no documented drain bound at context teardown; the
    real exposure is the window until the dead writer's context is torn down (a SIGSTOPped or hung rank's kernels keep
    issuing stores); survivors own (export) the memory, the dead rank only imported it, so after teardown it cannot
    address survivor memory.
  * Confinement (verified, swarm 2, quoting moeAlltoAllOp.cpp:629-677, .cu:790-795): under fixed slots the payload base
    offsets depend only on the fixed capacity and the payload widths, so when every call on a workspace has the same
    payload widths a dead source's late payload write lands only in its own slot, which receivers ignore (valid count
    0); counts and flags are per-source words. With MIXED widths on one workspace (key = hidden_size, dtype, top_k:
    nvlink_one_sided.py:599-612, not the quant mode) a late write can land in live sources' rows. So the earlier
    "slots never confine" statement was wrong in the uniform case and right only for mixed widths.
  * So fixed slots + per-source words give TIMING-INDEPENDENT containment, provided each workspace has one payload
    layout. Minimal hardening: key the workspace by its payload widths (or assert uniform widths at bind). The packed
    layout has no timing-independent containment (row addresses move with runtime_max): this settles the slot
    question: KEEP the fixed slots.
  * CLAIMS 100's late-write cases were INVALID by construction (the fence ended the stopped victim before its delayed
    write; resume came 26 s later). Decisive test designed: a spray kernel on the victim writing sentinels into
    survivors' rows (own and live sources' slots) until SIGKILL / SIGSTOP+fence; survivors scan rows for 50 steps after
    resume; 20 kills; negative control (fence off, victim spraying after resume) must show hits; ~75 min on 8 nodes.
- 19:00 SLOT DECISION (swarm 3 + monitor's own kernel read agree): KEEP the fixed per-source slots. Every remote write
  by a dead rank d lands in a part indexed by d (completion flag column d, recv_counters[..+d], EPLB stats row d,
  payload rows d*slot_tokens+t; combine pulls), and receivers ignore a masked source (count 0, ids -1, weights 0), so
  containment is timing-independent IF every call on a workspace uses the same payload signature (payload base offsets
  are packed per call by widths, moeAlltoAllOp.cpp:652-678). The packed layout has no zero-cost timing-independent fix
  (rows move with runtime_max; only a fresh workspace per recovery). Per-row tags cost ~6% bandwidth: rejected.
  Minimal fix: one payload signature per workspace, either fixed per-payload offsets at allocation
  (calculateWorkspaceLayout) or record-and-check at the first bound dispatch; then narrow source_write_ranges to d's
  real slot rows (more room for restore staging). Healthy cost 0. Rank re-admission (Phase 2) will need an epoch jump
  instead of zeroing flag_val, or a fresh workspace. Pending: which R1 calls (if any) share a workspace with different
  payload layouts (subagent), then code + unit tests; the spray test validates it on GPU.
- 19:05 FOUND (verified: moeAlltoAllOp.cpp:650-677 packs payload offsets per call; rank-17.log of STD-8121996: one
  workspace, 59 NVLinkOneSided communicators, layer 61 (MTP, excluded from FP4 quant) runs Cutlass unquantized):
  in DeepSeek-R1 the MTP layer's dispatch (bf16 hidden 14336 B, no sf) and the 58 NVFP4 layers' dispatch (3584 B +
  448 B sf) share one workspace with different payload layouts, so the NVFP4 layers' sf/slots/scales regions lie inside
  the MTP call's hidden region. A dead rank's late write from one layer type can land in live sources' rows of the
  other: the fixed-slot containment does NOT hold for R1 as deployed. Fix (P0, C32): per-role payload offsets fixed per
  workspace (hidden, sf, slots, scales regions sized for the max case, disjoint), ~40-60 lines C++ + Python layout,
  ~+117 MB/rank; graph-safe (constants of the workspace). Then source_write_ranges narrows to d's real rows.
- 19:20 C32 fix review (senior + red team, both converge on C):
  A (fixed per-role regions) is NOT structural as specified: a row's offset is s*cap*width + i*width, so a width
  change between calls still aliases (NVFP4 source s lands in BF16 source s/4); making it structural needs a max-width
  row pitch -> non-contiguous views -> ~5 reader sites incl. kernels and swizzled-SF tiles (silent corruption risk).
  B (signature in the workspace key) needs no reader changes but costs +7.53 GB/rank (+6.12 GB with a signature-sized
  NVFP4 workspace) = ~107k KV tokens/rank (bf16 basis) with FT on only; also progress_comm must pick a workspace that
  runs every step.
  C (chosen, pending the monitor's OK): one workspace, a separate fixed dispatch sub-region per payload signature
  (combine and control regions shared; combine is not part of the hazard: fence-path combine pulls). Cost ~1.07 GB/rank
  (NVFP4 hidden 939.5 MB + sf 117.4 MB + ids/scales 16.8 MB) = ~15k KV tokens/rank, FT on only; readers unchanged
  (contiguous views); signature bound to its sub-region at bind time, assert it never changes (fires before graph
  capture); sub-regions sized from every layer bound to the workspace. Same review found the EPLB stats row stride is
  the call's eplb_stats_num_experts (moeAlltoAllKernels.cu:659,875,1042): fix it at the layout's capacity or include it
  in the signature. Tests: mixed-signature late-write unit test (kill mid-NVFP4 dispatch, BF16 dispatch, survivor rows
  bit-exact), mixed EPLB widths, assert-fires test incl. a post_quant_comm fallback, source_write_ranges coverage, R1
  NVFP4+MTP GPU kill (accuracy, MTP acceptance), healthy A/B FT on vs off (doc states the KV-pool difference).
- 19:20 P9 / C34 (verified, authority.py:562-565, 470-478; peer_loss_recovery.py:448-454, 929, 989, 1213-1233;
  moe_load_balancer.py:1245-1249): nothing limits a loss to one node; a cumulative loss is recovered iff it is a union of
  admitted domains the survivors' slots cover. For R1 the ghost sizing covers exactly one node (ceil(256/28)=10 slots
  per rank; 24x10=240 < 256), so a second node ends the job: survivors raise before Stopped/Propose and the authority
  aborts at recovery_timeout. Decision: narrow the doc to the enforced rule; do not add a one-domain policy (it would
  refuse recoveries the slots support and break sequential single-rank recovery).
- 19:25 C32 design-C implementation (wt-v5, uncommitted, +446/-136 incl. tests): signature probed per layer at bind
  (quantize_input on one token), bind_layers gives each signature its dispatch sub-region and reallocates the workspace
  when the layout changes (before any dispatch/warmup/capture; same order on every rank); op takes the sub-region
  offset/size (unbound path byte-identical); bound dispatch asserts its signature; EPLB width checked in the bound path.
  Red-team review: ship with fixes. Applied A (check the override size before detaching) and B (probe with ones, not
  zeros: dynamic quantization derives its input scale from the probe). Adding C (2-GPU test of the reallocation path)
  and D (probe == the scheduler's real payloads for NVFP4 and BF16 layers). Alternatives rejected: static reservation
  (~+1.3 GB more), per-layer quant config at construction (factory sees only the global quant config), lazy
  allocation (modest saving, redesign now). Note: two-model speculative decoding on one workspace would bind twice:
  refuse it (follow-up).
- 19:35 SKEW8 (CLAIMS 155, n=1): launch skew is negligible (~40 us spread); the 8/GPU cost is closed-loop refill
  spread that the ULFM transport roughly reproduces with FT off (ulfm ~ ft) -> C22 is a transport effect: the
  sm-intra-node transport fix (C33) is its candidate. The 32/GPU decode cost is FT-only (ft +3.7%, ulfm +0.7%) -> C5 is
  FT's own per-step work (bound a2a path / host hooks), not transport. Next for C5: a 50-step nsys window FT vs off at
  32 (the window length that worked before), read as the per-step difference by kernel and host phase, plus the
  per-kernel graph timeline; repeat the 32/GPU A/B for n>=2.
- 19:45 d (FT shutdown) diagnosis (FINVERB 8139740, mvp/analysis/finverb.md; Open MPI v5.0.x source read on GitHub):
  every rank completes shrink, barrier and the detector's quit exchange (the "own rank" quit message is by design);
  then 16 of 32 ranks never return from the threaded detector's shutdown (likely a lost wakeup in the detector thread's
  loopbreak + join), the other 16 wait for them in the PMIx fence, and all leave by our give-up timer. Same stall in
  8138133 (3 s timer) and in the degraded world. FT off: no stall (17.0 s). The 137 after a loss is mpirun reporting
  the first abnormal exit (the SIGKILLed victim), unrelated to the stall. Open MPI's ULFM docs (v5.0.x) state the
  in-band detector (mpi_ft_detector) is DEPRECATED and experimental: failure detection is done at the PRTE level.
  Fix options (to the fix-review rule next): (1) root cause on our side: stop enabling the deprecated in-band detector
  (rely on PRTE-level detection + our own hang detection for live hangs): needs a check of what it currently buys us
  (it was adopted as "decision 5", CLAIMS 63: no healthy cost; why it was needed for detection latency must be
  re-checked) and a kill/SIGSTOP retest; (2) keep a bounded give-up derived from the detector timeout (stopgap);
  (3) upstream: document the lost-wakeup repro (not filed). Exit code 137 vs 143 after a loss: launcher/harness
  handling, separate item.
- 19:55 detector-off trade-off (subagent, from CLAIMS 57/97/102/103/109/111/117/119/121): the in-band detector adds
  nothing measured for SIGKILL (detection 0.01-0.09 s comes from the runtime + PeerFailureWatch) or process-level node
  kills, and nothing for live hangs (our hang path). It is the ONLY fast path for a frozen (SIGSTOP) process: detect
  4.2-4.8 s, recover 5.5-6.8 s with it (#119/#121, n=3) vs ~35+ s without (heartbeat_timeout_s 5 + hang_confirm_s 30,
  unmeasured at R1); freezes of 5-30 s would no longer be fenced. It also covers node-mates under an sm transport while
  the launcher's bash wrapper hides failure events (SM-DIAG2: ~3.1 s). Host power-off and partitions are unmeasured
  either way. DECISION NEEDED (monitor/user): (A) detector off: clean shutdown (no timer), frozen process ~35 s;
  (B) detector on + a give-up bound derived from its timeout (stopgap for an Open MPI shutdown bug), frozen ~5 s;
  either way document the Open MPI lost-wakeup repro. If (A): rerun the kill matrix, hang set, SHUTDOWN-R1 and the
  integration suite with it off.
- 20:05 option C (detector off + our silent-rank fast path), evaluated (hang_monitor.py:145-210, authority.py:688-690,
  heartbeatSender.cpp:153-176): the monitor already tracks silent (no fresh heartbeat) vs waiting, but confirms both
  after 30 s. C = for a rank that is not fresh and is waited on by every peer, confirm after a short silent threshold
  (>= 5 s, = the detector's semantics; 3 s would fence the freeze3 case): ~15 lines + a config field (or reuse
  heartbeat_timeout_s) + schema/manifest. Required fix with it: the authority loop must drain the heartbeat backlog
  before expire() (today it expires first and reads <= 256 datagrams per pass: an authority stall would make undrained
  ranks look silent -> false fence). Native UDP heartbeats are GIL-free (GC/long steps safe) but have no retry.
  Expected frozen-rank recovery ~1-2 s slower than with the detector (5 s + 1 s fence grace + a heartbeat). Same-node
  dead rank under sm: confirmed in ~5 s, then the runtime kill; whether node-mates then get the failure event depends
  on the launcher wrapper (exec fix). Tests (detector off, 5 s): SIGSTOP rank and node, same-node kill under sm;
  negatives: freeze3, host45, gil45 @0.5 s, GC loop, cgroup CPU throttle, ~3 s UDP drop, authority SIGSTOP 5-10 s idle
  and under load; unit test for drain-before-expire.
- 20:10 USER DECISION d = option A: Open MPI's deprecated in-band detector is off (harness launch/task.sh: off by
  default, WIDEEP_DETECTOR=1 turns it on for comparisons; feature-doc launch command updated in v5); no shutdown timer:
  a02f262be7 reverted on v5 (eb498be62d); option C not coded (documented follow-up: frozen rank ~35 s with hang
  detection -> ~7 s with a silent-rank fast path). Kill tests rerun with the detector off inside the v5 kill matrix:
  rank, node, SIGSTOP with hang detection, clean shutdown; the doc's frozen-rank number becomes the measured value.
- 20:20 C5 MECHANISM (NSYS32 8141738, rank-0 cuda_gpu_kern_sum, 50 steps x 58 layers; monitor's read, verified from
  /tmp/claude-161238/ns32/*.csv): the 32/GPU cost is the CuteDSL routing sort, not the a2a.
  routingIndicesClusterKernel<...,(int)256,...> 5.9 us (off) vs <...,(int)512,...> 16.7-17.1 us (ft): +10.8 us x 58 =
  ~0.63 ms/step vs measured decode +0.49 ms; on the critical path. FT's 320 slots exceed the 256 tier -> 512.
  The a2a is not the cost: ft dispatch 18.3 us replaces off's dispatch + sanitize (29.2 + 24.4 us); GPU kernel totals
  ft 911 vs off 959 ms. Corrects the earlier "moe_sort 384 class < 0.5 us" (subagent, 16:00) and the doc's "routing
  tier ... off the critical path". Swarm: tier options (384/320 tier, local id space) and why lay320 (FT off, 320
  slots) did not show it.
- 20:35 CORRECTION of 20:20 (two independent reads of ns32/{off,ft}.sqlite: mine and a subagent's, per stream): the
  routing kernel's 16.7 us under FT is PDL wait, not work. FT's sort launches ~10.4 us before the dispatch it depends on
  ends (dispatch triggers programmatic completion early, moeAlltoAllKernels.cu:611-612) and its own time after the
  dispatch ends is 5.9 us (off: 4.9-6.5 us after sanitize/dispatch). Dispatch start -> sort end: ft 24.5 us vs off
  35.5 us (FT faster). The 512 tier costs at most ~1 us/layer (~0.06 ms/step): no softmax runs (post-topK sort over
  received slot ids, scores pointer null), so no correctness issue. A Tier<384,8> stays a small optional main-side
  improvement. NSYS32 steady (n=1): decode median off 13.789 vs ft 14.140 ms (+2.5%) but mean 14.396 vs 14.121
  (FT -1.9%) and client tok/s 2944.7 vs 3009.8 (FT +2.2%); the 50-step profiled GPU span was shorter with FT
  (769 vs 835 ms). So at 32/GPU the FT "cost" is a median shift that the mean and throughput do not show in this run;
  next: compare full step-time distributions (median, mean, p10/p90, client tok/s) for FT vs off at 32 across every
  run that has the pair (v2 finals, LAY32, ATTR32, SKEW8, NSYS32, ADPWAIT, PDL-AB).

## 2026-10-21 21:13 Cancel NVL-DBG once it had its answer; finish the v5 gate as its own 1-node job
NVL-DBG's first pass found the failure: a test bug in the new bind_layers test (fixed in ca011e89fc, test-only). Its
second pass (the 16 unrun FT files, about 20-30 min) held the second QOS slot that HOSTBISECT32 (P0, 16 nodes) needs, so it was
cancelled. GATE-V5C reruns test_nvlink_one_sided.py and all 16 files on the same wheel and Python overlay. TRANSPORT-SM
stays held until GATE-V5C passes. The build's failure of the first test_nvlink_one_sided case, run from /checkout,
did not reproduce from /tmp/ut. It stays an open lead; if GATE-V5C reproduces it, it gets a separate read.

## 2026-10-21 21:40 Transport proof first, on v3 + overlay; fold HOSTBISECT32 into the host-fix A/B
The monitor relayed the user's "prove it, fix robustly, fast". C33 (exec'd FT worker + ob1 self,sm,tcp/CMA) has no C++
change, so it runs now on the v3 wheel: short-lived branch v3-c33 (worktree /tmp/claude-161238/wt-v3c33) = v3 +
fa884734ad, 258bed7510, b075b66bb4, ac9ea4e776, 409d2bd1f8, which excludes 2ccc3e8e1c (needs the v5 C++). Overlay
v3c33-0fed8d8141, with its unit files gated in-job before GPU use. It folds into v5 once proven, and the branch retires.
Two 16-node jobs can't run concurrently now (QOSGrpNodeLimit), so ONLINE-ULFM's slot goes to one job.
TRANSPORT-V3 8145956 takes it, and HOSTBISECT32 (pending, never started) was cancelled. Its bisect arms (ft-noH,
ft-noT, phase timers) join the next job beside the host-fix arm, which saves a 2 h slot cycle; the host fixes are ready
by then. New harness arm: std-online-ft-sm.
Deferred input-copy wait (review/hostfix/inputcopy.md, verified at py_executor.py:5994-6003): FT-only, and it fires
only after a step that had context requests. It waits on the same event main syncs later in the iteration. Estimated at
<= ~20 us/step amortized: not the 0.35 ms decode cost and not the online ctx cost. Not coded unless TRANSPORT-V3's
phase timers show its wait is material (lean code: no fix without a measured cost).

## 2026-10-22 01:20 Validate v5 once, on the build that freezes
VAL-V5 8150639 (pending) cancelled: the receive-kernel, watch and gate fixes may still land on v5, so validating f96a8c1a53 now would be redone. The VAL-V5 launch file (rank5/node1/rank0, twokill, stop, SHUTDOWN trio, doc launch command) runs on the freeze candidate. Queue by decision value: PACK1 (LOADCURVE + WATCHAB) takes ONLINE128-SM's slot; RECVAB takes BUILD-RECVFIX's slot once the build's unit tests pass; GATEAB after.

## 2026-10-22 01:39 FT's host transport defaults to TCP; shared memory is an opt-in (monitor-approved)
CLAIMS 158 and 161: shared memory within a node cuts FT's host-collective p50 but shows no serving throughput gain at
8/32/GPU, and at online 128/GPU it may lose ~2% (n=2). The kill benefit came from exec'ing the worker, which holds over
TCP. v5 5ebf7f72a2: the launcher exports pml ob1 and btl self,tcp unless the user sets them, startup still checks ob1,
and the feature doc lists sm as an opt-in that recovers from losses. The design doc's transport bullet now states the
measured cost (host collectives +20-40% p50 vs UCX, no throughput difference beyond noise).

## 2026-10-22 05:49 Attention-DP gate fix: keep the simple rule (option a, monitor-approved)
GATEAB (CLAIMS 166): the quiet-burst release removes the 32/GPU timeout releases and cuts the 8/GPU bistable mode
4-20x, with no 8/32 throughput regression. The residual is a straggler cohort that formed mid-wave, which can persist
at a lower rate; it will be documented, not patched with a second formula (the log-form window was option b). Before
landing: GUARD2 repeats the 128/512 guard at n=4 per arm on the formatted branch head 2d08fc1144 (vgate-2d08fc1144,
in-job ADP unit gate). If the +0.9-1.0 % holds, find out why the rule fires at high load before landing.

## 2026-10-22 06:15 The receive fixes (A, B', Tier<384,8>) stay off v5; the next target is the static-LB / slot-routing layer
ABFIX (CLAIMS 167): the three fixes together recover 0.03-0.07 ms of a ~0.27-0.32 ms gap, against a pre-registered
1/3. No arm isolates B' or the tier, so neither has its own effect, and A alone showed none (CLAIMS 165). Lean code
wins: v5-recvfix stays as a record branch and does not fold in. With FT on but MoE comms unbound, the gap is
+0.15-0.21 ms. Correction 06:35 (code read, moe_scheduler.py:566-571): the unbound arm runs main's route() kernel
(~0.12 per CLAIMS 164) that the bound path does not, so the bound path's own cost is ~0.16-0.22 ms, not 0.04-0.10. The in-place slot read stays (removing it costs +1.4 ms). Next: attribute the bound path's ~0.2 ms to kernels from the existing NSYS32 traces (job-free) before any A/B.

## 2026-10-22 06:36 The bound path's ~0.2 ms gets one more same-node split before freeze (monitor: option 2)
Two cost models mispredicted the bound-path floor (moe_path/plateau 0.10-0.15 vs ABFIX 0.25), so the >1% gap is not
yet understood. One 16-node job, both sets mirrored, trim decode at 32/GPU: off | ft | ft-noghost (bound, 256 slots) |
ft-unbound, plus up to two bound-path pieces that a harness-only hook can switch off. A read-only inventory of the
bound path's per-layer work runs in parallel. Freeze prep (GUARD2 -> gate fix, C43 harness fix, v5 commits) continues.
C43: C43-REPRO 8155494 hung with nvlink_one_sided alone; the build script now unsets srun's PMIx variables for the
extras (C43-FIX 8156357 verifies).

## 2026-10-22 07:10 C43 closed: a harness bug, fixed in tools/build_test_all.sh
C43-FIX 8156357 (same tree and command as C43-REPRO 8155494 plus the env unset for the extras): late_writes 9 passed,
nvlink_one_sided 33 passed / 19 skipped (rc=0 vs rc=124). The build's extras ran under srun's PMIx variables, which
make MPIPoolExecutor's MPI_Comm_spawn hang. No product change; CI is unaffected.

## 2026-10-22 07:20 v5-exp built for one measurement (monitor: build one kernel with all three restructurings)
v5-exp eb3950a3f8 = 5ebf7f72a2 + token-major fixed slots (packed slice for every backend; in-place slot plumbing removed,
net -112 lines) + hoisted mask/round/acquire loads + receive PDL trigger after the receive (review/hostfix/exp_design.md).
Red team: build as is (no blocker/major); minors owed before any landing: a bind-test round with T < capacity and two
live sources, token_major saved at dispatch, 3D comment, memory clobbers on the existing fences. BUILD-EXP 8157331 on
snapshot build-src-eb3950a3f8 (repurposed from the unused build-src-8d0ec8cf1c). ABEXP (launch/abexp_cmd.txt):
off | ft-v5 | ft-exp at 24/32, mirrored; pre-registered: >= 2/3 recovered at 32 in both groups -> bisect the commits.
Amended 07:22 (monitor): 1/3-2/3 recovery also bisects; if exp is perf-neutral or better and passes the full gate +
kills, propose landing it for the design alone (minors fixed); a surviving 32/GPU gap goes to the user with options.

## 2026-10-22 07:55 Gate fix landed on v5 (2e8d6d7da1); main-side branch prmain/08-adp-gate-quiet (c7740d7e83)
GUARD2 (CLAIMS 168) passes its pre-registered rule at 128 and 512 (n=4). v5 gets the tested code byte-identical to
2d08fc1144 as one commit; the main port uses len(responses_list) for the rank count; the residual is in PR_PLAN.md.

## 2026-10-22 08:05 SPLIT lost its unit gate to the known C12 failure; SPLIT2 runs first, ABEXP after the build
SPLIT 8156552's in-job gate failed on test_remote_mpi_session[flashinfer_temporary_cleanup] (FREEZE_CHECKLIST C12,
zmq ipc path > 107 chars; pre-existing on v2/v3, deselected by every other gate). My gate omitted the deselect, so no
case ran. SPLIT2 8158171 adds it and moves to the freeze line v5h (2e8d6d7da1, incl. the gate fix). ABEXP 8157937 was
cancelled while pending: its build gate would reject the same known failure in BUILD-EXP's extras.
require_build_pass.sh now accepts it only with KNOWN_FAIL and when it is the single FAILED line. ABEXP is resubmitted
when BUILD-EXP ends (~08:20) and runs after SPLIT2, so no slot idles.
08:14 (monitor, for the user): ABEXP first. SPLIT2 8158171 cancelled before it started; ABEXP 8158388 holds the
allocation and waits in-job for BUILD-EXP's BUILD_TEST_ALL_DONE (gate with KNOWN_FAIL=C12); SPLIT2 8158390 queued behind.
08:31: BUILD-EXP's extras finished (all rc=0 except the known C12, which the gate accepts), but the build went on to
its 4-GPU integration tests (launch file lacked SKIP_INTEGRATION), so ABEXP 8158388 would have waited in-job up to
40 min for BUILD_TEST_ALL_DONE. Cancelled it (and SPLIT2 8158390, pending) and resubmitted ABEXP 8158617 with the
build gate only (passes now), then SPLIT2 8158618. BUILD-EXP's integration run continues as landing evidence.

## 2026-10-22 08:47 v5-land 9f2f1892f0 assembled: the freeze candidate if ABEXP reads neutral or better
v5 2e8d6d7da1 + token-major slots (619c9442bb) + red-team fixes + fence clobbers (all six fence asm lines) + one
commit removing the unused receive-slot read (Cutlass/CuteDSL/tests, back to main's code, -487 net) + tidy-ups +
strengthened tests (distinct late-write rows; ep4 round trips masking dead rank 1). No load hoists, no late PDL
trigger (greps in wt-land match v5). Landing-quality review: clean after that cleanup. BUILD-LAND launch file and
presubmit ready (launch/build_land_cmd.txt, snapshot build-src-9f2f1892f0 repurposed from the record snapshot
b0ad558bcb, contains eb3950a3f8 for an incremental build); submits when BUILD-EXP frees the build root.

## 2026-10-22 09:17 Rebase onto current main (user via monitor); BUILD-MAIN at 0829689c52 held
Upstream main 9cc37929e8 (47 commits past 0829689c52). The monitor's agent rebases v5-land -> v5-land-rebase (wt-rebase,
backup backup-v5-land-20261022, local only). v5-zero branches off it; the C30 baseline becomes main at 9cc37929e8, so
BUILD-MAIN (0829689c52) is not submitted. ABEXP / BUILD-LAND / SPLIT2 continue unchanged.
09:20 (user via monitor, supersedes 09:17): finalize code first, rebase once before the freeze build (rebuild, gate, short
confirm A/B; freeze hash = rebased build). v5-zero branches off v5-land; BUILD-MAIN back on 0829689c52, submitted when a
slot allows. After freeze: one branch wide-ep-ft-mvp on the fork + one deletion list for approval.

## 2026-10-22 09:42 ABEXP read out (CLAIMS 169): exp recovers ~half at 32; no bisect
The 1/3-2/3 band said bisect; the monitor called no bisect since v5-zero removes the bound path entirely and ZERO-AB keeps
an ft-v5land arm as the fallback check. v5-land's failing late_writes row check is a test bug (rows are in slot-claim
order from atomicAdd(send_counters), not token order; combine is exact), found while re-scoping the test on v5-zero.

## 2026-10-22 09:50 BUILD-ZERO 8160496 submitted (v5-zero 6645ce1848) beside BUILD-LAND's test tail
v5-zero = 9f2f1892f0 + G1a + G1b + G4 + six red-team fix commits (net about -650 lines). BUILD-LAND's build root is no
longer written (its wheel and gtest binary are done; the rest is tests on the installed wheel), so BUILD-ZERO builds now
instead of waiting ~30 min. Generator checks for G4's config change run at the end of the build job. BUILD-LAND's extras:
late_writes rc=1 (test bug, slot-claim order) and test_moe_slot_gather rc=4 (file deleted on v5-land; stale list entry).

## 2026-10-22 10:19 No GPU time on the old design (user OK via monitor): SPLIT2 cancelled (was pending)
The two arms that still decide something move into ZERO-AB on the v5-zero build: lb (the indirection floor) and
ft-zero-noghost (decides the routing-tier fix). ZERO-AB = off | lb | ft-zero | ft-zero-noghost | ft-v5land at 32 and 128,
submitted once BUILD-ZERO's gate passes. The transport A/B goes into KILL-SMOKE's spare time or right after.
10:37: BUILD-MAIN 8161178 and BUILD-ZERO 8160496 share overlays/build-root-v2. Let both run: BUILD-ZERO's build-root
users (wheel, gtest) were done; its remaining extras/integration/generators use only the installed wheel and /checkout
(run_integration.sh and build_test_all.sh have no build-root reference). Cost: the next v5-zero rebuild starts from main's
last_built_commit (more touched files, ccache-backed).

## 2026-10-22 10:44 ZERO-AB 8161823 and KILL-SMOKE 8161824 submitted on v5-zero (wheel 6645ce1848 + overlay 981fd845e2)
BUILD-ZERO's extras failed only test bugs: late_writes (fixed 6a3eba107e) and test_ring_host_copies (G1b changed
_kernel_writes' signature; fixed 4a30682ead). Both gates run the fixed files from tests overlays vzero-981fd845e2-tests +
vzero-4a30682ead-tests, plus test_moe_a2a_recovery (4 GPUs). ZERO-AB: off | lb | ft-zero | ft-zero-noghost | ft-v5land at
32/128, ~106 min against 117. KILL-SMOKE: G0 rank5/node1/rank0, G8 stop + rank5acc (GSM8K >= 0.90 after a kill).
Late-write GPU rows dropped: under fence-always a stopped rank is killed through the runtime before survivors propose, so
its queued writes die with its context (DRH2-R1 assumed the in-band detector); late writes vs the remap are covered by
the 4-GPU unit tests only. TRANSPORT-CTX (launch/transport_ctx_cmd.txt, presubmit tctx-1f654ad6fd) queues after these.
10:50 correction (monitor, code-cited): "no late writes under fence-always" was wrong. kill_rank returns when PRRTE answers
the PMIx job-control request (pmix_fence.py:79-112), not when the process is reaped; the authority proposes right after
(authority.py:504-533); PRRTE's kill is SIGCONT -> SIGTERM -> SIGKILL after a grace period (standard, unverified here). A
stopped rank is woken first and can replay a graph with its old mask while survivors recover: the remap covers that
window. KILL-SMOKE stop-row readout adds: the victim's last iteration and timestamps after SIGCONT (did it replay after the
survivors resumed?) and every survivor's "Remapped ... in X ms" line. Doc wording: late writes after fencing are contained
by the recovery remap; proven by the 4-GPU recovery test; system-level timing observed in KILL-SMOKE's stop row.

## 2026-10-22 11:12 ZERO-AB 8161823's in-job gate FAILED (no GPU case ran); KILL-SMOKE 8161824 cancelled while pending
The follow-up commits' tests ran for the first time: test_peer_loss_recovery (2), test_modeling_deepseekv41 (10, the
[False-*] cases), test_mnnvl_alltoall_workspace (1), test_mnnvl_memory_remap (1), late_writes [no-remap] (1), and
test_moe_a2a_recovery [kill]/[graph-mask] in group 0 only (group 8 passed all 4). The gate did its job. A triage agent
classifies each (test bug / product bug / flaky) and fixes on v5-zero; ZERO-AB and KILL-SMOKE resubmit on the fixed overlay.
11:14 (monitor): no idle slot. TRANSPORT-OFF 8162824 = TRANSPORT-CTX's FT-OFF arms only, on the v5-land wheel (no overlay;
no v5-zero code): std-online at 128 | -ulfm (ob1 tcp eth0) | -ulfm-2nic (eth0,eth1) | -ulfm-uct (if uct_check passes),
mirrored, with the bcast timing hook. New job.sh arms std-online-ulfm-2nic / -ulfm-uct (noft-ulfm + WIDEEP_MPI_MCA,
honoured by launch/task.sh:50). The FT arms follow once v5-zero passes its gate.

## 2026-10-22 11:20 All six ZERO-AB gate failures were test bugs; ZERO-AB 8162956 + KILL-SMOKE 8162957 resubmitted
v5-zero 95d6a64a73 (Python unchanged from 981fd845e2; 7 test commits): disagg stub (read-only property), departed=0 in the
non-speculative ADP gather assert, WeakSet-held client, re-pin in finally (unraisable KeyError), late_writes no-remap control
alternates its write pattern (constant bytes were indistinguishable after the reset), and two survivor races in
test_moe_a2a_recovery ([kill]: peers' next-round counters/flags at offsets 32-43/132/140, barrier after the check;
[graph-mask]: from_blob receive views overwritten by the fastest survivor's next replay, clone before combine; production
safe: experts read rows before combine and a peer's next dispatch needs this rank's combine). KILL-SMOKE's gate runs the
recovery test 5x per group. Overlays vzero-95d6a64a73 / -tests. Queue: TRANSPORT-OFF 8162824, ZERO-AB, KILL-SMOKE.
11:22 (monitor): ZERO-AB 8162956 and KILL-SMOKE 8162957 go ahead of TRANSPORT-OFF; TRANSPORT-OFF 8162824 cancelled while
pending (presubmit toff-9f2f1892f0-943f0a3f75 kept) and resubmitted when one of the two finishes.
11:46 (monitor): no kernel refactor now; C30 (VAL-P) decides shared kernels vs split FT entry points (threshold in PR_PLAN).
The C++ batch (c08deefb55..7a03252e92) rides the rebuild after ZERO-AB/KILL-SMOKE results.

## 2026-10-22 12:09 TRANSPORT-CTX/OFF downgraded (monitor; CLAIMS 170): no 16-node slot for the transport A/B now
Job-free reanalysis of STD-8147471 shows no FT context-step cost above the ~17 % run-to-run spread at online 128 on v3 and
no refill concentration. C21/C22 move into VAL-H: online 32 and 128 context/whole-window, >= 3 same-node ABBA repeats per
arm, refill/backlog/decode classification; FT - off within the off-vs-off spread closes C21 as not reproducible on v5+.
TRANSPORT-OFF stays cancelled (launch file and arms kept). mpi4py 4.x bump (install_mpi4py.sh +101) needs infra sign-off
in the PR split (P0d).

## 2026-10-22 12:13 ZERO-AB 8162956 cancelled after its group-0 gate failed on the late_writes NEGATIVE CONTROL only
Group 0: test_moe_a2a_late_writes[no-remap-None] "the victim's writes missed the survivor's live rows without the remap"
(timing: one snapshot comparison); every other gate file and every remap case passed; group 8 passed everything and had
started its first case. Keeping it would have idled group 0's 8 nodes for ~100 min and left an unmirrored n=1 A/B.
Resubmitted as ZERO-AB 8164194; both zero gates (ZERO-AB, KILL-SMOKE 8162957) deselect the no-remap control
(-k "not no-remap"): it measures the test's detection power, not the product. A polling control is being written on
v5-zero and goes through a 1-node point run first (PREFLIGHT 18).

## 2026-10-22 12:26 BUILD-ZERO2 8164323 submitted: v5-zero f9e6bc0f8b in one build
= the C++ batch (FT-off main identity) + docs a0273e4845 + staging-exclusion removal b4672a55d0 (net -433) + EPLB warm-up
after recovery 1d675a2281 (setWarmUpIterCount honoured mid-run: moeLoadBalancer.cpp:1222-1238) + polling late_writes control
f9e6bc0f8b. The Python can't ride an overlay on 6645ce1848's wheel (C++ changed in between), so the rebuild runs every
changed test in its extras first (PREFLIGHT 18), test_moe_a2a_recovery 3x. A read-only red team checks the deletions
(every staging target remapped first on every path; in-flight copies of a rank lost mid-staging; protocol fields).
Snapshot build-src-f9e6bc0f8b repurposed from build-src-2ccc3e8e1c (no queued job uses it).
12:37: KILL-SMOKE 8162957 gave no kill evidence: G8's gate ran before the no-remap deselect (late_writes control failed);
G0 passed test_moe_a2a_recovery 5x, then rank5-1 hit SMOKE_FAIL (PRRTE daemon on nvl72020-T04 exit 255; infra).
Resubmitted as KILL-SMOKE 8164530 (same 95d6a64a73 code as ZERO-AB 8164194); nvl72020-T04 in mvp/EXCLUDE_NODES.txt and the
launch refuses listed nodes at start (the broker has no --exclude). ZERO-AB 8164194: gates PASS in both groups, cases from 12:33.

## 2026-10-22 12:56 Shared build root shipped stale Python into wheels; fixed, BUILD-MAIN must be redone
BUILD-ZERO2 8164323's extras failed at import (_utils.is_mpi_peer_failure missing): its wheel carried main's _utils.py.
Cause: setuptools' build_py copies a .py into build-root-v2/wheel-staging/build/lib only when newer, and a repurposed
snapshot keeps git's old mtimes (10-21 08:14) for files its checkout did not change, so the previous build's copy
survived. tools/verify_wheel.py (wheel .py vs git blobs) on the existing wheels: build-v5zero-6645ce1848 and
build-v5land-9f2f1892f0 match their commits exactly (0 differ; the ZERO-AB/KILL-SMOKE wheel is clean);
build-main-0829689c52 has 8 files with v5 content (communicator, py_executor, nvlink_one_sided, moe_scheduler, mnnvl_*,
cute_dsl/cpp custom ops) plus FT-only modules -> unusable as the C30 baseline; rebuild it. Fix in build_and_test.sh:
touch the package before build_wheel, then verify the wheel against the commit and fail the build on any difference.
BUILD-ZERO2 cancelled and resubmitted as 8165145; BUILD-MAIN rebuild follows it (same root).

## 2026-10-22 13:55 ZERO-AB read (CLAIMS 171): row B, ft-zero ~ lb; ZERO-AB2 8166627 submitted
Validity checks pass. ft-zero - off +0.112/+0.156 at 32 (lb +0.125/+0.110), v5-land +0.22; noghost vs ft-zero -0.020/-0.076
(sets disagree, so no G2 call). Stall and client within spread/noise. ZERO-AB2 (pre-staged, BUILD-ZERO2 wheel f9e6bc0f8b +
overlay b6f895cbc5, gate waits for BUILD-ZERO2's extras): slot-table proof log in every arm, ft-zero-noslottable, noghost,
mirrored at 32. Queue: KILL-SMOKE 8164530 first, then ZERO-AB2.
14:28: BUILD-ZERO2 8165145 done: wheel build-v5zero-f9e6bc0f8b verified (0 differ); every unit group, extra and integration
test rc=0 except the known C12 (late_writes polling control and test_moe_a2a_recovery 3x pass). UNITS-POINT 8167245 runs
b6f895cbc5's new tests (two-round remap case) 3x on that wheel before any 16-node gate takes them. BUILD-MAIN rebuild next.

## 2026-10-22 14:41 The KILL-SMOKE losses were the Occupied Idle Job Reaper, not the test or the node
sacct: KILL-SMOKE 8162957 and 8164530 are "CANCELLED by 146504" (not us). Both ran an in-job unit gate on node 0 of each
group (~25+ min with test_moe_a2a_recovery 5x and test_mpi_session) while the other 7 nodes idled: 28/32 GPUs idle past
the reaper's 30 min. The "tools vanished / rc=127" and PRRTE exit 255 were the container being torn down by the cancel.
UNITS-POINT 8167245 ran test_moe_a2a_recovery and then late_writes in the same container without loss, consistent with this.
nvl72020-T04 is off the exclude list. ZERO-AB2 8166627 (same long gate, 5 min in) was cancelled and resubmitted as
8167505 with a short gate (the overlay's changed modules; the wheel's full FT suite passed in BUILD-ZERO2 8165145);
KILL-SMOKE resubmitted as 8167545 on the same code as ZERO-AB2 with the same short gate. Rule: an in-job gate on a
multi-node job stays under ~10 min; long suites run in the build job or a 1-node point job.
14:54: UNITS-POINT 8167245 PASS on build-v5zero-f9e6bc0f8b + vzero-b6f895cbc5: test_ring_host_copies 45; test_moe_a2a_recovery
5/5 x3 (incl. the two-round re-plan case c134ce0284); late_writes 5/5 x3 (polling control). The new tests may enter gates.
ZERO-AB2 8167505 and KILL-SMOKE 8167545: short gates PASS in all four groups; cases running in parallel.

## 2026-10-22 15:12 KILL-SMOKE stop-1 FAIL root-caused: a latent classifier bug (pre-v5-zero) + detector-off timing
Rank 16 (other node) was lost because, when the runtime ended SIGSTOPped rank 2 during the ADP tp_allgather, Allgatherv
returned on rank 16 with garbage in rank 2's slot; pickle.loads raised ValueError("unsupported pickle protocol: 6"), which
_is_transport_error (communicator.py, since 2e3136a7dd) did not count as a transport error, so the executor crashed and
MPI_Abort'ed. The 10 s detect->stopped is that crash's hard-kill grace (survivors in shrink waiting for rank 16). 128
failures = 64/GPU on two lost ranks. Detection took 36 s because Open MPI's in-band detector is off by default since
2026-10-21 (user, option A): only the 30 s hang confirm detects a SIGSTOP. Fix 4ea75fe97d: any error raised inside
_deserialize_recvbuf / safe_broadcast is a transport error (goes through the failure-evidence check); a ValueError elsewhere
still raises as the rank's own (existing test kept); new test with a real protocol-6 payload. H23 for stop rows: resume is
to be measured from hang confirmation, confirm time reported separately. Rerun the stop row on the fixed overlay.
15:35: ZERO-AB2 8167505's ft-zero-noslottable arm died at its first dispatch in both groups (harness bug: the NO_SLOT_TABLE /
UNBOUND CALL proof lines read NVLinkOneSided.dispatch_source_slots(), which G1b removed; its TACTIC_CHECK cache_miss=32 is
from the early death, not comparable). Fixed: both proof lines and their expected patterns (job.sh, t_bind) drop
source_slots; t_bind 7/7, t_slot all PASS. No dedicated rerun (table proven used, ft-zero ~ lb); a noslottable arm rides
the next perf job with room.

## 2026-10-22 15:36 ZERO-AB2 read (CLAIMS 173): FT = indirection floor; G2 dropped; KILL-SMOKE kill rows PASS
ft-zero ~ lb in both runs (FT's own increment ~+0.005 ms over the static-LB floor at 32/GPU), noghost = ft-zero (no ghost
cost), slot table proven in use. Per the user's direction (indirection as main is acceptable), decode perf work on the
no-EPLB healthy path stops; G2's 384 tier is not needed. KILL-SMOKE 8167545: rank5 1.71 s/0.949, node1 3.7 s/0.894,
rank5acc 1.71 s + GSM8K 0.949, rank0 PASS (ends 36 s); stop-1 FAIL = CLAIMS 172 bug (fixed 4ea75fe97d), STOP-RERUN 8168466.
BUILD-MAIN 8161178's contaminated wheel is rebuilt now (job submitted) for C30.
15:38 (monitor): order to code final: (1) STOP-RERUN 8168466 must PASS; (2) the user's frozen-rank detection choice (A keep
30 s + stated limit, recommended / B fast path ~7 s / C shorter confirm; no speculative B); (3) in parallel, job-free: VAL-H/K/P
redraft for the freeze candidate (agent) and the PR-split squash on a new branch v5-split (agent; v5-zero stays the
integration branch, final tree must equal v5-zero); (4) then code final -> rebase (rerere) -> freeze build + verify_wheel -> VAL.

## 2026-10-22 15:40 User decision: frozen-rank detection option A (via monitor)
Open MPI's in-band detector stays off; a frozen (SIGSTOPped) rank is detected by the hang confirm window (30 s default), a
crashed or killed rank in ~2 s. The FT doc's known limits state it ("frozen rank ~30 s detection, crashed rank ~2 s"); the
silent-rank fast path is a documented follow-up. Code final now waits only on STOP-RERUN 8168466. The doc line lands on
v5-zero after the v5-split squash agent finishes (it is matching v5-zero's tree), then rides the freeze build.
15:41: STOP-RERUN 8168466: the CLAIMS 172 fix works (only rank 2 lost; resume 3.43 s after the 33.65 s confirm). The stop
row's h23_tput is None because the 30 s confirm + recovery left no post-resume window (harness gap): the row gets a longer
run so >= 20 requests complete after resume, and H23 reports tput N/A instead of FAIL when the after-window is empty.
Monitor counts stop as a kill-path PASS: CODE FINAL (v5-zero) pending only the option-A doc line.
15:42: BUILD-MAIN 8168840 (main 0829) cancelled as obsolete: C30's baseline is main at the rebase target, built right after the rebase.
16:05: Option-A doc line committed: v5-zero fc7bfab236, carried to v5-split 2aaa7f1c4f; trees equal (diff empty). Split verified: 26+1 signed commits, git diff 65a39b411d..v5-split was empty before the doc commit.
16:05: Rebase the 27-commit v5-split, not the 389-commit v5-zero, onto upstream main 9cc37929e8. Branch v5-split-rebase f4a142987a in wt-split; same net change, a reviewable history for the single fork branch. Conflicts: mpi_session.py, its test, and the openai_server.py import. Resolution: upstream's polling _wait_for_pending_futures now also ends the wait when ULFM reports a departed rank (_num_failed_ranks, which returns 0 with FT off). The blocking _live_worker_futures is removed, so under FT the stop client is acked during the wait (the old notes' NEEDS-REVIEW). Tests adapted, untested until the freeze build's extras. A red team on silent semantic merges is running. The freeze hash is set after it reports.
16:05: Both slots run shakedowns on the current candidate (harness validation, not freeze evidence): VALK-SHAKE 8169315 (both gates PASS at ~6 min) and VALH2-SHAKE 8169329.
16:14: Rebase red team: no rebase-induced break. Pre-existing gap fixed: KV cache events' attention-DP gather has no departed= (kv_cache_manager_v2.py:1537), so PeerLossRecovery.create now refuses KV cache events (5330b899f0, test + doc; event_buffer_max_size default 0, benchmarks unaffected). FREEZE HASH 5330b899f0 (v5-split-rebase, 28 commits on main 9cc37929e8), monitor OK. FT-off-not-upstream items PX (gate fix, CLAIMS 168) and P0c stay as intended PRs; C30 readout states them. Follow-up: mgmn os._exit can drop a queued shutdown ack in a degraded world.
16:38: VAL-H2 no longer pins the v2 seed std-seed2.json (C14, monitor): each group tunes std-online + std-online-ft at 128 on the freeze build, then 2 rotated rounds (n=2/arm/load, ~96 min/group). The MPIRUN_RC=1 / CASE rc=1 ending is normal for every real std case (124 cases, 16 jobs, FT-off included). Process slip: one command cd'd into logs/bundle; returned at once.
16:40: VAL-H2: both groups at online 32/GPU (monitor decision; n=4/arm at the C21 cell), tune + 2 rounds per group. tools/ctx_classify.py (CLAIMS 170 port, t_ctx_classify PASS) joins the VAL-H readout.
16:47: VALH2-SHAKE 8169329 cancelled as agreed after round 1 proved the harness: ctx_stall_cost's 3 tables are filled at 32 and 128; ctx_classify at 32 needed --full-pos 5 (waves are 5-8 context steps at 32/GPU vs 15+ at 128; t_ctx_classify still PASS). Shakedown numbers, n=1, not evidence. BUILD-FREEZE 8170816 submitted (5330b899f0, build-root-v2, snapshot build-src-5330b899f0).
17:26: BUILD-FREEZE 8170816 cancelled after 35 min with no output from the import smoke (BUILD-ZERO2's whole build took 26 min). The smoke loads this tree's Python on the v3 wheel's binaries, which a tree on main 9cc does not match (the reason BUILD-MAIN never ran it); no longer a valid pre-check here. Dropped it from build_freeze_cmd.txt; the build's unit tests import every FT module. Resubmitted as BUILD-FREEZE 8172055 (container named for exec). BUILD-MAIN-9CC 8171992 submitted (build-root-feat). VALK-SHAKE 8169315 all rows PASS (monitor read; harness validation only). Slip: the slot idled ~17:15-17:22 because I did not check VALK-SHAKE's end.
17:58: Root cause of the "stuck" builds: overlays/build-root-v2 symlinks cpp-build-oot, venv-3.12, fmha-gen and ccache to overlays/build-root-feat, so BUILD-FREEZE 8172055 (v2) and BUILD-MAIN-9CC 8171992 (feat) configured into ONE CMake tree at once (two pybind11 clones into the same _deps/pybind11-src; network itself fine: ls-remote OK). Both cancelled. Resubmitted apart: BUILD-FREEZE 8173212 (v2/feat tree alone; snapshot now has f9e6 so it rebuilds the 16 changed cpp files, not every file) and BUILD-MAIN-9CC 8173382 in the independent original root $R/build-root (last built fcc86b10b2, fetched into the main snapshot: 112 changed cpp files). Rule: never run two builds whose roots resolve to the same cpp-build-oot.
21:07: BUILD-FREEZE 8173212 cpu group: 10 failures. 8 are test_kv_pool_rebalance drain tests (no-events and events): the MagicMock(spec=PyExecutor) fake gives a truthy _peer_loss_recovery.enabled, leaves publish_responses unbound and gathers without departed=. 2 are test_multimodal_scheduler: a SimpleNamespace fake without _step_changes. All test-only: real FT-off has RecoveryHook().enabled False and _step_changes None, the same order as main. Fixed in the fakes, commit 759787cb17. tensorrt_llm/ and cpp/ are identical to 5330b899f0, so FREEZE = 759787cb17 on the 8173212 wheel (verify_wheel 0 differ; its single stale file is sampler/two_model_spec_dec.py, removed upstream and imported by nothing). Snapshot build-src-5330b899f0 now holds 759787cb17. The two fixed files were added to the VAL in-job gates (they run before any GPU case). VAL-K 8178388 + VAL-H1 8178389 submitted (pending, Resources).
21:07: The main-9cc wheel (8173382) is NOT pristine: 78 stale non-generated .py files from setuptools' build/lib in $R/build-root (an Oct-18 FT build), FT modules included (executor/recovery/*, peer_loss_recovery, execution_state, hang_evidence, fault_tolerance_status). The 631 "not in commit" = these 78 + generated kernel/vendored trees. Both roots' wheel-staging/build were moved aside to build.stale-1022-2106 (deletion queued). BUILD-MAIN must be rerun before VAL-P (the C++ is current, so it should mostly repackage).
22:23: Monitor/user: VAL-H2 and VAL-P dropped as separate jobs (finals H covers online 32 n=8 600 s + ctx_classify -> C21/C22; finals P = main parity on the same groups -> C30). Finals smoke = SMOKE=1 inside each fin_* file (same code paths, shorter) via a thin driver. Order: BUILD-REFREEZE (8c52d11469) + BUILD-MAIN-9CC repackage -> VAL-K/H1 rerun on the refreeze -> FIN-SMOKE -> finals with readout after each job. FINALS_PLAN v4.
22:28: Series restructured: v5-final 5c48020365, 26 commits on 9cc37929e8; tree byte-identical to v5-review-merged 8c52d11469 (git diff empty), 0 fixups, SERIES_SMOKE PASS (26 commits, 0 problems), B3/B6 hand-checked. Executor hook merged into 'Recover the PyTorch executor' (could not stand alone); B4 kernel part stays in the rank-mask commit (templates interleave). REFREEZE = 5c48020365 on the BUILD-REFREEZE wheel (built from 8c52d11469, same tree). Helper branches v5-sq/x1/x3/x4/x5 queued for deletion.
00:02: BUILD-REFREEZE 8179984 (8c52d11469): compiled, verify_wheel 0 differ / 0 stale, gtests incl. hostAccessibleDeviceAllocatorTest PASS, groups pass except 2 new-test bugs + known C12. (1) The garbled-payload test used dist.allgather (world comm); the fake errors on the TP comm, so it exercised the post-check, not the error path; it now uses tp_allgather(small_payload=True). Product unchanged: the error path already raises "from error". (2) The one-slot table test ran without FT mode; it now sets TLLM_FAULT_TOLERANCE_MODE=1 and adds an FT-off case asserting no table (main's kernel). Folded as fixups: v5-final e25e638054, diff vs 8c52d11469 = 2 test files, SERIES_SMOKE PASS. The wheel stands (tests aren't packaged). UNITS-REFREEZE 8181837 reruns them. BUILD-MAIN-9CC 8181612 repackage. Monitor: VAL-K/H1 are not rerun; FIN-SMOKE validates the refreeze. Finals regenerated to FINALS_PLAN v6 (overnight: 300 s windows, n=4 main + folded 8/512, recovery n=2, one pinned table per mode tuned on the refreeze).
09:26: PRE-REG amendments for the finals readout (monitor; fixed before any rehearsal/finals data, logged per PRE-REG header). Canonical tool: tools/finals_stats.py (fixed copy installed; original kept as finals_stats.py.pre-fix-20261023; reproduces trim_readout and CLAIMS 169/171/173 exactly). mvp/harness/finals_readout.py (10-21 median protocol) is a cross-check only. (a) Full mean step = mean of all steady decode steps (differs from "trim + stall" by trim x stall fraction, +0.14..0.25 points per pair). (b) End-to-end metric = client output tok/s/GPU over the bench run (result.json), plus rank-0 generated tok/s/GPU over the row window as a second view; no separate "steady client" metric. (c) Claim/IUT and per-mode summaries use loads 32 and 128 only, with every mode x load cell n >= 2; 8/512 are appendix cells. (d) Parity (main 9cc vs our FT off, same group): descriptive table; if the one-sided 95% bound in the "ours slower" direction is >= +1% in any 32/128 cell, the headline also reports FT on vs main. (e) Xid/NVRM is not recorded in cost cases (harness gap): validity relies on rank-log fault/hang lines; disclosed. (f) --window-s is mandatory (= planned window minus 10%).
10:53: PRE-REG amendment (g) (monitor; before any finals data; evidence = stall decomposition over 7 set-pairs + same-arm repeats, rehearsal STD-8193512 and STD-8164194/8167505/8178389 G0/G1). Stall ms/step is a small-count statistic: 5-40 stall events per window, each a context step spilling into the next decode step, sized only by that step's prefilled requests (identical across off/lb/ft/main); same-arm same-node repeats move it by up to 0.25 ms/step (~1.8% of the step). So the full-mean step cannot discriminate a <1% cost at n=4. Rule change: the per-step "FT cost < 1%" claim is decided on the trim-mean decode step (IUT over every mode x 32/128 cell, unchanged otherwise). Full mean and stall ms/step are reported per cell as estimate +/- 95% CI with the stall-event counts, and labelled as workload-sampling-dominated. No-EPLB is reported with its decomposition (off -> LB floor -> FT on); prior n=4 v5-zero: lb-off trim +0.110 ms, ft-lb trim +0.005 ms, stall diffs ~0 within +/-0.1. Previous rule (claim requires full mean too) is superseded. The user is told in the next report and may restore it.
10:54: Amendment (g) WITHDRAWN (monitor, after the dev's process objection, which is correct): (g) was written at ~10:55 PT, AFTER the rehearsal's n=1 reading of no-EPLB/32 full mean +2.52% / stall +103% against FT, so adopting it unilaterally would read as goalpost moving, and the claim criterion is the user's decision. PRE-REG 3 stands as written (per-step claim needs trim AND full mean). The spill analysis (identical spill sizes across off/lb/ft/main; same-arm swing up to 0.25 ms/step; v5-zero n=4 lb-off trim +0.110 / ft-lb +0.005 ms, stall diffs ~0 +/-0.1) is the pre-stated explanation that the finals test, reported with stall-event counts. (g) is put to the user as a question with this timing disclosed; any change needs the user's explicit OK before finals readout. tools/finals_stats.py reverted to the PRE-REG 3 rule.
12:11: USER DECISION (10-23 ~12:15 PT): report the per-step '<1%' verdict under BOTH rules side by side, each labelled: rule A = pre-registered (trim AND full mean, IUT over every mode x 32/128 cell); rule B = trim mean only (proposed after the rehearsal's n=1 full-mean reading; that timing is disclosed wherever B appears). Full mean and stall are always reported per cell with CI and stall-event counts. tools/finals_stats.py prints both verdicts.
13:15: fin6_j1 (healthy-only) starts before RC ends, in the idle slot (monitor, on the dev's proposal). Its data depends on the code only, not on RC's outcome: RC pass = no code change, j1 valid; RC failure or any product commit = refreeze, j1 DISCARDED under the freeze rule. Healthy-relevant gonogo items already PASS (A build identity incl. REFREEZE4 integration 14/14; B every arm serves; C1 fairness; D1 readout; G1 schedule). Recovery/finals j4-j8 still wait for the full gonogo.
13:55: USER DECISIONS on review/CLEANUP_PLAN.md: D1 P items proven by AST equality; S items by unit tests + a 1-node GPU kill smoke for recovery/communicator changes. D2 split of peer_loss_recovery.py later (move-only follow-up); FailureDomain layering move now. D3 FT config stays env vars (as main: TLLM_FAULT_TOLERANCE_MODE, TRTLLM_ENABLE_PDL, ...). D4 restack in dependency order: the API change 34611ef619 carries the minimal update of main's existing caller (device mask), so each commit/PR builds and passes CI alone. D5 all wrong comments fixed incl. C++: one post-finals rebuild validated by the D1 bar. Cleanup on branch v5-pr; finals measure 1117f7e0e7.
13:55: RC online rank0-1 FAIL = harness timeout (CASE_TIMEOUT_S=600 at job.sh rank0/deadline < online ready ~550-670 s; no server_ready, 0 iters, rc 124; kill never attempted). Fix: timeout 1500 s (harness only). Online rank0/deadline share the code path of the no-EPLB rows that PASSED on 1117f7e0e7; their first repeat in finals j6 doubles as validation; a failure there reruns only those rows.
14:10: ONE last refreeze (refreeze5), scoped to VERIFIED real errors only (user: 'if it's really errors... a good feature'; dev: quality items -> follow-ups). Candidates: the online node-loss CuteDSL compile after resume (root cause pending) + 7 red-team items under code verification (HMAC key in env; mpi_failed_ranks swallow; hcoll not refused; SIGALRM finalize alarm; mask read before grid-dependency sync; ghost releasePending error path; quiesce relying on caller join). Process: fix on v5-final with failing-before tests -> monitor review -> one build incl. C++ -> validation of every arm and every kill row in every mode on that build -> finals. Finals j2+ held; j1 (refreeze4) becomes rehearsal data.
14:18: USER DECISION: refreeze5 validation = TARGETED (build tests + integration, then one ~1.5 h job: FT-on healthy arm per mode + every kill-row type incl. online node1/rank0/deadline with the warm burst + all-rank compile check + SHUTDOWN), since refreeze5 = 3 startup/teardown Python/launcher fixes (HMAC key scrub, hcoll refusal + launcher default, SIGALRM hardening) and the full rehearsal validated the unchanged paths. Verified NOT bugs: mpi_failed_ranks swallow, mask read before grid sync, ghost releasePending, quiesce precondition.
14:39: fin6_j2 (healthy-only) runs alongside VAL5 on refreeze5 (same rule as j1: valid if VAL5 passes with no code change; discarded on any product fix). Recovery finals j4-j8 wait for VAL5 PASS. Shortest realistic --time per job for backfill; VAL5/j2/j3 pre-staged (dry run + presubmit) as soon as the wheel exists.
14:52: USER DECISION: keep frozen-rank detection (hang monitor/evidence, heartbeats incl. C++ sender, PMIx fence; ~1.3k product + ~1k test lines) in the frozen build and the design/presentation; order the PR series core-first (dead/lost-rank recovery on the fence path) with frozen-rank detection as a separate later PR in the same series for its own owner design review. Its healthy cost is inside the FT-on finals numbers (hang detection on in every FT arm).
16:54: GO for the finals on build-freeze-58caa0fde6 (REFREEZE = v5-final 58caa0fde6): VAL5 8202863 = 12/12 kill rows PASS (no-EPLB rank5/node1/twokill/stop/rank5pf/rank0/deadline; online node1/twokill/rank5upd/rank0/deadline) with 0 compiles after resume on every survivor, + SHUTDOWN ft-clean PASS; registration on all ranks; FT healthy arms in 3 modes served. Gonogo A-G PASS (F from refreeze2 RB; refreeze5 changes only startup/teardown). Recovery finals j4-j8 may start; j2 (running) and j3 count.
17:55: USER DECISION: online low-load FT cost (~1.1% at 8/32, ~0 at 128) gets a decomposition A/B right after the finals: same-node arms std-online / -ft / -ft-empty (EXTRA_SLOTS=empty: storage + copy path) / -ft-frozen (updates 0: EPLB update path) at 8/32/512 + short nsys per arm. If the cause is a small fix (copy path / allocation flags): fix + refreeze + rerun the online healthy cells before the PRs; if a bigger redesign: pitch the current design with the measured cost, cause and proposed improvement. DIAG-ONLINE-LAY (testhook) withdrawn: it would fail on the first online weight update.
19:17: USER: the online low-load ~1.1% is acceptable for now; after the A/B analysis it goes to next steps (no refreeze unless the A/B shows a trivial, clearly safe fix, which would still be put to the user). Report it in the doc/PRs with its measured cause.
19:18: USER (clarification): after the A/B, the monitor decides fix vs next-steps by judgment: fix now if the cause is clear, the fix is small and verifiable, and a rerun of the online cells fits; otherwise next steps. Report the decision and evidence.
21:05: PRE-REG 5 window threshold corrected (monitor; a check parameter, not a claim rule): the finals windows are COUNT-based (COST_LOAD_ITERATIONS 32=28, 128=12 waves), not 300 s; offline 128 waves take ~22.3 s, so its complete 12-wave window measures ~268 s. --window-s 270 (300 - 10%) wrongly flagged J6 G8's complete offline-128 pair (268/269 s). Threshold set to 240 s = the shortest planned window (~268 s) minus 10%; it still catches truncated windows. Disclosed here; applies to every finals readout.
21:07: --window-s 240 WITHDRAWN (dev objection correct; same process error as amendment g): the change came after the 270 rule flagged J6 G8's offline pair and after J6 costs were read. Rule of record stays 270 (amendment f). J6 G8 offline pair = INVALID per PRE-REG 5; rerun once per PRE-REG 6 (one 8-node offline STD pair at 32/128, same seed/pins). Both readouts go to the user side by side: 270 + rerun (rule of record) vs 240 keeping the pair (proposed after J6 data; windows are count-based and the pair completed all 12 waves).
22:22: USER DECISION: healthy-window validity = all planned waves completed (completed == concurrency x iterations), replacing the 270 s threshold. Adopted after J6 (J6 G8 offline pair: complete 12 waves in 268/269 s); disclosed. Every cell keeps n=4 pairs; J6R not needed (cancel). tools/finals_stats.py implements it (--window-s now optional, default off).
08:48: PLAN CHANGE (monitor relaying the user's question "are we testing the decisive cause?"): DECOMP-B 8239875 (online updates frozen) cancelled 5 min after its start (still in startup, no data read; the stalls also show in offline and no-EPLB, even when idle, so online updates can't be the main cause). Replaced by KNOB-AB 8240302: offline std-offline / -ft / -ft-nohang (hang detection off) / -ft-poll1 (peer_failure_poll_s 1.0), one variable per arm, cold first-150 + short 32/128 windows. DECOMP-NSYS 8240037 (pending) cancelled so KNOB-AB takes the freed slot first; NSYS is resubmitted after KNOB-AB starts. DECOMP-A 8239874 keeps running.
09:43: PLAN CHANGE (monitor OK after its numa_maps capture: each FT rank maps 364 GB of shm, ~58 GB same-node page migration after the post-ready page-cache drop, rank-0 stalls follow): GHOST-EMPTY 8241110 cancelled while PENDING (ghost vs empty moot: both prefault) and replaced by POPULATE-AB (offline FT vs FT without the peer prefault, diagnostic overlay diag-nopopulate-0eb6e82ac0, ABBA x2 per group + one online rank5 kill on the overlay per group for the restore-time cost). The user approved fixing the cold burst if the fix is small and validated.
11:38: GO for the fix (no peer prefault), decided ~11:35 after POPULATE-AB 8241601's non-fresh same-node cases (monitor read, dev-confirmed: excess stalled s over 90 s after the drop ON 21/31/46 vs OFF 6/10/11, ~3x less, not removed; fresh-node cases clean in both arms). User had agreed to fix if clear. Fix = v5-final d71f169e4b (new commit on 6688790d56; frozen history untouched). Rerun on overlay fix-nopopulate-d71f169e4b: wave 1 FIX-REC (validation first; rerun data discarded if any row fails) + FIX-W1A, wave 2 FIX-W1B + FIX-W2A; n=2 rank5/node1, n=1 twokill/stop/rank5pf/rank5upd (n=2 PASS on the frozen build); 8/512 dropped. SLOTS-AB 8242414 (pending) cancelled to queue behind the fix waves.
13:00: USER DECISION (monitor relay): today's rerun (waves 1-2) stays on the tested fix d71f169e4b; 087d1e7f19 (drop own host-copy mappings) does not enter today's numbers, stays on branch fix-dropmap, v5-final stays at d71f169e4b; 087d gets its own 8-node A/B (offline FT d71 vs FT 087d vs FT off, non-fresh cases, + rank5/node1 x1) after the waves or when it backfills, then becomes a measured follow-up commit.
13:07: USER OK (monitor relay): result tables come from the fix-1 build (waves 1+2 on d71f169e4b). Fix 2 (087d1e7f19, drop own host-copy mappings) enters the PR as a second commit only if DROPMAP-AB is clean: cold burst ~ FT off (same nodes), steady 32/128 = d71 within spread, rank5 restore unchanged; otherwise it stays a follow-up.
13:08: USER DECISION (monitor relay, supersedes the earlier fix-2 gate): fix 2 (087d1e7f19, drop own host-copy mappings) must be in the PRs. DROPMAP-AB is its landing validation, not a go/no-go: a problem (restore regression, steady change, burst not gone) is fixed and re-validated, not dropped. Doc: one or two lines under startup/known costs (+ optional figure); main tables stay on the fix-1 build (d71f169e4b).
19:03: Fix 2 landed (DROPMAP-AB 8253006 validation passed, CLAIMS 187; monitor reviewed the commit): v5-final = 27442bd91d (fast-forward from d71f169e4b, no rewrite); v5-pr tip = 19b0e90251 (fix 1 1a683f2a3b + fix 2, each its own commit). Healthy result tables stay on the fix-1 build (13:07 decision).
21:25: USER DEADLINE (monitor relay): final code + data by morning. (A) LOCKED deliverable: v5-final 27442bd91d, v5-pr (both fixes), fix-build tables CLAIMS 185-187, review/DESIGN_DOC_v5final_tonight.md; bursts disclosed as a measured kernel-compaction / host-copy page-lock interaction (seen once FT-off), next step. (B) fix attempt with hard cutoffs: mechanism + small safe fix by 23:30; same-node A/B (FT vs FT+fix vs off, cost sequence, 2 rounds) 00:00-01:30; if clean, healthy-cell rerun 01:30-05:30, numbers swapped by 07:00. Any failure/ambiguity at a cutoff -> stop, A stands.
21:44: Pin NO-GO tonight (COMPACT-PROBE: pinned 5.7 s vs mapped 29.4 s / lazy 13.1 s of >10 ms copies under forced compaction; partial). Final data = rerun on v5-final 27442bd91d (V5F-REC 8269102 first, V5F-W1A 8269103, then W1B/W2A); MIDBURST-DIAG3/4 cancelled for slots.
23:35: Option B premise INCONCLUSIVE (HOSTMEM-PROBE Part B, 8270321): under forced compaction, >10 ms copy time: shm-mapped control 2.8 s vs hostmem 26.4 / 18.3 / 7.1 s (pinned/gpu/anon dst); readers still in migration_entry_wait with a hostmem source; but modes ran sequentially with drifting fragmenter intensity (compact_isolated 3.9 M control vs 7.7-13.0 M hostmem) and Python readers touch their own movable heap pages. Not committing B. Next: PROBE v3 (C readers over pre-faulted buffers, sources interleaved in ~5 s ABAB blocks under one fragmenter phase, D samples tagged by block) for the next 1-node slot. Final data stays the fix-2 rerun.
23:35: PROBE v3 decision rule (pre-registered by the monitor before data): hostmem better iff, across the frag phase, its per-block >10 ms copy time is lower than both shm-mapped and shm-lazy in >=80% of block-triplets AND hostmem reader D samples ~0; anything else = inconclusive, option B stays out tonight. Slot: after PROBES-COMBO, then V5F-W2A.
23:56: PROBE3 verdict per the pre-registered rule: INCONCLUSIVE (forced fragmentation did not stall shm readers in C either; the only stall block was kcompactd migrating freshly filled shm while it was read, S1 block 0). Option B stays out tonight; next-step evidence.
00:01: PROBE4 pre-registered rule (monitor): hostmem better iff >=80% of triplets with migration >0.1 M pages show hostmem slow-copy time lower than both shm modes, and hostmem D ~0; else inconclusive. Probe's summary: PROBE4 decisive_triplets=X/Y (Y: S1 and S2 read windows both >100k migrated).
00:25: PROBE4 by the pre-registered rule: clause 1 met (4/4 eligible triplets), clause 2 (hostmem reader D ~0) not met (22 migration_entry_wait samples vs S1 110 / S2 65) -> INCONCLUSIVE, B stays out tonight. Evidence: hostmem cut slow-copy time ~10x under targeted compaction but did not eliminate migration waits. Validation complete on v5-final (rank5upd 1.19 s PASS).
00:29: B-AB (full-scale online test of option B) designed AFTER PROBE4's data (disclosed); not tonight's final numbers; morning input for the user's adoption decision.
01:31: Option B NOT adopted tonight. Reason: at full scale (B-AB 8272666) cuMemCreate HOST_NUMA ran out of memory during model loading (NUMA node MemFree 55 GB, checkpoint page cache 231 GB; driver allocations do not reclaim page cache). Fix designed (monitor): reserve each rank's host-copy segments at load-balancer setup, before checkpoint loading, and fill later; clean error if the reservation fails; no shm fallback. Design note + diff on ft-hostmem + a 1-node warm-page-cache repro for review in the morning. B-AB keeps running for its v5-final vs FT-off windows.

# Operating prompt: ONE current plan (CONSOLIDATION, rewritten 2026-10-18 evening). The layered history,
# verbatim, is in OPERATING_PROMPT_ARCHIVE.md; nothing there overrides this file.
# After EVERY compaction or restart re-read: OPERATING_PROMPT.md, TODO.md, RETRO.md, review/FINDINGS_2026-10-18.md,
# mvp/analysis/V7_EXTRACT.md.
# Update the progress line at the top of review/HANDOFF.md (findings closed / total; phase items done) at every
# wake-up.

## v7 INVARIANTS (design doc; details and PDF path in mvp/analysis/V7_EXTRACT.md; check every design against them)
1. Fixed slot count, tensor shapes, addresses, token buckets and selected tactics.
2. Same graphs: replays read stable device descriptors; never a mask captured by value.
3. Zero capture/compile/tuning during or after recovery (or while serving).
4. Suspicion never mutates membership: one coordinator proposes; READY from every admitted survivor.
5. Quiescence is proven, not assumed (abort + terminal survivor work + no failed-issuer writes to reused storage).
6. One logical commit; never roll back a committed generation.
7. Requests: no duplicate streamed tokens, no uncertain KV; retry only from a validated boundary or error.
8. Coverage proven per layer/expert per admitted failure domain before routing is published.
9. Healthy cost measured against the identical configuration.

## GOAL
Make WideEP fault tolerance shippable on GB200 for wide-EP MoE decode, on DeepSeek-R1 (the only model this
phase). Aims: coverage of real failures (rank, node, hang), a healthy-path cost close to zero, recovery in
milliseconds, and lean main-style code owners would merge. GLM-5.3 resumes only after the owner review.

## HARD RULES (fixed; changing any needs the user, asked in HANDOFF.md)
- R1-only scope. No PRs ever (the user opens them); push to the fork (`fork`) at milestones, never origin.
- No changes to main's design. No rm (mv to TRASH; deletions go to PENDING_APPROVAL.md). Upstream issues are
  documented, never filed. Out of scope: rank-0/node-0 loss (fail-closed), DeepSeek-V4.1-Flash, GB300-only models,
  disaggregated serving, Phase 2 implementation, transport work.
- Code bar for every new piece of code (re-admission, PMIx fencing, borrowed rebuild, circuit breaker, health
  API, ...): design note + persona review before coding; no new env flags (the FaultToleranceConfig object
  only); a real test (GPU or integration, not mock-only) per behavior; error paths that clean up; tests never
  weakened. Code that misses the bar by the gate stays on a side branch, out of the mergeable series.
- Audit gate: when consolidation is done, note it in HANDOFF.md and wait for the user's independent audit before
  the FINAL COMPARISON (keep working on non-final items meanwhile).
- Every number in a doc or message to the user cites its mvp/CLAIMS.md row (run id, scale, n, metric).
- `git commit -s`, no co-author/AI lines; no git stash; reuse main's machinery and justify each new component
  in DECISIONS.md.

## SETTLED DESIGN DECISIONS (user)
- Config: one validated FaultToleranceConfig read once at startup; MODE/AUTHORITY/KEY stay env; no LlmArgs change
  yet. Defaults:
  - extra slots: online EPLB on -> replicas; online EPLB off -> BORROWED (no device memory while healthy;
    ~250 ms more to resume); EMPTY is opt-in. Borrowed becomes the code default only after it meets the code bar
    (VMM rebuild or reason recorded, clean error paths, no leak on re-plan, R1 DEP32 GPU kill test).
  - failure_domain: NODE (4 GPUs; R1 needs 320 slots, checked at startup: s*(N-K) >= E); `rank` is opt-in.
    Nodes come from hostnames.
  - hang_confirm_s: the shortest value with zero false fences across the negative controls (40-60 s stalls,
    long prefill, GIL-starved heartbeat), preferring device-wait evidence; record value and evidence in CLAIMS.
- FT's own healthy cost: <=1% at every load (FT on vs FT off, same layout, mean-based, n>=3). A miss after the
  trims is reported with its root cause in CLAIMS/HANDOFF; it does not block the gate.
- Layout cost (FT off, 288/320 slots vs eplb0): skip EPLB's slot-route kernel when every expert has one slot,
  and attribute the 128/GPU cost (combine) before fixing it.
- Re-admission (IR1): requests of a lost rank that delivered nothing rerun on survivors only after a node loss
  (NR1: after a single-rank loss the request may be the cause and a second loss is not recoverable). Phase B
  (streaming continuations) per mvp/analysis/READMISSION.md.
- Fencing: self-fence via the worker's agent on by default; a frozen process is killed with
  PMIx_Job_control(PMIX_JOB_CTRL_KILL) on that one proc, never PMIX_JOB_CTRL_SIGNAL; the configurable fence
  command stays; no peer /proc kill; every SIGKILL site documented.
- Detection: Open MPI's threaded in-band detector (MCA settings in the docs), adopted provisionally (CLAIMS #63);
  hang detection plus fencing stay.
- N-K: process-level N-K at R1 DEP32 (several ranks, a whole node with the node alive, re-plan on a second
  failure). True node death aborts under mpirun/PRRTE: one time-boxed (~1 day) attempt at a survivable launch
  path, else documented as v7's launcher-survival gate.
- Host copies / DR-M1: sharded copies stay parked on wideep-ft-pending as an owner question, with restore from
  the checkpoint on disk as the alternative (CLAIMS #65 is an estimate).
- Builds: a build may use a GPU slot, one ~1 h build per milestone, scheduled when a slot would otherwise idle.

## SOURCE OF TRUTH AND ORDER
review/FINDINGS_2026-10-18.md: CQ1-20, EV1-13, DR-*, IR1-6, NR1-12. Every Status gets FIXED (commit), TESTED
(run/test), DOCUMENTED (where), REJECTED (reason in DECISIONS.md) or DEFERRED (only with the user's approval,
asked in HANDOFF.md). Verify each finding in the code first. The phase isn't done while a Status is blank.
Phase items (A done):
A. Honesty pass (CLAIMS.md). Done; maintained continuously.
B. Measurement standard: mean (cost claims) and p99 next to the median; cut-off streams counted; confounds
   isolated; n>=3 with CIs within a job; profiles only under full load; real text for EPLB/slot claims; one
   tactic draw per arm per round (pinned autotuner cache for both arms when attributing).
C. Perf diagnosis: the 128/GPU layout cost (EV5) and the first-load slow-step tail (EV6), each time-boxed to 2 more
   jobs after the page-cache control.
D. CQ1-20. E. DR-*. F. IR1-6 (IR6 = Phase 2, recorded). Plus NR1-12 by severity.
G. Evidence at R1 DEP32: overlap two-step kill, hangs, node loss, re-plan, KV-reuse kill, re-admission;
   checked-in integration cases; kills during a restore copy, an EPLB update, Install-to-Commit and warmup;
   authority death; a multi-hour random fault-injection soak and a multi-hour healthy soak (leaks, memory,
   KV, threads, fds). Validation runs use one integration-branch build per milestone; overlays only for probes.
Review packet (by Monday): DESIGN.md + review/OWNER_REVIEW_SUMMARY.md: the ULFM/Open MPI dependency, risks and
alternatives; every kill site; new components vs reuse of main; the PR cut with sizes and owners.
Then (after the audit gate): the FINAL COMPARISON on R1 (spec below), docs, push to the fork, stop.

FINAL COMPARISON (user, 2026-10-17): baseline = main FT off on the standard aggregated deployment (online EPLB
288 with layer_updates_per_iter 1, MTP with forced acceptance, overlap scheduler, attention-DP balancing); ours =
the same with FT on. (1) Healthy curves per BENCHMARK_STANDARD.md (1k/1k, 8k/1k). (2) FT-on failure timelines
(per-second throughput/latency): one GPU killed idle and under load, one hang (fenced), one whole non-rank-0
node: time to resume, restore time, failed requests, p99 spike, throughput dip, steady N-K throughput vs
31/32 and 28/32. (3) FT-off counterpart: same kills, time to serve again (cold and warm restart), lost requests,
capacity lost while down.

## PRIORITY AND PACE (user, 2026-10-18 evening)
- After C-PAGECACHE the top GPU job is a credible FT-cost number: one job isolating the confounds (FT off on
  main's wheel; FT off on the stack wheel; FT off under ULFM/TCP; FT on), 1k/1k, n>=3 per arm, mean-based, same
  nodes, randomized order, smoke stage first. CLAIMS reports wheel, transport and FT as separate deltas.
- Scope stays as planned (everything discussed gets finished), done faster: independent pieces in parallel
  (subagents in separate worktrees: re-admission, PMIx fencing hardening, borrowed rebuild fixes, health/config
  fixes, circuit breaker), each driven to done (note, review, code, real test), not many half-open; verify each
  result myself. Both GPU slots stay on the decision-ranked queue.
- A finding is FIXED only with a passing test run cited (unit, plus GPU where the behavior is runtime);
  otherwise "FIX COMMITTED, UNTESTED".

Steering (user, 2026-10-18 night): bookkeeping only when a result or merge lands; no plan/handoff rewrites or
recounts; check-ins <=5 bullets; low-severity findings swept in one batch per subagent; the user messages at most
twice a day, batched: change only the lines a message changes.

Steering (user, 2026-10-18 night, batch 2): pack 2 independent DEP32 experiments per job (8 nodes each, own
segment; cap 16 nodes, never 64; if a 16-node job pends >20 min, cancel and run the parts as 8-node jobs; log the wait in RETRO); wheel builds only for C++ changes
(Python fixes ship as overlays, unit tests inside an existing GPU job); keep the next job presubmitted; several
small fixes per job, smoke first. Fold the fixups now on a local-disk clone ($TMPDIR), incl. CQ3/CQ7 splits, then
one build + test pass; afterwards fold each fixup as it lands. Don't reinvent the wheel: check main first,
justify any new component in DECISIONS.md. Borrowed slots: fix CQ4 now; test whether empty slots need the
scratch mapping at all (else leave them unmapped and cut the code); VMM-manager extension vs thin class is an
owner question with a written plan. CQ6, CQ19, CQ16: OWNER QUESTION + PLAN; CQ18/DR-T1 integration tests now.
Small findings in one subagent sweep. Target: the audit gate on Tue Oct 20 or earlier.

## OVERNIGHT PLAN (user-approved, relayed 2026-10-18 night; design-shaping first, polish last; code bar applies)
Decisions:
1. Rename `borrowed` -> `ghost` everywhere (code, docs, CLAIMS: "320 = 256 active + 64 ghost"); with online EPLB allow
   EPLB slots + ghost.
2. Ghost memory moves onto CudaVirtualMemoryManager (tagged ghost pieces, materialized at recovery); keep only the
   minimal CQ4 leak fix until then. Note first; any small manager addition is an owner question.
3. Don't reinvent: audit every FT component against main (table in DECISIONS.md); start with the coordinator's
   messaging -> ZeroMqQueue (executor/ipc.py, HMAC).
4. Restore source: online EPLB -> its CPU copies; without it each node keeps the next node's experts (CPU -> that
   node's GPU -> NVLink -> target). No budget knob. SHARDED_HOST_COPIES.md: 320 slots, not 512. NVLink is the
   portable default (target allocates one fabric-shareable block at recovery); RDMA via NIXL optional. Restore times
   measured, never promised.
5. Hangs: fence only when every peer waits on one rank AND it made zero progress for T (any progress restarts T);
   all ranks stalled -> report the deployment dead (orchestrator restarts). Heartbeat from the C++ thread. T = 30 s.
6. Repeated failures in scope: lift the one-recovery guard, test a second recovery; then single-rank losses re-run
   requests once, with quarantine.
7. Spec-dec agnostic recovery: discard draft tokens and roll back draft KV through the same KV manager; untested
   methods warn, not refuse. Kill-test MTP + overlap at R1 DEP32 first.
8. Healthy-cost gate: ~0% within noise, per model, mean and p99.
9. 8k OOM: happens with FT off too: dropped.
10. Config -> LLM args later: short prep note only.
11. FT on = the FT-off deployment + ghost slots; don't extend main for configs main lacks.
12. Docs: 64 ghost slots = 1 node, up to 6 GPUs (not 2 nodes); startup fails closed on coverage, copies fit, IMEX up;
    coordinator loss = serving continues unprotected.
13. Owner packet adds: CudaVMM addition, excluding ghost slots from EPLB, launcher change, config form, mixed
    EPLB+ghost, ring copies; prior-art check (EEP, SGLang, FluxMoE, vLLM, NVIDIA-internal) before claiming novelty.
14. Jobs set container_name (read-only checks by the monitoring session); sysinfo adds ibstat.
15. Checklist: v7 Phase 1 items (1c.6, 1b.4-1b.7, 1d.x).
GPU queue (pack <=16 nodes): 1 FT-COST-A analysis (FT-COST-B only if FT's own cost is above noise); 2 C-COMBINE2 with
4 arms (eplb0, static288+ghost, 288 fixed replicas without rebalancing, online288), all 32 GPUs profiled, n>=2,
bursts attributed too; then fix and re-measure; 3 CQ4 gtest debug only if still relevant after the CudaVMM note;
4 MTP + overlap kill tests at R1 DEP32; 5 hang tests once the hang code lands + FT-COST with hang detection on.
Job-free (by path impact): CudaVMM note + move; repeated-failures note + code; hang code; ghost rename; spec-dec
agnostic check; reuse audit (ZeroMqQueue); mixed EPLB+ghost note; sharded-copies note update; CPU unit tests; owner
packet. Morning: report in HANDOFF.md (results with CLAIMS rows, decisions applied, blocked, fresh-session handoff).

NETWORK (user, 2026-10-18 ~21:40 PT, replaces the earlier no-prompt rule): WebFetch/WebSearch, git, gh, curl,
github.com/arxiv.org and `git push fork` are allowed. Still: no rm, no push to origin, no PRs. If a prompt appears,
queue that one action in PENDING_APPROVAL.md and keep working. Never WebFetch (org settings force a prompt): research via
WebSearch or `curl -sL` to github.com / raw.githubusercontent.com / arxiv.org, saved under $TMPDIR.

## MORNING STEERING (user via monitor, 2026-10-19 ~08:40 PT)
1. Don't reinvent: use main's tools and change main where needed if an FT-off A/B shows no slowdown and tests pass -
   sort gate in main's sort, MTP in-place reads, one hang/progress reader on main's watchdog, sharded copies via
   HostMoeTensorSharer, health/metrics via main's plumbing, final numbers via trtllm-bench/benchmark_serving, tests in
   main's integration framework; update the DECISIONS reuse table for each.
2. FT healthy cost ~0% within the same graphs: rank-0 bookkeeping only on change, per-block count writes, check the
   per-step system fence. ONE ladder job: FT off; FT on; + each fix in turn; 8/32/128, pinned, >=6 runs/arm, mean+p99, bursts.
3. Ghost layout cost: ghost 320 vs eplb0 pinned, >=6 runs, bursts; must also reach ~0%.
4. Final comparisons also with auto-tuned arms. 5. Patched PRRTE on GPU nodes, kill a node's daemon (DR-H1, 1d.1).
6. Clean-shutdown test (exit 0, no leftovers) after a clean run and after a recovery. 7. Keep TODO/HANDOFF/FINDINGS current.
8. Ghost resume ~0.8-1.6 s acceptable; faster mapping low priority. 9. Both slots + parallel subagents; prefer
   experiments that settle a design question.

## CURRENT ORDER (2026-10-19 evening; supersedes the 09:30 order, which is done: folds, one build, ghost-cost attribution, features)
1 Final wheel final-design3 (all features + fixes A/B + MTP in-place) -> its tests (integration first) -> final cost packs
  (both deployments: online EPLB on 288+32 ghost, off 256+64 ghost; 8/32/128; n=6; FT-off A/A + main's build) -> kill packs
  (4 cases x 2 deployments, n=3) + SHUTDOWN-R1. Gate: FT on within noise of FT off at every load on both deployments.
2 If a cost gate misses: profile at once and fix at the source before kill runs.
3 Then merge into wideep-ft-dev, push to the fork, branch deletion list. Design doc numbers go to the monitoring session
  (it owns mvp/DESIGN_DOC.md and mvp/figures/).
Decided today (DECISIONS 2026-10-19): ring host copies without a knob when online EPLB is off; ghost-first even-spread restore;
phase B parked; both heartbeats kept; hang_confirm_s 30 s; spawn-based tests only under mpirun.

## HOW TO WORK
- Measure before explaining; like with like (same job and nodes, alternating/randomized order, n>=3); a control
  and a precondition check in every experiment (INVALID, not FAIL); claim only what the data shows.
- Efficiency (slot-hours are the scarcest resource; two concurrent GPU jobs):
  1. Gate every job with one line in TODO: settle it without GPUs if possible (code, existing data, unit test,
     local stub); else 1 node / 4 GPUs if enough; DEP32 only for scale (healthy-cost A/Bs, DEP32 correctness,
     final comparison). Bundle small jobs.
  2. Order by decision value: what can change the plan or a user decision first; confirmatory runs last.
  3. Factorial jobs (several arms/suspects, randomized order, a control in the same job). Time-box a diagnosis
     to 2 jobs; if unresolved, record it as open in CLAIMS/HANDOFF and move on.
  4. Fail fast: multi-node jobs start with a <5 min smoke stage (server up + one request per arm).
  5. Never idle the CPU side: close job-free findings continuously, with subagents in separate worktrees for
     independent fixes (verify each myself); the findings count moves at every wake-up.
  6. Reports <=5 lines per wake-up (detail in HANDOFF/CLAIMS); batch git operations.
- Presubmit (tools/presubmit.py) passes before every job; its token gates the job.
- Plan flexibility: re-order, drop or replace items when data shows they are wrong or less valuable; record it in
  DECISIONS.md, update this file and TODO.md, say so in the next report. The HARD RULES above stay fixed.
- Git: one concern per commit, fixes folded in as fixups; run tests on a snapshot; never edit scripts a running
  job executes (replace atomically).
- Autonomy: no questions to the user except DEFERRED approvals and the items listed in HANDOFF.md (they never
  block); judgment calls go in HANDOFF.md / DECISIONS.md.

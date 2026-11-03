# Execution plan to CODE FREEZE and final benchmarks (2026-10-21, 15:40 PT)

Gate: review/FREEZE_CHECKLIST.md (every row DONE or user-dropped). Bar: no big or random gaps; small explained
gaps on a trend are fine. Phase 1 (finalize + test) | FREEZE | Phase 2 (benchmark the frozen build only).

## Critical path (GPU-bound, in order)
1. **LAY32** (16 nodes): off@256 | off@320 empty ghosts | FT@320 at 32/GPU. Decides C5/C9.
2. **C5 fix point test** (8 nodes): the prepared fix (below) vs off@256 at 32 and 128/GPU, plus 1 rank kill + 1 node kill
   (the fix touches the slots recovery restores into). If C++: build first (job-free prep makes it one build).
3. **v5 build + full suite** (gtest, units, FT files, mpirun a2a, integration). One build carrying C4/C5/C8 + b/c.
4. **VALIDATION** on v5 (16 nodes, one job): every outlier row + H4-H11 + C7 SIGSTOP + C8 update-kill + C10 deadline
   case + knob at 8/32/128/512 + hang detection on in the healthy FT arm + GSM8K on v5.
5. **FREEZE** (hash in HANDOFF) → **Phase 2 finals**: 3 healthy jobs (one per mode, same nodes, 3 repeats) + recovery
   jobs (A: failure types × timing; B: post-loss across loads; warm restarts), ~13 h wall on 2 slots.

Off the critical path, packed alongside when a half-slot is free: FINSTACK + PT-D (C4), NCU32 only if LAY32 does
not explain C5.

## Slot schedule
- Now: ADPWAIT + RESTART-MODES (running).
- Next free slot: LAY32. The other: FINSTACK + PT-D (8+8).
- Then: C5 point test (+ C8 hook smoke packed in) | v5 build.
- Then: v5 suite | (idle → NCU32 only if needed).
- Then: VALIDATION (both slots if split by mode).
- Every job: readout script and pass rule written and presubmitted BEFORE the slot frees (no idle slot > 5 min).

## CPU work now (not blocked by GPUs), in parallel
Dev + subagents:
- **C5 fix, speculative**: design + code the "empty ghost slots cost ~0" fix now (keep shapes/graphs fixed: e.g. the
  kernels skip slots marked empty via the existing descriptor, or ghosts excluded from the per-step loops until a
  restore activates them). Unit test + compile check ready the moment LAY32 confirms.
- C8 EPLB-update kill hook; H4/H6/H7/H8/H9/H11 harness changes; dry-run each on CPU.
- VALIDATION launch file + automatic verdict readout (one PASS/FAIL per checklist row).
- Phase-2 frozen launch files + readout + figure inputs, presubmitted.
- v5 restack plan (squash b fix, d, C5 fix) and per-PR build-only job ready.
Monitor (me):
- Verify every result from run files; keep FREEZE_CHECKLIST current.
- Figure scripts R1-R6 on v2 data, so phase 2 regenerates in one command; SLIDES.md; doc skeleton.

## Red team (things that could bite)
1. The C5 fix touches the fixed-shape/graph invariant and the slots recovery restores into → kills must be retested
   on it (in step 2), not just perf.
2. Hang detection ON in the finals means the healthy FT arm must also run with it on (same config for healthy and
   recovery). Its healthy cost must be measured in validation.
3. The batching knob changes the config for both arms: it must be checked at 128/512 too (no throughput loss), and
   documented as a deviation from the R1 reference config.
4. print_iter_log off for the headline means the step-cost appendix needs its own runs (or local-disk logging if
   the A/B shows no cost).
5. Accuracy (GSM8K) must be on v5, not v2.
6. 4 h job limit: every phase-2 job must fit with margin; a 16-node job can pend longer; pre-plan the packing.
7. Online EPLB's restart includes rebuilding 390 GB host copies: explained as main's online-EPLB cost.
8. Doc, slides and PRs must carry only phase-2 numbers; v2 numbers retired.
9. If validation fails a row: back to phase 1 for that row only, then re-validate that row before the freeze.

## Dev red team (16:05 PT, two read-only subagents, claims verified; details as rows C14-C15, H14-H21 in FREEZE_CHECKLIST)
- FINSTACK and PT-D cannot share an allocation: both write runs/SHUTDOWN-R1-<job> (same out dir, same autotuner file).
  They run as separate jobs, in order; PT-D can ride in the C5 point-test or VALIDATION allocation.
- One build, not two: the C5 point-test build is the v5 candidate (carries d, C8, C10 and the C5 fix); run the v5 suite in
  the same allocation as the point test.
- The C5 point test needs COST_DIAG_OVERLAY=1 (STD) and KF_ALLOW_OVERLAY=1 (kills) if it runs as an overlay; dry-run first.
- LAY32 is n=2 per arm (mirrored); the C5 decision gets its n>=3 from the C5 point test at 32 and 128 (C15).
- Unit gates need GPU nodes (CPU-partition pytest fails on cuInit, logs/cpu-pytest-8078075.log).
- NCU32: count raised to ~200 launches (one step), WIDEEP_LAUNCHER_GRACE_S now overridable for the report write.

## Monitor red team (16:25 PT): assumptions to resolve or state plainly
1. FT on = FT code + ULFM Open MPI launcher + authority; FT off = main on its default launcher. Healthy cost therefore
   includes the transport/launcher delta. Fair as "deployable cost", but each gap's explanation must say which part
   (FT code vs launcher). If unclear for no-EPLB 32: one arm FT off on the ULFM launcher.
2. Victims are always rank 5 / node 1. Recovery plan time depends on which experts are lost. Use 2 victims across
   repeats (e.g. rank 5 and rank 22; node 1 and node 6), same victims in every mode; rank 0 loss stays out of scope (state it).
3. "Throughput kept" after a loss includes the bigger per-GPU batch on the survivors (fixed client concurrency).
   Report it as such; optionally also per-GPU throughput.
4. Exact-token continuation of surviving requests: the evidence must be a token-equality check on a real kill
   (not only GSM8K). Confirm it exists on v5, or add it to validation.
5. Restart baseline excludes scheduler wait, image pull and model download (same allocation, warm cache), so it
   favours FT off. Say so on the slide.
6. Retryable errors are failures for clients that don't retry. The request-fate figure separates "retryable error"
   from "failed".
7. Only 1k/1k at DEP32 on one model: state as scope; no claims beyond it.
8. Hang detection on: its healthy cost (watch thread, heartbeats) is measured in validation (H19).

## Standing red-team rotation (user, 2026-10-21 17:00 PT)
Every few hours, 2-3 read-only subagents from perspectives not yet used, each <= 10 ranked findings with evidence and
an action; findings folded into FREEZE_CHECKLIST / EXEC_PLAN / SLIDES by priority, top 5 to the monitor.
Used so far: plan method + schedule (15:50); design claims, harness, statistics, TRT-LLM usage (16:20); mentor+audience,
business+ops, code owner+scientist, low-level GB200 failure semantics, "what will people question" -> mvp/QNA_PREP.md (17:00).
Next rotation candidates: process retro (where today's time went), a competitor/prior-art reviewer, a new team member
trying to reproduce one claim from the README alone, a security/robustness reviewer (hostile or malformed protocol input).

## Standing rule: fix design review (user, 2026-10-21 17:50 PT)
When attribution names a cause and a fix is chosen: (a) a senior-engineer subagent proposes the most general, robust,
clean solution in main's machinery; (b) a red-team subagent attacks it (band-aid for our config/benchmark? other loads,
models, EP sizes, transports? what breaks? what is simpler?). Root-cause fixes over knobs or timeouts tuned to our setup;
test hooks / env knobs only for diagnosis, never the shipped fix. Both conclusions + the chosen fix go in DECISIONS and to
the monitor before coding. Applies to A (a2a), B (prefill/transport), C (stalls), d (shutdown Finalize bound: running now).
- Standing rule (user, 18:25 PT): an elusive root cause gets a parallel swarm of subagents (first principles, code
  reading, TRT-LLM design principles, senior engineering, red team) instead of serial GPU probing; the ~3 cap is lifted
  for swarms; claims verified before acting; converge on one design. Running now for minimal FT: inventory, minimal a2a,
  transport, stall path, late-write drain guarantee.

## Working rules (user via monitor, 2026-10-22 08:20 PT; also mvp/PREFLIGHT.md 12-17)
- Measure first: one multi-hypothesis experiment, then bisect; no fix built on one predictive model.
- One shared gate (tools/known_failures.txt + every expected file ran and passed) for every build and gate job.
- One canonical readout (tools/trim_readout.py) for step-time and client metrics.
- A packed job always ready; no idle slot or held allocation without a reason.
- Milestone-only, decision-bearing messages; routine logs in HANDOFF.

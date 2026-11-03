- (2026-10-21) Overlay on a final-type case (KILL-FINAL*, cost packs, SHUTDOWN): the case preflight refuses MVP_OVERLAY unless its knob is set (KF_ALLOW_OVERLAY=1 / COST_DIAG_OVERLAY=1); grep the case's preflight before submitting (PT-BC 8129542 died at preflight).
- (2026-10-21) Point test: never run the GPU step if the unit step did not actually run and pass (a harness failure is not a pass). PT-BC2's kill ran on a unit step that had failed on a path bug, and the unit test that would have caught the b bug never ran.

## Hard rules (2026-10-21, user: learn from today's mistakes)
1. A unit step that did not run is a FAIL; never run GPU kills on top of it (PT-BC, PT-BC3: test_common/conftest path bugs, then a gate that waited without checking pass).
2. Never submit without a CPU dry run of the exact launch command (PT-BC died in 1 s on every node after a 30 min queue).
3. Unit tests run on every patch before any GPU job (the `return` above `install_mask()` would have failed locally).
4. n=1 is a lead; no plan or fix is built on it (CAP1 stride, the ULFM premise).
5. A microbench must match serving's exact config (capacity, PDL, routing) before its results are trusted.
6. Before profiling, check CLAIMS for an existing explanation (ghost-layout cost #75/#107 sat unused since Oct 18).
7. Check the harness against the claim it measures (cold restart from the page-cache drop; print_iter_log vs the recipe; logs on Lustre).
8. Same nodes for any A/B; never attribute group spread to FT.
9. Build: import smoke + library-file checks before a build job (v3 failed twice on missing libs, then on a test needing mpirun).
10. Slots: the next job is presubmitted before a slot frees; check each job ~2 min and ~10 min after start; cancel and fix at once on a startup failure.
11. Every number states its metric (recovery_s vs r0_interrupt_ms vs resume_s) and its job.
12. (2026-10-22, user: work more efficiently) Perf: measure first. One experiment that separates several hypotheses,
    then bisect. No fix is built on a single predictive model (two models mispredicted the bound-path floor).
13. Every build and gate job uses the shared gate: tools/known_failures.txt is the only known-failure list (point_units
    deselects it, require_build_pass accepts only it), and every expected file must run and pass. Tonight's lost hours
    were gate and harness slips (overlay paths, the C12 deselect, the C43 env).
14. One canonical metric readout: tools/trim_readout.py (dist32 case_stats: trim mean, stall ms/step, client tok/s, gap
    vs off). Cite it so dev and monitor never disagree on a metric.
15. A packed job is always ready; no idle slot and no held allocation without a stated reason.
16. Messages to the monitor are milestones that carry a decision or result; routine logs go to HANDOFF only.
17. Code-read swarms run from the dev session (its cwd is /tmp); the monitor's Lustre-cwd subagents stall.
18. (2026-10-22 retro) A new or changed test file runs first on a cheap 1-node job (UNITS-POINT: point_units.sh on 4
    GPUs, ~20 min), or in the rebuild job's own extras, before it enters a 16-node in-job gate. Two 16-node cycles
    (ZERO-AB 8161823, KILL-SMOKE 8161824) were lost to never-run tests failing in the gate.

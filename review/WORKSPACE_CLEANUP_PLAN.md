# Workspace and fork cleanup plan (read-only inventory, 2026-10-23 ~19:30 PT)

Apply only AFTER the finals (J3-J8) and the approved online decomposition jobs (DECOMP-A/B/NSYS) end. Nothing was
moved, edited or deleted to write this. Policy (README, memory never-run-rm): superseded files are moved with `mv` to
`TRASH/2026-10-24/<same subpath>` or `archive/2026-10-24/<same subpath>` (log every move as in
`tools/archive_moves-2026-10-21.log`). Real deletions are only queued for the user. Moving inside the same Lustre
filesystem frees no space. Only the user's queue-delete batch frees space.

Actions:
- **keep**: leave in place.
- **archive**: move to `archive/2026-10-24/`. These are history worth keeping, but not current.
- **TRASH**: move to `TRASH/2026-10-24/`. These are superseded or scratch, and become deletable after one quiet day.
- **queue-delete**: the user deletes them. Large items are listed with their size.

Sizes come from `du`: from `du -sk` per run directory (736 runs, 232.8 GB in total), and for the build roots from the
sum of their subdirectories. "CLAIMS" means the item is cited in mvp/CLAIMS.md (job id or path).

**Never touched:**
- Anything the finals or the decomposition use:
  - `mvp/harness/*` (live parts)
  - `launch/fin6_*`
  - `launch/gen_finals_v6.py`
  - `tools/*` (live parts)
  - `build-freeze-58caa0fde6`
  - `build-main-9cc37929e8`
  - `overlays/build-root-v2`
  - `overlays/build-src-8c52d11469` (the snapshot at 58caa0fde6)
  - `mvp/EXCLUDE_NODES.txt`
  - `bundle.sh` and `run_job.sh`
  - `ompi-ft-ucx/`
  - `cache/`
- The finals inputs:
  - `mvp/runs/STD-8193512` and `mvp/runs/STD-G1-8193512` (online tactic pins)
  - `mvp/runs/STD-8194123` (no-EPLB pin)
  - `mvp/runs/STD-8109649/std-offline-placement.yaml`
  - every `mvp/runs/*-82*`
- Every run whose job id is cited by CLAIMS: 473 dirs, 138.0 GB.

## 0. Bottom line

| Tier | What | Reclaimable |
|---|---|---|
| A: safe queue-delete (superseded, cited by nothing current) | 8 uncited wheels (6.4 G); 6 stale source snapshots + old feature build root (31.5 G); `wt/` worktrees (1.6 G) | **~39.5 GB** |
| B: user decision (history, not cited by CLAIMS) | main build root + its snapshot (20.8 G); runs cited by no doc (20.4 G); `runs/_diagnostics,_failed,_superseded` (11.1 G); old git bundles once a fresh one exists (net ~8.4 G) | **~60.7 GB** |
| C: user decision (old archive/ trees) | `archive/codex` 73.5 G; `archive/exploration-2026-10-15` 36.4 G; `archive/2026-10-21` (build-root-main 15.4 G + overlays >= 20.5 G); `archive/builds` (30.6 G); `archive/2026-10-13` 2.4 G; `archive/ep32-runs` (du timed out after 25 min, large) | **>= 179 GB** |
| Home (`~`, ~10 GB quota) | 9.5 GB used. Safe to clear: `~/.cache/pip` 0.94 G, `~/.nv/ComputeCache` 0.84 G, 3 old `~/.cursor-server/bin/linux-arm64/*` 1.5 G | **~3.3 GB** |

Docs and scripts are small, about 10 MB in total. Their cleanup is for clarity, not space.

## 1. Top level of $R

| Path | Size | Modified | Action | Reason |
|---|---|---|---|---|
| README.md | 3 K | 10-21 | keep, **edit** | Stale. It still says "branch of record v3, next build v5" and "source of truth: EXEC_PLAN/FREEZE_CHECKLIST", and lists `build-v3-bcea8da496` as the current wheel. Rewrite the table for the frozen build 58caa0fde6, v5-final/v5-pr, FINALS_PLAN, finals_stats.py and the doc set in section 6. |
| bundle.sh, build_and_test.sh, install_wheel.sh, overlay_python.sh, restore_build.sh | 15 K | 10-16..10-22 | keep | job entry and build path |
| .bashrc, .zshrc, .gitconfig, .mcp.json, ... (0-byte, read-only) | 0 | 10-12/10-19 | keep | sandbox mount stubs (also in review/, tools/, logs/); they would be recreated |
| TRASH/2026-10-22/build-v5r2-b0ad558bcb | 8 K | 10-22 | queue-delete | empty `dist/` |
| archive/ | >= 260 G | | see section 5 | |
| cache/ | 58 M | 10-21 | keep | mpi4py, wheelhouse, tokenizers |
| dev/ | 0.8 G | | keep | the worktree with `fork`; push path for wide-ep-ft-mvp |
| wt/reuse, wt/sortgate | 0.8 G + 0.8 G | 10-19 | queue-delete (`git -C $R/dev worktree remove`) | README kept them "until the code freeze"; the freeze is done. Branches fix-reuse/fix-sort-gate are clean (0 dirty files) and long superseded. |
| logs/ | 46 M | | keep | bundle logs are cited by CLAIMS |
| ompi-ft-ucx/ | 132 M | | keep | `WIDEEP_MPI_PREFIX` |
| probe/ | 0.4 M | | keep | cited in HANDOFF |
| launch/, review/, tools/, mvp/, overlays/, build-*/ | | | sections 2-5 | |

## 2. Builds (`build-*/`, about 0.8 G each, wheel + logs)

| Path | Size | Action | Reason |
|---|---|---|---|
| build-freeze-58caa0fde6 | 0.8 G | **keep** | the frozen build (finals, CLAIMS 174) |
| build-main-9cc37929e8 | 0.8 G | **keep** | main parity arm in the finals |
| build-v2-cef4670789, build-v3-bcea8da496, build-v5-2ccc3e8e1c, build-v5r-b643ad8a29, build-v5r-b0ad558bcb, build-v5x-eb3950a3f8, build-v5land-9f2f1892f0, build-v5zero-6645ce1848, build-v5zero-f9e6bc0f8b, build-pr0-fc52f0adee | 10 x 0.8 G | keep | CLAIMS rows and the CLAIMS wheel audit name them. Optional later: each wheel is reproducible from its commit (verify_wheel showed 0 differ), and the build-final-* wheels cited by older rows are already gone, so the user could queue-delete the `dist/*.whl` of each and keep the logs (~8 G). |
| build-freeze-8c52d11469 | 0.8 G | TRASH after the finals readout | the tactic tables were tuned on it (tune_freeze_cmd); not run by the finals; not cited by CLAIMS (CLAIMS 246 names the review hash only) |
| build-freeze-1117f7e0e7 | 0.8 G | TRASH after the finals readout | rehearsal RC + J1 (J1 discarded); superseded by refreeze5 |
| build-freeze-7e8dfb37df | 0.8 G | TRASH after the finals readout | rehearsal RA/RB; superseded |
| build-freeze-0e5a22ef51 | 0.8 G | TRASH | refreeze3, never benchmarked, superseded |
| build-freeze-5330b899f0 | 0.8 G | TRASH | first freeze (759787cb17), VAL-K/H point units only; superseded |
| build-main-0829689c52 | 0.8 G | TRASH | NOT clean (CLAIMS wheel audit: 8 files with v5 content); never used by a CLAIMS row |
| build-c43-b0ad558bcb, build-c43fix-b0ad558bcb | 2 x 0.8 G | TRASH | C43 build-repro diagnostics; CLAIMS cites the hash only through build-v5r-b0ad558bcb |

Tier A builds: 8 x 0.8 = **6.4 G**, as TRASH and then queue-delete.

## 3. overlays/ (build roots and snapshots dominate)

| Path | Size | Action | Reason |
|---|---|---|---|
| build-root-v2 | 17.7 G | **keep** | the incremental root of the frozen build (last_built_commit 58caa0fde6); any refreeze or C++ cleanup rebuilds here |
| build-src-8c52d11469 | 2.6 G | **keep** | its source snapshot (HEAD 58caa0fde6) |
| empty | 12 K | keep | used by launch files, CLAIMS 27x |
| build-root-feat | ~16 G | queue-delete | old feature root (last built 8d0ec8cf1c, 10-20/21 era); referenced only by 2 old launch files |
| build-src-5330b899f0 | 3.2 G | queue-delete | first-freeze snapshot (759787cb17), superseded |
| build-src-9f2f1892f0 | 2.6 G | queue-delete | v5-land snapshot, superseded |
| build-src-aae00e6c65 | 3.1 G | queue-delete | wideep-ft-dev tip; referenced by nothing |
| build-src-cef4670789 | 3.2 G | queue-delete | v2 snapshot (a2a microbench launch, 10-21) |
| src-final8-4de25459ce | 3.3 G | queue-delete | v2-era source copy; referenced by nothing |
| tests-8ec46695cc | 69 M | TRASH | one old launch (test_wheel_066) |
| build-src-9cc37929e8 | 2.6 G | **user (Tier B)** | main snapshot; goes with `build-root` below |
| CLAIMS-cited Python overlays: py-final8-4de25459ce, v3c33-0fed8d8141, v4bc-bdb01feb4f, vgate-2d08fc1144, vgate-733ef6d445, vzero-65a39b411d, vzero-95d6a64a73, vzero-b6f895cbc5 | ~5 M | keep | CLAIMS names them |
| the other ~50 small Python overlays (v4*, v5*, vzero-*, vexp-*, all-*, merge-*, mtp-*, stack-*, final-*, *-tests) | ~20 M | archive | only older launch files reference them |

Tier A overlays: about **31.5 G**.

`build-root/` (top level, 18.2 G; last_built_commit 9cc37929e8; used by build_main9cc). **User decision (Tier B):** queue-delete it together with `overlays/build-src-9cc37929e8` (20.8 G in total) if no further main wheel build is planned. The main wheel `build-main-9cc37929e8` already exists. A later main rebuild would then be a cold build.

## 4. mvp/

| Path | Size | Action | Reason |
|---|---|---|---|
| CLAIMS.md, DECISIONS.md, PREFLIGHT.md, RETRO.md, OPERATING_PROMPT.md, EXCLUDE_NODES.txt | | keep | current. HANDOFF line 2 points to `mvp/analysis/DECISIONS.md`: fix it to `mvp/DECISIONS.md` |
| DESIGN_DOC_v5final_draft.md | 27 K | keep | the current design doc draft (code of record 7e8dfb37df; update to 58caa0fde6) |
| DESIGN_DOC.md | 40 K | keep until the v5final draft replaces it, then archive | many CLAIMS rows cite "DESIGN_DOC Evaluation" (v2-era numbers) |
| MORNING_SUMMARY.md (10-21), QNA_PREP.md (10-21 midpoint), SLIDES.md (10-21 midpoint draft) | 30 K | archive | midpoint material, superseded by the finals plan |
| analysis/DECISIONS.md | 90 K | archive (as `DECISIONS_2026-10-17_19.md`) | the pre-10-21 decisions log; mvp/DECISIONS.md starts on 10-21; distinct content, not a duplicate |
| analysis/{CPU_PYTEST_2026-10-19, FIXB_HANG, FT_COST_PROFILE_2026-10-19, FT_ONE_SWITCH, GHOST_PLACEMENT_RULE, INPLACE_SLOTS_BACKENDS, MTP_INPLACE, RANK0_MASK_TEST, V7_EXTRACT}.md | 37 K | archive | 10-18/19 notes, not cited by CLAIMS |
| analysis/ctx_stall_cost.py.pre-stall15-20261023 | 5 K | TRASH | backup |
| analysis/* (cited: ctx_stall_cost.*, dist32.*, skew8.*, stall_timing.md, online_ulfm.md, transport_v3_skew.md, readout_final.*, kills_final_8115497.json; scripts) | 2 M | keep | CLAIMS / figures; `ctx_stall_cost.py` is imported by tools/finals_stats.py |
| figures/, archive/ (2026-10-19 design-doc drafts) | 14 M, 1 M | keep | mvp/archive is already an archive |
| runs/ | 232.8 G | see section 4.1 | |

### 4.1 mvp/runs (736 dirs)

| Class | Dirs | Size | Action |
|---|---|---|---|
| Finals inputs + J1-J8/VAL5/rehearsal (82*) | 21 | 20.9 G | keep |
| Job id cited by CLAIMS | 473 | 138.0 G | keep (never flagged) |
| Cited only by HANDOFF/review/analysis docs (e.g. STD-8199989 = J1 discarded 2.9 G x2, rehearsal 8197167/8198951, build/VAL runs) | 168 | 42.4 G | keep for now. After the finals readout, the user may review the J1/rehearsal dirs (~8 G). |
| Cited by **no** doc (CLAIMS, HANDOFF, review, analysis, tools) | 66 | 20.4 G | **Tier B queue-delete.** Largest: L1-512-7996104 5.1 G, L1-512-8002665 3.4 G, L1-512-8005424 2.6 G, L1-A-7993389 1.8 G, L1-B-7993390 1.7 G, STD-B-8028211 1.4 G, NOISE-8029902 1.2 G, FAIL-2/3-79936xx 1.4 G, PROFILE-LOSS-* 0.65 G; the rest are < 0.3 G each (TRIM-*, MTP-ORACLE*, T2-*, INTEGRATION-build-*, P1-LITE-*, ...). Full list in section 8. |
| `_diagnostics/`, `_failed/`, `_superseded/` | 3 | 11.1 G | Tier B. Pre-freeze diagnostics (10-13..10-17), described in runs/INDEX.md; not final results. |
| INDEX.md, overlay-*.txt, ULFM-WRAP, ULFM-PROBE, RESP-*, COMPACT-BENCH | 9 | ~0 | keep |

Optional and space-heavy without deleting anything: the raw per-request bench JSONs dominate the large STD runs
(`bench/healthy-512.json` ~260 MB and `healthy-128.json` ~150 MB per arm). Gzip would cut them about 10x. This
needs the readout scripts to accept `.gz`, so it is not proposed for now.

### 4.2 mvp/harness (frozen until the decomposition jobs end)

| Path | Size | Action | Reason |
|---|---|---|---|
| job.sh, serve.sh, rank.sh, bench.py, run_job.sh, testhooks.py, timed_client.py, steady_state.py, ab_stats.py, burst_counts.py, accuracy_*.py, add_markers.py, pin_busy.py, drop_page_cache.py, configs/, eplb/, tokens/, kbench/, launch/ (task.sh is used by job.sh), draft_cases/ (job.sh runs val_verdicts, ft_watch, expert_hist, pin_tactics, transplant_tactics from it), probe/ | | keep | live |
| **finals_readout.py** | 30 K | **keep (NOT dead)** | `tools/finals_stats.py` imports `mvp/analysis/ctx_stall_cost.py`, which imports `finals_readout` (t_ppf, tactic_status, STALL_X). The canonical readout depends on it. Also CLAIMS 3x. |
| job.sh.pre-{endcase-timeout,kf-warm-wave3,shutdown-skip,v6,xid-trigger}-20261023, bench.py.pre-kf-warm-wave3-20261023, finals_readout.py.pre-stall15-20261023, serve.sh.pre-effective-env-20261023 | 1.75 M | TRASH | `*.pre-*` backups (also in git history of the harness branch if pushed) |
| .job.W1ce | 242 K | TRASH | 10-20 editor/mktemp leftover |
| EXCLUDE_NODES | 114 B | TRASH | 10-17 copy; superseded by mvp/EXCLUDE_NODES.txt; referenced by nothing |
| build_bench.sh, burst_counts_burnin.py, ctx_steps.py, curves.py, ftoff_sanity.py/.sh, k3_verify.sh, nsys_ab.py, perf_table.py, prs_build.sh, run_integration.sh, status.py, step_readout.py, t2_build_ompi.sh, ttft_band.py, ulfm_transport_check.sh | ~90 K | archive | no reference from job.sh/serve.sh/bench.py/launch/tools; not cited by CLAIMS |
| itl_stalls.py, kv_bins.py | 5 K | keep | CLAIMS 139 / kv bins cite them |
| draft_cases/*.patch, *.md (applied drafts: FINAL_CASES, FT-COST-BISECT, SHUTDOWN-R1, testhooks*, ghost-rename) | ~100 K | archive | already applied. job.sh:2763 mentions FT-COST-BISECT.md in a comment, so archiving it only leaves a dangling comment. |

## 5. launch/, review/, tools/, archive/

### 5.1 launch/ (164 entries, 6.4 M)

| Path | Action | Reason |
|---|---|---|
| fin6_j1..j8, fin6_val5, fin6_smoke, gen_finals_v6.py | keep | finals |
| fin6_ra, fin6_rb, fin6_rc | archive after the finals readout | rehearsal (7e8dfb37df / 1117f7e0e7) |
| build_refreeze5_cmd.txt, build_main9cc_cmd.txt, tune_freeze_cmd.txt, tune_freeze_noeplb_cmd.txt, series_smoke_v5pr_cmd.txt | keep | provenance of the frozen wheel, main wheel and pinned tactic tables; latest smoke |
| std/ (std-seed.json, std-seed2.json) | keep | serve.sh reads std-seed |
| gen_finals_v6.py.pre-refreeze2-20261023 | TRASH | backup |
| old-v4-20261023/ (27 fin_* v4 launch files) | archive | superseded by v6 |
| all other ~125 files (`*_cmd.txt` 10-20..10-23 incl. build_{v3,v5,land,zero*,recvfix*,exp,final,freeze,refreeze..4,main}_cmd, val_*, gate_*, zero_ab*, kf_rows-*, overlay-*.list, node_agent.py, remote_exec.py, task.sh, build_v3.sh) | archive | provenance of CLAIMS runs; no current job uses them. Move with the same subpath, so CLAIMS-era references still resolve under archive/2026-10-24/launch/. |

### 5.2 review/

See section 6 for the doc set. Files that are not docs:

| Path | Action | Reason |
|---|---|---|
| decomp-a/b/nsys_cmd.draft.txt, harness-online-decomp.patch, std-online-frozen.yaml.draft | keep until DECOMP is installed and launched, then archive | approved post-finals jobs |
| diag_online_lay_cmd.draft.txt, harness-std-online-lay.patch | archive | replaced by DECOMP (HANDOFF 17:55-17:58) |
| fin6_val5_cmd.draft.txt | TRASH | installed as launch/fin6_val5_cmd.txt |
| harness-kf-warm-wave3.diff | archive | already installed (job.sh/bench.py 14:33) |
| refreeze3-*.diff, refreeze4-*.diff, refreeze5-6dde0496d5.diff, refreeze5-d91ce454b3.diff | archive | intermediate refreeze diffs |
| refreeze5-58caa0fde6.diff, v5-pr-batch3-58caa0fde6.diff | keep | the frozen build's last delta; current v5-pr batch |
| DESIGN_DOC_audit_edits.diff | archive | 10-21 doc audit, applied |
| .wtest (0 B, 10-23 09:10) | TRASH | write test |
| finals/ (v2 F1 readouts) | keep | CLAIMS cites review/finals/ |
| redteam/ | keep | CLAIMS 6x |
| hostfix/ (17 notes, 10-22) | archive | host-path investigation notes; cited by nothing |
| figures/ (5.2 M) | keep | design-doc figure data (design + finals_figures) |

### 5.3 tools/

| Path | Action | Reason |
|---|---|---|
| finals_stats.py, finals_figures.py, finals_manifest.py, fill_design_doc.py, trim_readout.py, ctx_classify.py (FINALS_PLAN block C), verify_wheel.py, units_gate.sh, point_units.sh, dry_run.sh, presubmit.py, require_presubmit.sh, require_build_pass.sh, import_smoke.sh, series_import_smoke.sh, series_smoke.py, failfirst_tests.sh, pytest_plugins/, tests/ (t_*.py unit checks of the harness/tools), nsys + nsight-systems/ (567 M), pylib/ (168 M), a2a_microbench.py + a2a_expert_load.json + a2a_mpirun.sh (launch-referenced, CLAIMS-era), wheels/ (numpy, 15 M) | keep | live, or referenced by current scripts or memory |
| finals_stats.py.pre-fix-20261023 | TRASH | backup |
| finals_readout.py vs finals_stats.py | (not dead; see 4.2) | finals_stats.py is the canonical readout and reuses finals_readout through ctx_stall_cost |
| a2a_microbench.sh, a2a_microbench_cap.sh, a2a_microbench_ep.sh | archive | 10-21 microbench wrappers; unreferenced |
| archive_apply.sh, archive_plan.tsv, archive_moves-2026-10-21.log | keep (log) / archive (script + tsv) | archive_apply is a template for this cleanup's apply script |
| build_test_all.sh, cleanup_stale.sh, commit_staged.sh, stage_hunks.py, cpu_build.sh, cpu_pytest.sh, gtest_debug.sh, overlay_verify.sh, test_wheel.sh, uct_check.sh, v2_test_completion.sh, v3_tests_rest.sh, watch_job.sh, setup.log | archive | no reference from current scripts, launch, docs or memory (README lists cleanup_stale.sh but it was last used 10-21) |

### 5.4 archive/ (already archived; space)

| Path | Size | Action | Reason |
|---|---|---|---|
| archive/codex | 73.5 G | user (Tier C) | Codex-era evidence (memory codex-history-location). Many entries exist both as an extracted dir and as `.tar.gz` (backend-*-followup-*): delete the extracted copies, or keep the tarballs only. |
| archive/exploration-2026-10-15 | 36.4 G | user (Tier C) | exploration worktree/evidence (ep32, kbench, final, evidence) |
| archive/2026-10-21/build-root-main | 15.4 G | queue-delete (Tier C) | a build tree (not evidence) |
| archive/2026-10-21/overlays (12 build-src-* snapshots 1.5-3.1 G, build-root-short, tests-*, ck-*) | >= 20.5 G (2 dirs unmeasured) | queue-delete the build-src-*/build-root-short (Tier C); keep the small Python overlays | source snapshots are reproducible from git |
| archive/builds (21 wheels x 0.8 G + build-root-main 14.4 G) | 30.6 G | queue-delete (Tier C) | old wheels (build-final-*, build-stage-*); no CLAIMS row cites these names |
| archive/2026-10-21/ompi-ft-ucx-prrte-dl, -dl2 | 0.26 G | queue-delete | old Open MPI download/build trees |
| archive/git (10-14 history bundle 1.8 G), git-2026-10-21 (3 bundles 5.4 G), git-2026-10-21b (2 bundles 3.6 G) | 10.2 G | Tier B: first write ONE fresh all-refs bundle of /tmp/claude-161238/work/repo (~1.8 G, before deleting branches; section 7), then queue-delete the 6 older bundles | net ~8.4 G; the new bundle supersedes them |
| archive/2026-10-13 | 2.4 G | user (Tier C) | first-week builds and slurm |
| archive/ep32-runs (74 runs) | unmeasured (du > 25 min) | user (Tier C) | 10-1x EP32 runs |
| archive/kbench-runs 85 M, logs 12 M, docs, harness, probes, reviews, stale-2026-10-21, pr-descriptions-11-commits | ~100 M | keep | small |

## 6. Minimal current doc set

Keep in place, as the current set:

| Doc | Role |
|---|---|
| review/HANDOFF.md | dated log (273 K). Optional: cut everything before 2026-10-22 18:00 into `archive/2026-10-24/review/HANDOFF_to_2026-10-22.md` and leave a pointer line. CLAIMS cites "HANDOFF <time>" (107x), so the archived part must keep its timestamps. |
| review/FINALS_PLAN.md (v6 + PRE-REG), review/FINALS_SCHEDULE_v6.md | the finals |
| mvp/CLAIMS.md, mvp/DECISIONS.md | evidence, decisions |
| mvp/DESIGN_DOC_v5final_draft.md (+ mvp/figures, review/figures) | the design doc |
| review/CODE_AUDIT_20261023.md, review/REDTEAM_QUALITY_20261023.md, review/CLEANUP_PLAN.md (code cleanup), review/RESTACK_v5-pr.md, review/PR_DRAFTS.md, review/HYGIENE_COMMITS_HEADERS.md (new, 10-23) | current code/PR work |
| review/REHEARSAL_GONOGO.md, review/REHEARSAL_READOUT_RA_RB.md, review/DEVIATIONS.md | the GO evidence for the finals; archive after the finals readout |
| review/WORKSPACE_CLEANUP_PLAN.md | this plan |
| mvp/PREFLIGHT.md, mvp/RETRO.md, mvp/OPERATING_PROMPT.md, README.md (rewrite) | rules |

Keep only because CLAIMS cites them (they are otherwise superseded): review/OWNER_REVIEW_SUMMARY.md (13x),
review/FINDINGS_2026-10-18.md, review/DECISION_TREE.md, review/PR_FIX_LIST.md, review/finals/, review/redteam/,
mvp/DESIGN_DOC.md.

Archive to `archive/2026-10-24/review/`:
- review/EXEC_PLAN.md and review/FREEZE_CHECKLIST.md (pre-freeze plan/gate, done; README still calls them the source of truth)
- review/FINALS_SCHEDULE.md (v4/v5)
- review/PR_PLAN.md and review/PR_SPLIT.md (superseded by PR_DRAFTS + RESTACK_v5-pr)
- review/MONITOR_HANDOFF.md (10-22 08:17) and review/MONITOR_TO_DEV.md
- review/VAL_PLAN.md, review/VAL_COVERAGE.md, review/INTEGRATION_PLAN.md, review/HEALTHY_ZERO.md, review/c33_land.md
- review/FREEZE_REVIEW_FIXES.md (done)
- review/DOC_CODE_AUDIT.md, review/DOC_FIXES_v5zero.md, review/doc_audit_1..3.md (10-21 audits, applied)
- review/finals_readout_F1.md (v2 finals; the data is in review/finals/)
- review/hostfix/
- the mvp items listed in section 4

## 7. Git

### 7.1 /tmp/claude-161238/work/repo: local branches (57) and worktrees (34)

Keep: `v5-final` (6688790d56; contains the frozen 58caa0fde6, 1117f7e0e7 and main 9cc37929e8), `v5-pr` (28ad75b7bb,
worktree /tmp/claude-161238/wt-pr), `wideep-ft-dev` (this checkout). No backup tags exist locally besides
`wideep-ft-before-rebase-20261002` and `wideep-ft-exploration-v0`, which are kept.

**Provenance first.** None of the other branches is an ancestor of v5-final, v5-pr or wideep-ft-dev (all were
rebased). Their tips hold the commits that CLAIMS wheels were built from. So before deleting, tag 5 tips locally
(not pushed) and write the fresh bundle from 5.4:

| Tag (proposed) | At branch | Keeps these CLAIMS commits reachable |
|---|---|---|
| claims/v5-zero | v5-zero fc7bfab236 | 9f2f1892f0, 6645ce1848, f9e6bc0f8b, 65a39b411d, 95d6a64a73, b6f895cbc5 |
| claims/v5-recvfix | v5-recvfix a3d2440044 | b643ad8a29, b0ad558bcb |
| claims/v5-exp | v5-exp 69c4ea430a | eb3950a3f8, 2ccc3e8e1c |
| claims/adp-gate-quiet | prmain/adp-gate-quiet 2d08fc1144 | cef4670789 (v2), bcea8da496 (v3), a02f262be7, bdb01feb4f, 733ef6d445, 4de25459ce |
| claims/v3-c33 | v3-c33 db3af1f98c | 0fed8d8141 |

Proposed deletion list for the user. Remove worktrees with `git worktree remove <path>`. Branches use `git branch -D`
because none is merged.

| Worktree | Branch | State | Note |
|---|---|---|---|
| /tmp/claude-161238/restack/wt-asm-{E,I,K,R,X}, wt-asm-R-{c01,c02,c04,c05,w08,w09} (11) | detached 6837295b9c (10-20 merge) | **dirty: 7-115 tracked files each** | 10-20 restack assembly scratch (restack/assemble.py), superseded by v5-pr. Uncommitted changes are lost on removal: if wanted, keep a `git diff` per worktree first. `/tmp/claude-161238/restack/` (scripts, logs) can go to TRASH. |
| /tmp/claude-161238/work/wt-rsg-{E,K,R,X} | rsg-v2/{E,K,R,X} | clean | 10-20 restack staging |
| /tmp/claude-161238/wt-v2, wt-v2cpp, wt-v2py | v2, v2-cpp, v2-py | clean | |
| /tmp/claude-161238/wt-v3, wt-v3c33 | v3, v3-c33 | clean | v3-c33 -> tag first |
| /tmp/claude-161238/wt-v5 | v5 | clean | memory subagent-usage says agents start in wt-v5: update that memory to wt-split/wt-pr first |
| /tmp/claude-161238/wt-exp, wt-recvfix, wt-zero | v5-exp, v5-recvfix, v5-zero | clean | tag first |
| /tmp/claude-161238/wt-land, wt-rebase | v5-land, v5-land-rebase | clean | |
| /tmp/claude-161238/wt-rvA, wt-rvB, wt-rvC | v5-review-A/B/C | clean | |
| /tmp/claude-161238/wt-prmain-gate | prmain/08-adp-gate-quiet | clean | |
| /tmp/claude-161238/wt-smoke | detached f1d7074efb | clean | |

Branches without a worktree: backup-v5-land-20261022, fix-final8-lean, ftcore/* (9), ftwave2/* (2), prmain/01..07 (7),
prmain/adp-gate-quiet (tag first), rsg/{E,K,R,X}, v4b1, v4c, v5-final-pre-fix2, v5-restructure, v5-review,
v5-review-merged, v5-sq (the last four all at 8c52d11469), v5-split, v5-split-rebase, v5-x1, v5-x3, v5-x4, v5-x5.

Remote-tracking refs `origin/*` (56, from $R/dev): `git fetch --prune` drops them once $R/dev's branches are pruned.

### 7.2 $R/dev and the main Lustre checkout (one repo, shared refs)

| Item | Action |
|---|---|
| worktrees $R/wt/reuse (fix-reuse), $R/wt/sortgate (fix-sort-gate) | remove (clean) and delete the branches |
| /tmp/claude-161238/harness-wt (wide-ep-ft-harness 85b564ac19, clean) | keep if the harness branch is the harness record; otherwise remove |
| branches ftcore/* (9), ftwave2/* (2), prmain/01..07 (7) | delete (10-20 split, superseded by v5-pr) |
| wide-ep-ft-mvp (cef4670789; the main checkout's worktree), wideep-ft-dev (191988d1fb; $R/dev) | keep |

### 7.3 Fork (git@github.com:arusli-nv/TensorRT-LLM.git, `ls-remote` 2026-10-23 ~19:00 PT)

| Ref | SHA | Target state |
|---|---|---|
| refs/heads/main | fc2f8543e0 | keep |
| refs/heads/wide-ep-ft-mvp | cef4670789 (= v2) | keep; still v2. Per branch-layout: after validation, first tag and push `wide-ep-ft-mvp-v2-backup` at cef4670789, then update to the validated code (v5-pr / v5-final) |
| refs/tags/wideep-ft-exploration-v0 | 0cb091c255 | **queue-delete on the fork** (user: `git -C $R/dev push fork :refs/tags/wideep-ft-exploration-v0`); keep the local tag |
| refs/tags/wide-ep-ft-mvp-v2-backup | **missing** | create + push before the wide-ep-ft-mvp update |

So the fork is already down to 2 branches. The remaining gaps: the backup tag does not exist yet, and one extra tag
remains.

## 8. Home (`~`, ~10 GB quota; 9.5 GB used now, so memory writes are at risk)

| Path | Size | Action |
|---|---|---|
| ~/.cursor-server | 3.0 G (bin 2.0 G = 4 server builds of ~0.5 G: c4730f.. 10-01, 2d2987.. 10-05, 37b865.. 10-12, 5d4df9.. 10-18) | queue-delete the 3 older builds (~1.5 G); keep 5d4df9 |
| ~/.cache/pip | 0.94 G | queue-delete (`pip cache purge`) |
| ~/.nv/ComputeCache | 0.84 G | queue-delete (CUDA JIT cache; rebuilt on demand) |
| ~/.cache/tensorrt_llm (flashinfer) | 0.84 G | optional (JIT cache) |
| ~/.cache/pre-commit | 0.32 G | keep (memory: precommit-readonly-cache) |
| ~/.codex | 1.8 G (sessions 0.95 G) | keep (Codex history); `.codex/.tmp` 0.11 G can go |
| ~/.claude | 1.6 G (projects 1.2 G) | keep |
| ~/.local 0.25 G, other < 0.1 G | | keep |

## 9. Runs cited by no doc (Tier B, 66 dirs, 20.4 G)

L1-512-7996104 5.14G; L1-512-8002665 3.42G; L1-512-8005424 2.55G; L1-A-7993389 1.75G; L1-B-7993390 1.74G;
STD-B-8028211 1.41G; NOISE-8029902 1.24G; FAIL-2-7993641 0.88G; FAIL-3-7993645 0.56G; PROFILE-LOSS-8007445 0.44G;
PREFLIGHT-B2-8031198 0.29G; PROFILE-LOSS-7996764 0.21G; B2Y-TOTAL-1K-8034754 0.10G; PREFLIGHT-FINAL-8042599 0.10G;
C-PROF-8048260 0.08G; GLM-FT-8039095 0.07G; GLM-AB2-8042845 0.04G; P1-LITE-8002082 0.04G; STD-B-8027195 0.03G;
TRIM-8036233, TRIM-8038953-old/-new, TRIM-8038506-old/-new, TRIM-8036560-old/-new, TRIM-8035759; P1-LITE-8001014,
-8003031, -8005415; P1-INFLIGHT-8004343; GLM-SCOUT-8028883; SM-DIAG-8026964; T1-SM-8012103; T2-UCT-8015271, -8026608,
-8026609; T2-LAUNCH-8016097, -8016545, -8016810; BORROW-KILL-8044971, BORROW-KILL-EMPTY-8044971; B2Y-TOTAL-1K-8034135;
OVERLAP-ORACLE-8031737; HANG-LITE-8008291, -8008468; MTP-ORACLE-8038953, MTP-ORACLE2-8039436, MTP-ORACLE3-8042301,
-8042745; MTP-DEBUG-8040904; ONLINE-EPLB-LITE-8040904; NODE-LOSS-8035182; FTOFF-8025460; HOSTCOLL2-7995106;
KBENCH-MAIN-8012681, -8016816; INTEGRATION-build-{final-aae00e6c65-8085510, pr8-937403a6cc-8020167,
pr8-937403a6cc-8027202, pr8-257db47f5f-8016198, pr9-d9777a8612-7993946, main-port-ac54232eff-8012964,
k3c-c706fdfa84-7989798}; TEST-WHEEL-build-final-aae00e6c65-8085510; TEST-WHEEL-build-pr0-fc52f0adee-8096829.
(The 7993xxx/7996xxx ids may appear in mvp/runs/INDEX.md, which was not part of the citation corpus: check it before deleting.)

## 10. Apply order (monitor, after the finals and DECOMP)

1. Readout done; CLAIMS rows for the finals written.
2. Write `tools/cleanup_apply-2026-10-24.sh`, modeled on `tools/archive_apply.sh`: plan-only by default, `--yes` to
   act, `mv` only, with a log. Apply the archive/TRASH rows of sections 1-6.
3. Edit the docs: README rewrite, HANDOFF line 2 pointer, and the memory update for wt-v5 -> wt-split/wt-pr.
4. Git: write the 5 provenance tags and one fresh bundle, then hand the user the branch/worktree list (7.1, 7.2) and the
   fork tag deletion (7.3).
5. Hand the user one queue-delete batch:
   - Tier A (~39.5 G)
   - Tier B items they approve (~60.7 G)
   - Tier C items they approve (>= 179 G)
   - the home items (~3.3 G)

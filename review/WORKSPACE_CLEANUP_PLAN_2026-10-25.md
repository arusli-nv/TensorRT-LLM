# Workspace cleanup plan (read-only inventory, 2026-10-25 ~04:20 PT)

Replaces `review/WORKSPACE_CLEANUP_PLAN.md` (2026-10-23 19:30 PT), which was never applied (no `TRASH/2026-10-24` or
`archive/2026-10-24` exists). Nothing was moved, edited or deleted to write this. Policy (README, memory never-run-rm):
superseded files are moved with `mv` to `TRASH/2026-10-25/<same subpath>` or `archive/2026-10-25/<same subpath>`, and
every move is logged to `tools/archive_moves-2026-10-25.log` (format as `tools/archive_moves-2026-10-21.log`). Real
deletions are only queued for the user. Moving inside the same Lustre filesystem frees no space. Only the user's
queue-delete batch frees space.

Actions:
- **keep**: leave in place.
- **archive**: move to `archive/2026-10-25/`. These are history worth keeping, but not current.
- **TRASH**: move to `TRASH/2026-10-25/`. These are superseded or scratch, and become deletable after one quiet day.
- **queue-delete**: the user deletes them. Sizes are listed.

Sizes come from `du -sk` (KiB) run on 2026-10-25, given here in GiB. Two roots timed out at 300 s as a whole
(`build-root`, `overlays/build-root-feat`); their sizes are the sums of per-subdirectory `du -sk` runs, which all finished.
"CLAIMS" means the item is cited in mvp/CLAIMS.md (rows 1-197) by job id, path or name.

State as of 2026-10-25:
- The code of record is v5-final 27442bd91d (wt-split). It is benchmarked as `overlays/fix-dropmap-9f1fd4cab4` (same
  product files) on `build-freeze-58caa0fde6`, and compared against `build-main-9cc37929e8`. The logs of CLAIMS 193 (V5F-REC 8269102),
  194 (V5F-W1A/W1B/W2A 8269103/8270779/8271451), 196 (B-AB2 8273757) and 197 (ONLINE-BURST 8273195) confirm this:
  every one of them has `build-freeze-58caa0fde6` and `OVERLAY=.../overlays/fix-dropmap-9f1fd4cab4` (8273195 sets it
  through `OVERLAY_A` in `launch/online_burst_cmd.txt`). B-AB2 8273757 also has `OVERLAY=.../overlays/ft-hostmem-804cd32049`
  (the option-B arm).
- PR branch: `v5-pr` (19b0e90251). Backup branch: `v5-pr-backup-19b0e90251` (same commit). wt-pr is checked out on the new restack
  `v5-pr-folded` (b27007aca1, HANDOFF 04:03). Its fold test V5PR-FOLD-TEST 8275622 finished at 04:12 (49 passed,
  `V5PR_FOLD_TEST_DONE`).
- Option B (`ft-hostmem` 804cd32049, wt-fix) is not adopted (HANDOFF 04:00/04:01, CLAIMS 196). Its overlays, scratch
  dir and probe outputs are archived as evidence. None of them goes to TRASH.

## 0. Bottom line

| Tier | What | Frees |
|---|---|---|
| **A: queue-delete** (cited by no CLAIMS row and no v5-final-era launch file) | 7 superseded wheels (5.5); main build root `build-root` (18.3); 6 old source snapshots (17.7); the current TRASH content (<0.01) | **41.4 GiB** |
| **B: user decision** (the frozen build's C++ root, delete all three or none) | `overlays/build-root-v2` (6.7) + `overlays/build-root-feat` (15.5) + `overlays/build-src-8c52d11469` (2.6) | **24.8 GiB** |
| Carried forward from the 10-23 plan (10-23 sizes, not re-measured) | `wt/reuse` + `wt/sortgate` worktrees (~1.6 G); uncited runs (~20.4 G); `runs/_diagnostics,_failed,_superseded` (~11.1 G); old `archive/` trees (>= 179 G); home caches (~3.3 G) | user decision, as in that plan |

**Moves** (no space freed; exact commands in section 7):
- archive (now): 7 option-B/pin/fix overlays, 5 scratch dirs (+2 below), `build-main-0829689c52`, 49 old small Python overlays,
  4 mvp docs, 10 mvp/analysis notes.
- archive (now, because 8275622 is done): `scratch-fixtest`, `scratch-v5pr`.
- TRASH: `overlays/fix-nopopulate-dad375768d`, `overlays/tests-8ec46695cc`,
  `mvp/analysis/ctx_stall_cost.py.pre-stall15-20261023`.

**Correction to the 10-23 plan:** `overlays/build-root-feat` is not an independent old root. `overlays/build-root-v2`'s
`ccache`, `cpp-build-oot`, `fmha-gen` and `venv-3.12` are symlinks into it (11.0 GiB of its 15.5). Deleting only
`build-root-feat` (as the 10-23 plan proposed) would break the incremental root of the frozen build. That is why
build-root-v2, build-root-feat and build-src-8c52d11469 form one all-or-nothing Tier B unit.

**Withdrawn from the 10-23 plan:** these directories are now **keep as a whole**: `launch/`, `review/`, `tools/` and
`mvp/harness/`. So the 10-23 archive/TRASH rows inside them (the `*.pre-*` backups, old launch files, review docs,
unused tools) are dropped. If you want the backups gone anyway, the candidates are 14 `mvp/harness/*.pre-*-2026102[34]`
files, `tools/finals_stats.py.pre-fix-20261023` and `launch/gen_finals_v6.py.pre-refreeze2-20261023`. They need your OK.

## 1. Keep (never touched)

- Builds: `build-freeze-58caa0fde6` (0.78), `build-main-9cc37929e8` (0.78).
- The v5-final overlay: `overlays/fix-dropmap-9f1fd4cab4` (CLAIMS 187, 193, 194, 196, 197; used by every V5F job).
- `overlays/empty` (CLAIMS 27x, incl. 182/189; v5-final-era launch files decomp_a, ghost_empty, decomp_nsys,
  launchtail_diag).
- Dirs: `tools/`, `launch/`, `mvp/harness/`, `review/` (incl. `review/diag/`, which holds the option-B and probe
  outputs), `ompi-ft-ucx/`, `cache/`, `TRASH/`, `archive/`.
- Scripts: `bundle.sh`, `run_job.sh` (in mvp/harness), `install_wheel.sh`, `overlay_python.sh`; also `build_and_test.sh`,
  `restore_build.sh` (build path).
- `mvp/CLAIMS.md`, `mvp/DECISIONS.md`, `mvp/EXCLUDE_NODES.txt`, `mvp/PREFLIGHT.md`, `mvp/RETRO.md`,
  `mvp/OPERATING_PROMPT.md`, `mvp/DESIGN_DOC_v5final.md`, `mvp/DESIGN_DOC.md` (CLAIMS cites "DESIGN_DOC Evaluation"),
  `mvp/USER_DESIGN_DOC_STYLE_REF_20261020.md`, `mvp/figures/`, `mvp/archive/`, `mvp/analysis/` (except section 5).
- `mvp/runs/` (842 dirs): all kept, see section 4.
- `dev/` (fork push worktree), `logs/` (bundle logs cited by CLAIMS), `probe/` (HANDOFF), the 0-byte sandbox stubs.
- CLAIMS-cited wheels, kept in place: `build-v2-cef4670789` (CLAIMS 140, 142), `build-v3-bcea8da496` (144-146),
  `build-v5-2ccc3e8e1c` (165, 167-169), `build-v5r-b643ad8a29` (165), `build-v5r-b0ad558bcb` (167),
  `build-v5x-eb3950a3f8` (169), `build-v5land-9f2f1892f0` (171), `build-v5zero-6645ce1848` (171),
  `build-v5zero-f9e6bc0f8b` (172, 173), `build-pr0-fc52f0adee` (128, 139). That is 10 x ~0.78 = 7.8 GiB. Each is reproducible
  from its commit (10-23 verify_wheel). Optional later: you could delete their `dist/*.whl` and keep the logs.
- CLAIMS-cited small overlays, kept in place: `py-final8-4de25459ce` (138), `v3c33-0fed8d8141` (158, 161),
  `v4bc-bdb01feb4f` (146, 152), `vgate-2d08fc1144` (168), `vgate-733ef6d445` (166), `vzero-65a39b411d` (172),
  `vzero-95d6a64a73` (171), `vzero-b6f895cbc5` (172, 173).
- `README.md`: keep, but **edit**. It still describes v3/v5, EXEC_PLAN and `build-v3-bcea8da496` as current. Rewrite it for
  v5-final 27442bd91d = fix-dropmap overlay on build-freeze-58caa0fde6, main 9cc37929e8, v5-pr/v5-pr-folded, and the
  CLAIMS 193-197 readouts.

## 2. Builds: top-level `build-*` and the C++ build roots

Check per root: is it cited by a CLAIMS row, or referenced by a v5-final-era launch file (launch files written on or
after 2026-10-24)?

| Path | Size (GiB) | CLAIMS | v5-final-era launch | Action | Reason |
|---|---|---|---|---|---|
| build-freeze-58caa0fde6 | 0.78 | 174-191, 193, 194, 196, 197 | 49 files | **keep** | the base of every v5-final run |
| build-main-9cc37929e8 | 0.78 | (as "main") | 38 files | **keep** | the main arm |
| build-c43-b0ad558bcb | 0.78 | no (hash only via build-v5r-b0ad558bcb, row 167) | no | queue-delete | C43 repro diagnostic |
| build-c43fix-b0ad558bcb | 0.78 | no (same) | no | queue-delete | same |
| build-freeze-0e5a22ef51 | 0.78 | no | no | queue-delete | refreeze3, never benchmarked |
| build-freeze-1117f7e0e7 | 0.78 | no (J1 8199989 is cited as discarded; the build path is not) | no | queue-delete | refreeze4: rehearsal RC + discarded J1 |
| build-freeze-5330b899f0 | 0.78 | no | no | queue-delete | first freeze (759787cb17) |
| build-freeze-7e8dfb37df | 0.78 | no | no | queue-delete | rehearsal RA/RB |
| build-freeze-8c52d11469 | 0.78 | no (CLAIMS line 269 names the review hash 8c52d11469, not this build) | no | queue-delete | tactic-tuning build, superseded by the pinned tables on 58caa0fde6 |
| build-main-0829689c52 | 0.77 | **yes**: CLAIMS wheel-audit note (line 266, "NOT clean") names it | no | **archive** | cited, so not deleted; not a numbered row (see section 8) |
| CLAIMS-cited wheels (10, section 1) | 7.8 | yes | no | keep | cited |
| build-root (top level) | **18.28** (ccache 1.13, cpp-build-oot 7.05, package 2.25, venv-3.12 3.35, wheel-staging 4.45, fmha-gen 0.05) | no | no (12 older launch files) | **queue-delete** | main's incremental root, last built 9cc37929e8. The main wheel already exists, so a later main rebuild would be cold. |
| overlays/build-src-9cc37929e8 | 2.58 | no | no | queue-delete | main's source snapshot; goes with build-root |
| overlays/build-src-5330b899f0 | 3.13 | no | no | queue-delete | first-freeze snapshot |
| overlays/build-src-9f2f1892f0 | 2.58 | hash only (row 171, via build-v5land-9f2f1892f0) | no | queue-delete | source snapshot, reproducible from git (keep 9f2f1892f0 reachable: branch v5-land or the claims/v5-zero tag from 10-23 section 7.1) |
| overlays/build-src-aae00e6c65 | 3.07 | hash only (rows 118-128, via build-final-aae00e6c65) | no | queue-delete | reproducible: aae00e6c65 is wideep-ft-dev's checkout HEAD |
| overlays/build-src-cef4670789 | 3.10 | hash only (rows 140, 142, via build-v2) | no | queue-delete | reproducible: branch v2 / fork wide-ep-ft-mvp |
| overlays/src-final8-4de25459ce | 3.20 | hash only (row 138, via py-final8 overlay) | no | queue-delete | reproducible (claims/adp-gate-quiet tag, 10-23 section 7.1) |
| overlays/build-root-v2 | 6.69 (own: package + wheel-staging; the rest are symlinks into build-root-feat) | no | no (last used by `launch/build_refreeze5_cmd.txt`, 10-23 14:25, which built 58caa0fde6) | **user decision (Tier B)** | incremental root of the frozen build (last_built_commit 58caa0fde6). Needed for any C++ rebuild of v5-final/v5-pr. |
| overlays/build-root-feat | 15.51 (ccache 0.54, cpp-build-oot 7.05, venv-3.12 3.38, fmha-gen 0.05 = 11.03 used by build-root-v2; package 2.25 + wheel-staging 2.23 own) | no | no | **user decision (Tier B)**, only together with build-root-v2 | see the correction in section 0 |
| overlays/build-src-8c52d11469 | 2.59 | no | no | **user decision (Tier B)** | source snapshot at 58caa0fde6 that build-root-v2 builds from |

Tier A builds and snapshots: 5.46 (7 wheels) + 18.28 + 17.67 (6 snapshots) = **41.41 GiB**. Tier B: **24.79 GiB**.

## 3. overlays/ (Python overlays) and scratch-*

### 3.1 Overlays added since 10-23

| Path | Size | CLAIMS | Action | What |
|---|---|---|---|---|
| fix-dropmap-9f1fd4cab4 | 100 K | 187, 193, 194, 196, 197 | **keep** | v5-final 27442bd91d product files (fix 1 + fix 2) |
| ft-hostmem-804cd32049 | 220 K | 196 | **archive** | option B (B-AB2 8273757), not adopted |
| ft-hostmem-70a2a9f869 | 108 K | no (HANDOFF: B-AB 8272666, B arm OOM at startup) | **archive** | option B, first version |
| fix-pin-5b42dabe65 | 104 K | no (HANDOFF 22:43, PROBES-COMBO 8270321 unit tests) | archive | page-lock option, rejected (kernel 6.8 isolates pinned shmem folios). V5P-* launch files were never run. |
| fix-pingpu-85ea85de95 | 104 K | no (same) | archive | page-lock + GPU copy option, rejected. V5G-* launch files were never run. |
| diag-nopopulate-0eb6e82ac0 | 104 K | 184 | archive | POPULATE-AB 8241601 diagnostic arm |
| fix-nopopulate-d71f169e4b | 100 K | 185, 186, 187 | archive | fix 1 alone (FIX-REC/FIX-W*), superseded by v5-final |
| fix-dropmap-087d1e7f19 | 100 K | no (HANDOFF 12:05-13:08: first fix-2 commit; its test ran via scratch-fixtest2 in FAILFIRST-FIX2 8255907) | archive | superseded by 9f1fd4cab4 |
| fix-nopopulate-dad375768d | 100 K | no; no launch file | **TRASH** | reviewed fix-1 draft, superseded by d71f169e4b (HANDOFF 11:39) |

### 3.2 Older overlays (carried from 10-23)

- keep: the 8 CLAIMS-cited overlays in section 1, and `empty`.
- archive: the other 49 small overlays: `all-4b7fcd1920 all-71ac3036a0 final-40a0c5fb5c merge-a470d307bb mtp-aacdb2c918
  mtp-debug stack-3e9027f4e4 stack-4c2929329c v3c33-0fed8d8141-tests v3c33-db3af1f98c-tests v4b-ee76f03e74
  v4b1-d00215dd68 v4b2-38c2cab828 v4b3-3d75bf5da1 v4bc-957ac92c5f v4bc-957ac92c5f-tests v4bc-bdb01feb4f-tests
  v4bcd-a02f262be7 v4bcd-a02f262be7-tests v4c-f0b6d3f1c9 v5-fa884734ad v5-fa884734ad-tests v5b-409d2bd1f8
  v5b-409d2bd1f8-tests v5c-ca011e89fc-tests v5d-e452975950-tests v5e-c7b13ceb85 v5e-c7b13ceb85-tests v5f-f96a8c1a53
  v5f-f96a8c1a53-tests v5g-5ebf7f72a2 v5g-5ebf7f72a2-tests v5h-2e8d6d7da1 v5h-2e8d6d7da1-tests vexp-d1065858eb
  vexp-d1065858eb-tests vgate-2d08fc1144-tests vgate-733ef6d445-tests vzero-4a30682ead-tests vzero-5ff84d0da9
  vzero-5ff84d0da9-tests vzero-65a39b411d-tests vzero-95d6a64a73-tests vzero-981fd845e2 vzero-981fd845e2-tests
  vzero-b6f895cbc5-tests vzero-d14c00b06f-tests vzero-f9e6bc0f8b vzero-f9e6bc0f8b-tests` (~20 M). Some of their commit hashes
  appear in CLAIMS (final-40a0c5fb5c 131, v4bcd-a02f262be7 152, v5f-f96a8c1a53 161, v5g-5ebf7f72a2 169,
  vzero-981fd845e2 171, vzero-f9e6bc0f8b 172/173), so they go to archive, not TRASH.
- TRASH: `tests-8ec46695cc` (69 M; one old launch file, test_wheel_066; not cited).
- queue-delete: the build-src-*/src-final8 snapshots (section 2).

### 3.3 scratch-* (all new on 10-24/25)

| Path | Size | What it was for | CLAIMS | Action |
|---|---|---|---|---|
| scratch-a2abench | 52 K | `a2a_bench.py` (+ `.v3`): one-node EP4 fence-path A2A microbench, off vs FT rank mask (A2A-MICRO 8266985/8267120/8267268/8267451) | **191** (path) | archive |
| scratch-compact | 76 K | `compact_probe.py` (+ `_v1`): /dev/shm readers under forced compaction, mapped/lazy/pinned (COMPACT-PROBE 8268725, COMPACT-PROBE2 in 8270321) | **192** (path) | archive |
| scratch-probe3 | 44 K | `probe3.c`, `probe4.c`: C readers by source type (shm pre-faulted / lazy / HOST_NUMA) under targeted compaction (PROBE3 8271169, PROBE4 8271450) | **195** (path) | archive |
| scratch-hostmem | 188 K | option B: `hostmem_probe.py` (HOSTMEM-PROBE in 8270321), `warm_cache_probe.py` (HOSTMEM-RESERVE-COMBO 8273280) | no (HANDOFF 4x) | archive (option-B evidence) |
| scratch-fixtest | 40 M | fix 1 (d71f169e4b) fail-first/pass: `run.sh`, `new/` (fixed moe_load_balancer.py), `tests/`, `requirements-dev*.txt`. Used by FIX-REC/V5F-REC preludes (8244019 INVALID, 8269102), FAILFIRST-FIX 8253005 and FAILFIRST-FIX2 8255907; its requirements file was also used by V5PR-FIXTEST 8264887 and V5PR-FOLD-TEST 8275622 | job 8255907 (row 187) | archive (8275622 is done) |
| scratch-fixtest2 | 40 M | fix 2 (087d1e7f19) fail-first/pass, `run.sh` + `new/` + `tests/` (FAILFIRST-FIX2 8255907) | job 8255907 (row 187) | archive |
| scratch-v5pr | 397 M | `run.sh` + `git archive` exports per commit for the unit tests: 1a683f2a3b, 19b0e90251 (V5PR-FIXTEST 8264887); 7c6fef3528, 6922263d92 (V5PR-FOLD-TEST 8275622); 5b42dabe65, 85ea85de95 (pin options, PROBES-COMBO 8270321); 70a2a9f869, hostmem-reserve (option B units, HOSTMEM-UNIT 8271796, HOSTMEM-RESERVE-COMBO 8273280) | no | archive (git-reproducible, but holds the option-B reserve draft as tested) |

## 4. mvp/runs

842 dirs. 98 are new since the 10-23 plan: 87 of them are cited by CLAIMS, and 11 are cited only by HANDOFF. **All are kept.**
Runs are never moved: HANDOFF and readouts cite them by path. The 11 runs without a CLAIMS citation:

| Run | Size (GiB) | What |
|---|---|---|
| STD-8241603, STD-G1-8241603 | 0.32 + 0.38 | NSYS (one FT case lost to a harness 409) |
| STD-8244358 | 0.27 | NSYS-STEADY: captured the burn-in window by mistake |
| STD-8266440, STD-G1-8266440 | 2.39 + 0.60 | MIDBURST-DIAG2 (its readout is cited by CLAIMS 192, not the job id) |
| STD-8267635 | 1.26 | MIDBURST-DIAG3, cancelled 21:28 |
| STD-8268513 | 0.06 | MIDBURST-DIAG4 |
| STD-8272666, STD-G1-8272666, KILL-FINAL-ON-8272666 | 0.05 + 0.23 + 0.01 | B-AB: B arm OOM at startup (option-B evidence) |
| KILL-FINAL-ON-A-8271210 | 0.10 | **V5F-REC-UPD**: the v5-final online rank5upd row that CLAIMS 193 lists as "pending". There is no CLAIMS row for it yet (section 8). |

The 10-23 Tier B lists (66 runs cited by no doc, 20.4 G; `_diagnostics/_failed/_superseded`, 11.1 G) carry forward
unchanged. They were not re-measured.

## 5. mvp docs and TRASH/

| Path | Action | Reason |
|---|---|---|
| mvp/MORNING_SUMMARY.md, QNA_PREP.md, SLIDES.md (10-21) | archive | midpoint material. The current morning summary is review/MORNING_SUMMARY.md. |
| mvp/DESIGN_DOC_v5final_draft.md (10-23 09:27) | archive | superseded by mvp/DESIGN_DOC_v5final.md and review/DESIGN_DOC_v5final_tonight.md; not cited by CLAIMS |
| mvp/analysis/DECISIONS.md | archive | pre-10-21 decisions log (mvp/DECISIONS.md starts 10-21) |
| mvp/analysis/{CPU_PYTEST_2026-10-19, FIXB_HANG, FT_COST_PROFILE_2026-10-19, FT_ONE_SWITCH, GHOST_PLACEMENT_RULE, INPLACE_SLOTS_BACKENDS, MTP_INPLACE, RANK0_MASK_TEST, V7_EXTRACT}.md | archive | 10-18/19 notes. FT_COST_PROFILE and MTP_INPLACE are cited by CLAIMS, so they go to archive, not TRASH. |
| mvp/analysis/ctx_stall_cost.py.pre-stall15-20261023 | TRASH | backup |
| TRASH/2026-10-22/build-v5r2-b0ad558bcb (empty dist/), TRASH/dryrun-stray-1-20261024/ (a harness copy a dry run wrote by mistake), TRASH/review_diag_q2.py.accidental-copy-from-tmp | queue-delete | already in TRASH for more than a quiet day; 6.8 M in total |

## 6. Git (short update; the 10-23 section 7 lists still apply)

- Keep: `v5-final` 27442bd91d (wt-split), `v5-pr` 19b0e90251, `v5-pr-backup-19b0e90251`, `v5-pr-folded` b27007aca1 (wt-pr; restack
  awaiting your decision on whether it replaces v5-pr), `wideep-ft-dev`.
- New since 10-23, provenance of CLAIMS rows and not ancestors of v5-final: `fix-dropmap` 9f1fd4cab4 (CLAIMS 187; the
  v5-final overlay's source), `ft-hostmem` 804cd32049 (contains 70a2a9f869; CLAIMS 196, wt-fix), `diag-nopopulate`
  0eb6e82ac0 (CLAIMS 184, wt-diag), `fix-pin` 5b42dabe65 and `fix-pin-gpucopy` 85ea85de95 (not adopted). Tag
  each locally (for example `claims/fix-dropmap`, `claims/ft-hostmem`, `claims/diag-nopopulate`, `evidence/fix-pin`,
  `evidence/fix-pin-gpucopy`) before deleting a branch or removing a worktree. Do the same for the earlier 5 claims/* tags,
  which do not exist yet. Then write one fresh all-refs bundle.
- `fix-nopopulate` and `a2a-ftcost` (both d71f169e4b) are ancestors of v5-final. You can delete them, and remove wt-a2a.
- The worktree and branch deletion lists of 10-23 section 7 are unchanged (none of them was applied).

## 7. Exact move commands (for later; NOT run)

Apply after the 10-25 morning review. Modeled on `tools/archive_apply.sh`. These commands only move; they delete nothing.

```bash
set -uo pipefail
R=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude
A=$R/archive/2026-10-25
T=$R/TRASH/2026-10-25
LOG=$R/tools/archive_moves-2026-10-25.log
mkdir -p "$A" "$T"
mvto() {  # mvto <dest root> <subpath relative to $R>
    local dst=$1 rel=$2
    if [[ ! -e $R/$rel && ! -L $R/$rel ]]; then echo "MISSING $rel" >> "$LOG"; return; fi
    mkdir -p "$dst/$(dirname "$rel")" && mv "$R/$rel" "$dst/$rel" \
        && echo "MOVED $rel -> ${dst#$R/}/$rel" >> "$LOG" || echo "FAILED $rel" >> "$LOG"
}

# archive: option B (not adopted), page-lock options, superseded fix overlays
for o in ft-hostmem-804cd32049 ft-hostmem-70a2a9f869 fix-pin-5b42dabe65 fix-pingpu-85ea85de95 \
         diag-nopopulate-0eb6e82ac0 fix-nopopulate-d71f169e4b fix-dropmap-087d1e7f19; do
    mvto "$A" "overlays/$o"
done
# archive: scratch dirs (probe drivers cited by CLAIMS 191/192/195, option-B probes, fail-first tests, v5-pr exports)
for s in scratch-a2abench scratch-compact scratch-probe3 scratch-hostmem scratch-fixtest scratch-fixtest2 scratch-v5pr; do
    mvto "$A" "$s"
done
# archive: CLAIMS wheel-audit evidence (not clean main wheel)
mvto "$A" build-main-0829689c52
# archive: old small Python overlays (49)
for o in all-4b7fcd1920 all-71ac3036a0 final-40a0c5fb5c merge-a470d307bb mtp-aacdb2c918 mtp-debug \
         stack-3e9027f4e4 stack-4c2929329c v3c33-0fed8d8141-tests v3c33-db3af1f98c-tests v4b-ee76f03e74 \
         v4b1-d00215dd68 v4b2-38c2cab828 v4b3-3d75bf5da1 v4bc-957ac92c5f v4bc-957ac92c5f-tests \
         v4bc-bdb01feb4f-tests v4bcd-a02f262be7 v4bcd-a02f262be7-tests v4c-f0b6d3f1c9 v5-fa884734ad \
         v5-fa884734ad-tests v5b-409d2bd1f8 v5b-409d2bd1f8-tests v5c-ca011e89fc-tests v5d-e452975950-tests \
         v5e-c7b13ceb85 v5e-c7b13ceb85-tests v5f-f96a8c1a53 v5f-f96a8c1a53-tests v5g-5ebf7f72a2 \
         v5g-5ebf7f72a2-tests v5h-2e8d6d7da1 v5h-2e8d6d7da1-tests vexp-d1065858eb vexp-d1065858eb-tests \
         vgate-2d08fc1144-tests vgate-733ef6d445-tests vzero-4a30682ead-tests vzero-5ff84d0da9 \
         vzero-5ff84d0da9-tests vzero-65a39b411d-tests vzero-95d6a64a73-tests vzero-981fd845e2 \
         vzero-981fd845e2-tests vzero-b6f895cbc5-tests vzero-d14c00b06f-tests vzero-f9e6bc0f8b \
         vzero-f9e6bc0f8b-tests; do
    mvto "$A" "overlays/$o"
done
# archive: superseded mvp docs and 10-18/19 analysis notes
for f in mvp/MORNING_SUMMARY.md mvp/QNA_PREP.md mvp/SLIDES.md mvp/DESIGN_DOC_v5final_draft.md \
         mvp/analysis/DECISIONS.md mvp/analysis/CPU_PYTEST_2026-10-19.md mvp/analysis/FIXB_HANG.md \
         mvp/analysis/FT_COST_PROFILE_2026-10-19.md mvp/analysis/FT_ONE_SWITCH.md \
         mvp/analysis/GHOST_PLACEMENT_RULE.md mvp/analysis/INPLACE_SLOTS_BACKENDS.md mvp/analysis/MTP_INPLACE.md \
         mvp/analysis/RANK0_MASK_TEST.md mvp/analysis/V7_EXTRACT.md; do
    mvto "$A" "$f"
done

# TRASH
for x in overlays/fix-nopopulate-dad375768d overlays/tests-8ec46695cc \
         mvp/analysis/ctx_stall_cost.py.pre-stall15-20261023; do
    mvto "$T" "$x"
done

echo "Moved $(grep -c ^MOVED "$LOG") entries; problems:"; grep -v ^MOVED "$LOG" || echo "  none"
```

That is 7 + 7 + 1 + 49 + 14 = 78 archive moves and 3 TRASH moves.

**Queue-delete batch for the user** (paths only; the user deletes them; 41.4 GiB):
`build-c43-b0ad558bcb`, `build-c43fix-b0ad558bcb`, `build-freeze-0e5a22ef51`, `build-freeze-1117f7e0e7`,
`build-freeze-5330b899f0`, `build-freeze-7e8dfb37df`, `build-freeze-8c52d11469`, `build-root`,
`overlays/build-src-9cc37929e8`, `overlays/build-src-5330b899f0`, `overlays/build-src-9f2f1892f0`,
`overlays/build-src-aae00e6c65`, `overlays/build-src-cef4670789`, `overlays/src-final8-4de25459ce`,
`TRASH/2026-10-22`, `TRASH/dryrun-stray-1-20261024`, `TRASH/review_diag_q2.py.accidental-copy-from-tmp`.
If you decide no further C++ build of v5-final/v5-pr is needed, also delete (Tier B, all three together, 24.8 GiB)
`overlays/build-root-v2`, `overlays/build-root-feat` and `overlays/build-src-8c52d11469`.
Before deleting any source snapshot, create the provenance tags from section 6 / 10-23 section 7.1.

## 8. Open points

- **V5F-REC-UPD 8271210** (`KILL-FINAL-ON-A-8271210/kill_final.txt`) has no CLAIMS row, but CLAIMS 193 still says
  "pending V5F-REC-UPD". It is kept either way. It needs a CLAIMS row or a note.
- `build-main-0829689c52` is named only in the CLAIMS wheel-audit note (line 266), not in a numbered row. It is archived to be
  safe. If you count only numbered rows, it would be queue-delete (0.77 GiB).
- Option-B probe outputs (`review/diag/hostmem-*`, `warm-cache-probe-8273280.txt`, `ft_hostmem_*.diff/.md`,
  `probe3/probe4-*.txt`) stay where they are in `review/diag/`. `review/` is keep as a whole, which preserves them, so they
  are not moved to archive.
- Moving CLAIMS-cited paths (scratch-a2abench/compact/probe3, fix-nopopulate-d71f169e4b, diag-nopopulate-0eb6e82ac0,
  ft-hostmem-804cd32049) changes their location. The same subpath is kept under `archive/2026-10-25/`, and the move log maps
  each one. If you prefer cited paths to resolve unchanged, mark those 7 as keep instead.
- The 10-23 sizes for the archive/ trees, uncited runs and home caches were not re-measured.

## 9. Applied (04:20 PT, monitor)

Section 7 was run with one change: the CLAIMS-cited paths stay in place so that citations resolve unchanged. These are
scratch-a2abench, scratch-compact, scratch-probe3, scratch-fixtest, scratch-fixtest2, and the overlays
ft-hostmem-804cd32049, fix-nopopulate-d71f169e4b, diag-nopopulate-0eb6e82ac0 and fix-dropmap-087d1e7f19.
69 archive moves + 3 TRASH moves = 72 moves; all are logged in tools/archive_moves-2026-10-25.log with no problems. Nothing was deleted.
The queue-delete batch and Tier B in section 7 are for the user.

## 10. Added after section 9 (2026-10-25 afternoon)

Created by today's PR test and diagnostic jobs. Keep while the user audits the PR branches; archive afterwards:
scratch-mpisession, scratch-v5pr-commits, scratch-v5pr-fixes, scratch-v5pr-tests.txt, scratch-v5pr-tip, review/drafts/.
Stale unused copy: review/figures/data/fig9_downtime.csv (rehearsal values; the doc uses figures/finals).

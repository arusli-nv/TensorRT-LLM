# WideEP fault tolerance: working area

Untracked scratch for the WideEP FT work; code lives in git (branch of record `v3`, next build `v5` =
v3 + fence-always, worktrees `/tmp/claude-161238/wt-v3`, `/tmp/claude-161238/wt-v5`). Cleaned 2026-10-21: everything
not used now is in `archive/2026-10-21/` (same subpaths; list in `tools/archive_moves-2026-10-21.log`).

**Source of truth:** `review/EXEC_PLAN.md` (plan to code freeze and finals), `review/FREEZE_CHECKLIST.md` (the gate),
`review/HANDOFF.md` (dated results log). Evidence for every number: `mvp/CLAIMS.md`. Decisions: `mvp/DECISIONS.md`.

| Path | What |
|---|---|
| `review/` | EXEC_PLAN, FREEZE_CHECKLIST, HANDOFF, FINDINGS_2026-10-18 (review findings and status), PR_PLAN, PR_FIX_LIST, OWNER_REVIEW_SUMMARY; `redteam/` (cited by CLAIMS), `finals/` |
| `mvp/` | DESIGN_DOC.md (+ `figures/`, incl. `slides/`), SLIDES.md, CLAIMS.md, DECISIONS.md, PREFLIGHT.md (hard rules), RETRO.md, OPERATING_PROMPT.md |
| `mvp/harness/` | The benchmark harness: `job.sh` (job blocks), `serve.sh` / `rank.sh` (one case), `bench.py` (client driver), `timed_client.py`, `testhooks.py` (kill and diagnosis hooks), `steady_state.py`, `finals_readout.py`, `ab_stats.py`, `configs/` (serving YAMLs). Jobs run a `/tmp` copy |
| `mvp/runs/` | Run directories `<JOB>-<slurm id>/<case>/`; kept until the finals (CLAIMS cites them) |
| `mvp/analysis/` | Readouts and figure scripts (`slide_figs.py`, `readout_final.txt`, `ctx_stall_cost.md`, ...) |
| `launch/` | Launch files (`*_cmd.txt`: a `bundle.sh` command per job); `std/std-seed2.json` pinned tactics |
| `tools/` | `dry_run.sh` (CPU dry run of a launch file), `presubmit.py` + `require_presubmit.sh`, `units_gate.sh`, `point_units.sh`, `import_smoke.sh`, `cleanup_stale.sh`, `archive_apply.sh`, local `nsys` |
| `bundle.sh`, `build_and_test.sh`, `install_wheel.sh`, `overlay_python.sh`, `restore_build.sh` | Job entry (8-node groups per Slurm job), build + unit tests, per-node wheel install, Python overlay onto a wheel |
| `build-v3-bcea8da496/` | Current wheel (v3); `build-v2-cef4670789/` (v2 finals), `build-pr0-fc52f0adee/` (main, FT-off reference), `build-root/` (incremental build tree) |
| `overlays/` | Python overlays named in current launch files |
| `ompi-ft-ucx/` | ULFM Open MPI the FT launcher uses (`WIDEEP_MPI_PREFIX`) |
| `cache/` | mpi4py, wheelhouse, tokenizers |
| `logs/` | `bundle/` (per-group job logs), `slurm/` (Slurm outputs from 2026-10-21) |
| `dev/`, `wt/` | Git worktrees (`dev`, `wt/reuse`, `wt/sortgate`): kept until the code freeze |
| `probe/` | `ghostmap` probe (cited in HANDOFF) |
| `TRASH/` | Empty; superseded files go here, never deleted by the agents |
| `archive/` | Superseded material, by date |

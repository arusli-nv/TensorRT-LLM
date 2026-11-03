# Final benchmark cases (draft, 2026-10-19; patch: FINAL_CASES.patch vs job.sh md5 8cb31dee, dry-run clean)
Apply: cp job.sh $TMPDIR/j && patch $TMPDIR/j < FINAL_CASES.patch && mv over job.sh (new inode; RETRO 02:25).
Set FINAL_BUILD in job.sh (TODO line 13) = the build of record; every new label's preflight needs MVP_BUILD=FINAL_BUILD.

| Label | Arms / pieces | Est. (QOS 2 h) | Runs |
|---|---|---|---|
| LADDER (L1) | noft, ft-base, ft @8/32/128; tune noft+ft; +1 noft-main tail | ~109 min | 2 rounds (3 if seeded) |
| LADDER-B (L2) | same, FT_COST_E2E_SEED=L1 pinned.json (required) | ~113 min | 3 rounds -> 5/arm total |
| GHOST-COST (G1) | lay-eplb0 vs lay-ghost320, FT off, 32+128, tuned @128 | ~91-112 min | 3-4 rounds |
| GHOST-COST-B (G2) | ft (online288) vs ft-mixed (online288+32 ghost), FT on | ~106 min | 3 rounds |
| FINAL-AUTO (A1) | noft vs ft, tune once per arm (FT-COST-AUTO logic, *-AUTO) | ~97 min | 2 rounds |
| KILL-MATRIX (K1) | pieces: KILL-MTP-R1 (all 6), SHUTDOWN-R1 (ft-clean, ft-loss rc 0), TWOKILL-R1B (2) | ~109 min | 1 each |
| KILL-MATRIX-B (K2) | KILL-FEATURES (5 new, stubs), HANG-SET-C30 (stop-c30, gil45-default-c30), DETECTOR-NEG (host45-node1, if room) | ~111 min | 1 each |
Budgets: FT-COST-E2E-8059717/8064963 (~12.5 min/3-load case), C-COMBINE4-PIN-8066483, KILL-MTP-R1-8059722,
TWOKILL-R1B-8062252, HANG-SET-C30-8063189, DETECTOR-NEG-8061184 (result.json mtimes). Round/piece guards skip work that
does not fit, so no job overruns; the counts above are what fits.

Packing (16 nodes, segment 8, 2:00:00, bundle LIMIT_S 7080): [L1|G1], [L2|G2], [A1|K1], [K2|G1' or G2'].
G' = seeded rerun of a GHOST label (FT_COST_E2E_SEED=its first job's pinned.json): >=6/arm for ONE pair only;
both pairs at >=6 needs a 5th packed job. Ladder: 5/arm unless L1 is seeded from a preflight tuning (then 6).

Still TODO (preflights refuse to run until set):
- FINAL_BUILD (job.sh default build-final-TODO); ft-base: FINAL_BASE_BUILD (nofix wheel) or FINAL_BASE_OVERLAY.
- FINAL_MIXED_CONFIG: configs/ yaml for online 288 + 32 ghost (owner decision 13) - G2 and K2.
- KILL-FEATURES stubs: FINAL_RING_ENV (ring host-copy knob), PHASEB_TOKENS_ARGS (frontend.py streaming-continuation
  mode); per-case PASS/FAIL lines (now KILL_FEATURES_FACTS only, expectations in the block comment).
- presubmit.py checks one --job per token: run it once per group, both tokens in the command.

Submit lines (C=/lustre/.../.wideep-ft-runs/claude; R=$C/mvp/harness/run_job.sh; B=MVP_BUILD=<FINAL_BUILD>):
  python3 tools/presubmit.py --name L1 --nodes 16 --segment 8 --job LADDER --build <FINAL_BUILD>   (-> <TOK_L1>; same for G1 ...)
  [L1|G1] bash /art/tools/require_presubmit.sh <TOK_L1> && bash /art/tools/require_presubmit.sh <TOK_G1> && bash $C/bundle.sh "0:8:7080:$B FINAL_BASE_OVERLAY=<dir> bash $R LADDER" "8:8:7080:$B bash $R GHOST-COST"
  [L2|G2] ... <TOK_L2> <TOK_G2> ... "0:8:7080:$B FINAL_BASE_OVERLAY=<dir> FT_COST_E2E_SEED=$C/mvp/runs/LADDER-<L1 id>/autotuner/pinned.json bash $R LADDER-B" "8:8:7080:$B FINAL_MIXED_CONFIG=<yaml> bash $R GHOST-COST-B"
  [A1|K1] ... <TOK_A1> <TOK_K1> ... "0:8:7080:$B bash $R FINAL-AUTO" "8:8:7080:$B bash $R KILL-MATRIX"
  [K2|G'] ... <TOK_K2> <TOK_Gx> ... "0:8:7080:$B FINAL_MIXED_CONFIG=<yaml> FINAL_RING_ENV=<k=v> PHASEB_TOKENS_ARGS=<args> bash $R KILL-MATRIX-B" "8:8:7080:$B FT_COST_E2E_SEED=<G pinned.json> bash $R GHOST-COST[-B]"

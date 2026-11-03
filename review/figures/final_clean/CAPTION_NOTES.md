# Text moved off the figures, to go into the doc captions (round 14)

## Fig 9 / kill timeline (file fig9_kill_timeline)
- The y axis starts at 35k (marked with an axis break).
- The pre-kill band is ±2 SD of the 12 pre-kill 10 s bins (2 min) per panel, with the SD pooled over the 3 panels (±9.9%).
- 10 s bins of served tokens can exceed the offered load after a backlog.

## Fig A3 / FT check cost
- Row glosses:
  - mask check in stats loop = per-peer check in the expert-statistics loop;
  - release-flag read = read of the released-round flag before the fence;
  - lane mask checks = per-lane live-rank checks;
  - abort-flag poll = abort poll in the round-flag wait;
  - abort OR = block-wide OR of the aborted flag.
- Each check was removed alone, so the savings overlap and are not additive. "All checks" means removed together in the ablation build. The measured FT total compares the FT and no-FT kernels.
- Bars are the mean of 3 launches; launch-to-launch ranges are ≤ 0.04 µs wide.

## Fig A4 / restart vs FT
- Phases are the mean of the n runs.
- Restart = kill → first request answered, excluding the relaunch gap. FT on = kill → first decode step.
- Online model load includes main's host-copy build (CLAIMS 211).

## Fig A5 / remap phases
- These are all 5 kills of the REMAP-TIMER diagnostic job (8308514).
- The whisker is the min–max total over the 31 survivors. Online EPLB ranges, too narrow to draw: kill 1 396.2–397.0 ms, kill 2 399.1–407.3 ms.

## Design figures
- Fig 3: lost experts e16–e23 (rank loss) are restored one per survivor; the tags were removed from the figure.
- Fig 1 and Fig 4: a thicker green outline = added or changed by FT.
- Fig 2 and the recovery sequence: a dashed red outline = lost or failed rank.

## Main charts (from the make_figs.py agent, pending)
- **Fig 5:**
  - change = (FT on − FT off) / FT off of the displayed means;
  - dots = each run, and a line joins each pair.
- **Fig 6:**
  - 95% t-CI (n − 1 df) on the per-pair % differences;
  - CI half-width 0.9–2.1% for tok/s and up to 17.6% for TTFT, so the test is not powered to rule out effects of a few %;
  - 128 requests/GPU is in Appendix F.
- **Fig 7:**
  - n per row:
    - Lose 1 rank: No / Offline / Online EPLB = 4 / 3 / 5;
    - Lose 1 node: 4 / 3 / 4;
    - Frozen rank: No / Online EPLB = 2 / 2;
    - restart (FT off): No / Offline / Online EPLB = 1 / 2 / 2.
  - Filled marker = includes the 30 s hang confirmation (hang_confirm_s); open = without it.
- **Fig 7b:** parts may not sum exactly to the total because of rounding.
- **Fig 8:**
  - before = mean of the full plateaus before the kill;
  - after = the first full plateau starting ≥ 2 s after the resume;
  - proportional = idealized linear scaling with the GPUs left;
  - pts = percentage points of the displayed values;
  - mode shown by outline shade (dark / mid / light green).
- **Fig 10 (labelled Figure 9 in the doc):**
  - p50/p99 per run, then the mean of the 2 runs;
  - change is of the displayed means;
  - "~0%" where the two pairs change in opposite directions;
  - the load is ≈79–81% of the calibrated closed-loop request rate.
- **Fig A1:**
  - upper CI ≥ 1% for No EPLB at 32 and 128 requests/GPU;
  - 95% t-CI with n − 1 df;
  - the 1% threshold was pre-registered for the trim-and-full-mean rule (this plot is trim only).

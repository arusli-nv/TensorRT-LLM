# FT on vs off: step-time distribution shape at 32 requests/GPU (DeepSeek-R1 DEP32, no EPLB, MTP)

Source: `python3 -I mvp/analysis/dist32.py` (uses harness `steady_state.iterations()`; steady steps as in `steady()`:
rank-0, row window, ctx == 0, num_active >= 0.9*load). "trim mean" = mean of steps <= 1.5x median; "stall ms/step" =
summed ms of steps > 1.5x median / steady step count (their contribution to the mean). Client = result.json
`output_throughput`/32. d% = on vs off. Single run per arm, same nodes per pair. STD-8137627 ulfm arm has no rows (skipped).
Tokens/step = 2 x requests (MTP draft included) in every arm, so they cannot differ; accepted tokens are not in the log.
Generated 2026-10-21 PT.

## Table 1. healthy-32, rank-0 full-load decode steps (ms), off -> on

| run | on arm | n steps off/on | row s off/on | median | mean | trim mean (<=1.5x med) | p10 | p25 | p75 | p90 | p99 | >1.5x med | stall ms/step | lag-1 ac | tok/step | client tok/s/GPU | client d% | trim d% | median d% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STD-8141738 | std-noeplb-ft-1 | 7541 / 7616 | 189 / 185 | 13.79 / 14.14 | 14.40 / 14.12 | 13.53 / 13.83 | 11.79 / 12.34 | 12.67 / 12.99 | 14.47 / 14.71 | 15.07 / 14.99 | 16.40 / 16.32 | 0.49% / 0.22% | 0.93 / 0.32 | -0.02 / 0.00 | 64 / 64 | 2944.7 / 3009.8 | +2.2 | +2.2 | +2.5 |
| STD-8139740 | std-noeplb-ft-1 | 7321 / 7592 | 190 / 187 | 13.79 / 14.13 | 14.35 / 14.59 | 13.58 / 13.91 | 11.84 / 12.43 | 12.74 / 13.05 | 14.55 / 14.82 | 15.12 / 15.37 | 16.52 / 16.71 | 0.51% / 0.42% | 0.83 / 0.74 | -0.02 / -0.02 | 64 / 64 | 2919.8 / 2954.8 | +1.2 | +2.4 | +2.5 |
| STD-8139740 | std-noeplb-ulfm-1 | 7321 / 7426 | 190 / 189 | 13.79 / 13.90 | 14.35 / 14.26 | 13.58 / 13.57 | 11.84 / 11.94 | 12.74 / 12.69 | 14.55 / 14.46 | 15.12 / 15.01 | 16.52 / 16.18 | 0.51% / 0.47% | 0.83 / 0.75 | -0.02 / -0.01 | 64 / 64 | 2919.8 / 2917.3 | -0.1 | -0.1 | +0.8 |
| STD-8137627 | std-noeplb-ft-1 | 7612 / 7617 | 182 / 186 | 13.76 / 14.22 | 13.99 / 14.46 | 13.53 / 13.89 | 12.07 / 12.39 | 12.69 / 13.00 | 14.43 / 14.75 | 14.83 / 15.19 | 15.96 / 16.42 | 0.26% / 0.32% | 0.50 / 0.61 | -0.01 / -0.02 | 64 / 64 | 3070.9 / 2990.4 | -2.6 | +2.7 | +3.3 |
| STD-8134758 | std-noeplb-ft-1 | 7472 / 7616 | 184 / 189 | 13.64 / 14.25 | 14.06 / 14.76 | 13.51 / 13.86 | 12.08 / 12.31 | 12.66 / 13.01 | 14.41 / 14.78 | 14.97 / 15.32 | 16.04 / 16.57 | 0.36% / 0.51% | 0.60 / 0.97 | -0.02 / -0.02 | 64 / 64 | 3018.2 / 2925.2 | -3.1 | +2.6 | +4.4 |
| STD-G1-8134758 | std-noeplb-ft-1 | 7546 / 7376 | 182 / 189 | 13.56 / 14.15 | 13.81 / 14.49 | 13.50 / 13.86 | 12.01 / 12.21 | 12.63 / 12.96 | 14.52 / 14.80 | 14.97 / 15.35 | 16.00 / 16.63 | 0.25% / 0.46% | 0.34 / 0.70 | 0.01 / -0.02 | 64 / 64 | 3074.1 / 2932.9 | -4.6 | +2.6 | +4.4 |
| STD-8133265 | std-noeplb-ft-loc-1 | 7533 / 7607 | 183 / 187 | 13.84 / 14.14 | 13.73 / 14.39 | 13.44 / 13.72 | 11.82 / 12.09 | 12.49 / 12.73 | 14.36 / 14.71 | 14.83 / 15.16 | 16.15 / 16.30 | 0.19% / 0.39% | 0.32 / 0.73 | 0.01 / -0.02 | 64 / 64 | 3044.9 / 2977.9 | -2.2 | +2.0 | +2.1 |
| STD-8116111 | std-noeplb-ft-1 | 7569 / 7613 | 184 / 188 | 13.84 / 14.22 | 13.83 / 14.67 | 13.47 / 13.83 | 11.83 / 12.23 | 12.61 / 12.99 | 14.39 / 14.79 | 14.81 / 15.27 | 15.93 / 16.40 | 0.21% / 0.46% | 0.39 / 0.90 | -0.01 / -0.02 | 64 / 64 | 3023.4 / 2965.0 | -1.9 | +2.6 | +2.8 |
| STD-G1-8116111 | std-noeplb-ft-1 | 7604 / 7552 | 185 / 185 | 13.92 / 14.13 | 14.31 / 14.18 | 13.55 / 13.85 | 11.89 / 12.35 | 12.62 / 12.96 | 14.44 / 14.73 | 14.98 / 15.22 | 16.29 / 16.46 | 0.45% / 0.19% | 0.82 / 0.36 | -0.02 / -0.00 | 64 / 64 | 2999.6 / 3010.9 | +0.4 | +2.2 | +1.5 |
| STD-G1-8118134 | std-noeplb-ft-1 | 7542 / 7620 | 187 / 186 | 13.80 / 14.34 | 14.12 / 14.69 | 13.56 / 13.90 | 11.86 / 12.38 | 12.78 / 13.09 | 14.46 / 14.73 | 14.97 / 15.11 | 16.13 / 16.49 | 0.32% / 0.41% | 0.60 / 0.85 | -0.02 / -0.01 | 64 / 64 | 2966.4 / 2978.9 | +0.4 | +2.5 | +4.0 |
| STD-G1-8118573 | std-noeplb-ft-1 | 7563 / 7542 | 186 / 186 | 13.69 / 14.03 | 14.23 / 14.14 | 13.54 / 13.83 | 12.05 / 12.17 | 12.71 / 12.92 | 14.48 / 14.75 | 15.00 / 15.35 | 16.12 / 16.42 | 0.40% / 0.21% | 0.75 / 0.34 | -0.02 / -0.00 | 64 / 64 | 2998.3 / 2985.1 | -0.4 | +2.2 | +2.4 |
| STD-8121996 | std-noeplb-ft-1 | 7606 / 7475 | 182 / 191 | 13.88 / 14.15 | 13.88 / 14.42 | 13.52 / 13.84 | 11.84 / 12.11 | 12.65 / 12.99 | 14.41 / 14.76 | 14.90 / 15.37 | 16.00 / 16.46 | 0.17% / 0.33% | 0.39 / 0.62 | 0.02 / -0.02 | 64 / 64 | 3061.9 / 2908.4 | -5.0 | +2.4 | +2.0 |
| STD-8121996 | std-noeplb-ft-nopdl-1 | 7577 / 7573 | 191 / 190 | 14.21 / 14.40 | 14.76 / 14.72 | 13.97 / 14.18 | 12.50 / 12.57 | 13.06 / 13.34 | 14.86 / 15.12 | 15.43 / 15.55 | 16.73 / 16.70 | 0.50% / 0.30% | 0.86 / 0.59 | -0.03 / -0.02 | 64 / 64 | 2912.7 / 2925.6 | +0.4 | +1.5 | +1.3 |

## Table 2. healthy-32 step-time profile across each request round (on - off, ms; 10 equal position bins of full-load steps)

| run | on arm | b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7 | b8 | b9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| STD-8141738 | std-noeplb-ft-1 | +0.22 | +0.38 | +0.27 | +0.38 | +0.44 | +0.35 | +0.47 | +0.46 | +0.42 | +0.55 |
| STD-8139740 | std-noeplb-ft-1 | +0.41 | +0.30 | +0.55 | +0.29 | +0.16 | +0.19 | +0.35 | +0.39 | +0.23 | +0.23 |
| STD-8139740 | std-noeplb-ulfm-1 | +0.08 | +0.16 | +0.23 | +0.26 | +0.03 | -0.01 | +0.20 | +0.23 | +0.15 | +0.18 |
| STD-8137627 | std-noeplb-ft-1 | +0.49 | +0.50 | +0.44 | +0.39 | +0.39 | +0.44 | +0.37 | +0.43 | +0.38 | +0.41 |
| STD-8134758 | std-noeplb-ft-1 | +0.74 | +0.57 | +0.69 | +0.46 | +0.64 | +0.61 | +0.55 | +0.57 | +0.62 | +0.68 |
| STD-G1-8134758 | std-noeplb-ft-1 | +0.63 | +0.58 | +0.57 | +0.63 | +0.55 | +0.56 | +0.59 | +0.45 | +0.58 | +0.55 |
| STD-8133265 | std-noeplb-ft-loc-1 | +0.20 | +0.29 | +0.26 | +0.32 | +0.22 | +0.28 | +0.22 | +0.25 | +0.27 | +0.28 |
| STD-8116111 | std-noeplb-ft-1 | +0.23 | +0.42 | +0.47 | +0.37 | +0.42 | +0.38 | +0.38 | +0.36 | +0.38 | +0.48 |
| STD-G1-8116111 | std-noeplb-ft-1 | +0.20 | +0.27 | +0.28 | +0.27 | +0.32 | +0.29 | +0.30 | +0.22 | +0.30 | +0.28 |
| STD-G1-8118134 | std-noeplb-ft-1 | +0.54 | +0.51 | +0.59 | +0.58 | +0.56 | +0.57 | +0.59 | +0.63 | +0.55 | +0.59 |
| STD-G1-8118573 | std-noeplb-ft-1 | +0.32 | +0.38 | +0.31 | +0.49 | +0.32 | +0.40 | +0.37 | +0.27 | +0.30 | +0.23 |
| STD-8121996 | std-noeplb-ft-1 | +0.37 | +0.39 | +0.28 | +0.27 | +0.29 | +0.25 | +0.21 | +0.22 | +0.27 | +0.27 |
| STD-8121996 | std-noeplb-ft-nopdl-1 | +0.12 | +0.37 | +0.08 | +0.38 | +0.15 | +0.40 | +0.21 | +0.34 | +0.29 | +0.25 |

## Table 3. FT on vs off at every load (FT arms only; ulfm/nopdl arms excluded from pooling)

| load | pairs | client tok/s/GPU d% mean (sd, #>0) | window tok/s d% | trim-mean step d% (#>0) | median step d% | mean step d% | stall ms/step off/on |
|---|---|---|---|---|---|---|---|
| 8 | 7 | -0.99 (4.52, 2/7) | -1.28 (2.84, 1/7) | -0.21 (0.25, 2/7) | -0.13 (0.11, 1/7) | +0.51 (2.73, 5/7) | 0.71 / 0.80 |
| 32 | 11 | -1.42 (2.36, 4/11) | -1.33 (1.90, 4/11) | +2.41 (0.22, 11/11) | +2.90 (0.98, 11/11) | +2.75 (2.74, 8/11) | 0.59 / 0.65 |
| 64 | 1 | +1.51 (0.00, 1/1) | +0.84 (0.00, 1/1) | +1.19 (0.00, 1/1) | +0.66 (0.00, 1/1) | -3.13 (0.00, 0/1) | 1.21 / 0.33 |
| 128 | 5 | -0.80 (0.56, 0/5) | -0.66 (0.47, 1/5) | +0.43 (0.19, 5/5) | +0.45 (0.16, 5/5) | +0.80 (0.39, 5/5) | 0.40 / 0.50 |
| 512 | 4 | +0.66 (1.12, 3/4) | +0.47 (0.64, 3/4) | -0.74 (0.34, 0/4) | -0.55 (0.34, 0/4) | -0.48 (0.28, 0/4) | 1.00 / 1.18 |

## Table 4. per-pair client tok/s/GPU at every load (off / on, d%)

| run | on arm | load | client off / on | d% | trim step d% |
|---|---|---|---|---|---|
| STD-8139740 | std-noeplb-ft-1 | 8 | 1103.9 / 1072.0 | -2.9 | -0.2 |
| STD-8139740 | std-noeplb-ulfm-1 | 8 | 1103.9 / 1078.3 | -2.3 | -0.4 |
| STD-8137627 | std-noeplb-ft-1 | 8 | 1070.7 / 1157.4 | +8.1 | +0.1 |
| STD-8137627 | std-noeplb-ft-1 | 64 | 4015.2 / 4075.7 | +1.5 | +1.2 |
| STD-8137627 | std-noeplb-ft-1 | 128 | 4916.5 / 4901.4 | -0.3 | +0.6 |
| STD-8133265 | std-noeplb-ft-loc-1 | 8 | 1154.3 / 1154.5 | +0.0 | -0.6 |
| STD-8116111 | std-noeplb-ft-1 | 8 | 1103.0 / 1051.1 | -4.7 | -0.2 |
| STD-8116111 | std-noeplb-ft-1 | 128 | 4937.9 / 4913.7 | -0.5 | +0.2 |
| STD-8116111 | std-noeplb-ft-1 | 512 | 5689.1 / 5820.0 | +2.3 | -1.1 |
| STD-G1-8116111 | std-noeplb-ft-1 | 8 | 1126.3 / 1120.0 | -0.6 | -0.2 |
| STD-G1-8116111 | std-noeplb-ft-1 | 128 | 4943.0 / 4894.8 | -1.0 | +0.4 |
| STD-G1-8116111 | std-noeplb-ft-1 | 512 | 5814.7 / 5800.1 | -0.3 | -0.8 |
| STD-G1-8118134 | std-noeplb-ft-1 | 8 | 1076.4 / 1063.0 | -1.2 | +0.1 |
| STD-G1-8118134 | std-noeplb-ft-1 | 128 | 4921.3 / 4837.6 | -1.7 | +0.3 |
| STD-G1-8118134 | std-noeplb-ft-1 | 512 | 5817.1 / 5835.2 | +0.3 | -0.8 |
| STD-G1-8118573 | std-noeplb-ft-1 | 8 | 1140.1 / 1076.0 | -5.6 | -0.3 |
| STD-G1-8118573 | std-noeplb-ft-1 | 128 | 4902.7 / 4875.7 | -0.6 | +0.6 |
| STD-G1-8118573 | std-noeplb-ft-1 | 512 | 5777.5 / 5794.4 | +0.3 | -0.3 |

## Findings

- Shape: both arms have the same two-mode distribution (modes near 12.5-13 ms and 14.5-15 ms, trough at 13.5-14 ms).
  Lag-1 autocorrelation is about 0 in every arm, so steps do not alternate fast/slow. FT moves both modes right. p10..p90
  each rise 1.6-3% and the round-position profile (Table 2) is up by a near-constant 0.2-0.7 ms in every bin of every pair.
  This is an added per-step cost, not a narrower distribution.
- The median lies in the trough, so a small shift moves it a lot: median +2.9% (sd 0.98) against trim mean +2.4% (sd 0.22, 11/11).
- Mean and client are driven by rare stalls: 0.2-0.8% of steps at > 1.5x median add 0.3-1.0 ms to every step's mean
  (2-7%). Which arm gets them looks random (FT 0.65 vs off 0.59 ms/step at 32). In NSYS32 (STD-8141738) the off arm had 0.93 vs 0.32
  ms/step of stalls. That gap alone flips its mean (-1.9%) and client (+2.2%), even though its trim mean is +2.2% like every other pair.
- Steady steps are about 60% of the healthy-32 window's step time. So a 2.4% steady cost predicts about 1.4% client loss, and the
  measured value is -1.4% (sd 2.4, 11 pairs).

## Conclusion

At 32/GPU, FT adds a real, near-constant cost of about 0.35 ms per decode step: trim mean +2.4% (sd 0.22, 11/11 pairs),
with the same shift at every point in the round. It does not change the distribution's shape. NSYS32's lower mean and higher
client throughput come from one arm's random stalls, not from FT. Users see this cost diluted by context and ramp time: client
tok/s/GPU is -1.4% (sd 2.4, 11 pairs) and whole-window tok/s is -1.3%. Both are consistent with the 2.4% steady cost on about 60% of
the window. FT therefore has a real cost at 32/GPU that is above the 1% budget.

Proposed pre-registered headline: client output tok/s/GPU on (FT on vs off) for each load. Report it as the mean of
same-node pair deltas with its SE, because a single pair has sd 2-4.5% from random stalls. Pair it with the rank-0 steady
trimmed-mean step delta (sd 0.2-0.3%) as the low-noise diagnostic that explains it. Keep the median and plain mean as
diagnostics only: the median sits in the trough between two modes and overstates the shift, and the mean is driven by stalls.

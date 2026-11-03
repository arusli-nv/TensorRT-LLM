# FT healthy-cost finals readout
runs: STD-8116111, STD-G1-8116111; groups: 2; cost % = (FT on - off) / off, positive = FT slower

## 1. Inclusion

| group | case | PIN_CHECK | autotune | startup ok | healthy rows | excluded |
|---|---|---|---|---|---|---|
| STD-8116111 | std-noeplb-1 | OK | 0 | yes | 8,32,128,512 | no |
| STD-8116111 | std-noeplb-ft-1 | OK | 0 | yes | 8,32,128,512 | no |
| STD-8116111 | std-offline-1 | OK | 0 | yes | 8,32,128,512 | no |
| STD-8116111 | std-offline-ft-1 | OK | 0 | yes | 8,32,128,512 | no |
| STD-G1-8116111 | std-noeplb-1 | OK | 0 | yes | 8,32,128,512 | no |
| STD-G1-8116111 | std-noeplb-ft-1 | OK | 0 | yes | 8,32,128,512 | no |
| STD-G1-8116111 | std-online-1 | OK | 0 | yes | 8,32,128,512 | no |
| STD-G1-8116111 | std-online-ft-1 | OK | 0 | yes | 8,32,128,512 | no |

## 2. Primary: rank-0 host loop-step median, FT on vs off (group-paired)

| mode | load | n | per-group cost % | mean % | sd | one-sided UB95 % | pass (<+1%) | two-sided 95% CI | delta us | off step ms | on step ms | off tok/s/GPU | on tok/s/GPU | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noeplb | 8 | 2 | -0.01 -0.19 | -0.10 | 0.12 | +0.46 | PASS | [-1.22, +1.02] | -12 | 11.900 | 11.888 | 1345 | 1346 | 0.229 |
| noeplb | 32 | 2 | +2.76 +1.53 | +2.14 | 0.87 | +6.01 | fail | [-5.64, +9.93] | +297 | 13.880 | 14.177 | 4611 | 4514 | 1.000 |
| noeplb | 128 | 2 | +0.20 +0.54 | +0.37 | 0.24 | +1.44 | fail | [-1.78, +2.52] | +92 | 24.643 | 24.734 | 10389 | 10350 | 0.669 |
| noeplb | 512 | 2 | -0.96 -0.41 | -0.68 | 0.39 | +1.06 | not tested | [-4.20, +2.83] | -487 | 71.439 | 70.952 | 14335 | 14434 | - |
| offline | 8 | 1 | -0.20 | -0.20 | - | - | n<2 | [-, -] | -25 | 12.468 | 12.444 | 1283 | 1286 | 1.000 |
| offline | 32 | 1 | +1.02 | +1.02 | - | - | n<2 | [-, -] | +144 | 14.197 | 14.341 | 4508 | 4463 | 1.000 |
| offline | 128 | 1 | -0.14 | -0.14 | - | - | n<2 | [-, -] | -33 | 24.285 | 24.252 | 10541 | 10556 | 1.000 |
| offline | 512 | 1 | -0.18 | -0.18 | - | - | not tested | [-, -] | -126 | 68.651 | 68.525 | 14916 | 14943 | - |
| online | 8 | 1 | +0.63 | +0.63 | - | - | n<2 | [-, -] | +85 | 13.378 | 13.463 | 1196 | 1188 | 1.000 |
| online | 32 | 1 | +0.40 | +0.40 | - | - | n<2 | [-, -] | +57 | 14.482 | 14.539 | 4419 | 4402 | 1.000 |
| online | 128 | 1 | -0.22 | -0.22 | - | - | n<2 | [-, -] | -49 | 21.999 | 21.951 | 11637 | 11662 | 1.000 |
| online | 512 | 1 | +1.44 | +1.44 | - | - | not tested | [-, -] | +871 | 60.561 | 61.432 | 16908 | 16669 | - |

Every-cell claim (IUT, all 9 headline cells UB95 < +1%): n<2 in some cell. Holm (9 cells, H0 cost >= 1%) established < 1%: none.

Load attainment over the row's decode steps: share of steps with num_active >= 0.9 x load (and share of decode time), median num_active; per arm:

- noeplb/512 std-noeplb: STD-8116111 43.7% of steps (75.5% of time), median active 129; STD-G1-8116111 50.6% of steps (80.6% of time), median active 500
- noeplb/512 std-noeplb-ft: STD-8116111 44.9% of steps (73.0% of time), median active 379; STD-G1-8116111 52.4% of steps (81.1% of time), median active 512
- offline/512 std-offline: STD-8116111 52.0% of steps (80.5% of time), median active 512
- offline/512 std-offline-ft: STD-8116111 50.9% of steps (79.2% of time), median active 511
- online/512 std-online: STD-G1-8116111 48.0% of steps (76.2% of time), median active 326
- online/512 std-online-ft: STD-G1-8116111 48.3% of steps (76.6% of time), median active 349

CLAIMS-ready lines:
- noeplb/8/GPU: FT cost -0.10% [one-sided 95% UB +0.46%], n=2 groups (STD-8116111, STD-G1-8116111), rank-0 host loop-step median, decode-only num_active >= 0.9 load
- noeplb/32/GPU: FT cost +2.14% [one-sided 95% UB +6.01%], n=2 groups (STD-8116111, STD-G1-8116111), rank-0 host loop-step median, decode-only num_active >= 0.9 load
- noeplb/128/GPU: FT cost +0.37% [one-sided 95% UB +1.44%], n=2 groups (STD-8116111, STD-G1-8116111), rank-0 host loop-step median, decode-only num_active >= 0.9 load
- noeplb/512/GPU: FT cost -0.68% [one-sided 95% UB +1.06%], n=2 groups (STD-8116111, STD-G1-8116111), rank-0 host loop-step median, decode-only num_active >= 0.9 load, beyond recipe, not tested
- offline/8/GPU: FT cost -0.20% [one-sided 95% UB -%], n=1 groups (STD-8116111), rank-0 host loop-step median, decode-only num_active >= 0.9 load
- offline/32/GPU: FT cost +1.02% [one-sided 95% UB -%], n=1 groups (STD-8116111), rank-0 host loop-step median, decode-only num_active >= 0.9 load
- offline/128/GPU: FT cost -0.14% [one-sided 95% UB -%], n=1 groups (STD-8116111), rank-0 host loop-step median, decode-only num_active >= 0.9 load
- offline/512/GPU: FT cost -0.18% [one-sided 95% UB -%], n=1 groups (STD-8116111), rank-0 host loop-step median, decode-only num_active >= 0.9 load, beyond recipe, not tested
- online/8/GPU: FT cost +0.63% [one-sided 95% UB -%], n=1 groups (STD-G1-8116111), rank-0 host loop-step median, decode-only num_active >= 0.9 load
- online/32/GPU: FT cost +0.40% [one-sided 95% UB -%], n=1 groups (STD-G1-8116111), rank-0 host loop-step median, decode-only num_active >= 0.9 load
- online/128/GPU: FT cost -0.22% [one-sided 95% UB -%], n=1 groups (STD-G1-8116111), rank-0 host loop-step median, decode-only num_active >= 0.9 load
- online/512/GPU: FT cost +1.44% [one-sided 95% UB -%], n=1 groups (STD-G1-8116111), rank-0 host loop-step median, decode-only num_active >= 0.9 load, beyond recipe, not tested

## 3. Client (descriptive; rows with > 5% failed/short requests dropped): median over groups [min-max]

| mode | load | arm | n | TPOT median ms | TPOT mean ms | out tok/s/GPU |
|---|---|---|---|---|---|---|
| noeplb | 8 | std-noeplb | 2 | 6.40 [6.27-6.54] | 6.41 [6.33-6.49] | 1114.65 [1102.99-1126.31] |
| noeplb | 8 | std-noeplb-ft | 2 | 6.64 [6.40-6.87] | 6.58 [6.39-6.78] | 1085.56 [1051.11-1120.00] |
| noeplb | 32 | std-noeplb | 2 | 8.53 [8.43-8.64] | 8.59 [8.51-8.66] | 3011.48 [2999.61-3023.36] |
| noeplb | 32 | std-noeplb-ft | 2 | 8.53 [8.47-8.58] | 8.65 [8.61-8.68] | 2987.98 [2965.04-3010.93] |
| noeplb | 128 | std-noeplb | 2 | 19.18 [19.13-19.24] | 19.11 [19.09-19.13] | 4940.46 [4937.95-4942.97] |
| noeplb | 128 | std-noeplb-ft | 2 | 19.38 [19.37-19.38] | 19.39 [19.31-19.46] | 4904.24 [4894.81-4913.66] |
| noeplb | 512 | std-noeplb | 2 | 62.71 [61.93-63.49] | 62.07 [61.79-62.36] | 5751.87 [5689.06-5814.68] |
| noeplb | 512 | std-noeplb-ft | 2 | 62.17 [62.12-62.21] | 61.76 [61.63-61.89] | 5810.01 [5800.06-5819.95] |
| offline | 8 | std-offline | 1 | 6.49 [6.49-6.49] | 6.45 [6.45-6.45] | 1152.79 [1152.79-1152.79] |
| offline | 8 | std-offline-ft | 1 | 6.50 [6.50-6.50] | 6.51 [6.51-6.51] | 1095.97 [1095.97-1095.97] |
| offline | 32 | std-offline | 1 | 8.06 [8.06-8.06] | 8.11 [8.11-8.11] | 3366.14 [3366.14-3366.14] |
| offline | 32 | std-offline-ft | 1 | 8.24 [8.24-8.24] | 8.29 [8.29-8.29] | 3308.92 [3308.92-3308.92] |
| offline | 128 | std-offline | 1 | 16.49 [16.49-16.49] | 16.46 [16.46-16.46] | 6303.57 [6303.57-6303.57] |
| offline | 128 | std-offline-ft | 1 | 16.72 [16.72-16.72] | 16.57 [16.57-16.57] | 6351.52 [6351.52-6351.52] |
| offline | 512 | std-offline | 1 | 49.85 [49.85-49.85] | 49.39 [49.39-49.39] | 7963.61 [7963.61-7963.61] |
| offline | 512 | std-offline-ft | 1 | 50.13 [50.13-50.13] | 49.70 [49.70-49.70] | 7972.64 [7972.64-7972.64] |
| online | 8 | std-online | 1 | 7.43 [7.43-7.43] | 7.38 [7.38-7.38] | 958.64 [958.64-958.64] |
| online | 8 | std-online-ft | 1 | 7.32 [7.32-7.32] | 7.30 [7.30-7.30] | 992.94 [992.94-992.94] |
| online | 32 | std-online | 1 | 8.80 [8.80-8.80] | 8.91 [8.91-8.91] | 3012.03 [3012.03-3012.03] |
| online | 32 | std-online-ft | 1 | 8.84 [8.84-8.84] | 8.91 [8.91-8.91] | 2933.68 [2933.68-2933.68] |
| online | 128 | std-online | 1 | 16.95 [16.95-16.95] | 17.02 [17.02-17.02] | 5672.69 [5672.69-5672.69] |
| online | 128 | std-online-ft | 1 | 17.54 [17.54-17.54] | 17.48 [17.48-17.48] | 5432.17 [5432.17-5432.17] |
| online | 512 | std-online | 1 | 48.22 [48.22-48.22] | 48.68 [48.68-48.68] | 7452.84 [7452.84-7452.84] |
| online | 512 | std-online-ft | 1 | 48.57 [48.57-48.57] | 48.79 [48.79-48.79] | 7446.52 [7446.52-7446.52] |

## 4. Stall share (% of steady time in steps > 2x the row median), FT on - off in points

Rule: Hodges-Lehmann <= +0.5 points and no single pair > +2 points. Wilcoxon p descriptive (floor 1/2^n).

| mode | load | n | off % per group | on % per group | HL | max pair | rule | Wilcoxon p (floor) |
|---|---|---|---|---|---|---|---|---|
| noeplb | 8 | 2 | 7.9 4.4 | 12.3 5.6 | +2.81 | +4.40 | fail | 0.250 (0.250) |
| noeplb | 32 | 2 | 2.8 5.7 | 6.2 2.5 | +0.07 | +3.34 | fail | 0.500 (0.250) |
| noeplb | 128 | 2 | 1.6 1.1 | 2.2 2.2 | +0.83 | +1.08 | fail | 0.250 (0.250) |
| noeplb | 512 | 2 | 1.5 1.0 | 2.3 1.4 | +0.59 | +0.73 | fail | 0.250 (0.250) |
| offline | 8 | 1 | 0.8 | 1.9 | +1.05 | +1.05 | fail | 0.500 (0.500) |
| offline | 32 | 1 | 3.1 | 3.3 | +0.18 | +0.18 | PASS | 0.500 (0.500) |
| offline | 128 | 1 | 0.5 | 1.3 | +0.77 | +0.77 | fail | 0.500 (0.500) |
| offline | 512 | 1 | 0.7 | 0.9 | +0.24 | +0.24 | PASS | 0.500 (0.500) |
| online | 8 | 1 | 6.7 | 5.0 | -1.66 | -1.66 | PASS | 1.000 (0.500) |
| online | 32 | 1 | 3.5 | 3.6 | +0.12 | +0.12 | PASS | 0.500 (0.500) |
| online | 128 | 1 | 1.2 | 0.7 | -0.47 | -0.47 | PASS | 1.000 (0.500) |
| online | 512 | 1 | 0.8 | 0.3 | -0.48 | -0.48 | PASS | 1.000 (0.500) |

## 5. 32/GPU cost within KV-fill bins (5 quantile bins of both arms' cached_kv_tokens, weighted by min steps per bin)

| mode | metric | per-group binned cost % | n | mean % | one-sided UB95 % | two-sided 95% CI |
|---|---|---|---|---|---|---|
| noeplb | host loop-step | STD-8116111:+2.70 STD-G1-8116111:+1.87 | 2 | +2.29 | +4.91 | [-2.99, +7.56] |
| noeplb | device step | STD-8116111:+2.45 STD-G1-8116111:+2.27 | 2 | +2.36 | +2.93 | [+1.21, +3.50] |
| offline | host loop-step | STD-8116111:+1.10 | 1 | +1.10 | - | [-, -] |
| offline | device step | STD-8116111:+0.90 | 1 | +0.90 | - | [-, -] |
| online | host loop-step | STD-G1-8116111:+0.58 | 1 | +0.58 | - | [-, -] |
| online | device step | STD-G1-8116111:+0.11 | 1 | +0.11 | - | [-, -] |

## 6. Sensitivity

Job-averaged: the groups of one job averaged first, t over jobs. Slow-mode launch: 8/GPU step median > 1.05 x the median of its mode's FT-off cases; its group pair is dropped for that mode.

- slow-mode launches noeplb: none
- slow-mode launches offline: none
- slow-mode launches online: none

| mode | load | variant | n | mean % | one-sided UB95 % | pass | agrees with primary |
|---|---|---|---|---|---|---|---|
| noeplb | 8 | job-averaged | 1 | -0.10 | - | n<2 | - |
| noeplb | 8 | without slow-mode | 2 | -0.10 | +0.46 | PASS | yes |
| noeplb | 32 | job-averaged | 1 | +2.14 | - | n<2 | - |
| noeplb | 32 | without slow-mode | 2 | +2.14 | +6.01 | fail | yes |
| noeplb | 128 | job-averaged | 1 | +0.37 | - | n<2 | - |
| noeplb | 128 | without slow-mode | 2 | +0.37 | +1.44 | fail | yes |
| noeplb | 512 | job-averaged | 1 | -0.68 | - | not tested | - |
| noeplb | 512 | without slow-mode | 2 | -0.68 | +1.06 | not tested | - |
| offline | 8 | job-averaged | 1 | -0.20 | - | n<2 | - |
| offline | 8 | without slow-mode | 1 | -0.20 | - | n<2 | - |
| offline | 32 | job-averaged | 1 | +1.02 | - | n<2 | - |
| offline | 32 | without slow-mode | 1 | +1.02 | - | n<2 | - |
| offline | 128 | job-averaged | 1 | -0.14 | - | n<2 | - |
| offline | 128 | without slow-mode | 1 | -0.14 | - | n<2 | - |
| offline | 512 | job-averaged | 1 | -0.18 | - | not tested | - |
| offline | 512 | without slow-mode | 1 | -0.18 | - | not tested | - |
| online | 8 | job-averaged | 1 | +0.63 | - | n<2 | - |
| online | 8 | without slow-mode | 1 | +0.63 | - | n<2 | - |
| online | 32 | job-averaged | 1 | +0.40 | - | n<2 | - |
| online | 32 | without slow-mode | 1 | +0.40 | - | n<2 | - |
| online | 128 | job-averaged | 1 | -0.22 | - | n<2 | - |
| online | 128 | without slow-mode | 1 | -0.22 | - | n<2 | - |
| online | 512 | job-averaged | 1 | +1.44 | - | not tested | - |
| online | 512 | without slow-mode | 1 | +1.44 | - | not tested | - |

## 7. Static-LB control (std-noeplb vs std-noeplb-lb vs std-noeplb-ft), host loop-step median

| group | load | noeplb ms | lb ms | ft ms | lb vs noeplb % | ft vs lb % | ft vs noeplb % |
|---|---|---|---|---|---|---|---|
| (no group with std-noeplb-lb) |||||||| 

json: /tmp/claude-161238/fr_f1.json

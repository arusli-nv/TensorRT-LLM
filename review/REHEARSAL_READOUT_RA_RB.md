## Validity (PRE-REG 5/6, decided before any cost)

Window check: >= 60 s; Xid/NVRM checked on all 8 nodes in 10 of 10 arms.

- all pairs valid
- note STD-8193512 std-noeplb-ft-1: 60 fault lines after the last row (server teardown; not a healthy-window fault)
- note STD-8194123 std-offline-ft-1: 57 fault lines after the last row (server teardown; not a healthy-window fault)
- note STD-G1-8193512 std-online-ft-1: 58 fault lines after the last row (server teardown; not a healthy-window fault)
- note STD-G2-8193512 std-noeplb-ulfm-1: 56 fault lines after the last row (server teardown; not a healthy-window fault)

## Pairs (% = (on - off) / off; step metrics + = FT slower, tok/s - = FT slower)

| group | first node | mode | load | first | rounds | trim % | full % | stall % | stall ms diff | client % | window tok/s % | trim off/on ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8193512@n0 | n0 | noeplb | 8 | off | 1 | +0.34 | -0.46 | -11.38 | -0.103 | +0.29 | +0.42 | 11.365/11.404 |
| 8193512@n0 | n0 | noeplb | 32 | off | 1 | +0.86 | +2.52 | +103.50 | +0.248 | -0.99 | -1.21 | 13.825/13.944 |
| 8193512@n0 | n0 | noeplb | 128 | off | 1 | +0.34 | +0.63 | +36.17 | +0.081 | +0.98 | +0.96 | 27.043/27.134 |
| 8193512@n0 | n0 | noeplb | 512 | off | 1 | -0.13 | -1.18 | -80.05 | -1.204 | +0.68 | -0.12 | 79.344/79.244 |
| 8193512@n8 | n8 | online | 8 | off | 1 | +0.99 | -0.26 | -34.75 | -0.176 | -0.16 | +0.00 | 12.856/12.984 |
| 8193512@n8 | n8 | online | 32 | off | 1 | +0.67 | +1.75 | +69.21 | +0.176 | -2.39 | -2.62 | 14.562/14.659 |
| 8193512@n8 | n8 | online | 128 | off | 1 | -0.02 | -1.09 | -84.63 | -0.281 | +1.25 | +1.07 | 24.572/24.566 |
| 8193512@n8 | n8 | online | 512 | off | 1 | +0.57 | +0.56 | -2.69 | -0.008 | +1.60 | +1.05 | 67.316/67.699 |
| 8194123@n0 | n0 | offline | 8 | off | 1 | +0.10 | -0.41 | -11.81 | -0.064 | +1.22 | +0.90 | 11.933/11.945 |
| 8194123@n0 | n0 | offline | 32 | off | 1 | +0.21 | +1.49 | +294.17 | +0.199 | +0.35 | -0.19 | 14.120/14.151 |
| 8194123@n0 | n0 | offline | 128 | off | 1 | +0.35 | +0.31 | -100.00 | -0.027 | -1.34 | -1.16 | 26.411/26.503 |
| 8194123@n0 | n0 | offline | 512 | off | 1 | +0.20 | +0.21 | +3.61 | +0.012 | -0.59 | -0.01 | 77.504/77.656 |

## Cells (mean % over pairs, 95% t-CI, one-sided 95% bounds)

| mode | load | metric | n | mean % | 95% CI | one-sided 95% LB | one-sided 95% UB |
|---|---|---|---|---|---|---|---|
| noeplb | 8 (appendix) | trim mean step | 1 | +0.34 | [-, -] | - | - |
| noeplb | 8 (appendix) | full mean step | 1 | -0.46 | [-, -] | - | - |
| noeplb | 8 (appendix) | stall ms/step | 1 | -11.38 | [-, -] | - | - |
| noeplb | 8 (appendix) | client tok/s/GPU | 1 | +0.29 | [-, -] | - | - |
| noeplb | 8 (appendix) | rank-0 window tok/s/GPU | 1 | +0.42 | [-, -] | - | - |
| noeplb | 32 | trim mean step | 1 | +0.86 | [-, -] | - | - |
| noeplb | 32 | full mean step | 1 | +2.52 | [-, -] | - | - |
| noeplb | 32 | stall ms/step | 1 | +103.50 | [-, -] | - | - |
| noeplb | 32 | client tok/s/GPU | 1 | -0.99 | [-, -] | - | - |
| noeplb | 32 | rank-0 window tok/s/GPU | 1 | -1.21 | [-, -] | - | - |
| noeplb | 128 | trim mean step | 1 | +0.34 | [-, -] | - | - |
| noeplb | 128 | full mean step | 1 | +0.63 | [-, -] | - | - |
| noeplb | 128 | stall ms/step | 1 | +36.17 | [-, -] | - | - |
| noeplb | 128 | client tok/s/GPU | 1 | +0.98 | [-, -] | - | - |
| noeplb | 128 | rank-0 window tok/s/GPU | 1 | +0.96 | [-, -] | - | - |
| noeplb | 512 (appendix) | trim mean step | 1 | -0.13 | [-, -] | - | - |
| noeplb | 512 (appendix) | full mean step | 1 | -1.18 | [-, -] | - | - |
| noeplb | 512 (appendix) | stall ms/step | 1 | -80.05 | [-, -] | - | - |
| noeplb | 512 (appendix) | client tok/s/GPU | 1 | +0.68 | [-, -] | - | - |
| noeplb | 512 (appendix) | rank-0 window tok/s/GPU | 1 | -0.12 | [-, -] | - | - |
| offline | 8 (appendix) | trim mean step | 1 | +0.10 | [-, -] | - | - |
| offline | 8 (appendix) | full mean step | 1 | -0.41 | [-, -] | - | - |
| offline | 8 (appendix) | stall ms/step | 1 | -11.81 | [-, -] | - | - |
| offline | 8 (appendix) | client tok/s/GPU | 1 | +1.22 | [-, -] | - | - |
| offline | 8 (appendix) | rank-0 window tok/s/GPU | 1 | +0.90 | [-, -] | - | - |
| offline | 32 | trim mean step | 1 | +0.21 | [-, -] | - | - |
| offline | 32 | full mean step | 1 | +1.49 | [-, -] | - | - |
| offline | 32 | stall ms/step | 1 | +294.17 | [-, -] | - | - |
| offline | 32 | client tok/s/GPU | 1 | +0.35 | [-, -] | - | - |
| offline | 32 | rank-0 window tok/s/GPU | 1 | -0.19 | [-, -] | - | - |
| offline | 128 | trim mean step | 1 | +0.35 | [-, -] | - | - |
| offline | 128 | full mean step | 1 | +0.31 | [-, -] | - | - |
| offline | 128 | stall ms/step | 1 | -100.00 | [-, -] | - | - |
| offline | 128 | client tok/s/GPU | 1 | -1.34 | [-, -] | - | - |
| offline | 128 | rank-0 window tok/s/GPU | 1 | -1.16 | [-, -] | - | - |
| offline | 512 (appendix) | trim mean step | 1 | +0.20 | [-, -] | - | - |
| offline | 512 (appendix) | full mean step | 1 | +0.21 | [-, -] | - | - |
| offline | 512 (appendix) | stall ms/step | 1 | +3.61 | [-, -] | - | - |
| offline | 512 (appendix) | client tok/s/GPU | 1 | -0.59 | [-, -] | - | - |
| offline | 512 (appendix) | rank-0 window tok/s/GPU | 1 | -0.01 | [-, -] | - | - |
| online | 8 (appendix) | trim mean step | 1 | +0.99 | [-, -] | - | - |
| online | 8 (appendix) | full mean step | 1 | -0.26 | [-, -] | - | - |
| online | 8 (appendix) | stall ms/step | 1 | -34.75 | [-, -] | - | - |
| online | 8 (appendix) | client tok/s/GPU | 1 | -0.16 | [-, -] | - | - |
| online | 8 (appendix) | rank-0 window tok/s/GPU | 1 | +0.00 | [-, -] | - | - |
| online | 32 | trim mean step | 1 | +0.67 | [-, -] | - | - |
| online | 32 | full mean step | 1 | +1.75 | [-, -] | - | - |
| online | 32 | stall ms/step | 1 | +69.21 | [-, -] | - | - |
| online | 32 | client tok/s/GPU | 1 | -2.39 | [-, -] | - | - |
| online | 32 | rank-0 window tok/s/GPU | 1 | -2.62 | [-, -] | - | - |
| online | 128 | trim mean step | 1 | -0.02 | [-, -] | - | - |
| online | 128 | full mean step | 1 | -1.09 | [-, -] | - | - |
| online | 128 | stall ms/step | 1 | -84.63 | [-, -] | - | - |
| online | 128 | client tok/s/GPU | 1 | +1.25 | [-, -] | - | - |
| online | 128 | rank-0 window tok/s/GPU | 1 | +1.07 | [-, -] | - | - |
| online | 512 (appendix) | trim mean step | 1 | +0.57 | [-, -] | - | - |
| online | 512 (appendix) | full mean step | 1 | +0.56 | [-, -] | - | - |
| online | 512 (appendix) | stall ms/step | 1 | -2.69 | [-, -] | - | - |
| online | 512 (appendix) | client tok/s/GPU | 1 | +1.60 | [-, -] | - | - |
| online | 512 (appendix) | rank-0 window tok/s/GPU | 1 | +1.05 | [-, -] | - | - |

## Per mode (headline loads [32, 128] averaged within each pair first; modes not pooled)

| mode | metric | n | mean % | 95% CI | one-sided 95% LB | one-sided 95% UB |
|---|---|---|---|---|---|---|
| noeplb | trim mean step | 1 | +0.60 | [-, -] | - | - |
| noeplb | full mean step | 1 | +1.58 | [-, -] | - | - |
| noeplb | stall ms/step | 1 | +69.83 | [-, -] | - | - |
| noeplb | client tok/s/GPU | 1 | -0.00 | [-, -] | - | - |
| noeplb | rank-0 window tok/s/GPU | 1 | -0.12 | [-, -] | - | - |
| offline | trim mean step | 1 | +0.28 | [-, -] | - | - |
| offline | full mean step | 1 | +0.90 | [-, -] | - | - |
| offline | stall ms/step | 1 | +97.08 | [-, -] | - | - |
| offline | client tok/s/GPU | 1 | -0.49 | [-, -] | - | - |
| offline | rank-0 window tok/s/GPU | 1 | -0.68 | [-, -] | - | - |
| online | trim mean step | 1 | +0.32 | [-, -] | - | - |
| online | full mean step | 1 | +0.33 | [-, -] | - | - |
| online | stall ms/step | 1 | -7.71 | [-, -] | - | - |
| online | client tok/s/GPU | 1 | -0.57 | [-, -] | - | - |
| online | rank-0 window tok/s/GPU | 1 | -0.77 | [-, -] | - | - |

## Claim (PRE-REG 3, IUT over every mode x [32, 128] cell)

- cells missing or n < 2 (no bound, so no claim): [('noeplb', 32), ('noeplb', 128), ('offline', 32), ('offline', 128), ('online', 32), ('online', 128)]
- trim mean: every upper bound < 1.0%: no; full mean: no
- rule A (pre-registered: trim AND full mean): per-step 'FT cost < 1%' NOT claimed
- rule B (trim mean only; added 10-23 after the rehearsal's n=1 reading): per-step 'FT cost < 1%' NOT claimed
- client tok/s: estimate +/- CI only (no < 1% claim at this n)

## Checks only (PRE-REG 7: not used to adjust)

| mode | load | metric | off first: n, mean % | ft first: n, mean % | n0: n, mean % | n8: n, mean % |
|---|---|---|---|---|---|---|
| noeplb | 32 | trim | 1, +0.86 | 0, - | 1, +0.86 | 0, - |
| noeplb | 32 | full | 1, +2.52 | 0, - | 1, +2.52 | 0, - |
| noeplb | 128 | trim | 1, +0.34 | 0, - | 1, +0.34 | 0, - |
| noeplb | 128 | full | 1, +0.63 | 0, - | 1, +0.63 | 0, - |
| offline | 32 | trim | 1, +0.21 | 0, - | 1, +0.21 | 0, - |
| offline | 32 | full | 1, +1.49 | 0, - | 1, +1.49 | 0, - |
| offline | 128 | trim | 1, +0.35 | 0, - | 1, +0.35 | 0, - |
| offline | 128 | full | 1, +0.31 | 0, - | 1, +0.31 | 0, - |
| online | 32 | trim | 1, +0.67 | 0, - | 0, - | 1, +0.67 |
| online | 32 | full | 1, +1.75 | 0, - | 0, - | 1, +1.75 |
| online | 128 | trim | 1, -0.02 | 0, - | 0, - | 1, -0.02 |
| online | 128 | full | 1, -1.09 | 0, - | 0, - | 1, -1.09 |

## Descriptive (PRE-REG 8, not in the claim): main -> our FT off (parity); % = (second - first) / first


| group | mode | load | rounds | trim % | full % | client % | trim ms first/second |
|---|---|---|---|---|---|---|---|
| 8193512@n0 | noeplb | 32 | 1 | -0.03 | +0.05 | +1.13 | 13.829/13.825 |
| 8193512@n0 | noeplb | 128 | 1 | -0.16 | -0.48 | -0.11 | 27.086/27.043 |
| 8193512@n8 | online | 32 | 1 | -0.24 | -1.04 | +2.25 | 14.598/14.562 |
| 8193512@n8 | online | 128 | 1 | +0.17 | +0.60 | +0.15 | 24.530/24.572 |

| mode | load | metric | n | mean % | 95% CI | one-sided 95% LB | one-sided 95% UB |
|---|---|---|---|---|---|---|---|
| noeplb | 32 | trim mean step | 1 | -0.03 | [-, -] | - | - |
| noeplb | 32 | full mean step | 1 | +0.05 | [-, -] | - | - |
| noeplb | 32 | stall ms/step | 1 | +6.98 | [-, -] | - | - |
| noeplb | 32 | client tok/s/GPU | 1 | +1.13 | [-, -] | - | - |
| noeplb | 32 | rank-0 window tok/s/GPU | 1 | +1.02 | [-, -] | - | - |
| noeplb | 128 | trim mean step | 1 | -0.16 | [-, -] | - | - |
| noeplb | 128 | full mean step | 1 | -0.48 | [-, -] | - | - |
| noeplb | 128 | stall ms/step | 1 | -28.43 | [-, -] | - | - |
| noeplb | 128 | client tok/s/GPU | 1 | -0.11 | [-, -] | - | - |
| noeplb | 128 | rank-0 window tok/s/GPU | 1 | -0.02 | [-, -] | - | - |
| online | 32 | trim mean step | 1 | -0.24 | [-, -] | - | - |
| online | 32 | full mean step | 1 | -1.04 | [-, -] | - | - |
| online | 32 | stall ms/step | 1 | -35.37 | [-, -] | - | - |
| online | 32 | client tok/s/GPU | 1 | +2.25 | [-, -] | - | - |
| online | 32 | rank-0 window tok/s/GPU | 1 | +1.48 | [-, -] | - | - |
| online | 128 | trim mean step | 1 | +0.17 | [-, -] | - | - |
| online | 128 | full mean step | 1 | +0.60 | [-, -] | - | - |
| online | 128 | stall ms/step | 1 | +37.57 | [-, -] | - | - |
| online | 128 | client tok/s/GPU | 1 | +0.15 | [-, -] | - | - |
| online | 128 | rank-0 window tok/s/GPU | 1 | -0.06 | [-, -] | - | - |

## Descriptive (PRE-REG 8, not in the claim): main -> our FT on; % = (second - first) / first


| group | mode | load | rounds | trim % | full % | client % | trim ms first/second |
|---|---|---|---|---|---|---|---|
| 8193512@n0 | noeplb | 32 | 1 | +0.83 | +2.57 | +0.13 | 13.829/13.944 |
| 8193512@n0 | noeplb | 128 | 1 | +0.18 | +0.15 | +0.87 | 27.086/27.134 |
| 8193512@n8 | online | 32 | 1 | +0.42 | +0.69 | -0.19 | 14.598/14.659 |
| 8193512@n8 | online | 128 | 1 | +0.15 | -0.49 | +1.40 | 24.530/24.566 |

| mode | load | metric | n | mean % | 95% CI | one-sided 95% LB | one-sided 95% UB |
|---|---|---|---|---|---|---|---|
| noeplb | 32 | trim mean step | 1 | +0.83 | [-, -] | - | - |
| noeplb | 32 | full mean step | 1 | +2.57 | [-, -] | - | - |
| noeplb | 32 | stall ms/step | 1 | +117.70 | [-, -] | - | - |
| noeplb | 32 | client tok/s/GPU | 1 | +0.13 | [-, -] | - | - |
| noeplb | 32 | rank-0 window tok/s/GPU | 1 | -0.19 | [-, -] | - | - |
| noeplb | 128 | trim mean step | 1 | +0.18 | [-, -] | - | - |
| noeplb | 128 | full mean step | 1 | +0.15 | [-, -] | - | - |
| noeplb | 128 | stall ms/step | 1 | -2.54 | [-, -] | - | - |
| noeplb | 128 | client tok/s/GPU | 1 | +0.87 | [-, -] | - | - |
| noeplb | 128 | rank-0 window tok/s/GPU | 1 | +0.94 | [-, -] | - | - |
| online | 32 | trim mean step | 1 | +0.42 | [-, -] | - | - |
| online | 32 | full mean step | 1 | +0.69 | [-, -] | - | - |
| online | 32 | stall ms/step | 1 | +9.36 | [-, -] | - | - |
| online | 32 | client tok/s/GPU | 1 | -0.19 | [-, -] | - | - |
| online | 32 | rank-0 window tok/s/GPU | 1 | -1.18 | [-, -] | - | - |
| online | 128 | trim mean step | 1 | +0.15 | [-, -] | - | - |
| online | 128 | full mean step | 1 | -0.49 | [-, -] | - | - |
| online | 128 | stall ms/step | 1 | -78.86 | [-, -] | - | - |
| online | 128 | client tok/s/GPU | 1 | +1.40 | [-, -] | - | - |
| online | 128 | rank-0 window tok/s/GPU | 1 | +1.01 | [-, -] | - | - |

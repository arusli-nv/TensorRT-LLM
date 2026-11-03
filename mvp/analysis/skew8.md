# Launch skew, host collectives, admission and client refill: STD-8139740, noeplb, off vs ft vs ulfm (n=1)

Run: $R/mvp/runs/STD-8139740, one group, same nodes, cases std-noeplb-1 (off: FT off, UCX), std-noeplb-ft-1 (ft: FT on,
ULFM ob1/TCP), std-noeplb-ulfm-1 (ulfm: FT off on FT's ULFM launcher). DeepSeek-R1 DEP32, no EPLB, rows healthy-8
(256 users, 20 requests each, OSL 1024) and healthy-32 (1024 users, 15 each). Every number is inside the row's
[start_wall_s, start_wall_s + elapsed_s] window. n = 1 per arm: no repeat, no CI.
Generator: `python3 mvp/analysis/skew8.py mvp/runs/STD-8139740` (tables below are its output; the verdict is by hand).
Remote slow step / own / waves-by-rank-0-burst follow stall_timing.md (rank-0 bursts give clean waves only at load 8;
at load 32 use sections 4-5, which use client waves, 19 at load 8 and 14 at load 32 in every arm).

## 1. Rank-0 steps (print_iter_log in the healthy window)

| load | arm | decode median ms | ctx median ms (n) | slow >=150 ms /min | own /min | remote /min | full waves | remote slow per wave | waves with >=1 | 180-210 ms decode /min | client out tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | off | 11.90 | 197.7 (60) | 37.62 | 29.82 | 7.81 | 20 | 1.05 | 12/20 | 3.19 | 35326 |
| 8 | ft | 11.87 | 198.1 (61) | 36.41 | 27.74 | 8.67 | 21 | 1.19 | 19/21 | 8.32 | 34303 |
| 8 | ulfm | 11.88 | 205.9 (62) | 40.46 | 28.95 | 11.51 | 19 | 1.53 | 11/19 | 5.23 | 34506 |
| 32 | off | 13.28 | 279.1 (128) | 51.78 | 43.57 | 8.21 | 51 | 0.47 | 16/51 | 4.42 | 93433 |
| 32 | ft | 13.77 | 348.3 (116) | 45.55 | 40.42 | 5.13 | 28 | 0.50 | 12/28 | 4.81 | 94555 |
| 32 | ulfm | 13.37 | 244.6 (136) | 52.05 | 46.34 | 5.71 | 58 | 0.29 | 16/58 | 4.13 | 93354 |

### 1b. Wave cycle vs benchmark duration (client waves of section 5)

| load | arm | iters in window | mean step ms (all) | mean decode step ms (<150 ms, no ctx) | wave period s | client duration s | last start after last main wave s | last token after last main wave s (1 s res) |
|---|---|---|---|---|---|---|---|---|
| 8 | off | 10961 | 14.95 | 11.616 | 7.361 | 148.4 | 2.23 | 9 |
| 8 | ft | 11359 | 14.77 | 11.602 | 7.354 | 152.8 | 7.25 | 14 |
| 8 | ulfm | 11243 | 14.85 | 11.543 | 7.358 | 151.9 | 6.39 | 12 |
| 32 | off | 8709 | 20.77 | 13.308 | 11.196 | 168.3 | 3.61 | 12 |
| 32 | ft | 8568 | 20.85 | 13.656 | 10.987 | 166.3 | 4.74 | 13 |
| 32 | ulfm | 8728 | 20.71 | 13.333 | 11.230 | 168.5 | 0.47 | 11 |

## 2. Host collectives (WIDEEP_COLL, every rank; per rank: median over 1000-call lines in the window)

| load | method | arm | ranks | lines | p50 us median over ranks | p50 min..max (rank) | p50 spread max-min | p99 us median over ranks | p99 max rank |
|---|---|---|---|---|---|---|---|---|---|
| 8 | tp_allgather | off | 32 | 352 | 1098.7 | 530.8 (r0)..1155.7 (r31) | 624.9 | 4847.1 | 4921.5 |
| 8 | tp_allgather | ft | 32 | 384 | 1448.2 | 742.3 (r0)..1512.8 (r31) | 770.5 | 5138.0 | 5202.6 |
| 8 | tp_allgather | ulfm | 32 | 384 | 1399.1 | 729.1 (r0)..1470.2 (r31) | 741.1 | 4564.0 | 4630.9 |
| 8 | tp_allgather_int64 | off | 32 | 1760 | 297.2 | 281.2 (r21)..341.2 (r0) | 60.0 | 482.0 | 530.4 |
| 8 | tp_allgather_int64 | ft | 32 | 1824 | 360.1 | 339.4 (r25)..395.7 (r1) | 56.3 | 554.2 | 613.6 |
| 8 | tp_allgather_int64 | ulfm | 32 | 1824 | 350.8 | 332.4 (r21)..386.5 (r1) | 54.1 | 537.2 | 590.1 |
| 8 | tp_gather | off | 32 | 704 | 453.5 | 413.0 (r17)..623.1 (r0) | 210.1 | 740.1 | 1885.1 |
| 8 | tp_gather | ft | 32 | 736 | 648.6 | 597.4 (r3)..810.2 (r0) | 212.8 | 954.8 | 2041.0 |
| 8 | tp_gather | ulfm | 32 | 736 | 613.5 | 555.4 (r3)..797.5 (r0) | 242.1 | 893.5 | 1786.1 |
| 32 | tp_allgather | off | 32 | 288 | 2380.4 | 559.1 (r0)..2431.0 (r31) | 1871.9 | 6718.4 | 6784.1 |
| 32 | tp_allgather | ft | 32 | 256 | 2926.2 | 782.1 (r0)..2991.5 (r31) | 2209.4 | 7581.2 | 7666.6 |
| 32 | tp_allgather | ulfm | 32 | 256 | 2699.5 | 757.8 (r0)..2778.3 (r31) | 2020.5 | 7065.3 | 7171.6 |
| 32 | tp_allgather_int64 | off | 32 | 1408 | 308.7 | 284.9 (r21)..334.3 (r1) | 49.4 | 536.7 | 612.1 |
| 32 | tp_allgather_int64 | ft | 32 | 1376 | 367.8 | 340.5 (r25)..400.4 (r8) | 59.9 | 574.6 | 638.9 |
| 32 | tp_allgather_int64 | ulfm | 32 | 1408 | 360.9 | 340.9 (r13)..390.3 (r8) | 49.4 | 625.1 | 685.0 |
| 32 | tp_gather | off | 32 | 576 | 470.6 | 432.2 (r3)..629.8 (r0) | 197.5 | 868.6 | 2225.6 |
| 32 | tp_gather | ft | 32 | 544 | 678.5 | 623.6 (r3)..850.8 (r0) | 227.2 | 1058.4 | 2323.7 |
| 32 | tp_gather | ulfm | 32 | 544 | 650.2 | 586.2 (r3)..809.9 (r0) | 223.7 | 1039.8 | 2434.7 |

## 3. Launch skew (WIDEEP_SKEW: host gap last collective -> graph replay; per 500-replay window k, across ranks)

| load | arm | ranks | windows | gap p50 median us | p50 spread (max-min) median us | p50 spread max | gap p99 median us | p99 spread median us | p99 spread max | line t span median s | ranks most often slowest p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | off | 32 | 22 | 254.7 | 39.0 | 44.5 | 291.6 | 59.7 | 88.0 | 0.00 | r24:9, r13:4, r12:3 |
| 8 | ft | 32 | 23 | 251.4 | 45.0 | 54.5 | 289.3 | 76.5 | 95.5 | 0.00 | r25:15, r24:8 |
| 8 | ulfm | 32 | 23 | 255.2 | 39.5 | 41.9 | 292.1 | 59.7 | 76.5 | 0.00 | r25:20, r17:2, r24:1 |
| 32 | off | 32 | 17 | 341.2 | 47.4 | 51.1 | 386.2 | 69.3 | 211.1 | 0.00 | r25:7, r24:6, r20:2 |
| 32 | ft | 32 | 17 | 338.1 | 49.9 | 56.3 | 382.5 | 78.8 | 198.3 | 0.00 | r21:8, r25:7, r24:1 |
| 32 | ulfm | 32 | 17 | 341.1 | 42.1 | 48.0 | 386.5 | 72.3 | 210.5 | 0.00 | r25:8, r24:6, r17:3 |

## 4. Admission (WIDEEP_FETCH / WIDEEP_ADMIT, all ranks; per full wave, mean)

Wave = the client waves of section 5 (main start group to the next, from 0.3 s before each), mapped to fetch calls by the hooks' wall times. Main refill = fetches within [-5, +10] calls of the wave's median request call; off-phase = the rest, late = off-phase after it. Timeout release = an ADMIT release of context held >= 50 balance calls (every such hold was 50-59 calls: the 60-iteration gate timeout). Remote slow = rank-0 steps >= 150 ms with no rank-0 context in that or the 2 prior steps, mapped to fetch calls (call = iter + 3).

| load | arm | waves | calls per wave | requests | fetch calls delivering | main-phase calls | off-phase calls | off-phase requests | off-phase rank-fetches | late calls | late requests | timeout releases (rank-events) | distinct release calls | released requests | waves with >=1 release | rank-0 remote slow steps | holds <10 / 10-49 / 50-69 / >=70 calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | off | 19 | 516 | 256.0 | 5.00 | 3.74 | 1.26 | 5.00 | 5.00 | 1.00 | 3.47 | 1.68 | 0.16 | 1.95 | 3/19 | 1.11 | 119 / 0 / 57 / 0 |
| 8 | ft | 19 | 517 | 255.1 | 6.47 | 3.79 | 2.68 | 19.89 | 19.05 | 2.00 | 15.47 | 8.63 | 0.63 | 11.16 | 12/19 | 1.26 | 159 / 13 / 164 / 0 |
| 8 | ulfm | 19 | 515 | 256.0 | 5.26 | 3.37 | 1.89 | 8.47 | 8.47 | 1.84 | 8.42 | 4.95 | 0.47 | 4.95 | 9/19 | 1.53 | 47 / 0 / 94 / 0 |
| 32 | off | 14 | 542 | 1035.7 | 26.00 | 5.00 | 21.00 | 153.14 | 149.07 | 11.93 | 102.93 | 3.21 | 0.29 | 3.71 | 4/14 | 1.71 | 1176 / 500 / 45 / 0 |
| 32 | ft | 14 | 528 | 1027.6 | 19.50 | 6.07 | 13.43 | 82.50 | 77.50 | 5.93 | 53.00 | 5.36 | 0.29 | 7.21 | 3/14 | 1.00 | 696 / 337 / 75 / 0 |
| 32 | ulfm | 14 | 544 | 1035.5 | 25.29 | 5.21 | 20.07 | 145.14 | 140.86 | 11.86 | 100.00 | 2.21 | 0.14 | 2.71 | 2/14 | 1.21 | 1242 / 242 / 57 / 0 |

### 4b. What precedes each rank-0 remote slow step (release on another rank at call c-1 or c)

| load | arm | remote slow steps | after a timeout release | after an immediate release only | none |
|---|---|---|---|---|---|
| 8 | off | 22 | 4 | 18 | 0 |
| 8 | ft | 25 | 12 | 12 | 1 |
| 8 | ulfm | 33 | 9 | 23 | 1 |
| 32 | off | 26 | 2 | 23 | 1 |
| 32 | ft | 16 | 4 | 12 | 0 |
| 32 | ulfm | 18 | 1 | 17 | 0 |

## 5. Client request starts per wave (bench/healthy-<load>.tokens.jsonl; closed loop, start = a completion)

Main group = starts with gaps <= 0.5 s holding >= half the concurrency; wave = main group to the next. Off-phase = starts between the main group's end and the next main group.

| load | arm | waves | main group size (median) | main span median ms | main span max ms | off-phase starts per wave (mean) | waves with off-phase | off-phase delay after wave start median s |
|---|---|---|---|---|---|---|---|---|
| 8 | off | 19 | 250 | 281.0 | 330.8 | 3.47 | 11/19 | 1.70 |
| 8 | ft | 19 | 231 | 275.5 | 688.4 | 15.47 | 17/19 | 4.59 |
| 8 | ulfm | 19 | 240 | 248.0 | 301.4 | 8.42 | 10/19 | 4.71 |
| 32 | off | 14 | 948 | 577.2 | 977.8 | 103.14 | 11/14 | 8.87 |
| 32 | ft | 14 | 996 | 613.2 | 1621.8 | 53.00 | 11/14 | 9.84 |
| 32 | ulfm | 14 | 935 | 635.8 | 1330.3 | 100.00 | 12/14 | 9.78 |

## Verdict (n=1)

- Throughput cost at load 8 reproduced at the client (out tok/s off 35326, ft 34303 = -2.9%, ulfm 34506 = -2.3%), not
  at load 32 (ft +1.2%, ulfm -0.1%). It is not a per-step or per-wave cost: the server wave period is 7.361 / 7.354 /
  7.358 s and decode mean 11.62 / 11.60 / 11.54 ms (1b). The load-8 duration gap (148.4 / 152.8 / 151.9 s) is the tail:
  the last request starts 2.2 / 7.3 / 6.4 s after the last main wave, because off-phase users (closed loop) run
  1.7 / 4.6 / 4.7 s behind the wave (5).
- Stalls: remote slow steps at load 8 7.81 / 8.67 / 11.51 per min, 180-210 ms band 3.19 / 8.32 / 5.23 per min, gate
  timeout release calls per wave 0.16 / 0.63 / 0.47 (3/19, 12/19, 9/19 waves). The FT increase is reproduced in kind;
  ulfm carries most of it. Each remote slow step follows a release on another rank one call earlier (4b), from late
  requests: late requests per wave 3.47 / 15.47 / 8.42 at the server (4) = off-phase client starts 3.47 / 15.47 / 8.42
  (5). So the stalls track refill spread at the client (users that fell out of phase and stay out of phase).
- Host collectives: ft and ulfm are both ~20-45% slower than off at p50 (tp_allgather_int64 297 / 360 / 351 us,
  tp_gather 454 / 649 / 614, tp_allgather 1099 / 1448 / 1399 at load 8); ft is within 2-6% of ulfm. Spread across ranks
  is unchanged (int64 60 / 56 / 54 us). The slowdown is the transport, and it is hidden by the overlap scheduler
  (decode medians equal at load 8).
- Launch skew: p50 spread across ranks 39 / 45 / 40 us (load 8), 47 / 50 / 42 us (load 32): tens of microseconds on a
  ~12 ms step; ft +6 us, ulfm none. Not an explanation.
- Rank-0 work (ft worse than ulfm): not at load 8. At load 32 the decode median is 13.28 / 13.77 / 13.37 ms (mean
  13.31 / 13.66 / 13.33), an ft-only +2.6-3.7% that ulfm does not show; one sample, worth a repeat.

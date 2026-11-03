## 1. Rank-0 steps (print_iter_log in the healthy window)

| load | arm | decode median ms | ctx median ms (n) | slow >=150 ms /min | own /min | remote /min | full waves | remote slow per wave | waves with >=1 | 180-210 ms decode /min | client out tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | off | 11.88 | 194.7 (62) | 31.22 | 25.26 | 5.96 | 20 | 0.85 | 10/20 | 4.56 | 34953 |
| 8 | ft | 11.88 | 206.5 (59) | 31.63 | 29.08 | 2.54 | 19 | 0.26 | 3/19 | 7.63 | 36288 |
| 8 | sm | 11.88 | 202.7 (62) | 37.49 | 28.21 | 9.28 | 19 | 1.21 | 12/19 | 5.71 | 35518 |
| 32 | off | 13.64 | 349.7 (114) | 48.31 | 42.80 | 5.51 | 27 | 0.56 | 11/27 | 2.59 | 95949 |
| 32 | ft | 13.80 | 353.5 (113) | 46.97 | 40.63 | 6.35 | 31 | 0.58 | 10/31 | 6.98 | 94120 |
| 32 | sm | 13.86 | 250.1 (127) | 46.97 | 45.07 | 1.90 | 49 | 0.10 | 5/49 | 2.54 | 93663 |

### 1b. Wave cycle vs benchmark duration (client waves of section 5)

| load | arm | iters in window | mean step ms (all) | mean decode step ms (<150 ms, no ctx) | wave period s | client duration s | last start after last main wave s | last token after last main wave s (1 s res) |
|---|---|---|---|---|---|---|---|---|
| 8 | off | 11335 | 14.62 | 11.673 | 7.200 | 150.0 | 7.13 | 14 |
| 8 | ft | 10938 | 14.65 | 11.585 | 7.149 | 144.5 | 2.47 | 9 |
| 8 | sm | 10939 | 14.96 | 11.579 | 7.305 | 147.6 | 2.47 | 10 |
| 32 | off | 8590 | 20.50 | 13.266 | 10.840 | 163.9 | 4.62 | 13 |
| 32 | ft | 8571 | 20.96 | 13.704 | 11.070 | 167.1 | 4.38 | 12 |
| 32 | sm | 8684 | 20.77 | 13.561 | 11.135 | 167.9 | 4.26 | 13 |

## 2. Host collectives (WIDEEP_COLL, every rank; per rank: median over 1000-call lines in the window)

| load | method | arm | ranks | lines | p50 us median over ranks | p50 min..max (rank) | p50 spread max-min | p99 us median over ranks | p99 max rank |
|---|---|---|---|---|---|---|---|---|---|
| 8 | tp_allgather | off | 32 | 384 | 1140.8 | 545.5 (r0)..1198.2 (r31) | 652.8 | 4120.8 | 4187.2 |
| 8 | tp_allgather | ft | 32 | 352 | 1496.7 | 769.6 (r0)..1557.9 (r31) | 788.3 | 4580.5 | 4641.6 |
| 8 | tp_allgather | sm | 32 | 352 | 1375.4 | 699.7 (r0)..1413.5 (r31) | 713.8 | 4544.4 | 4587.0 |
| 8 | tp_allgather_int64 | off | 32 | 1792 | 296.9 | 269.0 (r29)..336.9 (r3) | 67.9 | 489.9 | 535.3 |
| 8 | tp_allgather_int64 | ft | 32 | 1760 | 355.1 | 328.7 (r28)..408.6 (r3) | 79.9 | 552.0 | 607.9 |
| 8 | tp_allgather_int64 | sm | 32 | 1760 | 304.0 | 272.8 (r20)..344.1 (r3) | 71.3 | 514.4 | 557.9 |
| 8 | tp_gather | off | 32 | 736 | 478.3 | 433.9 (r1)..622.0 (r0) | 188.1 | 763.8 | 1637.4 |
| 8 | tp_gather | ft | 32 | 704 | 670.2 | 614.8 (r5)..815.1 (r0) | 200.4 | 950.2 | 1898.2 |
| 8 | tp_gather | sm | 32 | 704 | 598.4 | 544.4 (r11)..721.6 (r0) | 177.2 | 914.5 | 1799.9 |
| 32 | tp_allgather | off | 32 | 256 | 2572.8 | 585.5 (r0)..2632.9 (r31) | 2047.4 | 6849.7 | 6932.0 |
| 32 | tp_allgather | ft | 32 | 288 | 3036.4 | 814.1 (r0)..3105.2 (r15) | 2291.1 | 7716.8 | 7833.6 |
| 32 | tp_allgather | sm | 32 | 288 | 2700.9 | 735.9 (r0)..2753.6 (r30) | 2017.7 | 7390.6 | 7472.2 |
| 32 | tp_allgather_int64 | off | 32 | 1376 | 307.5 | 279.4 (r13)..339.4 (r3) | 60.0 | 518.8 | 553.3 |
| 32 | tp_allgather_int64 | ft | 32 | 1376 | 365.8 | 337.9 (r28)..403.1 (r3) | 65.2 | 576.8 | 640.4 |
| 32 | tp_allgather_int64 | sm | 32 | 1408 | 320.8 | 286.4 (r29)..347.1 (r2) | 60.8 | 546.2 | 610.0 |
| 32 | tp_gather | off | 32 | 544 | 506.6 | 448.2 (r1)..649.3 (r0) | 201.1 | 868.8 | 2241.6 |
| 32 | tp_gather | ft | 32 | 544 | 699.5 | 626.6 (r1)..834.1 (r0) | 207.5 | 1069.8 | 2457.1 |
| 32 | tp_gather | sm | 32 | 576 | 609.3 | 545.7 (r11)..733.8 (r0) | 188.0 | 996.4 | 2325.4 |

## 3. Launch skew (WIDEEP_SKEW: host gap last collective -> graph replay; per 500-replay window k, across ranks)

| load | arm | ranks | windows | gap p50 median us | p50 spread (max-min) median us | p50 spread max | gap p99 median us | p99 spread median us | p99 spread max | line t span median s | ranks most often slowest p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | off | no WIDEEP_SKEW lines | | | | | | | | | |
| 8 | ft | no WIDEEP_SKEW lines | | | | | | | | | |
| 8 | sm | no WIDEEP_SKEW lines | | | | | | | | | |
| 32 | off | no WIDEEP_SKEW lines | | | | | | | | | |
| 32 | ft | no WIDEEP_SKEW lines | | | | | | | | | |
| 32 | sm | no WIDEEP_SKEW lines | | | | | | | | | |

## 4. Admission (WIDEEP_FETCH / WIDEEP_ADMIT, all ranks; per full wave, mean)

Wave = the client waves of section 5 (main start group to the next, from 0.3 s before each), mapped to fetch calls by the hooks' wall times. Main refill = fetches within [-5, +10] calls of the wave's median request call; off-phase = the rest, late = off-phase after it. Timeout release = an ADMIT release of context held >= 50 balance calls (every such hold was 50-59 calls: the 60-iteration gate timeout). Remote slow = rank-0 steps >= 150 ms with no rank-0 context in that or the 2 prior steps, mapped to fetch calls (call = iter + 3).

| load | arm | waves | calls per wave | requests | fetch calls delivering | main-phase calls | off-phase calls | off-phase requests | off-phase rank-fetches | late calls | late requests | timeout releases (rank-events) | distinct release calls | released requests | waves with >=1 release | rank-0 remote slow steps | holds <10 / 10-49 / 50-69 / >=70 calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | off | 19 | 515 | 253.4 | 5.05 | 3.21 | 1.84 | 20.68 | 20.68 | 1.68 | 19.11 | 20.58 | 0.84 | 20.68 | 9/19 | 0.89 | 2 / 48 / 391 / 0 |
| 8 | ft | 19 | 515 | 256.0 | 3.37 | 3.16 | 0.21 | 0.95 | 0.95 | 0.21 | 0.95 | 0.53 | 0.11 | 0.53 | 2/19 | 0.26 | 1 / 0 / 10 / 0 |
| 8 | sm | 19 | 515 | 256.0 | 4.32 | 3.21 | 1.11 | 4.63 | 4.63 | 1.11 | 4.63 | 0.47 | 0.11 | 0.47 | 2/19 | 1.21 | 19 / 0 / 9 / 0 |
| 32 | off | 14 | 530 | 1029.0 | 17.71 | 6.07 | 11.64 | 73.93 | 69.71 | 5.43 | 42.93 | 4.29 | 0.29 | 6.29 | 3/14 | 1.00 | 466 / 199 / 60 / 0 |
| 32 | ft | 14 | 529 | 1034.7 | 19.21 | 5.71 | 13.50 | 122.43 | 112.64 | 6.07 | 67.21 | 9.29 | 0.50 | 16.50 | 7/14 | 1.29 | 532 / 494 / 130 / 0 |
| 32 | sm | 14 | 536 | 1033.6 | 21.00 | 5.64 | 15.36 | 118.71 | 112.43 | 7.36 | 71.07 | 2.57 | 0.14 | 2.57 | 2/14 | 0.36 | 1099 / 246 / 38 / 0 |

### 4b. What precedes each rank-0 remote slow step (release on another rank at call c-1 or c)

| load | arm | remote slow steps | after a timeout release | after an immediate release only | none |
|---|---|---|---|---|---|
| 8 | off | 17 | 16 | 1 | 0 |
| 8 | ft | 7 | 2 | 5 | 0 |
| 8 | sm | 26 | 2 | 22 | 2 |
| 32 | off | 17 | 4 | 13 | 0 |
| 32 | ft | 20 | 6 | 14 | 0 |
| 32 | sm | 6 | 2 | 4 | 0 |

## 5. Client request starts per wave (bench/healthy-<load>.tokens.jsonl; closed loop, start = a completion)

Main group = starts with gaps <= 0.5 s holding >= half the concurrency; wave = main group to the next. Off-phase = starts between the main group's end and the next main group.

| load | arm | waves | main group size (median) | main span median ms | main span max ms | off-phase starts per wave (mean) | waves with off-phase | off-phase delay after wave start median s |
|---|---|---|---|---|---|---|---|---|
| 8 | off | 19 | 256 | 232.7 | 630.2 | 19.11 | 8/19 | 3.33 |
| 8 | ft | 19 | 256 | 252.6 | 295.4 | 0.95 | 2/19 | 1.58 |
| 8 | sm | 19 | 248 | 249.6 | 291.2 | 4.63 | 11/19 | 1.59 |
| 32 | off | 14 | 1017 | 559.6 | 1832.0 | 42.93 | 7/14 | 9.70 |
| 32 | ft | 14 | 1000 | 590.5 | 1526.2 | 67.21 | 9/14 | 5.06 |
| 32 | sm | 14 | 977 | 583.3 | 1389.9 | 71.07 | 11/14 | 9.58 |

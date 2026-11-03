# Mid-window rank-0 stall bursts over all historical served arms (read-only log analysis)

Files: HISTORY_MIDBURST.py (scanner, `--meta`, `--report`), HISTORY_MIDBURST.csv (1259 sweep arms),
HISTORY_MIDBURST.extra_meta.csv + HISTORY_MIDBURST.extra.csv (100 later arms missing from the sweep CSV, through
10-24 evening), HISTORY_MIDBURST.report.txt (all tables below, with Fisher p).

## Method
- Windows: bench.log labels `healthy-N*` (start_wall_s, elapsed_s). Rank-0 `host_step_time` lines, timestamps PT.
- 10-s buckets of sum(step ms) over steps 40 ms < t <= 5 s; first and last bucket dropped. Burst = >= 2 consecutive
  buckets > window median + 2.5 s. Per arm: count, label, offset in window, offset from the page-cache drop
  (min dropcache-N.log start=), excess, min steps per bucket.
- `mid` = burst starts >= 60 s into the window; `late` = >= 120 s after the drop (not the post-drop refill).
- Transport from timeline `MPIRUN_START mca=[...]` (all arms have it): ob1-tcp = `--mca pml ob1 --mca btl self,tcp`
  (no sm, so TCP within a node too); default = UCX pml. Arm types: FT-ob1, OFF-ucx, OFF-ob1 (ULFM control,
  `*-ulfm*` arms: FT off over the FT transport).
- Excluded: kill arms, arms without healthy windows, profiled arms (nsys/ncu reports; two FT-off arms on 10-24
  evening, STD-8255773/std-online-1 and STD-8265699/std-offline-1, had 65-94 s "bursts" under the profiler).
- Arms with low-load windows (8-64): 323 FT-ob1, 371 OFF-ucx, 27 OFF-ob1, 3 FT-over-UCX (`*-ft-sm`), 2 FT ob1+sm.

## 1-2. Rate by group (arms with >= 1 burst / arms), load 8-64
| scope | kind | FT-ob1 | OFF-ucx | OFF-ob1 (ULFM) | p FT vs OFF | p FT vs ULFM |
|---|---|---|---|---|---|---|
| all modes | any | 11/323 | 4/371 | 0/27 | 0.039 | 1.0 |
| all modes | mid | 4/323 | 0/371 | 0/27 | 0.046 | 1.0 |
| all modes | late | 5/323 | 1/371 (GLM-5.3 scout) | 0/27 | 0.10 | 1.0 |
| EPLB online+offline | late | 5/208 | 0/199 | 0/21 | 0.061 | 1.0 |
| same run dir, FT vs off (160 runs) | late or mid | FT-only 4 runs | off-only 0 | - | sign test 0.125 | |

The ULFM control cannot decide transport vs FT code: at FT's late rate (2.4%), 21 ULFM EPLB arms expect 0.5 bursts;
P(0 | same rate) = 0.60. About 120 ULFM EPLB arms would be needed for 0 bursts to mean p < 0.05. FT over UCX: 3 arms,
no information. So the history says "FT config (code + ob1/TCP together)", and cannot split the two.

"any" bursts are mostly at window start 20-70 s after the drop (page-cache refill), in FT and off arms alike.

## 3. Within FT-ob1 EPLB arms (late bursts / arms)
- Hang detection: True 3/55, False 1/109, settings not logged (older build) 1/44. Bursts occur without hang detection
  (STD-8110144 hang=False; L1-LOSS 10-16 predates it): hang detection is not necessary. True vs False p ~0.1, and it is
  confounded with build freeze-58caa0fde6 and dates 10-23/24.
- extra_slots: ghost 4/111, not logged 1/81, empty 0/10, borrowed 0/6.
- Host copies: every burst arm had full (>150 G) /dev/shm or unknown; EPLB FT arms never used ring copies, so ring
  cannot be tested. no-EPLB and static320 (where ring was used): 0/117 FT, 0/172 off. Mode, not ring, is the variable.
- Mode: online 3/139, offline 2/69, noeplb 0/90, static320 0/27.
- Samplers on 1/111, off 4/97. ULFM detector thread on 1/104, off 4/104 (p ~0.37). Neither is needed.
- Date/build: first late burst 10-16 (build-final-b1d7fa789d, L1-LOSS-7993394), then 10-20 (rc1-78836d1723),
  10-23 x2 and 10-24 (freeze-58caa0fde6, 3/53). None in 10-17..10-19 (0/72) or 10-21/22 (0/24). The freeze build is
  denser, but its arms are also the only late-window load-32 online pairs with long windows.
- Load 128-512: 35/222 FT vs 65/330 off (p 0.26), concentrated on 10-16..10-18 in both groups (harness era, not FT).
  From 10-19: 1/173 FT vs 1/252 off (late). There is no FT signal at high load.

## 4. Older builds lacking a feature
Late bursts appeared in build-final-b1d7fa789d (no hang detection, settings not logged) and build-rc1 (hang off),
so neither hang detection nor the no-prefault/no-populate fixes (8245492 had fix-nopopulate) are required.

## Burst list (load 8-64, late or mid)
| arm | type | mode | build | window | win+ | drop+ | len | excess | min steps |
|---|---|---|---|---|---|---|---|---|---|
| L1-LOSS-7993394/sweep-ft-loss | FT-ob1 | offline | final-b1d7fa789d | healthy-64 | 150 | 517 | 20 | 6.3 | 262 |
| STD-8110144/std-offline-ft-1 | FT-ob1 | offline | rc1-78836d1723 | healthy-8 | 100 | 125 | 20 | 7.9 | 388 |
| STD-8202865/std-online-ft-1 | FT-ob1 | online | freeze-58caa0fde6 | healthy-32 | 240 | 697 | 20 | 8.1 | 198 |
| STD-8206860/std-online-ft-1 | FT-ob1 | online | freeze-58caa0fde6 | healthy-32 | 20 | 159 | 30 | 11.5 | 158 |
| STD-8245492/std-online-ft-1 | FT-ob1 | online | freeze-58caa0fde6 | healthy-32 | 270 | 400 | 40 | 13.1 | 219 |
| GLM53-SCOUT-8045347/glm53-noft | OFF-ucx | GLM | borrow | healthy-64 | 30 | 140 | 20 | 12.9 | 6 |

## Confounds
Node sets differ between runs (pairs share nodes within a run dir only). Dates and builds overlap with
harness changes. ULFM arms are 10-16..10-23, mostly the FT-COST online series. n is small. FT and the ob1/TCP
transport were never separated in an EPLB pair at scale.

## Decisive next step
Run same-node online pairs with three arms: FT-off/UCX, FT-off/ob1-tcp (`std-online-ulfm`), and FT-on/ob1-tcp. Keep
the per-node D-state sampler on. At a 2-3% per-arm event rate, arm counts alone need about 100+ arms per type, which is
impractical. The cheaper decisive route is the mechanism: catch one burst with the D-state/wchan sampler and see
whether the blocked threads are TCP/ob1 progress or FT threads. An FT-on/UCX arm (`*-ft-sm` mca) is the converse control.

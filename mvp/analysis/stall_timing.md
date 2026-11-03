# Stall timing: ADP balance-gate timeouts, noeplb load 8, FT off vs on

Source: rank-0.log print_iter_log lines inside each healthy-8 window (result.json start_wall_s, elapsed_s), cases std-noeplb-1 (off) and std-noeplb-ft-1 (on), runs STD-8116111, STD-G1-8116111, STD-G1-8118134, STD-G1-8118573 (same nodes per group). Generator: ad hoc script over steady_state.iterations()-style parsing plus num_unscheduled. Slow step = host_step_time >= 150 ms. Own = rank 0 ran context in this step or one of the 2 steps before it (the overlap scheduler logs 1-2 trailing slow steps). Remote = slow with no rank-0 context nearby, so some other rank ran context. Burst = rank-0 context steps no more than 5 iterations apart.

## 1. Rate per minute (healthy-8 window)

| group | arm | window s | iters | slow >=150 ms /min | own /min | remote /min | 180-210 ms decode /min (ctx_stall_cost band) |
|---|---|---|---|---|---|---|---|
| STD-8116111 | off | 169 | 10996 | 40.11 | 28.40 | 11.71 | 2.84 |
| STD-8116111 | on | 177 | 11055 | 48.80 | 26.43 | 22.37 | 13.56 |
| STD-G1-8116111 | off | 166 | 10939 | 33.61 | 28.55 | 5.06 | 2.89 |
| STD-G1-8116111 | on | 167 | 10937 | 35.20 | 27.66 | 7.54 | 7.18 |
| STD-G1-8118134 | off | 173 | 11333 | 35.37 | 25.31 | 10.06 | 4.16 |
| STD-G1-8118134 | on | 174 | 11330 | 38.96 | 25.86 | 13.10 | 9.65 |
| STD-G1-8118573 | off | 164 | 10876 | 32.55 | 29.26 | 3.29 | 0.37 |
| STD-G1-8118573 | on | 173 | 11330 | 34.67 | 24.97 | 9.71 | 6.24 |
| **mean** | **off** | | | 35.41 | 27.88 | 7.53 | 2.56 |
| **mean** | **on** | | | 39.41 | 26.23 | 13.18 | 9.16 |

The 180-210 ms band rate reproduces ctx_stall_cost.md (2.56 -> 9.16 /min). The FT increase is all remote slow steps; own (rank-0 context) slow steps are flat.

## 2. Histograms, pooled over 4 groups

### 2a. Iterations since rank 0 last ran context, per slow step

| gap | off own | off remote | on own | on remote |
|---|---|---|---|---|
| 0 | 242 | 0 | 246 | 0 |
| 1-2 | 70 | 0 | 56 | 0 |
| 56-64 | 0 | 35 | 0 | 59 |
| 116-124 | 0 | 20 | 0 | 46 |
| 176-184 | 0 | 24 | 0 | 21 |
| 185-500 | 0 | 6 | 0 | 27 |

### 2b. Iterations since the previous slow step, remote slow steps

| gap | off | on |
|---|---|---|
| 56-64 | 67 | 140 |
| 65-500 | 18 | 13 |

### 2c. Remote slow steps by offset from the start of the rank-0 burst of that wave

Every remote slow step falls exactly on 62 + 60k iterations after rank 0 admits its refill (k = 0, 1, 2, ...): a chain of 60-iteration gate timeouts, each releasing one lone context request on another rank.

| offset (iters) | off | on |
|---|---|---|
| 62 | 34 | 59 |
| 63 | 1 | 0 |
| 122 | 20 | 46 |
| 182 | 23 | 21 |
| 183 | 1 | 0 |
| 242 | 1 | 19 |
| 243 | 1 | 0 |
| 302 | 1 | 2 |
| 362 | 1 | 2 |
| 422 | 1 | 2 |
| 482 | 1 | 2 |

### 2d. Rank-0 hold before admission (iterations with num_unscheduled > 0 right before a rank-0 context step)

| hold iters | off | on |
|---|---|---|
| 1 | 8 | 4 |
| 2 | 63 | 74 |
| 3 | 6 | 4 |

Rank 0 itself never waits for the timeout: its refills release after 1-3 iterations.

## 3. Bursts per wave

A wave = one rank-0 refill cycle (all 8 rank-0 requests finish together and refill in one burst, ~520 iterations apart). Full waves exclude the first (startup) and last (partial) burst. Timeout releases = remote slow steps in the wave; each is one more fetch on another rank that arrived after the burst.

| group | arm | rank-0 bursts | rank-0 requests admitted | iters per burst | full waves | rank-0 bursts per wave | timeout releases per wave (mean) | waves with >=1 timeout | max chain |
|---|---|---|---|---|---|---|---|---|---|
| STD-8116111 | off | 21 | 161 | 524 | 19 | 1.00 | 1.58 | 11/19 | 3 |
| STD-8116111 | on | 21 | 161 | 526 | 19 | 1.00 | 3.26 | 17/19 | 4 |
| STD-G1-8116111 | off | 21 | 161 | 521 | 19 | 1.00 | 0.63 | 8/19 | 2 |
| STD-G1-8116111 | on | 21 | 161 | 521 | 19 | 1.00 | 1.00 | 10/19 | 2 |
| STD-G1-8118134 | off | 22 | 163 | 515 | 20 | 1.00 | 1.45 | 17/20 | 3 |
| STD-G1-8118134 | on | 22 | 163 | 515 | 20 | 1.00 | 1.90 | 18/20 | 4 |
| STD-G1-8118573 | off | 21 | 161 | 518 | 19 | 1.00 | 0.42 | 8/19 | 1 |
| STD-G1-8118573 | on | 22 | 163 | 515 | 20 | 1.00 | 1.40 | 18/20 | 2 |
| **mean** | **off** | | | | | 1.00 | 1.02 | | |
| **mean** | **on** | | | | | 1.00 | 1.89 | | |

Rank-0 bursts and requests admitted are the same FT off and on (21-22 bursts, 161-163 requests; almost every burst is all 8 requests within 3 iterations). FT changes how many other-rank lone requests trail the burst and wait out 60-iteration timeouts.


"releases per wave" counts remote slow steps, so several ranks released by the same timeout count once. Rank 0 is the only rank whose context admissions the log shows, so spreading on other ranks is inferred from the 60-iteration release chain, not seen directly.

## Verdict

- Peaks: every remote slow step is a 60-iteration timeout release, never an immediate release (2a/2b: 0 remote steps within 0-55 iterations of rank-0 context; all at 62 + 60k after the rank-0 burst). Own steps are immediate (gap 0-2; rank-0 holds 1-3 iterations).
- FT off -> on: remote timeout releases 1.02 -> 1.89 per wave (4/4 groups up). More releases fall at offsets 122 and 242, i.e. longer chains, and more waves have at least one (44/77 -> 63/78).
- Rank-0 bursts per wave: 1.00 in both arms. Rank 0 does not spread its refills.
- Supports a narrower version of the hypothesis: under FT, more lone context requests on other ranks arrive after the wave's refill burst and each waits out a 60-iteration timeout, as a chain. Rank 0's own fetch pattern does not change, so "spreads refills across more fetches" holds only for other ranks and is inferred from the chain. Per-rank fetch logs (other ranks' num_fetched/num_unscheduled) would confirm it.

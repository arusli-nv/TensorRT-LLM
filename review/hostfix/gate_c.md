# Gate C: ADP balance-gate stragglers -- mechanism, general fix, A/B, red team

Code read: wt-v5 (branch v5) `py_executor.py` `_balance_adp_requests` (L7048-7123), `_schedule` (L7380),
`_fetch_new_requests` (L6787-6930), `scheduler/adp_router.py` (DefaultADPRouter min-heap). upstream/main 0829689c52
has the same gate (minus the v5 `live_ranks` filter). Data: CLAIMS 153 (stall_timing.md, 4 groups), 155 (skew8.md,
STD-8139740, n=1/arm), 158 (transport_v3_skew.md, STD-8145956, n=1/arm). Config: max_batch_size 512,
`attention_dp_config` timeout_iters 60, batching_wait_iters 0.

## 1. Why a straggler waits the full timeout

The gate allgathers (num_ctx, num_gen, tokens, num_real_gen) per rank every `_schedule`, then:
- release now iff `all(num_gen < max_batch_size)` and `all(num_ctx > 0)` (every live rank has context);
- else hold all context, `adp_ctx_waiting_iters_count += 1`, release when the count reaches 60, or at once if some
  rank has `num_gen == 0` (the only early exit; the env low-occupancy knob is off/0 in our runs).
- The count is one global clock: it is identical on every rank (same gathered input), it ticks even when no rank
  has context, and it resets on any release. So it runs from the refill burst (where all ranks get context and it
  resets) and fires at burst + 60k. A request that misses the burst by 1-10 iterations waits 50-59: CLAIMS 153 (every
  remote slow step at 62 + 60k after rank 0's burst, none in 0-55) and 158 (off arm: 391 of 441 holds were 50-69 calls).

Why the condition cannot be met: 8/GPU closed loop = 256 users on 32 ranks, 8 in flight per rank, queue empty.
Mid-wave, a rank receives a new request only when one of its own users completes, and all of a wave's users complete
together ~520 iterations later. So "all 32 ranks hold context" can only happen at the next burst; "some rank has
zero generation" never happens mid-wave (every rank holds 8). The hold is futile by construction: it ends only by
timeout, or when other stragglers happen to cover all 32 ranks (they don't: 1-2 late fetch calls per wave).

Why bistable (closed-loop feedback):
1. Seed: a user's next request reaches the server a few iterations after the burst (client/response-path jitter;
   FT's slower host collectives plausibly widen it, CLAIMS 155: tp_gather p50 454 -> 649 us).
2. The gate holds it to the next 60-boundary (~0.7 s at ~12 ms/step), so it completes ~60 iterations behind its wave.
3. Closed loop: its next request starts at that completion, so it arrives mid-wave again -> lone again -> held again
   (another 0-60 iterations; drifts later every wave). The router (min-heap on active count) sends it back to the
   rank that just freed a slot, so the same ranks repeat.
4. Nothing pulls it back: rejoining needs the drift to wrap a full ~7.2 s wave, about the length of the benchmark.
   So off-phase users accumulate and persist: a launch with early seeds settles at ~20 held per wave (CLAIMS 158 off:
   20.58 timeout releases/wave, 9/19 waves); a launch without stays near 0.5 (ft 0.53). Per-launch, arm-random,
   FT raises the seed rate (CLAIMS 153: 1.02 -> 1.89 per wave over 4 groups).
Note the gate does batch the cohort (20.68 released requests in 0.84 distinct release calls per wave), so the cost is
not mainly extra stall steps: it is ~0.7 s TTFT added to ~8% of users each wave, the closed-loop tail (CLAIMS 155:
last start 2.2-7.3 s after the last main wave), and the chained 150-200 ms steps at 62 + 60k.

## 2. Fix: release a hold once arrivals have stopped (arrival-rate-adaptive idle limit)

Options considered:
- Empty waiting queue on every rank: rejected. At low/moderate load the queue is empty almost always, so this
  disables the gate; it would also split the refill burst itself (rank 0 needs 1-3 iterations for all ranks to get
  context, stall_timing 2d) into 2-3 context steps.
- "All ranks with pending context are waiting": tautological inside the hold branch; carries no new information.
- Global count instead of per-rank: a lone straggler has global count 1; does not help.
- Timeout tied to step time: step time does not predict whether context will arrive; futility is about arrivals.
- Chosen: the hold is worth keeping only while requests keep arriving. Every rank already sees the same routed
  requests each iteration (`route_requests` output is identical on all ranks), so it can track, with no new collective,
  how many requests were routed in the last `timeout_iters` iterations (A) and how many iterations since the last one.
  Release when the arrival-free gap exceeds W = ceil(T * ln(1 + A) / A) (T = timeout_iters; W = T when A = 0).
  W is where a Poisson stream at the observed rate (A/T per iteration) has its first gap of that length after T
  iterations on average, so Poisson arrivals keep the legacy hold on average; a gap longer than that means arrivals
  stopped. No new knob: T is the existing timeout. Values: A=0 -> 60 (legacy), 1 -> 42, 19 -> 10, 30 -> 7,
  120 -> 3, 256 (refill burst in window) -> 2.
- Effect here: a straggler 1-10 iterations after a burst sees A ~256 -> released after 2 idle iterations instead of
  50-59 -> its next request lands near the next burst (it even gains the burst's 1-3 iteration alignment) -> no new
  off-phase users. An existing mid-wave cohort (A ~19 -> W 10) is released after ~10 instead of up to 60.

Sketch (main-quality; replaces nothing, adds one release reason):
```python
# __init__, in `if self.attention_dp_enable_balance:`
self._adp_recent_arrivals = deque(maxlen=self.attention_dp_time_out_iters)
self._adp_iters_since_arrival = 0
self._adp_num_routed_requests = 0

# _fetch_new_requests, step 6, after route_requests (identical on every rank)
self._adp_num_routed_requests = sum(len(r) for r in all_ranks_new_requests.values())

def _adp_arrival_idle_limit(self) -> int:
    """Arrival-free iterations after which waiting for the other ranks' context is futile.

    A Poisson stream at the rate routed over the last timeout_iters iterations first shows
    a gap this long after timeout_iters iterations on average; a longer gap means arrivals
    have stopped. With no recent arrivals this is the timeout itself.
    """
    timeout = self.attention_dp_time_out_iters
    arrivals = sum(self._adp_recent_arrivals)
    if arrivals == 0:
        return timeout
    return min(timeout, math.ceil(timeout * math.log1p(arrivals) / arrivals))

# _balance_adp_requests, before the `if self.attention_dp_enable_balance:` block
routed, self._adp_num_routed_requests = self._adp_num_routed_requests, 0
self._adp_recent_arrivals.append(routed)
self._adp_iters_since_arrival = 0 if routed else self._adp_iters_since_arrival + 1
# ... in the hold branch:
arrivals_stopped = (any(all_ranks_num_scheduled_context_requests)
                    and self._adp_iters_since_arrival >= self._adp_arrival_idle_limit())
if timeout_reached or arrivals_stopped or not all_ranks_have_gen_requests:
```
`any(...)` keeps the free-running clock unchanged when no rank holds context. Reset the consumed count so an
iteration without a fetch counts as no arrival. The aligned branch (batching_wait_iters) is untouched.

Unit tests (CPU), extend `TestAdpBalanceExcludesPadDummies._make_executor` in
`tests/unittest/_torch/executor/test_py_executor.py` (L3833) with the three new attributes, then add:
1. `test_idle_limit_tracks_recent_arrivals`: A = 0/1/30/256 -> 60/42/7/2; never above timeout_iters.
2. `test_straggler_after_burst_released_once_arrivals_stop`: window [256, 0...], ranks [(1,8,8),(0,8,8),(0,8,8)],
   routed 1 then 0, 0: held, held, released on call 3; waiting count reset.
3. `test_steady_arrivals_keep_legacy_timeout`: window primed with 1 arrival every 2 iterations (A=30, W=7), routed
   alternating 1/0 for 59 calls: held throughout; released by the timeout on call 60.
4. `test_no_recent_arrivals_keeps_full_timeout`: empty window, routed 0: released exactly at 60.
5. `test_idle_rule_does_not_reset_clock_without_context`: no rank has context: counter keeps ticking to 60.
6. `test_all_ranks_with_context_release_unchanged`: aligned case and batching_wait_iters behave as today.
7. `test_departed_rank_does_not_block_idle_release`: live_ranks (0, 2), departed row zeros: released after W.
8. `test_fetch_counts_requests_routed_to_every_rank`: route_requests -> {0: [a], 1: [b, c]}: count 3 on rank 0
   (decision inputs must be rank-identical). Run: `pytest tests/unittest/_torch/executor/test_py_executor.py -k Adp`.

GPU A/B (same nodes, standard 1k/1k R1 NVFP4 DEP32, FT off, config identical; arms = baseline vs baseline+fix
overlay; optional control arm = baseline with timeout_iters 10, config only, to confirm the mechanism):
- The mode is per launch, so the unit is a launch: >= 4 launches per arm, alternating arm order, each launch runs
  closed-loop rows 8/GPU and 32/GPU, then 128 and 512 (pack in one job per launch pair).
- Primary (8 and 32): gate timeout releases per wave (ADMIT holds >= 50 calls), hold histogram (<10/10-49/50-69),
  off-phase client starts per wave (skew8.py sections 4-5), remote slow steps per minute and mean step ms
  (stall ms/step), context steps per wave (guard: the fix must not add more than ~1 per wave).
- Decisive: fix arm has no launch above 2 timeout releases/wave and no 50-69 holds beyond the first wave, while
  baseline shows the bimodal split; context steps/wave and decode median within noise.
- Non-regression: client output tok/s at 128 and 512 within 1% of baseline (mean of launches), and 32/GPU
  context steps per minute not higher (the main high-load risk below).
- Open-loop check (one row each arm): Poisson arrivals at the 8/GPU rate; context steps per minute and TTFT p50/p99.

## 3. Red team

- High load (32/128/512): the window holds many arrivals, so W = 1-3. A 2-3 iteration lull while not every rank has
  context now releases a partial set where the legacy gate kept waiting -> more context steps, each with fewer ranks
  (MoE padding to the busiest rank). For Poisson arrivals the expected hold is unchanged by construction, but closed-loop
  traffic is bursty; 32/GPU today shows many short holds (466-1176 holds < 10 calls per run, CLAIMS 155/158), so this
  is the risk to measure (context steps/min at 32 and tok/s at 128/512). If it regresses, condition the idle release on
  "fewer than half the live ranks hold context" rather than retuning W.
- Open loop: neutral on average for Poisson; for bursty clients it releases when a burst ends, which is the intent.
  At very low load (A -> 0) W -> T, identical to today. TTFT improves for lone requests.
- Disaggregated: context-only workers have no generation requests, so the existing "some rank has 0 gen" release
  fires first (unchanged); generation workers schedule no new context, so the gate is idle. Generation-init arrivals
  are counted but harmless. The benchmark-disagg fill ramp routes in growing batches: arrivals, so holds persist.
- FT recovery: the release decision must stay rank-identical. Inputs are the gathered counts (already filtered to
  `live_ranks`) and the routed requests (identical on survivors), so it is; a departed rank's zero row is excluded
  as today. Requests re-routed after a peer loss count as arrivals and shorten W, which releases faster during
  recovery (desired). A rank whose state diverged (e.g. a future rejoin with an empty window) would only mis-time its
  own release: the gate filters this rank's context list and no collective depends on it, so the cost is one
  unbatched context step, not a hang (worth one assertion in review).
- Interaction: the env low-occupancy knob and the aligned batching wait are untouched; the fix adds one release
  reason. The free-running clock is unchanged when no rank holds context, so the 60k boundaries still batch any
  holds that the idle rule does not end.
- Limits: an existing mid-wave cohort still releases after ~10 idle iterations and drifts ~10/wave until it wraps;
  the fix stops new seeds and cuts each hold ~5x, it does not pull an established cohort back. Seeds themselves
  (late responses) are out of scope here.

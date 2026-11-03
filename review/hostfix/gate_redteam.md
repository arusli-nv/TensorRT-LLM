# Red team: gate_c.md (arrival-rate-adaptive release of the ADP balance gate)

Code read: wt-v5 f96a8c1a53, `py_executor.py` `_balance_adp_requests` L7048-7123, `_fetch_new_requests` L6787-6930,
`_fetch_and_activate_new_requests` L6998, `_schedule` L7380, `_pp_schedule_and_propagate` L3050,
`_pad_attention_dp_dummy_request` L7911, `scheduler/adp_router.py` (split_unservable L263, DefaultADPRouter L330).
**Verdict: sound with changes.** The mechanism and rank-identity hold. The "keeps today's mean hold under Poisson"
claim is wrong, and there is one real high-load hole (saturation reads as "arrivals stopped").

## 1. Hold condition, counter, futility
- Correct. Release now iff all live ranks have gen < max_batch_size and ctx > 0. Otherwise hold, count += 1, release at the
  timeout (or the low-occupancy timeout, off by default: MIN_GENERATION_REQUESTS=0) or when some rank has gen == 0. The counter
  increments in the else branch even when no rank has context and resets only on a release, so it is a free-running
  clock that is rank-identical. Holds end at burst + 60k, as claimed.
- Nits: `num_gen` counts ADP pad dummies, so a padded idle rank does NOT trigger the gen==0 exit (only a rank with ctx and
  no gen does). Router heap is keyed on (active tokens, count), not count; same thing with equal 1k prompts.
- Futility: right for fixed-OSL closed loop at 8/GPU (wave finishes together, empty queue). Benchmark-specific: with variable
  OSL mid-wave holds are not futile, so the rule must stay safe there (sec 5).

## 2. Is A rank-identical? Yes, for the decision inputs as designed
- RequestBroadcaster puts identical items in every rank's waiting queue. `_pop_from_waiting_queue` uses the gathered
  `all_ranks_num_active_requests`. `split_unservable_requests` and `route_requests` are deterministic in
  (gathered states, new_requests) and key only serving ranks. The `all_ranks_new_requests` dict is the same on every
  rank, and the existing "symmetric skip" comment already relies on that invariant. KVCacheAware uses
  `random.Random(req_id)`, which is seeded and therefore identical.
- Validation failures happen after routing, only on the target rank (`_respond_if_invalid`). Counting routed rather than
  activated requests keeps A identical; don't switch to counting `validated_requests`. Unservable requests are excluded
  on every rank alike. Departed ranks are absent from `all_rank_states`, and there's no rejoin path in v5.
- One fetch per `_schedule` holds in the overlap/non-overlap loops (`_prepare_and_schedule_batch`). In the PP loop, `_schedule`
  runs only on the first PP rank, but every TP peer of that stage runs it, so the decision is still consistent.
  Required: `+=` in fetch, zero on consume; the sketch's `=` drops a count if two fetches precede one schedule.
- Divergence would not hang (no collective depends on the balanced ctx count; `_can_queue` gathers, forward pads); test anyway.

## 3. Poisson derivation: math right, conclusion wrong, and it matters
- E[first arrival-free gap of length W] = (e^{lW}-1)/l = T gives W = T ln(1+A)/A: right (discrete form differs < 1 iter).
- But the idle rule is OR'ed with the timeout. The time to the first gap is roughly exponential with mean T, so
  P(idle release before T) ~ 1-1/e ~ 63% and the mean hold drops to about 0.63 T. In the timeout-bound open-loop regime
  (e.g. l = 0.5/iter, A = 30, W = 7, never all 32 ranks within 60) that is about 1.6x more context steps, each covering fewer
  ranks. "Neutral on average" is false. This regime is where the gate earns its keep: one lone 1k prefill slows every rank's MoE step.
- Fix: size W so a false release within T is rare. Example: W = min(T, ceil(T ln(1+10A)/A)), so E[first gap] = 10T
  and P(early) ~ 10%. Values: A=256 -> 2, 120 -> 4, 30 -> 12, 19 -> 17, 6 -> 41, <=3 -> 60. The straggler case (A ~ 256)
  keeps its 2-iteration release, while open-loop batching stays close to today's.

## 4. Failure modes
- **Saturation (required change).** When a rank sits at max_batch_size / max_num_active_requests, the router stops routing
  and requests queue up. A routed count of 0 then means "no capacity", not "no demand". With A large, W = 2-3, so the idle
  rule fires in the hold branch that exists because a rank has no free slot (`all_ranks_have_free_ctx_slots` False). That
  hits exactly the 512/GPU row. Gate the idle rule on `all_ranks_have_free_ctx_slots and not waiting_queue` (the queue is
  rank-identical under ADP). A backlog means arrivals haven't stopped.
- **Intra-burst gaps.** The refill burst takes 1-3 iterations to cover all ranks (stall_timing 2d). With W = 2, two quiet
  iterations inside a burst split it into two context steps. Floor W at k_min ~ 3-4 (validate against the measured intra-burst
  gap distribution) or require A < live ranks for W < k_min.
- **Dense arrivals / partial releases (32-64/GPU).** Today, holds under 10 calls mostly end aligned. Any 3-iteration lull
  inside such a hold now releases a partial set. This is the main GPU risk; see the counters in sec 6.
- **Bursty open loop.** Releases at burst end (intent); bursts with internal gaps >= W split (same floor fix).
- **One rank holding while others release:** impossible, the decision is global. **batching_wait_iters:** aligned branch
  untouched. **Low-occupancy knob:** W uses configured T; harmless, timeout_reached still fires.
- **Disaggregated.** ctx-only workers have gen == 0, so they release immediately as today. Gen-only workers count gen-init arrivals
  but have no context, and `any(ctx)` blocks the rule. Benchmark-disagg ramp: arrivals keep holds alive. OK.
- **Chunked prefill.** Continuation chunks are scheduled context but not arrivals. Today each continuation re-holds up to 60
  (the counter resets on release). With the rule, a continuation after arrivals have stopped releases at once, which is an
  improvement. A test is still needed because this changes per-chunk timing.
- **Spec decoding / cancellations:** no interaction (counts are requests; cancels only remove ctx). **Startup:** empty
  deque -> W ~ T (legacy, safe). **FT recovery:** re-routed requests count as arrivals, identical on survivors.
- A future rejoin (empty deque) would mis-time, not hang; reset deque+gap with `adp_ctx_waiting_iters_count` on membership change.

## 5. Simpler rules (coordinator a)
- "Reset the clock at burst completion": already happens (aligned release zeroes it) and produces the 50-59 wait. No help.
- Fixed k quiet iterations (k = 2-5), always: this fixes 8/GPU, but it breaks low/moderate open loop. At l = 0.1/iter,
  P(4 quiet) = 0.67, so nearly every request gets its own context step, versus about one batched step per 60 today.
  This is the regime the rate term protects.
- Fixed k "once the burst has passed": detecting "a burst in the window" amounts to A >= c*live_ranks, which is
  the rate rule with a step function. Simplest equivalent that keeps the gate's purpose:
  `release if hold and any(ctx) and free_slots and queue empty and gap >= (k if A >= live_ranks else T)`, k = 4.
  It fixes the post-burst straggler (A ~ 256 >= 32), never fires for sparse open-loop traffic (A < 32 -> legacy T), and has
  no log/derivation to defend.
- Recommendation: the two-level rule above (k = 4, threshold = live ranks). It's easier to review and test than the
  ln form, and it has the same effect on the measured failure. It does not shorten holds for an existing mid-wave cohort
  (A ~ 19 < 32 -> 60), but since stragglers stop seeding, cohorts shouldn't form. If the GPU A/B shows established
  cohorts persisting, use the 10T log form from sec 3 instead.

## 6. Moderate load guard (coordinator b) and the counters to read
- 32-64/GPU closed loop: A >= 32 so k = 4 applies; a partial release needs 4 iterations with no routed request anywhere while
  ctx is held, slots free, queue empty. 1024-2048 users complete every ~0.25-0.5 iter, so rare; fixed-OSL lulls: measure.
- Open-loop Poisson below 32 arrivals/60 iterations: legacy behaviour exactly. At 32-120/60 iterations (l 0.5-2/iter),
  P(4 quiet) = e^{-4l} = 0.1-13%/iter: the risk band; the 10T log form is safer there.
- Counters (rank 0, per run; add a release-reason counter aligned/timeout/gen==0/idle to the ADMIT log): context
  steps/min and /wave; histogram of live ranks holding ctx at each release (regression = more releases < 50% of ranks);
  idle releases/min; hold-length histogram; remote slow steps/min; mean step ms, TPOT; tok/s 128/512; open-loop TTFT p50/p99.

## 7. Minimal unit tests (CPU, extend `TestAdpBalanceExcludesPadDummies`, L3834)
1. Post-burst straggler (A >= live ranks, 1 lone ctx): held k-1 quiet calls, released on call k, counter reset.
2. Sparse arrivals (A < live ranks, routed 1 every 5 calls): no idle release; released by the timeout exactly at T.
3. Saturation: one rank gen == max_batch_size, routed 0, waiting queue non-empty: held until T.
4. Intra-burst gap: routed [100, 0, 0, 100] with partial rank coverage, then all ranks: one aligned release, no idle release.
5. No rank has ctx: the clock keeps ticking to T. Aligned and batching_wait_iters paths are unchanged.
6. Chunked continuation after arrivals stop: released on the first call.
7. Rank identity: drive two executors (tp_rank 0 and 1) with the same route_requests output and gathered rows.
   -> identical release sequences; two fetches before one schedule accumulate.
8. A seeded CPU simulation of the gate state machine with Poisson arrivals across 32 ranks at l in {0.1, 0.5, 2}/iter.
   Context steps per 1000 iterations must be within 10% of legacy. This is the cheap guard for sec 3.

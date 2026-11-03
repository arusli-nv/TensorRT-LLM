# Draft case for mvp/harness/job.sh (not merged; paste as one more `case $job in ... esac` block).
# C-COMBINE3: why the static 288-slot layouts cost ~2-6% step time at 128/GPU vs eplb0 (CLAIMS #89,
# analysis/C_COMBINE2_RESULT.md: the cost is combine-kernel WAIT from a wider cross-rank spread of the expert GEMMs;
# unused slots are not the cause), and whether online288 == eplb0 (#83, C-HOSTACC, no nsys) or +5.6% (#89, nsys on
# all 32 ranks). 8 nodes, FT off, 1k/1k, 128/GPU only, NO nsys (clean timing).
# Needs: MVP_BUILD=build-dev-4c27cd5a98 (C-COMBINE2's and C-HOSTACC's wheel; the seeded tuner caches come from it),
#   configs/dep32-lb256.yaml (new), and for the optional hist cases draft_cases/testhooks-c3.patch applied to
#   testhooks.py (forwards main's EXPERT_STATISTIC_* through the launcher; without it the hist cases are skipped).
#
# Arms (C-COMBINE2's env per arm):
#   eplb0     dep32-eplb0.yaml: no load balancer, 256 slots, 8 local experts.
#   lb256     dep32-lb256.yaml: a STATIC load balancer over 256 slots, default placement = eplb0's (rank r holds
#             experts 8r..8r+7). Same slots, experts and routing as eplb0; adds the LB path (moe_load_balance_routing
#             per layer, EPLB-enabled all-to-all setup). Expressible in main as is: num_slots only has to divide
#             ep_size (MoeLoadBalancerConfig.setup); num_slots >= num_experts is asserted only for explicit
#             initial_global_assignments. Static routing starts no LB threads and calls no start_iter/end_iter
#             (MoeLoadBalancer::finalizeModel starts them only for layer_updates_per_iter > 0;
#             MoeLoadBalancerIterContext skips static), so suspect (b)'s per-iteration worker wake
#             (moeLoadBalancer.cpp startIter) exists only in online288; for the static arms (b) = route kernel +
#             EPLB all-to-all setup.
#   ghost288  dep32-b2-288.yaml + TLLM_TEST_EMPTY_EXTRA_SLOTS=1 + borrowed: 9 local slots, the extra one emptied.
#             _without_replicas keeps each expert's lowest local slot (its home slot 0..7), so routing and
#             tokens per rank/expert are eplb0's: suspect (c) cannot apply to ghost288 by construction.
#             Also differs from lb256 by: 9-group grouped GEMMs and their tactics, host copies (the hook turns on
#             peer-loss recovery), borrowed slot mapping.
#   online288 dep32-eplb288.yaml: online EPLB 288 (stats dispatch, LB worker threads, replica rebalancing).
#
# Autotuner: tactics are keyed per rank (the MoE runner key holds num_slots, num_local_experts AND the rank's
#   local_expert_offset), so 8- and 9-slot layouts can never share entries. eplb0 and lb256 have identical keys
#   (256, 8, 8, 8r, ...) and share one cache; the 288 arms keep their own. Every cache is seeded from C-COMBINE2's
#   (runs/C-COMBINE2-8055947/autotuner/<arm>.json; lb256 gets eplb0's), so no case tunes (check: WIDEEP_AUTOTUNE
#   count 0 per case) and each arm runs exactly C-COMBINE2's tactics: warmed identically, and C-COMBINE3 vs
#   C-COMBINE2 differ by nsys only. Remaining confound for (a): 8 vs 9 groups are separate keys with separately
#   tuned tactics (e.g. FC2 at the 256-row bucket); "9 groups" here means "9 groups with their own tactics".
#   Found in C-COMBINE2's caches: ranks disagree on the tactic for the same shape (eplb0 20 of 77 MoE shapes,
#   ghost288 28 of 81, online288 29 of 81; draft_cases/pin_tactics.py), a candidate for the cross-rank GEMM spread
#   itself -> optional C-COMBINE3-PIN below.
#
# Order: smoke = lb256-1 (the new config) first; then 3 blocked rounds, each the 4 arms shuffled (seed = job id),
#   so ab_stats pairs round k of each arm. If lb256-1 fails, lb256 is dropped and the next case is the smoke.
# Expected duration (C-HOSTACC 8054606 at 128/GPU, 2 waves: eplb0 ~6 min, 288 arms 9-11 min incl. startup;
#   tuning skipped here): 3 x (6 + 6 + 10 + 9) ~ 93 min + job start ~5 min. Then, only while the job has >= 12 min
#   left (node 0 decides, all nodes follow), one hist case each for eplb0, replicas288, online288 (~6-9 min each).
#
# Metric: rank-0 steady step time (ab_stats.py steady_mean_step_ms and steady_step_ms, paired by round, 95% CI);
#   noise floor = eplb0's spread across its 3 rounds (C-COMBINE2: 17.71 vs 17.49 ms, 1.3%). An effect counts when
#   its paired CI excludes 0 and it exceeds eplb0's max pairwise gap.
# Decision rules (128/GPU):
#   (b) LB presence     = lb256 - eplb0. >= +1.5%: the LB path costs (route kernel / EPLB a2a setup); next: nsys
#                         window on lb256 vs eplb0 (one rank group). Within noise: (b) out for static layouts.
#   (a) 9 vs 8 slots    = ghost288 - lb256 (LB path held equal). Carries the C-COMBINE2 ghost cost (+2.2%, ~0.4 ms)
#                         -> slot count / 9-group GEMM + tactics; next: C-COMBINE3-PIN, then ncu of FC1/FC2 at 8 vs
#                         9 groups with equal rows. ghost288 ~ lb256 > eplb0 -> it was (b) all along.
#   (c) token skew      = not testable with ghost288 (eplb0's routing, see above); applies to replicas288 (routed
#                         second copies) and online288. Hist cases: rank rows max/mean and largest slot/mean of
#                         replicas288 vs eplb0 (expert_hist.py). replicas288 higher -> (c) explains replicas288's
#                         extra cost over ghost288 (C-COMBINE2: +0.5 ms, route kernel +5 us of it).
#   online discrepancy  = online288 - eplb0 here (no nsys, C-COMBINE2's online tactics). Within noise -> #89's +5.6%
#                         was nsys on all ranks (online's extra stream/host work traced), #83 stands. >= +3% -> #83
#                         (n=1) was the outlier; online288 costs at 128/GPU.
# Burst counts per row at the end (burst_counts.py), as in C-COMBINE2.
case $job in
C-COMBINE3)
    export CASE_TIMEOUT_S=1080
    seed=$C/mvp/runs/C-COMBINE2-8055947/autotuner
    if [[ $dist != *4c27cd5a98* || ! -s $seed/eplb0.json ]]; then
        echo "SMOKE_FAIL preflight: MVP_BUILD must be build-dev-4c27cd5a98 (seed caches $seed)"; echo STD_DONE; exit 1
    fi
    mkdir -p "$out/autotuner"
    if [[ $SLURM_NODEID == 0 ]]; then
        for f in eplb0 ghost288 online288 replicas288; do cp "$seed/$f.json" "$out/autotuner/.$f.tmp"; done
        mv "$out/autotuner/.eplb0.tmp" "$out/autotuner/lb.json"  # eplb0 and lb256: same keys, one cache
        for f in ghost288 online288 replicas288; do mv "$out/autotuner/.$f.tmp" "$out/autotuner/$f.json"; done
    fi
    for _ in $(seq 120); do [[ -s $out/autotuner/replicas288.json && -s $out/autotuner/lb.json ]] && break; sleep 1; done
    arm_env() {  # ARM: the arm's config, hooks and tuner cache (call in the case's subshell)
        export BENCH_CONFIG=dep32-eplb0.yaml TLLM_AUTOTUNER_CACHE_PATH=$out/autotuner/lb.json
        case $1 in
        lb256) export BENCH_CONFIG=dep32-lb256.yaml ;;
        ghost288) export BENCH_CONFIG=dep32-b2-288.yaml TLLM_TEST_EMPTY_EXTRA_SLOTS=1 TLLM_FAULT_TOLERANCE_EXTRA_SLOTS=borrowed
                  export TLLM_AUTOTUNER_CACHE_PATH=$out/autotuner/ghost288.json ;;
        replicas288) export BENCH_CONFIG=dep32-b2-288.yaml TLLM_AUTOTUNER_CACHE_PATH=$out/autotuner/replicas288.json ;;
        online288) export BENCH_CONFIG=dep32-eplb288.yaml TLLM_AUTOTUNER_CACHE_PATH=$out/autotuner/online288.json ;;
        esac
    }
    load=(--mode sweep --loads 128 --iterations 2 --isl 1024 --osl 1024 --warm-each-load)
    (arm_env lb256; run_case lb256-1 noft "${load[@]}")
    case_index=$((case_index + 1))
    arms="eplb0 ghost288 online288 lb256"
    if ! grep -q '"state": "DONE"' "$out/lb256-1/result.json" 2>/dev/null; then
        echo "LB256_FAIL lb256-1: $(grep -o '"state": "[^"]*"' "$out/lb256-1/result.json" 2>/dev/null); dropping lb256"
        arms="eplb0 ghost288 online288"
    fi
    order=""
    for r in 1 2 3; do
        order="$order $(printf '%s\n' $arms | shuf --random-source=<(yes "$SLURM_JOB_ID-$r") | tr '\n' ' ')"
    done
    order=${order/lb256 /}  # lb256's round 1 ran as the smoke
    echo "C_COMBINE3_ORDER lb256(smoke) $order"
    declare -A round
    round[lb256]=1
    smoke=$([[ $arms == *lb256* ]] && echo 0 || echo 1)
    for a in $order; do
        round[$a]=$((${round[$a]:-0} + 1)); k=${round[$a]}
        (arm_env $a; run_case $a-$k noft "${load[@]}")
        case_index=$((case_index + 1))
        [[ $smoke == 1 ]] && require_done $a-$k
        smoke=0
    done
    # (c) histograms: main's ExpertStatistic records iterations 1600-1649 (mid first wave at full load in
    # C-HOSTACC's eplb0/empty288 logs: full load ~1200-2100) eagerly (CUDA graphs off for those steps only), so
    # these cases are not timed. Run only while >= 12 min remain (node 0 decides from SLURM_JOB_END_TIME).
    if grep -q TLLM_TEST_EXPERT_STATISTIC "$H/testhooks.py"; then
        for a in eplb0 replicas288 online288; do
            go=$out/hist-$a.go
            if [[ $SLURM_NODEID == 0 ]]; then
                end=${SLURM_JOB_END_TIME:-$(( $(date +%s) - SECONDS + 7200 ))}  # else: 2 h from job.sh start
                left=$(( end - $(date +%s) ))
                (( left >= 720 )) && echo 1 > "$go.tmp" || echo 0 > "$go.tmp"
                mv "$go.tmp" "$go"
            fi
            for _ in $(seq 120); do [[ -s $go ]] && break; sleep 1; done
            [[ $(cat "$go" 2>/dev/null) == 1 ]] || { echo "HIST_SKIP $a (time)"; break; }
            (arm_env $a
             export TLLM_TEST_EXPERT_STATISTIC_ITER_RANGE=1600-1650 TLLM_TEST_EXPERT_STATISTIC_PATH=$out/hist-$a/expert_stat
             run_case hist-$a noft --mode sweep --loads 128 --iterations 1 --isl 1024 --osl 1024 --warm-each-load)
            case_index=$((case_index + 1))
        done
    else
        echo "HIST_SKIP testhooks-c3.patch not applied"
    fi
    if [[ $SLURM_NODEID == 0 ]]; then
        for b in lb256 ghost288 online288; do
            python3 $H/ab_stats.py "$out" eplb0 $b > "$out/ab_stats-eplb0-vs-$b.txt" 2>&1; done
        python3 $H/ab_stats.py "$out" lb256 ghost288 > "$out/ab_stats-lb256-vs-ghost288.txt" 2>&1
        python3 $H/burst_counts.py "$out" > "$out/burst_counts.txt" 2>&1
        for d in "$out"/*-[0-9]*/ "$out"/hist-*/; do
            [[ -d $d ]] && echo "AUTOTUNE_COUNT $(basename "$d") $(cat "$d"/rank-*.log 2>/dev/null | grep -c WIDEEP_AUTOTUNE)"
        done
        python3 $H/draft_cases/expert_hist.py "$out"/hist-*/ 2>&1
        grep -h "WIDEEP_EMPTY_EXTRA_SLOTS\|MoE load balancer enabled" "$out"/*/rank-0.log | cut -c1-160 | sort | uniq -c
    fi
    echo STD_DONE
    ;;
esac
# Optional follow-up C-COMBINE3-PIN (hypothesis (d): ranks running different tactics for the same shape widen the
# cross-rank GEMM spread). Same wheel and seeds; the -pin arms read a cache in which every rank's MoE entry carries
# the tactic most ranks chose (pin_tactics.py). eplb0 vs eplb0pin and ghost288 vs ghost288pin, 3 blocked rounds
# (~96 min). Decision: a pin arm faster by more than eplb0's round-to-round gap -> per-rank tactic choice is a cause
# (fix direction: tune once and share tactics across ranks); ghost288pin - eplb0pin is then the 9-vs-8 cost proper.
case $job in
C-COMBINE3-PIN)
    export CASE_TIMEOUT_S=1080
    seed=$C/mvp/runs/C-COMBINE2-8055947/autotuner
    if [[ $dist != *4c27cd5a98* || ! -s $seed/eplb0.json ]]; then
        echo "SMOKE_FAIL preflight: MVP_BUILD must be build-dev-4c27cd5a98 (seed caches $seed)"; echo STD_DONE; exit 1
    fi
    mkdir -p "$out/autotuner"
    if [[ $SLURM_NODEID == 0 ]]; then
        for f in eplb0 ghost288; do
            cp "$seed/$f.json" "$out/autotuner/.$f.tmp" && mv "$out/autotuner/.$f.tmp" "$out/autotuner/$f.json"
            python3 $H/draft_cases/pin_tactics.py "$seed/$f.json" "$out/autotuner/.${f}pin.tmp" \
                && mv "$out/autotuner/.${f}pin.tmp" "$out/autotuner/${f}pin.json"
        done
    fi
    for _ in $(seq 120); do [[ -s $out/autotuner/ghost288pin.json ]] && break; sleep 1; done
    order=""
    for r in 1 2 3; do
        order="$order $(printf '%s\n' eplb0 eplb0pin ghost288 ghost288pin | shuf --random-source=<(yes "$SLURM_JOB_ID-$r") | tr '\n' ' ')"
    done
    echo "C_COMBINE3_PIN_ORDER $order"
    declare -A round
    n=0
    for a in $order; do
        n=$((n + 1)); round[$a]=$((${round[$a]:-0} + 1)); k=${round[$a]}
        (export BENCH_CONFIG=dep32-eplb0.yaml TLLM_AUTOTUNER_CACHE_PATH=$out/autotuner/$a.json
         [[ $a == ghost288* ]] && export BENCH_CONFIG=dep32-b2-288.yaml TLLM_TEST_EMPTY_EXTRA_SLOTS=1 \
             TLLM_FAULT_TOLERANCE_EXTRA_SLOTS=borrowed
         run_case $a-$k noft --mode sweep --loads 128 --iterations 2 --isl 1024 --osl 1024 --warm-each-load)
        case_index=$((case_index + 1))
        [[ $n == 1 ]] && require_done $a-$k
    done
    if [[ $SLURM_NODEID == 0 ]]; then
        python3 $H/ab_stats.py "$out" eplb0 eplb0pin > "$out/ab_stats-eplb0-vs-eplb0pin.txt" 2>&1
        python3 $H/ab_stats.py "$out" ghost288 ghost288pin > "$out/ab_stats-ghost288-vs-ghost288pin.txt" 2>&1
        python3 $H/ab_stats.py "$out" eplb0pin ghost288pin > "$out/ab_stats-eplb0pin-vs-ghost288pin.txt" 2>&1
        python3 $H/burst_counts.py "$out" > "$out/burst_counts.txt" 2>&1
        for d in "$out"/*-[0-9]*/; do echo "AUTOTUNE_COUNT $(basename "$d") $(cat "$d"/rank-*.log | grep -c WIDEEP_AUTOTUNE)"; done
    fi
    echo STD_DONE
    ;;
esac

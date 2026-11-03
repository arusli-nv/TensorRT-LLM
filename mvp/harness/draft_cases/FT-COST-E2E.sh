# Draft case for mvp/harness/job.sh (not merged; paste as one more `case $job in ... esac` block).
# FT-COST-E2E: the user-facing FT healthy cost. FT on as deployed vs main's FT-off deployment, same wheel, same
# layout, same tactics. CLAIMS #85/#90 compare ft with noft-ulfm, so both of their arms run ULFM ob1/TCP plus the MPI
# FT detector thread: that is the FT mechanism only. This job compares:
#   noft-stack  mode noft: plain mpirun, Open MPI defaults (UCX), no detector, no TLLM_FAULT_TOLERANCE_MODE: main's
#               FT-off launch on the FT wheel. serve.sh still exports TLLM_FAULT_TOLERANCE_CONFIG, which nothing
#               reads with FT off (moe_load_balancer/peer_loss_recovery read it only once recovery is enabled).
#   ft          mode ft: ULFM mpirun, ob1/TCP, detector thread (period 1 s, timeout 5 s), authority, FT on,
#               failure_domain rank (serve.sh's explicit setting; the wheel's default, node, refuses 288 slots).
#   noft-ulfm   optional bridge to #85/#90 (FT_COST_E2E_ARMS="noft-stack noft-ulfm ft"); off by default because two
#               arms with more rounds are worth more than three arms with fewer.
# Shared by every arm: wheel MVP_BUILD=build-next-fcc86b10b2 (snapshots/src-fcc86b10b2), no overlay;
#   dep32-final.yaml (online EPLB 288, MTP forced acceptance, overlap, ADP balancing); 1k/1k at 8, 32 and 128/GPU,
#   2 measured waves per load, each load warmed. Extra slots hold online EPLB's replicas in both arms: the wheel's
#   extra_slots default (ghost) applies only to static placements, and setting it with online EPLB is an error, so
#   the case sets no TLLM_FAULT_TOLERANCE_EXTRA_SLOTS. TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT is left unset (main's
#   default; the nodes' driver 580 < 615 selects fence-based dispatch in both arms anyway, checked below).
#
# Tactics (CLAIMS #91): the autotuner keys MoE entries per rank (the runner config holds the rank's local expert
#   offset) and tunes each rank and each arm separately. FT-COST-LOW-8057805's two caches have identical key sets
#   (81 per-rank shapes incl. the gather grouped GEMM, 108 shared keys; ft and noft-ulfm both run the gather GEMM),
#   yet differ in 371 of 2592 per-rank entries and 36 of 108 shared ones; ranks disagree on 29 of 81 shapes.
#   Here: one tuning pass on this wheel (tune-noft-stack, tune-ft: fresh caches, one short 8/GPU wave each; these
#   are also the smoke stage of both launch modes), then merge_pin_tactics.py merges them into pinned.json, which
#   holds one (runner, tactic) per shape on every rank for every key either arm has (majority over 2 arms x 32
#   ranks; shared keys: majority over arms). Every measured case reads its own copy of pinned.json (the autotuner
#   writes its cache back at the end of each warmup pass) and must tune nothing: per case WIDEEP_AUTOTUNE = 0,
#   "Priming cached tactic failed" = 0, cache-miss fallbacks = 0, and PIN_CHECK OK (the copy after the case still
#   equals pinned.json). A failed check marks the case's rows as confounded in the report, it does not stop the job.
#   FT_COST_E2E_SEED=<an earlier E2E job's autotuner/pinned.json, same wheel> skips the tuning pass (the second job
#   of a 3+3 pair then runs the first job's tactics, and pooling the two jobs adds no tactic draw); the first
#   measured case is then the smoke.
#
# Order: blocked rounds; round 1's arm order shuffled (seed = job id), later rounds rotate it (two arms: each goes
#   first in every other round, so a drift within a round cancels over pairs); ab_stats pairs round k of each arm.
#   A round starts only while node 0 sees >= FT_COST_E2E_ROUND_MIN min left (default 16 per arm: its cases plus
#   analysis), so rounds stay complete; at most FT_COST_E2E_ROUNDS rounds (default 6).
# Duration (FT-COST-A-8055539 / FT-COST-LOW-8057805 / C-COMBINE2: startup arrival -> server_ready 7.2-9.7 min, one
#   16 min outlier, tuning included or not; healthy rows 8/GPU 27+67 s, 32/GPU 20-36+48-91 s, 128/GPU 30+78 s):
#   a measured case ~8 + 5 = 13 min, a tuning case ~10 min. Job: ~2 min start + 2 x 10 + 3 rounds x 2 x 13 = ~100 min
#   (one slow startup: ~108); a 4th round does not fit in 2 h. With FT_COST_E2E_SEED: ~80 min for 3 rounds, a 4th
#   fits (~106 min) when no startup is slow. So 3 loads x 2 arms x n=3 fits one 2 h job; no split needed. For the
#   BENCHMARK_STANDARD proposal's >= 6 rounds, run it twice (the second with FT_COST_E2E_SEED from the first).
#
# Hypotheses:
#   H1 the user-facing cost (transport + detector + FT mechanism) on the burst-free step is within noise at 32 and
#      128/GPU; at 8/GPU, where steps are shortest (~13.5 ms) and the detector thread's 1 s heartbeats and TCP host
#      collectives weigh most, it may show (noft-ulfm vs noft-stack: L1-REF check 1 measured the transport alone).
#   H2 FT on does not add slow-step bursts (#86, #90: counts overlap between arms).
# Metrics and decision rules (per load):
#   (1) steady_step_ms (median-based, ab_stats.py ft vs noft-stack): cost % and paired 95% CI over rounds; noise
#       floor = noft-stack's max pairwise gap across rounds. PASS: CI upper bound < 1% (decision 8), or the cost is
#       within the noise floor. FAIL: CI excludes 0 and cost > 1% -> attribute with the bridge job (noft-ulfm vs
#       noft-stack = transport + detector; ft vs noft-ulfm = mechanism, #85/#90).
#   (2) bursts per row (burst_ranksum.py, steady steps > 2x the row median): per arm, exact rank-sum. n=3 vs 3
#       cannot reach p < 0.05 (smallest p = 0.1); report counts, decide on the pooled 6 + 6 of two jobs: FT adds
#       bursts if p < 0.05 with ft higher.
#   (3) mean step, p99 step, client throughput/TPOT (ab_stats.py): context, not gated.
#   Any measured case failing the tactic checks: its pair is reported separately, not pooled.
case $job in
FT-COST-E2E)
    export CASE_TIMEOUT_S=1500 BENCH_CONFIG=dep32-final.yaml TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS=1
    if [[ $dist != *fcc86b10b2* || -n "${MVP_OVERLAY:-}" ]]; then
        echo "SMOKE_FAIL preflight: MVP_BUILD must be build-next-fcc86b10b2 without MVP_OVERLAY ($dist, overlay '${MVP_OVERLAY:-}')"
        echo STD_DONE; exit 1
    fi
    arms=${FT_COST_E2E_ARMS:-noft-stack ft}
    load=(--mode sweep --loads 8,32,128 --iterations 2 --isl 1024 --osl 1024 --warm-each-load)
    tuner=$out/autotuner; pinned=$tuner/pinned.json
    mkdir -p "$tuner"
    arm_case() {  # NAME ARM CACHE [bench.py args...]: one case of ARM reading CACHE
        local name=$1 arm=$2 cache=$3; shift 3
        (export TLLM_AUTOTUNER_CACHE_PATH=$cache
         case $arm in
         noft-stack) run_case "$name" noft "$@" ;;
         noft-ulfm) run_case "$name" noft-ulfm "$@" ;;
         ft) run_case "$name" ft "$@" ;;
         esac)
        case_index=$((case_index + 1))
    }
    left_min() { local end=${SLURM_JOB_END_TIME:-$(( $(date +%s) - SECONDS + 7200 ))}; echo $(( (end - $(date +%s)) / 60 )); }
    node0_decides() {  # FILE COMMAND...: node 0 runs COMMAND and writes its status to FILE; every node returns it
        if [[ $SLURM_NODEID == 0 ]]; then
            if "${@:2}"; then echo 1 > "$1.tmp"; else echo 0 > "$1.tmp"; fi
            mv "$1.tmp" "$1"
        fi
        for _ in $(seq 600); do [[ -s $1 ]] && break; sleep 1; done
        [[ $(cat "$1" 2>/dev/null) == 1 ]]
    }
    # Tuning pass (or the seed): produces pinned.json.
    smoke=1
    if [[ -n "${FT_COST_E2E_SEED:-}" ]]; then
        echo "FT_COST_E2E_SEED $FT_COST_E2E_SEED"
        seed_ok() { [[ -s $FT_COST_E2E_SEED ]] && cp "$FT_COST_E2E_SEED" "$pinned.tmp" && mv "$pinned.tmp" "$pinned"; }
        node0_decides "$tuner/pinned.go" seed_ok || { echo "SMOKE_FAIL seed $FT_COST_E2E_SEED"; echo STD_DONE; exit 1; }
    else
        for a in noft-stack ft; do
            arm_case tune-$a $a "$tuner/tune-$a.json" --mode sweep --loads 8 --iterations 1 --isl 1024 --osl 1024
            require_done tune-$a
        done
        merge() { python3 $H/draft_cases/merge_pin_tactics.py merge "$pinned.tmp" "$tuner"/tune-noft-stack.json \
            "$tuner"/tune-ft.json > "$out/merge_pin.txt" 2>&1 && mv "$pinned.tmp" "$pinned"; }
        node0_decides "$tuner/pinned.go" merge || { echo "SMOKE_FAIL merge (merge_pin.txt)"; echo STD_DONE; exit 1; }
        smoke=0
        [[ $SLURM_NODEID == 0 ]] && cat "$out/merge_pin.txt"
    fi
    # Measured rounds: the first round's order is shuffled (seed = job id), later rounds rotate it, so with two
    # arms each one runs first in every other round.
    read -ra first <<< "$(printf '%s\n' $arms | shuf --random-source=<(yes "$SLURM_JOB_ID") | tr '\n' ' ')"
    for r in $(seq "${FT_COST_E2E_ROUNDS:-6}"); do
        room() { (( $(left_min) >= ${FT_COST_E2E_ROUND_MIN:-$(( 16 * ${#first[@]} ))} )); }
        node0_decides "$out/round-$r.go" room || { echo "FT_COST_E2E_STOP before round $r ($(left_min) min left)"; break; }
        order=$(for i in "${!first[@]}"; do echo "${first[(i + r - 1) % ${#first[@]}]}"; done | tr '\n' ' ')
        echo "FT_COST_E2E_ROUND $r $order"
        for a in $order; do
            cache=$tuner/case-$a-$r.json
            [[ $SLURM_NODEID == 0 ]] && cp "$pinned" "$cache.tmp" && mv "$cache.tmp" "$cache"
            arm_case $a-$r $a "$cache" "${load[@]}"
            [[ $smoke == 1 ]] && require_done $a-$r
            smoke=0
            if [[ $SLURM_NODEID == 0 ]]; then
                logs=("$out/$a-$r"/rank-*.log)
                echo "TACTIC_CHECK $a-$r autotune=$(cat "${logs[@]}" 2>/dev/null | grep -c WIDEEP_AUTOTUNE)" \
                    "priming_failed=$(cat "${logs[@]}" 2>/dev/null | grep -c 'Priming cached tactic failed')" \
                    "cache_miss=$(cat "${logs[@]}" 2>/dev/null | grep -c 'using the fallback tactic, due to cache miss')" \
                    "$(python3 $H/draft_cases/merge_pin_tactics.py check "$pinned" "$cache" 2>&1 | tail -1)" \
                    | tee -a "$out/tactic_check.txt"
            fi
        done
    done
    if [[ $SLURM_NODEID == 0 ]]; then
        for b in $arms; do
            [[ $b == ft ]] || python3 $H/ab_stats.py "$out" $b ft > "$out/ab_stats-$b-vs-ft.txt" 2>&1; done
        [[ $arms == *noft-ulfm* ]] && python3 $H/ab_stats.py "$out" noft-stack noft-ulfm > "$out/ab_stats-noft-stack-vs-noft-ulfm.txt" 2>&1
        python3 $H/burst_counts.py "$out" > "$out/burst_counts.txt" 2>&1
        python3 $H/draft_cases/burst_ranksum.py noft-stack ft "$out/burst_counts.txt" > "$out/burst_ranksum.txt" 2>&1
        cat "$out/ab_stats-noft-stack-vs-ft.txt" "$out/burst_ranksum.txt"
        # Deployment per case: the mpirun transport line, whether an authority ran, the a2a dispatch path.
        for d in "$out"/*-[0-9]*/ "$out"/tune-*/; do
            [[ -d $d ]] || continue
            echo "DEPLOY $(basename "$d") $(grep -o 'mca=\[[^]]*\]' "$d/timeline" | head -1)" \
                "authority=$([[ -e $d/authority.log ]] && echo yes || echo no)" \
                "cft_off_driver=$(grep -c 'CFT counted writes disabled: NVIDIA driver' "$d/rank-0.log" 2>/dev/null)"
        done
    fi
    echo STD_DONE
    ;;
esac

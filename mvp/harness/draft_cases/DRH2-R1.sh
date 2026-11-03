# Draft case for mvp/harness/job.sh (not merged; paste as one more `case $job in ... esac` block).
# FINDINGS DR-H2 / v7 invariant 5 ("no failed-issuer writes to reused storage") / checklist 1d.4(a): late writes
# between a lost rank and the survivors, both directions, at R1 DEP32 (8 nodes). Plan: analysis/SWEEP_2026-10-18.md
# "DR-H2" (done here at R1 DEP32 instead of V3-Lite DEP8; the whole NVL72 rack is one fabric domain either way).
# Needs: MVP_BUILD = the wheel under test (build-dev-28ea730a19 or later) and draft_cases/testhooks.patch applied to
#   mvp/harness/testhooks.py (TLLM_TEST_DELAY_ABORT_S, TLLM_TEST_LATE_WRITE_IN_DECODE). Verdicts: draft_cases/ft_watch.py.
# Exact tokens: tokens/frontend.py --inflight-after on config-small-overlap.yaml (overlap scheduler, CUDA graphs,
#   static 288 slots, empty extra slots), routing excludes the victim (rank 5, node 1) from startup, so the survivors'
#   480-token greedy outputs after the loss must equal the same run's healthy pass (no separate reference run).
# Expected duration: ~7-8 min per tokens case (KILL-MTP-R1 estimate), 6 cases ~45-50 min; capped at
#   6 x CASE_TIMEOUT_S = 90 min (< 2 h).
#
# 1) drh2-ref (smoke): healthy pass only; PASS_HEALTHY (deterministic) or the job ends.
# 2) drh2-control: rank 5 SIGKILLed in two uncommitted overlap steps (TLLM_TEST_KILL_WHEN_BUSY: the most kernels
#    queued on the survivors), no delay: the like-with-like control of 3).
# 3) drh2-delay10-{1,2} (survivors -> freed victim memory, the DR-H2 claim): the same kill, and every survivor
#    delays its reaction (MoE abort request, revoke, recovery) by 10 s after it first notices the loss, so the
#    kernels still queued at the abort run their write prologue into the victim's workspace after the victim's GPU
#    memory was freed (~5 s after SIGKILL). Precondition (else INVALID): every survivor's abort lands > T_free + 3 s,
#    T_free from the victim GPU's memory.used sampled every ~0.3 s on node 1.
# 4) drh2-latewrite-{9,12} (lost rank -> survivors' reused storage): rank 5's GPU sleeps 9 / 12 s before one decode
#    replay, then its process SIGSTOPs itself with that replay queued. Open MPI's detector (timeout 5 s) declares it
#    lost and the survivors recover and resume, while its CUDA context lives: after the sleep its GPU runs the
#    replay's all-to-all prologue into the peers' receive regions, flags and counters (the storage the survivors
#    reuse). A detached child SIGKILLs it 30 s after the stop. Preconditions (else INVALID): the victim was alive at
#    the estimated late write (no RANK_EXIT before it) and the late write followed every survivor's resume.
# PASS per case (ft_watch.py drh2, DRH2_VERDICT line): the 31 survivors resume; exact tokens after the loss; the kill
#   landed mid-generation; no CUDA error in any survivor log (illegal address, launch failure, device assert, ...);
#   no new Xid line in any node's kernel log (dmesg snapshot before/after each case, where readable).
# FAIL in 3) confirms DR-H2 (SWEEP fix: observe_abort at each rank-mask kernel's entry, before the remote prologue).
# Not checkable yet: (a) dmesg may be unreadable in the container: kernel-<node>-before.txt then says "not
#   permitted" and the Xid criterion falls back to CUDA errors in the rank logs (verdict lists the readable nodes);
#   (b) the late write's time is estimated (t_arm + delay at an assumed ~2 GHz SM clock; the victim's host is stopped,
#   so nothing can log when its GPU actually wrote); (c) whether the late write overlaps the compared decode is
#   reported (t_late_write_est vs kill_wall_s + recovered_requests_done_s), not enforced: with 480 tokens the
#   survivors finish ~5-6 s after resuming, so the two delays bracket that window.
case $job in
DRH2-R1)
    export CASE_TIMEOUT_S=900 TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT=0
    export TOKENS_CONFIG=config-small-overlap.yaml TLLM_FAULT_TOLERANCE_EXTRA_SLOTS=empty
    export TLLM_TEST_ORACLE_EXCLUDE_RANK=5
    victim_node=1 victim_gpu=1  # rank 5 = node 1, local rank 1
    # Preflight (seconds): the hooks this case needs, and a wheel.
    missing=""
    for hook in TLLM_TEST_DELAY_ABORT_S TLLM_TEST_LATE_WRITE_IN_DECODE; do
        grep -q "$hook" "$H/testhooks.py" || missing="$missing $hook"
    done
    compgen -G "$dist/*.whl" > /dev/null || missing="$missing wheel"
    if [[ -n $missing ]]; then echo "SMOKE_FAIL preflight: missing$missing"; echo STD_DONE; exit 1; fi
    require_state() {  # NAME STATE: end the job on every node unless case NAME's result.json has STATE
        local result=$out/$1/result.json
        for _ in $(seq 60); do [[ -s $result ]] && break; sleep 1; done
        grep -q "\"state\": \"$2\"" "$result" 2>/dev/null && return 0
        echo "SMOKE_FAIL $1: $(grep -o '"state": "[^"]*"' "$result" 2>/dev/null)"
        echo STD_DONE
        exit 1
    }
    kernel_log() {  # RUN PHASE: this node's Xid / NVRM kernel-log lines (or why they are unreadable)
        local file=$1/kernel-$SLURM_NODEID-$2.txt msg
        if msg=$(dmesg -T 2>&1); then
            { echo "dmesg ok $(date +%s)"; grep -iE 'xid|nvrm' <<< "$msg" | tail -200; } > "$file"
        else
            echo "dmesg not permitted: ${msg:0:160}" > "$file"
        fi
    }
    drh2_case() {  # NAME: one tokens case (TOKENS_ARGS and hooks from the caller), kernel log around it
        local name=$1 run=$out/$1 sampler=""
        mkdir -p "$run"
        kernel_log "$run" before
        if [[ $SLURM_NODEID == "$victim_node" ]]; then
            # When the victim's GPU memory is freed (T_free), every ~0.3 s: epoch, MiB used.
            ( while [[ ! -e $run/frontend.done && ! -e $run/done ]]; do
                  echo "$(date +%s.%N), $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $victim_gpu 2>&1)"
                  sleep 0.2
              done > "$run/victim-gpumem.csv" ) &
            sampler=$!
        fi
        run_case "$name" ft
        kernel_log "$run" after
        [[ -n $sampler ]] && kill "$sampler" 2>/dev/null
    }
    drh2_verdict() {  # NAME MODE [DELAY_S]: node 0, once every node's kernel log after the case is in
        [[ $SLURM_NODEID == 0 ]] || return 0
        for _ in $(seq 300); do
            (( $(ls "$out/$1"/kernel-*-after.txt 2>/dev/null | wc -l) >= SLURM_NNODES )) && break
            sleep 1
        done
        python3 "$H/draft_cases/ft_watch.py" drh2 --run "$out/$1" --victim 5 --mode "$2" --delay-s "${3:-0}" \
            2>&1 | tee -a "$out/drh2-verdicts.log" | cut -c1-600
    }
    inflight="--victim 5 --inflight-after 1 --max-tokens 480 --no-probe"
    # 1) Smoke: healthy, deterministic.
    (export TOKENS_ARGS="--victim 5 --no-kill --max-tokens 480"; drh2_case drh2-ref)
    case_index=$((case_index + 1))
    require_state drh2-ref PASS_HEALTHY
    # 2) Control and 3) delayed abort: SIGKILL in two uncommitted overlap steps.
    for name in drh2-control drh2-delay10-1 drh2-delay10-2; do
        delay=0; [[ $name == drh2-delay10-* ]] && delay=10
        trigger=$out/trigger-$name
        (export TLLM_TEST_KILL_RANK=5 TLLM_TEST_KILL_IN_DECODE=$trigger TLLM_TEST_KILL_WHEN_BUSY=1
         export TLLM_TEST_TRIGGER_NEEDS_REQUESTS=1 TLLM_TEST_DELAY_ABORT_S=$delay
         export TOKENS_ARGS="$inflight --trigger $trigger"
         drh2_case $name)
        case_index=$((case_index + 1))
        mode=control; [[ $delay != 0 ]] && mode=delay-abort
        drh2_verdict $name $mode
    done
    # 4) Late writes of a lost-but-alive victim.
    for delay in 9 12; do
        name=drh2-latewrite-$delay trigger=$out/trigger-drh2-latewrite-$delay
        (export TLLM_TEST_KILL_RANK=5 TLLM_TEST_LATE_WRITE_IN_DECODE=$trigger TLLM_TEST_LATE_WRITE_DELAY_S=$delay
         export TLLM_TEST_LATE_WRITE_KILL_S=30 TLLM_TEST_TRIGGER_NEEDS_REQUESTS=1
         export TOKENS_ARGS="$inflight --trigger $trigger"
         drh2_case $name)
        case_index=$((case_index + 1))
        drh2_verdict $name late-write $delay
    done
    if [[ $SLURM_NODEID == 0 ]]; then
        grep -h "DRH2_VERDICT" "$out/drh2-verdicts.log" | cut -c1-400
        grep -h "WIDEEP_ABORT_DELAY\|WIDEEP_ABORT_SET\|WIDEEP_SELF_\|WIDEEP_LATE_WRITE\|resumed serving without\|Running .* requests of the lost ranks again\|RuntimeError" \
            "$out"/drh2-*/rank-{0,5}.log 2>/dev/null | sed 's/ t=[0-9.]*//' | cut -c1-200 | sort | uniq -c | head -30
        grep -H "RANK_EXIT" "$out"/drh2-*/timeline 2>/dev/null | sed 's/^.*runs\///' | head -12
        head -3 "$out"/drh2-ref/kernel-*-before.txt 2>/dev/null | cut -c1-200
    fi
    echo STD_DONE
    ;;
esac

# Draft case for mvp/harness/job.sh (not merged; paste as one more `case $job in ... esac` block).
# v7 checklist 1b.4 / 1b.5 / 1c.6 / 1d.2, overnight decision 6 (repeated failures, rerun-once with quarantine;
# analysis/REPEATED_FAILURES.md "Tests: GPU"). R1 DEP32, 8 nodes.
# Needs: MVP_BUILD=build-dev-28ea730a19 (ee7e9f0cab "Recover a WideEP deployment from further losses"; the
#   preflight below refuses a wheel without it), and draft_cases/testhooks.patch applied to mvp/harness/testhooks.py
#   (TLLM_TEST_KILL_TRIGGER_FOR, TLLM_TEST_KILL_RERUN_OWNER). Side driver: draft_cases/ft_watch.py (node 0).
# Deployment: the 320 ghost layout (dep32-b2-320.yaml, extra slots borrowed = "ghost" after the rename patch, node
#   failure domain = the code default), as BORROW-KILL-R1, so one config serves all three cases; 128/GPU, 1k/1k,
#   3 waves (~165 s of load: room for two kills and two recoveries).
# Expected duration: ~16 min per case (R1 startup ~10-11 min as BORROW-KILL-R1-8054606, warmup ~45 s, load ~3 min,
#   teardown ~1 min), ~50 min for the three; capped at 3 x CASE_TIMEOUT_S = 75 min (< 2 h).
# Smoke: a preflight (seconds: patched hooks + wheel) ends the job at once on failure; case 1 is the smoke for
#   cases 2-3 (require_verdict ends the job unless it PASSes). A <5 min smoke is impossible at R1 (startup alone ~10 min).
#
# 1) twokill-rank5-rank9: rank 5 SIGKILLs itself in a decode step under load; once 31 rank logs report "resumed
#    serving ... at membership generation 1" plus a 10 s margin, ft_watch.py creates rank 9's own trigger and rank 9
#    SIGKILLs itself in a decode step (load still running, checked; else INVALID).
#    PASS (ft_watch.py verdict.json, TWOKILL_VERDICT line): all 30 survivors log generation 2 without ranks [5, 9];
#    /fault_tolerance (sampled 1/s) last shows generation 2, recoveries 2, failed_ranks [5, 9], 30 live ranks;
#    /health (sampled every 50 ms) shows degraded and never terminal (every state seen is recorded as a run-length
#    sequence, e.g. healthy -> recovering -> degraded -> recovering -> degraded; "recovering" is reported, not
#    required: it lasts ~1 s and the probe answers from a cached authority status); the load finishes (bench state
#    DONE); zero WIDEEP_CAPTURE / COMPILE / AUTOTUNE after the first resume on every survivor (v7 invariant 3).
# 2) twokill-rerun-quarantine (rerun-once, quarantine): rank 5 SIGKILLs itself in a prefill forward of the first
#    wave (its context requests delivered nothing, so rank 0 runs them again); rank 0 writes their ids before
#    readmitting them, and the first non-0 rank that prefills one SIGKILLs itself in that forward (the new owner
#    of a re-run request, before it delivers anything). PASS: as 1) with failed ranks [5, owner], plus rank 0
#    logs WIDEEP_QUARANTINE count >= 1 at the second recovery (the request "had already run again" and fails with
#    a retryable error instead of running a third time). The client-side error text is reported, not required.
# 3) twokill-node1-rank9 (node failure domain variant): node 1 (ranks 4-7) SIGKILL themselves in one decode step
#    (process-level node loss), then rank 9 after generation 1 + 10 s. PASS: as 1) with 27 survivors and failed
#    ranks [4, 5, 6, 7, 9] (27 x 10 slots >= 256 experts).
# Not checkable yet: "recovering" may be missed by 50 ms sampling (reported only); the client never sees which
#   request was quarantined (a stream the server ends with an error can look like a short request to
#   benchmark_serving), so the quarantine proof is rank 0's WIDEEP_QUARANTINE count.
case $job in
TWOKILL-R1)
    export CASE_TIMEOUT_S=1500 TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT=0
    export BENCH_CONFIG=dep32-b2-320.yaml TLLM_FAULT_TOLERANCE_EXTRA_SLOTS=borrowed WIDEEP_FAILURE_DOMAIN=node
    export TLLM_AUTOTUNER_CACHE_PATH=$out/autotuner/ft.json TLLM_TEST_TRIGGER_NEEDS_REQUESTS=1
    mkdir -p "$out/autotuner"
    # Preflight (seconds): the hooks this case needs, and a wheel that recovers more than once.
    preflight=$(python3 - "$H/testhooks.py" "$dist" <<'PY'
import glob, sys, zipfile
hooks = open(sys.argv[1]).read()
missing = [h for h in ("TLLM_TEST_KILL_TRIGGER_FOR", "TLLM_TEST_KILL_RERUN_OWNER") if h not in hooks]
wheels = glob.glob(sys.argv[2] + "/*.whl")
if not wheels:
    missing.append("wheel")
else:
    source = zipfile.ZipFile(wheels[0]).read("tensorrt_llm/_torch/pyexecutor/peer_loss_recovery.py").decode()
    missing += [] if "_own_loss_covered" in source else ["repeated recovery (ee7e9f0cab) in the wheel"]
print(" ".join(missing))
PY
)
    if [[ -n $preflight ]]; then echo "SMOKE_FAIL preflight: missing $preflight"; echo STD_DONE; exit 1; fi
    require_verdict() {  # NAME: end the job on every node unless ft_watch.py judged case NAME PASS
        local verdict=$out/$1/verdict.json
        for _ in $(seq 180); do [[ -s $verdict ]] && break; sleep 1; done
        grep -q '"verdict": "PASS"' "$verdict" 2>/dev/null && return 0
        echo "SMOKE_FAIL $1: $(grep -o '"verdict": "[^"]*"' "$verdict" 2>/dev/null)"
        echo STD_DONE
        exit 1
    }
    twokill_case() {  # NAME WATCH_ARGS BENCH_ARGS...: one case with ft_watch.py beside it on node 0
        local name=$1 watch_args=$2 watch=""; shift 2
        if [[ $SLURM_NODEID == 0 ]]; then
            # shellcheck disable=SC2086
            python3 "$H/draft_cases/ft_watch.py" twokill --run "$out/$name" $watch_args > "$out/watch-$name.log" 2>&1 &
            watch=$!
        fi
        run_case "$name" ft "$@"
        [[ -n $watch ]] && { timeout 300 tail --pid="$watch" -f /dev/null; cat "$out/watch-$name.log"; }
    }
    load=(--mode failover --load 128 --iterations 3 --kill-window-s 30,45)
    # 1) rank 5, then rank 9 after the first recovery committed.
    name=twokill-rank5-rank9
    (export WIDEEP_VICTIM=5 TLLM_TEST_KILL_RANK=5 TLLM_TEST_KILL_TRIGGER=$out/trigger-$name-rank5
     export TLLM_TEST_KILL_IN_DECODE=$TLLM_TEST_KILL_TRIGGER TLLM_TEST_KILL_TRIGGER_FOR=9=$out/trigger-$name-rank9
     twokill_case $name "--first-count 31 --final-count 30 --expect-failed 5,9 --second-trigger $out/trigger-$name-rank9" \
         "${load[@]}" --kill decode)
    case_index=$((case_index + 1))
    require_verdict $name
    # 2) rank 5 in a first-wave prefill, then the new owner of one of its re-run requests (quarantine).
    name=twokill-rerun-quarantine
    (export WIDEEP_VICTIM=5 TLLM_TEST_KILL_RANK=5 TLLM_TEST_KILL_TRIGGER=$out/trigger-$name-rank5
     export TLLM_TEST_KILL_IN_PREFILL=$TLLM_TEST_KILL_TRIGGER TLLM_TEST_KILL_RERUN_OWNER=$out/rerun-$name
     twokill_case $name "--first-count 31 --final-count 30 --expect-failed 5 --rerun-owner $out/rerun-$name" \
         --mode failover --load 128 --iterations 3 --kill-window-s 2,6 --kill prefill)
    case_index=$((case_index + 1))
    # 3) node 1 (ranks 4-7), then rank 9.
    name=twokill-node1-rank9
    (export WIDEEP_VICTIM=4 WIDEEP_LOST=4 TLLM_TEST_KILL_RANK=4,5,6,7 TLLM_TEST_KILL_TRIGGER=$out/trigger-$name-node1
     export TLLM_TEST_KILL_IN_DECODE=$TLLM_TEST_KILL_TRIGGER TLLM_TEST_KILL_TRIGGER_FOR=9=$out/trigger-$name-rank9
     twokill_case $name "--first-count 28 --final-count 27 --expect-failed 4,5,6,7,9 --second-trigger $out/trigger-$name-rank9" \
         "${load[@]}" --kill decode)
    case_index=$((case_index + 1))
    if [[ $SLURM_NODEID == 0 ]]; then
        grep -h "TWOKILL_VERDICT" "$out"/watch-*.log | cut -c1-600
        grep -h "Restored .* expert slots\|resumed serving without\|Running .* requests of the lost ranks again\|WIDEEP_RERUN\|WIDEEP_QUARANTINE\|failed during the recovery\|not recoverable\|RuntimeError" \
            "$out"/*/rank-0.log 2>/dev/null | sed 's/^.*\(resumed serving\)/\1/' | cut -c1-200 | sort | uniq -c | head -30
        grep -h "WIDEEP_SELF_KILL\|RANK_EXIT\|MPIRUN_EXIT" "$out"/*/rank-{4,5,9}.log "$out"/*/timeline 2>/dev/null | cut -c1-160 | head -20
        grep -H '"state"' "$out"/*/result.json | sed 's/^.*runs\///' | cut -c1-200
    fi
    echo STD_DONE
    ;;
esac

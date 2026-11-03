# Draft case for mvp/harness/job.sh (not merged; paste as one more `case $job in ... esac` block).
# Overnight plan decision 7 / GPU queue item 4: MTP + overlap kill tests at R1 DEP32 (analysis/SPEC_DEC_RECOVERY.md).
# 8 nodes, ~65 min expected (tokens cases ~7 min, load cases ~13 min as BORROW-KILL-R1-8054606), capped at 2 h by
# CASE_TIMEOUT_S=1200 x 6. MVP_BUILD = the wheel under test (today's code for the G4 evidence; a wheel with the
# SPEC_DEC_RECOVERY change 1-3 overlaid via MVP_OVERLAY to test it). Real MTP acceptance throughout (no
# TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS): rejected drafts exercise the KV rewind next to the discard.
# New config: harness/tokens/config-small-mtp-overlap.yaml. Existing: configs/dep32-final.yaml, dep32-final-320.yaml.
case $job in
KILL-MTP-R1)
    # 1) Smoke + exact tokens (tokens/frontend.py, routing excludes rank 2 from startup, small MTP + overlap config):
    #    a healthy run (PASS_HEALTHY = deterministic; ends the job otherwise), then twice a kill of rank 2 in the
    #    overlap scheduler's two uncommitted steps (GPU held ~2 s at step k, SIGKILL at step k+1) while it serves
    #    a request: survivors' 400-token outputs must equal the healthy pass. Rerun-once: the victim's undelivered
    #    request runs again once on a survivor and must be answered ("Madrid"). frontend.py still marks the case
    #    FAIL because it expects the victim's request to fail; RERUN_VERDICT below is the verdict for these cases.
    # 2) Load (bench.py failover, 128/GPU, 1k/1k) on the FINAL deployment (online EPLB 288, MTP, overlap, ADP
    #    balancing): rank 5 SIGKILLed in a decode step; rank 2 SIGKILLed in a prefill forward (context requests
    #    finishing in a discarded step, gap G2); node 1 (ranks 4-7, process-level node loss, 320 slots, node
    #    failure domain) SIGKILLed in a decode step. Pass: every survivor resumes, the load finishes (state DONE),
    #    no WIDEEP_CAPTURE / WIDEEP_COMPILE / WIDEEP_AUTOTUNE after "resumed serving" (v7 invariant 3).
    export CASE_TIMEOUT_S=1200 TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT=0
    unset TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS
    mkdir -p "$out/autotuner"
    require_state() {  # NAME STATE: end the job on every node unless case NAME's result.json has STATE (tokens cases)
        local result=$out/$1/result.json
        for _ in $(seq 60); do [[ -s $result ]] && break; sleep 1; done
        grep -q "\"state\": \"$2\"" "$result" 2>/dev/null && return 0
        echo "SMOKE_FAIL $1: $(grep -o '"state": "[^"]*"' "$result" 2>/dev/null)"
        echo STD_DONE
        exit 1
    }
    # Static 288 for the oracle; empty extra slots: the long-tested restore path, so only spec-dec is new here.
    export TOKENS_CONFIG=config-small-mtp-overlap.yaml TLLM_FAULT_TOLERANCE_EXTRA_SLOTS=empty
    (export TLLM_TEST_ORACLE_EXCLUDE_RANK=2 TOKENS_ARGS="--no-kill --max-tokens 400"; run_case mtp-tokens-ref ft)
    case_index=$((case_index + 1))
    require_state mtp-tokens-ref PASS_HEALTHY
    for k in 1 2; do
        trigger=/tmp/wideep-kill-$SLURM_JOB_ID-mtp-inflight-$k
        (export TLLM_TEST_ORACLE_EXCLUDE_RANK=2 TLLM_TEST_KILL_RANK=2 TLLM_TEST_KILL_IN_DECODE=$trigger
         export TLLM_TEST_KILL_WHEN_BUSY=1 TLLM_TEST_TRIGGER_NEEDS_REQUESTS=1
         export TOKENS_ARGS="--inflight-after 1 --max-tokens 400 --no-probe --trigger $trigger"
         run_case mtp-tokens-inflight-$k ft)
        case_index=$((case_index + 1))
    done
    unset TOKENS_CONFIG TLLM_FAULT_TOLERANCE_EXTRA_SLOTS
    export BENCH_CONFIG=dep32-final.yaml TLLM_AUTOTUNER_CACHE_PATH=$out/autotuner/ft288.json
    export TLLM_TEST_TRIGGER_NEEDS_REQUESTS=1
    load=(--mode failover --load 128 --iterations 2 --kill-window-s 30,45)
    (export WIDEEP_VICTIM=5 TLLM_TEST_KILL_RANK=5 TLLM_TEST_KILL_TRIGGER=$out/trigger-rank5
     export TLLM_TEST_KILL_IN_DECODE=$TLLM_TEST_KILL_TRIGGER
     run_case mtp-decode-rank5 ft "${load[@]}" --kill decode)
    case_index=$((case_index + 1))
    kill_case mtp-prefill-rank2 prefill 128 2
    (export BENCH_CONFIG=dep32-final-320.yaml TLLM_AUTOTUNER_CACHE_PATH=$out/autotuner/ft320.json
     export WIDEEP_FAILURE_DOMAIN=node WIDEEP_VICTIM=4 WIDEEP_LOST=4 TLLM_TEST_KILL_RANK=4,5,6,7
     export TLLM_TEST_KILL_TRIGGER=$out/trigger-node1 TLLM_TEST_KILL_IN_DECODE=$out/trigger-node1
     run_case mtp-node1-decode ft "${load[@]}" --kill decode)
    case_index=$((case_index + 1))
    if [[ $SLURM_NODEID == 0 ]]; then
        # Rerun-once verdict per in-flight case: exact survivor tokens, victim request answered (not failed), the
        # kill landed mid-generation, and rank 0 logged exactly one rerun of the lost rank's requests.
        for k in 1 2; do
            name=mtp-tokens-inflight-$k
            reruns=$(grep -c "Running 1 requests of the lost ranks again" "$out/$name/rank-0.log" 2>/dev/null)
            python3 -c '
import json, sys
r = json.load(open(sys.argv[1]))
fields = dict(exact=r.get("exact_tokens_after_loss"), victim_failed=r.get("victim_request_failed"),
              answered=r.get("victim_request", {}).get("plausible"),
              finished_before_kill=r.get("victim_finished_before_kill"), reruns_logged=sys.argv[3])
ok = fields == dict(exact=True, victim_failed=False, answered=True, finished_before_kill=False, reruns_logged="1")
print("RERUN_VERDICT", sys.argv[2], "PASS" if ok else "FAIL", *(f"{k}={v}" for k, v in fields.items()))
' "$out/$name/result.json" "$name" "${reruns:-0}" 2>&1 | cut -c1-240
        done
        grep -H '"state"\|exact_tokens_after_loss\|victim_request_failed\|victim_held_no_requests' "$out"/*/result.json | sed 's/^.*runs\///' | cut -c1-240
        grep -h "discarding\|Running .* requests of the lost ranks again\|resumed serving without\|Restored .* expert slots\|not recoverable\|lacks\|RuntimeError" \
            "$out"/*/rank-0.log 2>/dev/null | cut -c1-240 | sort | uniq -c | head -20
        # v7 invariant 3: nothing captured, compiled or tuned once serving resumed (counters from testhooks.py).
        for log in "$out"/*/rank-0.log; do
            echo "AFTER_RESUME $(basename "$(dirname "$log")"): $(awk '/resumed serving without/ {r = 1} r && /WIDEEP_(CAPTURE|COMPILE|AUTOTUNE)/ {n++} END {print n + 0}' "$log")"
        done
    fi
    echo STD_DONE
    ;;
esac

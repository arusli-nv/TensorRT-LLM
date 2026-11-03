#!/bin/bash
# CPU dry run of a launch file (PREFLIGHT hard rule 2): runs it once per bundle group as that group's first node, with
# WIDEEP_DRY_RUN set (serve.sh prints each case's launch and writes a stub result; job.sh runs in a scratch directory),
# so preflights, arm parsing, knobs and quoting run end to end without GPUs. Prints the DRY_RUN_CASE lines and every
# SMOKE_FAIL / DRY_RUN_FAIL / error; exits 1 on any.
# Usage: dry_run.sh LAUNCH_FILE NNODES FIRST_NODE...   (e.g. dry_run.sh launch/lay32_cmd.txt 16 0 8)
set -uo pipefail
launch=$1 nnodes=$2; shift 2
C=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude
scratch=$(mktemp -d "${TMPDIR:-/tmp}/dryrun.XXXXXX")
id=dry$(date +%s)
bad=0
for first in "$@"; do
    log=$scratch/node$first.log
    SLURM_JOB_ID=$id SLURM_NODEID=$first SLURM_NNODES=$nnodes WIDEEP_DRY_RUN=$scratch MVP_STALL_MIN=600 \
        timeout 900 bash "$launch" > "$log" 2>&1 &
    pid=$!
    # bundle.sh waits for every node at the end: stop once this group's task reported.
    for _ in $(seq 900); do grep -q "^BUNDLE_TASK first=" "$log" 2>/dev/null && break; kill -0 $pid 2>/dev/null || break; sleep 1; done
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    # The group's task writes to its bundle log (logs/bundle/<job>-<first node>.log).
    task=$C/logs/bundle/$id-$first.log
    echo "== group first=$first"
    grep -h "DRY_RUN_CASE\|BUNDLE_TASK first" "$task" "$log" 2>/dev/null | cut -c1-400
    # Known in a dry run: the summary's cat of ab_stats files and the ncu CSV export (no stub writes them).
    if grep -h "SMOKE_FAIL\|DRY_RUN_FAIL\|syntax error\|command not found\|unbound variable\|No such file" "$task" "$log" 2>/dev/null \
        | grep -v "ab_stats-.*No such file\|ncu-rank1.csv: No such file"; then
        bad=1
    fi
    grep -q "DRY_RUN_CASE" "$task" 2>/dev/null || { echo "DRY_RUN_FAIL group $first launched no case"; bad=1; }
done
echo "DRY_RUN $([[ $bad == 0 ]] && echo PASS || echo FAIL) logs=$scratch"
exit $bad

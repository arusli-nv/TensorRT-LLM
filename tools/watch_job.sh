#!/bin/bash
# Wait for one Slurm job's end as seen from its output file, then print the lines that matter.
# Bash here cannot reach the Slurm controller (only the broker can), so the end is any of: the job's own end
# marker (MARKER, an extended regex), Slurm's cancel / time-limit / node-failure messages, or the hard cap of
# the job's time limit plus 5 minutes (queue time not counted: the cap starts when the output file appears,
# or after 2 h of waiting for it). The caller checks the job state with the broker when this returns.
# Usage: watch_job.sh JOBID TIME_LIMIT_MIN MARKER [GREP_REGEX_FOR_SUMMARY]
set -uo pipefail
job=$1; limit_min=$2; marker=$3; summary=${4:-$3}
log=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude/logs/slurm/slurm-$job.out
waited=0
until [[ -f $log ]] || (( waited > 7200 )); do sleep 30; waited=$((waited + 30)); done
deadline=$((SECONDS + limit_min * 60 + 300))
reason=CAP
while (( SECONDS < deadline )); do
    if grep -qaE "$marker" "$log" 2>/dev/null; then reason=MARKER; break; fi
    if grep -qaE "CANCELLED|DUE TO TIME LIMIT|NODE_FAIL|PREEMPT" "$log" 2>/dev/null; then reason=SLURM_END; break; fi
    sleep 20
done
echo "WATCH_END job=$job reason=$reason"
grep -aE "$summary" "$log" 2>/dev/null | tail -60

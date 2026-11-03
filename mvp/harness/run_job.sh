#!/bin/bash
# Slurm entry point on every node: runs mvp/harness/job.sh under a stall watchdog, so a wedged job ends
# instead of holding idle GPUs. Usage: run_job.sh JOB
set -uo pipefail
C=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude
session=$(ps -o sid= -p $$ | tr -d ' ')
stall_min=${MVP_STALL_MIN:-30}
(
    sleep $((stall_min * 60))
    while sleep 60; do
        if [[ -z "$(find "$C"/mvp/runs/*-"$SLURM_JOB_ID" "$C"/mvp/runs/*-"$SLURM_JOB_ID"-fill* -newermt "-$stall_min minutes" -print -quit 2>/dev/null)" ]]; then
            echo "JOB_STALL $(hostname) $(date +%T): no output for $stall_min min; ending the job"
            pkill -TERM -s "$session"; sleep 30; pkill -KILL -s "$session"
        fi
    done
) &
watchdog=$!
bash $C/mvp/harness/job.sh "$@"
rc=$?
kill $watchdog 2>/dev/null
exit $rc

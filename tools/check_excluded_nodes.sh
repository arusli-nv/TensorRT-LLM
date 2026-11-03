#!/bin/bash
# Ends the job at once when any allocated node is listed in mvp/EXCLUDE_NODES.txt (the broker's slurm_submit has no
# --exclude), and when Slurm requeued the job (the broker has no --no-requeue; a rerun under the same job id would write
# into the first attempt's run directories). Run at the top of a launch file.
# Usage: check_excluded_nodes.sh   (reads SLURM_JOB_NODELIST, SLURM_RESTART_COUNT)
R=$(cd "$(dirname "$0")/.." && pwd)
(( ${SLURM_RESTART_COUNT:-0} == 0 )) || { echo "REQUEUED_JOB restart=$SLURM_RESTART_COUNT: not rerunning into this job's run dirs"; exit 1; }
[[ -n ${WIDEEP_DRY_RUN:-} && -z ${SLURM_JOB_NODELIST:-} ]] && { echo "EXCLUDED_NODE_CHECK_SKIPPED dry run"; exit 0; }
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null)
if [[ -z $nodes ]]; then
    nodes=$(python3 - "$SLURM_JOB_NODELIST" <<'PY'
import re, sys
# prefix[a-b,c]... groups separated by commas outside brackets
for prefix, ranges in re.findall(r"([^,\[]+)(?:\[([^\]]+)\])?,?", sys.argv[1]):
    if not ranges:
        print(prefix); continue
    for part in ranges.split(","):
        lo, _, hi = part.partition("-")
        for i in range(int(lo), int(hi or lo) + 1):
            print(f"{prefix}{i:0{len(lo)}d}")
PY
)
fi
[[ -n $nodes ]] || { echo "EXCLUDED_NODE_CHECK_FAIL no nodes from '$SLURM_JOB_NODELIST'"; exit 1; }
bad=$(grep -o '^[^ #]\+' "$R/mvp/EXCLUDE_NODES.txt" | grep -Fx -f - <(echo "$nodes") | paste -sd,)
[[ -z $bad ]] || { echo "EXCLUDED_NODE $bad in $SLURM_JOB_NODELIST"; exit 1; }
echo "EXCLUDED_NODE_CHECK_OK $(wc -l <<< "$nodes") nodes"

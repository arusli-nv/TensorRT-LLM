#!/bin/bash
# Move the approved MOVE entries of tools/archive_plan.tsv (monitor-approved 2026-10-21) into archive/2026-10-21/<same
# subpath>, and Slurm logs older than 2026-10-21 into archive/2026-10-21/logs/slurm. Nothing is deleted.
# Kept despite the plan: wt/ (holds git worktrees wt/reuse, wt/sortgate) and logs/cpu-pytest-8078075.log (cited in
# EXEC_PLAN). Added: mvp/launcher. Prints the plan; acts only with --yes. Log: tools/archive_moves-2026-10-21.log.
# Usage: bash archive_apply.sh [--yes]
set -uo pipefail
C=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude
A=$C/archive/2026-10-21
log=$C/tools/archive_moves-2026-10-21.log
yes=${1:-}

mapfile -t moves < <(awk -F'\t' '$1=="MOVE"{print $2}' "$C/tools/archive_plan.tsv" \
    | grep -v -e '^wt$' -e '^logs/slurm/' -e '^logs/cpu-pytest-8078075.log$')
moves+=(mvp/launcher)
mapfile -t old_logs < <(find "$C/logs/slurm" -maxdepth 1 -type f ! -newermt 2026-10-21)

echo "MOVE ${#moves[@]} entries to $A:"; printf '  %s\n' "${moves[@]}"
echo "MOVE ${#old_logs[@]} Slurm logs older than 2026-10-21 to $A/logs/slurm"
[[ $yes == --yes ]] || { echo "Plan only. Run with --yes to move."; exit 0; }

: > "$log"
for rel in "${moves[@]}"; do
    if [[ ! -e $C/$rel && ! -L $C/$rel ]]; then echo "MISSING $rel" >> "$log"; continue; fi
    mkdir -p "$A/$(dirname "$rel")" && mv "$C/$rel" "$A/$rel" && echo "MOVED $rel" >> "$log" || echo "FAILED $rel" >> "$log"
done
mkdir -p "$A/logs/slurm"
n=0
for f in "${old_logs[@]}"; do mv "$f" "$A/logs/slurm/" && n=$((n + 1)); done
echo "MOVED logs/slurm: $n files older than 2026-10-21" >> "$log"
echo "Moved $(grep -c ^MOVED "$log") entries; problems:"; grep -v ^MOVED "$log" || echo "  none"

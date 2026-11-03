#!/bin/bash
# GPU trim on one node (/checkout = the trim worktree): incremental wheel build, the compaction unit test, then
# trim_bench.sh on the trim wheel and on the reference wheel, same node, new then old (each run alternates FT off/on itself).
# Usage: trim_kbench.sh TAG REFERENCE_BUILD   (wheel in /art/build-TAG; results in mvp/runs/TRIM-<id>-{new,old}*)
set -uo pipefail
tag=$1 ref=$2
out=/art/build-$tag
SKIP_TESTS=1 bash /art/build_and_test.sh "$tag" || exit 1
SKIP_BUILD=1 GROUPS_ONLY=gpu_moe bash /art/build_and_test.sh "$tag" 2>&1 | grep -v "^commit="
grep -E "passed|failed" "$out/test-gpu_moe.log" | tail -2
id=$SLURM_JOB_ID
for run in new:build-$tag old:$ref; do
    echo "=== TRIM-KBENCH ${run%%:*} (${run#*:})"
    SLURM_JOB_ID=$id-${run%%:*} bash /art/mvp/harness/kbench/trim_bench.sh "${run#*:}"
done
# EPLB routing alone (main's kernel, on the reference wheel installed last): placement in GB200 NUMA memory
# (default) vs plain device memory (GDRCopy path), alternating.
for round in 1 2; do
    for gdr in 0 1; do
        (cd /tmp && TRTLLM_EPLB_FORCE_GDRCOPY=$gdr timeout 300 python3 /art/mvp/harness/kbench/route_bench.py gdr$gdr-$round 2>&1 \
            | grep -E "ROUTE|Error|error" | tail -6)
    done
done
echo TRIM_KBENCH_DONE

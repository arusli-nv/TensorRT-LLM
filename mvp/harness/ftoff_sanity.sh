#!/bin/bash
# FT-off sanity, main vs the stack, on one 4-GPU node: each wheel serves the same prompts twice, interleaved
# (A B A B), with recovery off. Usage: ftoff_sanity.sh MAIN_BUILD STACK_BUILD DRIVER_DIR
# (builds are directory names under the harness root; results in mvp/runs/FTOFF-<slurm id>/)
set -uo pipefail
C=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude
main=$1; stack=$2; driver=$3
out=$C/mvp/runs/FTOFF-$SLURM_JOB_ID
mkdir -p "$out"
model=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/common/DeepSeek-V3-Lite/nvfp4_moe_only
cp $C/mvp/harness/ftoff_sanity.py /tmp/ftoff_sanity.py
# Ranks come from the LLM API's own MPI spawn, not from Slurm.
for v in $(env | grep -oE '^(SLURM_|PMIX_|PMI_)[A-Za-z0-9_]*'); do unset "$v"; done
unset TLLM_FAULT_TOLERANCE_MODE TLLM_FAULT_TOLERANCE_AUTHORITY
# Requirements and the ULFM mpi4py once; install_wheel.sh skips a wheel it already installed, so each run
# below force-reinstalls its own wheel.
bash $C/install_wheel.sh "$C/$main/dist" "$out/install-deps.log" || { echo "INSTALL_FAILED deps"; exit 1; }
for run in main-1 stack-1 main-2 stack-2; do
    build=$main; [[ $run == stack-* ]] && build=$stack
    pip install --no-deps --force-reinstall "$C/$build"/dist/*.whl > "$out/install-$run.log" 2>&1 \
        || { echo "INSTALL_FAILED $run"; exit 1; }
    # STACK_OVERLAY + STACK_SNAPSHOT: the stack's Python on top of its wheel (Python-only changes since the build).
    if [[ $run == stack-* && -n "${STACK_OVERLAY:-}" ]]; then
        bash $C/overlay_python.sh "$STACK_SNAPSHOT" "$STACK_OVERLAY" > "$out/overlay-$run.log" 2>&1 \
            || { echo "OVERLAY_FAILED $run"; exit 1; }
    fi
    echo "RUN $run $build $(date +%T)"
    (cd /tmp && python3 /tmp/ftoff_sanity.py "$driver" "$model" "$out/$run.json") > "$out/$run.log" 2>&1
    echo "rc=$? $(grep FTOFF_DONE "$out/$run.log")"
done
python3 - "$out" <<'PY'
import json, statistics, sys
out = sys.argv[1]
r = {name: json.load(open(f"{out}/{name}.json")) for name in ("main-1", "stack-1", "main-2", "stack-2")}
same = all(r[n]["tokens"] == r["main-1"]["tokens"] for n in r)
for n in r:
    print(n, "median s", round(statistics.median(r[n]["seconds"]), 3))
print("TOKENS_IDENTICAL" if same else "TOKENS_DIFFER",
      {n: sum(a != b for a, b in zip(r[n]["tokens"], r["main-1"]["tokens"])) for n in r})
PY
echo FTOFF_ALL_DONE

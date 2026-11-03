#!/bin/bash
# Unit gate for a multi-node point test: node 0 runs point_units.sh, every node waits for its verdict and exits 1 unless
# it is PASS (a file that did not run counts as a failure), so the GPU step runs only after the units passed.
# Usage: units_gate.sh GATE_DIR point_units.sh-args...   (PYTEST_ADDOPTS passes through, e.g. -k to skip a known failure)
gate=$1; shift
mkdir -p "$gate"
if [[ -n ${WIDEEP_DRY_RUN:-} ]]; then
    # A CPU dry run checks the arguments only.
    for f in "$1" "$2" ${3//:/ } "$4"; do [[ -e $f ]] || { echo "DRY_RUN_FAIL units_gate: missing $f"; exit 1; }; done
    echo "DRY_RUN_UNITS $* PYTEST_ADDOPTS=${PYTEST_ADDOPTS:-}"
    echo PASS > "$gate/verdict"
elif [[ $SLURM_NODEID == 0 ]]; then
    bash /lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude/tools/point_units.sh "$@" \
        | tee "$gate/units.txt"
    grep -q '^POINT_UNITS_PASS' "$gate/units.txt" && echo PASS > "$gate/verdict.tmp" || echo FAIL > "$gate/verdict.tmp"
    mv "$gate/verdict.tmp" "$gate/verdict"
fi
for _ in $(seq 3600); do [[ -s $gate/verdict ]] && break; sleep 1; done
echo "UNITS_GATE $(cat "$gate/verdict" 2>/dev/null || echo TIMEOUT) node $SLURM_NODEID"
[[ $(cat "$gate/verdict" 2>/dev/null) == PASS ]]

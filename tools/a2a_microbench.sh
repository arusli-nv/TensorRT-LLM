#!/bin/bash
# Runs tools/a2a_microbench.py on one node (EP 4), FT off then on, at 32 experts (DEP32's per-rank layout) and 256.
# Usage: a2a_microbench.sh LABEL   (results in /art/build-LABEL/a2a-microbench/{off,ft}-e{32,256}.json and .log)
set -uo pipefail
label=$1
export HOME=/tmp/wideep-home PYTHONUSERBASE=/tmp/wideep-home/.local PIP_CACHE_DIR=/tmp/wideep-home/.cache/pip
mkdir -p "$PYTHONUSERBASE" "$PIP_CACHE_DIR"
out=/art/build-$label/a2a-microbench
mkdir -p "$out"
bash /art/install_wheel.sh "/art/build-$label/dist" "$out/install.log" || { echo INSTALL_FAILED; exit 1; }
for var in $(compgen -e | grep -E '^(PMIX_|PMI_|SLURM_|OMPI_MCA_ess|PRTE_MCA_ras)'); do unset "$var"; done
for experts in 32 256; do
    for mode in off ft; do
        name=$mode-e$experts
        (cd /tmp && TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT=0 PYTHONUNBUFFERED=1 timeout -k 30 600 \
            mpirun --allow-run-as-root --oversubscribe -n 4 -x TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT \
            -x PYTHONUNBUFFERED -x LD_LIBRARY_PATH python3 -u /art/tools/a2a_microbench.py "$mode" "$out/$name.json" $experts \
            16 32 48 64 96 128 256 > "$out/$name.log" 2>&1)
        echo "microbench $name rc=$?"
        grep -E "tokens/rank|Error|error" "$out/$name.log" | head -12
    done
done
echo A2A_MICROBENCH_DONE

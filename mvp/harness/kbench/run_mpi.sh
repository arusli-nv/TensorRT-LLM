#!/bin/bash
# One Slurm task per node: EP-wide a2a_bench_mpi.py for FT off/on, PDL off/on, nsys on one rank.
# Usage: run_mpi.sh DIST_DIR OUT_DIR
set -uo pipefail
dist=$1; out=$2
C=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude
K=$(dirname "$(realpath "$0")")
mkdir -p "$out"
bash $C/install_wheel.sh "$dist" "$out/install-$SLURM_NODEID.log" || exit 1
unset PYTHONPATH
export PYTHONUNBUFFERED=1 TLLM_LOG_LEVEL=warning TRTLLM_MOE_A2A_FORCE_CFT=0
export TLLM_TEST_NSYS_RANK=5 TLLM_TEST_NSYS_DELAY=0 TLLM_TEST_NSYS_DURATION=900
for var in $(compgen -e | grep -E '^(PMIX_|PMI_|OMPI_MCA_ess|PRTE_MCA_ras)'); do unset "$var"; done
for pdl in 0 1; do
    for mode in off on; do
        run=$out/$mode-pdl$pdl
        (
            export TRTLLM_ENABLE_PDL=$pdl
            if [[ $mode == on ]]; then
                export TLLM_FAULT_TOLERANCE_MODE=1 TLLM_FAULT_TOLERANCE_AUTHORITY=tcp://127.0.0.1:1
                export TLLM_FAULT_TOLERANCE_AUTHORITY_KEY=$(printf '0%.0s' {1..64})
            fi
            mkdir -p "$run"
            bash $K/../launch/task.sh "$run" 4 bash $K/rank.sh "$run" python3 $K/a2a_bench_mpi.py $mode
        )
        [[ "$SLURM_NODEID" == 0 ]] && { echo "RUN mode=$mode pdl=$pdl"; grep -h BENCH "$run"/rank-0.log; }
    done
done
if [[ "$SLURM_NODEID" == 0 ]]; then
    mkdir -p "$out/analysis"
    for d in "$out"/{off,on}-pdl{0,1}; do
        rep=$(ls "$d"/nsys-rank*.nsys-rep 2>/dev/null | head -1)
        [[ -n "$rep" ]] && timeout 300 nsys export --type sqlite -o "$out/analysis/nsys-$(basename "$d").sqlite" -f true \
            "$rep" > /dev/null 2>&1
    done
    timeout 300 python3 $K/analyze.py "$out/analysis"
fi
echo NODE_END

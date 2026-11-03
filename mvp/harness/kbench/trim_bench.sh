#!/bin/bash
# Price the pieces of FT's GPU mechanism per MoE layer (mvp/analysis/GPU_TRIM.md) on one 4-GPU node, same wheel:
#   off           FT off (main's path: sanitize)
#   on            FT on (fence dispatch with the rank mask, compaction, sort, combine)
#   on-nocompact  FT on without the compaction (measurement only), to price the compaction alone
# Two alternating rounds; nsys on every rank, rank 0's report exported (kernels_by_name.py reads it).
# Usage: trim_bench.sh BUILD   (results in mvp/runs/TRIM-<slurm id>/)
set -uo pipefail
C=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude
K=$C/mvp/harness/kbench
build=$1
out=$C/mvp/runs/TRIM-$SLURM_JOB_ID
mkdir -p "$out"
unset PYTHONPATH
export PYTHONUNBUFFERED=1 TLLM_LOG_LEVEL=warning TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT=0
for v in $(compgen -e | grep -E '^(SLURM_|PMIX_|PMI_|OMPI_MCA_ess|PRTE_MCA_ras)'); do unset "$v"; done
bash $C/install_wheel.sh "$C/$build/dist" "$out/install.log" || { echo INSTALL_FAILED; exit 1; }
for round in 1 2; do
    modes="off on on-nocompact"; [[ $round == 2 ]] && modes="on-nocompact on off"
    for mode in $modes; do
        run=$out/$mode-$round
        mkdir -p "$run"
        (
            ft=${mode%%-*}
            [[ $mode == on-nocompact ]] && export KBENCH_FT_COMPACT=0
            if [[ $ft == on ]]; then
                export TLLM_FAULT_TOLERANCE_MODE=1 TLLM_FAULT_TOLERANCE_AUTHORITY=tcp://127.0.0.1:1
                export TLLM_FAULT_TOLERANCE_AUTHORITY_KEY=$(printf '0%.0s' {1..64})
            fi
            prog=(nsys profile -o "$run/nsys-rank%q{OMPI_COMM_WORLD_RANK}" -f true --trace=cuda,nvtx --sample=none
                  --cpuctxsw=none --cuda-graph-trace=node python3 $K/a2a_bench_mpi.py $ft)
            cd /tmp && timeout 900 mpirun --allow-run-as-root --oversubscribe -np 4 "${prog[@]}" > "$run/log" 2>&1
        )
        nsys export --type sqlite --force-overwrite true -o "$run/nsys-rank0.sqlite" "$run/nsys-rank0.nsys-rep" \
            > /dev/null 2>&1
        echo "TRIM $mode round=$round"; grep -h BENCH "$run/log"
    done
done
python3 $K/kernels_by_name.py off=$out/off-1/nsys-rank0.sqlite on=$out/on-1/nsys-rank0.sqlite \
    nocompact=$out/on-nocompact-1/nsys-rank0.sqlite > "$out/kernels.txt" 2>&1
echo TRIM_DONE

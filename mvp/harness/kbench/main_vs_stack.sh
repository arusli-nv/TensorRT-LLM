#!/bin/bash
# FT-off evidence on one 4-GPU node: main's wheel against the stack's.
#  1. SASS of every moeA2A* kernel in both wheels (sm_100), compared per demangled name.
#  2. The routing sort on one GPU (sort_bench.py): moe_sort on main and the stack, moe_sort_source_slots.
#  3. The EP4 all-to-all per MoE layer (a2a_bench_mpi.py): main FT off, stack FT off, stack FT on; A B A B.
# Usage: main_vs_stack.sh MAIN_BUILD STACK_BUILD   (results in mvp/runs/KBENCH-MAIN-<slurm id>/)
set -uo pipefail
C=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude
K=$C/mvp/harness/kbench
main=$1; stack=$2
out=$C/mvp/runs/KBENCH-MAIN-$SLURM_JOB_ID
mkdir -p "$out"
unset PYTHONPATH
# The fence path in every mode: FT on requires it, so FT off is compared on the same kernels.
export PYTHONUNBUFFERED=1 TLLM_LOG_LEVEL=warning TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT=0
for v in $(compgen -e | grep -E '^(SLURM_|PMIX_|PMI_|OMPI_MCA_ess|PRTE_MCA_ras)'); do unset "$v"; done

# 1. SASS
T=$(mktemp -d /tmp/sass.XXXX)
for build in $main $stack; do
    unzip -q -o "$C/$build"/dist/*.whl 'tensorrt_llm/libs/*' -d "$T/$build"
    lib=$(grep -l -a moeA2ADispatchKernel "$T/$build"/tensorrt_llm/libs/*.so | head -1)
    echo "SASS lib $build: $(basename "$lib")"
    # Addresses, encodings and constant-bank (kernel parameter) offsets are masked: a larger parameter
    # struct moves every parameter load without changing the code.
    cuobjdump -sass -arch sm_100 "$lib" | awk '/Function : /{keep = ($3 ~ /moeA2A/)} keep' \
        | sed -E 's#/\*[0-9a-f]{4,}\*/##; s#/\* 0x[0-9a-f]+ \*/##; s#c\[0x0\]\[0x[0-9a-f]+\]#c[0x0][P]#g; s/[[:space:]]+/ /g; /^ ?$/d' \
        > "$T/$build.sass"
    awk '/Function : /{f=$3; next} {print > (dir "/" f ".k")}' dir="$(mkdir -p "$T/$build.k" && echo "$T/$build.k")" "$T/$build.sass"
done
filt=$(command -v cu++filt || echo "${CUDA_HOME:-/usr/local/cuda}/bin/cu++filt")
[[ -x $filt ]] || filt=c++filt
{
    for f in "$T/$main.k"/*.k; do
        name=$(basename "$f" .k)
        if [[ -e "$T/$stack.k/$name.k" ]]; then
            cmp -s "$f" "$T/$stack.k/$name.k" && r=IDENTICAL || r="DIFFERS($(diff "$f" "$T/$stack.k/$name.k" | grep -c '^[<>]') lines)"
            [[ $r == DIFFERS* && ! -e "$out/sass-diff-sample.txt" ]] && { echo "$name"; diff "$f" "$T/$stack.k/$name.k" | head -60; } > "$out/sass-diff-sample.txt"
        else
            r=MAIN_ONLY
        fi
        echo "$r $(echo "$name" | $filt)"
    done
    for f in "$T/$stack.k"/*.k; do
        [[ -e "$T/$main.k/$(basename "$f")" ]] || echo "STACK_ONLY $(basename "$f" .k | $filt)"
    done
} > "$out/sass.txt"
echo "SASS $(grep -c ^IDENTICAL "$out/sass.txt") identical, $(grep -c ^DIFFERS "$out/sass.txt") differ, $(grep -c ^MAIN_ONLY "$out/sass.txt") main-only, $(grep -c ^STACK_ONLY "$out/sass.txt") stack-only"
[[ -n "${SASS_ONLY:-}" ]] && { echo KBENCH_DONE; exit 0; }

# 2 + 3, A B A B
bash $C/install_wheel.sh "$C/$main/dist" "$out/install-deps.log" || { echo INSTALL_FAILED; exit 1; }
for round in 1 2; do
    for build in $main $stack; do
        pip install --no-deps --force-reinstall "$C/$build"/dist/*.whl > "$out/install-$build-$round.log" 2>&1 \
            || { echo "INSTALL_FAILED $build"; exit 1; }
        label=main; [[ $build == "$stack" ]] && label=stack
        (cd /tmp && python3 $K/sort_bench.py $label-$round) | sed "s/^/SORT /"
        modes=off; [[ $label == stack ]] && modes="off on"
        for mode in $modes; do
            run=$out/a2a-$label-$mode-$round
            mkdir -p "$run"
            (
                if [[ $mode == on ]]; then
                    export TLLM_FAULT_TOLERANCE_MODE=1 TLLM_FAULT_TOLERANCE_AUTHORITY=tcp://127.0.0.1:1
                    export TLLM_FAULT_TOLERANCE_AUTHORITY_KEY=$(printf '0%.0s' {1..64})
                fi
                prog=(python3 $K/a2a_bench_mpi.py $mode)
                # KBENCH_NSYS=1: every rank under nsys (graph kernels as nodes); rank 0's report exported.
                [[ -n "${KBENCH_NSYS:-}" ]] && prog=(nsys profile -o "$run/nsys-rank%q{OMPI_COMM_WORLD_RANK}" -f true
                    --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node "${prog[@]}")
                cd /tmp && timeout 900 mpirun --allow-run-as-root --oversubscribe -np 4 "${prog[@]}" > "$run/log" 2>&1
            )
            [[ -n "${KBENCH_NSYS:-}" ]] && nsys export --type sqlite --force-overwrite true \
                -o "$run/nsys-rank0.sqlite" "$run/nsys-rank0.nsys-rep" > /dev/null 2>&1
            echo "A2A $label $mode round=$round rc=$?"; grep -h BENCH "$run/log"
        done
    done
done
echo KBENCH_DONE

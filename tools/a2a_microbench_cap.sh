#!/bin/bash
# Capacity x routing sweep of tools/a2a_microbench.py at EP 32 (8 nodes), 3 repeats: the workspace sized for 128 tokens per rank (the bench default)
# vs 8192 (serving's max_num_tokens), FT off then on, 8 experts per rank (as DEP32), through launch/task.sh (plain mpirun).
# FT's bound receive slots span the full capacity per source; unbound slots span the runtime token count.
# Usage: a2a_microbench_cap.sh LABEL   (results in build-LABEL/a2a-microbench-cap/cap<N>-{off,ft}.json)
set -uo pipefail
label=$1
C=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude
export HOME=/tmp/wideep-home PYTHONUSERBASE=/tmp/wideep-home/.local PIP_CACHE_DIR=/tmp/wideep-home/.cache/pip
mkdir -p "$PYTHONUSERBASE" "$PIP_CACHE_DIR"
out=$C/build-$label/a2a-microbench-cap${OUT_SUFFIX:-}
mkdir -p "$out"
node=$SLURM_NODEID
bash "$C/install_wheel.sh" "$C/build-$label/dist" "$out/install-$node.log" || { echo "INSTALL_FAILED $node"; exit 1; }
export TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT=0 TRTLLM_ENABLE_PDL=1 PYTHONUNBUFFERED=1 WIDEEP_NO_ULFM=1
# mpirun launches through the node agents (launch/task.sh), not through Slurm.
export PRTE_MCA_ras=^slurm PRTE_MCA_plm=^slurm
# Three repeats of each arm; routing uniform, then replaying the served expert load (tools/a2a_expert_load.json).
for rep in 1 2 3; do
    for routing in uniform served; do
        load=; [[ $routing == served ]] && load=$C/tools/a2a_expert_load.json
        for cap in 128 8192; do
            for mode in off ft; do
                name=r$rep-$routing-cap$cap-$mode
                run=$out/$name
                mkdir -p "$run"
                TLLM_TEST_A2A_EXPERT_LOAD=$load TLLM_TEST_A2A_CAPACITY=$cap timeout -k 30 900 bash "$C/launch/task.sh" "$run" 4 \
                    python3 -u "$C/tools/a2a_microbench.py" "$mode" "$out/$name.json" 256 32 64 > "$run/task-$node.log" 2>&1
                echo "microbench $name node $node rc=$?"
                (( node == 0 )) && grep -h "tokens/rank" "$run/task-0.log" | head -2
            done
        done
    done
done
echo A2A_MICROBENCH_CAP_DONE

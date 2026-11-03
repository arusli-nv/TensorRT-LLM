#!/bin/bash
# EP-size sweep of tools/a2a_microbench.py: EP 4/8/16/32 on the first 1/2/4/8 nodes (4 GPUs each), FT off then on,
# 8 experts per rank (8 local slots, 10 with FT, as DEP32), multi-node through launch/task.sh (plain mpirun).
# One Slurm task per node; every node walks the same sequence, and nodes outside a run's EP wait for it to finish.
# Usage: a2a_microbench_ep.sh LABEL   (results in /art/build-LABEL/a2a-microbench-ep/ep<N>-{off,ft}.json)
set -uo pipefail
label=$1
C=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude
export HOME=/tmp/wideep-home PYTHONUSERBASE=/tmp/wideep-home/.local PIP_CACHE_DIR=/tmp/wideep-home/.cache/pip
mkdir -p "$PYTHONUSERBASE" "$PIP_CACHE_DIR"
out=$C/build-$label/a2a-microbench-ep
mkdir -p "$out"
node=$SLURM_NODEID
bash "$C/install_wheel.sh" "$C/build-$label/dist" "$out/install-$node.log" || { echo "INSTALL_FAILED $node"; exit 1; }
export TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT=0 PYTHONUNBUFFERED=1 WIDEEP_NO_ULFM=1
# mpirun launches through the node agents (launch/task.sh), not through Slurm.
export PRTE_MCA_ras=^slurm PRTE_MCA_plm=^slurm
for nodes in 1 2 4 8; do
    ep=$((4 * nodes))
    for mode in off ft; do
        run=$out/ep$ep-$mode
        mkdir -p "$run"
        if (( node < nodes )); then
            SLURM_NNODES=$nodes timeout -k 30 900 bash "$C/launch/task.sh" "$run" 4 \
                python3 -u "$C/tools/a2a_microbench.py" "$mode" "$out/ep$ep-$mode.json" $((8 * ep)) 16 32 64 128 \
                > "$run/task-$node.log" 2>&1
            echo "microbench ep$ep-$mode node $node rc=$?"
        else
            for _ in $(seq 900); do [[ -e $run/done ]] && break; sleep 1; done
        fi
        (( node == 0 )) && grep -h "tokens/rank" "$run/task-0.log" | head -8
    done
done
echo A2A_MICROBENCH_EP_DONE

#!/bin/bash
# One Slurm task per node. Installs the wheel, starts the recovery authority on node 0,
# then launches EP32 under ULFM mpirun through the in-container launch agent.
# Usage: run.sh DIST_DIR RUN_DIR [frontend.py args...]
set -euo pipefail
dist=$1; run=$2; shift 2
E=$(dirname "$(realpath "$0")"); L=$E/../launch
mkdir -p "$run"
bash "$E/../install_wheel.sh" "$dist" "$run/install-$SLURM_NODEID.log"
[[ -n "${OVERLAY:-}" ]] && bash "$E/../overlay_python.sh" "$OVERLAY" >> "$run/install-$SLURM_NODEID.log" 2>&1
if [[ -n "${TLLM_TEST_ORACLE_EXCLUDE_RANK:-}${TLLM_TEST_TRACE_CAPTURE:-}${TLLM_TEST_KILL_IN_REPLAY:-}${TLLM_TEST_DUMP_STACKS:-}${TLLM_TEST_NO_COMMIT_SYNC:-}${TLLM_TEST_NO_PEER_WATCH:-}" ]]; then
    site=$(python3 -c "import site; print(site.getsitepackages()[0])")
    cp "$E/oracle/wideep_eplb_oracle.py" "$site/"
    echo "import wideep_eplb_oracle" > "$site/wideep_eplb_oracle.pth"
fi
export LLM_MODELS_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/common
export TRTLLM_ENABLE_PDL=1 TRTLLM_KVCACHE_POOL_USE_FABRIC_MEMORY=0 TRTLLM_MOE_A2A_FORCE_CFT=0
export PYTHONUNBUFFERED=1 TLLM_LOG_LEVEL=${TLLM_LOG_LEVEL:-warning}
node=$SLURM_NODEID
for var in $(compgen -e | grep -E '^(PMIX_|PMI_|OMPI_MCA_ess|PRTE_MCA_ras)'); do unset "$var"; done
if [[ "$node" == 0 && -z "${WIDEEP_NO_RECOVERY:-}" ]]; then
    (umask 077; python3 -c "import os;print(os.urandom(32).hex())" > "$run/recovery.key")
    export TLLM_FAULT_TOLERANCE_MODE=1
    export TLLM_FAULT_TOLERANCE_AUTHORITY_KEY=$(cat "$run/recovery.key")
    export TLLM_FAULT_TOLERANCE_AUTHORITY=tcp://$(hostname):${WIDEEP_AUTHORITY_PORT:-24680}
    CUDA_VISIBLE_DEVICES= python3 -m tensorrt_llm.executor.recovery.authority \
        --endpoint "tcp://*:${WIDEEP_AUTHORITY_PORT:-24680}" --world-size 32 \
        > "$run/authority.log" 2>&1 &
fi
model=$LLM_MODELS_ROOT/DeepSeek-R1-0528-FP4-V2
exec bash "$L/task.sh" "$run" 4 bash "$E/rank.sh" "$run" \
    trtllm-llmapi-launch python3 "$E/frontend.py" --model "$model" --config "$E/${WIDEEP_CONFIG:-config-recover.yaml}" --output "$run" "$@"

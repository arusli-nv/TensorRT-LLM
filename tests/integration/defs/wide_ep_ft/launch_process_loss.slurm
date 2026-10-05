#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#SBATCH --job-name=wideep-process-loss
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:25:00

set -euo pipefail
: "${WIDEEP_CONTAINER_IMAGE:?Set the qualified image}"
: "${WIDEEP_MODEL_PATH:?Set the checkpoint}"
: "${WIDEEP_CONFIG_PATH:?Set the generated native configuration}"
: "${WIDEEP_BUILD_MANIFEST:?Set the bound build manifest}"
: "${WIDEEP_RAY_DEPENDENCIES:?Set the qualified Ray virtual environment directory}"
: "${WIDEEP_RUN_ROOT:?Set the shared artifact root}"
source_root="${WIDEEP_SOURCE_ROOT:-${SLURM_SUBMIT_DIR}}"
export ENROOT_ALLOW_DEV=yes LLM_MODELS_ROOT="$(dirname "$WIDEEP_MODEL_PATH")" PYTHONUNBUFFERED=1
export TRTLLM_EPLB_SHM_NAME="wideep_process_loss_$SLURM_JOB_ID"
export TRTLLM_FORCE_COMM_METHOD=NVLINK_ONE_SIDED TRTLLM_MOE_A2A_FORCE_CFT=0 TLLM_FAULT_TOLERANCE_MODE=0 TLLM_LOG_LEVEL=info
unset PYTHONPATH TLLM_RANK_CRASH_HARD_KILL_GRACE TLLM_LOG_LEVEL_BY_MODULE
export PYTHONPATH="${WIDEEP_HOST_PYTHONPATH:-}"
overlay=()
if [[ "${WIDEEP_PYTHON_OVERLAY:-0}" == 1 ]]; then
    overlay=(--python-overlay)
fi
python3 "$source_root/tests/integration/defs/wide_ep_ft/process_loss_controller.py" \
    --image "$WIDEEP_CONTAINER_IMAGE" --model "$WIDEEP_MODEL_PATH" --config "$WIDEEP_CONFIG_PATH" \
    --build-manifest "$WIDEEP_BUILD_MANIFEST" --ray-dependencies "$WIDEEP_RAY_DEPENDENCIES" \
    --output-dir "$WIDEEP_RUN_ROOT/process-loss-$SLURM_JOB_ID" \
    --scenario "${WIDEEP_SCENARIO:-healthy}" "${overlay[@]}" \
    --startup-timeout "${WIDEEP_STARTUP_TIMEOUT:-720}" \
    --client-timeout "${WIDEEP_CLIENT_TIMEOUT:-30}" \
    --cleanup-timeout "${WIDEEP_CLEANUP_TIMEOUT:-180}" &
controller_pid=$!
trap 'kill -USR1 "$controller_pid" 2>/dev/null || true' USR1 INT TERM
status=0
while true; do
    if wait "$controller_pid"; then
        status=0
        break
    else
        status=$?
    fi
    if ! kill -0 "$controller_pid" 2>/dev/null; then
        break
    fi
done
if (( status != 0 )); then
    python3 "$source_root/tests/integration/defs/wide_ep_ft/process_loss_controller.py" \
        --mode=cleanup --output-dir "$WIDEEP_RUN_ROOT/process-loss-$SLURM_JOB_ID" \
        --cleanup-timeout "${WIDEEP_CLEANUP_TIMEOUT:-180}"
fi
exit "$status"

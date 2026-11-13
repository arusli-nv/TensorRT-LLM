#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=04:00:00
#SBATCH --job-name=wide-ep-fault-tolerance

# Serve a model with Wide-EP fault tolerance on Slurm with pyxis/enroot containers.
#
# Every node runs one container with a remote_exec.py server. Node 0's container runs
# `mpirun --enable-recovery`, whose PRRTE ssh launcher starts PRRTE's daemon in each other node's
# container through remote_exec.py. The ranks are then children of PRRTE's daemons, not Slurm
# tasks, so a failed rank does not end the job step. trtllm-serve answers HTTP on node 0.
#
# A remote_exec.py server listens only on the address of its node's Slurm name and runs any command
# signed with the job's REMOTE_EXEC_KEY, with a fresh nonce per connection so that a recorded request
# does not run again. Requests are not encrypted, and the rendezvous store does not authenticate its
# values, so run the job on a trusted network.
#
# Run sbatch from this directory, on a file system that every node mounts, with these variables set:
#   CONTAINER_IMAGE   a TensorRT-LLM release container
#   CONTAINER_MOUNTS  the pyxis mounts that hold MODEL and CONFIG (this directory is added)
#   MODEL             the model directory
#   CONFIG            the trtllm-serve YAML file, with enable_wide_ep_fault_tolerance: true
# Optional: GPUS_PER_NODE (4), PORT (8000), RENDEZVOUS_PORT (29600) and REMOTE_EXEC_PORT (29700).
# GPUS_PER_NODE must equal the GPUs that each node's container sees, since each node runs one rank
# per visible GPU. Keep both ports outside the nodes' ephemeral port range. On GB200 NVL72, allocate
# the nodes in one NVLink domain, for example with --segment=<nodes> where the cluster supports it.
# Each container needs the job's IMEX channel in /dev/nvidia-caps-imex-channels; ask your cluster
# admin if it is missing.
#
# Example:
#   CONTAINER_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:<tag> CONTAINER_MOUNTS=/models:/models \
#   MODEL=/models/DeepSeek-R1-0528-FP4 CONFIG=/models/config.yaml sbatch launch.sh
#
# To stop the deployment, send SIGINT to trtllm-serve on node 0, the first node of the job. It
# drains the requests in flight and shuts the deployment down, and the job then ends. The job's
# output names its PID in "Rank0 task guard PID: <guard PID>; workload PGID: <server PID>":
#   srun --overlap --jobid=<job id> --nodes=1 --ntasks=1 --nodelist=<node 0> kill -INT <server PID>
# Or match trtllm-serve's own command line, not those of mpirun, the node supervisor or
# trtllm-llmapi-launch, which also hold "trtllm-serve":
#   srun --overlap --jobid=<job id> --nodes=1 --ntasks=1 --nodelist=<node 0> \
#       pkill -INT -o -u "$(id -u)" -f '^([^ ]*python[^ ]* )?([^ ]*/)?trtllm-serve( |$)'
# Whether scancel or the time limit drains is untested.

set -euo pipefail
: "${CONTAINER_IMAGE:?}" "${CONTAINER_MOUNTS:?}" "${MODEL:?}" "${CONFIG:?}"

SCRIPT_DIR=$SLURM_SUBMIT_DIR
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd,)
NODES=$SLURM_JOB_NUM_NODES
WORLD_SIZE=$((NODES * GPUS_PER_NODE))
PORT=${PORT:-8000}

# The keys stay in the environment, never on a command line.
REMOTE_EXEC_KEY=$(openssl rand -hex 32)
REMOTE_EXEC_PORT=${REMOTE_EXEC_PORT:-29700}
TLLM_FAULT_TOLERANCE_SHARED_SECRET=$(openssl rand -hex 32)
TLLM_FAULT_TOLERANCE_RENDEZVOUS=${HOSTS%%,*}:${RENDEZVOUS_PORT:-29600}
TRTLLM_EPLB_SHM_NAME=trtllm_eplb_${SLURM_JOB_ID}
export SCRIPT_DIR GPUS_PER_NODE HOSTS NODES WORLD_SIZE PORT MODEL CONFIG REMOTE_EXEC_KEY REMOTE_EXEC_PORT \
    TLLM_FAULT_TOLERANCE_SHARED_SECRET TLLM_FAULT_TOLERANCE_RENDEZVOUS TRTLLM_EPLB_SHM_NAME

# The step ends 60 s after node 0's task, which ends with mpirun.
srun --ntasks-per-node=1 --wait=60 \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$CONTAINER_MOUNTS,$SCRIPT_DIR:$SCRIPT_DIR" \
    bash -c '
        node_id=$SLURM_NODEID
        node_name=$SLURMD_NODENAME
        # PRRTE, not Slurm, launches the ranks.
        for var in $(compgen -e | grep -E "^(SLURM_|PMIX_|PMI_)"); do unset "$var"; done
        if (( node_id > 0 )); then
            exec python3 "$SCRIPT_DIR/remote_exec.py" --serve --bind "$node_name"
        fi
        mpirun --enable-recovery \
            --prtemca plm ssh --prtemca plm_ssh_agent "python3 $SCRIPT_DIR/remote_exec.py" \
            --prtemca plm_ssh_no_tree_spawn 1 \
            --host "$HOSTS" -np "$NODES" --map-by ppr:1:node --bind-to none \
            -x TLLM_FAULT_TOLERANCE_SHARED_SECRET -x TLLM_FAULT_TOLERANCE_RENDEZVOUS -x TRTLLM_EPLB_SHM_NAME \
            trtllm-llmapi-launch trtllm-serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
            --tp_size "$WORLD_SIZE" --ep_size "$WORLD_SIZE" --gpus_per_node "$GPUS_PER_NODE" \
            --config "$CONFIG"
    '

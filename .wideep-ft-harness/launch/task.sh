#!/bin/bash
# One Slurm task per node. Every node runs the agent; node 0 runs mpirun with the agent as PRTE's launcher.
# Usage: task.sh RUN_DIR PROCS_PER_NODE program args...
set -euo pipefail
run=$1; ppn=$2; shift 2
L=$(dirname "$(realpath "$0")")
node=$SLURM_NODEID nnodes=$SLURM_NNODES
# PRTE owns the MPI launch: drop the outer Slurm step's PMIx/Slurm environment.
for var in $(compgen -e | grep -E '^(PMIX_|PMI_|SLURM_|OMPI_MCA_ess|PRTE_MCA_ras)'); do unset "$var"; done
export WIDEEP_AGENT_PORT=${WIDEEP_AGENT_PORT:-23456} WIDEEP_AGENT_KEY_FILE=$run/agent.key
mkdir -p "$run/agents"
if [[ "$node" == 0 ]]; then
    (umask 077; python3 -c "import os;print(os.urandom(32).hex())" > "$run/agent.key.tmp")
    mv "$run/agent.key.tmp" "$run/agent.key"
fi
while [[ ! -s "$run/agent.key" ]]; do sleep 0.2; done
python3 "$L/node_agent.py" "$WIDEEP_AGENT_PORT" "$run/agents/$node" > "$run/agent-$node.log" 2>&1 &
agent=$!
trap 'kill $agent 2>/dev/null || true' EXIT
if [[ "$node" != 0 ]]; then
    while [[ ! -e "$run/done" ]]; do sleep 1; done
    exit 0
fi
while [[ $(ls "$run/agents" | wc -l) -lt $nnodes ]]; do sleep 0.5; done
hosts=$(for f in $(ls "$run/agents" | sort -n); do printf '%s:%s,' "$(cat "$run/agents/$f")" "$ppn"; done)
np=$((nnodes * ppn))
forward=()
for var in $(compgen -e | grep -E '^(TLLM_|TRTLLM_|LLM_MODELS_ROOT$|PYTHONUNBUFFERED$|HOME$)'); do forward+=(-x "$var"); done
set +e
# A degraded ULFM world can leave mpirun waiting after every rank has exited; give up on it
# two minutes after the frontend finished.
(
    while [[ ! -e "$run/frontend.done" ]]; do sleep 5; done
    sleep 120
    pkill -TERM -f "mpirun --with-ft ulfm" && echo "MPIRUN_RC=launcher-timeout"
) &
guard=$!
mpirun --with-ft ulfm --prtemca plm ssh --prtemca plm_ssh_agent "python3 $L/remote_exec.py" \
    --prtemca plm_ssh_no_tree_spawn 1 --mca pml ob1 --mca btl self,tcp --mca coll '^hcoll,ucc' \
    -x WIDEEP_AGENT_PORT -x WIDEEP_AGENT_KEY_FILE -x PATH -x LD_LIBRARY_PATH "${forward[@]}" \
    --host "${hosts%,}" -np "$np" --map-by ppr:$ppn:node --bind-to none "$@"
rc=$?
kill $guard 2>/dev/null
touch "$run/done"
echo "MPIRUN_RC=$rc"
exit $rc

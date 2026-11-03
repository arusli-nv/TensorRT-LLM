#!/bin/bash
# One Slurm task per node. Every node runs the agent; node 0 runs mpirun with the agent as PRTE's launcher.
# Usage: task.sh RUN_DIR PROCS_PER_NODE program args...
set -euo pipefail
run=$1; ppn=$2; shift 2
L=$(dirname "$(realpath "$0")")
node=$SLURM_NODEID nnodes=$SLURM_NNODES
# PRTE owns the MPI launch: drop the outer Slurm step's PMIx/Slurm environment.
for var in $(compgen -e | grep -E '^(PMIX_|PMI_|SLURM_|OMPI_MCA_ess|PRTE_MCA_ras)'); do unset "$var"; done
export WIDEEP_AGENT_DIR=$run/agents WIDEEP_AGENT_KEY_FILE=$run/agent.key
mkdir -p "$run/agents"
if [[ "$node" == 0 ]]; then
    (umask 077; python3 -c "import os;print(os.urandom(32).hex())" > "$run/agent.key.tmp")
    mv "$run/agent.key.tmp" "$run/agent.key"
fi
# Every cross-node wait is bounded, so a node that fails early cannot leave the others waiting.
waited() { (( SECONDS - start > ${WIDEEP_REGISTER_TIMEOUT_S:-600} )); }
start=$SECONDS
while [[ ! -s "$run/agent.key" ]]; do waited && { echo "AGENT_KEY_TIMEOUT"; exit 1; }; sleep 0.2; done
python3 "$L/node_agent.py" "$run/agents/$node" > "$run/agent-$node.log" 2>&1 &
agent=$!
sleep 2
if ! kill -0 $agent 2>/dev/null; then echo "AGENT_FAILED $node"; touch "$run/done"; exit 1; fi
trap 'kill $agent 2>/dev/null || true' EXIT
if [[ "$node" != 0 ]]; then
    while [[ ! -e "$run/done" ]]; do sleep 1; done  # bounded by the batch's per-case timeout
    exit 0
fi
while [[ $(ls "$run/agents" | grep -vc "\.tmp$") -lt $nnodes ]]; do
    waited && { echo "AGENT_REGISTER_TIMEOUT"; touch "$run/done"; exit 1; }
    sleep 0.5
done
hosts=$(for f in $(ls "$run/agents" | grep -v "\.tmp$" | sort -n); do printf '%s:%s,' "$(cut -d' ' -f1 "$run/agents/$f")" "$ppn"; done)
np=$((nnodes * ppn))
forward=()
for var in $(compgen -e | grep -E '^(TLLM_|TRTLLM_|LLM_MODELS_ROOT$|PYTHONUNBUFFERED$|HOME$)'); do forward+=(-x "$var"); done
set +e
ft=(--with-ft ulfm); [[ -n "${WIDEEP_NO_ULFM:-}" ]] && ft=()
# WIDEEP_MPIRUN_EXTRA: extra mpirun arguments (diagnostics, e.g. ULFM detector settings).
read -ra extra <<< "${WIDEEP_MPIRUN_EXTRA:-}"
echo "MPIRUN_START $(date +%s.%N)" >> "$run/timeline"
mpirun "${ft[@]}" "${extra[@]}" --prtemca plm ssh --prtemca plm_ssh_agent "python3 $L/remote_exec.py" \
    --prtemca plm_ssh_no_tree_spawn 1 --mca pml ob1 --mca btl self,tcp --mca coll '^hcoll,ucc' \
    -x WIDEEP_AGENT_DIR -x WIDEEP_AGENT_KEY_FILE -x PATH -x LD_LIBRARY_PATH "${forward[@]}" \
    --host "${hosts%,}" -np "$np" --map-by ppr:$ppn:node --bind-to none "$@" &
launcher=$!
# After a rank loss, mpirun can outlive every rank even after MPI_Abort. Stop it
# WIDEEP_LAUNCHER_GRACE_S (default 60) after the frontend finished, escalating to SIGKILL, so the
# allocation never idles. Profiled runs need longer to write their reports.
(
    while [[ ! -e "$run/frontend.done" ]] && kill -0 $launcher 2>/dev/null; do sleep 5; done
    for _ in $(seq "${WIDEEP_LAUNCHER_GRACE_S:-60}"); do kill -0 $launcher 2>/dev/null || exit 0; sleep 1; done
    echo "MPIRUN_RC=launcher-timeout"
    kill -TERM $launcher; sleep 10; kill -KILL $launcher 2>/dev/null
) &
guard=$!
# Stall guard: a rank that fails to start leaves the others blocked in a collective with idle GPUs.
# Stop the launcher once any rank reports a failed start, or once no run output has changed for
# WIDEEP_STALL_S seconds (default 720).
(
    while kill -0 $launcher 2>/dev/null && [[ ! -e "$run/frontend.done" ]]; do
        sleep 15
        newest=$(stat -c %Y "$run"/rank-*.log "$run"/result.json 2>/dev/null | sort -n | tail -1)
        if grep -qs "Failed to initialize executor" "$run"/rank-*.log \
            || (( $(date +%s) - ${newest:-$(date +%s)} > ${WIDEEP_STALL_S:-720} )); then
            echo "STALL_STOP $(date +%s.%N)" >> "$run/timeline"
            kill -TERM $launcher; sleep 10; kill -KILL $launcher 2>/dev/null; exit 0
        fi
    done
) &
stall=$!
# Teardown measurement without FT: once the frontend has killed a rank, wait the given seconds
# (standing in for an operator or health check noticing), then terminate the job and record it.
if [[ -n "${WIDEEP_TEARDOWN_AFTER_KILL_S:-}" ]]; then
    (
        until grep -q kill_wall_s "$run/result.json" 2>/dev/null; do kill -0 $launcher 2>/dev/null || exit 0; sleep 0.2; done
        sleep "$WIDEEP_TEARDOWN_AFTER_KILL_S"
        echo "TEARDOWN_TERM $(date +%s.%N)" >> "$run/timeline"
        kill -TERM $launcher
        for _ in $(seq 300); do kill -0 $launcher 2>/dev/null || exit 0; sleep 0.1; done
        echo "TEARDOWN_KILL $(date +%s.%N)" >> "$run/timeline"
        kill -KILL $launcher
    ) &
fi
wait $launcher
rc=$?
echo "MPIRUN_EXIT $(date +%s.%N) rc=$rc" >> "$run/timeline"
kill $guard $stall 2>/dev/null
touch "$run/done"
echo "MPIRUN_RC=$rc"
exit $rc

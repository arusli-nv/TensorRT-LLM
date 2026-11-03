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
# WIDEEP_MPI_PREFIX: an Open MPI build (with its mpi4py under python/) used instead of the container's, for the
# agents, mpirun and every rank. The *_PREFIX variables matter: the container's OPAL_PREFIX would otherwise make
# mpirun exec the container's prterun and launch the container's prted.
if [[ -n "${WIDEEP_MPI_PREFIX:-}" ]]; then
    export OPAL_PREFIX=$WIDEEP_MPI_PREFIX PRTE_PREFIX=$WIDEEP_MPI_PREFIX PMIX_PREFIX=$WIDEEP_MPI_PREFIX
    export PATH=$WIDEEP_MPI_PREFIX/bin:$PATH LD_LIBRARY_PATH=$WIDEEP_MPI_PREFIX/lib:${LD_LIBRARY_PATH:-}
    export PYTHONPATH=$WIDEEP_MPI_PREFIX/python${PYTHONPATH:+:$PYTHONPATH}
fi
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
for var in $(compgen -e | grep -E '^(TLLM_|TRTLLM_|WIDEEP_WORLD$|LLM_MODELS_ROOT$|PYTHONUNBUFFERED$|HOME$|PYTHONPATH$)'); do forward+=(-x "$var"); done
set +e
ft=(--with-ft ulfm); [[ -n "${WIDEEP_NO_ULFM:-}" ]] && ft=()
# Host MPI transport (WIDEEP_MPI_MCA overrides). Without ULFM: Open MPI's defaults (UCX), as deployed. ULFM needs
# the ob1 PML, whose only transport between nodes here is TCP; TCP within a node too, since ULFM's shrink hangs at
# 32 ranks with the shared-memory BTL (runs/_diagnostics/ULFMSCALE-7992972). One NIC: faster small collectives (HOSTCOLL).
if [[ -z "${WIDEEP_MPI_MCA+set}" ]]; then
    WIDEEP_MPI_MCA=""
    (( ${#ft[@]} )) && WIDEEP_MPI_MCA="--mca pml ob1 --mca btl self,tcp --mca btl_tcp_if_include eth0 --mca coll ^hcoll,ucc"
    # Open MPI's in-band failure detector is off (DECISIONS 2026-10-21 20:05, user: option A; deprecated upstream, and its
    # threaded shutdown stalls MPI_Finalize). WIDEEP_DETECTOR=1 turns it on (period 1 s, timeout 5 s) for comparisons.
    (( ${#ft[@]} )) && [[ -n "${WIDEEP_DETECTOR:-}" ]] && WIDEEP_MPI_MCA="$WIDEEP_MPI_MCA --mca mpi_ft_detector true --mca mpi_ft_detector_thread true --mca mpi_ft_detector_period 1 --mca mpi_ft_detector_timeout 5"
fi
echo "MPIRUN_START $(date +%s.%N) mca=[$WIDEEP_MPI_MCA]" >> "$run/timeline"
mpirun "${ft[@]}" --prtemca plm ssh --prtemca plm_ssh_agent "python3 $L/remote_exec.py" \
    --prtemca plm_ssh_no_tree_spawn 1 $WIDEEP_MPI_MCA \
    -x WIDEEP_AGENT_DIR -x WIDEEP_AGENT_KEY_FILE -x PATH -x LD_LIBRARY_PATH "${forward[@]}" \
    --host "${hosts%,}" -np "$np" --map-by ppr:$ppn:node --bind-to none "$@" > "$run/mpirun.log" 2>&1 &
launcher=$!
frontend_pids() {  # this case's trtllm-serve process on this node (not the shells that launched it, not its forks)
    local p argv all=() pid
    for p in /proc/[0-9]*; do
        grep -qzxF "WIDEEP_CASE_TAG=$WIDEEP_CASE_TAG" "$p/environ" 2>/dev/null || continue
        mapfile -d '' -t argv < "$p/cmdline" 2>/dev/null || continue
        [[ ${argv[0]##*/} == trtllm-serve || ( ${argv[0]##*/} == python* && ${argv[1]:-} == */trtllm-serve ) ]] \
            && all+=("${p#/proc/}")
    done
    for pid in "${all[@]}"; do
        [[ " ${all[*]} " == *" $(awk '/^PPid:/ {print $2}' "/proc/$pid/status" 2>/dev/null) "* ]] || echo "$pid"
    done
}
# Once the frontend has finished, stop mpirun (after a rank loss it can outlive every rank even after
# MPI_Abort): SIGTERM, then SIGKILL after 30 s. Profiled runs first get WIDEEP_LAUNCHER_GRACE_S to
# finish writing their reports. Clean-shutdown cases (WIDEEP_GRACEFUL_STOP=TERM|INT) first stop the deployment
# as an orchestrator does, by signalling trtllm-serve, and give mpirun WIDEEP_GRACEFUL_STOP_S to exit on its own.
(
    while [[ ! -e "$run/frontend.done" ]] && kill -0 $launcher 2>/dev/null; do sleep 1; done
    if [[ -n "${WIDEEP_GRACEFUL_STOP:-}" ]] && kill -0 $launcher 2>/dev/null; then
        pids=$(frontend_pids | paste -sd' ')
        echo "GRACEFUL_STOP $(date +%s.%N) signal=$WIDEEP_GRACEFUL_STOP pids=[$pids]" >> "$run/timeline"
        if [[ -n $pids ]]; then
            kill -s "$WIDEEP_GRACEFUL_STOP" $pids 2>/dev/null || true
            for _ in $(seq "${WIDEEP_GRACEFUL_STOP_S:-120}"); do kill -0 $launcher 2>/dev/null || exit 0; sleep 1; done
            echo "GRACEFUL_TIMEOUT $(date +%s.%N)" >> "$run/timeline"
        else
            echo "GRACEFUL_NO_FRONTEND $(date +%s.%N)" >> "$run/timeline"
        fi
    fi
    if [[ -n "${TLLM_TEST_NSYS_RANKS:-}" ]]; then
        for _ in $(seq "${WIDEEP_LAUNCHER_GRACE_S:-60}"); do kill -0 $launcher 2>/dev/null || exit 0; sleep 1; done
    fi
    kill -TERM $launcher 2>/dev/null || exit 0
    for _ in $(seq 30); do kill -0 $launcher 2>/dev/null || exit 0; sleep 1; done
    echo "MPIRUN_RC=launcher-kill"
    kill -KILL $launcher
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

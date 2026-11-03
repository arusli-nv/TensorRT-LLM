#!/bin/bash
# Several independent tasks in one allocation (one Slurm task per node). Usage: bundle.sh SPEC...
# Each SPEC is "FIRST_NODE:NODES:LIMIT_S:COMMAND"; the nodes FIRST_NODE..FIRST_NODE+NODES-1 run COMMAND, which sees
# SLURM_NODEID / SLURM_NNODES renumbered within its group, so a 1-node harness case runs as on its own job.
# Each task logs to logs/bundle/<job>-<first node>.log on its group's first node; the summary line per task is
# "BUNDLE_TASK first=<n> rc=<rc>".
# Slurm ends a step's remaining tasks soon after the first one exits (whatever its exit code), so every node,
# including one in no group, marks itself done and waits until all nodes are done before exiting.
# BUNDLE_FILL_<first node>=COMMAND (optional): once its task ends, that group reruns COMMAND, one round at a time, for
# as long as another group's task still runs, so a group that ends early (or fails) does not leave its GPUs idle.
# The group's first node decides each round for the whole group; a round is bounded by BUNDLE_FILL_LIMIT_S (1800).
# A round shorter than BUNDLE_FILL_MIN_S (300) failed early (e.g. at its smoke check), so the group stops filling.
set -uo pipefail
C=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude
node=$SLURM_NODEID nnodes=$SLURM_NNODES
done_dir=$C/logs/bundle/$SLURM_JOB_ID.done
# A task that initializes MPI through the step's PMIx starts a fence across every task of the step, which the
# other groups never join: Slurm then cancels the whole step. Tasks run without the step's PMIx/PMI environment.
for var in $(compgen -e | grep -E '^(PMIX_|PMI_|OMPI_MCA_ess|PRTE_MCA_ras)'); do unset "$var"; done
# A task's own mpirun must run on its node only, not on the bundle's allocation (it would spawn daemons on the
# other nodes through srun): Slurm allocation discovery and launch are off for every task.
export PRTE_MCA_ras=^slurm PRTE_MCA_plm=^slurm
mkdir -p "$done_dir"
for spec in "$@"; do
    IFS=: read -r first count limit cmd <<< "$spec"
    (( node >= first && node < first + count )) || continue
    log=$C/logs/bundle/$SLURM_JOB_ID-$first.log
    [[ $node == "$first" ]] || log=$log.node$((node - first))
    echo "BUNDLE_TASK_START first=$first node=$node $(date +%T) cmd=$cmd" | tee -a "$log"
    SLURM_NODEID=$((node - first)) SLURM_NNODES=$count timeout --kill-after=60 "$limit" bash -c "$cmd" >> "$log" 2>&1
    rc=$?
    echo "BUNDLE_TASK first=$first node=$node rc=$rc $(date +%T)" | tee -a "$log"
    fill_var=BUNDLE_FILL_$first
    fill=${!fill_var:-}
    touch "$done_dir/task.$node"
    round=0 short=0
    while [[ -n $fill ]]; do
        round=$((round + 1)) decision=$done_dir/fill.$first.$round
        if [[ $node == "$first" ]]; then
            if (( short == 0 && $(ls "$done_dir" | grep -c '^task\.') < nnodes )); then echo go > "$decision.tmp"; else echo stop > "$decision.tmp"; fi
            mv "$decision.tmp" "$decision"
        fi
        while [[ ! -s $decision ]]; do sleep 5; done
        [[ $(cat "$decision") == go ]] || break
        echo "BUNDLE_FILL_START first=$first node=$node round=$round $(date +%T)" | tee -a "$log"
        started=$SECONDS
        SLURM_NODEID=$((node - first)) SLURM_NNODES=$count BUNDLE_FILL_ROUND=$round \
            timeout --kill-after=60 "${BUNDLE_FILL_LIMIT_S:-1800}" bash -c "$fill" >> "$log" 2>&1
        echo "BUNDLE_FILL first=$first node=$node round=$round rc=$? $(date +%T)" | tee -a "$log"
        (( SECONDS - started < ${BUNDLE_FILL_MIN_S:-300} )) && short=1
    done
    break
done
touch "$done_dir/task.$node" "$done_dir/$node"
while (( $(ls "$done_dir" | grep -c '^[0-9]*$') < nnodes )); do sleep 10; done
echo "BUNDLE_NODE_EXIT node=$node $(date +%T)"

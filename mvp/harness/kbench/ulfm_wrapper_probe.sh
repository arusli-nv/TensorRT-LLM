#!/bin/bash
# Why does mpirun (PRRTE) exit ~1 s after a re-planned recovery in REPLAN-LITE, while the bare-python ULFM probe
# survives two SIGKILLs? Hypothesis: the victims run under trtllm-llmapi-launch, which forks the worker, so PRRTE
# sees a process that EXITED WITH NON-ZERO STATUS (137) instead of one killed by a signal. Variants, each with two
# SIGKILLed victims (ranks 2 then 3) and a single one: bare python; a forking bash wrapper; trtllm-llmapi-launch;
# an exec-ing bash wrapper. PROBE_OK from every survivor = the job survived. Usage: ulfm_wrapper_probe.sh OUT_DIR
out=$1; mkdir -p "$out"
probe=/art/mvp/harness/kbench/ulfm_replan_probe.py
run() {  # NAME SECOND -- launcher words...
    local name=$1 second=$2; shift 3
    echo "=== $name second=$second"
    timeout 300 mpirun --with-ft ulfm --mca pml ob1 --mca btl self,tcp --mca coll ^hcoll,ucc --allow-run-as-root \
        --oversubscribe --prtemca errmgr_base_verbose 5 -np 4 "$@" "$probe" kill "$second" 8 \
        > "$out/wrap-$name-$second.log" 2>&1
    echo "rc=$?"; grep -E "PROBE_OK|LOST_CONNECTION|NULL communicator|errmgr" "$out/wrap-$name-$second.log" | head -10
}
for second in -1 3; do
    run bare "$second" -- python3
    run fork "$second" -- bash /art/mvp/harness/kbench/fork_wrapper.sh python3
    run exec "$second" -- bash /art/mvp/harness/kbench/exec_wrapper.sh python3
    run forksignal "$second" -- bash /art/mvp/harness/kbench/fork_signal_wrapper.sh python3
done
echo ULFM_WRAPPER_PROBE_DONE

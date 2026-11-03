#!/bin/bash
# Variants of ulfm_replan_probe.py under the ULFM launch of test_mpi_survivors_ulfm.py (on the installed wheel).
# Usage: ulfm_replan_probe.sh OUT_DIR
out=$1; mkdir -p "$out"
for variant in "kill -1 8" "exit 3 8" "kill 3 8"; do
    set -- $variant
    echo "=== mode=$1 second=$2 hold=$3"
    timeout 300 mpirun --with-ft ulfm --mca pml ob1 --mca btl self,tcp --mca coll ^hcoll,ucc --allow-run-as-root \
        --oversubscribe -np 4 python3 /art/mvp/harness/kbench/ulfm_replan_probe.py "$@" > "$out/probe-$1-$2.log" 2>&1
    echo "rc=$?"; grep -E "PROBE_OK|PMIx|NULL communicator|ERRORS_ARE_FATAL|second shrink" "$out/probe-$1-$2.log" | head -8
done
echo ULFM_PROBE_DONE

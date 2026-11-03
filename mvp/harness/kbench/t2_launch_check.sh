#!/bin/bash
# T2 transport check (one Slurm task per node, 2+ nodes): does the Open MPI build in WIDEEP_MPI_PREFIX run the
# executor's host-collective pattern (t2_check.py: object broadcasts up to 32 MB, then a split-communicator
# allgather, on a worker thread) through our launcher, per transport setting, with and without ULFM? 180 s each.
# T2_VARIANTS / T2_FT pick the settings. Usage: t2_launch_check.sh OUT_DIR
set -uo pipefail
out=$1
K=$(dirname "$(realpath "$0")")
mkdir -p "$out"
unset PYTHONPATH
for var in $(compgen -e | grep -E '^(PMIX_|PMI_|OMPI_MCA_ess|PRTE_MCA_ras)'); do unset "$var"; done
coll="--mca coll ^hcoll,ucc"
uct="--mca pml ob1 --mca btl self,uct,tcp --mca btl_tcp_if_include eth0"
declare -A mca=(
    [tcp]="--mca pml ob1 --mca btl self,tcp --mca btl_tcp_if_include eth0 $coll"
    [uct-default]="$uct $coll"
    [uct-ib]="$uct --mca btl_uct_memory_domains $(ls /sys/class/infiniband 2>/dev/null | paste -sd,) $coll"
    [uct-nohan]="$uct --mca coll ^hcoll,ucc,han"
)
for name in ${T2_VARIANTS:-tcp uct-default uct-ib uct-nohan}; do
    for ft in ${T2_FT:-ulfm none}; do
        run=$out/$name-$ft
        mkdir -p "$run"
        (
            export WIDEEP_MPI_MCA="${mca[$name]}"
            [[ $ft == none ]] && export WIDEEP_NO_ULFM=1
            timeout 180 bash $K/../launch/task.sh "$run" 4 bash $K/rank.sh "$run" python3 $K/t2_check.py
        )
        [[ "$SLURM_NODEID" == 0 ]] && echo "CHECK $name-$ft $(grep -h CHECK_OK "$run"/rank-0.log 2>/dev/null || echo HUNG_OR_FAILED) $(grep -h MPIRUN_EXIT "$run"/timeline 2>/dev/null)"
    done
done
echo NODE_END

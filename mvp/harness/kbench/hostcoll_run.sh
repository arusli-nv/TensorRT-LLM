#!/bin/bash
# One Slurm task per node: host-collective latency over 32 ranks for each MPI transport setting.
# Usage: hostcoll_run.sh DIST_DIR OUT_DIR
set -uo pipefail
dist=$1; out=$2
C=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude
K=$(dirname "$(realpath "$0")")
mkdir -p "$out"
bash $C/install_wheel.sh "$dist" "$out/install-$SLURM_NODEID.log" || exit 1
{ grep -H . /sys/class/net/*/speed /sys/class/net/*/mtu; grep -H . /sys/class/infiniband/*/ports/1/link_layer \
    /sys/class/infiniband/*/ports/1/rate; ls -l /sys/class/infiniband/*/device/net; } > "$out/nics-$SLURM_NODEID.txt" 2>&1
unset PYTHONPATH
for var in $(compgen -e | grep -E '^(PMIX_|PMI_|OMPI_MCA_ess|PRTE_MCA_ras)'); do unset "$var"; done
coll="--mca coll ^hcoll,ucc"
[[ "$SLURM_NODEID" == 0 ]] && ompi_info | grep -E "MCA (btl|pml|mtl)" > "$out/ompi-components.txt" 2>&1
# Settings: NAME=MCA. HOSTCOLL_SET picks a list; the default covers the transports compared so far.
declare -A mca=(
    [ucx]=""
    [ob1-sm-tcp-eth0]="--mca pml ob1 --mca btl self,sm,tcp --mca btl_tcp_if_include eth0 $coll"
    # Which ranks learn of a failure from PMIx (lever A, sm): ULFM's failure messages per rank.
    [ob1-sm-tcp-eth0-verbose]="--mca pml ob1 --mca btl self,sm,tcp --mca btl_tcp_if_include eth0 $coll --mca mpi_ft_verbose 10"
    [ob1-tcp-eth0]="--mca pml ob1 --mca btl self,tcp --mca btl_tcp_if_include eth0 $coll"
    [ob1-smcuda-tcp-eth0]="--mca pml ob1 --mca btl self,smcuda,tcp --mca btl_tcp_if_include eth0 $coll"
    [cm-ofi]="--mca pml cm --mca mtl ofi $coll"
    [ob1-sm-tcp]="--mca pml ob1 --mca btl self,sm,tcp $coll"
    [ob1-tcp]="--mca pml ob1 --mca btl self,tcp $coll"
    # btl uct (WIDEEP_MPI_PREFIX build): RDMA under ob1, which ULFM supports; tcp stays as a fallback BTL.
    # Default memory domains, or every mlx5 device. "all" also takes the CUDA/shared-memory domains and hangs in
    # MPI_Init (runs/T2-LAUNCH-8016810).
    [ob1-uct]="--mca pml ob1 --mca btl self,uct,tcp --mca btl_tcp_if_include eth0 $coll"
    [ob1-uct-ib]="--mca pml ob1 --mca btl self,uct,tcp --mca btl_tcp_if_include eth0 --mca btl_uct_memory_domains $(ls /sys/class/infiniband 2>/dev/null | paste -sd,) $coll"
)
for name in ${HOSTCOLL_SET:-ucx ob1-sm-tcp ob1-tcp}; do
    for ft in ${HOSTCOLL_FT:-ulfm none}; do
        run=$out/$name-$ft
        mkdir -p "$run"
        (
            export WIDEEP_MPI_MCA="${mca[$name]}"
            [[ $ft == none ]] && export WIDEEP_NO_ULFM=1
            # HOSTCOLL_WRAPPED=1: Python runs as a child of a bash process that stays alive, as
            # trtllm-llmapi-launch runs the server's workers, instead of replacing it.
            wrap=(); [[ -n "${HOSTCOLL_WRAPPED:-}" ]] && wrap=(bash -c '"$@"; exit $?' wrapper)
            timeout 600 bash $K/../launch/task.sh "$run" 4 bash $K/rank.sh "$run" "${wrap[@]}" python3 $K/${HOSTCOLL_BENCH:-hostcoll_bench.py} "$name-$ft"
        )
        [[ "$SLURM_NODEID" == 0 ]] && grep -h HOSTCOLL "$run"/rank-0.log
    done
done
echo NODE_END

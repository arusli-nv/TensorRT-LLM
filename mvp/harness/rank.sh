#!/bin/bash
# Per-rank wrapper under mpirun: one log per MPI rank, memory bound to the node's Grace NUMA nodes (the GB200
# wide-EP recipe). Ranks listed in TLLM_TEST_NSYS_RANKS (r+r+...) run under nsys and capture only the window
# the driver opens through the server's /start_profile endpoint (profile cases only: writing a report stalls
# the profiled rank). A rank that exits records it in the run's timeline.
# Usage: rank.sh RUN_DIR program args...
run=$1; shift
rank=$OMPI_COMM_WORLD_RANK
# TLLM_TEST_NSYS_RANKS=all profiles every rank; TLLM_TEST_NSYS_PYTHON=1 adds CPU and Python backtrace sampling.
if [[ -n "${TLLM_TEST_NSYS_RANKS:-}" && ( $TLLM_TEST_NSYS_RANKS == all || "+$TLLM_TEST_NSYS_RANKS+" == *"+$rank+"* ) ]]; then
    collect=(--trace=cuda,nvtx --sample=none --cpuctxsw=none)
    [[ ${TLLM_TEST_NSYS_PYTHON:-} == 1 ]] && collect=(--trace=cuda,nvtx,osrt --sample=process-tree --cpuctxsw=process-tree \
        --python-sampling=true --python-sampling-frequency=1000 --resolve-symbols=false)
    # repeat:N:async: one report per window, written when it closes (a case with several windows, or one that ends
    # in MPI_Abort after a loss).
    set -- nsys profile -c cudaProfilerApi --capture-range-end="${TLLM_TEST_NSYS_CAPTURE_END:-stop}" \
        -o "$run/nsys-rank$rank" -f true --wait=primary \
        --cuda-graph-trace=node "${collect[@]}" "$@"
fi
# TLLM_TEST_NCU_RANKS (r+r+...): those ranks run under Nsight Compute, which profiles the first TLLM_TEST_NCU_COUNT
# kernels matching TLLM_TEST_NCU_KERNELS (a regex) inside the /start_profile window, collecting TLLM_TEST_NCU_ARGS
# (pick metrics that fit one pass: a replayed MoE all-to-all kernel would wait on its peers' flags again).
if [[ -n "${TLLM_TEST_NCU_RANKS:-}" && "+$TLLM_TEST_NCU_RANKS+" == *"+$rank+"* ]]; then
    ncu=$(command -v ncu || ls /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)
    if [[ -n $ncu ]]; then
        read -ra ncu_args <<< "${TLLM_TEST_NCU_ARGS:-}"
        set -- "$ncu" --profile-from-start off --target-processes all -k "regex:${TLLM_TEST_NCU_KERNELS:?}" \
            -c "${TLLM_TEST_NCU_COUNT:-40}" "${ncu_args[@]}" -o "$run/ncu-rank$rank" -f "$@"
        echo "NCU_RANK $rank $ncu" >> "$run/timeline"
    else
        echo "NCU_MISSING $rank" >> "$run/timeline"
    fi
fi
command -v numactl > /dev/null && set -- numactl -m 0,1 "$@"
# TLLM_TEST_THP_DISABLE=1: the rank runs with transparent huge pages off (prctl PR_SET_THP_DISABLE, inherited
# across exec), to test whether THP collapse or compaction migrates pages under the EPLB copies.
[[ -n "${TLLM_TEST_THP_DISABLE:-}" ]] && set -- python3 -c 'import ctypes, os, sys; ctypes.CDLL(None).prctl(41, 1, 0, 0, 0); os.execvp(sys.argv[1], sys.argv[1:])' "$@"
# TLLM_TEST_LOCAL_LOGS=1: the log goes to node-local /tmp, with a link in the run dir for readers on its node;
# serve.sh copies it over the link when the case ends (line-buffered logs on Lustre can block a rank).
log=$run/rank-$rank.log
if [[ -n "${TLLM_TEST_LOCAL_LOGS:-}" ]]; then
    log=/tmp/wideep-ranklogs/${run//\//_}/rank-$rank.log
    mkdir -p "${log%/*}" && ln -sfn "$log" "$run/rank-$rank.log"
fi
stdbuf -oL -eL "$@" > "$log" 2>&1
rc=$?
echo "RANK_EXIT $rank rc=$rc $(date +%s.%N)" >> "$run/timeline"
exit $rc

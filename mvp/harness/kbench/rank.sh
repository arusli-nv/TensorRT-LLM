#!/bin/bash
# Per-rank wrapper under mpirun: one log per MPI rank. With TLLM_TEST_NSYS_RANK=r[+r...], those ranks run
# under nsys for TLLM_TEST_NSYS_DURATION seconds after TLLM_TEST_NSYS_DELAY seconds.
run=$1; shift
if [[ -n "${TLLM_TEST_NSYS_RANK:-}" && "+$TLLM_TEST_NSYS_RANK+" == *"+$OMPI_COMM_WORLD_RANK+"* ]]; then
    # TLLM_TEST_NSYS_CPU=1 also samples CPU stacks and thread scheduling (which threads use the CPU).
    sampling="--trace=cuda,nvtx --sample=none --cpuctxsw=none --backtrace=none"
    [[ -n "${TLLM_TEST_NSYS_CPU:-}" ]] && sampling="--trace=cuda,nvtx,osrt --sample=process-tree --cpuctxsw=process-tree --backtrace=none --python-sampling=true"
    set -- nsys profile -o "$run/nsys-rank$OMPI_COMM_WORLD_RANK" -f true $sampling --cuda-graph-trace=node --kill none \
        --delay "${TLLM_TEST_NSYS_DELAY:-200}" --duration "${TLLM_TEST_NSYS_DURATION:-20}" "$@"
fi
if [[ -n "${TLLM_TEST_STARTUP_LOG:-}" ]]; then
    # Startup-stage timings only; everything else stays at the default level.
    export TLLM_LOG_LEVEL_BY_MODULE="info:_startup"
fi
if [[ -n "${TLLM_TEST_CUDA_COREDUMP:-}" ]]; then
    # Diagnostic: a GPU exception writes a lightweight coredump naming the faulting kernel and PC.
    export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1
    export CUDA_COREDUMP_FILE="$run/cudacore.%h.%p"
fi
if [[ "${1:-}" == nsys ]]; then
    # Export the profile for analysis on hosts without nsys.
    stdbuf -oL -eL "$@" > "$run/rank-$OMPI_COMM_WORLD_RANK.log" 2>&1
    rc=$?
    nsys export --type sqlite --force-overwrite true -o "$run/nsys-rank$OMPI_COMM_WORLD_RANK.sqlite" \
        "$run/nsys-rank$OMPI_COMM_WORLD_RANK.nsys-rep" > /dev/null 2>&1
    exit $rc
fi
exec stdbuf -oL -eL "$@" > "$run/rank-$OMPI_COMM_WORLD_RANK.log" 2>&1

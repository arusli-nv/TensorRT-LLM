#!/bin/bash
# Per-rank wrapper under mpirun: one log per MPI rank. With TLLM_TEST_NSYS_RANK=r, rank r runs
# under nsys for TLLM_TEST_NSYS_DURATION seconds after TLLM_TEST_NSYS_DELAY seconds.
run=$1; shift
if [[ -n "${TLLM_TEST_NSYS_RANK:-}" && "$OMPI_COMM_WORLD_RANK" == "$TLLM_TEST_NSYS_RANK" ]]; then
    set -- nsys profile -o "$run/nsys-rank$OMPI_COMM_WORLD_RANK" -f true --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --backtrace=none --kill none \
        --delay "${TLLM_TEST_NSYS_DELAY:-200}" --duration "${TLLM_TEST_NSYS_DURATION:-20}" "$@"
fi
exec stdbuf -oL -eL "$@" > "$run/rank-$OMPI_COMM_WORLD_RANK.log" 2>&1

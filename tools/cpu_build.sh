#!/bin/bash
# Wheel build on the cpu partition (OPERATING_PROMPT ADDITIONS 4). The broker requests N tasks of 1 CPU each, so
# task 0 widens its CPU affinity (the kernel keeps it inside the job's cpuset) and builds; the others exit.
# Usage (job command): bash /art/tools/cpu_build.sh LABEL
[[ ${SLURM_PROCID:-0} == 0 ]] || exit 0
[[ $(uname -m) == aarch64 ]] || { echo "CPU_BUILD_WRONG_ARCH $(uname -m)"; exit 1; }
taskset -pc "0-$(($(nproc --all) - 1))" $$ > /dev/null || exit 1
echo "CPU_BUILD nproc=$(nproc)"
(( $(nproc) >= 16 )) || { echo "CPU_BUILD_TOO_FEW_CPUS"; exit 1; }
JOB_COUNT=$(nproc) SKIP_TESTS=1 exec bash /art/build_and_test.sh "$1"

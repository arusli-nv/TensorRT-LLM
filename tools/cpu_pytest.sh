#!/bin/bash
# Pure-Python unit tests on the cpu partition (no GPU): install a wheel, copy a checkout's changed Python over it,
# run pytest on the given test files from that checkout. Arguments starting with tests/ are taken relative to
# CHECKOUT; the rest (-k EXPR, -v, ...) go to pytest as given. Usage (job command):
#   bash /art/tools/cpu_pytest.sh DIST_DIR OVERLAY_DIR CHECKOUT PYTEST_ARG...
set -uo pipefail
[[ ${SLURM_PROCID:-0} == 0 ]] || exit 0
dist=$1 overlay=$2 checkout=$3; shift 3
unset PYTHONPATH
log=/art/logs/cpu-pytest-${SLURM_JOB_ID:-local}.log
ts() { echo "CPU_PYTEST $(date +%T) $*"; }
# `import tensorrt_llm` runs MPI_Init; under a multi-task srun step its PMIx fence waits for the other tasks,
# which exited above, so it hangs (cpu-pytest5). Drop the launcher environment: MPI starts as a singleton.
unset $(compgen -e | grep -E '^(PMIX_|PMI_|OMPI_|SLURM_)')
ts start
bash /art/install_wheel.sh "$dist" /tmp/cpu-pytest-install.log || exit 1
ts "wheel installed"
pkg=$(cd / && python3 -c "import importlib.util as u, os; print(os.path.dirname(u.find_spec('tensorrt_llm').origin))")
# OVERLAY_DIR "-": the wheel already matches the checkout.
[[ $overlay != - ]] && (cd "$overlay/tensorrt_llm" && find . -name '*.py' -o -name 'trtllm-llmapi-launch') | while read -r f; do
    cp "$overlay/tensorrt_llm/$f" "$pkg/$f"
done
# CPU nodes have no driver: load the CUDA stub as libcuda.so.1.
stub=$(ls /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/targets/*/lib/stubs/libcuda.so 2>/dev/null | head -1)
mkdir -p /tmp/cudastub && ln -sf "$stub" /tmp/cudastub/libcuda.so.1
export LD_LIBRARY_PATH=/tmp/cudastub:${LD_LIBRARY_PATH:-}
# The same short list the GPU test jobs install (build_and_test.sh). Installing all of requirements-dev.txt
# --no-deps onto PYTHONPATH shadowed image packages and broke tensorrt_llm's decoding ops (cpu-pytest3/4).
pip install "pytest>=8.4,<9.1" pytest-asyncio pytest-timeout pytest-mock pytest-env pytest-forked parameterized mako \
    > /tmp/pytest-install.log 2>&1 || { echo PYTEST_INSTALL_FAILED; tail -5 /tmp/pytest-install.log; }
ts "deps installed; stub=$stub"
ver=$(cd /tmp && timeout 300 python3 -c 'import tensorrt_llm; print(tensorrt_llm.__version__)' 2>&1 | tail -1)
ts "import: $ver"
[[ $ver == 1.* ]] || { echo "CPU_PYTEST_DONE rc=1 (import failed or timed out)"; exit 1; }
args=(); for a in "$@"; do [[ $a == tests/* ]] && a=$checkout/$a; args+=("$a"); done
cd /tmp && PYTHONPATH=$checkout/tests/unittest timeout 1800 python3 -m pytest -v -p no:cacheprovider --timeout=300 --durations=10 "${args[@]}" > "$log" 2>&1
rc=$?
ts "pytest finished"
tail -40 "$log"
echo "CPU_PYTEST_DONE rc=$rc log=$log"

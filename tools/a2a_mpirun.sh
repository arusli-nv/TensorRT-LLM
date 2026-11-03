#!/bin/bash
# FT all-to-all unit files that start ranks with MPIPoolExecutor (MPI spawn). A singleton process cannot spawn in this
# container, so pytest runs as rank 0 of `mpirun -n 1`, whose runtime serves the spawn. Unbuffered, a log per file,
# 300 s per test. The Slurm step's PMIx env is unset: a timed-out spawn must not reach the step's error handler.
# Snapshot at /checkout, wheel from /art/build-LABEL. Usage: a2a_mpirun.sh LABEL TEST_FILE...
set -uo pipefail
label=$1; shift
export HOME=/tmp/wideep-home PYTHONUSERBASE=/tmp/wideep-home/.local PIP_CACHE_DIR=/tmp/wideep-home/.cache/pip
mkdir -p "$PYTHONUSERBASE" "$PIP_CACHE_DIR"
out=/art/build-$label/a2a-mpirun
mkdir -p "$out"
bash /art/install_wheel.sh "/art/build-$label/dist" "$out/install.log" || { echo INSTALL_FAILED; exit 1; }
pip install "pytest>=8.4,<9.1" pytest-asyncio pytest-timeout pytest-mock pytest-env parameterized mako \
    > "$out/pytest-install.log" 2>&1 || echo PYTEST_INSTALL_FAILED
for var in $(compgen -e | grep -E '^(PMIX_|PMI_|SLURM_|OMPI_MCA_ess|PRTE_MCA_ras)'); do unset "$var"; done
for t in "$@"; do
    name=$(basename "$t" .py)
    (cd /tmp && PYTHONPATH=/checkout/tests/unittest PYTHONUNBUFFERED=1 timeout -k 30 780 \
        mpirun --allow-run-as-root --oversubscribe -n 1 -x PYTHONPATH -x PYTHONUNBUFFERED -x LD_LIBRARY_PATH \
        python3 -u -m pytest -v -p no:cacheprovider --timeout=300 "/checkout/$t" > "$out/$name.log" 2>&1)
    echo "a2a $name rc=$?"
    grep -E '^(FAILED|ERROR)|[0-9]+ (passed|failed)' "$out/$name.log" | tail -8
done
echo A2A_MPIRUN_DONE

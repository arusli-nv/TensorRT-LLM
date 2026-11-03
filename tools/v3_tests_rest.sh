#!/bin/bash
# The v3 tests build 8124637 did not reach. It stopped at test_nvlink_one_sided.py, a spawn test that needs the mpirun
# launch and timed out until PMIx ended the step. This script runs the remaining FT unit files plus the fixed
# test_moe_a2a_abort.py on the v3 wheel, then the two spawn files through a2a_mpirun.sh, then the WideEP integration test.
# Snapshot at /checkout (v3 d2ed209a27; the wheel from bcea8da496 differs only in that test file).
# Usage: v3_tests_rest.sh LABEL   (wheel and logs in /art/build-LABEL)
set -uo pipefail
label=$1
export HOME=/tmp/wideep-home PYTHONUSERBASE=/tmp/wideep-home/.local PIP_CACHE_DIR=/tmp/wideep-home/.cache/pip
mkdir -p "$PYTHONUSERBASE" "$PIP_CACHE_DIR"
out=/art/build-$label
bash /art/install_wheel.sh "$out/dist" "$out/install-rest.log" || { echo INSTALL_FAILED; exit 1; }
pip install "pytest>=8.4,<9.1" pytest-asyncio pytest-timeout pytest-mock pytest-env pytest-forked parameterized mako \
    > "$out/pytest-install-rest.log" 2>&1 || echo PYTEST_INSTALL_FAILED
job_id=${SLURM_JOB_ID:-0}
for var in $(compgen -e | grep -E '^(PMIX_|PMI_|SLURM_|OMPI_MCA_ess|PRTE_MCA_ras)'); do unset "$var"; done
echo "=== extra"
for t in $EXTRA_TESTS; do
    (cd /tmp && PYTHONPATH=/checkout/tests/unittest timeout -k 30 1200 python3 -m pytest -q -p no:cacheprovider \
        --timeout=600 "/checkout/$t" >> "$out/test-extra-rest.log" 2>&1)
    echo "extra $t rc=$?"
done
grep -E '^(FAILED|ERROR)' "$out/test-extra-rest.log" | head -20
echo "=== mpirun"
bash /art/tools/a2a_mpirun.sh "$label" tests/unittest/_torch/moe/multi_gpu/test_nvlink_one_sided.py \
    tests/unittest/_torch/moe/test_moe_comm.py
echo "=== integration"
SLURM_JOB_ID=$job_id TESTS=/checkout bash /art/mvp/harness/run_integration.sh "build-$label"
echo V3_TESTS_REST_DONE

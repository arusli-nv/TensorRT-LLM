#!/bin/bash
# Completes the v2 build's tests that build 8114247 did not reach: the load-balancer gtest (libs from the installed
# wheel), the remaining FT unit files, then the WideEP integration test. Snapshot at /checkout, wheel from /art/build-LABEL.
# Usage: v2_test_completion.sh LABEL   (one line per step to stdout)
set -uo pipefail
label=$1
export HOME=/tmp/wideep-home PYTHONUSERBASE=/tmp/wideep-home/.local PIP_CACHE_DIR=/tmp/wideep-home/.cache/pip
mkdir -p "$PYTHONUSERBASE" "$PIP_CACHE_DIR"
out=/art/build-$label
bash /art/install_wheel.sh "$out/dist" "$out/install-completion.log" || { echo INSTALL_FAILED; exit 1; }
pip install "pytest>=8.4,<9.1" pytest-asyncio pytest-timeout pytest-mock pytest-env pytest-forked parameterized mako \
    > "$out/pytest-install-completion.log" 2>&1 || echo PYTEST_INSTALL_FAILED
# A test that starts MPI ranks (mpi4py spawn) must not see the Slurm step's PMIx: when such a test times out, the
# step's PMIx error handler cancels the whole job (8114247, 8115508).
job_id=${SLURM_JOB_ID:-0}
for var in $(compgen -e | grep -E '^(PMIX_|PMI_|SLURM_|OMPI_MCA_ess|PRTE_MCA_ras)'); do unset "$var"; done
pkg=$(cd / && python3 -c "import importlib.util as u, os; print(os.path.dirname(u.find_spec('tensorrt_llm').origin))")
echo "=== gtest"
LD_LIBRARY_PATH=$pkg/libs:${LD_LIBRARY_PATH:-} /art/overlays/build-root-feat/cpp-build-oot/tests/unit_tests/runtime/moeLoadBalancerTest \
    --gtest_filter="${GTEST_FILTER}" > "$out/gtest.log" 2>&1
echo "gtest rc=$?"; grep -E "^\[  (PASSED|FAILED) |FAILED  \]" "$out/gtest.log" | head -20
echo "=== extra"
for t in $EXTRA_TESTS; do
    (cd /tmp && PYTHONPATH=/checkout/tests/unittest timeout -k 30 900 python3 -m pytest -q -p no:cacheprovider \
        --timeout=600 "/checkout/$t" >> "$out/test-extra-completion.log" 2>&1)
    echo "extra $t rc=$?"
done
grep -E '^(FAILED|ERROR)' "$out/test-extra-completion.log" | head -20
echo "=== integration"
SLURM_JOB_ID=$job_id TESTS=/checkout bash /art/mvp/harness/run_integration.sh "build-$label"
echo V2_TEST_COMPLETION_DONE

#!/bin/bash
# One milestone pass on a snapshot mounted at /checkout: wheel build + gtests + unit groups (restore_build.sh),
# extra unit tests (EXTRA_TESTS, repo-relative), then the WideEP integration tests on 4 GPUs.
# Usage: build_test_all.sh LABEL   (wheel in /art/build-LABEL)
set -uo pipefail
label=$1
# Nothing writes to the user's home (/root here; ~10 GB quota): user-site installs and pip caches stay in the container.
export HOME=/tmp/wideep-home PYTHONUSERBASE=/tmp/wideep-home/.local PIP_CACHE_DIR=/tmp/wideep-home/.cache/pip
mkdir -p "$PYTHONUSERBASE" "$PIP_CACHE_DIR"
GTEST_FILTER=${GTEST_FILTER:-'GhostSlotMemoryTest.*:BorrowedSlotMemoryTest.*:MoeLoadBalancerRestoreTest.*:MoeLoadBalancerMaskOnlyTest.*:MoeLoadBalancerTests/*'} \
    TEST_GROUPS=${TEST_GROUPS:-'gpu_moe cpu executor'} bash /art/restore_build.sh "$label" || exit 1
if [[ -n "${EXTRA_TESTS:-}" ]]; then
    echo "=== extra"
    # EXTRA_SNAPSHOT (diagnosis): a command run before the extra tests, e.g. to record leftover processes and GPU use.
    [[ -n ${EXTRA_SNAPSHOT:-} ]] && bash -c "$EXTRA_SNAPSHOT"
    # One pytest per file under an outer timeout: a hang that pytest-timeout cannot interrupt fails that file only.
    for t in $EXTRA_TESTS; do
        # As in point_units.sh: srun's PMIx/Slurm variables make MPI_Comm_spawn (MPIPoolExecutor) hang (C43).
        (for var in $(compgen -e | grep -E '^(PMIX_|PMI_|SLURM_|OMPI_MCA_ess|PRTE_MCA_ras)'); do unset "$var"; done
            cd /tmp && PYTHONPATH=/checkout/tests/unittest timeout -k 30 1200 python3 -m pytest -q -p no:cacheprovider -v \
            --durations=10 --timeout=600 "/checkout/$t" >> "/art/build-$label/test-extra.log" 2>&1)
        echo "extra $t rc=$?"
    done
    grep -E '^(FAILED|ERROR)|passed|failed' "/art/build-$label/test-extra.log" | tail -20
fi
if [[ -n "${SKIP_INTEGRATION:-}" ]]; then echo "=== integration skipped (own job)"; echo BUILD_TEST_ALL_DONE; exit 0; fi
echo "=== integration"
TESTS=/checkout bash /art/mvp/harness/run_integration.sh "build-$label"
echo BUILD_TEST_ALL_DONE

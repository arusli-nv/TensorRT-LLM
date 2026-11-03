#!/bin/bash
# Restore core pieces on the stack (/checkout = port2, branch wideep-ft-restore): incremental wheel build, the
# load balancer gtests (restore + mask-only), then the stack's unit tests (cpu group) and the EPLB GPU tests.
set -uo pipefail
job=$1
out=/art/build-$job
SKIP_TESTS=1 bash /art/build_and_test.sh "$job" || exit 1
oot=/art/build-root/cpp-build-oot
make -C "$oot" moeLoadBalancerTest -j64 > "$out/gtest-build.log" 2>&1 || { echo GTEST_BUILD_FAILED; tail -30 "$out/gtest-build.log"; exit 1; }
bin=$(find -L "$oot" -type f -name moeLoadBalancerTest -perm -u+x | head -1)
echo "=== gtest $bin"
"$bin" --gtest_filter="${GTEST_FILTER:-MoeLoadBalancerRestoreTest.*:MoeLoadBalancerMaskOnlyTest.*:MoeLoadBalancerTests/*}" > "$out/gtest.log" 2>&1
echo "gtest rc=$?"; grep -E "^\[  (PASSED|FAILED) |FAILED  \]" "$out/gtest.log" | head -20
# EXTRA_GTESTS: more gtest binaries (CMake targets) to build and run whole, e.g. hostAccessibleDeviceAllocatorTest.
for target in ${EXTRA_GTESTS:-}; do
    make -C "$oot" "$target" -j64 >> "$out/gtest-build.log" 2>&1 || { echo "GTEST_BUILD_FAILED $target"; exit 1; }
    "$(find -L "$oot" -type f -name "$target" -perm -u+x | head -1)" >> "$out/gtest.log" 2>&1
    echo "gtest $target rc=$?"
done
SKIP_BUILD=1 GROUPS_ONLY="${TEST_GROUPS:-cpu}" bash /art/build_and_test.sh "$job" 2>&1 | grep -v "^commit="
echo "=== eplb_gpu"
cd /tmp && PYTHONPATH=/checkout/tests/unittest python3 -m pytest -q -p no:cacheprovider --timeout=300 \
    /checkout/tests/unittest/bindings/test_bindings_moe.py > "$out/test-eplb_gpu.log" 2>&1
echo "rc=$?"; tail -3 "$out/test-eplb_gpu.log"
echo RESTORE_BUILD_DONE

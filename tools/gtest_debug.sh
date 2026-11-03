#!/bin/bash
# Debug the borrowed-slot gtests on the built tree (/art/build-root): the full filter minus one test, then that test
# alone under compute-sanitizer memcheck. Usage: gtest_debug.sh TEST_NAME OUT_DIR
set -uo pipefail
test=$1 out=$2; mkdir -p "$out"
bin=$(find /art/build-root/cpp-build-oot -type f -name moeLoadBalancerTest -perm -u+x | head -1)
echo "BIN $bin"
"$bin" --gtest_filter="BorrowedSlotMemoryTest.*:MoeLoadBalancerRestoreTest.*:MoeLoadBalancerMaskOnlyTest.*:MoeLoadBalancerTests/*:-BorrowedSlotMemoryTest.$test" > "$out/without.log" 2>&1
echo "WITHOUT rc=$?"; grep -E "^\[  (PASSED|FAILED) " "$out/without.log" | head
"$bin" --gtest_filter="BorrowedSlotMemoryTest.$test" > "$out/alone.log" 2>&1
echo "ALONE rc=$?"; grep -E "Failure|FAILED|PASSED|illegal" "$out/alone.log" | head -5
sanitizer=$(command -v compute-sanitizer || ls /usr/local/cuda/bin/compute-sanitizer 2>/dev/null)
"$sanitizer" --tool memcheck --show-backtrace device "$bin" --gtest_filter="BorrowedSlotMemoryTest.$test" > "$out/memcheck.log" 2>&1
echo "MEMCHECK rc=$?"; grep -m12 -E "Invalid|Address|at 0x|by thread|Saved host backtrace|ERROR SUMMARY" "$out/memcheck.log"
echo GTEST_DEBUG_DONE

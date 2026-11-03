#!/bin/bash
# Build + unit tests of /checkout, then the routing-sort microbenchmark on the new wheel and on reference wheels.
# Usage: build_bench.sh NAME [REFERENCE_BUILD_DIR...]   (one GB200 node; output in /art/build-NAME/)
name=$1; shift
bash /art/build_and_test.sh "$name"
out=/art/build-$name
bench() { python3 /art/mvp/harness/kbench/sort_bench.py "$1" >> "$out/sort_bench.log" 2>&1; }
bench "$name"
for ref in "$@"; do
    bash /art/install_wheel.sh "/art/$ref/dist" "$out/install-$ref.log" && bench "$ref"
done
cat "$out/sort_bench.log"

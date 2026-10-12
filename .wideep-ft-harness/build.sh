#!/bin/bash
set -euo pipefail
job="$1"
commit=$(git -c safe.directory=/checkout -C /checkout rev-parse HEAD)
build_root=/art/build-root
out=/art/build-$job
mkdir -p "$build_root" "$out/dist"
unset PYTHONPATH
cd /checkout
python3 scripts/build_wheel.py --build_root "$build_root" --out-of-tree --use_ccache \
    --cuda_architectures 100-real --job_count 48 --yes --dist_dir "$out/dist"
sha256sum "$out"/dist/*.whl > "$out/wheel-sha256.txt"
echo "BUILD_SUCCESS commit=$commit"

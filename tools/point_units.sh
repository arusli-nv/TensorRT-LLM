#!/bin/bash
# Point test of Python-only changes: unit test files on a wheel with an overlay, no rebuild.
# Usage: point_units.sh WHEEL_DIST OVERLAY TESTS_OVERLAY SNAPSHOT OUT TEST_FILE...
#   OVERLAY holds tensorrt_llm/... copied over the installed package; SNAPSHOT/tests is copied to /tmp/ut (the rest of
#   SNAPSHOT linked beside it) and TESTS_OVERLAY/tests/... over it (TESTS_OVERLAY: a directory or a colon list of them). Logs in OUT; prints "<file> rc=<rc>" per file, then POINT_UNITS_DONE.
set -uo pipefail
C=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude
dist=$1 overlay=$2 tests_overlay=$3 snapshot=$4 out=$5; shift 5
export HOME=/tmp/wideep-home PYTHONUSERBASE=/tmp/wideep-home/.local PIP_CACHE_DIR=/tmp/wideep-home/.cache/pip
mkdir -p "$PYTHONUSERBASE" "$PIP_CACHE_DIR" "$out"
bash "$C/install_wheel.sh" "$dist" "$out/install.log" || { echo INSTALL_FAILED; exit 1; }
pip install "pytest>=8.4,<9.1" pytest-asyncio pytest-timeout pytest-mock pytest-env pytest-forked parameterized mako \
    > "$out/pytest-install.log" 2>&1 || echo PYTEST_INSTALL_FAILED
pkg=$(cd / && python3 -c "import importlib.util as u, os; print(os.path.dirname(u.find_spec('tensorrt_llm').origin))")
(cd "$overlay/tensorrt_llm" && find . -name '*.py' | while read -r rel; do cp "$rel" "$pkg/$rel"; echo "overlaid $rel"; done)
# A copy of the snapshot's tests (the conftests import several test packages) beside links to the rest of the tree.
mkdir -p /tmp/ut && cp -r "$snapshot/tests" /tmp/ut/
for entry in "$snapshot"/*; do [[ $(basename "$entry") == tests || $(basename "$entry") == tensorrt_llm ]] || ln -sfn "$entry" /tmp/ut/; done
# The source package: a tree of links to the snapshot with OVERLAY's files (any type) copied into it, so tests that read
# source files (e.g. the trtllm-llmapi-launch script) see the overlaid version. Nothing is written through a link.
src=/tmp/ut/.src/tensorrt_llm
[[ -d $src ]] || { mkdir -p /tmp/ut/.src && cp -rs "$snapshot/tensorrt_llm" /tmp/ut/.src/; } || { echo SRC_TREE_FAILED; exit 1; }
ln -sfn "$src" /tmp/ut/tensorrt_llm
(cd "$overlay" && find tensorrt_llm -type f | while read -r rel; do cp --remove-destination -p "$rel" "/tmp/ut/.src/$rel"; done)
for overlay_dir in ${tests_overlay//:/ }; do  # a colon list: later directories win
    (cd "$overlay_dir" && find tests -name '*.py' | while read -r rel; do cp "$rel" "/tmp/ut/$rel"; echo "test overlaid $rel"; done)
done
for var in $(compgen -e | grep -E '^(PMIX_|PMI_|SLURM_|OMPI_MCA_ess|PRTE_MCA_ras)'); do unset "$var"; done
for t in "$@"; do
    # tools/pytest_plugins/known_failures.txt is the one list of known failures every gate deselects.
    (cd /tmp && PYTHONPATH=/tmp/ut/tests/unittest:$C/tools/pytest_plugins timeout -k 30 1500 python3 -m pytest -q \
        -p no:cacheprovider -p known_failures --timeout=600 "/tmp/ut/$t" > "$out/$(basename "$t" .py).log" 2>&1)
    rc=$?
    echo "$t rc=$rc $(grep -E '(passed|failed|error)' "$out/$(basename "$t" .py).log" | tail -1)"
    (( rc == 0 )) || failed=1
done
# A file that did not run (import or usage error) fails the step like a failing test.
[[ -z ${failed:-} ]] && echo POINT_UNITS_PASS || echo POINT_UNITS_FAIL
echo POINT_UNITS_DONE

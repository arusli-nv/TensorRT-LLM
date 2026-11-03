#!/bin/bash
# Python-only fix check on a built wheel: overlay the checkout's changed Python files, run the given unit test
# files, then the WideEP integration test. Usage (in a 1-node GPU job, checkout mounted at /checkout):
#   OVERLAY=LIST overlay_verify.sh BUILD_DIR_NAME UNIT_TEST_FILE...
set -uo pipefail
C=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude
build=$1; shift
export TESTS=/checkout
out=$C/mvp/runs/OVERLAY-UNIT-$build-$SLURM_JOB_ID; mkdir -p "$out"
bash $C/install_wheel.sh "$C/$build/dist" "$out/install.log" || exit 1
bash $C/overlay_python.sh /checkout "$OVERLAY" > "$out/overlay.log" 2>&1 || { echo OVERLAY_FAILED; exit 1; }
grep -v '^-r' /checkout/requirements-dev.txt > "$out/requirements-dev.txt"
pip install -r "$out/requirements-dev.txt" > "$out/pip-dev.log" 2>&1 || { echo PIP_DEV_FAILED; exit 1; }
(cd /checkout/tests/unittest && python3 -m pytest -q -p no:cacheprovider "$@" > "$out/unit.log" 2>&1; echo "unit rc=$?" >> "$out/unit.log")
tail -3 "$out/unit.log"
bash $C/mvp/harness/run_integration.sh "$build"
echo OVERLAY_VERIFY_DONE

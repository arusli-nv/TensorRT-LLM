#!/bin/bash
# The series' WideEP peer-loss integration test (DeepSeek-V3-Lite NVFP4, EP4 on one GB200 node) on a built wheel.
# Usage: [TESTS=DIR] run_integration.sh BUILD_DIR_NAME   (results in mvp/runs/INTEGRATION-<build>-<slurm id>/)
set -uo pipefail
C=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude
out=$C/mvp/runs/INTEGRATION-$1-$SLURM_JOB_ID
mkdir -p "$out"
bash $C/install_wheel.sh "$C/$1/dist" "$out/install.log" || exit 1
# OVERLAY: a file list whose Python files are copied from TESTS over the installed package (Python-only changes).
[[ -n "${OVERLAY:-}" ]] && { bash $C/overlay_python.sh "$TESTS" "$OVERLAY" > "$out/overlay.log" 2>&1 || { echo OVERLAY_FAILED; exit 1; }; }
# The integration conftest needs the dev requirements, as in CI (the wheel already brought requirements.txt).
grep -v '^-r' ${TESTS:-$C/dev}/requirements-dev.txt > "$out/requirements-dev.txt"
pip install -r "$out/requirements-dev.txt" > "$out/pip-dev.log" 2>&1 || { echo PIP_DEV_FAILED; tail -20 "$out/pip-dev.log"; exit 1; }
export LLM_MODELS_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/common
# TESTS: the checkout (or snapshot) whose integration test runs; default dev/.
cd ${TESTS:-$C/dev}/tests/integration/defs
python3 -m pytest -v -p no:cacheprovider --timeout=3600 --basetemp="$out/pytest" wide_ep_ft/test_peer_loss_recovery.py > "$out/test.log" 2>&1
echo "rc=$?" >> "$out/test.log"
tail -5 "$out/test.log"

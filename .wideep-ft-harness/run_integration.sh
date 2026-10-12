#!/bin/bash
# Install the wheel with the checkout's Python and run the single-node WideEP FT integration test.
# Usage: run_integration.sh DIST_DIR OUT_DIR [pytest -k expression]
set -uo pipefail
dist=$1; out=$2; k=${3:-}
mkdir -p "$out"
unset PYTHONPATH
bash /art/install_wheel.sh "$dist" "$out/install.log" || exit 1
bash /art/overlay_python.sh /checkout >> "$out/install.log" 2>&1
pip install -r /checkout/requirements-dev.txt > "$out/pip-dev.log" 2>&1 || echo DEV_REQS_PARTIAL
cd /checkout/tests/integration/defs
export LLM_MODELS_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/common
python3 -m pytest -v -p no:cacheprovider --basetemp="$out/tmp" wide_ep_ft/test_peer_loss_recovery.py ${k:+-k "$k"} \
    > "$out/pytest.log" 2>&1
echo "pytest rc=$?"; tail -15 "$out/pytest.log"

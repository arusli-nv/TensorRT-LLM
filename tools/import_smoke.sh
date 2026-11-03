#!/bin/bash
# Import smoke test of a source tree's Python on a built wheel, in the real container (before any build of that tree).
# Usage: import_smoke.sh WHEEL_DIST_DIR CHECKOUT   (installs the wheel as install_wheel.sh does, copies the checkout's tensorrt_llm/*.py
# over it, then imports the package and the FT-touched modules; prints IMPORT_SMOKE_OK or the traceback)
set -euo pipefail
export HOME=/tmp/wideep-home PYTHONUSERBASE=/tmp/wideep-home/.local PIP_CACHE_DIR=/tmp/wideep-home/.cache/pip
mkdir -p "$PYTHONUSERBASE" "$PIP_CACHE_DIR"
# The wheel and its absent requirements, as every case installs them.
bash /lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude/install_wheel.sh "$1" /tmp/smoke-install.log \
    || { tail -20 /tmp/smoke-install.log; exit 1; }
site=$(cd / && python3 -c "import site; print(site.getsitepackages()[0])")
(cd "$2" && find tensorrt_llm -name '*.py' -print0 | while IFS= read -r -d '' f; do mkdir -p "$site/$(dirname "$f")"; cp "$f" "$site/$f"; done)
cd /tmp && python3 - <<'PY'
import importlib
# (mgmn_worker_node / mgmn_leader_node are launcher entry scripts: importing them runs them.)
for name in ["tensorrt_llm", "tensorrt_llm.llmapi.mpi_session", "tensorrt_llm._torch.moe.fused_moe.moe_load_balancer",
             "tensorrt_llm._torch.pyexecutor.py_executor", "tensorrt_llm._torch.pyexecutor.peer_loss_recovery",
             "tensorrt_llm.executor.recovery.authority", "tensorrt_llm.serve.openai_server"]:
    importlib.import_module(name)
    print("imported", name, flush=True)
from tensorrt_llm.llmapi.mpi_session import MpiSession  # noqa: F401
print("IMPORT_SMOKE_OK")
PY

#!/bin/bash
# Rerun the GPU MoE unit tests on an existing wheel (unless SKIP_UNIT), then the integration test.
# Usage: k3_verify.sh BUILD_DIR_NAME
C=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude
[[ -n "${SKIP_UNIT:-}" ]] || SKIP_BUILD=1 GROUPS_ONLY=gpu_moe DIST_DIR=/art/$1/dist bash /art/build_and_test.sh "$1-rerun"
bash $C/mvp/harness/run_integration.sh "$1"

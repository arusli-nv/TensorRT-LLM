#!/bin/bash
# First step of every job command: refuses a job whose spec did not pass tools/presubmit.py.
# Usage: require_presubmit.sh TOKEN
C=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude
[[ -f "$C/logs/presubmit/$1.pass" ]] || { echo "PRESUBMIT_MISSING $1: run tools/presubmit.py first"; exit 1; }
echo "PRESUBMIT_TOKEN $1"

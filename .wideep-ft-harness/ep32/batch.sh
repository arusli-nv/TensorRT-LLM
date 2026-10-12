#!/bin/bash
# Runs several EP32 cases back to back in one allocation. Every node runs this script; each case is
# one run.sh invocation with its own run directory, so the nodes stay in step through run.sh.
# Usage: batch.sh DIST_DIR BATCH_DIR CASE...   where CASE is NAME:ENV=V,ENV=V:frontend args
# e.g. "ref:TLLM_TEST_ORACLE_EXCLUDE_RANK=2:--no-kill"; CASE_DIST=dir overrides the wheel.
set -uo pipefail
dist=$1; batch=$2; shift 2
E=$(dirname "$(realpath "$0")")
for case in "$@"; do
    IFS=: read -r name envs args <<< "$case"
    run=$batch/$name
    case_index=$((${case_index:-0} + 1))
    echo "BATCH_CASE $name start $(date +%T)"
    (
        # A fresh authority port per case, so a slow-exiting authority cannot block the next one.
        export WIDEEP_AUTHORITY_PORT=$((24680 + case_index))
        IFS=, read -ra pairs <<< "$envs"
        for pair in "${pairs[@]}"; do [[ -n "$pair" ]] && export "${pair//@RUN@/$run}"; done
        # shellcheck disable=SC2086
        bash "$E/run.sh" "${CASE_DIST:-$dist}" "$run" ${args//@RUN@/$run}
    )
    echo "BATCH_CASE $name rc=$? $(date +%T)"
done

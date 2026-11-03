#!/bin/bash
# Incremental build of /checkout, install the wheel, run the WideEP unit tests.
set -uo pipefail
job="$1"
out=/art/build-$job
mkdir -p "$out/dist"
unset PYTHONPATH
cd /checkout
commit=$(git -c safe.directory='*' rev-parse HEAD)
echo "commit=$commit"
if [[ -n "${SKIP_BUILD:-}" ]]; then echo "SKIP_BUILD"; else
root=${BUILD_ROOT:-/art/build-root}
# Different worktrees build into this root from the same /checkout path, so make's mtime check can miss a file
# whose content differs from what the root last built: touch every file that changed since that commit.
last=$(cat "$root/last_built_commit" 2>/dev/null)
if [[ -n $last ]] && git -c safe.directory='*' cat-file -e "$last^{commit}" 2>/dev/null; then
    changed=$(git -c safe.directory='*' diff --name-only "$last" HEAD -- cpp)
    echo "touching $(echo "$changed" | grep -c .) files changed since $last"
    [[ -n $changed ]] && echo "$changed" | while read -r f; do [[ -f $f ]] && touch "$f"; done
elif [[ -d $root ]]; then
    echo "no last_built_commit in $root: touching every cpp file"; find cpp -type f -exec touch {} +
fi
# setuptools' build_py copies a .py into the root's build/lib only when it is newer, and a snapshot keeps git's old mtimes
# for files its last checkout did not change: touch the package so no other tree's copy survives in the wheel.
find tensorrt_llm -type f -exec touch {} +
python3 scripts/build_wheel.py --build_root "$root" --out-of-tree --use_ccache \
    --cuda_architectures 100-real --job_count "${JOB_COUNT:-64}" --yes --dist_dir "$out/dist" > "$out/build.log" 2>&1 \
    || { echo BUILD_FAILED; tail -40 "$out/build.log"; exit 1; }
python3 -I /art/tools/verify_wheel.py "$out"/dist/*.whl /checkout "$commit" | tee "$out/verify_wheel.log"
grep -q " 0 differ" "$out/verify_wheel.log" || { echo "BUILD_FAILED wheel Python differs from $commit"; exit 1; }
grep -q " 0 stale" "$out/verify_wheel.log" || { echo "BUILD_FAILED wheel carries stale Python from an earlier build in $root"; exit 1; }
echo "$commit" > "$root/last_built_commit"
echo "BUILD_SUCCESS commit=$commit compiled=$(grep -c "Building C" "$out/build.log")"; fi
[[ -n "${SKIP_TESTS:-}" ]] && { echo ALL_DONE; exit 0; }
dist=${DIST_DIR:-$out/dist}
bash /art/install_wheel.sh "$dist" "$out/install.log" || exit 1
[[ -n "${OVERLAY:-}" ]] && { bash /art/overlay_python.sh /checkout "$OVERLAY" >> "$out/install.log" 2>&1 || { echo OVERLAY_FAILED; exit 1; }; }
pip install "pytest>=8.4,<9.1" pytest-asyncio pytest-timeout pytest-mock pytest-env pytest-forked parameterized mako \
    > "$out/pytest-install.log" 2>&1 || echo PYTEST_INSTALL_FAILED
cd /tmp
export PYTHONPATH=/checkout/tests/unittest
run() { name=$1; shift; [[ -n "${GROUPS_ONLY:-}" && " $GROUPS_ONLY " != *" $name "* ]] && return
    # Test files not in this checkout (an earlier commit of the series) are left out.
    local args=() paths=0; for a in "$@"; do [[ $a == /checkout/* ]] && { [[ -e ${a%%::*} ]] || continue; paths=1; }; args+=("$a"); done
    (( paths )) || { echo "=== $name: no test files"; return; }
    echo "=== $name"; python3 -m pytest -q -p no:cacheprovider --timeout=300 "${args[@]}" > "$out/test-$name.log" 2>&1; rc=$?; tail -3 "$out/test-$name.log"; echo "rc=$rc"; }
T=/checkout/tests/unittest
# GPU tests first, so the GPUs do not idle behind the CPU groups after a build.
[[ -n "${SKIP_GPU_MOE:-}" ]] || run gpu_moe -v $T/_torch/moe/test_moe_a2a_abort.py $T/_torch/moe/test_moe_execution_state.py $T/_torch/moe/test_moe_slot_gather.py $T/_torch/moe/test_moe_sort_source_slots.py $T/_torch/moe/multi_gpu/test_moe_a2a_late_writes.py $T/_torch/executor/test_kv_cache_revert.py \
    $T/_torch/modules/test_alltoall_watchdog.py $T/_torch/test_mnnvl_alltoall_workspace.py
run cpu $T/executor/test_recovery_authority.py \
    $T/_torch/distributed/test_mpi_survivors.py $T/_torch/distributed/test_mpi_survivors_ulfm.py \
    $T/_torch/executor/test_peer_loss_recovery.py \
    $T/_torch/executor/test_adp_router.py $T/_torch/executor/test_request_utils.py \
    $T/llmapi/apps/test_stream_error_events.py \
    $T/_torch/executor/test_adp_absent_ranks.py $T/_torch/executor/test_resident_placement.py $T/_torch/moe/test_resident_placement.py $T/_torch/disaggregation/test_benchmark_disagg.py \
    $T/_torch/executor/test_multimodal_scheduler.py $T/_torch/executor/kv_cache/test_kv_pool_rebalance.py \
    $T/_torch/executor/test_executor_request_queue.py $T/_torch/executor/test_adp_host_exchange.py \
    $T/_torch/moe/test_ep_group_health.py $T/_torch/moe/fused_moe/test_configurable_moe.py $T/_torch/moe/test_moe_load_balancer.py $T/executor/test_rpc.py \
    $T/executor/test_rpc_worker_mixin.py $T/executor/test_recovery_hang_monitor.py $T/executor/test_recovery_config.py $T/_torch/executor/test_hang_evidence.py
# The rest of test_mpi_session.py spawns MPI processes, which the Slurm container cannot; run ours.
run session $T/llmapi/test_mpi_session.py -k "departed or departure or degraded"
run executor $T/_torch/executor/test_py_executor.py $T/_torch/test_mnnvl_memory_lifecycle.py
echo ALL_DONE

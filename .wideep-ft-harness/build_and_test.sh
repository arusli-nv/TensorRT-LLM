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
python3 scripts/build_wheel.py --build_root ${BUILD_ROOT:-/art/build-root} --out-of-tree --use_ccache \
    --cuda_architectures 100-real --job_count 64 --yes --dist_dir "$out/dist" > "$out/build.log" 2>&1 \
    || { echo BUILD_FAILED; tail -40 "$out/build.log"; exit 1; }
echo "BUILD_SUCCESS commit=$commit compiled=$(grep -c "Building C" "$out/build.log")"; fi
[[ -n "${SKIP_TESTS:-}" ]] && { echo ALL_DONE; exit 0; }
dist=${DIST_DIR:-$out/dist}
bash /art/install_wheel.sh "$dist" "$out/install.log" || exit 1
[[ -n "${OVERLAY:-}" ]] && bash /art/overlay_python.sh /checkout >> "$out/install.log" 2>&1
pip install "pytest>=8.4,<9.1" pytest-asyncio pytest-timeout pytest-mock pytest-env pytest-forked parameterized mako \
    > "$out/pytest-install.log" 2>&1 || echo PYTEST_INSTALL_FAILED
cd /tmp
export PYTHONPATH=/checkout/tests/unittest
run() { name=$1; shift; [[ -n "${GROUPS_ONLY:-}" && " $GROUPS_ONLY " != *" $name "* ]] && return; echo "=== $name"; python3 -m pytest -q -p no:cacheprovider --timeout=300 "$@" > "$out/test-$name.log" 2>&1; rc=$?; tail -3 "$out/test-$name.log"; echo "rc=$rc"; }
T=/checkout/tests/unittest
run cpu $T/executor/test_recovery_authority.py \
    $T/_torch/distributed/test_mpi_survivors.py $T/_torch/distributed/test_mpi_survivors_ulfm.py \
    $T/_torch/executor/test_peer_loss_recovery.py \
    $T/_torch/executor/test_adp_router.py $T/_torch/executor/test_request_utils.py \
    $T/_torch/executor/test_executor_request_queue.py $T/_torch/executor/test_adp_host_exchange.py \
    $T/_torch/moe/test_ep_group_health.py $T/executor/test_rpc.py \
    $T/executor/test_rpc_worker_mixin.py
# The rest of test_mpi_session.py spawns MPI processes, which the Slurm container cannot; run ours.
run session $T/llmapi/test_mpi_session.py -k "departed or departure or degraded"
run executor $T/_torch/executor/test_py_executor.py $T/_torch/test_mnnvl_memory_lifecycle.py
[[ -n "${SKIP_GPU_MOE:-}" ]] || run gpu_moe -v $T/_torch/moe/test_moe_a2a_abort.py $T/_torch/moe/test_moe_execution_state.py $T/_torch/moe/multi_gpu/test_moe_a2a_late_writes.py $T/_torch/executor/test_kv_cache_revert.py \
    $T/_torch/modules/test_alltoall_watchdog.py $T/_torch/test_mnnvl_alltoall_workspace.py
echo ALL_DONE

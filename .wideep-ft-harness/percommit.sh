#!/bin/bash
# For each commit of the folded series: extract its tree, overlay its Python on the tip wheel, and
# run the CPU-side test groups. Usage: percommit.sh DIST OUT SHA...
set -uo pipefail
dist=$1; out=$2; shift 2
mkdir -p "$out"
unset PYTHONPATH
bash /tmp/h/install_wheel.sh "$dist" "$out/install.log" || exit 1
pip install "pytest>=8.4,<9.1" pytest-asyncio pytest-timeout pytest-mock pytest-env pytest-forked parameterized mako \
    > "$out/pytest-install.log" 2>&1
git config --global --add safe.directory '*'
for sha in "$@"; do
    tree=/tmp/tree-$sha; rm -rf "$tree"; mkdir -p "$tree"
    git -C /repo archive "$sha" tensorrt_llm tests | tar -x -C "$tree"
    bash /tmp/h/overlay_python.sh "$tree" > /dev/null 2>&1
    T=$tree/tests/unittest
    tests=$(cd "$T" && ls executor/test_recovery_authority.py _torch/distributed/test_mpi_survivors.py \
        _torch/executor/test_peer_loss_recovery.py _torch/executor/test_adp_absent_ranks.py \
        _torch/executor/test_resident_placement.py _torch/executor/test_adp_host_exchange.py \
        _torch/executor/test_adp_router.py _torch/executor/test_py_executor.py \
        _torch/moe/test_ep_group_health.py executor/test_rpc.py executor/test_rpc_executor_mixin.py \
        executor/test_rpc_worker_mixin.py 2>/dev/null)
    (cd /tmp && PYTHONPATH=$T timeout 1500 python3 -m pytest -q -p no:cacheprovider --timeout=300 \
        $(for t in $tests; do echo "$T/$t"; done) > "$out/$sha.log" 2>&1)
    rc=$?
    (cd /tmp && PYTHONPATH=$T python3 -m pytest -q -p no:cacheprovider --timeout=300 \
        "$T/llmapi/test_mpi_session.py" -k "departed or departure or degraded" > "$out/$sha-session.log" 2>&1)
    echo "$sha rc=$rc session_rc=$? $(tail -1 "$out/$sha.log")"
done
echo PERCOMMIT_DONE

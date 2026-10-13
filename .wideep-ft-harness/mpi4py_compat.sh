#!/bin/bash
# Run MPI-dependent TRT-LLM tests with the image's mpi4py, then with the series' mpi4py 4.1.2.
# Usage: mpi4py_compat.sh OUT_DIR   (inside the devel image, /checkout = worktree at the series tip)
set -uo pipefail
out=$1; mkdir -p "$out"
unset PYTHONPATH
for v in $(compgen -e | grep -E '^(SLURM_|PMIX_|PMI_|OMPI_MCA_ess|PRTE_MCA_ras)'); do unset "$v"; done
C=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude
bash /tmp/h/install_wheel.sh "$C"/build-final-67c2f4f949/dist "$out/wheel.log" || exit 1
bash /tmp/h/overlay_python.sh /checkout >> "$out/wheel.log" 2>&1
# Baseline: main's mpi4py 3.1.5, built with main's own script.
git -c safe.directory='*' -C /checkout show origin/main:docker/common/install_mpi4py.sh > /tmp/h/install_mpi4py_main.sh
pip uninstall -y mpi4py > /dev/null 2>&1; rm -rf /usr/local/lib/python3.12/dist-packages/mpi4py*
bash /tmp/h/install_mpi4py_main.sh > "$out/mpi4py-main-install.log" 2>&1 || { echo MAIN_MPI4PY_INSTALL_FAILED; exit 1; }
pip install "pytest>=8.4,<9.1" pytest-asyncio pytest-timeout pytest-mock pytest-env pytest-forked parameterized mako > /dev/null 2>&1
T=/checkout/tests/unittest
TESTS="$T/_torch/moe/multi_gpu/test_moe_a2a.py $T/_torch/moe/multi_gpu/test_moe_a2a_workspace.py $T/_torch/executor/kv_cache/test_kvcm2_integration.py $T/_torch/disaggregation/test_disagg_inflight_cancel_gate.py $T/llmapi/test_mpi_session.py"
run() {
  tag=$1
  python3 -c "import mpi4py; print('mpi4py', mpi4py.__version__)" > "$out/$tag.version"
  for t in $TESTS; do
    name=$(basename $t .py)
    (cd /tmp && PYTHONPATH=$T timeout 2400 mpirun -np 1 --allow-run-as-root --oversubscribe \
        python3 -m pytest -q -p no:cacheprovider --timeout=600 $t > "$out/$tag-$name.log" 2>&1)
    echo "$tag $name rc=$? $(grep -E '[0-9]+ (passed|failed)' "$out/$tag-$name.log" | tail -1)"
  done
}
run base
pip uninstall -y mpi4py > /dev/null 2>&1; rm -rf /usr/local/lib/python3.12/dist-packages/mpi4py*
bash /checkout/docker/common/install_mpi4py.sh > "$out/mpi4py-install.log" 2>&1 || { echo MPI4PY_INSTALL_FAILED; exit 1; }
run v412
echo COMPAT_DONE

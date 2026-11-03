#!/bin/bash
# Tests of a built wheel in their own 1-node GPU job (source snapshot mounted at /checkout): extra unit test files, each
# under an outer timeout, then the WideEP integration tests. Usage: test_wheel.sh BUILD_DIR_NAME [TEST_FILE[@K_EXPR]...]
# (@K_EXPR = pytest -k filter, e.g. test_mpi_session.py@'departed or departure or degraded': most of it spawns MPI).
set -uo pipefail
C=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude
build=$1; shift
ck=${CHECKOUT:-/checkout}  # the source snapshot whose tests run
out=$C/mvp/runs/TEST-WHEEL-$build-$SLURM_JOB_ID${CHECKOUT:+-$(basename "$CHECKOUT")}; mkdir -p "$out"
bash $C/install_wheel.sh "$C/$build/dist" "$out/install.log" || { echo INSTALL_FAILED; exit 1; }
grep -v '^-r' $ck/requirements-dev.txt > "$out/requirements-dev.txt"
pip install -r "$out/requirements-dev.txt" > "$out/pip-dev.log" 2>&1 || { echo PIP_DEV_FAILED; exit 1; }
# OVERLAY: Python files copied from /checkout over the wheel (run_integration.sh reinstalls and re-applies it).
[[ -n "${OVERLAY:-}" ]] && { bash $C/overlay_python.sh $ck "$OVERLAY" > "$out/overlay.log" 2>&1 || { echo OVERLAY_FAILED; exit 1; }; }
# Integration first: a unit file whose hang ends the Slurm step can no longer take the integration run with it.
[[ -n "${SKIP_INTEGRATION:-}" ]] || TESTS=$ck bash $C/mvp/harness/run_integration.sh "$build"
for t in "$@"; do
    # mpi:FILE runs pytest under a local mpirun, so mpi4py's MPIPoolExecutor can spawn workers (its Spawn fails under
    # the srun step's PMIx); the step's PMIx/Slurm launch settings are dropped for that mpirun, as in bundle.sh.
    launch=(); if [[ $t == mpi:* ]]; then t=${t#mpi:}
        launch=(env $(compgen -e | grep -E '^(PMIX_|PMI_|OMPI_MCA_ess|PRTE_MCA_ras)' | sed 's/^/-u /') PRTE_MCA_ras=^slurm
                PRTE_MCA_plm=^slurm mpirun --allow-run-as-root --oversubscribe -n 1); fi
    k=(); [[ $t == *@* ]] && k=(-k "${t#*@}")
    (cd /tmp && PYTHONPATH=$ck/tests/unittest timeout -k 30 1200 "${launch[@]}" python3 -m pytest -p no:cacheprovider -rA \
        --durations=10 --timeout=${UNIT_TIMEOUT_S:-600} ${UNIT_PYTEST_ARGS:-} "$ck/${t%%@*}" "${k[@]}" >> "$out/unit.log" 2>&1)
    echo "unit $t rc=$?"
done
grep -E '^(FAILED|ERROR)|passed|failed' "$out/unit.log" | tail -20
echo TEST_WHEEL_DONE

#!/bin/bash
# Per-commit import + test-collection smoke of a commit series, in the real container. Installs WHEEL_DIST once (its
# compiled parts stand in for every commit's); for each commit BASE..TIP builds a fresh package directory holding links to
# the wheel's non-Python files plus exactly that commit's tensorrt_llm/*.py (a module only a later commit adds is absent),
# imports the package, the FT modules and every module the commit touches, and collects the unit tests it touches.
# Usage: series_import_smoke.sh WHEEL_DIST GIT_DIR BASE TIP   (prints SERIES_SMOKE_OK or SERIES_SMOKE_FAIL lines)
set -uo pipefail
dist=$1 repo=$2 base=$3 tip=$4
export HOME=/tmp/wideep-home PYTHONUSERBASE=/tmp/wideep-home/.local PIP_CACHE_DIR=/tmp/wideep-home/.cache/pip
mkdir -p "$PYTHONUSERBASE" "$PIP_CACHE_DIR"
for var in $(compgen -e | grep -E '^(PMIX_|PMI_|SLURM_|OMPI_MCA_ess|PRTE_MCA_ras)'); do unset "$var"; done
bash /lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude/install_wheel.sh \
    "$dist" /tmp/series-smoke-install.log || { tail -20 /tmp/series-smoke-install.log; exit 1; }
site=$(cd / && python3 -c "import site; print(site.getsitepackages()[0])")
git() { GIT_LFS_SKIP_SMUDGE=1 command git -c safe.directory='*' -C "$repo" "$@"; }
# The unit tests' conftest needs the dev requirements, as in CI (the wheel brought requirements.txt).
git show "$tip:requirements-dev.txt" | grep -v '^-r' > /tmp/series-smoke-requirements-dev.txt
pip install -q -r /tmp/series-smoke-requirements-dev.txt > /tmp/series-smoke-pip.log 2>&1 \
    || { echo "SERIES_SMOKE_FAIL pip"; tail -5 /tmp/series-smoke-pip.log; exit 1; }
fails=0
for c in $(git rev-list --reverse "$base..$tip"); do
    tag="$(git log -1 --format='%h %s' "$c" | cut -c1-90)"
    work=$(mktemp -d /tmp/series-smoke.XXXX)
    # The commit's Python (and unit tests) only.
    git archive "$c" tensorrt_llm tests/unittest | tar -x -C "$work" --wildcards '*.py' 2>/dev/null
    pkg=$work/pkg; mkdir -p "$pkg"
    (cd "$site" && find tensorrt_llm -type f ! -name '*.py' -print0) | while IFS= read -r -d '' f; do
        mkdir -p "$pkg/$(dirname "$f")"; ln -s "$site/$f" "$pkg/$f"
    done
    (cd "$work" && find tensorrt_llm -name '*.py' -print0) | while IFS= read -r -d '' f; do
        mkdir -p "$pkg/$(dirname "$f")"; cp "$work/$f" "$pkg/$f"
    done
    [[ -e $site/triton_kernels ]] && ln -s "$site/triton_kernels" "$pkg/triton_kernels"
    mods=$(git diff --name-only "$c^" "$c" -- 'tensorrt_llm/*.py' | grep -v -E 'mgmn_(worker|leader)_node|__main__' \
        | sed -E 's|/__init__\.py$||; s|\.py$||; s|/|.|g' | paste -sd' ')
    tests=$(git diff --name-only --diff-filter=AM "$c^" "$c" -- 'tests/unittest/*test_*.py' | paste -sd' ')
    out=$(cd /tmp && PYTHONPATH=$pkg python3 - $mods <<'PY' 2>&1
import importlib, sys
for name in ["tensorrt_llm", "tensorrt_llm._torch.pyexecutor.py_executor", "tensorrt_llm._torch.moe.fused_moe.moe_load_balancer",
             "tensorrt_llm.serve.openai_server", *sys.argv[1:]]:
    importlib.import_module(name)
print("IMPORTS_OK")
PY
)
    if ! grep -q IMPORTS_OK <<< "$out"; then
        echo "SERIES_SMOKE_FAIL import $tag"; tail -8 <<< "$out"; fails=$((fails + 1))
    fi
    if [[ -n $tests ]]; then
        col=$(cd /tmp && PYTHONPATH=$pkg:$work/tests/unittest timeout 900 python3 -m pytest -q -p no:cacheprovider --collect-only \
            $(for t in $tests; do echo "$work/$t"; done) 2>&1)
        if ! grep -qE "^[0-9]+ tests? collected|tests collected" <<< "$col" || grep -qE "^ERROR|errors? in" <<< "$col"; then
            echo "SERIES_SMOKE_FAIL collect $tag"; grep -E "^ERROR|Error" <<< "$col" | head -6; fails=$((fails + 1))
        fi
    fi
    echo "SERIES_SMOKE_COMMIT $tag mods=$(wc -w <<< "$mods") tests=$(wc -w <<< "$tests")"
done
(( fails == 0 )) && echo SERIES_SMOKE_OK || echo "SERIES_SMOKE_DONE fails=$fails"

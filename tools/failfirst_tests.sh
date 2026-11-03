#!/bin/bash
# Failing-first check, in the real container after the wheel is installed: runs each named test (taken from NEW's test
# tree) against OLD's Python and launcher on the installed wheel's compiled parts. Every test must fail there.
# Usage: failfirst_tests.sh GIT_DIR OLD NEW TEST...   (TEST: tests/unittest/<file>.py::<function name>, all its params)
# Prints FAILFIRST_OK / FAILFIRST_UNEXPECTED_PASS per test and FAILFIRST_DONE bad=<n>.
set -uo pipefail
repo=$1 old=$2 new=$3; shift 3
for var in $(compgen -e | grep -E '^(PMIX_|PMI_|SLURM_|OMPI_MCA_ess|PRTE_MCA_ras)'); do unset "$var"; done
site=$(cd / && python3 -c "import site; print(site.getsitepackages()[0])")
git() { GIT_LFS_SKIP_SMUDGE=1 command git -c safe.directory='*' -C "$repo" "$@"; }
work=$(mktemp -d /tmp/failfirst.XXXX); mkdir -p "$work/old" "$work/new" "$work/pkg"
git archive "$old" tensorrt_llm | tar -x -C "$work/old" --wildcards '*.py' 2>/dev/null
# The unit conftest imports integration.defs: the whole test tree, with tests/ on the path too.
git archive "$new" tests | tar -x -C "$work/new" --wildcards '*.py' '*.yml' '*.yaml' '*.txt' 2>/dev/null
# The launcher tests find trtllm-llmapi-launch beside the test tree: OLD's.
mkdir -p "$work/new/tensorrt_llm/llmapi"
git show "$old:tensorrt_llm/llmapi/trtllm-llmapi-launch" > "$work/new/tensorrt_llm/llmapi/trtllm-llmapi-launch"
chmod +x "$work/new/tensorrt_llm/llmapi/trtllm-llmapi-launch"
(cd "$site" && find tensorrt_llm -type f ! -name '*.py' -print0) | while IFS= read -r -d '' f; do
    mkdir -p "$work/pkg/$(dirname "$f")"; ln -s "$site/$f" "$work/pkg/$f"
done
(cd "$work/old" && find tensorrt_llm -name '*.py' -print0) | while IFS= read -r -d '' f; do
    mkdir -p "$work/pkg/$(dirname "$f")"; cp "$work/old/$f" "$work/pkg/$f"
done
[[ -e $site/triton_kernels ]] && ln -s "$site/triton_kernels" "$work/pkg/triton_kernels"
bad=0
for t in "$@"; do
    file=${t%%::*} name=${t#*::}
    log=$work/$(basename "$file" .py)-$name.log
    (cd /tmp && PYTHONPATH=$work/pkg:$work/new/tests/unittest:$work/new/tests timeout 600 python3 -m pytest -q -p no:cacheprovider \
        "$work/new/$file" -k "$name" > "$log" 2>&1)
    rc=$?
    summary=$(grep -E "passed|failed|error" "$log" | tail -1)
    if (( rc == 1 )) && ! grep -q "no tests ran\|deselected, 0" <<< "$summary"; then
        echo "FAILFIRST_OK $t: $summary"
    else
        (( rc == 0 )) && kind=UNEXPECTED_PASS || kind=ERROR  # rc 2-5: interrupted, collection or usage error
        echo "FAILFIRST_$kind $t (rc=$rc): $summary"; tail -15 "$log"; bad=$((bad + 1))
    fi
done
echo "FAILFIRST_DONE bad=$bad"

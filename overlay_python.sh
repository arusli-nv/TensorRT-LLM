#!/bin/bash
# Copy changed Python files of a checkout over the installed tensorrt_llm package, for testing Python-only
# changes on an existing wheel. Usage: overlay_python.sh CHECKOUT FILE_LIST   (FILE_LIST: repo-relative paths)
set -euo pipefail
checkout=$1; list=$2
pkg=$(cd / && python3 -c "import importlib.util as u, os; print(os.path.dirname(u.find_spec('tensorrt_llm').origin))")
while read -r path; do
    [[ $path == tensorrt_llm/*.py ]] || continue
    target=$pkg/${path#tensorrt_llm/}
    mkdir -p "$(dirname "$target")"
    cp "$checkout/$path" "$target"
    echo "OVERLAY $path"
done < "$list"

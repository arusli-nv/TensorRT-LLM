#!/bin/bash
# Copy the Python sources of a checkout over the installed tensorrt_llm package, so a wheel
# built from an earlier commit with identical native code runs the checkout's Python. Modules the
# checkout deleted stay installed but are no longer imported.
# Usage: overlay_python.sh CHECKOUT
set -euo pipefail
site=$(cd / && python3 -c "import importlib.util as u, os; print(os.path.dirname(os.path.dirname(u.find_spec('tensorrt_llm').origin)))")
cd "$1"
find tensorrt_llm -name '*.py' -print0 | xargs -0 cp --parents -t "$site"
echo "overlaid python from $1 onto $site"

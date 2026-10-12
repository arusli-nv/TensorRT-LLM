#!/bin/bash
# Install a TRT-LLM wheel into the devel image: the wheel without deps, then only the
# requirements that are absent (never touching the image's torch stack), then mpi4py 4.
# Usage: install_wheel.sh DIST_DIR LOG
set -uo pipefail
dist=$1; log=$2
pip install --no-deps --force-reinstall "$dist"/*.whl > "$log" 2>&1 || { echo INSTALL_FAILED; tail -20 "$log"; exit 1; }
missing=$(python3 - <<'PY'
from importlib.metadata import PackageNotFoundError, requires, version
from packaging.requirements import Requirement
for line in requires("tensorrt_llm") or []:
    req = Requirement(line)
    if req.marker is not None and not req.marker.evaluate({"extra": ""}):
        continue
    try:
        version(req.name)
    except PackageNotFoundError:
        print(str(req))
PY
)
echo "missing requirements: $missing" >> "$log"
if [[ -n "$missing" ]]; then
    printf '%s\n' "$missing" > "$log.req"
    pip install -c <(pip list --format=freeze | grep -iE '^(torch|torchvision|torchaudio|torchao|triton|tensorrt|nvidia-)') \
        -r "$log.req" >> "$log" 2>&1 || { echo DEPS_INSTALL_FAILED; tail -20 "$log"; exit 1; }
fi
# mpi4py through TRT-LLM's own build script (the branch bumps it to 4.0 for ULFM).
mpi4py_script=${MPI4PY_SCRIPT:-$(dirname "$(realpath "$0")")/wt/docker/common/install_mpi4py.sh}
# The script compiles mpi4py; reuse the package it installed when the script is unchanged.
site=$(cd / && python3 -c "import site; print(site.getsitepackages()[0])")
cache=$(dirname "$(realpath "$0")")/cache/mpi4py-$(sha256sum "$mpi4py_script" | cut -c1-16).tar
if [[ -s "$cache" ]]; then
    rm -rf "$site"/mpi4py "$site"/mpi4py-*.dist-info
    tar -xf "$cache" -C "$site" >> "$log" 2>&1 || { echo MPI4PY_CACHE_FAILED; exit 1; }
else
    bash "$mpi4py_script" >> "$log" 2>&1 || { echo MPI4PY_INSTALL_FAILED; tail -20 "$log"; exit 1; }
    mkdir -p "$(dirname "$cache")"
    (cd "$site" && tar -cf "$cache.$$" mpi4py mpi4py-*.dist-info && mv "$cache.$$" "$cache")
fi
python3 -c "import mpi4py; mpi4py.rc.initialize = False; from mpi4py import MPI; assert mpi4py.__version__.startswith('4.'); MPI.Comm.Shrink; MPI.Comm.Revoke" >> "$log" 2>&1 \
    || { echo MPI4PY_ULFM_MISSING; tail -5 "$log"; exit 1; }

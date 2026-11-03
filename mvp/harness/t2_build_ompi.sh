#!/bin/bash
# T2: Open MPI 5.0.x with ULFM and UCX (so ob1 gets btl uct, the RDMA transport ULFM supports), plus
# mpi4py 4.1.2 built against it, into a user prefix. One node, inside the TensorRT-LLM container.
# The exact recipe (versions, configure line, ompi_info checks) goes to $PREFIX/RECIPE.txt: the infra request.
# Usage: t2_build_ompi.sh PREFIX [OMPI_VERSION]
set -euo pipefail
prefix=$1; version=${2:-5.0.10}
work=$(mktemp -d /tmp/ompi-build.XXXX)
mkdir -p "$prefix"
recipe=$prefix/RECIPE.txt
{
    echo "date: $(date -u +%FT%TZ)"; echo "container ompi: $(ompi_info --version 2>/dev/null | head -1)"
    echo "ucx: $(command -v ucx_info) $(ucx_info -v 2>/dev/null | head -2 | tr '\n' ' ')"
} > "$recipe"
ucx_prefix=$(dirname "$(dirname "$(command -v ucx_info)")")
cuda=${CUDA_HOME:-/usr/local/cuda}
cd "$work"
curl -sSL "https://download.open-mpi.org/release/open-mpi/v${version%.*}/openmpi-$version.tar.bz2" | tar -xj
cd "openmpi-$version"
configure=(./configure --prefix="$prefix" --with-ft=ulfm --with-ucx="$ucx_prefix" --with-cuda="$cuda"
    --disable-man-pages)
echo "configure: ${configure[*]}" >> "$recipe"
"${configure[@]}" > "$prefix/configure.log" 2>&1
make -j"$(nproc)" > "$prefix/make.log" 2>&1
make install > "$prefix/install.log" 2>&1
export PATH=$prefix/bin:$PATH LD_LIBRARY_PATH=$prefix/lib:${LD_LIBRARY_PATH:-}
{
    echo "built: $(ompi_info --version | head -1)"
    echo "ftmpi: $(ompi_info | grep -ci ftmpi) lines; btl: $(ompi_info | grep 'MCA btl' | awk '{print $3}' | tr '\n' ' ')"
    echo "pml: $(ompi_info | grep 'MCA pml' | awk '{print $3}' | tr '\n' ' ')"
} >> "$recipe"
# mpi4py 4.1.2 against this MPI, installed into the prefix (jobs add it to PYTHONPATH).
MPICC=$prefix/bin/mpicc pip install --no-cache-dir --no-binary mpi4py --target "$prefix/python" mpi4py==4.1.2 \
    > "$prefix/mpi4py.log" 2>&1
echo "mpi4py: $(PYTHONPATH=$prefix/python python3 -c 'import mpi4py; print(mpi4py.__version__)')" >> "$recipe"
cat "$recipe"
echo T2_BUILD_DONE

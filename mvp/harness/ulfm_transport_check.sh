#!/bin/bash
# Which ob1 BTL settings ULFM can shrink under: the repo's ULFM survivor test (4 ranks, one exits) per setting.
# Usage: ulfm_transport_check.sh DIST_DIR OUT_DIR   (one GB200 node)
C=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude
dist=$1; out=$2; mkdir -p "$out"
bash $C/install_wheel.sh "$dist" "$out/install.log" || exit 1
pip install "pytest>=8.4,<9.1" > "$out/pytest-install.log" 2>&1
test=$C/dev/tests/unittest/_torch/distributed/test_mpi_survivors_ulfm.py
# Ranks come from mpirun, not from the enclosing Slurm launch.
for var in $(compgen -e | grep -E '^(SLURM_|PMIX_|PMI_|OMPI_MCA_ess|PRTE_MCA_ras)'); do unset "$var"; done
for btl in "self,tcp" "self,tcp eth0" "self,sm,tcp" "self,sm,tcp eth0"; do
    set -- $btl; name=${1//,/-}${2:+-$2}
    extra=(); [[ -n "${2:-}" ]] && extra=(--mca btl_tcp_if_include "$2")
    start=$(date +%s.%N)
    timeout 180 mpirun --with-ft ulfm --allow-run-as-root --oversubscribe --mca pml ob1 --mca btl "$1" "${extra[@]}" \
        --mca coll ^hcoll,ucc -np 4 python3 "$test" > "$out/$name.log" 2>&1
    rc=$?
    echo "ULFM_CHECK btl=$btl rc=$rc ok_ranks=$(grep -c ULFM_SURVIVOR_OK "$out/$name.log") s=$(awk "BEGIN{print $(date +%s.%N)-$start}")"
done

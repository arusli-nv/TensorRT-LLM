#!/bin/bash
# Does the installed Open MPI run ob1 over btl/uct under ULFM on this group's nodes? Every node of a bundle group runs
# it: if ompi_info lists btl uct, the group runs kbench/t2_launch_check.sh (variant uct-ib: ob1 self,uct,tcp on the mlx5 devices,
# ULFM on; the executor's host-collective pattern; <= 180 s). Node 0 writes the verdict (ok | rejected); every node
# prints UCT_CHECK <verdict> and exits 0 only for ok, so a caller adds the uct arm with `uct_check.sh DIR && ...`.
# Usage: uct_check.sh OUT_DIR   (WIDEEP_DRY_RUN: no MPI run, verdict ok)
out=$1
C=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude
mkdir -p "$out"
if [[ -n ${WIDEEP_DRY_RUN:-} ]]; then
    [[ $SLURM_NODEID == 0 ]] && echo ok > "$out/verdict"
elif ! ompi_info 2>/dev/null | grep -q "MCA btl: uct"; then
    # Every node sees the same container: no node starts the check.
    [[ $SLURM_NODEID == 0 ]] && { ompi_info 2>&1 | grep -E "MCA (btl|pml)" > "$out/components.txt"; echo rejected > "$out/verdict"; }
else
    T2_VARIANTS=uct-ib T2_FT=ulfm bash "$C/mvp/harness/kbench/t2_launch_check.sh" "$out/t2" > "$out/node$SLURM_NODEID.log" 2>&1
    if [[ $SLURM_NODEID == 0 ]]; then
        grep -q "^CHECK uct-ib-ulfm CHECK_OK" "$out/node0.log" && echo ok > "$out/verdict.tmp" || echo rejected > "$out/verdict.tmp"
        mv "$out/verdict.tmp" "$out/verdict"
    fi
fi
for _ in $(seq 300); do [[ -s $out/verdict ]] && break; sleep 1; done
verdict=$(cat "$out/verdict" 2>/dev/null || echo timeout)
echo "UCT_CHECK $verdict node $SLURM_NODEID"
[[ $verdict == ok ]]

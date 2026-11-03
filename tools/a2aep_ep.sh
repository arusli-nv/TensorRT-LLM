#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# A2A-EP sweep driver, one Slurm task per node (tools/a2a_microbench_ep.sh's scheme, A2A-EP 8121785): EP 4/8/16/32 on
# the first 1/2/4/8 nodes (4 GPUs each) through launch/task.sh (plain mpirun); nodes outside a launch's EP wait for it.
# Code under test: v5-final 27442bd91d = build-freeze-58caa0fde6 wheel + overlays/fix-dropmap-9f1fd4cab4, checked by
# file hash. PDL on (TRTLLM_ENABLE_PDL=1, as serve.sh exports it).
# Three repeats; each runs every EP with EPLB stats (online instantiation) and without (offline); repeat 2 runs the EPs
# in reverse order and the stats arms swapped. Each launch is a fresh process set and MNNVL workspace.
# Usage: a2aep_ep.sh OUT_DIR   (per launch OUT_DIR/r<rep>-ep<EP>-s<stats>.json; node 0 writes OUT_DIR/summary.txt)
# Smoke knobs (defaults = the job): A2AEP_NODES ("1 2 4 8": nodes per EP), A2AEP_REPS (3), A2AEP_CYCLES (20).
# Other builds (MASK-SPILL): A2AEP_WHEEL (wheel dir), A2AEP_LOADS ("32,64,256"), A2AEP_RES_EXPECT (ok|spill: the
# installed libth_common.so's `cuobjdump -res-usage` must give tools/maskspill_res.py's verdict, else the node fails).
# A2A FT ablation (a2a-ablate): A2AEP_RES_TOOL (tools/maskspill_res.py; the expected line is <TOOL NAME>_<EXPECT>, e.g.
# tools/ablate_res.py + ok -> ABLATE_RES_OK), A2AEP_ABLATE_VARIANTS (unset: no ablation; else e.g. "0 S S1 F": every
# (EP, stats) runs one launch per variant, a2aep_bench.py --ablate V, OUT_DIR/r<rep>-ep<EP>-s<stats>-v<V>.json; the
# variant order rotates by A2AEP_ABLATE_ROTATE (3) positions per repeat).
set -uo pipefail
out=$1
C=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude
BENCH=$C/tools/a2aep_bench.py
WHEEL=${A2AEP_WHEEL:-$C/build-freeze-58caa0fde6/dist}
# A2AEP_OVERLAY: a Python overlay over the wheel (default v5-final's; empty: none). A2AEP_CODE_CHECK=0 skips the
# v5-final file-hash check (for another tree, e.g. the PR stack).
OVERLAY=${A2AEP_OVERLAY-$C/overlays/fix-dropmap-9f1fd4cab4}
export HOME=/tmp/wideep-home PYTHONUSERBASE=/tmp/wideep-home/.local PIP_CACHE_DIR=/tmp/wideep-home/.cache/pip
mkdir -p "$PYTHONUSERBASE" "$PIP_CACHE_DIR" "$out"
node=$SLURM_NODEID nnodes=$SLURM_NNODES
# Every wait on other nodes is bounded, so one failed node ends the job instead of idling it.
wait_for() { for _ in $(seq "$2"); do [[ -e $1 ]] && return 0; sleep 1; done; return 1; }

{
    hostname -s
    nvidia-smi -L
    nvidia-smi -q | grep -iE "ClusterUUID|Cluster UUID|CliqueId|Clique ID" | sort | uniq -c
} > "$out/fabric-$node.txt" 2>&1
ok=1
bash "$C/install_wheel.sh" "$WHEEL" "$out/install-$node.log" || ok=0
pkg=$(cd / && python3 -c "import importlib.util as u, os; print(os.path.dirname(u.find_spec('tensorrt_llm').origin))")
# The overlay as serve.sh applies MVP_OVERLAY (serve.sh:181-187): the wheel's file kept as .wideep-orig.
[[ -n $OVERLAY ]] && while read -r rel; do
    [[ -e "$pkg/$rel.wideep-orig" || ! -e "$pkg/$rel" ]] || cp "$pkg/$rel" "$pkg/$rel.wideep-orig"
    cp "$OVERLAY/tensorrt_llm/$rel" "$pkg/$rel"
done < <(cd "$OVERLAY/tensorrt_llm" && find . -name '*.py' | sed 's|^\./||')
# sha256[:16] of the files at 27442bd91d (git show 27442bd91d:<path> | sha256sum).
[[ ${A2AEP_CODE_CHECK:-1} == 0 ]] && echo "CODE_CHECK_SKIPPED" > "$out/code-$node.txt" || python3 - "$pkg" > "$out/code-$node.txt" 2>&1 <<'PY' || ok=0
import hashlib, sys
expected = {
    "_torch/moe/fused_moe/communication/nvlink_one_sided.py": "3d418d5659a7c51d",
    "_torch/moe/fused_moe/communication/execution_state.py": "df9dacb0ae0a424d",
    "_torch/distributed/mnnvl_memory.py": "86aacee3ff14c2f4",
    "_torch/moe/fused_moe/moe_load_balancer.py": "0a4c2f3249f50688",
    "_torch/moe/fused_moe/ep_group_health.py": "65f5809f39b2c361",
}
bad = []
for rel, want in expected.items():
    got = hashlib.sha256(open(f"{sys.argv[1]}/{rel}", "rb").read()).hexdigest()[:16]
    print(rel, got, "ok" if got == want else f"MISMATCH want {want}")
    bad += [rel] if got != want else []
print("CODE_V5F_OK" if not bad else f"CODE_V5F_MISMATCH {bad}")
sys.exit(1 if bad else 0)
PY
if [[ -n ${A2AEP_RES_EXPECT:-} ]]; then
    lib=$pkg/libs/libth_common.so
    res_tool=${A2AEP_RES_TOOL:-$C/tools/maskspill_res.py}
    res_name=$(basename "$res_tool" .py)
    { sha256sum "$lib"; "$(command -v cuobjdump || echo /usr/local/cuda/bin/cuobjdump)" -res-usage "$lib" \
        | python3 "$res_tool"; } > "$out/res-$node.txt" 2>&1
    grep -q "${res_name^^}_${A2AEP_RES_EXPECT^^}" "$out/res-$node.txt" || ok=0
    # A build that staged its library hash (launch/maskspill_build_cmd.txt) must be the one installed.
    want=$(cat "$WHEEL/../LIB_SHA256" 2>/dev/null)
    if [[ -n $want && $(sha256sum < "$lib" | cut -c1-64) != "$want" ]]; then
        echo "LIB_SHA256_MISMATCH want $want" >> "$out/res-$node.txt"; ok=0
    fi
fi
(( ok )) && touch "$out/ready-$node" || touch "$out/failed-$node"
for n in $(seq 0 $((nnodes - 1))); do
    for _ in $(seq 1200); do [[ -e $out/ready-$n || -e $out/failed-$n ]] && break; sleep 1; done
    [[ -e $out/ready-$n ]] || { echo "A2A_EP_NODE_NOT_READY node=$n (see install-$n.log, code-$n.txt)"; exit 1; }
done
(( node == 0 )) && cat "$out/code-0.txt" && cat "$out/res-0.txt" 2>/dev/null

export TRTLLM_ENABLE_PDL=1 TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT=0 PYTHONUNBUFFERED=1 WIDEEP_NO_ULFM=1
# mpirun launches through the node agents (launch/task.sh), not through Slurm.
export PRTE_MCA_ras=^slurm PRTE_MCA_plm=^slurm
read -ra variants <<< "${A2AEP_ABLATE_VARIANTS:-}"
(( ${#variants[@]} )) || variants=("")
for rep in $(seq "${A2AEP_REPS:-3}"); do
    eps=${A2AEP_NODES:-"1 2 4 8"} arms="1 0"
    (( rep == 2 )) && eps=$(tr ' ' '\n' <<< "$eps" | sort -rn | xargs) arms="0 1"
    shift_by=$(( (rep - 1) * ${A2AEP_ABLATE_ROTATE:-3} % ${#variants[@]} ))
    order=("${variants[@]:shift_by}" "${variants[@]:0:shift_by}")
    for nodes in $eps; do
        ep=$((4 * nodes))
        for stats in $arms; do
            for v in "${order[@]}"; do
                run=$out/r$rep-ep$ep-s$stats${v:+-v$v}
                mkdir -p "$run"
                if (( node < nodes )); then
                    SLURM_NNODES=$nodes timeout -k 30 420 bash "$C/launch/task.sh" "$run" 4 \
                        python3 -u "$BENCH" --out "$run.json" --loads "${A2AEP_LOADS:-32,64,256}" --cycles "${A2AEP_CYCLES:-20}" --calls 200 \
                        $( (( stats )) && echo --eplb) ${v:+--ablate "$v"} > "$run/task-$node.log" 2>&1
                    echo "a2aep r$rep ep$ep stats=$stats${v:+ ablate=$v} node $node rc=$?"
                else
                    wait_for "$run/done" 480 || echo "a2aep r$rep ep$ep stats=$stats${v:+ ablate=$v} node $node: no done after 480 s"
                fi
                (( node == 0 )) && grep -hE "^A2A_EP(_CHECK)? " "$run/task-0.log"
            done
        done
    done
done
if (( node == 0 )); then
    python3 "$BENCH" --summarize "$out" > "$out/summary.txt" 2>&1
    cat "$out/summary.txt"
fi
echo A2A_EP_DONE

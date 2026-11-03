#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Writes the finals v6 launch files (review/FINALS_PLAN.md v6 + PRE-REG) into OUT_DIR: fin6_j1..j8, fin6_smoke and the
dress rehearsal fin6_ra, fin6_rb, fin6_rc.

Usage: gen_finals_v6.py OUT_DIR [RA_JOB RB_JOB]
RA_JOB / RB_JOB: the rehearsal jobs whose tuning passes wrote the finals' pinned tables (no-EPLB STD-<RA>, online
STD-G1-<RA>, offline STD-<RB>). Without them the finals' seeds name no file and every seeded case fails its preflight.
"""
import argparse
from pathlib import Path

R = "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude"
BUILD, HEAD = "build-freeze-58caa0fde6", "58caa0fde6"
GATE_TESTS = ("tests/unittest/_torch/executor/kv_cache/test_kv_pool_rebalance.py "
              "tests/unittest/_torch/executor/test_multimodal_scheduler.py tests/unittest/_torch/moe/test_ring_host_copies.py "
              "tests/unittest/executor/test_recovery_authority.py tests/unittest/_torch/moe/test_moe_execution_state.py "
              "tests/unittest/_torch/moe/test_moe_load_balancer.py tests/unittest/_torch/distributed/test_mpi_survivors.py "
              "tests/unittest/_torch/distributed/test_mpi_survivors_ulfm.py tests/unittest/_torch/executor/test_py_executor.py")

COMMON = """R={R}
grep -q "^$(hostname -s) " $R/mvp/EXCLUDE_NODES.txt && {{ echo "EXCLUDED_NODE $(hostname -s)"; exit 1; }}
B={BUILD}; MB=build-main-9cc37929e8
SMOKE=${{SMOKE:-0}}
if [[ $SMOKE == 1 ]]; then
    IT_MAIN=32=6,128=3 IT_HX=8=24,32=6,128=3,512=1 BURN=32:2 KI=4 KI32=8 KSTOP=4 CRR=rank5
else
    IT_MAIN=32=28,128=12 IT_HX=8=39,32=28,128=12,512=4 BURN=32:6 KI=4 KI32=16 KSTOP=8 CRR='rank5 rank5'
fi
SNO={seed_no}
SON={seed_on}
SOFF={seed_off}
HD="TLLM_FAULT_TOLERANCE_HANG_DETECTION=1 TLLM_FAULT_TOLERANCE_HANG_CONFIRM_S=30"
E="MVP_BUILD=$B FINAL_BUILD=$B MAIN_BUILD=$MB WIDEEP_NO_SAMPLERS=1 WIDEEP_POLLUTION_CHECKS=1 WIDEEP_DMON=1 WIDEEP_BURNIN=$BURN STD_PLACEMENT=$R/mvp/runs/STD-8109649/std-offline-placement.yaml $HD FT_COST_E2E_ROUNDS=1 COST_MAX_WAVES=40 COST_CASE_TIMEOUT_S=3000"
MAIN="COST_LOADS=32,128 COST_LOAD_ITERATIONS=$IT_MAIN"
HX="COST_LOADS=8,32,128,512 COST_LOAD_ITERATIONS=$IT_HX"
KS=200
M="--mca btl_tcp_if_include eth0 --mca coll ^hcoll,ucc"
K="MVP_BUILD=$B FINAL_BUILD=$B WIDEEP_MPI_MCA='$M' $HD WIDEEP_FAILURE_DOMAIN=node KF_ITERATIONS=$KI WIDEEP_POLLUTION_CHECKS=1 KF_GRAPH_SIZES=136,148 WIDEEP_KILL_IN_DECODE_STEADY=1 KF_STOP_ITERATIONS=$KSTOP"
KNO="$K KF_BENCH_CONFIG=std-noeplb-ft.yaml KF_VICTIM_RANKS='5 22' KF_VICTIM_NODES='1 5'"
KON="$K KF_BENCH_CONFIG=std-online.yaml KF_VICTIM_RANKS='13 29' KF_VICTIM_NODES='3 7'"
D="MVP_BUILD=$B FINAL_BUILD=$B CR_KEEP_PAGE_CACHE=1 WIDEEP_KILL_AFTER_STEPS=$KS"
A="MVP_BUILD=$B FINAL_BUILD=$B $HD"
run() {{ echo "bash $R/mvp/harness/run_job.sh $1"; }}
"""


def std(label, seed, loads, arms, round_min):
    return (f'"$E {loads} FT_COST_E2E_SEED=${seed} FT_COST_E2E_ROUND_MIN=$((SMOKE ? 10 : {round_min})) '
            f"FT_COST_E2E_ARMS='{arms}' $(run {label})\"")


def kill(label, kvar, rows, extra=""):
    return f"\"${kvar} KF_ROWS='{rows}' KF_REPEATS=$((SMOKE ? 1 : 2)) {extra}$(run {label})\""


def tune(label, loads, arms, measured):
    """This mode's tuning pass at 128/GPU (STD_TUNE_ARMS: ``arms``), then one round of ``measured`` on its pinned table.

    The pass's burn-in is the finals' (32:6) even under SMOKE: its pinned table seeds the finals.
    """
    return f"\"$E {loads} WIDEEP_BURNIN=32:6 STD_TUNE_ARMS='{arms}' FT_COST_E2E_ROUND_MIN=10 FT_COST_E2E_ARMS='{measured}' $(run {label})\""


# Groups: (header, [pieces]); a piece is a shell string expression. Minutes from the fin_h1 v4 header's measured case times:
# case = ready + burn-in 1.1 + warm-ups + windows + ~1 + ~0.75 per load; ready no-EPLB 5.0 (ft 5.5), offline 5.5 / 8.0,
# online 9.5 / 11.0; 300 s windows (32: 28 x 10.7-11.1 s, 128: 12 x 24-27 s, 8: 39 x 7.7 s, 512: 4 x 91 s);
# 32+128 case = ready + 14.2, 8..512 case = ready + 28.3. Kill rows (+2 per KILL-FINAL invocation): no-EPLB rank5 12,
# node1 10, rank0 9, rank5pf 10, deadline 9.4, twokill 12, stop 11, rank5@32 11.5; online +1.5, rank5upd 13.
JOBS = {
    "j1": ("H wave 1 (headline): no-EPLB HX pair off-first + main + LB floor | online HX pair off-first + main", [
        ("no-EPLB HX off-first (67) + main, LB floor @32/128 (38.4) + 2 = 107", [
            std("STD", "SNO", "$HX", "std-noeplb std-noeplb-ft", 70),
            std("STD-G2", "SNO", "$MAIN", "std-noeplb-main std-noeplb-lb", 40)]),
        ("online HX off-first (77.1) + main @32/128 (23.7) + 2 = 103", [
            std("STD-G1", "SON", "$HX", "std-online std-online-ft", 80),
            std("STD-G3", "SON", "$MAIN", "std-online-main", 25)])]),
    "j2": ("H wave 1 (headline): online HX pair ft-first + main | no-EPLB HX pair ft-first + LB floor + main", [
        ("online HX ft-first (77.1) + main (23.7) + 2 = 103", [
            std("STD", "SON", "$HX", "std-online-ft std-online", 80),
            std("STD-G2", "SON", "$MAIN", "std-online-main", 25)]),
        ("no-EPLB HX ft-first (67) + LB floor, main (38.4) + 2 = 107", [
            std("STD-G1", "SNO", "$HX", "std-noeplb-ft std-noeplb", 70),
            std("STD-G3", "SNO", "$MAIN", "std-noeplb-lb std-noeplb-main", 40)])]),
    "j3": ("H: no-EPLB pairs 3-4 @32/128 with main, LB floor, FT-off ulfm", [
        ("no-EPLB ft-first + main + LB + ulfm @32/128: 2 + 19.7 + 19.2 x 3 + 19.5 = 99", [
            std("STD", "SNO", "$MAIN", "std-noeplb-ft std-noeplb std-noeplb-main std-noeplb-lb std-noeplb-ulfm", 100)]),
        ("no-EPLB off-first + ulfm + main + LB @32/128 = 99", [
            std("STD-G1", "SNO", "$MAIN", "std-noeplb std-noeplb-ft std-noeplb-ulfm std-noeplb-main std-noeplb-lb", 100)])]),
    "j4": ("H: online pairs 3-4 @32/128 with main; R online rank5, node1 (n=2)", [
        ("online off-first + main (72.6) + 2 + online rank5 x2 (2 + 27) = 104", [
            std("STD", "SON", "$MAIN", "std-online std-online-ft std-online-main", 75),
            kill("KILL-FINAL-ON", "KON", "rank5")]),
        ("online ft-first + main (72.6) + 2 + online node1 x2 (2 + 23) = 100", [
            std("STD-G1", "SON", "$MAIN", "std-online-ft std-online std-online-main", 75),
            kill("KILL-FINAL-ON-A", "KON", "node1")])]),
    "j5": ("H: offline HX pairs; R online twokill, stop (n=2)", [
        ("offline HX off-first (70.1) + 2 + online twokill x2 (2 + 27) = 101", [
            std("STD", "SOFF", "$HX", "std-offline std-offline-ft", 72),
            kill("KILL-FINAL-ON", "KON", "twokill")]),
        ("offline HX ft-first (70.1) + 2 + online stop x2 (2 + 25) = 99", [
            std("STD-G1", "SOFF", "$HX", "std-offline-ft std-offline", 72),
            kill("KILL-FINAL-ON-A", "KON", "stop")])]),
    "j6": ("H: offline pairs 3-4 @32/128; R online rank5pf, rank5upd, rank0, deadline (n=2)", [
        ("offline ft-first (41.9) + 2 + online rank5pf, rank5upd x2 (2 + 48) = 94", [
            std("STD", "SOFF", "$MAIN", "std-offline-ft std-offline", 45),
            kill("KILL-FINAL-ON", "KON", "rank5pf rank5upd")]),
        ("offline off-first (41.9) + 2 + online rank0, deadline x2 (2 + 43) = 89", [
            std("STD-G1", "SOFF", "$MAIN", "std-offline std-offline-ft", 45),
            kill("KILL-FINAL-ON-A", "KON", "rank0 deadline")])]),
    "j7": ("R no-EPLB: every row n=2 (victims 5/22, nodes 1/5)", [
        ("no-EPLB rank5, node1, twokill, stop x2 = 2 + 90 = 92", [
            kill("KILL-FINAL-OFF", "KNO", "rank5 node1 twokill stop")]),
        ("no-EPLB rank5pf, rank0, deadline x2 (2 + 57) + rank5@32 x2 (2 + 23) = 84", [
            kill("KILL-FINAL-OFF-A", "KNO", "rank5pf rank0 deadline"),
            kill("KILL-FINAL-OFF-B", "KNO", "rank5", "KF_LOAD=32 KF_ITERATIONS=$KI32 ")])]),
    "j8": ("A GSM8K appendix + D warm-restart baseline (3 modes, n=2) + R online rank5@32 (n=2)", [
        ("A: GSM8K kill x2 + nokill + ref, then ref + nokill (63) + D no-EPLB (2 + 22) = 87", [
            "\"$A ACC_REPS=$((SMOKE ? 1 : 2)) ACC_REFS='ref-final nokill-final' $(run ACCURACY-KILL-ON)\"",
            "\"$A ACC_REPS=0 ACC_REFS='ref-final nokill-final' $(run ACCURACY-KILL-OFF)\"",
            "\"$D CR_BENCH_CONFIG=std-noeplb.yaml CR_RUNS='$CRR' $(run COLD-RESTART-A)\""]),
        ("D online (2 + 32) + D offline (2 + 24) + online rank5@32 x2 (2 + 26) = 90", [
            "\"$D CR_BENCH_CONFIG=std-online.yaml CR_RUNS='rank5 rank5' $(run COLD-RESTART)\"",
            "\"$D CR_BENCH_CONFIG=std-offline-ft-filled.yaml CR_RUNS='rank5 rank5' $(run COLD-RESTART-B)\"",
            kill("KILL-FINAL-ON", "KON", "rank5", "KF_LOAD=32 KF_ITERATIONS=$KI32 ")])]),
}

# Dress rehearsal (SMOKE=1 windows): every finals arm, load and kill row once on the refreeze wheel, and the tuning passes
# whose pinned tables seed the finals. RA and RB run together, RC after. Seeded pieces read this job's own tables.
OWN_SEEDS = ["SNO=$R/mvp/runs/STD-$SLURM_JOB_ID/autotuner/pinned.json",
             "SON=$R/mvp/runs/STD-G1-$SLURM_JOB_ID/autotuner/pinned.json"]
REHEARSAL = {
    "ra": ("REHEARSAL A: no-EPLB and online tuning + HX pairs, main/LB/ulfm seeded, D no-EPLB", [
        ("no-EPLB tune (27) + HX off, ft (31) + seeded main, LB, ulfm @32/128 (32) + D no-EPLB x1 (13) = ~103", [
            tune("STD", "$HX", "std-noeplb std-noeplb-ft", "std-noeplb std-noeplb-ft"),
            std("STD-G2", "SNO", "$MAIN", "std-noeplb-main std-noeplb-lb std-noeplb-ulfm", 0),
            "\"$D CR_BENCH_CONFIG=std-noeplb.yaml CR_RUNS='$CRR' $(run COLD-RESTART-A)\""]),
        ("online tune (35) + HX off, ft (41) + seeded main @32/128 (15) = ~91", [
            tune("STD-G1", "$HX", "std-online std-online-ft", "std-online std-online-ft"),
            std("STD-G3", "SON", "$MAIN", "std-online-main", 0)])]),
    "rb": ("REHEARSAL B: offline tuning + HX pair, D offline + online, A GSM8K, no-EPLB kill rows 1-4", [
        ("offline tune (30) + HX off, ft (34) + D offline x1 (13) + D online x1 (17) = ~94", [
            tune("STD", "$HX", "std-offline std-offline-ft", "std-offline std-offline-ft"),
            "\"$D CR_BENCH_CONFIG=std-offline-ft-filled.yaml CR_RUNS='$CRR' $(run COLD-RESTART-B)\"",
            "\"$D CR_BENCH_CONFIG=std-online.yaml CR_RUNS='$CRR' $(run COLD-RESTART)\""]),
        ("A GSM8K kill x1 + refs (35) + refs (22) + no-EPLB rank5, node1, twokill, stop (2 + 45) = ~104", [
            "\"$A ACC_REPS=1 ACC_REFS='ref-final nokill-final' $(run ACCURACY-KILL-ON)\"",
            "\"$A ACC_REPS=0 ACC_REFS='ref-final nokill-final' $(run ACCURACY-KILL-OFF)\"",
            kill("KILL-FINAL-OFF", "KNO", "rank5 node1 twokill stop")])]),
    "rc": ("REHEARSAL C: the remaining kill rows (no-EPLB second victims, every online row)", [
        ("no-EPLB rank5pf, rank0, deadline (22/5, late 26) (2 + 29) + rank5@32 (2 + 11.5) + online rank5, node1, twokill (2 + 36) = ~83", [
            kill("KILL-FINAL-OFF-A", "KNO", "rank5pf rank0 deadline", "KF_VICTIM_RANKS=22 KF_VICTIM_NODES=5 "),
            kill("KILL-FINAL-OFF-B", "KNO", "rank5", "KF_LOAD=32 KF_ITERATIONS=$KI32 "),
            kill("KILL-FINAL-ON", "KON", "rank5 node1 twokill")]),
        ("online stop, rank5pf, rank5upd, rank0, deadline (29/7, late 25) (2 + 61) + rank5@32 (2 + 13) = ~78", [
            kill("KILL-FINAL-ON-A", "KON", "stop rank5pf rank5upd rank0 deadline", "KF_VICTIM_RANKS=29 KF_VICTIM_NODES=7 "),
            kill("KILL-FINAL-ON-B", "KON", "rank5", "KF_LOAD=32 KF_ITERATIONS=$KI32 ")])]),
}

SMOKE = ("FIN6 SMOKE (SMOKE=1, short windows; refreeze validation + every new knob before the finals)", [
    ("gate (<10 min) + no-EPLB all arm kinds, seeded, 8 (24 waves: the lifted 20-wave cap) + 32: ~66", [
        std("STD", "SNO", "COST_LOADS=8,32 COST_LOAD_ITERATIONS=8=24,32=6",
            "std-noeplb std-noeplb-ft std-noeplb-main std-noeplb-lb std-noeplb-ulfm", 0)]),
    ("gate + online-ft and offline-ft seeded @32 + no-EPLB node1 (node 5), twokill (22 then 26), deadline (22, late 26): ~66", [
        std("STD-G1", "SON", "COST_LOADS=32 COST_LOAD_ITERATIONS=32=6", "std-online-ft", 0),
        std("STD-G3", "SOFF", "COST_LOADS=32 COST_LOAD_ITERATIONS=32=6", "std-offline-ft", 0),
        "\"$K KF_BENCH_CONFIG=std-noeplb-ft.yaml KF_VICTIM_RANKS=22 KF_VICTIM_NODES=5 KF_ROWS='node1 twokill deadline' KF_REPEATS=1 $(run KILL-FINAL-OFF)\""])])


def write(out: Path, name: str, title: str, groups, smoke: bool, common: str, extra=(), gate=None) -> None:
    gate = smoke if gate is None else gate
    lines = [f"# FIN6 {name.upper()} (16 nodes, two 8-node groups, segment 8, limit 7000 s per group): {title}.",
             "# Generated by gen_finals_v6.py (review/FINALS_PLAN.md v6 + PRE-REG; schedule in review/FINALS_SCHEDULE_v6.md).",
             f"# Wheel {BUILD} (= v5-final {HEAD}), no overlay. One pinned tactic table per mode from the",
             "# rehearsal's tuning passes (FT_COST_E2E_SEED). 300 s windows (COST_MAX_WAVES=40 lifts the 20-wave cap).",
             "# Kill victims rotate per repeat: no-EPLB ranks 5/22, nodes 1/5; online ranks 13/29, nodes 3/7 (KF_VICTIM_*).",
             *(f"# G{8 * i}: {h}" for i, (h, _) in enumerate(groups)),
             common.rstrip().replace("SMOKE=${SMOKE:-0}", "SMOKE=${SMOKE:-1}" if smoke else "SMOKE=${SMOKE:-0}"),
             *extra]
    if gate:
        lines += [f'T="{GATE_TESTS}"',
                  'g() { echo "bash $R/tools/units_gate.sh $R/logs/bundle/fin6smoke-gate$1-\\$SLURM_JOB_ID $R/$B/dist '
                  '$R/overlays/build-src-8c52d11469 $R/overlays/build-src-8c52d11469 $R/overlays/build-src-8c52d11469 '
                  '$R/$B/point-units-fin6smoke-g$1 $T"; }']
    for i, (_, pieces) in enumerate(groups):
        for j, p in enumerate(pieces, 1):
            lines.append(f"P{8 * i}_{j}={p}")
        lines.append(f'G{8 * i}="{{ ' + "; ".join(f"$P{8 * i}_{j}" for j in range(1, len(pieces) + 1)) + '; }"')
    lines += ['if [[ -n ${FIN_GROUP:-} ]]; then',
              '    v=G$FIN_GROUP; [[ -n ${FIN_PART:-} ]] && v=P${FIN_GROUP}_$FIN_PART',
              '    [[ -n ${!v:-} ]] || { echo "DRY_RUN_FAIL no $v in $0"; exit 1; }',
              '    bash -c "${!v}"; exit',
              'fi']
    if gate:
        lines.append('bash $R/bundle.sh "0:8:7000:$(g 0) && $G0" "8:8:7000:$(g 8) && $G8"')
    else:
        lines.append('bash $R/bundle.sh "0:8:7000:$G0" "8:8:7000:$G8"')
    (out / f"fin6_{name}_cmd.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("ra_job", nargs="?", default="UNSET")
    parser.add_argument("rb_job", nargs="?", default="UNSET")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    pinned = "$R/mvp/runs/{}/autotuner/pinned.json"
    common = COMMON.format(R=R, BUILD=BUILD, seed_no=pinned.format(f"STD-{args.ra_job}"),
                           seed_on=pinned.format(f"STD-G1-{args.ra_job}"), seed_off=pinned.format(f"STD-{args.rb_job}"))
    for name, (title, groups) in JOBS.items():
        write(out, name, title, groups, smoke=False, common=common)
    write(out, "smoke", SMOKE[0], SMOKE[1], smoke=True, common=common)
    for name, (title, groups) in REHEARSAL.items():
        write(out, name, title, groups, smoke=True, common=common, extra=OWN_SEEDS, gate=False)
    print("wrote", sorted(p.name for p in out.glob("fin6_*")))


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A2A FT ablation verdict (rule pre-registered in launch/ablate_bench_cmd.txt before any data).

Usage: python3 ablate_verdict.py RUN_DIR [EP] [T]     (decision cell: EP 32, T 64 by default)
RUN_DIR: a2aep runs with --ablate (r<rep>-ep<EP>-s<stats>-v<V>.json). Each launch has the graphs off, ft (the variant V),
off2 and ft0 (the unchanged FT kernel). Per launch: D = M_ft - M_off, D0 = M_ft0 - M_off, cut = M_ft0 - M_ft (= D0 - D,
the control's FT delta minus the variant's), N = M_off2 - M_off. Per (stats, V) cell: medians over launches, Nmax = max
|N|. Control D = median of D0 over every launch of that stats arm.
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2aep_bench import ABLATE_BITS, cell  # noqa: E402

VARIANTS = ("0", "S", "S1", "F", "F8", "L", "A", "SO", "ALL")
SINGLES = ("S", "F", "L", "A", "SO")  # the items ALL removes together
EXPLAINS_US = 0.3
# Expectations from review/diag/a2a_ft_work_coderead.md, printed next to the result (not part of the rule).
EXPECT = {
    ("S", 1): "~2.1", ("S", 0): "~0", ("S1", 1): "~2.1 (= S)", ("S1", 0): "~0",
    ("F", 1): "0.15-0.35", ("F", 0): "0.15-0.35", ("F8", 1): "<= F", ("F8", 0): "<= F",
    ("L", 1): "0.1-0.2", ("L", 0): "0.1-0.2", ("A", 1): "0.02-0.05", ("A", 0): "0.02-0.05",
    ("SO", 1): "0.01-0.02", ("SO", 0): "0.01-0.02", ("ALL", 1): "~D0", ("ALL", 0): "~D0", ("0", 1): "0", ("0", 0): "0",
}


def variant_mismatch(run: dict, tokens: str) -> list:
    """Graphs whose recorded TRTLLM_A2A_FT_ABLATE differs from the launch's intent (off/off2 unset, ft = the
    variant, ft0 = 0, bits as ABLATE_BITS); a run without the record mismatches."""
    want = {"off": None, "off2": None, "ft": run["ablate"], "ft0": "0"}
    rec = run.get("graph_env", {}).get(tokens, {})
    bad = []
    for graph, env in want.items():
        got = rec.get(graph)
        bits = ABLATE_BITS.get(env) if env is not None else None
        if got is None or got.get("env") != env or got.get("bits") != bits:
            bad.append(f"{graph}: want env={env} bits={bits}, recorded {got}")
    return bad


CLAIMS_205_CTRL = {1: 2.96, 0: 0.85}  # CTRL EP32 T=64 FT delta (MASKSPILL 8289096)


def main() -> None:
    run_dir = sys.argv[1]
    ep = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    tokens = str(int(sys.argv[3]) if len(sys.argv) > 3 else 64)
    launches = {}
    refused = []
    for path in sorted(Path(run_dir).rglob("*.json")):
        run = json.loads(path.read_text())
        if run.get("ep") != ep or "ablate" not in run or tokens not in run.get("results", {}):
            continue
        bad = variant_mismatch(run, tokens)
        if bad:
            print(f"ABLATE_VARIANT_MISMATCH {path.name} ablate={run['ablate']} (launch refused): {'; '.join(bad)}")
            refused.append(path.name)
            continue
        launches.setdefault((int(run["eplb_stats"]), run["ablate"]), []).append(cell(run["results"][tokens]))
    med = statistics.median
    ok = not refused
    for stats in (1, 0):
        arm = [c for (s, _), cs in launches.items() if s == stats for c in cs]
        if not arm:
            print(f"ABLATE_VERDICT INCONCLUSIVE no launches at ep={ep} T={tokens} stats={stats}")
            ok = False
            continue
        d_ctrl = med(c["delta0"] for c in arm)
        print(f"ABLATE_CONTROL ep={ep} T={tokens} stats={stats} D0={d_ctrl:+.2f}us over {len(arm)} launches "
              f"(CLAIMS 205 CTRL {CLAIMS_205_CTRL[stats]:+.2f})"
              + (" ABLATE_NOTE control differs from CLAIMS 205 by > 0.4 us"
                 if ep == 32 and abs(d_ctrl - CLAIMS_205_CTRL[stats]) > 0.4 else ""))
        cuts = {}
        for v in VARIANTS:
            cs = launches.get((stats, v), [])
            if not cs:
                print(f"ABLATE_ITEM {v} stats={stats} MISSING")
                ok = False
                continue
            d, d0, cut = med(c["delta"] for c in cs), med(c["delta0"] for c in cs), med(c["cut"] for c in cs)
            n, nmax = med(c["noise"] for c in cs), max(abs(c["noise"]) for c in cs)
            valid = abs(n) < 0.3 and nmax < 0.5
            cuts[v] = cut
            each = " ".join(f"{c['cut']:+.2f}" for c in cs)
            if v == "0":
                result = "NULL" + (" ABLATE_NOTE |cut| >= 0.3 between identical kernels" if abs(cut) >= 0.3 else "")
            elif not valid:
                result = "INCONCLUSIVE (noise)"
            else:
                result = "EXPLAINS" if cut >= EXPLAINS_US else "NO"
            ok = ok and valid
            print(f"ABLATE_ITEM {v} stats={stats} n={len(cs)} D={d:+.2f}us D0={d0:+.2f}us cut={cut:+.2f}us ({each}) "
                  f"N={n:+.2f}us Nmax={nmax:.2f}us -> {result} [expect cut {EXPECT[(v, stats)]}]")
        if all(v in cuts for v in SINGLES + ("ALL",)):
            print(f"ABLATE_SUM stats={stats} sum of single cuts {'+'.join(SINGLES)} = "
                  f"{sum(cuts[v] for v in SINGLES):+.2f}us vs ALL cut {cuts['ALL']:+.2f}us (informational)")
        if ("ALL" in cuts) and launches.get((stats, "ALL")):
            residual = med(c["delta"] for c in launches[(stats, "ALL")])
            print(f"ABLATE_ALL_RESIDUAL stats={stats} D(ALL)={residual:+.2f}us -> "
                  + ("UNLISTED (something not in the list contributes)" if residual > EXPLAINS_US else "COVERED"))
    print("ABLATE_VALID" if ok else "ABLATE_INVALID (see INCONCLUSIVE / MISSING / ABLATE_VARIANT_MISMATCH lines)")


if __name__ == "__main__":
    main()

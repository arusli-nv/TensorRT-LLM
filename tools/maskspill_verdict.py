# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MASK-SPILL verdict (rule pre-registered in launch/maskspill_bench_cmd.txt before any data).

Usage: python3 maskspill_verdict.py NEW_DIR CTRL_DIR
NEW_DIR: a2aep runs on the mask-spill wheel; CTRL_DIR: the same nodes, same job, on the freeze wheel (spilling mask).
Per cell (EP, stats, T): D = median over launches of (M_ft - M_off), N = median of (M_off2 - M_off), Nmax = max |.|,
as a2aep_bench.py --summarize. Decision at EP32, T=64.
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2aep_bench import cell  # noqa: E402

CLAIMS_200 = {(32, 1, 64): 3.02, (32, 0, 64): 1.00}


def cells(directory: str) -> dict:
    runs = {}
    for path in sorted(Path(directory).rglob("*.json")):
        run = json.loads(path.read_text())
        for tokens, load in run.get("results", {}).items():
            runs.setdefault((run["ep"], int(run["eplb_stats"]), int(tokens)), []).append(cell(load))
    return {
        key: {
            "D": statistics.median(r["delta"] for r in rs),
            "N": statistics.median(r["noise"] for r in rs),
            "Nmax": max(abs(r["noise"]) for r in rs),
            "off": statistics.median(r["off"] for r in rs),
            "n": len(rs),
        }
        for key, rs in runs.items()
    }


def main() -> None:
    new, ctrl = cells(sys.argv[1]), cells(sys.argv[2])
    for name, table in (("new", new), ("ctrl", ctrl)):
        for (ep, stats, tokens), c in sorted(table.items()):
            print(f"MASKSPILL_CELL {name} ep={ep} stats={stats} T={tokens} n={c['n']} off={c['off']:.2f}us "
                  f"D={c['D']:+.2f}us N={c['N']:+.2f}us Nmax={c['Nmax']:.2f}us")
    keys = [(32, 1, 64), (32, 0, 64)]
    missing = [(name, k) for name, t in (("new", new), ("ctrl", ctrl)) for k in keys if k not in t]
    if missing:
        print(f"MASKSPILL_VERDICT INCONCLUSIVE missing cells {missing}")
        return
    noisy = [(name, k) for name, t in (("new", new), ("ctrl", ctrl)) for k in keys
             if abs(t[k]["N"]) >= 0.3 or t[k]["Nmax"] >= 0.5]
    for k in keys:
        if abs(ctrl[k]["D"] - CLAIMS_200[k]) > 0.4:
            print(f"MASKSPILL_NOTE ctrl {k} D={ctrl[k]['D']:+.2f} differs from CLAIMS 200 {CLAIMS_200[k]:+.2f} by > 0.4 us")
    d1, d0 = new[(32, 1, 64)]["D"], new[(32, 0, 64)]["D"]
    s_new = d1 - d0
    s_ctrl = ctrl[(32, 1, 64)]["D"] - ctrl[(32, 0, 64)]["D"]
    drop = 1 - s_new / s_ctrl if s_ctrl > 0 else float("nan")
    print(f"MASKSPILL_STATS_ONLY new={s_new:+.2f}us ctrl={s_ctrl:+.2f}us drop={100 * drop:.0f}%")
    if noisy:
        verdict = f"INCONCLUSIVE noise {noisy}"
    elif d1 <= 1.5 and drop >= 0.7:
        verdict = "CONFIRMED"
    elif d1 >= 2.5:
        verdict = "REFUTED"
    else:
        verdict = "PARTIAL"
    print(f"MASKSPILL_VERDICT {verdict} (EP32 T=64 stats D new={d1:+.2f} ctrl={ctrl[(32, 1, 64)]['D']:+.2f} us)")
    secondary = "EXPLAINED" if d0 <= 0.6 else "NOT_SPILL" if d0 >= 0.85 else "PARTIAL"
    print(f"MASKSPILL_NOSTATS {secondary} (EP32 T=64 no-stats D new={d0:+.2f} ctrl={ctrl[(32, 0, 64)]['D']:+.2f} us)")


if __name__ == "__main__":
    main()

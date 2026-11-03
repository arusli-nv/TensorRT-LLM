"""PR-description performance table from a finals_readout.py --json file: FT off first, then FT on, per mode and load.
Usage: python3 perf_table.py READOUT.json
"""
import json
import statistics
import sys

MODES = {"noeplb": "w/o EPLB", "offline": "offline EPLB", "online": "online EPLB"}
ARMS = {"noeplb": "std-noeplb", "offline": "std-offline", "online": "std-online"}

d = json.load(open(sys.argv[1]))
print("| Config (DeepSeek-R1 NVFP4, DEP32 GB200, ISL/OSL 1k/1k, MTP-1) | Metric | FT off | FT on | Delta "
      "[one-sided 95% UB] | n groups |")
print("|---|---|---|---|---|---|")
for mode, label in MODES.items():
    for load in (8, 32, 128):
        cell = d["cells"].get(f"{mode}/{load}")
        if not cell or not cell.get("n"):
            continue
        ub = f"{cell['ub95']:+.2f}%" if cell.get("ub95") is not None else "n<2"
        print(f"| {label}, {load} req/GPU | rank-0 decode step ms | {cell['off_step_ms']:.3f} | {cell['on_step_ms']:.3f} "
              f"| {cell['mean']:+.2f}% [{ub}] | {cell['n']} |")
        off = d["client"].get(f"{mode}/{load}/{ARMS[mode]}")
        on = d["client"].get(f"{mode}/{load}/{ARMS[mode]}-ft")
        if off and on:
            a, b = statistics.median(off["tpot_median"]), statistics.median(on["tpot_median"])
            print(f"| same | client TPOT median ms | {a:.2f} | {b:.2f} | {100 * (b - a) / a:+.2f}% (unpaired: medians over groups) "
                  f"| {min(off['n'], on['n'])} |")

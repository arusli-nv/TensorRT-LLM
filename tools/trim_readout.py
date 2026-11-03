"""Canonical decode step-time readout for same-node A/B runs (mvp/PREFLIGHT.md rule 14).

For every arm directory in each RUN_DIR (a case dir with result.json and rank-0.log) and every healthy-<load> row:
rank-0 steady full-load decode steps (mvp/analysis/dist32.case_stats), trim mean (mean of steps <= 1.5x median),
stall ms/step (steps > 1.5x median, summed / steady steps), client output tok/s per GPU (result.json / 32), and each
arm's trim-mean gap to the off arm of the same run dir and load.
Usage: python3 -I tools/trim_readout.py RUN_DIR... [--off ARM_DIR_NAME]   (default off arm: std-noeplb-1)
"""
import importlib.util
import json
import pathlib
import sys

C = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dist32", C / "mvp" / "analysis" / "dist32.py")
dist32 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dist32)


def main():
    args = sys.argv[1:]
    off = "std-noeplb-1"
    if "--off" in args:
        i = args.index("--off")
        off = args[i + 1]
        del args[i:i + 2]
    print("| run | arm | load | n steps | trim mean ms | gap vs off ms | stall ms/step | client tok/s/GPU |")
    print("|---|---|---|---|---|---|---|---|")
    for run in map(pathlib.Path, args):
        arms = sorted(d for d in run.iterdir() if (d / "result.json").exists())
        stats = {d.name: dist32.case_stats(d) for d in arms}
        client = {}
        for d in arms:
            for row in json.loads((d / "result.json").read_text())["rows"]:
                client[(d.name, row.get("label"))] = row.get("output_throughput", 0) / 32
        for arm in [off] + [d.name for d in arms if d.name != off]:
            for label, s in sorted(stats.get(arm, {}).items()):
                base = stats.get(off, {}).get(label)
                gap = f"{s['trim_mean'] - base['trim_mean']:+.3f}" if base and arm != off else ""
                print(f"| {run.name} | {arm} | {label} | {s['n']} | {s['trim_mean']:.3f} | {gap} | "
                      f"{s['stall_ms_per_step']:.2f} | {client.get((arm, label), 0):.1f} |")


if __name__ == "__main__":
    main()

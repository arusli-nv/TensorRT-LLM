"""Slow decode steps per healthy row: count of steady steps above 40 ms and above 2x the median, and their excess time.
With --since-ready BIN_S, also each row's excess time per BIN_S-second bin of time since the case's first row started
(first traffic, right after the server is ready), to show whether the stalls cluster early.
Usage: python3 burst_counts.py [--since-ready BIN_S] RUN_DIR [RUN_DIR ...]
"""
import json, pathlib, statistics, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import steady_state as ss
args = sys.argv[1:]
bin_s = None
if args[:1] == ["--since-ready"]:
    bin_s, args = float(args[1]), args[2:]
for run in args:
    for case in sorted(pathlib.Path(run).iterdir()):
        if not (case / "result.json").exists() or not (case / "rank-0.log").exists():
            continue
        all_rows = json.loads((case / "result.json").read_text()).get("rows", [])
        rows = [r for r in all_rows if r.get("label", "").startswith(("healthy", "burnin"))]
        ready = min((r["start_wall_s"] for r in all_rows), default=None)
        its = ss.iterations(case / "rank-0.log")
        for row in rows:
            start, end = row["start_wall_s"], row["start_wall_s"] + row["elapsed_s"]
            floor = 0.9 * row["load_per_gpu"]
            steps = [(wall, ms) for wall, ms, tok, ctx, active in its if start <= wall <= end and ctx == 0 and active >= floor]
            picked = [ms for _, ms in steps]
            if len(picked) < 20:
                continue
            med = statistics.median(picked)
            slow = [ms for ms in picked if ms > 40]
            slow2 = [ms for ms in picked if ms > 2 * med]
            excess = sum(ms - med for ms in slow2)
            print(f"{run.split('/')[-1]:22s} {case.name:12s} load={row['load_per_gpu']:4d} steps={len(picked):5d} med={med:6.2f} >40ms={len(slow):3d} >2xmed={len(slow2):3d} excess_ms={excess:8.1f} ({100*excess/sum(picked):5.1f}% of time) max={max(picked):7.1f}")
            if bin_s:
                bins = {}
                for wall, ms in steps:
                    b = int((wall - ready) // bin_s)
                    total, extra = bins.get(b, (0.0, 0.0))
                    bins[b] = (total + ms, extra + (ms - med if ms > 2 * med else 0.0))
                print("    since ready: " + " ".join(f"{b * bin_s:.0f}s:{100 * x / t:.1f}%" for b, (t, x) in sorted(bins.items())))

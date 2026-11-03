"""Rank-0 healthy-row step-time distributions, FT off vs on, per same-node run pair.

Uses harness steady_state.iterations(); steady steps as in steady(): in the row window, ctx == 0, num_active >= 0.9*load.
Prints JSON: per pair and load, both arms' distribution stats, plus the step-time profile across each round.
Usage: python3 -I dist32.py > out.json
"""
import json, pathlib, statistics, sys
R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "harness"))
from steady_state import iterations  # noqa: E402

PAIRS = [("STD-8141738", "std-noeplb-1", "std-noeplb-ft-1"), ("STD-8139740", "std-noeplb-1", "std-noeplb-ft-1"),
         ("STD-8139740", "std-noeplb-1", "std-noeplb-ulfm-1"), ("STD-8137627", "std-noeplb-1", "std-noeplb-ft-1"),
         ("STD-8137627", "std-noeplb-1", "std-noeplb-ulfm-1"),
         ("STD-8134758", "std-noeplb-1", "std-noeplb-ft-1"), ("STD-G1-8134758", "std-noeplb-1", "std-noeplb-ft-1"),
         ("STD-8133265", "std-noeplb-loc-1", "std-noeplb-ft-loc-1"), ("STD-8116111", "std-noeplb-1", "std-noeplb-ft-1"),
         ("STD-G1-8116111", "std-noeplb-1", "std-noeplb-ft-1"), ("STD-G1-8118134", "std-noeplb-1", "std-noeplb-ft-1"),
         ("STD-G1-8118573", "std-noeplb-1", "std-noeplb-ft-1"), ("STD-8121996", "std-noeplb-1", "std-noeplb-ft-1"),
         ("STD-8121996", "std-noeplb-nopdl-1", "std-noeplb-ft-nopdl-1")]
NBIN = 10


def pct(xs, q):
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def stats(its, row):
    s, e = row["start_wall_s"], row["start_wall_s"] + row["elapsed_s"]
    floor = 0.9 * row["load_per_gpu"]
    win = [x for x in its if s <= x[0] <= e]
    rounds, cur = [], []  # full-load decode steps per request round (rounds split at context steps)
    for wall, ms, tok, ctx, active in win:
        if ctx > 0:
            if cur:
                rounds.append(cur)
            cur = []
        elif active >= floor:
            cur.append((ms, tok))
    rounds.append(cur)
    st = [x for r in rounds for x in r]
    if len(st) < 20:
        return None
    ms = [m for m, _ in st]
    srt, med, mean = sorted(ms), statistics.median(ms), statistics.mean(ms)
    var = sum((v - mean) ** 2 for v in ms)
    prof = [[] for _ in range(NBIN)]
    for r in rounds:
        if len(r) >= 100:
            for i, (m, _) in enumerate(r):
                prof[i * NBIN // len(r)].append(m)
    return {"n": len(ms), "win_steps": len(win), "row_s": row["elapsed_s"], "med": med, "mean": mean,
            "trim_mean": statistics.mean(v for v in ms if v <= 1.5 * med),
            "stall_ms_per_step": sum(v for v in ms if v > 1.5 * med) / len(ms),
            "p10": pct(srt, .10), "p25": pct(srt, .25), "p75": pct(srt, .75), "p90": pct(srt, .90),
            "p99": pct(srt, .99), "gt15": sum(v > 1.5 * med for v in ms) / len(ms),
            "ac1": sum((ms[i] - mean) * (ms[i + 1] - mean) for i in range(len(ms) - 1)) / var if var else 0.0,
            "tok_step": statistics.median(t for _, t in st),
            "win_tok_s": sum(x[2] for x in win) / sum(x[1] for x in win) * 1e3,
            "client": row["output_throughput"] / row["gpus"], "tpot": row["median_tpot_ms"],
            "profile": [statistics.median(b) if b else None for b in prof]}


def case_stats(case):
    rows = json.loads((case / "result.json").read_text())["rows"]
    its = iterations(case / "rank-0.log")
    return {r["label"]: stats(its, r) for r in rows if r.get("label", "").startswith("healthy")}


def main():
    out = []
    for run, off, on in PAIRS:
        d = R / "runs" / run
        if not all((d / c / "rank-0.log").exists() and (d / c / "result.json").exists() for c in (off, on)):
            print("skip", run, off, on, file=sys.stderr)
            continue
        a, b = case_stats(d / off), case_stats(d / on)
        for label in a:
            if a[label] and b.get(label):
                out.append({"run": run, "off": off, "on": on, "label": label, "a": a[label], "b": b[label]})
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()

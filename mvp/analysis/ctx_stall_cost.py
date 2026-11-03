"""FT on vs off cost outside the headline (context steps, stalls, whole-window throughput), group-paired.

Usage: python3 ctx_stall_cost.py RUN_DIR... > out.md  (reuses steady_state / finals_readout parsing)
"""
import os
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "harness"))
import finals_readout as fr  # noqa: E402
import steady_state as ss  # noqa: E402

LOADS = (8, 32, 128, 512)
MODES = ("noeplb", "offline", "online")
BAND = (180.0, 210.0)
TRAIN_GAP = 10  # band steps within this many logged iterations form one train


def case_metrics(case, row):
    its = ss.iterations(case / "rank-0.log")
    start, end = row["start_wall_s"], row["start_wall_s"] + row["elapsed_s"]
    win = [(i, ms, tok, ctx) for i, (wall, ms, tok, ctx, act) in enumerate(its) if start <= wall <= end]
    ctx_ms = [ms for _, ms, _, ctx in win if ctx > 0]
    dec = [(i, ms) for i, ms, _, ctx in win if ctx == 0]
    band = [i for i, ms in dec if BAND[0] <= ms <= BAND[1]]
    trains = sum(1 for k, i in enumerate(band) if k == 0 or i - band[k - 1] > TRAIN_GAP)
    minutes = row["elapsed_s"] / 60
    return {"ctx_n": len(ctx_ms), "ctx_med": statistics.median(ctx_ms) if ctx_ms else fr.NAN,
            "ctx_mean": statistics.mean(ctx_ms) if ctx_ms else fr.NAN,
            "ctx_share": 100 * sum(ctx_ms) / 1e3 / row["elapsed_s"],
            "band_pm": len(band) / minutes, "trains_pm": trains / minutes,
            "tok_s": 32 * sum(t for _, _, t, _ in win) / row["elapsed_s"],
            "client_tok_s": row.get("output_throughput", fr.NAN)}


# The arm pair compared (CTX_OFF_SUFFIX / CTX_ON_SUFFIX): FT off vs on by default; e.g. "" vs "-ulfm" for the
# launcher alone, or "-ulfm" vs "-ft" for FT under the same launcher.
OFF_SUFFIX = os.environ.get("CTX_OFF_SUFFIX", "")
ON_SUFFIX = os.environ.get("CTX_ON_SUFFIX", "-ft")


def main():
    groups = {}
    for run in sys.argv[1:]:
        groups.update(fr.load_group(pathlib.Path(run).resolve(), list(LOADS), False))
    cells = {}
    for mode in MODES:
        for load in LOADS:
            off_arm, on_arm = f"std-{mode}{OFF_SUFFIX}", f"std-{mode}{ON_SUFFIX}"
            for g, ma, mb in fr.pairs(groups, off_arm, on_arm, load):
                arms = groups[g]
                off = case_metrics(arms[off_arm]["case_path"], arms[off_arm]["_rows"][load])
                on = case_metrics(arms[on_arm]["case_path"], arms[on_arm]["_rows"][load])
                off["stall"], on["stall"] = ma.get("stall_share", fr.NAN), mb.get("stall_share", fr.NAN)
                cells.setdefault((mode, load), []).append((g, off, on))
    return groups, cells


def agg(vals):
    v = [x for x in vals if x == x]
    return (f"{statistics.mean(v):+.2f} | {min(v):+.2f}..{max(v):+.2f}") if v else "- | -"


def table(cells, title, cols, cost):
    """cols: [(header, fn(off,on)->str)]; cost: fn(off,on)->float for mean/min..max."""
    print(f"\n### {title}\n")
    print("| mode | load | group | " + " | ".join(h for h, _ in cols) + " |")
    print("|---|---|---|" + "---|" * len(cols))
    for (mode, load), rows in sorted(cells.items(), key=lambda kv: (MODES.index(kv[0][0]), kv[0][1])):
        for g, off, on in rows:
            print(f"| {mode} | {load} | {g} | " + " | ".join(f(off, on) for _, f in cols) + " |")
        print(f"| **{mode}** | **{load}** | **mean / min..max of cost** | " + agg([cost(o, n) for _, o, n in rows])
              + " |" + " |" * (len(cols) - 2))


if __name__ == "__main__":
    groups, cells = main()
    p = fr.pct
    print("groups per cell:", {f"{m}/{l}": [g for g, _, _ in r] for (m, l), r in sorted(cells.items())})
    table(cells, "1. Context (prefill-containing) steps, rank-0 host step ms, FT off -> on",
          [("n off/on", lambda a, b: f"{a['ctx_n']}/{b['ctx_n']}"),
           ("median ms off/on", lambda a, b: f"{a['ctx_med']:.1f}/{b['ctx_med']:.1f}"),
           ("median cost %", lambda a, b: f"{p(b['ctx_med'], a['ctx_med']):+.2f}"),
           ("mean ms off/on", lambda a, b: f"{a['ctx_mean']:.1f}/{b['ctx_mean']:.1f}"),
           ("mean cost %", lambda a, b: f"{p(b['ctx_mean'], a['ctx_mean']):+.2f}"),
           ("wall share % off/on", lambda a, b: f"{a['ctx_share']:.1f}/{b['ctx_share']:.1f}")],
          lambda a, b: p(b["ctx_mean"], a["ctx_mean"]))
    table(cells, f"2. Stall share: % of decode full-load step time in steps > {fr.STALL_X:g}x median, FT off -> on",
          [("stall % off", lambda a, b: f"{a['stall']:.2f}"), ("stall % on", lambda a, b: f"{b['stall']:.2f}"),
           ("diff pp", lambda a, b: f"{b['stall'] - a['stall']:+.2f}"),
           ("180-210 ms steps/min off/on", lambda a, b: f"{a['band_pm']:.2f}/{b['band_pm']:.2f}"),
           ("trains/min off/on", lambda a, b: f"{a['trains_pm']:.2f}/{b['trains_pm']:.2f}")],
          lambda a, b: b["stall"] - a["stall"])
    table(cells, "3. Whole-window throughput: 32 x rank-0 generated tokens / window wall s, FT off -> on",
          [("tok/s off", lambda a, b: f"{a['tok_s']:.0f}"), ("tok/s on", lambda a, b: f"{b['tok_s']:.0f}"),
           ("cost %", lambda a, b: f"{-p(b['tok_s'], a['tok_s']):+.2f}"),
           ("client out tok/s off/on", lambda a, b: f"{a['client_tok_s']:.0f}/{b['client_tok_s']:.0f}"),
           ("client cost %", lambda a, b: f"{-p(b['client_tok_s'], a['client_tok_s']):+.2f}")],
          lambda a, b: -p(b["tok_s"], a["tok_s"]))

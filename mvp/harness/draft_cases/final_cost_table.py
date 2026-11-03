"""FT healthy cost tables for the final cost pack, pooled over jobs.

Each job has two run dirs: LADDER-<job> (online EPLB on; arms noft, noft-aa, ft, tail case noft-main-<r>) and
GHOST-COST-<job> (online EPLB off; arms lay-eplb0, lay-eplb0-aa, ft-ghost320). Cases are <arm>-<round>/ with
result.json and rank-0.log. Rows and steady-state fields are taken as in ab_stats.py (the last clean healthy-<load>
row; steady_state.py fields from rank-0.log). Per deployment x load and metric: arm means, cost % (positive = the
test arm is worse) from paired differences, pairs = same (run dir, round), pooled over all run dirs, 95% Student t
interval. The A/A pair (base vs base-aa) is the noise reference; base vs the main-build arm and any extra arms are
reported the same way.

Slow-mode (DECISIONS): a launch whose 8/GPU row spends >15% of its steady time in bursts (steps >2x the row median,
excess over the median, as burst_counts.py) is flagged slow-mode. Tables are printed with all launches and, when any
launch is flagged, again without the flagged launches (--slow-scope row: drop only the flagged rows of any load).
Verdict per metric: "COST +/-" if the cost CI excludes 0; "noise" if it includes 0 and its half-width is <= max(1.0
point, the A/A half-width); else "unresolved" (also n<2). The one-line verdict per deployment x load uses the headline
metrics (throughput, TTFT, TPOT; step time is support): cost if any is COST, else unresolved if any is, else within noise.

Usage: python3 final_cost_table.py [--ladder BASE,TEST,AA,MAIN] [--ghost BASE,TEST,AA] [--ladder-extra ARM ...]
           [--ghost-extra ARM ...] [--loads 8,32,128] JOB_ID_OR_RUN_DIR [...]
A job id expands to RUNS/LADDER-<id> and RUNS/GHOST-COST-<id>; a run dir whose name starts with GHOST is the ghost
deployment, any other the ladder. Leave AA/MAIN empty (e.g. noft,ft,,) when the runs have no such arm.
"""

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import ab_stats  # noqa: E402
import steady_state  # noqa: E402

RUNS = pathlib.Path(__file__).resolve().parents[2] / "runs"
SLOW_FRACTION = 0.15
METRICS = (("output_throughput", "out tok/s", 1), ("median_ttft_ms", "TTFT p50 ms", -1),
           ("p99_ttft_ms", "TTFT p99 ms", -1), ("median_tpot_ms", "TPOT p50 ms", -1),
           ("p99_tpot_ms", "TPOT p99 ms", -1), ("steady_step_ms", "step ms (support)", -1))
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
       11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086}


def burst_fraction(its: list, row: dict) -> float:
    start, end = row["start_wall_s"], row["start_wall_s"] + row["elapsed_s"]
    picked = [ms for wall, ms, tok, ctx, active in its if start <= wall <= end and ctx == 0 and active >= 0.9 * row["load_per_gpu"]]
    if len(picked) < steady_state.MIN_STEADY:
        return float("nan")
    med = statistics.median(picked)
    return sum(ms - med for ms in picked if ms > 2 * med) / sum(picked)


def load_case(case: pathlib.Path) -> dict:
    """load -> row (ab_stats.load_rows selection) with steady fields and burst_frac."""
    log = case / "rank-0.log"
    its = steady_state.iterations(log) if log.exists() else []
    rows = {}
    for row in json.loads((case / "result.json").read_text()).get("rows", []):
        if row.get("label") == f"healthy-{row['load_per_gpu']}" and ab_stats.clean(row):
            extra = {**steady_state.steady(its, row), "burst_frac": burst_fraction(its, row)} if its else {}
            rows[row["load_per_gpu"]] = {**row, **extra}
    return rows


def load_run(run: pathlib.Path, arms: set) -> dict:
    """(arm, (run name, round)) -> rows. Rounds are keyed by run dir so pairs stay within a job."""
    cases = {}
    for result in sorted(run.glob("*-[0-9]*/result.json")):
        arm, _, rnd = result.parent.name.rpartition("-")
        if arm in arms:
            cases[(arm, (run.name, int(rnd)))] = load_case(result.parent)
    return cases


def paired(cases: dict, base: str, test: str, load: int, metric: str, sign: int, drop: set):
    def values(arm):
        return {key: rows[load][metric] for (a, key), rows in cases.items()
                if a == arm and (a, key, load) not in drop and metric in rows.get(load, {})}
    off, on = values(base), values(test)
    keys = [k for k in off if k in on]
    costs = [sign * (off[k] - on[k]) / off[k] * 100 for k in keys]
    mean = statistics.mean(costs) if costs else float("nan")
    half = (T95.get(len(costs) - 1, 2.0) * statistics.stdev(costs) / len(costs) ** 0.5
            if len(costs) > 1 else float("nan"))
    return {"off": statistics.mean(off.values()) if off else float("nan"),
            "on": statistics.mean(on.values()) if on else float("nan"), "cost": mean, "half": half, "n": len(keys)}


def fmt(c: dict) -> str:
    if c["n"] == 0:
        return "-"
    if c["n"] < 2:
        return f"{c['cost']:+.2f} (n=1)"
    return f"{c['cost']:+.2f} [{c['cost'] - c['half']:+.2f}, {c['cost'] + c['half']:+.2f}] n={c['n']}"


def verdict(c: dict, aa) -> str:
    if c["n"] < 2:
        return "unresolved"
    if abs(c["cost"]) > c["half"]:
        return "COST +" if c["cost"] > 0 else "COST -"
    limit = max(1.0, aa["half"] if aa and aa["n"] >= 2 else 0.0)
    return "noise" if c["half"] <= limit else "unresolved"


def table(name, cases, roles, extras, load, drop) -> str:
    base, test, aa, main = roles
    counts = {}
    for (arm, key), rows in cases.items():
        if load in rows and (arm, key, load) not in drop:
            counts[arm] = counts.get(arm, 0) + 1
    if not counts.get(base) or not counts.get(test):
        return None
    others = [(f"A/A {aa}", aa)] if aa else []
    others += [(f"vs {arm}", arm) for arm in ([main] if main else []) + list(extras)]
    print(f"\n#### {name} load={load}/GPU  n per arm: " + ", ".join(f"{a}={counts.get(a, 0)}" for a in
                                                                    (base, test, aa, main, *extras) if a))
    print(f"| metric | {base} | {test} | cost % [95% CI] | " + "".join(o[0] + " cost % | " for o in others)
          + "verdict |")
    print("|---" * (5 + len(others)) + "|")
    verdicts = []
    for metric, label, sign in METRICS:
        c = paired(cases, base, test, load, metric, sign, drop)
        ref = [paired(cases, base, arm, load, metric, sign, drop) for _, arm in others]
        v = verdict(c, ref[0] if aa else None)
        verdicts.append((label, v))
        print(f"| {label} | {c['off']:.2f} | {c['on']:.2f} | {fmt(c)} | " + "".join(fmt(r) + " | " for r in ref)
              + f"{v} |")
    support = verdicts.pop()
    costly = [f"{lab} {v[-1]}" for lab, v in verdicts if v.startswith("COST")]
    unresolved = [lab for lab, v in verdicts if v == "unresolved"]
    headline = (f"cost ({', '.join(costly)})" if costly else
                f"unresolved ({', '.join(unresolved)})" if unresolved else "within noise")
    headline += f"; step time (support): {support[1]}"
    ref_note = "" if aa else " (no A/A arm: noise bar 1.0 pt)"
    return f"VERDICT {name} load={load}: {headline}" + ref_note


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("targets", nargs="+")
    p.add_argument("--ladder", default="noft,ft,noft-aa,noft-main")
    p.add_argument("--ghost", default="lay-eplb0,ft-ghost320,lay-eplb0-aa,")
    p.add_argument("--ladder-extra", nargs="*", default=[])
    p.add_argument("--ghost-extra", nargs="*", default=[])
    p.add_argument("--loads", default="8,32,128")
    p.add_argument("--slow-scope", choices=("launch", "row"), default="launch")
    args = p.parse_args()
    runs = {"LADDER (online EPLB on)": [], "GHOST-COST (online EPLB off)": []}
    for t in args.targets:
        path = pathlib.Path(t)
        dirs = [path] if path.is_dir() else [RUNS / f"LADDER-{t}", RUNS / f"GHOST-COST-{t}"]
        for d in dirs:
            if not d.is_dir():
                print(f"missing run dir {d}", file=sys.stderr)
                continue
            runs["GHOST-COST (online EPLB off)" if d.name.startswith("GHOST") else "LADDER (online EPLB on)"].append(d)
    loads = [int(x) for x in args.loads.split(",")]
    for name, dirs in runs.items():
        if not dirs:
            continue
        ghost = name.startswith("GHOST")
        roles = ((args.ghost if ghost else args.ladder).split(",") + ["", "", ""])[:4]
        extras = args.ghost_extra if ghost else args.ladder_extra
        cases = {}
        for d in dirs:
            cases.update(load_run(d, {r for r in roles if r} | set(extras)))
        print(f"\n### {name}: {', '.join(d.name for d in dirs)}; cost % = {roles[1]} worse than {roles[0]}")
        flagged_rows = {(arm, key, load) for (arm, key), rows in cases.items() for load, row in rows.items()
                        if row.get("burst_frac", 0) > SLOW_FRACTION}
        for arm, key, load in sorted(flagged_rows):
            print(f"slow-mode row: {key[0]} {arm}-{key[1]} load={load} bursts "
                  f"{100 * cases[(arm, key)][load]['burst_frac']:.1f}% of time")
        if args.slow_scope == "launch":
            slow = {(arm, key) for arm, key, load in flagged_rows if load == 8}
            drop = {(arm, key, load) for (arm, key) in slow for load in loads}
            print(f"slow-mode launches (8/GPU row): {len(slow)} of {len(cases)}: "
                  + (", ".join(f"{k[0]} {a}-{k[1]}" for a, k in sorted(slow)) or "none"))
            arms = sorted({a for a, _ in cases})
            print("slow-mode launches per arm (DECISIONS: >1/3 gets one more round): " + ", ".join(
                f"{a} {sum(s == a for s, _ in slow)}/{sum(c == a for c, _ in cases)}" for a in arms))
        else:
            drop = flagged_rows
        lines = []
        for load in loads:
            lines.append(table(name, cases, roles, extras, load, set()))
            if any(l == load for _, _, l in drop):
                lines.append(table(name + " WITHOUT slow-mode", cases, roles, extras, load, drop))
        print()
        print("\n".join(line for line in lines if line))


if __name__ == "__main__":
    main()

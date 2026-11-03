#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fill the [FINALS: ...] placeholders of the design doc draft from the finals outputs, never inventing a number.

Sources (only these): the printed tables of tools/finals_stats.py (--stats-out, its stdout saved to a file) checked
against its --csv (every printed cell mean is recomputed from the CSV; a mismatch disables every stats-fed value);
KILL_FINAL_VERDICT lines (kill_final.txt), COLD_RESTART lines (cold_restart.txt) and ACCURACY_KILL_CASE / _VERDICT
lines (accuracy_kill.txt) of the run dirs given. Kill / restart modes come from the case's config-0.yaml, as in
tools/finals_figures.py. A placeholder without a mapped source, or whose source is missing, stays as it is and the
reason is printed; a placeholder only partly covered keeps a narrower [FINALS: ...] for the uncovered part.

Writes a NEW file (never the draft) and <out stem>_provenance.md: every filled value -> source file:line.
Usage: python3 -I tools/fill_design_doc.py --stats-out STATS.txt --stats-csv STATS.csv [--doc DRAFT] [--out FILLED]
           [--label TEXT] [--list] RUN_DIR...
"""
import argparse
import csv
import importlib.util
import json
import pathlib
import re
import statistics
import sys

C = pathlib.Path(__file__).resolve().parents[1]
MODES = ("noeplb", "offline", "online")
MODE_LABEL = {"noeplb": "no EPLB", "offline": "offline EPLB", "online": "online EPLB", "?": "mode unknown"}
KIND_LABEL = {"rank5": "rank kill", "rank5pf": "rank kill in prefill", "rank5upd": "rank kill in EPLB update",
              "node1": "node kill", "twokill": "two sequential losses", "stop": "frozen rank",
              "rank5acc": "rank kill + GSM8K", "rank0": "rank 0 kill (must end)", "deadline": "deadline (must end)"}
PROCESS_LOSS = ("rank5", "rank5pf", "rank5upd", "node1", "twokill", "rank5acc")  # detected by ULFM, not hang
METRIC_KEY = {"trim mean step": "trim", "full mean step": "full", "stall ms/step": "stall",
              "client tok/s/GPU": "client", "rank-0 window tok/s/GPU": "window"}
PLACEHOLDER = re.compile(r"\[FINALS: (.*?)\]", re.S)


def norm(text):
    return " ".join(text.split())


def rel(path):
    p = pathlib.Path(path).resolve()
    return str(p.relative_to(C)) if p.is_relative_to(C) else str(p)


def config_mode(case):
    """tools/finals_figures.py config_mode, so the doc and the figures name a case's mode alike."""
    cfg = case / "config-0.yaml"
    if not cfg.exists():
        return "?"
    text = cfg.read_text(errors="replace")
    if "load_balancer:" not in text:
        return "noeplb"
    upd = re.search(r"layer_updates_per_iter:\s*(\d+)", text)
    slots = re.search(r"num_slots:\s*(\d+)", text)
    if upd and int(upd[1]) > 0:
        return "online"
    if "initial_global_assignments" in text or (slots and int(slots[1]) > 256):
        return "offline"
    return "noeplb"


def kill_load(case):
    """The kill row's per-GPU load: its result.json load row, as the harness verdict reads it."""
    try:
        rows = json.loads((case / "result.json").read_text()).get("rows", [])
    except (OSError, ValueError):
        return None
    return next((r.get("load_per_gpu") for r in rows if r.get("label") in ("load", "warm-kill")), None)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def g(x):
    """A number as the source would print it (no trailing zeros)."""
    return f"{x:.3f}".rstrip("0").rstrip(".") if x is not None else "-"


def spread(values, unit=""):
    """median (min-max, n) of the values; the formula every time/ratio cell uses."""
    if len(values) == 1:
        return f"{g(values[0])}{unit} (n=1)"
    return (f"median {g(statistics.median(values))}{unit} (min-max {g(min(values))}-{g(max(values))}{unit}, "
            f"n={len(values)})")


# ---------------------------------------------------------------- sources


class Stats:
    """finals_stats printed tables (cells, per mode, claim, parity cells), each row with its file:line."""

    def __init__(self, out_path, csv_path):
        self.cells, self.per_mode, self.parity, self.claim, self.invalid = {}, {}, {}, {}, []
        self.error = None
        if not out_path or not csv_path:
            self.error = "finals_stats --stats-out / --stats-csv not given"
            return
        section, src = "", rel(out_path)
        for i, line in enumerate(pathlib.Path(out_path).read_text().splitlines(), 1):
            if line.startswith("## "):
                section = line
                continue
            if line.startswith("- INVALID"):
                self.invalid.append((line, f"{src}:{i}"))
            m = re.match(r"- (rule [AB]) .*?'FT cost < 1%' (CLAIMED|NOT claimed)", line)
            if m and section.startswith("## Claim"):
                self.claim[m[1]] = (m[2], f"{src}:{i}")
            cols = [c.strip() for c in line.strip().strip("|").split("|")] if line.startswith("|") else []
            if not cols or cols[0] not in MODES:  # headers, separators and the per-pair tables
                continue
            row = {"n": cols[-5], "mean": cols[-4], "ci": cols[-3], "ub": cols[-1], "src": f"{src}:{i}"}
            if len(cols) == 8 and section.startswith("## Cells"):
                self.cells[(cols[0], int(cols[1].split()[0]), cols[2])] = dict(row, appendix="appendix" in cols[1])
            elif len(cols) == 7 and section.startswith("## Per mode"):
                self.per_mode[(cols[0], cols[1])] = row
            elif len(cols) == 8 and section.startswith("## Descriptive") and "our FT off (parity)" in section:
                self.parity[(cols[0], int(cols[1].split()[0]), cols[2])] = row
        self.csv = rel(csv_path)
        with open(csv_path) as fh:
            rows = list(csv.DictReader(fh))
        for (mode, load, metric), cell in self.cells.items():
            vals = [float(r[METRIC_KEY[metric]]) for r in rows if r["mode"] == mode and int(r["load"]) == load]
            if str(len(vals)) != cell["n"] or format(statistics.mean(vals), "+.2f") != cell["mean"]:
                self.error = (f"printed cell {mode} {load} {metric} (n {cell['n']}, mean {cell['mean']}) does not match "
                              f"the CSV (n {len(vals)}) at {cell['src']}: stats output and CSV are not one run")
                return
        if not self.cells:
            self.error = f"no cell table in {src}"

    def cell(self, mode, load, metric):
        return self.cells.get((mode, load, metric))

def invalid_cases(run):
    """Case names a run dir's INVALID_CASES file excludes (one per line; "*" = every case), e.g. after a node fault."""
    path = run / "INVALID_CASES"
    return {line.strip() for line in path.read_text().splitlines() if line.strip()} if path.exists() else set()



class Runs:
    def __init__(self, runs):
        self.kills, self.colds, self.acc, self.acc_cases, self.dirs = [], [], [], {}, []
        for run in runs:
            for name, tag, sink in (("kill_final.txt", "KILL_FINAL_VERDICT ", self.kills),
                                    ("cold_restart.txt", "COLD_RESTART ", self.colds),
                                    ("accuracy_kill.txt", "ACCURACY_KILL_VERDICT ", self.acc),
                                    ("accuracy_kill.txt", "ACCURACY_KILL_CASE ", None)):
                path = run / name
                if not path.exists():
                    continue
                if run not in self.dirs:
                    self.dirs.append(run)
                for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                    if not line.startswith(tag):
                        continue
                    parts = line.split()
                    if tag != "COLD_RESTART " and (parts[1] in invalid_cases(run) or "*" in invalid_cases(run)):
                        continue
                    rec = {"run": run, "src": f"{rel(path)}:{i}", "line": line,
                           "f": dict(re.findall(r"(\w+)=(\S+)", line))}
                    if tag == "KILL_FINAL_VERDICT ":
                        rec.update(case=parts[1], kind=parts[1].rsplit("-", 1)[0], ok=parts[2] == "PASS",
                                   mode=config_mode(run / parts[1]), load=kill_load(run / parts[1]))
                    elif tag == "COLD_RESTART ":
                        rec.update(kind=parts[1], mode=config_mode(run / rec["f"].get("relaunch", "-")))
                    elif not (run / parts[1]).is_dir():
                        continue  # accuracy_kill.txt repeats the allocation's other run dir's lines
                    elif tag == "ACCURACY_KILL_CASE ":
                        self.acc_cases[(run, parts[1])] = rec
                        continue
                    else:
                        rec.update(case=parts[1], ok=parts[2] == "PASS")
                    sink.append(rec)

    def kill_values(self, kinds, field, mode=None, load=None):
        out = [(v, k["src"]) for k in self.kills if k["kind"] in kinds and (mode is None or k["mode"] == mode)
               and (load is None or k["load"] == load) for v in [num(k["f"].get(field))] if v is not None]
        return [v for v, _ in out], [s for _, s in out]

    def phases(self, kinds, phase):
        out = []
        for k in self.kills:
            m = re.search(rf"(?:^|,){phase}=([0-9.]+)", k["f"].get("phases1", ""))
            if k["kind"] in kinds and m:
                out.append((float(m[1]), k["src"]))
        return [v for v, _ in out], [s for _, s in out]


# ---------------------------------------------------------------- fillers: (text or None, sources, why)


def modes_present(kills, kind):
    return [m for m in MODES + ("?",) if any(k["kind"] == kind and k["mode"] == m for k in kills)]


def cells_of(kills, kind):
    """(mode, load) of a kill row's cases, modes in order."""
    return sorted({(k["mode"], k["load"]) for k in kills if k["kind"] == kind},
                  key=lambda c: ((MODES + ("?",)).index(c[0]), c[1] or 0))


def resume(runs, kind):
    vals, srcs = runs.kill_values((kind,), "resume_s")
    if not vals:
        return None, [], f"no KILL_FINAL_VERDICT {kind} line with a resume_s"
    rows = [k for k in runs.kills if k["kind"] == kind]
    loads = "/".join(str(x) for x in sorted({k["load"] for k in rows if k["load"]}))
    return f"{spread(vals, ' s')}; {sum(k['ok'] for k in rows)}/{len(rows)} PASS; {loads} req/GPU", srcs, ""


def stats_cells(stats, metrics, loads=None, appendix=False, short=None):
    if stats.error:
        return None, [], stats.error
    parts, srcs = [], []
    for mode in MODES:
        for (m, load, metric), c in sorted(stats.cells.items(), key=lambda kv: kv[0][1]):
            if m != mode or metric not in metrics or (c["appendix"] and not appendix) or (loads and load not in loads):
                continue
            tag = " (appendix)" if c["appendix"] else ""
            label = (short or {}).get(metric, metric)
            parts.append(f"{MODE_LABEL[mode]} {load}/GPU{tag}{' ' + label if label else ''} {c['mean']}% "
                         f"(95% CI {c['ci']}, n={c['n']})")
            srcs.append(c["src"])
    if not parts:
        return None, [], f"no cell for {metrics} in the stats output"
    return "; ".join(parts), srcs + [stats.csv], ""


def healthy_tput(stats, runs):
    text, srcs, why = stats_cells(stats, ("client tok/s/GPU",), short={"client tok/s/GPU": ""})
    return (f"client output tok/s/GPU, FT on vs off, % = (on - off) / off (negative = FT slower): {text}"
            if text else None), srcs, why


def per_step(stats, runs):
    if stats.error:
        return None, [], stats.error
    parts, srcs = [], []
    for mode in MODES:
        t, f = stats.per_mode.get((mode, "trim mean step")), stats.per_mode.get((mode, "full mean step"))
        if t and f:
            parts.append(f"{MODE_LABEL[mode]} trim {t['mean']}% / full {f['mean']}% (one-sided 95% UB {t['ub']} / "
                         f"{f['ub']}, n={t['n']})")
            srcs += [t["src"], f["src"]]
    if not parts:
        return None, [], "no per-mode rows in the stats output"
    return "; ".join(parts), srcs + [stats.csv], ""


def detect(stats, runs):
    vals, srcs = runs.phases(PROCESS_LOSS, "detect")
    if not vals:
        return None, [], "no phases1 detect= on a process-loss KILL_FINAL_VERDICT line"
    return f"{spread(vals, ' s')} from the victim's kill to rank 0's phase=detect (rank, node, two-loss rows)", srcs, ""


def matrix(stats, runs):
    parts, srcs = [], []
    for kind in KIND_LABEL:
        rows = [k for k in runs.kills if k["kind"] == kind]
        if rows:
            per = ", ".join(f"{MODE_LABEL[m]} {sum(k['ok'] for k in rows if k['mode'] == m)}/"
                            f"{sum(k['mode'] == m for k in rows)}" for m in modes_present(rows, kind))
            parts.append(f"{KIND_LABEL[kind]} {sum(k['ok'] for k in rows)}/{len(rows)} PASS ({per})")
            srcs += [k["src"] for k in rows]
    acc = [a for a in runs.acc if "kill" in a["case"] and "nokill" not in a["case"]]
    if acc:
        parts.append(f"GSM8K with a rank kill {sum(a['ok'] for a in acc)}/{len(acc)} PASS")
        srcs += [a["src"] for a in acc]
    if runs.colds:
        parts.append(f"restarts (FT off) n={len(runs.colds)}")
        srcs += [c["src"] for c in runs.colds]
    if not parts:
        return None, [], "no kill, accuracy or restart lines"
    jobs = sorted({re.search(r"-(\d{6,})", r.name)[1] for r in runs.dirs if re.search(r"-(\d{6,})", r.name)})
    return ("kill runs: " + "; ".join(parts) + f"; jobs {', '.join(jobs)}; "
            "[FINALS: integration cases on the frozen build]"), srcs, "integration cases: no mapped source"


def parity(stats, runs):
    if stats.error:
        return None, [], stats.error
    parts, srcs = [], []
    for (mode, load, metric), c in sorted(stats.parity.items(), key=lambda kv: (MODES.index(kv[0][0]), kv[0][1])):
        if metric in ("trim mean step", "client tok/s/GPU"):
            parts.append(f"{MODE_LABEL[mode]} {load}/GPU {'trim step' if metric[0] == 't' else 'client tok/s'} "
                         f"{c['mean']}% (95% CI {c['ci']}, n={c['n']})")
            srcs.append(c["src"])
    if not parts:
        return None, [], "no 'main -> our FT off (parity)' cell table in the stats output"
    return "our FT off vs main, % = (ours - main) / main: " + "; ".join(parts), srcs, ""


def healthy_cost(stats, runs, loads=(8, 32, 128)):
    if stats.error:
        return None, [], stats.error
    parts, srcs = [], []
    for mode in MODES:
        for load in loads:
            cs = [(label, stats.cell(mode, load, metric)) for label, metric in
                  (("client tok/s", "client tok/s/GPU"), ("trim step", "trim mean step"), ("full step", "full mean step"))]
            if not all(c for _, c in cs):
                continue
            tag = " (appendix)" if cs[0][1]["appendix"] else ""
            parts.append(f"{MODE_LABEL[mode]} {load}/GPU{tag}: " + ", ".join(f"{label} {c['mean']}% (CI {c['ci']})"
                                                                         for label, c in cs) + f", n={cs[0][1]['n']}")
            srcs += [c["src"] for _, c in cs]
    if not parts:
        return None, [], f"no cells at loads {loads} in the stats output"
    text = "; ".join(parts)
    srcs.append(stats.csv)
    claims = [f"{rule} {stats.claim[rule][0]}" for rule in ("rule A", "rule B") if rule in stats.claim]
    srcs += [stats.claim[rule][1] for rule in ("rule A", "rule B") if rule in stats.claim]
    return (f"FT on vs off, % = (on - off) / off: {text}; per-step 'FT cost < 1%': {', '.join(claims) or 'no claim line'}"
            "; [FINALS: TPOT and TTFT per mode at 8/32/128 req/GPU]"), srcs, "TPOT/TTFT: not in finals_stats output"


def recovery(stats, runs):
    parts, srcs = [], []
    for kind in ("rank5", "node1", "twokill", "stop", "rank5pf", "rank5upd"):
        per = []
        for mode, load in cells_of(runs.kills, kind):
            vals, s = runs.kill_values((kind,), "resume_s", mode, load)
            if vals:
                per.append(f"{MODE_LABEL[mode]} {load}/GPU {spread(vals, ' s')}")
                srcs += s
        if per:
            parts.append(f"{KIND_LABEL[kind]}: " + ", ".join(per))
    if not parts:
        return None, [], "no KILL_FINAL_VERDICT resume_s"
    phase = []
    for p in ("detect", "proposed", "ready", "committed", "first_step"):
        vals, s = runs.phases(("rank5", "node1"), p)
        if vals:
            phase.append(f"{p} {g(statistics.median(vals))} s")
            srcs += s
    text = "kill -> all ranks resumed: " + "; ".join(parts)
    if phase:
        text += f"; rank-0 phases from the kill (median, rank and node kills): {', '.join(phase)}"
    restart = []
    for mode in MODES + ("?",):
        cs = [c for c in runs.colds if c["mode"] == mode and num(c["f"].get("down_s")) is not None]
        if cs:
            restart.append(f"{MODE_LABEL[mode]} {spread([num(c['f']['down_s']) for c in cs], ' s')}")
            srcs += [c["src"] for c in cs]
    if restart:
        text += f"; restart in the same allocation, FT off, kill -> serving again (down_s): {', '.join(restart)}"
    return text + "; [FINALS: the authority's stage durations (AUT:585-597)]", srcs, "authority stages: no mapped source"


def after_loss(stats, runs):
    parts, srcs = [], []
    for kind in ("rank5", "node1"):
        for mode, load in cells_of(runs.kills, kind):
            vals, s = runs.kill_values((kind,), "h23_tput_ratio", mode, load)
            live = {k["f"].get("h23_live") for k in runs.kills if k["kind"] == kind and k["mode"] == mode}
            if vals:
                parts.append(f"{KIND_LABEL[kind]} {MODE_LABEL[mode]} {load}/GPU {spread(vals, 'x')} "
                             f"(live {', '.join(sorted(live))})")
                srcs += s
    if not parts:
        return None, [], "no KILL_FINAL_VERDICT h23_tput_ratio"
    return "rank-0 decode tok/s x live ranks, after / before the loss: " + "; ".join(parts), srcs, ""


def memory(stats, runs):
    parts, srcs = [], []
    for kind in ("rank5", "node1"):
        top, med, s = [], [], []
        for k in runs.kills:
            m = re.match(r"max(\d+)@[0-9,]+/med(\d+)/min(\d+)", k["f"].get("restored_mb", ""))
            if k["kind"] == kind and m:
                top.append(float(m[1]))
                med.append(float(m[2]))
                s.append(k["src"])
        if top:
            parts.append(f"{KIND_LABEL[kind]}: busiest survivor {spread(top, ' MB')}, median survivor "
                         f"{spread(med, ' MB')}")
            srcs += s
    if not parts:
        return None, [], "no restored_mb on KILL_FINAL_VERDICT lines"
    return ("[FINALS: ghost slot healthy bytes per GPU]; memory behind restored expert slots per survivor "
            "(restored_mb): " + "; ".join(parts) + "; [FINALS: KV lent to the remap and when returned]"), srcs, \
        "ghost healthy bytes, KV lent: no mapped source"


def accuracy(stats, runs):
    parts, srcs = [], []
    for a in runs.acc:
        if "nokill" in a["case"] or "kill" not in a["case"]:
            continue
        m = re.search(r"vs (\S+) strict=(\S+) ref=(\S+) delta=(\S+) 2se=(\S+)", a["line"])
        ref = runs.acc_cases.get((a["run"], m[1])) if m else None
        if not m or not ref or ref["f"].get("ft") != "False" or ref["f"].get("kill") != "0":
            return None, [], f"{a['src']}: reference is not an FT-off no-kill ACCURACY_KILL_CASE in the same run dir"
        parts.append(f"strict-match {m[2].replace('+-', '±')} with a rank kill vs {m[3].replace('+-', '±')} FT off "
                     f"without (delta {m[4]}, 2 SE {m[5]}, {a['line'].split()[2]}; n={a['f'].get('n')} questions)")
        srcs += [a["src"], ref["src"]]
    if not parts:
        return None, [], "no ACCURACY_KILL_VERDICT for a kill case"
    return "GSM8K " + "; ".join(parts), srcs, ""


def unmapped(why):
    return lambda stats, runs: (None, [], why)


# Placeholder text (whitespace-normalized) -> filler. Order is the doc's.
MAP = {
    "rank-kill time to resume, median/min-max, n": (
        lambda s, r: resume(r, "rank5"), "KILL_FINAL_VERDICT rank5 rows: median/min/max of resume_s, n, k/n PASS"),
    "node-kill time to resume, median/min-max, n": (
        lambda s, r: resume(r, "node1"), "KILL_FINAL_VERDICT node1 rows: median/min/max of resume_s, n, k/n PASS"),
    "FT on vs off output throughput, per mode and load": (
        healthy_tput, "finals_stats Cells, client tok/s/GPU, headline loads: mean %, 95% t-CI, n (CSV-checked)"),
    "per-step cost, no EPLB / offline / online": (
        per_step, "finals_stats Per mode: trim and full mean step mean % and one-sided 95% UB, n"),
    "kill-to-detect": (
        detect, "KILL_FINAL_VERDICT process-loss rows (rank5/pf/upd/acc, node1, twokill): phases1 detect= median/min/max, n"),
    "test matrix (integration cases, kill runs per mode, accuracy with a kill), runs and n": (
        matrix, "KILL_FINAL_VERDICT k/n PASS per row and mode; ACCURACY_KILL_VERDICT kill k/n; COLD_RESTART n; job ids; "
                "integration cases unmapped"),
    "FT off vs main parity, per mode and load": (
        parity, "finals_stats Descriptive 'main -> our FT off (parity)' cells: trim step and client tok/s mean %, CI, n"),
    "FT off on the ULFM launcher vs FT off on UCX, to separate the transport from FT": (
        unmapped("finals_stats ignores the -ulfm arm (no pair table for it); no mapped source"), "unmapped"),
    "output throughput, TPOT, TTFT per mode at 8/32/128 req/GPU; per-step decode cost": (
        healthy_cost, "finals_stats Cells (8 appendix, 32, 128): client tok/s, trim, full mean %, CI, n; Claim rule A/B; "
                      "TPOT/TTFT unmapped"),
    "time from kill to resume, rank and node kills, per mode; phase breakdown from the `phase=` log lines "
    "(PLR:1125-1130) and the authority's stage durations (AUT:585-597)": (
        recovery, "KILL_FINAL_VERDICT resume_s median/min/max/n per row x mode; phases1 medians (rank5+node1); "
                  "COLD_RESTART down_s median/min/max/n; authority stages unmapped"),
    "decode throughput after vs before, rank and node loss": (
        after_loss, "KILL_FINAL_VERDICT h23_tput_ratio (est_tok_s_after / est_tok_s_before) median/min/max/n per row x mode"),
    "ghost slot healthy bytes per GPU; bytes mapped at restore; KV lent to the remap and when returned": (
        memory, "KILL_FINAL_VERDICT restored_mb max/med per case, median/min/max/n (rank5, node1); ghost bytes, KV "
                "lent unmapped"),
    "GSM8K with a kill vs FT off without": (
        accuracy, "ACCURACY_KILL_VERDICT kill case vs its FT-off no-kill reference: strict-match acc±SE, delta, 2 SE"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="*", type=pathlib.Path)
    p.add_argument("--doc", type=pathlib.Path, default=C / "mvp" / "DESIGN_DOC_v5final_draft.md")
    p.add_argument("--out", type=pathlib.Path, default=C / "mvp" / "DESIGN_DOC_v5final_filled.md")
    p.add_argument("--stats-out", type=pathlib.Path, help="finals_stats.py stdout, saved to a file")
    p.add_argument("--stats-csv", type=pathlib.Path, help="finals_stats.py --csv output of the same run")
    p.add_argument("--label", default="", help='e.g. "TEST - rehearsal data, not finals": a banner on both outputs')
    p.add_argument("--list", action="store_true", help="list the placeholders and their mapping, write nothing")
    args = p.parse_args()
    doc = args.doc.read_text()
    found = [(m, doc.count("\n", 0, m.start()) + 1) for m in PLACEHOLDER.finditer(doc)]
    print(f"{len(found)} placeholders in {rel(args.doc)}")
    for i, (m, line) in enumerate(found, 1):
        print(f"P{i} line {line}: [FINALS: {norm(m[1])}]\n    -> {MAP.get(norm(m[1]), (None, 'UNMAPPED'))[1]}")
    if args.list:
        return
    if args.out.resolve() == args.doc.resolve():
        sys.exit("--out must be a new file, not the draft")

    stats, runs = Stats(args.stats_out, args.stats_csv), Runs([r.resolve() for r in args.runs])
    if stats.error:
        print(f"NOTE stats: {stats.error}")
    manifest = importlib.util.spec_from_file_location("finals_manifest", C / "tools" / "finals_manifest.py")
    fm = importlib.util.module_from_spec(manifest)
    manifest.loader.exec_module(fm)
    builds = {}
    for run in runs.dirs:
        for case in run.iterdir():
            for dist in fm.case_builds(case)[0].values() if case.is_dir() else ():
                builds.setdefault(fm.build_name(dist), set()).add(run.name)
    for b, names in sorted(builds.items()):
        print(f"build {b}: {', '.join(sorted(names))}")
    mixed = f"WARNING: inputs span {len(builds)} builds ({', '.join(sorted(builds))})" if len(builds) > 1 else ""
    if mixed:
        print(mixed)

    out, prov, pos = [], [], 0
    for i, (m, line) in enumerate(found, 1):
        key = norm(m[1])
        filler = MAP.get(key)
        text, srcs, why = filler[0](stats, runs) if filler else (None, [], "no mapping for this placeholder text")
        out.append(doc[pos:m.start()])
        pos = m.end()
        if text is None:
            out.append(m[0])
            print(f"LEFT P{i} line {line}: {why}")
            continue
        out.append(text)
        prov.append((i, line, key, text, srcs))
        print(f"FILLED P{i} line {line}" + (f" (partly; left: {why})" if why else ""))
    out.append(doc[pos:])
    banner = "".join(f"> **{x}**\n\n" for x in (args.label, mixed) if x)
    args.out.write_text(banner + "".join(out))
    pv = args.out.with_name(args.out.stem + "_provenance.md")
    lines = [f"# Provenance for {args.out.name}", "", banner.strip(), "",
             f"Draft: `{rel(args.doc)}`; stats: `{rel(args.stats_out) if args.stats_out else '-'}` + "
             f"`{rel(args.stats_csv) if args.stats_csv else '-'}`; builds: "
             + "; ".join(f"{b} ({', '.join(sorted(n))})" for b, n in sorted(builds.items())), "",
             "A kill row's mode is read from its case's config-0.yaml, its load from the case's result.json "
             "(load row); a restart's mode from its relaunch case's config-0.yaml.", "",
             "| P | draft line | placeholder | value | sources (file:line) |", "|---|---|---|---|---|"]
    for i, line, key, text, srcs in prov:
        cell = lambda s: s.replace("|", "\\|")
        lines.append(f"| P{i} | {line} | {cell(key)} | {cell(text)} | {'<br>'.join(dict.fromkeys(srcs))} |")
    pv.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} and {pv}: {len(prov)} of {len(found)} placeholders filled")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finals manifest: per job, the build, the autotuner seeds and what each run dir produced.

For each job id: its run dirs (mvp/runs/*-<job>[-fill<k>]), each case's build (the dist= line of every
env-<node>.txt; a case on another build than the job's, or with nodes that disagree, is flagged), the tactic seed
(FT_COST_E2E_SEED from the job's launch command in logs/bundle/<job>-*.log, else the merge_pin.txt inputs, else
"tuned in this run"), the sha256 of every autotuner/pinned.json, and the rows each run dir produced: FT off/on pairs
and descriptive arms (std-* cases), kill rows (KILL_FINAL_VERDICT), D rows (COLD_RESTART) and A rows
(ACCURACY_KILL_CASE / _VERDICT). Read-only on run dirs; one section per job, so a refresh replaces only the jobs given.

Usage: python3 -I tools/finals_manifest.py [--out review/FINALS_MANIFEST.md] [--label TEXT] JOB_ID...
"""
import argparse
import hashlib
import pathlib
import re
import sys
from collections import Counter

C = pathlib.Path(__file__).resolve().parents[1]
RUNS, BUNDLE = C / "mvp" / "runs", C / "logs" / "bundle"
STD_CASE = re.compile(r"std-(noeplb|offline|online)(-ft|-main|-lb|-ulfm)?-(\d+)")
BEGIN, END = "<!-- manifest job {} -->", "<!-- end manifest job {} -->"
SUMMARY = re.compile(r"<!-- builds -->.*?<!-- end builds -->\n", re.S)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def short(path):
    """A path relative to the claude root when under it."""
    p = pathlib.Path(path)
    return str(p.relative_to(C)) if p.is_absolute() and p.is_relative_to(C) else str(path)


def run_dirs(job):
    return sorted(p for p in RUNS.glob(f"*-{job}*") if p.is_dir() and re.search(rf"-{job}(?:-fill\d+)?$", p.name))


def prefix_of(run, job):
    return re.sub(rf"-{job}(?:-fill\d+)?$", "", run.name)


def case_builds(case):
    """{node: dist} from env-<node>.txt, plus the non-empty MVP_OVERLAY values."""
    builds, overlays = {}, set()
    for env in sorted(case.glob("env-*.txt")):
        text = env.read_text(errors="replace")
        m = re.search(r"\bdist=(\S+)", text)
        builds[env.stem[4:]] = m[1] if m else "?"
        o = re.search(r"^MVP_OVERLAY=(\S+)", text, re.M)
        if o:
            overlays.add(o[1])
    return builds, overlays


def build_name(dist):
    return pathlib.Path(dist).parent.name if dist.endswith("/dist") else dist


def launch_seeds(job):
    """{run-dir prefix: FT_COST_E2E_SEED} from the `... run_job.sh <prefix>` commands in the job's bundle logs."""
    seeds = {}
    for log in sorted(BUNDLE.glob(f"{job}-*.log")):
        head = log.read_text(errors="replace").split("\nJOB_END", 1)[0]
        for cmd, prefix in re.findall(r"(.*?)run_job\.sh ([\w-]+)", head, re.S):
            m = re.search(r"FT_COST_E2E_SEED=(\S+)", cmd)
            if m:
                seeds[prefix] = m[1]
    return seeds


def seed_lines(run, seed):
    tuner, out = run / "autotuner", []
    pinned = tuner / "pinned.json"
    if seed:
        same = ""
        if pinned.exists() and pathlib.Path(seed).exists():
            same = " (pinned.json == seed)" if sha256(seed) == sha256(pinned) else " (**pinned.json differs from seed**)"
        out.append(f"FT_COST_E2E_SEED `{short(seed)}`{same}")
    elif (run / "merge_pin.txt").exists():
        inputs = re.findall(r"^MERGE_PIN input \d+ (\S+?):", (run / "merge_pin.txt").read_text(errors="replace"), re.M)
        out.append("merged in this run from " + ", ".join(f"`{short(i)}`" for i in inputs) + " (merge_pin.txt)")
    elif tuner.is_dir():
        caches = sorted(p.name for p in tuner.glob("*.json"))
        out.append("no seed: tuned at startup in this run (" + ", ".join(caches) + ")" if caches else "no cache")
    if pinned.exists():
        out.append(f"pinned.json sha256 `{sha256(pinned)}`")
    return out


def kv(line):
    return dict(re.findall(r"(\w+)=(\S+)", line))


def produced(run):
    """Rows this run dir produced: [str]."""
    rows = []
    cases = {c.name for c in run.iterdir() if c.is_dir()}
    std = [STD_CASE.fullmatch(c).groups() for c in sorted(cases) if STD_CASE.fullmatch(c)]
    arms = {(m, r): set() for m, _, r in std}
    for m, v, r in std:
        arms[(m, r)].add(v or "")
    for (mode, rnd), vs in sorted(arms.items()):
        if {"", "-ft"} <= vs:
            rows.append(f"off/on pair {mode} round {rnd} (std-{mode}-{rnd} + std-{mode}-ft-{rnd})")
        elif vs & {"", "-ft"}:
            rows.append(f"**unpaired** {mode} round {rnd}: only {sorted(v or 'off' for v in vs & {'', '-ft'})}")
        for v in sorted(vs - {"", "-ft"}):
            rows.append(f"descriptive arm std-{mode}{v}-{rnd}")
    for name, tag, kind in (("kill_final.txt", "KILL_FINAL_VERDICT ", "kill row"),
                            ("cold_restart.txt", "COLD_RESTART ", "D row"),
                            ("accuracy_kill.txt", "ACCURACY_KILL_CASE ", "A row"),
                            ("accuracy_kill.txt", "ACCURACY_KILL_VERDICT ", "A verdict")):
        path = run / name
        if not path.exists():
            continue
        for line in path.read_text(errors="replace").splitlines():
            if not line.startswith(tag):
                continue
            parts, f = line.split(), kv(line)
            if kind == "kill row":
                bad = parts[1] in invalid_cases(run) or "*" in invalid_cases(run)
                rows.append(f"kill row {parts[1].rsplit('-', 1)[0]} `{parts[1]}` {'INVALID (INVALID_CASES)' if bad else parts[2]}"
                            f" (failed_ranks {f.get('failed_ranks', '-')})")
            elif kind == "D row":
                rows.append(f"D row {parts[1]} run {f.get('run')} (relaunch `{f.get('relaunch')}`, state {f.get('state')})")
            elif parts[1] in cases:  # accuracy_kill.txt also repeats the allocation's other run dir's verdicts
                rows.append(f"{kind} `{parts[1]}`" + (f" ft={f.get('ft')} kill={f.get('kill')} state={f.get('state')}"
                                                     if kind == "A row" else f" {parts[2]}"))
    return rows

def invalid_cases(run):
    """Case names a run dir's INVALID_CASES file excludes (one per line; "*" = every case), e.g. after a node fault."""
    path = run / "INVALID_CASES"
    return {line.strip() for line in path.read_text().splitlines() if line.strip()} if path.exists() else set()



def job_section(job, label):
    runs = run_dirs(job)
    out = [BEGIN.format(job), f"## Job {job}" + (f" ({label})" if label else ""), ""]
    if not runs:
        return out + [f"No run dirs mvp/runs/*-{job}.", "", END.format(job)], [f"{job}: no run dirs"]
    seeds, cases, flags = launch_seeds(job), {}, []
    for run in runs:
        for case in sorted(p for p in run.iterdir() if p.is_dir() and any(p.glob("env-*.txt"))):
            cases[case] = case_builds(case)
    count = Counter(d for c, (b, _) in cases.items() if "-main-" not in c.name for d in set(b.values()))
    job_build = count.most_common(1)[0][0] if count else "?"
    out += [f"Build: `{build_name(job_build)}` (`{short(job_build)}`), from env-<node>.txt dist= of {len(cases)} cases",
            ""]
    for case, (builds, overlays) in cases.items():
        where = f"{case.parent.name}/{case.name}"
        dists = set(builds.values())
        if len(dists) > 1:
            flags.append(f"{where}: nodes on different builds {sorted(map(build_name, dists))}")
        elif dists != {job_build}:
            note = " (main parity arm: main build expected)" if "-main-" in case.name else ""
            flags.append(f"{where}: build `{build_name(next(iter(dists)))}`{note}")
        if overlays:
            flags.append(f"{where}: MVP_OVERLAY {sorted(overlays)}")
    out += ["Flags: " + ("none" if not flags else ""), *[f"- {f}" for f in flags], ""]
    out += ["| run dir | cases | seed / autotuner cache | produced |", "|---|---|---|---|"]
    for run in runs:
        names = [c.name for c in cases if c.parent == run]
        seed = seed_lines(run, seeds.get(prefix_of(run, job)))
        out.append(f"| `{short(run)}` | {', '.join(names) or '-'} | {'<br>'.join(seed) or '-'} | "
                   f"{'<br>'.join(produced(run)) or '-'} |")
    return out + ["", END.format(job)], [f"{job}: {f}" for f in flags]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("jobs", nargs="+")
    p.add_argument("--out", type=pathlib.Path, default=C / "review" / "FINALS_MANIFEST.md")
    p.add_argument("--label", default="", help='e.g. "TEST - rehearsal, not finals"; stamped on each job section')
    args = p.parse_args()
    text = args.out.read_text() if args.out.exists() else "# Finals manifest\n\nWritten by tools/finals_manifest.py.\n"
    all_flags = []
    for job in args.jobs:
        if not re.fullmatch(r"\d+", job):
            sys.exit(f"not a job id: {job}")
        lines, flags = job_section(job, args.label)
        all_flags += flags
        block = re.compile(re.escape(BEGIN.format(job)) + r".*?" + re.escape(END.format(job)), re.S)
        new = "\n".join(lines)
        text = block.sub(lambda _: new, text) if block.search(text) else text.rstrip("\n") + "\n\n" + new + "\n"
    # Every job in the file, not only the ones refreshed: the finals must all run one build.
    builds = dict(re.findall(r"^## Job (\d+).*\n\nBuild: `([^`]+)`", text, re.M))
    mixed = len(set(builds.values())) > 1
    summary = ("<!-- builds -->\nBuilds across jobs: " + ", ".join(f"{j} `{b}`" for j, b in builds.items())
               + (" -- **MIXED BUILDS**" if mixed else "") + "\n<!-- end builds -->\n")
    text = SUMMARY.sub(lambda _: summary, text) if SUMMARY.search(text) else text.replace("\n\n<!--", f"\n\n{summary}\n<!--", 1)
    if mixed:
        all_flags.append(f"jobs on different builds: {builds}")
    args.out.write_text(text)
    print(f"wrote {args.out} ({len(args.jobs)} jobs)")
    for f in all_flags:
        print(f"FLAG {f}")


if __name__ == "__main__":
    main()

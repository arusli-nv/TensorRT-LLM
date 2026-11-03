#!/usr/bin/env python3
"""Mid-window rank-0 stall bursts over every served arm of HISTORY_STALL_SWEEP.csv (read-only).

Burst: inside a measured window (bench.log label healthy-N), 10-s buckets of sum(host_step_time) over steps
40 ms < t <= 5 s (first and last bucket dropped); a burst is >= 2 consecutive buckets above the window's
median bucket + 2.5 s. Rank-0 log timestamps are local time (PT), as the bench start_wall_s.
Usage: python3 -I review/HISTORY_MIDBURST.py review/HISTORY_STALL_SWEEP.csv review/HISTORY_MIDBURST.csv
       python3 -I review/HISTORY_MIDBURST.py --meta SWEEP.csv EXTRA_META.csv  (arms under mvp/runs missing from
       SWEEP.csv, outside _failed/_superseded: metadata via HISTORY_STALL_SWEEP's scan + enrich, no wheel map)
       python3 -I review/HISTORY_MIDBURST.py --report review/HISTORY_MIDBURST.csv review/HISTORY_MIDBURST.extra.csv
"""
import csv
import json
import multiprocessing
import pathlib
import re
import statistics
import sys
import time

C = pathlib.Path(__file__).resolve().parents[1]
TS = re.compile(rb"^\[(\d\d)/(\d\d)/(\d{4})-(\d\d):(\d\d):(\d\d)\]")
STEP = re.compile(rb"host_step_time = ([\d.]+)ms")
MCA = re.compile(r"MPIRUN_START \S+ mca=\[(.*?)\]")
DROP = re.compile(r"start=([\d.]+)")
THRESH_MS, BIG_MS, EXCESS_MS = 40.0, 5000.0, 2500.0


def kv(text, key):
    m = re.search(rf"^{key}=(.*)$", text, re.M)
    return m.group(1) if m else None


def read(p):
    try:
        return p.read_text(errors="replace")
    except OSError:
        return ""


def transport(mca):
    if mca is None:
        return "unknown"
    if "pml ob1" not in mca:
        return "default(ucx)" if "pml" not in mca else "other:" + mca
    btl = re.search(r"btl (\S+)", mca)
    btl = btl.group(1) if btl else ""
    return "ob1-sm-tcp" if ("sm" in btl.split(",") or "vader" in btl) else "ob1-tcp"


def scan(r):
    case = C / r["path"]
    out = {"path": r["path"]}
    tl = read(case / "timeline")
    m = MCA.search(tl)
    out["mca"] = m.group(1) if m else ""
    out["transport_eff"] = transport(m.group(1) if m else None)
    out["ulfm_detector"] = "mpi_ft_detector true" in out["mca"]
    env = read(case / "env-0.txt")
    out["samplers"] = "on" if (case / "hostpressure-0.log").exists() or (case / "gpumem-0.csv").exists() else "off"
    out["host_cache_size"] = kv(env, "WIDEEP_HOST_CACHE_SIZE") or ""
    out["hang_confirm_s"] = kv(env, "TLLM_FAULT_TOLERANCE_HANG_CONFIRM_S") or ""
    drops = [float(DROP.search(read(p)).group(1)) for p in sorted(case.glob("dropcache-[0-9]*.log"))
             if DROP.search(read(p))]
    drop = min(drops) if drops else None
    out["drop_start"] = drop or ""
    wins = []
    for line in read(case / "bench.log").splitlines():
        try:
            b = json.loads(line)
        except ValueError:
            continue
        if str(b.get("label", "")).startswith("healthy-") and b.get("start_wall_s"):
            wins.append((b["label"], float(b["start_wall_s"]), float(b["elapsed_s"])))
    out["windows"] = ";".join(w[0] for w in wins)
    if not wins or r["kill_case"] == "True":
        out["status"] = "no-window" if not wins else "kill"
        return out
    lo, hi = min(w[1] for w in wins), max(w[1] + w[2] for w in wins)
    buckets = {}
    cache = {}
    with open(case / "rank-0.log", "rb") as f:
        for line in f:
            if b"host_step_time" not in line:
                continue
            m = TS.match(line)
            s = STEP.search(line)
            if not (m and s):
                continue
            key = m.group(0)
            t = cache.get(key)
            if t is None:
                mo, d, y, hh, mm, ss = (int(x) for x in m.groups())
                t = time.mktime((y, mo, d, hh, mm, ss, 0, 0, -1))
                cache[key] = t
            if t < lo or t > hi:
                continue
            ms = float(s.group(1))
            for i, (label, st, el) in enumerate(wins):
                if st <= t < st + el:
                    k = (i, int((t - st) // 10))
                    b = buckets.setdefault(k, [0.0, 0])
                    b[1] += 1
                    if THRESH_MS < ms <= BIG_MS:
                        b[0] += ms
                    break
    bursts, nb_total = [], {}
    for i, (label, st, el) in enumerate(wins):
        nb = int(el // 10)
        sums = [buckets.get((i, j), [0.0, 0]) for j in range(1, nb - 1)]
        load = re.match(r"\d+", label.split("-", 1)[1]).group(0)
        nb_total[load] = nb_total.get(load, 0) + len(sums)
        if len(sums) < 4 or sum(x[1] for x in sums) == 0:
            continue
        med = statistics.median(x[0] for x in sums)
        out[f"med_{load}"] = round(med / 1000, 2)
        j = 0
        while j < len(sums):
            if sums[j][0] > med + EXCESS_MS:
                k = j
                while k < len(sums) and sums[k][0] > med + EXCESS_MS:
                    k += 1
                if k - j >= 2:
                    t0 = st + 10 * (j + 1)
                    bursts.append({"label": label, "load": load, "win_off_s": round(10 * (j + 1)), "len_s": 10 * (k - j),
                                   "drop_off_s": round(t0 - drop) if drop else "",
                                   "excess_s": round(sum(x[0] - med for x in sums[j:k]) / 1000, 1),
                                   "min_steps": min(x[1] for x in sums[j:k])})
                j = k
            else:
                j += 1
    out["status"] = "ok" if buckets else "no-steps-in-window"
    out["n_buckets"] = json.dumps(nb_total)
    for load in ("8", "16", "32", "64", "128", "256", "512"):
        out[f"bursts_{load}"] = sum(b["load"] == load for b in bursts) if load in nb_total else ""
    out["bursts"] = json.dumps(bursts)
    return out


def safe(r):
    try:
        return scan(r)
    except Exception as e:  # noqa: BLE001 - one malformed historical arm must not stop the sweep
        return {"path": r["path"], "status": "error:" + repr(e)[:120]}


def meta(sweep_csv, dst):
    import importlib.util
    import os
    import tempfile
    spec = importlib.util.spec_from_file_location("sweep", C / "review" / "HISTORY_STALL_SWEEP.py")
    sweep = importlib.util.module_from_spec(spec)
    sys.modules["sweep"] = sweep
    spec.loader.exec_module(sweep)
    have = {r["path"] for r in csv.DictReader(open(sweep_csv))}
    cases = sorted({p.parent for p in (C / "mvp" / "runs").rglob("bench.log")
                    if (p.parent / "rank-0.log").exists() and str(p.parent.relative_to(C)) not in have
                    and not {"_failed", "_superseded"} & set(p.parts)})
    with multiprocessing.Pool(16) as pool:
        rows = pool.map(sweep.safe_scan, cases, chunksize=2)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    fd, raw = tempfile.mkstemp(suffix=".csv", dir=os.environ.get("TMPDIR"))
    with os.fdopen(fd, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if "scan_error" in r:
            print("scan_error", r["path"], r["scan_error"])
    sweep.enrich(raw, dst)


def fisher(a, b, c, d):
    """Two-sided Fisher exact p for [[a, b], [c, d]]."""
    import math
    n, r1, c1 = a + b + c + d, a + b, a + c
    p = [math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1) for x in range(min(r1, c1) + 1)]
    lo = max(0, c1 - (n - r1))
    return sum(q for x, q in enumerate(p) if x >= lo and q <= p[a] * (1 + 1e-9))


def report(*csvs):
    """Arm-level burst rates. late = burst starting >= 120 s after the page-cache drop; mid = >= 60 s into the
    window. Profiled arms (nsys/ncu reports in the arm dir) are excluded."""
    import collections
    rs = [r for f in csvs for r in csv.DictReader(open(f)) if r.get("status") == "ok"
          and not any((C / r["path"]).glob("n[sc][yu]*-rank*"))]
    grp = lambda r: (r["arm_type"], r["transport_eff"])  # noqa: E731
    def bl(r, lo, hi, kind):
        return [b for b in json.loads(r["bursts"]) if lo <= int(b["load"]) <= hi and
                (kind == "all" or (kind == "mid" and b["win_off_s"] >= 60) or
                 (kind == "late" and b["drop_off_s"] != "" and b["drop_off_s"] >= 120))]
    def has(r, lo, hi):
        return any(lo <= int(k) <= hi for k in json.loads(r["n_buckets"]))
    def tab(key, sub, lo, hi, kind):
        c = collections.defaultdict(lambda: [0, 0])
        for r in rs:
            if sub(r) and has(r, lo, hi):
                c[key(r)][0] += 1
                c[key(r)][1] += bool(bl(r, lo, hi, kind))
        return c
    eplb = lambda r: r["mode_cmp"] in ("online", "offline")  # noqa: E731
    for name, sub in (("all modes", lambda r: True), ("EPLB online+offline", eplb),
                      ("all modes, from 10-19", lambda r: r["date"] >= "2026-10-19")):
        for lo, hi in ((8, 64), (128, 512)):
            for kind in ("all", "mid", "late"):
                t = tab(grp, sub, lo, hi, kind)
                F, O, U = t[("ft-on", "ob1-tcp")], t[("off-default", "default(ucx)")], t[("off-ob1(ulfm)", "ob1-tcp")]
                print(f"{name:24} load {lo}-{hi} {kind:4} FT-ob1 {F[1]}/{F[0]}  OFF-ucx {O[1]}/{O[0]}  "
                      f"OFF-ob1(ulfm) {U[1]}/{U[0]}  p(FT,OFF)={fisher(F[1], F[0]-F[1], O[1], O[0]-O[1]):.3f}  "
                      f"p(FT,ULFM)={fisher(F[1], F[0]-F[1], U[1], U[0]-U[1]):.3f}  "
                      f"p(ULFM,OFF)={fisher(U[1], U[0]-U[1], O[1], O[0]-O[1]):.3f}")
    print("low-load late or mid bursts:")
    for r in rs:
        for b in bl(r, 8, 64, "late") or bl(r, 8, 64, "mid"):
            print(" ", r["date"], r["run"], r["arm"], grp(r), r["mode_cmp"], r["build"], r["hang_detection"][:8],
                  r["extra_kind"], r["shm"], "samplers=" + r["samplers"], r["rank0_host"], b)
    ft = lambda r: grp(r) == ("ft-on", "ob1-tcp") and eplb(r)  # noqa: E731
    for name, key in (("hang_detection", lambda r: r["hang_detection"][:12]), ("extra_kind", lambda r: r["extra_kind"]),
                      ("shm", lambda r: r["shm"]), ("mode", lambda r: r["mode_cmp"]), ("samplers", lambda r: r["samplers"]),
                      ("ulfm_detector", lambda r: "mpi_ft_detector true" in r["mca"]), ("day", lambda r: r["date"][:10]),
                      ("build", lambda r: r["build"])):
        t, h = tab(key, ft, 8, 64, "late"), tab(key, ft, 8, 64, "mid")
        print(f"FT-ob1 EPLB arms, load 8-64, by {name}: " +
              "; ".join(f"{k}: late {v[1]}/{v[0]} mid {h[k][1]}" for k, v in sorted(t.items(), key=str)))


def main():
    if sys.argv[1] == "--report":
        report(*sys.argv[2:])
        return
    if sys.argv[1] == "--meta":
        meta(sys.argv[2], sys.argv[3])
        return
    src, dst = sys.argv[1:3]
    rows = list(csv.DictReader(open(src)))
    with multiprocessing.Pool(32) as pool:
        res = pool.map(safe, rows, chunksize=2)
    keep = ["run", "arm", "date", "build", "build_trust", "model", "world", "ft", "mode_cmp", "hang_detection",
            "peer_failure_poll_s", "extra_kind", "shm", "node0_shared_gib_max", "overlay", "rank0_host", "kill_case"]
    out = []
    for r, x in zip(rows, res):
        o = {k: r[k] for k in keep}
        o.update(x)
        t = x.get("transport_eff", "")
        o["arm_type"] = ("ft-on" if r["ft"] == "on" else "off-ob1(ulfm)" if t.startswith("ob1") else
                         "off-main" if "main" in r["build"] else "off-default")
        out.append(o)
    keys = []
    for o in out:
        keys += [k for k in o if k not in keys]
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(out)
    print(len(out), "arms")


if __name__ == "__main__":
    main()

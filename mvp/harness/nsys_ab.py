#!/usr/bin/env python3
"""A/B comparison of per-rank nsys decode windows (steady steps, all ranks).

Usage:
  python3 nsys_ab.py CASE_A CASE_B --window K [--layers 58] [--steps N] [--top 25] [--jobs 8]
                     [--label-a A --label-b B] [--nsys PATH] [--cache DIR] [--json OUT.json]
  CASE_A/CASE_B: case dirs holding nsys-rank<r>.<K>.nsys-rep (one per rank; K = load window index,
  e.g. C-COMBINE2: .1 = 128/GPU, .2 = 256/GPU).  Delta = B minus A (positive = B slower).
  Reports are exported once to sqlite under --cache (default $TMPDIR/nsys_ab_cache) and reused.

Method:
  * Steps are cut at every L-th moeA2ADispatch kernel (L = MoE layers, --layers, default 58; checked against
    the NVTX "_forward_step" count).  The first window step and the last (open-ended) one are dropped; steps
    whose period exceeds 1.3x that rank's median (nsys flush artifacts, bursts) are dropped on all ranks;
    --steps N keeps the first N of the remainder.
  * Wall decomposition per layer (sums exactly to the GPU step period / L): dispatch kernel + expert section
    (dispatch end -> combine start) + combine kernel + rest (combine end -> next dispatch; includes attention,
    dense, norms, route, and the step-boundary host gap on the last layer).
  * Combine: transfer floor = min over ranks of the combine duration for the same (step, layer); wait = mean
    over ranks minus floor.  Same for dispatch.  Expert-section spread = max over ranks minus mean.
  * Kernel classes by shortName regex (CLASSES below); per-kernel-name table for the unclassified detail.
    Class sums add durations (overlapping streams double count); "gpu idle" = step span minus union of
    kernel+memcpy intervals.
  * Host: step period from consecutive NVTX "_forward_step" starts (host wall per step); NVTX ranges
    (digits normalised) summed over kept steps, per step, averaged over ranks (plus rank 0 alone).
  * Cross-rank spread: per-rank mean step period (min/max/std over ranks).
"""
import argparse
import collections
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import re
import sqlite3
import statistics as st
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CLAIM_ROOT = os.path.dirname(os.path.dirname(HERE))  # .../claude
CLASSES = [  # first match wins
    ("dispatch", r"^moeA2ADispatchKernel"),
    ("combine", r"^moeA2ACombineKernel"),
    ("prepare_dispatch", r"^moeA2APrepareDispatch"),
    ("prepare_combine", r"^moeA2APrepareCombine"),
    ("sanitize", r"Sanitize"),
    ("memset", r"memset|Memset"),
    ("route", r"moeComputeRoute|routingIndices|topk|Topk|TopK|routing"),
    ("expert_gemm", r"cutlass_kernel_tensorrt|grouped|Grouped|moe.*gemm|silu_and_mul"),
    ("gather_compact", r"gather|Gather|compact|Compact|Scatter|scatter|CatArray|index_"),
    ("attention", r"fmha|MLARope|applyMLA|attention|Attention|splitKreduce"),
    ("dense_gemm", r"nvjet|cutlass_kernel_flashinf|gemm|Gemm"),
    ("quant_norm", r"quantize|rmsNorm|RmsNorm|layernorm"),
    ("allreduce_nccl", r"nccl|AllReduce|allreduce"),
]
CLASS_RE = [(n, re.compile(p)) for n, p in CLASSES]
CLASS_ORDER = [n for n, _ in CLASSES] + ["memcpy", "other"]


def classify(name):
    for n, r in CLASS_RE:
        if r.search(name):
            return n
    return "other"


def find_nsys(explicit):
    if explicit:
        return explicit
    for p in glob.glob(os.path.join(CLAIM_ROOT, "tools", "**", "nsys"), recursive=True):
        if os.access(p, os.X_OK) and os.path.isfile(p):
            return p
    return "nsys"


def norm_nvtx(name):
    if name.startswith("[Executor] _forward_step"):
        return "[Executor] _forward_step"
    return re.sub(r"\d+", "#", name)


def export(rep, cache, nsys):
    h = hashlib.md5(os.path.realpath(rep).encode()).hexdigest()[:10]
    out = os.path.join(cache, f"{os.path.basename(rep)[:-9]}.{h}.sqlite")
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(rep):
        subprocess.run([nsys, "export", "--type=sqlite", "--force-overwrite=true", "-o", out, rep],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def union_len(iv, lo, hi):
    tot, cur_e = 0, lo
    for s, e in sorted(iv):
        s, e = max(s, lo), min(e, hi)
        if e <= s:
            continue
        if s > cur_e:
            cur_e = s
        if e > cur_e:
            tot += e - cur_e
            cur_e = e
    return tot


def extract_rank(args):
    rep, cache, nsys, L, launch = args
    db = export(rep, cache, nsys)
    c = sqlite3.connect(db)
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        return {"error": "no CUDA kernels in the report"}
    # --launch appends the grid and block shape, so one kernel under two tactics shows as two rows.
    name = ("s.value||' g'||k.gridX||'x'||k.gridY||'x'||k.gridZ||' b'||k.blockX" if launch else "s.value")
    ks = c.execute(f"SELECT k.start,k.end,{name} FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON "
                   "k.shortName=s.id ORDER BY k.start").fetchall()
    mc = c.execute("SELECT start,end,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start").fetchall()
    nv = c.execute("SELECT n.start,n.end,s.value FROM NVTX_EVENTS n JOIN StringIds s ON n.textId=s.id "
                   "WHERE n.end>0 ORDER BY n.start").fetchall()
    c.close()
    disp = [k for k in ks if k[2].startswith("moeA2ADispatchKernel")]
    comb = [k for k in ks if k[2].startswith("moeA2ACombineKernel")]
    fwd = [n for n in nv if n[2].startswith("[Executor] _forward_step")]
    r = {"n_disp": len(disp), "n_comb": len(comb), "n_fwd": len(fwd)}
    if len(disp) != len(comb) or len(disp) < 2 * L:
        r["error"] = f"dispatch={len(disp)} combine={len(comb)} (L={L})"
        return r
    nst = len(disp) // L
    ds = [d[0] for d in disp]
    cls_of = {}
    steps = []
    ki = 0
    for k in range(nst - 1):
        lo, hi = ds[k * L], ds[(k + 1) * L]
        cls, names, iv = collections.defaultdict(float), collections.defaultdict(float), []
        while ki < len(ks) and ks[ki][0] < lo:
            ki += 1
        j = ki
        while j < len(ks) and ks[j][0] < hi:
            s, e, nm = ks[j]
            cl = cls_of.get(nm) or cls_of.setdefault(nm, classify(nm))
            cls[cl] += e - s
            names[re.sub(r"_\d+\.\.\.$", "_#...", nm)] += e - s
            iv.append((s, e))
            j += 1
        for s, e, _b in mc:
            if lo <= s < hi:
                cls["memcpy"] += e - s
                names["[memcpy]"] += e - s
                iv.append((s, e))
        lay = []
        for l in range(L):
            i = k * L + l
            nd = ds[i + 1]
            lay.append((disp[i][1] - disp[i][0], comb[i][0] - disp[i][1], comb[i][1] - comb[i][0],
                        nd - comb[i][1]))
        steps.append({"period": hi - lo, "idle": (hi - lo) - union_len(iv, lo, hi), "cls": dict(cls),
                      "names": dict(names), "layers": lay})
    fs = [n[0] for n in fwd]
    r["steps"] = steps
    r["fwd_starts"] = fs
    # host NVTX ranges per kernel-step index (kernel step k <-> k-th forward_step start; host windows)
    hs = []
    if len(fs) > 1:
        import bisect
        nv_s = [n[0] for n in nv]
        for k in range(min(len(steps), len(fs) - 1)):
            a, b = bisect.bisect_left(nv_s, fs[k]), bisect.bisect_left(nv_s, fs[k + 1])
            d = collections.defaultdict(float)
            cnt = collections.defaultdict(int)
            for s, e, nm in nv[a:b]:
                key = norm_nvtx(nm)
                d[key] += e - s
                cnt[key] += 1
            hs.append({"period": fs[k + 1] - fs[k], "ranges": dict(d), "counts": dict(cnt)})
    r["host"] = hs
    return r


def load_case(case, window, L, cache, nsys, jobs, launch=False):
    reps = {}
    for p in glob.glob(os.path.join(case, f"nsys-rank*.{window}.nsys-rep")):
        m = re.search(r"nsys-rank(\d+)\.\d+\.nsys-rep$", p)
        reps[int(m.group(1))] = p
    if not reps:
        sys.exit(f"no nsys-rank*.{window}.nsys-rep in {case}")
    os.makedirs(cache, exist_ok=True)
    ranks = sorted(reps)
    with cf.ProcessPoolExecutor(jobs) as ex:
        res = list(ex.map(extract_rank, [(reps[r], cache, nsys, L, launch) for r in ranks]))
    out = dict(zip(ranks, res))
    bad = {r: v["error"] for r, v in out.items() if "error" in v}
    if bad:
        print(f"WARNING {case}: ranks with unusable windows dropped: {bad}", file=sys.stderr)
    return {r: v for r, v in out.items() if "error" not in v}


def summarize(data, L, nsteps):
    ranks = sorted(data)
    nmin = min(len(data[r]["steps"]) for r in ranks)
    keep = []
    for k in range(1, nmin):  # drop first window step
        ok = True
        for r in ranks:
            per = [s["period"] for s in data[r]["steps"][:nmin]]
            if per[k] > 1.3 * st.median(per):
                ok = False
        if ok:
            keep.append(k)
    dropped = nmin - 1 - len(keep)
    if nsteps:
        keep = keep[:nsteps]
    n = len(keep)
    S = {"ranks": len(ranks), "steps": n, "dropped": dropped, "L": L}
    us = 1e-3
    per_rank_period = [st.mean(data[r]["steps"][k]["period"] for k in keep) * us for r in ranks]
    S["period_us"] = st.mean(per_rank_period)
    S["rank_period_us"] = {"min": min(per_rank_period), "max": max(per_rank_period),
                           "std": st.pstdev(per_rank_period)}
    S["idle_us"] = st.mean(data[r]["steps"][k]["idle"] for r in ranks for k in keep) * us
    hp = [st.mean(data[r]["host"][k]["period"] for k in keep if k < len(data[r]["host"])) * us
          for r in ranks if data[r]["host"]]
    S["host_period_us"] = st.mean(hp) if hp else float("nan")
    S["cls_us"] = {c: st.mean(data[r]["steps"][k]["cls"].get(c, 0) for r in ranks for k in keep) * us
                   for c in CLASS_ORDER}
    names = collections.defaultdict(float)
    for r in ranks:
        for k in keep:
            for nm, d in data[r]["steps"][k]["names"].items():
                names[nm] += d
    S["names_us"] = {nm: d / (len(ranks) * n) * us for nm, d in names.items()}
    # layer segments (disp, expert, comb, rest) mean over ranks/steps/layers
    seg = [[st.mean(data[r]["steps"][k]["layers"][l][i] for r in ranks for k in keep for l in range(L)) * us
            for i in range(4)]][0]
    S["seg_us_layer"] = dict(zip(["dispatch", "expert_section", "combine", "rest"], seg))
    # cross-rank per (step, layer)
    w = {"disp_wait": [], "comb_wait": [], "comb_floor": [], "expert_spread": [], "expert_mean": []}
    for k in keep:
        for l in range(L):
            d = [data[r]["steps"][k]["layers"][l][0] for r in ranks]
            e = [data[r]["steps"][k]["layers"][l][1] for r in ranks]
            c = [data[r]["steps"][k]["layers"][l][2] for r in ranks]
            w["disp_wait"].append(st.mean(d) - min(d))
            w["comb_wait"].append(st.mean(c) - min(c))
            w["comb_floor"].append(min(c))
            w["expert_spread"].append(max(e) - st.mean(e))
            w["expert_mean"].append(st.mean(e))
    S["cross_us_layer"] = {a: st.mean(v) * us for a, v in w.items()}
    # host ranges
    def host(rs):
        tot, cnt = collections.defaultdict(float), collections.defaultdict(float)
        for r in rs:
            for k in keep:
                if k < len(data[r]["host"]):
                    for nm, d in data[r]["host"][k]["ranges"].items():
                        tot[nm] += d
                        cnt[nm] += data[r]["host"][k]["counts"][nm]
        return ({nm: d / (len(rs) * n) * us for nm, d in tot.items()},
                {nm: c / (len(rs) * n) for nm, c in cnt.items()})
    S["host_us"], S["host_cnt"] = host(ranks)
    S["host0_us"], _ = host([ranks[0]])
    return S


def table(title, rows, a_l, b_l, unit, top=None, fmt="{:10.1f}", thresh=0.0):
    rows = sorted(rows, key=lambda r: -abs(r[2] - r[1]))
    print(f"\n## {title}")
    print(f"{'item':56s} {a_l:>10s} {b_l:>10s} {'B-A':>10s}  [{unit}]")
    for nm, a, b in rows[:top]:
        if abs(b - a) < thresh and top:
            continue
        print(f"{nm[:56]:56s} {fmt.format(a)} {fmt.format(b)} {fmt.format(b - a)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case_a")
    ap.add_argument("case_b")
    ap.add_argument("--window", default="1")
    ap.add_argument("--layers", type=int, default=58)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--nsys", default=None)
    ap.add_argument("--cache", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "nsys_ab_cache"))
    ap.add_argument("--json", default=None)
    ap.add_argument("--launch", action="store_true", help="split kernel rows by grid and block shape")
    a = ap.parse_args()
    la = a.label_a or os.path.basename(a.case_a.rstrip("/"))
    lb = a.label_b or os.path.basename(a.case_b.rstrip("/"))
    nsys = find_nsys(a.nsys)
    L = a.layers
    S = {}
    for lab, case in ((la, a.case_a), (lb, a.case_b)):
        d = load_case(case, a.window, L, a.cache, nsys, a.jobs, a.launch)
        nf = {r: (v["n_disp"] / max(v["n_fwd"], 1)) for r, v in d.items()}
        if any(abs(x - L) > 0.5 for x in nf.values()):
            print(f"NOTE {lab}: dispatches per _forward_step NVTX range differ from --layers={L}: "
                  f"{sorted(set(round(x, 2) for x in nf.values()))}", file=sys.stderr)
        S[lab] = summarize(d, L, a.steps)
    A, B = S[la], S[lb]
    print(f"# nsys A/B  window .{a.window}  layers={L}   A={la}   B={lb}   (delta = B - A)")
    for lab in (la, lb):
        s = S[lab]
        print(f"{lab}: ranks={s['ranks']} kept steps={s['steps']} (dropped {s['dropped']} slow)  "
              f"GPU step {s['period_us']/1e3:.3f} ms  host step {s['host_period_us']/1e3:.3f} ms  "
              f"gpu idle {s['idle_us']:.0f} us/step ({100*s['idle_us']/s['period_us']:.1f}%)")
    dp = B["period_us"] - A["period_us"]
    print(f"GPU step delta {dp:+.1f} us ({100*dp/A['period_us']:+.2f}%), "
          f"{dp/L:+.2f} us/layer; host step delta {B['host_period_us']-A['host_period_us']:+.1f} us")
    print(f"cross-rank mean-step spread (us): A min/max/std {A['rank_period_us']['min']:.0f}/"
          f"{A['rank_period_us']['max']:.0f}/{A['rank_period_us']['std']:.1f}  "
          f"B {B['rank_period_us']['min']:.0f}/{B['rank_period_us']['max']:.0f}/{B['rank_period_us']['std']:.1f}")
    table("Wall decomposition (us/layer; sums to GPU step/L)",
          [(k, A["seg_us_layer"][k], B["seg_us_layer"][k]) for k in A["seg_us_layer"]], la, lb, "us/layer",
          fmt="{:10.2f}")
    table("Cross-rank combine/dispatch wait vs floor (us/layer)",
          [(k, A["cross_us_layer"][k], B["cross_us_layer"][k]) for k in A["cross_us_layer"]], la, lb, "us/layer",
          fmt="{:10.2f}")
    table("Kernel classes (us/step, mean per rank; summed durations)",
          [(k, A["cls_us"][k], B["cls_us"][k]) for k in CLASS_ORDER] + [("gpu_idle", A["idle_us"], B["idle_us"])],
          la, lb, "us/step")
    names = set(A["names_us"]) | set(B["names_us"])
    table(f"Kernel names, top {a.top} by |delta| (us/step)",
          [(n, A["names_us"].get(n, 0), B["names_us"].get(n, 0)) for n in names], la, lb, "us/step", top=a.top)
    hn = set(A["host_us"]) | set(B["host_us"])
    table(f"Host NVTX ranges, mean over ranks (us/step), top {a.top}",
          [(n, A["host_us"].get(n, 0), B["host_us"].get(n, 0)) for n in hn], la, lb, "us/step", top=a.top)
    hn = set(A["host0_us"]) | set(B["host0_us"])
    table(f"Host NVTX ranges, rank 0 only (us/step), top {a.top}",
          [(n, A["host0_us"].get(n, 0), B["host0_us"].get(n, 0)) for n in hn], la, lb, "us/step", top=a.top)
    if a.json:
        json.dump({la: A, lb: B}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

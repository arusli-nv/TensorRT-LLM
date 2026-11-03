#!/usr/bin/env python3
"""Per-decode-step idle/busy analysis of nsys SQLite exports (FT on vs off).

usage: nsys32_steps.py LABEL=off.sqlite LABEL=ft.sqlite ...   (prints markdown tables)

Segmentation: every decode step is one cudaGraphLaunch; graph kernels carry
graphId and correlationId 0 (--cuda-graph-trace node), so they are bound to a
step by order: on each graph stream the k-th block of n_s kernels is step k
(n_s = kernels on that stream / number of launches; checked divisible).
Step start = earliest graph kernel start of the step.
period  = start(N+1) - start(N)
busy    = union of all kernel + memcpy + memset intervals (all streams, graph and
          eager) clipped to [start(N), start(N+1))
idle    = period - busy;  split into idle inside the graph window and idle
          between graph end(N) and graph start(N+1)
lateness= cudaGraphLaunch(N+1) API start - graph_end(N) (graph_end = last graph
          kernel end of step N). >0: GPU already idle when the host issued the launch.
"""
import sqlite3, sys
import numpy as np

def union_len(iv, lo, hi):
    tot, cur_s, cur_e = 0, None, None
    for s, e in iv:
        s, e = max(s, lo), min(e, hi)
        if e <= s: continue
        if cur_e is None or s > cur_e:
            if cur_e is not None: tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None: tot += cur_e - cur_s
    return tot

def load(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    q = db.execute
    gl = q("""select r.start,r.end,r.globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId
              where s.value like 'cudaGraphLaunch%' order by r.start""").fetchall()
    gk = q("select streamId,start,end from CUPTI_ACTIVITY_KIND_KERNEL where graphId is not null order by streamId,start").fetchall()
    allk = q("select start,end from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    allk += q("select start,end from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
    allk += q("select start,end from CUPTI_ACTIVITY_KIND_MEMSET").fetchall()
    allk.sort()
    api = q("""select r.start,r.end,s.value from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId
               order by r.start""").fetchall()
    return gl, gk, allk, api

def analyze(path):
    gl, gk, allk, api = load(path)
    n = len(gl)
    streams = {}
    for sid, s, e in gk: streams.setdefault(sid, []).append((s, e))
    starts = np.full(n, np.inf); ends = np.zeros(n); gsum = np.zeros(n); nk = np.zeros(n)
    for sid, lst in streams.items():
        assert len(lst) % n == 0, (sid, len(lst), n)
        per = len(lst) // n
        for k in range(n):
            blk = lst[k*per:(k+1)*per]
            starts[k] = min(starts[k], blk[0][0]); ends[k] = max(ends[k], max(b[1] for b in blk))
            gsum[k] += sum(b[1]-b[0] for b in blk); nk[k] += per
    # steps 0..n-2 have a successor
    m = n - 1
    rows = []
    api_s = np.array([a[0] for a in api])
    for k in range(m):
        lo, hi = starts[k], starts[k+1]
        busy = union_len(allk, lo, hi)
        inwin = union_len(allk, lo, ends[k])
        a0, a1 = gl[k][0], gl[k+1][0]
        i0, i1 = np.searchsorted(api_s, a0), np.searchsorted(api_s, a1)
        calls = {}
        for s, e, nm in api[i0:i1]:
            c = calls.setdefault(nm.split('_v')[0], [0, 0]); c[0] += 1; c[1] += e - s
        rows.append(dict(period=hi-lo, busy=busy, idle=hi-lo-busy, win=ends[k]-lo, idle_in=(ends[k]-lo)-inwin,
                         idle_between=(hi-ends[k])-(busy-inwin), gap=hi-ends[k],
                         late=gl[k+1][0]-ends[k], launch_lead=starts[k+1]-gl[k+1][0], gl_dur=gl[k+1][1]-gl[k+1][0],
                         gsum=gsum[k], nk=nk[k], calls=calls))
    return rows, n

def st(x):
    x = np.asarray(x) / 1e6
    return f"{np.median(x):.3f} | {x.mean():.3f} | {np.percentile(x,10):.3f} | {np.percentile(x,90):.3f}"

def report(label, rows, n):
    per = np.array([r['period'] for r in rows]); med = np.median(per)
    keep = [r for r in rows if r['period'] <= 2*med]
    out = [f"\n#### {label}: {len(rows)} steps ({n} graph launches), {len(rows)-len(keep)} stall steps (>2x median {med/1e6:.2f} ms) excluded in the second block\n",
           "| metric (ms) | median | mean | p10 | p90 |", "|---|---|---|---|---|"]
    for title, rs in (("all steps", rows), ("excluding stalls", keep)):
        out.append(f"| **{title}** (n={len(rs)}) | | | | |")
        for key, nm in (('period','period'),('busy','GPU busy (union)'),('idle','GPU idle'),('win','graph window (first-last graph kernel)'),
                        ('idle_in','idle inside graph window'),('gap','graph end -> next graph start'),
                        ('late','launch lateness (cudaGraphLaunch start - prev graph end)'),
                        ('launch_lead','next first kernel start - its cudaGraphLaunch start'),
                        ('gl_dur','cudaGraphLaunch API duration'),('gsum','sum of graph kernel durations (PDL-inflated, info only)')):
            out.append(f"| {nm} | {st([r[key] for r in rs])} |")
    names = sorted({k for r in keep for k in r['calls']})
    out += ["", f"Host API calls per step ({label}, non-stall steps): mean count / mean total ms", "", "| API | calls/step | ms/step |", "|---|---|---|"]
    for nm in names:
        c = np.mean([r['calls'].get(nm, [0,0])[0] for r in keep]); t = np.mean([r['calls'].get(nm, [0,0])[1] for r in keep])/1e6
        out.append(f"| {nm} | {c:.2f} | {t:.3f} |")
    out.append(f"\ngraph kernels per step: {rows[0]['nk']:.0f}")
    return "\n".join(out), keep

def attribution(path, topn=14):
    """Sweep-line attribution: walk graph kernels by end time; each kernel is charged
    end - max(start, running max end), so overlapped (PDL-spinning) time is charged once
    and the per-name sums add up to the graph window busy time. Returns {name: ms/step}."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    n = db.execute("select count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where s.value like 'cudaGraphLaunch%'").fetchone()[0]
    ks = db.execute("""select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName
                       where k.graphId is not null order by k.end""").fetchall()
    acc, hi = {}, 0
    for st_, en, nm in ks:
        c = en - max(st_, hi) if en > hi else 0
        if c > 0: acc[nm] = acc.get(nm, 0) + c
        hi = max(hi, en)
    return {k: v / n / 1e6 for k, v in acc.items()}

def attr_table(a_off, a_ft, topn=14):
    names = sorted(set(a_off) | set(a_ft), key=lambda k: -abs(a_ft.get(k, 0) - a_off.get(k, 0)))[:topn]
    out = ["| kernel (sweep-attributed ms/step) | off | ft | ft-off |", "|---|---|---|---|"]
    for k in names:
        out.append(f"| {k[:70]} | {a_off.get(k,0):.3f} | {a_ft.get(k,0):.3f} | {a_ft.get(k,0)-a_off.get(k,0):+.3f} |")
    out.append(f"| TOTAL graph attributed | {sum(a_off.values()):.3f} | {sum(a_ft.values()):.3f} | {sum(a_ft.values())-sum(a_off.values()):+.3f} |")
    return "\n".join(out)

if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--attr":
        print(attr_table(attribution(sys.argv[2]), attribution(sys.argv[3]))); sys.exit()
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        rows, n = analyze(path)
        txt, _ = report(label, rows, n)
        print(txt)

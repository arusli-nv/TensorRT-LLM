"""Per decode step and per MoE layer timings (medians) from the nsys traces of the kernel benchmark."""
import json, sqlite3, statistics as st, sys
from multiprocessing import Pool

A2A = ("moeA2ADispatchKernel", "moeA2ACombineKernel", "moeA2ASanitizeExpertIdsKernel", "moeA2AGatherLocalKernel")

def analyze(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = db.execute("select s.value, k.start, k.end, k.streamId from CUPTI_ACTIVITY_KIND_KERNEL k "
                      "join StringIds s on s.id=k.shortName order by k.start").fetchall()
    # Per MoE layer: dispatch, fix-up after dispatch end (sanitize or gather), combine, span.
    layer = {"dispatch": [], "fixup_after_dispatch": [], "combine": [], "span": []}
    open_ = {}
    for n, s, e, sid in rows:
        if n == "moeA2ADispatchKernel":
            open_[sid] = (s, e)
            layer["dispatch"].append(e - s)
        elif n in A2A[2:] and sid in open_:
            layer["fixup_after_dispatch"].append(e - open_[sid][1])
        elif n == "moeA2ACombineKernel" and sid in open_:
            layer["combine"].append(e - s)
            layer["span"].append(e - open_.pop(sid)[0])
    # Per decode step (exactly 58 MoE layers), bounded by the once-per-step metadata copy kernel.
    marks = [s for n, s, e, _ in rows if n == "copyBatchBlockOffsetsToDeviceKernel"]
    steps, busy, i = [], [], 0
    for a, b in zip(marks, marks[1:]):
        while i < len(rows) and rows[i][1] < a: i += 1
        j, iv, nd = i, [], 0
        while j < len(rows) and rows[j][1] < b:
            n, s, e, _ = rows[j]; iv.append((max(s, a), min(e, b))); nd += n == "moeA2ADispatchKernel"; j += 1
        if nd != 58: continue
        tot, cs, ce = 0, None, None
        for s, e in sorted(iv):
            if ce is None or s > ce:
                if ce is not None: tot += ce - cs
                cs, ce = s, e
            else: ce = max(ce, e)
        if ce is not None: tot += ce - cs
        steps.append(b - a); busy.append(tot)
    us = lambda v: st.median(v) / 1000
    return path, {**{k: us(v) for k, v in layer.items()}, "layers": len(layer["span"]),
                  "step": us(steps), "gpu_busy": us(busy), "gpu_idle": us([s - b for s, b in zip(steps, busy)]),
                  "decode_steps": len(steps)}

if __name__ == "__main__":
    runs = sys.argv[1]
    cases = {f"{load}/{mode}": f"{runs}/{job}/{d}/nsys-rank5.sqlite"
             for load, job in (("8", "kexp-small-7921296"), ("64", "kexp-prod-7921297"))
             for mode, d in (("off", "noft"), ("on", "gather"))}
    with Pool(4) as p:
        res = dict(zip(cases, (r for _, r in p.map(analyze, cases.values()))))
    res["_source"] = {k: v.split("runs/")[1] for k, v in cases.items()}
    json.dump(res, open("eval_overhead.json", "w"), indent=1)
    for k, v in res.items():
        if not k.startswith("_"): print(k, {a: round(b, 2) for a, b in v.items()})

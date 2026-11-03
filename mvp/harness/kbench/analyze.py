"""Per-kernel timings per token count from the a2a_bench nsys reports.

Usage: python3 analyze.py RUN_DIR   (expects RUN_DIR/nsys-{off,on}-pdl{0,1}.sqlite; writes RUN_DIR/kernels.json)
"""
import json
import sqlite3
import statistics as st
import sys

DISPATCH, COMBINE = "moeA2ADispatchKernel", "moeA2ACombineKernel"
FIXUP = ("moeA2ASanitizeExpertIdsKernel", "moeA2AGatherLocalKernel", "moeA2AExpertInputKernel")


def analyze(path: str) -> dict:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    ranges = db.execute("select text, start, end from NVTX_EVENTS where text like 'tokens=%'").fetchall()
    kernels = db.execute("select s.value, k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k "
                         "join StringIds s on s.id = k.shortName order by k.start").fetchall()
    out = {}
    for text, start, end in ranges:
        rows = [k for k in kernels if start <= k[1] < end]
        layer = {"dispatch": [], "fixup": [], "fixup_past_dispatch": [], "sort": [], "sort_end_past_dispatch": [],
                 "combine": [], "span": []}
        last_dispatch = None
        sort = []
        for name, s, e in rows:
            if name == DISPATCH:
                last_dispatch = (s, e)
                layer["dispatch"].append(e - s)
                sort = []
            elif name in FIXUP and last_dispatch:
                layer["fixup"].append(e - s)
                layer["fixup_past_dispatch"].append(e - last_dispatch[1])
            elif not name.startswith("moeA2A") and last_dispatch:
                sort.append((s, e))  # the routing sort's kernels, between the dispatch side and combine
            elif name == COMBINE and last_dispatch:
                if sort:
                    layer["sort"].append(sum(e2 - s2 for s2, e2 in sort))
                    layer["sort_end_past_dispatch"].append(sort[-1][1] - last_dispatch[1])
                layer["combine"].append(e - s)
                layer["span"].append(e - last_dispatch[0])
                last_dispatch = None
        out[text.split("=")[1]] = {k: st.median(v) / 1e3 for k, v in layer.items() if v}
    return out


if __name__ == "__main__":
    run = sys.argv[1]
    result = {}
    for mode in ("off", "on"):
        for pdl in ("0", "1"):
            try:
                result[f"{mode}/pdl{pdl}"] = analyze(f"{run}/nsys-{mode}-pdl{pdl}.sqlite")
            except sqlite3.Error as error:
                print(f"skip {mode}/pdl{pdl}: {error}")
    json.dump(result, open(f"{run}/kernels.json", "w"), indent=1)
    for key, per_tokens in result.items():
        print(key)
        for tokens, t in sorted(per_tokens.items(), key=lambda kv: int(kv[0])):
            print(f"  tokens={tokens:>5} " + " ".join(f"{k}={v:6.2f}" for k, v in t.items()))

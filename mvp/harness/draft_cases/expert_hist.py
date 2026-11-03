"""Per-rank expert load from main's ExpertStatistic dumps (C-COMBINE3 hist cases).

Each rank's rank<r>.safetensors holds, per "<iter>_<layer>", the bincount over all EPLB slots of the slots its own
tokens were routed to. Summed over ranks this is the number of token rows each slot receives (the grouped GEMM's
input per local expert). Per (iter, layer): per-rank rows (sum of its slots) and the rank's largest slot; reported
as max over ranks / mean over ranks, averaged over the recorded steps and MoE layers.
Usage: expert_hist.py CASE_DIR [CASE_DIR ...]  (reads CASE_DIR/expert_stat/)
"""

import array
import json
import pathlib
import statistics
import struct
import sys

DTYPES = {"I64": "q", "I32": "i"}


def read_safetensors(path):
    data = path.read_bytes()
    (n,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8:8 + n])
    body = data[8 + n:]
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        out[name] = array.array(DTYPES[meta["dtype"]], body[start:end])
    return out


def summarize(case):
    stat = pathlib.Path(case) / "expert_stat"
    files = sorted(stat.glob("rank*.safetensors"), key=lambda p: int(p.stem[4:]))
    if not files:
        return f"{case}: no expert_stat dumps"
    ep = len(files)
    total = {}
    for f in files:
        for key, counts in read_safetensors(f).items():
            acc = total.setdefault(key, [0] * len(counts))
            for i, c in enumerate(counts):
                acc[i] += c
    top_k = json.loads((stat / "meta_info.json").read_text())["num_experts_per_token"]
    rank_ratio, slot_ratio, empty, tokens = [], [], [], []
    for counts in total.values():
        per = len(counts) // ep
        ranks = [counts[r * per:(r + 1) * per] for r in range(ep)]
        rows = [sum(r) for r in ranks]
        mean = statistics.mean(rows)
        if mean == 0:
            continue
        rank_ratio.append(max(rows) / mean)
        slot_ratio.append(max(max(r) for r in ranks) / (mean / per))
        empty.append(sum(c == 0 for c in counts))
        tokens.append(sum(counts) // top_k)
    return (f"{pathlib.Path(case).name:14s} steps*layers={len(rank_ratio):5d} slots={len(counts)} "
            f"tokens/step median={statistics.median(tokens):6.0f} rank rows max/mean={statistics.mean(rank_ratio):.3f} "
            f"(p90 {sorted(rank_ratio)[int(0.9 * len(rank_ratio))]:.3f}) largest slot/mean slot="
            f"{statistics.mean(slot_ratio):.2f} empty slots={statistics.mean(empty):.1f}")


if __name__ == "__main__":
    for case in sys.argv[1:]:
        print("EXPERT_HIST", summarize(case))

"""One autotuner cache for every arm of an A/B job, with one tactic per shape on every rank (FT-COST-E2E).

merge: OUT.json IN.json [IN.json ...]
    Merges the caches that each arm's tuning case wrote (same wheel, same layout). Per-rank entries (the MoE runners)
    are keyed by the rank's local expert offset; every (op, runner, config without the offset, shapes) gets the
    (runner_id, tactic) most (input, rank) entries chose, on every rank of every input that has the key (ties: the
    first input's lowest rank). Shared entries (the dense GEMMs, tuned once and broadcast) get the majority over
    inputs (ties: the first input). Keys only one input has are kept and pinned across ranks the same way. Prints
    the counts and every coverage gap: a rank missing a key another rank or input has would be tuned in a measured
    case.
check: PINNED.json CASE.json
    After a measured case: every entry of CASE.json equals PINNED.json's (the case read the cache, tuned nothing
    and wrote nothing new). Exit 1 otherwise.
"""

import ast
import collections
import json
import re
import sys

# The MoE runner config (key[2]) starts (num_slots, top_k, num_local_experts, local_expert_offset, ...).
OFFSET = re.compile(r"^\((\d+), (\d+), (\d+), \d+, ")


def shape_key(key):
    op, runner, config, *rest = ast.literal_eval(key)
    return repr((op, runner, OFFSET.sub(r"(\1, \2, \3, X, ", config, count=1), *rest))


def choice(entry):
    return json.dumps([entry["runner_id"], entry["tactic"]])


def rank_sections(cache):
    return sorted((k for k in cache if k.startswith("rank_")), key=lambda k: int(k[5:]))


def merge(dst, *srcs):
    caches = [json.load(open(src)) for src in srcs]
    shared_votes, rank_votes = collections.defaultdict(list), collections.defaultdict(list)
    for cache in caches:
        for key, entry in cache.get("shared", {}).items():
            shared_votes[key].append(choice(entry))
        for rank in rank_sections(cache):
            for key, entry in cache[rank].items():
                rank_votes[shape_key(key)].append(choice(entry))
    # Counter.most_common keeps first-seen order among ties: the first input's lowest rank wins.
    shared_pin = {k: collections.Counter(v).most_common(1)[0][0] for k, v in shared_votes.items()}
    rank_pin = {k: collections.Counter(v).most_common(1)[0][0] for k, v in rank_votes.items()}

    out = {"metadata": caches[0]["metadata"], "shared": {}}
    changed = collections.Counter()
    for cache in caches:
        for key, entry in cache.get("shared", {}).items():
            runner_id, tactic = json.loads(shared_pin[key])
            out["shared"].setdefault(key, dict(entry, runner_id=runner_id, tactic=tactic))
        for rank in rank_sections(cache):
            section = out.setdefault(rank, {})
            for key, entry in cache[rank].items():
                runner_id, tactic = json.loads(rank_pin[shape_key(key)])
                changed["rank"] += [entry["runner_id"], entry["tactic"]] != [runner_id, tactic]
                section.setdefault(key, dict(entry, runner_id=runner_id, tactic=tactic))
    for key, entry in out["shared"].items():
        for cache in caches:
            if key in cache.get("shared", {}):
                changed["shared"] += choice(cache["shared"][key]) != choice(entry)

    ranks = rank_sections(out)
    shapes_per_rank = {r: {shape_key(k) for k in out[r]} for r in ranks}
    all_shapes = set(rank_pin)
    gaps = {r: len(all_shapes - s) for r, s in shapes_per_rank.items() if all_shapes - s}
    split_ranks = sum(len(set(v)) > 1 for v in rank_votes.values())
    split_shared = sum(len(set(v)) > 1 for v in shared_votes.values())
    print(f"MERGE_PIN {dst} from {len(srcs)} caches: {len(ranks)} ranks, {len(rank_pin)} per-rank shapes "
          f"({split_ranks} with disagreeing tactics, {changed['rank']} entries changed), {len(shared_pin)} shared "
          f"keys ({split_shared} disagreeing across inputs, {changed['shared']} input entries overridden)")
    for i, (src, cache) in enumerate(zip(srcs, caches)):
        own = {shape_key(k) for r in rank_sections(cache) for k in cache[r]}
        print(f"MERGE_PIN input {i} {src}: {len(rank_sections(cache))} ranks, {len(own)} per-rank shapes "
              f"({len(all_shapes - own)} only in other inputs), {len(cache.get('shared', {}))} shared keys "
              f"({len(set(shared_pin) - set(cache.get('shared', {})))} only in other inputs)")
    print(f"MERGE_PIN coverage gaps (rank: shapes missing): {gaps or 'none'}")
    json.dump(out, open(dst, "w"))


def check(pinned_path, case_path):
    pinned, case = json.load(open(pinned_path)), json.load(open(case_path))
    added = differ = 0
    for section, entries in case.items():
        if section == "metadata":
            continue
        for key, entry in entries.items():
            ref = pinned.get(section, {}).get(key)
            added += ref is None
            differ += ref is not None and choice(ref) != choice(entry)
    ok = added == 0 and differ == 0
    print(f"PIN_CHECK {'OK' if ok else 'FAIL'} {case_path}: {added} entries not in the pinned cache, "
          f"{differ} with another tactic")
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "merge" and len(sys.argv) >= 4:
        merge(*sys.argv[2:])
    elif sys.argv[1] == "check" and len(sys.argv) == 4:
        sys.exit(0 if check(*sys.argv[2:]) else 1)
    else:
        sys.exit(__doc__)

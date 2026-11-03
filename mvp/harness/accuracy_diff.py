"""Question-by-question GSM8K diff of two accuracy cases (accuracy_client.py's OUTPUT.samples.jsonl).

Per filter: questions both runs graded, how many flipped right->wrong and wrong->right, and how many answers differ.
Greedy decoding on the same server config should give identical answers; flips beyond the reference's own
nondeterminism point at a wrong result after recovery.
Usage: python3 accuracy_diff.py CASE.samples.jsonl REF.samples.jsonl
"""
import json
import sys


def load(path):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        rows[(r["filter"], r["doc_id"])] = r
    return rows


case, ref = load(sys.argv[1]), load(sys.argv[2])
for flt in sorted({k[0] for k in ref}):
    keys = [k for k in ref if k[0] == flt and k in case]
    lost = sum(1 for k in keys if ref[k]["exact_match"] and not case[k]["exact_match"])
    gained = sum(1 for k in keys if not ref[k]["exact_match"] and case[k]["exact_match"])
    differ = sum(1 for k in keys if ref[k]["answer"] != case[k]["answer"])
    print(f"ACCURACY_DIFF filter={flt} graded={len(keys)} right_to_wrong={lost} wrong_to_right={gained} answers_differ={differ}")

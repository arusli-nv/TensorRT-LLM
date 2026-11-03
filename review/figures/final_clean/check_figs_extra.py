"""Consistency check for make_figs_extra.py: every derived number printed on Fig 9 / A3-A5 reproduces from the other
numbers printed on the same chart (within the printed rounding). Reads the CSVs the script writes.

usage: /usr/bin/python3 check_figs_extra.py
"""
import csv
import os

OUT = os.path.dirname(os.path.abspath(__file__))


def rows(name):
    return list(csv.DictReader(open(f"{OUT}/{name}.csv")))


ok = True


def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg)
    ok &= bool(cond)


# Fig A5: printed parts add up to the printed total; the median survivor's total lies inside the printed whisker.
for r in rows("figA5_remap_phases"):
    parts = [int(r[k]) for k in ("other_printed_ms", "import_map_printed_ms", "outcome_printed_ms")]
    total = int(f"{float(r['total_ms']):.0f}")
    check(sum(parts) == total, f"A5 {r['case']}: {' + '.join(map(str, parts))} = {sum(parts)} vs printed {total}")
    check(float(r["min_total_ms"]) <= float(r["total_ms"]) <= float(r["max_total_ms"]),
          f"A5 {r['case']}: whisker {r['min_total_ms']}-{r['max_total_ms']} contains {r['total_ms']}")

# Fig A4: the printed mean equals the mean of the printed run totals (one unit of the last printed digit).
a4 = rows("figA4_restart_vs_ft")
for mode in ("noeplb", "offline", "online"):
    runs = [float(r["total_s"]) / 60 for r in a4 if r["what"] == "restart" and r["mode"] == mode
            and not r["source"].startswith("MEAN")]
    mean = [float(r["total_s"]) / 60 for r in a4 if r["mode"] == mode and r["source"].startswith("MEAN")][0]
    printed_runs = [round(x, 1) for x in runs]
    derived = sum(printed_runs) / len(printed_runs)
    check(abs(round(mean, 1) - derived) <= 0.05 + 1e-9,
          f"A4 {mode}: printed mean {mean:.1f} min vs mean of printed runs {printed_runs} = {derived:.3f}")
    phases = [r for r in a4 if r["mode"] == mode and r["source"].startswith("MEAN") and r["what"] == "restart"][0]
    s = sum(float(phases[k]) for k in ("model_load_s", "autotune_s", "graph_capture_s", "other_s"))
    check(abs(s - float(phases["total_s"])) < 0.25, f"A4 {mode}: phase means sum {s:.1f} s = total {phases['total_s']} s")
fig7 = {r["mode"]: r["value_s"].split()[0] for r in rows("fig7_recovery")
        if r["panel"] == "a" and r["failure"] == "rank" and r["source"].startswith("MEAN")}
for r in a4:
    if r["what"] == "FT on recovery":
        check(f"{float(r['total_s']):.2f}" == f"{float(fig7[r['mode']]):.2f}",
              f"A4 {r['mode']}: FT label {float(r['total_s']):.2f} s = fig7 mean {fig7[r['mode']]}")

# Fig A3: each printed median lies inside its printed min-max (the medians are measured, not derived).
for r in rows("figA3_a2a_ablation"):
    check(float(r["min_us"]) - 0.005 <= float(r["us_saved"]) <= float(r["max_us"]) + 0.005,
          f"A3 {r['check_removed']} {r['case']}: {r['us_saved']} in [{r['min_us']}, {r['max_us']}]")

# Fig 9: the band is offered x (1 +- the same 2 SD fraction) in every panel; 12 pre-kill bins per panel.
f9 = rows("fig9_kill_timeline")
fr = {(float(r["noise_band_hi"]) / float(r["offered_out_tok_s"]) - 1) for r in f9}
check(max(fr) - min(fr) < 1e-3, f"fig9: band half-width fraction {min(fr):.4f}-{max(fr):.4f} is one value")
for key in {(r["test"], r["kill"]) for r in f9}:
    n = sum(1 for r in f9 if (r["test"], r["kill"]) == key and int(r["bin_start_s_from_kill"]) < 0)
    check(n == 12, f"fig9 {key}: {n} pre-kill bins")
print("ALL PASS" if ok else "SOME FAILED")

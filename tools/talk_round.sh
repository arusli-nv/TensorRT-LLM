#!/bin/bash
# One blind quality round for the slide figures: install the rendered *_talk figures, shuffle them with public TRT-LLM
# blog charts/diagrams, run one headless judge per set, unblind, and print our scores and the judge's fixes for ours.
# Usage: talk_round.sh <round> <render dir with *_talk.png/svg>   (judge briefs: $TMPDIR/judges_r1.5PJ5/{charts,design}_v2.txt)
set -euo pipefail
N=$1; SRC=$2
R=$(cd "$(dirname "$0")/.." && pwd); F=$R/review/figures/final_clean; M=/tmp/claude-161238/wt-split/docs/source/blogs/media
B=/tmp/claude-161238/judges_r1.5PJ5
cp "$SRC"/*_talk.png "$SRC"/*_talk.svg "$F/"; mkdir -p "$R/FINAL/figures/talk"; cp "$F"/*_talk.png "$R/FINAL/figures/talk/"
OURS_C=$(ls "$F"/{restart,recovery,phases,after_loss,kill_timeline,healthy,step_cost,realistic_latency,remap_phases,a2a_ablation,ft_change}_talk.png)
REF_C="$(ls "$F"/reference/tech_blog30_*.png) $M/tech_blog27_hit_rate_eviction_cliff.png $M/tech_blog27_batch_size_sweep_fixed_concurrency.png $M/tech_blog21_e2e_tept8_osl1k_bar.png $M/tech_blog26_dsv4_performance_evolution_8k1k.png $M/tech_blog10_tps_ttft_pareto_curve.png $M/tech_blog21_synthetic_scaling.png $M/tech_blog23_queuing_delays.png"
OURS_D="$F/detect_talk.png $F/phases_rise_talk.png $F/fig1_architecture_final.png $F/fig2_ghost_slots_final.png $F/fig3_placement_final.png $F/figH_oct20_late_writes.png $F/fig4_late_writes_final.png $F/fig_recovery_sequence_final.png"
REF_D="$M/tech_blog18_dispatch_moe_combine.png $M/tech_blog19_dwdp_overview.png $M/tech_blog22_helix_execution_flow.png $M/tech_blog18_rank_major_vs_expert_major.png $M/tech_blog15_ds32_wide_ep.png"
mk() {
    local K=$1; shift; local D=/tmp/claude-161238/blind_talk_r${N}_$K KEY=$R/review/diag/blind_key_talk_r${N}_$K.txt i=0
    mkdir -p "$D"; : > "$KEY"
    for f in $(printf '%s\n' "$@" | shuf); do i=$((i+1)); n=$(printf "%s%02d.png" "${K:0:1}" $i); cp "$f" "$D/$n"; echo "$n $f" >> "$KEY"; done
}
mk charts $OURS_C $REF_C; mk design $OURS_D $REF_D
for K in charts design; do
    ( cd /tmp/claude-161238/blind_talk_r${N}_$K && timeout 1500 claude -p "$(cat $B/${K}_v2.txt)" --allowedTools "Read" > $B/r${N}_$K.txt 2>&1 ) &
done
wait
python3 -I - "$R" "$N" "$B" <<'EOF'
import re, sys
R, N, B = sys.argv[1:]
for kind in ("charts", "design"):
    key = {l.split()[0][:3]: l.split()[1] for l in open(f"{R}/review/diag/blind_key_talk_r{N}_{kind}.txt")}
    txt = open(f"{B}/r{N}_{kind}.txt").read()
    rows = {m.group(1): (float(m.group(2)), m.group(3)) for m in re.finditer(
        r"^\|\s*\*{0,2}([cd]\d\d)(?:\.png)?\*{0,2}\s*\|\s*\*{0,2}(\d+(?:\.\d+)?)\*{0,2}\s*\|(.*)$", txt, re.M)}
    ours = {k: v for k, v in rows.items() if "final_clean/" in key[k] and "reference/" not in key[k]}
    refs = {k: v for k, v in rows.items() if k not in ours}
    mean = lambda d: sum(v[0] for v in d.values()) / max(1, len(d))
    print(f"== round {N} {kind}: ours {mean(ours):.2f} (n={len(ours)}), refs {mean(refs):.2f} (n={len(refs)})")
    for k, (score, rest) in sorted(ours.items(), key=lambda x: x[1][0]):
        print(f"  {score:4.1f} {key[k].split('/')[-1]:32s} {rest.strip()[:330]}")
EOF

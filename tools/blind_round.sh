#!/bin/bash
# Usage: blind_round.sh <round> [refs|norefs] [design]  -> prints the blinded image dir; key in review/diag/blind_key_round<N>.txt
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); F=$R/review/figures/final_clean; N=$1; MODE=${2:-refs}; DES=${3:-}
K=$R/review/diag/blind_key_round$N.txt; S=$(mktemp -d); mkdir "$S/imgs"; : > "$K"
L=$(ls $F/fig5_healthy.png $F/fig6_ft_change.png $F/fig7_recovery.png $F/fig7b_phases.png $F/fig8_after_loss.png $F/fig9_kill_timeline.png $F/fig10_realistic_latency.png $F/figA1_step_cost.png $F/figA3_a2a_ablation.png $F/figA4_restart_vs_ft.png $F/figA5_remap_phases.png)
[ "$MODE" = refs ] && L="$L $(ls $F/reference/tech_blog30_*.png)"
[ -n "$DES" ] && L="$L $(ls $F/fig1_architecture_final.png $F/fig2_ghost_slots_final.png $F/fig3_placement_final.png $F/fig4_late_writes_final.png $F/fig_recovery_sequence_final.png)"
i=0; for f in $(echo $L | tr ' ' '\n' | shuf); do i=$((i+1)); n=$(printf "r%02d_%02d.png" "$N" $i); cp "$f" "$S/imgs/$n"; echo "$n $f" >> "$K"; done
echo "$S/imgs"

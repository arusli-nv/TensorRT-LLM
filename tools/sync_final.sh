#!/bin/bash
# Rebuild FINAL/ (the clean, user-facing folder) from the working copies under review/ and mvp/.
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
F=$R/FINAL
mkdir -p "$F/figures" "$F/oct20/figures/draft"
for f in fig1_architecture_final fig2_ghost_slots_final fig3_placement_final fig4_late_writes_final \
         fig_recovery_sequence_final fig5_healthy fig6_ft_change fig7_recovery fig7b_phases fig8_after_loss figH_oct20_late_writes fig9_kill_timeline figA3_a2a_ablation figA4_restart_vs_ft figA5_remap_phases fig10_realistic_latency figD_failure_detection_final figA1_step_cost figA2_throughput; do
    cp "$R/review/figures/final_clean/$f.png" "$F/figures/"
done
# Only image links move; the doc's other file paths are relative to the runs root, as its citation line says.
sed -e 's#](figures/final_clean/#](figures/#g' \
    "$R/review/DESIGN_DOC_final.md" > "$F/DESIGN_DOC_final.md"
cp "$R/review/PRESENTER_NOTES.md" "$F/PRESENTER_NOTES.md"
cp "$R/review/MORNING_SUMMARY.md" "$F/SUMMARY.md"
cp "$R/review/OVERNIGHT_CHANGES_2026-10-26.md" "$F/OVERNIGHT_CHANGES.md"
cp "$R/mvp/DESIGN_DOC.md" "$F/oct20/DESIGN_DOC_oct20.md"
cp "$R/mvp/figures/"{fig1_architecture,fig2_ghost_slots,fig3_late_writes,fig5_placement}.png "$F/oct20/figures/"
cp "$R/mvp/figures/draft/"{fig7_recovery_bars,fig_after_loss,fig_healthy_delta,fig_healthy_metrics,fig_step_cost,fig_throughput}.png "$F/oct20/figures/draft/"
mkdir -p "$F/figures/slides"
for f in "$R"/review/figures/final_clean/*_slide.png; do [ -e "$f" ] && cp "$f" "$F/figures/slides/"; done
mkdir -p "$F/figures/talk"
for f in "$R"/review/figures/final_clean/*_talk.png; do [ -e "$f" ] && cp "$f" "$F/figures/talk/"; done
echo "FINAL synced $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M PT')"

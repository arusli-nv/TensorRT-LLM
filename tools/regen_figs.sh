#!/bin/bash
# Rebuild every figure from its data, then sync FINAL/: doc charts (make_figs.py, make_figs_extra.py, their --slide
# variants), the restart chart (make_fig_today.py), the slide charts (make_talk_figs.py, from the CSVs the doc scripts
# write), and the design figures (fig_design_final.py, make_talk_design.py, which need Lato through fontconfig).
# Renders into a scratch copy first, so a failing script leaves the installed figures untouched.
# Usage: regen_figs.sh
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); F=$R/review/figures/final_clean
S=$(mktemp -d "${TMPDIR:-/tmp}/regen.XXXXXX")
cp "$F"/*.py "$F"/*.csv "$S/"
mkdir -p "$S/reference"
export MPLCONFIGDIR=$S/mpl
cd "$S"
for args in "make_fig_today.py" "make_figs.py" "make_figs.py --slide" "make_figs_extra.py" "make_figs_extra.py --slide" \
            "make_talk_figs.py" "make_appendix.py"; do
    /usr/bin/python3 $args > "$S/log-${args// /_}.txt" 2>&1 || { echo "REGEN_FAIL $args (log $S/log-${args// /_}.txt)"; exit 1; }
done
FC=$S/fonts.conf
cat > "$FC" <<EOF
<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig><dir>$R/mvp/analysis/fonts</dir><include ignore_missing="yes">/etc/fonts/fonts.conf</include><cachedir>$S/fccache</cachedir></fontconfig>
EOF
FONTCONFIG_FILE=$FC PYTHONPATH=/tmp/claude-161238/figlib:$S /usr/bin/python3 fig_design_final.py "$S" > "$S/log-design.txt" 2>&1 \
    || { echo "REGEN_FAIL fig_design_final.py"; exit 1; }
FONTCONFIG_FILE=$FC PYTHONPATH=/tmp/claude-161238/figlib:$S /usr/bin/python3 make_talk_design.py "$S" > "$S/log-talk-design.txt" 2>&1 \
    || { echo "REGEN_FAIL make_talk_design.py"; exit 1; }
# PNGs and CSVs decide what changed (SVGs carry a timestamp); a changed PNG brings its SVG along. The previous file
# goes to TRASH/<date>/regen-<time>/ so every regeneration can be reverted.
changed=0; T=$R/TRASH/$(date +%F)/regen-$(date +%H%M%S)
for f in "$S"/*.png "$S"/*.csv; do
    b=$(basename "$f")
    if ! cmp -s "$f" "$F/$b" 2>/dev/null; then
        mkdir -p "$T"; [[ -f $F/$b ]] && cp "$F/$b" "$T/"
        cp "$f" "$F/$b"; [[ $b == *.png && -f ${f%.png}.svg ]] && cp "${f%.png}.svg" "$F/"
        echo "REGEN_CHANGED $b"; changed=$((changed + 1))
    fi
done
bash "$R/tools/sync_final.sh" > /dev/null
echo "REGEN_DONE changed=$changed scratch=$S"

# WideEP FT: final package

Open these:

| File | What it is |
|---|---|
| `DESIGN_DOC_final.md` | **The final design doc.** Same structure as the Oct 20 doc, final data, figures in `figures/`. |
| `SLIDES_DRAFT.md` | Mid-point slide draft: text and figure per slide, speaker notes with sources. |
| `PRESENTER_NOTES.md` | Talking points and likely questions for the presentation. |
| `SUMMARY.md` | One-page summary: numbers, PR status, open items. |
| `OVERNIGHT_CHANGES.md` | What changed overnight (10-25 -> 10-26), and the decisions waiting on you. |
| `OVERNIGHT_LOG.md` | What changed overnight 10-26 -> 10-27 (slides, figures, doc, new data), with a morning summary and how to revert each change. |
| `DATA.md` | All current data in one file: context, configs, metric definitions, raw and aggregate values per figure, and a script that redraws every figure (`python3 recreate.py DATA.md out/`). |
| `oct20/DESIGN_DOC_oct20.md` | Your Oct 20 doc, unchanged, with its own figures, for comparison. |

Figures:
- **Design (Figs 1-4, R):** your Oct 20 style, redrawn for the final design (same-address remap, five-phase
  recovery). Generator: `review/figures/final_clean/fig_design_final.py`.
- **Evaluation (Figs 5-10, A1-A5):** final data. Generators: `review/figures/final_clean/make_figs.py` and `make_figs_extra.py`.

Everything else (run logs, CLAIMS, readouts, plans, drafts) is working material:
- evidence rows: `../mvp/CLAIMS.md`;
- analyses: `../review/diag/`;
- run directories: `../mvp/runs/`.

The working copy of the doc is `../review/DESIGN_DOC_final.md`. This folder is rebuilt from the working copies by
`../tools/sync_final.sh`, so do not edit files here.

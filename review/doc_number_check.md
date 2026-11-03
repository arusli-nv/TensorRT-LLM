# Number check: DESIGN_DOC_v5final_tonight.md (04:14 PT), APPENDIX_E_compaction_draft.md, MORNING_SUMMARY.md

Read-only check, 2026-10-25 ~04:20 PT. Sources: mvp/CLAIMS.md rows 87, 176-197 (CLAIMS mtime 04:03); review/v5f_readout_full.txt
(= v5f_readout_dev.txt, identical in every table); review/v5f_tpot_ttft.txt; kill_final.txt in KILL-FINAL-{ON,ON-A,OFF,OFF-A}-8269102
and KILL-FINAL-ON-A-8271210; ACCURACY-KILL-ON-8269102/accuracy_kill.txt; for frozen-build rows the doc's own cited
review/DESIGN_DOC_v5final_filled_provenance.md (P-rows -> kill_final files of 8206860-8215035); review/diag/probe4-8271450.txt;
HANDOFF.md where CLAIMS has no row.

Build key: final = v5-final 27442bd91d (overlay fix-dropmap-9f1fd4cab4; V5F jobs 8269102/8269103/8270779/8271451/8271210).
frozen = build-freeze-58caa0fde6 (finals). fix-1 = d71f169e4b (CLAIMS 185/186). No fix-1 number appears in any of the three files.

Burst count check: CLAIMS 194 (4 FT + 4 off) + 196 (v5-final ovA 2 + off 2; the 2 option-B windows excluded) + 197 (2 FT + 3 off)
= 0/8 FT, 0/9 FT off, all on 27442bd91d product files, all online healthy-32 windows. Count correct.

Verdicts: OK / MISMATCH / UNSOURCED / STALE. "Note" = value right, but labeling or strength issue.

## DESIGN_DOC_v5final_tonight.md

| doc:line | quoted | source | verdict | note |
|---|---|---|---|---|
| 3 | "Updated: Oct 23 \| Code: v5-final `58caa0fde6`" | CLAIMS 193/194 (v5-final = 27442bd91d) | MISMATCH | 58caa0fde6 is the frozen finals build; v5-final is 27442bd91d (the doc says so at line 239). Date also stale |
| 23 | "1.0-2.2 s (median 1.1-1.7 s by mode)" | provenance P1/P8 (frozen); CLAIMS 176-178 | OK | Frozen build; v5-final rank rows (1.06-1.71 s, CLAIMS 193) fall inside |
| 24 | "1.7-3.2 s" node | provenance P2 (1.72-3.21, frozen) | OK | v5-final 1.72-2.71 inside |
| 25 | "fenced after ~35 s" | CLAIMS 176/178/193: confirm 30.0-34.1 s | OK | Fencing is at confirm (30-34 s); resume at 33-38 s. "~35" is loose |
| 26 | "within +-1.2% output tok/s in every cell except offline at 32/GPU (-2.0%)" | v5f_tpot_ttft.txt output_throughput | Note (stronger than data) | Point estimates are +0.50/-1.15/-0.35/-0.18/-0.05, max abs 1.15, OK; but the 95% CIs reach -4.13/+3.43 (offline 128) and -3.30 (no-EPLB 128). "within +-1.2%" reads as a bound |
| 83 | "no EPLB +0.96% / +0.53%, offline +0.59% / +0.34%, online +0.48% / +0.39%" | v5f_readout_full.txt:52,57,62,67,72,77 | OK | Final build |
| 150 | "~0.02 s (median, rank and node kills)" | provenance P5 (frozen, n=24) | OK | |
| 196 | "1.7 s (1.2-2.2, n=2) / 1.1 s (1.0-1.2, n=2)" | P8: 1.725 (1.24-2.21) / 1.11 (1.01-1.21) | OK | Frozen build (whole Recovery section; the section does not say so, line 188 cites the frozen provenance) |
| 197 | "1.7 s (1.2-2.2, n=3) / 1.1 s (1.0-1.2, n=2)" | P8: 1.71 (1.19-2.21, n=3) / 1.115 (1.02-1.21) | OK | Frozen |
| 198 | "1.6 s (1.1-2.1, n=3) / 1.2 s (n=2)" | P8: 1.58 (1.09-2.09, n=3) / 1.18 | OK | Frozen |
| 199 | "1.2 s (n=2)" EPLB update | P8: 1.195 | OK | Frozen |
| 200 | "3.2 s (n=2) / 1.7 s (n=2)" node | P8: 3.205 / 1.72 | OK | Frozen |
| 201 | "Two losses ... 1.2 s each (n=2)" no EPLB | KILL-FINAL-OFF-8211797: h23_resume_s=1.22,1.75 and 1.2,1.93 (resume2 median 1.84); CLAIMS 178 | MISMATCH | Second loss no-EPLB is 1.75/1.93 s (median 1.8), not 1.2. Online 1.22,1.12 / 1.2,1.12 -> "1.2 / 1.1 s" (minor). P8 quotes only the first loss |
| 202 | "34.4 s total (34.1-34.6, n=2) / 36.6 s total (36.1-37.1, n=2)" | P8: 34.35 (34.1-34.6) / 36.595 (36.09-37.1) | OK | Frozen |
| 203 | "318 s (n=1) / 605 s (585-625, n=2)" | P8 cold_restart: 318.1 / 604.75 (584.4-625.1) | MISMATCH (rounding) | 584.4 rounds to 584, not 585 |
| 205 | "detect 0.02 s, plan 0.28 s, ready 1.3 s, commit 1.35 s, first step 1.43 s" | P8 phases | OK | Frozen |
| 214 | "0.95x / 0.96x" | P9: 0.946 / 0.956 (n=3 / n=2) | OK | Frozen; v5-final 128 rank5 0.94/0.956 and 0.955/0.949 consistent |
| 215 | "0.88x / 0.88x" | P9: 0.878 / 0.875 | OK | Frozen |
| 223 | "0.9469 / 0.9416 / +0.005 (2 SE 0.018)" | P11 (ACCURACY-KILL-ON-8206870) | OK | Frozen build; table not labeled (final-build run is at line 270) |
| 224 | "0.9431 / 0.9409 / +0.002 (2 SE 0.018)" | P11 (8215035) | OK | Frozen |
| 228 | "0.4 GB ... 1.6 GB (busiest survivor)" | P10: 444 / 1614 MB | OK | Frozen; v5-final busiest 396-444 / 1476-1752 MB |
| 233 | "-0.26% to +0.37% at 32 and 128/GPU (n=4)" parity | CLAIMS 180 parity (no-EPLB -0.05/-0.01, online -0.26/+0.37) | OK | Frozen-build finals; v5-final ran no main arms. Not labeled as frozen |
| 239 | "Build v5-final 27442bd91d. Source: CLAIMS 194" | CLAIMS 194 | OK | |
| 243 | "+0.50% (-1.40, +2.41) \| -2.41% \| -1.99%" | v5f_tpot_ttft.txt:5,3,6 | OK | |
| 244 | "-1.15% (-3.30, +1.01) \| -0.55% \| +0.75%" | v5f_tpot_ttft.txt:9,7,10 | OK | readout_full has -3.31 (client tok/s/GPU); tpot file -3.30 |
| 245 | "-2.03% (-3.56, -0.49) \| +2.21% \| +3.55%" | v5f_tpot_ttft.txt:13,11,14; CLAIMS 194 | OK | |
| 246 | "-0.35% (-4.13, +3.43) \| -0.27% \| +0.00%" | v5f_tpot_ttft.txt:17,15,18 | OK | |
| 247 | "-0.18% (-1.94, +1.57) \| -0.29% \| -0.31%" | v5f_tpot_ttft.txt:21,19,22 | OK | |
| 248 | "-0.05% (-1.67, +1.57) \| -0.11% \| +0.41%" | v5f_tpot_ttft.txt:25,23,26 | OK | |
| 251 | "about -2% throughput and +3.6% p99 TPOT ... (no burst)" | CLAIMS 194 (-2.03, +3.55, "no burst by the 1.5 s rule") | OK | |
| 252, 284 | "~390 GB of host copies per node" | CLAIMS 194 (hostmem 390 GB vs 0.5 MB; per node not stated); HANDOFF 17:45 "~390 GB/node" | OK | |
| 253 | "TTFT ... (CIs +-6% to +-35%)" | v5f_tpot_ttft.txt median_ttft rows | MISMATCH (minor) | CI half-widths are +-6.7 to +-28.6; +35.30 is one outer bound (online 32 [-21.90, +35.30]). All CIs include 0, so "no significant change" is OK |
| 254 | "No mid-window stall burst in 8 online windows (4 FT, 4 FT off)" | CLAIMS 194 | OK | Healthy-32 windows |
| 260-265 | full/trim cells, e.g. "+0.18% (-2.08, +2.43) \| +0.96% (+0.68, +1.25)" | v5f_readout_full.txt:52-53,57-58,62-63,67-68,72-73,77-78 | OK | All 12 cells checked, values and CIs match, no mode swaps |
| 268 | "not claimed: no-EPLB at 32/GPU is +0.96% with a 95% CI up to +1.25%" | v5f_readout_full.txt:52 (and :72, :103-107) | Note (incomplete) | +0.96 is the right cell and +1.25 its two-sided upper. But +1.25 is not the largest upper bound: online 32 trim reaches +1.30 (:72). The claim rule uses one-sided UBs: no-EPLB 32 +1.18 and online 32 +1.09 both exceed 1%, so rule B fails in two cells, not one; rule A also fails on full mean (UBs up to +3.76) |
| 269-270 | "every restore row passes (CLAIMS 193) ... 0.9477 vs 0.9469" | CLAIMS 193; accuracy_kill.txt (delta +0.0008, 2 SE 0.0174) | Note | GSM8K OK. CLAIMS 193 still says rank5upd pending; the row passed in KILL-FINAL-ON-A-8271210 (1.19 s, ratio 0.922, client_failed 0), which no CLAIMS row cites yet |
| 274 | "the kernel compacts memory for ~20-90 s" | CLAIMS 183: 45-90 s pass | MISMATCH | 20-30 s is CLAIMS 184's stall burst with fix 1, not the compaction duration. Appendix E (line 380) says 45-90 s |
| 277-278 | "before the fixes 21-46 s (n=3, offline); with both, 2.7 / 5.7 s against 2.2 / 3.2 s with FT off (CLAIMS 184, 187)" | CLAIMS 184 (21/31/46), 187 (2.7/5.7; off 2.2 fresh / 3.2 used) | Note (cross-method) | Values match, but 21-46 is from 184's readout (buckets 2-9 over 90 s, job 8241601) and 2.7/5.7 from 187's (50 ms-5 s steps, 10-90 s, job 8253006). The two methods disagree on fix 1 alone (6-11 s in 184, 15.0-17.5 s in 187), so "21-46 -> 2.7/5.7" is not a same-method before/after. The FT-off 2.2 is a fresh node, 3.2 used |
| 279 | "The burst falls before the measured windows." | none in CLAIMS 183-197 | UNSOURCED | |
| 280 | "+0.3% to +1.0% per decode step (trim mean)" | v5f_readout_full cells +0.34..+0.96 | OK | |
| 281 | "Not from hang detection, the peer-failure poll, the spare-slot memory" | CLAIMS 181, 182 (frozen build, n=1 per arm per group, "within spread") | Note (stronger than data) | n=1 diagnostics on the frozen build |
| 281 | "the extra slots (320 vs 288: ~0)" | no CLAIMS row; HANDOFF 17:18 / review/slots_ab_readout.txt (SLOTS-AB 8246041, lay320, offline, n=1) | UNSOURCED (in CLAIMS) | |
| 282 | "~0.6-0.8 us per dispatch+combine under graph replay (4 ranks, one node)" | CLAIMS 191 (+0.57..+0.81) | OK | Frozen build, EP4 |
| 283 | "at 59 per step that is ~0.04 ms (~0.3%)" | derived; 59 not in a source | UNSOURCED (59) | 59 x 0.7 us = 0.041 ms = ~0.3% of ~13.9 ms, so arithmetic OK |
| 285 | "ring copies (~49 GB ...)" | CLAIMS 194 | OK | |
| 286 | "Mid-window stall bursts (30-80 s, one node's threads waiting on kernel compaction) hit 3 of 12 online FT windows" | CLAIMS 188 context, 194, 197 (3/12); 30-80 s from MIDWINDOW_BURSTS.md | Note | 3/12 OK. The mechanism is "candidate" in CLAIMS 192, stated here as fact |
| 287 | "0 in every window checked (0/8 FT, 0/9 FT off). CLAIMS 194, 196, 197" | CLAIMS 194 + 196 + 197 | OK | Count verified (4+2+2 / 4+2+3). 0/8 vs the earlier 3/12 is not statistically decisive (Fisher one-sided p ~0.19); the wording claims only "0 seen", which is fine |
| 288-289 | "one FT-off burst seen in a diagnostic run" | BURST_FORENSICS.md ("the only burst is in the FT-off arm") | OK | Not in a CLAIMS row |
| 290 | "driver 580 is below the 615 that CFT needs" | CLAIMS 87 | OK | |
| 301 | "frozen build ... rank kill 9/9, in prefill 5/5, EPLB update 2/2, node kill 4/4, two sequential 4/4, frozen 4/4, rank 0 4/4, deadline 4/4; GSM8K 2/2" | provenance P14 | OK | |
| 301 | "On the final build (v5-final): 15/15 restore rows and GSM8K with a kill PASS (CLAIMS 193)" | CLAIMS 193: 14 rows ran (7 online + 7 no-EPLB), rank5upd pending; 15th = KILL-FINAL-ON-A-8271210 rank5upd PASS | Note (citation) | 15/15 is true from the run files, but CLAIMS 193 supports 14/14 + pending. Needs CLAIMS 193 updated with 8271210 (or cite it) |
| 380 | "45-90 s compaction pass that moves ~100 GB per memory node" | CLAIMS 183 | OK | |
| 382 | "3 of 12 online FT windows had a 30-80 s burst" | CLAIMS 188 context | OK | |
| 385-386 | "21-46 s -> 6-11 s (n=3 each)" | CLAIMS 184 | OK | Same method within 184 |
| 387-388 | "16.8 / 15.0 s -> 2.7 / 5.7 s, FT off 2.2 / 3.2 s" | CLAIMS 187 (monitor values) | OK | Note: fix 1 reads 6-11 s two lines up and 15.0-16.8 s here (different methods, unexplained to the reader) |
| 390-391 | "0/4 FT + 0/4 off; 0/2 FT + 0/3 off; B-AB2 0/2 FT + 0/2 off" | CLAIMS 194, 197, 196 | OK | |
| 394 | "Pinning ... cuts compaction stalls ~5x" | CLAIMS 192 (29.4 -> 5.7 s) | OK | |
| 395 | "Pinning every copy starved nodes in an earlier run (job 8121589). CLAIMS 192." | HANDOFF 07:37 (HPIN-AB 8121589: nodes could not exec binaries) | Note (citation) | Value sourced in HANDOFF, not CLAIMS 192. CLAIMS 192 reports a 1-node probe with 240 GiB pinned that stayed healthy |
| 400 | "~10x less slow-copy time than /dev/shm" | CLAIMS 195; probe4-8271450.txt per-source totals | MISMATCH (stronger than data) | Summed over the 10 blocks: S1 71.1 s, S2 43.6 s, S3 14.9 s, so 4.8x vs prefaulted shm and 2.9x vs lazy shm. ~10x holds only per typical block, with the 8.9 s block excluded |
| 400 | "residual waits (22 vs 110 / 65 samples)" | CLAIMS 195; probe4 DSTATE/PROBE4_SRC lines | MISMATCH (mixed basis, inherited from CLAIMS 195) | 110 and 22 are all-block counts, 65 is S2 over its 4 eligible blocks. Same basis: migration_entry_wait samples, all blocks, S1 109 / S2 86 / S3 22; or eligible-block reader_D 110 / 65 / 9 |
| 402 | "slow copies (8.9 s) were within the /dev/shm range (3.8-11.6 s)" | CLAIMS 195 | OK | |
| 403 | "adding ~200 s to startup" | CLAIMS 196 (758 vs 547 s ready; load 423 vs 207 s) | OK | |
| 406 | "-1.1% to +0.3% at 32/GPU, -2.5% to -0.8% at 128/GPU ... higher TTFT, 0 bursts in either" | CLAIMS 196 | OK | TTFT source is B vs FT off (+1.8 to +13.7%), not B vs the final code |

## APPENDIX_E_compaction_draft.md

Same text as the doc's lines 371-407 (the only difference: the doc adds "CLAIMS 194, 196, 197" at line 391). Same verdicts:
draft:9 OK, :11-12 OK, :15-16 OK, :17-18 OK (cross-method note), :20-21 OK, :24 OK, :25 Note (8121589 cited to CLAIMS 192; source is
HANDOFF), :30 "~10x" MISMATCH (totals 4.8x / 2.9x), :30 "22 vs 110 / 65" MISMATCH (mixed basis), :32 OK, :33 OK, :36 OK.

## MORNING_SUMMARY.md

| line | quoted | source | verdict | note |
|---|---|---|---|---|
| 7 | "v5-final 27442bd91d = finals build + fix 1 + fix 2" | CLAIMS 193/194 | OK | |
| 8-9 | "every recovery row passes (15/15) ... (0.9477 vs 0.9469). CLAIMS 193" | CLAIMS 193 (14 ran + rank5upd pending); KILL-FINAL-ON-A-8271210; accuracy_kill.txt | Note (citation) | Same as doc:301: 15/15 needs 8271210, which CLAIMS 193 does not cite yet |
| 14 | "+0.50% / -1.15% \| -2.4% / -0.6% \| -2.0% / +0.8%" | v5f_tpot_ttft.txt | OK | -0.55 -> -0.6 and +0.75 -> +0.8, rounding |
| 15 | "-2.03% / -0.35% \| +2.2% / -0.3% \| +3.6% / 0.0%" | v5f_tpot_ttft.txt | OK | |
| 16 | "-0.18% / -0.05% \| -0.3% / -0.1% \| -0.3% / +0.4%" | v5f_tpot_ttft.txt | OK | |
| 18-19 | "3 of 12 ... 0/4 FT + 0/4 off; 0/2 FT + 0/3 off; B-AB2 0/2 FT + 0/2 off; 0/2 on option B" | CLAIMS 194, 196, 197 | OK | |
| 25-26 | "~2% ... ~390 GB/node ... ~49 GB" | CLAIMS 194; HANDOFF | OK | |
| 27 | "+0.3% to +1.0% (trim mean). '<1% per step' not claimed" | v5f_readout_full | OK | |
| 32-33 | "-1.1% to +0.3% at 32, -2.5% to -0.8% at 128 ... +200 s startup" | CLAIMS 196 | OK | |
| 34 | "The 1-node probe's ~10x fewer stalls" | probe4-8271450.txt | MISMATCH (stronger than data) | Same as doc:400; also "fewer stalls" vs "less slow-copy time" (copies > 10 ms over all blocks: S1 2593 / S2 1907 / S3 331, i.e. 7.8x / 5.8x) |
| 35 | "CLAIMS 195, 196" | | OK | |

# Overnight plan 2026-10-27 (slides + data)

User, 10-26 ~23:30 PT: use both GPU slots all night; highest-EV fixes and data for the slides first, more robust work
later. Update figures, slides and doc automatically when the result is clearly better (more runs, less noise, cleaner);
when it is ambiguous, decide and say so in the log. Everything must be revertible: nothing deleted; superseded files go
to TRASH/2026-10-27/ or keep a .pre-<tag> copy.

## Rules
- Code frozen at v5-final 27442bd91d (overlay fix-dropmap-9f1fd4cab4 on build-freeze-58caa0fde6). Measurements only.
  Red-team findings that need code go to next steps.
- Rules before data: every launch file states what it decides and its pass rule before submit.
- Every new number gets a CLAIMS row before it reaches a figure, slide or doc.
- Every change to a presented number goes into FINAL/OVERNIGHT_LOG.md as old -> new, source, why.
- Slides: no cuts and no new main slides (backup slides and appendix figures are fine). Graph-only figures.
- Jobs: presubmit token, segment 8, common:ro mount, <= 2 h, both slots busy; poll ~3-5 min (~2 min at startup).

## Queue (highest EV first)
GPU:
1. RECOV2 (16 nodes): thin recovery rows on slides 11/18: prefill kill (no EPLB, online, offline), kill in an EPLB
   update (online), frozen rank with offline EPLB (never run).
2. RESTART-V5F (16 nodes): FT-off warm restarts on the presented build, 3 per mode (slide 1, CLAIMS 224 has 3 builds).
3. HEALTHY-128 (16 nodes): 4 more FT on/off pairs per mode at 128/GPU (slide 15 right panel n=4 -> 8).
4. REALISTIC-S1B (16 nodes): 2 more S1 FT on/off pairs (slide 17 n=2 -> 4, CI possible).
5. PAD-TEST: post-loss CUDA graph size at the survivors' batch (explains slide 13's 1-2 points).

CPU (between job checks):
A. Fill missing graph parts from existing data (slide 11 frozen rank, slide 12 small-segment values).
B. Consistency red team: slides vs doc vs CLAIMS vs figure CSVs; fix and log.
C. Blind figure rounds (tools/talk_round.sh) and cold-read deck reviews; polish within the current structure.
D. "Why" analyses from existing logs for the open questions; add findings to slide notes / backup slides.
E. Appendix graphs from existing data (TTFT over time around a kill, per-survivor remap, restored memory per survivor).
F. One regenerate script: CSVs -> figures -> FINAL -> slide-number check.

## State
(appended by the loop)
- 01:00 RECOV2 8353506 + RESTART2 8353534 submitted (both PENDING); slide 11 frozen-rank panel added.
- 01:06 RECOV2 running (both groups loading first case); RESTART2 pending; HEALTHY128 presubmitted (token HEALTHY128-092401fe5b, REPL_JOB=1..3); numbers red-team agent running (redteam.wH87); tools/regen_figs.sh written, test run in progress.
- 01:10 Both slots running (RECOV2 8353506 first cases loading ~8 min; RESTART2 8353534 started). Added B13 (TTFT around a kill, CLAIMS 225, make_appendix.py in regen). regen_figs.sh verified deterministic (0 changes).
- 01:15 Ready for next slots: HEALTHY128 (REPL_JOB=1..3, token HEALTHY128-092401fe5b), STRESS-S1B (STRESS_RATE=51.0 bash launch/stress_s1_cmd.txt, token STRESS-S1B-6eb9e1b3cc, --time 02:00). Numbers auditor rerunning (redteam.wH87, first launch died).
- 01:20 RECOV2 online rank5pf-1 PASS (script resume 1.2 s, ratio 0.94); stop-1 offline at server_ready. RESTART2 servers loading. Numbers audit applied (28 fixes, logged).
- 01:26 RECOV2: online rank5pf-1 PASS 1.2 s; offline stop-1 PASS (confirm 30.1 s + 4.0 s, first offline frozen run). On RECOV2 done: make_figs.py STOP_FILES["offline"]=["KILL-FINAL-OFF-B-8353506"], UPD += "KILL-FINAL-ON-A-8353506" (online rank n=6), prefill rows -> slide 18 + doc prefill table; CLAIMS 226; regen_figs.sh. GPU queue after: HEALTHY128 j1-3, STRESS-S1B, then PAD-TEST. Persona deck reviews running (personas.*).
- 01:38 RECOV2 3/7 PASS (offline rank5pf 1.53 s); RESTART2 no-EPLB run1 down 416 s (weights 186 s). EXACT presubmitted (token EXACT-5002f99253; job.sh EXACT-FINAL labels -B/-C/-D added, backup TRASH); queued next. B14 Q&A slide added from persona reviews.
- 01:45 RECOV2 6/7 so far PASS; RESTART2 no-EPLB 3 runs, online 2 so far. Slide 13 'why below proportional' diagnostic (CLAIMS 226) added to notes.
- 01:45 correction to the line above: RECOV2 4 cases PASS, 2 running (offline stop-2, online rank5upd-1), 1 queued (no-EPLB rank5pf).
- 01:51 RECOV2 5 PASS (offline stop-2 29.5+3.6 s), 2 running; RESTART2 no-EPLB 3/3 done (416/369/310 s), online+offline running. Blind round 10 running. make_figs.py handles offline FROZEN once STOP_FILES has it.
- 02:13 RECOV2 8353506 done 7/7 PASS -> CLAIMS 227, figures regenerated, slides 5/9/11/13/18 + doc + summary updated, logged. EXACT 8354988 submitted in the freed slot (running). RESTART2: online 3/4 (693/580/663 s), offline 2/3 (334/325 s). Blind round 10: ours 6.73 vs refs 5.08 charts, 7.00 vs 4.60 design.
- 02:31 RESTART2 8353534 done (10/10 runs) -> CLAIMS 228, slide 1 + Fig A4 + doc pooled (n=6/7/7; 5.7/6.2/10.4 min). HEALTHY128 REPL_JOB=1 submitted 8355466. EXACT 8354988 running (reference servers loading). New: tools/restart_phases.py; regen order make_fig_today before make_figs_extra.
- 02:36 EXACT 8354988 at the kill phase (no EPLB); HEALTHY128-1 8355466 running (online FT first). CLAIMS 229 (prefill-kill client dip, offline stop) -> slide 18 notes + doc. Red team #2 (redteam2.Jkqn) and blind round 11 running. HEALTHY128 readout plan: tools/finals_stats.py --loads 128 on the new jobs, pool with v5f_pairs_full.csv 128 rows (as CLAIMS 206 did at 32).
- 02:50 EXACT 8354988 done: FAIL/cannot decide (std no-EPLB reference not deterministic; online start assertion from the oracle exclusion; offline not run) -> CLAIMS 230, slides 3/B9/B14 + doc reworded. HEALTHY128-2 8355979 submitted in the freed slot. Red team #2 fixes applied (restart headline -> 6-10 min on average, decided).
- 02:54 EXACT-SMALL prepared (launch/exact_small_cmd.txt: config-small-mtp-overlap, victim 30, n=2, rules before data; job.sh EXACT_VICTIM harness-only change, backup in TRASH); dry run PASS; presubmit token EXACT-SMALL-dccb805c6e; queued next GPU slot (before HEALTHY128-3 and STRESS-S1B).
- 03:08 HEALTHY128-1 8355466 and -2 8355979 running (first pairs done, rc=1 normal, state DONE); EXACT-SMALL 8356173 pending. Slide 21 notes: tonight's findings as next-step candidates. Deck cold-read #3 running (deck3.RlNi). PAD-TEST deferred: STD cost cases refuse extra CUDA graph sizes (harness preflight), so it needs a harness change; lower EV than the queued jobs.
- 03:16 Deck cold-read #3 applied (wording only, logged); Fig 2 restored-memory drawn in free GPU memory; morning summary section added to FINAL/OVERNIGHT_LOG.md; blind round 12 running. Jobs: HEALTHY128-1/-2 on pair 2; EXACT-SMALL pending.
- 03:20 HEALTHY128 readout pipeline validated on the original n=4: tools/finals_stats.py --loads 128 --appendix-loads '' --csv (reproduces CLAIMS 194 128/GPU cells) and new tools/pair_latency_ci.py --load 128 (reproduces v5f_tpot_ttft.txt 128 rows exactly). Pooled = original dirs STD*-8269103/8270779/8271451 + new STD*-8355466/8355979/<job3>; make_figs: H[128], TRIM[128], CI[128] to the pooled files. Blind round 12: judge mis-keyed several rows (descriptions of other images) -> treat as noise; stop blind rounds unless a figure changes a lot.
- 03:46 HEALTHY128-1 8355466 done 03:33 (4/4 pairs valid per finals_stats; no readout until all 3 jobs). EXACT-SMALL 8356173 PENDING on QOSGrpNodeLimit (short QOS busy with other teams' large jobs); HEALTHY128-2 running. Slide 18: optional recovery-by-situation chart (fig7c CSV + other_situations_talk), logged.
- 04:00 Only one of our 16-node jobs runs at a time now (short QOS group node limit; others' large jobs). Queue: EXACT-SMALL 8356173 (est start 04:33), HEALTHY128-3 8357606 submitted (pending); STRESS-S1B after. Doc Fig 7b online remap range updated (0.50-0.55 s).
- 04:34 HEALTHY128-2 8355979 done 04:10 (4/4 pairs valid). EXACT-SMALL 8356173 and HEALTHY128-3 8357606 PENDING (Priority), est start ~04:35.
- 04:47 EXACT-SMALL 8356173 started 04:41; both groups' references starting with config-small-mtp-overlap and oracle rank 30 (verified in env/program line). HEALTHY128-3 pending (QOS). FINAL/DATA.md got a staleness note; FINAL/README lists OVERNIGHT_LOG.md.
- 04:54 EXACT-SMALL: both references deterministic on config-small (state PROBE, deterministic=true); pert running, kill next (~05:10).
- 05:11 EXACT-SMALL done: all token checks PASS in 2/2 (exact 400 tokens after the kill), rule verdict FAIL on the placement proof (no print_iter_log in config-small) -> CLAIMS 231, notes/doc updated. STRESS-S1B 8359063 submitted (pending); HEALTHY128-3 8357606 pending.
- 05:23 Running: HEALTHY128-3 8357606 (started 04:47, ends ~06:15), STRESS-S1B 8359063 (started 05:11, ends ~07:10). Fig 7 CSV restart rows now pooled (n=20); regen runs make_fig_today first. DATA.md blocks updated (recreate checks pass). Slide 1 notes: build provenance corrected (15 of 20 restarts on the v5-final-equivalent FT-off wheel).
- 05:50 EXACT-SMALL2 8360292 submitted (pending): same as EXACT-SMALL with print_iter_log on (new harness config tokens/config-small-mtp-overlap-iterlog.yaml) so the placement proof is readable; rule unchanged; decided after 8356173 (disclosed in the launch header). Dry run PASS, token EXACT-SMALL2-6d6f1b9756.
- 06:19 HEALTHY128 pooled (CLAIMS 232) -> slides 15/16, doc, summary, notes; make_figs bench glob widened to 83xxxxx jobs (it silently matched only 82xxxxx). Running: STRESS-S1B 8359063, EXACT-SMALL2 8360292.
- 06:21 RECOV3 8361282 submitted (pending): offline twokill x2 (first v5-final offline two-loss cells), online rank5upd x2, no-EPLB rank5pf x1; rules before data in launch/recov3_cmd.txt; dry run PASS; token RECOV3-a3fd604cd2.
- 06:31 EXACT-SMALL2 done: 4/4 exact overall; placement check is a harness bug (cumulative counter) -> CLAIMS 233; notes/doc/summary updated. Waiting: STRESS-S1B 8359063, RECOV3 8361282 (pending).
- 06:34 RECOV3 8361282 running since 06:28 (offline twokill on std-offline-ft-filled, online rank5upd on std-online: verified). STRESS-S1B ends ~07:10.
- 07:16 S1B done -> CLAIMS 234, Fig 9/slide 17 n=4 (generic n code in make_figs; same-step TPOT labelled in ms), doc/notes/summary updated. RECOV3: offline twokill x2 PASS, online rank5upd x2 PASS, no-EPLB rank5pf running.
- 07:20 RECOV3 done (5/5 PASS) -> CLAIMS 235, figures regenerated, slides 9/11/13/18 + doc tables + summary updated. All queued GPU work done; no job running.
- 07:23 DATA.md fully current (recovery, restart, healthy 128/GPU, realistic traffic blocks; recreate.py checks pass). Final consistency red team (redteam3) running. No GPU job running; queue empty (HEALTHY8 considered and skipped: one job gives n=1-2 per mode, too weak for a claim).
- 07:27 Final red team fixes applied and logged; CLAIMS 198/224 marked as updated by 228. Overnight plan complete: all queued GPU jobs done, no job running, loop stopped (no new GPU work started into the user's day without asking).

## Morning additions (user 10-27 ~07:40: decisions OK, may start new GPU jobs)
Rules, written before the data:
- EXACT-SMALL3 (launch/exact_small3_cmd.txt): EXACT-SMALL2 rerun after fixing the placement-check bug (per-iteration
  num_scheduled_requests); EXACT_FINAL rule intent unchanged; its own verdict reported as is; CLAIMS row; if PASS in 2/2,
  slide 3 / B9 / B14 notes say "token-exact on a deterministic small config (pre-set rule PASS)".
- STRESS-S1C (launch/stress_s1_cmd.txt, STRESS_RATE=51.0): 2 more realistic-traffic pairs, same readout rules; pool with
  S1 and S1B (n=6), report mean and 95% t-CI per metric (descriptive, no pass bar); slide 17 / Fig 9 update; CLAIMS row.
- 07:41 EXACT-SMALL3 and STRESS-S1C presubmitted (tokens EXACT-SMALL3-61780ca854, STRESS-S1C-f13eeed2e7); dry run PASS.
- 07:42 Submitted EXACT-SMALL3 8363137 and STRESS-S1C 8363138.
- 07:58 User 07:50: prioritize runs that change claims or graphs; more consistency / vocabulary checks. Done on CPU: slide 14 timeline + dip now over all 7 rank kills (CLAIMS 236; was 3 losses, no selection rule); vocabulary/style review applied (text only). GPU: RECOV4 8363511 submitted (offline rank/node n 3->5 for slides 11/13, frozen +1 per mode), rules in launch/recov4_cmd.txt. EXACT-SMALL3 8363137 (notes only) and STRESS-S1C 8363138 (slide 17) running.
- 08:31 EXACT-SMALL3 PASS 2/2 -> CLAIMS 237, notes/doc updated. RECOV4 8363511 running since 08:12 (configs verified: offline 288 slots, no EPLB 256). S1C 8363138 running (ends ~09:45).
- 08:39 Cross-document consistency fixes applied (doc, summary, presenter notes, slides header). Integration plan for RECOV4: make_figs KILL_FILES offline += OFF-B-8363511; STOP_FILES noeplb += OFF-8363511, online += ON-B-8363511, offline += OFF-A-8363511 (labels are dir names only); client_tput_after_kill.py on the 4 new offline decode kills, lines appended to review/diag/client_tput_after_kill.txt for Fig 8; DATA.md blocks via upd.py.
- 09:22 RECOV4 done (7/7) -> CLAIMS 238, 239; figures regenerated; slides 11/13/14/18/20 + doc + summary + notes updated. Offline rank-kill client median 95.4% -> 92.5% (two new kills 0.900, 0.916). S1C 8363138 running (ends ~09:45). GPU queue empty otherwise.
- 10:56 S1C 8363138 done (4/4 PASS) -> CLAIMS 240; slide 17 etc. now n=6 (p99 CIs include 0, one outlier run each way). Slide 2 rechecked vs sources (+Tarragon row); deck restructured per user (outline, step/section bars, 5b authority, 9b resume). GPU queue empty; no job running.

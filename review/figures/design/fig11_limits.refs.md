# fig11_limits: refs

Source: DESIGN_DOC_v5final_draft.md Sections 1, 7, 11, 12 (wording condensed, no new claims), plus review/FINALS_PLAN.md item 11 for
"untested failure types". Code refs as cited in the draft at 7e8dfb37df.

| Row | Draft source | Code ref (from the draft) | Status |
|---|---|---|---|
| Detection | Section 12 bullet 1; Section 7 last paragraph | CFG:73-94; hang_monitor.py:46-62, 145-164; hang_detector.py:444 | from draft; device-timeout clause from Section 7 |
| Second loss | Section 12 bullet 2; Section 8 "A further loss before Install" | PLR:1043-1046, 1263-1302 | verified (PLR lines read) |
| Not recoverable | Section 12 bullet 3 | AUT:440-458 (never rank 0); LB:1428-1436 (ring copy node) | verified |
| Out of FT's reach | Section 12 bullet 3 (host death, GPU loss, NVLink fabric, silent wrong results) | none (by design statement) | from draft, UNVERIFIED in code (behaviour depends on PRRTE) |
| No rejoin | Section 1 | none | from draft |
| Memory | Section 12 bullets 2 and 4 | PLR:1323-1340, 1426-1457 | verified |
| Spec. decoding | Section 12 bullet 6; Section 11 last paragraph | PLR:105-116, 465-475 | from draft, not re-read |
| Shutdown | Section 12 bullet 5 | mpi_session.py:141-190 (_FINALIZE_GRACE_S = 30.0) | verified |
| Platform | Section 12 bullet 7; Section 5 Online EPLB | LB:99-121, 1278-1297 | from draft, not re-read |
| Reload | Section 12 bullet 8 | LB:313-315 | from draft, not re-read |
| Scope (refused) | Section 11 table | PLR:416-463; COM:941-975; N1S:740-745 | N1S (fence path) and COM (ob1) verified; rest from draft |
| Footer | Section 12 last bullet placeholder; FINALS_PLAN item 11 | none | placeholder, filled from finals |

# fig10d_late_write_remap: refs

Refs are path:line at the draft's code of record 7e8dfb37df (verified by reading the lines). The frozen build 1117f7e0e7 differs only in
peer_loss_recovery.py (+1 line after 1088, about +19 after 1558) and mnnvl_memory.py (remap +5 lines); no change to what is drawn.
Abbreviations as in DESIGN_DOC_v5final_draft.md (PLR, LB, ES, N1S, KRN, GSM, MLB, MNV, COM, AUT, CFG, CLI, PYE).

| Element | Claim drawn | Ref | Status |
|---|---|---|---|
| Subtitle | failed rank's writes into pages it mapped, unbounded timing; pages never reused | PLR:1378-1386; MNV:938-945 | verified |
| Panel 1 | each rank's row imported by every rank (MNNVL workspace rows) | MNV:960-988 (rows per rank, rank_stride) | verified in substance; the "every rank imports every row" detail is from the MNNVL design, not re-read line by line |
| Panel 2 | mask clears departed ranks first; each survivor unmaps every row, releases handles, keeps addresses | ES:113-125; MNV:938-958 | verified |
| Panel 2 note | pages stay allocated while any importer maps them; late write lands in pages no survivor maps | MNV:942-943; ES:131 | verified |
| Panel 3 | barrier; room from KV if need be; fresh pages at same addresses among live ranks only; departed rows unmapped | PLR:1402-1413; MNV:960-1012; ES:127-137 | verified |
| Panel 3 note | all map or all raise | MNV:967-998 | verified |
| Panel 3 note | rows reset, barrier before any peer writes (ring staging) | PLR:1415-1418; ES:129 | verified |
| Panel 3 note | graphs replay on fresh pages; installed mask keeps collectives off row f | ES:36-39, 131-132 | verified |
| Afterwards note | old pages freed when f ends; KV returned at commit if free, else idle (<= 1/s) or later commit | PLR:96, 1426-1457, 1097 | verified |
| Afterwards note | survivor that cannot release/map ends the job; peer failure during remap exchanges re-plans | PLR:1390-1394; draft Section 9 | verified (docstring) |

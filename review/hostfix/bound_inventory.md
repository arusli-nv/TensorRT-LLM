# Bound-path inventory: per MoE layer per step work the bound path does that unbound/off does not (2026-10-22 06:45 PT)

Read-only subagent report on wt-recvfix a3d2440044 (contains the measured b0ad558bcb); spot-checked by the main session:
item 6 (FC1 `_source_slot_row`, a runtime Int32 divide per gathered row, gk:710-718/1707-1724) confirmed in code.
Same-run target: bound - unbound = +0.04/+0.10 ms (CLAIMS 167); adding route()'s ~0.12 (CLAIMS 164, other run) gives
0.16-0.22. Treat it as ~0.1-0.2 ms.

Carriers ranked:
1. FC1 in-place slot read on the slowest rank (fixed-slot stride s*8192+i: 32 regions ~29 MB apart vs ~7 MB packed;
   runtime divide per row). Models used device-0 medians; the step is max over 32 ranks. STRIDE32 changed off too and
   both strides exceed a page. Restructure: token-major interleave row = i*ep+s, or pass log2 and shift.
2. Serial fixed-latency chains untouched by A/B': per-target rank-mask loads after the dispatch wait in the last token
   CTA (a2a:918/934/963), a released_round load after fence.release.sys (a2a:957; SASS unverified), fence.acquire.sys
   per source after the last arrival (a2a:675). Restructure: hoist mask/round loads before the wait/fence.
3. Routing cluster resident early via the receive CTAs' PDL trigger at entry (a2a:650-651); the trigger move
   ("Change T") was never run.
Not carriers (evidence): lookup table (hidden since B'), abort polling (absent in steady state), compact copy (A: no
gain), source_valid_counts (1 store), per-signature regions (host only), fence release/combine (identical).
Unbound hook keeps the 320 slots and ghosts, so routing tier and FC2 extras are equal between ft and ft-unbound.

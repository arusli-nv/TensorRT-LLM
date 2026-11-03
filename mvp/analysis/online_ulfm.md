# ONLINE-ULFM readout (STD-8140516 + STD-G1-8140516; CLAIMS 157). Default pairing off -> ft; rerun with CTX_OFF_SUFFIX/CTX_ON_SUFFIX for off -> ulfm and ulfm -> ft.

groups per cell: {'online/32': ['STD-8140516/r1', 'STD-G1-8140516/r1'], 'online/128': ['STD-8140516/r1', 'STD-G1-8140516/r1']}

### 1. Context (prefill-containing) steps, rank-0 host step ms, FT off -> on

| mode | load | group | n off/on | median ms off/on | median cost % | mean ms off/on | mean cost % | wall share % off/on |
|---|---|---|---|---|---|---|---|---|
| online | 32 | STD-8140516/r1 | 106/109 | 378.0/370.2 | -2.06 | 529.9/524.8 | -0.97 | 29.6/29.6 |
| online | 32 | STD-G1-8140516/r1 | 106/107 | 366.3/407.1 | +11.12 | 525.0/531.4 | +1.22 | 28.7/29.6 |
| **online** | **32** | **mean / min..max of cost** | +0.13 | -0.97..+1.22 | | | | |
| online | 128 | STD-8140516/r1 | 142/144 | 478.0/486.1 | +1.69 | 582.1/579.1 | -0.50 | 43.3/43.0 |
| online | 128 | STD-G1-8140516/r1 | 146/141 | 467.1/494.9 | +5.95 | 555.5/592.9 | +6.74 | 42.2/43.3 |
| **online** | **128** | **mean / min..max of cost** | +3.12 | -0.50..+6.74 | | | | |

### 2. Stall share: % of decode full-load step time in steps > 2x median, FT off -> on

| mode | load | group | stall % off | stall % on | diff pp | 180-210 ms steps/min off/on | trains/min off/on |
|---|---|---|---|---|---|---|---|
| online | 32 | STD-8140516/r1 | 1.93 | 3.17 | +1.24 | 1.58/3.42 | 1.58/3.42 |
| online | 32 | STD-G1-8140516/r1 | 5.71 | 2.74 | -2.96 | 5.26/2.81 | 5.26/2.50 |
| **online** | **32** | **mean / min..max of cost** | -0.86 | -2.96..+1.24 | | | |
| online | 128 | STD-8140516/r1 | 1.10 | 4.20 | +3.10 | 1.88/2.47 | 1.88/2.47 |
| online | 128 | STD-G1-8140516/r1 | 3.58 | 2.23 | -1.35 | 2.50/1.86 | 2.50/1.24 |
| **online** | **128** | **mean / min..max of cost** | +0.87 | -1.35..+3.10 | | | |

### 3. Whole-window throughput: 32 x rank-0 generated tokens / window wall s, FT off -> on

| mode | load | group | tok/s off | tok/s on | cost % | client out tok/s off/on | client cost % |
|---|---|---|---|---|---|---|---|
| online | 32 | STD-8140516/r1 | 83068 | 81824 | +1.50 | 93965/92785 | +1.26 |
| online | 32 | STD-G1-8140516/r1 | 81541 | 82211 | -0.82 | 92031/93130 | -1.19 |
| **online** | **32** | **mean / min..max of cost** | +0.34 | -0.82..+1.50 | | | |
| online | 128 | STD-8140516/r1 | 154088 | 151831 | +1.46 | 174596/172117 | +1.42 |
| online | 128 | STD-G1-8140516/r1 | 153338 | 152487 | +0.55 | 173978/173136 | +0.48 |
| **online** | **128** | **mean / min..max of cost** | +1.01 | +0.55..+1.46 | | | |

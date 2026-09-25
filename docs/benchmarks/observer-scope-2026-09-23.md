# Two-project observer fixture after observer-aware reconciliation (2026-09-23)

The run of record after observer-aware reconciliation: both arms, twenty seeds, fourteen days, `single` store mode, the real embedding model, zero LLM calls. The write-up and the comparison with the first run are in [`observer-scope.md`](observer-scope.md).

## Arm `lines`

Seeds 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 · 14 days · arm `lines` · store mode `single` · generator v1 · zero LLM calls by construction.

| Measure | Rate |
|---|---|
| Own lines in view for their project | 100.0% (4273 / 4273) |
| … the same lines ACTIVE anywhere (no lens) | 100.0% (4273 / 4273) |
| Own lines retired by the other project's activity | 0.0% (0 / 4273) |
| Winner in view after a cross-project supersession | n/a (0 eligible) |
| Lines in view a project never stated (lens leak) | 0.0% (0 / 4273) |
| In view, `fact` lines | 100.0% (2240 / 2240) |
| In view, `rule` lines | 100.0% (1473 / 1473) |
| In view, `own` lines | 100.0% (560 / 560) |

**Why an own line was not in view** (counts over own-line checkpoints):

- none

**What the write path did**, each retirement attributed by the retired claim's scope just before the extraction that retired it:

| Measure | Count |
|---|---|
| Own-value supersessions (`SUPERSEDED_BY_UPDATE`) | 63 |
| Cross-project supersessions | 0 |
| Own generation-cascade retirements | 104 |
| Cross-project cascade retirements | 0 |
| Candidates born superseded (rung 2.5 mirror) | 0 |
| Pairs declined by the observer precondition | 936 |
| `CONTRADICTS` relations (`OBSERVER_DIVERGENCE`) | 157 |
| Declined pairs with no recorded relation | 0 |
| Global line contested by a project → held in review | 100.0% (40 / 40) |

| Seed | Deposits | Own in view | Store-wide | Causes |
|---|---|---|---|---|
| 1 | 15 | 100.0% (211 / 211) | 100.0% (211 / 211) | — |
| 2 | 14 | 100.0% (232 / 232) | 100.0% (232 / 232) | — |
| 3 | 15 | 100.0% (213 / 213) | 100.0% (213 / 213) | — |
| 4 | 12 | 100.0% (163 / 163) | 100.0% (163 / 163) | — |
| 5 | 14 | 100.0% (198 / 198) | 100.0% (198 / 198) | — |
| 6 | 17 | 100.0% (195 / 195) | 100.0% (195 / 195) | — |
| 7 | 17 | 100.0% (194 / 194) | 100.0% (194 / 194) | — |
| 8 | 13 | 100.0% (214 / 214) | 100.0% (214 / 214) | — |
| 9 | 20 | 100.0% (214 / 214) | 100.0% (214 / 214) | — |
| 10 | 17 | 100.0% (212 / 212) | 100.0% (212 / 212) | — |
| 11 | 13 | 100.0% (197 / 197) | 100.0% (197 / 197) | — |
| 12 | 20 | 100.0% (240 / 240) | 100.0% (240 / 240) | — |
| 13 | 19 | 100.0% (234 / 234) | 100.0% (234 / 234) | — |
| 14 | 18 | 100.0% (240 / 240) | 100.0% (240 / 240) | — |
| 15 | 14 | 100.0% (218 / 218) | 100.0% (218 / 218) | — |
| 16 | 16 | 100.0% (220 / 220) | 100.0% (220 / 220) | — |
| 17 | 16 | 100.0% (205 / 205) | 100.0% (205 / 205) | — |
| 18 | 13 | 100.0% (208 / 208) | 100.0% (208 / 208) | — |
| 19 | 21 | 100.0% (242 / 242) | 100.0% (242 / 242) | — |
| 20 | 16 | 100.0% (223 / 223) | 100.0% (223 / 223) | — |

## Arm `chunked`

Seeds 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 · 14 days · arm `chunked` · store mode `single` · generator v1 · zero LLM calls by construction.

| Measure | Rate |
|---|---|
| Own lines in view for their project | 100.0% (4273 / 4273) |
| … the same lines ACTIVE anywhere (no lens) | 100.0% (4273 / 4273) |
| Own lines retired by the other project's activity | 0.0% (0 / 4273) |
| Winner in view after a cross-project supersession | n/a (0 eligible) |
| Lines in view a project never stated (lens leak) | 0.0% (0 / 4273) |
| In view, `fact` lines | 100.0% (2240 / 2240) |
| In view, `rule` lines | 100.0% (1473 / 1473) |
| In view, `own` lines | 100.0% (560 / 560) |

**Why an own line was not in view** (counts over own-line checkpoints):

- none

**What the write path did**, each retirement attributed by the retired claim's scope just before the extraction that retired it:

| Measure | Count |
|---|---|
| Own-value supersessions (`SUPERSEDED_BY_UPDATE`) | 63 |
| Cross-project supersessions | 0 |
| Own generation-cascade retirements | 104 |
| Cross-project cascade retirements | 0 |
| Candidates born superseded (rung 2.5 mirror) | 0 |
| Pairs declined by the observer precondition | 262 |
| `CONTRADICTS` relations (`OBSERVER_DIVERGENCE`) | 157 |
| Declined pairs with no recorded relation | 0 |
| Global line contested by a project → held in review | 100.0% (40 / 40) |
| Chunks carried forward / all chunks | 926 / 1432 |

| Seed | Deposits | Own in view | Store-wide | Causes |
|---|---|---|---|---|
| 1 | 15 | 100.0% (211 / 211) | 100.0% (211 / 211) | — |
| 2 | 14 | 100.0% (232 / 232) | 100.0% (232 / 232) | — |
| 3 | 15 | 100.0% (213 / 213) | 100.0% (213 / 213) | — |
| 4 | 12 | 100.0% (163 / 163) | 100.0% (163 / 163) | — |
| 5 | 14 | 100.0% (198 / 198) | 100.0% (198 / 198) | — |
| 6 | 17 | 100.0% (195 / 195) | 100.0% (195 / 195) | — |
| 7 | 17 | 100.0% (194 / 194) | 100.0% (194 / 194) | — |
| 8 | 13 | 100.0% (214 / 214) | 100.0% (214 / 214) | — |
| 9 | 20 | 100.0% (214 / 214) | 100.0% (214 / 214) | — |
| 10 | 17 | 100.0% (212 / 212) | 100.0% (212 / 212) | — |
| 11 | 13 | 100.0% (197 / 197) | 100.0% (197 / 197) | — |
| 12 | 20 | 100.0% (240 / 240) | 100.0% (240 / 240) | — |
| 13 | 19 | 100.0% (234 / 234) | 100.0% (234 / 234) | — |
| 14 | 18 | 100.0% (240 / 240) | 100.0% (240 / 240) | — |
| 15 | 14 | 100.0% (218 / 218) | 100.0% (218 / 218) | — |
| 16 | 16 | 100.0% (220 / 220) | 100.0% (220 / 220) | — |
| 17 | 16 | 100.0% (205 / 205) | 100.0% (205 / 205) | — |
| 18 | 13 | 100.0% (208 / 208) | 100.0% (208 / 208) | — |
| 19 | 21 | 100.0% (242 / 242) | 100.0% (242 / 242) | — |
| 20 | 16 | 100.0% (223 / 223) | 100.0% (223 / 223) | — |

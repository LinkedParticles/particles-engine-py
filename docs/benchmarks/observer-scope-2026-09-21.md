# Two-project observer fixture (gate B)

Seeds 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 · 14 days · store mode `single` · generator v1 · zero LLM calls by construction.

| Measure | Rate |
|---|---|
| Own lines in view for their project | 79.4% (3393 / 4273) |
| … the same lines ACTIVE anywhere (no lens) | 80.2% (3428 / 4273) |
| Own lines retired by the other project's activity | 20.6% (880 / 4273) |
| Winner in view after a cross-project supersession | 0.0% (0 / 810) |
| Lines in view a project never stated (lens leak) | 0.0% (0 / 3393) |
| In view — `fact` lines | 62.4% (1397 / 2240) |
| In view — `rule` lines | 97.5% (1436 / 1473) |
| In view — `own` lines | 100.0% (560 / 560) |

**Why an own line was not in view** (counts over own-line checkpoints):

- `superseded_by_update`: 810
- `cascade`: 35
- `active_elsewhere`: 35

| Seed | Deposits | Own in view | Store-wide | Causes |
|---|---|---|---|---|
| 1 | 15 | 72.5% (153 / 211) | 72.5% (153 / 211) | cascade 2, superseded_by_update 56 |
| 2 | 14 | 74.1% (172 / 232) | 76.3% (177 / 232) | active_elsewhere 5, cascade 4, superseded_by_update 51 |
| 3 | 15 | 81.7% (174 / 213) | 82.6% (176 / 213) | active_elsewhere 2, cascade 1, superseded_by_update 36 |
| 4 | 12 | 68.1% (111 / 163) | 68.1% (111 / 163) | superseded_by_update 52 |
| 5 | 14 | 77.8% (154 / 198) | 77.8% (154 / 198) | cascade 1, superseded_by_update 43 |
| 6 | 17 | 77.9% (152 / 195) | 78.5% (153 / 195) | active_elsewhere 1, superseded_by_update 42 |
| 7 | 17 | 84.5% (164 / 194) | 86.6% (168 / 194) | active_elsewhere 4, superseded_by_update 26 |
| 8 | 13 | 79.4% (170 / 214) | 79.4% (170 / 214) | superseded_by_update 44 |
| 9 | 20 | 76.6% (164 / 214) | 78.0% (167 / 214) | active_elsewhere 3, cascade 1, superseded_by_update 46 |
| 10 | 17 | 73.1% (155 / 212) | 73.1% (155 / 212) | cascade 2, superseded_by_update 55 |
| 11 | 13 | 75.1% (148 / 197) | 77.2% (152 / 197) | active_elsewhere 4, superseded_by_update 45 |
| 12 | 20 | 81.2% (195 / 240) | 81.2% (195 / 240) | cascade 3, superseded_by_update 42 |
| 13 | 19 | 85.5% (200 / 234) | 88.5% (207 / 234) | active_elsewhere 7, cascade 1, superseded_by_update 26 |
| 14 | 18 | 83.3% (200 / 240) | 83.8% (201 / 240) | active_elsewhere 1, cascade 5, superseded_by_update 34 |
| 15 | 14 | 84.4% (184 / 218) | 84.9% (185 / 218) | active_elsewhere 1, cascade 5, superseded_by_update 28 |
| 16 | 16 | 80.9% (178 / 220) | 83.2% (183 / 220) | active_elsewhere 5, cascade 4, superseded_by_update 33 |
| 17 | 16 | 75.6% (155 / 205) | 75.6% (155 / 205) | cascade 5, superseded_by_update 45 |
| 18 | 13 | 84.1% (175 / 208) | 84.1% (175 / 208) | superseded_by_update 33 |
| 19 | 21 | 79.8% (193 / 242) | 80.2% (194 / 242) | active_elsewhere 1, cascade 1, superseded_by_update 47 |
| 20 | 16 | 87.9% (196 / 223) | 88.3% (197 / 223) | active_elsewhere 1, superseded_by_update 26 |

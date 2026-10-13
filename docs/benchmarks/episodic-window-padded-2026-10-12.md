# Episodic window questions on a padded store (2026-10-12)

A re-run of the [first episodic-window measurement](episodic-window-2026-10-12.md) at a realistic store size. The first run's store held 30 active beliefs, fewer than the default `top_k` of 40, so every query handed the answer model the whole store. This run adds about 300 unrelated beliefs so that the default budget has to choose. The per-question answers, verdicts and retrieval ranks are in [`episodic-window-padded-2026-10-12.json`](episodic-window-padded-2026-10-12.json).

## Setup

- **Fixture:** the same eight sessions of `tests/fixtures/trip_vs_move/` as the first run (04, 11, 12-moved, 13 to 17), audited first into a fresh scratch store.
- **Padding:** a second `particles audit` over 60 files. Two are `tests/fixtures/ledgerly/` and three are `tests/fixtures/memory_dir/`, worth about 20 beliefs between them. These are the only checked-in session-style fixtures that do not restate the trip case: `tests/fixtures/moved_city/` has the same user moving from Lajpat Nagar to Bandra, which would contradict the fixture, and `reanchor/` and `update_probe/` hold claim pairs rather than sessions. The other 55 are sessions written by `scripts/episodic_window_padding.py`: 220 seeded facts about the same user (books, cooking, work, travel, errands), dated 2025-01 to 2026-09, which never name the fixture's places or people and never state a home, a hotel, a move or a diet. No interchange export of a fixture-built store is checked in, so none was used.
- **Store:** `particles memory consolidate --scope store` over the whole store, as in the first run. The result holds **335 active beliefs: 34 from the fixture and 301 distractors.**
- **Questions and arms:** `window_questions.json`, unchanged, through `scripts/episodic_window_probe.py` with the first run's arms, plus its `--extra` mode for full-store ranks and answers at `top_k` 10. The requested extra arm, `top_k` 40 with the dates spelled out, is the probe's agent two-step arm, so it needed no new code.
- **Model and spend:** every purpose on `claude-sonnet-5-5`, as in the first run. US$2.96 at list price, owner-approved.

**Observer scope was checked, not assumed.** All 68 corpus entries carry the same tags, `claude-code` and `memory-file`, and none carries a `project:` tag, so fixture and distractors are equally unattributed. The probe queries with `all_projects` set and the scratch store binds no project, so the observer filter is never engaged. The full-store ranking returns all 335 active beliefs as candidates for every question, which confirms that no distractor was filtered out.

## Result

**On the padded store the default query fails one window question of six.** The first-weekend question returns "the knowledge base doesn't say directly". Its three gold claims rank 45, 92 and 131 of 335, all outside the default budget of 40, while the first-day-at-office claim ranks 2nd because it shares the word "first". The other five window questions and all six controls stay correct.

**The failure is retrieval, not date arithmetic.** In every answer that received the dated claims, the answer model placed them correctly, including working out that 2026-10-10 was a Saturday in the failing answer itself. Rewriting the question with the dates (the agent two-step) lifts the keys claim from rank 92 to 1 and the Carter Road walk from 45 to 4, but the Gauri's Thali find stays at 42, two places outside the budget, so that answer is partial.

**At `top_k` 10 the store size makes things worse than in the first run.** Only 1 of 6 is correct, against 2 of 6 before. The question about where the user lived on 2026-10-01 now fails outright, because the ten move-week claims outrank the Lajpat Nagar claim (rank 17).

| Arm (6 window questions) | This run: correct | Partial | Wrong or empty | First run: correct | Partial | Wrong or empty |
|---|---|---|---|---|---|---|
| Default query, `top_k` 40 | 5 | 0 | 1 | 6 | 0 | 0 |
| `as_of` | 0 | 0 | 6 | 0 | 0 | 6 |
| `include_non_asserted` | 5 | 0 | 1 | 6 | 0 | 0 |
| `recency_window_days` 7 | 5 | 0 | 1 | 6 | 0 | 0 |
| `subject_id` Mumbai | 1 | 1 | 4 | 1 | 1 | 4 |
| Agent two-step (`top_k` 40, dates spelled out) | 5 | 1 | 0 | 6 | 0 | 0 |
| `top_k` 10 | 1 | 3 | 2 | 2 | 3 | 1 |
| `top_k` 10, agent two-step | 3 | 3 | 0 | 5 | 1 | 0 |

The first run's store held 30 active beliefs; this one holds 335. The knobs behave as before: `as_of` is assertion time and cannot express a world-time window, `include_non_asserted` and `recency_window_days` change nothing, and `subject_id` Mumbai misses the events linked to Pali Hill, Gauri's Thali and the office.

## Per question: gold ranks and verdicts

Ranks are positions among all 335 active beliefs for the plain question, then for the rewritten one (the agent two-step's wording), from the query path's own ranking with no answer model. A rank of 40 or better is inside the default retrieval set. Out-of-window claims a correct answer must not present as in scope are listed after the gold claims.

**What happened in the week after the user moved to Mumbai?** Default: correct. Two-step: correct.

| Claim | Plain | Rewritten |
|---|---|---|
| Keys to the Pali Hill flat (10-10) | 27 | 3 |
| Carter Road walk (10-11) | 29 | 4 |
| Found Gauri's Thali (10-11) | 80 | 70 |
| First day at the office (10-12) | 3 | 5 |
| Ananya's visit (10-14) | 17 | 20 |
| Dinner at Gauri's (10-14) | 31 | 13 |
| Movers delivered (10-15) | 10 | 18 |
| Out of window: Powai internship (2019) | 35 | 94 |

All five events reached the answer, with Gauri's present as the dinner venue, as the first run's rules allow. The Powai claim sat inside the default set and was not presented as in the window.

**Where was the user staying on 2026-10-08?** Default: correct. Two-step: correct. The hotel-in-Bandra claim ranks 11th plain and 3rd rewritten. At `top_k` 10 it falls one place outside the set and the answer loses the hotel's location.

**What did the user do on their first weekend in Mumbai?** Default: **wrong**. Two-step: partial.

| Claim | Plain | Rewritten |
|---|---|---|
| Keys to the Pali Hill flat (10-10) | 92 | 1 |
| Carter Road walk (10-11) | 45 | 4 |
| Found Gauri's Thali (10-11) | 131 | 42 |
| Out of window: first day at the office (10-12) | 2 | 5 |
| Out of window: Ananya's visit (10-14) | 48 | 20 |

**Who visited the user during their first week in Mumbai?** Default: correct. Two-step: correct. Ananya's visit ranks 10th plain and 8th rewritten.

**Where was the user living on 2026-10-01?** Default: correct. Two-step: correct.

| Claim | Plain | Rewritten |
|---|---|---|
| The user lives in Lajpat Nagar, Delhi | 17 | 4 |
| The user's home is in Delhi | 15 | 3 |
| Sandeep's Curry House is in Lajpat Nagar | 322 | 84 |

**What did the user do between 2026-10-12 and 2026-10-14?** Default: correct. Two-step: correct.

| Claim | Plain | Rewritten |
|---|---|---|
| First day at the office (10-12) | 4 | 5 |
| Dinner with Ananya at Gauri's (10-14) | 13 | 11 |
| Out of window: keys (10-10) | 3 | 2 |
| Out of window: movers (10-15) | 18 | 19 |
| Out of window: Powai internship (2019) | 111 | 133 |

The two out-of-window claims rank above the dinner, and the default answer named both as outside the window. At `top_k` 10 the dinner falls out of the set and the answer reports nothing for 10-14.

## Two differences from the first run's store

- **The move retired nothing.** The first run's consolidation retired the two Delhi residence claims. Here they stay active. The update pass probed 100 of 1,002 qualifying pairs, its configured cap, so the Delhi-to-Mumbai pair plausibly fell outside a budget the padding filled. The run record of that cycle does not state whether the cap was reached, so this is not confirmed. Since 1.185.1 the record and the consolidation summary state the qualifying count, the probed count, the cap, whether it was reached, and how many pairs it left, together with the order the pass probes in: content cosine, highest first, with ties in the subject index's order. The order is fixed rather than arbitrary, and a pair the cap leaves unprobed ranks below every probed pair; it also means a value update, whose wording differs in exactly the word that carries the value, tends to sit behind near-paraphrases of the same claim (a leverage ordering is proposed). Whether a given pair was probed is read from the probe-verdict ledger, keyed by each claim's content hash. It made the 2026-10-01 question easier, since the residence claim answers it directly, and it left the current-home control hedging over Delhi as an earlier statement, still correct.
- **The trip was again not retired.** The hotel and the return flight stayed active, as in the first run, so the retired-claim case was not exercised at all this time.

## What this does and does not show

- It shows a window question failing at the default budget on a store of realistic size, for the reason the first run predicted: retrieval ranks by the question's wording, and nothing orders or filters by the dates the claims carry.
- It shows that spelling out the dates mostly repairs retrieval, and that the answer model handles the date arithmetic once the claims arrive. The remaining two-step failure is a claim ranked 42nd.
- It does not show how often a window question fails in real use. One of six questions failed, on one fixture, with synthetic padding that is topically far from the fixture. A store whose distractors share the fixture's places and people would likely push more gold claims out.
- The verdicts are the drafting session's, under the first run's rules, against the same labels. The owner confirmed the labels and these verdicts on 2026-10-12. Each arm ran once, and the answer model sampled at its default temperature.

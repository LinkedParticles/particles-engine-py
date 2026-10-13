# Episodic window questions on the trip-vs-move fixture (2026-10-12)

A measurement before any design. It asks whether the query path answers a question about a world-time interval, such as "what happened in the week after the user moved to Mumbai?", and which existing knob helps. The per-question answers, verdicts and retrieval ranks are in [`episodic-window-2026-10-12.json`](episodic-window-2026-10-12.json).

## Setup

- **Fixture:** `tests/fixtures/trip_vs_move/` sessions 04, 11 and 12-moved (the trip case), plus five sessions dated 2026-10-10 to 2026-10-16. Session 16 states on 2026-10-14 a fact about 2019, and session 17 reports on Friday what happened on Thursday.
- **Questions:** `window_questions.json` beside the fixture. Six window questions and six controls (lookup, current state, relationship, exact term), labelled with gold answers and with facts a correct answer must leave out.
- **Run:** `particles audit`, then `particles memory consolidate --scope store`, into a fresh scratch store, every purpose on `claude-sonnet-5-5`. Then `scripts/episodic_window_probe.py`, which asks each question through the MCP query tool and the query operation.
- **Spend:** US$0.60 at list price.

## Result

**On this store the default query answers every window question correctly, and so does every control.** The window path does not exist, but nothing needed it here, for two reasons.

1. **The extractor resolved every date into the claim prose.** "Yesterday" in a Friday session became "on 2026-10-15", and the 2019 internship stayed dated 2019, so no answer placed it in the week after the move.
2. **The store is smaller than the retrieval budget.** It holds 30 active beliefs and the default `top_k` is 40, so retrieval hands the answer model the whole store. The answer model then does the date arithmetic itself, including working out that the first weekend was 10 and 11 October.

**With retrieval limited to 10, the window questions start to fail.** Only 2 of 6 stay correct. The first-weekend question returns "the knowledge base doesn't say", because the word "first" ranks the first-day-at-the-office claim at the top and the weekend's claims at 11, 19 and 21. The question about where the user stayed on 2026-10-08 loses the hotel's location, because the hotel claim ranks 16th.

**An agent's two-step recovers most of it.** Asking for the move date first, then re-asking with the dates spelled out, is correct on 6 of 6 at the default budget and 5 of 6 at a budget of 10. The anchor step itself was hedged: it gave 2026-10-09 as best supported but called the move a transition from 10-08 to 10-15.

| Arm (6 window questions) | Correct | Partial | Wrong or empty |
|---|---|---|---|
| Default query, `top_k` 40 | 6 | 0 | 0 |
| `as_of` | 0 | 0 | 6 |
| `include_non_asserted` | 6 | 0 | 0 |
| `recency_window_days` 7 | 6 | 0 | 0 |
| `subject_id` Mumbai | 1 | 1 | 4 |
| Agent two-step | 6 | 0 | 0 |
| `top_k` 10 | 2 | 3 | 1 |
| `top_k` 10, agent two-step | 5 | 1 | 0 |

## What each knob does to a window question

- **`as_of` cannot express world time.** It is assertion time, and every belief was asserted at the run instant, 2026-10-12, not at its session's date. A date before the run returns "the store held no beliefs"; a date after it is rejected as future. It failed all six.
- **`include_non_asserted` and `recency_window_days` change nothing.** The store holds no non-asserted claim, and a recency window cuts on the same run-instant `asserted_at`.
- **`subject_id` makes it worse.** The week's events are linked to Pali Hill, Gauri's Thali and the office, not to Mumbai, so filtering to Mumbai leaves only the trip, the hotel, the flight and the move.

## The retired claim

The brief expected a retired trip to hide what held during the window. On this model the move did not retire the trip, the hotel or the return flight (item 5), so that case was not exercised. Those claims stayed active, and the default answers hedge over the "flight back to Delhi" without getting the window wrong.

The move did retire the Delhi residence claims. The question about where the user lived on 2026-10-01 was still answered correctly, through the re-anchor restatement "as of 2026-09-04" and the movers' Delhi origin. The retired claims themselves never reached an answer: the query path reads active claims only, and no knob reaches a retired one.

## What this does and does not show

- It shows that the existing path can answer a window question **only when the window's claims are retrieved**, and that retrieval ranks them by the question's wording, not by their dates.
- It does not show a failure at the default budget. A store of 30 beliefs cannot show one, because it fits in a single retrieval.
- The partial and wrong answers at a budget of 10 are the evidence a later design should cite. The first-weekend case is the clearest one.
- The verdicts are the drafting session's. The owner confirmed them and the labels as proposed on 2026-10-12. Each arm ran once, and the answer model sampled at its default temperature.

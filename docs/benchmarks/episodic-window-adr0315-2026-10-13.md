# The window route on the episodic window questions (2026-10-13)

The acceptance runs, the world-time window (`particles query
--during` / `--on`). The two earlier measurements
([the first run](episodic-window-2026-10-12.md), [the padded
run](episodic-window-padded-2026-10-12.md)) asked the owner-confirmed window
questions of `tests/fixtures/trip_vs_move/window_questions.json` through
the knobs that existed and recorded the failure; this one asks them through
the route the record built, on freshly built stores. The per-question
hits, readings, coverage counts and answers are in
[`episodic-window-adr0315-2026-10-13.json`](episodic-window-adr0315-2026-10-13.json).

## Setup

- **Stores.** Each run audits sessions 04, 11, 12-moved and 13 to 17 of
  `tests/fixtures/trip_vs_move/` into a fresh scratch store, then its
  padding, then runs `particles memory consolidate --scope store`,
  `particles validity` (the prose-date backfill), and
  `scripts/episodic_window_probe.py --window`, which asks every window
  question with the window its anchor names (`WINDOW` in the script) and,
  for comparison, without one.
- **Item 1**, the padded store: the first measurement's padding (55
  seeded sessions plus the Ledgerly and memory-dir fixtures), every purpose
  on `claude-sonnet-5-5`. **Item 2**, the trip case on `claude-sonnet-4-6`,
  the model on which the move retires the trip. **Item 6**, the
  hard padding: 60 sessions of 240 facts that share the fixture's places and
  people, dated before the trip (`scripts/episodic_window_padding.py
  --hard`), on `claude-sonnet-5-5`.
- **Two rounds.** The first round ran the record as approved. It found the
  two point questions failing on every store for one reason, below, and the
  owner chose to amend §2.1 rather than activate with them failing; items
  1 and 2 were then rebuilt under the amended extractor (0.17.0 with the
  `as-of` anchor). Item 6 was not rebuilt; its first-round result stands as
  the prefilter measurement it was for.
- **Verdicts** are the implementing session's, against the confirmed
  labels, under the first run's rules. Each arm ran once at the model's
  default sampling.
- **Spend:** US$13.4 at list price over the five runs (3.49, 1.13, 4.45,
  3.37, 0.91), owner-approved in two tranches.

## Result

| Window question | Item 1, round 1 | Item 2, round 1 | Item 6, round 1 | Item 1, round 2 | Item 2, round 2 |
|---|---|---|---|---|---|
| w1 the week after the move | correct | correct | correct | correct | correct |
| w2 staying on 2026-10-08 | wrong (city only) | correct | wrong (city only) | wrong (city only) | correct |
| w3 the first weekend | correct | correct | correct | correct | correct |
| w4 who visited | correct | correct | correct | correct | correct |
| w5 living on 2026-10-01 | empty | empty | empty | correct | correct |
| w6 2026-10-12 to 14 | correct | correct | correct | correct | correct |
| **Correct** | **4 of 6** | **5 of 6** | **4 of 6** | **5 of 6** | **6 of 6** |

For reference, the same questions without a window on the same stores
(the `today` arm) were correct on 2, 3 and 3 of 6 in round 1 (w3 wrong on
all three; w6 presenting out-of-window events on two).

**The headline failure is fixed.** The first-weekend question, whose three
gold claims ranked 45, 92 and 131 of 335 on the padded store, is correct on
every store in every round: the window admits 4 of 26 dated claims on the
padded store and 5 of 15 on the hard padding, and the three gold claims are
among them. The Oct 12 to 14 question no longer presents the keys (10-10)
or the movers (10-15) as in the window, which the default path did on every
store. The prefilter does the work: on the hard padding it admits 12 of 275
candidates, and on item 2's point question it returns 5 or 7 claims rather
than 12.

**The retired-claim case (acceptance item 2) works as specified.** On
`claude-sonnet-4-6` the move retires the trip; "Where was the user staying
on 2026-10-08?" returns the hotel stay and the two retired trip claims, each
marked retired and closed, and the week-after-the-move listing does not
show them as in force. In round 2 the retired Delhi residence comes back
for 2026-10-01 on both models, closed at its retirement instant on
`claude-sonnet-5-5` (whose extraction gives the move no anchor) and through
its re-anchored restatement on `claude-sonnet-4-6`.

**Why the point questions failed in round 1, on every store.** The claims
they need are present-tense states with no date in the source: "I'm
staying at a hotel in Bandra" (said 2026-10-08), "I live in Lajpat Nagar"
(said 2026-09-04). The record as approved withheld a start from any claim
whose only cue was the session date, so they were undated, and an undated
claim is never placed. On `claude-sonnet-4-6` the extractor dated the hotel
stay anyway, which is why w2 passed there. The owner chose to let the
extractor, which reads tense, anchor a present-tense state at the reference
date under the `as-of` basis (held from then, never began), and to reject
the deterministic session-date fallback as before.

**Round 2.** Under the amended extractor, `claude-sonnet-4-6` anchored the
Delhi residence (`as-of`, cue "I live in Delhi, in Lajpat Nagar") and the
hotel stay (`stated`, cue "staying at a hotel in Bandra") and is 6 of 6.
`claude-sonnet-5-5` anchored the residence but not the hotel stay, so w2
still fails there: the rule is in the prompt and the model applied it to
one of two eligible claims. That is a compliance gap in one model, not a
rule gap, and the record activates with it stated.

## Three things the runs showed beyond the table

1. **The anchor is only as good as the reference date.** For a dated
   session file the reference date is the capture time unless the model
   reads the heading itself: on item 2 the residence's `as-of` anchor landed
   on the run day (2026-10-13), after the move, so it closed at once and
   w5 was answered through the restatement instead. On a real Claude Code
   harvest the reference date is the session's time and this does not
   arise; on a fixture file it is ground.
2. **The backfill's `as-of` cue is the weakest basis.** prose
   habit writes "As of D, the user finished reading X", an event phrased as a
   state, which then reads as held from D onward; and every standing state
   ("pays 349 rupees a month, as of 2025-06-07") is in force in every later
   window, so a point question on the padded store returned a dozen
   subscriptions as context beside its answer. The answers were right; the
   precision cost is real and recorded.
3. **The hard padding tested the undated exclusion, not the prefilter's
   discrimination.** Its 240 facts were said on dated sessions but carry no
   date in their text, so the extractor dated 12 of 275 claims and the
   window admitted only those. A padding that states its dates would be the
   harder test.

## Coverage

| Store | ACTIVE | dated by the extractor | dated by the backfill | total dated |
|---|---|---|---|---|
| Item 1, round 1 | 334 | 11 | 17 | 28 (8%) |
| Item 2, round 1 | 42 | 18 | 4 | 22 (52%) |
| Item 6, round 1 | 275 | 12 | 3 | 15 (5%) |
| Item 1, round 2 | 338 | 14 | 12 | 26 (8%) |
| Item 2, round 2 | 46 | 23 | 5 | 28 (61%) |

The fixture's own claims are the dated ones; the padding is mostly undated,
which is what the window's disclosure counts (`undated` on every response)
are for. On a migrated copy of the owner's store (acceptance item 3) the
backfill dated 736 of 48,213 ACTIVE claims (1.5%), 30 of which the owner
checked: 24 confirmed, four coarse `in <year>` anchors that should read as
events rather than states (the cue's kind was changed on that ruling), one
`since` that dated a report's coverage window, and one sentence holding two
claims; going forward the extractor dates what it extracts.

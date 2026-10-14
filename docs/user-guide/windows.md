# Asking about a period

*What happened to the user in the week after they moved?* A question like
that asks for no single fact. It asks for the claims whose subject matter
falls in an interval of the world's time, and the store can answer it only
if it knows when each claim is *about*, which is a different thing from when
it learned the claim. The window of `--during` and `--on` reads
that, and nothing else about the ordinary query path changes.

```bash
uv run particles query "What happened in the week after the move?" --during 2026-10-10..2026-10-16
uv run particles query --during 2026-10-10..2026-10-16
uv run particles query "Where was the user staying?" --on 2026-10-08
```

The bounds are ISO-8601 dates, inclusive; either side of `..` may be empty
(`2026-10-10..` is everything from that day on). The same parameter exists on
all three surfaces: the CLI flags, `QueryRequest.window` on `POST /query`,
and `during_start` / `during_end` / `on` on the MCP `query` tool.

## What a window selects

Every claim may carry two fields beside the `valid_until` it already had:
`valid_from`, when it begins to hold, and `valid_kind`, whether it is a
moment (`EVENT`: "picked up the keys on 2026-10-10", about that day and no
other) or a span (`STATE`: "lives in Pali Hill since 2026-10-10", holding
from then until it closes). A window admits a claim under one of two
readings, and the listing labels each:

- **began**: its start lies in the window, whatever its kind. This is what
  "what happened in the week" asks.
- **in force**: a span that started before the window and had not closed.
  This is the standing context: where the user lived, that they were on a
  trip.

An event before the window matches neither, so the store's dated past stays
out of the week after the move. Two further labels mark a weaker anchor:
**held at**, a span known to hold at a date with its start unknown ("as of
2026-10-08"), and **within**, a claim anchored to a month or a year and
no finer ("published in 1999"), which sits in any window its unit meets and
in no other. The point form `--on` admits an event on the date and a span whose
interval covers it.

With a question, the window cuts the candidate set **before** ranking, where
`--subject` cuts, so a dated claim competes only with the claims of its
period; ranking, the answer and every disclosure are then as usual. Without
a question the result is deterministic: the admitted claims in date order,
no embedding, no language model.

## Retired claims hold for their period

A claim the store has since retired was not false during the period it
names. A window read therefore admits a retired claim for that period,
marked with its retirement, and closes a retired span at the earliest of its
own end, the start of the claim that superseded it, and its retirement
instant. "Where was the user living on 2026-10-01?" returns the retired
Delhi residence, closed on the day of the move; the week after the move does
not show it as in force. When a retired claim's successor is in the window
too, the successor stands in for it and names it: a restatement reads better
than the original it restates. Outside a window every read is as it was.

## Where the dates come from

The extractor emits a start when the source states one ("since March",
"on 2026-10-10"). A present-tense state whose start the text does not give
("I'm staying at a hotel in Bandra") is anchored at the day it was said,
marked `as-of`: known to hold then, start unknown, read as held from then
on and never as having begun then. A past-tense claim with no date gets
nothing, since when it was said is not when it happened. For the claims
already in a store, `particles validity` reads the absolute dates the
extractor writes into claim prose and sets a default from a short list of
cues (`on`, `since`, `from`, `as of`, `in <year>`), at no model cost; a claim
with two dates, or a date behind no listed cue, is skipped and counted.
`particles particle date` sets one claim by hand. `particle show` prints the
interval and how it was set.

An undated claim is never placed in a window; the footer says how many the
read could not judge. Run `particles validity --dry-run` to see the
coverage.

## Resolving the anchor

"The week after the move" names a period by an event. The route takes
absolute dates, so resolve the anchor first and ask with dates:

```bash
uv run particles query "When did the user move to Mumbai?"
uv run particles query "What happened in the week after the move?" --during 2026-10-10..2026-10-16
```

The first answer may hedge, and that hedge belongs in the open, in your turn,
rather than inside a retrieval call. An agent over MCP does the same two
steps.

## Composing with as-of

`--during` selects by world time; `--as-of` keeps what the store believed at
T ([As-of time travel](as-of.md)). Together they answer "what did the store
believe on 2026-10-12 about the week of the move". Structural claim filters
intersect with the window as they do with every other filter.

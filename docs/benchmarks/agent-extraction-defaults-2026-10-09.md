# Agent-side extraction defaults: what local data says (2026-10-09)

**A dated snapshot, measured 2026-10-09.** Agent-side extraction
asks the in-session Claude Code agent to extract its own transcript at a Stop
hook. Its ask rule and its coverage rule each need defaults, and the record
requires every default to be measured before it is chosen. This page holds the
first two of the three measurements: an offline replay of past sessions and a
read-only store query. Neither makes a model call. The third, a shadow week,
needs the feature built.

Both come from `scripts/simulate_agent_extraction_asks.py`, which prints
aggregates only and never prints transcript or claim text.

## Verdict

- **The final stretch of a session is a small loss; the backlog is the large
  one.** Text left at the end of a session below the ask threshold is 1 to 5 %
  of all distilled text. What the rule mostly fails to cover is backlog: a
  long session grows faster between Stops than one ask's span can carry.
- **The binding constraint is how few Stops there are.** The median session
  has 3 Stops, because agentic turns are long. Each Stop is a chance to ask,
  so a per-ask span cap leaves large sessions behind whatever the interval.
- **A 15 000-character span cap leaves 18 to 31 % uncovered.** That cap is the
  general extractor's single-pass limit, which the record named as the
  starting point. At 40 000 characters with catch-up asks, 9 to 12 % is left.
- **An ask must produce a lot of output.** The general extractor yields a
  median of 4.1 claims per 1 000 characters of transcript. A 15 000-character
  span therefore expects about 60 claims, each with a verbatim quote, and a
  40 000-character span about 165. That is the agent's extra turn, paid for in
  output tokens on the user's plan.

## Corpus

| | Value |
|---|---|
| Transcripts (`~/.claude/projects/*/*.jsonl`, modified since 2026-09-01) | 423 |
| Active hours (gaps over 30 minutes counted as 30) | 730 |
| Distilled characters per session, median / p90 | 27 955 / 119 681 |
| Distilled characters, total | 23.4 M |
| Stops per session, median | 3 |

A Stop is taken to fire at the last assistant record before each typed user
prompt, and at the end of the file. Subagent records are excluded. The replay
assumes every ask covers its span, so the uncovered figures are lower bounds.

## The ask rule, replayed

No per-session cap. "Catch-up" waives the interval while more than one span
is owed. Token figures use 4 characters per token.

| `min_new_chars` | Interval | Span cap | Catch-up | Asks per session, median / p90 | Asks per active hour | Span p90 | ~Tokens of span text per ask, p90 | Uncovered | of which final stretch |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| 4 000 | 10 min | 15 000 | off | 2 / 8 | 2.18 | 15 000 | 3 750 | 24.3 % | 1.3 % |
| 4 000 | 10 min | 15 000 | on | 2 / 8 | 2.35 | 15 000 | 3 750 | 18.1 % | 1.4 % |
| 4 000 | 10 min | 40 000 | off | 2 / 7 | 2.10 | 30 419 | 7 605 | 12.1 % | 1.5 % |
| 4 000 | 10 min | 40 000 | on | 2 / 8 | 2.15 | 31 311 | 7 828 | 9.2 % | 1.5 % |
| 4 000 | 20 min | 40 000 | on | 2 / 6 | 1.85 | 36 120 | 9 030 | 10.8 % | 1.4 % |
| 8 000 | 10 min | 15 000 | off | 2 / 6 | 1.81 | 15 000 | 3 750 | 26.2 % | 4.3 % |
| 8 000 | 10 min | 15 000 | on | 2 / 7 | 1.95 | 15 000 | 3 750 | 20.3 % | 4.5 % |
| 8 000 | 10 min | 40 000 | off | 1 / 5 | 1.64 | 36 678 | 9 170 | 14.3 % | 4.6 % |
| 8 000 | 10 min | 40 000 | on | 1 / 6 | 1.67 | 38 868 | 9 717 | 11.3 % | 4.7 % |
| 8 000 | 20 min | 40 000 | on | 1 / 5 | 1.57 | 40 000 | 10 000 | 12.0 % | 4.4 % |

A cap of 6 asks per session raises the uncovered share to 35 to 51 % in every
combination tried, which confirms the record's choice of no cap on a keyless
install. With no span cap at all the uncovered share falls to 3 to 5 %, but
the largest single span is about 720 000 characters, which no single ask can
carry.

## Claim density on transcripts already in the store

For each COMPLETE snapshot of a `claude-code://session/` entry: the particles
the general extractor first observed on that snapshot, divided by the bytes it
extracted past the previous COMPLETE snapshot. Delta reads record
`extracted_through`; whole reads use the blob's size instead.

| Read | Snapshots | Thousand characters | Pooled claims per 1 000 characters | Per snapshot p10 / median / p90 |
|---|---:|---:|---:|---|
| Delta reads | 76 | 2 610 | 4.07 | 2.74 / 4.14 / 5.96 |
| Whole reads | 246 | 10 468 | 2.85 | 1.08 / 3.45 / 19.07 |

The delta reads are the cleaner measure, since they are exactly the quantity
an ask produces. The whole reads spread more widely, partly because a whole
re-read of a grown transcript is divided by the growth alone.

## What this means for the defaults

Proposed for the shadow week, to be checked against real asks there:

- **`min_claims_per_kchar` at 2.7**, the 10th percentile of the delta reads.
  A submission below it delivers fewer claims than 90 % of the background
  extractor's own deltas.
- **No per-session cap on a keyless install**, as the record already decides.
- **Catch-up asks on.** They cost little in ask frequency and recover a
  quarter to two fifths of the backlog at the 15 000 cap.
- **`min_new_chars` 4 000 to 8 000, interval 10 minutes, span cap 40 000.**
  This means about 1.7 to 2.2 asks per active hour and leaves 9 to 11 % of
  text uncovered. The remaining choice is between ask frequency and output per
  ask, and the shadow week is what shows which one users notice.

Two design questions follow that the record does not yet answer:

1. **Output per ask.** At about 4 claims per 1 000 characters, a 40 000
   character span asks the agent for around 160 claims in one turn. Whether a
   session model produces that reliably, and how long it takes, decides
   whether the span cap should be set by expected claim count rather than by
   characters.
2. **Covering long autonomous sessions.** With few Stops, any per-ask cap
   leaves a backlog in the largest sessions. Asking in several slices within
   one turn, or accepting the backlog on keyless installs, are the
   two directions.

## How to reproduce

```bash
uv run python scripts/simulate_agent_extraction_asks.py replay --since 2026-09-01 \
    --min-new-chars 4000,8000 --min-interval 600,1200 --max-span-chars 15000,40000
uv run python scripts/simulate_agent_extraction_asks.py density --db <store.db>
```

## The fitted calibration record

The isotonic fit for the `claude-code-agent` extractor on `anthropic:claude-sonnet-5-5` (fitted 2026-10-10, 162 claims on `prose-calibration-001`) is kept verbatim in [`claude-code-agent-calibration-sonnet-5-5-2026-10-09.json`](claude-code-agent-calibration-sonnet-5-5-2026-10-09.json). On the held-out `prose-article-seed-001` suite (266 claims) it reduced the expected calibration error from 0.112 to 0.022. The same fit for `claude-opus-5-5` was refused by the held-out guard (0.077 before, 0.126 after), so no Opus record exists. The record is stored here so it survives until it ships as package data, and so a store can be seeded with it without paying for the fit again.

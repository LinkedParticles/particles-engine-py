# The relevance floor, measured on real questions

A query against a store always retrieves *something*: top-k similarity search
has no notion of "nothing was close", so the query operation holds the best
retrieved cosine to a **relevance floor** (`query.relevance_floor`, default
0.25). Below it, the answer step is skipped and a server-built refusal is
returned, with the nearest beliefs still listed and the similarity disclosed.

Every gate like this has the same hazard: **a false negative is invisible.** A
question the store could have answered, refused, produces a confident
non-answer that nobody knows was answerable. Disclosure in the answer text is
not measurement. `particles benchmark relevance-floor` is the measurement.

It complements the [memory-rot benchmark](rot.md), which already sweeps the
same floor over *synthetic* probes for free. This page is the other half: real
questions, against a real store.

## What it does

**1. Harvest.** The product keeps no query log, so the only record of what a
person actually asked is their agent transcripts. `harvest` reads three
sources and tags each one, because they are not the same kind of evidence:

| Source | What it is |
|---|---|
| `mcp_query` | The question given to the MCP `query` tool |
| `cli_query` | `particles query "<question>"` run in a shell call (skipped when the command addresses a different store) |
| `user_prompt` | A question-shaped sentence the operator typed to their agent. A **proxy**: a real information need in the store's domain, but nobody addressed it to the store |

Secrets are redacted before a question is kept. The held-out set is private by
construction, so it is written outside any repository, and the verb refuses to
write it (or the JSON report, which carries question text) inside a git work
tree without an explicit flag. **The table on this page is the publishable
artifact**: the renderer prints counts and rates, never a question.

**2. Replay, free.** Every question goes through the query operation's ranking
half, and the maximum raw cosine over the rendered top-k is recorded, which is
exactly the quantity the floor reads. No LLM call. This alone gives the
**refusal curve**: the share of questions the gate would refuse at each floor
from 0.10 to 0.40.

**3. Answer and judge, paid and gated.** With `--judge`, each question runs
through the real query operation **with the gate switched off** (a floor of
0.0 is the documented off switch), so the product's own answer step runs over
the top-k the floor would have suppressed. A judge then labels the answer
*grounded* (supported by the retrieved beliefs) *and useful* (it resolves the
question asked), or not. From those labels the 2×2 table is swept over the
same floors, under both conditionings:

| Rate | Reads as |
|---|---|
| answerable → refused | Of the questions the store could answer, the share the gate refuses. The false-negative rate |
| unanswerable → passed | Of the questions it could not, the share the gate lets through to the answer step |
| refused that were answerable | Of what the gate refused, the share that was answerable. What a person reading a refusal wants to know |
| passed that were unanswerable | Of what the gate passed, the share that was not answerable |

An empty denominator prints `n/a`, never a number: a floor that refuses
nothing has no "share of refusals that were wrong". There is no aggregate
score.

The projected cost prints before any LLM call, priced from each question's
*actual* retrieved context; `--estimate` stops there.

## Read before comparing

* **The judge has no reference answer.** A real question has no gold answer,
  so "answerable" is a label from an LLM judge, not ground truth. It is the
  same judge model the [agent-memory benchmark](../benchmarks.md) uses, asked a
  different, reference-free question. The prompt is versioned and on the run
  tuple.
* **Typed prompts inflate the unanswerable side.** Many sentences typed to an
  agent depend on the conversation around them ("why did that fail?") and are
  unanswerable standalone. That moves *unanswerable → passed*. It cannot move
  the *answerable → refused* numerator, which is the number this benchmark
  exists for. The per-source columns are there so a reader can restrict to
  explicit queries.
* **The replay is against the store as it is now**, not as it was when the
  question was asked.
* **The scale is encoder-specific.** The floor reads raw cosine. Every figure
  here is for the encoder named in its table, and the run is to be repeated
  alongside any encoder change.
* **One store, one person.** This is a single-operator dogfood store of
  software-project memory. A small store, or one in a different domain, will
  sit elsewhere on the curve: the memory-rot page shows the same floor against
  a 12-attribute synthetic persona, where it passes everything.

## Results: the free replay (2026-09-20)

338 real questions harvested from 459 agent transcripts (1 MCP query, 3 CLI
queries, 334 typed prompts), replayed against a single-operator dogfood store
of 27,980 active beliefs. Encoder `all-MiniLM-L6-v2`, `top_k` 40, stock
configuration. No LLM call was made.

**Nothing in the transcripts had ever been refused.** Of the explicit query
calls whose result a transcript captured (5), none carried the refusal, so
there was no recorded population of refused questions to study; the refusals
below come from replaying real questions, which is the only way to find the
ones the gate would turn away.

Maximum cosine over the rendered top-k:

| p0 | p10 | p25 | p50 | p75 | p90 | p100 |
|---|---|---|---|---|---|---|
| 0.154 | 0.321 | 0.435 | 0.543 | 0.654 | 0.748 | 0.972 |

The refusal curve, which is what the gate *does*, unjudged:

| Floor | Refused | |
|---|---|---|
| 0.10 | 0.0% (0/338) | |
| 0.15 | 0.0% (0/338) | |
| 0.20 | 0.9% (3/338) | |
| **0.25** | **3.3% (11/338)** | configured default |
| 0.30 | 6.8% (23/338) | |
| 0.35 | 13.0% (44/338) | |
| 0.40 | 18.6% (63/338) | |
| 0.50 | 38.2% (129/338) | |
| 0.60 | 60.7% (205/338) | |

Three things this establishes without a judge:

* **The default is on the flat part of the curve.** At 0.25 the gate refuses
  about one real question in thirty, and refusals roughly double over each of
  the next two 0.05 steps. The false-negative count at the default is therefore bounded above
  by 11 of 338 (3.3%) before a single answer is judged, and at 0.20 by 3.
* **On a large store the floor passes almost everything.** 96.7% of the
  questions clear 0.25: with 28,000 beliefs, some belief is nearly always
  close. The synthetic sweep on the memory-rot page found the same thing from
  the other side (36 of 36 unanswerable probes passed). Whether what passes is
  answerable is the judged sweep's question, below.
* **There is no empty band here.** The floor was chosen against a small
  demonstration store where off-topic questions sat at or below 0.15 and
  on-topic ones at or above 0.6. Real questions against a real store fill the
  whole range between, with the median at 0.54. Any floor in that range is a
  trade, which is the reason to judge it and not guess.

## Results: the judged sweep (2026-09-20)

The same 338 questions, each answered by the real query operation with the
gate switched off and then judged. Answer and judge model `claude-sonnet-5`,
judge protocol 1. All 338 were scored; none was excluded. The judge found
**71 of 338 (21.0%) answerable**. Of the 267 that were not, the product's own
answer step declined 141 outright ("nothing relevant") and the judge rejected
the other 126 as ungrounded or not resolving the question.

| Floor | Answerable → refused ↓ | Unanswerable → passed ↓ | Refused that were answerable ↓ | Passed that were unanswerable ↓ | |
|---|---|---|---|---|---|
| 0.10 | 0.0% (0/71) | 100.0% (267/267) | n/a (0 refused) | 79.0% (267/338) | |
| 0.15 | 0.0% (0/71) | 100.0% (267/267) | n/a (0 refused) | 79.0% (267/338) | |
| 0.20 | 0.0% (0/71) | 98.9% (264/267) | 0.0% (0/3) | 78.8% (264/335) | |
| **0.25** | **0.0% (0/71)** | 95.9% (256/267) | **0.0% (0/11)** | 78.3% (256/327) | configured default |
| 0.30 | 0.0% (0/71) | 91.4% (244/267) | 0.0% (0/23) | 77.5% (244/315) | |
| 0.35 | 0.0% (0/71) | 83.5% (223/267) | 0.0% (0/44) | 75.9% (223/294) | |
| 0.40 | 1.4% (1/71) | 76.8% (205/267) | 1.6% (1/63) | 74.5% (205/275) | |
| 0.45 | 4.2% (3/71) | 65.9% (176/267) | 3.2% (3/94) | 72.1% (176/244) | |
| 0.50 | 8.5% (6/71) | 53.9% (144/267) | 4.7% (6/129) | 68.9% (144/209) | |
| 0.60 | 29.6% (21/71) | 31.1% (83/267) | 10.2% (21/205) | 62.4% (83/133) | |

Maximum cosine by judged label:

| | n | min | p10 | median | p90 |
|---|---|---|---|---|---|
| Answerable | 71 | 0.379 | 0.511 | 0.642 | 0.745 |
| Unanswerable | 267 | 0.154 | 0.309 | 0.517 | 0.749 |

What this says:

* **At the default, the gate refused no question the store could answer.**
  All 11 questions it refuses at 0.25 were judged unanswerable, and the
  lowest-scoring answerable question sits at 0.379. On this store and encoder
  the floor could rise to 0.35 without a single judged false negative; the
  first one appears at 0.40, and the cost climbs quickly after 0.50. With 71
  answerable questions, zero observed refusals bounds the default's
  false-negative rate at roughly 4% with 95% confidence (the rule of three);
  it does not show that the rate is zero.
* **The floor is a coarse filter, and it is not the gate doing most of the
  work.** At the default, 95.9% of unanswerable questions still pass it. What
  stops them becoming confident wrong answers is the second gate, the
  answer step's own instruction to decline when the retrieved beliefs do not
  bear on the question: it declined 141 of the 256 that passed. The two
  populations overlap across almost the whole cosine range (medians 0.64 and
  0.52), so no floor separates them; raising it to 0.35 would turn away 44
  unanswerable questions without an LLM call, against 11 today.
* **The remaining 126 are the honest caveat.** These cleared the floor, were
  answered, and were judged ungrounded or unhelpful. Almost all of this
  population is typed prompts, many of which plausibly depend on conversational
  context the store never saw, so the figure probably overstates what a
  deliberate memory query would meet (this run cannot separate the two), and the
  judge is strict by design (a partial answer that does not resolve the
  question is a "no"). It is reported, not explained away.

The explicit-query sources are too small to read alone (the one MCP query was
answerable; none of the three CLI queries was). That is the limit of this run:
it measures the gate on real information needs, mostly not on real memory
queries, because almost none of those were ever recorded.

## Reproducing

```bash
# Build the private held-out set from your own transcripts (no LLM call)
particles benchmark relevance-floor harvest

# The free replay: the refusal curve. Save the report of record somewhere private
particles benchmark relevance-floor --format json --output ~/private/replay.json

# Re-render any saved report as the aggregate table, over any floors (free)
particles benchmark relevance-floor resweep ~/private/replay.json --floor 0.45 --floor 0.5

# Price the judged stage from the saved replay, then run it
particles benchmark relevance-floor --judge --estimate --replay-from ~/private/replay.json
particles benchmark relevance-floor --judge --replay-from ~/private/replay.json
```

Knobs live under `benchmark_relevance_floor` in `config.yaml`; prices come from
`benchmark_memory.price_per_mtok`.

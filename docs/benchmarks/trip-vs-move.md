# Trip versus move: a value that changes, and one that does not

A reader on the r/AIMemory forum proposed a test every agent memory should
pass:

> "I live in Delhi, but I'm in Mumbai for a week." Tonight's dinner search
> should use Mumbai, while the home address in Delhi stays current. Then a
> month later the same user says "I moved to Mumbai." If the store treats every
> newer location as a permanent replacement, it'll fail the first half; if it
> treats every location as travel, it'll fail the second.

This page publishes that case, the checks it is scored by, the results, and the
commands to run all of it yourself. It also publishes a second, harder case
that earlier versions of this page presented as the reader's: it is a
different test, and the page says how.

## The cases

Short session notes, dated in their headings, ship with the package.
`particles benchmark trip-vs-move --case` chooses which ones a run deposits.

### The reader's case (`--case reader`, the default)

| Note | Date | What the user says |
|---|---|---|
| `reader-01.md` | 2026-09-01 | "I live in Delhi, but I'm in Mumbai for a week." Then: "Somewhere for dinner tonight would be great." |
| `reader-02-moved.md` | 2026-10-01 | "I moved to Mumbai." |

The first line is the reader's sentence, unchanged. The assistant's replies in
the notes name no city. The **move** variant deposits both notes, one at a
time, with `particles audit`, then runs `particles memory consolidate --scope
store`. The **trip alone** variant deposits only the first. The SHA-256 of
each note is recorded in every report, so a reader can show they ran these
bytes.

### Change of plans mid-trip (`--case mid-trip`)

| Note | Date | What the user says |
|---|---|---|
| `session-04.md` | 2026-09-04 | They live in Delhi, in Lajpat Nagar; their favourite restaurant is a ten-minute walk from their flat; they are vegetarian |
| `session-11.md` | 2026-10-08 | They are in Mumbai for a week for work, at a hotel in Bandra, and fly back home to Delhi next Thursday |
| `session-12-moved.md` | 2026-10-09 | "I moved to Mumbai." |

This is the case every result before 2026-10-14 measured. It differs from the
reader's in three ways, and each makes it harder:

- **The move comes in the middle of the trip,** the day after the trip note,
  in four words, with no word that the flight home is off. The store has to
  work out that the stay, the hotel and the flight are over.
- **Home and trip arrive five weeks apart,** in separate notes, so the store
  must reconcile a stored residence with a later trip rather than read both
  from one sentence.
- **There is more to restate:** a restaurant placed by the walk from the old
  flat, a hotel and a return flight, each of which must stop reading as
  current.

A run of this case on or after 2026-10-15 is not comparable with the results
below. The extractor keeps an end date only while it is still in the future
when the note is read, and the trip note's own Thursday (2026-10-15) has now
passed, so a later run stores the trip without it.

## The checks

Each run's store is read back and checked. The checks read only the store's
rows and events, never a model's opinion of them, and they were fixed before
any run of the case they score. Version 2 is current; version 1, the checks
of the first mid-trip results, stays selectable with `rescore
--checks-version 1`.

Each check selects claims by the note they came from: the **home note** states
the residence (`reader-01.md`; `session-04.md`), the **trip note** the week
away (`reader-01.md`; `session-11.md`), and the **move note** the move. A
**dated restatement** is a claim the store rewrote as history, with an `as of`
ISO date and a past-tense `was` or `were`.

**Trip alone.** A run passes when both hold:

- (a) an ACTIVE claim from the home note names Delhi with a residence word
  (`live`, `lives`, `living`, `home`, `flat`, `reside`, `resides`,
  `residence`): the home stays current;
- (b) an ACTIVE claim from the trip note that is not a restatement names
  Mumbai: the trip is current, which is what tonight's dinner search needs.

One claim may satisfy both ("The user lives in Delhi but is in Mumbai for a
week.").

**Move.** A run passes when all of these hold:

1. No ACTIVE claim names Delhi with a residence word or a flight word (`fly`,
   `flies`, `flying`, `flight`, `back to`) unless it is a dated restatement.
2. No restatement is kept back for contradicting the claim that retired its
   anchor, and every restatement the store wrote is ACTIVE.
3. No ACTIVE claim from the trip note reads the week as current: none matches
   `for a week`, `week-long`, `trip`, `visit`, `visiting`, `temporary` or
   `temporarily` unless it is a dated restatement or, in the reader's case,
   its own end date falls on or before the move.
4. An ACTIVE claim from the move note names Mumbai: the move is kept.

**The bar.** Three move runs and two trip-alone runs per model, every one
passing, with every LLM purpose on the one model under test. For the reader's
case the bar and the run counts were fixed on 2026-10-14, before its first run.

**One limit of a replay.** The reader's week ends on about 2026-09-08. The
extractor keeps an end date only when it is still in the future at the moment
the note is read, so a note replayed weeks later stores the week with no end.
In this benchmark the trip can therefore stop reading as current only through
a later claim: the move retiring it, or the store rewriting it as dated
history. Check 3 accepts an end date as well, and would credit one if the
store kept it.

## Results

### The reader's case

Run on 2026-10-14 on the build released as 1.191.0, every purpose on one
model, checks version 2. The notes' digests were posted publicly before the
first run.

| | `claude-sonnet-5-5` | `claude-sonnet-4-6` |
|---|---|---|
| Move, runs passing | **3 of 3** | **3 of 3** |
| Trip alone, runs passing | **2 of 2** | **2 of 2** |
| Bar | **met** | **met** |
| Cost per move run (list price) | US$0.04 to US$0.07 | US$0.06 to US$0.07 |
| Cost per trip-alone run | US$0.02 | US$0.02 |
| Cost of the protocol | US$0.21 | US$0.22 |

The per-run stores are in
[`trip-vs-move-reader-claude-sonnet-5-5-2026-10-14.json`](trip-vs-move-reader-claude-sonnet-5-5-2026-10-14.json)
and
[`trip-vs-move-reader-claude-sonnet-4-6-2026-10-14.json`](trip-vs-move-reader-claude-sonnet-4-6-2026-10-14.json).

**The first half.** Every trip-alone run kept two ACTIVE claims, one for each
half of the sentence: "The user lives in Delhi." and "The user is in Mumbai for
a week, as of 2026-09-01." Neither model wrote the sentence as one claim
naming both cities.

**The second half.** In every move run the move ("The user moved to Mumbai.")
retired both claims as updated by it, and stayed the one current statement of
where the user lives. On `claude-sonnet-4-6` two of the three move runs wrote
the week in the past tense ("The user was in Mumbai on 2026-09-01."), and the
move retired it all the same.

Two observations sit outside the checks and are reported, not scored:

- **The week ended because the move ended it.** No run stored an end date on
  the week, as the limit above predicts, so in this benchmark the trip stops
  reading as current only when a later claim retires it.
- **A dated dinner request can stay.** On `claude-sonnet-4-6` the extractor
  also wrote the dinner request as a claim ("The user wanted a dinner
  recommendation for the evening of 2026-09-01 in Mumbai."), which stayed
  ACTIVE after the move in two runs. It is dated and past, so it does not read
  the week as current, and check 3 does not count it.

### Change of plans mid-trip

Every purpose on one model. The first runs (2026-10-12, on the implementation
of 1.187.0, US$1.46 in all) used a harness this suite replaced; the second
(2026-10-13, the build released as 1.190.0, US$1.69 in all) used this suite,
with the same notes and checks version 1. Both 2026-10-13 reports were then
re-scored under checks version 2 at no cost, from the rows and events they
saved.

| | `claude-sonnet-5-5` | `claude-sonnet-4-6` |
|---|---|---|
| Move, runs passing (2026-10-12, checks v1) | **3 of 3** | 2 of 3 |
| Move, runs passing (2026-10-13, checks v1) | **3 of 3** | 2 of 3 |
| Move, runs passing (2026-10-13, re-scored under checks v2) | **3 of 3** | 2 of 3 |
| Trip alone, runs passing (every row above) | **2 of 2** | **2 of 2** |
| Cost per move run, 2026-10-13 (list price) | US$0.15 to US$0.18 | US$0.21 to US$0.26 |
| Cost per trip-alone run, 2026-10-13 | US$0.09 | US$0.13 to US$0.15 |

The per-run stores are in
[`trip-vs-move-end-to-end-2026-10-12.json`](trip-vs-move-end-to-end-2026-10-12.json),
[`trip-vs-move-claude-sonnet-5-5-2026-10-13.json`](trip-vs-move-claude-sonnet-5-5-2026-10-13.json)
and
[`trip-vs-move-claude-sonnet-4-6-2026-10-13.json`](trip-vs-move-claude-sonnet-4-6-2026-10-13.json),
and the re-scored reports beside them with `-checks-v2` in their names.

On `claude-sonnet-5-5` every move run retired the residence, the home claim,
the trip and the hotel stay as updated by the move, and kept the return flight
and the walk to the restaurant as dated history:

> As of 2026-10-08, when the user was in Mumbai for a week for work, they
> planned to fly back to Delhi on Thursday, 2026-10-15.

The failing `claude-sonnet-4-6` run did everything else right, on both days,
and failed the same way under both check versions. Its re-anchor check judged
the restated walk ("was a ten-minute walk from the flat in Lajpat Nagar,
Delhi, where the user lived, as of 2026-09-04") to contradict another standing
claim about the restaurant, so it kept the original sentence, which still
reads as the user's current flat. That is a known residual of the re-anchor
check on that model, not of the update rule this case tests. The reader's
notes name no restaurant, so it cannot arise there.

Both 2026-10-13 reports were run on the build released as 1.190.0 and record
the number it carried before its release, `sdk_version` 1.189.0.

## Run it yourself

The suite ships with the engine, so a released install is enough. Choosing a
case needs 1.191.0 or later:

```bash
pip install linkedparticles
export ANTHROPIC_API_KEY=sk-ant-...
particles benchmark trip-vs-move --model claude-sonnet-5-5 --estimate
particles benchmark trip-vs-move --model claude-sonnet-5-5 --output trip-vs-move.json
particles benchmark trip-vs-move --case mid-trip --model claude-sonnet-5-5 --estimate
```

The first command prints what a run of the reader's case will cost and makes no
call. The second asks before its first call, then runs the published protocol:
three move runs and two trip-alone runs. The third prices the mid-trip case.
Each estimate is priced from measured usage at your own list prices: about
US$0.25 for the reader's case on either model, and for the mid-trip case about
US$0.75 on `claude-sonnet-5-5` and US$1.10 on `claude-sonnet-4-6`. The
commands name `claude-sonnet-5-5` on purpose, so the result does not depend on
what your default model is.

Three properties of a run matter:

- **It never touches your own store.** Each run is a separate process with its
  own home directory, store, blob directory and configuration. Only your API
  key, your provider endpoints, your prices and your network policy reach it.
- **Every LLM purpose runs on the model you name.** A usage row on any other
  model marks the run invalid rather than passed. `--pin PURPOSE=MODEL` runs
  one purpose on another model, and the report names the mixture.
- **Spend is bounded.** An estimate prints before any call, every live run
  asks first (`--yes` confirms in advance), and `--max-usd` sets a ceiling per
  run, checked against measured spend before each step. A run that reaches it
  stops and is reported incomplete, never passed or failed. Consolidation runs
  as it did for the published results, bounded by its own probe caps.

Two other forms are worth knowing:

```bash
particles benchmark trip-vs-move --case mid-trip --model claude-sonnet-5-5 --no-same-slot-admits
particles benchmark trip-vs-move --arm scripted
```

`--no-same-slot-admits` runs the update rule as it was before 1.187.0, which
reproduces the failure the mid-trip case exposed: the residence and the trip
stay current after the move. Under the scripted arm the reader's case fails
the same way under that rule. `--arm scripted` answers every model call from a
table in the package, costs nothing and needs no key. It tests the harness,
the isolation and the checks; it says nothing about any model. Its table
covers both cases from 1.191.2 on.

`particles benchmark trip-vs-move rescore trip-vs-move.json` re-runs the checks
over a saved report with no call: every report saves each particle and the
events the checks read. `--checks-version` chooses the version.

## The registered pair sets

Passing the mid-trip case could mean the fix was tuned to it. The update rule was also
measured on labelled claim pairs, registered on 2026-10-12 before any model
was asked a row: a development set (38 pairs), a held-out set worded apart
from it (19), a set of past values (15) and a held-out set for the past-value
marker (17). Each pair is an older and a newer claim about one thing, labelled
with whether the newer one replaces the older.

```bash
particles benchmark update-gate --model claude-sonnet-5-5 --estimate
particles benchmark update-gate --model claude-sonnet-5-5 --output update-gate.json
```

Each pair is asked three times. Each sample asks the contradiction probe and
the update probe, and three rules are scored from the same answers: the
**production** rule 1.187.0 ships, the update verdict **alone**, and the rule
**before 1.187**. The registered bar applies to the production rule: every
labelled update retired in at least two of three samples (one may miss on a
held-out set), and no retirement in any control sample.

Results of record, production rule, updates retired and control retirements:

| Set | `claude-sonnet-5-5` | `claude-sonnet-4-6` |
|---|---|---|
| Development | **10 of 10**, 0 of 57 | **10 of 10**, 0 of 57 |
| Held-out | **8 of 8**, 0 of 33 | **8 of 8**, 3 of 33 |
| Past values | **5 of 5**, 0 of 24 | **5 of 5**, 1 of 24 |
| Past-value marker, held-out | **5 of 5**, 0 of 27 | **5 of 5**, 2 of 27 |

The rule before 1.187 retired 1 of 10 and 1 of 8 dated updates on
`claude-sonnet-5-5`, and 4 of 10 and 5 of 8 on `claude-sonnet-4-6`.

Every control the production rule retires on `claude-sonnet-4-6` is one the
older rule retired on the same samples: a claim that relied on the old home,
and two notes written in the shape of a dated restatement. The report states
this as a second reading beside each bar, `regression`, which holds when the
production rule adds no control retirement to the older one. It holds on
every set on both models. One row is reported and never barred: a past value
stated without a marker ("In 2023 the user's rent was 640 euros a month.") was
retired by a later rent in three samples of three on both models.

The samples are in
[`update-gate-acceptance-2026-10-12.json`](update-gate-acceptance-2026-10-12.json),
[`update-gate-past-value-2026-10-12.json`](update-gate-past-value-2026-10-12.json)
and [`update-gate-fallback-2026-10-12.json`](update-gate-fallback-2026-10-12.json).
At the 2026-10-12 list prices all four sets cost about US$1.75 on
`claude-sonnet-5-5` and US$1.85 on `claude-sonnet-4-6`.

The first replay through `particles benchmark update-gate`, on the build released as 1.190.0, on
2026-10-13, reproduced every production-rule tally in the table above on both
models, with the same three `claude-sonnet-4-6` controls retired and no control
the production rule adds to the older rule's. It cost US$1.75 and US$1.86:
[`update-gate-claude-sonnet-5-5-2026-10-13.json`](update-gate-claude-sonnet-5-5-2026-10-13.json)
and
[`update-gate-claude-sonnet-4-6-2026-10-13.json`](update-gate-claude-sonnet-4-6-2026-10-13.json).

**What this page cannot show.** The sets were registered before the runs that
score them, and the dates above say so. A published release is one commit, so
this repository cannot prove that order; the development history that records
it is not public.

## The same rule in a simulated world

The [memory-rot benchmark](rot.md) measures the update rule in both directions
on scripted worlds where the truth is known. On 2026-10-12, with the production
rule, each world retired exactly the updates the ground truth retires, on both
models, and every retrieval metric matched the ground truth's
([`update-gate-rot-adr0313-2026-10-12.json`](update-gate-rot-adr0313-2026-10-12.json)).

## Reading a report

Both suites write one JSON report per model. `trip-vs-move` reports carry the
model of every purpose, the digests of the three notes, each run's checks,
every particle and checked event of its store, and its measured usage.
`update-gate` reports carry every sample and each set's tallies under all three
rules. Neither report has an aggregate score. The full option list is in the
[CLI reference](../cli-reference.md).

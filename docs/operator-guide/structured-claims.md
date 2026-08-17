# Structured claims

A particle's `content` is prose — a claim written as a sentence. Semantic search
is good at finding prose and bad at the questions that are really *relational*:

- which coins weigh more than 3 g,
- everything with the predicate *was minted at*,
- how many subjects have a *date of birth* claim.

Each of those is a comparison or a count, not a similarity ranking, and none is
answerable without an LLM call.

A **structured claim** is a second rendering of the same claim as one
subject-predicate-object triple, stored beside the prose:

```
content            "Sirius has spectral type A1V."
structured_claim   Sirius —[has spectral type]→ "A1V"
```

## It is an annotation, not an assertion

This is the rule everything else follows from.

`content`, `confidence` and provenance are **asserted** — a source said them,
and they are never rewritten. The triple is **derived** — tooling produced it
from `content`, and it can be regenerated at any time. So:

- Generating, regenerating, or failing to generate a triple **cannot change a
  belief**. Confidence is untouched, status is untouched, contradiction
  resolution is untouched.
- A bad triple is a tooling defect to regenerate, never evidence that the source
  is less trustworthy.
- Every stored triple carries a stamp — which structurizer produced it, at which
  version, and when — so you can always tell what a given annotation came from
  and re-do the ones from an older generator.

## Absence is fine, permanently

Some prose has no honest triple: *"the migration was harder than we expected"*
is a real claim and not a relation between two things. The extractor and the
backfill are both told to emit nothing rather than invent one, because a
fabricated triple is a false statement the exporters will publish.

So there is **no coverage target and no lint finding for a missing
annotation**. Nothing in the system degrades without it. Coverage is a number
you can look at, not a goal you have to hit.

## Getting them

**New particles get theirs for free.** Since v1.96.0 the general extractor asks
for the triple as one more field of the reply it was already making — no extra
LLM call, no extra cost. It is on by default; `structured_claim.enabled: false`
in `config.yaml` turns it off and restores the previous prompt exactly.

**Existing particles need a backfill**, and that one *does* cost an LLM call per
particle. Hence a verb with a rate limit and a resumable cap:

```bash
particles structure --dry-run
```

Reports the **whole** backlog — not the batch cap — plus how many runs that
implies at the current `--limit`, and current coverage. Writes nothing, spends
nothing:

```json
{
  "dry_run": true,
  "backlog": 21506,
  "batch_limit": 200,
  "runs_needed": 108,
  "coverage": { "active": 21506, "annotated": 0, "by_structurizer": {} }
}
```

`backlog` is the number to size the job from, and it is exactly one LLM call
each — a particle is asked about **once**, whether or not a triple comes back.
Measured on a 21.5k-particle store, throughput is bounded by model latency
(~1.3 s/claim) rather than by the rate cap, so a full backfill is on the order
of 8 hours. That is why the pass is capped, resumable, and interruptible rather
than something you start by accident.

```bash
particles structure --limit 0
```

`--limit 0` means **the whole backlog in one run**. That is safe to start and
safe to interrupt: the pass commits every
`structured_claim.backfill_commit_interval` particles (default 25), so Ctrl-C
costs you at most the current handful, never the hours already paid for. Restart
and it picks up where it stopped.

A smaller `--limit N` takes one slice; the summary reports `remaining`, so a
capped run never reads as "done".

Particles whose prose the structurizer declines to triple-ize come back as
`skipped`, not `failed` — the designed outcome, not an error. **The decline is
recorded**, so those particles leave the backlog and are never re-asked (and
never re-charged). They stay un-annotated, which is a legal permanent state;
what is recorded is only *that we asked, and at which structurizer version*. A
later version re-asks them by construction — that is what the version stamp on
the attempt buys.

### The four storage states

Worth knowing when reading a row directly:

| Stored triple | Structurizer stamp | Meaning |
|---|---|---|
| — | — | never attempted; in the backlog |
| — | set | attempted, declined; **out** of the backlog |
| set | set | annotated |
| set | — | unreachable (a corrupt row; `to_model()` says so) |

## Regenerating after a structurizer upgrade

When the structurizer's prompt or parsing changes in a way that makes old
triples worth redoing, its version is bumped. Find and regenerate the old ones
exactly as you would for an extractor upgrade:

```bash
particles structure --structurizer-version 1.0.0
```

That selects annotated particles stamped with a version *other* than `1.0.0` —
the mirror of `reindex --extractor-version`. The claims themselves are not
re-extracted; only the annotation is rewritten.

**Structure-canonical particles are excluded from both scopes.** A particle
whose triple came from a structure-native extractor — `rdf-extractor` for a
deposited RDF document, `wikidata-extractor` for a Wikibase statement — carries
`canonical_form: STRUCTURED`, meaning the triple is the assertion and the prose
is the rendering. There is nothing for the content-structurizer to derive there,
and rewriting it would replace what the source said with a guess at it. Redo one
the way you redo any extractor output instead:

```bash
particles reindex --extractor-version 0.1.0
```

## Checking coverage

```bash
particles quality
```

ends with a structured-claims section: how many ACTIVE particles carry an
annotation, as a fraction, broken down by which structurizer and version
produced them. A store part-way through a backfill will show two or three
stamps at once, which is normal.

## When something looks wrong

`particles lint` reports `STRUCTURED_CLAIM_SUBJECT_MISMATCH` (check `L-STR-11`)
when a triple's subject is not one of the subjects the particle is about — the
cheapest signal that the structurizer picked the wrong entity. The fix is always
to regenerate the annotation; the claim needs no change:

```bash
particles structure --structurizer-version <the version in the finding>
```

A triple whose subject resolved to *no* Subject at all is **not** flagged. That
is the honest state for a claim about something your store has no Subject for,
and reporting it would turn a coverage gap into a recurring alarm.

## What exporters do with them

They report coverage. They do **not** generate triples inline — generation
happens only in the extraction pipeline and in `particles structure`, both of
which write the stamp. An exporter that structurized on the fly would produce an
unstamped, unstored, unrepeatable triple and put an LLM call inside a render
path you expect to be fast and deterministic.

Structured claims do travel with the interchange format (`particles interchange
export` / `import`), stamp included, so copying a store does not throw away work
you paid for.

# The vocabulary report on the owner's store (2026-10-02)

**A dated snapshot, as of 2026-10-02.** An ontology is built from evidence:
which relations occur, how often, what values they take, which kinds of thing
they attach to, which entities line up with an external knowledge base, and
which modelling choices a person has confirmed. A store of beliefs holds all
of that already. This page runs the report that reads it out, `particles
query --vocabulary`, on a read-only copy of the owner's live store, and checks
that it reproduces the figures first measured by hand with SQL on 2026-09-30.

## What the report computes

The report is computed on every call and never stored. It has two parts.

**A store header.**

- Subjects in total, the subjects aligned to each external namespace, and,
  within a namespace, how many fall in each link-confidence band. The bands
  are the ones the subject resolver already records: `asserted`
  (confidence 1.0, from a structured extractor or an operator), `scored` (an
  embedding score at or above `subjects.wikidata_link_suppress_threshold`),
  `unscored` (the 0.5 a link gets when no content could score it), and
  `suppressed` (below the threshold, stored but hidden from export and
  query output).
- Subjects carrying a class, by class and by the namespace the class is
  minted in.
- Structured claims in the store, the claims in view after the query's
  default exclusions, and their object kinds and value shapes.
- The confirmed modelling decisions in the operator event log, by event type:
  merged or unmerged duplicates, closed contradictions, resolved reviews, and
  subject merges, splits, aliases, reclassifications and link rulings.

**One row per canonical predicate.** Surface forms are grouped under the
deterministic canonical form that predicate profiles compare, so
`has status`, `had status` and `have status` are one row with all three
listed. Each row carries its claim count, its object value shapes (text,
numeric, dated, URI, token), the classes of the subjects its claims are
about, and its alignment: the term an adopted vocabulary document mints or
aliases for the predicate, and that term's links to external properties.
The alignment column is empty on every row of this run, since
the store has adopted no vocabulary document, and the report says so rather
than leaving the column out.

A value shape is a reading of form, for inferring a datatype: an untyped
literal `"1987"` counts as numeric. The structural query filters still
compare only typed values.

The report reads claims through a project observer when one is named, like
every other read that selects beliefs. The subject header and the
decision counts are store-wide, since subjects and operator events are not
beliefs a project observes. With `--as-of`, the claims are those believed at
the instant, the header counts the subjects created by then (with their
classes and links as they stand now), and the decisions are those recorded by
then.

## How it was run

On a copy of the store taken with SQLite's backup interface from a read-only
connection:

```bash
DATABASE_URL=sqlite+aiosqlite:///store-copy.db uv run particles query --vocabulary
DATABASE_URL=sqlite+aiosqlite:///store-copy.db uv run particles query --vocabulary --as-of 2026-10-01
DATABASE_URL=sqlite+aiosqlite:///store-copy.db uv run particles query --vocabulary \
    --as-of 2026-10-01 --include-document-meta --include-non-asserted
```

Each run took about eight seconds, with no embedding and no LLM call. The
same report is served as JSON by `--format json` and by `GET /vocabulary`.

## The figures first measured, reproduced

The 2026-09-30 measurement was taken by hand before midnight UTC. Reading the
store as of 2026-10-01 at 00:00 UTC reproduces every figure.

| Figure | Measured 2026-09-30 | Report as of 2026-10-01 |
|---|---|---|
| Distinct predicates | 6,358 | 6,358 |
| Structured claims the listing counts | 11,074 | 11,074 |
| ACTIVE structured claims, no default exclusions | 11,462 | 11,462 |
| Literal objects among them | 11,338 | 11,338 |
| URI objects among them | 124 | 124 |
| Subjects | 9,161 | 9,161 |
| Subjects aligned to Wikidata | 582 | 582 |
| Subjects classed in the local `artifact:` namespace | 2,243 | 2,243 |
| `DUPLICATES_MERGED` | 181 | 181 |
| `INCONSISTENCY_CLOSED` | 198 | 198 |
| `REVIEW_RESOLVED` | 14 | 14 |
| `SUBJECTS_MERGED` | 4 | 4 |

The literal and URI counts were measured over every ACTIVE claim, so they are
reproduced by the third command, which lifts the default exclusions of
document-structure and non-asserted claims. The report also counts three
`SUBJECT_ALIASED` events the first measurement did not look for, so its
decision total as of 2026-10-01 is 400 against the 397 the four types sum to.

## Today's store

| Header figure | 2026-10-02 |
|---|---|
| Subjects | 9,850 |
| Aligned to any namespace | 2,945 |
| Classed | 2,363, all in `artifact:` |
| ACTIVE structured claims | 13,817 |
| Claims in view | 13,387 |
| Literal / URI / token objects | 13,243 / 144 / 0 |
| Text / numeric / dated values | 12,452 / 671 / 120 |
| Distinct predicates | 7,504 |
| Canonical predicates | 6,758 |
| Canonical predicates with one claim | 5,381 |
| Share of claims under the 20 commonest canonical predicates | 15.1% |
| Modelling decisions | 402 |

Alignment by namespace and link band:

| Namespace | Subjects | Asserted | Scored | Unscored | Suppressed |
|---|---|---|---|---|---|
| `artifact` | 2,363 | 2,363 | 0 | 0 | 0 |
| `wikidata` | 582 | 3 | 340 | 41 | 198 |

The 6,758 canonical forms differ by one from the 6,759 the
[canonicalisation page](predicate-canonicalisation-2026-10-01.md) reports
for the core stem on the same listing. That page measured the stem before
`run` was treated as its own participle; the report uses the shipped
function.

## The co-occurrence signal

The subject class a predicate attaches to is the evidence for its domain. The
commonest canonical predicates on each class of subject, by claims:

| Class | Claims | Commonest predicates |
|---|---|---|
| `artifact:record` | 1,910 | `has status` (138), `covers` (53), `concerns` (51), `is` (40), `addresses` (31) |
| `artifact:file` | 1,224 | `contains` (60), `is located at` (39), `implements` (37), `is` (25), `requires` (15) |
| `artifact:symbol` | 379 | `has status` (19), `reports` (11), `returns` (8), `is defined in` (7) |
| `artifact:command` | 212 | `contains count` (12), `uses` (5), `supports flag` (4), `supports option` (3) |
| Unclassed subject | 7,409 | `is` (205), `passed` (97), `contains` (97), `is located at` (71) |
| No resolved subject | 2,253 | `is` (81), `requires` (45), `covers` (25), `returns` (24) |

## Reading

- **The evidence is real today and needs no new machinery.** Records carry
  status and scope, files carry containment and location, commands carry
  flags. A reviewer could write each of those down as a domain from this
  table alone.
- **Most claims attach to no class.** 9,662 of the 13,387 claims in view are
  about an unclassed subject or none, so the co-occurrence signal covers
  about 28% of the store. Wikidata-resolved subjects record no class, which
  is the largest single gap.
- **Alignment is thin and mostly estimated.** 579 of the 582 Wikidata links
  carry a confidence below 1, and 198 of them are below the suppression
  threshold. An export that wrote them all as identity would assert what the
  resolver only estimated.
- **Values are free text.** 93% of values are text, and no literal in the
  store carries a datatype: the 671 numeric and 120 dated values are untyped
  literals whose form says what a datatype would.

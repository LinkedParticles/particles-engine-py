# Subject resolution: the prefix-expansion fix (2026-10-01)

**A dated snapshot, as of 2026-10-01.** This page measures one change to the
subject resolver: the Wikidata filter that drops a search hit whose label
continues the name as a longer word now keeps that hit when the entity lists
the name among its own aliases. It uses the same suite, column and
method as the [2026-09-30 baseline](subject-resolution-2026-09-30.md), and
re-measures that baseline on the same day so the before and after share one
state of live Wikidata.

## The failure

Wikidata's search does prefix matching. A short name therefore often matches a
longer label for a different thing: "micrograd", a small autograd library,
matches "Microgradients of microbial oxygen consumption …". The resolver drops
any hit whose label continues the name with a letter or digit, which is what
keeps "micrograd" off that paper.

The same rule dropped the right answer for "Postgres". Wikidata's top hit is
PostgreSQL (`Q192490`), and "PostgreSQL" continues "Postgres" with a letter.
The resolver fell through to the next hit, `Q28975208`, POSTGRES, the
discontinued predecessor, and renamed the Subject "POSTGRES".

## The rule

A word-continuation hit is now kept when its aliases include the name exactly,
case and all. PostgreSQL lists "Postgres", so it is the same entity under its
fuller name. "PostgreSQL License" also continues "Postgres" and lists no such
alias, so it is still dropped.

Two details decide whether the rule works at all:

- **The alias read is a second call.** The search response does not say why
  a hit matched: for "Postgres" it reports a label match on "PostgreSQL" and
  carries no aliases. One batched `wbgetentities` call reads the aliases of
  the continuation hits only, so a name with no continuation hit costs
  nothing extra. If that call fails, every continuation hit is dropped, which
  is the old behaviour.
- **English aliases alone miss it.** PostgreSQL has no English alias. Its
  "Postgres" alias sits under `mul`, the language-neutral code Wikidata uses
  for names that are the same in every language. The read takes `en` and
  `mul`.

The rule is a pure function of the name, the label and the aliases, so the same
responses always give the same answer.

### The rules that lost

Three rules were candidates. Each was checked against the live search results
for all 35 gold names plus a dozen common software names ("micrograd", "Go",
"Mongo", "Java" and others), looking at every hit the current filter drops.

| Rule | Keeps PostgreSQL for "Postgres" | Breaks |
|---|---|---|
| The hit's aliases include the name exactly, case-sensitive | yes | nothing found |
| The label adds one unbroken word to the name | yes | "Go": Goiás, a Brazilian state, is the top hit and would be linked |
| The name is at least N characters long | only for N ≤ 8 | "micrograd" is 9 characters, so any N that keeps PostgreSQL also keeps "Microgradients …" |

The alias rule also had to be case-sensitive. Goiás lists the alias "GO", so a
case-insensitive match would link "Go" to the state. Keeping a continuation
only when it is the top hit fails on Goiás the same way and was not pursued.

## Result

| Parameter | Value |
|---|---|
| Suite | `prose-article-seed-001` v0.3.0, its `gold_subjects` block |
| Resolver | `particles` 1.168.4 (before) and 1.168.7 (after), default `top_hit` selection |
| Encoder and thresholds | unchanged from the baseline page |
| Wikidata | live, 2026-10-01 |
| Repeats | 2 per arm, byte-identical per-subject results within each arm |

| Metric | Before | After | Change |
|---|---:|---:|---:|
| `resolution_accuracy` | 0.886 (31) | **0.914** (32) | +1 subject |
| `resolution_wrong_ref` | 0.086 (3) | 0.057 (2) | −1 subject |
| `resolution_bare_local` | 0.029 (1) | 0.029 (1) | none |

The re-measured baseline matches the 2026-09-30 page row for row. One row
changes:

| Subject | Gold | Before | After |
|---|---|---|---|
| Postgres | `Q192490` | `Q28975208` POSTGRES (0.33), **wrong** | `Q192490` PostgreSQL (0.20), correct |

The other 34 rows are identical, so none of the 31 correct links moved. The
Subject is now named "PostgreSQL" with "Postgres" as an alias.

The correct link scores 0.20 against the claim, which puts it in the band
between the two thresholds: stored and joined, hidden from exports and flagged
by lint, like the five correct links the baseline page lists there. The wrong
link it replaces scored higher, at 0.33. That is the baseline page's finding
again: the description-to-claim score does not separate right from wrong.

The three remaining failures are untouched and need different fixes:

- **Harbor and Lantern** still link to the common nouns. That needs a
  judgement per ambiguous name, measured on
  [a later page](subject-resolution-judge-2026-10-01.md).
- **Rust** still abstains, a correct top hit scored below the 0.15 floor.

## Claim metrics from the same runs

The benchmark also ran the extractor in each pass. The resolver change does not
touch extraction, and the spread below is sampling, not the fix.

| Pass | Recall | Precision | Calibration error |
|---|---:|---:|---:|
| Before, 1 | 0.88 | 0.82 | 0.10 |
| Before, 2 | 0.92 | 0.84 | 0.08 |
| After, 1 | 0.90 | 0.83 | 0.09 |
| After, 2 | 0.92 | 0.84 | 0.09 |

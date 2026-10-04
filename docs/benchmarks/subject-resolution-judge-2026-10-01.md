# Subject resolution: a model judges the ambiguous names (2026-10-01)

**A dated snapshot, as of 2026-10-01.** This page measures a third way for the
subject resolver to choose among Wikidata's search candidates: for an ambiguous
name only, one model call reads the claim and the candidates and picks one, or
answers that none of them is meant. It uses the same suite, column
and method as the [2026-09-30 baseline](subject-resolution-2026-09-30.md) and
the [prefix-expansion fix](subject-resolution-2026-10-01.md), and re-measures
the default resolver on the same day so both arms share one state of live
Wikidata.

## The failure

Two invented companies in the suite are named with ordinary English words. For
"Harbor has acquired Lantern.", the resolver linked Harbor to the body of water
(`Q283202`) and Lantern to the lighting device (`Q862454`), because each is
Wikidata's top hit for its name. The baseline page showed why no score of
description against claim can catch this: on that set the correct links scored
between 0.16 and 0.58, and the wrong ones between 0.20 and 0.33.

## The selector

The setting `subjects.wikidata_candidate_selection: llm_judge` changes one step
of the cascade, and only for a name that is ambiguous. The test is a pure
function of the search response and one local score:

| Usable candidates | Ambiguous | Why |
|---|---|---|
| none | no | nothing to choose; the name becomes a bare local Subject |
| one, scored at or above 0.25 | no | the lone candidate's description matches the claim at the line where a link is shown |
| one, scored below 0.25 | yes | a lone common noun against a claim about something else |
| one, not scorable | no | with no description or no claim the model has nothing to weigh |
| several | yes | the search ranks by popularity, and the description score cannot pick among them |

An ambiguous name gets exactly one call, never one per candidate. The call
carries the name, the claim, and each candidate's id, label and description,
all of which the one search response already holds, so it costs no further
Wikidata request. The model answers with a QID or "none".

- **A pick is adopted** at its description score or at 0.5, whichever is
  higher. The judge's reading of the claim is stronger evidence than a score
  this page's predecessors showed cannot separate right from wrong, and 0.5 is
  the value the resolver already attaches when a link cannot be scored. A
  judged link therefore never falls under the 0.15 abstention floor.
- **"None" leaves a bare local Subject** under the extracted name with no
  external ref. The rejected candidates are kept in the verdict record only:
  a stored ref, at any confidence, would join every later mention of that QID
  to the invented company, and the column would count it as a wrong link.
- **A failed or unusable reply takes the top hit**, which is the default
  resolver's answer.

Every usable answer is recorded in the store's probe-verdict ledger, keyed by
the name and claim, the candidate set in rank order, the prompt version and
the model, and read back before any call. The same input therefore resolves the
same way, and a second resolution of it costs nothing. The call runs on its own
completion purpose, `llm.subject_resolution` (the default model when unset), so
an extraction run's usage line and spend record include it.

## Result

| Parameter | Value |
|---|---|
| Suite | `prose-article-seed-001` v0.3.0, its `gold_subjects` block |
| Resolver | `particles` 1.168.16, `top_hit` against `llm_judge` |
| Judge | `anthropic/claude-sonnet-4-6`, temperature 0, prompt version `3614d60fbfb2fb1d` |
| Encoder and thresholds | unchanged from the baseline page |
| Wikidata | live, 2026-10-01 |
| Repeats | 2 per arm, byte-identical per-subject results within each arm |

| Metric | `top_hit` | `llm_judge` | Change |
|---|---:|---:|---:|
| `resolution_accuracy` | 0.914 (32) | **0.971** (34) | +2 subjects |
| `resolution_wrong_ref` | 0.057 (2) | 0.029 (1) | −1 subject |
| `resolution_bare_local` | 0.029 (1) | 0.000 (0) | −1 subject |

The `top_hit` arm matches the 2026-10-01 page row for row. Two outcomes change,
and none of the 32 correct links is lost:

| Subject | Gold | `top_hit` | `llm_judge` |
|---|---|---|---|
| Harbor | none | `Q283202` body of water (0.30), **wrong** | no ref, correct |
| Rust | `Q575650` | no link: the correct top hit scored below 0.15 | `Q575650` programming language (0.50), correct |
| Lantern | none | `Q862454` lighting device (0.20), **wrong** | `Q28770319` "device under development" (0.50), **wrong** |

Lantern stays wrong under a different QID. The claim "Harbor has acquired
Lantern." says only that Lantern is something a company can acquire, and one
candidate is a device under development, which fits. The claim gives no
evidence for the gold reading, a payroll vendor, that the judge could use.

### Confidence moves for every judged link

A judged link is stored at no less than 0.5, so six correct links that sat in
the hidden band between 0.15 and 0.25 under `top_hit` (Python, Go, Postgres,
Kafka, Signal, Notion) are now shown by exporters and no longer flagged by the
link-mismatch lint. The column counts a link at any confidence, so this moves
no outcome, but it changes what a reader of an export sees.

### How many names reach the model, and the cost

| Measure | Value |
|---|---:|
| Gold names | 35 |
| Names that reached the model | 14 (40%) |
| Calls per pass | 14 |
| Tokens per pass, input / output | 8,666 / 171 (pass 1), 8,586 / 171 (pass 2) |
| Cost per pass, list price | US$0.0286 (pass 1), US$0.0283 (pass 2) |
| Cost per 100 names resolved | about US$0.08 |
| Cost per 100 names judged | about US$0.20 |

The fourteen are the twelve real names with more than one usable candidate or
a weak lone one, plus Harbor and Lantern. Duluth did not reach the model: its
one candidate scored 0.43. Twenty of the twenty-two invented names returned no
usable candidate and never reached it. The share on this suite is high because
the suite was built to hold ambiguous names; a store of private referents will
send fewer. Input tokens differ between passes by the per-call fence nonce,
which tokenizes differently each time.

### A prompt variant that changed nothing

After the first pass, one clause was tried that told the judge a wrong pick is
worse than none, and to answer "none" when the claim does not show which
candidate it means. Two passes with it gave the same 35 rows, Lantern
included, so it was not kept: the shipped prompt is the one measured above.

## A second gold set: ordinary prose

The suite above was written around known resolver failures, which is the
right way to show a failure and the wrong way to estimate how a change does
on prose nobody wrote for it. A second gold set was therefore built for the
default decision: `tests/benchmark/resolution/ordinary-prose-001.yaml`, eight
short passages in eight genres (local news, a tech blog, travel, business,
science, sports, a home blog, and team notes). Facts about real entities are
true and well known, and the people are invented. Every proper-named entity in
a passage is a gold subject, 49 in all: 43 with a Wikidata item and 6 without.
Each ref was chosen from live Wikidata by reading the candidates'
descriptions, and the set was committed before either selection value ran on
it.

| Parameter | Value |
|---|---|
| Gold set | `ordinary-prose-001` v0.1.0, 49 subjects, source type `WEB_PAGE` |
| Resolver, judge, encoder | as above |
| Repeats | 2 per arm |
| Command | `uv run python scripts/measure_subject_resolution.py tests/benchmark/resolution/ordinary-prose-001.yaml --selection <value> --out <file>` |

| Metric | `top_hit` | `llm_judge`, pass 1 | `llm_judge`, pass 2 |
|---|---:|---:|---:|
| `resolution_accuracy` | 0.735 (36) | 0.898 (44) | **0.918** (45) |
| `resolution_wrong_ref` | 0.041 (2) | 0.020 (1) | 0.000 (0) |
| `resolution_bare_local` | 0.224 (11) | 0.082 (4) | 0.082 (4) |

The two `top_hit` passes gave identical rows. The two `llm_judge` passes
differ in one row, Braga, which is the next section. No correct `top_hit` link
was lost in either pass.

**What the judge fixed.** The default resolver's two wrong links were
Chesterton, linked to the Oxfordshire village rather than the Cambridge
suburb, and Chelsea, linked to the London district in a sentence about the
football club's ground. The judge linked the suburb, and answered "none" for
Chelsea because the club is not among the five candidates the search returns.
Of the default's eleven missing links, eight were abstentions on a top hit
scored under the 0.15 floor. For Kourou, Salesforce and Braga the top hit was
the right item; for Vite, NASA, Holloway, Delta and Brown Palace it was another
sense (NASA's top hit is a plant genus, Delta's a Nigerian state), with the
right item lower in the list. The judge linked all eight, Braga to the gold
item in one pass of two.

**What it could not fix.** Four names stay bare local under both values:
Jest, Lodge, Chelsea and Ottolenghi. In each, the right item is not among the
five usable candidates the one search returns. The JavaScript test framework
ranks below a gesture and a carnival club for "Jest", and the chef does not
appear for his bare surname. The judge answered "none" for all four, which is
the correct answer from the candidates it was shown. This is a limit of the
search, not of the judgement.

**The invented names.** All six (Mark Ellison, Lena Marsh, Rui Matos, Priya,
Jordan, and Vitest, a real tool with no Wikidata item) resolve correctly under
both values. Under `top_hit` that is the abstention floor dropping a
low-scored top hit, or, for Lena Marsh, no candidate at all; under `llm_judge`
the judge answered "none" for every one that had candidates.

### Braga: the judge is not deterministic across stores

Pass 1 linked Braga to `Q3344946`, the "city seat of Braga municipality", and
pass 2 to `Q83247`, the municipality, which the gold names. Both items denote
the same city, so the judgement was defensible both times, but the same input
at temperature 0 gave two answers. Within one store this cannot happen, because
the first answer is recorded and every later resolution reads it. Two separate
stores can still resolve the same name and claim differently. On this set it
happened once in 45 judged names.

### Reach and cost on ordinary prose

| Measure | Value |
|---|---:|
| Names that reached the model | 45 of 49 (92%) |
| Tokens per pass, input / output | 28,777 / 667 (pass 1) |
| Cost per pass, list price | US$0.0963 (pass 1), US$0.0967 (pass 2) |
| Cost per 100 names resolved | about US$0.20 |
| Cost per judged name | about US$0.002 |

The share is much higher than on the first suite, where most invented names
returned no candidate. On prose about notable entities, nearly every real name
has several Wikidata candidates and reaches the model. The four that did not
were GitHub Actions, Douro River and Canadian Space Agency, each with a single
well-scored candidate, and Lena Marsh, with none.

## Decision

`llm_judge` is the default from 1.168.16, decided by the owner on these two
measurements. Across both gold sets the judge gained ten or eleven
subjects and lost none:

| Gold set | `top_hit` | `llm_judge` |
|---|---:|---:|
| `prose-article-seed-001` v0.3.0, 35 subjects | 0.914 | 0.971 |
| `ordinary-prose-001` v0.1.0, 49 subjects | 0.735 | 0.898 to 0.918 |

The costs accepted with it: on ordinary prose about notable entities it is
asked about nearly every real name, which is about US$0.20 per 100 names and
one sequential call per new name during extraction, and its answer for a name
with two valid items can differ between stores. `top_hit` remains available for
a store that should make no model call during resolution, and it is what
`llm_judge` falls back to when no LLM is reachable. The four names neither value
links, whose right item is not among the five search results, are the next
limit to work on.

## Method

The column was run on its own, through `run_resolution`, the function the
benchmark runner calls for it (the second gold set through
`scripts/measure_subject_resolution.py`, which wraps the same call), with `subjects.wikidata_candidate_selection` set
per arm and a usage scope open around each pass. No extractor call was made, so
this page reports no claim metrics. Each gold subject resolves in a throwaway
store of its own, which holds an empty verdict ledger, so every judged name was
asked afresh in every pass and the two passes compare the model's answers, not
the ledger's.

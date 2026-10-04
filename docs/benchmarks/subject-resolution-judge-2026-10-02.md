# Subject resolution: the judge sees a deeper search (2026-10-02)

**A dated snapshot, as of 2026-10-02.** This page measures one change to the
subject-link judge: it is now shown seven Wikidata search hits instead of five,
and never a hit that stands for a name or a disambiguation page. It
follows the [2026-10-01 judge page](subject-resolution-judge-2026-10-01.md),
uses its two gold sets and its measurement script, and re-measures the
unchanged judge on the same day so before and after share one state of live
Wikidata.

## The failure

On the ordinary-prose gold set the judge left four real names bare local:
Jest, Lodge, Chelsea and Ottolenghi. The resolver's one search asked
`wbsearchentities` for five hits, and none of the four right items was among
them. Shown five hits, the judge answered "none" each time, which was correct
from what it saw.

Ten hits from the same search show where each item sits:

| Name | Right item | Rank in the search |
|---|---|---|
| Jest | `Q65121527`, the JavaScript test framework | 6 |
| Chelsea | `Q9616`, the football club | 6 |
| Lodge | `Q17034962`, the cookware maker | not in the first 10 |
| Ottolenghi | `Q8056413`, the chef | not in the first 10 |

## The change

**A deeper search for the judge only.** A new setting,
`subjects.wikidata_judge_search_limit` (default 7), sets how many hits the one
search request asks for under `llm_judge`. The ambiguity gate still reads only
the hits within the first five of the response, and `top_hit` and
`best_description` still search five. The set of names that reach the model is
therefore unchanged, and so is the resolution of every name that never reaches
it. The extra hits come back in the same request and pass through the
prefix-expansion filter and its alias read exactly as the first five do.

**No name or disambiguation items in what the judge is shown.** A search hit
whose description is a given name, a family name, "name", or "Wikimedia
disambiguation page" is dropped from the judge's list. Such an item never
denotes the person, place or thing a claim names. The rule reads the
description the search response already carries, so it costs no request.
When every candidate is such an item, the name stays bare local with no model
call, which is the answer the judge could only have given.

The second rule exists because the first one alone broke the
no-correct-link-lost rule, as the next section shows.

## What the deeper search did alone

The first round raised the depth and changed nothing else. Ten runs per name,
each in a fresh store, on the names whose outcome moved:

| Name, correct runs of 10 | 5 hits | 7 hits | 10 hits |
|---|---:|---:|---:|
| Jest | 0 | 10 | 10 |
| Chelsea | 0 | 10 | 10 |
| Jordan, an invented colleague | 10 | 3 | 3 |
| Brown Palace | 10 | 7 | 8 |
| Braga | 1 | 1 | 0 |

Jest and Chelsea linked every time. Jordan did not hold: "Jordan, who joined
from Salesforce last year" was linked to `Q14021944`, "unisex given name", in 7
of 10 runs at either depth, against none at five. The deeper list adds two
painters named Jordaens and Giordano, and with people in view the judge reads
the given-name item as a fit. That lost a correct outcome, so the plain depth
change could not ship.

Brown Palace's misses were a pick of `Q4976226`, a disambiguation page. Its
search returns the same five hits at any depth, so the prompts at 5, 7 and 10
hits were identical apart from the per-call nonce. Its misses are the judge's
own variation, which the depth did not cause.

A clause in the prompt saying that a given-name, family-name or disambiguation
item is never what a claim names fixed Jordan in 10 of 10 runs. It did not
hold for Priya, "my neighbour Priya", whose "female given name" item was
picked in 2 of 10 runs at seven hits and in 1 of 2 full passes. A rule the
model must follow is a rate, not a guarantee, so the clause was dropped and the
filter above was built instead. The shipped prompt is the 2026-10-01 one,
version `3614d60fbfb2fb1d`, unchanged.

## Result

| Parameter | Value |
|---|---|
| Gold sets | `ordinary-prose-001` v0.1.0 (49 names), `prose-article-seed-001` v0.3.0 (35 names) |
| Resolver | `particles` 1.169.1 with this change (released in 1.169.6), `llm_judge` |
| Judge | `anthropic/claude-sonnet-4-6`, temperature 0, prompt `3614d60fbfb2fb1d` |
| Wikidata | live, 2026-10-02 |
| Repeats | 2 full passes per arm, plus 10 runs per name on 10 names |
| Command | `uv run python scripts/measure_subject_resolution.py <gold> --selection llm_judge --judge-search-limit <n> --out <file>` |

**Ordinary prose.** "Before" is the shipped resolver: five hits, no filter.

| Arm | Accuracy | Wrong ref | Bare local | Passes match |
|---|---:|---:|---:|---|
| Before | 0.898 / 0.898 | 1 / 2 | 4 / 3 | no |
| 5 hits, filter | 0.918 / 0.918 | 0 / 0 | 4 / 4 | yes |
| **7 hits, filter (shipped)** | **0.959 / 0.959** | **0 / 0** | **2 / 2** | **yes** |
| 10 hits, filter | 0.959 / 0.939 | 0 / 1 | 2 / 2 | no |

The two values per cell are the two passes. Against the before arm, the
shipped arm links Jest and Chelsea and holds Braga on the municipality the gold
names, and it loses no correct outcome in either pass. The two names left bare
local are Lodge and Ottolenghi.

- **Chelsea** under the before arm was bare local in one pass and linked to the
  London district in the other: shown the district and four places, the judge
  sometimes took the district for "Chelsea's ground". Ten runs gave 7 bare and
  3 wrong. With the club in view it linked the club 10 times of 10.
- **Braga** has two items for the one city, the municipality (the gold) and its
  city seat. The filter drops a family-name hit from its list, and with that
  gone the judge chose the municipality in 10 of 10 runs at five and seven
  hits, against 1 of 10 before. At ten hits, which add three civil parishes, it
  chose the city seat in 2 of 10.

**The first suite.** All four arms scored 0.971 (34 of 35) in both passes, and no
correct link was lost. Its one miss is Lantern, an invented payroll vendor,
which the judge links to a wrong item in every arm; the 2026-10-01 page explains
why the claim gives it nothing to go on.

**Ten runs per name at seven hits, with the filter:** all ten names held their
outcome 10 times of 10. That covers the six invented names (Mark Ellison,
Vitest, Lena Marsh, Rui Matos, Priya and Jordan), all left without a link, and
Jest, Chelsea, Braga and Brown Palace, all linked to the gold item.

### Reach and cost

The gate is unchanged, so the same names reached the model in every arm: 45 of
49 on ordinary prose, 14 of 35 on the first suite.

| Measure, per judged name | Before | 7 hits, filter | 10 hits, filter |
|---|---:|---:|---:|
| Input tokens, ordinary prose | 636 | 659 | 711 |
| Input tokens, first suite | 615 | 637 | 679 |
| Cost per 100 names, ordinary prose | US$0.199 | US$0.201 | US$0.217 |
| Cost per 100 names, first suite | US$0.081 | US$0.084 | US$0.089 |

Seven hits add about 20 input tokens per judged name, because the filter takes
back some of what the two extra hits add. The search itself is the same one
request; a deeper response is a few hundred bytes more.

### Seven, not ten

Ten hits link the same two names as seven and cost 40 to 50 more input tokens
per judged name. In the measured runs they also did worse on the one name with
two valid items: Braga went to the city seat in one full pass and in 2 of 10
repeats, against none at seven. Seven reaches the rank-6 items with one hit to
spare.

## The two names a deeper search cannot reach

Lodge and Ottolenghi are not in the first ten hits for the bare name. The
register entry proposed a second query built from the claim: the name plus a
word that says what kind of thing it is. Its premise was measured before
anything was built, by searching for the name plus each content word of its
sentence and checking whether the right item came back in the top five.

| Name | Claim words tried | Entity search finds it | Full-text search finds it |
|---|---:|---|---|
| Lodge | 8 | with no word | only with "cast-iron" |
| Ottolenghi | 1 ("recipe") | with no word | with no word |
| Jest | 7 | with no word | with no word |
| Chelsea | 4 | with no word | only with "Fulham" |

The word that finds Ottolenghi ("chef") and the word that finds Lodge through
the entity search ("cookware", or the label "Lodge Manufacturing") are not in
the sentence. A query drawn from the claim therefore buys Lodge at most, only
through the full-text endpoint, which returns no labels or descriptions and so
needs a third request, and only if the right word of eight is chosen. It was
not built. The two names stay open in the register with this measurement.

## Upgrading

The candidate set is part of the key the judge's verdicts are recorded under.
A name whose search returns more than five hits, or whose list included a name
or disambiguation item, has a new key, so the first resolution of it after the
upgrade asks the model once more. On ordinary prose that is nearly every
judged name. The old verdict rows stay in the ledger and are never read again;
the ledger only grows by one row per judged name each time its candidate set
changes, as it already did whenever Wikidata's search response changed.

## Method

Each arm ran the column on its own through
`scripts/measure_subject_resolution.py`, with
`subjects.wikidata_judge_search_limit` set per arm, and every gold subject was
resolved in a throwaway store with an empty verdict ledger, so every judged
name was asked afresh in every pass. The ten-run repeats used the same
function, each repeat in its own store. The before arm was measured on the
shipped code, the rounds with the clause and the filter on the code they
describe, all on 2026-10-02.

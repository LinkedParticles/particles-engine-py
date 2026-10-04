# Predicate canonicalisation: how far the free-text predicates collapse (2026-10-01)

**A dated snapshot, as of 2026-10-01.** This page measures the first step a
predicate profile needs: turning the free-text predicates of the
owner's live store into a smaller set of canonical predicates. It compares two
methods that cost no LLM call. Both were run read-only, on a copy of the store
taken with SQLite's backup command, over the listing that `particles query
--predicates` prints.

## What was measured

The store has grown since the 2026-09-30 count the register row quotes
(6,358 predicates over 11,074 claims).

| | 2026-10-01 |
|---|---|
| ACTIVE particles | 37,382 |
| Particles carrying a structured claim | 13,817 |
| Claims the listing counts | 13,387 |
| Distinct predicates | 7,504 |
| Kind | all `TOKEN`, none `URI` |
| Predicates that occur once | 6,084 (81%) |
| Share of claims under the 20 commonest predicates | 13.0% |

Two methods:

- **Normalisation.** Lowercase the predicate, drop articles, drop leading
  auxiliaries and modals (`is`, `was`, `can`, and `has` when a participle
  follows it), then lemmatise the head verb. `moved` becomes `move` and
  `lives` becomes `live`. Negation and prepositions are kept, so `does not
  support` stays apart from `supports`, and `moved to` stays apart from
  `moved from`. Lemmas are taken from the listing's own head verbs where one
  fits, with an irregular-verb table and suffix rules as the fallback.
- **Embedding clustering.** Encode each predicate with the store's own local
  encoder (`all-MiniLM-L6-v2`) and cluster with average-linkage agglomerative
  clustering on cosine distance. The cut is made at cosine similarity 0.85 and
  at 0.75.

The script is `scripts/measure_predicate_clusters.py`. It reads the listing,
not the store:

```bash
uv run particles query --predicates > predicates.txt
uv run python scripts/measure_predicate_clusters.py predicates.txt \
    --threshold 0.85 --threshold 0.75 --json clusters.json
```

## Cluster counts

| Method | Clusters | Reduction | Clusters with 2+ terms | Claims in those | Largest cluster |
|---|---|---|---|---|---|
| None (as stored) | 7,504 | | | | |
| Normalisation | 6,792 | 9.5% | 520 | 5,030 (37.6%) | 12 terms |
| Embedding at 0.85 | 6,336 | 15.6% | 869 | 5,192 (38.8%) | 14 terms |
| Embedding at 0.75 | 5,191 | 30.8% | 1,303 | 8,406 (62.8%) | 26 terms |

No method comes close to a vocabulary a reviewer could confirm term by term.
The long tail is the reason: four predicates in five occur once, and most of
those are one-off phrasings ("passed with", "has passing unit test count")
that no surface method can tie to a commoner form without knowing what they
mean.

## Where the methods go wrong

A merge is only useful to a profile if every member means the same relation
in the same direction. A cardinality rule applied to a cluster that mixes
`renamed to` with `renamed from` would treat the old name and the new name as
one value. The check below counts clusters with 2+ terms that mix a negated
form with a plain one, or a `to` / `into` form with a `from` form.

| Method | Mix negated and plain | Claims | Mix `to` and `from` | Claims |
|---|---|---|---|---|
| Normalisation | 0 | 0 | 0 | 0 |
| Embedding at 0.85 | 23 | 151 | 26 | 221 |
| Embedding at 0.75 | 73 | 1,007 | 27 | 282 |

The script flagged two normalisation clusters as mixed polarity. Both are
false alarms, since every member is a negative form (`lacks` / `lack` /
`lacked`, and the bare `is not` / `was not` / `does not`), so the table counts
them as zero. The embedding errors are real. At 0.85 the encoder merges
`was renumbered from` with `was renumbered to`, `renamed to` with `renamed
from`, `version bumped from` with `version bumped to`, and `isolate` with `do
not isolate`. At 0.75 it also merges `is` with `is not`, `includes` with `does
not include`, and `returns` with `never returns`.

Normalisation makes the opposite trade. Every merge it makes is a tense,
number or modal variant of one verb with the same preposition: `has status` /
`had status` / `have status`, `contains` / `contained` / `contain`. It cannot
see a synonym (`is located at` and `lives in` stay apart), and it merges
modals that change meaning: `passes` and `must pass` land together, though
one is a fact and the other a requirement. Its largest cluster is the bare
copula (`is`, `was`, `are`, `is a`, `can be`, `must be`…), which is not a
relation at all.

## The residence family, worked through

The register row cites `lives in` (4), `moved to` (7), `was moved to` (8) and
`moved from` (6) as the residence slot spread over four phrasings, the 4 to 9
way variance measured when the update rung was designed. On
today's copy those predicates land as follows:

| Predicate | Normalisation | Embedding at 0.85 | Embedding at 0.75 |
|---|---|---|---|
| `lives in` (4) | with `live in`, `lived in` | alone | with `live in`, `lived in` |
| `moved to` (7) | with `was moved to`, `must be moved to` | with `moved from`, `moved` | one cluster of 9: `to`, `from`, `into`, `by`, `under` |
| `was moved to` (11) | with `moved to`, `must be moved to` | with `was moved from`, `was moved into` | the same cluster of 9 |
| `moved from` (6) | with `was moved from` | with `moved to`, `moved` | the same cluster of 9 |

Reading the claims behind them changes the example. In this store none of
the four is about where a person lives. `moved to` and `was moved to` record
a decision record's file moving from `proposed/` to `active/`, or a register
row moving to its closed section. `moved from` names the directory it left.
`lives in` says where a module or setting is kept ("the scoring package lives
in the Client layer"). The residence slot comes from persona stores such as
the rot and LongMemEval ones, where the subject is a person.

Two conclusions follow for the design:

- **The slot is a predicate on a class, not a predicate.** `move to` on a
  decision record is one value at a time (a record sits in one directory),
  and on a person it is residence, also one at a time but a different slot.
  `lives in` on a module and `lives in` on a person are two slots. A
  canonical predicate has to be keyed by the subject's class, which is the
  per-class co-occurrence the vocabulary report would carry.
- **Direction is part of the predicate.** `move to` names the new value and
  `move from` names the old one. Normalisation keeps them apart, which is
  right. A profile can then say that `move to` gives a slot its current value
  and `move from` records a past one, which no merged cluster can say.

## Reading

- **Normalisation is the safe first step.** On this store it merges nothing
  it should not, and it is deterministic, free and explainable. It is also
  weak: it cuts the vocabulary by a tenth and leaves 6,272 singletons.
- **Embedding similarity is a suggestion source, never a merge.** At 0.85 it
  finds real aliases normalisation misses (`has passing test count` and its
  eight variants, `was pushed to` / `pushed to` / `were pushed to`), and in
  the same pass it joins opposite directions and opposite polarities. Its
  candidates are worth ranking for a reviewer by the claims they would cover.
  Applied without review, it would hand a cardinality rule the wrong values.
- **Canonicalisation will not cover the tail.** Even at 0.75, where the
  errors above are frequent, 3,888 clusters are singletons. A profile can only
  ever govern the head of the distribution, and every claim outside it keeps
  the probe path. That is why the probe itself was changed first (see
  [the slot check's kind](../benchmarks.md#the-slot-checks-kind-2026-10-01)).

## Added 2026-10-02: the core form, and how far a profile reaches

The predicate profiles proposal needs a canonical form computed
from one predicate alone. The normalisation above picked each lemma by
checking which base forms the listing contains, so adding one predicate to
the store could change another's form. The SDK's function instead reduces
the head verb to an inflectional stem (`moved`, `moves`, `moving` all
become `mov`). On the same listing:

| | Normalisation above | Core stem |
|---|---|---|
| Clusters | 6,792 | 6,759 |
| Merges in common | 1,039 | 1,039 |
| Merges only this one makes | 3 | 50 |
| Clusters mixing negated and plain, or `to` and `from` | 0 | 0 |

The stem lost three merges, `had run for` with its variants, which treating
`run` as its own participle fixes. The 50 it adds are inflections the
attestation rule missed, such as `created` and `creates`.

How far could a profile reach on this store? On a read-only copy taken
2026-10-02, the update sweep's own candidate search, with no LLM call,
finds 1,985 same-subject pairs it would consider.

| Filter | Pairs |
|---|---|
| Both claims carry a triple | 578 |
| Both triples about the same resolved subject | 450 |
| That subject has a class | 62 (3.1%) |
| Both predicates share a canonical form | 11 (0.55%), on 4 keys |

2,363 of 9,850 subjects have a class, nearly all from the code-artifact
authority (files, records, symbols, commands). A profile can only act on a
classed pair, so on this store it could settle at most 62 of the pairs the
sweep considers, and 11 without reviewed aliases.

## Added 2026-10-02: the vocabulary proposal step

The SDK now stores reviewed predicate rulings in a vocabulary document
(`particles vocab`), and its proposal step suggests alias and profile
candidates from a store's own predicates. It follows the design above: a
canonical predicate is a normalised form on a subject class, so only claims
whose triple is about a classed subject take part, and an alias candidate is an
embedding cluster at cosine 0.85 within one class, split so that no candidate
mixes a negated form with a plain one, or a `to` form with a `from` form. A
candidate is only a suggestion. Nothing enters the document until a reviewer
confirms it, and nothing was confirmed here.

It was run on a fresh read-only copy of the owner's store, taken with SQLite's
backup command and migrated, with no LLM call:

```bash
particles vocab create owner --prefix own --namespace https://vocab.example.org/owner/
particles vocab propose owner --top 30 --json proposals.json
```

| | 2026-10-02 |
|---|---|
| Structured claims | 13,817 |
| Distinct predicates | 7,718 |
| Claims about a classed subject | 3,740 (27%) |
| Subject classes those claims fall in | 4 (`artifact:` record, file, symbol, command) |
| Predicate and class pairs | 2,293 |
| Normalised forms on a class | 2,173 |
| Alias candidates | 96 (79 of two forms, 17 of three to five) |
| Distinct predicates in an alias candidate | 236 (3.1% of 7,718) |
| Claims an alias candidate covers | 569 |
| Profile candidates | 2,173, of which 87 cover 5 or more claims |
| Profile candidates with a past-value partner | 7 (`renumber to` beside `renumber from`, `mov to` beside `mov from`) |

The store grew by 214 predicates since 2026-10-01. The run took 29 seconds,
most of it loading the encoder.

Three readings:

- **The class requirement is what limits reach.** Seven claims in ten have no
  canonical predicate, because the subject their triple is about has no class:
  7,137 claims are about a bare local subject, 2,486 name no resolved subject at
  all, and 427 are about a subject linked to Wikidata. Class-free clustering of
  the same predicates at 0.85 found 869 multi-term clusters on 2026-10-01;
  within a class it finds 96. Recording a Wikidata subject's class would reach
  only the 427. The larger lever is a class for the bare local subjects, which
  nothing assigns today.
- **The split by direction and polarity holds.** No candidate below joins
  `renamed to` with `renamed from`, or a negated form with a plain one, which
  were the embedding errors measured on 2026-10-01. Direction partners surface
  instead as a profile candidate's past-value partner.
- **Many candidates still need a no.** Rows 10 (`passed` with `passed after`),
  12 (`was closed as` with `was closed in`), 19 (`shipped` with `shipped as`)
  and 26 (`states in §3.2` with `states in §2.4`) join forms whose
  preposition, or a value written into the predicate, changes the relation.
  A reviewer declines those, and a declined candidate is never proposed again.

### The top 30 alias candidates

By claims covered. Members are surface predicates with their claim counts; the
similarity is the lowest pairwise cosine in the candidate (average linkage can
admit a pair below 0.85).

| # | Claims | Predicates | Min sim. | Class | Members (claims) |
|---|---|---|---|---|---|
| 1 | 140 | 3 | 0.85 | record | `has status` (136), `had status` (3), `has status/version` (1) |
| 2 | 38 | 2 | 0.89 | file | `implements` (37), `correctly implements` (1) |
| 3 | 25 | 2 | 0.89 | record | `specifies` (24), `specifies that` (1) |
| 4 | 19 | 2 | 0.91 | record | `status` (13), `status is` (6) |
| 5 | 14 | 6 | 0.88 | record | `was minted for` (4), `was minted` (3), `was minted as` (3), `minted` (2), `was minted because` (1), `was minted in` (1) |
| 6 | 13 | 3 | 0.90 | record | `has spec_impact` (10), `has spec_impact of` (2), `has likely spec_impact of` (1) |
| 7 | 13 | 5 | 0.92 | record | `was promoted to` (7), `promoted to` (2), `was promoted as` (2), `is promoted to` (1), `was promoted on` (1) |
| 8 | 10 | 2 | 0.88 | record | `tracks` (9), `tracks question` (1) |
| 9 | 8 | 3 | 0.96 | file | `contains section` (6), `contains sections` (1), `will contain section` (1) |
| 10 | 8 | 4 | 0.87 | file | `passed after` (3), `passed` (2), `passes` (2), `passed with` (1) |
| 11 | 8 | 3 | 0.88 | record | `became active at version` (4), `is active at version` (3), `active at version` (1) |
| 12 | 8 | 5 | 0.81 | record | `was closed as` (4), `is closed as` (1), `was closed in` (1), `was closed instead of` (1), `was closed to` (1) |
| 13 | 8 | 6 | 0.95 | record | `is marked` (2), `marked` (2), `is marked as` (1), `was marked` (1), `was marked as` (1), `will be marked` (1) |
| 14 | 7 | 3 | 0.93 | file | `declares prefix` (5), `declares only prefixes` (1), `declares prefixes` (1) |
| 15 | 7 | 2 | 0.89 | file | `was edited` (4), `was edited in` (3) |
| 16 | 7 | 3 | 0.85 | record | `landed on` (5), `landed at` (1), `was landed on` (1) |
| 17 | 7 | 2 | 0.92 | symbol | `returns result block type` (5), `produces result block type` (2) |
| 18 | 6 | 4 | 0.87 | file | `has passing test count` (3), `has passing test count of` (1), `has passing tests count` (1), `passing test count` (1) |
| 19 | 6 | 3 | 0.87 | record | `shipped` (4), `shipped as` (1), `ships as` (1) |
| 20 | 6 | 5 | 0.89 | record | `shipped as version` (2), `shipped at version` (1), `shipped at versions` (1), `shipped in version` (1), `was shipped as version` (1) |
| 21 | 6 | 2 | 0.94 | symbol | `has field` (4), `has fields` (2) |
| 22 | 5 | 2 | 0.96 | file | `asserts` (3), `asserts on` (2) |
| 23 | 5 | 4 | 0.87 | file | `auto-merged` (2), `auto-merged cleanly during` (1), `auto-merged cleanly with` (1), `auto-merged with` (1) |
| 24 | 5 | 2 | 0.95 | record | `establishes rule` (3), `establishes rule that` (2) |
| 25 | 5 | 2 | 0.93 | record | `has trigger status` (4), `trigger status` (1) |
| 26 | 5 | 4 | 0.87 | record | `states in §3.2` (2), `states in §2.4` (1), `states in §2.9` (1), `states in §7` (1) |
| 27 | 5 | 2 | 0.91 | record | `trigger fired on` (4), `trigger fired due to` (1) |
| 28 | 4 | 2 | 0.96 | file | `version bumped to` (2), `was bumped to version` (2) |
| 29 | 4 | 3 | 0.90 | file | `was committed as` (2), `committed as` (1), `was committed with` (1) |
| 30 | 4 | 2 | 0.85 | file | `enforces` (3), `enforces policy` (1) |

## Added 2026-10-03: the reviewed profiles meet no sweep pair

The owner reviewed the 60 proposals on their own store (a document `personal`
with 16 aliases and 28 profiles). On a fresh read-only copy with that document
adopted, a measurement script (since removed with the proposal) lined the profiles up against the
1,985 same-subject pairs the update sweep would consider, before any LLM call:

| Why a pair cannot reach a profile | Pairs |
|---|---|
| One side has no structured claim | 1,407 |
| Subject has no class | 388 |
| The two claims are about different or unresolved subjects | 128 |
| Classed, but the predicate has no profile | 57 |
| One side profiled, the other a different predicate | 5 |
| Both sides under one profile | 0 |

On this store the profiles would therefore save no check and prevent no
retirement, whatever the checks answer. The predicates with the most claims, which
the proposal step ranks first, rarely appear on both sides of a sweep pair:
the sweep mostly pairs two different relations about one record or file. The
only classed pairs that share a predicate are `restates` on files (6) and
`self-certifies` on commands (3).

As a ceiling test, the only two shared predicates were profiled on the copy
too, and the update checks were run on all 1,985 pairs (`claude-haiku-4-5`,
about US$3.2). The contradiction check confirmed 71 pairs, so saving 4 checks
would have met the 5% bar. Every one of the 9 pairs the two profiles cover is
the same claim worded twice, which the contradiction check does not confirm.
The profiles saved no check and prevented no retirement.

The same bound was then taken on the owner's older development store, whose
claims come from Wikipedia articles, Reddit posts and journal entries. Its
claims had no structured triples and its subjects no classes, so on a copy the
claims in the sweep pairs about a Wikidata-linked subject were structured and
those subjects classed by their Wikidata "instance of" value. Of the 3,407
pairs the sweep would consider, 36 then had triples about one classed subject,
and in 35 of them the two claims use different predicates. The 6 pairs a
profile could reach, with reviewed aliases, all state one fact twice ("Tim
Berners-Lee invented the web" and "is the creator of the Web"), and the
contradiction check confirms none of them.

## The top 30 clusters

By claims covered, members by claim count, at most eight shown per cluster.

#### normalise

| # | Claims | Terms | Members (claims) |
|---|---|---|---|
| 1 | 359 | 12 | `is` (257), `was` (41), `are` (31), `is a` (10), `were` (6), `has been` (4), `can be` (2), `must be` (2), +4 more |
| 2 | 239 | 3 | `has status` (224), `had status` (8), `have status` (7) |
| 3 | 172 | 3 | `contains` (153), `contained` (14), `contain` (5) |
| 4 | 149 | 3 | `is located at` (140), `located at` (8), `are located at` (1) |
| 5 | 141 | 4 | `requires` (131), `require` (7), `required` (2), `would require` (1) |
| 6 | 113 | 4 | `passed` (60), `passes` (49), `must pass` (3), `pass` (1) |
| 7 | 103 | 7 | `uses` (86), `use` (8), `used` (4), `must use` (2), `has used` (1), `should use` (1), `will use` (1) |
| 8 | 101 | 3 | `covers` (97), `covered` (3), `cover` (1) |
| 9 | 99 | 4 | `implements` (95), `implemented` (2), `implement` (1), `is implemented` (1) |
| 10 | 66 | 4 | `includes` (55), `include` (5), `included` (3), `must include` (3) |
| 11 | 63 | 4 | `has` (49), `had` (9), `have` (4), `would have` (1) |
| 12 | 59 | 3 | `returns` (51), `returned` (7), `must return` (1) |
| 13 | 53 | 2 | `concerns` (52), `concern` (1) |
| 14 | 52 | 2 | `has version` (45), `had version` (7) |
| 15 | 47 | 3 | `lacks` (43), `lack` (2), `lacked` (2) |
| 16 | 44 | 3 | `defines` (42), `define` (1), `defined` (1) |
| 17 | 44 | 6 | `produced` (24), `produces` (16), `can produce` (1), `is producing` (1), `produce` (1), `will produce` (1) |
| 18 | 41 | 2 | `addresses` (36), `addressed` (5) |
| 19 | 41 | 4 | `reports` (21), `reported` (18), `must report` (1), `report` (1) |
| 20 | 36 | 3 | `added` (20), `adds` (15), `will add` (1) |
| 21 | 31 | 1 | `has model ID` (31) |
| 22 | 31 | 1 | `has title` (31) |
| 23 | 29 | 2 | `provides` (28), `provide` (1) |
| 24 | 29 | 5 | `runs` (19), `ran` (5), `is running` (2), `was run` (2), `will run` (1) |
| 25 | 28 | 6 | `carries` (16), `carry` (6), `carried` (3), `can carry` (1), `carries a` (1), `must carry` (1) |
| 26 | 27 | 2 | `result` (26), `results` (1) |
| 27 | 27 | 1 | `specifies` (27) |
| 28 | 26 | 5 | `remains` (17), `remain` (5), `remained` (2), `must remain` (1), `should remain` (1) |
| 29 | 25 | 1 | `has URL` (25) |
| 30 | 25 | 4 | `is located in` (21), `are located in` (2), `located in` (1), `was located in` (1) |

#### embed@0.85

| # | Claims | Terms | Members (claims) |
|---|---|---|---|
| 1 | 257 | 1 | `is` (257) |
| 2 | 231 | 2 | `has status` (224), `have status` (7) |
| 3 | 155 | 2 | `contains` (153), `contains only` (2) |
| 4 | 149 | 3 | `is located at` (140), `located at` (8), `are located at` (1) |
| 5 | 132 | 2 | `requires` (131), `requires only` (1) |
| 6 | 98 | 2 | `covers` (97), `cover` (1) |
| 7 | 96 | 2 | `implements` (95), `correctly implements` (1) |
| 8 | 94 | 2 | `uses` (86), `use` (8) |
| 9 | 68 | 4 | `passed` (60), `passed with` (4), `passed after` (3), `passed for` (1) |
| 10 | 60 | 2 | `includes` (55), `include` (5) |
| 11 | 53 | 2 | `concerns` (52), `concern` (1) |
| 12 | 52 | 3 | `passes` (49), `passes in` (2), `passes with` (1) |
| 13 | 51 | 1 | `returns` (51) |
| 14 | 49 | 1 | `has` (49) |
| 15 | 47 | 3 | `has version` (45), `has available version` (1), `has release version` (1) |
| 16 | 45 | 6 | `was renumbered from` (20), `was renumbered to` (16), `were renumbered to` (6), `renumbered from` (1), `renumbered to` (1), `was not renumbered to` (1) |
| 17 | 43 | 1 | `lacks` (43) |
| 18 | 42 | 1 | `defines` (42) |
| 19 | 41 | 1 | `was` (41) |
| 20 | 40 | 2 | `produced` (24), `produces` (16) |
| 21 | 37 | 2 | `status` (25), `status is` (12) |
| 22 | 36 | 1 | `addresses` (36) |
| 23 | 36 | 8 | `was bumped to version` (11), `bumped version to` (6), `version bumped from` (6), `version bumped to` (5), `was re-bumped to version` (3), `bumped to version` (2), `re-bumped version to` (2), `was bumped from version` (1) |
| 24 | 31 | 1 | `are` (31) |
| 25 | 31 | 1 | `has model ID` (31) |
| 26 | 31 | 1 | `has title` (31) |
| 27 | 29 | 7 | `was pushed to` (18), `pushed to` (5), `were pushed to` (2), `pushed` (1), `pushed with` (1), `was pushed` (1), `was pushed with` (1) |
| 28 | 28 | 1 | `provides` (28) |
| 29 | 28 | 2 | `specifies` (27), `specifies that` (1) |
| 30 | 27 | 14 | `has passing test count` (8), `had passing test count of` (3), `has passing test count of` (3), `passed test count` (2), `passing test count` (2), `had passing test count` (1), `had passing tests count` (1), `had passing tests count of` (1), +6 more |

#### embed@0.75

| # | Claims | Terms | Members (claims) |
|---|---|---|---|
| 1 | 278 | 7 | `has status` (224), `status` (25), `status is` (12), `had status` (8), `have status` (7), `reached status` (1), `status marked as` (1) |
| 2 | 264 | 2 | `is` (257), `is not` (7) |
| 3 | 180 | 12 | `contains` (153), `contains function` (10), `contain` (5), `contains only` (2), `contains test` (2), `contains tests for` (2), `contains check` (1), `contains class` (1), +4 more |
| 4 | 149 | 3 | `is located at` (140), `located at` (8), `are located at` (1) |
| 5 | 133 | 3 | `requires` (131), `requires no` (1), `requires only` (1) |
| 6 | 109 | 4 | `implements` (95), `implements verb` (12), `correctly implements` (1), `implements method` (1) |
| 7 | 98 | 2 | `covers` (97), `cover` (1) |
| 8 | 94 | 2 | `uses` (86), `use` (8) |
| 9 | 72 | 6 | `passed` (60), `passed with` (4), `passed after` (3), `passed on` (3), `passed for` (1), `passed in` (1) |
| 10 | 63 | 3 | `includes` (55), `include` (5), `does not include` (3) |
| 11 | 62 | 3 | `has` (49), `had` (9), `have` (4) |
| 12 | 60 | 6 | `passes` (49), `passes on` (6), `passes in` (2), `pass` (1), `passes through` (1), `passes with` (1) |
| 13 | 56 | 4 | `returns` (51), `always returns` (3), `never returns` (1), `returns to` (1) |
| 14 | 53 | 2 | `concerns` (52), `concern` (1) |
| 15 | 52 | 3 | `defines` (42), `is defined as` (9), `defined` (1) |
| 16 | 51 | 26 | `merged` (8), `is merged into` (5), `was merged via` (5), `merged into` (3), `merged via` (3), `was merged` (3), `was merged into` (3), `was merged on` (3), +18 more |
| 17 | 47 | 3 | `has version` (45), `has available version` (1), `has release version` (1) |
| 18 | 47 | 3 | `lacks` (43), `lack` (2), `lacked` (2) |
| 19 | 47 | 8 | `was renumbered from` (20), `was renumbered to` (16), `were renumbered to` (6), `renumbered from` (1), `renumbered to` (1), `version renumbered to` (1), `was not renumbered to` (1), `were renumbered, count` (1) |
| 20 | 45 | 12 | `was bumped to version` (11), `bumped version to` (6), `version bumped from` (6), `version bumped to` (5), `was bumped to` (5), `was re-bumped to version` (3), `bumped to version` (2), `re-bumped version to` (2), +4 more |
| 21 | 43 | 2 | `was` (41), `was not` (2) |
| 22 | 42 | 4 | `produced` (24), `produces` (16), `is produced by` (1), `is producing` (1) |
| 23 | 40 | 23 | `has passing test count` (8), `had passing test count of` (3), `has passing test count of` (3), `test suite passed count` (3), `has passing unit test count` (2), `has unit test count` (2), `passed test count` (2), `passing test count` (2), +15 more |
| 24 | 38 | 8 | `excludes` (17), `is excluded from` (12), `are excluded from` (3), `excluded from` (2), `excluded` (1), `excluded by` (1), `is excluded for` (1), `was excluded from` (1) |
| 25 | 36 | 1 | `addresses` (36) |
| 26 | 35 | 5 | `result` (26), `result is` (4), `result was` (2), `results in` (2), `results` (1) |
| 27 | 34 | 3 | `has model ID` (31), `has API model id` (2), `has api model id` (1) |
| 28 | 34 | 9 | `was moved to` (11), `moved to` (7), `moved from` (6), `was moved from` (3), `moved by` (2), `was moved into` (2), `moved` (1), `was moved by` (1), +1 more |
| 29 | 33 | 4 | `supports` (23), `supports option` (5), `supports only` (3), `support` (2) |
| 30 | 33 | 9 | `was committed as` (18), `committed as` (4), `was committed to` (3), `is committed to` (2), `was committed at` (2), `committed at` (1), `had committed` (1), `was committed on` (1), +1 more |

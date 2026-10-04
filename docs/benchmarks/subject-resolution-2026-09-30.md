# Subject resolution: baseline and the embedding-only selector (2026-09-30)

**A dated snapshot, as of 2026-09-30.** This page records how often the subject
resolver links a name to the right external entity, measured for the first
time. The baseline was taken on the resolver as it shipped in 1.168.1, before
any change. The same page then measures the first candidate fix,
choosing among Wikidata's search candidates by how well each description
matches the claim text, and records that it made the resolver
worse.

!!! info "Later"
    The Postgres failure below was fixed on 2026-10-01; see
    [the prefix-expansion fix](subject-resolution-2026-10-01.md). Harbor and
    Rust were fixed by an optional model judge the same day; see
    [the judged selector](subject-resolution-judge-2026-10-01.md).

It measures the **subject resolver** (the cascade that turns an extracted name
such as "Harbor" into a Subject, consulting the local store, then in-name
identifiers, then Wikidata, then falling back to a bare local Subject). It does
not measure the extractor and it is not the agent-memory benchmark on the
[Benchmarks](../benchmarks.md) page.

!!! note "What the column can and cannot see"
    Twenty-two of the thirty-five gold subjects are invented entities with no
    Wikidata item, and most of them return no Wikidata search hit at all, so
    they score correct for any resolver that does not invent a link. The
    column's ability to separate one resolver from another rests on the other
    thirteen and on the handful of invented names that are also ordinary
    words. Read a change of one subject as a change of 2.9 points, and read the
    per-subject table rather than the headline alone.

## What was measured

| Parameter | Value |
|---|---|
| Suite | `prose-article-seed-001` v0.3.0, its `gold_subjects` block |
| Gold subjects | 35: 13 with a Wikidata QID, 22 with no item (`ref: null`) |
| Resolver | the production cascade, `particles` 1.168.1 |
| Context per subject | its gold mention (the first gold claim naming it) |
| Store | a throwaway SQLite store per subject |
| Encoder | `all-MiniLM-L6-v2` (the default) |
| Thresholds | `subjects.external_link_abstain_threshold` 0.15, `subjects.wikidata_link_suppress_threshold` 0.25 (defaults) |
| Wikidata | live `wbsearchentities`, 2026-09-30 |
| Repeats | 2, byte-identical per-subject results |

The two cases v0.3.0 adds exist for this column. `web-article-003` is a trade
brief about two invented companies named with ordinary English words, Harbor
and Lantern: the recorded "POET" failure in a form a fixture can hold.
`web-article-004` is an engineering post naming real software (Go,
Python, Postgres, Kafka, Rust, Signal, Notion) whose names collide with other
Wikidata items.

Which names are gold follows one rule: every proper-named entity a gold claim
names, whether or not today's extractor emits it as a subject. Each subject is
resolved from its mention rather than from the extractor's output, so
extraction sampling never moves the column.

A link counts **at any confidence**. Exporters hide a link below 0.25 and lint
flags it, but the store's join (`find_by_external_ref`) reads no confidence, so
the next mention that resolves to the same QID lands on that Subject either
way. Counting only the displayed links would score this baseline 0.771 and
hide one wrong link and five right ones.

## Baseline

| Metric | Value | Count |
|---|---:|---:|
| `resolution_accuracy` | **0.886** | 31 / 35 |
| `resolution_wrong_ref` | 0.086 | 3 |
| `resolution_bare_local` | 0.029 | 1 |

The four failures, and why each happens:

- **Harbor → `Q283202`, "sheltered body of water" (0.30).** This is the
  recorded "POET" failure exactly. The company has no item, the common noun is the top hit, and
  its description scores 0.30 against "Harbor has acquired Lantern.", above
  both thresholds. The Subject is renamed "harbor".
- **Lantern → `Q862454`, "fixed or portable enclosed lighting device" (0.20).**
  The same failure in the band between the two thresholds: the link is hidden
  from exports and flagged by lint, but it is stored, and the Subject is
  renamed "lantern".
- **Postgres → `Q28975208`, "discontinued database software, predecessor to
  PostgreSQL" (0.33).** Wikidata's top hit is PostgreSQL (`Q192490`), the right
  answer. The resolver's prefix-expansion filter discards it, because
  "PostgreSQL" continues "Postgres" with a letter, the pattern the filter uses
  to reject "micrograd" → "Microgradients …". The next hit is taken instead.
- **Rust → no link.** The top hit is right (`Q575650`, the programming
  language), but its description scores below 0.15 against "Ostrander Freight
  considered Rust and chose Go for the billing service rewrite because …", so
  the resolver abstains.

Five correct links (Python 0.20, Go 0.23, Kafka 0.20, Signal 0.16, Notion 0.24)
sit in the band where exporters hide them. They are correct and they join, so
they count.

### Every gold subject

| Subject | Gold | Stored ref (confidence) | Outcome |
|---|---|---|---|
| Fernwood Systems | none | none | correct |
| SHA-256 | `Q110651361` | `Q110651361` (0.32) | correct |
| Halifax, Nova Scotia | `Q2141` | `Q2141` (0.58) | correct |
| Dana Okonkwo | none | none | correct |
| Halcyon Grid Cooperative | none | none | correct |
| Duluth, Minnesota | `Q485708` | `Q485708` (0.43) | correct |
| Pike Lake array | none | none | correct |
| Tomas Ilves | none | none | correct |
| Priya Raghunathan | none | none | correct |
| Bracken & Doyle | none | none | correct |
| Salt Marsh Notes | none | none | correct |
| Maren Kaldestad | none | none | correct |
| Kvitholm Institute | none | none | correct |
| Meridian Rose | none | none | correct |
| Stray Light Foundation | none | none | correct |
| Bergen | `Q26793` | `Q26793` (0.27) | correct |
| Jonas Brekke | none | none | correct |
| Tidewrack Quarterly | none | none | correct |
| Harbor | none | `Q283202` (0.30) | **wrong ref** |
| Lantern | none | `Q862454` (0.20) | **wrong ref** |
| Manchester | `Q18125` | `Q18125` (0.56) | correct |
| Leeds | `Q39121` | `Q39121` (0.27) | correct |
| Odile Fanshawe | none | none | correct |
| Wendell Asquith | none | none | correct |
| Counting House Review | none | none | correct |
| Ostrander Freight | none | none | correct |
| Python | `Q28865` | `Q28865` (0.20) | correct |
| Go | `Q37227` | `Q37227` (0.23) | correct |
| Postgres | `Q192490` | `Q28975208` (0.33) | **wrong ref** |
| Kafka | `Q16235208` | `Q16235208` (0.20) | correct |
| Rust | `Q575650` | none | *bare local* |
| Signal | `Q19718090` | `Q19718090` (0.16) | correct |
| Notion | `Q60747998` | `Q60747998` (0.24) | correct |
| Ines Marchetti | none | none | correct |
| Rewriting billing in Go | none | none | correct |

## After: choosing the candidate by its description

The selector, `subjects.wikidata_candidate_selection: best_description`,
changes one step. Where the resolver took Wikidata's top search hit, it now
scores every candidate the same search returns (up to five; each carries its
English description, so this costs no further network call) against the claim
text, with the local encoder and no LLM call. The best score is adopted only at
or above 0.25, the line above which a link is shown. Below it the Subject keeps
the extracted name and records the best candidate as a low-confidence link,
and below 0.15 the resolver still records nothing. The choice is a
function of the search response, the claim text and the encoder alone, so it
is as reproducible as the baseline: two passes gave identical rows here too.

| Metric | `top_hit` (baseline) | `best_description` | Change |
|---|---:|---:|---:|
| `resolution_accuracy` | **0.886** (31) | 0.800 (28) | −3 subjects |
| `resolution_wrong_ref` | 0.086 (3) | 0.171 (6) | +3 subjects |
| `resolution_bare_local` | 0.029 (1) | 0.029 (1) | none |

It fixed none of the four baseline failures and broke three correct links:

| Subject | Gold | `top_hit` | `best_description` |
|---|---|---|---|
| SHA-256 | `Q110651361` | `Q110651361` "cryptographic hash function" (0.32), correct | `Q124624061` "double SHA-256" (0.46), **wrong** |
| Leeds | `Q39121` | `Q39121` "city in West Yorkshire" (0.27), correct | `Q1128631` "Leeds United F.C." (0.38), **wrong** |
| Signal | `Q19718090` | `Q19718090` "privacy-focused encrypted messaging app" (0.16), correct | `Q828130` "signal transduction" (0.18), **wrong** |
| Lantern | none | `Q862454` lighting device (0.20), wrong | `Q2618398` lighthouse lantern room (0.32), wrong and now shown |
| Harbor | none | `Q283202` (0.30), wrong | unchanged |
| Postgres | `Q192490` | `Q28975208` (0.33), wrong | unchanged |
| Rust | `Q575650` | no link | unchanged |

Every other row is unchanged. The one difference the column does not count is
naming: below the 0.25 line the Subject now keeps the extracted name, so
"Kafka" stays "Kafka" where the baseline renamed it "Apache Kafka".

### Why it cannot work with this encoder

The losses share one cause. A longer, more specific description that shares
words with the claim outscores a shorter, correct one: "double SHA-256" beats
"cryptographic hash function" on a sentence about SHA-256 hashes, and a
football club in Leeds beats the city on a sentence about a company in Leeds.
Wikidata's search rank already encodes which sense is usual, and the argmax
discards it.

A floor cannot rescue the policy either. On the baseline, the eleven correct
links score between 0.16 and 0.58 against their claims and the three wrong
ones between 0.20 and 0.33. Seven of the eleven correct links score below the
highest wrong one, so any cut that removes the Harbor link (0.30) also removes
Bergen, Leeds, Python, Go, Kafka, Signal and Notion. The description-to-claim
cosine does not separate right from wrong on this set.

The literal "POET" case behaves the same way. Probed by hand with the claim
"POET Technologies is shipping its optical interposer platform to two
data-center customers.", the occupation (`Q49757`, "person who writes poetry")
is both Wikidata's top hit and the best-scoring candidate at 0.34, so either
selector links it. The company's own item (`Q30298339`) is not among the
candidates a search for "POET" returns.

### Decision

Embedding-only selection is not enough, and on this measurement it is a
regression, so it is not the default. It ships as the non-default value of
`subjects.wikidata_candidate_selection` so this comparison can be re-run and a
later selector can be measured against both. The default, `top_hit`, is the
baseline resolver, unchanged (re-measured with the setting in place: identical
rows).

The four baseline failures point at three different fixes, none of them a
better cosine:

- **Harbor and Lantern** need a judgement that a payroll company is not a body
  of water when no candidate is right. That is the LLM-assisted step, a
  separate change with its own cost per name.
- **Postgres** is a defect in the prefix-expansion filter, which discards the
  correct top hit because "PostgreSQL" continues "Postgres" with a letter.
- **Rust** is the abstention floor's recall cost, a correct top hit scored
  below 0.15 against a claim that mentions it in passing.

## Claim metrics from the same suite version

One full `particles extractor benchmark general-extractor --suite
prose-article-seed-001` run on 2026-09-30 (`claude-sonnet-4-6`, extractor
0.16.0, embedding judge at 0.80) scored recall 0.94, precision 0.83 and
calibration error 0.09 over the six cases. It is a single run, and v0.3.0 has
two more cases than the v0.2.0 the provider-survey pages measured, so it is
comparable with neither.

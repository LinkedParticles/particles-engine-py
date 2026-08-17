# Querying

## Semantic search

The default query path embeds your question, runs cosine similarity
against ACTIVE particles, and synthesises a natural-language answer
with citations:

```bash
uv run particles query "What is the composition of the 1 Pfennig 1948-1950?"
```

Results are ranked by a combined score of cosine similarity and
`effective_confidence` (stored confidence × extractor trust × source
trust × age decay). See [Concepts → confidence](concepts.md#confidence).

To ask what the store believed at a **past instant** — and see what replaced
each since-retired belief, and when — pass `--as-of`. See
[As-of time travel](as-of.md) for the full walkthrough.

## Creating a taxonomy

Before you can filter by tag, a taxonomy has to exist and particles
have to be tagged. A taxonomy is a JSON file you **deposit
like any other source** — there is no special command.

Author the file. Tag paths use `/` as the hierarchy separator; a root
tag has `parent: null`, and every child's `parent` must equal its own
path with the last segment stripped:

```json
{
  "name": "cloud-native",
  "version": "0.1.0",
  "author": "jeff",
  "domain": "Cloud-native infrastructure and platform engineering",
  "tags": [
    { "tag": "cloud-native",               "parent": null,           "aliases": ["cncf"] },
    { "tag": "cloud-native/observability", "parent": "cloud-native", "aliases": ["monitoring", "o11y"] },
    { "tag": "cloud-native/orchestration", "parent": "cloud-native" }
  ]
}
```

Deposit it. The local-file importer recognises the top-level
`name` / `version` / `tags` keys and tags the corpus entry with
`source_type = TAXONOMY_DEFINITION` automatically:

```bash
uv run particles deposit ./cloud-native-taxonomy.json
# entry_id:    10869035-...
# snapshot_id: 09a0ab0b-...
```

Extract to materialise it into the queryable tag tables:

```bash
uv run particles extract 10869035-...
# Extracted 0 particles.
```

`0 particles` is expected and correct — a taxonomy is configuration,
not a set of knowledge claims, so the `TaxonomyExtractor` populates the
`taxonomies` / `tag_nodes` tables and produces no particles. Confirm it
landed with the `list_taxonomies` MCP tool.

## Tagging particles

Tags arrive on particles manually (Phase A). Each `--tag` is validated
against the active taxonomies; `--tag` is repeatable:

```bash
uv run particles particle tag <particle-id> \
    --tag cloud-native/observability
uv run particles particle untag <particle-id> \
    --tag cloud-native/observability
```

Use `--force` to apply an ad-hoc tag that isn't in any active taxonomy.

## Tag filters

Apply taxonomy tags to constrain the search to a subtree.
Tags are repeatable:

```bash
uv run particles query "What changed in 1990?" \
    --tag "numismatics/germany" \
    --tag "history/cold-war"
```

A tag-filtered query restricts the candidate pool *before* embedding
similarity — fewer particles, sharper results.

Expansion is **subtree-only**: `--tag cloud-native` matches particles
tagged `cloud-native/observability` (a descendant), but `--tag
cloud-native/observability` does **not** match a particle tagged at the
bare `cloud-native` root. Tag at the most specific level you'll want to
query.

## Structural claim filters

Since many particles carry a derived S-P-O **structured claim**
annotation beside their prose; the structured filter makes it readable on the same
`query` verb. Structural conditions select **claims by their form**, not
truths — the output speaks of *claims* throughout, and counts count
claims, never entities.

**Which flags make a query deterministic?** Any query with structural
flags and **no question** — including `--count`, `--group-by`, and
`--predicates` — runs with **no embedding and no LLM call**: results are
matching claims ordered by `effective_confidence` (tie: newest
`asserted_at` first), and the aggregates are exact. Add a question and
the flags instead **prefilter** the ordinary semantic path — cosine
ranking, confidence composition, and the LLM answer are untouched; the
filters only narrow the candidate set, exactly like `--subject` and
`--tag`.

Discover the vocabulary first — the predicate filter is an exact string
(case-insensitive), so a CURIE and its expanded IRI are different terms:

```bash
uv run particles query --predicates
```

Filter and compare (deterministic — no question given):

```bash
uv run particles query --predicate nmo:hasWeight --object-gt 3
```

Objects compare **typed**: numeric xsd datatypes normalize to numbers,
`xsd:date` / `xsd:dateTime` to dates, at read time (nothing is stored).
There is no unit parsing — an untyped `"3 grams"` does not compare to
`3.0`; such claims are excluded from `--object-gt` / `--object-lt` and
the exclusion count is **disclosed** in the footer, never silently
dropped. `--object-eq` falls back to case-insensitive text when either
side doesn't normalize; `--object-contains` is substring match on any
term kind.

Deterministic aggregates (a simultaneous question is rejected):

```bash
uv run particles query --predicate nmo:hasWeight --count
uv run particles query --predicate nmo:hasWeight --group-by subject
```

Counts come with the effective-confidence min/median/max of the counted
rows. There is **no default confidence floor** — pass
`--min-effective-confidence` to exclude rows explicitly (the exclusion
is disclosed). `--group-by subject` groups by the resolved Subject
(`claim.subject_id`, falling back to the particle's subject links) — and
counting the subject buckets, not the claims, is the honest answer to
"how many coins".

Every structural result ends with a coverage footer — "matched against
the N of M ACTIVE particles carrying a structured claim" — because the
filter only ever saw the annotated fraction of the store: absence of a
hit is not absence of a belief. All of this composes with `--store`,
`--as-of`, `--tag`, `--subject`, and `--min-confidence` unchanged.

## MCP server

For LLM agents that want to query the store directly, Particles ships
a read-only MCP server:

```bash
uv run particles mcp serve
```

The MCP tools are: `query`, `particle_show`, `particles_list`,
`particle_search`, `subjects_list`, `subjects_search`, `subjects_show`,
`lint`, `quality_report`, `list_corpus_entries`, `list_taxonomies`,
`links_suggest`, `events_list`, `event_show`. All read-only — the MCP
surface cannot deposit, extract, or change particle status.

## Ranking — what the SDK actually does

1. **Embed the question.** SentenceTransformers (default
   `all-MiniLM-L6-v2`).
2. **Filter to ACTIVE particles**, optionally constrained by tag /
   subject.
3. **Cosine similarity** between the question embedding and each
   particle's embedding. Top-k truncation (default 40; configurable
   under `query.default_top_k` in `config.yaml`).
4. **Rank by combined score** — `similarity_weight` × cosine
   similarity + `confidence_weight` × `effective_confidence`
   (defaults 0.6 / 0.4, §9.3). Particles whose `effective_confidence`
   falls below `min_confidence` (default 0.0) are dropped.
5. **Relevance floor** — if the best raw cosine similarity
   over the rendered top-k falls below `query.relevance_floor` (default
   0.25, calibrated to the reference embedding model; `0.0` disables),
   the answer is a deterministic "the store holds no beliefs relevant
   to this question" — no LLM call. The nearest beliefs are still
   returned, labelled as likely unrelated, and the structured
   `relevance` field on the response carries `max_similarity` / `floor`
   / `below_floor`. Disclosure only: the floor never changes ranking,
   confidence, or filtering.
6. **Synthesise** a natural-language answer with citations. Audience
   tier (`EXPERT`, `GENERAL`, `REGULATORY`) shapes the prose; default
   `GENERAL`.
7. **Coverage gap disclosure** — if no particles meet the
   threshold, the response says "the store doesn't contain relevant
   evidence" rather than confabulate.

## The contested badge

Every query result carries a composed **contested badge** by default
: a claim renders *contested* when at least one of three
named bases fires, and the badge always names which —

| Basis | Fires when |
|---|---|
| `stance` | someone is on record **disputing** the claim — a `DISPUTES` edge in its query-time stance distribution. Endorsements alone never fire; when this basis fires the badge carries the unverified-holder caveat. |
| `divergence` | the claim's effective confidence **spreads across your trust policies** (local + adopted lenses) by at least `contestedness.callout_threshold`. Absent — not merely quiet — until you have two or more policies. |
| `inconsistency` | an **open INCONSISTENCY particle** references the claim; the badge keeps that particle's id as the drill-down. |

The CLI prints one `⚠ contested (…)` line per badged result;
`--contestedness` still prints the full per-policy readings, and the
agreement distribution stays behind its own opt-in. A claim with no
basis fired carries no badge — absence of measurement is never rendered
as "uncontested". The badge is disclosure only: it never changes
ranking, confidence, or filtering. Disable it with
`contestedness.badge_enabled: false` in `config.yaml`. The same badge
appears in the session-start digest, the `MEMORY.md` projection, and the
MCP `query` / `particles_list` responses (as `contested_bases`, beside
the existing INCONSISTENCY-id `contested` key).

## Tuning the ranking

Operator-side concern; see the [operator guide → tuning](../operator-guide/tuning.md)
for `min_confidence`, `query.top_k`, trust weights, and audience
defaults.

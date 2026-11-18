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

To ask what the store believed at a **past instant**, and see what replaced
each since-retired belief and when, pass `--as-of`. See
[As-of time travel](as-of.md) for the full walkthrough. To ask what
**happened or held during a period of world time**, pass `--during` or
`--on`. See [Asking about a period](windows.md).

## Seeing the source passage

Every belief links back to the source it was extracted from. To read that
source text next to the answer, add `--show-source`:

```bash
uv run particles query "When does the nightly job run?" --show-source
```

After the answer, the top hits are listed with the passage behind each. For a
single belief, use its ID:

```bash
uv run particles particle source 3f9a1c2e
```

The passage is re-derived from the stored snapshot each time it is shown:
nothing extra is stored at ingest, and the original file or URL is never
re-read, so anything removed before archiving stays removed. Each passage is
labelled with how it was found, and the labels are not interchangeable:

| Label | Meaning |
|---|---|
| `exact` | The chunk the extractor was shown, verified against the hash recorded on the belief's provenance. This is a quote-level guarantee. |
| `located` | The paragraph (or line) sharing the most terms with the belief, with the share shown as a percentage. A reading aid, **not** verification. |
| `whole source` | No single passage stood out, so the source text is shown instead of a guess. |
| `unavailable` | There is nothing to show, and a note says why: the belief derives from other beliefs, the stored content is missing, or the source is an image. |

Expect `located` for short documents. Only a source long enough to be
extracted in several chunks records a per-chunk hash; a note, a memory file,
or a short page is extracted in one pass and records none. A recorded hash can
also stop matching if `extraction.html_chunk_size` has changed since the
belief was extracted; the passage then falls back to `located` and a note
says so.

The passage is for reading only. It is fetched after the results are ranked
and never influences which beliefs are retrieved or how they are ordered, and
the `located` percentage is not a confidence. `--show-source` is not available
together with `--store` federation; run `particle source` against the store
that holds the belief. Over HTTP the same result is served by
`GET /particles/{particle_id}/source`.

Three settings under `source_passage` in `config.yaml` tune the display:
`locate_min_overlap` (the share of the belief's terms a paragraph must contain
to be offered, default `0.5`), `max_passage_chars` (default `6000`), and
`query_show_limit` (how many top hits `--show-source` covers, default `5`).

## Grounded answers

The answer is composed by a model from the retrieved beliefs, and a model
adds things of its own: a motive, a status, a conclusion, now and then a fact
it learned elsewhere. In the default answer that addition reads exactly like
the recalled beliefs around it. A grounded answer makes it visible instead:

```bash
uv run particles query "Why was the answer token budget raised?" --grounded
```

Every sentence ends with a tag saying where it came from:

| Tag | Meaning |
|---|---|
| `[p-1a2b3c4d]` | The retrieved beliefs the sentence rests on, by ID. `particles particle show p-1a2b3c4d` opens one. |
| `[inference]` | No belief states it. The model drew it over the cited beliefs: a motive, a status, a conclusion, or a remark about what the store holds. |
| `[background]` | It comes from the model's own general knowledge and from no belief at all. |
| `[unattributed]` | The model gave no tag, or cited only IDs it was never given. |

The model declares the labels and the query checks the citations. Every cited
ID is matched against the beliefs actually retrieved for this answer; an ID
outside that set is never trusted, a warning on stderr reports it, and a
sentence left with no valid citation is labelled `[unattributed]`. Nothing is
dropped: a labelled sentence is still in the answer, so you can weigh it
yourself. Nothing the model adds is ever stored as a belief either. The
labels exist on this one response and nowhere else.

Below the answer, an attribution line counts the sentences of each kind and
lists each cited belief once under the ID the answer uses:

```text
Attribution: 3 cited · 2 inference · 0 background · 0 unattributed
  [p-10cdc518] The token caps have been changed to 4096 for answer and 1024 for judge.
  [p-f548e2f1] On models with adaptive thinking by default, thinking spend counts …
```

A refusal is unchanged: when nothing retrieved bears on the question, the
answer says so and carries no tags.

Grounded mode is off by default. Set `query.grounded_answers: true` in
`config.yaml` to make it the default for every surface, and use
`--grounded` or `--ungrounded` to choose per query. Over HTTP the request field
is `grounded` and the response carries the per-sentence labels and full cited
IDs in `answer_attribution`; the MCP `query` tool takes the same `grounded`
argument. On the owner's store it lowered the
share of silently unsupported sentences only from about 25% to about 22%
(see [the leakage benchmark](../benchmarks/leakage.md#grounded-answers-and-the-reason-first-judge-2026-10-03)),
which is why it is not on by default.

## Creating a taxonomy

Before you can filter by tag, a taxonomy has to exist and particles
have to be tagged. A taxonomy is a JSON file you **deposit
like any other source**; there is no special command.

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
`source_type = TAXONOMY_DEFINITION` automatically, for a file of at most
1 MiB; deposit a larger taxonomy with `--source-type TAXONOMY_DEFINITION`:

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

`0 particles` is expected and correct: a taxonomy is configuration,
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
similarity: fewer particles, sharper results.

Expansion is **subtree-only**: `--tag cloud-native` matches particles
tagged `cloud-native/observability` (a descendant), but `--tag
cloud-native/observability` does **not** match a particle tagged at the
bare `cloud-native` root. Tag at the most specific level you'll want to
query.

## Structural claim filters

Since many particles carry a derived S-P-O **structured claim**
annotation beside their prose; the structured filter makes it readable on the same
`query` verb. Getting and maintaining those annotations (`particles structure`,
coverage, what to do about a wrong triple) is
[Operator guide → structured claims](../operator-guide/structured-claims.md). Structural conditions select **claims by their form**, not
truths: the output speaks of *claims* throughout, and counts count
claims, never entities.

**Which flags make a query deterministic?** Any query with structural
flags and **no question** (including `--count`, `--group-by`, and
`--predicates`) runs with **no embedding and no LLM call**: results are
matching claims ordered by `effective_confidence` (tie: newest
`asserted_at` first), and the aggregates are exact. Add a question and
the flags instead **prefilter** the ordinary semantic path. Cosine
ranking, confidence composition, and the LLM answer are untouched; the
filters only narrow the candidate set, exactly like `--subject` and
`--tag`.

Discover the vocabulary first; the predicate filter is an exact string
(case-insensitive), so a CURIE and its expanded IRI are different terms:

```bash
uv run particles query --predicates
```

Filter and compare (deterministic, since no question is given):

```bash
uv run particles query --predicate nmo:hasWeight --object-gt 3
```

Objects compare **typed**: numeric xsd datatypes normalize to numbers,
`xsd:date` / `xsd:dateTime` to dates, at read time (nothing is stored).
There is no unit parsing: an untyped `"3 grams"` does not compare to
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
rows. There is **no default confidence floor**; pass
`--min-effective-confidence` to exclude rows explicitly (the exclusion
is disclosed). `--group-by subject` groups by the resolved Subject
(`claim.subject_id`, falling back to the particle's subject links).
Counting the subject buckets, not the claims, is the honest answer to
"how many coins".

Every structural result ends with a coverage footer ("matched against
the N of M ACTIVE particles carrying a structured claim") because the
filter only ever saw the annotated fraction of the store: absence of a
hit is not absence of a belief. All of this composes with `--store`,
`--as-of`, `--during` / `--on`, `--tag`, `--subject`, and `--min-confidence`
unchanged.

## MCP server

For LLM agents that want to query the store directly, Particles ships
a read-only MCP server:

```bash
uv run particles mcp serve
```

The read tools are: `query`, `particle_show`, `particles_list`,
`particle_search`, `subjects_list`, `subjects_search`, `subjects_show`,
[`lint`](../operator-guide/lint-and-review.md) and `quality_report`,
`list_corpus_entries`, `list_taxonomies`,
[`links_suggest`](../operator-guide/co-evidential.md),
[`corpus_links_suggest`](../operator-guide/citation-signals.md),
[`events_list` / `event_show`](../operator-guide/auditing.md), and
[`graph_view`](graph-view.md#served-get-graph-and-the-mcp-graph_view-tool).
These are read-only: they cannot deposit, extract, or change particle
status. A store can additionally be *write-enabled* which adds
the assert / supersede / retract / deposit tools.

[Connecting your coding agent](coding-agents.md) has the config block for
each harness (Claude Code, Claude Desktop, Codex, Cursor, Windsurf, Zed,
OpenCode) and says which of them can go further than tools. Two such routes
are worth knowing about: the
[Claude Code integration](claude-code.md) wires a store in as managed agent
memory rather than as a tool an agent must choose to call, and the
[memory-server swap](memory-server-swap.md) presents the store through the
reference knowledge-graph memory server's own tool contract.

### What the `query` tool returns

An agent pays for every character a tool returns, so the MCP `query` tool
returns a compact result by default:

- **The answer**, as plain prose, with every disclosure that applies
  (relevance, truncation, coverage, as-of and window notes).
- **The beliefs the answer cites**, in the order it first cites them, at most
  `mcp.query_hits` (default 10) of them. The tool asks for a grounded answer
  to learn which ones those are, then removes the citation tags from the text.
  When the answer cites nothing, the top-ranked beliefs are listed instead.
- **For each belief**: its `id` (for `particle_show`), `content`,
  `effective_confidence`, a one-line `source` (source type, URI and
  publication date of its first source, with `(+N more)` when it has
  several), `valid_from`, and, when it is contested, the open conflict's id
  and the badge's bases. A key whose value is null, empty, zero or false is
  left out: an absent key means unset.
- **`hits_omitted`**, when the answer read more beliefs than the list shows.
  The answer still reads all `top_k` (default `query.default_top_k`, 40).

A call can ask for more. `detail="full"` returns the whole `QueryResponse`
with full particle records, every retrieved belief and the per-sentence
attribution; `hits` sets how many beliefs are listed; `particle_show`
returns one belief's full record. The REST `/query` response and the CLI are
unchanged, so the MCP tool is the one surface whose answer is composed
grounded by default.

To restore the full result for every client, set `mcp.query_detail: full`.
To keep the compact shape but list the top-ranked beliefs under an ungrounded
answer, set `mcp.query_list: ranked`.

## Ranking: what the SDK actually does

1. **Embed the question.** SentenceTransformers (default
   `all-MiniLM-L6-v2`).
2. **Filter to ACTIVE particles**, optionally constrained by tag /
   subject.
3. **Cosine similarity** between the question embedding and each
   particle's embedding. Top-k truncation (default 40; configurable
   under `query.default_top_k` in `config.yaml`). The response warns
   that top-k cut a plateau only when the last result scores within
   `query.truncation_plateau_drop` (default 10%) of the first and more
   than `query.truncation_near_count` (default 5) excluded particles
   score as well, and names the `--top-k` that takes them in, or asks
   for a narrower question when none can.
4. **Rank by combined score**: `similarity_weight` × cosine
   similarity + `confidence_weight` × `effective_confidence`
   (defaults 0.6 / 0.4, §9.3). Particles whose `effective_confidence`
   falls below `min_confidence` (default 0.0) are dropped. When an
   operator sets `query.exact_token_weight` (off by default),
   a question that names something by number (`is RFC 9110 obsolete?`,
   `what changed in 1.231.0?`) also lifts the particles whose text
   contains that number, before the top-k cut. The embedding barely
   tells `RFC 9110` from `RFC 9112`; the lift is what brings the
   particles about the document you named into the answer. See
   [the operator guide](../operator-guide/tuning.md#exact-identifier-token-addend-adr-0368).
5. **Relevance floor**: if the best raw cosine similarity
   over the rendered top-k falls below `query.relevance_floor` (default
   0.25, calibrated to the reference embedding model; `0.0` disables),
   the answer is a deterministic "the store holds no beliefs relevant
   to this question", with no LLM call. The structured `relevance`
   field on the response carries `max_similarity` / `floor` /
   `below_floor`. Disclosure only: the floor never changes ranking,
   confidence, or filtering.
6. **Synthesise** a natural-language answer with citations. Audience
   tier (`EXPERT`, `GENERAL`, `REGULATORY`) shapes the prose; default
   `GENERAL`. Each particle the answer step sees carries its date: the
   day the claim began to hold, or else the day its source was written.
   Where particles give different values for the same thing, the most
   recently dated is taken as current and the earlier value is mentioned,
   however many older particles repeat it. `query.answer_dates: false`
   turns the dates off. When the retrieved particles do not bear on the
   question, the answer step declines rather than answering from them,
   and the response is a refusal like the floor's.
7. **Coverage gap disclosure**: if no particles meet the
   threshold, the response says "the store doesn't contain relevant
   evidence" rather than confabulate.

### When the answer refuses

A refusal from either step sets `answer_refused` on the response. Its
nearest beliefs are then withheld by default: they are likely
unrelated, and an agent reading the whole result would otherwise spend
its context on them. The response keeps the `relevance` verdict, reports
the number withheld in `withheld_candidates`, and the answer text ends
with one sentence saying how many were withheld.

To receive them anyway, set `include_refused_candidates` on the request:
the `include_refused_candidates` parameter of the MCP `query` tool, the
same field in the `POST /query` body, or `--show-particles` on
`particles query`, which then prints them under "Nearest beliefs, likely
unrelated". The web UI asks for them and shows them under the same
heading. Nothing is withheld when the answer did not refuse, including
when answer generation failed and the response lists the retrieved
beliefs in its place.

## The contested badge

Every query result carries a composed **contested badge** by default:
a claim renders *contested* when at least one of three
named bases fires, and the badge always names which.

| Basis | Fires when |
|---|---|
| `stance` | someone is on record **disputing** the claim: a `DISPUTES` edge in its query-time stance distribution. Endorsements alone never fire; when this basis fires the badge carries the unverified-holder caveat. |
| `divergence` | the claim's effective confidence **spreads across your trust policies** (local + adopted lenses) by at least `contestedness.callout_threshold`. Absent (not merely quiet) until you have two or more policies. Adopting a second policy is [Operator guide → lenses](../operator-guide/tuning.md#lenses-adopting-someone-elses-trust-policy). |
| `inconsistency` | an **open INCONSISTENCY particle** references the claim; the badge keeps that particle's id as the drill-down. Clearing it is [the review workflow](../operator-guide/lint-and-review.md#the-review-workflow). |

The CLI prints one `⚠ contested (…)` line per badged result;
`--contestedness` still prints the full per-policy readings, and the
agreement distribution stays behind its own opt-in. A claim with no
basis fired carries no badge; absence of measurement is never rendered
as "uncontested". The badge is disclosure only: it never changes
ranking, confidence, or filtering. Disable it with
`contestedness.badge_enabled: false` in `config.yaml`. The same badge
appears in the session-start digest, the `MEMORY.md` projection, the
[graph view](graph-view.md), and the
MCP `query` / `particles_list` responses (as `contested_bases`, beside
the existing INCONSISTENCY-id `contested` key).

## Tuning the ranking

Operator-side concern; see the [operator guide → tuning](../operator-guide/tuning.md)
for `min_confidence`, `query.top_k`, trust weights, and audience
defaults.

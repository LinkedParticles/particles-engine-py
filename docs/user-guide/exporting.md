# Exporting

This page walks the four document-set exporters: Obsidian (vault),
Anki (deck), wiki (per-subject cited articles), and Logseq (graph). Each
writes to the `output` path you pass on the CLI.

Three more `export` targets have different shapes and live elsewhere:

- **`graph`** — a scoped, self-contained HTML render of the store's
  epistemics rather than a document set. It has its own page:
  [Graph view](graph-view.md).
- **`notion`** — an idempotent upsert into a Notion database over the HTTP
  API, so it writes no files at all (`--dry-run` plans it with zero API
  writes). It needs a credential; see
  [Operator guide → configuration → secrets](../operator-guide/configuration.md#secrets).
- **`jsonl`** — line-delimited JSON for downstream tooling.

`particles export --help` and the
[command reference](../cli-reference.md) carry the full, always-current flag
list for each.

## Obsidian vault

```bash
uv run particles export obsidian ./my-vault
```

Writes one `.md` per Subject plus a `_index.md`. Coin subjects get
the Numismatic infobox template; pivot subjects get a minimal
linking template; everything else gets the generic callout template.

**Useful flags:**

- `--min-particles N` — skip subjects with fewer than N ACTIVE
  particles after the quality filter.
- `--min-links N` — skip subjects with fewer than N graph links (in
  + out combined). Reduces phantom-node noise in the graph view.
- `--with-synthesis` — splice an LLM-synthesised prose article into
  each note. Requires `ANTHROPIC_API_KEY`. Shares the article-cache
  with `export wiki`, so running both pays the LLM cost once per
  subject. Also emits one cited-prose note per NARRATIVE
  under `Narratives/` — a journal entry rendered as its whole-entry
  narrative. Disable with `obsidian.emit_narrative_notes:
  false`; with no key these notes fall back to a deterministic cited
  listing.
- `--invalidate-stale-links` — drop the article-cache hash from any
  note whose `[[X]]` wikilinks reference a renamed subject.
- `--min-particle-confidence F` — cross-exporter quality filter
  . Drops particles whose `effective_confidence` is below
  `F` from every rendered note. To set a standing floor for every export
  instead of passing it each run, see
  [Operator guide → cross-exporter quality threshold](../operator-guide/tuning.md#cross-exporter-quality-threshold);
  for what `effective_confidence` is composed of, see
  [Concepts → confidence](concepts.md#confidence).

## Anki deck

```bash
uv run particles export anki ./deck.txt --deck-name=Numismatics
```

Writes a tab-delimited file. One card per particle, grouped into
decks by subject. Structured particles emit one card per `properties`
key; descriptive particles emit one card from the content.

**Useful flags:**

- `--deck-name NAME` — root deck name prefix.
- `--min-particle-confidence F` — same cross-exporter filter.

(The exporter also accepts a `max_cards_per_subject` option via the
Python / HTTP API; it is not exposed as a CLI flag.)

## Wiki articles

```bash
uv run particles export wiki ./my-wiki
```

Writes a flat directory of per-Subject Markdown articles, each with
LLM-synthesised prose and inline citations back to the corpus
entries. Plus a top-level `index.md`.

**Useful flags:**

- `--dry-run` — report cache hits + regen count + estimated token
  spend; no LLM calls or file writes.
- `--regenerate-all` — bypass the per-subject input-hash cache.
- `--invalidate-stale-links` — same wikilink-staleness behaviour as
  Obsidian.
- `--subjects "A,B,C"` — limit to specific canonical names.
- `--min-particles N` — minimum post-filter particle count
  (default 3 per `config.wiki.min_particles`).
- `--min-particle-confidence F` — cross-exporter quality filter.

The export also writes one cited article per NARRATIVE under
`Narratives/`, using the same render path and cache as the Obsidian
narrative notes. Disable with `wiki.emit_narrative_notes: false`.
Narratives are subject-less, so `--subjects` suppresses them.

## Logseq graph

```bash
uv run particles export logseq ./my-graph
```

Writes `pages/<subject>.md` in Logseq's native bullet-outline format
. Each particle is emitted as a block whose `id::` is the
particle ID, enabling cross-page citation via `((<particle_id>))`
syntax.

**Useful flags:**

- `--with-synthesis` — same synthesis splice as Obsidian; shares the
  article cache, so running multiple synthesising
  exporters pays the LLM cost once per subject. It also emits one
  cited-prose page per NARRATIVE in Logseq's `Narratives/` page
  namespace (on disk `pages/Narratives___<slug>.md`) and adds a
  `## Narratives` backlink block to each subject page whose claims
  take part in one. Disable with `logseq.emit_narrative_notes: false`.
- `--invalidate-stale-links`, `--min-particles`, `--min-links`,
  `--min-particle-confidence` — same semantics as Obsidian.

## The dry-run summary

Each exporter returns a typed Pydantic summary. The CLI
prints it as `key: value` lines after the run. Conditional fields
(synthesis counts, `articles_written`) only appear when the producing
step actually ran. For machine consumption, the same model is
available as `*.model_dump_json()` for downstream tooling.

## What about other formats?

The plugin registry is designed so an exporter is one file plus
one registry line. If the format you want isn't in the list above, writing it
is a small job — see
[Plugin-author guide → writing an exporter](../plugin-author-guide/exporters.md),
which walks the contract, the worked examples in the tree, and the
[cross-exporter options](../plugin-author-guide/exporters.md#cross-exporter-contract)
every exporter must honour.

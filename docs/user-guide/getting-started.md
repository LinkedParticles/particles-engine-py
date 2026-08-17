# Getting started

Install, initialise a store, deposit your first source, extract
particles, and query.

## Install

Particles uses [uv](https://docs.astral.sh/uv/) for dependency
management. Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone the repo and sync:

```bash
git clone https://github.com/LinkedParticles/particles-engine-py.git
cd particles-engine-py
uv sync
```

## Secrets

The extractor and semantic lint both call the Anthropic API. Set:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

For operator-level secrets handling, see the
[operator guide](../operator-guide/configuration.md).

## Initialise the store

```bash
uv run particles db init
```

This creates the SQLite database (default: `./particles.db`) and
the blob directory.

## Deposit a source

`deposit` takes a URL or a local file path:

```bash
uv run particles deposit https://en.numista.com/catalogue/pieces8562.html
# entry_id:    3f2a1c8e-...
# snapshot_id: 9b4d7e2a-...

uv run particles deposit ./article.pdf
```

For link-shaped sources (Reddit / Hacker News / Mastodon), the
deposit also follows the post's primary URL and records the
relationship — see the [operator guide](../operator-guide/troubleshooting.md)
for the follow-edges behaviour.

### Depositing RDF

An RDF document — Turtle, N-Triples, TriG, N-Quads, JSON-LD or RDF/XML —
is recognised by its extension and parsed rather than read by an LLM:

```bash
uv run particles deposit ./coins.ttl
```

Extraction is then deterministic and free: one particle per triple, no API
call, every triple covered. Because a `.json` file could be many things, a
JSON-LD document needs either a `.jsonld` extension or an explicit
`--source-type RDF_GRAPH`.

These particles are the one place the store works in reverse: the **triple is
the assertion** and the readable `content` is generated from it, so an
imported graph is still findable by ordinary semantic query instead of sitting
in the store as opaque URIs. Labels come from the document itself, so a graph
that carries `rdfs:label`s reads as prose (`5 Pfennigs was minted at: Berlin
Mint`) while a bare triple dump reads as URIs. Entity URIs from a recognised
namespace — Wikidata, for instance — bind straight to the matching Subject
rather than being name-matched, which is why imported RDF tends to align with
what you already know instead of forking it.

Confidence comes from your trust policy rather than from the file, since a
parser has no opinion of its own. The exception is a document that annotates
its own confidence (an RDF 1.1 reification bundle or a named graph carrying a
confidence predicate); those values are read directly. Configure which
predicates count under `rdf.confidence_predicates` — there is no standard one
in RDF, so publishers differ.

## Extract particles

```bash
uv run particles extract 3f2a1c8e-...
# or to extract every pending snapshot
uv run particles extract --all-pending
```

The extractor calls Claude to produce structured claim-granularity
particles with confidence + provenance + subject resolution.

## Query

```bash
uv run particles query "What is the composition of the 1 Pfennig 1948-1950?"
```

Add `--tag <path>` to restrict to a taxonomy subtree. See
[Querying](querying.md) for how to create a taxonomy, tag particles,
and the tag patterns and ranking that follow.

## Export

```bash
uv run particles export obsidian ./my-vault
uv run particles export anki ./deck.txt
uv run particles export wiki ./my-wiki
uv run particles export logseq ./my-graph
```

See [Exporting](exporting.md) for the dry-run / cache / synthesis
options each exporter supports.

## What next

- [Concepts](concepts.md) — particle, subject, status, confidence,
  provenance.
- [Querying](querying.md) — tag filters, MCP, ranking.
- [Exporting](exporting.md) — exporter-specific workflows.
- The full CLI reference is at [`cli-reference.md`](../cli-reference.md);
  the workflow-oriented index at [`cli.md`](../cli.md).

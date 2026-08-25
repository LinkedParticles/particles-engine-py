# Getting started

Install, initialise a store, deposit your first source, extract
particles, and query — with citations.

## Install

```bash
pip install linkedparticles
```

That installs the `particles` CLI and the full engine. (Python 3.11+;
`pipx install linkedparticles` or `uv tool install linkedparticles` work
too and keep it isolated.)

Working on the engine itself, or want the bleeding edge? See
[Development setup](#development-setup) below — everything else on this
page assumes the installed package.

## Configure an LLM

Extraction and the semantic lint read your sources with an LLM. The
default provider is Anthropic:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Providers are configuration, never code: any OpenAI-compatible endpoint —
including a local model — can serve any purpose via `config.yaml`
(see the [operator guide](../operator-guide/configuration.md)). Everything
else stays on your machine: the store is a local SQLite database, and only
the source text being extracted (or a question being answered) goes to the
model you configured.

## Initialise the store

```bash
particles db init
```

This creates the SQLite database (default: `./particles.db`) and
the blob directory.

## Deposit a source

`deposit` takes a URL or a local file path and writes it into the
append-only corpus, snapshotted and content-addressed:

```bash
particles deposit https://en.wikipedia.org/wiki/Douglas_Lenat
# entry_id:    3f2a1c8e-...
# snapshot_id: 9b4d7e2a-...

particles deposit ./article.pdf
```

For link-shaped sources (Reddit / Hacker News / Mastodon), the
deposit also follows the post's primary URL and records the
relationship — see the [operator guide](../operator-guide/troubleshooting.md)
for the follow-edges behaviour.

### Depositing RDF

An RDF document — Turtle, N-Triples, TriG, N-Quads, JSON-LD or RDF/XML —
is recognised by its extension and parsed rather than read by an LLM:

```bash
particles deposit ./coins.ttl
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
particles extract 3f2a1c8e-...
# or to extract every pending snapshot
particles extract --all-pending
```

The extractor calls the configured LLM to produce structured
claim-granularity particles with confidence + provenance + subject
resolution.

## Query

```bash
particles query "What was Lenat's role in building Cyc?" --show-particles
```

`--show-particles` prints the retrieved claims above the answer, ranked by
effective confidence — each one traceable to the exact snapshot it came
from. The [walkthrough](https://linkedparticles.org/walkthrough/) runs this
same example end to end, including setting per-source trust and watching
the ranking follow.

Add `--tag <path>` to restrict to a taxonomy subtree. See
[Querying](querying.md) for how to create a taxonomy, tag particles,
and the tag patterns and ranking that follow. Add `--as-of <date>` to ask
the same question of a past instant ([As-of time travel](as-of.md)).

## Export

```bash
particles export obsidian ./my-vault
particles export anki ./deck.txt
particles export wiki ./my-wiki
particles export logseq ./my-graph
```

See [Exporting](exporting.md) for the dry-run / cache / synthesis
options each exporter supports.

## What next

- [Concepts](concepts.md) — particle, subject, status, confidence,
  provenance.
- [Querying](querying.md) — tag filters, structural filters, MCP, ranking.
- [Exporting](exporting.md) — exporter-specific workflows.
- [Graph view](graph-view.md) — the store's epistemics as a picture.
- [Claude Code memory](claude-code.md) — wire the store into an agent so
  deposits and recall happen without you running the verbs.
- The full CLI reference is at [`cli-reference.md`](../cli-reference.md);
  the workflow-oriented index at [`cli.md`](../cli.md).

Running this long-term rather than trying it out? The
[operator guide](../operator-guide/index.md) covers
[configuration](../operator-guide/configuration.md),
[lint hygiene](../operator-guide/lint-and-review.md), and
[tuning](../operator-guide/tuning.md).

## Development setup

To work on the engine itself, install [uv](https://docs.astral.sh/uv/) and
run from a checkout — and prefix every command above with `uv run`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/LinkedParticles/particles-engine-py.git
cd particles-engine-py
uv sync
uv run particles db init
```

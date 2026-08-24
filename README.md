# linkedparticles

> **Particles is shared memory for humans and AI agents.** Each particle is one
> claim, plus what you need to judge it: who said it, where, when, and how
> confident they were. Facts, opinions, and memories are all claims, recorded
> the same way as particles. Particles are not edited or deleted. Particles are
> superseded, retracted, or disputed in the open. How much to trust it is a
> perspective applied at query time, never baked into the record.

`linkedparticles` is the **Engine** of the Particles reference implementation —
the package you install to actually run a belief store. Give it documents,
pages, chat logs, or an agent's own session notes; get back claim-granularity
beliefs, each carrying calibrated confidence, an uncertainty kind, a resolved
subject, and provenance back to the exact source bytes. Ask a question and the
answer cites the beliefs it was built from. Nothing is ever overwritten, so you
can ask the store what it believed a year ago and why it stopped.

**When your agent is wrong, you can see exactly why, and fix it at the source.**

The Engine is a library first. The HTTP API, the CLI, the read-only MCP server,
and the resident daemon are all *surfaces* over it.

## Install

```bash
pip install linkedparticles
```

Python 3.11+. `linkedparticles-core` — the store-free Client layer — is pulled
in automatically.

## Sixty seconds

```bash
export ANTHROPIC_API_KEY=sk-ant-...
particles db init

# Deposit a source — file, URL, or literal text — into the append-only corpus
particles deposit https://en.wikipedia.org/wiki/Pluto

# Extract claim-granularity beliefs with confidence, subjects, and provenance
particles extract --all-pending

# Ask in natural language; the answer cites the particles behind it
particles query "Why is Pluto not a planet?"

# Health-check the store: contradictions, staleness, orphaned links
particles lint
```

Because nothing is overwritten, the store can replay its own history. `--as-of`
is the **assertion-time** lens — *what did the store believe at T, and why did
it stop* — not what was true of the world at T:

```bash
particles query "How many planets are in the Solar System?" --as-of 2000-01-01
# → the belief as it stood then, and why it no longer stands: SUPERSEDED,
#   retired 2006-08-24, superseded by "Pluto is a dwarf planet."
```

There is a clickable version of that belief history on the
[front door](https://linkedparticles.org). Step by step:
[getting started](https://docs.linkedparticles.org/user-guide/getting-started/),
then [as-of time travel](https://docs.linkedparticles.org/user-guide/as-of/).

### Wiring it into an agent

```bash
particles mcp serve                 # read-only MCP server (stdio), any MCP client
particles init claude-code          # session-start digest, session-end harvest
particles engine serve              # HTTP API, for a shared or remote engine
particles export obsidian ./vault   # also: anki, wiki, logseq, notion, graph, jsonl
```

It can also stand in for the reference memory server, and it plugs into
LangChain as a retriever and a tool set — see
[Claude Code memory](https://docs.linkedparticles.org/user-guide/claude-code/),
[swapping in for the reference memory server](https://docs.linkedparticles.org/user-guide/memory-server-swap/),
and [using from LangChain](https://docs.linkedparticles.org/user-guide/integrations/).

## Does it hold up?

Measured on **LongMemEval**, the multi-session agent-memory benchmark, with a
full-context oracle and a no-memory floor run as controls in the same harness:

| | Recall@10 | End-to-end QA |
|---|---:|---:|
| **Particles memory** | **0.940** | **0.733** |
| Full-context oracle (baseline) | — | 0.793 |
| No memory (floor) | — | 0.080 |

Method, per-question-type breakdowns, comparator memories, and the
budget-matched arm — including the arms where Particles *loses* — are on the
[benchmarks page](https://docs.linkedparticles.org/benchmarks/), with the
report JSONs of record beside them.

## Security posture

A memory layer is an *input-shaping* attack surface: it does not merely hold
data, it shapes what the agent believes and does next, so a poisoned claim is
an instruction on a delay timer. Before this code was opened we ran an
adversarial application-security audit over the whole package — the HTTP, CLI,
and MCP surfaces, the filesystem writers, the egress layer, and the build
pipeline. The verdict, verbatim: **GO-WITH-FIXES**, with **33 findings — 2
High, 7 Medium, 20 Low, 4 Info**. The ranked must-fix set merged the following
day; the last open finding closed in `v1.128.0`.

What that bought, and what it did not:

- **Prompt injection is contained, not eliminated.** Every LLM call site that
  touches attacker-controllable text keeps trusted instructions in the system
  turn and wraps the untrusted material in a per-call, 128-bit-nonce data
  fence, with structural backstops behind it — a JSON contract enforced at the
  parser, a citation-id membership gate on synthesized prose. This raises the
  bar materially. It is hardening, not immunity.
- **The model is never given tools, and no model output is executed.** No
  function-calling; nothing the model emits becomes a shell command, a SQL
  fragment, or a fetch. An injection can at worst distort *claims* — never
  trigger *actions*.
- **Outbound fetches are validated per hop.** The host is resolved, the address
  is checked against a blocklist, and the connection is made to *that vetted
  address*, re-resolved and re-validated on every redirect — closing DNS
  rebinding and redirect SSRF, not just the first lookup. The two fetches that
  run as subprocesses reach the same guarantee by pinning to addresses this
  process vetted.
- **Fail-closed auth, and no raw SQL.** The API refuses to boot on a
  non-loopback bind without a real bearer key, and the token comparison is
  constant-time. Every query in the data layer is typed ORM with bound
  parameters: no `text()`, no string-built SQL, no dynamic `ORDER BY`.
- **Local-first by default.** Loopback bind, a local SQLite file, telemetry
  off, and the MCP server as a locally-spawned stdio child rather than a
  network service. The only outbound traffic in the default posture is what you
  configured.

The real answer to the residual is epistemic rather than technical: every
particle carries its provenance, source trust is a read-time lens that
discounts a distrusted source without rewriting anything, and `lint` / `review`
turn your rulings on contradictions into a reusable trust policy. The store is
a record of *what sources said, weighted by how much you trust them* — not an
oracle — so a poisoned source is something you can see, discount, and retract
with the audit trail intact.

The limitations we ask you to read before relying on any of this — the
unauthenticated read surface, verbatim storage of whatever you deposit
(secrets included), the single-operator trust model, and the MCP write boundary
— are in
[SECURITY.md](https://github.com/LinkedParticles/particles-engine-py/blob/main/SECURITY.md).
That is also where to report a vulnerability; please report privately.

## Why Particles?

<!-- BEGIN PROJECTED: what-is (manifest: docs/projection/readme.yaml) -->
Most Retrieval-Augmented Generation systems re-derive knowledge from scratch on every query, operating over raw text chunks without any persistent record of what was learned, how confidently it was held, or where it came from. Particles takes a different approach: rather than retrieving passages, it stores discrete, citable, revisable claims — natural-language sentences paired with structured metadata for confidence, provenance, and uncertainty — so that an agent's knowledge accumulates as an auditable ledger rather than evaporating between sessions. Corpus-entry-level provenance is required at Core, meaning every belief can be traced back to the source that produced it, and the particle store itself is a derived view that can be rebuilt from the corpus by re-running extractors, giving the system both durability and reproducibility.

The Particles standard defines the particle schema, the source-corpus model, the extraction protocol, and a set of operations — deposit, extract, query, lint, review, and reindex — that together form the spine along which an agent's knowledge is built, queried, audited, and revised one belief at a time. A developer working with this loop deposits a source into the append-only corpus, extracts claim-granularity particles with resolved subjects, queries by effective-confidence ranking, lints for contradictions and staleness, and reviews inconsistencies into a reusable trust policy. Because particles carry explicit confidence and provenance, Particles can answer queries that require finding relevant claims about a subject and synthesising them with provenance intact, rather than returning a ranked list of chunks whose epistemic status is opaque.

Particles is a minimal interoperable substrate for claim-granularity agent knowledge — narrower than a formal-ontology knowledge graph such as RDF/SPARQL and more structured than augmented markdown. Anchoring every particle to a subject is what makes this precision possible: the Subject store is the standard's catalogue of real-world entities against which particles are indexed, aligning beliefs to canonical entities that can, in principle, be mapped to external ontologies. A developer choosing Particles is therefore choosing a substrate where each belief is a first-class object with an identity, a confidence score, a source, and a subject — not a position in a vector space.

<!-- sources: p-21f11766, p-3f71ca93, p-7931228a, p-80f9b726, p-a5a42fe0, p-ba8cfa2c, p-cb10c54c, p-d3fd6c56 -->
<!-- END PROJECTED: what-is -->

## Design rationale

<!-- BEGIN PROJECTED: design-rationale (manifest: docs/projection/readme.yaml) -->
The foundation of Particles is the guarantee that nothing already written is ever overwritten. The Source Corpus is constituted as an append-only archive of all materials from which particles have been or may be extracted, meaning that every ingested source and every derived claim accumulates in place rather than displacing what came before. Provenance is carried by each particle from the moment of its creation, and when a source must be handled in a mutable mode rather than a purely append-only one, the system does not silently absorb the change: particles whose provenance location falls within the changed regions are flagged with `PROVENANCE_STALE`, making the disturbance visible rather than burying it. Supersession and retraction therefore propagate deterministically through the provenance graph, touching every claim whose origin is implicated, so the ledger remains a faithful record of what was asserted and when, never a quietly revised version of the past.

Trust in Particles is not a property baked into a stored record but a lens applied at the moment of reading. The raw confidence value is stored immutably, calibrated at the time a particle is created and never subsequently altered; what changes with each query is the effective confidence, which is computed at query time rather than stored. Trust weighting, recency decay, and the as-of instant are all applied per observer at query time, never written back into the record. This means that a belief does not need to be retracted or rewritten to sink in the rankings as its source ages or as the extractor that produced it loses credibility: the stored particle remains intact while the effective confidence computed over it falls, allowing stale beliefs to recede naturally without any mutation of the underlying ledger. When `as_of` is set to a past instant T, the query evaluates the store on the assertion-time axis — what the store believed at T — rather than on any world-time validity dimension, so the same immutable corpus can answer questions about present belief and historical belief without ambiguity.

Contradictions in Particles are treated as first-class citizens of the knowledge graph rather than anomalies to be suppressed. When two sources or two principals disagree, the conflict-resolution ladder surfaces an `INCONSISTENCY` record for review instead of silently overwriting either side, preserving both competing claims in the ledger and making the disagreement legible to any downstream consumer. Contestedness, the property that marks a particle as being in tension with one or more others, must be computed at read time and is never stored; like effective confidence, it is a read-time lens rather than a durable annotation, which means that as new particles arrive or old ones are superseded, the contested status of any given claim is always freshly derived from the current state of the graph rather than from a stale flag written at some earlier moment.

Memory in Particles maintains itself through a periodic, automated health discipline. The Lint operation performs health checks for contradictions, stale claims, orphan pages, and missing cross-references, scanning the corpus on a scheduled basis so that problems accumulate into visible findings rather than silently degrading the quality of the belief store. Because nothing is overwritten and contradictions are surfaced rather than resolved by fiat, Lint operates on a stable substrate: it traverses the same immutable record that queries read, and its findings feed back into the trust-policy review cycle, closing the loop between ingestion, querying, and the ongoing stewardship of the ledger's integrity.

<!-- sources: p-0296afc0, p-8476c2ac, p-8767ad66, p-9cf41a46, p-c3b1cef6, p-cea0186c, p-d0d41702, p-f1ced6c4, p-f35e1dee -->
<!-- END PROJECTED: design-rationale -->

## Architecture

<!-- BEGIN PROJECTED: architecture (manifest: docs/projection/readme.yaml) -->
A Particles deployment is organized around two distinct storage subsystems: a Source Corpus store and a Particle store. The Source Corpus store is an append-only object store, compatible with a local filesystem, an S3-compatible backend, or a content-addressed store, while corpus entries and snapshot metadata are held in queryable form, either relational or document store. Together these two subsystems form the durable foundation on which all operations act.

The Core SDK exposes both a clean Python API and a CLI, and the six core operations — Deposit, Extract, Query, Lint, Reindex, and Review — each map to a CLI subcommand and a `POST` API endpoint in the reference SDK. A developer therefore has two symmetric surfaces for every operation: a scriptable command line suitable for pipelines and a programmatic endpoint suitable for application integration.

The journey from a raw source to a set of particles begins with Deposit, which writes the source into the append-only corpus, and continues with Extract, which drives the extraction pipeline. That pipeline selects the most specific registered extractor whose applicability specification matches the deposited source; the general extractor is used if and only if no domain-specific extractor applies. Once particles exist in the Particle store, Query retrieves them ranked by effective confidence, Lint checks for contradictions and staleness, Reindex rebuilds derived indexes, and Review consolidates inconsistencies into a reusable trust policy. Each step in this sequence is a discrete, named operation rather than an implicit side effect, which means a developer can invoke any stage independently and observe its inputs and outputs in isolation.

<!-- sources: p-3bd3740c, p-5a941962, p-8f670063, p-a4bdde07, p-ac28fba0, p-cb1cd190, p-e5639e78 -->
<!-- END PROJECTED: architecture -->

> The *Why Particles?*, *Design rationale*, and *Architecture* sections are
> **cited projections of a particle store**, not hand-authored prose. Each
> marked block is rendered from
> [`docs/projection/readme.yaml`](https://github.com/LinkedParticles/particles-engine-py/blob/main/docs/projection/readme.yaml)
> and the corpus bundle committed beside it, and the trailer under each block
> names the exact claims it was built from. Edit the manifest or the underlying
> particles and re-render — never the prose between the sentinels. Check it
> yourself: `python scripts/projection_drift.py` restores the bundle into a
> throwaway store, re-derives the deterministic render, and fails if any block
> has drifted from the claims it cites.
>
> This is the project eating its own dogfood: the README argues for
> claim-granularity knowledge with provenance, and is itself assembled that way.

## Documentation

Everything below is served at
**[docs.linkedparticles.org](https://docs.linkedparticles.org)**.

| Guide | What it covers |
|---|---|
| [User guide](https://docs.linkedparticles.org/user-guide/) | Depositing, extracting, querying, as-of time travel, exporting, agent integrations |
| [Operator guide](https://docs.linkedparticles.org/operator-guide/) | Configuration, tuning, lint and review, remote engines, containers, observability, troubleshooting |
| [Plugin-author guide](https://docs.linkedparticles.org/plugin-author-guide/) | Writing extractors, exporters, and benchmark suites |
| [CLI reference](https://docs.linkedparticles.org/cli-reference/) | Every verb and flag |
| [HTTP API](https://docs.linkedparticles.org/api/http/) | The OpenAPI contract |
| [Benchmarks](https://docs.linkedparticles.org/benchmarks/) | LongMemEval results, method, and comparators |

The standard itself — whitepaper, technical specification, and the normative
schema, context, and vocabulary artifacts — lives at
**[linkedparticles.org](https://linkedparticles.org)**.

## The three repositories

| Repo | What it is |
|---|---|
| [`particles-standard`](https://github.com/LinkedParticles/particles-standard) | The standard: whitepaper, technical specification, normative schema + SHACL artifacts, conformance fixtures |
| [`particles-core-py`](https://github.com/LinkedParticles/particles-core-py) | The Python Client layer ([`linkedparticles-core`](https://pypi.org/project/linkedparticles-core/)) |
| [`particles-engine-py`](https://github.com/LinkedParticles/particles-engine-py) | **This repo** — the Python Engine layer + surfaces (`linkedparticles`) |

## Deployment

Container and chart artifacts live under
[`deploy/`](https://github.com/LinkedParticles/particles-engine-py/tree/main/deploy).
The engine ships as a single-writer service; see
[running in a container](https://docs.linkedparticles.org/operator-guide/container-deployment/).

## Contributing

See
[CONTRIBUTING.md](https://github.com/LinkedParticles/particles-engine-py/blob/main/CONTRIBUTING.md)
and
[ARCHITECTURE.md](https://github.com/LinkedParticles/particles-engine-py/blob/main/ARCHITECTURE.md).
Contributions are accepted under a Developer Certificate of Origin sign-off —
there is no CLA.

## License

Apache-2.0. See
[LICENSE](https://github.com/LinkedParticles/particles-engine-py/blob/main/LICENSE)
and
[NOTICE](https://github.com/LinkedParticles/particles-engine-py/blob/main/NOTICE).

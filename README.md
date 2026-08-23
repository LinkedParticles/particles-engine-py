# linkedparticles

> **Particles is shared memory for humans and AI agents.** Each particle is one
> claim, plus what you need to judge it: who said it, where, when, and how
> confident they were. Facts, opinions, and memories are all claims, recorded
> the same way as particles. Particles are not edited or deleted. Particles are
> superseded, retracted, or disputed in the open. How much to trust it is a
> perspective applied at query time, never baked into the record.

The **Engine layer** of the Particles reference implementation — the
state-holding SDK and its surfaces. This is the package you install to actually
run a Particles knowledge store. The core loop:

- **deposit** source material into an append-only corpus;
- **extract** claim-granularity particles with subject resolution and
  confidence/provenance metadata;
- **query** with effective-confidence ranking and subject filtering;
- **lint** for contradictions and staleness;
- **review** inconsistencies into a reusable source-trust policy.

When your agent is wrong, you can see exactly why, and fix it at the source.

The Engine is a library first. The FastAPI server, the CLI, the read-only MCP
server, and the resident daemon are all *surfaces* over it.

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

## Install

```bash
pip install linkedparticles
```

This depends on `linkedparticles-core` (the store-free Client layer) — it is
pulled in automatically.

## Architecture

<!-- BEGIN PROJECTED: architecture (manifest: docs/projection/readme.yaml) -->
A Particles deployment is organized around two distinct storage subsystems: a Source Corpus store and a Particle store. The Source Corpus store is an append-only object store, compatible with a local filesystem, an S3-compatible backend, or a content-addressed store, while corpus entries and snapshot metadata are held in queryable form, either relational or document store. Together these two subsystems form the durable foundation on which all operations act.

The Core SDK exposes both a clean Python API and a CLI, and the six core operations — Deposit, Extract, Query, Lint, Reindex, and Review — each map to a CLI subcommand and a `POST` API endpoint in the reference SDK. A developer therefore has two symmetric surfaces for every operation: a scriptable command line suitable for pipelines and a programmatic endpoint suitable for application integration.

The journey from a raw source to a set of particles begins with Deposit, which writes the source into the append-only corpus, and continues with Extract, which drives the extraction pipeline. That pipeline selects the most specific registered extractor whose applicability specification matches the deposited source; the general extractor is used if and only if no domain-specific extractor applies. Once particles exist in the Particle store, Query retrieves them ranked by effective confidence, Lint checks for contradictions and staleness, Reindex rebuilds derived indexes, and Review consolidates inconsistencies into a reusable trust policy. Each step in this sequence is a discrete, named operation rather than an implicit side effect, which means a developer can invoke any stage independently and observe its inputs and outputs in isolation.

<!-- sources: p-3bd3740c, p-5a941962, p-8f670063, p-a4bdde07, p-ac28fba0, p-cb1cd190, p-e5639e78 -->
<!-- END PROJECTED: architecture -->

> The *Why Particles?*, *Design rationale*, and *Architecture* sections are
> **cited projections of a particle store**, not hand-authored prose. Each
> marked block is rendered from [`docs/projection/readme.yaml`](docs/projection/readme.yaml)
> and the corpus bundle committed beside it, and the trailer under each block
> names the exact claims it was built from. Edit the manifest or the underlying
> particles and re-render — never the prose between the sentinels. Check it
> yourself: `python scripts/projection_drift.py` restores the bundle into a
> throwaway store, re-derives the deterministic render, and fails if any block
> has drifted from the claims it cites.
>
> This is the project eating its own dogfood: the README argues for
> claim-granularity knowledge with provenance, and is itself assembled that way.

## The three repositories

| Repo | What it is |
|---|---|
| [`particles-standard`](https://github.com/LinkedParticles/particles-standard) | The standard: whitepaper, technical specification, normative schema + SHACL artifacts, conformance fixtures |
| [`particles-core-py`](https://github.com/LinkedParticles/particles-core-py) | The Python Client layer (`linkedparticles-core`) |
| [`particles-engine-py`](https://github.com/LinkedParticles/particles-engine-py) | **This repo** — the Python Engine layer + surfaces (`linkedparticles`) |

## Deployment

Container and chart artifacts live under [`deploy/`](deploy/). The engine ships
as a single-writer service; see the
[operator guide](docs/operator-guide/index.md).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [ARCHITECTURE.md](ARCHITECTURE.md).
Contributions are accepted under a Developer Certificate of Origin sign-off —
there is no CLA.

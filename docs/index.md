<!--
  The landing page for the engine repository's own documentation.

  Authored in the private development upstream under publish/overlays/ and laid
  into this tree by the release export: edit it there, not on the public repo (a
  direct edit here is overwritten by the next export).

  These docs are the SDK's guides and CLI/HTTP reference. The project's landing
  page — the belief-ledger lead, the demos, the three repository doors — is the
  standard repository's site at https://linkedparticles.org, so this page is
  deliberately a short index and not a second front door.

  The API reference is generated from source docstrings across BOTH
  distributions: the build materialises the Client half into the tree first, so
  the reference is whole rather than Engine-only. See the header of mkdocs.yml.
-->
# Particles SDK documentation

The guides for **running and extending** the Particles reference
implementation. The project's front door, the whitepaper, the technical
specification, and the normative schema artifacts live at
[linkedparticles.org](https://linkedparticles.org).

Particles stores an AI system's knowledge as a git-like ledger of claims: every
belief is sourced, dated, and confidence-scored, nothing is overwritten, and
trust and staleness are applied as a lens at query time. If that framing is new,
start with the [whitepaper](https://linkedparticles.org/spec/whitepaper/).

## Where to go

- **[User guide](user-guide/index.md)** — deposit, extract, query, lint,
  export. Start at [getting started](user-guide/getting-started.md).
- **[Operator guide](operator-guide/index.md)** — running a store long-term:
  configuration, tuning, review, consolidation, troubleshooting.
- **[Plugin-author guide](plugin-author-guide/index.md)** — adding an
  extractor, an exporter, or a benchmark suite.
- **[CLI](cli.md)** — the workflow-oriented command index, with the
  [full command reference](cli-reference.md) beside it.
- **[API reference](api/schema.md)** — generated from the source docstrings,
  across both distributions: the schema models and extraction from the Client
  layer, the corpus, store, and operations from the Engine.
- **[HTTP API](api/http.md)** — every operation as a typed endpoint, rendered
  from the committed OpenAPI contract.

## The standard

This SDK is one implementation of an open standard. The normative documents and
artifacts are published separately, so a second implementation has something to
conform to:

- [Whitepaper](https://linkedparticles.org/spec/whitepaper/) and
  [technical specification](https://linkedparticles.org/spec/technical-specification/)
- [Vocabulary](https://linkedparticles.org/vocab) — every term at the
  identifier it resolves to
- [`particle.schema.json`](https://linkedparticles.org/schemas/particle.schema.json),
  [`context.jsonld`](https://linkedparticles.org/schemas/context.jsonld), and
  the SHACL shapes, served at the identifiers published particles carry
- [The standard's repository](https://github.com/LinkedParticles/particles-standard)

## The three repositories

| Repository | What it is |
|---|---|
| [particles-standard](https://github.com/LinkedParticles/particles-standard) | The standard: spec prose, normative artifacts, conformance fixtures |
| [particles-engine-py](https://github.com/LinkedParticles/particles-engine-py) | This repository — the Engine and its surfaces (`linkedparticles`) |
| [particles-core-py](https://github.com/LinkedParticles/particles-core-py) | The store-free Client layer (`linkedparticles-core`) |

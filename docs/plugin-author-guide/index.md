# Plugin author's guide

You're writing a new extractor, Subject Authority, exporter, benchmark
suite, or conformance fixture. Each page below is that family's
contract: what to build, where it plugs in, and the rules no function
signature states — identity constants you must never change, naming,
registration, lifecycle.

| You want to | Start here | Protocol lives in |
|---|---|---|
| Write a new extractor (Reddit-like source, domain API, file format) | [Extractors](extractors.md) | [`particles/extraction/`](https://github.com/LinkedParticles/particles-core-py/tree/main/particles/extraction) |
| Resolve subjects against a new external ontology | [Subject authorities](subject-authorities.md) | [`particles/ingest/authorities/`](https://github.com/LinkedParticles/particles-engine-py/tree/main/particles/ingest/authorities) |
| Write a new exporter (Notion, JSON Lines, …) | [Exporters](exporters.md) | [`particles/exporters/`](https://github.com/LinkedParticles/particles-engine-py/tree/main/particles/exporters) |
| Add a benchmark suite for an extractor you already have | [Benchmark suites](benchmark-suites.md) | [`particles/benchmark/`](https://github.com/LinkedParticles/particles-engine-py/tree/main/particles/benchmark) |
| Add a conformance fixture for an extractor | [Conformance](conformance.md) | [`particles/conformance/`](https://github.com/LinkedParticles/particles-core-py/tree/main/particles/conformance) |

**Extraction and conformance live in the other distribution.** They are
Client-layer, so they ship in `linkedparticles-core` and their source is in
the `particles-core-py` repository; Subject Authorities, exporters and
benchmark are Engine-layer and live here. Both distributions import as the
same `particles` package, so this changes where you *read* the code, not how
you import it.

## The guide is the contract

Each family page states that family's contract in full: the protocol's
shape, and the normative rules around it that a signature cannot carry —
which module constants are identity and must stay stable across versions,
how a plugin is named and registered, what it may and may not do at
extraction or export time, and what a change to an existing plugin
obliges. Those rules are binding; a plugin that follows the protocol's
signatures but breaks one of them is a broken plugin.

The protocol itself lives once, in `particles/<package>/registry.py` (or
the family's `schema.py` / `contract.py`), and the guide links it rather
than copying it. Read the two together: the page for the rules, the source
for the exact signatures. If they ever disagree, that is a bug in one of
them — please report it.

## Where the plumbing lives

Every plugin family follows the same shape:

```
particles/<package>/
├── registry.py          # XxxPlugin Protocol + get_xxxs() registry
└── <name>.py            # Your new plugin module (FORMAT = "name")
```

Two-file rule: a new plugin is *one new module* + *one line added
to the registry's `_make_*` function*. No CLI changes, no `app.py`
changes, no FastAPI changes.

## What about a brand-new plugin family?

If you want to add an entirely new family (e.g. "exporters for
custom storage backends"), that's an architectural change rather than a
plugin. Raise it as a proposal before building, per the contribution
process in the repository's `CONTRIBUTING.md`.

## Front-end clients (not SDK plugins)

Front-end *clients* of the engine (distinct from the in-package
plugin families above) live in the top-level `clients/` directory and
are built separately from the Python SDK. They consume the frozen
FastAPI contract,
not the `XxxPlugin` registries.

- **Obsidian lint-callout plugin** (`clients/obsidian-plugin/`):
  a
  TypeScript Obsidian community plugin that renders
  [`POST /lint`](../operator-guide/lint-and-review.md) findings as in-vault
  callouts and lets you act on each one (link / confirm / retract) over the
  engine. It complements, and never replaces, the read-only `obsidian`
  [exporter](../user-guide/exporting.md#obsidian-vault).
- **Web UI** (`clients/web-ui/`): the browser surface the engine serves at
  `/app`; see
  [Operator guide → running in a container](../operator-guide/container-deployment.md#the-web-ui).

## The other two doors

Plugins are written against a running system, so the other two guides are
where the behaviour you are extending is described:

- [User guide](../user-guide/index.md): what deposit, extract, query and
  export look like from the outside. [Exporting](../user-guide/exporting.md)
  in particular shows the flags your exporter will be invoked with.
- [Operator guide](../operator-guide/index.md): the knobs an operator turns
  against your plugin; [tuning](../operator-guide/tuning.md) for extractor
  trust and calibration, [configuration](../operator-guide/configuration.md)
  for how your config sub-model is loaded.

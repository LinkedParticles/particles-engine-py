# Plugin author's guide

You're writing a new extractor, exporter, benchmark suite, or
conformance fixture. This guide gets you to "where do I start" in
five minutes; the authority for each plugin family is the protocol
declared in that package's `registry.py`.

| You want to | Start here | Protocol lives in |
|---|---|---|
| Write a new extractor (Reddit-like source, domain API, file format) | [Extractors](extractors.md) | [`particles/extraction/`](https://github.com/LinkedParticles/particles-core-py/tree/main/particles/extraction) |
| Write a new exporter (Notion, JSON Lines, …) | [Exporters](exporters.md) | [`particles/exporters/`](https://github.com/LinkedParticles/particles-engine-py/tree/main/particles/exporters) |
| Add a benchmark suite for an extractor you already have | [Benchmark suites](benchmark-suites.md) | [`particles/benchmark/`](https://github.com/LinkedParticles/particles-engine-py/tree/main/particles/benchmark) |
| Add a conformance fixture for an extractor | [Conformance](conformance.md) | [`particles/conformance/`](https://github.com/LinkedParticles/particles-core-py/tree/main/particles/conformance) |

**Two of those four live in the other distribution.** Extraction and
conformance are Client-layer, so they ship in `linkedparticles-core` and
their source is in the `particles-core-py` repository; exporters and
benchmark are Engine-layer and live here. Both distributions import as the
same `particles` package, so this changes where you *read* the code, not how
you import it.

## The two-source rule

This guide is the **welcome mat**: it explains the shape of each plugin
family and links to a worked example. The **code is the contract** —
protocol signatures, normative naming, lifecycle rules. When this guide and
the code disagree, the code wins.

Why split: the guide is for a stranger landing cold; the code is for a
contributor already inside it. Different audiences, different reading paths.
We deliberately don't duplicate the protocol definitions — your
`<X>Plugin` protocol lives once, in `particles/<package>/registry.py`, and
that file is what to read when this page runs out.

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
plugin — raise it as a proposal before building, per the contribution
process in the repository's `CONTRIBUTING.md`.

## Front-end clients (not SDK plugins)

Front-end *clients* of the engine — distinct from the in-package
plugin families above — live in the top-level `clients/` directory and
are built separately from the Python SDK. They consume the frozen
FastAPI contract,
not the `XxxPlugin` registries.

- **Obsidian lint-callout plugin** (`clients/obsidian-plugin/`) — a
  TypeScript Obsidian community plugin that renders
  [`POST /lint`](../operator-guide/lint-and-review.md) findings as in-vault
  callouts and lets you act on each one (link / confirm / retract) over the
  engine. It complements — never replaces — the read-only `obsidian`
  [exporter](../user-guide/exporting.md#obsidian-vault).
- **Web UI** (`clients/web-ui/`) — the browser surface the engine serves at
  `/app`; see
  [Operator guide → running in a container](../operator-guide/container-deployment.md#the-web-ui).

## The other two doors

Plugins are written against a running system, so the other two guides are
where the behaviour you are extending is described:

- [User guide](../user-guide/index.md) — what deposit, extract, query and
  export look like from the outside. [Exporting](../user-guide/exporting.md)
  in particular shows the flags your exporter will be invoked with.
- [Operator guide](../operator-guide/index.md) — the knobs an operator turns
  against your plugin: [tuning](../operator-guide/tuning.md) for extractor
  trust and calibration, [configuration](../operator-guide/configuration.md)
  for how your config sub-model is loaded.

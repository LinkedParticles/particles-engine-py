# Plugin author's guide

You're writing a new extractor, exporter, benchmark suite, or
conformance fixture. This guide gets you to "where do I start" in
five minutes; the canonical contract for each plugin family lives
in the relevant package's `AGENTS.md`, which is the authority.

| You want to | Start here |
|---|---|
| Write a new extractor (Reddit-like source, domain API, file format) | [Extractors](extractors.md) → `particles/extraction/AGENTS.md` |
| Write a new exporter (Notion, JSON Lines, …) | [Exporters](exporters.md) → `particles/exporters/AGENTS.md` |
| Add a benchmark suite for an extractor you already have | [Benchmark suites](benchmark-suites.md) → `particles/benchmark/AGENTS.md` |
| Add a conformance fixture for an extractor | [Conformance](conformance.md) → `particles/conformance/AGENTS.md` |

## The two-source rule

This guide is the **welcome mat**: it explains the shape of each
plugin family and links to a worked example. The `AGENTS.md` file
inside each `particles/<package>/` is the **contract**: protocol
signatures, normative naming, lifecycle rules. When the two disagree,
`AGENTS.md` wins.

Why split: the guide is for a stranger landing cold; `AGENTS.md` is
for a contributor (human or LLM) inside the codebase. Different
audiences, different reading paths. We deliberately don't duplicate
the protocol definitions — your `<X>Plugin` protocol lives once, in
`particles/<package>/registry.py`, and is described in
`particles/<package>/AGENTS.md`.

## Where the plumbing lives

Every plugin family follows the same shape:

```
particles/<package>/
├── registry.py          # XxxPlugin Protocol + get_xxxs() registry
├── <name>.py            # Your new plugin module (FORMAT = "name")
└── AGENTS.md            # The contract, including the two-file rule
```

Two-file rule: a new plugin is *one new module* + *one line added
to the registry's `_make_*` function*. No CLI changes, no `app.py`
changes, no FastAPI changes.

## What about a brand-new plugin family?

If you want to add an entirely new family (e.g. "exporters for
custom storage backends"), that's an ADR-class change, not a
plugin. Open a `proposed/` ADR per the
[doc-lifecycle policy](../AGENTS.md).

## Front-end clients (not SDK plugins)

Front-end *clients* of the engine — distinct from the in-package
plugin families above — live in the top-level `clients/` directory and
are built separately from the Python SDK. They consume the frozen
FastAPI contract,
not the `XxxPlugin` registries.

- **[Obsidian lint-callout plugin](https://github.com/LinkedParticles/particles-engine-py/tree/main/clients/obsidian-plugin)**
  (`clients/obsidian-plugin/`)
  — a TypeScript Obsidian community plugin that renders `POST /lint`
  findings as in-vault callouts and lets you act on each one (link /
  confirm / retract) over the engine. The canonical build/install/scope
  detail lives in that directory's `README.md`. It complements — never
  replaces — the read-only `obsidian` exporter.

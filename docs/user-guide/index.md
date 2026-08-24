# User guide

You want to use Particles to manage knowledge — deposit sources,
extract claims, query, export.

This guide is task-oriented: each page walks you through one job
end-to-end. Read [Getting started](getting-started.md) first; the
other pages are reference-style and you can dip into whichever you
need.

| Page | When to read |
|---|---|
| [Getting started](getting-started.md) | First-time setup → first query |
| [Concepts](concepts.md) | What's a particle, subject, status, confidence, provenance |
| [Querying](querying.md) | Semantic search, tag filters, structural filters, MCP, what the ranking does |
| [As-of time travel](as-of.md) | What the store believed at a past instant, and what replaced it |
| [Exporting](exporting.md) | Obsidian vault, Anki deck, wiki articles, Logseq graph |
| [Graph view](graph-view.md) | The scoped epistemic subgraph as a self-contained HTML file |
| [Using from LangChain](integrations.md) | Consume a Particles store as LangChain tools / a retriever |
| [Claude Code memory](claude-code.md) | Wire a store in as managed agent memory — session-start push, session-end harvest |
| [Swapping in for the reference memory server](memory-server-swap.md) | Drop-in replacement for `@modelcontextprotocol/server-memory` |
| [Depositing from your phone](inbox.md) | Deposit URLs from the iOS Share Sheet via HTTP or an iCloud inbox file |

For operator-side concerns (long-term tuning, lint hygiene,
troubleshooting), see the [operator guide](../operator-guide/index.md).
For writing a new extractor or exporter, see the
[plugin-author guide](../plugin-author-guide/index.md).

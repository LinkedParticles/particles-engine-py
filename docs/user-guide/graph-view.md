# Graph view — seeing the epistemics

`particles export graph` renders a **scoped subgraph** of your store as a
single self-contained HTML file: no server, no build toolchain, no CDN — the
graph library (Cytoscape.js, MIT) and all data are inlined, so the file opens
over `file://` and works air-gapped.

What makes it different from every other memory product's graph view is what
it renders. The schema already *is* a graph — Subjects are nodes,
multi-subject particles are edges, single-subject particles are node
properties — so the view has no topology extraction step to get wrong.
Instead it spends its visual channels on the epistemics:

| Channel | Meaning |
|---|---|
| **Opacity** | Effective confidence, computed at render time (never stored). Decay renders as literal fading. |
| **Form** | Status: solid = ACTIVE; dashed ghost = SUPERSEDED (with its successor chain in the panel); dotted amber = PROVENANCE_STALE; ☒ = RETRACTED tombstone. |
| **⚠ badge** | Contested — open the panel for the fired bases (`stance` / `divergence` / `inconsistency`) and the drill-down ids. |
| **Node size** | Utility evidence: how often the belief was demonstrably used. Display only — it never changes a confidence. |
| **Node shade** | The best-supported claim on that subject (a labeled display aggregate). |
| **Bold blue** | A retrieval hit, in query scope. |

Click any node or edge to open the detail panel: every visual claim is one
click from its underlying particles — content, stored vs effective
confidence, status, provenance link, supersession chain.

## Scoped, always

A whole-store render does not exist — that is the anti-hairball rule
, and it is deliberate: a 10,000-node force layout answers no
question. Every render is anchored:

```bash
# One Subject's neighbourhood (1 hop by default; --hops up to graph.max_hops)
uv run particles export graph pluto.html --subject <subject-id>

# One query's retrieval set — the picture of the knowledge a query consults
uv run particles export graph answer.html --query "is Pluto a planet?"
```

Renders are bounded by `graph.max_nodes` and `graph.max_particles_per_subject`
(see `config.yaml.sample`). When a cap binds, the page says so in a banner —
"showing 150 of 412 subjects" — and the machine-readable census in the file
carries the exact counts. A capped view is a disclosed lower bound, never a
silent truncation.

## History and time travel

```bash
# Include retired ancestors as ghosts (page gets a "show history" toggle)
uv run particles export graph pluto.html --subject <subject-id> --history

# Render the graph as believed at an instant (as-of lens)
uv run particles export graph pluto-2000.html --subject <subject-id> --as-of 2000-01-01
uv run particles export graph pluto-2007.html --subject <subject-id> --as-of 2007-01-01
```

`--history` walks each in-scope particle's supersession chain and renders the
retired ancestors as dashed ghosts, with the directed chain in the panel.

`--as-of` renders the store's beliefs *as they stood at T*: visibility and
decay evaluate at T, while trust and the contested marker stay current
(temporal-vs-judgment rule). Retirements the store cannot date are
excluded fail-closed and counted in the banner. Export the same subject at two
instants — before and after a supersession — and you can watch a belief get
demoted between the two files. (Seed the demo store with
`scripts/seed_pluto_demo.py` and try the 2000 vs 2007 renders above.)

## Served: `GET /graph` and the MCP `graph_view` tool

The same render is available on the wire (shipped):
`GET /graph` on the FastAPI engine returns the identical `GraphData` contract
the static export embeds — one build, two presentations. Scope is mandatory
here too (an unscoped request is 422). All four scopes are
served: `scope=subject&subject_id=…`, `scope=query&q=…`,
`scope=inconsistency&inconsistency_id=…` (a contradiction's evidence: the
INCONSISTENCY record as the anchor plus its disputants with their true
statuses — the quarantined loser included; accepts a full id or a unique
prefix), and `scope=projection&manifest=…&section=…` (a manifest
section's deterministic selection, addressed by region id or exact title).
The same `hops` / `history` / `as_of` / `max_nodes` params apply, plus
`store` to target a non-default store. The endpoint is bearer-gated and
rate-limited like `POST /query` (query scope drives a paid embedding).

In the web UI, contested rows link straight into the inconsistency scope:
the Curate tab's CONTESTED card and any particle row's contested text carry
a "show the conflict" link that opens the evidence render with the panel
already listing the INCONSISTENCY and both disputants.

Over MCP, the read-registered `graph_view` tool returns the scoped
`GraphData` inline — how an agent hands you the picture of the knowledge it
consulted — and, when `engine.base_url` is configured, adds a `url` field
deep-linking the same scope on the unified web UI (`/app#/browse?…`), where
the render is interactive and carries the as-of scrubber.

## Options

| Flag | Effect |
|---|---|
| `--subject <id-or-name>` | Neighbourhood scope; accepts a subject id or an exact case-insensitive canonical name / alias (scopes are mutually exclusive) |
| `--query "<q>"` | Retrieval-set scope (top `graph.query_top_k` hits) |
| `--inconsistency <id>` | A contradiction's evidence: the INCONSISTENCY anchor + its disputants with their true statuses (full id or unique prefix) |
| `--manifest <path> --section <name>` | A projection manifest section's deterministic selection (region id or exact title) |
| `--hops N` | Neighbourhood radius (clamped to `graph.max_hops`) |
| `--history` | Include supersession-chain ghosts + a client-side toggle |
| `--as-of <ISO-8601>` | Single-instant as-of lens |
| `--max-nodes N` | Per-run node cap (clamped to `graph.max_nodes`) |
| `--min-particle-confidence X` | Cross-exporter floor on effective confidence |
| `--include-non-asserted` | Keep DECLINED / HYPOTHETICAL particles |

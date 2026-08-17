# Remote engine (client–server)

By default the CLI runs every verb **in-process** against the local SQLite
store. The client–server mode lets you instead work from a thin
**local client** against an always-on **remote engine** that owns one canonical
archive — so several machines share a single store instead of each drifting into
its own copy.

The engine runs extraction, embedding, query, and lint **server-side**, so the
client machine needs neither the LLM key nor the embedding model.

## Running the engine

On the always-on host (e.g. a Mac Mini):

```bash
# Loopback-only (local dev, no key needed):
particles engine serve localhost:8000

# Exposed on the LAN / a private mesh — a real key is mandatory:
export PARTICLES_API_KEY=<a-long-random-secret>
particles engine serve 0.0.0.0:8000
```

`engine serve` derives `api.bind_host` from the bind and runs the fail-closed gate
**before** the socket opens: a non-loopback bind without a real
`PARTICLES_API_KEY` is **refused up front** rather than serving the write verbs
unauthenticated. (Raw `uvicorn particles.api.app:app` still works and is still
gated, but you must keep `--host` and `api.bind_host` in sync yourself —
`engine serve` removes that footgun.)

## Pointing a client at the engine

On the client machine, set the engine URL (config or env) and the bearer token
(secret — env only, never `config.yaml`):

```yaml
# config.yaml
engine:
  base_url: "http://mac-mini.local:8000"
  timeout_seconds: 60
```

```bash
export PARTICLES_ENGINE_TOKEN=<the-same-secret-the-engine-expects>
particles query "what do we know about X?"   # runs on the engine
```

With `engine.base_url` set, the daily verbs (`deposit`, `extract`, `query`,
`lint`, `review`, `quality`, `reindex`) target the engine. Unset (the default),
everything runs locally as before — the two modes share the same Pydantic models
and the same `operations/` code, reached either in-process or over HTTP.

The client token (`PARTICLES_ENGINE_TOKEN`) is **distinct** from the engine's
`PARTICLES_API_KEY`: the engine reads the key it expects, the client reads the
token it presents. For a single operator they may hold the same value.

## Network exposure

For the MVP, expose the engine over a **private path**, not a public TLS
endpoint:

- **Tailscale** (or another private mesh) — the bind stays effectively private;
  point `engine.base_url` at the engine's tailnet name.
- **SSH local-forward** — `ssh -L 8000:localhost:8000 mac-mini` and set
  `engine.base_url=http://localhost:8000`. The engine binds loopback on its host;
  nothing is exposed publicly.

Both keep the engine off the public internet and sidestep reverse-proxy
client-identification (the per-request loopback gate is not proxy-aware
— deferred work). A public TLS endpoint is out of scope for the MVP.

## What is not remoted yet

Since, the
operator verbs **route or refuse** — with an engine configured, none silently
read or write the *local* store. The endpoint-backed operator verbs (`trust set`
/ `statement-set`, `subjects alias` / `merge` / `split` / `list` / `search` /
`show`, `links add` / `remove` / `suggest`, `particle tag` / `untag` / `show`,
`corpus retract` / `show` / `links suggest` / `dismiss`, `events list` / `show`)
now execute against the engine. **`import vault`** and **`import project`** also
route in remote mode:
the verb walks the client's tree locally and uploads each matched file to the
engine via `deposit_file`, so a thin client can bulk-seed the canonical store in
one command (idempotent, continue-on-per-file-error).

The verbs whose endpoint does not exist yet **fail loud** with an actionable
error naming the verb — run them on the engine host directly, or unset
`engine.base_url` to operate on the local store:

- **`export`** — renders the whole store to the local filesystem; no remote
  bulk-read route yet.
- **`import web-clipper`** — restores per-capture `WEB_PAGE` provenance
  (`uri_r`, publication date) from frontmatter, which `deposit_file` cannot
  carry; stays local-only.
- **`corpus`** — `list`, `delete`, `prune-orphans`, and `links list`.
- **`subjects`** — `set-class`, `delete`, `gc` / `prune-empty`, `confirm`,
  `unlink`, `fix-labels`, `find-duplicates`, and `list --phantoms-only`.
- **`trust`** — `list`, `show`, `set-entry`, `cascade`, and the lens verbs
  (`lens list` / `show` / `adopt` / `unadopt`).
- **`links list`**, and **`particle search`** / **`particle narrative`**.

## The MCP server follows the engine too

Since, the local
stdio MCP server (`particles mcp serve`) routes **all** its tools through the
same backend seam. With `engine.base_url` set, the agent's read tools (`query`,
`particle_show` / `particles_list` / `particle_search`, `subjects_*`,
`events_*`, `quality_report`, `lint`, `list_taxonomies`, `list_corpus_entries`,
`links_suggest`, `corpus_links_suggest`) and its **belief writes**
(`particle_assert` / `particle_supersede` / `particle_retract` / `deposit_text`)
operate on the **canonical engine store**, not the laptop's local one — the agent
recalls and asserts against the same archive the CLI does. The
`particles://digest/<store>` session-start memory resource is rendered by the
engine over HTTP as well.

The MCP server stays **local stdio** (unchanged install path); the
local-stdio write-trust model and the bearer auth carry over unchanged —
the local server presents the same `PARTICLES_ENGINE_TOKEN` the CLI does. The
belief-write tools are still offered only when `mcp.write.enabled_stores` is set
locally; the **engine** independently gates the write server-side on its own
write-enablement. The belief-write endpoints (`POST /particles/{assert,
supersede,retract}`) are now reachable by any authenticated HTTP client, not only
the MCP surface. A *network-exposed* MCP transport (for a non-co-located client
that cannot run a local stdio proxy) remains future work.

Other deliberately-local surfaces (§ Deferred):

- Display-only extras degrade gracefully in remote mode: `extract`'s page-stats /
  carry-forward summary, `deposit`'s follow-target list, `review`'s
  author-enriched detail, and the subject-name / provenance-URI / narrative
  blocks of `particle show` and `corpus show` are shown in local mode only. The
  routed MCP `particle_show` / `subjects_show` likewise degrade their enrichment
  (subject names, provenance URIs) over a remote engine.

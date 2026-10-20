# Remote engine (client-server)

By default the CLI runs every verb **in-process** against the local SQLite
store. The client-server mode lets you instead work from a thin
**local client** against an always-on **remote engine** that owns one canonical
archive, so several machines share a single store instead of each drifting into
its own copy.

The engine runs extraction, embedding, query, and lint **server-side**, so the
client machine needs neither the LLM key nor the embedding model.

## Running the engine

On the always-on host (e.g. a Mac Mini):

```bash
# Loopback-only (local dev, no key needed):
particles engine serve localhost:8000

# Exposed on the LAN / a private mesh; a real key is mandatory:
export PARTICLES_API_KEY=<a-long-random-secret>
particles engine serve 0.0.0.0:8000
```

`engine serve` derives `api.bind_host` from the bind and runs the
fail-closed gate
**before** the socket opens: a non-loopback bind without a real
`PARTICLES_API_KEY` is **refused up front** rather than serving the write verbs
unauthenticated. (Raw `uvicorn particles.api.app:app` still works and is still
gated, but you must keep `--host` and `api.bind_host` in sync yourself;
`engine serve` removes that footgun.)

## Pointing a client at the engine

On the client machine, set the engine URL (config or env) and the bearer token
(a secret: env only, never `config.yaml`):

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
everything runs locally as before. The two modes share the same Pydantic models
and the same `operations/` code, reached either in-process or over HTTP.

The client token (`PARTICLES_ENGINE_TOKEN`) is **distinct** from the engine's
`PARTICLES_API_KEY`: the engine reads the key it expects, the client reads the
token it presents. For a single operator they may hold the same value.

## Network exposure

For the MVP, expose the engine over a **private path**, not a public TLS
endpoint:

- **Tailscale** (or another private mesh): the bind stays effectively private;
  point `engine.base_url` at the engine's tailnet name.
- **SSH local-forward**: `ssh -L 8000:localhost:8000 mac-mini` and set
  `engine.base_url=http://localhost:8000`. The engine binds loopback on its host;
  nothing is exposed publicly.

Both keep the engine off the public internet with no reverse proxy in front
of it, so the per-request loopback gate sees the real peer. If you do put the
engine behind a proxy, list it in `api.trusted_proxies` so the gate reads
`X-Forwarded-For` from that hop only. A public TLS endpoint is out of
scope.

The SSH forward has one further property: an engine bound to loopback with no
`PARTICLES_API_KEY` runs with the development-key skip, and every tunnelled
request arrives from loopback, so the client needs no token for reads, deposits,
extraction, query, lint, and an agent's own belief writes. Use it only when you
are the sole user of the engine host. The operator writes below need a real key
even then.

## Operator writes need a key, even on loopback

Some routes act with the operator's authority rather than an agent's, and the
development-key skip never applies to them, not even from loopback:

| Route | Gesture |
|---|---|
| `POST /review/{id}` | resolve an INCONSISTENCY (`particles review` against an engine) |
| `POST /particles/{id}/supersede` | operator edit of any belief, extracted ones included |
| `POST /particles/{id}/retract` | operator retract of any single belief |
| `POST /particles/{id}/subjects` | assign a subject to an orphan belief |
| `POST /curation/affirm`, `POST /curation/snooze` | the curation queue's "still true", snooze, and dismiss |
| `POST /corpus/{id}/retract` | whole-source retract (a `dry_run` plan is still served) |
| `POST /subjects/relink-gated` | batch relink (a `dry_run` plan is still served) |
| `POST /trust/rules` | add or change a source-trust rule (`particles trust set` against an engine) |
| `POST /trust/statements` | write a trust statement and run its cascade, which can resolve held conflicts (`particles trust statement-set` against an engine) |

The skip treats a loopback caller as the operator, and an agent with a shell on
the engine host is a loopback caller too. With `PARTICLES_API_KEY` unset or
`dev-key`, these routes answer `503` with a message naming the fix; no token the
client sends can change that, because the engine has no key to check it
against. To use them over HTTP, start the engine with a key and give each client
the same value:

```bash
export PARTICLES_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
particles engine serve localhost:8000
```

| Client | Where the token goes |
|---|---|
| `particles` CLI with `engine.base_url` set | `PARTICLES_ENGINE_TOKEN` |
| Web app at `/app` | the bearer token field in its settings |
| Obsidian plugin | the bearer token field in the plugin settings |

The CLI without `engine.base_url`, and the MCP server, run these operations
in-process against the local store and need no key. The MCP server's write
tools are the agent's own-belief writes, which the skip still covers.

## What is not remoted yet

Since, the
operator verbs **route or refuse**: with an engine configured, none silently
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
error naming the verb. Run them on the engine host directly, or unset
`engine.base_url` to operate on the local store:

- **`export`**: renders the whole store to the local filesystem; no remote
  bulk-read route yet.
- **`import web-clipper`**: restores per-capture `WEB_PAGE` provenance
  (`uri_r`, publication date) from frontmatter, which `deposit_file` cannot
  carry; stays local-only.
- **`corpus`**: `list`, `delete`, `prune-orphans`, and `links list`.
- **`subjects`**: `set-class`, `delete`, `gc` / `prune-empty`, `confirm`,
  `unlink`, `fix-labels`, `find-duplicates`, and `list --phantoms-only`.
- **`trust`**: `list`, `show`, `set-entry`, `cascade`, and the lens verbs
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
operate on the **canonical engine store**, not the laptop's local one. The agent
recalls and asserts against the same archive the CLI does. The
`particles://digest/<store>` session-start memory resource is rendered by the
engine over HTTP as well.

The MCP server stays **local stdio** (unchanged install path); the
local-stdio write-trust model and the bearer auth carry
over unchanged: the local server presents the same `PARTICLES_ENGINE_TOKEN` the
CLI does. The belief-write tools are still offered only when
`mcp.write.enabled_stores` is set locally; the **engine** independently gates the write server-side on its own
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

## Claude Code hooks against a remote engine

The hooks installed by `particles init claude-code` follow `engine.base_url` as
well, with three surfaces that still need the store on the same machine:

| Hook step | With an engine configured |
|---|---|
| SessionStart digest | Served by the engine (`GET /digest/{store}`). The freshness check is skipped, so every session receives the full digest |
| SessionEnd harvest | Refused unless `claude_code.harvest.allow_remote: true`, because transcripts are the most sensitive material the SDK handles. With it, the distilled, redacted transcript and the memory files are deposited on the engine, and the engine's own consolidation extracts them |
| SessionEnd MEMORY.md re-render | Skipped. The projected region keeps its last local render |
| SessionEnd utility mining | Skipped |
| Stop hook (agent-side extraction ask) | Skipped, and the MCP `extraction_submit` tool refuses a remote engine |

Every hook still fails open: an unreachable engine, a refused token or a slow
link ends the hook with exit 0 and no output, inside the
`claude_code.hook_deadline_seconds` budget (10 s by default), and the reason is
written to the local hook log (`particles hook log`). `particles hook doctor`
checks only a local store today, so on a thin client read the hook log
instead.

Claude Code starts hooks and MCP servers with the environment their settings
entries declare. A `PARTICLES_ENGINE_TOKEN` for a non-loopback engine therefore
has to be written into those entries, in plaintext; the SSH forward described
above avoids that.

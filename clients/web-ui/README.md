# Particles — unified web UI

One same-origin **Progressive Web App** over the Particles FastAPI engine
: the bus-stop **curation queue**, a **query** surface
over `POST /query`, and the **scoped epistemic graph view** (`GET /graph`) in a single shell with hash routing — plus deposit (URL / text /
file) and settings as global actions. The client holds **no store, no schema
logic, and no reconciliation**: every state change runs on the engine, and
every epistemic quantity it renders (effective confidence, contested badges,
as-of visibility) is **server-computed** — the client only renders (generalized app-wide). *Rigor in the substrate, casualness
in the surface.*

Like every client in `clients/`, it is a typed HTTP client of the frozen
FastAPI contract and lives **outside** the Python package's mypy /
ruff / import-linter scope; it is not part of the wheel. It was born as the curation PWA (`clients/curation-pwa/`) and renamed here.

## Routes

All views are deep-linkable — the route and its parameters live in the hash,
so the engine serves one static bundle and the fragment never hits the server:

| Route | Surface |
|---|---|
| `/app#/curate` | The swipeable, leverage-ranked "today's N" curation feed (`GET /curation`) with the full gesture set (0161) |
| `/app#/query?q=…` | Non-streaming query over `POST /query`: staged progress, the cited answer, subject chips, and an inline lazy **knowledge-consulted graph** |
| `/app#/browse?scope=subject&subject_id=…` `/app#/browse?scope=query&q=…` | The scoped epistemic graph over `GET /graph`, with census + disclosures, history ghosts, and the per-instant **as-of scrubber** (`&as_of=…`, `&hops=…`, `&history=true`) |

The browse route's parameters transcribe the `GET /graph` wire params
one-to-one, so a deep link *is* the API call it drives. The MCP `graph_view`
tool emits these links (its `url` field) when `engine.base_url` is configured.
The pre-rename route names (`#/queue`, `#/graph`) are permanent aliases.

## Same-origin by construction (no CORS)

The engine serves this app's built bundle **from its own origin** at `GET /app`
(`particles/api/web_app.py`). The **shell itself is unauthenticated**
 — a browser navigation cannot carry an `Authorization` header, and
gating it made the app unopenable in any browser once a real key was set,
since the settings view where you paste the token is inside the withheld
bundle. Every API path the loaded app calls stays behind the fail-closed bearer. Because the app shell and the API it calls
(`/curation`, `/query`, `/graph`, …) share one origin, the authenticated
`fetch` calls carry no cross-origin `Origin` and trigger **no CORS
preflight** — so the engine adds no CORS middleware and gains no CORS surface
. This is the same failure mode (the "Failed to fetch"
preflight → unhandled `OPTIONS` → 405) **designed out, not patched around**. A
separately-hosted (different-origin) deployment would force a first-class
engine-CORS change and is deferred.

## Run (operators: no build step)

**`dist/` is committed**: `git pull` delivers the working
bundle, and the engine serves it as-is — operators never need npm. Start the
engine and open the app at its own origin:

```bash
# from the repo root
uv run particles engine serve localhost:8000
# then browse to http://127.0.0.1:8000/app
```

This also works against an engine running with a real `PARTICLES_API_KEY`
(including the container): the shell loads unauthenticated and the
app prompts for the bearer, which it then sends on every call.

The engine mounts `/app` only when `clients/web-ui/dist/` is present (it is,
in a source checkout; an installed wheel does not ship it and simply skips
the mount). The app footer shows `web-ui <build stamp> · engine <version>` —
the ground truth for which bundle and engine are actually live; a rebuilt
bundle lands in an open tab automatically on its next visit (the service
worker reloads the page once when a new build takes over).

## Build (UI development only)

From this directory:

```bash
npm install            # once per checkout
npm run build          # gen-api → src/openapi.d.ts, then esbuild → dist/
```

`build` first regenerates the typed client (`src/openapi.d.ts`) from the
committed `artifacts/openapi.json` via `openapi-typescript`, then bundles
`src/` + `public/` into `dist/` — along with the **vendored Cytoscape.js**
(`artifacts/graph/cytoscape.min.js`, the same pinned build the static graph
exporter embeds; same-origin asset, no CDN).

**A change under `src/` or `public/` (or to the vendored Cytoscape / the
OpenAPI contract) must commit the rebuilt `dist/` in the same change.** The
build stamp is deterministic — package version + a hash of the build inputs,
no timestamp — so rebuilding identical sources is a no-op diff, and the
committed bundle changes exactly when the code does. Other scripts:

| Script | What it does |
|---|---|
| `npm run gen-api` | Regenerate `src/openapi.d.ts` from `artifacts/openapi.json` |
| `npm run typecheck` | `gen-api` + `tsc --noEmit` — the contract-conformance gate |
| `npm run dev` | esbuild watch/serve build (development) |
| `npm run build` | `gen-api` + production esbuild bundle into `dist/` |

## Configuration & auth

The app presents the **client bearer** on every call. On first launch
it shows a settings screen; the engine base URL and bearer token are entered
there and persisted client-side (browser storage) — secrets never leave the
device and are never compiled in. Fail-closed both ways (posture,
mirrored client-side): with no token configured the app makes no call at all,
and the engine refuses to serve an unauthenticated non-loopback request. No
cookies, no sessions — bearer-only, the mobile-parity rule (see
`PARITY.md`).

When belief writes are not enabled on the target store
(`mcp.write.enabled_stores` default-deny), a write returns 403 and
the queue **degrades to a read-only review mode** — it still renders; write
gestures are hidden. The query and graph routes are read surfaces and are
unaffected.

## Gestures → endpoints

Each queue card surfaces only the gestures its `CardKind` offers. Every gesture maps onto an endpoint that already ships — the client adds
no new contract surface:

| Gesture | Endpoint |
|---|---|
| affirm ("still true") | `POST /curation/affirm` |
| snooze / dismiss (belief cards) | `POST /curation/snooze` |
| comment | `POST /review/{id}` |
| merge (duplicate pair) | `POST /links` · `POST /subjects/{id}/merge` |
| deposit (uncited URL) | `POST /corpus/deposit/url` (engine fetches + reconciles the citing mentions); manual-paste fallback → `POST /corpus/deposit/text` when the URL can't be fetched |
| dismiss (uncited URL) | `POST /corpus/links/dismiss` |
| supersede / edit | operator `POST /particles/{id}/supersede`; own-beliefs `POST /particles/supersede` |
| retract (single belief) | operator `POST /particles/{id}/retract` |
| assign-subject (NO_SUBJECT card) | `POST /particles/{id}/subjects` |
| whole-source retract | `POST /corpus/{id}/retract` |
| reindex | `POST /reindex` |

The global deposit sheet drives `POST /corpus/deposit/url`,
`POST /corpus/deposit/text`, and — since — the multipart
`POST /corpus/deposit/file` for file uploads.

**Still deferred:** **agree / vouch** — the endorsement primitive is still proposed
— not offered at all.

## Mobile parity (`PARITY.md`)

`PARITY.md` records the app's screen inventory and interaction contract — the
spec the future React Native app mirrors 1:1. Its normative rules
bind this app too: bearer-only auth (no cookies/sessions), no web-only
endpoint affordances, every endpoint in the OpenAPI snapshot, and
server-computed epistemics only.

## Layout

```
clients/web-ui/
  package.json            esbuild + openapi-typescript; npm scripts
  tsconfig.json
  esbuild.config.mjs      bundles src/ + copies public/ + vendored Cytoscape → dist/
  PARITY.md               screen inventory + interaction contract (the spec)
  src/
    main.ts               app entry — shell: nav, routes, global actions
    router.ts             hash routing (#/curate, #/query, #/browse; deep-linkable)
    api.ts                typed HTTP client of the engine (bearer)
    feed.ts               the swipeable leverage-ranked card stack (curate route)
    queryView.ts          the #/query surface
    graphView.ts          the #/browse route: scope picker, seeded landing, as-of scrubber
    graphRender.ts        the shared Cytoscape mount (encodings) used by
                          #/browse and the query page's inline retrieval-set graph
    deposit.ts            global deposit sheet (URL / text / file)
    sheet.ts              per-card gesture sheet
    gestures.ts           gesture → endpoint dispatch
    settings.ts           client-side settings/bearer persistence
    settingsView.ts       settings screen (incl. Appearance: system / light / dark)
    theme.ts              the <html data-theme> stamp + token reads for Cytoscape
    openapi.d.ts          GENERATED (npm run gen-api) — do not edit
  public/
    index.html  styles.css  manifest.webmanifest  service-worker.js  icon.svg
```

**Theming.** Colour, type, space and radius come from the repo-wide design
tokens, `design/tokens.css` (see `design/README.md`); the build copies that
file into `dist/tokens.css` and `index.html` loads it before `styles.css`,
which only aliases its old `--bg` / `--accent` / … names onto the `--p-*`
tokens and holds no literal colour. Light is the default, dark follows the
OS, and Settings → Appearance stamps an explicit `<html data-theme>`. The
graph's Cytoscape styles read the same tokens at render (`theme.ts`), so
node/edge colours and the support-shade ramp flip with the theme too. To
change a colour, edit `design/tokens.css` and rebuild — never `styles.css`.

The service worker caches the **app shell only**, never API responses — a
curated belief, a query answer, or a graph render is never served stale from a
client cache.

See `docs/ADR/active/0225-unified-web-ui.md` (the unification) and
`docs/ADR/active/0154-bus-stop-curation-client.md` (the original curation-PWA
design) for the full rationale.

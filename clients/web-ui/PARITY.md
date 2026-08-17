# PARITY.md — screen inventory & interaction contract

This document is the **normative spec the future React Native mobile app
 mirrors 1:1** — parity by construction, not by retrofit. It
records what the unified web UI does entirely in terms of the FastAPI contract, so a native client can reproduce every capability without
reverse-engineering this codebase. When the web UI gains or changes a
surface, this file changes **in the same commit**.

## Normative rules (binding on this app and every future client)

1. **Bearer-only auth.** Every call presents the client bearer
   (`Authorization: Bearer …`). **No cookies, no sessions, no web-only auth
   affordance.** The credential is entered by the operator, stored
   client-side only (browser storage here; the platform keystore on
   mobile), never compiled in, never in a URL. Fail-closed: with no token
   configured, the client makes **no call at all**.
2. **No web-only endpoint affordances.** Every capability used here is an
   ordinary endpoint of the contract, callable identically by a
   native client. The engine serves this app same-origin at `/app` purely
   as a convenience; nothing about the API depends on that.
3. **All epistemics render server-computed state** (
   generalized app-wide). Effective confidence, contested badges, as-of
   visibility, ranking — computed by the engine per request, **never** by
   the client. A client draws bars and sets opacities from numbers it was
   given; it performs no confidence math of its own.
4. **Every endpoint a client needs is in the OpenAPI snapshot**
   (`artifacts/openapi.json`). New endpoints added for any UI land there in
   the same change, so generated clients cover them. This app's
   own types are generated from the snapshot (`npm run gen-api`); the
   mobile app should generate its client the same way.
5. **Read-only degrade.** A 403 on a belief write means the store has not
   opted into writes (`mcp.write.enabled_stores` default-deny): the client hides write gestures and keeps every read surface
   working. It never retries writes around the gate.
6. **Deep links are API transcriptions.** A shareable location (`#/browse`
   params here; the equivalent screen params on mobile) names exactly the
   API call that renders it — no client-private addressing scheme. Route
   renames never break links: the pre-rename names (`#/queue`, `#/graph`)
   are accepted forever as aliases of `#/curate` / `#/browse`.

## Screen inventory

### 1. Curate (`/app#/curate`) — bus-stop curation

The swipeable, leverage-ranked "today's N" card feed.

| Interaction | API call |
|---|---|
| Load / refresh queue (kind filter, limit, semantic toggle) | `GET /curation?limit&kind&semantic` |
| affirm ("still true") | `POST /curation/affirm` |
| snooze / dismiss (belief card) | `POST /curation/snooze` |
| comment / resolve inconsistency | `POST /review/{particle_id}` |
| merge duplicate pair | `POST /links` (CO_EVIDENTIAL) or `POST /subjects/{source_id}/merge` |
| deposit cited URL | `POST /corpus/deposit/url`; manual-paste fallback `POST /corpus/deposit/text` |
| dismiss uncited URL | `POST /corpus/links/dismiss` |
| edit (operator supersede) | `POST /particles/{id}/supersede` |
| retract single belief | `POST /particles/{id}/retract` |
| assign subject to orphan | `POST /particles/{id}/subjects` |
| whole-source retract (dry-run first, then confirm) | `POST /corpus/{entry_id}/retract` |
| reindex failed snapshots | `POST /reindex` |

Interaction contract: cards surface only the engine-computed
`suggested_gestures`; the client invents none. The primary swipe gesture is
the cheapest card-resolving v1 gesture per kind (see `src/gestures.ts` for
the preference table). Destructive gestures (retract) confirm; whole-source
retract shows the dry-run blast radius first. Unknown gestures from a newer
engine render disabled, never guessed.

### 2. Query (`/app#/query?q=…`) — ask the store

| Interaction | API call |
|---|---|
| Ask a question | `POST /query` (body `{question}`) |

Non-streaming v1: one call per question; the client shows **staged local
progress** (the stages pace expectation only — they do not report engine
state; streaming is deferred). Renders: the NL `answer`; per-hit rows with
`status`, stored `confidence.value` vs `effective_confidences[i]`, the `contested[i]` badge (bases shown, never blended);
`truncation_warning`, `as_of` echo, and `subject_coverage_gaps` as notices.

Cross-links (the unification's payoff — mobile must mirror both):
- **Inline retrieval-set graph** — a collapsed "show the knowledge
  consulted" section under the answer. Expanding fires
  `GET /graph?scope=query&q=<question>` (lazily — never as a hidden tax on
  every query, since query scope re-runs retrieval engine-side) and mounts
  the same graph render as the Browse screen, without the as-of scrubber.
  Under an `answer_refused` response the section inherits the refusal
  framing ("nearest beliefs — likely unrelated"): the graph is
  transparency about retrieval, never visual support for a refused answer.
- **Subject chip** on a hit → browse screen with
  `scope=subject&subject_id=…`.

### 3. Browse (`/app#/browse?…`) — scoped epistemic subgraph

| Interaction | API call |
|---|---|
| Render a scope | `GET /graph?scope=subject\|query\|inconsistency\|projection` + its selector (`subject_id` / `q` / `inconsistency_id` / `manifest`+`section`) `&hops&history&as_of&max_nodes&store` |
| As-of scrubber (per instant) | `GET /graph` re-issued with `as_of=<ISO date>` |

Screen params mirror the wire params one-to-one. The engine enforces the
anti-hairball invariant with 422 (a whole-store render does not exist —
clients must not try to assemble one).

**Inconsistency scope** — a contradiction's evidence: the
INCONSISTENCY record as the anchor plus its disputants with their **true
statuses** (the quarantined loser included), all `retrieval_hit`-flagged.
The client auto-opens the detail panel with the foreground listing
("Conflict evidence", anchor first) — on pre-subject-binding conflicts
there may be zero nodes (engine-disclosed), and the panel is the whole
render. Entry points a client must offer: the CONTESTED curation card's
`inconsistency_id` and the contested drill-down text on any particle row
both link `scope=inconsistency&inconsistency_id=…`. Projection scope
renders a manifest section's deterministic selection the same
way (foreground-flagged); the manifest path resolves engine-side.

A scope-less visit **seeds** the render instead of showing an empty form:
`GET /subjects?order=degree&limit=1` picks the store's most-connected
subject (the ordering is server-computed — the client holds no ranking of
its own), rendered as its 1-hop neighbourhood behind a banner that names
the auto-chosen anchor, with the scope picker kept visible below the
render. An empty store, or a seed-lookup failure (e.g. an older engine
without `order=degree`), degrades to the picker alone. Broken scope params
(e.g. `scope=subject` with no id) also get the picker, not the seed — the
operator was mid-edit.

Render contract (all values arrive in `GraphData`):
- effective confidence → edge opacity / cargo-row bar (decay = fading);
- status → form (ACTIVE solid; SUPERSEDED dashed ghost + supersession
  chain; PROVENANCE_STALE dotted; RETRACTED ☒ tombstone);
- contested → ⚠ badge with the fired bases on the detail panel;
- utility evidence → node size only (never opacity);
- node shade → `max_effective_confidence` display aggregate;
- `disclosures[]` → banner verbatim; `census` → the counts line. A capped
  render is a disclosed lower bound — never hide the disclosure.
- detail panel: click any node/edge → the underlying particles (content,
  stored vs effective confidence, status, source link, chain), because
  every visual claim is one click from its substrate.
- history toggle: client-side show/hide of the ghost set from one
  `history=true` response.
- as-of scrubber: slider over [earliest rendered assertion − 1 day … now];
  release re-queries; the "now" end drops `as_of`. Bounds derive from the
  render's own timestamps (presentation), never from epistemic computation.

### 4. Deposit (global sheet)

| Input | API call |
|---|---|
| URL | `POST /corpus/deposit/url` (engine fetch + importer routing + SSRF guard) |
| Pasted text | `POST /corpus/deposit/text` |
| File | `POST /corpus/deposit/file` (multipart) — the mobile share-sheet path is separately deferred |

Exactly one input per deposit; precedence file > URL > text. On mobile,
the platform share sheet should land on these same endpoints (URL/text via
share payloads; files via the multipart endpoint).

### 5. Settings (global)

Engine base URL (optional; empty = same-origin here — a native client
always sets it), the bearer, and the reviewer id recorded on
`POST /review` resolutions. Stored client-side only. A diagnostics footer
shows the client build version and the engine version from `GET /health`.

**Appearance** (system / light / dark) is a presentation preference, not
connection config: it applies on change, persists client-side, and gates
nothing. The palette and the epistemic encodings it themes are the
design-system tokens (`design/tokens.css`) — a native client uses the same
token values (light and dark) so a particle looks the same on every surface.

## Error-mapping contract (uniform across screens)

| Engine response | Client behaviour |
|---|---|
| 401 | "engine rejected the token" — direct to settings; never retry-loop |
| 403 (belief write) | flip to read-only degrade (rule 5) |
| 404 | not-found notice with the engine's `detail` |
| 422 | request-shape error; show the engine's `detail` verbatim (e.g. graph scope errors) |
| 429 | rate-limited (`Retry-After`); back off, do not auto-retry writes |
| 503 | dev-key loopback refusal — surface the `detail` |
| network failure | "could not reach the engine" with the configured base URL |

## Out of scope (deferred, tracked)

- Streaming query answers — deferred.
- Cursor pagination on browse endpoints — deferred.
- Separate-origin hosting + CORS — deferred.
- Endorse/vouch gesture — proposed.
- HTTP share-sheet deposit (iOS Shortcut → `POST /corpus/deposit/url`) →
  a later milestone.
- The native app itself is deferred — held for a later wrapper; this
  file is the spec it inherits).

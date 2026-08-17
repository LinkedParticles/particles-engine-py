# Security Policy

> For the narrative version — the audit record, the controls that held under
> attack, and why this posture is credible — see the
> [security and trust posture page](docs/security.md).

## Supported versions

Particles is pre-1.0 and under active development. Until the first stable
release (`1.0.0`), only the latest published version receives security
fixes.

| Version | Supported |
|---|---|
| Latest `0.x` release | ✅ |
| Older `0.x` releases | ❌ |

Once `1.0.0` ships, the most recent `1.x` minor release will be supported.

## Deployment posture: authentication and the read surface

The HTTP engine (`particles.api.app`) has an **asymmetric authentication
model** that operators must understand before exposing it on a network.

**The bearer token gates writes, not reads.** Setting `PARTICLES_API_KEY` to a
secret enables bearer-token auth (`Authorization: Bearer <key>`) on every
*mutating* route — deposits, extraction, retraction/supersession, subject
splits/merges, trust changes, link edits, and the agent-write verbs. The
**read** routes are, by default, **unauthenticated even when a real key is
set**: `/particles`, `/corpus`, `/subjects*`, `/quality`, `/lint/report`,
`/taxonomies`, the narrative reads, and the read-only `POST /corpus/links/suggest`.
A client that can reach the port can read the full belief store and corpus
without presenting any credential.

**Three high-value reads are always bearer-gated**, regardless of the knob
below, because the data-confidentiality and cost stakes are higher:

- **`POST /query`** — bills the operator's `ANTHROPIC_API_KEY`. The natural-language
  answer drives a paid Anthropic completion with no per-request rate limit, so an
  open `/query` is an **unbounded-spend** vector, distinct from the
  data-confidentiality concern.
- **`GET /events`** (and `/events/{id}`) — the **operator audit log**:
  retractions, subject splits/merges, trust changes, reviews.
- **`GET /digest/{store}`** — the provenance-ranked roll-up of the full belief
  store, including which beliefs are contested.

**Loopback is the assumed boundary.** The default bind is `127.0.0.1`, and the
fail-closed startup gate refuses to boot when bearer auth is disabled
(`PARTICLES_API_KEY` unset/`dev-key`) **and** the bind is non-loopback — so the
default install is safe. The unauthenticated read surface becomes a real
exposure only when you deliberately bind beyond loopback (e.g.
`engine serve 0.0.0.0:8000`, a documented LAN/Tailscale mode) **with**
a real key set. In that posture, treat the network itself (a private subnet, a
Tailscale tailnet, a reverse proxy that enforces its own auth) as the read-access
control, or close the reads in-app (below).

**Closing the read surface.** Set `api.require_auth_for_reads: true` (config
field, env `PARTICLES_API_REQUIRE_AUTH_FOR_READS`; default `false`) to require
the **same** bearer on every read route. The `dev-key` loopback skip still
applies, so local development is unaffected; only non-loopback / real-key callers
must then present the bearer. This is recommended for any non-loopback
deployment where the network is not itself a sufficient trust boundary.

## Trust model and known limitations

Beyond the network/auth posture above, three design choices are deliberate and
worth understanding before you deploy Particles or feed it sensitive material.

### Ingested content is untrusted — and is stored verbatim, including any secrets

Deposited source material (URLs, files, pasted text, social-media threads) is
**fully untrusted** by design: it flows into the LLM extraction prompts, the
embedding model, the belief store, and exporter/projection output. Particles
tracks provenance and confidence for that content but does **not** sanitise or
scan it. In particular, **any secret in a deposited document — an API key, a
password, a token — is persisted content-addressed in the corpus and can
resurface** in particle content, query answers, the session digest, and
exporter files. Do not deposit material you are unwilling to store unredacted,
and treat the corpus blob store and the belief store as **as sensitive as the
most sensitive document you have deposited**.

(Prompt-injection from deposited content is *contained but not eliminated*:
untrusted text is nonce-fenced before every LLM call, no tool-use /
function-calling is exposed to the model, and no model output is parsed into an
executed action — but an injected document can still influence the *claims
extracted from it*. The belief store is a record of what sources said, weighted
by trust — not an oracle.)

### The MCP write surface is a local-process trust boundary, not a network one

The stdio MCP server (`particles mcp serve`) has **no token**. Its write tools
are gated only by the default-deny `mcp.write.enabled_stores` allowlist (writes
are off unless a store is explicitly enabled). The trust boundary is
therefore the **host**: any local user who can read `config.yaml` / the database
and spawn the MCP server against a write-enabled store can mutate that store.
This is appropriate for the **single-operator, single-host** model Particles
targets — the MCP server is a locally-spawned child process, not a network
service. On a shared host, treat write-enabling a store as granting belief-write
to every local user, and keep `mcp.write.enabled_stores` empty (the default) on
stores that must stay read-only.

### Belief mutation uses a single configured identity, not per-caller principals

Agent writes are attributed to one config-bound identity
(`mcp.write.asserter_identity`, default `mcp:claude-code`), and the
"own-beliefs-only" mutation guard keys on that single identity. There
is therefore **no isolation between multiple callers that share one engine** —
they all write and revise as the same principal. Particles is a
**single-operator** tool; a multi-tenant deployment (distinct principals,
per-caller authorization, write isolation) is **out of scope** and not provided.
If you need that, put it in front of the engine — a per-tenant proxy with its
own authorization, or one engine per tenant.

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub
issues, discussions, or pull requests.**

Instead, use either of these private channels:

- **GitHub Security Advisory** — open a private report via the
  [Security Advisories](https://github.com/LinkedParticles/particles-engine-py/security/advisories/new)
  page for this repository (preferred).
- **Email** — contact the maintainers privately if you cannot use GitHub.

Please include enough detail to reproduce the issue: affected version, a
description of the vulnerability, reproduction steps or a proof of concept,
and the impact you foresee.

## Response expectations

- **Acknowledgement** within 5 business days of your report.
- A follow-up with an assessment and remediation plan, or a request for more
  information, within 10 business days.
- We will keep you informed of progress and coordinate disclosure timing
  with you once a fix is available.

## Scope

In scope:

- The Particles **reference SDK** (`particles/`) — the CLI, API, extraction
  pipeline, storage, and operations.
- The Particles **standard's schema and conformance artifacts**
  (`artifacts/schemas/`).

Out of scope:

- **Third-party extractor and exporter plugins** distributed separately from
  this repository. Report those to their respective maintainers.
- Vulnerabilities in dependencies, unless the SDK uses the dependency in a
  way that is itself insecure.
- Issues that require a misconfigured deployment, a compromised host, or a
  malicious `ANTHROPIC_API_KEY` / `DATABASE_URL` already under the
  attacker's control.
- **Multi-tenant / multi-principal deployments** — Particles is a
  single-operator tool (see *Trust model and known limitations*); per-caller
  authorization and write isolation are not provided.

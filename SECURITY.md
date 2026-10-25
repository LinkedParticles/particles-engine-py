# Security Policy

> For the narrative version (the audit record, the controls that held under
> attack, and why this posture is credible) see the
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
*mutating* route: deposits, extraction, retraction/supersession, subject
splits/merges, trust changes, link edits, and the agent-write verbs. The
**read** routes are, by default, **unauthenticated even when a real key is
set**: `/particles`, `/corpus`, `/subjects*`, `/quality`, `/lint/report`,
`/taxonomies`, the narrative reads, and the read-only `POST /corpus/links/suggest`.
They include the raw deposited bytes (`GET /corpus/blob/{selector}`) and the
source passage behind a belief (`GET /particles/{id}/source`). A client that
can reach the port can read the full belief store and corpus without presenting
any credential.

**Three high-value reads always require the bearer once a real key is set**,
regardless of the knob below, because the data-confidentiality and cost stakes
are higher. (Under the `dev-key` skip they are served to loopback clients like
every other route; see below.)

- **`POST /query`** bills the operator's model-provider key. Each client
  address (an IPv6 client by its `/64`) is held to `api.rate_limit_per_minute`,
  so every loopback caller shares one allowance, but every answered request
  still drives a paid completion, so an open `/query` would be a **spend**
  vector as well as a data-confidentiality one.
- **`GET /events`** (and `/events/{id}`) is the **operator audit log**:
  retractions, subject splits/merges, trust changes, reviews.
- **`GET /digest/{store}`** is the provenance-ranked roll-up of the full belief
  store, including which beliefs are contested.

**Loopback is the assumed boundary, and it is not a boundary against other
local users.** The default bind is `127.0.0.1`. With bearer auth disabled
(`PARTICLES_API_KEY` unset or `dev-key`), the startup gate refuses
to boot on a non-loopback bind, and the per-request bounds below refuse
everything else. With a real key set and reads left open, the read surface is
served to anything that can reach the port: on a loopback bind that includes
another user's process on the same host and a web page that uses DNS
rebinding (the browser treats such a page as same-origin, so the cross-site
refusal below does not stop its reads) unless `api.allowed_hosts` names the
hosts the engine answers on. Beyond loopback (for example
`engine serve 0.0.0.0:8000`, a documented LAN or Tailscale mode) it
includes everything on that network. In either posture, close the reads in-app
(below), or treat the network itself (a private subnet, a Tailscale tailnet, a
reverse proxy that enforces its own auth) as the read-access control.

**Closing the read surface.** Set `api.require_auth_for_reads: true` (config
field, env `PARTICLES_API_REQUIRE_AUTH_FOR_READS`; default `false`) to require
the **same** bearer on every read route. Under the `dev-key` skip it changes
nothing, since that mode serves only loopback clients; with a real key, every
reader must then present the bearer. This is recommended for any deployment that sets
a real key, unless the network is itself a sufficient trust boundary.
The container image ships with it set, in its baked config and in its
environment, because the image binds `0.0.0.0`; a config you mount over the
baked one does not turn it off.

**The `dev-key` skip is bounded to loopback on every path.** With bearer auth
disabled, the engine serves a request, read or write, only from a loopback
client and only when it names a loopback `Host` or an `api.allowed_hosts`
entry, whether or not it carries browser headers, `/health` included. A
DNS-rebinding page names its own hostname and is refused. The startup gate
reads the bind from the `uvicorn` command line when the engine is launched by
`uvicorn` rather than by `particles engine serve`. It refuses a bind it cannot
read: an empty `--host`, an inherited `--fd`, and a launch by gunicorn,
hypercorn, granian, daphne or `fastapi run`. For any other launcher, the
per-request bounds above are what keep a non-loopback client out.

**`api.allowed_hosts` bounds a keyed engine too.** With a real key it is empty
by default and no `Host` is refused. Listing the names a deployment answers on
refuses every other `Host` on every path, which closes the rebinding read
described above; the loopback names stay allowed for a local health probe.

**Cross-site browser requests are refused.** A web page on another site can
send a form or multipart POST to a loopback engine without a preflight. Every
gated route therefore refuses a request the browser marks as cross-site
(`Sec-Fetch-Site: cross-site` or `same-site`, or an `Origin` that is not the
engine's own) unless it carries a valid bearer, which such a page cannot attach.
The web UI at `/app` is same-origin and is unaffected, and the CLI and MCP
clients send neither header. `api.allowed_origins` lists any other browser
origin you trust.

## Trust model and known limitations

Beyond the network/auth posture above, the following design choices and open
limits are deliberate or known, and worth understanding before you deploy
Particles or feed it sensitive material.

### Settings come only from a file you name or own, never from the working directory

Particles reads its config from `PARTICLES_CONFIG`, else
`~/.particles/config.yaml`, else compiled defaults. It never reads a
`config.yaml` from the directory a command runs in or its parents, so a cloned
repository cannot point the engine, model-provider or telemetry endpoints your
secrets are sent to at its own host, cannot turn on the MCP write tools, and
cannot name local files for the memory projection to copy. A Particles
`config.yaml` found there is reported and ignored. A relative
`PARTICLES_CONFIG` is still honoured, read against the working directory, and
each command warns that it should be absolute. `particles init claude-code`
pins an absolute `PARTICLES_CONFIG` into every hook command it writes and into
an MCP server registration that has none; a registration already pinned to
another file is reported and left alone unless you pass `--config`.

The environment a process starts with outranks the file: `DATABASE_URL`,
`PARTICLES_ENGINE_BASE_URL`, `PARTICLES_OBSERVABILITY_ENDPOINT`,
`ANTHROPIC_BASE_URL`, the `OTEL_EXPORTER_OTLP_*` variables and the other
registered overrides all apply over a pinned config. `particles config
validate` and `particles hook doctor` list the overrides a process has and flag
each one that sends a key, the prompts or telemetry somewhere other than the
file says.

The client refuses to send the engine token over `http://` to a non-loopback
host unless `engine.allow_plaintext_token` is set. The same rule covers an
OpenAI-compatible provider entry's key (`allow_plaintext_key` on the entry) and
OTLP exporter headers (`observability.allow_plaintext_headers`). The Anthropic
client takes its endpoint from `ANTHROPIC_BASE_URL` when that is set, and
applies no such rule.

Releases before `1.211.0` searched upward from the working directory for a
`config.yaml`, to the nearest repository root, and applied what they found. A
hook or MCP server installed by an older release without a `PARTICLES_CONFIG`
pin gains one when you re-run `particles init claude-code`. `particles hook
doctor` warns about a missing or relative pin in `~/.claude/settings.json`, in
the current repository's `.claude` settings, and in user-scope MCP
registrations; it does not read project-scope or local-scope MCP entries.

The MCP server never serves a store from the directory it starts in either.
Started with `DATABASE_URL`, it uses that store. Started with
`PARTICLES_CONFIG` and no absolute store path, it refuses to start and names
the fix. Started with neither, it serves `~/.particles/particles.db` and says
so on stderr. The `particles` command line and `particles engine serve` still
use `./particles.db` when nothing names a store, because you choose the
directory you run them in. A cloned repository that ships a `particles.db`
therefore supplies the beliefs those commands read when they run inside it,
including when an agent runs them there. `particles config validate` names the
store file in use.

One dependency here is on the harness rather than on Particles. A repository
can still set `PARTICLES_CONFIG` for a process by shipping the command that
starts it, as a committed Claude Code project hook or MCP entry. Such an entry
already runs a command of the repository's choosing, and Claude Code asks
before it trusts a project's entries, so the pin adds nothing to what the
entry could do. That holds only while the harness keeps asking; it is Claude
Code's behaviour, and no test in this repository can check it.

### Ingested content is untrusted, and is stored verbatim, including any secrets

Deposited source material (URLs, files, pasted text, social-media threads) is
**fully untrusted** by design: it flows into the LLM extraction prompts, the
embedding model, the belief store, and exporter/projection output. Particles
tracks provenance and confidence for that content but does **not** sanitise or
scan it. In particular, **any secret in a deposited document (an API key, a
password, a token) is persisted content-addressed in the corpus and can
resurface** in particle content, query answers, the session digest, and
exporter files. Do not deposit material you are unwilling to store unredacted,
and treat the corpus blob store and the belief store as **as sensitive as the
most sensitive document you have deposited**.

The folder importers do not follow a symlinked file. `particles import vault`,
`particles import web-clipper` (including the daemon's captures watcher) and
`particles import project` skip one and log it. Rule-file intake (`particles
init claude-code`, `particles rules sync`) takes in a symlinked rule file only
when it resolves to a rule file inside the same repository or `~/.claude`. A
file you name yourself, as in `particles deposit FILE`, is still read through
a symlink.

(Prompt injection from deposited content is *contained but not eliminated*:
outside the benchmark harness, every model prompt that carries deposited,
stored or harvested text fences it with a per-call nonce, and no tool use or
function calling is exposed to the model. A test finds each call to the
model's entry points, following wrappers that pass a system turn on, and fails
the build on one nobody has classified; a call made through a function passed
as a value is outside what it can find. An injected document can still
influence the *claims extracted from it*, and model verdicts on those claims
change what is stored; see the next section. The belief store is a record of
what sources said, weighted by trust, not an oracle.)

### Model verdicts can retire a claim without review

Four automatic paths act on a model's reading of claims, each on by default:

- **Update.** When a newer source states a claim that may update an older one,
  the update probe decides whether it does, and on a yes the older claim is
  retired as superseded (`reconciliation.update_supersession.enabled`).
- **Re-anchor.** When an update retires a claim, the nightly consolidation asks
  the model whether other claims from the same passage relied on it, and
  replaces each that did with a dated restatement the model wrote and a second
  reading checked (`consolidation.reanchor.enabled`).
- **Release.** The nightly consolidation returns a held claim to active when the
  model no longer confirms the conflict that held it
  (`consolidation.release.enabled`).
- **Trust order.** When the contradiction probe confirms that two claims
  conflict and their sources' trust differs by `trust.differential_threshold`
  (0.15) or more, the claim from the less trusted source is retired, in a
  store with one trust order (`reconciliation.store_mode: single`, the
  default).

Each prompt fences the claims it reads, so a claim cannot answer the probe for
the model, and each change is recorded in the operator event log. The update
and re-anchor passes retire only extracted claims, and an agent's own earlier
claim when the same agent later states a new value; neither retires a claim the
operator asserted. A retired claim stays in the store with its history, but no
command returns it to active.

Part of the 2026-10-23 review's finding on this pass is still open. A rule
file (`AGENTS.md`, `CLAUDE.md`) retires only its own earlier claims, but memory
files under `~/.claude/projects` can count as one source with each other, and
so can Claude Code session transcripts, so a newer statement in one of them
can retire an older claim from another. The
confirming second reading asks the model the same question again rather than
re-reading the source passages, and automatic retirements are not summarised
at session start. Read `particles events list` after depositing material you do not
trust, or set `reconciliation.update_supersession.enabled: false` to turn the pass
off, so that a newer claim no longer retires an older one by date.

### Project scope is a relevance default, not isolation

With `claude_code.observer_scope: project` (the default), a session's digest,
its `MEMORY.md` region and its searches show global beliefs plus those observed
in its own project. That narrows what an agent is shown; it does not restrict
what it can read. A project-bound MCP server can still read any belief by id,
list subjects, events and corpus entries across the store, run lint, and read
the whole store when a tool is called with `all_projects`. The filter engages
only after `particles memory rescope` has run on the store. On a rescoped
store, a project-bound MCP server refuses to retract, supersede, tag, untag or
link a belief that only another project sees; on a store never rescoped it
does not. The memory-server facade (`particles memory serve`) has no project
binding at all: its reads cover every Subject and belief in the store, and it
may change only the Subjects it created itself. A project is keyed by its
path with every punctuation character turned into a dash, as Claude Code
names its memory directories, so two repositories whose paths differ only in
punctuation (`/src/a-b` and `/src/a/b`) share one project scope;
`particles hook doctor` names the repositories when that happens. Keep
material that must not reach an agent in a separate store.

### The Claude Code harness writes your sessions into the store and the store into your sessions

The SessionEnd hook deposits a distilled transcript of each session and the
project's memory files, and rule-file intake deposits `AGENTS.md` and
`CLAUDE.md` files. Each goes through a best-effort redaction pass first, which
masks `sk-` keys, AWS access key IDs, GitHub, Slack and Google tokens, JSON Web
Tokens, `Bearer` headers, PEM blocks, and the value of a secret-named
assignment such as `password=` or `api_key:`. A secret in any other shape is
deposited as written. A memory or rule file deposited with nothing to mask is
re-read by the nightly refresh, which stores the file's bytes without this
pass, so a secret added to such a file afterwards can be stored unmasked. In
the other direction, the `MEMORY.md` projection writes
ranked beliefs into `~/.claude/projects/<project>/memory/MEMORY.md`, which
Claude Code loads into every later session in that project. A poisoned belief
that ranks high enough therefore reaches an agent that has tools. Set
`agent_memory.projection.enabled: false` to stop the projection.

### The MCP write surface is a local-process trust boundary, not a network one

The stdio MCP server (`particles mcp serve`) has **no token**. Its eight write
tools (assert, supersede, retract, deposit text, link, unlink, tag, untag) are
offered only for stores in the default-deny `mcp.write.enabled_stores`
allowlist, and each writes only the store it names (writes are off unless a
store is explicitly enabled). Through a remote engine, the engine
checks its own allowlist again for the seven belief writes; `deposit_text`
reaches the engine's text-deposit route, which asks only for the bearer. The
trust boundary is therefore the **host**:
any local user who can start the MCP server with a config that write-enables a
store, and can open that store's database, can mutate it. This is appropriate
for the **single-operator, single-host** model Particles targets: the MCP
server is a locally-spawned child process, not a network service. On a shared
host, treat write-enabling a store as granting belief-write to every local
user, and keep `mcp.write.enabled_stores` empty (the default) on stores that
must stay read-only.

A looping agent can still spend. Each MCP server process allows at most
`mcp.rate_limit_per_minute` (60 by default) of the tool calls that can drive a
paid model call: a `query` with a question, `particle_assert`,
`particle_supersede`, and every write of the memory-server facade. A call over
the cap returns a tool error. Other tools are not counted, and `0` turns the
cap off.

### Belief mutation uses a single configured identity, not per-caller principals

Agent writes through the MCP server are attributed to one config-bound
identity (`mcp.write.asserter_identity`, default `mcp:claude-code`), and the
memory-server facade's to another (`mcp.memory_compat.asserter_identity`). The
"own-beliefs-only" guard on supersede, retract, link, unlink, tag and untag
keys on that identity; none of them may change an
operator-asserted belief. An agent's belief always cites a source the agent
deposited itself, so it carries the agent's trust
(`mcp.write.agent_trust_rank`) rather than borrowing an operator source's. There is therefore **no isolation between multiple
callers that share one engine**; they all write and revise as the same
principal. Particles is a **single-operator** tool; a multi-tenant deployment
(distinct principals, per-caller authorization, write isolation) is **out of
scope** and not provided. If you need that, put it in front of the engine: a
per-tenant proxy with its own authorization, or one engine per tenant.

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub
issues, discussions, or pull requests.**

Instead, use either of these private channels:

- **GitHub Security Advisory**: open a private report via the
  [Security Advisories](https://github.com/LinkedParticles/particles-engine-py/security/advisories/new)
  page for this repository (preferred).
- **Email**: [jeff@gage.org](mailto:jeff@gage.org), if you cannot use GitHub.

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

- The Particles **reference SDK** (`particles/`): the CLI, API, extraction
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
- **Multi-tenant / multi-principal deployments**: Particles is a
  single-operator tool (see *Trust model and known limitations*); per-caller
  authorization and write isolation are not provided.

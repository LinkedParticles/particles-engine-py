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
# Loopback-only, with no key: serves reads only:
particles engine serve localhost:8000

# Exposed on the LAN / a private mesh; a real key is mandatory:
export PARTICLES_API_KEY=<a-long-random-secret>
particles engine serve 0.0.0.0:8000
```

`engine serve` derives `api.bind_host` from the bind and runs the
fail-closed gate
**before** the socket opens: a non-loopback bind without a key (a real
`PARTICLES_API_KEY`, or a live per-client key minted as described below) is
**refused up front** rather than serving its reads to the network
unauthenticated. An empty or whitespace-only `PARTICLES_API_KEY` counts as
unset, so with no per-client key it is refused the same way. (Raw `uvicorn particles.api.app:app` is gated the same way:
the check reads the bind from uvicorn's own `--host`. Without a key, an empty
`--host`, an inherited `--fd`, or a launch by another ASGI server such as
gunicorn or `fastapi run` is refused, because its bind cannot be read, and a
launch with `--lifespan off` serves nothing until the same check passes on its
first request. A launch on a UNIX socket (`--uds`) is refused
too, since its connections carry no peer address. A script that calls
`uvicorn.run`, or `python -c`, is not refused: the check falls back to
`api.bind_host`, and the per-request loopback bound is what keeps other
clients out, and the first request such an engine receives on a
non-loopback address is logged. That bound answers `503` to a request with no
peer address, which is what a UNIX socket delivers, rather than reading it as
loopback. Use `engine serve`.)

## Pointing a client at the engine

On the client machine, set the engine URL (config or env) and the bearer token
(a secret: env only, never in a config file):

```yaml
# ~/.particles/config.yaml, or the file PARTICLES_CONFIG names
engine:
  base_url: "http://localhost:8000"   # an SSH forward to the engine; see below
  timeout_seconds: 60
```

```bash
export PARTICLES_ENGINE_TOKEN=<a-key-the-engine-accepts>   # see Per-client keys below
particles query "what do we know about X?"   # runs on the engine
```

With `engine.base_url` set, the daily verbs (`deposit`, `extract`, `query`,
`lint`, `review`, `quality`, `reindex`) target the engine. Unset (the default),
everything runs locally as before. The two modes share the same Pydantic models
and the same `operations/` code, reached either in-process or over HTTP.

The client refuses to send the token over `http://` to a host that is not
loopback, before any request is made: use `https://`, or reach the
engine through a loopback hop such as the SSH forward below. On a network you
trust end to end, `engine.allow_plaintext_token: true` sends it anyway with a
warning. A config file in the directory you run a command from is never read,
so a cloned repository cannot point `engine.base_url` somewhere else and
collect the token.

The client token (`PARTICLES_ENGINE_TOKEN`) is **distinct** from the engine's
`PARTICLES_API_KEY`: the engine reads the key it expects, the client reads the
token it presents. Give each client a key of its own (below); a client that
holds `PARTICLES_API_KEY` itself holds the operator's authority.

## Network exposure

For the MVP, expose the engine over a **private path**, not a public TLS
endpoint:

- **Tailscale** (or another private mesh): the bind stays effectively private.
  Serve the engine over HTTPS with `tailscale serve` and point `engine.base_url`
  at its `https://` tailnet name. A plain `http://` tailnet name is not
  loopback, so the client refuses to send the token to it unless you set
  `engine.allow_plaintext_token: true`; WireGuard encrypts that hop, but the
  client cannot see that it does.
- **SSH local-forward**: `ssh -L 8000:localhost:8000 mac-mini` and set
  `engine.base_url=http://localhost:8000`. The engine binds loopback on its host;
  nothing is exposed publicly.

Both keep the engine off the public internet with no reverse proxy in front
of it, so the per-request loopback gate sees the real peer. If you do put the
engine behind a proxy, list it in `api.trusted_proxies` so the gate reads
`X-Forwarded-For` from that hop only. An engine bound to loopback
answers only the loopback `Host` names and the `api.allowed_hosts` entries,
with a key or without one, so a proxy that forwards its
public `Host` needs that name listed, as does one in front of a keyed engine
on a UNIX socket ([SECURITY.md](https://github.com/LinkedParticles/particles-engine-py/blob/main/SECURITY.md#deployment-posture-authentication-and-the-read-surface) states the `Host` rule). A public TLS endpoint is out of scope.

On a non-loopback bind the read routes are open to anything that reaches the
port, raw corpus bytes included, unless `api.require_auth_for_reads` is `true`;
set it whenever the network is not itself the read-access control. The
container image sets it by default. Separately, every gated route refuses a
request a web page on another site sends (the browser marks it with
`Sec-Fetch-Site` or `Origin`) unless it carries a valid key. The CLI and the
web UI at `/app` are unaffected; `api.allowed_origins` admits another browser
origin you trust.

The SSH forward does not carry writes without a key. An engine with no
`PARTICLES_API_KEY` and no per-client key serves reads only: every
route that changes the store answers `503`, from loopback too, whatever token
the client sends. Through the tunnel the client can read, query and lint with
no token; to deposit, extract or write beliefs, give the engine a key (below).

## Writes need a key, even on loopback

An engine with no key at all (no `PARTICLES_API_KEY`, or the `dev-key`
sentinel, and no per-client key) serves reads only. Loopback is no evidence of who is calling: another local user, a
sandboxed process and an agent with a shell all reach `127.0.0.1`, and a write
names its own source, author and depositor. Every route that changes the store
therefore answers `503` with a message naming the fix, and no token the client
sends can change that, because the engine has no key to check it against.

| Class | Routes | On an engine with no key |
|---|---|---|
| Open | `GET /health`, `GET /openapi.json`, the web app at `/app` and `GET /`, which redirects to it | served to a loopback client |
| Reads | every `GET` route, plus `POST /query`, `POST /corpus/links/suggest`, `POST /particles/contested-badges`, `POST /memory/projection/{store}` | served to a loopback client |
| Writes | the deposits, `/extract`, `POST /subjects`, the agent belief writes (`/particles/assert` and its siblings), the operator writes (review, supersede, retract, curation, trust, subject merge and split, links and tags), `/db/init` | `503` |
| Write on one branch | `POST /lint` with `fix: true`, `POST /links/suggest` in `APPLY` mode, and `dry_run: false` on `/reconcile`, `/reindex`, `/subjects/merge`, `/corpus/{id}/retract`, `/subjects/relink-gated` | the report or dry run is served; the write answers `503` |

A report served to a request that carries no operator key, which on an
engine with no key is every request, leaves the store unchanged: a semantic
`POST /lint` report or `GET /curation?semantic=true` reads the model verdicts
already recorded but records no new ones, and takes no writer lock. Unless
`api.rate_limit_per_minute` is `0`, its model calls are also metered one call
at a time: the request's
own token pays for the first call and each further call spends another, so
one request cannot run a model pass over the whole store. A pass whose
allowance runs out comes back partial, and the
`X-Particles-Model-Calls-Refused` response header says by how many calls.
A request whose key has the `agent` or `reader` role is
treated the same way; an operator key, which carries the operator's
authority, records verdicts and is not metered by the call.

With a key, every write needs a key as the bearer, and what a key may do
depends on its role.

## Per-client keys and roles

Each HTTP client should hold a key of its own. A key names a **principal** (the
label the engine records as the depositor, author or asserter of what the
client writes) and one of three **roles**:

| Role | May call | Records its writes as |
|---|---|---|
| `reader` | every read route, and the report or dry run of a write-on-one-branch route | nothing; it writes nothing |
| `agent` | what a reader may, plus the agent writes: the deposits, `POST /particles/assert` and its siblings, `POST /memory/useful`, `POST /session-exposures` | its own principal, always |
| `operator` | every route: review, operator supersede and retract, curation, trust, subjects, links and tags, `/extract`, the write branches, `/db/init` | the operator, or any depositor or author it names |

`PARTICLES_API_KEY` keeps working as an operator key, so an engine that never
mints a per-client key behaves as before. Mint the others on the engine's host
(no HTTP route creates, lists or revokes a key):

```bash
particles engine keys create --role agent --principal mcp:claude-code --label "laptop MCP server"
particles engine keys create --role agent --principal claude-code-hook --reserved-principal --label "laptop hooks" \
  --memory-root /Users/you/.claude/projects --archive-root /Users/you/.particles/claude-code/projects
particles engine keys create --role reader --principal viewer
particles engine keys list
particles engine keys revoke 3f9a0c1b2d4e
```

`create` writes the key once, to a new file only you can read (`--output PATH`,
by default `<key id>.key` beside the key file), and prints it only with
`--print-token`, because whatever shows or keeps the command's output would
keep the key. The engine keeps only a SHA-256 digest of it, in
`api.keys_file` (`~/.particles/engine-keys.json` by default), which must be an
absolute path: a relative one is refused as an unusable file,
since it would be read against the directory the engine runs in. `create` and
`revoke` hold a lock on the file, so a revocation is never lost to a
concurrent mint.

**The hooks' key records the client's directories.** The SessionEnd harvest
deposits each memory file and each project's archive under its path on the
client, and the engine accepts one only under a directory the
hook key recorded when it was minted: `--memory-root` names the client's
Claude Code `projects` directory, `--archive-root` its
`<claude_code.state_dir>/projects`, each resolved and each repeatable. Run
`particles init claude-code` on the client with `engine.base_url` set and it
prints the exact `keys create` command with those directories filled in; run
that on the engine's host. It prints the command on every run (on stderr, and
in `--json` output among the next steps). The directories are resolved, so
when `CLAUDE_CONFIG_DIR` names a directory whose `projects` entry is a link,
the printed root is where that link leads, which may be broader than a
projects directory; check the command before you run it. With neither flag,
`create` records the minting host's own directories, which is right only when
the hooks run there. A hook
key minted before `1.223.0` records none, so its memory files are refused until
you mint it again; `engine serve` names each such key at startup, and
`particles hook doctor` on the client says whether the key records the
directories that client uses. A memory or archive file the engine refuses is
recorded in the hook log and the harvest goes on with the rest; `hook doctor`
reports how many files the newest session end had refused, and a session end
with a refused file skips the `MEMORY.md` projection.
For the first key, when the reads are open and `api.allowed_hosts` is empty,
it says that the read routes stay open without a key. A revoked key
answers `401` from the next request on, with no restart. A key file the
engine judges unusable (for example one with a permission bit for its group or
other users, or a malformed one) stops `engine serve`
at startup, and while the engine runs it makes every per-client key answer `503`,
with the reason in the engine's log. An engine that has read a key stays keyed
until it restarts, even after its last key is revoked. [SECURITY.md](https://github.com/LinkedParticles/particles-engine-py/blob/main/SECURITY.md#deployment-posture-authentication-and-the-read-surface) says
what counts as unusable; the check is best-effort, and its known exceptions
are tracked by register row. A group-writable key
directory is unusable unless it is setgid to the group `api.keys_dir_group`
names (env `PARTICLES_API_KEYS_DIR_GROUP`), the engine's user holds that
group and no other account does. The Helm chart sets it from its `fsGroup`,
but on that volume the kubelet's group pass makes the key file unusable at the
first pod restart, so per-client keys stop working there until it
is fixed, while `PARTICLES_API_KEY` keeps working.

**Reserved principals.** A label that some part of the engine trusts by name
is reserved: the operator's, the harvesters' (`claude-code-hook`, `audit`),
the rule-file intakes' (`rule-sync`, `init`), the memory-server facade's, and
every label a store writer records (each extractor id, the
extraction pipeline's, interchange import's and restore's, consolidation's,
the trust cascade's, the update sweep's, `web-clipper` and
the daemon inbox's `inbox`). `create` mints a non-operator key under one only
with `--reserved-principal`, which says you mean it. `engine serve` logs at
startup each key under a reserved name minted without it; mint that key again
with `--reserved-principal`, since it no longer authenticates.
`mcp:claude-code` is not reserved: it is the stdio MCP server's own
identity, and the HTTP MCP server's key takes it so that it keeps owning what
the stdio server asserted.

**What an agent key cannot do.** A deposit made with an agent key is recorded as
the key's principal, names no URI, source type or harvester tag of its own
(a harvester key only the harvest's shapes), and never changes or retypes an
entry another principal deposited. Its belief writes change only beliefs its
principal asserted as an agent, and an agent key under a store writer's label
changes none. [SECURITY.md](https://github.com/LinkedParticles/particles-engine-py/blob/main/SECURITY.md#deployment-posture-authentication-and-the-read-surface) states each rule, the harvester
principals' shared entries and lineage, the local Subject an
assertion mints for an unknown name.
A key whose role cannot call a route gets `403` with a message naming the role
it needs; the web app and the Obsidian plugin show that message.

**Binding an agent key to a project.** `--project KEY` binds an agent key's
writes, and the session exposures it records, to one project's
scope. An unbound agent key names the project of each write, and its deposit that names no project and carries no harvest tag is
marked unattributed, never global. Two repositories whose paths differ only
in characters other than ASCII letters and digits share one project key, so a
key bound to one writes as the other too. Reads are not confined: every key, a reader's included, reads
every project's beliefs, so a reader key is read-only, not confidential. Keys
are not scoped to a store either, an accepted limit that
[SECURITY.md](https://github.com/LinkedParticles/particles-engine-py/blob/main/SECURITY.md#deployment-posture-authentication-and-the-read-surface) states: run one engine per store when
clients must not all reach every store.

**Which key each client holds:**

| Client | Key | Where it goes |
|---|---|---|
| `particles` CLI with `engine.base_url` set, run by the operator | `operator` | `PARTICLES_ENGINE_TOKEN` |
| MCP server with `engine.base_url` set | `agent`, principal `mcp:claude-code` (the label its beliefs already carry) | `PARTICLES_ENGINE_TOKEN` in its environment; to keep the value out of Claude Code's settings, register a `chmod 700` wrapper that reads it from a `chmod 600` file |
| Hooks and the remote harvest | `agent`, principal `claude-code-hook`, minted with `--reserved-principal` | the same, through the hooks' own wrapper |
| Web app at `/app` | `operator` for curation, `reader` for a browser that only views | the bearer token field in its settings |
| Obsidian plugin | `operator` for its review, link and retract actions, `reader` for callouts only | the bearer token field in the plugin settings |

Give the agent-side clients (the MCP server and the hooks) their own keys
first, and take `PARTICLES_API_KEY` out of their environment: an agent with a
shell can read any key in that environment, so the operator's key should not
be there. `particles hook doctor` warns when the hooks hold an operator key or
the shared `PARTICLES_API_KEY` (the tokens pinned in the installed hook
commands first, your own settings' before the current repository's, else the
one in its own shell; a repository you cloned is yours, so its hook entries
count as installed, as [SECURITY.md](https://github.com/LinkedParticles/particles-engine-py/blob/main/SECURITY.md#settings-come-only-from-a-file-you-name-or-own-never-from-the-working-directory)
says), the MCP server logs the same warning when it starts, and `particles config validate` names the role of the client token it
finds. Each reads the key's prefix locally. To ask the engine instead,
`GET /auth/whoami` returns the presented key's principal, role, project and key
id. Every event a request writes records its principal and key id;
`particles events list --key KEY_ID` lists what one key did; a key that is
not an operator's sees only its own key id there, and every event's
principal.

The CLI, the hooks and the MCP server without `engine.base_url` write the local
store in-process and need no key. `particles config validate` prints whether an
engine started from your environment accepts writes, and `particles hook doctor`
prints whether the hooks write in-process or need a token for an engine.

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

The memory-server facade (`particles memory serve`) does not follow the
engine: it has no route to one. It refuses every tool call,
and its `memory://knowledge-graph` read, while `engine.base_url` is set, so
its memory never lands in a local store you do not use. Run it on the
engine's host against the engine's own store, or without `engine.base_url`.

The MCP server stays **local stdio** (unchanged install path); the
local-stdio write-trust model carries over, and the server presents
its own `PARTICLES_ENGINE_TOKEN`: an `agent` key under `mcp:claude-code`, not
the operator key the CLI holds (see the client table above). The belief-write
tools are still offered only when `mcp.write.enabled_stores` is set locally;
the **engine** independently gates the write server-side on its own
write-enablement. The belief-write endpoints (`POST /particles/{assert,
supersede,retract}`) admit any agent or operator key, not only the MCP
surface's, and answer `403` to a reader key. A *network-exposed* MCP transport (for a non-co-located client
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
well. Each hook write names the store its hook was installed for (`--store`),
and the engine writes that store; a handle the engine does not configure is
refused with a 404, recorded in the hook log. The hook steps behave as follows:

| Hook step | With an engine configured |
|---|---|
| SessionStart digest | Served by the engine (`GET /digest/{store}`). The freshness check runs as it does locally: the engine renders the region from this machine's manifest, and the hook compares it with the `MEMORY.md` here. A current region means no digest, a stale one means only the new lines, and an engine that fails or answers too slowly means the full digest |
| SessionEnd harvest | Refused unless `claude_code.harvest.allow_remote: true`, because transcripts are the most sensitive material the SDK handles. With it, the distilled, redacted transcript and the memory files are deposited on the engine, and the engine's own consolidation extracts them. The memory files are deposited with `fetch_policy: NEVER`, since the engine cannot re-read a file on this machine: the next SessionEnd harvest is what picks up an edit |
| SessionEnd MEMORY.md re-render | Rendered by the engine from this machine's own manifest (`POST /memory/projection/{store}`) and spliced into `MEMORY.md` here, with the same fold, backup and atomic write as a local render. It runs only when an installed hook names this store and pins this engine. A manifest section that includes a file is refused, with the reason in the hook log |
| SessionEnd utility mining | Skipped |

The re-render reads the engine a hook pins from the hook command in Claude
Code's settings: `PARTICLES_ENGINE_BASE_URL`, or the `engine.base_url` of the
`PARTICLES_CONFIG` the command pins. A wrapper script that sets
`PARTICLES_CONFIG` inside itself hides that from the check, and the memory
directory is then left alone. Pin the config in the command as well:

```bash
env PARTICLES_CONFIG=/path/to/client.yaml /path/to/particles-client hook session-end --store default
```

`particles init claude-code` also pins `PARTICLES_ENGINE_BASE_URL` empty in
each hook command and in the MCP server registration, which overrides nothing
and leaves the pinned config's `engine.base_url` in charge, so a hook or
registration that carries the pin never follows an engine the session's
environment names. Name the engine in the pinned config to make them thin
clients. A hook whose engine comes only from `PARTICLES_ENGINE_BASE_URL` still
refuses a local store that resolves against the working directory.
`particles hook doctor` warns about a hook or registration with no pin or
with an inherited one, and a re-run of `init claude-code` gives it the empty
pin. A registration written before `1.221.7` carries no such pin, and a
re-run from `1.194.0` to `1.221.6` could remove it: if yours is gone, run
`init claude-code` again.

Every hook still fails open: an unreachable engine, a refused token or a slow
link ends the hook with exit 0 and no output, inside the
`claude_code.hook_deadline_seconds` budget (10 s by default), and the reason is
written to the local hook log (`particles hook log`). On a thin client
`particles hook doctor` asks the engine (`/auth/whoami`) which directories the
hook key records, when the token in its shell is the one the hooks pin, and
reports the refused files of the newest session end; for every other failure,
read the hook log.

Claude Code starts hooks and MCP servers with the environment their settings
entries declare. A `PARTICLES_ENGINE_TOKEN` for a non-loopback engine therefore
has to reach those entries: written into them in plaintext, or read by the
wrapper the table above describes from a `chmod 600` file. The SSH forward
described above avoids both.

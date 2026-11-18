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
| Reads | general `GET` routes, plus `POST /query`, `POST /corpus/links/suggest`, `POST /particles/contested-badges`, `POST /memory/projection/{store}` | served to a loopback client |
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

The agent writes that make no model call (`POST /particles/retract`,
`/particles/link`, `/particles/unlink`, `/particles/tag`, `/particles/untag`
and `/memory/useful`) spend from a second allowance,
`api.write_rate_limit_per_minute` (300 by default), counted the same way per
key principal, or per client address without a key. It applies to every key,
an operator key included, and shares nothing with `api.rate_limit_per_minute`,
so a burst of tags or links neither uses up nor waits on the allowance that
deposits and queries spend. A request past it answers `429`; `0` turns it off.

A deposit makes no model call of its own: the extraction it queues runs later,
in consolidation pass 1, where no request allowance applies. A deposit made
with an agent key that is not one of the owner's harvesters is therefore
counted in the extraction queue instead, on every route that deposits (the
`source_excerpt` of an assertion or a supersession included). Its principal
may hold at most
`api.max_pending_extractions_per_principal` snapshots waiting to be read (50
by default); a deposit past that is refused with `429`, stores nothing, and
names the setting and the entry holding the most of them. Each consolidation
run reads at most `consolidation.max_pending_per_metered_principal` of a
principal's snapshots (5 by default), whatever `max_pending_entries` and
`extract_budget_usd` are, and leaves the rest for the next run, counted in the
run record as `pending_deferred_metered`. Slots a principal does not use are
not handed to its next snapshot, so raising the run's cap or budget never
raises what an agent key extracts. To let one agent deposit more, raise the
share, which raises its spend with it; raising the cap alone only lengthens
its queue. `particles extract --all-pending` drains every pending snapshot,
these included. Operator and harvester keys are never counted.

With a key, every write needs a key as the bearer, and what a key may do
depends on its role.

## Per-client keys and roles

Each HTTP client should hold a key of its own. A key names a **principal** (the
label the engine records as the depositor, author or asserter of what the
client writes) and its **role**.

| Role | May call | Records its writes as |
|---|---|---|
| `reader` | every read route, and the report or dry run of a write-on-one-branch route | nothing; it writes nothing |
| `agent` | what a reader may, plus the agent writes: the deposits, `POST /particles/assert` and its siblings, `POST /memory/useful`, `POST /session-exposures` | its own principal, always |
| `operator` | review, operator supersede and retract, curation, trust, subjects, links and tags, `/extract`, the write branches, and `/db/init`, in addition to ordinary reads and deposits | the operator or an explicitly named author or depositor on ordinary intake |
| `codex-capture` | Codex conversation deposit, exact-session continuity proof, and credential identity | `codex-hook` |

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

**Codex capture keys.** Mint one key per client on the engine host with
`particles engine keys create --role codex-capture --principal codex-hook --reserved-principal`.
Optionally pass `--project KEY` to restrict capture to a single canonical project.
Check the replacement key with `GET /auth/whoami` before revoking the old key.
The response gives the role, principal, project binding, and capture protocol
version. Keep the returned credential in its private file. A Codex capture key
grants write access to Codex conversation sources and read access to their
matching continuity metadata. General authenticated reads require a separate
key. The [Codex installer](../user-guide/codex.md#connect-codex-to-a-same-host-engine)
can enroll this key for same-host capture. If you disable remote capture,
revoke the retained credential on the engine host.

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
**The hooks' key also records the client's host id** (`--host-id`). Each
`file://` source in the store records the machine whose disk its path names,
and the engine stamps every memory and archive file the hooks deposit with the
host id their key records, never one from the request. Its own refresh then
leaves those files alone, even on a host that has a checkout at the same path.
`init claude-code` puts the client's id in the command it prints, and
`hook doctor` on the client compares the key's id with the client's own. A
hook key without one names no memory or archive file, so a key minted before
the release that added the flag is refused until you mint it again; `keys
create` refuses the engine host's own id. See
[Files on the client stay the client's](#files-on-the-client-stay-the-clients).

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
changes none. A Subject it mints for a name the store does not know, by an
assertion or by the extraction of a document it deposited, is provisional:
extraction, imports and operator writes do not resolve that name to it until
an operator accepts it. [SECURITY.md](https://github.com/LinkedParticles/particles-engine-py/blob/main/SECURITY.md#deployment-posture-authentication-and-the-read-surface) states each rule, and the harvester
principals' shared entries and lineage.

**Provisional Subjects.** `particles subjects list --provisional` lists the
Subjects agent principals minted that nobody has confirmed, oldest first, each
with any confirmed Subject that holds its name. `particles subjects accept ID`
(`POST /subjects/{id}/accept`, an operator key) confirms one, and so does an
operator's alias, merge into it, or class change. Every read marks a
provisional Subject with `provisional` (its principal and when it was minted
or shared). A store upgraded from before this keeps every Subject confirmed;
`particles subjects provisional-backfill --dry-run` counts the Subjects an
agent principal minted that no one else's claim, external reference or
operator act depends on, and the run without `--dry-run` marks them.
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
| MCP server with `engine.base_url` set | `agent`, principal `mcp:claude-code` (the label its beliefs already carry) | its token file, which `init claude-code --engine` pins (see [Installing Claude Code as a thin client](#installing-claude-code-as-a-thin-client)) |
| Hooks and the remote harvest | `agent`, principal `claude-code-hook`, minted with `--reserved-principal` | the hooks' own token file, pinned the same way |
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
id, and the stores that accept its belief writes (`belief_write_stores`). Every event a request writes records its principal and key id;
`particles events list --key KEY_ID` lists what one key did; a key that is
not an operator's sees only its own key id there, and every event's
principal.

The CLI, the hooks and the MCP server without `engine.base_url` write the local
store in-process and need no key. `particles config validate` prints whether an
engine started from your environment accepts writes, and `particles hook doctor`
prints whether the hooks write in-process or need a token for an engine.

## Files on the client stay the client's

A `file://` URI names a path, not a machine, and a store can move between
machines as a file. Every `file://` corpus entry therefore records the host
whose filesystem its path names: an entry a local command deposits records this
host, and one the hooks deposit over HTTP records the client host their key
names. The local refresh (`corpus refresh`, consolidation pass 0.5) reads only
the entries of the host it runs on. An engine in a container never looks for the
laptop's paths, and an engine on a machine with a checkout at the same path
never reads that copy into the laptop's history. `corpus refresh`, the bare
`particles rules` and the consolidation report count the entries they leave
alone ("another host's files; that host pushes them").

The host id lives in `host.id_file` (`~/.particles/host-id` by default; the
container pins `/data/host-id`). It is a random id stored beside the operating
system's machine identifier, so a copy of `~/.particles` carried to another
machine is not believed there: that machine mints an id of its own.

**Moving an existing store.** Entries deposited before this record of origin
existed carry none. `particles db init` gives them the id of the host it runs
on, once per store, and prints how many it marked by class; run it on the
machine whose files the store tracks, before you move the store.
`particles db init --origin-preview` prints the same counts and changes nothing.
The engine's own startup migration never marks them, so a store first migrated
in a container leaves them unmarked, its refresh reads none of them, and the
client's next deposit of each file claims it. If a store was marked on the wrong
machine, `particles corpus reorigin --from <id> --to <id>` on the engine host
moves the marks (`--dry-run` prints the count, `--uri-prefix DIR` narrows it).

A deposit that names another host's file is refused with `409`, and the
message names both host ids. Two machines that track one path in one store are
not supported: the first to deposit the path keeps it.

**Rule files are pushed by the client.** With an engine configured, each
SessionEnd sends the `AGENTS.md` and `CLAUDE.md` files of the session's project
(a worktree's main checkout, never the worktree's copies) and the two files
directly in `~/.claude`, inside a fifth of the hook deadline. A record in the
client's state directory keeps a quiet session from sending anything.
`particles rules sync` sends the whole set at any time, and `--dry-run` shows
what it would send. The hook key must record where those files lie:
`--rule-root` once per project root (a directory holding several projects is
refused), and `--global-rule-root` for `~/.claude`. `init claude-code` prints
both for the project you run it in, and `hook doctor` names any root the key
lacks. A rule file the engine refuses is recorded in the hook log and does not
hold up the `MEMORY.md` projection. A file changed outside any session (a
`git pull`) reaches the engine at the next session in that project or the next
`rules sync`.

## What is not remoted yet

Since, the
operator verbs **route or refuse**: with an engine configured, none silently
read or write the *local* store. The endpoint-backed operator verbs (`trust set`
/ `statement-set`, `subjects alias` / `merge` / `split` / `list` / `search` /
`show`, `links add` / `remove` / `suggest`, `particle tag` / `untag` / `show`,
`corpus retract` / `show` / `links suggest` / `dismiss`, `events list` / `show`,
and the `curate` listing with `--card` and `--refresh`) now execute against the
engine. The engine runs `curate --refresh`, and a queue request on a store with
no stored collection yet, synchronously; on a large store either can take longer
than `engine.timeout_seconds` (default 60), so raise it for that call or run the
rebuild on the engine host. **`import vault`** and **`import project`** also
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
- **`curate`**: `apply` and `--precision`.
- **`reconcile`**, in every mode.
- **`inbox`**: `process` and `watch` (`status` reads only the inbox file and
  still runs).
- **`interchange`**: `export`, `import` and `restore`.
- **`synthesis-cache`**: `list`, `show`, `vacuum` and `evict`.
- **`memory`**: `rescope` (every mode), `widen`, `redate`, `useful`,
  `rebuild-utility`, `sweep-rank-lift` and `sweep-owner-lift`. On a thin
  client, run them on the engine host (`docker compose exec engine particles
  memory rescope …` against the container).

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
| SessionStart digest | Served by the engine (`GET /digest/{store}`). The freshness check runs as it does locally: the engine renders the region from this machine's manifest, and the hook compares it with the `MEMORY.md` here. A current region means no digest, a stale one means only the new lines, and an engine that fails means the full digest. An engine that answers too slowly leaves a session that loaded the last rendered region with that region (logged `degraded`), and gives any other session the full digest |
| SessionEnd harvest | Refused unless `claude_code.harvest.allow_remote: true`, because transcripts are the most sensitive material the SDK handles. With it, the distilled, redacted transcript and the memory files are deposited on the engine, and the engine's own consolidation extracts them. The memory files are deposited with `fetch_policy: NEVER`, since the engine cannot re-read a file on this machine: the next SessionEnd harvest is what picks up an edit |
| SessionEnd MEMORY.md re-render | Rendered by the engine from this machine's own manifest (`POST /memory/projection/{store}`) and spliced into `MEMORY.md` here, with the same fold, backup and atomic write as a local render. It runs only when an installed hook names this store and pins this engine. A manifest section that includes a file (a `block:`) works too: this machine reads the file, under the same path rules as a local render, and sends its text with the render, at most 64 KiB in all and only with `claude_code.harvest.allow_remote` on; the engine renders it and keeps none of it. `particles hook doctor` names what each render sends, or why it would be refused |
| SessionEnd utility mining | None, on either topology: the engine's consolidation mines the harvested transcripts |

The re-render reads the engine a hook pins from the hook command in Claude
Code's settings: `PARTICLES_ENGINE_BASE_URL`, or the `engine.base_url` of the
`PARTICLES_CONFIG` the command pins. A wrapper script that sets
`PARTICLES_CONFIG` inside itself hides that from the check, and the memory
directory is then left alone. The thin-client install below pins the config in
every command, so it passes the check.

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

**Size the engine's rate limit for the harvest.** Each SessionEnd sends one
deposit per memory file, plus the archive, the session transcript, up to
`claude_code.harvest.catchup_limit` recent transcripts and each changed rule
file, and the engine counts every deposit against the hook key's
`api.rate_limit_per_minute` (60 by default), unchanged files included. A
project with 90 memory files therefore exceeds the default in one session end:
the first deposit past the limit answers 429, the harvest ends there with
the error in the hook log, and the MEMORY.md splice waits for a session end
that sends every file. Raise
`api.rate_limit_per_minute` in the engine's config to a few times the number
of memory files of your largest project. A rule-file push that meets a 429
stops and sends the remaining files at the next session end.

Every hook still fails open: an unreachable engine, a refused token or a slow
link ends the hook with exit 0 and no output, inside the
`claude_code.hook_deadline_seconds` budget (10 s by default), and the reason is
written to the local hook log (`particles hook log`). The SessionEnd hook only
hands the session off; the background worker makes the requests, bounded by
`claude_code.harvest.worker_deadline_seconds`, so a slow link delays the
harvest without the ending session waiting for it.

**The engine warms its render path at start.** After its startup checks pass,
the engine builds each served store's candidate cache and runs one
confidence-ranked retrieval in the background (`api.warm_up_on_start`, on by
default). The first `MEMORY.md` render after a start is then served warm. A
request that arrives during the warm-up waits for it rather than building a
second copy, so a session that starts in those seconds may still find its
freshness check out of time and keep the region it loaded. On a thin client
`particles hook doctor` asks the engine (`/auth/whoami`) which directories the
hook key records, when the token in its shell is the one the hooks pin, and
reports the refused files of the newest session end; for every other failure,
read the hook log.

### Checking the hooks with `hook doctor`

With `engine.base_url` set, `particles hook doctor --store <handle>` probes the
engine the hooks write instead of a local database file. A thin client with no
local store passes when the engine is healthy, and a stale local file no longer
stands in for the engine. Its exit code comes from the `reachability:`, `digest:` and `engine token:` lines:

| Line | What it checks |
|---|---|
| `reachability:` | `GET /health` answers, and how long it took against `claude_code.hook_deadline_seconds`. A reply slower than the deadline is a warning: every hook would give up before the engine answers. An unreachable engine or a failing `/health` fails the doctor |
| `engine token:` | One line for each consumer of the engine key, the installed hooks and each MCP registration, read from their pins: where the key comes from (`file <path>`, `plaintext in <settings file>`, `plaintext in ~/.claude.json`, or `none pinned`), the token-file check a file fails, the role the key's prefix carries and, when the engine answers, the principal and role `/auth/whoami` reports against the ones the consumer needs. A token file that fails a check, or a key the engine refuses, fails the doctor, since that consumer sends nothing. A wrong role or principal is a warning. A key stored in plaintext is a warning that says to re-run `particles init claude-code`, and a hook entry that runs the lifecycle hooks through another executable is named as the retired wrapper install. No line prints a key, a hash of one, or its length. The keys go only to the engine the pinned config names: with `PARTICLES_ENGINE_BASE_URL` set in the doctor's shell, the principal is not checked |
| `version:` | The version `/health` reports against this client's. The two ship in lockstep, so a mismatch is a warning naming both |
| `digest:` | An authenticated `GET /digest/{store}`, the read the SessionStart hook makes. A `401` or `403` fails the doctor and says whether this environment holds no `PARTICLES_ENGINE_TOKEN` or holds one the engine refuses; a `404` names a store the engine does not configure |
| `writes:` | Whether the engine accepts belief writes on the store from the key the `digest:` probe sent, as `/auth/whoami` reports it; the check makes no write. A store missing from the engine's `mcp.write.enabled_stores`, a reader key, or a caller with no key is a warning, since the SessionStart exposure write, the MCP write tools and the curation writes then answer `403` while the reads pass. An engine older than this client is not judged. Shown only when the `digest:` read passes |
| `tuning:` | Whether this client's tuning matches the engine's. The engine applies most of what an operator tunes (extraction, the nightly consolidation passes, query ranking, lint, the model routes), so a setting changed in this client's file only has no effect. The doctor reads a digest of each such setting of the engine's effective config from `GET /config/tuning` and compares it with the same digest of this client's, then names each one that differs, with this client's value and whether the engine takes its own from its config file or its environment. Settings that describe a process or its host (storage, the bind address, the engine URL, logging, host paths, a local model's URL) are never compared. A difference is a warning and leaves the exit code alone; list a deliberate one in `engine.tuning_drift_ignore`. The engine's extraction-budget warnings follow, marked as the engine's: consolidation runs there, so this client's own would judge the wrong config. An engine older than the route is not compared. Shown only when the `digest:` read passes |
| `hook refusal:` | Shown only when the hooks refuse to run at all: the engine comes only from `PARTICLES_ENGINE_BASE_URL` and the store's SQLite path is relative, the rule above. It fails the doctor |
| `harvest:` | Whether the SessionEnd harvest runs remotely. With `claude_code.harvest.allow_remote` false it is a warning, since every SessionEnd skips the harvest |
| `stop hook:` | The retired Stop hook does nothing locally or remotely; a leftover entry is a warning, as locally |
| `projection:` | Whether the engine renders the `MEMORY.md` region for this client to splice. While the harvest is off the render never runs, and the line warns |

The probes use the token in the doctor's own environment. When that holds
none and the installed hooks run with one key of their own (a token file or a
plaintext pin), the probes send that key instead, unless
`PARTICLES_ENGINE_BASE_URL` in the doctor's shell names the engine. When the
hooks pin a different token from the shell's, the `digest:` line says the
probe did not judge theirs. The `schema:` line and the local store checks (DSN, database file,
corpus tables, blobs, refresh roots, calibration) say they do not apply, since
the engine migrates and holds its own store. The `install:`, `engine:`,
`memory dir:`, `observer scope:` and `egress:` lines read this client and are
printed as locally.

### Installing Claude Code as a thin client

`particles init claude-code --engine URL` installs the hooks and the MCP server
as thin clients of the engine at `URL`. A config that already names
`engine.base_url` gets the same install without the flag.

```bash
particles init claude-code --engine http://127.0.0.1:8000 --dry-run
```

```bash
particles init claude-code --engine http://127.0.0.1:8000
```

What it does:

- writes `engine.base_url` into the pinned config, and adds the store to
  `mcp.write.enabled_stores` so the local MCP server offers its write tools;
- creates no local store, no blob directory and no `storage.stores` entry, and
  pins no `DATABASE_URL` in the hooks or the MCP registration;
- gives each client its own key in its own file: the hooks read
  `~/.particles/claude-code/engine-tokens/hooks`, the MCP server reads
  `~/.particles/claude-code/engine-tokens/mcp`. The settings and the
  registration hold the file paths (`PARTICLES_ENGINE_TOKEN_FILE`), never a key;
- asks for each missing key at a hidden prompt, or pins a file you name with
  `--hooks-token-file` and `--mcp-token-file`;
- refuses an operator key and the shared `PARTICLES_API_KEY` for either file,
  and a reader key for the hooks, then asks the engine (`/auth/whoami`) whether
  each key's principal is the one its client needs;
- on macOS, keeps the token directory out of Time Machine;
- lists under `not_done` (and `Not done:` in its output) what a thin client
  cannot do: the project-key rescope, the first-run audit and the rule-source
  registration.

It never turns on `claude_code.harvest.allow_remote`; set it yourself to ship
session transcripts to the engine.

Mint the two keys where the engine runs before you install; the install prints
both commands, with this machine's harvest roots filled in:

```bash
particles engine keys create --role agent --principal claude-code-hook --reserved-principal --memory-root /Users/you/.claude/projects --archive-root /Users/you/.particles/claude-code/projects --label "claude-code hooks" --output hooks.key
```

```bash
particles engine keys create --role agent --principal mcp:claude-code --label "claude-code MCP server" --output mcp.key
```

The roots are this machine's, spelled as absolute paths, since the commands
run on the engine's host. Against the container, run them with
`docker compose exec`. Copy each key
across, then give it to the install at its prompt or as a file.

**The token file.** `PARTICLES_ENGINE_TOKEN_FILE` names a file the engine key
is read from, and any thin client may use it in place of
`PARTICLES_ENGINE_TOKEN` (setting both is refused). The file must be an
absolute path, a regular file with one link, owned by you, with no group or
other permission (`0600` or `0400`), in a directory only you can write (and
every directory above it owned by you or root, writable by no one else unless
sticky, like `/tmp`), and
hold one line of at most 4 KiB. A file that fails a check is refused by name
and no request is sent: a hook logs `token-file-refused` in the hook log, the
MCP server refuses to start, and `particles config validate` and the doctor's
`engine token:` line name the check.
A secret mounted into a container as a root-owned file fails the owner check
by design; give that process `PARTICLES_ENGINE_TOKEN` instead.

The key is read once per process. A hook reads a rewritten file at its next
run; the MCP server reads it when it next starts, so after rotating its key,
reconnect the server or start a new session.

`particles init claude-code --remove` deletes the token files the install
wrote, leaves a file you named, and prints the `particles engine keys revoke`
command for each key, since deleting a file does not revoke its key.

**Moving off a wrapper script.** An install that runs the hooks through a
wrapper script (a `chmod 700` script that sources a `chmod 600` file) is not
one the installer can recognise, so a thin install refuses to add its hooks
beside the wrapper's and lists them, and `hook doctor` names the wrapper's
executable on an `engine token:` line. Delete those entries from
`~/.claude/settings.json`, reduce the sourced file to the bare key, and re-run
the install with `--hooks-token-file` naming it.

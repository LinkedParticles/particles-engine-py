# Security Policy

This page states the guarantees Particles makes and the limits it keeps. A
guarantee is stated for its main path; where a known exception exists, the
exception is named with the register row that tracks its fix. The history of
each fix, release by release, is in the changelog and the audit reports, not
here.

## Supported versions

Particles is under active development, and only the latest published release
receives security fixes. A fix ships as a new release; it is not backported.

| Version | Supported |
|---|---|
| Latest `1.x` release | ✅ |
| Every earlier release | ❌ |

A release number does not by itself prove that a fix is present, nor that its
commit passed CI: *Releases and the supply chain* below names the releases
that are exceptions.

## Deployment posture: authentication and the read surface

The HTTP engine (`particles.api.app`) has an **asymmetric authentication
model** that operators must understand before exposing it on a network.

**An engine with no key serves reads only, and only to loopback.** An engine
is keyless when `PARTICLES_API_KEY` is unset, empty, whitespace-only or the
`dev-key` sentinel and no live per-client key (below) exists. A keyless engine
refuses every route that changes the store, from loopback too and whatever
token the request carries: with `503` and a message naming the fix, or with
`403` when the request already fails the loopback, `Host` or cross-site bounds
below. Its report routes that drive the model (`POST /lint` without `fix`,
`GET /curation` with `semantic`) read the verdicts already stored and record
none. It serves a request only from a loopback client that names a loopback
`Host` or an `api.allowed_hosts` entry, `/health` included, so a DNS-rebinding
page is refused. Its startup gate refuses a non-loopback bind, and a bind it
cannot read (an empty `--host`, an inherited `--fd`, a UNIX socket, a launch by
another ASGI server); a request with no peer address is never read as
loopback, and a launch with `--lifespan off` serves nothing until the same
checks pass on its first request. A script that calls `uvicorn.run`, or
`python -c`, gives the gate no bind to read, so the gate falls back to
`api.bind_host`; the per-request bounds keep a non-loopback client out, and
the first request on a non-loopback local address is logged. On the routes
whose request body chooses between a report and a write (`POST /lint`,
`/links/suggest`, `/reindex`, `/reconcile` and a few others), the keyless
refusal comes once the body has chosen the write, by design. The `particles`
command line, the Claude Code hooks and the stdio MCP servers write the local
store in-process and need no key.

**A key file the engine cannot use never makes it keyless.** A key file that
exists but cannot be used (another owner, a permission bit for its group or
other users, a record that does not parse, a directory on its path the
engine's user cannot search) counts as holding keys: it stops `engine serve`
at startup, and if it goes bad while the engine runs, every per-client key
answers `503` while `PARTICLES_API_KEY` keeps working. Only a missing name, or
a path through a component that is missing or is not a directory, reads as no
file. Once the engine has read a live per-client key it stays keyed until it
restarts, whatever then happens to the key file. The `503` names no path; the
reason goes to the engine's log.

**The engine refuses a key file another account can change, best-effort.**
It opens the key file without following a link, requires a regular file with
no permission bit for its group or other users, and judges the directories the
file's name resolves through. A directory its group may write holds the key
file only when it is setgid to the group `api.keys_dir_group` names (env
`PARTICLES_API_KEYS_DIR_GROUP`, unset by default), the engine's user holds that
group and the account databases show no other account holding it; an unlisted
group, and on macOS every group, counts as shared. The Helm chart sets
`api.keys_dir_group` to its pod's `fsGroup`; leave it unset for a bind mount.
This check has known exceptions. On
the Helm chart's `fsGroup` volume the kubelet gives the key file group bits at
each remount, which the engine refuses, so per-client keys stop working after
the first pod restart while `PARTICLES_API_KEY` keeps working.

**With a key, a key gates writes, not reads.** Setting `PARTICLES_API_KEY` to
a secret, or minting a per-client key (below), enables bearer-token auth
(`Authorization: Bearer <key>`) on every route that changes the store, and
each needs a key whose role allows it. The **read** routes are, by default,
**unauthenticated whether or not a key is set**: among them `/particles`,
`/corpus`, the raw deposited bytes (`GET /corpus/blob/{selector}`), the source
passage behind a belief, `/subjects*`, `/quality`, `/lint/report`,
`/taxonomies`, `/vocabulary`, the narrative reads and the read-only
`POST /corpus/links/suggest` and `POST /particles/contested-badges`. A client
that can reach the port can read the full belief store and corpus without
presenting any credential, unless it is a cross-site browser request (below).
No read route opens a file the caller names on the engine's host: `GET /graph`
reads a projection manifest only by a relative name under
`graph.manifest_root`, judged on every request, and refuses that scope while
the setting is unset (the default) or names a broad root such as the home
directory. The in-process `graph_view` MCP tool applies the same rule.

**Three high-value reads always require a key once any key exists**,
regardless of the knob below. A key of any role passes, a reader's included,
and so do the graph, curation and review reads, the synthesized narrative and
the memory projection. On a keyless engine they are served to loopback
clients like every other read.

- **`POST /query`** bills the operator's model-provider key, so an open
  `/query` would be a **spend** vector as well as a data-confidentiality one.
- **`GET /events`** (and `/events/{id}`) is the **operator audit log**:
  retractions, subject splits and merges, trust changes, reviews. A key that
  is not an operator's reads no other key's id there; every event still names
  its principal to any key that reads the log.
- **`GET /digest/{store}`** is the provenance-ranked roll-up of the full belief
  store, including which beliefs are contested.

**The rate limit counts model calls for every caller but an operator key.**
A request that carries a key spends from its principal's
`api.rate_limit_per_minute` allowance, so every key of one principal shares
one allowance and a new address buys none. On a keyless engine each client
address (an IPv6 client by its `/64`) has its own allowance, and every
loopback address shares one. `particles engine serve` takes the client address
from the socket, reading `X-Forwarded-For` only from a peer listed in
`api.trusted_proxies`, which refuses any range that overlaps loopback or is
broader than `/8` (IPv4) or `/32` (IPv6). A raw `uvicorn` launch trusts
`X-Forwarded-For` from a loopback peer by default. A request without an
operator key spends one token for each model call it makes; a route that runs
a model pass over the store stops calling the model when the allowance is
spent, returns what it has, and reports the calls it refused in the
`X-Particles-Model-Calls-Refused` header. An operator key is charged per
request, by design, and its report requests record their verdicts. With
`api.rate_limit_per_minute` set to `0` nothing is metered. Three limits
remain: the extraction an HTTP deposit queues runs later, outside any caller's
allowance, bounded per run by `consolidation.max_pending_entries`;
the agent writes that make no model call have no volume limit;
and no check refuses one extraction call whose input exceeds a
stated budget, so a chunking defect in an extractor can send an oversized
call.

Only an operator's press on `POST /memory/useful` (`PARTICLES_API_KEY` or an
operator key) earns that belief its usefulness credit, which ranks it higher,
and every operator press over HTTP shares one credit a day per belief. An
agent key's press is recorded and credits nothing, and a reader key may not
press at all (`403`). A press an agent key made in an earlier release keeps
the credit it earned (an accepted limit).

**Loopback is the assumed boundary, and it is not a boundary against other
local users.** The default bind is `127.0.0.1`. What a keyless engine serves
to loopback is its read surface, which another local user can read, and its
model-driving report routes, through which another local user can spend the
operator's model key within the per-call limit above. A per-client key lifts
the startup gate as `PARTICLES_API_KEY` does. With a key and reads left open,
the read surface is served to anything that can reach the port, another user's
process on the same host included. An engine bound to loopback answers only
the loopback `Host` names and the `api.allowed_hosts` entries, with a key or
without one, and judges the local address each request arrived on, so a
loopback listener the startup checks read as wider keeps that rule. The `Host`
rule does not stop another local user, who can name a loopback `Host`. A
reverse proxy on the engine's own host that forwards without
`X-Forwarded-For` makes every client look like loopback. Beyond loopback (for
example `engine serve 0.0.0.0:8000`, a documented LAN or Tailscale mode),
the open reads reach everything on that network, and a rebinding
page is refused only when `api.allowed_hosts` names the engine's hosts. In
either posture, close the reads in-app (below), or treat the network itself
(a private subnet, a Tailscale tailnet, a reverse proxy that enforces its own
auth) as the read-access control.

**Another local account is the actor many of the rules on this page stand
against.** On a host only you use there is no such account. The rules matter
on a host several people log in to, on a checkout or volume a group shares,
and on a data directory mounted from the host into the container. The
container image keeps its store, key file and blobs under `/data`, which the
documented setup mounts as a Docker named volume that only the engine's user
(uid `10001`) writes. A host directory bind-mounted at `/data` brings the
host's other accounts back: give it a directory only that uid can write, or
any account that can write it can change the store, the key file and the
blobs under the engine.

**Closing the read surface.** Set `api.require_auth_for_reads: true` (config
field, env `PARTICLES_API_REQUIRE_AUTH_FOR_READS`; default `false`) to require
a key on every read route. A key of any role passes, so a `reader` key is how
a client reads without being able to write. On a keyless engine it changes
nothing, since that engine serves only loopback clients. `GET /health` still
answers without a key, so a readiness probe keeps working, and shows its
`daemon` block only to a request with a key the engine accepts. This is
recommended for any deployment that sets a key, unless the network is itself
a sufficient trust boundary. The container image ships with it set, in its
baked config and in its environment, because the image binds `0.0.0.0`; a
config you mount over the baked one does not turn it off, and only an explicit
environment override (`-e PARTICLES_API_REQUIRE_AUTH_FOR_READS=false`, or the
same variable in the Helm chart's `env` values) opens the reads. The engine
logs a warning at startup when the bind it can read is beyond loopback and the
reads are open.

**Each client can hold a key of its own, and a key's role bounds what it may
do.** An engine accepts per-client keys besides `PARTICLES_API_KEY`.
Each names a principal and one of three roles: a `reader` key
reads; an `agent` key also deposits and writes beliefs; an `operator` key may
take every action, including the routes that act with the operator's
authority (subject merges and splits, operator retractions and supersessions,
review rulings, trust policy, Subject aliases, the curation gestures, links
and tags, extraction, link dismissal). A key whose role cannot call a route
gets `403` before the request's body is read; on a route whose body chooses a
write branch, the `403` for that branch comes once the body has chosen it.

**An agent key acts only as its own principal.** Its writes are recorded as
its principal. Its deposit cannot name another depositor or author, a URI, a
source type the engine would not record itself, or a harvester's tag; it never
changes an entry another principal deposited (a deposit at a URI another
principal's entry holds answers `409` with nothing changed) and never retypes
an entry, its own included. An unbound agent key's deposit that names no
project and carries no harvest tag is marked unattributed, so it is never in
view for every project. The agent belief writes assert as the key's principal
and change only a belief that principal asserted as an agent (or, for the
memory-server facade's identity, one it migrated in), unless
`mcp.write.allow_cross_asserter` is set; no agent key changes what an
extractor, an import or a maintenance pass asserted. Every label a store
writer records, and every label a reader trusts by name, is a reserved
principal: `particles engine keys create` mints a non-operator key under one
only with `--reserved-principal`, the key file records that acknowledgement,
and a non-operator key under a reserved name whose record lacks it
authenticates nothing. A record `keys create` would refuse (an agent or reader
record under the operator's name, a padded principal, a project binding on a
reader or operator key, an operator record under a principal other than
`curation.operator_identity`) authenticates nothing either. `mcp:claude-code`,
the stdio MCP server's own identity, is deliberately not reserved, so the HTTP
MCP server's key keeps owning what the in-process server asserted;
every key that shares a principal shares its beliefs.

**A harvester key names only what the harvest sends.** A key minted for a
harvester principal (`claude-code-hook` or `audit`, with
`--reserved-principal`) may name only URIs of the harvest's shapes (a session
transcript, a memory file under a `projects/<project>/memory/` directory, a
project's memory archive), and a memory file or archive only under a
directory the key recorded when it was minted (`--memory-root`,
`--archive-root`), compared segment for segment with no link resolved on the
engine. A `file://` URI must be spelt as the hook spells the path, and may
hold no percent-encoded control character and no raw tab or line break. `particles init claude-code`
prints the mint command with the client's resolved directories filled in when
`engine.base_url` is set; when an inherited `CLAUDE_CONFIG_DIR` names a
directory whose `projects` entry is a link, the root it prints is where that
link leads, by design, so read the printed command before you run
it. The two harvester principals share each other's entries, by design,
because `particles audit` deposits the same transcripts the SessionEnd harvest
does. These limits are accepted: under its own recorded directories a
harvester key may name any project and file name and replace the content of
an entry the real client harvested there, and clients whose keys share a
principal and whose roots coincide share those entries; a harvester key
learns from the text route's `409` that another principal's entry holds a URI,
and its type. On a volume that folds Unicode normal forms or letter case, one
memory file can still be filed under two spellings.

**`--project` binds an agent key.** Its writes, the session exposures it
records included, carry that project and no other. An unbound agent key, and
an operator key, name the project a write is made for, and the engine trusts
that name. A project key is stored and compared in Unicode NFC. A key that
names no project (only invisible, blank-rendering, control, private-use or
unassigned characters) or that reads as another key is refused when a key is
minted with it, in the key file, in an agent deposit's `project:` tag (`403`)
and by `particles memory rescope`. On an agent route a key that reads as
another is refused (`422`), and one that names no project is read as omitted,
so the write is made as a bound key's own project or, for an unbound key,
unattributed. No write is stamped with a key of either class. A key reads
as another when it hides an invisible or format character among visible ones,
mixes scripts (digits included), holds a compatibility form NFKC would change,
a non-ASCII space, separator or control among visible characters, a dash or
other mark that reads as an ASCII one, or one of a hand-kept set of Latin
letters that read as ASCII ones. An operator key's project tags are made
canonical, never refused, by design. Two classes still name a project of their
own: a key written wholly in one script's letters that look like another
script's, or with a same-script lookalike letter the hand-kept set does not
hold, since that set is not Unicode's whole confusables table (an accepted
limit); and a key whose ASCII letter carries a combining mark.
Which rare characters count as unassigned follows the Unicode tables of the
Python the engine runs. Reads are not confined: every key, a reader's
included, reads every project's beliefs.

**A key is not scoped to a store, an accepted limit**. On an engine
that serves several stores, every key reaches each store its role allows.
`mcp.write.enabled_stores` is the only per-store switch, and it applies to
every caller alike. Run one engine per store when clients must not all reach
every store.

**Keys are managed on the engine's host only.** `particles engine keys`
mints, lists and revokes keys (no HTTP route manages them) under a lock on the
key file that stops the verb after two seconds rather than waiting without
bound, and the engine keeps only a SHA-256 digest of each, in
`api.keys_file`. That path must be absolute; a relative one is refused as an
unusable key file. `keys create` writes the new key to a new file only the
engine's user can read (`--output PATH`, by default `<key id>.key` beside
`api.keys_file`) and shows it nowhere else; with `--print-token` it prints the
key instead, and whatever captures a command's output keeps it. The harness's
secret mask covers the per-client key shape (see *The Claude Code harness*
below).

**`PARTICLES_API_KEY` is an operator key.** Any client holding it, or a
per-client operator key, holds the operator's authority, by design:
it can give a deposit any attribution, file text under an
`http(s)` URL the engine never fetched (which takes that domain's trust rule
and joins that host's update lineage), place a document in the collection of
your decision records, name a Subject over another's alias, and append to,
join, retype or re-key an entry another principal deposited. On the agent
routes an operator key acts as `mcp.write.asserter_identity`. Give it only to
clients you would give the operator's authority, and give every other client,
the MCP server and the hooks above all, a key of its own: a process that can
read a key holds that key's role, and an agent with a shell reads every key in
its environment.

**Deposits and session exposures.** The HTTP deposit routes refuse the
migration source type (`400`), which the memory-server facade reads as its own
memory, so only a local `particles import mcp-memory` run creates such an
entry. The agents' `POST /particles/deposit`, the generic deposit routes and
`POST /session-exposures` are held to `api.rate_limit_per_minute`, counted per
key principal. An uploaded `.json` or `.jsonld` file that names no type is
sniffed for a self-declared policy type (taxonomy, trust lens, vocabulary
document) only when it is at most 1 MiB, off the event loop and before the
writer lock, whichever key sent it; a larger policy file is
deposited as an ordinary file unless the operator names its type with
`--source-type`. `POST /session-exposures` needs its store write-enabled,
bounds the beliefs one request names, records the engine's own receipt time
and the key's principal, refuses a bound key's exposure for another project
and an unbound key's project key that names no project or reads as another
(with the exception named above), and stores the key in NFC. A session's frame
reads only the exposures recorded in process, by a harvester key or by an
operator key, so another agent key's rows under the same session id do not
enter it.

**`api.allowed_hosts` bounds a keyed engine too.** It is empty by default. On
an engine bound to loopback, or reached through a UNIX or inherited socket
whose requests name no local address, an empty list admits only the loopback
names, so a proxy in front of it must list its public names. On a wider bind
an empty list refuses no `Host`. Listing the names a deployment answers on
refuses every other `Host` on every path, which closes the rebinding read
described above; the loopback names stay allowed for a local health probe. A
wildcard entry must name a domain you control: one over a bare top-level
domain, a country-code second level (`*.co.uk`) or a listed shared-hosting
suffix (`*.github.io`, `*.ts.net`) is rejected, though the list is not the
full public-suffix list, so a wildcard over a suffix it does not hold, or over
a subdomain of a listed one (`*.s3.amazonaws.com`), is accepted.

**Cross-site browser requests are refused.** A web page on another site can
send a form or multipart POST to a loopback engine without a preflight. Every
gated route therefore refuses, before reading its body, a request the browser
marks as cross-site (`Sec-Fetch-Site: cross-site` or `same-site`, or an
`Origin` that is not the engine's own) unless it carries a valid key, which
such a page cannot attach. A top-level navigation the user opens from another
application (`Sec-Fetch-Site: none`), and a GET from a browser that sends no
fetch metadata, are still served on a keyless engine; such a request can start
rate-limited work that records nothing, but the page that sent it cannot read
the answer. The CLI and MCP clients send neither header. `api.allowed_origins`
lists any other browser origin you trust. CORS is not enabled. Raw deposited
bytes are served with `X-Content-Type-Options: nosniff`.

**A refused request takes no writer lock, and model calls run outside it.**
Every write gate (key, role, origin, `Host`) is checked before a route takes
the store's writer lock, and the belief-write and deposit routes make the
refusals that depend only on the request (its body, its key's binding, a store
with belief writes disabled) before it too. A refusal that depends on what the
store holds comes under the lock. The store's belief writers run model probes
before they take the lock and search the store again under it for a
conflicting claim, and a conflicting claim found only there keeps the belief
from being written ACTIVE beside it: the belief is held for review, or left
for a later run. A URL deposit fetches its URL before it takes the lock. A
writer in another process waits for the lock up to
`storage.write_lock.timeout_seconds` and then fails. One agent write's subject
names are bounded by `mcp.write.max_subject_names` (16 by default). This
property is best-effort, with known gaps in which writers take the lock,
search again under it, or keep to that bound.

**The web UI.** The web UI at `/app` is same-origin. Its content security
policy sets `default-src 'self'` and `connect-src 'self'`, so it sends
requests, and its key, only to the engine that served it. It keeps its key in
the browser's `localStorage` for its origin, where a later server on the same
host and port could read it. The policy covers `/app` only.

## Trust model and known limitations

Beyond the network and auth posture above, the following design choices and
open limits are deliberate or known, and worth understanding before you
deploy Particles or feed it sensitive material.

### Settings come only from a file you name or own, never from the working directory

Particles reads its config from `PARTICLES_CONFIG`, else
`~/.particles/config.yaml`, else compiled defaults. It never reads a
`config.yaml` from the directory a command runs in or its parents, so a cloned
repository cannot point the engine, model-provider or telemetry endpoints your
secrets are sent to at its own host, and cannot turn on the MCP write tools.
A Particles `config.yaml` found there is reported and ignored. A relative
`PARTICLES_CONFIG` is the one way the directory can still supply settings: the
stdio MCP servers, the lifecycle hooks and `particles init claude-code` refuse
one, while the `particles` command line reads it against the working directory
and warns that it should be absolute.

`particles init claude-code` pins an absolute `PARTICLES_CONFIG` into every
hook command it writes, and into the Particles MCP server registration. A
registration already pinned to another config file is reported and keeps its
pins until you re-run with `--config` (an accepted limit), which gives it
exactly the pins the install writes and names each pin it dropped. `--config`
rebinds only the first Particles registration; a further one keeps its pins,
unnamed. Without `--config`, a registration with no config pin
gains the install's pin and keeps its other variables, a `DATABASE_URL`
included; the install names a kept `DATABASE_URL` that differs from the one
the hooks pin.

The memory projection takes the local files it copies from the pinned
config, with one exception: a relative `agent_memory.projection.manifest`,
`claude_code.state_dir` or `claude_code.hook_log_path` there resolves against
the directory a hook runs in, so a repository you open a session in can supply
the projection manifest and receive the hook state. Keep those paths
absolute; their defaults are. A `particles project` manifest may read and write only inside the
repository that holds it (or its own directory, outside a repository), and
never through a link, a hard link or anything but a regular file at the
output, unless you pass `--allow-outside-paths`. The root is opened from `/`
one component at a time without following a link, the output's parent
directories are opened the same way and checked against the root before the
write, and a manifest kept directly in the home directory gets no root at all.
A manifest that fails validation is reported by location and error type, with
none of its values. On a platform whose `open` takes no directory descriptor,
only the output's last component is checked (an accepted limit). A manifest's
framing and direction text reach the synthesis model's instructions as
written, so project only with manifests you trust.

The environment a process starts with outranks the file: `DATABASE_URL`,
`PARTICLES_ENGINE_BASE_URL`, `PARTICLES_OBSERVABILITY_ENDPOINT`,
`ANTHROPIC_BASE_URL`, the `OTEL_EXPORTER_OTLP_*` variables and the other
registered overrides all apply over a pinned config. `particles config
validate` and `particles hook doctor` list the overrides a process has, flag
each one that sends a key, the prompts or telemetry somewhere other than the
file says, and flag the proxy variables spelled in upper or lower case, the CA
variables and a system proxy, each credential variable by name only, never by
value. A proxy variable spelled in mixed case (`Https_Proxy`) is honoured by
the clients below but flagged nowhere, and it also hides the system-proxy note.
They also flag a fixed list
of variables third-party SDKs read themselves (the OpenTelemetry SDK's header,
certificate and client-key variables, the Anthropic SDK's
`ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_CUSTOM_HEADERS`, the Hugging Face Hub's
`HF_ENDPOINT` and tokens, `SSLKEYLOGFILE`); another SDK's other variables are
not listed. Two limits are accepted: `NO_PROXY`, which sends a host around a
proxy, is not flagged, and the diagnostics see only the environment of the
shell that runs them, not the one your hooks and MCP servers start with. A
proxy variable reroutes the native Anthropic client, the OTLP exporter's
sessions and the Hugging Face encoder download, each for connections of the
variable's own scheme, and with no variable set those also follow the
operating system's proxy settings. A proxy named by an `http://` URL, in the
environment or the system settings, stops the native Anthropic client and the
encoder download client from being built at all. The shared fetch client and the engine and
OpenAI-compatible clients take no proxy from the environment, and the `curl`
and `git` subprocesses are started without these variables. The native
Anthropic client, the shared fetch client, the engine and OpenAI-compatible
clients and the OTLP exporter's sessions write no TLS key log, so an inherited
`SSLKEYLOGFILE` records none of their sessions or proxy hops. The native
Anthropic client's own telemetry is set to metadata only, with no trace
propagation, so the SDK's variables cannot put prompt content on exported
spans.

The client refuses to send the engine token over `http://` to a non-loopback
host unless `engine.allow_plaintext_token` is set. The same rule covers an
OpenAI-compatible provider entry's key (`allow_plaintext_key` on the entry),
OTLP exporter headers (`observability.allow_plaintext_headers`) and the
Anthropic key (`llm.anthropic_allow_plaintext_key`). A refusal, and the warning
an allowed plaintext send logs, names the endpoint without any credential
written into its URL. The engine, OpenAI-compatible and Anthropic clients, the
Numista lookup and every OTLP exporter that carries headers follow no
redirect, so a credential is never resent to a host a redirect names. The
shared fetch client drops a caller's own request headers on any hop that
leaves the first request's origin, and OTLP headers bound for a loopback
`http://` collector ignore an environment proxy. A configured secret, and each
credential a request sent, is replaced in the OpenAI-compatible adapter's
error text, in an account-level model error, in a failed consolidation pass's
recorded detail, in the hook log and in an error the Anthropic SDK raises (its
response body and the `ANTHROPIC_CUSTOM_HEADERS` and `ANTHROPIC_AUTH_TOKEN`
values included); the OpenAI-compatible adapter replaces every header value it
sent except `Accept`, `Content-Type` and `User-Agent`, and the OpenTelemetry
SDK's header values are replaced as written in their variable. A value
shorter than eight characters is never replaced (an accepted limit). An OTLP
header value written percent-encoded, as the OTLP format requires for a space,
is sent decoded, and that decoded form is not replaced.

`particles hook doctor` warns about a missing or relative config pin, an
absolute pin inside a git work tree, a hook or registration whose engine pin
is not empty (naming the engine without its userinfo), and a store pin the
refresh would refuse, in `~/.claude/settings.json`, the current repository's
`.claude` settings and the user-scope MCP registrations. It reports the first
problem it finds in each settings file, does not read project-scope or
local-scope MCP entries, and reads `~/.claude` whatever `CLAUDE_CONFIG_DIR`
names. It reads a repository's `.claude` settings files only as regular files
with one name that your account owns, inside the repository, without following
a link or blocking and under `local_files.max_bytes`. A repository you cloned
is owned by your account, so the hook entries its settings carry count as
installed hooks, and the doctor judges the engine token they pin; it reads a
key's role from the key's prefix.

### Which store a process opens

The stdio MCP servers (`particles mcp serve`, `particles memory serve`) never
serve a store from the directory they start in unless you name it.
Started with `DATABASE_URL`, a server uses that store as given, so a relative
`DATABASE_URL` in its registration opens against the directory it starts in;
write it absolute. Started with `PARTICLES_CONFIG` whose default store path is
relative, it refuses to start and names the fix. Started with neither, it
serves the store `~/.particles/config.yaml` names (a relative path there is
read under `~/.particles/`), and with no such store serves
`~/.particles/particles.db` and says so on stderr. A relative SQLite path
among the configured named stores, or for the store the memory-server facade
writes, is refused, and the facade refuses a write to any store other than the
one it reads. A leading `~` in a store path means the home directory.
`particles mcp serve` pointed at a remote engine (`engine.base_url`) opens no
local store; `particles memory serve` keeps these rules then, and its facade
refuses every tool call while `engine.base_url` is set.

The Claude Code lifecycle hooks apply the same rule before they run: a hook
whose store has a relative SQLite path logs the refusal and exits without
touching it. A hook counts as pointed at a remote engine only when its pinned
config names one, since `particles init claude-code` pins
`PARTICLES_ENGINE_BASE_URL` empty in every hook command and in the MCP
registration it writes; an engine the harness's environment names does not lift the
rule. `particles init claude-code` pins an absolute `DATABASE_URL` into the
hooks it writes for the default store, while a named store's hooks carry only
the `PARTICLES_CONFIG` pin; for a relative named store it warns that the hooks
will refuse it.

The `particles` command line and `particles engine serve` still use
`./particles.db` when nothing names a store, because you choose the directory
you run them in. A cloned repository that ships a `particles.db` therefore
supplies the beliefs those commands read when they run inside it, including
when an agent runs them there. Such a store chooses no host file: the local
refresh reads no `file://` entry of a store whose path resolves against the
working directory, and `particles corpus refresh` exits naming the
fix. Refusing a store named by an absolute path that lies inside a repository
checkout, or that another user owns, is best-effort; a
store unpacked from an archive and named by an absolute path is trusted as your
choice.

Beyond where a store came from, the refresh trusts a file only when you alone
can write it. That property governs the store, the SQLite files
beside it, the config file the process loaded (which chooses the roots the
refresh reads), and a rule file read through a recorded root: you own the
file, no other user may write it, and no user but you and root may change a
directory its name resolves through, including the directory a symbolic link
on the way lives in. A file or directory that cannot be judged fails it, and
so does one on a macOS volume mounted to ignore ownership. A store or config
file that fails stops the refresh with its cause and fix (`chmod go-w`, for
one). The config judged is the file the settings were loaded from: a config
whose name no longer holds that file, that lies inside a git work tree, or that
was another account's or writable by others when it was loaded, stops the
refresh until the process restarts. One check judges it for the refresh, the
counts `particles rules` and `particles hook doctor` print, and
`init claude-code`. With no effective user id the local refresh reads
nothing. Its limits:

- The walk up a name's directories stops at the first broad root: the
  filesystem root, the home directory, an ancestor of either, and any mount
  point on the home directory's filesystem, whether or not it holds home. A
  directory others may write at or above such a mount point is not judged,
  and neither is a home directory others may write; the work-tree rule above
  stops at the same root, so a `.git` there does not count.
- A config named through a symbolic link is judged at load by the link's
  directory, not the target's. The config's directory change time is not
  compared, an ACL entry on it removed before the first refresh is not seen,
  and the directories above it are judged as they are now.
- Off macOS only the mode bits are read, so a Linux POSIX ACL counts through
  its mask, and an NFSv4 ACL, an exFAT or SMB mount whose mode bits do not show
  who may write, and a FUSE mount with `allow_other` and no
  `default_permissions` are not seen (an accepted limit).
- The store is judged when the refresh starts, after it is already open, so a
  path swapped when the store was opened and put back before the check goes
  unseen; undoing such a swap needs the other user to hold, and then give up, a
  grant on a directory in the store's path that the walk judges.
- Only the refresh judges the config file: every other command reads the
  config it loads whoever may write it, and `particles hook doctor` and
  `init claude-code` only warn about a pinned one.
- A file read through a root you name in the config (`rule_sources.paths`,
  `local_refresh.allowed_roots`), a rule file under `~/.claude`, and a memory
  file under `~/.claude/projects` are not held to the property, since you chose
  that root.

A store's blob directory (`./corpus_blobs` beside it by default) is read and
written only inside itself: no symbolic link is followed, a blob is written
under a temporary name and linked into place, a write that finds another
regular file at a blob's name replaces that name and refuses anything else,
and a read returns a blob only when its bytes hash to its name. A blob
directory a repository ships therefore cannot direct a deposit's bytes, or a
read, to another file on the host. A file planted at a blob's name is hashed
in blocks before it is refused, never buffered whole, but it is hashed to its
end whatever its size, the raw-bytes route reads on the engine's event loop,
and a file that grows between the hash and the read is buffered whole, so a
shipped blob directory can set how long and how much one read takes.

`particles init claude-code` run with no store configured creates a separate
memory store at `~/.particles/memory.db`. Run with `--store default`, or when
`default` is the one write-enabled store, it pins `./particles.db` of the
directory it runs in, as an absolute path, into the hooks and the MCP
registration it writes, so run it where you mean the store to live or set
`DATABASE_URL` first. It warns when that path lies inside a git work tree,
names a default store that a `DATABASE_URL` in its environment, or the working
directory, chose, and names an inherited `PARTICLES_CONFIG`, warning when that
file lies inside a git work tree, is another account's or other users may
write it. `particles config validate` names the store file in use.

One dependency here is on the harness rather than on Particles. A repository
can still set `PARTICLES_CONFIG` or `DATABASE_URL` for a process by shipping
the command that starts it, as a committed Claude Code project hook or MCP
entry, or an `env` block in its project settings. Such an entry already runs a
command of the repository's choosing, and Claude Code asks before it trusts a
project's entries, so the pin adds nothing to what the entry could do. That
holds only while the harness keeps asking; it is Claude Code's behaviour, and
no test in this repository can check it. The install never pins
`PARTICLES_ENGINE_TOKEN` into a hook or the registration: a hook or MCP server
routed to an engine presents whatever token the environment Claude Code starts
it with holds, or none. Hiding credentials in what the install prints is
best-effort, and some are printed as written. `mcp serve` logs an engine its environment names
without its userinfo.

Without such an entry a repository still reaches Particles through what the
commands read from the directory: the `particles.db` above and the blob
directory beside it, the rule files `particles rules sync` and
`particles init claude-code` enrol, `.claude` settings that list stores the
wrong-store guard accepts, its `.git` entries (which choose the project key a
session is stamped with and whose `MEMORY.md` it renders; a `.git` file that is
a link is not followed to another repository), a projection manifest's framing
when you project it, and a `config.yaml` that is parsed only to decide whether
to warn. Keeping the config you pin out of the working directory's reach is
best-effort for its relative paths: some resolve against the working
directory, or beside the name the config was loaded by where a symbolic link
can move them, so keep every path in that config absolute. A
relative `rule_sources.paths` or `local_refresh.allowed_roots` entry is
skipped with a warning, and a relative `api.keys_file` is refused.

### Outbound connections

Outbound traffic is open by default (`network.egress: open`). Setting it to
`none` or `allowlist` is checked at each of Particles' own transports, with a
process-wide backstop behind them in the CLI, the hooks, the engine and the MCP
servers. The backstop sees what passes through Python's socket layer (TCP
connects, name and reverse lookups, and datagrams sent to an address) and
judges a connect that names a host before Python resolves the name. Code that
calls the raw `_socket.socket` type, and native code that opens its own
sockets, are outside it. Under a refusing posture every event loop the CLI,
the hooks, the engine and the MCP servers start arms the backstop, and a loop
whose sockets it cannot see is refused: `engine serve` runs the standard event
loop, an engine started on uvloop refuses to start, a CLI verb exits 1 with
one line, a hook logs the refusal and exits 0, and the MCP servers refuse to
start. One path Particles starts is not held to an allowlist: when the
allowlist names `huggingface.co`, the encoder download's Xet-backed files are
fetched by a native client from storage hosts the list does not name, and the
Hub client follows redirects without an egress check of its own.
Keep the encoder cached, or deny the download at the operating system. For a
guarantee against any path, also deny network access at the operating system.

The two fetches that run as subprocesses (`curl` for Reddit, `git` for a gist)
are pinned to addresses this process vetted, inherit no proxy, CA,
`SSLKEYLOGFILE` or `SSH_ASKPASS` variable, read no `~/.curlrc`, and `curl`
expands no URL glob, so one deposited URL is one request. They, and the other
programs Particles starts (`claude` for `init claude-code`, `git` for the
`MEMORY.md` history, `hook doctor` and `subjects fold-artifacts --repo`, the
query-latency benchmark's helpers), run with a trusted `PATH`: only absolute
entries that exist, that you or root own, and that no other user may write,
judged with every link an entry resolves through and the directories above
each, falling back to the trusted default path when every entry is dropped,
so a program is never looked up in the working directory. An entry that an
administrative group (`root`, `wheel`, `admin`, `sudo`) may write is kept (an
accepted limit).

Telemetry is off by default. With it on, spans describe outbound
work, so an exporter pointed at a third-party collector ships that metadata
off the host. Server spans record the route template, never the concrete path
or query string; httpx client spans record only the destination's scheme,
host and port; with `observability.traces` on, every span records an
exception's type and no message or status description, and the store's engine
names no bound value in a statement error. With observability on and traces
off, the httpx instrumentation still runs, and its client spans reach an
embedding application's own tracer provider with the exception message and
stack trace. With the SQLAlchemy release the project locks, the
instrumentation exports no database span. With `observability.logs`
on, an exported log record carries the message template without its
arguments, a logged exception's type and the trace correlation, and no
exception message, stack trace or `extra=` attribute; the console still shows
the record whole. A message whose own text already holds exception text, as an
f-string does, is exported as written, and so is an exception object logged
as the message with no arguments when no earlier handler formatted the record.
With telemetry on, the httpx instrumentation also adds a
`traceparent` header to each outbound request, so a fetched host, a deposited
URL's included, receives the trace id.

By default, outbound traffic also includes the link a deposited Reddit, Hacker
News or Mastodon post names, which is fetched as well, and for a Reddit
deposit the Arctic Shift archive, which serves the thread.

The Obsidian plugin refuses to send its engine key over `http://` to a
non-loopback host unless its *Allow token over plain HTTP* setting is on, and
it stores the key unencrypted in the vault's plugin data file (an accepted
limit), so give it a key of its own, a `reader` key when it only shows
callouts.

### Ingested content is untrusted, and is stored verbatim, including any secrets

Deposited source material (URLs, files, pasted text, social-media threads) is
**fully untrusted** by design: it flows into the LLM extraction prompts, the
embedding model, the belief store, and exporter and projection output.
Particles tracks provenance and confidence for that content but does **not**
sanitise or scan it. In particular, **any secret in a deposited document (an
API key, a password, a token) is persisted content-addressed in the corpus and
can resurface** in particle content, query answers, the session digest, and
exporter files. Only the Claude Code harness intakes (session transcripts,
memory files, rule files) pass a best-effort secret mask, described below, and
the nightly refresh applies it again only to an entry tagged as a rule or
memory file; an agent key's own text deposit is stored as sent. Do not deposit
material you are unwilling to store unredacted, and treat the corpus blob
store and the belief store as **as sensitive as the most sensitive document
you have deposited**.

Stored text reaches your terminal too. Showing each control and invisible
format character in stored text as a visible escape is best-effort for the
command-line verbs, and some print a stored line break as a new line, with
known exceptions. JSON the `corpus`, `reindex` and
`query` verbs print escapes every character outside ASCII. The `audit`
report, which is Markdown, prints stored text as written, controls included.

The folder importers do not follow a symlinked file: `particles import vault`,
`particles import web-clipper` (including the daemon's captures watcher),
`particles import project`, the memory-file harvest and `particles audit`'s
walk of the session transcripts skip one and log it, and against a remote
engine the importers upload the bytes they read rather than reopening the
path by name. Rule-file intake (`particles init claude-code`,
`particles rules sync`) takes in a symlinked rule file only when it resolves to
a rule file inside the same repository or `~/.claude`. Every read contained to
a root (walks, rule intake, web captures, the local refresh, bundle members)
refuses a hard-linked file. A file you name yourself, as in
`particles deposit FILE`, is still read through a symlink.

A deposit whose bytes match another depositor's local file entry gets an entry
of its own rather than joining it. A web capture and a deposit of the URL it
names never share an entry, in either order: a capture of a URL another
depositor's entry holds is deposited as a local file of its own, and a later
deposit of a URL a capture named first takes the URL from the capture's
entry, which keeps its snapshots, tags and depositor, with an event naming
both entries and the URL. A
keyless engine accepts no HTTP deposit, and an agent key's deposit is confined
to its own principal's entries (above). An operator key's deposit is read as
the depositor it names, by design, which leaves four limits for
operator keys: a text deposit can append to a local file entry the refresh
does not keep current, whoever enrolled it; a deposit can create an entry at a
path the harness later enrols, which the harvest then adopts with that
deposit's depositor (the adoption adds the intake's tag, so the refresh masks
the file); an upload that names an operator file entry's own depositor can
join it; and a local file entry enrolled over HTTP under a local depositor's
name by a release before `1.210.4` is still refreshed from the engine's disk
when it lies inside the rule-file or memory-file roots. A key minted with
`--reserved-principal` under a rule-file intake's name (`rule-sync`, `init`)
matches that name's entries of the same type, by design.

A refresh root that `rules sync` or `init claude-code` recorded, and the main
checkout of a recorded worktree, admits a file only when the refreshing user
owns it, checked on the opened file before a byte is read, and only while that
user owns the root directory; a root another user owns is refused and counted
with its fix. A rule file read through a recorded root must also meet the
property under *Which store a process opens*. A symlinked rule file
the refresh does not follow, a file that is not regular and one over
`local_files.max_bytes` are refused and counted; a `local_refresh.allowed_roots`
entry that is the filesystem root, the home directory or an ancestor of either
admits nothing, and `config validate` warns about it.

Interchange bundles and the YAML a deposit carries (a decision record's
frontmatter, a web capture's header, a memory file's header, the general
extractor's frontmatter) are read under bounds: a per-member cap for a
bundle's manifest, particles and subjects members and one shared cap for at
most 64 others; YAML read under an alias budget that counts the nodes an
expansion builds, with merge keys folded in linear time and a depth bound that
counts an alias's levels; and a YAML-LD bundle member read one top-level unit
at a time. A header that cannot be built is read as absent, and a web
capture's header must close within its first 8 KiB. JSON from a third-party
source is decoded with a depth guard that turns too-deep nesting into a
refusal; a few decode sites still raise past that refusal. A
capture whose read or deposit fails is logged and skipped without undoing the
others in the scan, and the daemon retries it on later passes.

The URL inbox's pass deposits under the store's writer lock, reads the inbox
off the event loop without following a link or blocking, refuses anything but
a regular UTF-8 file within `inbox.max_bytes` (1 MiB by default), and rewrites
the inbox from the file as it then stands, so a URL shared while the pass runs
stays pending. One shared in the moment between that last read and the
rewrite's rename can still be lost, since whatever appends to the inbox takes
no lock (an accepted limit). After a pass that raised, `inbox watch` stops and
the daemon waits for the next share before it tries again.

An imported unit's provenance that names an entry or a belief the target
store already holds, and a `supersedes` that names a belief it holds, are kept
only as origin metadata, so a bundle cannot choose which of your entries
attest its claims, widen a belief into another project's view or become your
belief's successor. Every imported particle is asserted by the import, with
the asserter the bundle claimed kept as an author. An import, and a restore
without `--trust-bundle`, stores each belief as falsifiable and keeps a
declared assertion modality, scope or polarity that would take the claim off
the conflict surface only as the bundle's claim. An import keeps each unit's
lifecycle status as the bundle states it, and a status other than
ACTIVE that the import kept, rather than one this store reached, holds no
later claim for review and absorbs none.

(Prompt injection from deposited content is *contained but not eliminated*:
outside the benchmark harness, every model prompt that carries deposited,
stored or harvested text fences it with a per-call nonce, and no tool use or
function calling is exposed to the model. A test finds each call to the
model's entry points, following wrappers that pass a system turn on, and fails
the build on one nobody has classified; a call made through a function passed
as a value is outside what it can find. That test checks that a fenced site
sends a system turn; behavioural tests drive each fenced prompt builder with a
claim that tries to close its fence. A projection manifest's framing and
direction are not fenced (see above). An injected document can still influence
the *claims extracted from it*, and model verdicts on those claims change what
is stored; see the next section. The belief store is a record of what sources
said, weighted by trust, not an oracle.)

### Model verdicts can retire a claim without review

Four automatic paths act on a model's reading of claims, and a fifth on what a
document declares about itself, each on by default:

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
  default). Extraction, interchange import, the reconcile sweep and the
  paraphrase pass each read the target store's own mode.
- **Document supersession.** A document that declares it replaces another (a
  decision record naming the record it supersedes) retires the claims of the
  document it names, read only within the declaring document's own collection:
  entries recorded with the same depositor, source type, host and directory
  tree. Over HTTP an operator key names those fields, so it can place a
  document in your collection, and a harvester key's harvest-shaped path
  chooses the source type and directory of a collection under its own
  principal. A URL with a short path, or a deposit with no URI, shares one
  collection per host or per depositor. The declaring frontmatter is read
  under the bounded alias budget, and its record type and number only as plain
  values.

Each prompt fences the claims it reads, so a claim cannot answer the probe for
the model. The re-anchor and release passes are belief writers, so a claim
committed while one runs is searched for under the writer lock before it
writes, best-effort as for the other belief writers (see the deployment
posture above). Each change these paths make is recorded in the operator event log,
and so is each retirement made by the abstraction pass, `lint --fix`,
`reindex`, the generation cascade and the trust cascade. The update and
re-anchor passes retire only extracted claims, and an agent's own earlier
claim when the same agent later states a new value; neither retires a claim
the operator asserted. A retired claim stays in the store with its history and
stays retired: the one status change back to active undoes a duplicate merge,
and a review can admit a held restatement of a retired value as a new claim.

Part of the 2026-10-23 review's finding on the update pass is still open.
A rule file (`AGENTS.md`, `CLAUDE.md`) retires only its own
earlier claims, only the harness's own deposits share the memory-file lineage,
and every deposit's date is held to its deposit time. Memory files under
`~/.claude/projects` can still count as one source with each other, and so
can Claude Code session transcripts, so a newer statement in one of them can
retire an older claim from another, and a harvester key's deposit can retire a
belief another harvest deposited. The confirming second reading asks the model
the same question again rather than re-reading the source passages, and
automatic retirements are not summarised at session start. An operator key's
text deposit that names a URL shares that host's update lineage with your
dated, unattributed captures from it. The session-date repair
(`particles memory redate`) repairs only the harness's own session deposits,
holds the date it writes to the first capture, and reads a transcript only
when the session id has a session id's shape and the transcript lies inside
the projects root. Read `particles events list` after depositing material you
do not trust, or set `reconciliation.update_supersession.enabled: false` to
turn the update pass off, so that a newer claim no longer retires an older one
by date.

### Project scope is a relevance default, not isolation

With `claude_code.observer_scope: project` (the default), a session's digest,
its `MEMORY.md` region and its searches show global beliefs plus those observed
in its own project. That narrows what an agent is shown; it does not restrict
what it can read. A project-bound MCP server can still read any belief by id,
list subjects, events and corpus entries across the store, run lint, and read
the whole store when a tool is called with `all_projects`. The filter engages
only after `particles memory rescope` has run on the store. On a rescoped
store, a project-bound MCP server refuses to retract, supersede, tag, untag or
link a belief that only another project sees; on a store never rescoped it does
not. An agent key minted with `--project` writes only as that project. On a
rescoped store an unbound agent key's or an operator key's call to the agent
HTTP routes that names no project is refused, but the engine trusts the project
such a key names. An operator key's deposit through the generic corpus routes
can add a project key to an entry another principal deposited, which moves
that entry's beliefs into another project's view; an agent key's deposit
cannot.

The memory-server facade (`particles memory serve`) has no project binding at
all: its reads cover every Subject and belief in the store, it may add
observations and relations to any Subject, and it changes the class and the
tombstones only of Subjects it created itself. It counts a belief as its own
memory only when it asserted the belief or the belief cites an export a local
`particles import mcp-memory` run deposited: the HTTP deposit routes refuse the
migration source type, every tag route and verb refuses the migration tag, an
import records a unit naming the facade's or the agent's identity as an
import, and `particles interchange restore` neutralises those identities, the
migration tag and the reserved calibrations a bundle carries unless you pass
`--trust-bundle`, which restores them as written.

A project is keyed by its path with every character other than an ASCII letter
or digit turned into a dash, as Claude Code names its memory directories, so
two repositories whose paths differ only in such characters (punctuation,
spaces, accented or non-Latin letters: `/src/a-b` and `/src/a/b`) share one
project scope, and an agent key bound to one of them writes as the other too.
Once the SessionEnd hook has recorded both repositories,
`particles hook doctor` run in either names them; the digest, `MEMORY.md` and
the MCP reads do not. A `.git` file that is a link to another checkout's does
not adopt that repository's key. Keep material that must not reach an agent in
a separate store.

### The Claude Code harness writes your sessions into the store and the store into your sessions

The SessionEnd hook deposits a distilled transcript of each session and the
project's memory files, and rule-file intake deposits `AGENTS.md` and
`CLAUDE.md` files. Each goes through a best-effort redaction pass first, and
so do the memory index lines the harvest folds into its per-project archive
and the transcript the SessionEnd utility judge reads. The pass masks common
credential shapes, and it is best-effort: a secret split across a line break
is left as written (an accepted limit), some secrets in other shapes or
spellings are left as written too, and a large crafted transcript can slow it.

A memory or archive file the engine refuses at the SessionEnd harvest is
recorded in the hook log and the harvest goes on with the other files, and
`particles hook doctor` reports how many files the newest session end had
refused; the `MEMORY.md` projection is skipped while any file was refused.
When the nightly refresh re-reads a file whose entry is tagged as a memory or
rule file, it applies the same mask; a file in which it masks something is
stored masked and is no longer refreshed. A memory file is then kept current
by the SessionEnd harvest, and a rule file only when you run
`particles rules sync` or `particles init claude-code` again. A harness
enrolment that adopts an entry another intake created first adds its tag, so
that file is masked on refresh too.

In the other direction, the `MEMORY.md` projection writes ranked beliefs into
`~/.claude/projects/<project>/memory/MEMORY.md`, and the SessionStart hook
writes a digest of them into the session, both of which Claude Code reads in
every later session in that project. A poisoned belief that ranks high enough
therefore reaches an agent that has tools. Each belief is one line of the
digest, its stored line breaks flattened, with the contested badge ahead of
the claim. Keeping a stored belief from imitating the line's own parts (its
confidence figure, its contested badge, its date, its subject list or the
order of lines) is best-effort. Each `MEMORY.md` bullet is held to
the same best-effort rule, and its Markdown stays live, since the rules it
carries quote commands. A `MEMORY.md` region edited since the projection
rendered it gets the full digest. A `particles project` manifest section
written without model synthesis is held to the same best-effort rule.
Set `agent_memory.projection.enabled: false` to stop the
projection.

`particles init claude-code` and every hook write make an existing state
directory owner-only, through a descriptor opened without following a link.
A project-scope install (`--project`) merges the project's settings file
through the `.claude` directory it checked at the start (a regular file with
one name, owned by you, read and written relative to that directory), refuses
a `.claude` directory another account owns or that any account may write in
(one its group may write is admitted, by design), and opens a `.claude`
directory it creates itself without following a link, checked empty. `init
--remove` replays a recorded registration only when its state file is private
to you and its program is an executable you or root own that no other account
can write, reached through directories, and links, that are yours or root's
and that no other account may write into unless sticky. With
`agent_memory.projection.git.sign` on, the `MEMORY.md` history commit is
skipped when the repository's own config sets any `gpg.*` key,
`user.signingKey`, `commit.gpgSign` or `tag.gpgSign`, so signing follows only
your global choice. That commit's `git` runs without an inherited
`GIT_CONFIG_GLOBAL`, `GIT_CONFIG_SYSTEM` or `GIT_ATTR_SOURCE`, but it still
reads the global config an inherited `HOME` or `XDG_CONFIG_HOME` names, and a
filter driver that config sets runs when `MEMORY.md` is staged.

### The MCP write surface is a local-process trust boundary, not a network one

The stdio MCP server (`particles mcp serve`) has **no token**. Its eight write
tools (assert, supersede, retract, deposit text, link, unlink, tag, untag) are
offered only for stores in the default-deny `mcp.write.enabled_stores`
allowlist, and each writes only the store it names (writes are off unless a
store is explicitly enabled). Through a remote engine, the engine
checks its own allowlist again for all eight, and sets the attribution of a
text deposit itself; the engine's generic corpus deposit routes are not behind
`mcp.write.enabled_stores`. The trust boundary is therefore the **host**: any
local user who can start the MCP server with a config that write-enables a
store, and can open that store's database, can mutate it. This is appropriate
for the **single-operator, single-host** model Particles targets: the MCP
server is a locally-spawned child process, not a network service. On a shared
host, treat write-enabling a store as granting belief-write to every local
user, and keep `mcp.write.enabled_stores` empty (the default) on stores that
must stay read-only.

A looping agent can still spend. The MCP servers on one store admit at most
`mcp.rate_limit_per_minute` (60 by default) units of model-capable work a
minute between them: a `query` with a question, `particle_assert`,
`particle_supersede` and `graph_view` each cost one, and a memory-server
facade write costs one for each assertion it carries, refused whole when the
allowance cannot cover it. A call over the cap returns a tool error. Other
tools are not counted, and `0` turns the cap off. The allowance is kept beside
a file-backed store (`<store>.mcp-rate`, owner-only, under a file lock, keyed
on the resolved store file and written to a fresh file renamed into place), so
every server process on that store shares it. A store with no file, and a
server routed to a remote engine (whose own rate limit bounds the calls it
forwards), keep one allowance per process. An uncounted `deposit_text`, and
the excerpt each assertion deposits, leaves a pending snapshot that the nightly
consolidation extracts through a model, bounded by
`consolidation.max_pending_entries` rather than by this allowance.

### Each agent principal owns its own beliefs; callers are not isolated

Agent writes through the stdio MCP server are attributed to one config-bound
identity (`mcp.write.asserter_identity`, default `mcp:claude-code`), and the
memory-server facade's to another (`mcp.memory_compat.asserter_identity`).
Over HTTP an agent key's writes assert as the key's own principal, and an
operator key's writes on the agent routes as `mcp.write.asserter_identity`.
The "own-beliefs-only" guard on supersede, retract, link, unlink,
tag and untag keys on that identity, so distinct agent principals
cannot change each other's beliefs, and it admits only a belief asserted as an
agent's, or, for the memory-server facade's identity, one migrated from a
memory-server export, unless `mcp.write.allow_cross_asserter` is set; no agent
key can change what an extractor, the interchange import or a maintenance pass
asserted, and none may change an operator-asserted belief. Keys that share a
principal share ownership: an HTTP MCP server's key under `mcp:claude-code`
owns what the stdio server asserted, by design. An agent's belief cites a
source the agent deposited itself, or one that reads exactly as such a deposit
(the agent's identity as depositor and author, a conversation source and the
agent's project marking), so it carries the agent's trust
(`mcp.write.agent_trust_rank`) rather than borrowing an operator source's. An
operator key can move such a belief to neutral source trust, by design: a
generic deposit of the same bytes as an agent excerpt retypes the agent's
entry, and an entry deposited under the agent's depositor label is reused by
the agent's next excerpt deposit of those bytes. An agent key under another
principal can do neither, and a reader key writes nothing. An agent's assertion that names a Subject the
store does not know, through an agent key, the stdio MCP server or the
memory-server facade's entity tools, mints a local Subject, which every
principal's later mention of that name, and a later extraction's, then
resolves to, so one agent or project can choose what another
project's mention of a new name means. The owner has chosen a fix for the
agent-key case (a new Subject an agent key names is minted provisional and
scoped to that key's principal until an operator or a second principal
confirms it); it needs its own decision record, and until it ships this is a
known limit.

Particles is a **single-operator** tool. Per-client keys bound what each
client may write and record who wrote it, but they do not isolate callers:
every key reads the whole store, every operator key holds the operator's
authority, a key is not scoped to a store (an accepted limit, see *Deployment
posture*), and per-tenant policy and finer authorization are not provided. A
multi-tenant deployment is **out of scope**. If you need it, put it in front
of the engine: a per-tenant proxy with its own authorization, or one engine
per tenant.

### Releases and the supply chain

Each published distribution is built and uploaded by its public repository's
release workflow: the job that holds the PyPI publishing identity installs
nothing, checks the files against the digests the build job recorded, and runs
a pinned upload action, which runs in its own container image. Each
distribution's SBOM is made from a wheels-only install of its closure, read
without starting the closure's interpreter, checked against the digest its
generating job recorded and signed over the distribution files. The provenance
and SBOM attestations run on a release event and on a manual run of the upload
workflow, attached to that version's Release once PyPI lists every file the
run built under the same digest, and a re-run of a release event proves PyPI
serves exactly the files it attests. No job that holds a write or signing
identity runs the container image. The build of the standard's site, which
serves the normative schemas, installs its toolchain from a hash-locked,
wheels-only requirements file, and its checkout keeps no credential. CI
installs both browser clients with their install scripts off, typechecks and
builds them, and fails when the committed web UI bundle differs from the
build. The review job that runs a model holds a read-only token and refuses
every command while a model credential is in its environment, and a separate
job that runs no model posts its review, editing only a comment that carries
the record's marker.

A version bump is tagged as a release only after a push, schedule or dispatch
CI run on that commit succeeds. A bump whose own run failed is never tagged,
and once a later bump's run passes it is retired. The tagger refuses to build
on a release tag it did not cut unless such a run succeeded on that tag's
commit. The release scripts (the one-command release and the by-hand
`publish_export.py --prepare-push` alike) refuse a version whose commit has no
such run, a version whose private tag names other code or that lacks a lower
release, and an untagged version unless the commit it exports is the one that
bumped to it. The export copies only tracked regular files out of the
overlays, and refuses a symbolic link there or among the files it would copy.
A published release and the private tag of the same number therefore name the
same code, provided nothing changes a file in the exporting checkout while the
export runs: the files are read from the working tree after that check, and a
file swapped for a link, or written through a hard link, in that window ships
bytes the recorded commit does not hold. A ruleset in the repository where release tags are cut lets only
the release-tagger app create, move or delete one, the PyPI publishing
environments of both code repositories admit only `main` and release tags with
no administrator bypass, and `main`'s required checks (`Unit tests`,
`Lint & type-check`) must pass on a branch that is up to date with `main`
(repository settings, checked 2026-10-30).

These limits remain:

- **The repository settings have gaps.** The release-tagger app's private key
  is a repository secret that a workflow on any branch can read, so anyone who
  can push a branch can act as the app. `main`'s branch protection
  is not enforced for administrators, who can merge without the required
  checks passing.
- **CI does not run everything on every commit.** The run on `main` may skip
  the unit-test shards and the image build when a pull request's run passed
  them on the same tree, and it vouches on the newest passing pull request
  run even when a newer run on that tree failed. The SDK-range job,
  which runs the Anthropic client's guard on the published core's `anthropic`
  floor (`1.17.0`) and the newest admitted 2.x, and the OpenTelemetry extra's
  real SDK, is advisory: it blocks no merge, and when it runs on `main` and
  fails, that run fails, so the bump is not tagged. Some OpenTelemetry
  log-export regression tests run in no CI job, and the dependency audit does
  not read the audit dependency group.
- **The container image's Python dependencies are not hash-locked**.
  They are resolved from PyPI when the image is built, under a date
  cutoff, and listed in the image's signed SBOM. The embedding encoder baked
  into the image is pinned to one revision, and every file of the encoder's
  cache is checked against that revision's digests before the image is tagged
  and again before its SBOM is signed.
- **The release gate's own installs are hash-locked only in part.** The
  split-build gate run on the owner's machine before a release installs each
  tree's closure from wheels only, and every package the core tree's lock names
  at its locked version and hashes. Its child processes get only the
  variables they need, with a proxy's userinfo removed, a fresh private `HOME`
  and no `uv` configuration file, and still run as your user. A package only
  the engine needs, and the import-boundary checker, install without hashes,
  because the engine tree carries no lock by decision (an accepted limit).
- **The published core's own dependency floors can lag the SDK's.** Its
  `pypdf` floor admits releases the SDK's floor excludes for advisories, so an
  install of the core can end up with one of them.
- **Not every release contains every release numbered before it.** A release
  tagged on a branch before its merge lacks the fixes merged ahead of it:
  `1.212.0` lacks `1.211.6` and `1.211.7`, `1.214.8` lacks `1.214.6` and
  `1.214.7`, `1.215.0` lacks `1.214.7`, and so do 19 older releases, from
  `0.61.0` to `1.206.0`. All 22 are listed as such, the export refuses each of
  their versions, and every release from `1.215.1` on contains every lower
  release.
- **Not every release passed CI.** `1.222.3`, `1.222.4` and `1.222.5` were
  tagged on commits whose CI run failed, `1.222.5` by hand.
- **The public repositories run the workflows of the last release exported to
  them.** Changes to these workflows reach them with the next export; no
  release after `1.214.1` had been published when this was checked on
  2026-10-30.

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub
issues, discussions, or pull requests.**

Instead, use either of these private channels:

- **GitHub private vulnerability reporting** (preferred), on the public
  repository the issue concerns:
  [particles-engine-py](https://github.com/LinkedParticles/particles-engine-py/security/advisories/new)
  for the engine, the command line, the MCP servers and the harness,
  [particles-core-py](https://github.com/LinkedParticles/particles-core-py/security/advisories/new)
  for the client library, and
  [particles-standard](https://github.com/LinkedParticles/particles-standard/security/advisories/new)
  for the schema and conformance artifacts. Private reporting is enabled on all
  three (checked 2026-10-30). When unsure which applies, use
  particles-engine-py.
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
- **Multi-tenant deployments**: Particles is a single-operator tool (see
  *Trust model and known limitations*). Per-client keys give three roles and
  attribute each write to its key's principal; read isolation, per-store key
  scope, per-tenant policy and finer authorization are not provided.

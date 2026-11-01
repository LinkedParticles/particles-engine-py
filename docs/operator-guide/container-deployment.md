# Running the engine in a container

Running Particles as a service used to take four hand-assembled parts: the
engine, a loop over `extract --all-pending`, `inbox watch`, and a launchd or
cron entry for `memory consolidate --if-due`. The consolidation design chose that
shape on purpose (external scheduler, no resident process), and its own §2
Correction then wrote down what it costs: a LaunchAgent inherits no working
directory and almost no environment, so a mis-pinned job runs "successful"
cycles against an empty store and scatters blob directories somewhere nobody
looks.

A container has no launchd and no cron, so this page is where that changes.
The engine grows an opt-in **resident daemon mode**, and the image bakes every
store-adjacent path absolute under one volume, which closes that failure class
by construction rather than by operator discipline.

**The launchd/cron recipes are still correct** for hosts that do not run a
container. Nothing on this page deprecates them; see
[Scheduled consolidation](scheduled-consolidation.md).

## Quick start (`docker run`)

The image is published: `ghcr.io/linkedparticles/engine`, one manifest for
`linux/amd64` and `linux/arm64`, tagged `latest` and immutably `vX.Y.Z` per
release. Its code is exactly the released PyPI pair (the image is the
release, containerized), and every push carries a build-provenance
attestation you can check:

```bash
gh attestation verify oci://ghcr.io/linkedparticles/engine:latest --owner LinkedParticles
```

Each platform image also carries a signed **SBOM**, a CycloneDX 1.6
inventory of its Debian packages, its Python closure, and the revision of the
embedding encoder baked into it, which the code pins and the image workflow
checks (see [What is in the image](#what-is-in-the-image)). The job that signs
it runs no code from the image or a package index, starts no builder
container, and signs only an SBOM that names that image's own digests and the
pinned encoder. No job that holds a write or signing identity runs the image:
the post-publish smoke test runs in a job of its own that can
only read, with each container cut off from the network.
Verify it and print it in one step:

```bash
gh attestation verify oci://ghcr.io/linkedparticles/engine:latest --owner LinkedParticles \
  --predicate-type https://cyclonedx.org/bom --format json \
  --jq '.[].verificationResult.statement.predicate'
```

The same documents are attached to each version's GitHub Release as
`engine-<version>-linux-<arch>.cdx.json`, beside the two wheels' SBOMs.

To run it:

```bash
export PARTICLES_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
docker run -d --name particles \
  -e PARTICLES_API_KEY -e ANTHROPIC_API_KEY \
  -p 127.0.0.1:8000:8000 -v particles-data:/data \
  ghcr.io/linkedparticles/engine:latest
```

## From a source checkout (compose)

For development, or to run unreleased changes, build from the tree instead:

```bash
export PARTICLES_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

```bash
docker compose -f deploy/compose.yaml up --build
```

`http://localhost:8000/health` is open (no auth gate). Everything else except
the API schema, the web UI's static shell and the `/` redirect requires the
bearer:

```bash
curl -s -H "Authorization: Bearer $PARTICLES_API_KEY" http://localhost:8000/curation
```

The web UI at `http://localhost:8000/app` is the exception: its static shell
loads with no header so a browser can reach it, and the app then
asks you for the bearer. See [The web UI](#the-web-ui).

To build the image on its own, from the repo root:

```bash
docker build -f deploy/Dockerfile --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" --build-arg VCS_REF="$(git rev-parse --short HEAD)" --build-arg VERSION="$(grep -m1 '^version' pyproject.toml | cut -d'"' -f2)" -t ghcr.io/linkedparticles/engine:dev .
```

The three build args are what let a container answer *what am I, and when was
I made?*; see [Which image is this?](#which-image-is-this). They are
optional: without them the build succeeds and produces an image that declines
to say. The build context is a strict allowlist that does not admit `.git`, so
the revision cannot be discovered during the build; it has to be passed in.

!!! note "The published image, and what still builds locally"
    `ghcr.io/linkedparticles/engine` is a placeholder: the repos are public
    and the name is consistent with the naming decision,
    and the image is published: the public engine
    repository's `publish-image.yml` workflow pushes it after each PyPI
    publish, built from the released distributions. Version tags are
    immutable by policy; `latest` follows the newest release. Locally built
    images (this section) remain the development path and are never pushed
    from here.

## The one volume

Everything the engine persists lives under `/data`, and the image's baked
`/etc/particles/config.yaml` pins each path absolute:

| Path | What it is | Config key |
|---|---|---|
| `/data/particles.db` | The store (SQLite) | `storage.database_url` |
| `/data/corpus_blobs` | Deposited source bytes | `storage.blob_dir` |
| `/data/state` | The consolidation lockfile | `claude_code.state_dir` |

Mount one volume at `/data` and the whole store is backed up, moved, or
inspected as a unit. There is no cwd-relative default left in the image to
mis-fire.

Use a Docker named volume, as the commands on this page do: only the image's
user (uid `10001`) writes it. A host directory bind-mounted at `/data`
(`-v /srv/particles:/data`) brings the host's other accounts back to the
store, the key file and the blobs, so give it a directory only uid `10001` can
write. The engine refuses a key file in a group-writable directory unless
`api.keys_dir_group` names that group, which the Helm chart sets from its
`fsGroup`; leave it unset for a bind mount. [SECURITY.md](https://github.com/LinkedParticles/particles-engine-py/blob/main/SECURITY.md#deployment-posture-authentication-and-the-read-surface)
says which protections that keeps.

## The key is mandatory

The image's entrypoint is `particles engine serve 0.0.0.0:8000 --daemon`.
Binding non-loopback arms the fail-closed gate, so **the
container refuses to start with no key**: no real `PARTICLES_API_KEY` and no
live per-client key in `/data/engine-keys.json`:

```
Refusing to start: bearer auth is disabled (PARTICLES_API_KEY is unset, empty
or 'dev-key') and the bind from api.bind_host, '0.0.0.0', is not a loopback
address …
```

That refusal is a tested property of the image, not a hope: the image cannot
come up keyless on `0.0.0.0` by accident. Set the variable to a real secret,
or mint a per-client key on the volume, which also satisfies the gate. Since
`1.221.7` the compose file leaves `PARTICLES_API_KEY` optional (with neither,
the container restarts on the refusal), and the Helm chart runs without it
when `auth.keysFileOnly` is set. `PARTICLES_API_KEY` is the operator's key, so
an engine run on per-client keys alone holds no shared operator secret.

`ANTHROPIC_API_KEY` is optional. Without it the engine serves every read and
write verb; the LLM-priced work (extraction, semantic lint, the consolidation
cycle's semantic passes) degrades to a **disclosed** structural-only run
rather than failing quietly.

## Reads require the bearer too

The image's baked config sets `api.require_auth_for_reads: true`, so every read
route asks for a key, as the writes do; a key of any role passes, a reader's
included. Only `GET /health`, the API
schema at `GET /openapi.json`, the web UI shell at `/app` and the redirect at
`/`, which carries no data, stay open. Without it, a client that reached the port could
list the store and read the raw deposited bytes (`GET /corpus/blob/{selector}`,
`GET /particles/{id}/source`) with no credential, because the image binds
`0.0.0.0`.

Every client already sends its key on every call: the CLI's remote backend
(`PARTICLES_ENGINE_TOKEN`), the web UI (the token in its settings), and the
Obsidian plugin. A script that read the engine with a bare `curl` now gets a
`401` and needs `-H "Authorization: Bearer <key>"`, a reader key when it only
reads.

**Upgrading.** This changed in 1.210.5. An image from an earlier release served
reads unauthenticated. A config you mount over `/etc/particles/config.yaml`
replaces the baked one, and the image also sets
`PARTICLES_API_REQUIRE_AUTH_FOR_READS=true` in its environment, which wins over
any file, so a mounted config no longer opens the reads by leaving the line out.
Open them only when the network itself is your read-access control (a private
mesh with no other tenants), with `-e PARTICLES_API_REQUIRE_AUTH_FOR_READS=false`.
`particles engine serve`, the image's command, prints a warning whenever it
binds beyond loopback with the reads open, and the engine logs one at startup
for the bind it can read: from `engine serve` or a `uvicorn` command line, and
on the first request of a launch with `--lifespan off`. A script that calls
`uvicorn.run` gives it no bind to read, so there the startup warning follows
`api.bind_host`, and the first request on a non-loopback
address logs it.

## Browser requests from other sites are refused

Every gated route refuses a request that a web page on another site sends, so
a page open in the operator's browser cannot deposit into the engine or start
paid work through it. The browser marks such a request itself
(`Sec-Fetch-Site: cross-site`, or an `Origin` that is not the engine's own), and
a request carrying a valid key is still served. The web UI at `/app` is
same-origin and is unaffected. To let another browser origin call the engine,
list it in `api.allowed_origins`.

A DNS-rebinding page is different: it reaches the engine's address under its
own hostname, and the browser treats it as same-origin. List the names the
engine answers on in `api.allowed_hosts` (for example `engine.example.org` or
`*.tailnet.example`) and every other `Host` is refused on every path. The
image binds `0.0.0.0`, so with the list empty it refuses no `Host` (an engine
bound to loopback answers only the loopback names whatever the list holds).
The loopback names stay allowed, so the image's own health check keeps working; a
Kubernetes probe that sends the pod address as its `Host` needs that address
listed, or a `Host` header set on the probe.

Exposure posture is unchanged by containerization: keep the engine on a private
mesh or a port-forward. A public TLS endpoint is out of
scope, and CORS is not enabled; the web UI is served same-origin from `/app`
precisely so no CORS surface is needed.

## Egress: also block it at the network layer

The engine fetches URLs on request, so every deposit is an outbound connection
it makes on a caller's behalf. In-process that is guarded end to end: every
fetch, through `httpx` or through the `curl` / `git`
subprocesses, resolves the host, checks each address
against a loopback / RFC 1918 / link-local / CGNAT blocklist, and connects to
*that vetted address*, so the validated address is the connected address on
every hop.

That guard lives in the SDK, which is exactly its limit: it is code the engine
runs, not a property of the network the engine runs on. **In a hosted
deployment, add an egress control at the network layer as well**: an egress
firewall or NetworkPolicy allowing only the hosts you actually deposit from,
and, on a cloud instance, disabling the link-local metadata endpoint (or
requiring IMDSv2 with a hop limit of 1). The two controls fail independently:
a defect in one is covered by the other, and only the network-layer one
constrains anything the engine's own process does not mediate.

None of this is needed for the default local-first posture, where the engine is
on loopback and the deposit URLs are the operator's own.

## The web UI

Open `http://localhost:8000/app` in a browser. The app shell (the static
`index.html` + JS + CSS bundle that boots the single-page app) is served
**unauthenticated**, so the page loads with no header.

Everything with data behind it is still gated. On first load the app asks for
the engine bearer; paste `$PARTICLES_API_KEY` (or an operator or reader key of
its own, below) into its settings and the `/curation`, `/query`, and `/graph`
calls it makes from then on carry it.

### Keys for each client

`PARTICLES_API_KEY` is the operator's key. Give every other client a key of its
own, minted inside the container so the key file lands on the data
volume (`api.keys_file: /data/engine-keys.json` in the baked config):

```bash
docker compose exec engine particles engine keys create --role agent --principal mcp:claude-code
docker compose exec engine particles engine keys list
```

`create` writes the new key to a file beside the key file, readable only by
uid `10001` (`/data/<key id>.key`, its path printed), and shows it nowhere
else; hand it to its client and delete the file, or pass `--print-token` to
print it instead.
`docker compose exec` and `kubectl exec` run as the image's user (uid 10001),
which is the engine's own, so the file they write is one the engine accepts.
The verbs refuse to create the file as root. See
[Remote engine → Per-client keys and roles](remote-engine.md#per-client-keys-and-roles)
for the roles and which key each client holds.

This was the other way round until, and it did not work: a browser
navigation cannot send an `Authorization` header, and the settings view where
you would paste the token lives inside the bundle the gate withheld, so with
a real key the UI could not be opened at all. The reversal un-gates the static
shell only; no API path changed.

## Which image is this?

A container you started weeks ago looks exactly like one you started today,
and the question that matters (*is this behind?*) has three answers, in
increasing order of precision.

The **footer of the web UI** names the engine version, and the build date when
the image carries one: `engine 1.129.3 (built 2026-08-08) · web-ui 0.2.1+…`.
It resolves before you have entered a bearer, because `GET /health` is
unauthenticated, so the settings screen can already tell you what you are
pointed at. `engine unreachable` there means the engine is down or the base
URL is wrong, never a rejected token, since none is sent. The version is the
SDK release, so compare it against `CHANGELOG.md` or the tag you expect. (The
trailing `web-ui` string is not a release; it is a hash of the bundle's own
inputs, useful only for spotting a service worker serving a stale bundle.)

The same two facts, without a browser:

```bash
curl -s http://localhost:8000/health
```

The same facts from outside the container, including for an image that is not running:

```bash
docker image inspect ghcr.io/linkedparticles/engine:dev --format '{{json .Config.Labels}}'
```

The standard OCI labels (`org.opencontainers.image.created`, `.revision`,
`.version`) are the precise answer, because `.revision` names the commit.
Two images can share a version and differ in content; they cannot share a
revision and differ. All three come from the build args above, so an image
built without them reports empty labels and `/health` omits `built_at`
entirely rather than guessing.

## What the daemon does

In daemon mode the FastAPI lifespan runs background tasks in the serving
process. Configure them under `daemon` in `config.yaml`:

| Task | Active when | Cadence |
|---|---|---|
| Consolidation tick | always in daemon mode | `daemon.consolidation_tick_minutes` (default 60) |
| Inbox watcher | `inbox.file_path` is set | `inbox.poll_interval_seconds` (default 30) |
| Web-clipper watcher | `daemon.web_clipper_dir` is set | `daemon.web_clipper_poll_minutes` (default 5) |

The tick calls the consolidation operation with `--if-due` semantics, so
`consolidation.min_interval_hours` (default 20) remains the real cadence and
ticking hourly is harmless. **Pending extraction rides consolidation pass 1**,
exactly as it does under cron; there is deliberately no second extract-drain
loop, because a second periodic writer is the shape daemon mode exists to
remove.

Both watchers are mtime-polls. There is no `watchdog` or FSEvents dependency;
rejection of filesystem-event watchers stands, as does its
decision that mutable-source refresh remains a consolidation pass. To use the
web-clipper watcher, mount your captures directory read-only and point
`daemon.web_clipper_dir` at it.

### Watching the daemon

`GET /health` is both the liveness and the readiness probe, and in daemon mode
it discloses the tasks. With reads closed, as the image ships them, it shows
them only to a request that carries a key the engine accepts, so a request
with no key (the image's own health check among them) gets the status without
the `daemon` block:

```json
{
  "status": "degraded",
  "version": "1.122.0",
  "daemon": {
    "enabled": true,
    "healthy": false,
    "tasks": [
      {
        "name": "consolidation",
        "interval_seconds": 3600.0,
        "state": "crashed",
        "runs": 4,
        "failures": 1,
        "last_outcome": "failed",
        "last_error": "OperationalError (message in the engine log)"
      }
    ]
  }
}
```

A failing *iteration* is caught, logged with its traceback, and counted in
`failures`; the task keeps its cadence, because every task here is
level-triggered (the corpus, the inbox file, the captures directory are the
state, so the next tick simply sees the same work again). A task that dies
outright is marked `crashed` and flips `status` to `degraded`. `last_error`
names the exception type only: `/health` carries no auth gate, and an error's
text can quote a database statement or a fetched URL, so the message stays in
the engine log.

`/health` deliberately stays **200** in that case: the API is still serving
requests, and taking the whole engine down because a scheduled tick died would
turn a background problem into an outage. Alert on the body; restart on the
connection.

## Locking

The consolidation lockfile is retained; it still guards
a host-side `particles memory consolidate` colliding with a daemon over a
shared mount. The lock is a kernel advisory lock, and a kernel lock
taken inside a container's VM (Docker Desktop's file sharing, for example) is
not always visible on the host. The lock file therefore also records the
holder's host and a `heartbeat_at` the holder refreshes every
`consolidation.lock_heartbeat_seconds` (default 60) from a background thread.
A caller on another host treats the lock as held while that heartbeat is
younger than `consolidation.lock_heartbeat_stale_minutes` (default 10), so a
long batch wait inside the container is never reclaimed, and a crashed
container frees the lock within ten minutes. The daemon serializes its own
passes in-process, and the cross-process write lock
continues to referee every writer.

Upgrade the image and any host-side venv together. A pre-change binary takes
no kernel lock and still reclaims a lock older than
`consolidation.lock_timeout_minutes`, even from a live cycle.

Run one engine per store. `replicas: 1` is load-bearing: SQLite has one writer.

## Consolidation pass 6 is off (and says so)

The image's baked config sets `agent_memory.projection.enabled: false`.
Pass 6 (agent-memory projection) renders `MEMORY.md` into host-coupled paths
(`~/.claude/projects/*/memory`) that do not exist in a container. The pass
**discloses the skip** in every run record; it is never silently absent.

To opt in, mount the projection targets into the container and re-enable the
pass in a mounted config:

```yaml
# my-config.yaml, mounted over /etc/particles/config.yaml
agent_memory:
  projection:
    enabled: true
```

```bash
docker run --rm \
  -e PARTICLES_API_KEY \
  -v particles-data:/data \
  -v "$HOME/.claude/projects:/home/particles/.claude/projects" \
  -v "$PWD/my-config.yaml:/etc/particles/config.yaml:ro" \
  -p 127.0.0.1:8000:8000 \
  ghcr.io/linkedparticles/engine:dev
```

## Overriding configuration

Two supported routes, in precedence order:

1. **Mount your own file over `/etc/particles/config.yaml`.** `PARTICLES_CONFIG`
   already points there, so nothing else changes. Copy `config.yaml.sample` as
   the starting point and keep the absolute `/data` pins.
2. **Set a registered env override**: `DATABASE_URL`, `PARTICLES_BLOB_DIR`,
   `PARTICLES_DAEMON_ENABLED`, `PARTICLES_DAEMON_STORE`,
   `PARTICLES_DAEMON_WEB_CLIPPER_DIR`, and the rest of `_ENV_OVERRIDES` in
   `particles/config.py`. Env wins over the file.

Secrets are never read from `config.yaml`; they come from the environment
(`PARTICLES_API_KEY`, `ANTHROPIC_API_KEY`, …). Do not put them in a mounted
config.

## What is in the image

- Python 3.11-slim, the wheel, and its locked dependency closure
  (`uv sync --frozen --no-dev`) in an image you build from a checkout. The
  published image installs the released pair from PyPI instead, with its
  dependencies resolved at build time under a date cutoff and not
  hash-locked; its SBOM lists what was resolved.
- The committed web-UI bundle, so `/app` always registers; the wheel excludes
  it, the image must not.
- **The embedding encoder, baked at build time** (`all-MiniLM-L6-v2`).
  This resolves the question routed here in favour of
  baking: a predictable cold start, air-gap friendliness, and the observation
  that an image is pulled far less often than a daemon restarts. There is **no
  slim/no-encoder variant** in v1; it was rejected until a measured size or
  cold-start need exists.

!!! info "Image size: 1.8 GB on disk, 400 MB compressed"

    Measured 2026-08-05 (`linux/arm64`). Most of that is `torch`, which
    `sentence-transformers` needs to run the encoder; the encoder weights
    themselves are ~90 MB.

    The image was **8.5 GB / 3.1 GB compressed** before the slimming work: PyPI's
    default Linux torch wheel declares the entire CUDA runtime as hard
    dependencies (43 `nvidia-*` packages plus `triton`), and this SDK never
    touches a GPU; torch is used only for `all-MiniLM-L6-v2` inference and no
    code path imports it directly. `pyproject.toml` now routes torch to
    PyTorch's CPU-only wheel index on every platform except macOS (whose PyPI
    wheel is already CPU-only), so `torch.cuda.is_available()` is `False` in
    the image by construction rather than by luck.

    Published images up to and including `v1.143.0` are the exception: the
    publication build installs the released wheel from PyPI, where
    `pyproject.toml`'s torch routing does not apply, so those tags carry the
    CUDA build at **~3.3 GB compressed**. Tags are immutable, so they stay that
    way; `v1.143.1` and later install torch from the CPU index too,
    and the publish workflow now fails any image that carries a CUDA torch
    build or an `nvidia-*` package.

- **An SBOM per platform image**, generated after each publish by scanning
  the pushed image, then signed and pushed to the registry beside it (see
  [Quick start](#quick-start-docker-run) for the verify command). A scanner
  cannot see model weights, so the encoder is added explicitly as a
  `machine-learning-model` component with its Hugging Face commit. The code
  pins that commit. Before it pushes the version and `latest` tags, the image
  workflow copies the encoder cache out of a container it never starts and
  hashes every file under the pinned snapshot against the commit's recorded
  file digests, failing on any other file, snapshot or cache of the encoder,
  and it refuses to sign an SBOM whose encoder is not the pin. Images built
  by the workflow of an earlier release were checked less strictly (the
  changelog says which); for an image before `1.215.1` the SBOM entry is the
  only record of which weights it baked. A workflow change reaches the public
  repository with the next release exported to it.
- The Alembic migrations, which the wheel now force-includes under
  `particles/_alembic`, so the first boot creates a *stamped* schema
  and later `alembic upgrade head` works.

Postgres and multi-replica are deferred: `asyncpg` is not a dependency today
and neither the image nor the chart pretends otherwise.

## Kubernetes

The minimal chart lives in `deploy/helm/`: a single-replica **StatefulSet**
with a PVC for `/data`, a ClusterIP Service, and **no Ingress**:

```bash
kubectl create secret generic particles-auth --from-literal=PARTICLES_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

```bash
helm install particles ./deploy/helm --set auth.existingSecret=particles-auth
```

`replicas: 1` is load-bearing and the chart says so where it is set: one SQLite
writer, and the daemon schedules consolidation *inside* the serving process, so
a second replica is a second scheduler on the same volume. The chart refuses to
render at all without a key configured, rather than letting you discover the
refusal as a CrashLoopBackOff. Postgres and multi-replica are deferred;
`asyncpg` is not a dependency today and the chart does not pretend
otherwise.

Full values table and rationale: [`deploy/helm/README.md`](https://github.com/LinkedParticles/particles-engine-py/blob/main/deploy/helm/README.md).

[adr122]: ../ADR/active/0122-validating-fetch-transport.md
[adr123]: ../ADR/active/0123-fail-closed-bearer-auth.md
[adr261]: ../ADR/active/0261-container-image-publication.md
[adr250]: ../ADR/active/0250-subprocess-fetch-connect-pinning.md
[adr137]: ../ADR/active/0137-remote-engine-thin-client.md
[adr177]: ../ADR/active/0177-single-writer-discipline-cross-process-write-lock.md
[adr194]: ../ADR/active/0194-dream-cycle-scheduled-consolidation.md
[adr206]: ../ADR/active/0206-mutable-local-source-refresh.md
[adr241]: ../ADR/active/0241-zero-friction-adoption-bundle.md
[adr157]: ../ADR/active/0157-naming-branding-and-schema-id-host.md

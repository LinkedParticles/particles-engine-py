# Running the engine in a container

Running Particles as a service used to take four hand-assembled parts: the
engine, a loop over `extract --all-pending`, `inbox watch`, and a launchd or
cron entry for `memory consolidate --if-due`. The consolidation design chose that
shape on purpose — external scheduler, no resident process — and its own §2
Correction then wrote down what it costs: a LaunchAgent inherits no working
directory and almost no environment, so a mis-pinned job runs "successful"
cycles against an empty store and scatters blob directories somewhere nobody
looks.

A container has no launchd and no cron, so this page is where that changes.
The engine grows an opt-in **resident daemon mode**, and the image bakes every
store-adjacent path absolute under one volume — which closes that failure class
by construction rather than by operator discipline.

**The launchd/cron recipes are still correct** for hosts that do not run a
container. Nothing on this page deprecates them; see
[Scheduled consolidation](scheduled-consolidation.md).

## Quick start (compose)

```bash
export PARTICLES_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

```bash
docker compose -f deploy/compose.yaml up --build
```

`http://localhost:8000/health` is open (no auth gate). Everything else,
**including the web UI's app shell**, requires the bearer:

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
I made?* — see [Which image is this?](#which-image-is-this). They are
optional: without them the build succeeds and produces an image that declines
to say. The build context is a strict allowlist that does not admit `.git`, so
the revision cannot be discovered during the build; it has to be passed in.

!!! note "Provisional names, local builds only"
    `ghcr.io/linkedparticles/engine` is the owner's placeholder while
    The naming decision (distribution name + schema `$id` host) is blocked on
    the repos going public. Final names ride that sign-off. Until publication
    these images are built and used **locally or in CI and never pushed to a
    public registry** — the same internal-marker rule applies to
    version tags.

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

## The key is mandatory

The image's entrypoint is `particles engine serve 0.0.0.0:8000 --daemon`.
Binding non-loopback arms the fail-closed gate, so **the
container refuses to start without a real `PARTICLES_API_KEY`**:

```
Refusing to start: bearer auth is disabled (PARTICLES_API_KEY is unset or
'dev-key') and api.bind_host='0.0.0.0' is not a loopback address …
```

That refusal is a tested property of the image, not a hope — the image cannot
come up open by accident. Set the variable to a real secret (compose's
`PARTICLES_API_KEY:?…` fails the `up` with the same message rather than letting
the container crash-loop).

`ANTHROPIC_API_KEY` is optional. Without it the engine serves every read and
write verb; the LLM-priced work — extraction, semantic lint, the consolidation
cycle's semantic passes — degrades to a **disclosed** structural-only run
rather than failing quietly.

Exposure posture is unchanged by containerization: keep the engine on a private
mesh or a port-forward. A public TLS endpoint is out of
scope, and CORS is not enabled — the web UI is served same-origin from `/app`
precisely so no CORS surface is needed.

## Egress: also block it at the network layer

The engine fetches URLs on request, so every deposit is an outbound connection
it makes on a caller's behalf. In-process that is guarded end to end: every
fetch — through `httpx` or through the `curl` / `git`
subprocesses — resolves the host, checks each address
against a loopback / RFC 1918 / link-local / CGNAT blocklist, and connects to
*that vetted address*, so the validated address is the connected address on
every hop.

That guard lives in the SDK, which is exactly its limit: it is code the engine
runs, not a property of the network the engine runs on. **In a hosted
deployment, add an egress control at the network layer as well** — an egress
firewall or NetworkPolicy allowing only the hosts you actually deposit from,
and, on a cloud instance, disabling the link-local metadata endpoint (or
requiring IMDSv2 with a hop limit of 1). The two controls fail independently:
a defect in one is covered by the other, and only the network-layer one
constrains anything the engine's own process does not mediate.

None of this is needed for the default local-first posture, where the engine is
on loopback and the deposit URLs are the operator's own.

## The web UI

Open `http://localhost:8000/app` in a browser. The app shell — the static
`index.html` + JS + CSS bundle that boots the single-page app — is served
**unauthenticated**, so the page loads with no header.

Everything with data behind it is still gated. On first load the app asks for
the engine bearer; paste `$PARTICLES_API_KEY` into its settings and the
`/curation`, `/query`, and `/graph` calls it makes from then on carry it.

This was the other way round until, and it did not work: a browser
navigation cannot send an `Authorization` header, and the settings view where
you would paste the token lives inside the bundle the gate withheld — so with
a real key the UI could not be opened at all. The reversal un-gates the static
shell only; no API path changed.

## Which image is this?

A container you started weeks ago looks exactly like one you started today,
and the question that matters — *is this behind?* — has three answers, in
increasing order of precision.

The **footer of the web UI** names the engine version, and the build date when
the image carries one: `engine 1.129.3 (built 2026-08-08) · web-ui 0.2.1+…`.
It resolves before you have entered a bearer, because `GET /health` is
unauthenticated, so the settings screen can already tell you what you are
pointed at. `engine unreachable` there means the engine is down or the base
URL is wrong — never a rejected token, since none is sent. The version is the
SDK release, so compare it against `CHANGELOG.md` or the tag you expect. (The
trailing `web-ui` string is not a release — it is a hash of the bundle's own
inputs, useful only for spotting a service worker serving a stale bundle.)

The same two facts, without a browser:

```bash
curl -s http://localhost:8000/health
```

And from outside the container, including for an image that is not running:

```bash
docker image inspect ghcr.io/linkedparticles/engine:dev --format '{{json .Config.Labels}}'
```

The standard OCI labels — `org.opencontainers.image.created`, `.revision`,
`.version` — are the precise answer, because `.revision` names the commit.
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
exactly as it does under cron — there is deliberately no second extract-drain
loop, because a second periodic writer is the shape daemon mode exists to
remove.

Both watchers are mtime-polls. There is no `watchdog` or FSEvents dependency;
rejection of filesystem-event watchers stands, as does its
decision that mutable-source refresh remains a consolidation pass. To use the
web-clipper watcher, mount your captures directory read-only and point
`daemon.web_clipper_dir` at it.

### Watching the daemon

`GET /health` is both the liveness and the readiness probe, and in daemon mode
it discloses the tasks:

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
        "last_error": "OperationalError: database is locked"
      }
    ]
  }
}
```

A failing *iteration* is caught, logged with its traceback, and counted in
`failures` — the task keeps its cadence, because every task here is
level-triggered (the corpus, the inbox file, the captures directory are the
state, so the next tick simply sees the same work again). A task that dies
outright is marked `crashed` and flips `status` to `degraded`.

`/health` deliberately stays **200** in that case: the API is still serving
requests, and taking the whole engine down because a scheduled tick died would
turn a background problem into an outage. Alert on the body; restart on the
connection.

## Locking

The consolidation lockfile is retained — it still guards
a host-side `particles memory consolidate` colliding with a daemon over a
shared mount. Inside a container its `os.kill(pid, 0)` stale-reclaim is
meaningless (pids are namespaced), so **`consolidation.lock_timeout_minutes` is
the reclaim authority there**: a lock older than that is reclaimed regardless of
what its recorded pid appears to be. The daemon serializes its own passes
in-process, and the cross-process write lock continues to
referee every writer.

Run one engine per store. `replicas: 1` is load-bearing — SQLite has one writer.

## Consolidation pass 6 is off (and says so)

The image's baked config sets `agent_memory.projection.enabled: false`.
Pass 6 (agent-memory projection) renders `MEMORY.md` into host-coupled paths
(`~/.claude/projects/*/memory`) that do not exist in a container. The pass
**discloses the skip** in every run record — it is never silently absent.

To opt in, mount the projection targets into the container and re-enable the
pass in a mounted config:

```yaml
# my-config.yaml — mounted over /etc/particles/config.yaml
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
2. **Set a registered env override** — `DATABASE_URL`, `PARTICLES_BLOB_DIR`,
   `PARTICLES_DAEMON_ENABLED`, `PARTICLES_DAEMON_STORE`,
   `PARTICLES_DAEMON_WEB_CLIPPER_DIR`, and the rest of `_ENV_OVERRIDES` in
   `particles/config.py`. Env wins over the file.

Secrets are never read from `config.yaml` — they come from the environment
(`PARTICLES_API_KEY`, `ANTHROPIC_API_KEY`, …). Do not put them in a mounted
config.

## What is in the image

- Python 3.11-slim, the wheel, and its locked dependency closure
  (`uv sync --frozen --no-dev`).
- The committed web-UI bundle, so `/app` always registers — the wheel excludes
  it, the image must not.
- **The embedding encoder, baked at build time** (`all-MiniLM-L6-v2`).
  This resolves the question routed here in favour of
  baking: a predictable cold start, air-gap friendliness, and the observation
  that an image is pulled far less often than a daemon restarts. There is **no
  slim/no-encoder variant** in v1; it was rejected until a measured size or
  cold-start need exists.

!!! info "Image size — 1.8 GB on disk, 400 MB compressed"

    Measured 2026-08-05 (`linux/arm64`). Most of that is `torch`, which
    `sentence-transformers` needs to run the encoder; the encoder weights
    themselves are ~90 MB.

    The image was **8.5 GB / 3.1 GB compressed** before the slimming work: PyPI's
    default Linux torch wheel declares the entire CUDA runtime as hard
    dependencies (43 `nvidia-*` packages plus `triton`), and this SDK never
    touches a GPU — torch is used only for `all-MiniLM-L6-v2` inference and no
    code path imports it directly. `pyproject.toml` now routes torch to
    PyTorch's CPU-only wheel index on every platform except macOS (whose PyPI
    wheel is already CPU-only), so `torch.cuda.is_available()` is `False` in
    the image by construction rather than by luck.

- The Alembic migrations, which the wheel now force-includes under
  `particles/_alembic`, so the first boot creates a *stamped* schema
  and later `alembic upgrade head` works.

Postgres and multi-replica are deferred: `asyncpg` is not a dependency today
and neither the image nor the chart pretends otherwise.

## Kubernetes

The minimal chart lives in `deploy/helm/` — a single-replica **StatefulSet**
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
render at all without a key configured, rather than letting you discover the refusal as a CrashLoopBackOff. Postgres and multi-replica are deferred
; `asyncpg` is not a dependency today and the chart does not pretend
otherwise.

Full values table and rationale: [`deploy/helm/README.md`](https://github.com/LinkedParticles/particles-engine-py/blob/main/deploy/helm/README.md).

[adr122]: ../ADR/active/0122-validating-fetch-transport.md
[adr123]: ../ADR/active/0123-fail-closed-bearer-auth.md
[adr250]: ../ADR/active/0250-subprocess-fetch-connect-pinning.md
[adr137]: ../ADR/active/0137-remote-engine-thin-client.md
[adr177]: ../ADR/active/0177-single-writer-discipline-cross-process-write-lock.md
[adr194]: ../ADR/active/0194-dream-cycle-scheduled-consolidation.md
[adr206]: ../ADR/active/0206-mutable-local-source-refresh.md
[adr241]: ../ADR/active/0241-zero-friction-adoption-bundle.md
[adr157]: ../ADR/active/0157-naming-branding-and-schema-id-host.md

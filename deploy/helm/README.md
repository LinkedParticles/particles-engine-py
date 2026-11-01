# linkedparticles-engine

A deliberately minimal Helm chart for the Particles engine: a
**single-replica StatefulSet** with a PVC for `/data`, a ClusterIP Service, and
no Ingress.

## Install

```bash
helm install particles ./deploy/helm --set auth.apiKey="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

Better, with a Secret you manage:

```bash
kubectl create secret generic particles-auth --from-literal=PARTICLES_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

```bash
helm install particles ./deploy/helm --set auth.existingSecret=particles-auth
```

Reach it (there is no Ingress by design):

```bash
kubectl port-forward svc/particles-linkedparticles-engine 8000:8000
```

## `replicas: 1` is load-bearing

Do not raise it. The store is SQLite, which has exactly one writer, and the
daemon schedules the consolidation cycle *inside* the serving process.
Two replicas would be two schedulers and two writers against one
PersistentVolume — precisely the multi-writer shape the design exists to
remove. Multi-replica/HA rides a Postgres backend that is **not a dependency
today**; it is deferred and this chart does not pretend otherwise.

The chart also sets `podManagementPolicy: OrderedReady` so a rolling update
never briefly runs two pods against the same volume.

Keep `/data` on the chart's PVC, which the pod writes as uid `10001` with
`fsGroup` `10001`. A `hostPath` volume, or a
storage class backed by a host directory, brings the node's other accounts
back to the store, the key file and the blobs, so give it a directory only
uid `10001` can write. SECURITY.md (*Deployment posture*) says which
protections that keeps.

## The key is mandatory

The image binds `0.0.0.0`, which arms the fail-closed gate: with no
real `PARTICLES_API_KEY` and no live per-client key in `/data/engine-keys.json`
the process refuses to start and the pod CrashLoopBackOffs on that refusal, by
design, so a cluster deploy cannot come up keyless by accident. The chart
fails at *template* time rather than at runtime if neither
`auth.existingSecret` nor `auth.apiKey` is set, unless `auth.keysFileOnly` says
the engine runs on per-client keys alone. An empty or whitespace-only key
counts as no key.

`PARTICLES_API_KEY` is the operator's key: any holder can make every write,
including the operator's, and name the provenance a deposit is recorded with.
Give it only to clients you would give the operator's authority. Give every
other client a key of its own, with the role it needs (`reader`, `agent` or
`operator`), minted inside the pod so the key file lands on the volume:

```bash
kubectl exec -it statefulset/particles-linkedparticles-engine -- particles engine keys create --role reader --principal viewer
```

A reader key writes nothing, and an agent key writes only as its own
principal. See the engine's remote-engine guide for the roles and which key
each client holds.

`ANTHROPIC_API_KEY` is optional; without it the LLM-priced work degrades to a
disclosed structural-only run.

## Reads require the bearer

The image sets `api.require_auth_for_reads: true` in its baked config and
sets `PARTICLES_API_REQUIRE_AUTH_FOR_READS=true` in its environment, so every
route except `/health`, `/openapi.json`, the `/app` shell and the `/`
redirect asks for a key (any role, a reader's included). A config you mount with `config.create` or
`config.existingConfigMap` replaces the baked file, but the pod inherits the
image's environment, which outranks any file: a mounted config that leaves the
line out does not open the reads. Only an explicit override in `env` does:

```yaml
env:
  PARTICLES_API_REQUIRE_AUTH_FOR_READS: "false"   # opens every read route
```

Set that only when the network itself is your read-access control.

Releases before 1.210.5 served reads unauthenticated; see the container
deployment guide's upgrade note.

## No Ingress

Exposure posture is a private mesh or `kubectl port-forward`.
There is no Ingress template at all — a public TLS endpoint stays out of scope
until the hosted-deployment SSRF closure and the observability
hardening land. Front this with your own ingress and authentication
only if you have made that call yourself.

## Values worth knowing

| Key | Default | Notes |
|---|---|---|
| `replicaCount` | `1` | Load-bearing. See above. |
| `image.repository` | `ghcr.io/linkedparticles/engine` | Placeholder name pending the naming decision; images are built locally/in CI and never pushed to a public registry before publication. Point this at your own registry. |
| `auth.existingSecret` | `""` | A Secret you manage, carrying `PARTICLES_API_KEY` (and optionally `ANTHROPIC_API_KEY`). |
| `auth.apiKey` | `""` | A literal the chart renders into its own Secret. Fine for `--set`; never commit it. |
| `auth.keysFileOnly` | `false` | Run on per-client keys alone: `PARTICLES_API_KEY` is not required (optional in `existingSecret`). The pod starts only once a live key is in `/data/engine-keys.json`. |
| `persistence.size` | `20Gi` | The PVC for `/data` — store, corpus blobs, and the consolidation lockfile. |
| `config.create` / `config.content` | `false` / `""` | Mount your own `config.yaml` over the image's baked one. Keep the absolute `/data` pins. The reads stay closed whatever the file says, since the image's environment sets `PARTICLES_API_REQUIRE_AUTH_FOR_READS=true`; only `env` opens them. |
| `env` | `{}` | Registered config overrides, e.g. `PARTICLES_DAEMON_STORE`. |

## Probes

All three probes hit `/health`, which carries no auth gate. In daemon mode the
response also reports the background tasks, and `status` flips to `degraded`
once one has crashed — but it stays a **200**, so a dead consolidation tick
never restarts a pod that is still serving requests. Alert on the body; restart
on the connection. The `startupProbe` is generous because the first boot runs
`alembic upgrade head` and loads the baked embedding encoder.

More context: [Running in a container](../../docs/operator-guide/container-deployment.md).

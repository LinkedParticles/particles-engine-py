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

Do not raise it. The store is SQLite, which has exactly one writer, and the daemon schedules the consolidation cycle *inside* the serving process.
Two replicas would be two schedulers and two writers against one
PersistentVolume — precisely the multi-writer shape the design exists to
remove. Multi-replica/HA rides a Postgres backend that is **not a dependency
today**; it is deferred and this chart does not pretend otherwise.

The chart also sets `podManagementPolicy: OrderedReady` so a rolling update
never briefly runs two pods against the same volume.

## The key is mandatory

The image binds `0.0.0.0`, which arms the fail-closed gate: without a
real `PARTICLES_API_KEY` the process refuses to start and the pod
CrashLoopBackOffs on that refusal — by design, so a cluster deploy cannot come
up unauthenticated by accident. The chart fails at *template* time rather than
at runtime if neither `auth.existingSecret` nor `auth.apiKey` is set.

`ANTHROPIC_API_KEY` is optional; without it the LLM-priced work degrades to a
disclosed structural-only run.

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
| `persistence.size` | `20Gi` | The PVC for `/data` — store, corpus blobs, and the consolidation lockfile. |
| `config.create` / `config.content` | `false` / `""` | Mount your own `config.yaml` over the image's baked one. Keep the absolute `/data` pins. |
| `env` | `{}` | Registered config overrides, e.g. `PARTICLES_DAEMON_STORE`. |

## Probes

All three probes hit `/health`, which carries no auth gate. In daemon mode the
response also reports the background tasks, and `status` flips to `degraded`
once one has crashed — but it stays a **200**, so a dead consolidation tick
never restarts a pod that is still serving requests. Alert on the body; restart
on the connection. The `startupProbe` is generous because the first boot runs
`alembic upgrade head` and loads the baked embedding encoder.

More context: [Running in a container](../../docs/operator-guide/container-deployment.md).

# Observability (OpenTelemetry)

Particles can emit **traces, metrics, and trace-correlated logs** via
[OpenTelemetry](https://opentelemetry.io/). It is **off by default**
and is an **opt-in optional dependency** — a stock install ships only the no-op
OTel API and pays nothing.

The motivating case: on the client–server topology (the
[remote engine](remote-engine.md)) a request runs in a different
process from the CLI that issued it. When a client `deposit` hangs while the
engine is busy, the engine logs nothing until the request *finishes* — so a hung
request is invisible. With observability on, the request is a `traceparent`-
propagated **span tree** that shows exactly where its time went — URL fetch vs
write-lock wait vs blob write vs LLM call — across the process boundary.

## Turn it on

1. **Install the extra** (the base install does not include the OTel SDK):

   ```bash
   pip install 'particles[otel]'      # or: uv sync --extra otel
   ```

2. **Enable it in `config.yaml`:**

   ```yaml
   observability:
     enabled: true
     exporter: console          # print spans/metrics to the log (zero infra)
     # exporter: otlp           # or ship to a collector / SaaS backend
     # endpoint: "http://localhost:4318"
     service_name: "particles"
     traces: true
     metrics: true
     logs: true
     sample_ratio: 1.0          # always-on (correct for single-operator volume)
   ```

   Or per-process, without editing the file:

   ```bash
   export PARTICLES_OBSERVABILITY_ENABLED=true
   export PARTICLES_OBSERVABILITY_EXPORTER=console
   ```

The bootstrap runs at every process entry point — the CLI, the MCP server, and
the FastAPI engine (`particles engine serve` wires the FastAPI server span). With
`enabled: false` **or** the `otel` extra absent, every span/metric call is a
cheap no-op.

## Exporters

`exporter` selects where signals go — one OTLP code path, three targets:

| `exporter` | Where it goes | Use |
|---|---|---|
| `console` | printed to the log | zero-infra, diagnose-now (resolves the hung-request case) |
| `otlp` | `endpoint` (OTLP/HTTP) | a local [collector](https://opentelemetry.io/docs/collector/) (`http://localhost:4318`) or a SaaS backend |
| `none` | nowhere (provider only) | tests / pure no-op |

The collector and SaaS cases are the **same** `otlp` setting pointed at a
different `endpoint`.

## The exporter credential is a secret

`endpoint` (a URL) is non-secret and lives in `config.yaml`. The credential an
authenticated collector or SaaS backend requires is a **secret** and is read
from the environment — never put it in `config.yaml`:

```bash
export PARTICLES_OTEL_EXPORTER_HEADERS="authorization=Bearer <token>"
```

## What you get

- **Traces** — auto-instrumented FastAPI (server spans), httpx (client spans +
  `traceparent` propagation), and SQLAlchemy/aiosqlite (DB spans, including the
  write-lock wait), plus hand-rolled `extract.snapshot` → `embed.batch` /
  `llm.complete` spans on the extraction path.
- **Metrics** — LLM- and embed-call duration histograms, the
  `particles.extracted` throughput counter, and the **`particles.sqlite.busy`**
  counter — incremented on every `database is locked` at the DB boundary, so it
  measures cross-process write-lock contention (a direct-I/O CLI verb on the
  engine host vs. the always-on engine) regardless of which writer lost the lock
  — plus the engine's per-request `http.server` duration from the FastAPI
  instrumentation.
- **Logs** — the existing stdlib logs, with the active trace/span ID injected so
  a log line ties back to its span.

## Grafana dashboard

A ready-made dashboard ships alongside this page:
[`particles-dashboard.json`](particles-dashboard.json). It covers HTTP RED
(request rate, errors, and latency by route), the **Particles-internal metrics**
(extraction throughput, SQLite write-lock contention, LLM + embedding call
latency), outbound httpx calls, a recent-traces table, log volume + a live log
panel, and host CPU / memory / load / network / disk.

**Import it:** Grafana → Dashboards → New → Import → upload the JSON, then pick
your Prometheus, Tempo, and Loki data sources when prompted — the dashboard uses
data source *variables*, so it adapts to your setup rather than hard-coding UIDs.

**Assumptions.** The metric panels expect OTLP metrics to reach **Prometheus via
a collector** (Grafana Alloy or the OTel Collector) that renders them in the
Prometheus idiom — `http_server_duration_milliseconds_*`,
`particles_sqlite_busy_total`, `particles_llm_duration_seconds_*`, … — and maps
`service.name` onto the `job` label (so `job="particles"`, the default
`service_name`). A pipeline that keeps dotted OTLP names, or exports straight to
a non-Prometheus backend, will name things differently and the queries won't
match. Host panels read `node_exporter` (`node_*`) metrics.

**What lights up when.** The HTTP-server and Particles-internal metrics come from
the **engine** (`particles engine serve`) and its extraction / LLM paths; the CLI
and MCP server emit traces and logs but not those metrics. OTLP metrics export
every 60 s by default — set `OTEL_METRIC_EXPORT_INTERVAL` (milliseconds) lower if
you want the `rate()` panels to fill in sooner. The live-logs panel needs a logs
pipeline into Loki; since OTLP log *export* is deferred (above), that means
tailing the engine's log file into Loki today.

For network exposure of the engine itself, see
[Remote engine](remote-engine.md); observability rides whatever channel
(Tailscale / SSH tunnel) that uses.

"""OpenTelemetry SDK bootstrap.

This is the **only** module that touches the OTel SDK / exporters /
auto-instrumentation. The rest of the codebase emits spans and metrics through
the no-op-by-default OTel **API** (``opentelemetry-api``, a base dependency);
this module — gated by ``observability.enabled`` and the optional ``otel``
extra — installs the provider that turns those no-ops into real telemetry.

Layering: this package is **Engine**. It reaches only *downward*
into ``particles.config`` / ``particles.secrets`` (both Client) and the
third-party OTel SDK. Client modules must never import it; their hand-rolled
spans use the OTel API directly. The Surface entry points (CLI ``run()``, MCP
``main()``, the FastAPI app / ``engine serve``) call ``setup_observability()``
once at process bootstrap.

The SDK imports are **deferred inside the functions** (AGENTS.md § Deferred
imports case 2 — lazy-init of an optional resource): with the ``otel`` extra
absent they raise ``ImportError``, which is caught and degraded to a logged
no-op, so a stock install that left ``observability.enabled`` on by mistake does
not crash. mypy treats the optional submodules as untyped (``follow_imports =
"skip"`` in ``pyproject.toml``); they appear here as ``Any``.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from particles.config import get_config
from particles.secrets import get_otel_exporter_headers_optional

if TYPE_CHECKING:
    from fastapi import FastAPI

    from particles.config import ObservabilityConfig

log = logging.getLogger(__name__)

# One bootstrap per process. The lock makes the idempotency thread-safe (the MCP
# server and the FastAPI engine both run multi-threaded).
_lock = threading.Lock()
_configured = False
_fastapi_instrumented = False
# Providers retained for the test-reset seam. Flush-on-exit is handled by the
# SDK itself (TracerProvider / MeterProvider register their own atexit shutdown,
# ``shutdown_on_exit=True``), so a short-lived CLI's spans export before exit
# without any hook here.
_providers: list[Any] = []


def setup_observability() -> None:
    """Install the OTel providers + global instrumentation, once per process.

    Idempotent and thread-safe: the first call with ``observability.enabled``
    true wires everything; later calls return immediately. A no-op when
    observability is disabled or the ``otel`` extra is not installed (the latter
    logs a one-line warning so a misconfiguration is visible).
    """
    global _configured
    with _lock:
        if _configured:
            return
        cfg = get_config().observability
        if not cfg.enabled:
            return
        try:
            _install(cfg)
        except ImportError:
            log.warning(
                "observability.enabled is true but the 'otel' extra is not "
                "installed; telemetry is disabled. Install it with "
                "`pip install particles[otel]`."
            )
            return
        _configured = True
        log.info(
            "observability enabled (exporter=%s, service=%s, sample_ratio=%s)",
            cfg.exporter,
            cfg.service_name,
            cfg.sample_ratio,
        )


def instrument_fastapi_app(app: FastAPI) -> None:
    """Add the FastAPI server-span middleware to ``app`` (called from ``engine serve``).

    Ensures the providers are installed (via :func:`setup_observability`), then
    instruments the app instance. Must run **before** the app starts serving (the
    instrumentor adds middleware, which Starlette forbids after startup) — hence
    the ``engine serve`` call site, before ``uvicorn.run``. Idempotent; a no-op
    when observability or traces are disabled or the extra is absent.
    """
    setup_observability()
    global _fastapi_instrumented
    with _lock:
        if _fastapi_instrumented:
            return
        cfg = get_config().observability
        if not cfg.enabled or not cfg.traces:
            return
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        except ImportError:
            return
        FastAPIInstrumentor.instrument_app(app)
        _fastapi_instrumented = True


def _install(cfg: ObservabilityConfig) -> None:
    """Wire providers + instrumentors. Raises ``ImportError`` if the extra is absent."""
    from opentelemetry.sdk.resources import Resource

    resource = Resource.create({"service.name": cfg.service_name})
    if cfg.traces:
        _install_traces(cfg, resource)
    if cfg.metrics:
        _install_metrics(cfg, resource)
    if cfg.logs:
        _install_logging()
    _install_instrumentors()


def _install_traces(cfg: ObservabilityConfig, resource: Any) -> None:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

    sampler = ParentBased(TraceIdRatioBased(cfg.sample_ratio))
    provider = TracerProvider(resource=resource, sampler=sampler)

    if cfg.exporter == "console":
        # Synchronous export so the diagnose-now CLI prints each span as it closes.
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif cfg.exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        otlp = OTLPSpanExporter(**_otlp_kwargs(cfg, "traces"))
        provider.add_span_processor(BatchSpanProcessor(otlp))

    trace.set_tracer_provider(provider)
    _providers.append(provider)


def _install_metrics(cfg: ObservabilityConfig, resource: Any) -> None:
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )

    readers: list[Any] = []
    if cfg.exporter == "console":
        readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter()))
    elif cfg.exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

        readers.append(
            PeriodicExportingMetricReader(OTLPMetricExporter(**_otlp_kwargs(cfg, "metrics")))
        )

    provider = MeterProvider(resource=resource, metric_readers=readers)
    metrics.set_meter_provider(provider)
    _providers.append(provider)


def _install_logging() -> None:
    """Correlate stdlib logs to the active trace/span ID (Phase 1).

    Injects ``otelTraceID`` / ``otelSpanID`` onto every log record and prefixes
    the active trace/span context onto each console log line. OTLP log *export*
    (shipping the logs themselves to a collector) is deferred, tracked
    separately.

    ``log_level=WARNING`` is load-bearing. ``set_logging_format=True`` makes the
    instrumentor call ``logging.basicConfig(...)`` to install the
    trace-correlating console formatter; left to its default that call also pins
    the level to **INFO**, which floods stderr with every library's INFO logs
    (httpx request lines, sentence-transformers model load) for any verb that
    never wired ``configure_logging`` to claim the root logger first — ``query``
    being the one that bit us. Pinning the level to WARNING here keeps the
    project's default-quiet convention (``configure_logging`` also defaults to
    WARNING) while preserving the Phase-1 correlation on the warnings/errors that
    *do* surface. We must not simply drop the flag: this instrumentor version
    also attaches an export-only ``LoggingHandler`` to the root logger, so
    without the ``basicConfig`` stream handler genuine WARNING/ERROR lines would
    be swallowed instead of reaching the console.
    """
    from opentelemetry.instrumentation.logging import LoggingInstrumentor

    LoggingInstrumentor().instrument(set_logging_format=True, log_level=logging.WARNING)


def _scrub_httpx_url(span: Any, request: Any) -> None:
    """Strip the query string (+ fragment + userinfo) from the httpx span URL.

    F8 (SECURITY_REVIEW_20260625-2). The httpx instrumentor's default
    ``url.full`` / ``http.url`` span attribute is ``redact_url(str(url))``,
    which removes embedded userinfo and a *fixed allowlist* of credential-shaped
    params (``Signature`` / ``AWSAccessKeyId`` / ``X-Amz-*``) but leaves
    **arbitrary** query strings intact. Our outbound subject-authority lookups
    carry user/corpus-derived content in the query string — e.g. the Wikidata
    ``?action=wbsearchentities&search=<subject name>`` search
    (``particles/ingest/authorities/wikidata.py``) and the Numista / Nomisma
    lookups — so the raw query would ship *which entities the operator's private
    corpus concerns* to whatever OTLP collector is configured. We overwrite the
    attribute with ``scheme://host[:port]/path`` only.

    The hook runs immediately after the span is created with the unscrubbed
    value (``handle_request`` / ``handle_async_request`` set the attribute at
    span start, then invoke this hook), so the overwrite wins. We reconstruct
    from the URL components rather than calling ``copy_with`` so that userinfo,
    query, and fragment are all dropped unconditionally; both the new-semconv
    (``url.full``) and old/default-semconv (``http.url``) keys are set so the
    query never escapes regardless of the active OTel semantic-convention mode.

    ``request`` is httpx's ``RequestInfo`` namedtuple
    ``(method, url, headers, stream, extensions)``; ``request.url`` is an
    ``httpx.URL``.
    """
    url = request.url
    netloc = url.host or ""
    if url.port is not None:
        netloc = f"{netloc}:{url.port}"
    safe = f"{url.scheme}://{netloc}{url.path}"
    span.set_attribute("url.full", safe)
    span.set_attribute("http.url", safe)


async def _scrub_httpx_url_async(span: Any, request: Any) -> None:
    """Async variant of :func:`_scrub_httpx_url` for ``httpx.AsyncClient``.

    The instrumentor routes async-transport spans through ``async_request_hook``
    (which it requires to be a coroutine function), so the sync hook cannot
    cover them. All of the project's outbound HTTP is async, so this is the path
    that actually fires; the sync hook is wired too for completeness.
    """
    _scrub_httpx_url(span, request)


def _install_instrumentors() -> None:
    """Global library auto-instrumentation: httpx (client spans + traceparent) + SQLAlchemy."""
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    # F8: scrub the query string out of every httpx client span (see
    # _scrub_httpx_url). Both hooks are passed; httpx.AsyncClient uses the async
    # one, httpx.Client the sync one.
    HTTPXClientInstrumentor().instrument(
        request_hook=_scrub_httpx_url,
        async_request_hook=_scrub_httpx_url_async,
    )

    # SQLAlchemy: global instrument() patches engine creation, so the per-store
    # async engines created lazily after bootstrap (particles/db.py) are traced —
    # including the write-lock wait that motivated this ADR. setup runs at process
    # start, before any DB access, so the patch is in place first.
    #
    # F25 (SECURITY_REVIEW_20260625-2): we pass NO kwargs, so the opt-in
    # SQL-comment / literal capture (`enable_commenter`) stays off. At the pinned
    # 0.63b1 the default ``db.statement`` records only the **parameterized** SQL
    # (placeholders, e.g. ``... WHERE name = ?``) — never bound user/LLM literals,
    # and never the DSN password (verified empirically; only driver/host/port/db
    # are captured). The instrumentor exposes no flag to drop ``db.statement``,
    # and the parameterized statement is exactly the write-lock-contention signal
    # needs, so we keep it; a regression test pins the no-literal
    # invariant. Do NOT enable `enable_commenter` with `literal_binds`, and avoid
    # `literal_binds=True` at query call sites — either would route user content
    # into the span.
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    SQLAlchemyInstrumentor().instrument()


def _otlp_kwargs(cfg: ObservabilityConfig, signal: str) -> dict[str, Any]:
    """Build OTLP/HTTP exporter kwargs: per-signal endpoint + the secret headers.

    ``observability.endpoint`` is the non-secret OTLP base (e.g.
    ``http://localhost:4318``); the signal path (``/v1/traces`` or
    ``/v1/metrics``) is appended. The auth credential is the
    ``PARTICLES_OTEL_EXPORTER_HEADERS`` secret, passed through to the
    exporter only when set. With ``endpoint`` unset, the SDK's own
    ``OTEL_EXPORTER_OTLP_*`` env defaults apply.
    """
    kwargs: dict[str, Any] = {}
    if cfg.endpoint:
        kwargs["endpoint"] = f"{cfg.endpoint.rstrip('/')}/v1/{signal}"
    headers = get_otel_exporter_headers_optional()
    if headers:
        kwargs["headers"] = headers
    return kwargs


def _reset_for_tests() -> None:
    """Reset the module's one-shot state so a test can re-bootstrap (test seam).

    Does not reset the OpenTelemetry global providers (those are write-once per
    process); the enabled-path emission test runs in a subprocess to avoid
    polluting the shared test process's globals.
    """
    global _configured, _fastapi_instrumented
    with _lock:
        _configured = False
        _fastapi_instrumented = False
        _providers.clear()

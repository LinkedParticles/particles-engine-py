# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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

import copy
import logging
import os
import re
import threading
from typing import TYPE_CHECKING, Any

from particles import egress
from particles.config import get_config
from particles.secrets import get_otel_exporter_headers_optional
from particles.url_safety import is_cleartext_remote, url_without_userinfo

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
        # D19: the server span keeps the route template, never the query or
        # the concrete ids in the path (see _scrub_server_span).
        FastAPIInstrumentor.instrument_app(app, server_request_hook=_scrub_server_span)
        # The instrumentor records an unhandled exception on the server span
        # itself, so the provider serving it must carry the type-only scrub
        # even when another component installed it (G12).
        from opentelemetry import trace

        _keep_exception_type_only(trace.get_tracer_provider())
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
    _keep_exception_type_only(provider)

    if cfg.exporter == "console":
        # Synchronous export so the diagnose-now CLI prints each span as it closes.
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif cfg.exporter == "otlp" and (kwargs := _otlp_kwargs(cfg, "traces")) is not None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        otlp = OTLPSpanExporter(**kwargs)
        provider.add_span_processor(BatchSpanProcessor(otlp))

    trace.set_tracer_provider(provider)
    _providers.append(provider)
    # set_tracer_provider keeps a provider installed earlier (an embedding
    # application's, or the auto-instrumentation agent's); the spans go there.
    _keep_exception_type_only(trace.get_tracer_provider())


# The exception-event attributes a span keeps: the type, never the message or
# the stacktrace, which carry a failed request's URL or a statement's text.
_KEPT_EXCEPTION_ATTRIBUTES = frozenset({"exception.type", "exception.escaped"})
_scrubbed_providers: list[Any] = []


class _ExceptionTypeOnly:
    """Span processor: every span records an exception's type, never its text (R7).

    The hand-rolled spans open with ``record_exception=False``, but the library
    instrumentors open theirs with the defaults: the FastAPI server span records
    an unhandled route exception's message and stacktrace as an event and as
    the status description, and a store error's message quotes the request it
    served (G12, G67, SECURITY_REVIEW_20261027-1). ``on_start`` wraps each new
    span's ``add_event`` and ``set_status``, so every later recording is
    reduced before any exporter sees the span, whatever processor order the
    provider holds: an ``exception`` event keeps ``exception.type`` and
    ``exception.escaped``, and a status keeps its code without a description.
    """

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        add_event = span.add_event
        set_status = span.set_status

        def add_event_type_only(
            name: str, attributes: Any = None, timestamp: int | None = None
        ) -> None:
            if name == "exception" and attributes:
                attributes = {
                    key: value
                    for key, value in attributes.items()
                    if key in _KEPT_EXCEPTION_ATTRIBUTES
                }
            add_event(name, attributes, timestamp)

        def set_status_code_only(status: Any, description: str | None = None) -> None:
            from opentelemetry.trace import Status

            if isinstance(status, Status):
                status = Status(status.status_code)
            set_status(status)

        span.add_event = add_event_type_only
        span.set_status = set_status_code_only

    def _on_ending(self, span: Any) -> None:
        return None

    def on_end(self, span: Any) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _keep_exception_type_only(provider: Any) -> None:
    """Add the :class:`_ExceptionTypeOnly` processor to ``provider``, once.

    A provider with no ``add_span_processor`` (the API's no-op or proxy
    provider) records nothing, so it needs none.
    """
    if not hasattr(provider, "add_span_processor"):
        return
    if any(seen is provider for seen in _scrubbed_providers):
        return
    provider.add_span_processor(_ExceptionTypeOnly())
    _scrubbed_providers.append(provider)


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
    elif cfg.exporter == "otlp" and (kwargs := _otlp_kwargs(cfg, "metrics")) is not None:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

        readers.append(PeriodicExportingMetricReader(OTLPMetricExporter(**kwargs)))

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
    handler = getattr(LoggingInstrumentor, "_logging_handler", None)
    if handler is None:
        # The instrumentor keeps its handler on a private attribute; should a
        # release move it, find the exporting handler on the root logger rather
        # than leave it unscrubbed (security review 2026-10-30-1, AA9).
        handler = next(
            (
                h
                for h in logging.getLogger().handlers
                if type(h).__name__ == "LoggingHandler"
                and type(h).__module__.startswith("opentelemetry")
            ),
            None,
        )
    if handler is not None:
        _keep_exception_type_only_in_logs(handler)


def _keep_exception_type_only_in_logs(handler: logging.Handler) -> None:
    """Make ``handler`` export a logged exception's type and no message or stacktrace.

    The instrumentor's ``LoggingHandler`` copies ``exception.message`` and
    ``exception.stacktrace`` from a record's ``exc_info``, and that text carries
    deposited URLs and request content, to whatever ``LoggerProvider`` an
    embedding application installed. The handler gets a copy of each record
    without the exception, carrying ``exception.type`` only, as spans do; the
    console's handlers still see the record whole (security review
    2026-10-28-1).

    A call that formats the exception into its message (``log.error("failed:
    %s", exc)``) carries the same text in its arguments, so the copy also drops
    the arguments and exports the message template alone, naming the type of
    an exception argument (security review 2026-10-28-2).

    The handler exports every attribute a record carries beyond the standard
    ones, so a field a log call attaches with ``extra=`` (an endpoint, a
    response body) would be shipped too. The copy keeps only the standard
    record attributes and the trace correlation the instrumentor adds
    (2026-10-30-1). A message a caller formatted before the call is exported as
    written: nothing tells its text apart from the template.
    """
    emit = handler.emit

    def _emit_without_exception_text(record: logging.LogRecord) -> None:
        extra = [k for k in vars(record) if k not in _EXPORTED_RECORD_ATTRIBUTES]
        if not (record.exc_info or record.exc_text or record.stack_info or record.args or extra):
            emit(record)
            return
        exctype: type[BaseException] | None = record.exc_info[0] if record.exc_info else None
        if exctype is None and record.args:
            exctype = _first_exception_type(record.args)
        exported = copy.copy(record)
        for key in extra:
            del exported.__dict__[key]
        exported.exc_info = None
        exported.exc_text = None
        exported.stack_info = None
        exported.args = ()
        if not isinstance(exported.msg, str):
            exported.msg = type(exported.msg).__name__
        if exctype is not None:
            exported.__dict__["exception.type"] = exctype.__name__
        emit(exported)

    handler.emit = _emit_without_exception_text  # type: ignore[method-assign]


#: The record attributes the exporting handler is given: those every
#: ``LogRecord`` has (without ``message`` and ``asctime``, which a console
#: formatter fills with the formatted text), plus the instrumentor's trace
#: correlation.
_EXPORTED_RECORD_ATTRIBUTES = frozenset(
    vars(logging.LogRecord("", logging.NOTSET, "", 0, "", (), None))
) | {"otelTraceID", "otelSpanID", "otelServiceName", "otelTraceSampled"}


def _first_exception_type(args: Any) -> type[BaseException] | None:
    """The type of the first exception among a log record's arguments, if any."""
    values = args.values() if isinstance(args, dict) else args or ()
    for value in values:
        if isinstance(value, BaseException):
            return type(value)
    return None


def _scrub_httpx_url(span: Any, request: Any) -> None:
    """Reduce the httpx span URL to ``scheme://host[:port]``.

    F8 (SECURITY_REVIEW_20260625-2) and D19 (SECURITY_REVIEW_20261023-1). The
    httpx instrumentor's default ``url.full`` / ``http.url`` span attribute is
    ``redact_url(str(url))``, which removes embedded userinfo and a *fixed
    allowlist* of credential-shaped params (``Signature`` / ``AWSAccessKeyId`` /
    ``X-Amz-*``) but leaves the rest intact. Our outbound lookups carry
    user/corpus-derived content in both halves of the URL: the Wikidata
    ``?action=wbsearchentities&search=<subject name>`` search
    (``particles/ingest/authorities/wikidata.py``) in the query, and the Numista
    ``/types/<coin id>`` and Nomisma ``/id/<concept>.jsonld`` lookups in the
    path. Either would ship *which entities the operator's private corpus
    concerns* to whatever OTLP collector is configured, so only the origin is
    kept; the span name already carries the method.

    The hook runs immediately after the span is created with the unscrubbed
    value (``handle_request`` / ``handle_async_request`` set the attribute at
    span start, then invoke this hook), so the overwrite wins. We reconstruct
    from the URL components rather than calling ``copy_with`` so that userinfo,
    path, query, and fragment are all dropped unconditionally; both the
    new-semconv (``url.full``) and old/default-semconv (``http.url``) keys are
    set so nothing escapes regardless of the active OTel semantic-convention
    mode.

    ``request`` is httpx's ``RequestInfo`` namedtuple
    ``(method, url, headers, stream, extensions)``; ``request.url`` is an
    ``httpx.URL``.
    """
    url = request.url
    netloc = url.host or ""
    if url.port is not None:
        netloc = f"{netloc}:{url.port}"
    safe = f"{url.scheme}://{netloc}"
    span.set_attribute("url.full", safe)
    span.set_attribute("http.url", safe)


def _scrub_server_span(span: Any, scope: dict[str, Any]) -> None:
    """Keep the request's route template, never its query or concrete path (D19).

    The ASGI instrumentor records the full target at span start: ``http.target``
    and ``http.url`` (old semconv) or ``url.path``, ``url.query`` and
    ``url.full`` (new), all with the query string, and the concrete path names
    the particle, subject or corpus entry asked about. This ``server_request_hook``
    runs right after and overwrites each with the route template the
    instrumentor matched (``/particles/{particle_id}``), or ``""`` for a request
    that matched no route; ``url.query`` becomes ``""``, since an attribute
    cannot be removed.
    """
    attributes = getattr(span, "attributes", None) or {}
    route = attributes.get("http.route")
    template = route if isinstance(route, str) else ""
    scheme = scope.get("scheme") or "http"
    server = scope.get("server")
    origin = ""
    if isinstance(server, tuple | list) and len(server) == 2:
        origin = f"{scheme}://{server[0]}:{server[1]}"
    span.set_attribute("http.target", template)
    span.set_attribute("url.path", template)
    span.set_attribute("url.query", "")
    span.set_attribute("http.url", f"{origin}{template}")
    span.set_attribute("url.full", f"{origin}{template}")


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

    # F8 / D19: reduce every httpx client span URL to its origin (see
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


_otlp_refusals_logged: set[str] = set()

#: The OTLP/HTTP exporter's own default base, used when neither
#: ``observability.endpoint`` nor an ``OTEL_EXPORTER_OTLP_*`` variable names one.
_SDK_DEFAULT_ENDPOINT = "http://localhost:4318"


def _warn_otlp_once(key: str, message: str, *args: Any) -> None:
    if key in _otlp_refusals_logged:
        return
    _otlp_refusals_logged.add(key)
    log.warning(message, *args)


def _otlp_endpoint(cfg: ObservabilityConfig, signal: str) -> str:
    """The URL the ``signal`` exporter sends to, resolved the way the SDK would (D21).

    ``observability.endpoint`` wins; otherwise the SDK's own order applies:
    ``OTEL_EXPORTER_OTLP_<SIGNAL>_ENDPOINT`` as given, then
    ``OTEL_EXPORTER_OTLP_ENDPOINT`` plus the signal path, then the SDK default.
    The result is both what is checked and what is passed to the exporter, so
    the two can never disagree. The ``OTEL_*`` names are the SDK's, read here
    only to mirror it, not as a Particles setting.
    """
    if cfg.endpoint:
        return f"{cfg.endpoint.rstrip('/')}/v1/{signal}"
    per_signal = os.environ.get(f"OTEL_EXPORTER_OTLP_{signal.upper()}_ENDPOINT")
    if per_signal:
        return per_signal
    base = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or _SDK_DEFAULT_ENDPOINT
    return f"{base.rstrip('/')}/v1/{signal}"


#: The item delimiter of an OTLP-headers string, as the OpenTelemetry parser splits it.
_HEADER_DELIMITER = re.compile(r"[ \t]*,[ \t]*")


def _drop_record(_record: logging.LogRecord) -> bool:
    """A log filter that passes nothing."""
    return False


def _otlp_headers() -> dict[str, str]:
    """``PARTICLES_OTEL_EXPORTER_HEADERS`` parsed into the mapping the exporter takes.

    The secret is the W3C OTLP-headers string (``key=value,…``); the exporter's
    ``headers=`` argument is a mapping, and a raw string fails its construction.

    The OpenTelemetry parser logs a rejected item verbatim at WARNING, and an
    item written in HTTP-header form (``x-api-key: <secret>``) is rejected, so
    each item is parsed on its own with that log suppressed, and a rejected
    one is reported by its position only (security review 2026-10-25, Q45).
    """
    raw = get_otel_exporter_headers_optional()
    if not raw:
        return {}
    from opentelemetry.util.re import parse_env_headers

    parser_log = logging.getLogger("opentelemetry.util.re")
    headers: dict[str, str] = {}
    rejected: list[int] = []
    parser_log.addFilter(_drop_record)
    try:
        for position, item in enumerate(_HEADER_DELIMITER.split(raw), start=1):
            if not item.strip():
                continue
            parsed = parse_env_headers(item, liberal=True)
            if parsed:
                headers.update(parsed)
            else:
                rejected.append(position)
    finally:
        parser_log.removeFilter(_drop_record)
    if rejected:
        log.warning(
            "PARTICLES_OTEL_EXPORTER_HEADERS: item(s) %s are not name=value and were "
            "ignored (their text is not logged)",
            ", ".join(str(p) for p in rejected),
        )
    return headers


def _sdk_env_headers(signal: str) -> bool:
    """Whether the SDK will add headers of its own from ``OTEL_EXPORTER_OTLP_*HEADERS``."""
    return bool(
        os.environ.get(f"OTEL_EXPORTER_OTLP_{signal.upper()}_HEADERS")
        or os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")
    )


def _otlp_kwargs(cfg: ObservabilityConfig, signal: str) -> dict[str, Any] | None:
    """OTLP/HTTP exporter kwargs for ``signal``, or ``None`` when it must not be built.

    Two checks, both on the one endpoint :func:`_otlp_endpoint` resolves and
    passes explicitly:

    * ``network.egress`` must allow it. The exporters run on
      their own threads over their own HTTP stack, so the check is made here,
      where they are built.
    * Headers (the ``PARTICLES_OTEL_EXPORTER_HEADERS`` secret, or the SDK's own
      ``OTEL_EXPORTER_OTLP_*HEADERS``) are not sent over a non-loopback
      ``http://`` endpoint unless ``observability.allow_plaintext_headers`` is
      set (N12; the engine token's rule). Headers bound for a
      loopback ``http://`` collector travel on a session that honours no
      environment proxy (Q47).

    A refusal installs no exporter for that signal and logs one WARNING.
    """
    endpoint = _otlp_endpoint(cfg, signal)
    host, port = egress.host_port(endpoint)
    if not egress.allows(host, port):
        _warn_otlp_once(
            f"egress:{host}:{port}",
            "network.egress refuses the OTLP endpoint %s:%s; no OTLP exporter is installed",
            host,
            port,
        )
        return None
    headers = _otlp_headers()
    if (headers or _sdk_env_headers(signal)) and is_cleartext_remote(endpoint):
        if not cfg.allow_plaintext_headers:
            _warn_otlp_once(
                f"plaintext:{endpoint}",
                "Refusing to send the OTLP exporter headers to %s over unencrypted "
                "http://; no OTLP exporter is installed. Use https://, or a loopback "
                "collector, or set observability.allow_plaintext_headers: true.",
                url_without_userinfo(endpoint),
            )
            return None
        _warn_otlp_once(
            f"plaintext:{endpoint}",
            "The OTLP exporter headers are being sent over an unencrypted "
            "connection to %s (observability.allow_plaintext_headers is set).",
            url_without_userinfo(endpoint),
        )
    kwargs: dict[str, Any] = {"endpoint": endpoint}
    if headers:
        kwargs["headers"] = headers
    if headers or _sdk_env_headers(signal):
        # The headers are judged against this endpoint only, and an exporter
        # older than 1.45.1 follows a collector's redirect with them, to an
        # origin neither check saw. Every header-carrying exporter therefore
        # gets a session that follows no redirect, whatever exporter version
        # is installed (security review 2026-10-25-2, R8).
        kwargs["session"] = _session_following_no_redirect(
            # The plaintext rule admits a loopback http:// collector, and the
            # exporter's session would still send the headers in clear to an
            # HTTP_PROXY the rule never judged (security review 2026-10-25,
            # Q47). A loopback collector never needs a proxy, so its session
            # reads none from the environment.
            trust_env=not _is_loopback_http(endpoint)
        )
    else:
        # The SDK's own session honours an inherited SSLKEYLOGFILE as well, so
        # a header-less exporter gets one that writes no key log either
        # (security review 2026-10-28-1).
        kwargs["session"] = _session_writing_no_key_log()
    return kwargs


def _is_loopback_http(endpoint: str) -> bool:
    """Whether ``endpoint`` is an ``http://`` URL the plaintext rule admits (loopback)."""
    return endpoint.lower().startswith("http://") and not is_cleartext_remote(endpoint)


def _session_following_no_redirect(*, trust_env: bool) -> Any:
    """A ``requests`` session that returns a 3xx as it is instead of following it.

    ``trust_env=False`` also makes it read no proxy (or other setting) from the
    environment.
    """
    import requests

    class _NoRedirectSession(requests.Session):
        def get_redirect_target(self, resp: Any) -> None:
            return None

    session = _NoRedirectSession()
    session.trust_env = trust_env
    _mount_no_key_log_adapter(session)
    return session


def _session_writing_no_key_log() -> Any:
    """A plain ``requests`` session whose TLS connections write no key log."""
    import requests

    session = requests.Session()
    _mount_no_key_log_adapter(session)
    return session


def _mount_no_key_log_adapter(session: Any) -> None:
    """Make every TLS connection ``session`` opens write no TLS key log.

    urllib3 sets ``keylog_filename`` from an inherited ``SSLKEYLOGFILE`` on
    every context it creates, so each session's secrets, the collector bearer
    and the span content included, would go to the file it names. The context
    here is urllib3's own with the key log turned off, the reset
    ``particles.http.tls_context`` applies to the httpx clients (security
    review 2026-10-28-1).

    ``requests`` builds the pool for an environment ``HTTPS_PROXY`` or
    ``ALL_PROXY`` in ``proxy_manager_for``, apart from ``init_poolmanager``, so
    the adapter hands that pool the same contexts: one for the tunnelled
    connection to the collector and one for the TLS hop to an ``https://``
    proxy (security review 2026-10-28-2).

    The adapter serves ``http://`` too: a plaintext collector reached through
    an ``https://`` proxy still has a TLS session, with the proxy. ``requests``
    turns verification off for a plain ``http://`` URL, which would leave that
    session unverified, so the adapter verifies the proxy against the bundle
    ``requests`` verifies with (2026-10-30-1).
    """
    from requests.adapters import HTTPAdapter

    class _NoKeyLogAdapter(HTTPAdapter):
        def cert_verify(self, conn: Any, url: str, verify: Any, cert: Any) -> None:
            super().cert_verify(conn, url, verify, cert)
            if verify and getattr(conn, "scheme", None) == "https" and url[:6].lower() != "https:":
                # The pool is the one to an ``https://`` proxy, forwarding a
                # plain ``http://`` request.
                from requests.utils import DEFAULT_CA_BUNDLE_PATH

                conn.cert_reqs = "CERT_REQUIRED"
                conn.ca_certs = DEFAULT_CA_BUNDLE_PATH
                conn.ca_cert_dir = None

        def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
            kwargs["ssl_context"] = _context_writing_no_key_log()
            super().init_poolmanager(*args, **kwargs)

        def proxy_manager_for(self, proxy: str, **proxy_kwargs: Any) -> Any:
            if proxy not in self.proxy_manager:
                proxy_kwargs["ssl_context"] = _context_writing_no_key_log()
                if not proxy.lower().startswith("socks"):
                    # urllib3 loads no CA into a supplied proxy context, so it
                    # gets the bundle requests verifies the collector with.
                    from requests.utils import DEFAULT_CA_BUNDLE_PATH

                    proxy_context = _context_writing_no_key_log()
                    proxy_context.load_verify_locations(DEFAULT_CA_BUNDLE_PATH)
                    proxy_kwargs["proxy_ssl_context"] = proxy_context
            return super().proxy_manager_for(proxy, **proxy_kwargs)

    session.mount("https://", _NoKeyLogAdapter())
    session.mount("http://", _NoKeyLogAdapter())


def _context_writing_no_key_log() -> Any:
    """urllib3's default client TLS context with the inherited key log turned off."""
    from urllib3.util.ssl_ import create_urllib3_context

    context = create_urllib3_context()
    context.keylog_filename = None  # type: ignore[assignment]
    return context


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
        _scrubbed_providers.clear()
        _otlp_refusals_logged.clear()

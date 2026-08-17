"""Tests for the OpenTelemetry observability bootstrap.

The off path (disabled / extra-absent) is the contract that matters most — it
must be a zero-cost no-op for every existing user. The real provider-install
path is exercised in a subprocess (guarded by ``importorskip``) so it never
pollutes the shared test process's write-once OTel globals.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from particles.config import get_config, reset_config
from particles.observability import instrument_fastapi_app, setup_observability
from particles.observability import setup as obs


@pytest.fixture(autouse=True)
def _reset_obs_state() -> None:
    """Reset the module's one-shot bootstrap flags around each test."""
    obs._reset_for_tests()
    yield
    obs._reset_for_tests()


def test_config_defaults_off() -> None:
    o = get_config().observability
    assert o.enabled is False
    assert o.exporter == "console"
    assert o.service_name == "particles"
    assert o.sample_ratio == 1.0


def test_disabled_is_noop() -> None:
    # Default config has observability disabled — setup installs nothing.
    setup_observability()
    assert obs._configured is False


def test_idempotent_installs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARTICLES_OBSERVABILITY_ENABLED", "true")
    reset_config()
    calls: list[object] = []
    monkeypatch.setattr(obs, "_install", lambda cfg: calls.append(cfg))

    setup_observability()
    setup_observability()

    assert len(calls) == 1
    assert obs._configured is True


def test_extra_absent_warns_and_noops(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Simulate the `otel` extra being absent: the deferred SDK import raises.
    monkeypatch.setenv("PARTICLES_OBSERVABILITY_ENABLED", "true")
    reset_config()

    def _boom(cfg: object) -> None:
        raise ImportError("No module named 'opentelemetry.sdk'")

    monkeypatch.setattr(obs, "_install", _boom)

    # Scope at_level to the emitting logger, not just the root one: a CLI test
    # that ran `--quiet` earlier in this worker would otherwise leave the
    # `particles` family pinned at ERROR and the warning would never reach
    # caplog (see the restore_logger_levels fixture in tests/conftest.py).
    with caplog.at_level("WARNING", logger="particles.observability.setup"):
        setup_observability()

    assert obs._configured is False
    assert "otel" in caplog.text.lower()


def test_instrument_fastapi_app_noop_when_disabled() -> None:
    fastapi = pytest.importorskip("fastapi")
    app = fastapi.FastAPI()
    # Disabled config: a no-op, and crucially no error from adding middleware.
    instrument_fastapi_app(app)
    assert obs._fastapi_instrumented is False


def test_otlp_kwargs_appends_signal_path_and_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARTICLES_OBSERVABILITY_ENDPOINT", "http://collector:4318/")
    monkeypatch.setenv("PARTICLES_OTEL_EXPORTER_HEADERS", "authorization=Bearer tok")
    reset_config()
    kwargs = obs._otlp_kwargs(get_config().observability, "traces")
    assert kwargs["endpoint"] == "http://collector:4318/v1/traces"
    assert kwargs["headers"] == "authorization=Bearer tok"


def test_otlp_kwargs_empty_without_endpoint_or_headers() -> None:
    # Default config: no endpoint, no secret headers → empty kwargs (SDK defaults).
    assert obs._otlp_kwargs(get_config().observability, "metrics") == {}


def test_console_exporter_emits_span_in_subprocess() -> None:
    # The real provider-install path: run in a subprocess so the write-once OTel
    # global providers are not set in the shared test process. Skips when the
    # optional `otel` extra is not installed (e.g. CI's `uv sync --frozen`).
    pytest.importorskip("opentelemetry.sdk")
    code = textwrap.dedent(
        """
        import os
        os.environ["PARTICLES_OBSERVABILITY_ENABLED"] = "true"
        os.environ["PARTICLES_OBSERVABILITY_EXPORTER"] = "console"
        os.environ["PARTICLES_CONFIG"] = "/nonexistent-observability-test.yaml"
        from particles.observability import setup_observability
        setup_observability()
        from opentelemetry import trace
        with trace.get_tracer("particles.test").start_as_current_span("probe-span") as span:
            span.set_attribute("probe", "yes")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    # ConsoleSpanExporter prints the span as JSON to stdout (SimpleSpanProcessor
    # is synchronous, so it flushes before exit).
    assert "probe-span" in result.stdout
    assert '"service.name": "particles"' in result.stdout


def test_cli_run_opens_root_span_in_subprocess() -> None:
    # `run()` must wrap the command in a `cli.<verb>` root span so
    # child spans nest into one trace instead of fragmenting into separate
    # single-span traces. Subprocess-isolated (write-once OTel globals + argv).
    pytest.importorskip("opentelemetry.sdk")
    code = textwrap.dedent(
        """
        import os, sys
        os.environ["PARTICLES_OBSERVABILITY_ENABLED"] = "true"
        os.environ["PARTICLES_OBSERVABILITY_EXPORTER"] = "console"
        os.environ["PARTICLES_CONFIG"] = "/nonexistent-observability-test.yaml"
        sys.argv = ["particles", "probeverb"]
        from particles.api.cli import run
        from opentelemetry import trace
        async def body():
            with trace.get_tracer("particles.test").start_as_current_span("child-span"):
                pass
        run(body())
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert '"name": "cli.probeverb"' in result.stdout
    assert '"name": "child-span"' in result.stdout
    # Exactly one root (the cli span); the child parents to it rather than being
    # its own rootless trace.
    assert result.stdout.count('"parent_id": null') == 1


def test_logging_install_keeps_root_at_warning_in_subprocess() -> None:
    # The logs feature must correlate logs to traces WITHOUT pinning the root
    # logger to INFO (which floods stderr with every library's INFO logs — the
    # httpx request lines + sentence-transformers model load that motivated this
    # regression test). Subprocess-isolated: LoggingInstrumentor + basicConfig
    # mutate process-global logging state. Skips when the `otel` extra is absent.
    pytest.importorskip("opentelemetry.instrumentation.logging")
    code = textwrap.dedent(
        """
        import os, logging
        os.environ["PARTICLES_OBSERVABILITY_ENABLED"] = "true"
        os.environ["PARTICLES_OBSERVABILITY_EXPORTER"] = "none"
        os.environ["PARTICLES_CONFIG"] = "/nonexistent-observability-test.yaml"
        from particles.observability import setup_observability
        setup_observability()
        print("ROOTLEVEL=" + logging.getLevelName(logging.getLogger().level))
        # A noisy library INFO line (suppressed) and a genuine WARNING (must
        # still reach stderr — the export-only LoggingHandler must not swallow it).
        logging.getLogger("httpx").info("NOISE-info-line")
        logging.getLogger("particles.probe").warning("PROBE-warning-line")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    # Default-quiet preserved: root stays at WARNING, not the instrumentor's INFO.
    assert "ROOTLEVEL=WARNING" in result.stdout
    # INFO noise suppressed on both streams; WARNING still surfaces on stderr.
    assert "NOISE-info-line" not in result.stderr
    assert "NOISE-info-line" not in result.stdout
    assert "PROBE-warning-line" in result.stderr


def test_httpx_span_url_has_no_query_string_in_subprocess() -> None:
    # F8 (SECURITY_REVIEW_20260625-2): the httpx instrumentor's default
    # url.full/http.url is redact_url(str(url)), which leaves arbitrary query
    # params intact — so a Wikidata-shaped ?search=<subject name> lookup would
    # ship corpus-derived content to the OTLP collector. setup_observability()
    # wires a request hook that overwrites the attribute with scheme://host/path.
    # We instrument, fire a real GET at a localhost server, and assert the
    # exported span carries the path but neither the query string nor its
    # user-content value. Subprocess-isolated (write-once OTel globals); skips
    # when the `otel` extra is absent.
    pytest.importorskip("opentelemetry.instrumentation.httpx")
    code = textwrap.dedent(
        """
        import os, asyncio, threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        os.environ["PARTICLES_OBSERVABILITY_ENABLED"] = "true"
        os.environ["PARTICLES_OBSERVABILITY_EXPORTER"] = "console"
        os.environ["PARTICLES_CONFIG"] = "/nonexistent-observability-test.yaml"
        from particles.observability import setup_observability
        setup_observability()

        import httpx

        class _H(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"search": []}')
            def log_message(self, *a):
                pass

        srv = HTTPServer(("127.0.0.1", 0), _H)
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()

        async def main():
            async with httpx.AsyncClient() as client:
                await client.get(
                    f"http://127.0.0.1:{port}/w/api.php",
                    params={
                        "action": "wbsearchentities",
                        "search": "SENTINELSUBJECTXYZ",
                        "format": "json",
                    },
                )

        asyncio.run(main())
        srv.shutdown()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    # The client span was exported with the host/path (proves the hook ran on a
    # real instrumented request, not that the span was simply absent)...
    assert "/w/api.php" in result.stdout
    # ...but the query string — including the user/corpus-derived search term —
    # is gone from every exported attribute.
    assert "SENTINELSUBJECTXYZ" not in result.stdout
    assert "wbsearchentities" not in result.stdout


def test_sqlalchemy_span_has_no_user_literal_in_subprocess() -> None:
    # F25 (SECURITY_REVIEW_20260625-2): the SQLAlchemy instrumentor records a
    # db.statement span attribute. setup_observability() enables no literal /
    # SQL-comment capture, so db.statement must hold only the parameterized SQL
    # (placeholders) — never the bound user/LLM value. Run a real query whose
    # bound parameter is a sentinel and assert the span keeps the placeholdered
    # statement but not the literal. Subprocess-isolated; skips without `otel`.
    pytest.importorskip("opentelemetry.instrumentation.sqlalchemy")
    code = textwrap.dedent(
        """
        import os, asyncio
        os.environ["PARTICLES_OBSERVABILITY_ENABLED"] = "true"
        os.environ["PARTICLES_OBSERVABILITY_EXPORTER"] = "console"
        os.environ["PARTICLES_CONFIG"] = "/nonexistent-observability-test.yaml"
        from particles.observability import setup_observability
        setup_observability()

        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        async def main():
            eng = create_async_engine("sqlite+aiosqlite:///:memory:")
            async with eng.begin() as conn:
                await conn.execute(text("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)"))
                await conn.execute(
                    text("INSERT INTO t (name) VALUES (:n)"), {"n": "SENTINELSUBJECTXYZ"}
                )
                await conn.execute(
                    text("SELECT id FROM t WHERE name = :n"), {"n": "SENTINELSUBJECTXYZ"}
                )
            await eng.dispose()

        asyncio.run(main())
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    # The DB span was exported with the parameterized statement (placeholder)...
    assert "db.statement" in result.stdout
    assert "WHERE name = ?" in result.stdout
    # ...and the bound user literal never reaches any exported attribute.
    assert "SENTINELSUBJECTXYZ" not in result.stdout


class TestTracingEmbeddingModel:
    """the embed.encode/embed.load tracing proxy over the encoder.

    The proxy must be transparent: encode passes through (the span is a no-op
    when observability is off) and every other attribute delegates to the wrapped
    model, since call sites only ever use ``.encode()``.
    """

    def test_encode_passes_through(self) -> None:
        from particles.embeddings import _TracingEmbeddingModel

        class _Inner:
            def encode(
                self,
                texts: list[str],
                convert_to_numpy: bool = True,
                normalize_embeddings: bool = True,
                **kwargs: object,
            ) -> list[list[float]]:
                return [[0.1]] * len(texts)

        proxy = _TracingEmbeddingModel(_Inner())
        assert proxy.encode(["a", "b"]) == [[0.1], [0.1]]

    def test_getattr_delegates_non_encode(self) -> None:
        from particles.embeddings import _TracingEmbeddingModel

        class _Inner:
            sentinel = "delegated"

            def encode(self, *args: object, **kwargs: object) -> list[object]:
                return []

        assert _TracingEmbeddingModel(_Inner()).sentinel == "delegated"

# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OpenTelemetry observability bootstrap.

The off path (disabled / extra-absent) is the contract that matters most — it
must be a zero-cost no-op for every existing user. The real provider-install
path is exercised in a subprocess (guarded by ``importorskip``) so it never
pollutes the shared test process's write-once OTel globals.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap

import pytest

from particles.config import get_config, reset_config
from particles.observability import instrument_fastapi_app, setup_observability
from particles.observability import setup as obs

#: The exporter session the OTLP path builds is a ``requests`` one, and
#: ``requests`` arrives with the ``otel`` extra (the OTLP HTTP exporter needs
#: it), so a base install never reaches that path: the bootstrap degrades first.
needs_otel_extra = pytest.mark.skipif(
    importlib.util.find_spec("requests") is None,
    reason="the otel extra (and its requests) is not installed",
)


def _endpoint_only(kwargs: dict[str, object] | None) -> dict[str, object] | None:
    """``kwargs`` without the session every exporter gets (it writes no TLS key log)."""
    if kwargs is None:
        return None
    return {k: v for k, v in kwargs.items() if k != "session"}


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


@needs_otel_extra
def test_otlp_kwargs_appends_signal_path_and_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARTICLES_OBSERVABILITY_ENDPOINT", "https://collector:4318/")
    monkeypatch.setenv("PARTICLES_OTEL_EXPORTER_HEADERS", "authorization=Bearer tok,x-team=a")
    reset_config()
    kwargs = obs._otlp_kwargs(get_config().observability, "traces")
    assert kwargs is not None
    assert kwargs["endpoint"] == "https://collector:4318/v1/traces"
    # The exporter takes a mapping; the W3C string is parsed (a raw string
    # fails OTLPSpanExporter's construction).
    assert kwargs["headers"] == {"authorization": "Bearer tok", "x-team": "a"}


@needs_otel_extra
def test_otlp_kwargs_name_the_sdk_default_without_endpoint_or_headers() -> None:
    # Default config: the SDK's own default, passed explicitly; no headers.
    assert _endpoint_only(obs._otlp_kwargs(get_config().observability, "metrics")) == {
        "endpoint": "http://localhost:4318/v1/metrics"
    }


@needs_otel_extra
class TestOtlpEndpointIsTheCheckedOne:
    """D21 (SECURITY_REVIEW_20261023-1): what is egress-checked is what is sent to."""

    @pytest.fixture(autouse=True)
    def _clear_otel_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
            "OTEL_EXPORTER_OTLP_HEADERS",
            "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
        ):
            monkeypatch.delenv(name, raising=False)

    def test_per_signal_env_endpoint_is_checked_and_passed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import EgressAllowEntry, NetworkConfig

        monkeypatch.setenv(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "https://collector.example.invalid/v1/traces"
        )
        cfg = get_config()
        # An allowlist that admits only the SDK default the old check assumed.
        monkeypatch.setattr(
            cfg,
            "network",
            NetworkConfig(
                egress="allowlist", allow=[EgressAllowEntry(host="localhost", port=4318)]
            ),
        )
        assert obs._otlp_kwargs(cfg.observability, "traces") is None
        # The metrics signal has no override, so it is the default and allowed.
        assert _endpoint_only(obs._otlp_kwargs(cfg.observability, "metrics")) == {
            "endpoint": "http://localhost:4318/v1/metrics"
        }

    def test_general_env_endpoint_gets_the_signal_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://collector.example.invalid/")
        kwargs = obs._otlp_kwargs(get_config().observability, "traces")
        assert _endpoint_only(kwargs) == {"endpoint": "https://collector.example.invalid/v1/traces"}

    def test_config_endpoint_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://collector.example.invalid")
        monkeypatch.setenv("PARTICLES_OBSERVABILITY_ENDPOINT", "http://127.0.0.1:4318")
        reset_config()
        kwargs = obs._otlp_kwargs(get_config().observability, "traces")
        assert _endpoint_only(kwargs) == {"endpoint": "http://127.0.0.1:4318/v1/traces"}


class TestOtlpHeadersOverCleartext:
    """N12 (SECURITY_REVIEW_20261024-1): exporter headers never cross the network in clear."""

    @pytest.fixture(autouse=True)
    def _clear_otel_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("OTEL_EXPORTER_OTLP_HEADERS", "OTEL_EXPORTER_OTLP_TRACES_HEADERS"):
            monkeypatch.delenv(name, raising=False)

    def test_secret_headers_to_a_remote_http_endpoint_are_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PARTICLES_OBSERVABILITY_ENDPOINT", "http://collector.example.invalid")
        monkeypatch.setenv("PARTICLES_OTEL_EXPORTER_HEADERS", "authorization=Bearer tok")
        reset_config()
        assert obs._otlp_kwargs(get_config().observability, "traces") is None

    def test_sdk_env_headers_count_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_OBSERVABILITY_ENDPOINT", "http://collector.example.invalid")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "authorization=Bearer tok")
        reset_config()
        assert obs._otlp_kwargs(get_config().observability, "traces") is None

    @needs_otel_extra
    @pytest.mark.parametrize(
        "endpoint", ["https://collector.example.invalid", "http://127.0.0.1:4318"]
    )
    def test_https_or_loopback_carries_them(
        self, monkeypatch: pytest.MonkeyPatch, endpoint: str
    ) -> None:
        monkeypatch.setenv("PARTICLES_OBSERVABILITY_ENDPOINT", endpoint)
        monkeypatch.setenv("PARTICLES_OTEL_EXPORTER_HEADERS", "authorization=Bearer tok")
        reset_config()
        kwargs = obs._otlp_kwargs(get_config().observability, "traces")
        assert kwargs is not None
        assert kwargs["headers"] == {"authorization": "Bearer tok"}

    @needs_otel_extra
    def test_allow_plaintext_headers_opts_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_OBSERVABILITY_ENDPOINT", "http://collector.example.invalid")
        monkeypatch.setenv("PARTICLES_OTEL_EXPORTER_HEADERS", "authorization=Bearer tok")
        reset_config()
        cfg = get_config().observability.model_copy(update={"allow_plaintext_headers": True})
        kwargs = obs._otlp_kwargs(cfg, "traces")
        assert kwargs is not None
        assert kwargs["headers"] == {"authorization": "Bearer tok"}

    @needs_otel_extra
    def test_no_headers_no_refusal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_OBSERVABILITY_ENDPOINT", "http://collector.example.invalid")
        reset_config()
        kwargs = obs._otlp_kwargs(get_config().observability, "traces")
        assert _endpoint_only(kwargs) == {"endpoint": "http://collector.example.invalid/v1/traces"}


class TestOtlpHeaderSecretHandling:
    """Q45 and Q47 (SECURITY_REVIEW_20261025-1): the header secret stays out of logs and proxies."""

    _SECRET = "otlp-audit-header-value"

    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "OTEL_EXPORTER_OTLP_HEADERS",
            "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
            "NO_PROXY",
            "no_proxy",
        ):
            monkeypatch.delenv(name, raising=False)

    def test_a_malformed_item_is_never_logged(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Q45: HTTP-header form (``name: value``) is rejected without its text."""
        monkeypatch.setenv(
            "PARTICLES_OTEL_EXPORTER_HEADERS", f"x-api-key: {self._SECRET},x-tenant=acme"
        )
        with caplog.at_level("WARNING"):
            headers = obs._otlp_headers()
        assert self._SECRET not in caplog.text
        assert "item(s) 1" in caplog.text
        assert headers == {"x-tenant": "acme"}

    @pytest.mark.parametrize("allow", [False, pytest.param(True, marks=needs_otel_extra)])
    def test_a_plaintext_warning_does_not_echo_the_userinfo(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, allow: bool
    ) -> None:
        """R36 (SECURITY_REVIEW_20261025-2): the endpoint is logged without its userinfo."""
        secret = "userinfo-secret-0123456789"
        monkeypatch.setenv(
            "PARTICLES_OBSERVABILITY_ENDPOINT", f"http://op:{secret}@collector.example.invalid"
        )
        monkeypatch.setenv("PARTICLES_OTEL_EXPORTER_HEADERS", f"x-api-key={self._SECRET}")
        reset_config()
        monkeypatch.setattr(get_config().observability, "allow_plaintext_headers", allow)
        obs._otlp_refusals_logged.clear()
        with caplog.at_level("WARNING"):
            kwargs = obs._otlp_kwargs(get_config().observability, "traces")
        assert (kwargs is not None) is allow
        assert "unencrypted" in caplog.text
        assert "collector.example.invalid" in caplog.text
        assert secret not in caplog.text

    def test_well_formed_items_parse_as_before(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "PARTICLES_OTEL_EXPORTER_HEADERS", "Authorization=Bearer%20tok , x-tenant=acme"
        )
        assert obs._otlp_headers() == {"authorization": "Bearer tok", "x-tenant": "acme"}

    @needs_otel_extra
    def test_a_loopback_collector_gets_a_session_that_ignores_a_proxy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Q47: the headers reach the loopback collector, never an HTTP_PROXY.

        Two loopback servers: one is the collector, the other stands where the
        environment's proxy would be. A request on the session the exporter is
        handed reaches the collector only.
        """
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        seen: dict[str, list[str | None]] = {"collector": [], "proxy": []}

        def server(role: str) -> HTTPServer:
            class Handler(BaseHTTPRequestHandler):
                def do_POST(self) -> None:  # noqa: N802
                    seen[role].append(self.headers.get("x-api-key"))
                    self.send_response(200)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

                def log_message(self, *_args: object) -> None:
                    return None

            httpd = HTTPServer(("127.0.0.1", 0), Handler)
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            return httpd

        collector, proxy = server("collector"), server("proxy")
        try:
            endpoint = f"http://127.0.0.1:{collector.server_address[1]}"
            monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy.server_address[1]}")
            monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{proxy.server_address[1]}")
            monkeypatch.setenv("PARTICLES_OBSERVABILITY_ENDPOINT", endpoint)
            monkeypatch.setenv("PARTICLES_OTEL_EXPORTER_HEADERS", f"x-api-key={self._SECRET}")
            reset_config()
            kwargs = obs._otlp_kwargs(get_config().observability, "traces")
            assert kwargs is not None and "session" in kwargs
            kwargs["session"].post(kwargs["endpoint"], headers=kwargs["headers"], data=b"x")
        finally:
            collector.shutdown()
            proxy.shutdown()
        assert seen == {"collector": [self._SECRET], "proxy": []}

    @needs_otel_extra
    def test_no_session_without_headers_and_the_proxy_kept_off_loopback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No header, a session that still follows redirects.

        A remote https collector's session keeps the environment proxy.
        """
        monkeypatch.setenv("PARTICLES_OBSERVABILITY_ENDPOINT", "http://127.0.0.1:4318")
        reset_config()
        import requests

        plain = (obs._otlp_kwargs(get_config().observability, "traces") or {})["session"]
        assert type(plain) is requests.Session
        monkeypatch.setenv("PARTICLES_OBSERVABILITY_ENDPOINT", "https://collector.example.invalid")
        monkeypatch.setenv("PARTICLES_OTEL_EXPORTER_HEADERS", f"x-api-key={self._SECRET}")
        reset_config()
        kwargs = obs._otlp_kwargs(get_config().observability, "traces")
        assert kwargs is not None and kwargs["session"].trust_env is True


@needs_otel_extra
class TestOtlpHeadersFollowNoRedirect:
    """R8 (SECURITY_REVIEW_20261025-2): the header secret never reaches a redirect target.

    An exporter older than 1.45.1, which the published engine's ``otel`` extra
    used to admit, follows a collector's 3xx with the session's headers. The
    session :func:`_otlp_kwargs` hands every header-carrying exporter follows
    none, so the guarantee does not rest on the installed exporter version.
    """

    _SECRET = "otlp-secret-0123456789abcdef"

    @pytest.mark.parametrize("remote", [False, True])
    def test_a_collector_redirect_is_not_followed(
        self, monkeypatch: pytest.MonkeyPatch, remote: bool
    ) -> None:
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        import requests

        got: list[str | None] = []

        class Target(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                got.append(self.headers.get("x-api-key"))
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *_args: object) -> None:
                return None

        target = HTTPServer(("127.0.0.1", 0), Target)
        tport = target.server_address[1]

        class Collector(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                self.send_response(307)
                self.send_header("Location", f"http://127.0.0.1:{tport}/v1/traces")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *_args: object) -> None:
                return None

        collector = HTTPServer(("127.0.0.1", 0), Collector)
        for httpd in (target, collector):
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            monkeypatch.setenv(
                "PARTICLES_OBSERVABILITY_ENDPOINT",
                # The remote case: an https collector, judged by the plaintext
                # rule as fine; its session is the one under test, so it posts
                # to the loopback collector directly.
                "https://collector.example.invalid"
                if remote
                else f"http://127.0.0.1:{collector.server_address[1]}",
            )
            monkeypatch.setenv("PARTICLES_OTEL_EXPORTER_HEADERS", f"x-api-key={self._SECRET}")
            reset_config()
            kwargs = obs._otlp_kwargs(get_config().observability, "traces")
            assert kwargs is not None
            session: requests.Session = kwargs["session"]
            response = session.post(
                f"http://127.0.0.1:{collector.server_address[1]}/v1/traces",
                headers=kwargs["headers"],
                data=b"x",
                # What an exporter before 1.45.1 sends: the requests default.
                allow_redirects=True,
            )
        finally:
            target.shutdown()
            collector.shutdown()
        assert response.status_code == 307
        assert got == []


def test_otlp_headers_follow_no_redirect_through_the_exporter_in_subprocess() -> None:
    # R8 with the installed exporter: a collector answering 307 never sees its
    # header secret resent to the Location. Subprocess-isolated (write-once OTel
    # globals). Skips when the `otel` extra is absent.
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    secret = "otlp-secret-0123456789abcdef"
    code = textwrap.dedent(
        """
        import json, os, threading, http.server, socketserver
        os.environ["PARTICLES_CONFIG"] = "/nonexistent-observability-test.yaml"
        got = []
        class Target(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("content-length") or 0))
                got.append({k.lower(): v for k, v in self.headers.items()})
                self.send_response(200); self.end_headers()
            def log_message(self, *a): pass
        target = socketserver.TCPServer(("127.0.0.1", 0), Target)
        tport = target.server_address[1]
        class Collector(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("content-length") or 0))
                self.send_response(307)
                self.send_header("Location", f"http://127.0.0.1:{tport}/v1/traces")
                self.end_headers()
            def log_message(self, *a): pass
        collector = socketserver.TCPServer(("127.0.0.1", 0), Collector)
        cport = collector.server_address[1]
        for s in (target, collector):
            threading.Thread(target=s.serve_forever, daemon=True).start()
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from particles.config import ObservabilityConfig
        from particles.observability import setup as st
        cfg = ObservabilityConfig(enabled=True, exporter="otlp", endpoint=f"http://127.0.0.1:{cport}")
        kwargs = st._otlp_kwargs(cfg, "traces")
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(**kwargs)))
        provider.get_tracer("probe").start_span("probe").end()
        print(json.dumps({"trust_env": kwargs["session"].trust_env, "redirect_target": got}))
        """
    )
    env = {**os.environ, "PARTICLES_OTEL_EXPORTER_HEADERS": f"x-api-key={secret}"}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out == {"trust_env": False, "redirect_target": []}


class _RecordingSpan:
    def __init__(self, sink: dict[str, object]) -> None:
        self._sink = sink

    def set_attribute(self, key: str, value: object) -> None:
        self._sink[key] = value

    def __enter__(self) -> _RecordingSpan:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


class _RecordingTracer:
    def __init__(self, sink: dict[str, object]) -> None:
        self._sink = sink

    def start_as_current_span(self, _name: str, *_a: object, **_k: object) -> _RecordingSpan:
        return _RecordingSpan(self._sink)


class TestCliRootSpanCarriesNoArguments:
    """Q16 (SECURITY_REVIEW_20261025-1): the CLI root span names the verb, never its arguments.

    The span is already named ``cli.<verb>``; ``cli.command`` used to carry
    the whole command line.

    The route-template rule of the server spans: a query's text, a deposited
    URL with its query string or a particle id would otherwise reach the
    collector past every URL scrub.
    """

    _QUESTION = "what did the audit find about project-zeta"

    @staticmethod
    async def _noop() -> None:
        return None

    def test_outside_a_command_the_verb_alone_is_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from opentelemetry import trace

        from particles.api.cli import run

        attributes: dict[str, object] = {}
        monkeypatch.setattr(trace, "get_tracer", lambda *_a, **_k: _RecordingTracer(attributes))
        monkeypatch.setattr(sys, "argv", ["particles", "query", self._QUESTION])
        run(self._noop())
        assert attributes == {"cli.command": "query"}

    def test_inside_a_command_group_the_verb_alone_is_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import typer
        from opentelemetry import trace
        from typer.testing import CliRunner

        from particles.api.cli import run

        attributes: dict[str, object] = {}
        monkeypatch.setattr(trace, "get_tracer", lambda *_a, **_k: _RecordingTracer(attributes))
        root = typer.Typer()
        group = typer.Typer()
        root.add_typer(group, name="particle")

        @group.command("show")
        def show(particle_id: str, verbose: bool = False) -> None:
            run(self._noop())

        @root.callback()
        def main() -> None:
            return None

        argv = ["particle", "show", "p-secret-id", "--verbose"]
        monkeypatch.setattr(sys, "argv", ["particles", *argv])
        result = CliRunner().invoke(root, argv, prog_name="particles")
        assert result.exit_code == 0, result.output
        assert attributes == {"cli.command": "particle"}


class _ExceptionRecordingSpan:
    """A span that behaves as the SDK's does when an exception leaves it.

    ``start_as_current_span`` records an ``exception`` event carrying the
    exception's message unless it is opened with ``record_exception=False``,
    and marks the span ERROR unless ``set_status_on_exception=False``.
    """

    def __init__(self, sink: dict[str, object], kwargs: dict[str, object]) -> None:
        self._sink = sink
        self._kwargs = kwargs

    def set_attribute(self, key: str, value: object) -> None:
        self._sink[key] = value

    def set_status(self, status: object) -> None:
        self._sink["status"] = getattr(getattr(status, "status_code", None), "name", status)

    def __enter__(self) -> _ExceptionRecordingSpan:
        return self

    def __exit__(self, _type: object, exc: BaseException | None, _tb: object) -> None:
        if exc is None:
            return None
        if self._kwargs.get("record_exception", True):
            self._sink["exception.message"] = str(exc)
        if self._kwargs.get("set_status_on_exception", True):
            self._sink["status"] = "ERROR"
        return None


class _ExceptionRecordingTracer:
    def __init__(self, sink: dict[str, object]) -> None:
        self._sink = sink

    def start_as_current_span(
        self, _name: str, *_a: object, **kwargs: object
    ) -> _ExceptionRecordingSpan:
        return _ExceptionRecordingSpan(self._sink, kwargs)


class TestCliRootSpanCarriesNoExceptionMessage:
    """R7 (SECURITY_REVIEW_20261025-2): a failed verb exports no exception message.

    A deposit of a URL answering 404 fails with ``raise_for_status``, whose
    text is the URL with its query string. Recorded as the root span's
    exception event, it would reach the collector past every URL scrub.
    """

    _ARG = "https://files.example.test/report.pdf?X-Amz-Signature=audit0123456789abcdef"

    def test_the_root_span_records_the_exception_type_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx
        from opentelemetry import trace

        from particles.api.cli import run

        sink: dict[str, object] = {}
        monkeypatch.setattr(trace, "get_tracer", lambda *_a, **_k: _ExceptionRecordingTracer(sink))
        monkeypatch.setattr(sys, "argv", ["particles", "deposit", self._ARG])

        async def verb() -> None:
            request = httpx.Request("GET", self._ARG)
            httpx.Response(404, request=request).raise_for_status()

        with pytest.raises(httpx.HTTPStatusError):
            run(verb())
        assert "X-Amz-Signature" not in repr(sink)
        assert sink == {
            "cli.command": "deposit",
            "error.type": "HTTPStatusError",
            "status": "ERROR",
        }


def test_cli_root_span_exports_no_argument_inside_a_recorded_exception_in_subprocess() -> None:
    # R7 with the real SDK: every exported span and event of a failed verb is
    # free of its argument. Subprocess-isolated (write-once OTel globals + argv).
    pytest.importorskip("opentelemetry.sdk")
    arg = "https://files.example.test/report.pdf?X-Amz-Signature=audit0123456789abcdef"
    code = textwrap.dedent(
        """
        import json, os, sys
        os.environ["PARTICLES_CONFIG"] = "/nonexistent-observability-test.yaml"
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        ARG = sys.argv[1]
        sys.argv = ["particles", "deposit", ARG]
        import httpx
        from particles.api.cli import run
        async def verb():
            request = httpx.Request("GET", ARG)
            httpx.Response(404, request=request).raise_for_status()
        try:
            run(verb())
        except BaseException:
            pass
        spans = []
        for s in exporter.get_finished_spans():
            spans.append({"name": s.name,
                          "status": s.status.status_code.name,
                          "attributes": {k: str(v) for k, v in s.attributes.items()},
                          "events": [{k: str(v) for k, v in e.attributes.items()}
                                     for e in s.events]})
        print(json.dumps(spans))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, arg], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    exported = result.stdout.strip().splitlines()[-1]
    assert "X-Amz-Signature" not in exported
    assert arg not in exported
    assert '"name": "cli.deposit"' in exported
    assert '"status": "ERROR"' in exported


class _FakeSpan:
    def __init__(self, attributes: dict[str, object] | None = None) -> None:
        self.attributes = dict(attributes or {})

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


class TestSpanUrlsCarryNoCorpusIdentifier:
    """D19 (SECURITY_REVIEW_20261023-1)."""

    def test_client_span_keeps_only_the_origin(self) -> None:
        import httpx

        request = httpx.Request("GET", "https://api.numista.com/v3/types/SENTINEL42?lang=en")
        span = _FakeSpan()
        obs._scrub_httpx_url(span, request)
        assert span.attributes["url.full"] == "https://api.numista.com"
        assert span.attributes["http.url"] == "https://api.numista.com"

    def test_server_span_keeps_the_route_template(self) -> None:
        span = _FakeSpan(
            {
                "http.route": "/particles/{particle_id}",
                "http.target": "/particles/SENTINEL42?q=private",
                "url.path": "/particles/SENTINEL42",
                "url.query": "q=private",
                "http.url": "http://127.0.0.1:8000/particles/SENTINEL42?q=private",
                "url.full": "http://127.0.0.1:8000/particles/SENTINEL42?q=private",
            }
        )
        scope = {"type": "http", "scheme": "http", "server": ("127.0.0.1", 8000)}
        obs._scrub_server_span(span, scope)
        assert span.attributes["http.target"] == "/particles/{particle_id}"
        assert span.attributes["url.path"] == "/particles/{particle_id}"
        assert span.attributes["url.query"] == ""
        assert span.attributes["url.full"] == "http://127.0.0.1:8000/particles/{particle_id}"
        assert not any("SENTINEL42" in str(v) for v in span.attributes.values())
        assert not any("private" in str(v) for v in span.attributes.values())

    def test_unrouted_server_span_keeps_no_path(self) -> None:
        span = _FakeSpan({"http.target": "/nowhere/SENTINEL42"})
        obs._scrub_server_span(span, {"scheme": "http", "server": ("127.0.0.1", 8000)})
        assert span.attributes["http.target"] == ""
        assert span.attributes["url.full"] == "http://127.0.0.1:8000"

    def test_fastapi_instrumentation_wires_the_server_hook(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import types

        fastapi = pytest.importorskip("fastapi")
        calls: list[dict[str, object]] = []

        class _Instrumentor:
            @staticmethod
            def instrument_app(_app: object, **kwargs: object) -> None:
                calls.append(kwargs)

        module = types.ModuleType("opentelemetry.instrumentation.fastapi")
        module.FastAPIInstrumentor = _Instrumentor  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "opentelemetry.instrumentation.fastapi", module)
        monkeypatch.setattr(obs, "setup_observability", lambda: None)
        monkeypatch.setenv("PARTICLES_OBSERVABILITY_ENABLED", "true")
        reset_config()
        instrument_fastapi_app(fastapi.FastAPI())
        assert calls == [{"server_request_hook": obs._scrub_server_span}]


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
    # wires a request hook that overwrites the attribute with scheme://host[:port].
    # We instrument, fire a real GET at a localhost server, and assert the
    # exported span carries the origin but neither the path, the query string nor its
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
    # The client span was exported with its origin (proves the hook ran on a
    # real instrumented request, not that the span was simply absent)...
    assert "http://127.0.0.1:" in result.stdout
    # ...but the path and the query string, including the user/corpus-derived
    # search term, are gone from every exported attribute (F8, D19).
    assert "/w/api.php" not in result.stdout
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
    # The bound user literal never reaches any exported attribute...
    assert "SENTINELSUBJECTXYZ" not in result.stdout
    if "falls outside of that range" in result.stderr:
        # The instrumentor declines a SQLAlchemy release it does not list as
        # supported and exports no DB span at all, so the literal cannot leak
        # through one; the placeholder half below has nothing to read.
        pytest.skip("the SQLAlchemy instrumentor does not support this SQLAlchemy")
    # ...and the DB span was exported with the parameterized statement.
    assert "db.statement" in result.stdout
    assert "WHERE name = ?" in result.stdout


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


class TestChildSpansCarryNoExceptionMessage:
    """R7 (SECURITY_REVIEW_20261026-1, W13): no span below the root records a message.

    The ``@traced`` operation spans, the extraction, embedding and LLM spans
    and the MCP tool spans each record the exception's type alone: a failed URL
    deposit's message is the URL with its query string.
    """

    _SECRET = "X-Amz-Signature=audit6childspan0123456789"
    _URL = f"https://files.example.test/report.pdf?{_SECRET}"

    def test_a_failed_url_deposit_span_carries_no_part_of_the_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed URL deposit's operation span carries no part of the deposited URL's query."""
        import asyncio
        from typing import Any

        import httpx

        from particles.corpus import deposit as deposit_mod
        from particles.observability import spans

        sink: dict[str, object] = {}
        monkeypatch.setattr(spans, "_tracer", _ExceptionRecordingTracer(sink))
        monkeypatch.setattr("particles.url_safety.validate_fetch_url", lambda _url: None)

        async def fake_get_capped(_client: Any, url: str, **_kw: Any) -> httpx.Response:
            return httpx.Response(404, request=httpx.Request("GET", url))

        monkeypatch.setattr(deposit_mod, "get_capped", fake_get_capped)

        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(deposit_mod.deposit_url(None, self._URL, source_type="web"))  # type: ignore[arg-type]
        assert self._SECRET not in repr(sink)
        assert sink == {"error.type": "HTTPStatusError", "status": "ERROR"}

    def test_every_hand_rolled_span_opens_without_exception_recording(self) -> None:
        """Every ``start_as_current_span`` call in the package passes both opt-outs."""
        import ast
        from pathlib import Path

        import particles

        root = Path(particles.__file__).parent
        unguarded: list[str] = []
        for path in sorted(root.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "start_as_current_span"
                ):
                    continue
                kwargs = {
                    kw.arg: kw.value.value
                    for kw in node.keywords
                    if isinstance(kw.value, ast.Constant)
                }
                if (
                    kwargs.get("record_exception") is not False
                    or kwargs.get("set_status_on_exception") is not False
                ):
                    unguarded.append(f"{path.relative_to(root.parent)}:{node.lineno}")
        assert not unguarded, unguarded

# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the egress and telemetry findings of security review 2026-10-27-1.

G12, G13, G67 (R7, W13, Z13, Z29): no span records an exception's text.
G15, G50 (W56, E6): the diagnostics name every SDK credential and TLS key-log
variable, and the clients Particles builds write no TLS key log.
G52 (K10): the pinned subprocess fetch searches no directory another user may write.
G56: both retract routes hold the store's writer lock.
W1: the per-client engine key is masked after a percent-encoded separator and
before the tool summary cap.
W12, W58, R36, E1: an engine URL is never shown with a credential.

Every test runs under a temporary ``HOME``, opens no real store and makes no
model call or outside connection.
"""

from __future__ import annotations

import json
import os
import secrets as _secrets
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from particles.config import environment_notes, get_config, reset_config
from particles.corpus.redaction import redact_secrets

_SECRET = "auditsevenspanleak1234567890"


def _engine_key(role_prefix: str = "agent") -> str:
    from particles.api import _keys

    token = f"ptk_{role_prefix}_{_secrets.token_hex(6)}_{_secrets.token_urlsafe(32)[:43]}"
    assert _keys.parse_token(token) is not None
    return token


# ---------------------------------------------------------------------------
# G12, G13, G67: exception text on spans
# ---------------------------------------------------------------------------


class _FakeSpan:
    """Records what a span would export, with the SDK span's two recording methods."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.status: Any = None

    def add_event(self, name: str, attributes: Any = None, timestamp: int | None = None) -> None:
        self.events.append((name, dict(attributes or {})))

    def set_status(self, status: Any, description: str | None = None) -> None:
        self.status = (status, description)

    def record_exception(self, exception: BaseException) -> None:
        # The SDK's record_exception calls the instance's add_event.
        self.add_event(
            "exception",
            {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
                "exception.stacktrace": f"Traceback: {exception}",
                "exception.escaped": "False",
            },
        )


def test_every_span_keeps_an_exception_type_and_never_its_text() -> None:
    """A span the processor saw start records an exception's type, not its message."""
    from opentelemetry.trace import Status, StatusCode

    from particles.observability.setup import _ExceptionTypeOnly

    span = _FakeSpan()
    _ExceptionTypeOnly().on_start(span)
    exc = RuntimeError(f"statement failed [parameters: ('{_SECRET}',)]")
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, f"RuntimeError: {exc}"))
    span.add_event("cache.miss", {"key": "kept"})

    assert _SECRET not in repr(span.events) + repr(span.status)
    assert span.events[0] == (
        "exception",
        {"exception.type": "RuntimeError", "exception.escaped": "False"},
    )
    assert span.events[1] == ("cache.miss", {"key": "kept"})
    status, description = span.status
    assert status.status_code is StatusCode.ERROR
    assert status.description is None and description is None


def test_the_scrub_is_added_to_a_provider_once() -> None:
    from particles.observability import setup

    added: list[Any] = []

    class _Provider:
        def add_span_processor(self, processor: Any) -> None:
            added.append(processor)

    provider = _Provider()
    setup._reset_for_tests()
    try:
        setup._keep_exception_type_only(provider)
        setup._keep_exception_type_only(provider)
        setup._keep_exception_type_only(object())  # a no-op provider records nothing
    finally:
        setup._reset_for_tests()
    assert len(added) == 1
    assert isinstance(added[0], setup._ExceptionTypeOnly)


def test_a_store_statement_error_names_no_bound_value() -> None:
    """The store's engine keeps bound parameters out of every statement error's message."""
    from particles.db import get_engine

    assert get_engine().sync_engine.hide_parameters is True


def _spans_from_subprocess(code: str, tmp_path: Path) -> list[dict[str, Any]]:
    env_code = textwrap.dedent(
        f"""
        import os
        os.environ["HOME"] = {str(tmp_path)!r}
        cfg = os.path.join({str(tmp_path)!r}, "config.yaml")
        with open(cfg, "w") as f:
            f.write("observability:\\n  enabled: true\\n  exporter: none\\n  metrics: false\\n"
                    "  logs: false\\nnetwork:\\n  egress: none\\n")
        os.environ["PARTICLES_CONFIG"] = cfg
        os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///" + os.path.join(
            {str(tmp_path)!r}, "s.db")
        for k in list(os.environ):
            if k.startswith(("ANTHROPIC_", "PARTICLES_LLM_API_KEY_")) or k in (
                "PARTICLES_API_KEY", "PARTICLES_ENGINE_TOKEN"):
                del os.environ[k]
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        """
    )
    dump = textwrap.dedent(
        """
        import json
        out = []
        for s in exporter.get_finished_spans():
            out.append({"name": s.name, "kind": s.kind.name,
                        "description": s.status.description or "",
                        "events": [{k: str(v) for k, v in e.attributes.items()}
                                   for e in s.events]})
        print("SPANS=" + json.dumps(out))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", env_code + textwrap.dedent(code) + dump],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    line = [ln for ln in result.stdout.splitlines() if ln.startswith("SPANS=")][-1]
    spans: list[dict[str, Any]] = json.loads(line[len("SPANS=") :])
    return spans


def test_a_failing_route_records_no_request_text_on_the_server_span(tmp_path: Path) -> None:
    """A route that fails with a store error exports no part of the request's query on any span.

    The server span is the FastAPI instrumentor's, opened outside the
    hand-rolled ``record_exception=False`` ratchet, under a provider an
    embedding application installed first. Subprocess-isolated (the OTel
    globals are write-once); skips without the ``otel`` extra.
    """
    pytest.importorskip("opentelemetry.sdk")
    pytest.importorskip("opentelemetry.instrumentation.fastapi")
    spans = _spans_from_subprocess(
        f"""
        from fastapi.testclient import TestClient
        from particles.api.app import app
        from particles.observability.setup import instrument_fastapi_app
        instrument_fastapi_app(app)
        client = TestClient(app, base_url="http://127.0.0.1:8000",
                            client=("127.0.0.1", 50000), raise_server_exceptions=False)
        # The scratch store was never migrated, so the search's statement fails.
        r = client.get("/subjects/search", params={{"q": {_SECRET!r}}})
        print("STATUS", r.status_code)
        """,
        tmp_path,
    )
    server = [s for s in spans if s["kind"] == "SERVER"]
    assert server, spans
    assert any(e.get("exception.type") for s in server for e in s["events"]), server
    assert _SECRET not in json.dumps(spans)


# ---------------------------------------------------------------------------
# G15, G50: the diagnostics and the TLS key log
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "OTEL_EXPORTER_OTLP_TRACES_CLIENT_KEY",
        "OTEL_EXPORTER_OTLP_TRACES_CLIENT_CERTIFICATE",
        "OTEL_EXPORTER_OTLP_METRICS_CLIENT_KEY",
        "OTEL_EXPORTER_OTLP_LOGS_CLIENT_CERTIFICATE",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "SSLKEYLOGFILE",
    ],
)
def test_every_sdk_credential_or_key_log_variable_is_named_without_its_value(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    value = "audit7diagvalue"
    monkeypatch.setenv(name, value)
    notes = environment_notes()
    assert any(name in note for note in notes), notes
    assert not any(value in note for note in notes)


@pytest.mark.asyncio
@pytest.mark.parametrize("which", ["egress", "particles"])
async def test_the_clients_particles_builds_write_no_tls_key_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    """The client carrying the engine token or a provider key writes no TLS session secrets."""
    from particles.http import egress_client, particles_client

    monkeypatch.setenv("SSLKEYLOGFILE", str(tmp_path / "keys.log"))

    def context_of(client: Any) -> Any:
        pool = getattr(client._transport, "_pool", None)
        return getattr(pool, "_ssl_context", None)

    if which == "egress":
        async with egress_client(purpose="engine") as client:
            context = context_of(client)
    else:
        async with particles_client() as client:
            context = context_of(client)
    assert context is not None
    assert context.keylog_filename is None
    assert context.verify_mode.name == "CERT_REQUIRED"


# ---------------------------------------------------------------------------
# G52: the pinned subprocess PATH
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [0o777, 0o1777, 0o757])
def test_the_pinned_env_keeps_no_path_entry_others_may_write(tmp_path: Path, mode: int) -> None:
    from particles.url_safety import pinned_subprocess_env

    shared = tmp_path / "shared-bin"
    shared.mkdir()
    shared.chmod(mode)
    env = pinned_subprocess_env({"PATH": f"{shared}{os.pathsep}/usr/bin"})
    assert env["PATH"].split(os.pathsep) == ["/usr/bin"]


def test_the_pinned_env_keeps_a_path_entry_only_this_user_writes(tmp_path: Path) -> None:
    from particles.url_safety import pinned_subprocess_env

    own = tmp_path / "own-bin"
    own.mkdir()
    own.chmod(0o755)
    missing = tmp_path / "missing-bin"
    env = pinned_subprocess_env({"PATH": os.pathsep.join([str(own), str(missing), "bin"])})
    assert env["PATH"].split(os.pathsep) == [str(own)]


# ---------------------------------------------------------------------------
# G56: the retract routes hold the writer lock
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/particles/retract", "/particles/x/retract"])
def test_every_retract_route_holds_the_store_writer_lock(
    path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A route that changes a belief's status holds its store's writer lock.

    Judged by behaviour: the routes open their write session in the handler,
    after every refusal (security review 2026-10-28-1, O47).
    """
    import asyncio
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    from fastapi.testclient import TestClient

    from particles import db
    from particles.api.app import app

    cfg = tmp_path / "config.yaml"
    cfg.write_text("mcp:\n  write:\n    enabled_stores: [default]\n", encoding="utf-8")
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'g56.db'}")
    monkeypatch.setenv("PARTICLES_API_KEY", "g56-operator-key")
    reset_config()

    async def _create() -> None:
        import particles._orm_modules  # noqa: F401

        async with db.get_engine().begin() as conn:
            await conn.run_sync(db.Base.metadata.create_all)
        await db.get_engine().dispose()

    asyncio.run(_create())
    taken: list[str] = []
    real = db.write_lock

    @asynccontextmanager
    async def _recording(store: str = "default") -> AsyncIterator[None]:
        taken.append(store)
        async with real(store):
            yield

    monkeypatch.setattr(db, "write_lock", _recording)
    client = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))
    body = {"particle_id": "x", "reason": "r"} if path == "/particles/retract" else {"reason": "r"}
    resp = client.post(path, json=body, headers={"Authorization": "Bearer g56-operator-key"})
    assert resp.status_code == 400, resp.text  # no such belief, judged under the lock
    assert taken == ["default"]


# ---------------------------------------------------------------------------
# W1: the engine key mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("before", ["%3D", "%20", "Bearer%20", "%22"])
def test_a_key_after_a_percent_encoded_separator_is_masked(before: str) -> None:
    from particles.secrets import redact_known_secrets

    key = _engine_key()
    text = f"curl https://engine.example/query?token{before}{key}"
    assert key[-43:] not in redact_secrets(text)
    assert key[-43:] not in redact_known_secrets(text)


def test_a_key_cut_by_the_tool_summary_cap_leaves_no_secret_run() -> None:
    from particles.api.cli._claude_code import distill_transcript

    token = _engine_key()
    secret = token[len("ptk_agent_") + 13 :]
    # The summary keeps 119 characters; this prefix puts 40 of the secret's 43 inside them.
    prefix = "cd /srv/app/releases/2026-10-26/current && ENGINE_KEY="
    prefix = "x" * (56 - len(prefix)) + prefix
    command = f"{prefix}{token} ./client sync --all --verbose --retry 3"
    line = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "name": "Bash", "input": {"command": command}}],
        },
    }
    masked = redact_secrets(distill_transcript(json.dumps(line), "s"))
    assert secret[:16] not in masked


# ---------------------------------------------------------------------------
# W12, W58, R36, E1: engine URLs without their credential
# ---------------------------------------------------------------------------

_PW = "4417/audit7pw"


@pytest.mark.parametrize(
    ("url", "shown"),
    [
        (f"https://svc:{_PW}@engine.invalid:8443", "https://engine.invalid:8443"),
        (f"https://svc:{_PW}@engine.invalid/v1?x=1", "https://engine.invalid/v1?x=1"),
        ("svc:audit7schemeless@engine.invalid:8443", "engine.invalid:8443"),
    ],
)
def test_a_port_shaped_or_scheme_less_credential_is_stripped(url: str, shown: str) -> None:
    from particles.url_safety import url_without_userinfo

    assert url_without_userinfo(url) == shown


@pytest.mark.parametrize(
    "base_url",
    [f"https://svc:{_PW}@engine.invalid:8443", "svc:audit7schemeless@engine.invalid:8443"],
)
def test_the_facade_refusal_returns_no_engine_credential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, base_url: str
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    from particles.mcp.memory_compat import server

    get_config().engine.base_url = base_url
    try:
        message = server._remote_engine_refusal()
    finally:
        reset_config()
    assert message is not None
    assert "engine.invalid" in message
    assert "audit7" not in message


def test_the_mcp_server_engine_origin_log_strips_userinfo(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    import logging

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "https://svc:audit7logpw@engine.invalid:8443")
    from particles.mcp import server

    with caplog.at_level(logging.WARNING):
        server._note_the_engine_origin()
    assert "engine.invalid" in caplog.text
    assert "audit7logpw" not in caplog.text


def test_the_install_note_strips_the_inherited_url_userinfo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "https://svc:audit7initpw@engine.invalid:8443")
    import particles.api.cli.init as init_mod

    result: dict[str, Any] = {"next_steps": []}
    init_mod._note_inherited_engine_url(result, tmp_path / "config.yaml")
    err = capsys.readouterr().err
    assert "engine.invalid" in " ".join(result["next_steps"])
    assert "audit7initpw" not in " ".join(result["next_steps"])
    assert "audit7initpw" not in err

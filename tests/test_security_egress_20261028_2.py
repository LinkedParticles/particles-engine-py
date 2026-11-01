# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the egress findings of security review 2026-10-28-2.

Each test runs under a temporary ``HOME``, writes only under ``tmp_path``,
makes no model call and connects only to loopback servers it starts itself.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import select
import socket
import stat
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from particles import url_safety
from particles.config import get_config, reset_config
from tests.test_security_egress_20261028 import _secret_lines, tls_server  # noqa: F401


@pytest.fixture(autouse=True)
def _temporary_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


def _load_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    cfg = tmp_path / "cfg" / "config.yaml"
    cfg.parent.mkdir(mode=0o700, exist_ok=True)
    cfg.write_text(text)
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    reset_config()


# --------------------------------------------------------------------------- #
# The OTLP sessions through an environment proxy write no TLS key log
# --------------------------------------------------------------------------- #


@pytest.fixture
def connect_proxy() -> Iterator[int]:
    """A loopback HTTP ``CONNECT`` proxy relaying to whatever host and port it is asked for."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    port = listener.getsockname()[1]
    stop = threading.Event()

    def relay(client: socket.socket) -> None:
        try:
            client.settimeout(5)
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = client.recv(4096)
                if not chunk:
                    return
                data += chunk
            target = data.split(b"\r\n", 1)[0].split()[1].decode()
            host, _, tport = target.rpartition(":")
            upstream = socket.create_connection((host, int(tport)), timeout=5)
            client.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            pair = [client, upstream]
            while not stop.is_set():
                ready, _, _ = select.select(pair, [], [], 0.5)
                for s in ready:
                    chunk = s.recv(65536)
                    if not chunk:
                        return
                    (upstream if s is client else client).sendall(chunk)
        except OSError:
            return
        finally:
            client.close()

    def serve() -> None:
        listener.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:
                continue
            threading.Thread(target=relay, args=(conn,), daemon=True).start()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        stop.set()
        thread.join(2)
        listener.close()


@pytest.mark.parametrize("headers", [True, False])
def test_the_otlp_session_through_an_environment_proxy_writes_no_tls_key_log(
    headers: bool,
    tls_server: tuple[int, Path],  # noqa: F811
    connect_proxy: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed handshake through the OTLP session Particles builds, over an
    environment ``HTTPS_PROXY``, with the collector bearer or without, writes no
    session secret to ``SSLKEYLOGFILE``."""
    pytest.importorskip("requests")
    from particles.observability import setup as obs

    port, cert_path = tls_server
    _load_config(
        tmp_path, monkeypatch, "network:\n  egress: allowlist\n  allow:\n    - host: 127.0.0.1\n"
    )
    for name in ("NO_PROXY", "no_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{connect_proxy}")
    monkeypatch.setenv("PARTICLES_OBSERVABILITY_ENDPOINT", f"https://127.0.0.1:{port}")
    if headers:
        monkeypatch.setenv("PARTICLES_OTEL_EXPORTER_HEADERS", "authorization=Bearer s2audit9")
    reset_config()
    kwargs = obs._otlp_kwargs(get_config().observability, "traces")
    assert kwargs is not None
    response = kwargs["session"].post(
        kwargs["endpoint"], data=b"x", verify=str(cert_path), timeout=5
    )
    assert response.status_code == 404
    assert _secret_lines(tmp_path / "keys.log") == []


def test_the_proxy_hop_context_verifies_against_the_bundle_requests_uses() -> None:
    """The key-log-free context for an ``https://`` proxy hop still verifies its certificate."""
    pytest.importorskip("requests")
    import ssl

    from particles.observability import setup as obs

    session = obs._session_writing_no_key_log()
    adapter = session.get_adapter("https://collector.example")
    manager = adapter.proxy_manager_for("https://proxy.example:3128")
    context = manager.proxy_ssl_context
    assert context.keylog_filename is None
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.cert_store_stats()["x509_ca"] > 0
    assert manager.connection_pool_kw["ssl_context"].keylog_filename is None


# --------------------------------------------------------------------------- #
# A PATH entry is judged along its whole link chain
# --------------------------------------------------------------------------- #


def test_a_path_entry_through_a_link_in_a_directory_others_may_write_is_dropped(
    tmp_path: Path,
) -> None:
    """A ``PATH`` entry whose link chain passes through a link in a directory other
    users may write without the sticky bit is dropped by ``pinned_subprocess_env``."""
    own = tmp_path / "own"
    own.mkdir(mode=0o700)
    open_dir = tmp_path / "open"
    open_dir.mkdir()
    middle = open_dir / "mid"
    middle.symlink_to("/usr/bin")
    entry = own / "bin"
    entry.symlink_to(middle)
    os.chmod(open_dir, 0o777)
    try:
        assert os.stat(open_dir).st_mode & stat.S_IWOTH
        env = url_safety.pinned_subprocess_env({"PATH": str(entry)})
    finally:
        os.chmod(open_dir, 0o755)
    assert str(entry) not in env["PATH"].split(os.pathsep), env


def test_a_path_entry_through_a_chain_of_trusted_links_is_kept(tmp_path: Path) -> None:
    """A chain of links this user owns, in directories only this user may write, is kept."""
    own = tmp_path / "own"
    own.mkdir(mode=0o700)
    target = tmp_path / "target"
    target.mkdir(mode=0o755)
    (own / "mid").symlink_to(target)
    entry = own / "bin"
    entry.symlink_to("mid")
    env = url_safety.pinned_subprocess_env({"PATH": str(entry)})
    assert env["PATH"].split(os.pathsep) == [str(entry)]


def test_a_path_entry_whose_link_chain_never_ends_is_dropped(tmp_path: Path) -> None:
    """A link loop is refused rather than walked forever."""
    own = tmp_path / "own"
    own.mkdir(mode=0o700)
    (own / "a").symlink_to(own / "b")
    (own / "b").symlink_to(own / "a")
    assert not url_safety._only_trusted_users_write(str(own / "a"))


# --------------------------------------------------------------------------- #
# Benchmark children run with the egress backstop armed
# --------------------------------------------------------------------------- #


def test_the_query_latency_child_arms_the_egress_backstop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The query-latency benchmark child refuses a connection ``network.egress: none`` forbids."""
    from particles import egress
    from particles.benchmark.query_latency import cell

    _load_config(tmp_path, monkeypatch, "network:\n  egress: none\n")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    outcome: dict[str, object] = {}

    async def _probe(spec: object) -> object:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                outcome["connected"] = True
        except egress.EgressRefused:
            outcome["connected"] = False
        return object()

    class _Spec:
        result_path = str(tmp_path / "result.json")

        @staticmethod
        def model_validate_json(_raw: str) -> _Spec:
            return _Spec()

    spec_path = tmp_path / "spec.json"
    spec_path.write_text("{}")
    monkeypatch.setattr(cell, "run_cell", _probe)
    monkeypatch.setattr(cell, "CellSpec", _Spec)
    monkeypatch.setattr(cell, "write_result", lambda *_a: None)
    try:
        cell.main([str(spec_path)])
    finally:
        egress.disarm_backstop()
        listener.close()
    assert outcome.get("connected") is False, outcome


def test_the_update_gate_child_runs_its_store_loops_armed() -> None:
    """The update-gate child starts no event loop of its own outside the armed runner."""
    import particles.benchmark.update_gate._child as child

    text = Path(child.__file__).read_text(encoding="utf-8")
    assert "asyncio.run(" not in text
    assert text.count("egress.run_armed(") == 2


# --------------------------------------------------------------------------- #
# An exported log record carries no exception text from its message
# --------------------------------------------------------------------------- #

_LOG_SECRET = "s2audit9-deposited-q-4b1e"


@pytest.fixture
def unset_logger_provider() -> Iterator[None]:
    """The process's global ``LoggerProvider`` unset for the test and put back after it.

    The global is set-once, so without this the test would see whichever
    provider an earlier test in the process installed, and leave its own for
    the next one.
    """
    pytest.importorskip("opentelemetry.sdk._logs")
    from opentelemetry._logs import _internal
    from opentelemetry.util._once import Once

    saved = (_internal._LOGGER_PROVIDER, _internal._LOGGER_PROVIDER_SET_ONCE)
    _internal._LOGGER_PROVIDER = None
    _internal._LOGGER_PROVIDER_SET_ONCE = Once()
    try:
        yield
    finally:
        _internal._LOGGER_PROVIDER, _internal._LOGGER_PROVIDER_SET_ONCE = saved


def test_an_exported_log_record_carries_no_exception_text_logged_in_its_message(
    monkeypatch: pytest.MonkeyPatch, unset_logger_provider: None
) -> None:
    """With an exporting ``LoggerProvider``, a failed model call's logged error ships no
    exception text in the exported record."""
    pytest.importorskip("opentelemetry.instrumentation.logging")
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor

    import particles.llm as llm_pkg
    from particles.llm import breaker
    from particles.observability import setup as setup_mod

    async def _failing(*_a: object, **_k: object) -> str:
        raise RuntimeError(f"upstream 500 for https://h.example/?q={_LOG_SECRET}")

    monkeypatch.setattr(llm_pkg, "complete", _failing)
    monkeypatch.setattr(breaker, "_is_account_level", lambda _exc: False)
    provider = LoggerProvider()
    exporter = InMemoryLogRecordExporter()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    set_logger_provider(provider)
    root = logging.getLogger()
    before = list(root.handlers)
    setup_mod._install_logging()
    try:
        asyncio.run(breaker._llm_call("prompt"))
        provider.force_flush()
    finally:
        LoggingInstrumentor().uninstrument()
        for h in list(root.handlers):
            if h not in before:
                root.removeHandler(h)
    records = exporter.get_finished_logs()
    assert records, "no record reached the application's logger provider"
    dumped = json.dumps(
        [dict(r.log_record.attributes or {}) | {"body": r.log_record.body} for r in records],
        default=str,
    )
    assert _LOG_SECRET not in dumped, dumped
    assert "RuntimeError" in dumped


def test_the_exporting_handler_gets_the_template_and_the_console_the_whole_message() -> None:
    """The exporting handler's copy drops the arguments and names an exception
    argument's type; the record the other handlers see still formats in full."""
    from particles.observability import setup as setup_mod

    seen: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record)

    handler = _Capture()
    setup_mod._keep_exception_type_only_in_logs(handler)
    exc = ValueError(_LOG_SECRET)
    record = logging.LogRecord("x", logging.ERROR, "f", 1, "failed: %s", (exc,), None)
    handler.emit(record)
    (exported,) = seen
    assert exported.getMessage() == "failed: %s"
    assert exported.__dict__["exception.type"] == "ValueError"
    assert _LOG_SECRET in record.getMessage()
    plain = logging.LogRecord("x", logging.WARNING, "f", 1, "no arguments", None, None)
    handler.emit(plain)
    assert seen[-1] is plain

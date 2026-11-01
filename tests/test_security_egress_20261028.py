# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the egress and telemetry findings of security review 2026-10-28-1.

G52 (K10): the pinned subprocess's PATH never searches the working directory,
and an entry another user can repoint or rename is dropped.
E22 (G10): no part of an endpoint password survives the printed URL, whatever
the host's shape.
G15: the native Anthropic client and the OTLP exporter session write no TLS
key log, checked with a completed handshake.
G11 (W1): the per-client engine key is masked after any escape.
The logs signal ships a logged exception's type and no text.

Every test runs under a temporary ``HOME``, opens no real store and makes no
model call or outside connection: the TLS server is a loopback thread with a
throwaway certificate, and ``network.egress`` allows only ``127.0.0.1``.
"""

from __future__ import annotations

import datetime
import ipaddress
import json
import logging
import os
import secrets
import socket
import ssl
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from particles import url_safety
from particles.config import EgressAllowEntry, NetworkConfig, get_config, reset_config
from particles.corpus.redaction import redact_secrets
from particles.url_safety import url_without_userinfo


@pytest.fixture(autouse=True)
def _temporary_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


# --------------------------------------------------------------------------- #
# G52: PATH entries for the pinned curl and git subprocesses
# --------------------------------------------------------------------------- #


def test_a_path_with_every_entry_dropped_never_searches_the_working_directory(
    tmp_path: Path,
) -> None:
    """An empty ``PATH`` is ``['']`` to the program lookup, which is the working directory."""
    untrusted = tmp_path / "open"
    untrusted.mkdir()
    os.chmod(untrusted, 0o777)
    try:
        env = url_safety.pinned_subprocess_env({"PATH": f"{untrusted}{os.pathsep}relative/bin"})
    finally:
        os.chmod(untrusted, 0o755)
    exec_path = os.get_exec_path(env)
    assert "" not in exec_path, env
    assert str(untrusted) not in exec_path and "relative/bin" not in exec_path


def test_a_path_entry_that_is_another_users_link_is_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A link another user owns can be repointed after the check, so it is dropped.

    Another user is simulated by patching the effective uid, so the link this
    test creates belongs to "someone else".
    """
    link = tmp_path / "bin"
    link.symlink_to("/usr/bin")
    real_uid = os.geteuid()
    monkeypatch.setattr(os, "geteuid", lambda: real_uid + 4242)
    env = url_safety.pinned_subprocess_env({"PATH": str(link)})
    assert str(link) not in env["PATH"].split(os.pathsep), env


def test_a_path_entry_under_a_directory_others_may_write_is_dropped(tmp_path: Path) -> None:
    """A parent others may write without the sticky bit lets them rename the entry away."""
    shared = tmp_path / "shared"
    shared.mkdir()
    entry = shared / "bin"
    entry.mkdir(mode=0o755)
    os.chmod(shared, 0o777)
    try:
        env = url_safety.pinned_subprocess_env({"PATH": str(entry)})
    finally:
        os.chmod(shared, 0o755)
    assert str(entry) not in env["PATH"].split(os.pathsep), env


def test_a_trusted_entry_is_kept(tmp_path: Path) -> None:
    own = tmp_path / "own"
    own.mkdir(mode=0o755)
    env = url_safety.pinned_subprocess_env({"PATH": str(own)})
    assert env["PATH"] == str(own)


def test_with_no_trusted_entry_left_no_program_is_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "defpath", "relative/only")
    env = url_safety.pinned_subprocess_env({"PATH": "also/relative"})
    assert env["PATH"] == os.devnull
    assert "" not in os.get_exec_path(env)


# --------------------------------------------------------------------------- #
# E22, G10: endpoint passwords before any host shape
# --------------------------------------------------------------------------- #

_PW_TAIL = "audit8pw"


@pytest.mark.parametrize(
    "url",
    [
        f"http://svc:4417/{_PW_TAIL}@engine/v1",
        f"http://svc:4417/{_PW_TAIL}@localhost",
        f"http://svc:4417?{_PW_TAIL}@engine.example",
        f"http://svc:4417#{_PW_TAIL}@engine.example",
        f"svc:pa/{_PW_TAIL}@engine.example:8443",
    ],
)
def test_no_password_shape_is_echoed(url: str) -> None:
    """No part of a password after its first separator survives, whatever the host's shape."""
    assert _PW_TAIL not in url_without_userinfo(url)


def test_the_memory_facade_refusal_returns_no_password_before_a_single_label_host() -> None:
    """The remote-engine refusal the model reads carries no password for a container-name host."""
    from particles.mcp.memory_compat import server

    get_config().engine.base_url = f"http://svc:4417/{_PW_TAIL}@engine"
    try:
        message = server._remote_engine_refusal()
    finally:
        reset_config()
    assert message is not None
    assert _PW_TAIL not in message


def test_an_endpoint_origin_never_reads_a_password_as_a_port() -> None:
    from particles.config import _endpoint_origin

    assert _endpoint_origin(f"http://svc:4417/{_PW_TAIL}@engine") == "(unparseable)"
    assert _endpoint_origin("https://proxy.example.test:3128") == "https://proxy.example.test:3128"


# --------------------------------------------------------------------------- #
# G15: a completed TLS handshake writes nothing to SSLKEYLOGFILE
# --------------------------------------------------------------------------- #


def _cert(tmp: Path) -> tuple[Path, Path]:
    x509 = pytest.importorskip("cryptography.x509")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = tmp / "cert.pem", tmp / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


@pytest.fixture
def tls_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[int, Path]]:
    """A loopback TLS server answering 404 to every request; ``SSLKEYLOGFILE`` set."""
    cert_path, key_path = _cert(tmp_path)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_path, key_path)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(8)
    port = sock.getsockname()[1]
    stop = threading.Event()

    def serve() -> None:
        sock.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = sock.accept()
            except OSError:
                continue
            try:
                with ctx.wrap_socket(conn, server_side=True) as tls:
                    tls.settimeout(2)
                    data = b""
                    while b"\r\n\r\n" not in data:
                        chunk = tls.recv(65536)
                        if not chunk:
                            break
                        data += chunk
                    head, _, rest = data.partition(b"\r\n\r\n")
                    length = 0
                    for line in head.split(b"\r\n"):
                        if line.lower().startswith(b"content-length:"):
                            length = int(line.split(b":", 1)[1])
                    while len(rest) < length:
                        chunk = tls.recv(65536)
                        if not chunk:
                            break
                        rest += chunk
                    body = b'{"type":"error","error":{"type":"not_found_error","message":"x"}}'
                    tls.sendall(
                        b"HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n"
                        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                        b"Connection: close\r\n\r\n" + body
                    )
            except OSError:
                pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    monkeypatch.setattr(
        get_config(),
        "network",
        NetworkConfig(egress="allowlist", allow=[EgressAllowEntry(host="127.0.0.1")]),
    )
    monkeypatch.setenv("SSLKEYLOGFILE", str(tmp_path / "keys.log"))
    try:
        yield port, cert_path
    finally:
        stop.set()
        thread.join(2)
        sock.close()


def _secret_lines(log: Path) -> list[str]:
    if not log.exists():
        return []
    return [ln for ln in log.read_text().splitlines() if ln and not ln.startswith("#")]


def test_the_native_anthropic_client_writes_no_tls_key_log(
    tls_server: tuple[int, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A handshake made with the context the client's pool holds writes no session secret.

    With ``SSL_CERT_FILE`` set, httpx2 would verify with
    ``ssl.create_default_context``, which honours ``SSLKEYLOGFILE``. The
    suite's guard refuses any request the SDK sends, so the handshake is made
    with the pool's own context directly.
    """
    from particles.llm import client as client_mod

    port, cert_path = tls_server
    monkeypatch.setenv("SSL_CERT_FILE", str(cert_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-audit8-keylog-0000000000")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"https://127.0.0.1:{port}")
    client_mod.set_client(None)
    try:
        context = client_mod.get_client()._client._transport._pool._ssl_context
    finally:
        client_mod.set_client(None)
    with (
        socket.create_connection(("127.0.0.1", port), timeout=5) as raw,
        context.wrap_socket(raw, server_hostname="127.0.0.1") as tls,
    ):
        tls.sendall(b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 0\r\n\r\n")
        tls.recv(65536)
    assert _secret_lines(tmp_path / "keys.log") == []


@pytest.mark.parametrize("headers", [True, False])
def test_the_otlp_exporter_session_writes_no_tls_key_log(
    headers: bool, tls_server: tuple[int, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed handshake through the session Particles builds for an OTLP
    exporter, with the collector bearer or without, writes no session secret."""
    pytest.importorskip("requests")
    from particles.observability import setup as obs

    port, cert_path = tls_server
    monkeypatch.setenv("PARTICLES_OBSERVABILITY_ENDPOINT", f"https://127.0.0.1:{port}")
    if headers:
        monkeypatch.setenv("PARTICLES_OTEL_EXPORTER_HEADERS", "authorization=Bearer audit8")
    network = get_config().network
    reset_config()
    monkeypatch.setattr(get_config(), "network", network)
    kwargs = obs._otlp_kwargs(get_config().observability, "traces")
    assert kwargs is not None
    response = kwargs["session"].post(
        kwargs["endpoint"], data=b"x", verify=str(cert_path), timeout=5
    )
    assert response.status_code == 404
    assert _secret_lines(tmp_path / "keys.log") == []


# --------------------------------------------------------------------------- #
# G11, W1: the engine-key shape after any escape
# --------------------------------------------------------------------------- #


def _key() -> str:
    return f"ptk_agent_{secrets.token_hex(6)}_{secrets.token_urlsafe(32)[:43]}"


@pytest.mark.parametrize("before", ["%253D", "&#61", "\\n", "\\u0022", "=3D", "x"])
def test_a_key_after_an_escape_ending_in_an_alphanumeric_is_masked(before: str) -> None:
    """A per-client engine key is masked by its shape whatever escape precedes it."""
    key = _key()
    text = f"curl -d 'x{before}{key}' https://engine.example/query"
    assert key[-43:] not in redact_secrets(text)


# --------------------------------------------------------------------------- #
# The logs signal ships an exception's type and no text
# --------------------------------------------------------------------------- #

_SECRET = "audit8-log-secret-7f3e2a"


def test_the_logs_signal_carries_no_exception_text_when_a_logger_provider_exports() -> None:
    """With an exporting LoggerProvider, a logged exception ships no message or stacktrace."""
    pytest.importorskip("opentelemetry.sdk._logs")
    pytest.importorskip("opentelemetry.instrumentation.logging")
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor

    from particles.observability import setup as setup_mod

    provider = LoggerProvider()
    exporter = InMemoryLogRecordExporter()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    set_logger_provider(provider)
    root = logging.getLogger()
    before = list(root.handlers)
    setup_mod._install_logging()
    try:
        try:
            raise RuntimeError(f"fetch failed for https://h.example/?q={_SECRET}")
        except RuntimeError:
            logging.getLogger("particles.corpus.deposit").exception("deposit failed")
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
    assert _SECRET not in dumped, dumped
    assert "RuntimeError" in dumped


def test_the_logs_handler_keeps_the_console_record_whole() -> None:
    """The handler gets a copy without the exception; the record other handlers see is unchanged."""
    from particles.observability import setup as setup_mod

    seen: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record)

    handler = _Capture()
    setup_mod._keep_exception_type_only_in_logs(handler)
    try:
        raise ValueError(_SECRET)
    except ValueError:
        import sys

        record = logging.LogRecord("x", logging.ERROR, "f", 1, "m", None, sys.exc_info())
    handler.emit(record)
    (copy,) = seen
    assert copy.exc_info is None and copy.__dict__["exception.type"] == "ValueError"
    assert record.exc_info is not None

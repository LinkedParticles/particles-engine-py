# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""The real-dependency halves of three regression tests whose doubles covered less.

Security review 2026-10-28-2 (T1): the native client's key-log test made its
handshake with the pool's context directly, the OTLP key-log test built the
traces exporter's session only, and the projection walk's tests patched the
platform's open flags. Each test here drives the real shape instead: a request
through the SDK client's own transport, every OTLP signal's session, and the
walk on this platform's own flags. Each runs under a temporary ``HOME`` and
connects only to the loopback servers the fixtures start.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from particles.config import get_config, reset_config
from tests.test_security_egress_20261028 import _secret_lines, tls_server  # noqa: F401


@pytest.fixture(autouse=True)
def _temporary_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


def test_a_request_through_the_native_clients_own_transport_writes_no_tls_key_log(
    tls_server: tuple[int, Path],  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request the SDK client's transport sends, verified the way the transport
    verifies it, completes its handshake and writes no session secret.

    The suite's guard refuses any request the SDK client sends, so the request
    goes to the client's own transport, past the client but through its pool,
    its TLS context and its certificate check.
    """
    import httpx2

    from particles.llm import client as client_mod

    port, cert_path = tls_server
    monkeypatch.setenv("SSL_CERT_FILE", str(cert_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-audit9-keylog-0000000000")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"https://127.0.0.1:{port}")
    client_mod.set_client(None)
    try:
        transport = client_mod.get_client()._client._transport
        request = httpx2.Request("POST", f"https://127.0.0.1:{port}/v1/messages", content=b"{}")
        response = transport.handle_request(request)
        response.read()
    finally:
        client_mod.set_client(None)
    assert response.status_code == 404
    assert _secret_lines(tmp_path / "keys.log") == []


def test_the_native_clients_transport_refuses_a_certificate_it_cannot_verify(
    tls_server: tuple[int, Path],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: the same transport, without the server's certificate trusted, refuses
    the handshake, so the test above passes on a verified connection."""
    import httpx2

    from particles.llm import client as client_mod

    port, _cert_path = tls_server
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-audit9-keylog-0000000000")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"https://127.0.0.1:{port}")
    client_mod.set_client(None)
    try:
        transport = client_mod.get_client()._client._transport
        request = httpx2.Request("POST", f"https://127.0.0.1:{port}/v1/messages", content=b"{}")
        with pytest.raises(httpx2.ConnectError):
            transport.handle_request(request)
    finally:
        client_mod.set_client(None)


@pytest.mark.parametrize("signal", ["traces", "metrics"])
@pytest.mark.parametrize("headers", [True, False])
def test_every_otlp_signals_session_writes_no_tls_key_log(
    signal: str,
    headers: bool,
    tls_server: tuple[int, Path],  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The session Particles builds for each OTLP exporter it creates, with the
    collector bearer or without, completes a handshake and writes no session secret."""
    pytest.importorskip("requests")
    from particles.observability import setup as obs

    port, cert_path = tls_server
    monkeypatch.setenv("PARTICLES_OBSERVABILITY_ENDPOINT", f"https://127.0.0.1:{port}")
    if headers:
        monkeypatch.setenv("PARTICLES_OTEL_EXPORTER_HEADERS", "authorization=Bearer audit9")
    network = get_config().network
    reset_config()
    monkeypatch.setattr(get_config(), "network", network)
    kwargs = obs._otlp_kwargs(get_config().observability, signal)
    assert kwargs is not None
    response = kwargs["session"].post(
        kwargs["endpoint"], data=b"x", verify=str(cert_path), timeout=5
    )
    assert response.status_code == 404
    assert _secret_lines(tmp_path / "keys.log") == []


def test_the_real_platform_walk_never_opens_the_root_by_name_under_an_unreadable_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On this platform's own open flags, unpatched, an ancestor searchable but not
    readable never makes the projection walk open the root by its absolute name."""
    from particles.operations.projection import paths

    locked = tmp_path / "locked"
    root = locked / "repo"
    (root / ".git").mkdir(parents=True)
    real_root = os.path.realpath(root)
    opened: list[tuple[str, int | None]] = []
    real_open = os.open

    def recording_open(path, flags, mode=0o777, *, dir_fd=None):  # type: ignore[no-untyped-def]
        opened.append((os.fsdecode(path), dir_fd))
        return real_open(path, flags, mode, dir_fd=dir_fd)

    locked.chmod(0o311)
    try:
        monkeypatch.setattr(paths.os, "open", recording_open)
        paths.write_text_no_follow(root / "sub" / "out.md", "projected\n", root=root)
    finally:
        monkeypatch.undo()
        locked.chmod(0o755)
    assert (root / "sub" / "out.md").read_text() == "projected\n"
    assert all(name != real_root for name, _fd in opened)
    assert all(fd is not None for name, fd in opened if name != os.sep)

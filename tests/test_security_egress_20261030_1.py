# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the tenth delta security audit's egress findings (2026-10-30-1).

TLS key logs on the hop to an ``https://`` proxy (the native Anthropic client,
the OTLP sessions, the Hugging Face download), the exporting log handler, the
``PATH`` the trusted subprocesses search, the query-latency child's module path,
the history commit's git environment, the shared-suffix list and the proxy
diagnostics. Each test runs under a temporary ``HOME``, makes no model call and
connects only to loopback servers it starts itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import socket
import ssl
import stat
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from particles import url_safety
from particles.config import get_config, reset_config
from tests.test_security_egress_20261028 import _cert, _secret_lines

_PROXY_VARS = (
    "NO_PROXY",
    "no_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
)


@pytest.fixture(autouse=True)
def _temporary_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for name in _PROXY_VARS:
        monkeypatch.delenv(name, raising=False)


def _load_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    cfg = tmp_path / "cfg" / "config.yaml"
    cfg.parent.mkdir(mode=0o700, exist_ok=True)
    cfg.write_text(text)
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    reset_config()


@pytest.fixture
def tls_proxy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[int, Path]]:
    """A loopback TLS listener answering 404 to any request (a stand-in ``https://`` proxy).

    ``SSLKEYLOGFILE`` names ``tmp_path/keys.log`` for the test.
    """
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
                    # Read the body too: urllib3 sends it as its own TLS record,
                    # and closing with it unread makes the kernel reset the
                    # connection, which can discard the 404 before the client
                    # reads it (a CI flake, 2026-10).
                    head, _, body = data.partition(b"\r\n\r\n")
                    length = 0
                    for line in head.split(b"\r\n")[1:]:
                        name, _, value = line.partition(b":")
                        if name.strip().lower() == b"content-length":
                            length = int(value.strip())
                    while len(body) < length:
                        chunk = tls.recv(65536)
                        if not chunk:
                            break
                        body += chunk
                    tls.sendall(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    )
            except OSError:
                pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    monkeypatch.setenv("SSLKEYLOGFILE", str(tmp_path / "keys.log"))
    try:
        yield port, cert_path
    finally:
        stop.set()
        thread.join(2)
        sock.close()


# --------------------------------------------------------------------------- #
# E28 (AB9): the native Anthropic client's hop to an https:// proxy
# --------------------------------------------------------------------------- #


def _anthropic_proxy_transport(monkeypatch: pytest.MonkeyPatch, proxy_port: int) -> object:
    import httpx2

    from particles.llm import client as client_mod

    monkeypatch.setenv("HTTPS_PROXY", f"https://127.0.0.1:{proxy_port}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-sec10-keylog-000000000000")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    client_mod.set_client(None)
    try:
        http = client_mod.get_client()._client
    finally:
        client_mod.set_client(None)
    return http._transport_for_url(httpx2.URL("https://destination.invalid/v1/messages"))


def test_the_native_anthropic_clients_hop_to_an_https_proxy_writes_no_tls_key_log(
    tls_proxy: tuple[int, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``SSL_CERT_FILE`` set, the native Anthropic client's TLS session with an
    ``https://`` proxy from ``HTTPS_PROXY`` writes no session secret to ``SSLKEYLOGFILE``."""
    import httpx2

    port, cert_path = tls_proxy
    monkeypatch.setenv("SSL_CERT_FILE", str(cert_path))
    transport = _anthropic_proxy_transport(monkeypatch, port)
    with pytest.raises(httpx2.TransportError):  # the stand-in proxy refuses CONNECT
        transport.handle_request(  # type: ignore[attr-defined]
            httpx2.Request("GET", "https://destination.invalid/v1/messages")
        )
    assert _secret_lines(tmp_path / "keys.log") == []


def test_without_a_cert_variable_the_anthropic_proxy_hop_writes_no_key_log(
    tls_proxy: tuple[int, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``SSL_CERT_FILE`` or ``SSL_CERT_DIR`` the proxy hop writes no key log either."""
    import httpx2

    port, _cert_path = tls_proxy
    transport = _anthropic_proxy_transport(monkeypatch, port)
    with pytest.raises(httpx2.TransportError):
        transport.handle_request(  # type: ignore[attr-defined]
            httpx2.Request("GET", "https://destination.invalid/v1/messages")
        )
    assert _secret_lines(tmp_path / "keys.log") == []


@pytest.mark.asyncio
async def test_the_httpx_clients_take_no_environment_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fetch client and ``egress_client`` build no proxy mount from the environment."""
    from particles.http import egress_client, particles_client

    monkeypatch.setenv("HTTPS_PROXY", "https://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "https://127.0.0.1:9")
    monkeypatch.setenv("ALL_PROXY", "https://127.0.0.1:9")
    async with particles_client() as fetch:
        assert fetch._mounts == {}
    client = egress_client(purpose="llm endpoint")
    try:
        assert client._mounts == {}
    finally:
        await client.aclose()


# --------------------------------------------------------------------------- #
# The completeness critic's E28-class item: the Hugging Face encoder download
# --------------------------------------------------------------------------- #


def test_the_hub_clients_hop_to_an_https_proxy_writes_no_tls_key_log(
    tls_proxy: tuple[int, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``SSL_CERT_FILE`` set, the encoder download's TLS session with an
    ``https://`` proxy writes no session secret to ``SSLKEYLOGFILE``."""
    pytest.importorskip("huggingface_hub")
    import httpx2

    from particles.http import hub_client

    port, cert_path = tls_proxy
    monkeypatch.setenv("SSL_CERT_FILE", str(cert_path))
    monkeypatch.setenv("HTTPS_PROXY", f"https://127.0.0.1:{port}")
    client = hub_client()
    try:
        transport = client._transport_for_url(httpx2.URL("https://huggingface.invalid/x"))
        with pytest.raises(httpx2.TransportError):
            transport.handle_request(httpx2.Request("GET", "https://huggingface.invalid/x"))
    finally:
        client.close()
    assert _secret_lines(tmp_path / "keys.log") == []


def test_the_hub_clients_own_tls_session_writes_no_key_log(
    tls_proxy: tuple[int, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``SSL_CERT_FILE`` set and no proxy, the download's own session writes no
    session secret, and the server it trusts still answers."""
    hub = pytest.importorskip("huggingface_hub")
    from particles.http import hub_client

    port, cert_path = tls_proxy
    monkeypatch.setenv("SSL_CERT_FILE", str(cert_path))
    # CI sets HF_HUB_OFFLINE, and the Hub's own request hook refuses every
    # request under it; this test sends one to a loopback server only.
    monkeypatch.setattr(hub.constants, "is_offline_mode", lambda: False)
    client = hub_client()
    try:
        response = client.get(f"https://127.0.0.1:{port}/model")
    finally:
        client.close()
    assert response.status_code == 404
    assert _secret_lines(tmp_path / "keys.log") == []


def test_the_encoder_load_installs_the_hub_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Before a download, the Hub is given the client that writes no key log."""
    hub = pytest.importorskip("huggingface_hub")
    from particles import embeddings
    from particles.http import hub_client

    installed: list[object] = []
    monkeypatch.setattr(hub, "set_client_factory", installed.append)
    embeddings._use_hub_client_writing_no_key_log()
    assert installed == [hub_client]


# --------------------------------------------------------------------------- #
# AA11 (AB6): the OTLP session's hop to an https:// proxy
# --------------------------------------------------------------------------- #


def _otlp_session_for(
    endpoint: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[object, str]:
    from particles.observability import setup as obs

    _load_config(
        tmp_path,
        monkeypatch,
        "network:\n  egress: allowlist\n  allow:\n    - host: collector.invalid\n"
        "    - host: 127.0.0.1\n",
    )
    monkeypatch.setenv("PARTICLES_OBSERVABILITY_ENDPOINT", endpoint)
    monkeypatch.delenv("PARTICLES_OTEL_EXPORTER_HEADERS", raising=False)
    reset_config()
    kwargs = obs._otlp_kwargs(get_config().observability, "traces")
    assert kwargs is not None
    return kwargs["session"], kwargs["endpoint"]


def test_an_http_otlp_endpoints_hop_to_an_https_proxy_writes_no_tls_key_log(
    tls_proxy: tuple[int, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The header-less OTLP session for an ``http://`` collector, sent through an
    ``https://`` ``HTTP_PROXY``, writes no session secret to ``SSLKEYLOGFILE``."""
    pytest.importorskip("requests")
    import requests.utils

    port, cert_path = tls_proxy
    monkeypatch.setattr(requests.utils, "DEFAULT_CA_BUNDLE_PATH", str(cert_path))
    monkeypatch.setenv("HTTP_PROXY", f"https://127.0.0.1:{port}")
    session, endpoint = _otlp_session_for("http://collector.invalid:4318", tmp_path, monkeypatch)
    response = session.post(endpoint, data=b"x", timeout=5)  # type: ignore[attr-defined]
    assert response.status_code == 404  # the request went through the TLS proxy
    assert _secret_lines(tmp_path / "keys.log") == []


def test_an_http_otlp_endpoints_hop_to_an_https_proxy_verifies_the_proxy(
    tls_proxy: tuple[int, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ``https://`` proxy whose certificate the bundle does not trust is refused
    for an ``http://`` collector, as it is for an ``https://`` one."""
    requests = pytest.importorskip("requests")

    port, _cert_path = tls_proxy
    monkeypatch.setenv("HTTP_PROXY", f"https://127.0.0.1:{port}")
    session, endpoint = _otlp_session_for("http://collector.invalid:4318", tmp_path, monkeypatch)
    with pytest.raises(requests.exceptions.RequestException):
        session.post(endpoint, data=b"x", timeout=5)  # type: ignore[attr-defined]


def test_an_https_otlp_endpoints_hop_to_an_https_proxy_writes_no_tls_key_log(
    tls_proxy: tuple[int, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``https://`` half: no session secret reaches ``SSLKEYLOGFILE``."""
    pytest.importorskip("requests")
    import requests.utils

    port, cert_path = tls_proxy
    monkeypatch.setattr(requests.utils, "DEFAULT_CA_BUNDLE_PATH", str(cert_path))
    monkeypatch.setenv("HTTPS_PROXY", f"https://127.0.0.1:{port}")
    session, endpoint = _otlp_session_for("https://collector.invalid:4318", tmp_path, monkeypatch)
    with pytest.raises(requests.exceptions.ProxyError):
        session.post(endpoint, data=b"x", timeout=5)  # type: ignore[attr-defined]
    assert _secret_lines(tmp_path / "keys.log") == []


# --------------------------------------------------------------------------- #
# AA9 (AB58): what the exporting log handler ships
# --------------------------------------------------------------------------- #

_LOG_SECRET = "sec10-userinfo-secret-7c2d"


def _exporting_handler() -> tuple[logging.Handler, object, object]:
    pytest.importorskip("opentelemetry.instrumentation.logging.handler")
    from opentelemetry.instrumentation.logging.handler import LoggingHandler
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor

    from particles.observability import setup as setup_mod

    provider = LoggerProvider()
    exporter = InMemoryLogRecordExporter()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    handler = LoggingHandler(logger_provider=provider)
    setup_mod._keep_exception_type_only_in_logs(handler)
    return handler, provider, exporter


def _exported(exporter: object) -> list[dict[str, object]]:
    records = exporter.get_finished_logs()  # type: ignore[attr-defined]
    assert records
    return [dict(r.log_record.attributes or {}) | {"body": r.log_record.body} for r in records]


def test_an_exported_log_record_carries_no_extra_field_a_log_call_attached() -> None:
    """A record whose ``extra=`` field holds an endpoint with userinfo is exported
    without that field's value."""
    handler, provider, exporter = _exporting_handler()
    logger = logging.getLogger("sec10.extra")
    record = logger.makeRecord(
        logger.name,
        logging.WARNING,
        __file__,
        1,
        "endpoint refused: %s",
        ("x",),
        None,
        extra={"endpoint": f"https://user:{_LOG_SECRET}@collector.example"},
    )
    handler.emit(record)
    provider.force_flush()  # type: ignore[attr-defined]
    assert _LOG_SECRET not in json.dumps(_exported(exporter), default=str)


def test_an_exported_log_record_without_arguments_drops_its_extra_fields_too() -> None:
    """The ``extra=`` field goes on a record with no arguments or exception as well,
    and the record's own template and level still reach the exporter."""
    handler, provider, exporter = _exporting_handler()
    logger = logging.getLogger("sec10.extra.plain")
    record = logger.makeRecord(
        logger.name,
        logging.WARNING,
        __file__,
        1,
        "endpoint refused",
        (),
        None,
        extra={"response_body": _LOG_SECRET},
    )
    handler.emit(record)
    provider.force_flush()  # type: ignore[attr-defined]
    exported = _exported(exporter)
    assert _LOG_SECRET not in json.dumps(exported, default=str)
    assert exported[0]["body"] == "endpoint refused"


# --------------------------------------------------------------------------- #
# AB5: a PATH link chain whose middle hop sits behind a directory link
# --------------------------------------------------------------------------- #


def test_a_path_entry_whose_middle_hop_lies_behind_a_link_into_an_open_directory_is_dropped(
    tmp_path: Path,
) -> None:
    """A ``PATH`` entry whose link chain passes through a hop whose directory is reached
    through a link into a directory other users may rename entries of is dropped."""
    own = tmp_path / "own"
    own.mkdir(mode=0o700)
    open_dir = tmp_path / "open"
    open_dir.mkdir()
    sub = open_dir / "sub"
    sub.mkdir(mode=0o755)
    (sub / "hop").symlink_to("/usr/bin")
    (own / "dirlink").symlink_to(sub)
    entry = own / "bin"
    entry.symlink_to(own / "dirlink" / "hop")
    os.chmod(open_dir, 0o777)
    try:
        assert os.stat(open_dir).st_mode & stat.S_IWOTH
        assert not os.stat(open_dir).st_mode & stat.S_ISVTX
        kept = url_safety._only_trusted_users_write(str(entry))
    finally:
        os.chmod(open_dir, 0o755)
    assert kept is False


def test_a_path_entry_whose_link_chain_stays_in_closed_directories_is_kept(
    tmp_path: Path,
) -> None:
    """Control: the same chain, with no directory others may write, is kept."""
    own = tmp_path / "own"
    own.mkdir(mode=0o700)
    closed = tmp_path / "closed"
    closed.mkdir(mode=0o755)
    sub = closed / "sub"
    sub.mkdir(mode=0o755)
    (sub / "hop").symlink_to("/usr/bin")
    (own / "dirlink").symlink_to(sub)
    entry = own / "bin"
    entry.symlink_to(own / "dirlink" / "hop")
    if any(url_safety._an_ancestor_is_open(str(p)) for p in (tmp_path / "x", Path("/usr/bin/x"))):
        pytest.skip("the temporary directory or /usr/bin sits under a directory others may write")
    assert url_safety._only_trusted_users_write(str(entry)) is True


# --------------------------------------------------------------------------- #
# E31 / AA13 (AB8, AB24): children that searched the inherited PATH
# --------------------------------------------------------------------------- #


def _plant(directory: Path, name: str, marker: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    prog = directory / name
    prog.write_text(f"#!/bin/sh\n/usr/bin/touch '{marker}'\nexit 0\n")
    prog.chmod(0o755)


@pytest.fixture
def open_path_dir(tmp_path: Path) -> Iterator[Path]:
    """A ``PATH`` directory every user may write, which ``pinned_subprocess_env`` drops."""
    planted = tmp_path / "shared-bin"
    planted.mkdir()
    os.chmod(planted, 0o777)
    try:
        assert not url_safety._only_trusted_users_write(str(planted))
        yield planted
    finally:
        os.chmod(planted, 0o755)


def test_the_hook_doctors_git_runs_no_program_from_an_entry_the_pin_drops(
    open_path_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hook doctor's git query finds git only on ``PATH`` entries the pin keeps."""
    from particles.api.cli import hook

    marker = tmp_path / "planted-git-ran"
    _plant(open_path_dir, "git", marker)
    monkeypatch.setenv("PATH", f"{open_path_dir}{os.pathsep}/usr/bin{os.pathsep}/bin")
    hook._git(tmp_path, "rev-parse", "--abbrev-ref", "HEAD")
    assert not marker.exists(), "hook doctor ran a git planted on an untrusted PATH entry"


def test_the_hook_doctors_git_runs_no_program_a_clone_ships_on_a_relative_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hook doctor's git query never runs a program found through a relative ``PATH``
    entry in a cloned repository."""
    from particles.api.cli import hook

    clone = tmp_path / "clone"
    marker = tmp_path / "planted-git-ran"
    _plant(clone / "bin", "git", marker)
    monkeypatch.chdir(clone)
    monkeypatch.setenv("PATH", "bin" + os.pathsep + os.environ.get("PATH", os.defpath))
    hook._git(tmp_path, "rev-parse", "HEAD")
    assert not marker.exists()


def test_the_hook_doctors_git_still_answers_from_a_kept_entry(tmp_path: Path) -> None:
    """Control: with a trusted ``PATH``, the doctor's git query still answers."""
    from particles.api.cli import hook

    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    subprocess.run(["git", "init", "-q", str(tmp_path / "r")], check=True, env=env)
    assert hook._git(tmp_path / "r", "rev-parse", "--git-dir") == ".git"


def test_the_benchmark_environment_runs_no_program_from_an_entry_the_pin_drops(
    open_path_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The query-latency benchmark's environment probe runs git and sysctl only from
    ``PATH`` entries the pin keeps."""
    from particles.benchmark.query_latency import cell

    marker = tmp_path / "planted-program-ran"
    _plant(open_path_dir, "git", marker)
    _plant(open_path_dir, "sysctl", marker)
    monkeypatch.setenv("PATH", f"{open_path_dir}{os.pathsep}/usr/bin{os.pathsep}/bin")
    cell.environment()
    assert not marker.exists(), "the benchmark ran a program planted on an untrusted PATH entry"


def test_the_benchmark_runners_probes_run_no_program_from_an_entry_the_pin_drops(
    open_path_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner's ``ps`` and ``sysctl`` come only from ``PATH`` entries the pin keeps."""
    from particles.benchmark.query_latency import runner

    marker = tmp_path / "planted-program-ran"
    _plant(open_path_dir, "ps", marker)
    _plant(open_path_dir, "sysctl", marker)
    monkeypatch.setenv("PATH", f"{open_path_dir}{os.pathsep}/usr/bin{os.pathsep}/bin")
    asyncio.run(runner._rss_bytes(os.getpid()))
    runner.swap_used_bytes()
    assert not marker.exists()


def test_the_projection_git_runs_no_program_from_an_entry_the_pin_drops(
    open_path_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The projection's git comes only from kept ``PATH`` entries."""
    from particles.api.cli import _projection_git

    marker = tmp_path / "planted-git-ran"
    _plant(open_path_dir, "git", marker)
    monkeypatch.setenv("PATH", f"{open_path_dir}{os.pathsep}/usr/bin{os.pathsep}/bin")
    repo = tmp_path / "repo"
    repo.mkdir()
    with contextlib.suppress(Exception):  # only which git ran matters
        asyncio.run(_projection_git._git(repo, "status"))
    assert not marker.exists()


# --------------------------------------------------------------------------- #
# AA60's remainder (AB28): config files an inherited variable names
# --------------------------------------------------------------------------- #


def test_a_filter_driver_from_an_inherited_global_config_file_never_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A filter driver named by the config file an inherited ``GIT_CONFIG_GLOBAL``
    points at never runs while the history commit stages ``MEMORY.md``."""
    from particles.api.cli._projection_git import commit_projection

    for name in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(name)
    home = Path(os.environ["HOME"])
    (home / ".gitconfig").write_text(
        "[user]\n\tname = Test\n\temail = test@example.invalid\n[commit]\n\tgpgSign = false\n"
    )
    repo = tmp_path / "repo"
    memory = repo / "memory"
    memory.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=dict(os.environ))
    (memory / "MEMORY.md").write_text("- a belief `p-abcd1234`\n")

    marker = tmp_path / "filter-ran"
    attributes = tmp_path / "planted.attributes"
    attributes.write_text("*.md filter=probe\n")
    planted = tmp_path / "planted.gitconfig"
    planted.write_text(
        f"[core]\n\tattributesFile = {attributes}\n"
        f"[filter \"probe\"]\n\tclean = /usr/bin/touch '{marker}'; cat\n"
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(planted))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(planted))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    result = asyncio.run(
        commit_projection(
            memory, store="default", outcome="ok", body="", snapshot=None, run_id="r1"
        )
    )
    assert not marker.exists()
    assert result.startswith("committed"), result  # git add ran, without the driver


# --------------------------------------------------------------------------- #
# AB10: the query-latency cell child and its working directory
# --------------------------------------------------------------------------- #


def test_the_query_latency_cell_child_takes_no_module_from_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cell child is started so that the directory the benchmark runs in cannot
    supply its modules."""
    from particles.benchmark.query_latency import runner

    captured: dict[str, object] = {}

    class _Stop(Exception):
        pass

    async def _capture(*args: object, **kwargs: object) -> object:
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise _Stop

    class _Spec:
        cell_id = "c1"

        @staticmethod
        def model_dump_json(indent: int = 2) -> str:
            return "{}"

    monkeypatch.setattr(runner.asyncio, "create_subprocess_exec", _capture)
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(_Stop):
        asyncio.run(runner._run_child(_Spec(), object(), work, tmp_path / "c.yaml"))  # type: ignore[arg-type]
    args = [str(a) for a in captured["args"]]  # type: ignore[attr-defined]
    kwargs = captured["kwargs"]
    flags = set(args[1 : args.index("-m")])
    isolated = bool(flags & {"-I", "-P"}) or (
        kwargs.get("env", {}).get("PYTHONSAFEPATH")  # type: ignore[attr-defined]
    )
    assert isolated or kwargs.get("cwd") is not None, (args, sorted(kwargs))  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# L28 (AB57): a proxy the diagnostics never saw
# --------------------------------------------------------------------------- #


def test_the_diagnostics_flag_a_proxy_the_anthropic_client_takes_from_the_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proxy the native Anthropic client routes through is named by the environment
    diagnostics, including one taken from the macOS system settings."""
    import urllib.request

    from anthropic._utils._httpx import get_environment_proxies

    from particles.config import environment_notes

    if not hasattr(urllib.request, "getproxies_macosx_sysconf"):
        pytest.skip("not macOS")
    monkeypatch.setattr(
        urllib.request,
        "getproxies_macosx_sysconf",
        lambda: {"https": "http://proxy.corp.example:3128"},
    )
    assert get_environment_proxies().get("https://") == "http://proxy.corp.example:3128"
    notes = environment_notes()
    assert any("proxy" in note.lower() and "proxy.corp.example" in note for note in notes), notes


def test_the_diagnostics_name_no_system_proxy_when_a_variable_sets_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proxy variable is named once, and the system settings it overrides are not."""
    from particles.config import _system_proxy_notes, environment_notes

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.env.example:8080")
    assert _system_proxy_notes() == []
    assert sum("proxy.env.example" in note for note in environment_notes()) == 1


# --------------------------------------------------------------------------- #
# Z65 (AB7): hosted-platform suffixes the fixed list omitted
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("suffix", ["azurestaticapps.net", "myshopify.com", "lovable.app"])
def test_a_hosted_platform_suffix_counts_as_shared(suffix: str) -> None:
    """A wildcard over a hosting platform's shared suffix is judged shared."""
    from particles.config import _is_shared_suffix

    assert _is_shared_suffix(suffix)


def test_an_operators_own_domain_is_not_a_shared_suffix() -> None:
    """Control: a wildcard over the operator's own two-label domain still validates."""
    from particles.config import _is_shared_suffix

    assert not _is_shared_suffix("example.com")

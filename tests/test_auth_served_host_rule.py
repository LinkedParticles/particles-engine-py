# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The keyed Host rule follows the served address, and forwarded hops are parsed.

Security review 2026-10-27-1: G1 (a launch the startup checks judge wide but
that listens on loopback keeps the loopback-only Host rule) and G45 (a
forwarded hop written with a port or in brackets never makes a remote client
read as loopback).
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from particles.api import _keys
from particles.api._keys import Role
from particles.config import ApiConfig, get_config, reset_config

REBIND = {"Host": "rebind.attacker.example:8000", "Sec-Fetch-Site": "same-origin"}


@pytest.fixture
def keys_file(cli_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / "engine-keys.json"
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(path))
    monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
    monkeypatch.delenv("CLAUDECODE", raising=False)
    for name in ("UVICORN_HOST", "UVICORN_FD", "UVICORN_UDS", "UVICORN_PORT"):
        monkeypatch.delenv(name, raising=False)
    reset_config()
    _keys.reset_key_cache()
    yield path
    _keys.reset_key_cache()


def _client(peer: str = "127.0.0.1", base: str = "http://127.0.0.1:8000") -> TestClient:
    from particles.api.app import app

    return TestClient(app, base_url=base, client=(peer, 50000))


class TestServedAddressHostRule:
    """G1, and the W10 / N11 / Z28 remainders it closes."""

    def test_a_keyed_fd_launch_served_on_loopback_keeps_the_loopback_host_rule(
        self, keys_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _keys.mint(Role.READER, "viewer")
        monkeypatch.setattr(sys, "argv", ["uvicorn", "particles.api.app:app", "--fd", "3"])
        resp = _client().get("/particles", headers=REBIND)
        assert resp.status_code == 403, f"rebinding Host served {resp.status_code} on loopback"

    def test_a_keyed_script_launch_on_loopback_under_a_wide_bind_host_keeps_the_host_rule(
        self, keys_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _keys.mint(Role.READER, "viewer")
        monkeypatch.setattr(get_config().api, "bind_host", "0.0.0.0")
        resp = _client().get("/corpus", headers=REBIND)
        assert resp.status_code == 403, f"rebinding Host served {resp.status_code} on loopback"

    def test_a_keyed_engine_served_on_a_wide_address_still_answers_its_names(
        self, keys_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control: a wide listener with an empty api.allowed_hosts answers any name."""
        _keys.mint(Role.READER, "viewer")
        monkeypatch.setattr(get_config().api, "bind_host", "0.0.0.0")
        wide = _client("192.0.2.7", base="http://192.0.2.1:8000")
        assert wide.get("/particles", headers={"Host": "engine.example"}).status_code == 200
        monkeypatch.setattr(sys, "argv", ["uvicorn", "particles.api.app:app", "--fd", "3"])
        assert wide.get("/particles", headers={"Host": "engine.example"}).status_code == 200

    def test_an_inherited_socket_with_no_served_address_keeps_the_host_rule(
        self, keys_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A UNIX socket or inherited socket that names no address is judged loopback."""
        from particles.api.auth import _host_is_allowed

        _keys.mint(Role.READER, "viewer")
        monkeypatch.setattr(sys, "argv", ["uvicorn", "particles.api.app:app", "--uds", "/tmp/s"])
        assert not _host_is_allowed("rebind.attacker.example", dev_key=False, server=None)
        assert _host_is_allowed("127.0.0.1:8000", dev_key=False, server=None)


def _forwarded(xff: str, peer: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/particles",
            "headers": [(b"host", b"127.0.0.1:8000"), (b"x-forwarded-for", xff.encode())],
            "client": (peer, 40000),
            "server": (peer, 8000),
            "query_string": b"",
        }
    )


class TestForwardedHopAddresses:
    """G45, the W78 regression: a hop that is not a bare address is never skipped."""

    @pytest.mark.parametrize("hop", ["203.0.113.7:4711", "[2001:db8::7]", "[2001:db8::7]:443"])
    def test_a_remote_client_forwarded_with_a_port_is_never_read_as_loopback(
        self, keys_file: Path, monkeypatch: pytest.MonkeyPatch, hop: str
    ) -> None:
        from particles.api.auth import _client_is_loopback, request_bound_refusal

        # Assigned past the validator, as the audit's proof did: the strongest case.
        get_config().api.trusted_proxies = ["127.0.0.0/8"]
        assert not _client_is_loopback(_forwarded(hop, "127.0.0.1"))
        assert request_bound_refusal(_forwarded(hop, "127.0.0.1")) is not None

    @pytest.mark.parametrize("hop", ["203.0.113.7:4711", "[2001:db8::7]:443", "unknown"])
    def test_a_client_written_loopback_hop_left_of_a_non_bare_hop_is_never_the_client(
        self, monkeypatch: pytest.MonkeyPatch, hop: str
    ) -> None:
        """Under a trusted_proxies value the validator accepts."""
        from particles.api.auth import _client_is_loopback, _real_client_host

        validated = ApiConfig(trusted_proxies=["10.0.0.0/8"]).trusted_proxies
        monkeypatch.setattr(get_config().api, "trusted_proxies", validated)
        request = _forwarded(f"127.0.0.1, {hop}", "10.0.0.5")
        assert not _client_is_loopback(request)
        assert _real_client_host(request) != "127.0.0.1"

    @pytest.mark.parametrize(
        ("hop", "address"),
        [
            ("203.0.113.7:4711", "203.0.113.7"),
            ("[2001:db8::7]", "2001:db8::7"),
            ("[2001:db8::7]:443", "2001:db8::7"),
            ("203.0.113.7", "203.0.113.7"),
            ("2001:db8::7", "2001:db8::7"),
        ],
    )
    def test_a_ported_or_bracketed_hop_names_its_address(
        self, monkeypatch: pytest.MonkeyPatch, hop: str, address: str
    ) -> None:
        from particles.api.auth import _real_client_host

        monkeypatch.setattr(get_config().api, "trusted_proxies", ["10.0.0.0/8"])
        assert _real_client_host(_forwarded(f"10.0.0.7, {hop}", "10.0.0.5")) == address

    def test_a_remote_client_forwarded_as_a_bare_address_is_refused_on_a_keyless_engine(
        self, keys_file: Path
    ) -> None:
        """Control: the same client forwarded as a bare address is refused."""
        from particles.api.auth import request_bound_refusal

        get_config().api.trusted_proxies = ["127.0.0.0/8"]
        refusal = request_bound_refusal(_forwarded("203.0.113.7", "127.0.0.1"))
        assert refusal is not None and refusal[0] == 503

    def test_a_non_address_hop_never_names_a_principal_bucket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """W78 still holds: the hop's own spelling never reaches a bucket key."""
        from particles.api._rate_limit import _client_bucket

        monkeypatch.setattr(get_config().api, "trusted_proxies", ["10.0.0.0/8"])
        bucket = _client_bucket(_forwarded("principal:operator:local, 10.0.0.7", "10.0.0.2"))
        assert not bucket.startswith("principal:"), bucket

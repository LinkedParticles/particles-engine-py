# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The per-request bounds of particles/api/auth.py (security review 2026-10-26, W8 to W10).

- **W10**: a loopback-bound engine answers only the loopback ``Host`` names
  while ``api.allowed_hosts`` is empty, whether or not a key exists, so minting
  a key at runtime does not open the reads to a DNS-rebinding page.
  ``engine keys create`` says what the first key leaves open.
- **W9**: with no usable key the keyless loopback-client and loopback-``Host``
  bounds hold whatever lies at the key-file path.
- **W8**: a request its route's gate refuses whatever its body says is refused
  before the body is read.
- **N9 / N11**: a request arriving on an address wider than the bind the
  startup checks judged is logged once.

Everything runs through ``TestClient`` against a scratch store: no network, no
model call, no Anthropic key.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from particles.api import _keys
from particles.api._keys import Role
from particles.config import get_config, reset_config

#: The headers a same-origin fetch from a page under a rebinding host name carries.
REBIND = {"Host": "rebind.attacker.example:8000", "Sec-Fetch-Site": "same-origin"}
CROSS_SITE = {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"}


@pytest.fixture
def keys_file(cli_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A keyless engine whose key file lives under this test's ``tmp_path``."""
    path = tmp_path / "engine-keys.json"
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(path))
    monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
    reset_config()
    _keys.reset_key_cache()
    yield path
    _keys.reset_key_cache()


def _client(peer: str = "127.0.0.1") -> TestClient:
    from particles.api.app import app

    return TestClient(app, base_url="http://127.0.0.1:8000", client=(peer, 50000))


class TestMintingAKeyKeepsTheLoopbackHostRule:
    def test_keyless_loopback_engine_refuses_a_rebinding_host(self, keys_file: Path) -> None:
        assert _client().get("/particles", headers=REBIND).status_code == 403

    def test_minting_a_reader_key_keeps_the_rebinding_host_refused(self, keys_file: Path) -> None:
        client = _client()
        assert client.get("/particles", headers=REBIND).status_code == 403
        _keys.mint(Role.READER, "web-ui-phone")  # no restart, no config change
        after = client.get("/particles", headers=REBIND)
        corpus = client.get("/corpus", headers=REBIND)
        assert after.status_code == 403, f"served {after.status_code} to a rebinding Host"
        assert corpus.status_code == 403
        assert "api.allowed_hosts" in after.json()["detail"]
        assert client.get("/particles").status_code == 200

    def test_a_listed_host_is_served_on_a_keyed_loopback_engine(
        self, keys_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _keys.mint(Role.READER, "web-ui-phone")
        monkeypatch.setattr(get_config().api, "allowed_hosts", ["engine.example"])
        client = _client()
        assert client.get("/particles", headers={"Host": "engine.example"}).status_code == 200
        assert client.get("/particles", headers=REBIND).status_code == 403

    def test_a_wider_bind_serves_any_host_while_the_list_is_empty(
        self, keys_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.app import app

        _keys.mint(Role.READER, "web-ui-phone")
        monkeypatch.setattr(get_config().api, "bind_host", "0.0.0.0")
        # Served on a wide address: a request on a loopback socket keeps the
        # loopback Host rule whatever api.bind_host says (2026-10-27-1, G1).
        wide = TestClient(app, base_url="http://192.0.2.1:8000", client=("192.0.2.7", 50000))
        assert wide.get("/particles", headers=REBIND).status_code == 200
        assert _client().get("/particles", headers=REBIND).status_code == 403


class TestFirstKeyNamesTheReadPosture:
    def _create(self, *args: str) -> Any:
        from particles.api.cli import app

        # The suite may itself run in a Claude Code shell, and the runner's
        # stdout is not a terminal; both refuse without --print-token.
        return CliRunner().invoke(
            app,
            [
                "engine",
                "keys",
                "create",
                "--role",
                "reader",
                "--principal",
                "viewer",
                "--print-token",
                *args,
            ],
            env={"CLAUDECODE": None},
        )

    def test_the_first_key_warns_that_reads_stay_open(self, keys_file: Path) -> None:
        first = self._create()
        assert first.exit_code == 0, first.output
        assert "api.allowed_hosts" in first.output
        second = self._create("--label", "second")
        assert second.exit_code == 0, second.output
        assert "api.allowed_hosts" not in second.output

    def test_no_warning_when_reads_need_the_key(
        self, keys_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PARTICLES_API_REQUIRE_AUTH_FOR_READS", "true")
        reset_config()
        result = self._create()
        assert result.exit_code == 0, result.output
        assert "api.allowed_hosts" not in result.output


class TestAnUnusableKeyFileKeepsTheKeylessBounds:
    @pytest.mark.parametrize("damage", ["world-writable", "unparseable"])
    def test_an_unusable_key_file_keeps_the_keyless_read_bounds(
        self, keys_file: Path, damage: str
    ) -> None:
        loop = _client()
        remote = _client("192.0.2.10")
        assert loop.get("/particles", headers=REBIND).status_code == 403
        assert remote.get("/particles").status_code == 503
        if damage == "world-writable":
            keys_file.write_text(json.dumps({"version": 1, "keys": []}), encoding="utf-8")
            os.chmod(keys_file, 0o666)
        else:
            keys_file.write_text("not json", encoding="utf-8")
        rebind = loop.get("/particles", headers=REBIND)
        far = remote.get("/particles")
        assert rebind.status_code == 403, f"rebinding Host served {rebind.status_code}"
        assert far.status_code == 503, f"remote client served {far.status_code}"
        # The file still counts as keys for the write refusal: no write is served.
        deposit = loop.post("/corpus/deposit/text", json={"text": "x", "title": "t"})
        assert deposit.status_code in (401, 503)

    def test_a_usable_key_file_lifts_the_loopback_client_bound(self, keys_file: Path) -> None:
        _keys.mint(Role.READER, "web-ui-phone")
        assert _client("192.0.2.10").get("/particles").status_code == 200


class TestRefusedBeforeTheBodyIsRead:
    @pytest.fixture
    def parsed(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        import starlette.formparsers as fp

        calls: list[int] = []
        original = fp.MultiPartParser.parse

        async def _spy(self: Any) -> Any:
            calls.append(1)
            return await original(self)

        monkeypatch.setattr(fp.MultiPartParser, "parse", _spy)
        return calls

    def _upload(self, client: TestClient, headers: dict[str, str]) -> Any:
        return client.post(
            "/corpus/deposit/file",
            files={"file": ("a.txt", b"x" * (2 * 1024 * 1024), "text/plain")},
            headers=headers,
        )

    def test_a_refused_cross_site_upload_is_never_parsed(
        self, keys_file: Path, parsed: list[int]
    ) -> None:
        resp = self._upload(_client(), CROSS_SITE)
        assert resp.status_code == 403, resp.text
        assert "cross-site" in resp.json()["detail"]
        assert parsed == [], "the refused request's multipart body was parsed before the gate"

    def test_a_keyless_write_is_refused_unparsed(self, keys_file: Path, parsed: list[int]) -> None:
        resp = self._upload(_client(), {})
        assert resp.status_code == 503
        assert parsed == []

    def test_an_unauthenticated_upload_to_a_keyed_engine_is_refused_unparsed(
        self, keys_file: Path, parsed: list[int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
        assert self._upload(_client(), {}).status_code == 401
        wrong = self._upload(_client(), {"Authorization": "Bearer nope"})
        assert wrong.status_code == 401
        assert parsed == []

    def test_a_reader_key_upload_is_refused_unparsed(
        self, keys_file: Path, parsed: list[int]
    ) -> None:
        token, _ = _keys.mint(Role.READER, "viewer")
        resp = self._upload(_client(), {"Authorization": f"Bearer {token}"})
        assert resp.status_code == 403
        assert "agent or operator key" in resp.json()["detail"]
        assert parsed == []

    def test_an_agent_key_upload_is_parsed_and_served(
        self, keys_file: Path, parsed: list[int]
    ) -> None:
        token, _ = _keys.mint(Role.AGENT, "mcp:test-agent")
        resp = self._upload(_client(), {**CROSS_SITE, "Authorization": f"Bearer {token}"})
        assert resp.status_code == 200, resp.text
        assert parsed == [1]


class TestAWiderServedAddressIsLogged:
    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.api import auth

        monkeypatch.setattr(auth, "_wide_bind_logged_for", None)

    def test_keyed_open_reads_on_a_wider_address_warn_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from particles.api.auth import note_served_address

        monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
        with caplog.at_level(logging.WARNING, logger="particles.api.auth"):
            note_served_address(("127.0.0.1", 8000))
            note_served_address(("192.0.2.5", 8000))
            note_served_address(("192.0.2.5", 8000))
        warnings = [r for r in caplog.records if "require_auth_for_reads" in r.getMessage()]
        assert len(warnings) == 1
        assert "192.0.2.5" in warnings[0].getMessage()

    def test_keyless_engine_on_a_wider_address_says_so(
        self, keys_file: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from particles.api.auth import note_served_address

        with caplog.at_level(logging.WARNING, logger="particles.api.auth"):
            note_served_address(("192.0.2.5", 8000))
        assert any("wider than loopback" in r.getMessage() for r in caplog.records)

    def test_a_judged_wide_bind_is_not_repeated(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from particles.api.auth import note_served_address

        monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
        monkeypatch.setattr(get_config().api, "bind_host", "0.0.0.0")
        with caplog.at_level(logging.WARNING, logger="particles.api.auth"):
            note_served_address(("192.0.2.5", 8000))
            note_served_address(("testserver", 80))
        assert caplog.records == []

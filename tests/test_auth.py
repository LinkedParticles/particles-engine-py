# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for particles/api/auth.py.

``_verify_key`` reads ``PARTICLES_API_KEY`` at call time, not at module-import
time — the L2 architecture-review fix. Before the fix, the cached module-level
constant meant the key could not be rotated without restarting the process, and
tests could not change it via ``monkeypatch.setenv`` once any code path had
imported the module.

The fail-closed bearer-auth behaviour is also covered here: the
constant-time compare, the per-request non-loopback gate under dev-key, the
``_is_loopback_host`` classifier, and the ``enforce_fail_closed_on_startup``
boot check. The operator gate is covered too: the operator-scope
writes and review resolution refuse the dev-key skip even from loopback, while
the read routes and the agent's own-belief writes keep it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
from starlette.requests import Request

from particles.api.auth import (
    _is_loopback_host,
    _verify_key,
    enforce_fail_closed_on_startup,
)
from particles.config import get_config, reset_config
from tests._engine_client import BEARER, TEST_ENGINE_KEY


def _creds(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def _request(
    client: tuple[str, int] | None = ("127.0.0.1", 12345),
    xff: str | None = None,
) -> Request:
    """Minimal ASGI request with a chosen network peer (default: loopback).

    Pass ``xff`` to set an ``X-Forwarded-For`` header (proxy-aware gate tests).
    """
    headers: list[tuple[bytes, bytes]] = []
    if xff is not None:
        headers.append((b"x-forwarded-for", xff.encode()))
    return Request({"type": "http", "headers": headers, "client": client})


def test_dev_key_bypasses_auth_without_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
    _verify_key(_request(), None)  # loopback peer → no exception


def test_dev_key_bypasses_auth_with_arbitrary_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
    _verify_key(_request(), _creds("anything-goes"))


def test_real_key_rejects_missing_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
    with pytest.raises(HTTPException) as exc:
        _verify_key(_request(), None)
    assert exc.value.status_code == 401


def test_real_key_rejects_wrong_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
    with pytest.raises(HTTPException) as exc:
        _verify_key(_request(), _creds("wrong"))
    assert exc.value.status_code == 401


def test_real_key_accepts_matching_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
    _verify_key(_request(), _creds("prod-secret"))  # no exception


def test_real_key_rejects_prefix_of_correct_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A matching prefix must still reject — constant-time compare, no short-circuit."""
    monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
    with pytest.raises(HTTPException) as exc:
        _verify_key(_request(), _creds("prod"))
    assert exc.value.status_code == 401


def test_non_ascii_key_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """compare_digest's str path is ASCII-only; we encode to bytes so a
    non-ASCII key/credential rejects cleanly instead of raising a 500."""
    monkeypatch.setenv("PARTICLES_API_KEY", "pröd-sécret")
    _verify_key(_request(), _creds("pröd-sécret"))  # accepts
    with pytest.raises(HTTPException) as exc:
        _verify_key(_request(), _creds("wröng"))
    assert exc.value.status_code == 401


def test_key_change_takes_effect_without_reimport(monkeypatch: pytest.MonkeyPatch) -> None:
    """The L2 regression target: rotating the key must not require restart.

    Before the fix, the module-level constant was captured once at import,
    so the second monkeypatch.setenv had no effect — the test would have
    failed because the old key would still be active.
    """
    monkeypatch.setenv("PARTICLES_API_KEY", "key-one")
    _verify_key(_request(), _creds("key-one"))

    monkeypatch.setenv("PARTICLES_API_KEY", "key-two")
    with pytest.raises(HTTPException):
        _verify_key(_request(), _creds("key-one"))
    _verify_key(_request(), _creds("key-two"))


# ---------------------------------------------------------------------------
# Per-request loopback gate under dev-key (§(a) mechanism ii)
# ---------------------------------------------------------------------------


class TestDevKeyLoopbackGate:
    def test_non_loopback_client_refused_503(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        with pytest.raises(HTTPException) as exc:
            _verify_key(_request(client=("203.0.113.7", 5000)), None)
        assert exc.value.status_code == 503
        assert "non-loopback" in exc.value.detail

    def test_non_loopback_client_refused_even_with_creds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Under dev-key there is no real key, so presenting creds cannot help.
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        with pytest.raises(HTTPException) as exc:
            _verify_key(_request(client=("8.8.8.8", 443)), _creds("whatever"))
        assert exc.value.status_code == 503

    def test_ipv6_loopback_client_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        _verify_key(_request(client=("::1", 5000)), None)  # no exception

    def test_no_peer_client_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A UNIX-socket connection carries no peer; it is not provably local (R1).
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        with pytest.raises(HTTPException) as exc:
            _verify_key(_request(client=None), None)
        assert exc.value.status_code == 503

    def test_real_key_ignores_gate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # With a real key the normal bearer check runs regardless of peer; a
        # non-loopback client with the right key is accepted (401 on wrong key,
        # never 503).
        monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
        _verify_key(_request(client=("203.0.113.7", 5000)), _creds("prod-secret"))
        with pytest.raises(HTTPException) as exc:
            _verify_key(_request(client=("203.0.113.7", 5000)), _creds("nope"))
        assert exc.value.status_code == 401


# ---------------------------------------------------------------------------
# _is_loopback_host classifier
# ---------------------------------------------------------------------------


class TestIsLoopbackHost:
    @pytest.mark.parametrize(
        "host",
        ["127.0.0.1", "127.5.6.7", "::1", "localhost", "LOCALHOST", "[::1]", " 127.0.0.1 "],
    )
    def test_loopback(self, host: str) -> None:
        assert _is_loopback_host(host) is True

    @pytest.mark.parametrize(
        "host",
        ["0.0.0.0", "::", "203.0.113.7", "10.0.0.1", "example.com", "", "testclient"],
    )
    def test_not_loopback(self, host: str) -> None:
        assert _is_loopback_host(host) is False


# ---------------------------------------------------------------------------
# enforce_fail_closed_on_startup — boot-time bind-host check (§(a)(i))
# ---------------------------------------------------------------------------


class TestEnforceFailClosedOnStartup:
    def test_dev_key_loopback_bind_allows_boot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        monkeypatch.setenv("PARTICLES_API_BIND_HOST", "127.0.0.1")
        reset_config()
        enforce_fail_closed_on_startup()  # no exception

    def test_dev_key_default_bind_allows_boot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No bind_host override → default 127.0.0.1.
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        monkeypatch.delenv("PARTICLES_API_BIND_HOST", raising=False)
        reset_config()
        enforce_fail_closed_on_startup()  # no exception

    def test_dev_key_non_loopback_bind_refuses_boot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        monkeypatch.setenv("PARTICLES_API_BIND_HOST", "0.0.0.0")
        reset_config()
        with pytest.raises(RuntimeError) as exc:
            enforce_fail_closed_on_startup()
        assert "0.0.0.0" in str(exc.value)
        assert "PARTICLES_API_KEY" in str(exc.value)
        # A keyless engine refuses every write, so the refusal names
        # what a wide bind would expose: the reads (security review 2026-10-28-1).
        assert "write verbs" not in str(exc.value)
        assert "read routes" in str(exc.value)

    def test_real_key_non_loopback_bind_allows_boot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A real key is set → the bind host is irrelevant; serving publicly is fine.
        monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
        monkeypatch.setenv("PARTICLES_API_BIND_HOST", "0.0.0.0")
        reset_config()
        enforce_fail_closed_on_startup()  # no exception


class TestEmptyKeyIsNoKey:
    """Security review 2026-10-25, the empty-key residual.

    An empty or blank ``PARTICLES_API_KEY`` used to count as a key: the startup
    gate let a non-loopback bind through, the loopback bounds did not apply,
    reads were served to the network and writes got 401. It is now no key, so
    the engine is keyless and every keyless rule applies.
    """

    @pytest.mark.parametrize("value", ["", "   "])
    def test_empty_key_on_a_non_loopback_bind_refuses_boot(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("PARTICLES_API_KEY", value)
        monkeypatch.setenv("PARTICLES_API_BIND_HOST", "0.0.0.0")
        reset_config()
        with pytest.raises(RuntimeError) as exc:
            enforce_fail_closed_on_startup()
        assert "0.0.0.0" in str(exc.value)
        assert "empty" in str(exc.value)

    def test_empty_key_on_a_non_loopback_command_line_serves_nothing(
        self, cli_db: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        from particles.api.app import app

        for name in ("UVICORN_HOST", "UVICORN_FD", "UVICORN_UDS", "UVICORN_PORT"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(
            sys,
            "argv",
            ["/opt/venv/bin/uvicorn", "particles.api.app:app", "--host", "0.0.0.0"],
        )
        monkeypatch.setenv("PARTICLES_API_KEY", "")
        reset_config()
        client = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))
        assert client.get("/corpus").status_code == 503

    def test_empty_key_under_the_dev_key_skip_serves_reads_and_refuses_writes(
        self, cli_db: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.app import app
        from particles.api.auth import KEYLESS_WRITE_REFUSAL

        monkeypatch.setenv("PARTICLES_API_KEY", "")
        monkeypatch.setenv("PARTICLES_API_BIND_HOST", "127.0.0.1")
        reset_config()
        client = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))
        assert client.get("/corpus").status_code == 200
        # Before the fix this was 401: the empty key was a key nobody could send.
        resp = client.post(
            "/corpus/deposit/text", json={"text": "x"}, headers={"Authorization": "Bearer "}
        )
        assert resp.status_code == 503
        assert resp.json()["detail"] == KEYLESS_WRITE_REFUSAL
        # The loopback bound applies: a non-loopback peer is refused.
        remote = TestClient(app, base_url="http://127.0.0.1:8000", client=("203.0.113.9", 50000))
        assert remote.get("/corpus").status_code == 503

    def test_every_reader_agrees(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.api.auth import (
            http_write_posture,
            open_reads_warning,
            warn_if_dev_auth_in_use,
        )

        monkeypatch.setenv("PARTICLES_API_KEY", " ")
        reset_config()
        assert http_write_posture().startswith("HTTP writes: refused")
        assert warn_if_dev_auth_in_use() is True
        assert open_reads_warning("0.0.0.0") is None


class TestServingChecksWithoutLifespan:
    """Security review 2026-10-25, Q10: a launch the startup gate refuses serves
    nothing, even when the lifespan never runs (``uvicorn --lifespan off``, or a
    TestClient used outside a ``with`` block, which is how these tests reach it)."""

    _UVICORN = "/opt/venv/bin/uvicorn"

    def _argv(self, monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
        import sys

        for name in ("UVICORN_HOST", "UVICORN_FD", "UVICORN_UDS", "UVICORN_PORT"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(sys, "argv", [self._UVICORN, "particles.api.app:app", *args])

    def test_refused_command_line_serves_nothing_with_lifespan_off(
        self, cli_db: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.app import app

        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        self._argv(monkeypatch, "--host", "0.0.0.0", "--lifespan", "off")
        reset_config()
        with pytest.raises(RuntimeError):
            enforce_fail_closed_on_startup()

        client = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))
        for path in ("/corpus", "/health", "/openapi.json", "/app/"):
            resp = client.get(path)
            assert resp.status_code == 503, (path, resp.status_code)
            assert "0.0.0.0" not in resp.text

    def test_a_loopback_launch_serves_with_lifespan_off(
        self, cli_db: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.app import app

        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        self._argv(monkeypatch, "--host", "127.0.0.1", "--lifespan", "off")
        reset_config()
        client = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))
        assert client.get("/health").status_code == 200

    def test_checks_run_once_per_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import particles.api.app as app_module

        calls: list[str] = []
        monkeypatch.setattr(app_module, "enforce_fail_closed_on_startup", lambda: calls.append("x"))
        monkeypatch.setattr(app_module, "_serving_checks_passed_for", None)
        client = TestClient(
            app_module.app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000)
        )
        client.get("/health")
        client.get("/health")
        assert calls == ["x"]
        # A reload is a new config: the checks run again under it.
        reset_config()
        client.get("/health")
        assert calls == ["x", "x"]

    def test_a_failing_check_is_retried_until_it_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import particles.api.app as app_module

        outcomes = [RuntimeError("refused"), None]

        def gate() -> None:
            outcome = outcomes.pop(0)
            if outcome is not None:
                raise outcome

        monkeypatch.setattr(app_module, "enforce_fail_closed_on_startup", gate)
        monkeypatch.setattr(app_module, "_serving_checks_passed_for", None)
        client = TestClient(
            app_module.app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000)
        )
        assert client.get("/health").status_code == 503
        assert client.get("/health").status_code == 200


class TestStartupOpenReadsWarning:
    """Security review 2026-10-24-2, N9: the warning reaches a raw uvicorn launch too."""

    def test_raw_uvicorn_launch_logs_the_warning_at_lifespan(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import asyncio
        import sys

        import particles.api.app as app_module

        monkeypatch.setenv("PARTICLES_API_KEY", "real-secret")
        monkeypatch.setattr(
            sys, "argv", ["/venv/bin/uvicorn", "particles.api.app:app", "--host", "0.0.0.0"]
        )
        reset_config()

        async def _noop() -> None:
            return None

        monkeypatch.setattr(app_module, "create_tables", _noop)
        monkeypatch.setattr(app_module, "start_daemon", lambda: None)
        monkeypatch.setattr(app_module, "stop_daemon", _noop)
        monkeypatch.setattr(app_module, "ensure_backstop", lambda: None)

        async def _run() -> None:
            async with app_module.lifespan(app_module.app):
                pass

        with caplog.at_level("WARNING"):
            asyncio.run(_run())
        assert any("require_auth_for_reads" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize(
        "argv",
        [
            ["/venv/bin/uvicorn", "particles.api.app:app", "--host", "127.0.0.1"],
            ["/venv/bin/uvicorn", "particles.api.app:app", "--uds", "/tmp/p.sock"],
        ],
    )
    def test_local_binds_are_not_warned(
        self, monkeypatch: pytest.MonkeyPatch, argv: list[str]
    ) -> None:
        import sys

        from particles.api.auth import startup_open_reads_warning

        monkeypatch.setenv("PARTICLES_API_KEY", "real-secret")
        monkeypatch.setattr(sys, "argv", argv)
        reset_config()
        assert startup_open_reads_warning() is None

    def test_read_auth_on_is_not_warned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        from particles.api.auth import startup_open_reads_warning

        monkeypatch.setenv("PARTICLES_API_KEY", "real-secret")
        monkeypatch.setenv("PARTICLES_API_REQUIRE_AUTH_FOR_READS", "true")
        monkeypatch.setattr(sys, "argv", ["uvicorn", "particles.api.app:app", "--host", "0.0.0.0"])
        reset_config()
        assert startup_open_reads_warning() is None


# ---------------------------------------------------------------------------
# warn_if_dev_auth_in_use — startup warning when bearer auth is disabled
# ---------------------------------------------------------------------------


class TestWarnIfDevAuthInUse:
    def test_warns_when_env_var_unset(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from particles.api.auth import warn_if_dev_auth_in_use

        monkeypatch.delenv("PARTICLES_API_KEY", raising=False)
        with caplog.at_level("WARNING", logger="particles.api.auth"):
            fired = warn_if_dev_auth_in_use()
        assert fired is True
        assert any("serves reads only" in rec.message for rec in caplog.records)

    def test_warns_when_explicit_dev_key(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from particles.api.auth import warn_if_dev_auth_in_use

        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        with caplog.at_level("WARNING", logger="particles.api.auth"):
            fired = warn_if_dev_auth_in_use()
        assert fired is True

    def test_silent_when_real_key_set(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from particles.api.auth import warn_if_dev_auth_in_use

        monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
        with caplog.at_level("WARNING", logger="particles.api.auth"):
            fired = warn_if_dev_auth_in_use()
        assert fired is False
        assert not any("serves reads only" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Proxy-aware loopback gate — api.trusted_proxies + X-Forwarded-For
# (§(a) mechanism ii)
# ---------------------------------------------------------------------------


class TestProxyAwareLoopbackGate:
    @staticmethod
    def _set_trusted(monkeypatch: pytest.MonkeyPatch, proxies: list[str]) -> None:
        # The autouse fixture reset config before this test; build the singleton
        # and patch the field so auth's get_config() sees the trusted set.
        monkeypatch.setattr(get_config().api, "trusted_proxies", proxies)

    def test_default_empty_ignores_xff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No trusted proxies (default): a spoofed X-Forwarded-For is ignored and
        # the loopback peer is honoured exactly as before → dev-key skip applies.
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        self._set_trusted(monkeypatch, [])
        _verify_key(_request(client=("127.0.0.1", 12345), xff="203.0.113.7"), None)

    def test_trusted_proxy_remote_xff_refused_503(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Behind a trusted loopback proxy, a remote real client (via XFF) must
        # NOT read as loopback — the dev-key skip refuses it (503).
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        self._set_trusted(monkeypatch, ["127.0.0.1"])
        with pytest.raises(HTTPException) as exc:
            _verify_key(_request(client=("127.0.0.1", 12345), xff="203.0.113.7"), None)
        assert exc.value.status_code == 503

    def test_trusted_proxy_loopback_xff_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Request originated locally and traversed the trusted proxy → real
        # client is loopback → dev-key skip applies.
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        self._set_trusted(monkeypatch, ["127.0.0.1"])
        _verify_key(_request(client=("127.0.0.1", 12345), xff="127.0.0.1"), None)

    def test_untrusted_peer_xff_not_consulted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Peer is NOT a trusted proxy → XFF is ignored even when it claims
        # loopback (anti-spoof); the remote peer governs → refused.
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        self._set_trusted(monkeypatch, ["10.0.0.0/8"])
        with pytest.raises(HTTPException) as exc:
            _verify_key(_request(client=("203.0.113.7", 5000), xff="127.0.0.1"), None)
        assert exc.value.status_code == 503

    def test_multi_hop_returns_nearest_untrusted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # XFF = "<real client>, <inner trusted proxy>": walking right-to-left
        # skips the trusted hop and returns the real remote client → refused.
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        self._set_trusted(monkeypatch, ["127.0.0.1", "10.0.0.0/8"])
        with pytest.raises(HTTPException) as exc:
            _verify_key(_request(client=("127.0.0.1", 12345), xff="203.0.113.7, 10.0.0.5"), None)
        assert exc.value.status_code == 503

    def test_cidr_trusted_proxy_with_loopback_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # CIDR proxy match: peer 10.0.0.5 ∈ 10.0.0.0/8, XFF loopback → allowed.
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        self._set_trusted(monkeypatch, ["10.0.0.0/8"])
        _verify_key(_request(client=("10.0.0.5", 5000), xff="127.0.0.1"), None)


# ---------------------------------------------------------------------------
# _verify_read_key — conditional read-route gate (security review F2)
# ---------------------------------------------------------------------------


class TestVerifyReadKey:
    """The read gate enforces the bearer only when ``api.require_auth_for_reads``.

    The flag-off path is the default read posture (reads open even with a real
    key); the flag-on path delegates to ``_verify_key`` so reads carry the same
    fail-closed bearer behaviour as writes.
    """

    def test_noop_when_flag_off_even_with_real_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.api.auth import _verify_read_key

        monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
        monkeypatch.setenv("PARTICLES_API_REQUIRE_AUTH_FOR_READS", "false")
        reset_config()
        _verify_read_key(_request(), None)  # no creds, no exception — reads open

    def test_delegates_to_verify_key_when_flag_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.api.auth import _verify_read_key

        monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
        monkeypatch.setenv("PARTICLES_API_REQUIRE_AUTH_FOR_READS", "true")
        reset_config()
        with pytest.raises(HTTPException) as exc:
            _verify_read_key(_request(), None)
        assert exc.value.status_code == 401
        _verify_read_key(_request(), _creds("prod-secret"))  # right key accepted

    def test_dev_key_loopback_skip_applies_when_flag_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Even with the flag on, the dev-key loopback skip keeps reads open for
        # local development (delegated through _verify_key).
        from particles.api.auth import _verify_read_key

        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        monkeypatch.setenv("PARTICLES_API_REQUIRE_AUTH_FOR_READS", "true")
        reset_config()
        _verify_read_key(_request(), None)  # loopback peer → no exception


# ---------------------------------------------------------------------------
# Operator-scope writes never ride the dev-key loopback skip
# ---------------------------------------------------------------------------

#: Every operator-scope route, with a body that passes request validation. The
#: dry-run-capable routes are listed in their writing form (``dry_run`` false,
#: ``fix`` true, ``mode`` APPLY).
_OPERATOR_ROUTES: list[tuple[str, str, dict[str, object]]] = [
    (
        "POST",
        "/particles/p-missing/supersede",
        {"content": "c", "subject_names": ["S"], "confidence": 0.5, "reason": "r"},
    ),
    ("POST", "/particles/p-missing/retract", {"reason": "r"}),
    ("POST", "/particles/p-missing/subjects", {"subject_name": "S"}),
    ("POST", "/curation/affirm", {"particle_id": "p-missing"}),
    ("POST", "/curation/snooze", {"card_key": "contested:p-missing"}),
    ("POST", "/review/p-missing", {"action": "DEFER", "reviewer_id": "operator"}),
    ("POST", "/corpus/e-missing/retract", {"reason": "r", "dry_run": False}),
    ("POST", "/subjects/relink-gated", {"dry_run": False}),
    ("POST", "/trust/rules", {"scope": "domain", "pattern": "example.org", "score": 0.5}),
    (
        "POST",
        "/trust/statements",
        {
            "domain": "general",
            "source_ref_type": "SOURCE_TYPE",
            "source_ref_value": "WEB_PAGE",
            "trust_rank": 0.5,
        },
    ),
    # Security review 2026-10-24, N2 and N5: the subject, link, tag, lint-fix,
    # reconcile, links-suggest APPLY and db-init writes.
    ("POST", "/subjects/merge", {"source_ids": ["s-a"], "target_id": "s-b", "dry_run": False}),
    ("POST", "/subjects/s-missing/merge", {"target_id": "s-b"}),
    ("POST", "/subjects/s-missing/split", {"particle_ids": ["p-missing"], "new_name": "X"}),
    ("POST", "/subjects/s-missing/aliases", {"aliases": ["a"]}),
    ("POST", "/particles/p-missing/tags", {"tags": ["t"]}),
    ("DELETE", "/particles/p-missing/tags", {"tags": ["t"]}),
    ("POST", "/links", {"particle_a": "p-a", "particle_b": "p-b"}),
    ("DELETE", "/links", {"particle_a": "p-a", "particle_b": "p-b"}),
    ("POST", "/lint", {"fix": True, "semantic": False}),
    ("POST", "/reconcile", {"dry_run": False}),
    ("POST", "/links/suggest", {"mode": "APPLY"}),
    ("POST", "/db/init", {}),
    # Security review 2026-10-24-2, P7: creating a Subject sets the same
    # resolution-steering aliases and external ids the alias route gates.
    ("POST", "/subjects", {"canonical_name": "Planted", "aliases": ["Acme Corp"]}),
    (
        "POST",
        "/subjects",
        {"canonical_name": "Planted", "external_ids": [{"namespace": "wikidata", "id": "Q1"}]},
    ),
    # P6: an HTTP deposit attributed to a principal other than its depositor,
    # or to the reserved operator label.
    (
        "POST",
        "/corpus/deposit/text",
        {"text": "x", "deposited_by": "mcp:claude-code", "author_id": "operator:local"},
    ),
    ("POST", "/corpus/deposit/text", {"text": "x", "deposited_by": "operator:local"}),
]


@pytest.fixture
def api_client(cli_db: object) -> Iterator[TestClient]:
    """A loopback ``TestClient`` over a file-backed store with every table created."""
    from particles.api.app import app

    yield TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))


class TestOperatorRoutesRefuseDevKey:
    """A loopback caller with no key cannot reach an operator-scope write.

    The dev-key skip serves loopback because a local process is taken to be
    the operator. A producing agent with a shell is also a local process, so
    the routes that act with the operator's authority need a real key.
    """

    @pytest.mark.parametrize(("method", "path", "body"), _OPERATOR_ROUTES)
    def test_loopback_dev_key_refused_503(
        self,
        api_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        method: str,
        path: str,
        body: dict[str, object],
    ) -> None:
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        resp = api_client.request(method, path, json=body)
        assert resp.status_code == 503
        assert "PARTICLES_API_KEY" in resp.json()["detail"]

    @pytest.mark.parametrize(("method", "path", "body"), _OPERATOR_ROUTES)
    def test_loopback_dev_key_refused_even_with_a_bearer(
        self,
        api_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        method: str,
        path: str,
        body: dict[str, object],
    ) -> None:
        # Under dev-key there is no key to check a bearer against, so presenting
        # one changes nothing.
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        resp = api_client.request(
            method, path, json=body, headers={"Authorization": "Bearer dev-key"}
        )
        assert resp.status_code == 503

    @pytest.mark.parametrize(("method", "path", "body"), _OPERATOR_ROUTES)
    def test_real_key_wrong_bearer_401(
        self,
        api_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        method: str,
        path: str,
        body: dict[str, object],
    ) -> None:
        monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
        resp = api_client.request(method, path, json=body, headers={"Authorization": "Bearer nope"})
        assert resp.status_code == 401

    @pytest.mark.parametrize(("method", "path", "body"), _OPERATOR_ROUTES)
    def test_real_key_right_bearer_passes_the_gate(
        self,
        api_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        method: str,
        path: str,
        body: dict[str, object],
    ) -> None:
        # Past the gate the route answers on its own terms (403 writes disabled,
        # 404 no such record, 200 for an empty relink): anything but an auth refusal.
        monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
        if path == "/db/init":
            # The scratch store's tables come from create_all, not Alembic.
            async def _no_migration() -> None:
                return None

            monkeypatch.setattr("particles.api.app.create_tables", _no_migration)
        resp = api_client.request(
            method, path, json=body, headers={"Authorization": "Bearer prod-secret"}
        )
        assert resp.status_code not in (401, 503), resp.text

    def test_every_operator_route_is_listed(self) -> None:
        # A route that takes ``OperatorAuthDep`` and is missing from the table
        # above would go untested; one listed here that lost it would fail above.
        from fastapi.routing import APIRoute

        from particles.api.app import app
        from particles.api.auth import _operator_gate

        def _uses(dependant: Any) -> bool:
            return any(d.call is _operator_gate or _uses(d) for d in dependant.dependencies)

        gated = {
            route.path
            for route in app.routes
            if isinstance(route, APIRoute) and _uses(route.dependant)
        }
        assert gated == {
            "/particles/{particle_id}/supersede",
            "/particles/{particle_id}/retract",
            "/particles/{particle_id}/subjects",
            "/curation/affirm",
            "/curation/snooze",
            "/review/{particle_id}",
            "/trust/rules",
            "/trust/statements",
            "/subjects/{source_id}/merge",
            "/subjects/{subject_id}/split",
            "/subjects/{subject_id}/aliases",
            "/particles/{particle_id}/tags",
            "/links",
            "/db/init",
            # each acts store-wide or on another principal's records.
            "/extract",
            "/subjects",
            "/corpus/links/dismiss",
            "/curation/rebuild",
        }


class TestDevKeyLoopbackSkipStillServes:
    """Everything that writes nothing keeps the loopback skip; every write is refused."""

    def test_dry_run_corpus_retract_served(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A dry run writes nothing, so it keeps the skip: 404 for an absent entry.
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        resp = api_client.post("/corpus/e-missing/retract", json={"dry_run": True})
        assert resp.status_code == 404

    def test_dry_run_relink_served(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        resp = api_client.post("/subjects/relink-gated", json={"dry_run": True})
        assert resp.status_code == 200

    @pytest.mark.parametrize(
        ("path", "body", "expected"),
        [
            # Security review 2026-10-24, N2: the forms that write nothing keep the skip.
            ("/subjects/merge", {"source_ids": ["s-a"], "target_id": "s-b", "dry_run": True}, 404),
            ("/lint", {"fix": False, "semantic": False}, 200),
            ("/reconcile", {"dry_run": True}, 200),
            ("/links/suggest", {"mode": "REPORT"}, 200),
        ],
    )
    def test_non_writing_forms_served(
        self,
        api_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        path: str,
        body: dict[str, object],
        expected: int,
    ) -> None:
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        resp = api_client.post(path, json=body)
        assert resp.status_code == expected, resp.text

    @pytest.mark.parametrize("path", ["/review", "/events", "/particles", "/subjects"])
    def test_read_routes_served(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch, path: str
    ) -> None:
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        resp = api_client.get(path)
        assert resp.status_code == 200, resp.text

    def test_agent_own_belief_write_is_refused_keyless(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # the agent's own-beliefs path is a write too, so a keyless
        # engine refuses it before the store gate is consulted.
        from particles.api.auth import KEYLESS_WRITE_REFUSAL

        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        resp = api_client.post(
            "/particles/retract", json={"particle_id": "p-missing", "reason": "r"}
        )
        assert resp.status_code == 503
        assert resp.json()["detail"] == KEYLESS_WRITE_REFUSAL


class TestHttpDepositAttribution:
    """Security review 2026-10-24-2, P6 (the HTTP half; S6 owns the MCP half).

    The agent path accepts an entry as the agent's own source by its
    ``deposited_by``, and the AUTHOR trust tier reads the snapshot's
    ``author_id``. Over HTTP both are the caller's to write, so an attribution to
    someone else needs operator authority. Since a keyless engine
    refuses every one of these deposits, so the agent's own remote deposit needs
    the key as well.
    """

    @pytest.fixture
    def belief_writes(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        from unittest.mock import AsyncMock, MagicMock

        import numpy as np

        import particles.ingest.subject_resolver as sr
        from particles import embeddings as ep
        from particles.db import DEFAULT_STORE

        get_config().mcp.write.enabled_stores = [DEFAULT_STORE]
        monkeypatch.setattr(sr, "resolve_subjects", AsyncMock(return_value=["sid-test"]))
        model = MagicMock()
        model.encode = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
        original = ep._embedding_model
        ep.set_embedding_model(model)
        try:
            yield
        finally:
            ep.set_embedding_model(original)

    def test_agent_cannot_cite_a_deposit_carrying_another_author(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch, belief_writes: None
    ) -> None:
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        identity = get_config().mcp.write.asserter_identity
        planted = api_client.post(
            "/corpus/deposit/text",
            json={
                "text": "A planted passage.",
                "deposited_by": identity,
                "author_id": get_config().curation.operator_identity,
            },
        )
        assert planted.status_code == 503, planted.text

    def test_the_agent_own_remote_deposit_is_still_its_source(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch, belief_writes: None
    ) -> None:
        # What HttpBackend.deposit_text sends for the agent: its own
        # deposit route, which attributes on the server (2026-10-24-2, P28).
        # A keyed engine: a keyless one refuses every write.
        monkeypatch.setenv("PARTICLES_API_KEY", TEST_ENGINE_KEY)
        api_client.headers.update(BEARER)
        dep = api_client.post("/particles/deposit", json={"text": "The agent's passage."})
        assert dep.status_code == 200, dep.text
        resp = api_client.post(
            "/particles/assert",
            json={
                "content": "The agent's claim holds.",
                "subject_names": ["agent"],
                "confidence": 0.9,
                "corpus_entry_id": dep.json()["entry_id"],
            },
        )
        assert resp.status_code == 200, resp.text

    def test_an_unattributed_deposit_is_refused_keyless(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # even a deposit naming nobody else is a write.
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        resp = api_client.post("/corpus/deposit/text", json={"text": "plain", "tags": []})
        assert resp.status_code == 503, resp.text

    def test_url_and_file_deposits_refuse_the_operator_label(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Refused before any fetch or upload is read.
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        operator = get_config().curation.operator_identity
        url = api_client.post(
            "/corpus/deposit/url", json={"url": "https://example.org/", "deposited_by": operator}
        )
        assert url.status_code == 503, url.text
        upload = api_client.post(
            "/corpus/deposit/file",
            files={"file": ("a.txt", b"x", "text/plain")},
            data={"deposited_by": operator},
        )
        assert upload.status_code == 503, upload.text

    def test_a_subject_with_only_a_name_is_refused_keyless(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Q3 (security review 2026-10-25): a canonical name alone can take
        # another Subject's alias, and a keyless engine refuses every write.
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        resp = api_client.post("/subjects", json={"canonical_name": "Plain", "description": "d"})
        assert resp.status_code == 503, resp.text

    def test_a_subject_with_only_a_name_is_served_with_the_key(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PARTICLES_API_KEY", TEST_ENGINE_KEY)
        resp = api_client.post(
            "/subjects", json={"canonical_name": "Plain", "description": "d"}, headers=BEARER
        )
        assert resp.status_code == 200, resp.text


class TestOperatorLinkAndTagMirrorsNeedWriteEnablement:
    """Security review 2026-10-24, N5: the operator link and tag mirrors had no write gate.

    They call the stores directly, so they are the agent verbs without the
    ownership, HUMAN_REVIEW and scope guards (D10). Beside the operator bearer,
    they now honour ``mcp.write.enabled_stores`` like the operator supersede.
    """

    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            ("POST", "/particles/p-missing/tags", {"tags": ["t"]}),
            ("DELETE", "/particles/p-missing/tags", {"tags": ["t"]}),
            ("POST", "/links", {"particle_a": "p-a", "particle_b": "p-b"}),
            ("DELETE", "/links", {"particle_a": "p-a", "particle_b": "p-b"}),
        ],
    )
    def test_refused_403_when_writes_are_disabled(
        self,
        api_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        method: str,
        path: str,
        body: dict[str, object],
    ) -> None:
        monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
        resp = api_client.request(
            method, path, json=body, headers={"Authorization": "Bearer prod-secret"}
        )
        assert resp.status_code == 403
        assert "enabled_stores" in resp.json()["detail"]


class TestUnixSocketBind:
    """Security review 2026-10-25-2, R1: a UNIX-socket connection carries no peer.

    uvicorn reports no client for one, so a reverse proxy on the socket forwards
    every caller as peerless, and the dev-key bound read each as loopback.
    """

    _UDS = ["/venv/bin/uvicorn", "particles.api.app:app", "--uds", "/tmp/particles-test.sock"]

    def _argv(self, monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
        import sys

        for name in ("UVICORN_HOST", "UVICORN_FD", "UVICORN_UDS", "UVICORN_PORT"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(sys, "argv", argv)

    def test_a_keyless_unix_socket_launch_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.api.auth import enforce_fail_closed_on_startup

        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        self._argv(monkeypatch, self._UDS)
        with pytest.raises(RuntimeError, match="UNIX socket"):
            enforce_fail_closed_on_startup()

    def test_a_keyed_unix_socket_launch_boots(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.api.auth import enforce_fail_closed_on_startup

        monkeypatch.setenv("PARTICLES_API_KEY", "real-secret")
        self._argv(monkeypatch, self._UDS)
        enforce_fail_closed_on_startup()

    @pytest.mark.parametrize("xff", [None, "127.0.0.1", "203.0.113.7"])
    def test_a_keyless_engine_serves_no_peerless_request(
        self, cli_db: Any, monkeypatch: pytest.MonkeyPatch, xff: str | None
    ) -> None:
        """Whatever launched it (``uvicorn.run(uds=…)`` is not seen at startup), and
        whatever a proxy forwarded, a request with no peer is not served keyless."""
        import asyncio

        import httpx

        from particles.api.app import app

        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        self._argv(monkeypatch, ["/venv/bin/uvicorn", "particles.api.app:app"])
        reset_config()
        headers = {} if xff is None else {"X-Forwarded-For": xff}

        async def _get() -> httpx.Response:
            transport = httpx.ASGITransport(app=app, client=None)  # type: ignore[arg-type]
            async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as c:
                return await c.get("/corpus", headers=headers)

        assert asyncio.run(_get()).status_code == 503

    def test_worker_and_reload_children_judge_the_parents_command_line(self) -> None:
        """The critic's open item: ``--workers`` / ``--reload`` children run the startup
        gate in a spawned process. ``multiprocessing``'s spawn hands the child the
        parent's ``sys.argv``, so the child parses the same uvicorn command line.
        Checked live against uvicorn's children too (PR body); this pins the
        mechanism the gate relies on."""
        import multiprocessing.spawn
        import sys

        from uvicorn._subprocess import spawn

        assert spawn.get_start_method() == "spawn"
        assert multiprocessing.spawn.get_preparation_data("x")["sys_argv"] == sys.argv

"""Tests for particles/api/auth.py.

``_verify_key`` reads ``PARTICLES_API_KEY`` at call time, not at module-import
time — the L2 architecture-review fix. Before the fix, the cached module-level
constant meant the key could not be rotated without restarting the process, and
tests could not change it via ``monkeypatch.setenv`` once any code path had
imported the module.

The fail-closed bearer-auth behaviour is also covered here: the
constant-time compare, the per-request non-loopback gate under dev-key, the
``_is_loopback_host`` classifier, and the ``enforce_fail_closed_on_startup``
boot check.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from starlette.requests import Request

from particles.api.auth import (
    _is_loopback_host,
    _verify_key,
    enforce_fail_closed_on_startup,
)
from particles.config import get_config, reset_config


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

    def test_no_peer_client_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        _verify_key(_request(client=None), None)  # no exception

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

    def test_real_key_non_loopback_bind_allows_boot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A real key is set → the bind host is irrelevant; serving publicly is fine.
        monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
        monkeypatch.setenv("PARTICLES_API_BIND_HOST", "0.0.0.0")
        reset_config()
        enforce_fail_closed_on_startup()  # no exception


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
        assert any("DISABLED" in rec.message for rec in caplog.records)

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
        assert not any("DISABLED" in rec.message for rec in caplog.records)


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

# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for particles/api/_rate_limit.py (token-bucket logic, in isolation).

The endpoint-level wiring (429 on /query, /reindex, the semantic /lint path,
disabled-when-0) is covered by TestRateLimit in tests/test_app.py. These pin the
bucket math directly with a controlled clock — no FastAPI, no DB.
"""

from __future__ import annotations

from typing import Any

import pytest

from particles.api import _rate_limit
from particles.config import reset_config


@pytest.fixture(autouse=True)
def _clean_buckets() -> Any:
    _rate_limit.reset_rate_limiter()
    yield
    _rate_limit.reset_rate_limiter()


def test_allow_spends_capacity_then_refuses() -> None:
    # Capacity == rate; at a frozen clock the bucket cannot refill.
    now = 1000.0
    allowed = [_rate_limit._allow("h", 3, now) for _ in range(3)]
    assert allowed == [True, True, True]
    assert _rate_limit._allow("h", 3, now) is False  # 4th over capacity


def test_allow_refills_over_time() -> None:
    now = 1000.0
    for _ in range(3):
        _rate_limit._allow("h", 3, now)
    assert _rate_limit._allow("h", 3, now) is False
    # 20 s later, rate=3/min ⇒ +1 token refilled.
    assert _rate_limit._allow("h", 3, now + 20.0) is True
    assert _rate_limit._allow("h", 3, now + 20.0) is False


def test_buckets_are_per_host() -> None:
    now = 1000.0
    assert _rate_limit._allow("a", 1, now) is True
    assert _rate_limit._allow("a", 1, now) is False
    # A different host has its own full bucket.
    assert _rate_limit._allow("b", 1, now) is True


def test_reset_clears_state() -> None:
    now = 1000.0
    assert _rate_limit._allow("h", 1, now) is True
    assert _rate_limit._allow("h", 1, now) is False
    _rate_limit.reset_rate_limiter()
    assert _rate_limit._allow("h", 1, now) is True


def test_reset_config_clears_state() -> None:
    # The reset hook registered at import time wires reset_config -> bucket clear.
    now = 1000.0
    assert _rate_limit._allow("h", 1, now) is True
    assert _rate_limit._allow("h", 1, now) is False  # bucket now empty
    reset_config()  # runs reset_rate_limiter via the registered hook
    assert _rate_limit._allow("h", 1, now) is True


def test_enforce_disabled_when_rate_not_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARTICLES_API_RATE_LIMIT_PER_MINUTE", "0")
    reset_config()

    class _Req:
        client = type("C", (), {"host": "1.2.3.4"})()
        headers: dict[str, str] = {}

    # No exception even when called far more than any positive cap would allow.
    for _ in range(50):
        _rate_limit.enforce_rate_limit(_Req())  # type: ignore[arg-type]


def test_enforce_noop_without_peer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARTICLES_API_RATE_LIMIT_PER_MINUTE", "1")
    reset_config()

    class _Req:
        client = None
        headers: dict[str, str] = {}

    # A request without a network peer (in-process transport) is treated as
    # local and never throttled.
    for _ in range(5):
        _rate_limit.enforce_rate_limit(_Req())  # type: ignore[arg-type]


class TestClientKeyAndBound:
    """Security review 2026-10-24, C9: address rotation took a fresh bucket each time."""

    @pytest.mark.parametrize(
        ("host", "key"),
        [
            ("2001:db8:1:2:aaaa::1", "2001:db8:1:2::/64"),
            ("2001:db8:1:2:ffff:ffff:ffff:ffff", "2001:db8:1:2::/64"),
            ("::ffff:203.0.113.7", "203.0.113.7"),
            ("203.0.113.7", "203.0.113.7"),
            ("not-an-ip", "not-an-ip"),
        ],
    )
    def test_ipv6_clients_share_their_slash_64(self, host: str, key: str) -> None:
        assert _rate_limit._bucket_key(host) == key

    def test_rotating_within_a_slash_64_does_not_refill(self) -> None:
        now = 1000.0
        first = _rate_limit._bucket_key("2001:db8::1")
        second = _rate_limit._bucket_key("2001:db8::2")
        assert _rate_limit._allow(first, 1, now) is True
        assert _rate_limit._allow(second, 1, now) is False

    def test_bucket_count_is_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_rate_limit, "_MAX_BUCKETS", 5)
        for i in range(50):
            _rate_limit._allow(f"10.0.0.{i}", 1, 1000.0 + i * 0.001)
        assert len(_rate_limit._buckets) <= 5
        # The newest client is still tracked: its spent token is remembered.
        assert _rate_limit._allow("10.0.0.49", 1, 1000.05) is False

    def test_full_buckets_are_dropped_before_active_ones(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_rate_limit, "_MAX_BUCKETS", 3)
        _rate_limit._allow("idle", 1, 0.0)
        _rate_limit._allow("busy-a", 1, 100.0)
        _rate_limit._allow("busy-b", 1, 100.0)
        _rate_limit._allow("new", 1, 100.5)
        assert "idle" not in _rate_limit._buckets
        assert {"busy-a", "busy-b", "new"} <= set(_rate_limit._buckets)

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

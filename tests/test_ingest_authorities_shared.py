# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-namespace rate limiter in ``particles.ingest.authorities._shared``.

``_RateLimiter.acquire`` spaces calls ``1 / requests_per_second`` apart. The
tests swap the module's ``asyncio`` for a fake clock whose ``sleep`` advances
time instead of waiting, so the spacing is asserted exactly and nothing
actually sleeps. ``PatternAuthority`` is covered in
``tests/test_subject_authority.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from types import SimpleNamespace

import pytest

from particles.ingest.authorities import _shared
from particles.ingest.authorities._shared import _RateLimiter, get_limiter, reset_limiters


class _FakeClock:
    """A loop clock that only moves when the limiter sleeps (or a test advances it)."""

    def __init__(self, start: float = 100.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    fake = _FakeClock()
    monkeypatch.setattr(
        _shared,
        "asyncio",
        SimpleNamespace(get_event_loop=lambda: fake, sleep=fake.sleep),
    )
    return fake


@pytest.fixture
def clean_limiters() -> Iterator[None]:
    reset_limiters()
    yield
    reset_limiters()


# ---------------------------------------------------------------------------
# _RateLimiter
# ---------------------------------------------------------------------------


def test_min_interval_is_reciprocal_of_rate() -> None:
    assert _RateLimiter(requests_per_second=2.0)._min_interval == pytest.approx(0.5)
    assert _RateLimiter(requests_per_second=0.25)._min_interval == pytest.approx(4.0)


async def test_first_acquire_does_not_wait(clock: _FakeClock) -> None:
    lim = _RateLimiter(requests_per_second=2.0)

    await lim.acquire()

    assert clock.sleeps == []
    assert lim._last == 100.0


async def test_back_to_back_acquire_waits_out_the_interval(clock: _FakeClock) -> None:
    lim = _RateLimiter(requests_per_second=2.0)
    await lim.acquire()

    clock.now += 0.2  # 0.2 s elapsed of the 0.5 s interval
    await lim.acquire()

    assert clock.sleeps == [pytest.approx(0.3)]
    # _last is re-read after the sleep, so the next interval starts from there.
    assert lim._last == pytest.approx(100.5)


async def test_acquire_after_interval_elapsed_does_not_wait(clock: _FakeClock) -> None:
    lim = _RateLimiter(requests_per_second=2.0)
    await lim.acquire()

    clock.now += 0.5
    await lim.acquire()

    assert clock.sleeps == []
    assert lim._last == pytest.approx(100.5)


async def test_concurrent_acquires_are_serialized(clock: _FakeClock) -> None:
    lim = _RateLimiter(requests_per_second=4.0)

    await asyncio.gather(lim.acquire(), lim.acquire(), lim.acquire())

    # The lock admits one caller at a time; each later caller waits one full
    # interval after the previous one.
    assert clock.sleeps == [pytest.approx(0.25), pytest.approx(0.25)]
    assert lim._last == pytest.approx(100.5)


# ---------------------------------------------------------------------------
# get_limiter / reset_limiters
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("clean_limiters")
def test_get_limiter_returns_one_instance_per_namespace() -> None:
    first = get_limiter("wikidata", 2.0)

    assert get_limiter("wikidata", 2.0) is first
    assert first.requests_per_second == 2.0
    assert get_limiter("numista", 1.0) is not first


@pytest.mark.usefixtures("clean_limiters")
def test_get_limiter_keeps_the_rate_it_was_created_with() -> None:
    first = get_limiter("wikidata", 2.0)

    # A later caller's rate does not rebuild an existing limiter; that is what
    # reset_limiters is for.
    again = get_limiter("wikidata", 10.0)

    assert again is first
    assert again._min_interval == pytest.approx(0.5)


@pytest.mark.usefixtures("clean_limiters")
def test_reset_limiters_drops_existing_instances() -> None:
    first = get_limiter("wikidata", 2.0)

    reset_limiters()
    second = get_limiter("wikidata", 10.0)

    assert second is not first
    assert second._min_interval == pytest.approx(0.1)

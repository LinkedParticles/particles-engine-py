"""Tests for the utility-evidence store (idempotent per-session events)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from particles.store.utility_store import (
    SOURCE_EXPLICIT,
    clear_utility_events,
    explicit_credit_key,
    get_reinforcement_scores,
    record_explicit_utility_event,
    record_utility_events,
)

_NOW = datetime(2026, 7, 1, tzinfo=UTC)


@pytest.mark.asyncio
async def test_record_and_score_single_session(db_session: object) -> None:
    await record_utility_events(
        db_session,  # type: ignore[arg-type]
        "sess-1",
        {"p-a": "literal", "p-b": "behavioural"},
        observed_at=_NOW,
    )
    scores = await get_reinforcement_scores(
        db_session,  # type: ignore[arg-type]
        ["p-a", "p-b", "p-c"],
        30.0,
        now=_NOW,
    )
    assert math.isclose(scores["p-a"], 1.0, abs_tol=1e-9)
    assert math.isclose(scores["p-b"], 1.0, abs_tol=1e-9)
    # A belief with no events is absent (caller treats as 0.0 → neutral).
    assert "p-c" not in scores


@pytest.mark.asyncio
async def test_re_mining_same_session_is_idempotent(db_session: object) -> None:
    await record_utility_events(db_session, "sess-1", {"p-a": "literal"}, observed_at=_NOW)  # type: ignore[arg-type]
    await record_utility_events(db_session, "sess-1", {"p-a": "literal"}, observed_at=_NOW)  # type: ignore[arg-type]
    scores = await get_reinforcement_scores(db_session, ["p-a"], 30.0, now=_NOW)  # type: ignore[arg-type]
    # Re-mining replaces, not duplicates: one event, reinforcement ~1.0.
    assert math.isclose(scores["p-a"], 1.0, abs_tol=1e-9)


@pytest.mark.asyncio
async def test_distinct_sessions_accumulate(db_session: object) -> None:
    await record_utility_events(db_session, "sess-1", {"p-a": "literal"}, observed_at=_NOW)  # type: ignore[arg-type]
    await record_utility_events(db_session, "sess-2", {"p-a": "literal"}, observed_at=_NOW)  # type: ignore[arg-type]
    scores = await get_reinforcement_scores(db_session, ["p-a"], 30.0, now=_NOW)  # type: ignore[arg-type]
    # Acted on in two distinct sessions → reinforcement ~2.0.
    assert math.isclose(scores["p-a"], 2.0, abs_tol=1e-9)


@pytest.mark.asyncio
async def test_recency_weighting(db_session: object) -> None:
    await record_utility_events(
        db_session,  # type: ignore[arg-type]
        "old",
        {"p-a": "literal"},
        observed_at=_NOW - timedelta(days=30),
    )
    scores = await get_reinforcement_scores(db_session, ["p-a"], 30.0, now=_NOW)  # type: ignore[arg-type]
    assert math.isclose(scores["p-a"], 0.5, abs_tol=1e-9)


@pytest.mark.asyncio
async def test_clear_removes_all(db_session: object) -> None:
    await record_utility_events(db_session, "sess-1", {"p-a": "literal"}, observed_at=_NOW)  # type: ignore[arg-type]
    removed = await clear_utility_events(db_session)  # type: ignore[arg-type]
    assert removed == 1
    scores = await get_reinforcement_scores(db_session, ["p-a"], 30.0, now=_NOW)  # type: ignore[arg-type]
    assert scores == {}


class TestExplicitChannel:
    """the second utility channel and its isolation from the miner."""

    @pytest.mark.asyncio
    async def test_explicit_credit_carries_its_channel_weight(self, db_session: object) -> None:
        await record_explicit_utility_event(db_session, "p-a", "explicit:op:2026-07-01", _NOW)  # type: ignore[arg-type]
        scores = await get_reinforcement_scores(
            db_session,  # type: ignore[arg-type]
            ["p-a"],
            30.0,
            now=_NOW,
            explicit_weight=25.0,
        )
        assert math.isclose(scores["p-a"], 25.0, abs_tol=1e-9)

    @pytest.mark.asyncio
    async def test_same_key_twice_is_one_credit(self, db_session: object) -> None:
        key = explicit_credit_key("cli:memory-useful", _NOW)
        assert await record_explicit_utility_event(db_session, "p-a", key, _NOW) is True  # type: ignore[arg-type]
        assert await record_explicit_utility_event(db_session, "p-a", key, _NOW) is False  # type: ignore[arg-type]
        scores = await get_reinforcement_scores(
            db_session,  # type: ignore[arg-type]
            ["p-a"],
            30.0,
            now=_NOW,
            explicit_weight=25.0,
        )
        assert math.isclose(scores["p-a"], 25.0, abs_tol=1e-9)

    @pytest.mark.asyncio
    async def test_distinct_days_accumulate(self, db_session: object) -> None:
        later = _NOW + timedelta(days=1)
        await record_explicit_utility_event(  # type: ignore[arg-type]
            db_session, "p-a", explicit_credit_key("op", _NOW), _NOW
        )
        await record_explicit_utility_event(  # type: ignore[arg-type]
            db_session, "p-a", explicit_credit_key("op", later), later
        )
        scores = await get_reinforcement_scores(
            db_session,  # type: ignore[arg-type]
            ["p-a"],
            30.0,
            now=later,
            explicit_weight=25.0,
        )
        # Two credits, the older decayed by one day — repeated endorsement over
        # time is genuine signal, which is what the daily key permits.
        assert scores["p-a"] > 25.0

    @pytest.mark.asyncio
    async def test_credit_key_is_per_principal(self, db_session: object) -> None:
        # Two interfaces are rate-limited independently.
        assert explicit_credit_key("cli:memory-useful", _NOW) != explicit_credit_key(
            "http:/memory/useful", _NOW
        )

    @pytest.mark.asyncio
    async def test_re_mining_cannot_destroy_an_explicit_credit(self, db_session: object) -> None:
        """The hazard scoped delete exists to prevent.

        The miner clears a session's rows before re-inserting. An unscoped delete
        would take an explicit credit filed under a colliding key with it.
        """
        collide = "sess-1"
        await record_explicit_utility_event(db_session, "p-a", collide, _NOW)  # type: ignore[arg-type]
        await record_utility_events(db_session, collide, {"p-b": "literal"}, observed_at=_NOW)  # type: ignore[arg-type]
        await record_utility_events(db_session, collide, {"p-b": "literal"}, observed_at=_NOW)  # type: ignore[arg-type]
        scores = await get_reinforcement_scores(
            db_session,  # type: ignore[arg-type]
            ["p-a"],
            30.0,
            now=_NOW,
            explicit_weight=25.0,
        )
        assert math.isclose(scores["p-a"], 25.0, abs_tol=1e-9)

    @pytest.mark.asyncio
    async def test_clear_can_scope_to_one_channel(self, db_session: object) -> None:
        await record_utility_events(db_session, "sess-1", {"p-a": "literal"}, observed_at=_NOW)  # type: ignore[arg-type]
        await record_explicit_utility_event(db_session, "p-a", "explicit:op:2026-07-01", _NOW)  # type: ignore[arg-type]
        assert await clear_utility_events(db_session, source=SOURCE_EXPLICIT) == 1  # type: ignore[arg-type]
        scores = await get_reinforcement_scores(db_session, ["p-a"], 30.0, now=_NOW)  # type: ignore[arg-type]
        assert math.isclose(scores["p-a"], 1.0, abs_tol=1e-9)

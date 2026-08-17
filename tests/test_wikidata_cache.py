"""Tests for the Wikidata label cache (particles/store/wikidata_cache.py).

A small persistent key/value cache (qid → label) — 0% covered in the
architecture-review baseline. Logic is trivial but worth pinning so the
get/set/update contract doesn't silently regress.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from particles.store.wikidata_cache import (
    WikidataLabelRow,
    _persistent_label_cache,
    get_label,
    set_label,
)


class TestGetLabel:
    @pytest.mark.asyncio
    async def test_missing_key_returns_none(self, db_session: Any) -> None:
        assert await get_label(db_session, "Q999999") is None

    @pytest.mark.asyncio
    async def test_returns_label_after_set(self, db_session: Any) -> None:
        await set_label(db_session, "Q64", "Berlin")
        assert await get_label(db_session, "Q64") == "Berlin"


class TestSetLabel:
    @pytest.mark.asyncio
    async def test_first_set_creates_row(self, db_session: Any) -> None:
        before = datetime.now(UTC).replace(tzinfo=None)
        await set_label(db_session, "P571", "inception")
        row = await db_session.get(WikidataLabelRow, "P571")
        assert row is not None
        assert row.label == "inception"
        # SQLite strips tzinfo on read-back; compare naive-to-naive against a
        # naive `before`. The check is just "timestamp is recent and sane".
        cached_at = row.cached_at.replace(tzinfo=None) if row.cached_at.tzinfo else row.cached_at
        assert cached_at >= before

    @pytest.mark.asyncio
    async def test_second_set_updates_label_and_timestamp(self, db_session: Any) -> None:
        await set_label(db_session, "Q42", "Old Label")
        row_before = await db_session.get(WikidataLabelRow, "Q42")
        assert row_before is not None
        before_naive = row_before.cached_at.replace(tzinfo=None)

        await set_label(db_session, "Q42", "New Label")
        row_after = await db_session.get(WikidataLabelRow, "Q42")
        assert row_after is not None
        assert row_after.label == "New Label"
        # Strip tzinfo on both sides — SQLite round-trip strips it on one side
        # but not the other (the second value comes from the freshly-assigned
        # datetime.now(UTC) which is still tz-aware in memory).
        after_naive = row_after.cached_at.replace(tzinfo=None)
        assert after_naive >= before_naive

    @pytest.mark.asyncio
    async def test_set_does_not_create_duplicate(self, db_session: Any) -> None:
        """Two consecutive set() calls with the same qid produce one row, not two."""
        from sqlalchemy import func, select

        await set_label(db_session, "Q1", "first")
        await set_label(db_session, "Q1", "second")
        count = (await db_session.execute(select(func.count(WikidataLabelRow.qid)))).scalar_one()
        assert count == 1


class TestPersistentLabelCache:
    """The L2 cache seam: DB hit/miss + own-session path."""

    @pytest.mark.asyncio
    async def test_db_hit_skips_fetch_live(self, db_session: Any) -> None:
        from unittest.mock import AsyncMock

        await set_label(db_session, "Q90", "Paris")
        await db_session.commit()
        fetch_live = AsyncMock(return_value="should-not-be-used")
        label = await _persistent_label_cache("Q90", fetch_live, db_session)
        assert label == "Paris"
        fetch_live.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_db_miss_runs_fetch_live_and_persists(self, db_session: Any) -> None:
        from unittest.mock import AsyncMock

        fetch_live = AsyncMock(return_value="Munich")
        label = await _persistent_label_cache("Q1726", fetch_live, db_session)
        assert label == "Munich"
        fetch_live.assert_awaited_once()
        # Persisted to the cache so a second call is a hit.
        assert await get_label(db_session, "Q1726") == "Munich"

    @pytest.mark.asyncio
    async def test_own_session_when_none_passed(self, db_session: Any) -> None:
        """session=None → the cache manages + commits its own session_scope()."""
        from unittest.mock import AsyncMock

        from particles.db import session_scope

        fetch_live = AsyncMock(return_value="Hamburg")
        label = await _persistent_label_cache("Q1055", fetch_live, None)
        assert label == "Hamburg"
        fetch_live.assert_awaited_once()
        # The own-session write committed — visible from an independent session.
        async with session_scope() as other:
            assert await get_label(other, "Q1055") == "Hamburg"

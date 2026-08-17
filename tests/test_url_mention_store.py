"""Tests for the URL-mention citation-signal store."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from particles.store.url_mention_store import (
    UrlMentionRow,
    build_deposited_url_map,
    get_suppressed_urls,
    list_undeposited_mentions,
    reconcile_url_to_entry,
    record_url_mentions,
    suppress_suggestion,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def _add_entry(session: AsyncSession, entry_id: str, uri_r: str | None) -> None:
    from particles.corpus.store import CorpusEntryRow

    session.add(
        CorpusEntryRow(
            entry_id=entry_id,
            uri_r=uri_r,
            source_type="WEB_PAGE",
            mutability="MUTABLE",
            fetch_policy="LAZY",
            created_at=datetime.now(UTC),
            deposited_by="test",
        )
    )
    await session.flush()


class TestRecord:
    async def test_inserts_new_mentions(self, db_session: AsyncSession) -> None:
        n = await record_url_mentions(
            db_session,
            source_entry_id="src1",
            canonical_urls=["https://a.example/x", "https://b.example/y"],
        )
        assert n == 2
        rows = await list_undeposited_mentions(db_session)
        assert {r.canonical_url for r in rows} == {"https://a.example/x", "https://b.example/y"}

    async def test_idempotent_on_reextraction(self, db_session: AsyncSession) -> None:
        await record_url_mentions(
            db_session, source_entry_id="src1", canonical_urls=["https://a.example/x"]
        )
        # Re-extracting the same source must not inflate counts.
        n = await record_url_mentions(
            db_session,
            source_entry_id="src1",
            canonical_urls=["https://a.example/x", "https://b.example/y"],
        )
        assert n == 1  # only the new URL
        rows = await list_undeposited_mentions(db_session)
        assert len(rows) == 2

    async def test_dedupes_within_call(self, db_session: AsyncSession) -> None:
        n = await record_url_mentions(
            db_session,
            source_entry_id="src1",
            canonical_urls=["https://a.example/x", "https://a.example/x"],
        )
        assert n == 1

    async def test_deposited_map_sets_target(self, db_session: AsyncSession) -> None:
        # A URL already deposited is born with target set → never a suggestion.
        n = await record_url_mentions(
            db_session,
            source_entry_id="src1",
            canonical_urls=["https://a.example/x"],
            deposited_map={"https://a.example/x": "entry-a"},
        )
        assert n == 1
        assert await list_undeposited_mentions(db_session) == []

    async def test_empty_is_noop(self, db_session: AsyncSession) -> None:
        assert await record_url_mentions(db_session, source_entry_id="s", canonical_urls=[]) == 0


class TestReconcile:
    async def test_binds_null_rows_and_returns_vias(self, db_session: AsyncSession) -> None:
        await record_url_mentions(
            db_session, source_entry_id="src1", canonical_urls=["https://a.example/x"]
        )
        await record_url_mentions(
            db_session, source_entry_id="src2", canonical_urls=["https://a.example/x"]
        )
        vias = await reconcile_url_to_entry(
            db_session, canonical_url="https://a.example/x", target_entry_id="entry-a"
        )
        assert sorted(vias) == ["src1", "src2"]
        assert await list_undeposited_mentions(db_session) == []

    async def test_second_call_is_noop(self, db_session: AsyncSession) -> None:
        await record_url_mentions(
            db_session, source_entry_id="src1", canonical_urls=["https://a.example/x"]
        )
        await reconcile_url_to_entry(
            db_session, canonical_url="https://a.example/x", target_entry_id="entry-a"
        )
        again = await reconcile_url_to_entry(
            db_session, canonical_url="https://a.example/x", target_entry_id="entry-a"
        )
        assert again == []

    async def test_leaves_other_urls_untouched(self, db_session: AsyncSession) -> None:
        await record_url_mentions(
            db_session,
            source_entry_id="src1",
            canonical_urls=["https://a.example/x", "https://b.example/y"],
        )
        await reconcile_url_to_entry(
            db_session, canonical_url="https://a.example/x", target_entry_id="entry-a"
        )
        remaining = await list_undeposited_mentions(db_session)
        assert [r.canonical_url for r in remaining] == ["https://b.example/y"]


class TestDepositedUrlMap:
    async def test_canonicalizes_entry_uris(self, db_session: AsyncSession) -> None:
        await _add_entry(db_session, "e1", "https://Example.com/Story?utm_source=x")
        await _add_entry(db_session, "e2", "https://other.example/p/")
        await _add_entry(db_session, "e3", None)  # local entry, no URL
        mapping = await build_deposited_url_map(db_session)
        assert mapping == {
            "https://example.com/Story": "e1",
            "https://other.example/p": "e2",
        }


class TestSuppression:
    async def test_suppressed_until_future_is_listed(self, db_session: AsyncSession) -> None:
        await suppress_suggestion(
            db_session,
            canonical_url="https://a.example/x",
            until=datetime.now(UTC) + timedelta(days=30),
        )
        assert await get_suppressed_urls(db_session) == {"https://a.example/x"}

    async def test_expired_snooze_not_listed(self, db_session: AsyncSession) -> None:
        await suppress_suggestion(
            db_session,
            canonical_url="https://a.example/x",
            until=datetime.now(UTC) - timedelta(days=1),
        )
        assert await get_suppressed_urls(db_session) == set()

    async def test_upsert_overwrites(self, db_session: AsyncSession) -> None:
        url = "https://a.example/x"
        await suppress_suggestion(
            db_session, canonical_url=url, until=datetime.now(UTC) - timedelta(days=1)
        )
        await suppress_suggestion(
            db_session, canonical_url=url, until=datetime.now(UTC) + timedelta(days=1)
        )
        assert await get_suppressed_urls(db_session) == {url}


async def test_row_repr_smoke(db_session: AsyncSession) -> None:
    # UrlMentionRow is importable and constructable (metadata-registration check).
    await record_url_mentions(
        db_session, source_entry_id="s", canonical_urls=["https://a.example/x"]
    )
    row = (await list_undeposited_mentions(db_session))[0]
    assert isinstance(row, UrlMentionRow)
    assert row.target_entry_id is None

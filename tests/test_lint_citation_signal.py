"""Tests for the L-CITE-01 undeposited-cited-source lint check."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from particles.operations.lint.citation_signal import _check_undeposited_cited_sources
from particles.store.url_mention_store import record_url_mentions

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def _add_entry(session: AsyncSession, entry_id: str, uri_r: str) -> None:
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


async def test_flags_url_above_lint_threshold(db_session: AsyncSession) -> None:
    # Default lint_min_distinct_sources is 3 — cite from 3 distinct sources.
    for i in range(3):
        await _add_entry(db_session, f"s{i}", f"https://s{i}.example/p")
        await record_url_mentions(
            db_session, source_entry_id=f"s{i}", canonical_urls=["https://press.example/x"]
        )
    findings = await _check_undeposited_cited_sources(db_session)
    assert len(findings) == 1
    f = findings[0]
    assert f.finding_type == "UNDEPOSITED_CITED_SOURCE"
    assert f.severity == "INFO"
    assert "https://press.example/x" in f.detail
    assert f.recommended_action is not None
    assert "particles deposit" in f.recommended_action


async def test_below_lint_threshold_is_silent(db_session: AsyncSession) -> None:
    # Two distinct sources — below the default lint threshold of 3.
    for i in range(2):
        await _add_entry(db_session, f"s{i}", f"https://s{i}.example/p")
        await record_url_mentions(
            db_session, source_entry_id=f"s{i}", canonical_urls=["https://press.example/x"]
        )
    assert await _check_undeposited_cited_sources(db_session) == []

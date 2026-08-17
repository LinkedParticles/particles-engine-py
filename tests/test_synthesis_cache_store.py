"""Tests for the shared synthesis cache store."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from particles.store.synthesis_cache_store import (
    evict_subject,
    list_cache_entries,
    lookup_cached_article,
    store_cached_article,
    vacuum_cache,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_then_lookup_returns_body(db_session: AsyncSession) -> None:
    await store_cached_article(
        db_session,
        subject_id="subj-1",
        input_hash="hash-aaa",
        prompt_version="v1",
        article_body="# Synthesised body\n\nCited claim [^p-1].",
    )
    body = await lookup_cached_article(db_session, "subj-1", "hash-aaa", "v1")
    assert body == "# Synthesised body\n\nCited claim [^p-1]."


@pytest.mark.asyncio
async def test_lookup_returns_none_on_miss(db_session: AsyncSession) -> None:
    body = await lookup_cached_article(db_session, "subj-1", "hash-aaa", "v1")
    assert body is None


# ---------------------------------------------------------------------------
# Composite key — different values isolate cache entries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_different_input_hash_isolates_entries(db_session: AsyncSession) -> None:
    await store_cached_article(db_session, "subj-1", "hash-aaa", "v1", "old body")
    await store_cached_article(db_session, "subj-1", "hash-bbb", "v1", "new body")
    assert await lookup_cached_article(db_session, "subj-1", "hash-aaa", "v1") == "old body"
    assert await lookup_cached_article(db_session, "subj-1", "hash-bbb", "v1") == "new body"


@pytest.mark.asyncio
async def test_different_prompt_version_isolates_entries(db_session: AsyncSession) -> None:
    """Same subject + same input under a new prompt version is a separate cache row.

    prompt_version is stored separately from input_hash so a
    future eviction sweep can target a single prompt version explicitly."""
    await store_cached_article(db_session, "subj-1", "hash-aaa", "v1", "v1 body")
    await store_cached_article(db_session, "subj-1", "hash-aaa", "v2", "v2 body")
    assert await lookup_cached_article(db_session, "subj-1", "hash-aaa", "v1") == "v1 body"
    assert await lookup_cached_article(db_session, "subj-1", "hash-aaa", "v2") == "v2 body"


@pytest.mark.asyncio
async def test_different_subject_isolates_entries(db_session: AsyncSession) -> None:
    await store_cached_article(db_session, "subj-1", "hash-aaa", "v1", "subj-1 body")
    await store_cached_article(db_session, "subj-2", "hash-aaa", "v1", "subj-2 body")
    assert await lookup_cached_article(db_session, "subj-1", "hash-aaa", "v1") == "subj-1 body"
    assert await lookup_cached_article(db_session, "subj-2", "hash-aaa", "v1") == "subj-2 body"


# ---------------------------------------------------------------------------
# Upsert — re-storing the same key updates in place
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_is_idempotent_upsert(db_session: AsyncSession) -> None:
    """Re-storing the same composite key replaces body + bumps generated_at.
    No duplicate-key violation, no second row."""
    await store_cached_article(
        db_session, "subj-1", "hash-aaa", "v1", "first body", layer_b_verdict="SUPPORTS"
    )
    await store_cached_article(
        db_session, "subj-1", "hash-aaa", "v1", "second body", layer_b_verdict="UNRELATED"
    )
    body = await lookup_cached_article(db_session, "subj-1", "hash-aaa", "v1")
    assert body == "second body"


@pytest.mark.asyncio
async def test_store_records_layer_b_verdict_and_quality_notes(
    db_session: AsyncSession,
) -> None:
    """layer_b_verdict and quality_notes are diagnostic — recorded but
    not returned by lookup. Verify they round-trip via direct ORM read."""
    from particles.store.synthesis_cache_store import SynthesisCacheRow

    await store_cached_article(
        db_session,
        "subj-1",
        "hash-aaa",
        "v1",
        "body",
        layer_b_verdict="SUPPORTS",
        quality_notes="layer A retried once",
    )
    row = await db_session.get(SynthesisCacheRow, ("subj-1", "hash-aaa", "v1"))
    assert row is not None
    assert row.layer_b_verdict == "SUPPORTS"
    assert row.quality_notes == "layer A retried once"


# ---------------------------------------------------------------------------
# Eviction (cross-subject stale-link path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evict_subject_removes_every_hash_and_prompt(
    db_session: AsyncSession,
) -> None:
    """evict_subject drops every cache row for the subject — across input_hash
    versions and prompt versions both. Wired into the
    --invalidate-stale-links path so renamed-subject invalidation forces a
    genuine re-synthesis next render."""
    await store_cached_article(db_session, "subj-1", "hash-aaa", "v1", "x")
    await store_cached_article(db_session, "subj-1", "hash-bbb", "v1", "y")
    await store_cached_article(db_session, "subj-1", "hash-aaa", "v2", "z")
    await store_cached_article(db_session, "subj-2", "hash-aaa", "v1", "other")

    count = await evict_subject(db_session, "subj-1")
    assert count == 3
    assert await lookup_cached_article(db_session, "subj-1", "hash-aaa", "v1") is None
    assert await lookup_cached_article(db_session, "subj-1", "hash-bbb", "v1") is None
    assert await lookup_cached_article(db_session, "subj-1", "hash-aaa", "v2") is None
    # Sibling subject's cache survives.
    assert await lookup_cached_article(db_session, "subj-2", "hash-aaa", "v1") == "other"


@pytest.mark.asyncio
async def test_evict_subject_returns_zero_on_unknown(db_session: AsyncSession) -> None:
    count = await evict_subject(db_session, "subj-never-seen")
    assert count == 0


# ---------------------------------------------------------------------------
# list + vacuum
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_cache_entries_returns_all_rows(db_session: AsyncSession) -> None:
    await store_cached_article(db_session, "subj-1", "hash-aaa", "v1", "a")
    await store_cached_article(db_session, "subj-2", "hash-bbb", "v1", "b")
    rows = await list_cache_entries(db_session)
    assert {r.subject_id for r in rows} == {"subj-1", "subj-2"}


async def _insert_subject(session: AsyncSession, subject_id: str) -> None:
    from particles.core.schema import Subject
    from particles.store.subject_store import insert_subject

    await insert_subject(
        session, Subject(id=subject_id, canonical_name=f"S {subject_id}", asserted_by="test")
    )


@pytest.mark.asyncio
async def test_vacuum_removes_stale_prompt_version(db_session: AsyncSession) -> None:
    await _insert_subject(db_session, "subj-1")
    await store_cached_article(db_session, "subj-1", "hash-aaa", "v-old", "stale")
    await store_cached_article(db_session, "subj-1", "hash-aaa", "v-cur", "current")

    counts = await vacuum_cache(db_session, current_prompt_version="v-cur")
    await db_session.commit()

    assert counts["stale_prompt_version"] == 1
    assert counts["orphaned_subject"] == 0
    assert await lookup_cached_article(db_session, "subj-1", "hash-aaa", "v-old") is None
    assert await lookup_cached_article(db_session, "subj-1", "hash-aaa", "v-cur") == "current"


@pytest.mark.asyncio
async def test_vacuum_removes_orphaned_subjects(db_session: AsyncSession) -> None:
    await _insert_subject(db_session, "subj-live")
    await store_cached_article(db_session, "subj-live", "hash-aaa", "v-cur", "kept")
    # No Subject row for "subj-ghost" → orphaned.
    await store_cached_article(db_session, "subj-ghost", "hash-bbb", "v-cur", "orphan")

    counts = await vacuum_cache(db_session, current_prompt_version="v-cur")
    await db_session.commit()

    assert counts["stale_prompt_version"] == 0
    assert counts["orphaned_subject"] == 1
    assert await lookup_cached_article(db_session, "subj-live", "hash-aaa", "v-cur") == "kept"
    assert await lookup_cached_article(db_session, "subj-ghost", "hash-bbb", "v-cur") is None


@pytest.mark.asyncio
async def test_vacuum_noop_when_all_reachable(db_session: AsyncSession) -> None:
    await _insert_subject(db_session, "subj-1")
    await store_cached_article(db_session, "subj-1", "hash-aaa", "v-cur", "body")
    counts = await vacuum_cache(db_session, current_prompt_version="v-cur")
    assert counts == {"stale_prompt_version": 0, "orphaned_subject": 0}

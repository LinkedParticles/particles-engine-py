"""Shared synthesis cache.

Per-Subject LLM-synthesised article bodies live in
``synthesis_cache``, keyed on ``(subject_id, input_hash,
prompt_version)``. Every prose exporter (wiki, obsidian, future
logseq) consults the table before invoking the LLM; cache hit
returns the stored body, cache miss synthesises and stores.

The key triple encodes "this exact input under this prompt version
produced this exact output." :func:`compute_input_hash` already
mixes ``prompt_version`` into the hash, but ``prompt_version``
is stored separately here so future eviction can target a single
prompt version cleanly without re-deriving it from the hash.

Eviction is operator-pull: the table never
auto-evicts; the ``particles synthesis-cache`` verb group
(``list`` / ``show`` / ``vacuum`` / ``evict``) is the
operator's read + cleanup surface over it. Storage cost is small —
~5 KB per cached article — so the table grows linearly with
synthesis events but doesn't need TTL.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, String, Text, delete, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.db import Base


class SynthesisCacheRow(Base):
    """One ``(subject_id, input_hash, prompt_version)`` cached article body."""

    __tablename__ = "synthesis_cache"

    subject_id: Mapped[str] = mapped_column(String, primary_key=True)
    input_hash: Mapped[str] = mapped_column(String, primary_key=True)
    prompt_version: Mapped[str] = mapped_column(String, primary_key=True)

    article_body: Mapped[str] = mapped_column(Text, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # Layer B verdict recorded at write time — diagnostic only.
    layer_b_verdict: Mapped[str | None] = mapped_column(String, nullable=True)
    quality_notes: Mapped[str] = mapped_column(Text, nullable=False, default="")


async def lookup_cached_article(
    session: AsyncSession,
    subject_id: str,
    input_hash: str,
    prompt_version: str,
) -> str | None:
    """Return the cached article body for this key, or ``None``."""
    result = await session.execute(
        select(SynthesisCacheRow.article_body).where(
            SynthesisCacheRow.subject_id == subject_id,
            SynthesisCacheRow.input_hash == input_hash,
            SynthesisCacheRow.prompt_version == prompt_version,
        )
    )
    return result.scalar_one_or_none()


async def store_cached_article(
    session: AsyncSession,
    subject_id: str,
    input_hash: str,
    prompt_version: str,
    article_body: str,
    *,
    layer_b_verdict: str | None = None,
    quality_notes: str = "",
) -> None:
    """Upsert the cache entry. Idempotent on the composite key."""
    existing = await session.execute(
        select(SynthesisCacheRow).where(
            SynthesisCacheRow.subject_id == subject_id,
            SynthesisCacheRow.input_hash == input_hash,
            SynthesisCacheRow.prompt_version == prompt_version,
        )
    )
    row = existing.scalar_one_or_none()
    now = datetime.now(UTC)
    if row is not None:
        row.article_body = article_body
        row.generated_at = now
        row.layer_b_verdict = layer_b_verdict
        row.quality_notes = quality_notes
        return
    session.add(
        SynthesisCacheRow(
            subject_id=subject_id,
            input_hash=input_hash,
            prompt_version=prompt_version,
            article_body=article_body,
            generated_at=now,
            layer_b_verdict=layer_b_verdict,
            quality_notes=quality_notes,
        )
    )


async def list_cache_entries(session: AsyncSession) -> list[SynthesisCacheRow]:
    """Every cache row, newest first. Backs ``synthesis-cache list``."""
    result = await session.execute(
        select(SynthesisCacheRow).order_by(SynthesisCacheRow.generated_at.desc())
    )
    return list(result.scalars().all())


async def vacuum_cache(session: AsyncSession, *, current_prompt_version: str) -> dict[str, int]:
    """Delete provably-unreachable cache rows. Caller commits.

    A row can never be served again when either:

    * its ``prompt_version`` is no longer the live one — :func:`lookup_cached_article`
      always keys on the current ``_PROMPT_VERSION``, so older versions are dead, or
    * its ``subject_id`` no longer exists (the Subject was deleted).

    Returns ``{reason: deleted_count}``. The reasons are disjoint — a stale-prompt
    row is deleted (and counted) first, so it is not re-counted as orphaned. A
    dry run is the caller rolling back instead of committing.
    """
    from particles.store.subject_store import SubjectRow

    stale: CursorResult[None] = await session.execute(  # type: ignore[assignment]
        delete(SynthesisCacheRow).where(SynthesisCacheRow.prompt_version != current_prompt_version)
    )
    stale_count = stale.rowcount or 0

    existing_subject_ids = set((await session.execute(select(SubjectRow.id))).scalars().all())
    cached_subject_ids = set(
        (await session.execute(select(SynthesisCacheRow.subject_id))).scalars().all()
    )
    orphan_ids = cached_subject_ids - existing_subject_ids
    orphaned = 0
    if orphan_ids:
        res: CursorResult[None] = await session.execute(  # type: ignore[assignment]
            delete(SynthesisCacheRow).where(SynthesisCacheRow.subject_id.in_(orphan_ids))
        )
        orphaned = res.rowcount or 0
    return {"stale_prompt_version": stale_count, "orphaned_subject": orphaned}


async def evict_subject(session: AsyncSession, subject_id: str) -> int:
    """Delete every cache row for ``subject_id``. Returns deletion count.

    Wired into the ``--invalidate-stale-links`` path so a
    renamed-subject invalidation evicts the DB entries alongside
    stripping the on-disk frontmatter hash. The next render genuinely
    re-runs the LLM rather than rehydrating stale prose from the
    cache.
    """
    result: CursorResult[None] = await session.execute(  # type: ignore[assignment]
        delete(SynthesisCacheRow).where(SynthesisCacheRow.subject_id == subject_id)
    )
    return result.rowcount or 0

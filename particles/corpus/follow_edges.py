"""Cross-entry follow provenance.

When `deposit_url` follows a link-shaped post's primary URL and
deposits the target as a separate corpus entry, the relationship is
recorded as a row in ``corpus_follow_edges``. The shape is a join
table because a single article URL is commonly reached from many
sources (press releases, viral links: the same article gets shared
on Reddit, HN, multiple tweets/statuses) — a fan-in pattern a
single column on ``corpus_entries`` could not represent.

The table is **not** foreign-keyed to ``corpus_entries`` on either
side: ``particles corpus delete <id>`` is operator-deliberate, and
the operator who deletes a Reddit thread doesn't necessarily want
the linked-article entry deleted (it may have useful particles,
may be the target of other follows). Deleting either side leaves
dangling edges; a future ``corpus delete --cascade-follows`` flag
can offer the opt-in sweep § Deferred.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, String, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.db import Base


class CorpusFollowEdgeRow(Base):
    """One ``(via, target, link_type)`` follow relationship."""

    __tablename__ = "corpus_follow_edges"
    __table_args__ = (Index("ix_corpus_follow_edges_target", "target_entry_id"),)

    via_entry_id: Mapped[str] = mapped_column(String, primary_key=True)
    target_entry_id: Mapped[str] = mapped_column(String, primary_key=True)
    link_type: Mapped[str] = mapped_column(String, primary_key=True)
    discovered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


# Link-type sentinel values. ``POST_LINK`` is the only type written
# today; ``COMMENT_LINK`` is reserved for the deferred
# comment-link-following feature so the column shape doesn't need a
# schema migration when that ships.
LINK_TYPE_POST = "POST_LINK"
LINK_TYPE_COMMENT = "COMMENT_LINK"


async def add_follow_edge(
    session: AsyncSession,
    *,
    via_entry_id: str,
    target_entry_id: str,
    link_type: str = LINK_TYPE_POST,
) -> None:
    """Record a follow edge, idempotently.

    A duplicate ``(via, target, link_type)`` triple is a no-op — the
    edge already exists. ``write_entry_and_snapshot``'s content-hash
    dedup means the same URL may be re-followed multiple times for
    the same source; we'd want to record that once, not N times.
    """
    from datetime import UTC

    existing = await session.execute(
        select(CorpusFollowEdgeRow).where(
            CorpusFollowEdgeRow.via_entry_id == via_entry_id,
            CorpusFollowEdgeRow.target_entry_id == target_entry_id,
            CorpusFollowEdgeRow.link_type == link_type,
        )
    )
    if existing.scalar_one_or_none() is not None:
        return
    session.add(
        CorpusFollowEdgeRow(
            via_entry_id=via_entry_id,
            target_entry_id=target_entry_id,
            link_type=link_type,
            discovered_at=datetime.now(UTC),
        )
    )


async def get_follow_targets(session: AsyncSession, via_entry_id: str) -> list[CorpusFollowEdgeRow]:
    """Every entry deposited as a follow-of ``via_entry_id``."""
    result = await session.execute(
        select(CorpusFollowEdgeRow).where(CorpusFollowEdgeRow.via_entry_id == via_entry_id)
    )
    return list(result.scalars())


async def get_follow_sources(
    session: AsyncSession, target_entry_id: str
) -> list[CorpusFollowEdgeRow]:
    """Every entry that linked to ``target_entry_id`` — the fan-in view.

    Single article reached from many sources (press release, viral
    link): N rows, one per source. The ``ix_corpus_follow_edges_target``
    index makes this a single lookup.
    """
    result = await session.execute(
        select(CorpusFollowEdgeRow).where(CorpusFollowEdgeRow.target_entry_id == target_entry_id)
    )
    return list(result.scalars())

"""Persistent cache for Wikidata property and item labels.

Labels are fetched once from the Wikidata REST API and stored here.
Subsequent runs read from the DB rather than calling the API.
The in-process dict cache in wikidata.py remains as the L1 layer;
this table is the L2 layer that survives between processes.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from sqlalchemy import DateTime, String, Text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.db import Base


class WikidataLabelRow(Base):
    __tablename__ = "wikidata_label_cache"

    qid: Mapped[str] = mapped_column(String, primary_key=True)  # "P571", "Q64", etc.
    label: Mapped[str] = mapped_column(Text, nullable=False)
    cached_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


async def get_label(session: AsyncSession, qid: str) -> str | None:
    row = await session.get(WikidataLabelRow, qid)
    return row.label if row else None


async def set_label(session: AsyncSession, qid: str, label: str) -> None:
    existing = await session.get(WikidataLabelRow, qid)
    if existing:
        existing.label = label
        existing.cached_at = datetime.now(UTC)
    else:
        session.add(
            WikidataLabelRow(
                qid=qid,
                label=label,
                cached_at=datetime.now(UTC),
            )
        )
    await session.flush()


async def _persistent_label_cache(
    entity_id: str,
    fetch_live: Callable[[], Awaitable[str]],
    session: AsyncSession | None,
) -> str:
    """L2 persistent label cache.

    On a DB miss, runs the Client-supplied ``fetch_live`` (the L3 REST call) and
    persists the result. Manages its own session when the caller passes none —
    preserving the prior ``_fetch_label`` behaviour, now Engine-side.
    """

    async def _lookup(sess: AsyncSession) -> str:
        db_label = await get_label(sess, entity_id)
        if db_label is not None:
            return db_label
        label = await fetch_live()
        await set_label(sess, entity_id, label)
        return label

    if session is not None:
        return await _lookup(session)

    from particles.db import session_scope

    async with session_scope() as own_session:
        result = await _lookup(own_session)
        await own_session.commit()
        return result


# Inverted persistent-label-cache coupling: the Engine
# registers its store-backed L2 cache with the Client-layer ``extraction.wikidata``
# module at import time, so the extractor resolves labels without importing any
# Engine module. ``_orm_modules`` imports this module (ORM registration) and
# ``ingest.pipeline`` imports it explicitly, so the cache is set before any
# Wikidata extraction runs. Engine→Client import is allowed.
from particles.extraction.wikidata import register_label_cache  # noqa: E402

register_label_cache(_persistent_label_cache)

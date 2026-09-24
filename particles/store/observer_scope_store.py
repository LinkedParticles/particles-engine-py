# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Observer-scope statements — lens-side policy that widens a belief.

"This rule learned in one project is how I work everywhere" is an operator's
judgement, so it is recorded the way trust judgements are: as a
store-resident statement the read lens consults, never as a change to the
claim or to its provenance. A statement targets one particle or one whole
corpus entry and says it is in view for every project observer.

Only widening exists. Statements are written by an operator verb and are
deliberately absent from the agent write surface: an agent that could widen its
own belief could put it in front of every future session (the
argument).
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import DateTime, String, delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.db import Base

_IN_CHUNK = 500


class ScopeTarget(StrEnum):
    """What a statement is about."""

    PARTICLE = "particle"
    CORPUS_ENTRY = "corpus_entry"


class ObserverScopeStatementRow(Base):
    __tablename__ = "observer_scope_statements"

    target_kind: Mapped[str] = mapped_column(String, primary_key=True)
    target_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_by: Mapped[str] = mapped_column(String, nullable=False)


async def add_widening(
    session: AsyncSession, kind: ScopeTarget, target_id: str, *, created_by: str
) -> bool:
    """Record that ``target_id`` is in view everywhere. Returns ``False`` if it already was."""
    if await session.get(ObserverScopeStatementRow, (kind.value, target_id)) is not None:
        return False
    session.add(
        ObserverScopeStatementRow(
            target_kind=kind.value,
            target_id=target_id,
            created_at=datetime.now(UTC),
            created_by=created_by,
        )
    )
    await session.flush()
    return True


async def remove_widening(session: AsyncSession, kind: ScopeTarget, target_id: str) -> bool:
    """Withdraw a widening. Returns ``False`` if there was none."""
    if await session.get(ObserverScopeStatementRow, (kind.value, target_id)) is None:
        return False
    await session.execute(
        delete(ObserverScopeStatementRow).where(
            ObserverScopeStatementRow.target_kind == kind.value,
            ObserverScopeStatementRow.target_id == target_id,
        )
    )
    await session.flush()
    return True


async def widened_ids(
    session: AsyncSession, kind: ScopeTarget, target_ids: Collection[str]
) -> set[str]:
    """The subset of ``target_ids`` an operator has widened."""
    ids: Sequence[str] = list(target_ids)
    found: set[str] = set()
    for start in range(0, len(ids), _IN_CHUNK):
        rows = await session.execute(
            select(ObserverScopeStatementRow.target_id).where(
                ObserverScopeStatementRow.target_kind == kind.value,
                ObserverScopeStatementRow.target_id.in_(ids[start : start + _IN_CHUNK]),
            )
        )
        found.update(str(row) for row in rows.scalars())
    return found


async def any_widening(session: AsyncSession) -> bool:
    """Whether the store holds any statement at all — lets a read skip two lookups."""
    row = await session.execute(select(ObserverScopeStatementRow.target_id).limit(1))
    return row.first() is not None

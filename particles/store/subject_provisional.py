# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The provisional mark on a Subject an agent principal minted.

A provisional Subject is an ordinary ``subjects`` row plus one row here. The
row exists only while the Subject is provisional: accepting it deletes the row
and writes an event, so a Subject with no row is confirmed, and every Subject
in a store that predates the table is. The mark is write-authority state of
the store node, changed only by the named transitions below, and never a
field of the ``Subject`` record or of the interchange format.

This module imports no other store module at its top, so ``subject_store``
can read and clear marks without a cycle.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import model_serializer
from sqlalchemy import DateTime, String, delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.core.provisional import ProvisionalMark
from particles.core.schema import Subject
from particles.db import Base
from particles.store import subject_cache
from particles.store.event_store import EventRefKind, OperatorEventType, record_event

_IN_CHUNK = 10_000


class SubjectProvisionalRow(Base):
    """One provisional Subject's mark."""

    __tablename__ = "subject_provisional"

    subject_id: Mapped[str] = mapped_column(String, primary_key=True)
    principal: Mapped[str] = mapped_column(String, nullable=False, index=True)
    minted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    shared_by: Mapped[str | None] = mapped_column(String, nullable=True)
    shared_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def to_mark(self) -> ProvisionalMark:
        return ProvisionalMark(
            principal=self.principal,
            minted_at=self.minted_at,
            shared_by=self.shared_by,
            shared_at=self.shared_at,
        )


class AcceptedVia(StrEnum):
    """How a provisional Subject came to answer everyone."""

    #: An operator's act addressed to the Subject by id.
    OPERATOR = "OPERATOR"
    #: An independent resolution reached the Subject's external reference.
    EXTERNAL_REF = "EXTERNAL_REF"


class SubjectView(Subject):
    """A Subject as a read surface returns it, with its provisional mark.

    ``provisional`` is present on a provisional Subject and the key is left
    out on a confirmed one, so a confirmed Subject reads exactly as before.
    """

    provisional: ProvisionalMark | None = None

    @model_serializer(mode="wrap")
    def _omit_confirmed(self, handler: Any) -> Any:
        data = handler(self)
        if isinstance(data, dict) and data.get("provisional") is None:
            data.pop("provisional", None)
        return data


def _clear_cache() -> None:
    subject_cache.clear(keep_negative=True)


async def marks_for(
    session: AsyncSession, subject_ids: Collection[str]
) -> dict[str, ProvisionalMark]:
    """The provisional marks among ``subject_ids``; a confirmed Subject is absent."""
    ids = sorted({sid for sid in subject_ids if sid})
    marks: dict[str, ProvisionalMark] = {}
    for start in range(0, len(ids), _IN_CHUNK):
        rows = await session.execute(
            select(SubjectProvisionalRow).where(
                SubjectProvisionalRow.subject_id.in_(ids[start : start + _IN_CHUNK])
            )
        )
        for row in rows.scalars():
            marks[row.subject_id] = row.to_mark()
    return marks


async def mark_of(session: AsyncSession, subject_id: str) -> ProvisionalMark | None:
    """The Subject's provisional mark, or ``None`` when it is confirmed."""
    row = await session.get(SubjectProvisionalRow, subject_id)
    return row.to_mark() if row is not None else None


async def any_provisional(session: AsyncSession) -> bool:
    """Whether the store holds any provisional Subject (a fast skip for the fold)."""
    found = await session.execute(select(SubjectProvisionalRow.subject_id).limit(1))
    return found.first() is not None


async def mark_provisional(
    session: AsyncSession,
    subject_id: str,
    principal: str,
    *,
    minted_at: datetime | None = None,
) -> ProvisionalMark:
    """Mark a newly minted Subject provisional, scoped to ``principal`` (§1).

    The mint itself is the act, so no event is written here; the backfill
    writes its own (§7).
    """
    row = SubjectProvisionalRow(
        subject_id=subject_id,
        principal=principal,
        minted_at=minted_at or datetime.now(UTC),
    )
    session.add(row)
    await session.flush()
    # No cache clear: a Subject just minted is in no cached answer, and the
    # backfill, which marks existing Subjects, clears once when it is done.
    return row.to_mark()


async def share(session: AsyncSession, subject_id: str, by_principal: str) -> bool:
    """Widen a scoped provisional Subject to every agent principal (§4 Share).

    Returns ``False`` (and writes nothing) when the Subject is confirmed,
    already shared, or ``by_principal`` is its own principal.
    """
    row = await session.get(SubjectProvisionalRow, subject_id)
    if row is None or row.shared_by is not None or row.principal == by_principal:
        return False
    row.shared_by = by_principal
    row.shared_at = datetime.now(UTC)
    await session.flush()
    await record_event(
        session,
        actor=by_principal,
        event_type=OperatorEventType.SUBJECT_SHARED,
        refs=[(EventRefKind.SUBJECT, subject_id)],
        payload={"principal": row.principal, "shared_by": by_principal},
    )
    # No cache clear: every answer cached for this Subject stays right, since
    # sharing only widens who it answers.
    return True


async def accept(
    session: AsyncSession,
    subject_id: str,
    *,
    via: AcceptedVia,
    actor: str,
    detail: dict[str, Any] | None = None,
) -> bool:
    """Confirm a provisional Subject for every resolution (§4 Confirm).

    Deletes the mark and writes ``SUBJECT_ACCEPTED`` naming ``via``. Returns
    ``False`` (and writes nothing) when the Subject is already confirmed.
    """
    row = await session.get(SubjectProvisionalRow, subject_id)
    if row is None:
        return False
    mark = row.to_mark()
    await session.delete(row)
    await session.flush()
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.SUBJECT_ACCEPTED,
        refs=[(EventRefKind.SUBJECT, subject_id)],
        payload={
            "via": via.value,
            "principal": mark.principal,
            "shared_by": mark.shared_by,
            **(detail or {}),
        },
    )
    _clear_cache()
    return True


async def drop_marks(session: AsyncSession, subject_ids: Collection[str]) -> None:
    """Remove the marks of Subjects that are being deleted (a merge source, a delete).

    No event: the delete or merge that removes the Subject records its own.
    """
    ids = sorted({sid for sid in subject_ids if sid})
    for start in range(0, len(ids), _IN_CHUNK):
        await session.execute(
            delete(SubjectProvisionalRow).where(
                SubjectProvisionalRow.subject_id.in_(ids[start : start + _IN_CHUNK])
            )
        )


async def list_marks(session: AsyncSession) -> list[tuple[str, ProvisionalMark]]:
    """Every provisional Subject's id and mark, oldest mint first."""
    rows = await session.execute(
        select(SubjectProvisionalRow).order_by(
            SubjectProvisionalRow.minted_at, SubjectProvisionalRow.subject_id
        )
    )
    return [(row.subject_id, row.to_mark()) for row in rows.scalars()]


async def with_marks(session: AsyncSession, subjects: Sequence[Subject]) -> list[SubjectView]:
    """``subjects`` as read surfaces return them, each with its mark (§5)."""
    marks = await marks_for(session, [s.id for s in subjects])
    return [SubjectView(**s.model_dump(), provisional=marks.get(s.id)) for s in subjects]


async def with_mark(session: AsyncSession, subject: Subject | None) -> SubjectView | None:
    """One Subject with its mark, or ``None``."""
    if subject is None:
        return None
    return (await with_marks(session, [subject]))[0]

# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""What each session was shown at session start.

One row per SessionStart that delivered context: the session id and time, the
SessionStart source (``startup`` / ``clear`` / ``compact``), the hook's action
(``full`` / ``diff`` / ``skip``), the session's resolved project key, whether
the digest was read through observer scope, and the beliefs the session
received, in the order shown. The hook writes the row after it has cut the
injected text to its byte budget, so a row holds what the session received and
never what the engine rendered.

Each row is a **primary record** (D1): it cannot be re-derived,
because the ranking behind a projection or a digest depends on config, lens and
utility state at that moment. It therefore lives in its own module, outside the
utility store, and no clear, rebuild, re-mine or consolidation pass touches it.
``db init --force`` keeps it (``PRESERVED_TABLES``). Rows are
only appended.

No read surface exposes the rows: :func:`session_exposures_for`
is the accessor for the readers that follow. Retention is uncapped for now.

The functions flush and never commit: the caller owns the transaction.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Collection
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import (
    Boolean,
    ColumnElement,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    func,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.core.observer_scope import canonical_project_key
from particles.db import Base

__all__ = [
    "PRESERVED_TABLES",
    "SessionExposure",
    "SessionExposureRow",
    "ShownBelief",
    "record_session_exposure",
    "session_exposures_for",
]

#: Tables a store rebuild keeps (``db init --force``): this record cannot be
#: re-derived from the corpus.
PRESERVED_TABLES: frozenset[str] = frozenset({"session_exposures"})

#: Length of a full particle id (a UUID string). Anything shorter is a display
#: prefix (``p-<shortid>``) and is resolved at write time.
_FULL_ID_LEN = 36


class ShownBelief(BaseModel):
    """One belief a session was shown, in the order it was shown."""

    particle_id: str = Field(
        description="The belief's id. A short id that matches no stored particle is kept as given."
    )
    shown_as: Literal["projection", "digest"] = Field(
        description="How it was shown: a projected MEMORY.md line, or a digest line."
    )
    contested_bases: list[str] = Field(
        default_factory=list,
        description="The contested bases the shown line displayed; empty when none.",
    )


class SessionExposure(BaseModel):
    """One SessionStart's delivered context."""

    session_id: str
    recorded_at: datetime
    source: str = Field(description="The SessionStart source: startup, clear or compact.")
    action: Literal["full", "diff", "skip"]
    project_key: str = Field(description="The session's resolved project key.")
    observer_scope_applied: bool = Field(
        description="Whether the session read the store through its project observer."
    )
    beliefs: list[ShownBelief] = Field(default_factory=list)


#: ``recorded_by`` on a row the in-process writer (the hook writing its own
#: store) appended. No key principal is blank (``keys create`` and the record
#: rule refuse one), so an HTTP write never records it.
IN_PROCESS_RECORDER = ""


class SessionExposureRow(Base):
    """One SessionStart that delivered context. Append-only."""

    __tablename__ = "session_exposures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    action: Mapped[str] = mapped_column(String(8), nullable=False)
    project_key: Mapped[str] = mapped_column(String, nullable=False)
    observer_scope_applied: Mapped[bool] = mapped_column(Boolean, nullable=False)
    #: The shown beliefs in order, as a JSON array of :class:`ShownBelief`.
    beliefs: Mapped[str] = mapped_column(Text, nullable=False)
    #: The engine key principal that wrote the row over HTTP;
    #: :data:`IN_PROCESS_RECORDER` for an in-process write. ``None`` on a row
    #: whose writer was never recorded (one older than the column, or carried
    #: through a downgrade of it), which no reader trusts.
    recorded_by: Mapped[str | None] = mapped_column(String, nullable=True)

    __table_args__ = (Index("ix_session_exposures_session", "session_id", "recorded_at"),)

    def to_model(self) -> SessionExposure:
        recorded_at = self.recorded_at
        if recorded_at.tzinfo is None:
            # SQLite drops the offset; every row is written in UTC.
            recorded_at = recorded_at.replace(tzinfo=UTC)
        return SessionExposure(
            session_id=self.session_id,
            recorded_at=recorded_at,
            source=self.source,
            action=self.action,  # type: ignore[arg-type]
            project_key=self.project_key,
            observer_scope_applied=self.observer_scope_applied,
            beliefs=[ShownBelief.model_validate(b) for b in json.loads(self.beliefs)],
        )


async def _resolve_ids(session: AsyncSession, ids: set[str]) -> dict[str, str]:
    """Map each display prefix in ``ids`` to the one stored particle id it names.

    A prefix that matches no particle, or more than one, is left out, and the
    caller keeps it as given. Every status is searched: a belief retracted
    since it was rendered was still shown.
    """
    from particles.store.particle_store import ParticleRow

    by_len: dict[int, set[str]] = defaultdict(set)
    for raw in ids:
        if len(raw) < _FULL_ID_LEN:
            by_len[len(raw)].add(raw)
    resolved: dict[str, str] = {}
    for length, prefixes in by_len.items():
        rows = await session.execute(
            select(ParticleRow.id).where(func.substr(ParticleRow.id, 1, length).in_(prefixes))
        )
        matches: dict[str, list[str]] = defaultdict(list)
        for (full,) in rows:
            matches[full[:length]].append(full)
        resolved.update({p: hits[0] for p, hits in matches.items() if len(hits) == 1})
    return resolved


async def record_session_exposure(
    session: AsyncSession, exposure: SessionExposure, *, recorded_by: str | None = None
) -> int:
    """Append one session-start row, resolving ``p-`` short ids to full ids.

    ``recorded_by`` is the engine key principal of an HTTP write; an in-process
    write (none given) records :data:`IN_PROCESS_RECORDER`. Returns the new
    row's id. Earlier rows, including ones for the same session, are never
    changed.
    """
    stripped = [
        b.model_copy(update={"particle_id": b.particle_id.removeprefix("p-")})
        for b in exposure.beliefs
    ]
    resolved = await _resolve_ids(session, {b.particle_id for b in stripped})
    beliefs = [
        b.model_copy(update={"particle_id": resolved.get(b.particle_id, b.particle_id)})
        for b in stripped
    ]
    row = SessionExposureRow(
        session_id=exposure.session_id,
        recorded_at=exposure.recorded_at,
        source=exposure.source,
        action=exposure.action,
        # Stored as reads compare it, so one project is one key whatever form
        # the writer passed (security review 2026-10-30-1, AB69).
        project_key=(
            canonical_project_key(exposure.project_key)
            if exposure.project_key
            else exposure.project_key
        ),
        observer_scope_applied=exposure.observer_scope_applied,
        beliefs=json.dumps([b.model_dump() for b in beliefs]),
        recorded_by=IN_PROCESS_RECORDER if recorded_by is None else recorded_by,
    )
    session.add(row)
    await session.flush()
    return row.id


def recorded_by_one_of(recorders: Collection[str]) -> ColumnElement[bool]:
    """The rows an in-process writer recorded or one of ``recorders`` did.

    A row with no ``recorded_by`` is not among them. Its writer was never
    recorded: before migration 060 an agent key's HTTP row looked exactly like
    the hook's own, and a downgrade of 060 forgets every writer (security
    review 2026-10-28-2, AA44).
    """
    return SessionExposureRow.recorded_by.in_([IN_PROCESS_RECORDER, *sorted(recorders)])


async def session_exposures_for(
    session: AsyncSession, session_id: str, *, recorders: Collection[str] | None = None
) -> list[SessionExposure]:
    """Every session-start row recorded for ``session_id``, oldest first.

    With ``recorders``, only the rows an in-process writer recorded or one of
    those principals did (:func:`recorded_by_one_of`).
    """
    query = select(SessionExposureRow).where(SessionExposureRow.session_id == session_id)
    if recorders is not None:
        query = query.where(recorded_by_one_of(recorders))
    rows = await session.execute(
        query.order_by(SessionExposureRow.recorded_at, SessionExposureRow.id)
    )
    return [row.to_model() for row in rows.scalars()]

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
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import Boolean, DateTime, Index, Integer, String, Text, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

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


async def record_session_exposure(session: AsyncSession, exposure: SessionExposure) -> int:
    """Append one session-start row, resolving ``p-`` short ids to full ids.

    Returns the new row's id. Earlier rows, including ones for the same
    session, are never changed.
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
        project_key=exposure.project_key,
        observer_scope_applied=exposure.observer_scope_applied,
        beliefs=json.dumps([b.model_dump() for b in beliefs]),
    )
    session.add(row)
    await session.flush()
    return row.id


async def session_exposures_for(session: AsyncSession, session_id: str) -> list[SessionExposure]:
    """Every session-start row recorded for ``session_id``, oldest first."""
    rows = await session.execute(
        select(SessionExposureRow)
        .where(SessionExposureRow.session_id == session_id)
        .order_by(SessionExposureRow.recorded_at, SessionExposureRow.id)
    )
    return [row.to_model() for row in rows.scalars()]

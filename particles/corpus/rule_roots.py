# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The rule-source roots an operator enrolled.

``rules sync`` and ``init claude-code`` record each root they resolve, in the
transaction that deposits its files. The local refresh tier's allowlist reads
these rows rather than the refreshing process's working directory, so a
refresh admits the same files from any directory (security review N38).

Rows are never inferred from corpus entries: an enrolled entry's existence is
not evidence the operator chose its root (security review D2). Nor are they
removed: a root that no longer exists admits nothing, since its files are gone,
and a later sync from the same place only touches ``recorded_at``.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import DateTime, String, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.db import Base


class RuleSourceRootRow(Base):
    """One enrolled rule-source root, keyed by its resolved path."""

    __tablename__ = "rule_source_roots"

    path: Mapped[str] = mapped_column(String, primary_key=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    recorded_by: Mapped[str] = mapped_column(String, nullable=False)


async def record_rule_source_roots(
    session: AsyncSession, roots: Iterable[Path], *, by: str
) -> None:
    """Record each root (resolved), or refresh ``recorded_at`` on one already recorded.

    Does not commit: the caller records the roots in the transaction that
    deposits their files.
    """
    now = datetime.now(UTC)
    for root in sorted({Path(r).resolve() for r in roots}):
        row = await session.get(RuleSourceRootRow, str(root))
        if row is None:
            session.add(RuleSourceRootRow(path=str(root), recorded_at=now, recorded_by=by))
        else:
            row.recorded_at = now
    await session.flush()


async def list_rule_source_roots(session: AsyncSession) -> list[Path]:
    """Every recorded root, in path order."""
    result = await session.execute(select(RuleSourceRootRow.path).order_by(RuleSourceRootRow.path))
    return [Path(p) for p in result.scalars()]

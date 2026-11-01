# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The rule-source roots an operator enrolled.

``rules sync`` and ``init claude-code`` record each root they resolve, in the
transaction that deposits its files. The local refresh tier's allowlist reads
these rows rather than the refreshing process's working directory, so a
refresh admits the same files from any directory (security review N38).

Rows are never inferred from corpus entries: an enrolled entry's existence is
not evidence the operator chose its root (security review D2). Nothing
removes one on its own: a root that no longer exists admits nothing, since its
files are gone, and a later sync from the same place only touches
``recorded_at``. The operator removes a row with ``particles rules forget``,
which finds it with :func:`find_rule_source_root` and deletes it
with :func:`delete_rule_source_root`.

A broad root (:func:`is_broad_root`: the filesystem root, the home directory,
or an ancestor of either) is never recorded, and a row naming one admits
nothing (security review K9, P49): under a home-directory dotfiles repository
the nearest ``.git`` ancestor is the home directory, and recording it would
admit every rule-named file the user owns, permanently.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import DateTime, String, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.config import is_broad_root
from particles.db import Base

log = logging.getLogger(__name__)


_EUID_WARNED = False


def effective_uid() -> int | None:
    """The refreshing process's effective uid, or ``None`` where the platform has none.

    With no effective uid the ownership checks cannot be made, so the local
    tier reads no host file, and the first call per process
    says so.
    """
    global _EUID_WARNED
    geteuid = getattr(os, "geteuid", None)
    if geteuid is not None:
        return int(geteuid())
    if not _EUID_WARNED:
        _EUID_WARNED = True
        log.warning(
            "This platform reports no effective uid, so the local refresh cannot check who "
            "may write its files, and reads no host file."
        )
    return None


def owned_by_refresher(path: Path) -> bool:
    """Whether the refreshing user owns ``path``.

    ``False`` when it cannot be stat'ed. ``True`` on a platform with no
    effective uid (:func:`effective_uid`).
    """
    uid = effective_uid()
    if uid is None:
        return True
    try:
        return os.stat(path).st_uid == uid
    except OSError:
        return False


def owner_name(path: Path) -> str:
    """The name of the user who owns ``path``, or their uid when it has none."""
    try:
        uid = os.stat(path).st_uid
    except OSError:
        return "an unknown user"
    try:
        import pwd  # POSIX only; a platform without it names the uid

        return pwd.getpwuid(uid).pw_name
    except (ImportError, KeyError):
        return f"uid {uid}"


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

    A broad root (:func:`is_broad_root`), or a directory another user owns
    (:func:`owned_by_refresher`), is skipped with a warning. Does not
    commit: the caller records the roots in the transaction that deposits
    their files.
    """
    now = datetime.now(UTC)
    for root in sorted({Path(r).resolve() for r in roots}):
        if is_broad_root(root):
            log.warning("rule_roots: %s is too broad to record as a refresh root; skipped", root)
            continue
        if root.is_dir() and not owned_by_refresher(root):
            log.warning(
                "rule_roots: %s is owned by %s, not by you, so it is not recorded as a "
                "refresh root. If you trust it, add it to "
                "local_refresh.allowed_roots.",
                root,
                owner_name(root),
            )
            continue
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


async def find_rule_source_root(session: AsyncSession, path: str) -> RuleSourceRootRow | None:
    """The recorded row ``path`` names, or ``None``.

    The argument is matched as typed against the stored key first, then made
    absolute, then resolved the way :func:`record_rule_source_roots` resolves.
    The literal match comes first because the rows an operator removes are
    usually the missing ones, whose path may no longer resolve, and a root
    recorded through a symlink that has since been repointed resolves
    somewhere else.
    """
    candidates = [path, os.path.abspath(path), str(Path(path).resolve())]
    for key in dict.fromkeys(candidates):
        row = await session.get(RuleSourceRootRow, key)
        if row is not None:
            return row
    return None


async def delete_rule_source_root(session: AsyncSession, row: RuleSourceRootRow) -> None:
    """Delete one recorded row. Does not commit; the caller records the event beside it."""
    await session.delete(row)
    await session.flush()

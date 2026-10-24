# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The refusal hold as the bulk extraction paths apply it.

``particles/corpus/refusals.py`` owns the table and the query; this module
supplies what the query needs from configuration (the pairing each purpose
resolves to now, and ``extraction.refusal_retry_days``), so every bulk path
(``extract --all-pending``, the consolidation catch-up, the audit) asks the
same question the same way.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.corpus.refusals import HeldSnapshot, list_held_snapshots


def current_refusal_pairings() -> dict[str, str]:
    """``{purpose: provider_model}`` a refusal is compared against now.

    Every extraction seam calls the ``extraction`` purpose, so that is the one
    purpose a refusal can name today.
    """
    sel = get_config().llm.for_purpose("extraction")
    return {"extraction": f"{sel.provider}:{sel.model}"}


async def held_snapshots(
    session: AsyncSession, *, snapshot_ids: Sequence[str] | None = None
) -> list[HeldSnapshot]:
    """The ``PENDING`` snapshots a current refusal holds, narrowed to ``snapshot_ids``."""
    return await list_held_snapshots(
        session,
        current_refusal_pairings(),
        get_config().extraction.refusal_retry_days,
        snapshot_ids=snapshot_ids,
    )


async def held_snapshot_ids(
    session: AsyncSession, *, snapshot_ids: Sequence[str] | None = None
) -> frozenset[str]:
    """The ids of :func:`held_snapshots`."""
    return frozenset(
        h.snapshot_id for h in await held_snapshots(session, snapshot_ids=snapshot_ids)
    )

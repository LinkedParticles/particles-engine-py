# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The guards on an automatic retirement by replacement (§3c).

A claim the operator restored ruling that both claims hold is never retired
by replacement again (:func:`restored_to_hold`). And one corpus entry wins
at most ``reconciliation.update_supersession.max_retirements_per_entry``
update retirements in any rolling 24 hours, counted over every route that
retires by update (the ladder, its extras, the paraphrase pass, the restatement lineage
and the sweep). The count is read from the ``UPDATE_SUPERSESSION`` events,
whose payload names the winner's entry (``winner_entry_id``), so re-running or
re-depositing the entry does not reset it. A pair past the bound continues to
rung 3 (on the sweep and the paraphrase pass, it is left for review), and the
run's record counts it as capped.

The entry is the unit an attacker controls: one hostile session or file. A
per-store bound would let it spend the budget the owner's real updates need.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.store.event_store import (
    EventRefKind,
    OperatorEventRefRow,
    OperatorEventRow,
    OperatorEventType,
)

log = logging.getLogger(__name__)

#: The rolling window the bound counts over.
WINDOW = timedelta(hours=24)

#: The ``UPDATE_SUPERSESSION`` payload key naming the winning claim's entry.
WINNER_ENTRY_KEY = "winner_entry_id"

#: The restore rulings: the replacement was wrong, or both hold.
WRONG = "wrong"
HOLDS = "holds"


@dataclass
class CappedCount:
    """Pairs the bound sent to rung 3 within one run (:func:`count_capped`)."""

    pairs: set[tuple[str, str]] = field(default_factory=set)

    @property
    def n(self) -> int:
        """Distinct pairs refused; a pair checked at two stages counts once."""
        return len(self.pairs)


_CAPPED: ContextVar[CappedCount | None] = ContextVar("particles_capped_updates", default=None)


@contextmanager
def count_capped() -> Iterator[CappedCount]:
    """Count, for the block, every pair the bound sends to rung 3.

    The extract run and the consolidation cycle open one, so their run records
    carry the count whichever route capped the pair.
    """
    count = CappedCount()
    token = _CAPPED.set(count)
    try:
        yield count
    finally:
        _CAPPED.reset(token)


async def entry_retirements(
    session: AsyncSession, entry_id: str, *, now: datetime | None = None
) -> int:
    """Update retirements ``entry_id`` won in the 24 hours before ``now``."""
    since = (now or datetime.now(UTC)) - WINDOW
    payloads = await session.scalars(
        select(OperatorEventRow.payload).where(
            OperatorEventRow.event_type == OperatorEventType.UPDATE_SUPERSESSION.value,
            OperatorEventRow.occurred_at >= since,
        )
    )
    return sum(1 for p in payloads if p and p.get(WINNER_ENTRY_KEY) == entry_id)


async def restored_to_hold(session: AsyncSession, particle_id: str) -> bool:
    """Whether ``particle_id`` was restored under a ``holds`` ruling.

    Read from the ``RETIREMENT_RESTORED`` events, as the hold set is
    read from retraction events: no suppression table, no particle field. Such
    a claim is never retired again by replacement; a later pair against it is
    held for review.
    """
    rows = await session.scalars(
        select(OperatorEventRow.payload)
        .join(OperatorEventRefRow, OperatorEventRefRow.event_id == OperatorEventRow.event_id)
        .where(
            OperatorEventRow.event_type == OperatorEventType.RETIREMENT_RESTORED.value,
            OperatorEventRefRow.ref_kind == EventRefKind.PARTICLE.value,
            OperatorEventRefRow.ref_id == particle_id,
        )
    )
    return any(
        p and p.get("successor_id") == particle_id and p.get("ruling") == HOLDS for p in rows
    )


async def admits_retirement(
    session: AsyncSession,
    winner_entry_id: str | None,
    *,
    pair: tuple[str, str],
    retired_id: str | None = None,
    pending: Counter[str] | None = None,
) -> bool:
    """Whether one more update retirement may happen (§3c).

    No, when the claim it would retire (``retired_id``; ``None`` for a
    candidate not stored yet) was restored under a ``holds`` ruling
    (:func:`restored_to_hold`), and no when ``winner_entry_id`` has already won
    its bound of update retirements in the last 24 hours.

    ``pending`` adds retirements a pass decided but has not recorded (a dry
    run writes no event). A refusal of ``pair`` (the two claims, as the caller
    names them) is counted once in the enclosing :func:`count_capped` block. A
    winner with no source entry (an assertion) is not an entry, and is never
    capped.
    """
    if retired_id is not None and await restored_to_hold(session, retired_id):
        log.info(
            "Restored claim %s is held by the operator's ruling; not retired by replacement",
            retired_id[:8],
        )
        return False
    if winner_entry_id is None:
        return True
    limit = get_config().reconciliation.update_supersession.max_retirements_per_entry
    won = await entry_retirements(session, winner_entry_id)
    if pending is not None:
        won += pending[winner_entry_id]
    if won < limit:
        return True
    tally = _CAPPED.get()
    if tally is not None:
        tally.pairs.add(pair)
    log.info(
        "Update cap: entry %s already won %d retirements in 24 h; held for review",
        winner_entry_id[:8],
        won,
    )
    return False

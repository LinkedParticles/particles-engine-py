# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""An agent key's deposits, counted in the extraction queue (§2).

A deposit makes no model call of its own: the extraction it queues runs later,
in consolidation pass 1, where no request bucket exists. So the engine marks
each entry a metered agent key deposits (``corpus_entries.metered_principal``)
and bounds two things from that mark: how many unclaimed snapshots the
principal may hold (here, at deposit time) and how many of them pass 1 reads
per run (``operations/consolidation.py``).

The engine's auth gates bind the metered principal for the whole request
(:func:`bind_metered_principal`), so no route or operation that reaches the
deposit seam can leave a metered key's deposit uncounted. Every deposit path
ends in ``deposit.write_entry_and_snapshot``, which calls :func:`meter_deposit`
once it knows the deposit stores a new snapshot, inside the write unit that
holds the store's writer lock, so two deposits arriving one below the cap
serialise and the second is refused.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar

from sqlalchemy import ColumnElement, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import ExtractionStatus
from particles.corpus.store import CorpusEntryRow, SnapshotRow

#: The principal whose deposits in the current scope are metered, or ``None``.
_metered_principal: ContextVar[str | None] = ContextVar("particles_metered_principal", default=None)


class PendingExtractionsFull(Exception):
    """A metered principal already holds its cap of unclaimed snapshots."""

    def __init__(self, principal: str, cap: int, held: int, stuck_entry_id: str | None) -> None:
        self.principal = principal
        self.cap = cap
        self.held = held
        self.stuck_entry_id = stuck_entry_id
        super().__init__(self.describe())

    def describe(self) -> str:
        """The refusal's text: the setting, the count, and the entry holding the most."""
        stuck = (
            f" Entry {self.stuck_entry_id} holds the most of them; an operator can read it "
            f"now with `particles extract {self.stuck_entry_id}` or retract it."
            if self.stuck_entry_id is not None
            else ""
        )
        return (
            f"This key's pending extraction is full: principal {self.principal} holds "
            f"{self.held} snapshot(s) waiting to be read, and "
            f"api.max_pending_extractions_per_principal is {self.cap}. Nothing was stored. "
            "Consolidation reads consolidation.max_pending_per_metered_principal of them a "
            f"run; deposit again after it has.{stuck}"
        )


def bind_metered_principal(principal: str | None) -> None:
    """Meter every deposit for the rest of this context against ``principal``.

    The engine's auth gates call it for every request, with the request key's
    principal when that key is metered (an agent key that is not a
    harvester's) and ``None`` otherwise, so the binding covers every route and
    every operation the request reaches, never only the routes that remember
    to wrap their deposit. A gate runs in the request's own
    task, so the binding ends with the request.
    """
    _metered_principal.set(principal)


@contextlib.contextmanager
def metered_deposits_for(principal: str) -> Iterator[None]:
    """Meter every deposit in this scope against ``principal``.

    For a caller outside an engine request (a test, a local tool); the engine
    binds the principal at its auth gate instead (:func:`bind_metered_principal`).
    A deposit outside either is never marked, counted or refused.
    """
    token = _metered_principal.set(principal)
    try:
        yield
    finally:
        _metered_principal.reset(token)


def _unclaimed(principal: str, also_entry: str | None) -> list[ColumnElement[bool]]:
    """The filter for a principal's unclaimed snapshots: PENDING and never claimed."""
    owner = CorpusEntryRow.metered_principal == principal
    return [
        SnapshotRow.extraction_status == ExtractionStatus.PENDING.value,
        or_(SnapshotRow.extraction_attempts == 0, SnapshotRow.extraction_attempts.is_(None)),
        or_(owner, CorpusEntryRow.entry_id == also_entry) if also_entry is not None else owner,
    ]


async def unclaimed_snapshot_count(
    session: AsyncSession, principal: str, *, also_entry: str | None = None
) -> tuple[int, str | None]:
    """The principal's unclaimed snapshots, and the entry holding the most of them.

    A snapshot counts while it is ``PENDING`` and pass 1 has never claimed it
    (``extraction_attempts`` 0): one claimed and handed back stops counting.
    ``also_entry`` counts an unmarked entry of the principal's
    own that this deposit is about to mark, so its pending snapshots count from
    the moment it is marked.
    """
    rows = (
        await session.execute(
            select(SnapshotRow.entry_id, func.count())
            .join(CorpusEntryRow, CorpusEntryRow.entry_id == SnapshotRow.entry_id)
            .where(*_unclaimed(principal, also_entry))
            .group_by(SnapshotRow.entry_id)
        )
    ).all()
    if not rows:
        return 0, None
    total = sum(int(n) for _, n in rows)
    stuck = max(rows, key=lambda row: (int(row[1]), str(row[0])))
    return total, str(stuck[0])


async def meter_deposit(session: AsyncSession, existing_entry_id: str | None) -> str | None:
    """Check a metered deposit against its cap and mark the entry it lands on.

    Called once a deposit is known to store a new snapshot (a deduplicated
    deposit never reaches here). Returns the principal a new entry is marked
    with, or ``None`` outside :func:`metered_deposits_for`. Raises
    :class:`PendingExtractionsFull`, before anything is written, when the
    principal already holds ``api.max_pending_extractions_per_principal``
    unclaimed snapshots; ``0`` turns the check off. An unmarked entry of the
    principal's own (one deposited before the mark existed, or one an operator
    deposited under its name) is marked here and never cleared.
    """
    principal = _metered_principal.get()
    if principal is None:
        return None
    row = (
        await session.get(CorpusEntryRow, existing_entry_id)
        if existing_entry_id is not None
        else None
    )
    unmarked = row is not None and row.metered_principal is None
    cap = get_config().api.max_pending_extractions_per_principal
    if cap > 0:
        held, stuck = await unclaimed_snapshot_count(
            session, principal, also_entry=existing_entry_id if unmarked else None
        )
        if held >= cap:
            raise PendingExtractionsFull(principal, cap, held, stuck)
    if row is not None and unmarked:
        row.metered_principal = principal
    return principal

# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Whether an agent principal's route wrote a corpus entry.

Extracting an entry an agent deposited runs its subject resolutions as the
depositing agent's, so a Subject the extraction mints for a name the store does
not know is provisional to that agent. The decision is made at deposit, not at
extraction: the agent routes write their deposits inside :func:`agent_deposit`,
and :func:`stamp_entry` records ``deposit_role = "agent"`` on every entry such a
deposit creates or writes a snapshot to. A key revoked or renamed later changes
nothing, and no Engine code reads the Surface's key store.

A harvester's deposit is never stamped: the owner's harvests run outside the
scope.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar

from sqlalchemy.ext.asyncio import AsyncSession

from particles.corpus.store import CorpusEntryRow
from particles.http import withhold_service_credentials

#: The stamp an agent route's entry carries.
AGENT_ROLE = "agent"

_agent_scope: ContextVar[bool] = ContextVar("particles_agent_deposit", default=False)


@contextlib.contextmanager
def agent_deposit() -> Iterator[None]:
    """Stamp every entry a deposit in this scope writes as an agent's.

    The agent chose what the deposit fetches, so the scope also withholds the
    operator's service credentials (security review 2026-11-17-1, RA1).
    """
    token = _agent_scope.set(True)
    try:
        with withhold_service_credentials("an agent principal's deposit"):
            yield
    finally:
        _agent_scope.reset(token)


def in_agent_deposit() -> bool:
    """Whether the current scope is an agent route's deposit."""
    return _agent_scope.get()


def stamp_entry(row: CorpusEntryRow) -> None:
    """Record the agent stamp on ``row`` when the scope is an agent's; never clears it."""
    if _agent_scope.get():
        row.deposit_role = AGENT_ROLE


async def stamp_entry_id(session: AsyncSession, entry_id: str) -> None:
    """:func:`stamp_entry` for an existing entry a deposit in this scope wrote to."""
    if not _agent_scope.get():
        return
    row = await session.get(CorpusEntryRow, entry_id)
    if row is not None and row.deposit_role != AGENT_ROLE:
        row.deposit_role = AGENT_ROLE
        await session.flush()


async def agent_depositor(session: AsyncSession, entry_id: str) -> str | None:
    """The depositing agent principal of a stamped entry, else ``None``."""
    row = await session.get(CorpusEntryRow, entry_id)
    if row is None or row.deposit_role != AGENT_ROLE:
        return None
    return row.deposited_by

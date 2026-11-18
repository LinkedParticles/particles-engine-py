# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""What the digest discloses of automatic retirements.

An automatic retirement by replacement used to be written to the event log
and shown to nobody. The session-start digest now names every claim a rung
retired as replaced since the observer's previous digest: one line per source
the replacing claims came from, with the command that restores them
(``particles particle restore``).

The scope is the demotions that name a replacing claim
(``REPLACEMENT_DEMOTIONS``), each read from the event that records it: an
update, a document supersession, a re-anchor restatement. A claim restored
since is no longer retired and drops out. A project observer sees only the
retirements whose retired claim is in view for it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import CorpusEntry, Particle
from particles.corpus.harvest_marks import harness_session_source, is_owner_harvest
from particles.corpus.store import get_entry
from particles.db import session_scope
from particles.ingest.conflict_plan import source_entry_id
from particles.ingest.retirement_cap import WINNER_ENTRY_KEY
from particles.ingest.second_reading import source_name
from particles.operations.query.observer_scope import filter_visible
from particles.render.markdown import RetirementSource, render_retirement_notice
from particles.store.event_store import OperatorEventRow, OperatorEventType
from particles.store.particle_store import (
    REPLACEMENT_DEMOTIONS,
    get_particles_by_ids,
    get_superseding_particle,
)

#: The command the section points at for the sources it does not list.
MORE_COMMAND = "particles events list --type UPDATE_SUPERSESSION"

_EVENTS = (
    OperatorEventType.UPDATE_SUPERSESSION,
    OperatorEventType.DOCUMENT_SUPERSESSION,
    OperatorEventType.DEPENDENT_CLAIM_EXAMINED,
)

#: The re-anchor outcomes that retire the original claim.
_REANCHOR_RETIRES = frozenset({"restated", "matched_existing"})


@dataclass(frozen=True)
class _Retirement:
    retired_id: str
    winner_id: str | None
    winner_entry_id: str | None
    at: datetime


def _as_aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _retirement(row: OperatorEventRow) -> _Retirement | None:
    payload: dict[str, Any] = row.payload or {}
    if row.event_type == OperatorEventType.DEPENDENT_CLAIM_EXAMINED.value:
        if payload.get("outcome") not in _REANCHOR_RETIRES:
            return None
        retired = payload.get("particle_id")
        winner = None
    else:
        retired, winner = payload.get("retired_id"), payload.get("winner_id")
    if not isinstance(retired, str) or not retired:
        return None
    entry = payload.get(WINNER_ENTRY_KEY)
    return _Retirement(
        retired_id=retired,
        winner_id=winner if isinstance(winner, str) and winner else None,
        winner_entry_id=entry if isinstance(entry, str) and entry else None,
        at=_as_aware(row.occurred_at),
    )


def _label(entry: CorpusEntry | None, entry_id: str | None) -> str:
    if entry is None:
        return f"source {entry_id[:8]}…" if entry_id else "an unrecorded source"
    uri = entry.uri_r or ""
    if harness_session_source(uri) is not None:
        return f"session {uri.rsplit('/', 1)[-1][:8]}…"
    if is_owner_harvest(entry):
        return f"memory file {source_name(uri)}"
    return f"{entry.source_type.lower().replace('_', ' ')} {source_name(uri)}"


def window_start(since: datetime | None, *, now: datetime | None = None) -> datetime:
    """Where the section's window opens: ``since``, else the configured look-back."""
    if since is not None:
        return _as_aware(since)
    days = get_config().mcp.recall.digest_retirement_window_days
    return (now or datetime.now(UTC)) - timedelta(days=days)


def window_phrase(since: datetime | None) -> str:
    """How the header names the window."""
    if since is not None:
        return "since your last session"
    return f"in the last {get_config().mcp.recall.digest_retirement_window_days} days"


async def gather_retirement_sources(
    session: AsyncSession, observer_project: str | None, since: datetime | None
) -> list[RetirementSource]:
    """The automatic retirements since ``since``, one :class:`RetirementSource` per source.

    Newest source first. A retirement whose claim is ACTIVE again, or no
    longer carries a replacement demotion (restored), is left out.
    """
    start = window_start(since)
    rows = (
        await session.execute(
            select(OperatorEventRow)
            .where(
                OperatorEventRow.event_type.in_([t.value for t in _EVENTS]),
                OperatorEventRow.occurred_at >= start,
            )
            .order_by(OperatorEventRow.occurred_at)
        )
    ).scalars()
    latest: dict[str, _Retirement] = {}
    for row in rows:
        found = _retirement(row)
        if found is not None:
            latest[found.retired_id] = found
    if not latest:
        return []
    particles = await get_particles_by_ids(
        session,
        [*latest, *(r.winner_id for r in latest.values() if r.winner_id is not None)],
    )
    retired: list[Particle] = [
        p
        for pid in latest
        if (p := particles.get(pid)) is not None and p.status_reason in REPLACEMENT_DEMOTIONS
    ]
    # The digest's own reading: standing memory only.
    scope = await filter_visible(session, retired, observer_project, standing_only=True)
    retired = [p for p in retired if p.id in scope.visible_ids]

    groups: dict[str | None, list[tuple[_Retirement, Particle, Particle | None]]] = defaultdict(
        list
    )
    for claim in retired:
        record = latest[claim.id]
        winner = particles.get(record.winner_id) if record.winner_id else None
        if winner is None:
            winner = await get_superseding_particle(session, claim.id)
        entry_id = record.winner_entry_id or (
            source_entry_id(winner) if winner is not None else None
        )
        groups[entry_id].append((record, claim, winner))

    sources: list[tuple[datetime, RetirementSource]] = []
    for entry_id, members in groups.items():
        members.sort(key=lambda m: m[0].at, reverse=True)
        record, claim, winner = members[0]
        entry = await get_entry(session, entry_id) if entry_id is not None else None
        if len(members) == 1 or entry_id is None:
            restore = f"particles particle restore {claim.id[:8]} --replacement wrong|holds"
        else:
            restore = (
                f"particles particle restore --from-entry {entry_id[:8]} --replacement wrong|holds"
            )
        sources.append(
            (
                record.at,
                RetirementSource(
                    label=_label(entry, entry_id),
                    date=record.at.date().isoformat(),
                    count=len(members),
                    example_retired=claim.content,
                    example_replacement=winner.content if winner is not None else "(not found)",
                    restore=restore,
                ),
            )
        )
    sources.sort(key=lambda s: s[0], reverse=True)
    return [s for _, s in sources]


async def retirement_lines(
    session: AsyncSession, observer_project: str | None, since: datetime | None
) -> list[str]:
    """The rendered section, or no lines when nothing was retired in the window."""
    sources = await gather_retirement_sources(session, observer_project, since)
    return render_retirement_notice(
        sources,
        window=window_phrase(since),
        max_sources=get_config().mcp.recall.digest_max_retirements,
        more_command=MORE_COMMAND,
    )


async def build_retirement_notice(
    store: str, observer_project: str | None = None, since: datetime | None = None
) -> str:
    """The section on its own, for a session start that injects no digest.

    The SessionStart hook skips the digest when the projected ``MEMORY.md``
    region is current; the disclosure still reaches the session.
    """
    async with session_scope(store) as session:
        lines = await retirement_lines(session, observer_project, since)
    return "\n".join(lines) + "\n" if lines else ""

# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Restore a claim an automatic rung retired as replaced.

``particles particle restore`` is the road back the techspec said a wrongly
superseded claim lacked (§6.4 step 1.7). It covers the demotions that name a
replacing claim (``REPLACEMENT_DEMOTIONS``: an update, a document
supersession, a re-anchor restatement) and returns the claim by **minting a
successor**, never by reviving the row: the recovery a quarantined loser
already gets. The retirement stays on record, and the restore is a
new judgment with its own event, ``RETIREMENT_RESTORED``.

The operator says what the replacement was, with no default:

* ``wrong``: the replacement is retracted under operator authority,
  which puts it in the hold set, so the same text from
  the same source is held for review instead of retiring the restored claim
  again.
* ``holds``: both claims are current. The successor is exempt from every route
  that retires by replacement (:func:`~particles.ingest.retirement_cap.restored_to_hold`);
  a later pair against it is held for review.

Each ruling is also kept as a labelled pair for the memory-rot benchmark
(``coexist`` for ``holds``, ``rejected`` for ``wrong``), as a curation ruling
is.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import Particle
from particles.core.status import Status, StatusReason, validate_transition
from particles.ingest.conflict_plan import source_entry_id
from particles.ingest.retirement_cap import HOLDS, WINNER_ENTRY_KEY, WRONG
from particles.operations.agent_write import retract_belief
from particles.operations.curation.rulings import Ruling, append_ruling, build_pair_ruling
from particles.store.event_store import (
    EventRefKind,
    OperatorEventRow,
    OperatorEventType,
    list_events,
    record_event,
)
from particles.store.particle_store import (
    REPLACEMENT_DEMOTIONS,
    copy_modality_stamp,
    copy_particle_embedding,
    get_particle,
    get_particles_by_ids,
    get_superseding_particle,
    insert_particle,
    update_particle_status,
)

log = logging.getLogger(__name__)

#: The rulings, as the CLI spells them.
RULINGS = (WRONG, HOLDS)

#: The label each ruling is kept under in the benchmark fixture file.
_FIXTURE_LABEL: dict[str, Ruling] = {WRONG: "rejected", HOLDS: "coexist"}

#: The events that name the claim a replacement demotion was retired by.
_RETIREMENT_EVENTS = (
    OperatorEventType.UPDATE_SUPERSESSION,
    OperatorEventType.DOCUMENT_SUPERSESSION,
)


class RestoreRefused(ValueError):
    """The claim cannot be restored; the message says why."""


@dataclass
class RestoreOutcome:
    """What one restore did."""

    retired: Particle
    successor: Particle
    replacement: Particle | None
    ruling: str
    #: Whether the replacement was retracted (``wrong`` on an ACTIVE replacement).
    retracted: bool = False
    #: Claims the re-anchor pass restated because they relied on the retired
    #: state; each is a judgment about a different claim, so the
    #: restore lists them and does not reach them.
    reanchored: list[str] = field(default_factory=list)
    #: The fixture file's disclosure when the label could not be written.
    fixture_note: str | None = None


async def replacement_of(session: AsyncSession, retired: Particle) -> Particle | None:
    """The claim that retired ``retired`` as its replacement, when one is on record.

    The retirement's own event first (an update or document supersession names
    its winner), else the claim whose ``supersedes`` names it (a re-anchor
    restatement, an update written before its event existed).
    """
    for event_type in _RETIREMENT_EVENTS:
        for event in await list_events(
            session,
            ref_kind=EventRefKind.PARTICLE,
            ref_id=retired.id,
            event_type=event_type,
            limit=20,
        ):
            payload = event.payload or {}
            winner = payload.get("winner_id")
            if payload.get("retired_id") == retired.id and isinstance(winner, str) and winner:
                return await get_particle(session, winner)
    return await get_superseding_particle(session, retired.id)


async def _already_restored(session: AsyncSession, retired_id: str) -> str | None:
    for event in await list_events(
        session,
        ref_kind=EventRefKind.PARTICLE,
        ref_id=retired_id,
        event_type=OperatorEventType.RETIREMENT_RESTORED,
        limit=20,
    ):
        payload = event.payload or {}
        if payload.get("retired_id") == retired_id:
            return str(payload.get("successor_id") or "")
    return None


async def reanchored_by(session: AsyncSession, retired_id: str) -> list[str]:
    """Claims the re-anchor pass restated as dated history of ``retired_id``."""
    out: list[str] = []
    for event in await list_events(
        session,
        ref_kind=EventRefKind.PARTICLE,
        ref_id=retired_id,
        event_type=OperatorEventType.DEPENDENT_CLAIM_EXAMINED,
        limit=200,
    ):
        payload = event.payload or {}
        original = payload.get("particle_id")
        if payload.get("retired_id") != retired_id or not isinstance(original, str):
            continue
        found = await get_particle(session, original)
        if (
            found is not None
            and found.status_reason is StatusReason.SUPERSEDED_BY_REANCHOR
            and original not in out
        ):
            out.append(original)
    return out


async def check_restorable(session: AsyncSession, retired: Particle) -> None:
    """Raise :class:`RestoreRefused` unless an automatic rung retired ``retired`` as replaced."""
    successor = await _already_restored(session, retired.id)
    if successor is not None:
        raise RestoreRefused(f"{retired.id[:8]} was already restored as {successor[:8]}")
    if retired.status is Status.ACTIVE or retired.status_reason not in REPLACEMENT_DEMOTIONS:
        reason = retired.status_reason.value if retired.status_reason else "none"
        raise RestoreRefused(
            f"{retired.id[:8]} is {retired.status.value} ({reason}); only a claim an "
            "automatic rung retired as replaced can be restored"
        )


async def restore_retirement(
    session: AsyncSession,
    retired_id: str,
    ruling: str,
    *,
    actor: str,
    store: str,
) -> RestoreOutcome:
    """Restore one retired claim under ``ruling``; the caller holds the writer lock and commits.

    Raises:
        RestoreRefused: The claim is not a replacement demotion, or was restored already.
        ValueError: ``ruling`` is not ``wrong`` or ``holds``.
    """
    if ruling not in RULINGS:
        raise ValueError(f"--replacement must be one of {', '.join(RULINGS)}, not {ruling!r}")
    retired = await get_particle(session, retired_id)
    if retired is None:
        raise RestoreRefused(f"{retired_id[:8]} not found")
    await check_restorable(session, retired)
    replacement = await replacement_of(session, retired)

    successor = retired.model_copy(
        update={
            "id": str(uuid.uuid4()),
            "status": Status.ACTIVE,
            "status_reason": None,
            "supersedes": retired.id,
            "asserted_at": datetime.now(UTC),
        }
    )
    validate_transition(None, Status.ACTIVE)
    await insert_particle(session, successor)
    # The vector and the modality stamp travel with the claim, as in a promotion.
    await copy_particle_embedding(session, retired.id, successor.id)
    await copy_modality_stamp(session, retired.id, successor.id)
    if retired.status is Status.PROVENANCE_STALE:
        await update_particle_status(
            session, retired.id, Status.SUPERSEDED, StatusReason.CONFLICT_RESOLVED
        )

    retracted = False
    if ruling == WRONG and replacement is not None and replacement.status is Status.ACTIVE:
        await retract_belief(
            session,
            store=store,
            particle_id=replacement.id,
            reason=f"particle restore: the replacement of {retired.id[:8]} was ruled wrong",
            operator=True,
            actor=actor,
        )
        retracted = True

    refs = [(EventRefKind.PARTICLE, successor.id), (EventRefKind.PARTICLE, retired.id)]
    if replacement is not None:
        refs.append((EventRefKind.PARTICLE, replacement.id))
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.RETIREMENT_RESTORED,
        reason=f"Restored {retired.id[:8]} as {successor.id[:8]}; replacement {ruling}",
        refs=refs,
        payload={
            "retired_id": retired.id,
            "replacement_id": replacement.id if replacement is not None else None,
            "successor_id": successor.id,
            "ruling": ruling,
        },
    )
    outcome = RestoreOutcome(
        retired=retired,
        successor=successor,
        replacement=replacement,
        ruling=ruling,
        retracted=retracted,
        reanchored=await reanchored_by(session, retired.id),
    )
    if replacement is not None:
        record = await build_pair_ruling(
            session,
            retired,
            replacement,
            _FIXTURE_LABEL[ruling],
            actor=actor,
            store=store,
            card_key=f"restore:{retired.id}",
        )
        outcome.fixture_note = append_ruling(record)
    log.info(
        "Restored %s as %s (replacement %s: %s)",
        retired.id[:8],
        successor.id[:8],
        replacement.id[:8] if replacement is not None else "none",
        ruling,
    )
    return outcome


async def retired_by_entry(
    session: AsyncSession, entry_id: str, *, now: datetime | None = None
) -> list[Particle]:
    """Claims still retired whose replacement came from ``entry_id`` within the digest window.

    Read from the retirement events: an update names its winner's entry, and a
    document supersession's winner is looked up by its first source ref. The
    window is ``mcp.recall.digest_retirement_window_days``, so the command the
    digest prints restores what the digest listed.
    """
    days = get_config().mcp.recall.digest_retirement_window_days
    since = (now or datetime.now(UTC)) - timedelta(days=days)
    rows = (
        await session.execute(
            select(OperatorEventRow.payload).where(
                OperatorEventRow.event_type.in_([t.value for t in _RETIREMENT_EVENTS]),
                OperatorEventRow.occurred_at >= since,
            )
        )
    ).scalars()
    retired_ids: list[str] = []
    unkeyed: dict[str, str] = {}
    for payload in rows:
        retired_id, winner_id = (payload or {}).get("retired_id"), (payload or {}).get("winner_id")
        if not isinstance(retired_id, str) or not isinstance(winner_id, str):
            continue
        keyed = (payload or {}).get(WINNER_ENTRY_KEY)
        if keyed == entry_id:
            retired_ids.append(retired_id)
        elif keyed is None:
            unkeyed[retired_id] = winner_id
    winners = await get_particles_by_ids(session, list(set(unkeyed.values())))
    for retired_id, winner_id in unkeyed.items():
        winner = winners.get(winner_id)
        if winner is not None and source_entry_id(winner) == entry_id:
            retired_ids.append(retired_id)
    found = await get_particles_by_ids(session, list(dict.fromkeys(retired_ids)))
    out: list[Particle] = []
    for pid in dict.fromkeys(retired_ids):
        particle = found.get(pid)
        if particle is None:
            continue
        try:
            await check_restorable(session, particle)
        except RestoreRefused:
            continue
        out.append(particle)
    return out

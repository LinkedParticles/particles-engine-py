# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The operator's side of provisional Subjects (§5, §7).

A Subject an agent principal minted for a name the store did not know answers
only agent principals until it is confirmed. This module holds the operator
acts that confirm one, the queue that lists them, and the opt-in backfill that
marks Subjects agents minted before the mark existed:

- :func:`accept_subject` is ``subjects accept`` and ``POST /subjects/{id}/accept``;
- :func:`operator_touched` confirms a provisional Subject an operator's act
  addressed by id (an alias, a class change, a merge into it), and folds into
  a confirmed bare-local Subject the provisional ones its new names reach;
- :func:`list_provisional` is ``subjects list --provisional``, each Subject
  with the confirmed Subjects that hold its name (the held pairs);
- :func:`backfill` is ``subjects provisional-backfill``.

An operator write that only *names* a name never confirms: the resolver's
operator resolutions do not see provisional Subjects at all.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.provisional import ProvisionalMark
from particles.core.schema import Subject
from particles.ingest.subject_resolver import (
    PERSONA_FOLD_ACTOR,
    RESOLVER_ALIAS_ACTOR,
    fold_or_hold,
)
from particles.store import subject_cache
from particles.store.event_store import (
    EventRefKind,
    OperatorEventRefRow,
    OperatorEventRow,
    OperatorEventType,
    record_event,
)
from particles.store.subject_provisional import (
    AcceptedVia,
    SubjectView,
    accept,
    list_marks,
    mark_of,
    mark_provisional,
    marks_for,
    with_mark,
)
from particles.store.subject_store import (
    ParticleSubjectRow,
    SubjectNotFoundError,
    SubjectRow,
    find_name_candidates,
    get_subject,
)

#: Actor on the backfill's event.
BACKFILL_ACTOR = "subjects-provisional-backfill"

#: The ``SUBJECT_ALIASED`` actors that are process writes, not an operator's.
_PROCESS_ALIAS_ACTORS = frozenset({RESOLVER_ALIAS_ACTOR, PERSONA_FOLD_ACTOR})

#: The operator acts that give a Subject new names, after which the fold runs:
#: an alias, and a merge, whose sources' names become aliases.
_NAME_GIVING_ACTS = frozenset({"alias", "merge"})

#: How many Subjects of each class a backfill report names.
_SHOWN = 20


async def accept_subject(
    session: AsyncSession, subject_id: str, *, actor: str
) -> tuple[SubjectView, bool]:
    """Accept a provisional Subject by id (``subjects accept``).

    Returns the Subject and whether it was provisional; accepting a confirmed
    Subject changes nothing. The caller commits.

    Raises:
        SubjectNotFoundError: No Subject has that id or id prefix.
        ValueError: The prefix matches more than one Subject.
    """
    subject = await get_subject(session, subject_id)
    if subject is None:
        raise SubjectNotFoundError(f"Subject {subject_id!r} not found.")
    accepted = await accept(
        session, subject.id, via=AcceptedVia.OPERATOR, actor=actor, detail={"act": "accept"}
    )
    view = await with_mark(session, subject)
    assert view is not None
    return view, accepted


async def operator_touched(session: AsyncSession, subject_id: str, *, actor: str, act: str) -> bool:
    """Record what an operator's act addressed to a Subject by id does to its mark.

    A provisional Subject is accepted (``via: OPERATOR``, naming ``act``): the
    operator looked at this node. After an act that gives it new names (an
    alias, a merge into it), the bare-local provisional Subjects those names
    reach fold into it (Fold). Returns whether it was accepted.
    """
    accepted = await accept(
        session, subject_id, via=AcceptedVia.OPERATOR, actor=actor, detail={"act": act}
    )
    if act in _NAME_GIVING_ACTS:
        subject = await get_subject(session, subject_id)
        if subject is not None:
            await fold_or_hold(session, subject)
    return accepted


@dataclass(frozen=True)
class ProvisionalEntry:
    """One provisional Subject in the operator's queue, with its held pairs."""

    subject: Subject
    mark: ProvisionalMark
    held_with: tuple[Subject, ...] = ()


async def list_provisional(session: AsyncSession) -> list[ProvisionalEntry]:
    """Every provisional Subject, oldest first, with the confirmed Subjects holding its name."""
    out: list[ProvisionalEntry] = []
    for subject_id, mark in await list_marks(session):
        subject = await get_subject(session, subject_id)
        if subject is None:
            continue
        held: dict[str, Subject] = {}
        for name in [subject.canonical_name, *subject.aliases]:
            for candidate, candidate_mark in await find_name_candidates(session, name):
                if candidate_mark is None and candidate.id != subject.id:
                    held.setdefault(candidate.id, candidate)
        out.append(ProvisionalEntry(subject=subject, mark=mark, held_with=tuple(held.values())))
    return out


@dataclass
class BackfillReport:
    """What ``subjects provisional-backfill`` found or did."""

    marked: list[Subject] = field(default_factory=list)
    already_shared: list[Subject] = field(default_factory=list)
    dry_run: bool = True

    def counts(self) -> dict[str, int]:
        return {"marked": len(self.marked), "already_shared": len(self.already_shared)}


async def _operator_touched_ids(session: AsyncSession, subject_ids: Collection[str]) -> set[str]:
    """The Subjects among ``subject_ids`` some event other than a process alias touched."""
    if not subject_ids:
        return set()
    touched: set[str] = set()
    ids = sorted(subject_ids)
    for start in range(0, len(ids), 5_000):
        rows = await session.execute(
            select(OperatorEventRefRow.ref_id, OperatorEventRow.event_type, OperatorEventRow.actor)
            .join(OperatorEventRow, OperatorEventRow.event_id == OperatorEventRefRow.event_id)
            .where(
                OperatorEventRefRow.ref_kind == EventRefKind.SUBJECT.value,
                OperatorEventRefRow.ref_id.in_(ids[start : start + 5_000]),
            )
        )
        for ref_id, event_type, actor in rows.all():
            if (
                event_type == OperatorEventType.SUBJECT_ALIASED.value
                and actor in _PROCESS_ALIAS_ACTORS
            ):
                continue
            touched.add(ref_id)
    return touched


async def backfill(
    session: AsyncSession,
    *,
    agent_principals: Collection[str],
    dry_run: bool,
) -> BackfillReport:
    """Mark provisional the Subjects agent principals minted before (§7).

    ``agent_principals`` are the labels an agent principal writes under (the
    stdio server's and the facade's identities, every non-operator key's
    principal), reserved labels already removed by the caller, which reads the
    Surface's key store. A Subject is marked, scoped to its ``asserted_by``,
    when that label is one of them, every particle linked to it was asserted
    under the same label, it carries no external reference, and no event
    other than a process alias touched it. Every other agent-minted Subject is
    counted as already shared and left confirmed. A live run writes one
    ``SUBJECT_PROVISIONAL_BACKFILLED`` event; the caller commits.
    """
    from particles.store.particle_store import ParticleRow

    report = BackfillReport(dry_run=dry_run)
    principals = sorted(set(agent_principals))
    if not principals:
        return report
    rows = await session.execute(
        select(SubjectRow)
        .where(SubjectRow.asserted_by.in_(principals))
        .order_by(SubjectRow.created_at, SubjectRow.id)
    )
    candidates = [row.to_model() for row in rows.scalars()]
    marks = await marks_for(session, [s.id for s in candidates])
    candidates = [s for s in candidates if s.id not in marks]
    touched = await _operator_touched_ids(session, [s.id for s in candidates])
    asserters: dict[str, set[str]] = {}
    ids = [s.id for s in candidates]
    for start in range(0, len(ids), 5_000):
        pairs = await session.execute(
            select(ParticleSubjectRow.subject_id, ParticleRow.asserted_by)
            .join(ParticleRow, ParticleRow.id == ParticleSubjectRow.particle_id)
            .where(ParticleSubjectRow.subject_id.in_(ids[start : start + 5_000]))
            .distinct()
        )
        for subject_id, asserted_by in pairs.all():
            asserters.setdefault(subject_id, set()).add(asserted_by)
    for subject in candidates:
        others = asserters.get(subject.id, set()) - {subject.asserted_by}
        if others or subject.external_ids or subject.id in touched:
            report.already_shared.append(subject)
        else:
            report.marked.append(subject)
    if dry_run or not (report.marked or report.already_shared):
        return report
    for subject in report.marked:
        if await mark_of(session, subject.id) is None:
            await mark_provisional(
                session, subject.id, subject.asserted_by, minted_at=subject.created_at
            )
    await record_event(
        session,
        actor=BACKFILL_ACTOR,
        event_type=OperatorEventType.SUBJECT_PROVISIONAL_BACKFILLED,
        refs=[(EventRefKind.SUBJECT, s.id) for s in report.marked],
        payload={
            **report.counts(),
            "principals": principals,
            "already_shared_shown": [s.id for s in report.already_shared[:_SHOWN]],
        },
    )
    subject_cache.clear(keep_negative=True)
    return report

# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Corpus delete operation: the operator hard-delete of one corpus entry.

The destructive sibling of ``corpus retract``. It removes the
entry, its snapshots, and its refs from every particle that cites it, which
is the one exception to append-only provenance (D1):

- a particle whose **only** ``SOURCE`` is the entry is deleted with it;
- a particle that **other entries also support** survives, minus the entry's
  ``SOURCE`` refs and edge row. When one of those refs was ``provenance[0]``,
  the next-earliest ref becomes the decay anchor;
- ``PARTICLE``-type refs (a derived claim's premises) are never stripped. A
  derived claim whose premise is deleted is picked up by the revalidation
  ladder on the next cycle.

Shaped gather / decide / apply (D2): :func:`decide_entry_deletion` is
a pure function over plain values, :func:`plan_entry_deletion` loads its
inputs, and :func:`delete_entry` writes and records one
``CORPUS_ENTRY_DELETED`` operator event. The event carries the entry id and
counts only, never deleted content (no claim text, URI or chunk hash), since
retaining content would defeat the delete's privacy purpose. Caller commits.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pydantic import BaseModel
from sqlalchemy import delete, func, or_, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import ProvenanceRef, ProvenanceRefType
from particles.corpus.store import CorpusEntryRow, SnapshotRow
from particles.store.event_store import EventRefKind, OperatorEventType, record_event
from particles.store.particle_store import (
    ParticleRow,
    ProvenanceEdgeRow,
    get_particles_for_entry,
    strip_entry_provenance_refs,
)
from particles.store.relation_store import ParticleRelationRow
from particles.store.subject_store import ParticleSubjectRow, SubjectRow
from particles.store.synthesis_cache_store import SynthesisCacheRow
from particles.store.taxonomy_store import ParticleTagEdgeRow

# ---------------------------------------------------------------------------
# Decide (pure)
# ---------------------------------------------------------------------------


class ProvenanceStrip(BaseModel):
    """One surviving particle and what the delete removes from it."""

    particle_id: str
    removed_refs: int
    # True when a removed ref was provenance[0], so the decay anchor moves to
    # the next-earliest remaining ref and the claim's age can change.
    anchor_moved: bool


class EntryDeletionDecision(BaseModel):
    """Which of an entry's particles are deleted and which are stripped."""

    to_delete: list[str] = []
    to_strip: list[ProvenanceStrip] = []


def decide_entry_deletion(
    entry_id: str, particles: Iterable[tuple[str, Sequence[ProvenanceRef]]]
) -> EntryDeletionDecision:
    """Split an entry's particles into delete vs strip (D1).

    Pure: no I/O. ``particles`` pairs each particle id with its provenance
    refs, in stored order. A particle is **deleted** when the distinct
    ``corpus_entry_id`` over its ``SOURCE`` refs is exactly ``{entry_id}``,
    and **stripped** when it also has a ``SOURCE`` ref to another entry.
    ``PARTICLE`` and ``AGENT`` refs never count as a source and are never
    stripped. A particle with no ``SOURCE`` ref naming the entry (linked only
    by an edge row) is neither: its refs are left alone and the edge row goes
    with the entry.
    """
    decision = EntryDeletionDecision()
    for particle_id, refs in particles:
        names_entry = [
            ref.type is ProvenanceRefType.SOURCE and ref.corpus_entry_id == entry_id for ref in refs
        ]
        if not any(names_entry):
            continue
        sources = {ref.corpus_entry_id for ref in refs if ref.type is ProvenanceRefType.SOURCE}
        if sources == {entry_id}:
            decision.to_delete.append(particle_id)
        else:
            decision.to_strip.append(
                ProvenanceStrip(
                    particle_id=particle_id,
                    removed_refs=sum(names_entry),
                    anchor_moved=names_entry[0],
                )
            )
    return decision


# ---------------------------------------------------------------------------
# Gather
# ---------------------------------------------------------------------------


class EntryDeletionPlan(BaseModel):
    """What a delete would do. Read-only, no DB writes."""

    entry_id: str
    snapshots: int
    decision: EntryDeletionDecision


class EntryDeletionResult(BaseModel):
    """Outcome of an applied delete."""

    entry_id: str
    particles_deleted: int
    particles_stripped: int
    anchors_moved: int
    snapshots_deleted: int
    subjects_orphaned: int
    synthesis_rows_deleted: int


async def plan_entry_deletion(session: AsyncSession, entry_id: str) -> EntryDeletionPlan:
    """Load the entry's particles and decide the delete-vs-strip split.

    Read-only: backs the CLI preview shown before the confirm prompt.
    """
    particles = await get_particles_for_entry(session, entry_id)
    snapshots = (
        await session.execute(
            select(func.count()).select_from(SnapshotRow).where(SnapshotRow.entry_id == entry_id)
        )
    ).scalar_one()
    decision = decide_entry_deletion(entry_id, ((p.id, p.provenance) for p in particles))
    return EntryDeletionPlan(entry_id=entry_id, snapshots=snapshots, decision=decision)


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


async def delete_entry(
    session: AsyncSession, entry_id: str, *, actor: str = "cli:corpus-delete"
) -> EntryDeletionResult:
    """Hard-delete ``entry_id`` per the D1 rule; record one event.

    Re-plans inside the call, so a particle that gained a second source after
    the preview is stripped rather than deleted. Caller commits.
    """
    plan = await plan_entry_deletion(session, entry_id)
    delete_ids = plan.decision.to_delete

    # Subjects linked to the doomed particles, captured before the join rows
    # go, so each can be re-checked for orphanhood afterwards.
    candidate_subject_ids: set[str] = set()
    if delete_ids:
        candidate_subject_ids = set(
            (
                await session.execute(
                    select(ParticleSubjectRow.subject_id).where(
                        ParticleSubjectRow.particle_id.in_(delete_ids)
                    )
                )
            ).scalars()
        )
        await session.execute(delete(ParticleRow).where(ParticleRow.id.in_(delete_ids)))
        await purge_particle_index_rows(session, delete_ids)

    for strip in plan.decision.to_strip:
        await strip_entry_provenance_refs(session, strip.particle_id, entry_id)

    subjects_orphaned, synth_removed = await purge_orphan_subjects(session, candidate_subject_ids)

    await session.execute(
        delete(ProvenanceEdgeRow).where(ProvenanceEdgeRow.corpus_entry_id == entry_id)
    )
    await session.execute(delete(SnapshotRow).where(SnapshotRow.entry_id == entry_id))
    await session.execute(delete(CorpusEntryRow).where(CorpusEntryRow.entry_id == entry_id))

    result = EntryDeletionResult(
        entry_id=entry_id,
        particles_deleted=len(delete_ids),
        particles_stripped=len(plan.decision.to_strip),
        anchors_moved=sum(1 for s in plan.decision.to_strip if s.anchor_moved),
        snapshots_deleted=plan.snapshots,
        subjects_orphaned=subjects_orphaned,
        synthesis_rows_deleted=synth_removed,
    )
    # Ids and counts only. The entry's URI is not recorded: a path or URL can
    # itself name what the operator asked to erase.
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.CORPUS_ENTRY_DELETED,
        refs=[
            (EventRefKind.CORPUS_ENTRY, entry_id),
            *[(EventRefKind.PARTICLE, pid) for pid in delete_ids],
            *[(EventRefKind.PARTICLE, s.particle_id) for s in plan.decision.to_strip],
        ],
        payload=result.model_dump(exclude={"entry_id"}),
    )
    return result


# ---------------------------------------------------------------------------
# Index sweeps (shared with ``corpus prune-orphans``)
#
# None of the index tables (particle_subjects, particle_tag_edges,
# particle_relations, synthesis_cache) declare a foreign key, and SQLite FK
# enforcement is off, so deleting particles or subjects leaves dangling rows
# unless they are swept explicitly here.
# ---------------------------------------------------------------------------


async def purge_particle_index_rows(session: AsyncSession, particle_ids: list[str]) -> None:
    """Delete every index row keyed on one of ``particle_ids``.

    Sweeps the ``particle_subjects`` join, ``particle_tag_edges``, and
    ``particle_relations`` (either endpoint). Caller deletes the
    ``ParticleRow`` rows themselves.
    """
    if not particle_ids:
        return
    await session.execute(
        delete(ParticleSubjectRow).where(ParticleSubjectRow.particle_id.in_(particle_ids))
    )
    await session.execute(
        delete(ParticleTagEdgeRow).where(ParticleTagEdgeRow.particle_id.in_(particle_ids))
    )
    await session.execute(
        delete(ParticleRelationRow).where(
            or_(
                ParticleRelationRow.particle_a.in_(particle_ids),
                ParticleRelationRow.particle_b.in_(particle_ids),
            )
        )
    )


async def purge_orphan_subjects(
    session: AsyncSession, candidate_ids: set[str] | None
) -> tuple[int, int]:
    """Delete subjects with no remaining ``particle_subjects`` link.

    Also drops any ``synthesis_cache`` rows keyed on the removed subjects.
    Returns ``(subjects_removed, synthesis_rows_removed)``.

    ``candidate_ids`` restricts the orphan check to a known set (the subjects
    linked to a just-deleted entry's particles), so a subject still linked to
    surviving particles is never touched. Pass ``None`` to scan every subject
    (the whole-DB ``prune-orphans`` sweep).
    """
    if candidate_ids is not None and not candidate_ids:
        return (0, 0)

    linked = select(ParticleSubjectRow.subject_id).distinct()
    stmt = select(SubjectRow.id).where(SubjectRow.id.not_in(linked))
    if candidate_ids is not None:
        stmt = stmt.where(SubjectRow.id.in_(candidate_ids))
    orphan_ids = list((await session.execute(stmt)).scalars())
    if not orphan_ids:
        return (0, 0)

    synth_result: CursorResult[None] = await session.execute(  # type: ignore[assignment]
        delete(SynthesisCacheRow).where(SynthesisCacheRow.subject_id.in_(orphan_ids))
    )
    await session.execute(delete(SubjectRow).where(SubjectRow.id.in_(orphan_ids)))
    return (len(orphan_ids), synth_result.rowcount or 0)

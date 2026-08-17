"""SQLAlchemy ORM models and repository for the Subject store.

Subjects are canonical real-world entities — the nodes of the knowledge graph.
Particles are statements (properties or edges) about subjects.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Literal

from sqlalchemy import DateTime, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.core.schema import ContributorRef, ExternalRef, Subject
from particles.db import Base
from particles.sql_safety import LIKE_ESCAPE, escape_like_pattern
from particles.store import subject_cache
from particles.store.event_store import EventRefKind, OperatorEventType, record_event

log = logging.getLogger(__name__)


class SubjectRow(Base):
    __tablename__ = "subjects"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    canonical_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    aliases_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    external_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    asserted_by: Mapped[str] = mapped_column(String, nullable=False)
    # Nomisma ontology class for exporter template selection
    subject_class: Mapped[str | None] = mapped_column(String, nullable=True)
    # Extension D/E contributor attribution. NULL ≡ none recorded.
    contributors_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    def to_model(self) -> Subject:
        ext_ids = [
            ExternalRef(
                namespace=r["namespace"],
                id=r["id"],
                uri=r.get("uri"),
                confidence=float(r.get("confidence", 1.0)),
            )
            for r in json.loads(self.external_ids_json)
        ]
        return Subject(
            id=self.id,
            canonical_name=self.canonical_name,
            description=self.description,
            aliases=json.loads(self.aliases_json),
            external_ids=ext_ids,
            created_at=self.created_at,
            asserted_by=self.asserted_by,
            subject_class=self.subject_class,
            contributors=(
                [ContributorRef(**c) for c in json.loads(self.contributors_json)]
                if self.contributors_json
                else None
            ),
        )

    @classmethod
    def from_model(cls, s: Subject) -> SubjectRow:
        return cls(
            id=s.id,
            canonical_name=s.canonical_name,
            description=s.description,
            aliases_json=json.dumps(s.aliases),
            external_ids_json=json.dumps(
                [
                    {"namespace": r.namespace, "id": r.id, "uri": r.uri, "confidence": r.confidence}
                    for r in s.external_ids
                ]
            ),
            created_at=s.created_at,
            asserted_by=s.asserted_by,
            subject_class=s.subject_class,
            contributors_json=(
                json.dumps([c.model_dump(mode="json") for c in s.contributors])
                if s.contributors is not None
                else None
            ),
        )


class ParticleSubjectRow(Base):
    """Join table: particle ↔ subject (enables O(k) lookup in both directions)."""

    __tablename__ = "particle_subjects"

    particle_id: Mapped[str] = mapped_column(String, primary_key=True)
    subject_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)


# ---------------------------------------------------------------------------
# Repository helpers
# ---------------------------------------------------------------------------


async def insert_subject(session: AsyncSession, subject: Subject) -> None:
    """Persist a new subject. Flushes but does not commit."""
    session.add(SubjectRow.from_model(subject))
    await session.flush()


async def get_subject(session: AsyncSession, subject_id: str) -> Subject | None:
    """Look up a subject by full UUID or unambiguous prefix.

    Args:
        subject_id: Full UUID or unique prefix (e.g. first 8 chars).

    Raises:
        ValueError: If the prefix matches more than one subject.
    """
    row = await session.get(SubjectRow, subject_id)
    if row:
        return row.to_model()
    # Prefix match — mirrors the truncated-ID display in subjects list.
    # Escape user-input LIKE wildcards so '%' / '_' don't broaden the match.
    pattern = f"{escape_like_pattern(subject_id)}%"
    result = await session.execute(
        select(SubjectRow).where(SubjectRow.id.like(pattern, escape=LIKE_ESCAPE))
    )
    rows = result.scalars().all()
    if len(rows) == 1:
        return rows[0].to_model()
    if len(rows) > 1:
        raise ValueError(
            f"Ambiguous subject prefix {subject_id!r} matches {len(rows)} subjects;"
            " use more characters."
        )
    return None


async def find_by_name(session: AsyncSession, name: str) -> Subject | None:
    """Case-insensitive lookup against canonical_name and aliases.

    Used by the subject resolver during extraction to avoid creating duplicate
    subjects for the same real-world entity.
    """
    from sqlalchemy import func

    name_lower = name.lower()

    # Exact canonical_name match first (fast path). This is an *exact* lookup, so
    # we compare with case-insensitive equality (func.lower(col) == name_lower),
    # NOT ilike(name): ilike treats '%' and '_' in the (untrusted, LLM-extracted)
    # name as wildcards, so a poisoned name like 'Acme_Inc' or '%' would match an
    # unrelated pre-existing subject and silently mis-attribute the claim
    # (security finding F4 — subject-graph poisoning). A shared canonical_name is
    # still a *valid* state — distinct real-world entities can collide on a
    # surface name (e.g. "Prometheus" the monitoring software vs the Greek Titan),
    # and the resolver may also leave true duplicates behind. Either way this must
    # never crash extraction, so we pick deterministically (earliest-created)
    # rather than using scalar_one_or_none(), which raises MultipleResultsFound
    # and aborts the whole snapshot. A duplicate is logged so an operator can merge.
    result = await session.execute(
        select(SubjectRow)
        .where(func.lower(SubjectRow.canonical_name) == name_lower)
        .order_by(SubjectRow.created_at, SubjectRow.id)
    )
    rows = result.scalars().all()
    if rows:
        if len(rows) > 1:
            log.warning(
                "Multiple subjects share canonical_name %r: %s; resolving to "
                "earliest (%s). Merge or disambiguate to silence this warning.",
                name,
                [r.id for r in rows],
                rows[0].id,
            )
        return rows[0].to_model()

    # Scan aliases (acceptable at v0.3 scale; add a dedicated alias table if needed)
    result = await session.execute(select(SubjectRow))
    for row in result.scalars():
        aliases: list[str] = json.loads(row.aliases_json)
        if any(a.lower() == name_lower for a in aliases):
            return row.to_model()
    return None


async def find_by_external_ref(
    session: AsyncSession, namespace: str, external_id: str
) -> Subject | None:
    """Look up a subject by external ontology reference."""
    result = await session.execute(select(SubjectRow))
    for row in result.scalars():
        refs: list[dict[str, str]] = json.loads(row.external_ids_json)
        for ref in refs:
            if ref["namespace"] == namespace and ref["id"] == external_id:
                return row.to_model()
    return None


async def get_subjects_for_particle(session: AsyncSession, particle_id: str) -> list[Subject]:
    result = await session.execute(
        select(SubjectRow)
        .join(ParticleSubjectRow, ParticleSubjectRow.subject_id == SubjectRow.id)
        .where(ParticleSubjectRow.particle_id == particle_id)
    )
    return [row.to_model() for row in result.scalars()]


async def get_particles_for_subject(
    session: AsyncSession,
    subject_id: str,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> list[str]:
    """Return particle IDs linked to this subject.

    Args:
        session: Active SQLAlchemy session.
        subject_id: Subject UUID.
        limit: Maximum particle IDs to return. ``None`` returns all (the
            existing exporter/lint callers depend on the full list).
        offset: Number of IDs to skip before returning. Used with ``limit``
            to drive pagination at the MCP surface, where a hot subject's
            full particle-id list can blow the per-tool-result cap.
    """
    stmt = select(ParticleSubjectRow.particle_id).where(ParticleSubjectRow.subject_id == subject_id)
    # Stable order so paginated callers get consistent slices.
    stmt = stmt.order_by(ParticleSubjectRow.particle_id)
    if offset:
        stmt = stmt.offset(offset)
    if limit is not None:
        stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars())


async def count_particles_for_subject(session: AsyncSession, subject_id: str) -> int:
    """Total count of particle IDs linked to this subject (any status).

    Companion to :func:`get_particles_for_subject` for callers that want
    the true total even when they only fetched a page of IDs.
    """
    from sqlalchemy import func

    result = await session.execute(
        select(func.count(ParticleSubjectRow.particle_id)).where(
            ParticleSubjectRow.subject_id == subject_id
        )
    )
    return int(result.scalar_one())


async def list_particle_subject_pairs(
    session: AsyncSession,
) -> list[tuple[str, str]]:
    """Return every ``(particle_id, subject_id)`` link in the join table.

    Both the wiki and Anki exporters need a full dump of the join table
    so they can group particles by subject (or vice versa) in memory.
    Two callers — kept here so a third exporter doesn't reach for a
    third bespoke ``select(ParticleSubjectRow)``. Filtering by particle
    status happens downstream because each caller groups in a different
    direction: wiki builds ``subject_id → [particle_id]`` and intersects
    with its already-loaded ACTIVE particle set; Anki builds the inverse
    ``particle_id → [subject_id]`` and looks names up per particle.
    """
    result = await session.execute(
        select(ParticleSubjectRow.particle_id, ParticleSubjectRow.subject_id)
    )
    return [(pid, sid) for pid, sid in result.all()]


async def link_particle_to_subjects(
    session: AsyncSession, particle_id: str, subject_ids: list[str]
) -> None:
    for sid in dict.fromkeys(subject_ids):  # deduplicate, preserving order
        session.add(ParticleSubjectRow(particle_id=particle_id, subject_id=sid))
    await session.flush()


async def add_aliases(
    session: AsyncSession,
    subject_id: str,
    new_aliases: list[str],
    *,
    actor: str = "subjects-alias",
) -> tuple[Subject, list[str]]:
    """Append aliases to a subject; return (updated subject, actually-added names).

    Idempotent: names already present (case-insensitive) are silently skipped.
    Invalidates the subject resolver cache so subsequent extractions pick up
    the new aliases immediately. Records a ``SUBJECT_ALIASED`` operator event
     when names are actually added.
    """
    row = await session.get(SubjectRow, subject_id)
    if row is None:
        raise ValueError(f"Subject {subject_id} not found")
    existing_lower = {row.canonical_name.lower()} | {
        a.lower() for a in json.loads(row.aliases_json)
    }
    added = [a for a in new_aliases if a.lower() not in existing_lower]
    if added:
        current: list[str] = json.loads(row.aliases_json)
        row.aliases_json = json.dumps(current + added)
        await session.flush()
        await record_event(
            session,
            actor=actor,
            event_type=OperatorEventType.SUBJECT_ALIASED,
            refs=[(EventRefKind.SUBJECT, subject_id)],
            payload={"added": added},
        )
    subject_cache.clear(keep_negative=True)
    return row.to_model(), added


async def add_external_ref(session: AsyncSession, subject_id: str, ref: ExternalRef) -> bool:
    """Attach an external ref to a subject. Idempotent; returns True if actually added."""
    row = await session.get(SubjectRow, subject_id)
    if row is None:
        return False
    refs: list[dict[str, object]] = json.loads(row.external_ids_json)
    if any(r["namespace"] == ref.namespace and r["id"] == ref.id for r in refs):
        return False
    refs.append(
        {
            "namespace": ref.namespace,
            "id": ref.id,
            "uri": ref.uri or "",
            "confidence": ref.confidence,
        }
    )
    row.external_ids_json = json.dumps(refs)
    await session.flush()
    return True


async def set_external_ref_confidence(
    session: AsyncSession,
    subject_id: str,
    namespace: str,
    ref_id: str,
    confidence: float,
    *,
    actor: str = "subjects-confirm",
) -> bool:
    """Set the confidence on an existing external ref. Returns True if found and updated.

    The ``subjects confirm`` verb's only caller; records a
    ``SUBJECT_LINK_CONFIRMED`` operator event on success.
    """
    row = await session.get(SubjectRow, subject_id)
    if row is None:
        return False
    refs: list[dict[str, object]] = json.loads(row.external_ids_json)
    found = False
    for r in refs:
        if r["namespace"] == namespace and r["id"] == ref_id:
            r["confidence"] = confidence
            found = True
            break
    if not found:
        return False
    row.external_ids_json = json.dumps(refs)
    await session.flush()
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.SUBJECT_LINK_CONFIRMED,
        refs=[(EventRefKind.SUBJECT, subject_id)],
        payload={"namespace": namespace, "ref_id": ref_id, "confidence": confidence},
    )
    return True


async def remove_external_ref(
    session: AsyncSession,
    subject_id: str,
    namespace: str,
    ref_id: str,
    *,
    actor: str = "subjects-unlink",
) -> bool:
    """Detach an external ref from a subject. Returns True if found and removed.

    Operator workflow for when the subject resolver bound a particle's
    subject to a wrong Wikidata entity (canonical name happens to match,
    description doesn't): drop the bad link with this function, leaving
    the subject + its particles intact. The next export will re-render
    the note without the misleading `wikidata-<QID>` tag in the
    frontmatter.

    The Subject's ``canonical_name`` is unchanged — operator can follow
    up with ``particles subjects merge`` if they want to move the
    particles to a freshly-extracted subject with the correct entity
    binding.
    """
    row = await session.get(SubjectRow, subject_id)
    if row is None:
        return False
    refs: list[dict[str, object]] = json.loads(row.external_ids_json)
    new_refs = [r for r in refs if not (r["namespace"] == namespace and r["id"] == ref_id)]
    if len(new_refs) == len(refs):
        return False
    row.external_ids_json = json.dumps(new_refs)
    await session.flush()
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.SUBJECT_LINK_REMOVED,
        refs=[(EventRefKind.SUBJECT, subject_id)],
        payload={"namespace": namespace, "ref_id": ref_id},
    )
    # Drop any cached resolution that was based on the removed ref, so
    # the next ``resolve_subject`` call doesn't return the stale binding.
    subject_cache.clear(keep_negative=True)
    return True


async def merge_subjects(
    session: AsyncSession, source_id: str, target_id: str, *, actor: str = "subjects-merge"
) -> tuple[Subject, list[str], int]:
    """Merge source subject into target. Source is deleted; its particles are re-linked.

    The source's canonical_name and aliases are added to the target as aliases.
    All particle_subjects join rows pointing to source are re-pointed to target.
    Irreversible — caller should confirm before committing.

    Returns:
        Tuple of (updated_target_subject, aliases_added, particles_relinked).

    Raises:
        ValueError: If either subject is not found, or source == target.
    """
    source_row = await session.get(SubjectRow, source_id)
    target_row = await session.get(SubjectRow, target_id)
    if source_row is None:
        raise ValueError(f"Source subject {source_id} not found")
    if target_row is None:
        raise ValueError(f"Target subject {target_id} not found")
    if source_id == target_id:
        raise ValueError("Source and target must be different subjects")

    # Collect source names → add as aliases on target
    source_names = [source_row.canonical_name] + json.loads(source_row.aliases_json)
    target_existing_lower = {target_row.canonical_name.lower()} | {
        a.lower() for a in json.loads(target_row.aliases_json)
    }
    added = [n for n in source_names if n.lower() not in target_existing_lower]
    if added:
        current_aliases: list[str] = json.loads(target_row.aliases_json)
        target_row.aliases_json = json.dumps(current_aliases + added)

    # Re-link particle_subjects join rows
    result = await session.execute(
        select(ParticleSubjectRow).where(ParticleSubjectRow.subject_id == source_id)
    )
    ps_rows = result.scalars().all()
    relinked = 0
    relinked_pids: list[str] = []
    for ps_row in ps_rows:
        existing_link = await session.get(ParticleSubjectRow, (ps_row.particle_id, target_id))
        if existing_link is None:
            session.add(ParticleSubjectRow(particle_id=ps_row.particle_id, subject_id=target_id))
        relinked_pids.append(ps_row.particle_id)
        await session.delete(ps_row)
        relinked += 1

    # Update subject_ids_json on particle rows
    from particles.store.particle_store import ParticleRow

    p_result = await session.execute(
        select(ParticleRow).where(ParticleRow.subject_ids_json.contains(source_id))
    )
    for p_row in p_result.scalars():
        ids: list[str] = json.loads(p_row.subject_ids_json)
        if source_id in ids:
            new_ids = [target_id if i == source_id else i for i in ids]
            p_row.subject_ids_json = json.dumps(new_ids)

    # Delete source subject
    await session.delete(source_row)
    await session.flush()

    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.SUBJECTS_MERGED,
        refs=[
            (EventRefKind.SUBJECT, source_id),
            (EventRefKind.SUBJECT, target_id),
            *[(EventRefKind.PARTICLE, pid) for pid in relinked_pids],
        ],
        payload={"aliases_added": added, "relinked": relinked},
    )

    subject_cache.clear(keep_negative=True)
    return target_row.to_model(), added, relinked


async def delete_subject(
    session: AsyncSession, subject_id: str, *, actor: str = "subjects-delete"
) -> tuple[Subject, int]:
    """Delete a subject and detach it from any particles still referencing it.

    Intended for phantom cleanup; the caller is responsible for the
    phantom / ``--force`` guard. Any ``particle_subjects`` join rows pointing at
    the subject are removed and the id is stripped from each particle's
    ``subject_ids_json`` so no particle is left pointing at a deleted subject.
    Irreversible — caller commits.

    Returns:
        Tuple of (deleted_subject, particles_detached).

    Raises:
        ValueError: If the subject is not found.
    """
    row = await session.get(SubjectRow, subject_id)
    if row is None:
        raise ValueError(f"Subject {subject_id} not found")

    # Drop join rows and strip the id from each particle's subject_ids_json.
    result = await session.execute(
        select(ParticleSubjectRow).where(ParticleSubjectRow.subject_id == subject_id)
    )
    detached_pids: list[str] = []
    for ps_row in result.scalars().all():
        detached_pids.append(ps_row.particle_id)
        await session.delete(ps_row)

    from particles.store.particle_store import ParticleRow

    p_result = await session.execute(
        select(ParticleRow).where(ParticleRow.subject_ids_json.contains(subject_id))
    )
    for p_row in p_result.scalars():
        ids: list[str] = json.loads(p_row.subject_ids_json)
        if subject_id in ids:
            p_row.subject_ids_json = json.dumps([i for i in ids if i != subject_id])

    deleted = row.to_model()
    await session.delete(row)
    await session.flush()

    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.SUBJECT_DELETED,
        refs=[
            (EventRefKind.SUBJECT, subject_id),
            *[(EventRefKind.PARTICLE, pid) for pid in detached_pids],
        ],
        payload={
            "canonical_name": deleted.canonical_name,
            "detached": len(detached_pids),
        },
    )

    subject_cache.clear(keep_negative=True)
    return deleted, len(detached_pids)


async def split_subject(
    session: AsyncSession,
    *,
    source_id: str,
    new_subject_id: str,
    particle_ids: list[str],
    actor: str = "subjects-split",
) -> tuple[list[str], list[str]]:
    """Re-link particles from ``source_id`` onto ``new_subject_id``.

    Both Subjects must already exist in the DB. The CLI verb is
    responsible for resolving / inserting the new Subject (via
    :func:`particles.ingest.subject_resolver.resolve_subject` or a
    by-ID metadata fetch) before calling this helper, so identity
    canonicalisation lives in the resolver layer where the Anthropic
    + Wikidata clients are already wired.

    

    - The source Subject is preserved with its remaining particles.
      It is NOT deleted even if every particle was split off — the
      audit trail of what the operator just corrected stays intact.
    - For each ``<pid>``: the ``(particle_id, source_id)`` row is
      removed from ``particle_subjects`` and a ``(particle_id,
      new_subject_id)`` row is inserted. The denormalised
      ``ParticleRow.subject_ids_json`` is updated in lockstep so query
      / export paths read the new binding immediately.
    - Multi-subject particles are partially moved: a particle bound to
      ``[source_id, other_id]`` becomes ``[new_subject_id, other_id]``.
      The other binding is untouched.
    - A particle_id that isn't actually bound to ``source_id`` is a
      no-op for the rebinding but is returned in ``not_bound`` so the
      caller can warn. We don't silently rewrite particles that were
      never bound to the source.

    Caller commits.

    Returns:
        Tuple of (relinked_pids, not_bound_pids).

    Raises:
        ValueError: If either Subject does not exist, or the source
            and new Subjects are the same.
    """
    if source_id == new_subject_id:
        raise ValueError("source_id and new_subject_id must differ")
    source_row = await session.get(SubjectRow, source_id)
    if source_row is None:
        raise ValueError(f"Source subject {source_id} not found")
    new_row = await session.get(SubjectRow, new_subject_id)
    if new_row is None:
        raise ValueError(f"New subject {new_subject_id} not found")

    from particles.store.particle_store import ParticleRow

    relinked: list[str] = []
    not_bound: list[str] = []

    for pid in particle_ids:
        ps_row = await session.get(ParticleSubjectRow, (pid, source_id))
        if ps_row is None:
            # Particle was never bound to the source — skip it. Caller
            # surfaces a warning.
            not_bound.append(pid)
            continue
        # If the particle is *already* bound to the new subject (e.g.
        # via an unrelated extraction earlier), don't re-insert — just
        # drop the source binding. The unique constraint would reject
        # the duplicate anyway.
        existing_new_link = await session.get(ParticleSubjectRow, (pid, new_subject_id))
        await session.delete(ps_row)
        if existing_new_link is None:
            session.add(ParticleSubjectRow(particle_id=pid, subject_id=new_subject_id))

        # Update the denormalised subject_ids_json on the particle row
        # in lockstep so downstream readers see the rebinding without
        # waiting for the next extraction pass.
        p_row = await session.get(ParticleRow, pid)
        if p_row is not None:
            ids: list[str] = json.loads(p_row.subject_ids_json)
            # Replace source_id with new_subject_id, but drop any
            # duplicates (handles the already-bound-to-new case).
            new_ids: list[str] = []
            for i in ids:
                target = new_subject_id if i == source_id else i
                if target not in new_ids:
                    new_ids.append(target)
            p_row.subject_ids_json = json.dumps(new_ids)

        relinked.append(pid)

    await session.flush()
    if relinked:
        await record_event(
            session,
            actor=actor,
            event_type=OperatorEventType.SUBJECTS_SPLIT,
            refs=[
                (EventRefKind.SUBJECT, source_id),
                (EventRefKind.SUBJECT, new_subject_id),
                *[(EventRefKind.PARTICLE, pid) for pid in relinked],
            ],
            payload={
                "source_id": source_id,
                "new_subject_id": new_subject_id,
                "not_bound": not_bound,
            },
        )
    subject_cache.clear(keep_negative=True)
    return relinked, not_bound


async def get_particle_count_for_subject(session: AsyncSession, subject_id: str) -> int:
    """Count ACTIVE CLAIM particles linked to a subject."""
    from sqlalchemy import func

    from particles.store.particle_store import ParticleRow

    result = await session.execute(
        select(func.count(ParticleRow.id))
        .join(ParticleSubjectRow, ParticleSubjectRow.particle_id == ParticleRow.id)
        .where(
            ParticleSubjectRow.subject_id == subject_id,
            ParticleRow.status == "ACTIVE",
            ParticleRow.particle_type == "CLAIM",
        )
    )
    return result.scalar_one() or 0


async def set_subject_class(session: AsyncSession, subject_id: str, subject_class: str) -> None:
    """Set (or update) the subject_class for a subject. Idempotent.

    The extraction pipeline's classification path (no operator event — an
    automated pipeline step fails inclusion criterion). For the
    operator-initiated override verb, use :func:`reclassify_subject`.
    """
    row = await session.get(SubjectRow, subject_id)
    if row is None:
        raise ValueError(f"Subject {subject_id} not found")
    if row.subject_class != subject_class:
        row.subject_class = subject_class
        await session.flush()


async def reclassify_subject(
    session: AsyncSession,
    subject_id: str,
    subject_class: str,
    *,
    actor: str = "subjects-set-class",
) -> tuple[Subject, str | None]:
    """Operator override of a subject's Nomisma class.

    Unlike :func:`set_subject_class` (the automated pipeline path), this records
    a ``SUBJECT_RECLASSIFIED`` operator event when the value actually changes —
    the override meets inclusion criterion (operator-initiated,
    mutates entity classification, no durable history of its own). A no-op
    (new class equals current) records nothing.

    Returns:
        Tuple of (updated_subject, previous_class). ``previous_class`` is the
        value before the call (``None`` if the subject had no class), letting the
        caller report old → new and detect the no-op.

    Raises:
        ValueError: If the subject is not found.
    """
    row = await session.get(SubjectRow, subject_id)
    if row is None:
        raise ValueError(f"Subject {subject_id} not found")
    previous = row.subject_class
    if previous != subject_class:
        row.subject_class = subject_class
        await session.flush()
        await record_event(
            session,
            actor=actor,
            event_type=OperatorEventType.SUBJECT_RECLASSIFIED,
            refs=[(EventRefKind.SUBJECT, subject_id)],
            payload={"previous_class": previous, "new_class": subject_class},
        )
        subject_cache.clear(keep_negative=True)
    return row.to_model(), previous


async def list_all_subjects(
    session: AsyncSession,
    *,
    limit: int | None = None,
    offset: int = 0,
    order: Literal["name", "degree"] = "name",
) -> list[Subject]:
    """List subjects, alphabetical by default.

    ``order="degree"`` sorts by descending count of ACTIVE particles linked
    via ``particle_subjects`` (canonical-name tie-break) — "most-connected
    first", the seed the web UI's Browse route opens on. Degree counts only
    ACTIVE links: a subject whose beliefs are all retired is not a good
    picture of the store's current shape.
    """
    stmt = select(SubjectRow)
    if order == "degree":
        from sqlalchemy import func

        from particles.store.particle_store import ParticleRow

        count_sq = (
            select(
                ParticleSubjectRow.subject_id,
                func.count(ParticleRow.id).label("cnt"),
            )
            .join(ParticleRow, ParticleRow.id == ParticleSubjectRow.particle_id)
            .where(ParticleRow.status == "ACTIVE")
            .group_by(ParticleSubjectRow.subject_id)
            .subquery()
        )
        stmt = stmt.outerjoin(count_sq, count_sq.c.subject_id == SubjectRow.id).order_by(
            func.coalesce(count_sq.c.cnt, 0).desc(), SubjectRow.canonical_name
        )
    else:
        stmt = stmt.order_by(SubjectRow.canonical_name)
    if offset:
        stmt = stmt.offset(offset)
    if limit is not None:
        stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return [row.to_model() for row in result.scalars()]


async def search_subjects(
    session: AsyncSession,
    query: str,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> list[Subject]:
    """Simple substring search over canonical_name (case-insensitive).

    Args:
        session: Active SQLAlchemy session.
        query: Substring to match against ``canonical_name``.
        limit: Maximum subjects to return. ``None`` returns every match
            (preserved for existing internal callers).
        offset: Number of matches to skip before returning. Used with
            ``limit`` to drive pagination — a generic substring like
            "the" can match thousands of subjects.

    Results are alphabetically ordered by ``canonical_name`` so paginated
    callers get stable slices.
    """
    # Escape user-input LIKE wildcards (security review F7): without this a
    # query of "%" matches every row in an unindexed full scan, and "_" matches
    # any single character. Mirrors the escaped prefix sites (e.g. get_subject).
    pattern = f"%{escape_like_pattern(query)}%"
    stmt = (
        select(SubjectRow)
        .where(SubjectRow.canonical_name.ilike(pattern, escape=LIKE_ESCAPE))
        .order_by(SubjectRow.canonical_name)
    )
    if offset:
        stmt = stmt.offset(offset)
    if limit is not None:
        stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return [row.to_model() for row in result.scalars()]


async def count_subjects(session: AsyncSession) -> int:
    """Total subject count."""
    from sqlalchemy import func

    result = await session.execute(select(func.count(SubjectRow.id)))
    return int(result.scalar_one())


async def count_subjects_without_active_particles(session: AsyncSession) -> int:
    """Subjects with zero ACTIVE particles linked via the particle_subjects join."""
    from sqlalchemy import func

    from particles.store.particle_store import ParticleRow

    covered_subject_ids = (
        select(ParticleSubjectRow.subject_id)
        .join(ParticleRow, ParticleRow.id == ParticleSubjectRow.particle_id)
        .where(ParticleRow.status == "ACTIVE")
        .distinct()
    )
    result = await session.execute(
        select(func.count(SubjectRow.id)).where(SubjectRow.id.not_in(covered_subject_ids))
    )
    return int(result.scalar_one())


def _active_claim_count_subquery() -> Any:
    """Subquery: (subject_id, count) of ACTIVE CLAIM particles per subject."""
    from sqlalchemy import func

    from particles.store.particle_store import ParticleRow

    return (
        select(
            ParticleSubjectRow.subject_id,
            func.count(ParticleRow.id).label("cnt"),
        )
        .join(ParticleRow, ParticleRow.id == ParticleSubjectRow.particle_id)
        .where(
            ParticleRow.status == "ACTIVE",
            ParticleRow.particle_type == "CLAIM",
        )
        .group_by(ParticleSubjectRow.subject_id)
        .subquery()
    )


async def get_phantom_subjects(session: AsyncSession) -> list[Subject]:
    """Return subjects with zero ACTIVE CLAIM particles linked."""
    count_sq = _active_claim_count_subquery()
    result = await session.execute(
        select(SubjectRow)
        .outerjoin(count_sq, count_sq.c.subject_id == SubjectRow.id)
        .where(count_sq.c.cnt.is_(None))
    )
    return [row.to_model() for row in result.scalars()]


async def find_duplicate_subjects(
    session: AsyncSession, *, threshold: float
) -> list[tuple[Subject, Subject, float]]:
    """Candidate-duplicate subject pairs by name/alias embedding similarity.

    For each subject the name-set ``{canonical_name} ∪ aliases`` is embedded; a
    pair is a candidate when the maximum cosine similarity across the two
    name-sets is ``>= threshold`` — so "Applied Optoelectronics" matches a
    second subject whose *alias* is "Applied Opto". Pairs are returned sorted by
    similarity descending. Returns an empty list when there are fewer than two
    subjects or the embedding model is unavailable (callers surface that).

    Read-only diagnostic: the operator reviews the pairs and decides
    what to ``subjects merge``. O(n²) over subjects × names — fine for the
    occasional-run scale this verb targets.
    """
    import numpy as np

    from particles.embeddings import cosine_similarity, get_embedding_model

    subjects = await list_all_subjects(session)
    if len(subjects) < 2:
        return []
    model = get_embedding_model()
    if model is None:
        return []

    # Flatten every name, remembering which subject (by index) owns each row.
    names: list[str] = []
    rows_by_subject: list[list[int]] = [[] for _ in subjects]
    for idx, s in enumerate(subjects):
        for name in [s.canonical_name, *s.aliases]:
            rows_by_subject[idx].append(len(names))
            names.append(name)
    # encode() returns L2-normalised rows by default, so cosine == dot product.
    vecs = np.asarray(model.encode(names), dtype=np.float32)

    pairs: list[tuple[Subject, Subject, float]] = []
    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            # normalized cosine clamped to [0, 1] — the score escapes
            # into the returned pairs, so keep it on the normative scale.
            sim = max(
                cosine_similarity(vecs[a], vecs[b])
                for a in rows_by_subject[i]
                for b in rows_by_subject[j]
            )
            if sim >= threshold:
                pairs.append((subjects[i], subjects[j], sim))
    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs


async def get_low_coverage_subjects(
    session: AsyncSession, threshold: int, canonical_only: bool = False
) -> list[tuple[Subject, int]]:
    """Return (subject, count) for subjects with 1 ≤ count < threshold ACTIVE CLAIM particles.

    Args:
        session: async session.
        threshold: subjects with strictly fewer ACTIVE CLAIM particles than
            this are returned.
        canonical_only: when True, restrict to subjects that carry at least
            one ``external_ids`` reference (Wikidata, Numista, ISBN, DOI,
            Nomisma, etc.). Author-name / handle-name subjects without any
            external authority link are typically expected to have sparse
            coverage and the low-coverage signal is noise on them.
    """
    if threshold <= 1:
        return []
    count_sq = _active_claim_count_subquery()
    stmt = (
        select(SubjectRow, count_sq.c.cnt)
        .join(count_sq, count_sq.c.subject_id == SubjectRow.id)
        .where(count_sq.c.cnt < threshold)
    )
    if canonical_only:
        # external_ids_json defaults to "[]"; treat any non-empty-list value
        # as canonical. Avoids a JSON1 extension dependency on SQLite.
        stmt = stmt.where(SubjectRow.external_ids_json != "[]")
    result = await session.execute(stmt)
    return [(row.to_model(), int(cnt)) for row, cnt in result]

"""SQLAlchemy ORM and helpers for typed particle relations (§6.10).

The relation table is the storage primitive for co-evidential links: typed
edges between two particles indicating that they assert the same underlying
claim. The transitive closure (the *group* of mutually co-evidential
particles) is computed at query time, not persisted, so a group of N
particles stores O(N²) edges in the worst case but real groups are small
(typically 2–5).

Symmetry is per-kind. ``_SYMMETRIC_KINDS`` enumerates the
kinds whose ``(particle_a, particle_b)`` endpoint order is invariant
("A ↔ B" = "B ↔ A"); writers canonicalise to ``(min, max)`` for those
so duplicate logical edges collide at the unique constraint regardless
of insertion order. Asymmetric kinds (``PART_OF`` / ``SEQUENCE_IN``
ACTIVE; ``BOOSTS`` / ``QUOTES`` / ``REPLIES_TO`` /
``MENTIONS`` reserved per registry) preserve the supplied
direction verbatim — for them "A is part of B" is a different fact than
"B is part of A" and the endpoint order carries that information.

The store deliberately does not enforce that referenced particles exist
(no FK). Real workflows insert a particle and a relation in the same
transaction; an FK would force ordering for no operational benefit, and
relations to deleted particles are surfaced as a lint finding rather than
prevented at insert time.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime

from sqlalchemy import DateTime, Float, Index, String, UniqueConstraint, delete, or_, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.core.schema import ParticleRelation, RelationCreatedBy, RelationType
from particles.core.stance import STANCE_KINDS
from particles.db import Base
from particles.store.event_store import EventRefKind, OperatorEventType, record_event

log = logging.getLogger(__name__)


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    """Return (a, b) sorted lexicographically so the unique constraint catches duplicates."""
    return (a, b) if a <= b else (b, a)


# Kinds whose ``(particle_a, particle_b)`` pair is order-invariant —
# "A ↔ B" represents the same fact as "B ↔ A". The store canonicalises
# the endpoint order for these kinds on every write so duplicate
# insertions of the same logical edge collide at the unique constraint.
# Asymmetric kinds (e.g. ``BOOSTS`` — "A boosts B" ≠ "B boosts A")
# preserve the operator-supplied direction verbatim.
_SYMMETRIC_KINDS: frozenset[RelationType] = frozenset(
    {RelationType.CO_EVIDENTIAL, RelationType.CONTRADICTS}
)


def _endpoints_for_write(a: str, b: str, kind: RelationType) -> tuple[str, str]:
    """Return the ``(particle_a, particle_b)`` pair the row should store.

    Canonicalises for symmetric kinds; preserves the supplied order for
    asymmetric kinds. The single seam every writer + deleter routes
    through so the kind-awareness rule is enforced in one place.
    """
    if kind in _SYMMETRIC_KINDS:
        return _canonical_pair(a, b)
    return (a, b)


class ParticleRelationRow(Base):
    __tablename__ = "particle_relations"

    # Composite primary key plus a unique constraint over the typed pair.
    # The pair is always stored canonicalised, so (a, b) collisions are
    # detected regardless of insertion order.
    particle_a: Mapped[str] = mapped_column(String, primary_key=True)
    particle_b: Mapped[str] = mapped_column(String, primary_key=True)
    relation_type: Mapped[str] = mapped_column(String, primary_key=True)
    created_by: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    __table_args__ = (
        UniqueConstraint(
            "particle_a", "particle_b", "relation_type", name="uq_particle_relations_pair_type"
        ),
        # Both endpoints indexed for O(1) neighbour lookup during BFS.
        Index("ix_particle_relations_a", "particle_a", "relation_type"),
        Index("ix_particle_relations_b", "particle_b", "relation_type"),
    )

    def to_model(self) -> ParticleRelation:
        return ParticleRelation(
            particle_a=self.particle_a,
            particle_b=self.particle_b,
            relation_type=RelationType(self.relation_type),
            created_by=RelationCreatedBy(self.created_by),
            created_at=self.created_at,
            confidence=self.confidence,
        )


async def create_relation(
    session: AsyncSession,
    particle_a: str,
    particle_b: str,
    relation_type: RelationType,
    created_by: RelationCreatedBy,
    confidence: float = 1.0,
) -> ParticleRelation:
    """Create a relation between two particles.

    The pair is canonicalised so insertion order does not matter. A relation
    between a particle and itself is rejected — a particle does not
    corroborate itself.

    Returns the persisted ``ParticleRelation``. Raises ``ValueError`` for
    self-relations.
    """
    if particle_a == particle_b:
        raise ValueError("Cannot create a relation between a particle and itself")

    a, b = _endpoints_for_write(particle_a, particle_b, relation_type)
    relation = ParticleRelation(
        particle_a=a,
        particle_b=b,
        relation_type=relation_type,
        created_by=created_by,
        confidence=confidence,
    )
    session.add(
        ParticleRelationRow(
            particle_a=a,
            particle_b=b,
            relation_type=relation_type.value,
            created_by=created_by.value,
            created_at=relation.created_at,
            confidence=confidence,
        )
    )
    await session.flush()
    # only the manual `links add` operator action is logged.
    # LLM_JUDGE / EXTRACTOR_DIRECT relations are not operator decisions.
    if created_by == RelationCreatedBy.MANUAL_CLI:
        await record_event(
            session,
            actor="links-add",
            event_type=OperatorEventType.RELATION_ADDED,
            refs=[(EventRefKind.PARTICLE, a), (EventRefKind.PARTICLE, b)],
            payload={"relation_type": relation_type.value, "confidence": confidence},
        )
    return relation


async def get_relations_for_particle(
    session: AsyncSession,
    particle_id: str,
    relation_type: RelationType | None = None,
) -> list[ParticleRelation]:
    """Return all relations incident to ``particle_id``, optionally typed."""
    stmt = select(ParticleRelationRow).where(
        or_(
            ParticleRelationRow.particle_a == particle_id,
            ParticleRelationRow.particle_b == particle_id,
        )
    )
    if relation_type is not None:
        stmt = stmt.where(ParticleRelationRow.relation_type == relation_type.value)
    result = await session.execute(stmt)
    return [row.to_model() for row in result.scalars()]


async def get_all_relations(
    session: AsyncSession, relation_type: RelationType
) -> list[ParticleRelation]:
    """Return every edge of one kind in the store, in insertion order.

    The store-wide read aggregation behind the contested finder: a
    hygiene surface asks "what in this store is disputed / co-evidential?",
    which is a question about the *edges*, not about a candidate list. Asking
    once and grouping in memory is what lets that finder evaluate every ACTIVE
    belief without the per-particle ``get_incoming`` /
    ``get_co_evidential_group`` walk the read path pays on a bounded top-k.
    """
    result = await session.execute(
        select(ParticleRelationRow)
        .where(ParticleRelationRow.relation_type == relation_type.value)
        .order_by(ParticleRelationRow.created_at)
    )
    return [row.to_model() for row in result.scalars()]


async def get_incoming(
    session: AsyncSession,
    particle_id: str,
    relation_type: RelationType,
) -> list[str]:
    """Return the ``particle_a`` ids of edges pointing *to* ``particle_id``.

    For an asymmetric kind the endpoint order is stored verbatim, so this is
    the directed "who points at me" query: every row whose ``particle_b`` is
    ``particle_id``. Used for narrative constituents — the PART_OF children of
    a NARRATIVE. Ids are returned sorted for deterministic output.
    """
    result = await session.execute(
        select(ParticleRelationRow.particle_a).where(
            ParticleRelationRow.particle_b == particle_id,
            ParticleRelationRow.relation_type == relation_type.value,
        )
    )
    return sorted(result.scalars())


async def get_outgoing(
    session: AsyncSession,
    particle_id: str,
    relation_type: RelationType,
) -> list[str]:
    """Return the ``particle_b`` ids of edges pointing *from* ``particle_id``.

    The directed "what do I point at" query for an asymmetric kind: every row
    whose ``particle_a`` is ``particle_id``. Used for the narratives a particle
    belongs to (its PART_OF parents) and for SEQUENCE_IN successors.
    Ids are returned sorted for deterministic output.
    """
    result = await session.execute(
        select(ParticleRelationRow.particle_b).where(
            ParticleRelationRow.particle_a == particle_id,
            ParticleRelationRow.relation_type == relation_type.value,
        )
    )
    return sorted(result.scalars())


async def get_co_evidential_group(
    session: AsyncSession, particle_id: str, min_confidence: float = 0.0
) -> set[str]:
    """Return the transitive closure of CO_EVIDENTIAL links from ``particle_id``.

    The result always includes ``particle_id`` itself. A particle with no
    CO_EVIDENTIAL links returns ``{particle_id}`` — a singleton group.

    Only edges whose ``effective_equivalence`` is ``>= min_confidence``
    are traversed — graded, threshold-gated collapse over the edge's link
    confidence. The default ``0.0`` reproduces the pre-0106 behaviour of
    collapsing on *any* CO_EVIDENTIAL link (identity lens; the per-observer term
    is reserved).

    Computed via BFS over ``particle_relations``. Real groups are small
    (typically 2–5 members) so the per-query overhead is bounded.
    """
    from particles.core.equivalence import effective_equivalence

    visited: set[str] = {particle_id}
    queue: deque[str] = deque([particle_id])
    while queue:
        current = queue.popleft()
        result = await session.execute(
            select(ParticleRelationRow).where(
                or_(
                    ParticleRelationRow.particle_a == current,
                    ParticleRelationRow.particle_b == current,
                ),
                ParticleRelationRow.relation_type == RelationType.CO_EVIDENTIAL.value,
            )
        )
        for row in result.scalars():
            if effective_equivalence(row.confidence) < min_confidence:
                continue
            neighbour = row.particle_b if row.particle_a == current else row.particle_a
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return visited


async def remove_particle_from_relations(session: AsyncSession, particle_id: str) -> int:
    """Delete every *non-stance* relation incident to ``particle_id``.

    Returns the deletion count.

    Called when a particle's status transitions to RETRACTED. Co-evidential
    and narrative edges are deleted: the group survives, but the retracted
    particle leaves it. If a co-evidential group falls to a single member as a
    result, it is dissolved by virtue of having no relations left.

    **Kind-aware exception:** ``ENDORSES`` / ``DISPUTES`` stance
    edges are *preserved* as *dangling* edges rather than hard-deleted. The
    edge is a stance particle's role marker; blindly deleting it
    when a target is retracted would orphan the surviving stance — stripping
    its role marker so it silently drops out of the §4 agreement distribution
    with no lint signal. The dangling edge is excluded from the distribution
    (an endpoint is no longer ACTIVE) but remains inspectable for stance-lint
    .
    """
    stance_kind_values = [k.value for k in STANCE_KINDS]
    result: CursorResult[None] = await session.execute(  # type: ignore[assignment]
        delete(ParticleRelationRow).where(
            or_(
                ParticleRelationRow.particle_a == particle_id,
                ParticleRelationRow.particle_b == particle_id,
            ),
            ParticleRelationRow.relation_type.not_in(stance_kind_values),
        )
    )
    await session.flush()
    return result.rowcount or 0


async def delete_relation(
    session: AsyncSession,
    particle_a: str,
    particle_b: str,
    relation_type: RelationType,
    *,
    created_by: RelationCreatedBy | None = None,
    actor: str = "links-remove",
) -> bool:
    """Delete a specific typed relation between two particles.

    Args:
        created_by: When given, delete only if the stored edge carries this
            provenance; an unmerge needs it: it withdraws the merge's
            own ``EXACT_DUPLICATE`` edges and must never touch a co-evidential
            link a human or the judge created for the same pair.
        actor: Recorded on the ``RELATION_REMOVED`` event, so a deletion made
            by a revert attributes to the revert rather than to `links remove`.

    Returns True if a row was deleted, False if no matching relation existed.
    """
    a, b = _endpoints_for_write(particle_a, particle_b, relation_type)
    conditions = [
        ParticleRelationRow.particle_a == a,
        ParticleRelationRow.particle_b == b,
        ParticleRelationRow.relation_type == relation_type.value,
    ]
    if created_by is not None:
        conditions.append(ParticleRelationRow.created_by == created_by.value)
    result: CursorResult[None] = await session.execute(  # type: ignore[assignment]
        delete(ParticleRelationRow).where(*conditions)
    )
    await session.flush()
    removed = (result.rowcount or 0) > 0
    if removed:
        payload: dict[str, str] = {"relation_type": relation_type.value}
        if created_by is not None:
            payload["created_by"] = created_by.value
        await record_event(
            session,
            actor=actor,
            event_type=OperatorEventType.RELATION_REMOVED,
            refs=[(EventRefKind.PARTICLE, a), (EventRefKind.PARTICLE, b)],
            payload=payload,
        )
    return removed

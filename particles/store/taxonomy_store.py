"""SQLAlchemy ORM models and repository for the taxonomy store.

A ``TaxonomyDefinition`` is a depositable corpus artefact. The
``TaxonomyExtractor`` materialises its content into the three tables
defined here:

* ``taxonomies`` — one row per published TaxonomyDefinition.
* ``tag_nodes`` — one row per tag in any taxonomy; the query-time
  hierarchy index walked by :func:`expand_tags`.
* ``particle_tag_edges`` — denormalised (particle, tag) index for fast
  "particles by tag" lookups. Mirrors the dual-representation pattern
  used by :mod:`particles.store.subject_store` — the canonical list is
  ``ParticleRow.tags_json``, and this table is the index.

Repository helpers ``flush()`` but do not ``commit()`` — the caller owns
the transaction boundary.
"""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import DateTime, Index, String, Text, delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.core.schema import TagNode, TaxonomyDefinition
from particles.db import Base
from particles.store.event_store import EventRefKind, OperatorEventType, record_event


class TaxonomyRow(Base):
    __tablename__ = "taxonomies"

    taxonomy_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str] = mapped_column(String, nullable=False)
    author: Mapped[str] = mapped_column(String, nullable=False)
    domain: Mapped[str | None] = mapped_column(String, nullable=True)
    corpus_entry_id: Mapped[str | None] = mapped_column(String, nullable=True)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class TagNodeRow(Base):
    """One tag in a TaxonomyDefinition. Composite PK across taxonomy_id + tag."""

    __tablename__ = "tag_nodes"

    taxonomy_id: Mapped[str] = mapped_column(String, primary_key=True)
    tag: Mapped[str] = mapped_column(Text, primary_key=True)
    parent: Mapped[str | None] = mapped_column(Text, nullable=True)
    aliases_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (Index("ix_tag_nodes_taxonomy_parent", "taxonomy_id", "parent"),)


class ParticleTagEdgeRow(Base):
    """Join table: particle ↔ tag (one row per assignment, denormalised
    so a tag can come from any active taxonomy or be operator-forced)."""

    __tablename__ = "particle_tag_edges"

    particle_id: Mapped[str] = mapped_column(String, primary_key=True)
    tag: Mapped[str] = mapped_column(Text, primary_key=True)

    __table_args__ = (Index("ix_particle_tag_edges_tag", "tag"),)


# ---------------------------------------------------------------------------
# Repository helpers
# ---------------------------------------------------------------------------


async def insert_taxonomy(session: AsyncSession, td: TaxonomyDefinition) -> None:
    """Persist a TaxonomyDefinition and its tag nodes.

    Idempotent on the ``(taxonomy_id, tag)`` PK — re-extracting the same
    artefact replaces nothing; the second call's row inserts are silently
    skipped at the DB level if they would conflict. Callers needing
    revision semantics should publish a new TaxonomyDefinition with a
    fresh ``taxonomy_id``.
    """
    session.add(
        TaxonomyRow(
            taxonomy_id=td.taxonomy_id,
            name=td.name,
            version=td.version,
            author=td.author,
            domain=td.domain,
            corpus_entry_id=td.corpus_entry_id,
            published_at=td.published_at,
        )
    )
    for node in td.tags:
        session.add(
            TagNodeRow(
                taxonomy_id=td.taxonomy_id,
                tag=node.tag,
                parent=node.parent,
                aliases_json=json.dumps(node.aliases),
                description=node.description,
            )
        )
    await session.flush()


async def get_taxonomy(session: AsyncSession, taxonomy_id: str) -> TaxonomyDefinition | None:
    row = await session.get(TaxonomyRow, taxonomy_id)
    if row is None:
        return None
    nodes_result = await session.execute(
        select(TagNodeRow).where(TagNodeRow.taxonomy_id == taxonomy_id)
    )
    nodes = [
        TagNode(
            tag=n.tag,
            parent=n.parent,
            aliases=json.loads(n.aliases_json),
            description=n.description,
        )
        for n in nodes_result.scalars()
    ]
    return TaxonomyDefinition(
        taxonomy_id=row.taxonomy_id,
        name=row.name,
        version=row.version,
        author=row.author,
        domain=row.domain,
        tags=nodes,
        published_at=row.published_at,
        corpus_entry_id=row.corpus_entry_id,
    )


async def list_taxonomies(
    session: AsyncSession,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> list[TaxonomyDefinition]:
    """List taxonomies ordered by name.

    Args:
        session: Active SQLAlchemy session.
        limit: Maximum taxonomies to return. ``None`` returns all (current
            default; preserved so existing internal callers don't have to
            pass a value).
        offset: Number of taxonomies to skip before returning results. Used
            with ``limit`` to drive pagination at the MCP surface, where the
            full tag tree per taxonomy can easily push a response past the
            per-tool-result token cap.
    """
    stmt = select(TaxonomyRow).order_by(TaxonomyRow.name)
    if offset:
        stmt = stmt.offset(offset)
    if limit is not None:
        stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    out: list[TaxonomyDefinition] = []
    for row in result.scalars():
        td = await get_taxonomy(session, row.taxonomy_id)
        if td is not None:
            out.append(td)
    return out


async def tag_exists(session: AsyncSession, tag: str) -> bool:
    """Return True if ``tag`` is declared in any taxonomy.

    Used by the CLI to gate the default ``particle tag`` path — operators
    pass ``--force`` to bypass this check for ad-hoc tags that aren't yet
    in any taxonomy.
    """
    result = await session.execute(select(TagNodeRow.tag).where(TagNodeRow.tag == tag).limit(1))
    return result.scalar_one_or_none() is not None


async def expand_tags(
    session: AsyncSession, requested: list[str], *, include_ancestors: bool = False
) -> set[str]:
    """Subtree-expand each requested tag across all active taxonomies.

    A request for ``coins`` returns the union of {``coins``,
    ``coins/by-region/germany``, ``coins/by-region/usa``, …} across every
    taxonomy that defines a ``coins`` root. Each requested tag is always
    included in the result even when it's not defined in any taxonomy —
    that covers the ``--force`` ad-hoc tag case (the particle was tagged
    with a string the operator made up).

    When ``include_ancestors`` is True, the result also
    includes each requested tag's *parent chain* — so a query for a specific
    node additionally matches particles tagged only with a broader ancestor.
    Ancestors are added as the bare chain, not their full subtrees, so sibling
    branches are not pulled in.

    Implemented as a breadth-first walk of the ``(taxonomy_id, parent)``
    index, one indexed ``SELECT`` per depth level. Avoids recursive-CTE
    syntax so the query is portable across SQLite and Postgres.
    """
    expanded: set[str] = set()
    frontier: set[str] = set(requested)
    expanded.update(frontier)
    while frontier:
        children_result = await session.execute(
            select(TagNodeRow.tag).where(TagNodeRow.parent.in_(frontier))
        )
        children = {tag for tag in children_result.scalars().all() if tag not in expanded}
        if not children:
            break
        expanded.update(children)
        frontier = children

    if include_ancestors:
        # Walk UP the parent chain from each requested tag (not the
        # subtree-expanded set — we want broader ancestors, not the
        # ancestors of every descendant).
        anc_frontier: set[str] = set(requested)
        while anc_frontier:
            parents_result = await session.execute(
                select(TagNodeRow.parent).where(
                    TagNodeRow.tag.in_(anc_frontier),
                    TagNodeRow.parent.is_not(None),
                )
            )
            parents = {p for p in parents_result.scalars().all() if p and p not in expanded}
            if not parents:
                break
            expanded.update(parents)
            anc_frontier = parents

    return expanded


async def set_particle_tags(session: AsyncSession, particle_id: str, tags: list[str]) -> None:
    """Replace the particle's tags (canonical JSON + edge table) in one shot.

    Both representations stay consistent: ``ParticleRow.tags_json`` is the
    canonical list (round-tripped to ``Particle.tags``), and the edge
    table is rebuilt for this particle.
    """
    from particles.store.particle_store import ParticleRow

    row = await session.get(ParticleRow, particle_id)
    if row is None:
        raise ValueError(f"Particle {particle_id} not found")

    deduped = list(dict.fromkeys(tags))  # preserve order, drop dupes
    row.tags_json = json.dumps(deduped) if deduped else None

    await session.execute(
        delete(ParticleTagEdgeRow).where(ParticleTagEdgeRow.particle_id == particle_id)
    )
    for tag in deduped:
        session.add(ParticleTagEdgeRow(particle_id=particle_id, tag=tag))
    await session.flush()


async def add_particle_tags(
    session: AsyncSession, particle_id: str, new_tags: list[str]
) -> list[str]:
    """Add tags to a particle, idempotent on existing tags. Returns added tags."""
    from particles.store.particle_store import ParticleRow

    row = await session.get(ParticleRow, particle_id)
    if row is None:
        raise ValueError(f"Particle {particle_id} not found")

    current: list[str] = json.loads(row.tags_json) if row.tags_json else []
    current_set = set(current)
    added = [t for t in dict.fromkeys(new_tags) if t not in current_set]
    if not added:
        return []
    merged = current + added
    row.tags_json = json.dumps(merged)
    for tag in added:
        session.add(ParticleTagEdgeRow(particle_id=particle_id, tag=tag))
    await session.flush()
    await record_event(
        session,
        actor="particle-tag",
        event_type=OperatorEventType.PARTICLE_TAGGED,
        refs=[(EventRefKind.PARTICLE, particle_id)],
        payload={"added": added},
    )
    return added


async def remove_particle_tags(
    session: AsyncSession, particle_id: str, tags: list[str]
) -> list[str]:
    """Remove tags from a particle, idempotent on missing tags. Returns removed tags."""
    from particles.store.particle_store import ParticleRow

    row = await session.get(ParticleRow, particle_id)
    if row is None:
        raise ValueError(f"Particle {particle_id} not found")

    current: list[str] = json.loads(row.tags_json) if row.tags_json else []
    to_remove = set(tags) & set(current)
    if not to_remove:
        return []
    remaining = [t for t in current if t not in to_remove]
    row.tags_json = json.dumps(remaining) if remaining else None
    await session.execute(
        delete(ParticleTagEdgeRow).where(
            ParticleTagEdgeRow.particle_id == particle_id,
            ParticleTagEdgeRow.tag.in_(to_remove),
        )
    )
    await session.flush()
    removed = sorted(to_remove)
    await record_event(
        session,
        actor="particle-untag",
        event_type=OperatorEventType.PARTICLE_UNTAGGED,
        refs=[(EventRefKind.PARTICLE, particle_id)],
        payload={"removed": removed},
    )
    return removed


async def get_particle_ids_for_tags(session: AsyncSession, tags: set[str]) -> set[str]:
    """Return particle IDs that carry at least one of the given tags."""
    if not tags:
        return set()
    result = await session.execute(
        select(ParticleTagEdgeRow.particle_id).where(ParticleTagEdgeRow.tag.in_(tags))
    )
    return set(result.scalars().all())


# Inverted taxonomy-persistence coupling: the Engine
# registers this store write with the Client-layer ``extraction.taxonomy`` module
# at import time, so ``TaxonomyExtractor.extract`` persists a parsed
# ``TaxonomyDefinition`` without ``extraction.taxonomy`` importing any Engine
# module. ``_orm_modules`` imports this module (ORM registration) and
# ``ingest.pipeline`` imports it explicitly, so the sink is set before any
# taxonomy extraction runs. Engine→Client import is allowed. Mirrors the
# carry-forward registration in ``particle_store``.
from particles.extraction.taxonomy import register_taxonomy_sink  # noqa: E402

register_taxonomy_sink(insert_taxonomy)

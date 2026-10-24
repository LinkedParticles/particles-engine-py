# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""SQLAlchemy ORM models and repository for the Subject store.

Subjects are canonical real-world entities — the nodes of the knowledge graph.
Particles are statements (properties or edges) about subjects.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from sqlalchemy import ColumnElement, DateTime, String, Text, delete, insert, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.config import get_config
from particles.core.artifact_fold import ARTIFACT_REF_NAMESPACE, artifact_ref_project
from particles.core.schema import ContributorRef, ExternalRef, Subject
from particles.db import Base
from particles.sql_safety import LIKE_ESCAPE, escape_like_pattern
from particles.store import subject_cache
from particles.store.event_store import EventRefKind, OperatorEventType, record_event

log = logging.getLogger(__name__)

_IN_CHUNK = 10_000
"""Ids per ``IN (...)`` query, well under SQLite's 32766-variable ceiling."""


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


class SubjectAliasRow(Base):
    """One recorded alias of a Subject, keyed for lookup.

    A derived index over ``SubjectRow.aliases_json``, which stays the system of
    record: :func:`rebuild_alias_index` re-derives it, so dropping it changes
    speed, never an answer (D1). Every read joins ``subjects``, so a
    row a missed delete left behind is invisible to answers and is reported by
    :func:`check_alias_index`.
    """

    __tablename__ = "subject_aliases"

    subject_id: Mapped[str] = mapped_column(String, primary_key=True)
    alias_key: Mapped[str] = mapped_column(Text, primary_key=True, index=True)
    alias: Mapped[str] = mapped_column(Text, nullable=False)


def alias_key(name: str) -> str:
    """The folded form the alias index compares.

    ``str.lower()`` computed in Python, the comparison :func:`find_by_name`,
    :func:`add_aliases` and the merge path make, so moving a lookup onto the
    index changes no binding. SQLite's ``lower()`` folds ASCII only, which is
    why the key is stored rather than computed in SQL.
    """
    return name.lower()


def _alias_index_rows(subject_id: str, aliases: Sequence[str]) -> list[dict[str, str]]:
    """The index rows for one Subject's aliases: one per key, first spelling kept."""
    rows: dict[str, dict[str, str]] = {}
    for alias in aliases:
        rows.setdefault(
            alias_key(alias),
            {"subject_id": subject_id, "alias_key": alias_key(alias), "alias": alias},
        )
    return list(rows.values())


async def _index_aliases(session: AsyncSession, subject_id: str, aliases: Sequence[str]) -> None:
    """Replace one Subject's index rows with ``aliases`` (core SQL, no identity map)."""
    await session.execute(delete(SubjectAliasRow).where(SubjectAliasRow.subject_id == subject_id))
    rows = _alias_index_rows(subject_id, aliases)
    if rows:
        await session.execute(insert(SubjectAliasRow), rows)


async def _set_aliases(session: AsyncSession, row: SubjectRow, aliases: list[str]) -> None:
    """Write a Subject's aliases and its index rows together.

    The one place ``aliases_json`` changes after a row is created;
    ``tests/test_alias_index_writes.py`` fails on a write anywhere else.
    """
    row.aliases_json = json.dumps(aliases)
    await _index_aliases(session, row.id, aliases)


async def drop_aliases(session: AsyncSession, subject_ids: Collection[str]) -> None:
    """Delete the index rows of Subjects being deleted."""
    ids = list(subject_ids)
    for i in range(0, len(ids), _IN_CHUNK):
        chunk = ids[i : i + _IN_CHUNK]
        await session.execute(delete(SubjectAliasRow).where(SubjectAliasRow.subject_id.in_(chunk)))


async def _rows_with_alias(session: AsyncSession, name: str) -> list[SubjectRow]:
    """Subjects holding ``name`` as an alias, earliest first, through the index.

    Joined to ``subjects``, so an index row whose Subject is gone never answers.
    """
    result = await session.execute(
        select(SubjectRow)
        .join(SubjectAliasRow, SubjectAliasRow.subject_id == SubjectRow.id)
        .where(SubjectAliasRow.alias_key == alias_key(name))
        .order_by(SubjectRow.created_at, SubjectRow.id)
    )
    return list(result.scalars())


@dataclass(frozen=True)
class AliasIndexDrift:
    """How the alias index differs from what ``aliases_json`` derives."""

    missing: int
    """Rows a rebuild would add."""
    extra: int
    """Rows a rebuild would remove whose Subject still exists."""
    orphaned: int
    """Rows whose Subject no longer exists."""
    subject_ids: list[str]
    """Every Subject id involved, sorted."""

    @property
    def clean(self) -> bool:
        return not (self.missing or self.extra or self.orphaned)


async def _expected_alias_rows(session: AsyncSession) -> set[tuple[str, str, str]]:
    result = await session.execute(select(SubjectRow.id, SubjectRow.aliases_json))
    return {
        (r["subject_id"], r["alias_key"], r["alias"])
        for sid, aliases_json in result
        for r in _alias_index_rows(sid, json.loads(aliases_json))
    }


async def check_alias_index(session: AsyncSession) -> AliasIndexDrift:
    """Compare the alias index with a rebuild of it, changing nothing."""
    expected = await _expected_alias_rows(session)
    actual_rows = await session.execute(
        select(SubjectAliasRow.subject_id, SubjectAliasRow.alias_key, SubjectAliasRow.alias)
    )
    actual = {(sid, key, alias) for sid, key, alias in actual_rows}
    subject_ids = set((await session.execute(select(SubjectRow.id))).scalars())
    surplus = actual - expected
    orphaned = {r for r in surplus if r[0] not in subject_ids}
    missing = expected - actual
    return AliasIndexDrift(
        missing=len(missing),
        extra=len(surplus - orphaned),
        orphaned=len(orphaned),
        subject_ids=sorted({r[0] for r in missing | surplus}),
    )


async def rebuild_alias_index(session: AsyncSession) -> int:
    """Re-derive the alias index from ``aliases_json``; return the rows written.

    The ``memory rebuild-utility`` shape: the index is disposable,
    and this is what makes it so (D1). Does not commit.
    """
    rows = [
        {"subject_id": sid, "alias_key": key, "alias": alias}
        for sid, key, alias in sorted(await _expected_alias_rows(session))
    ]
    await session.execute(delete(SubjectAliasRow))
    for i in range(0, len(rows), _IN_CHUNK):
        await session.execute(insert(SubjectAliasRow), rows[i : i + _IN_CHUNK])
    return len(rows)


# ---------------------------------------------------------------------------
# Repository helpers
# ---------------------------------------------------------------------------


async def insert_subject(session: AsyncSession, subject: Subject) -> None:
    """Persist a new subject and index its aliases. Flushes but does not commit."""
    session.add(SubjectRow.from_model(subject))
    await _index_aliases(session, subject.id, subject.aliases)
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


#: Candidates a refused ambiguous reference names before eliding the rest.
_AMBIGUOUS_SHOWN = 5


async def lookup_subject(session: AsyncSession, ref: str) -> Subject | None:
    """Resolve an operator-typed subject reference: ID, ID prefix, or name.

    The read surfaces that take a subject argument (``query --subject``,
    ``subjects show``, ``links``, the MCP ``subjects_show`` and
    ``particles_list``) call this instead of :func:`get_subject`, so a reader
    can type the name ``subjects list`` shows rather than copy an ID prefix.

    1. A full ID returns its Subject at once.
    2. Otherwise the ID-prefix matches and the name matches are pooled. A name
       matches case-insensitively and exactly, canonical names first; aliases
       answer only when no canonical name does (the :func:`find_by_name`
       tiering). The persona forms answer through the persona Subject's
       aliases, since this lookup binds nothing.
    3. One distinct Subject in the pool is the answer, and an empty pool
       returns ``None``. A larger pool raises: a reference that is both a hex
       prefix and a name ("cafe"), a prefix several IDs share, and a name two
       Subjects share are all refused with the candidates named, never settled
       by a silent tie-break.

    Raises:
        ValueError: If the reference matches more than one Subject.
    """
    from sqlalchemy import func

    ref = ref.strip()
    if not ref:
        return None
    row = await session.get(SubjectRow, ref)
    if row is not None:
        return row.to_model()

    pool: dict[str, tuple[SubjectRow, str]] = {}
    # Bounded: past one more than the shown candidates the reference is refused
    # anyway, and a one-character prefix can otherwise load most of the table.
    prefix_rows = await session.execute(
        select(SubjectRow)
        .where(SubjectRow.id.like(f"{escape_like_pattern(ref)}%", escape=LIKE_ESCAPE))
        .order_by(SubjectRow.id)
        .limit(_AMBIGUOUS_SHOWN + 1)
    )
    for r in prefix_rows.scalars():
        pool[r.id] = (r, "ID prefix")

    name_lower = ref.lower()
    by_canonical = await session.execute(
        select(SubjectRow)
        .where(func.lower(SubjectRow.canonical_name) == name_lower)
        .order_by(SubjectRow.created_at, SubjectRow.id)
    )
    name_rows: list[tuple[SubjectRow, str]] = [(r, "name") for r in by_canonical.scalars()]
    if not name_rows:
        name_rows = [(r, "alias") for r in await _rows_with_alias(session, ref)]
    for r, how in name_rows:
        pool.setdefault(r.id, (r, how))

    if not pool:
        return None
    if len(pool) == 1:
        return next(iter(pool.values()))[0].to_model()
    shown = list(pool.values())[:_AMBIGUOUS_SHOWN]
    listing = "; ".join(f"{r.id[:8]}… {r.canonical_name} (by {how})" for r, how in shown)
    more = ", and more" if len(pool) > _AMBIGUOUS_SHOWN else ""
    raise ValueError(
        f"Subject {ref!r} is ambiguous: it matches {listing}{more}."
        " Pass more characters of the ID you mean."
    )


async def get_subject_classes(
    session: AsyncSession, subject_ids: Collection[str]
) -> dict[str, str | None]:
    """``subject_class`` by subject id, one query per chunk; an unknown id is absent."""
    ids = sorted({sid for sid in subject_ids if sid})
    if not ids:
        return {}
    classes: dict[str, str | None] = {}
    for start in range(0, len(ids), _IN_CHUNK):
        result = await session.execute(
            select(SubjectRow.id, SubjectRow.subject_class).where(
                SubjectRow.id.in_(ids[start : start + _IN_CHUNK])
            )
        )
        classes.update(result.all())
    return classes


def is_persona_source(source_type: str | None, source_tags: Collection[str] = ()) -> bool:
    """Whether a corpus entry's first-person references are the store's speaker.

    True for a ``subjects.persona_source_types`` source type (a transcript, a
    journal) and, whatever the source type, for an entry carrying one of
    ``subjects.persona_source_tags``: a Claude Code memory file is
    deposited as ``LOCAL_MARKDOWN`` and tagged ``memory-file``, and its "I" is
    the same person a transcript's is. The one predicate both the fold and
    :func:`persona_alias_guard` read, so the two cannot disagree about scope.
    """
    cfg = get_config().subjects
    if source_type is not None and source_type in cfg.persona_source_types:
        return True
    return not set(source_tags).isdisjoint(cfg.persona_source_tags)


def persona_alias_guard(
    name: str, source_type: str | None, source_tags: Collection[str] = ()
) -> tuple[str, ...]:
    """Canonical names whose aliases must not answer ``name`` for this source.

    The persona fold records each surface form it folds ("user", "I", "the
    speaker") as an alias of the persona Subject, so a lookup that binds
    nothing — a graph anchor, an owner-lens name, a projection manifest —
    resolves them without having to know about the fold. A path
    that *binds a particle to a Subject* is the opposite case: "I" in a web
    page, or a Subject named "User" in an imported bundle, is not the person
    this store's transcripts call "I". Every such path passes this guard as
    ``find_by_name(..., skip_aliases_of=...)``, so outside a persona source
    (:func:`is_persona_source`) it binds exactly as it did before the aliases
    existed.

    Scoped to persona *forms*: an alias an operator added to the persona
    Subject ("Jeff") answers everywhere. Keyed on the configured canonical
    name, so renaming ``subjects.persona_canonical_name`` on a folded store
    wants a ``subjects merge`` of the old persona Subject into the new one —
    which it already needed, since the fold would otherwise split them.

    Returns:
        The ``skip_aliases_of`` value: empty inside a persona source (the
        resolver folds the name there, and never looks the raw form up) and
        for any name that is not a persona form.
    """
    cfg = get_config().subjects
    if is_persona_source(source_type, source_tags):
        return ()
    if name.strip().casefold() not in {a.casefold() for a in cfg.persona_aliases}:
        return ()
    return (cfg.persona_canonical_name,)


async def find_by_name(
    session: AsyncSession, name: str, *, skip_aliases_of: Collection[str] = ()
) -> Subject | None:
    """Case-insensitive lookup against canonical_name and aliases.

    Used by the subject resolver during extraction to avoid creating duplicate
    subjects for the same real-world entity.

    Args:
        session: Active SQLAlchemy session.
        name: The name to look up.
        skip_aliases_of: Canonical names (case-insensitive) of Subjects that
            may answer by canonical name only — their aliases are not
            consulted. A path that binds a particle to the result passes
            :func:`persona_alias_guard` here; a lookup that binds nothing
            leaves it empty.
    """
    from sqlalchemy import func

    name_lower = name.lower()
    skip_lower = {c.lower() for c in skip_aliases_of}

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

    # Aliases through the index. Two Subjects sharing an alias
    # resolve to the earliest, the canonical tier's tie-break.
    for row in await _rows_with_alias(session, name):
        if row.canonical_name.lower() in skip_lower:
            continue
        return row.to_model()
    return None


async def find_by_external_ref(
    session: AsyncSession, namespace: str, external_id: str
) -> Subject | None:
    """Look up a subject by external ontology reference."""
    query = select(SubjectRow)
    if external_id.isascii() and '"' not in external_id and "\\" not in external_id:
        # The stored JSON spells such an id verbatim, so a substring match is a
        # sound prefilter; the exact check below still decides. Without it every
        # lookup parsed every Subject, which a batch relink does thousands of
        # times.
        query = query.where(SubjectRow.external_ids_json.contains(external_id, autoescape=True))
    result = await session.execute(query)
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


async def attach_subjects_in_place(
    session: AsyncSession, particle_id: str, subject_ids: list[str]
) -> list[str]:
    """Link an unlinked particle to Subjects without superseding it.

    A subject link is an annotation, changed in place, as
    :func:`split_subject` and :func:`merge_subjects` already change it. Writes
    one ``particle_subjects`` row per Subject and the denormalised
    ``subject_ids_json`` in lockstep, so the particle keeps its id, its utility
    evidence, its observer scope and every card or relation naming it.

    Refuses a particle that already has a subject link: correcting a wrong
    subject is reassignment, not attachment. Records no event; the
    operation that calls this records one per batch. Caller commits.

    Returns:
        The Subject ids linked, deduplicated in order.

    Raises:
        ValueError: The particle or a Subject is missing, the particle is
            already linked, or ``subject_ids`` is empty.
    """
    from particles.store.particle_store import ParticleRow

    ids = list(dict.fromkeys(subject_ids))
    if not ids:
        raise ValueError("attach_subjects_in_place needs at least one subject id")
    row = await session.get(ParticleRow, particle_id)
    if row is None:
        raise ValueError(f"Particle {particle_id} not found")
    existing = await session.execute(
        select(ParticleSubjectRow.subject_id).where(ParticleSubjectRow.particle_id == particle_id)
    )
    if existing.first() is not None or json.loads(row.subject_ids_json):
        raise ValueError(f"Particle {particle_id} already has a subject link")
    for sid in ids:
        if await session.get(SubjectRow, sid) is None:
            raise ValueError(f"Subject {sid} not found")
        session.add(ParticleSubjectRow(particle_id=particle_id, subject_id=sid))
    row.subject_ids_json = json.dumps(ids)
    await session.flush()
    return ids


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
        await _set_aliases(session, row, current + added)
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


class SubjectNotFoundError(ValueError):
    """A subject named in a merge does not exist (absent, or already merged)."""


@dataclass(frozen=True)
class MergeSourcePlan:
    """One source of a merge: who it is, which particles it links, and its refs.

    ``refs_carried`` are the source's ``artifact`` refs the target gains, so a
    later qualified mention of the source's name finds the target.
    ``refs_withheld`` are the ``artifact`` refs of another project, which
    the merge drops as it always did.
    """

    subject_id: str
    canonical_name: str
    particle_ids: tuple[str, ...]
    refs_carried: tuple[ExternalRef, ...] = ()
    refs_withheld: tuple[ExternalRef, ...] = ()


@dataclass(frozen=True)
class SubjectMergePlan:
    """What an N→1 merge will do, computed before any row changes.

    The same plan drives the CLI's preview, the confirmation prompt, and the
    mutation itself, so what the operator confirmed is what is written.
    """

    target: Subject
    sources: tuple[MergeSourcePlan, ...]
    aliases_added: tuple[str, ...]

    @property
    def particles_relinked(self) -> int:
        """Particle links re-pointed, counted per source link (as the 1→1 merge did)."""
        return sum(len(s.particle_ids) for s in self.sources)


async def _missing_source_message(session: AsyncSession, source_id: str) -> str:
    """Name the merge that already folded ``source_id`` away, when one did."""
    from particles.store.event_store import list_events

    events = await list_events(
        session,
        ref_kind=EventRefKind.SUBJECT,
        ref_id=source_id,
        event_type=OperatorEventType.SUBJECTS_MERGED,
        limit=5,
    )
    for event in events:
        payload = event.payload or {}
        if source_id in payload.get("sources", []):
            return (
                f"Source subject {source_id} was already merged into "
                f"{payload.get('target')} (event {event.event_id})"
            )
    if events:
        # Events recorded before the N→1 form carry no ``sources`` list, so the
        # role of this subject in them is not recorded.
        return (
            f"Source subject {source_id} not found; merge event "
            f"{events[0].event_id} names it, so it was probably merged already"
        )
    return f"Source subject {source_id} not found"


async def plan_subject_merge(
    session: AsyncSession, source_ids: Sequence[str], target_id: str
) -> SubjectMergePlan:
    """Validate an N→1 merge and compute what it will move.

    Ids may be full UUIDs or unambiguous prefixes. Every source is checked
    before anything is written, so a bad source refuses the whole merge.

    Raises:
        SubjectNotFoundError: The target or a source does not exist. A source
            that an earlier merge folded away is named as such.
        ValueError: No sources, a source equal to the target, a source listed
            twice, or an ambiguous prefix.
    """
    if not source_ids:
        raise ValueError("At least one source subject is required")
    target = await get_subject(session, target_id)
    if target is None:
        raise SubjectNotFoundError(f"Target subject {target_id} not found")

    resolved: list[Subject] = []
    seen: set[str] = set()
    for sid in source_ids:
        source = await get_subject(session, sid)
        if source is None:
            raise SubjectNotFoundError(await _missing_source_message(session, sid))
        if source.id == target.id:
            raise ValueError(
                f"Subject {source.id} ({source.canonical_name}) is the merge target;"
                " it cannot also be a source"
            )
        if source.id in seen:
            raise ValueError(f"Source subject {source.id} is listed more than once")
        seen.add(source.id)
        resolved.append(source)

    existing_lower = {target.canonical_name.lower()} | {a.lower() for a in target.aliases}
    added: list[str] = []
    plans: list[MergeSourcePlan] = []
    held_refs = {(r.namespace, r.id) for r in target.external_ids}
    target_projects = {
        artifact_ref_project(r.id)
        for r in target.external_ids
        if r.namespace == ARTIFACT_REF_NAMESPACE
    }
    for source in resolved:
        for name in [source.canonical_name, *source.aliases]:
            if name.lower() not in existing_lower:
                existing_lower.add(name.lower())
                added.append(name)
        carried: list[ExternalRef] = []
        withheld: list[ExternalRef] = []
        for ref in source.external_ids:
            if ref.namespace != ARTIFACT_REF_NAMESPACE or (ref.namespace, ref.id) in held_refs:
                continue
            if artifact_ref_project(ref.id) in target_projects:
                held_refs.add((ref.namespace, ref.id))
                carried.append(ref)
            else:
                withheld.append(ref)
        result = await session.execute(
            select(ParticleSubjectRow.particle_id).where(ParticleSubjectRow.subject_id == source.id)
        )
        plans.append(
            MergeSourcePlan(
                subject_id=source.id,
                canonical_name=source.canonical_name,
                particle_ids=tuple(result.scalars().all()),
                refs_carried=tuple(carried),
                refs_withheld=tuple(withheld),
            )
        )
    return SubjectMergePlan(target=target, sources=tuple(plans), aliases_added=tuple(added))


async def merge_subjects_many(
    session: AsyncSession,
    source_ids: Sequence[str],
    target_id: str,
    *,
    actor: str = "subjects-merge",
    extra_payload: dict[str, Any] | None = None,
) -> tuple[Subject, SubjectMergePlan]:
    """Merge several source subjects into one target in a single pass.

    Each source's canonical_name and aliases are added to the target as
    aliases; a source's ``artifact`` refs in one of the target's projects are
    added to the target's refs; every particle link to a source
    is re-pointed to the target; the sources are deleted. One
    ``SUBJECTS_MERGED`` event records the whole merge. ``extra_payload`` is
    merged into that event's payload (the fold records its rule and
    the source as it was there).
    Every source is validated first (:func:`plan_subject_merge`), so the merge
    is all-or-nothing within the caller's transaction. Irreversible; the
    caller confirms, then commits.

    Returns:
        Tuple of (updated_target_subject, plan_that_was_applied).
    """
    plan = await plan_subject_merge(session, source_ids, target_id)
    target_id = plan.target.id
    source_set = {s.subject_id for s in plan.sources}

    target_row = await session.get(SubjectRow, target_id)
    assert target_row is not None  # plan_subject_merge checked it
    if plan.aliases_added:
        current_aliases: list[str] = json.loads(target_row.aliases_json)
        await _set_aliases(session, target_row, current_aliases + list(plan.aliases_added))
    carried = [ref for src in plan.sources for ref in src.refs_carried]
    if carried:
        target_refs: list[dict[str, object]] = json.loads(target_row.external_ids_json)
        target_refs.extend(
            {"namespace": r.namespace, "id": r.id, "uri": r.uri, "confidence": r.confidence}
            for r in carried
        )
        target_row.external_ids_json = json.dumps(target_refs)

    # Re-link particle_subjects join rows
    result = await session.execute(
        select(ParticleSubjectRow).where(ParticleSubjectRow.subject_id.in_(source_set))
    )
    relinked_pids: list[str] = []
    already_on_target: list[str] = []
    for ps_row in result.scalars().all():
        existing_link = await session.get(ParticleSubjectRow, (ps_row.particle_id, target_id))
        if existing_link is not None and ps_row.particle_id not in relinked_pids:
            already_on_target.append(ps_row.particle_id)
        if existing_link is None:
            session.add(ParticleSubjectRow(particle_id=ps_row.particle_id, subject_id=target_id))
            # Flush so a particle linked to two sources finds this row on the
            # next lookup instead of adding a second one with the same key.
            await session.flush()
        relinked_pids.append(ps_row.particle_id)
        await session.delete(ps_row)

    # Update subject_ids_json on particle rows. A particle that named two of
    # the merged subjects names the target once.
    from particles.store.particle_store import ParticleRow

    p_result = await session.execute(
        select(ParticleRow).where(
            or_(*(ParticleRow.subject_ids_json.contains(s) for s in source_set))
        )
    )
    for p_row in p_result.scalars():
        ids: list[str] = json.loads(p_row.subject_ids_json)
        if not source_set.intersection(ids):
            continue
        new_ids: list[str] = []
        for i in ids:
            mapped = target_id if i in source_set else i
            if mapped not in new_ids:
                new_ids.append(mapped)
        p_row.subject_ids_json = json.dumps(new_ids)

    for source in plan.sources:
        source_row = await session.get(SubjectRow, source.subject_id)
        if source_row is not None:
            await session.delete(source_row)
    await drop_aliases(session, source_set)
    await session.flush()

    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.SUBJECTS_MERGED,
        refs=[
            *[(EventRefKind.SUBJECT, s.subject_id) for s in plan.sources],
            (EventRefKind.SUBJECT, target_id),
            *[(EventRefKind.PARTICLE, pid) for pid in relinked_pids],
        ],
        payload={
            "aliases_added": list(plan.aliases_added),
            "relinked": plan.particles_relinked,
            "sources": [s.subject_id for s in plan.sources],
            "source_names": [s.canonical_name for s in plan.sources],
            "target": target_id,
            "refs_carried": [f"{r.namespace}:{r.id}" for r in carried],
            "already_on_target": already_on_target,
            **(extra_payload or {}),
        },
    )

    subject_cache.clear(keep_negative=True)
    return target_row.to_model(), plan


async def restore_merged_source(
    session: AsyncSession,
    *,
    source: Subject,
    target_id: str,
    particle_ids: Sequence[str],
    keep_on_target: Collection[str],
    aliases_added: Collection[str],
    refs_carried: Collection[str],
    actor: str,
    payload: dict[str, Any],
) -> list[str]:
    """Reverse one 1→1 merge from what its event recorded.

    Recreates ``source`` (same id, names, refs and class), takes back from the
    target the aliases and ``namespace:id`` refs the merge gave it, and moves
    each of ``particle_ids`` back onto the source. A particle in
    ``keep_on_target`` was linked to the target before the merge and keeps
    that link too. Records one ``SUBJECTS_SPLIT`` event carrying ``payload``.
    Caller commits.

    Returns:
        The particle ids relinked to the source.

    Raises:
        ValueError: The source exists again, or the target does not.
    """
    if await session.get(SubjectRow, source.id) is not None:
        raise ValueError(f"Subject {source.id} already exists; the merge was undone already")
    target_row = await session.get(SubjectRow, target_id)
    if target_row is None:
        raise ValueError(f"Target subject {target_id} no longer exists")

    await insert_subject(session, source)

    drop_lower = {a.lower() for a in aliases_added}
    aliases: list[str] = json.loads(target_row.aliases_json)
    await _set_aliases(session, target_row, [a for a in aliases if a.lower() not in drop_lower])
    refs: list[dict[str, Any]] = json.loads(target_row.external_ids_json)
    target_row.external_ids_json = json.dumps(
        [r for r in refs if f"{r['namespace']}:{r['id']}" not in refs_carried]
    )

    from particles.store.particle_store import ParticleRow

    keep = set(keep_on_target)
    relinked: list[str] = []
    for pid in particle_ids:
        p_row = await session.get(ParticleRow, pid)
        if p_row is None:
            continue
        if await session.get(ParticleSubjectRow, (pid, source.id)) is None:
            session.add(ParticleSubjectRow(particle_id=pid, subject_id=source.id))
        ids: list[str] = json.loads(p_row.subject_ids_json)
        if pid in keep:
            new_ids = [*ids, source.id] if source.id not in ids else ids
        else:
            target_link = await session.get(ParticleSubjectRow, (pid, target_id))
            if target_link is not None:
                await session.delete(target_link)
            new_ids = []
            for i in ids:
                mapped = source.id if i == target_id else i
                if mapped not in new_ids:
                    new_ids.append(mapped)
            if source.id not in new_ids:
                new_ids.append(source.id)
        p_row.subject_ids_json = json.dumps(new_ids)
        relinked.append(pid)
    await session.flush()

    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.SUBJECTS_SPLIT,
        refs=[
            (EventRefKind.SUBJECT, target_id),
            (EventRefKind.SUBJECT, source.id),
            *[(EventRefKind.PARTICLE, pid) for pid in relinked],
        ],
        payload={"source_id": target_id, "new_subject_id": source.id, **payload},
    )
    subject_cache.clear(keep_negative=True)
    return relinked


async def merge_subjects(
    session: AsyncSession, source_id: str, target_id: str, *, actor: str = "subjects-merge"
) -> tuple[Subject, list[str], int]:
    """Merge one source subject into target: the 1→1 form of :func:`merge_subjects_many`.

    Returns:
        Tuple of (updated_target_subject, aliases_added, particles_relinked).

    Raises:
        ValueError: If either subject is not found, or source == target.
    """
    subject, plan = await merge_subjects_many(session, [source_id], target_id, actor=actor)
    return subject, list(plan.aliases_added), plan.particles_relinked


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
    await drop_aliases(session, [row.id])
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

    The extraction pipeline's classification path (no operator event: an
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
    via ``particle_subjects`` (canonical-name tie-break): "most-connected
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
    """Case-insensitive substring search over canonical names and aliases.

    Args:
        session: Active SQLAlchemy session.
        query: Substring to match against ``canonical_name`` or any recorded
            alias (through the index).
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
    # A Subject also matches through any recorded alias, inside
    # the one statement, so limit/offset page the matches themselves.
    by_alias = select(SubjectAliasRow.subject_id).where(
        SubjectAliasRow.alias_key.like(
            f"%{escape_like_pattern(alias_key(query))}%", escape=LIKE_ESCAPE
        )
    )
    stmt = (
        select(SubjectRow)
        .where(
            or_(
                SubjectRow.canonical_name.ilike(pattern, escape=LIKE_ESCAPE),
                SubjectRow.id.in_(by_alias),
            )
        )
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


async def list_subject_classes_and_links(
    session: AsyncSession, *, created_by: datetime | None = None
) -> list[tuple[str | None, list[ExternalRef]]]:
    """Every Subject's class and external links, one query.

    The plain rows the vocabulary report's subject header folds; ids and names
    are not read, since the header counts and never lists. ``created_by``
    keeps the Subjects created at or before that instant, as they stand now:
    a class or link changed since is read as current, and a Subject deleted
    or merged away since is absent.
    """
    if created_by is not None and created_by.tzinfo is None:
        created_by = created_by.replace(tzinfo=UTC)
    result = await session.execute(
        select(SubjectRow.subject_class, SubjectRow.external_ids_json, SubjectRow.created_at)
    )
    rows: list[tuple[str | None, list[ExternalRef]]] = []
    for subject_class, ext_json, created_at in result.all():
        if created_by is not None:
            created = created_at if created_at.tzinfo else created_at.replace(tzinfo=UTC)
            if created > created_by:
                continue
        refs = [
            ExternalRef(
                namespace=r["namespace"],
                id=r["id"],
                uri=r.get("uri"),
                confidence=float(r.get("confidence", 1.0)),
            )
            for r in json.loads(ext_json or "[]")
        ]
        rows.append((subject_class, refs))
    return rows


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
    # Typed so the row unpacks as (SubjectRow, int): the subquery is ``Any``.
    cnt: ColumnElement[int] = count_sq.c.cnt
    stmt = (
        select(SubjectRow, cnt)
        .join(count_sq, count_sq.c.subject_id == SubjectRow.id)
        .where(cnt < threshold)
    )
    if canonical_only:
        # external_ids_json defaults to "[]"; treat any non-empty-list value
        # as canonical. Avoids a JSON1 extension dependency on SQLite.
        stmt = stmt.where(SubjectRow.external_ids_json != "[]")
    result = await session.execute(stmt)
    return [(row.to_model(), int(count)) for row, count in result]

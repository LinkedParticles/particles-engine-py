# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Reference knowledge-graph ↔ Particles store mapping (§8).

The reference memory server (``@modelcontextprotocol/server-memory`` v0.6.3)
models memory as a flat JSONL knowledge graph of three shapes::

    Entity   { name, entityType, observations[] }
    Relation { from, to, relationType }
    KnowledgeGraph { entities[], relations[] }

This module translates that vocabulary onto the store, and back, without
losing anything in either direction:

* **entity → Subject** — ``name`` is the canonical name, ``entityType`` the
  ``subject_class`` (an unconstrained ``str | None``, so arbitrary reference
  type strings round-trip). Resolution is deliberately **bare-local** (exact
  name, else create): the ``resolve_subject`` authority ladder can rewrite
  ``canonical_name`` via a live Wikidata lookup, which would break the exact
  round-trip the reference contract requires and make a local server
  network-dependent (§3).
* **observation → asserted particle** — one ACTIVE particle per observation
  string, linked to exactly one Subject, tagged :data:`OBSERVATION_TAG`, with
  provenance pointing at the deposit of the verbatim tool payload (§4).
* **relation → multi-subject particle** — one ACTIVE particle linked to two
  Subjects, tagged :data:`RELATION_TAG`. The native link surface cannot carry
  these: ``_parse_relation`` admits only ``CO_EVIDENTIAL`` and
  ``particle_relations`` is particle↔particle, not subject↔subject. Since
  ``particle_subjects`` has no role or ordering column, direction cannot ride
  ``subject_ids``; the triple is carried losslessly in percent-encoded reserved
  tags, with the corpus deposit holding the verbatim payload as the audit
  record (§5).
* **delete → retraction** — nothing is ever hard-deleted. Reads filter to
  ACTIVE, so a client that deletes then reads does not see the item, while the
  history survives (§6). An entity is tombstoned by an ACTIVE particle tagged
  :data:`TOMBSTONE_TAG`, so an entity with no observations still disappears.

Tags are unconstrained ``list[str]``, so the whole encoding rides existing
mechanisms and needs no schema change — which keeps its
``spec_impact: implementation`` classification.

**The tag vocabulary itself is defined Client-side**, in
``particles.extraction.mcp_memory`` — the module that owns the reference
*format* — and imported here (Surface → Client, the allowed direction).
It moved there when inbound migration shipped: a migrated
``memory.jsonl`` record and a record this façade writes must be *the same
record*, so a user who migrates sees their history through these same reads.
Two copies of the encoding kept in sync by hand would eventually diverge, and
the divergence would be invisible until a migrated graph silently failed to
appear. This module owns the store-reading projection on top; it does not own
the vocabulary.
"""

from __future__ import annotations

import logging
from typing import Any

from particles.config import get_config
from particles.core.schema import ProvenanceRefType
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status

# The reference vocabulary is defined Client-side, in the module that owns the
# reference *format*, so a migrated record and a record this façade
# writes are byte-for-byte the same record — one encoding, not two kept in
# sync. Surface → Client is the allowed direction.
from particles.extraction.mcp_memory import (
    IMPORT_TAG,
    OBSERVATION_TAG,
    RELATION_TAG,
    TOMBSTONE_TAG,
    entity_type_of,
    is_observation,
    is_relation,
    is_tombstone,
    relation_content,
    relation_from_particle,
    relation_tags,
)
from particles.extraction.mcp_memory import SOURCE_TYPE as MIGRATION_SOURCE_TYPE

__all__ = [
    "OBSERVATION_TAG",
    "RELATION_TAG",
    "TOMBSTONE_TAG",
    "Subgraph",
    "caps",
    "entity_type_of",
    "is_observation",
    "is_relation",
    "is_tombstone",
    "load_subgraph",
    "relation_content",
    "relation_from_particle",
    "relation_tags",
]

log = logging.getLogger(__name__)


def _facade_record(particle: Any, facade: str, migration_entries: frozenset[str]) -> bool:
    """Whether a tagged particle carries the façade's own provenance (P50).

    The tags are caller-supplied on the native write path, so a tag alone would
    let another principal tombstone, annotate or relate any Subject in the
    façade's view (security review 2026-10-24-2, P50). A record counts when the
    façade asserted it, or, for an observation or relation, when the
    migration imported it: the migration's tag, the ``IMPORTED`` calibration
    source, and a source ref to an export the local migration deposited
    (``migration_entries``, from :func:`_migration_entries`). The tag and the
    calibration alone are not enough, since an interchange bundle or a tag verb
    could once supply both (security review 2026-10-25-2, R26). Only the façade
    writes tombstones. A particle that carries none of the façade's tags is not
    a façade record either way.
    """
    if particle.asserted_by == facade:
        return True
    if is_tombstone(particle):
        return False
    return bool(
        IMPORT_TAG in (particle.tags or ())
        and particle.confidence.calibration_source == CalibrationSource.IMPORTED
        and any(
            ref.type is ProvenanceRefType.SOURCE and ref.corpus_entry_id in migration_entries
            for ref in particle.provenance
        )
    )


def _migration_candidates(particles: list[Any]) -> set[str]:
    """The entries a particle that may be a migrated record cites, for one lookup."""
    return {
        ref.corpus_entry_id
        for particle in particles
        if IMPORT_TAG in (particle.tags or ())
        and particle.confidence.calibration_source == CalibrationSource.IMPORTED
        for ref in particle.provenance
        if ref.type is ProvenanceRefType.SOURCE
    }


async def _migration_entries(session: Any, entry_ids: set[str]) -> frozenset[str]:
    """Which of ``entry_ids`` are memory-server exports the local migration deposited.

    Only ``particles import mcp-memory`` writes an entry of the migration's
    source type: the HTTP deposit routes refuse it (R26), so the type is a
    server-owned mark of a local migration run.
    """
    if not entry_ids:
        return frozenset()
    from sqlalchemy import select

    from particles.corpus.store import CorpusEntryRow

    found: set[str] = set()
    listed = sorted(entry_ids)
    for start in range(0, len(listed), 500):
        rows = await session.execute(
            select(CorpusEntryRow.entry_id).where(
                CorpusEntryRow.entry_id.in_(listed[start : start + 500]),
                CorpusEntryRow.source_type == MIGRATION_SOURCE_TYPE,
            )
        )
        found.update(rows.scalars())
    return frozenset(found)


class Subgraph:
    """An in-memory projection of the façade's view of the store.

    Built once per tool call from three bulk queries (subjects, the
    particle↔subject pairs, the particles themselves) rather than N+1 lookups.
    Everything downstream — ``read_graph``, ``search_nodes``, ``open_nodes``,
    and the write paths' dedup checks — reads this projection.
    """

    def __init__(
        self,
        subjects_by_id: dict[str, Any],
        particles: list[Any],
        pairs: list[tuple[str, str]],
        migration_entries: frozenset[str] = frozenset(),
    ) -> None:
        self._subjects_by_id = subjects_by_id
        self._by_name: dict[str, Any] = {}
        for subject in subjects_by_id.values():
            self._by_name[subject.canonical_name.lower()] = subject

        subject_ids_by_particle: dict[str, list[str]] = {}
        for particle_id, subject_id in pairs:
            subject_ids_by_particle.setdefault(particle_id, []).append(subject_id)

        self.observations_by_subject: dict[str, list[Any]] = {}
        self.relations: list[Any] = []
        self.tombstoned_subject_ids: set[str] = set()
        self._tombstones_by_subject: dict[str, list[Any]] = {}

        facade = get_config().mcp.memory_compat.asserter_identity
        for particle in sorted(particles, key=lambda p: (p.asserted_at, p.id)):
            if particle.status != Status.ACTIVE or not _facade_record(
                particle, facade, migration_entries
            ):
                continue
            linked = subject_ids_by_particle.get(particle.id, [])
            if is_tombstone(particle):
                self.tombstoned_subject_ids.update(linked)
                for subject_id in linked:
                    self._tombstones_by_subject.setdefault(subject_id, []).append(particle)
            elif is_relation(particle):
                self.relations.append(particle)
            elif is_observation(particle) and linked:
                self.observations_by_subject.setdefault(linked[0], []).append(particle)

    # -- lookup ---------------------------------------------------------

    def subject_by_name(self, name: str) -> Any | None:
        """Case-insensitive exact-name lookup, honouring tombstones.

        A tombstoned subject reads as absent — that is what makes
        delete-then-read behave like the reference's hard delete.
        """
        subject = self._by_name.get(name.lower())
        if subject is None or subject.id in self.tombstoned_subject_ids:
            return None
        return subject

    def raw_subject_by_name(self, name: str) -> Any | None:
        """Exact-name lookup **ignoring** tombstones — the revive path needs it."""
        return self._by_name.get(name.lower())

    def is_tombstoned(self, subject_id: str) -> bool:
        return subject_id in self.tombstoned_subject_ids

    def tombstone_particles_for(self, subject_id: str) -> list[Any]:
        """The ACTIVE tombstones on a Subject, retracted when the entity is re-created."""
        return list(self._tombstones_by_subject.get(subject_id, []))

    def live_subjects(self) -> list[Any]:
        """Every non-tombstoned Subject, ordered by canonical name (stable)."""
        return sorted(
            (s for s in self._subjects_by_id.values() if s.id not in self.tombstoned_subject_ids),
            key=lambda s: s.canonical_name,
        )

    def observation_texts(self, subject_id: str) -> list[str]:
        return [p.content for p in self.observations_by_subject.get(subject_id, [])]

    def observation_particles(self, subject_id: str) -> list[Any]:
        return list(self.observations_by_subject.get(subject_id, []))

    def relation_triples(self) -> list[tuple[dict[str, str], Any]]:
        """Every well-formed relation as ``(reference_dict, particle)``."""
        out: list[tuple[dict[str, str], Any]] = []
        for particle in self.relations:
            rel = relation_from_particle(particle)
            if rel is not None:
                out.append((rel, particle))
        return out

    def find_relation(self, from_name: str, to_name: str, relation_type: str) -> Any | None:
        """Exact triple match, the reference's dedup and delete key."""
        for rel, particle in self.relation_triples():
            if (
                rel["from"] == from_name
                and rel["to"] == to_name
                and rel["relationType"] == relation_type
            ):
                return particle
        return None

    # -- projection -----------------------------------------------------

    def entity_dict(
        self, subject: Any, *, observation_limit: int = 0
    ) -> tuple[dict[str, Any], int]:
        """Project one Subject as a reference ``Entity``.

        Returns the entity and the number of observations withheld by the cap,
        so the caller can disclose truncation rather than silently shrink it.
        """
        texts = self.observation_texts(subject.id)
        dropped = 0
        if observation_limit and len(texts) > observation_limit:
            dropped = len(texts) - observation_limit
            texts = texts[:observation_limit]
        return (
            {
                "name": subject.canonical_name,
                "entityType": entity_type_of(subject),
                "observations": texts,
            },
            dropped,
        )

    def project(
        self, subjects: list[Any], *, observation_limit: int = 0
    ) -> tuple[dict[str, list[dict[str, Any]]], int]:
        """Project a Subject set as ``{entities, relations}``.

        Relations follow the reference's current rule: include an edge when
        **at least one** endpoint is in the set, so a caller can discover
        connections to nodes outside the result.
        """
        entities: list[dict[str, Any]] = []
        dropped_observations = 0
        for subject in subjects:
            entity, dropped = self.entity_dict(subject, observation_limit=observation_limit)
            entities.append(entity)
            dropped_observations += dropped

        names = {s.canonical_name for s in subjects}
        relations = [
            rel for rel, _ in self.relation_triples() if rel["from"] in names or rel["to"] in names
        ]
        return {"entities": entities, "relations": relations}, dropped_observations

    # -- search ---------------------------------------------------------

    def search(self, query: str) -> list[Any]:
        """The reference's search: case-insensitive **substring**, not semantic.

        Matches against entity name, entityType, and observation content —
        exactly the three fields ``searchNodes`` scans. Deterministic, offline,
        and keyless, like the server it replaces.
        """
        needle = query.lower()
        hits = []
        for subject in self.live_subjects():
            haystacks = [subject.canonical_name, entity_type_of(subject)]
            haystacks.extend(self.observation_texts(subject.id))
            if any(needle in h.lower() for h in haystacks):
                hits.append(subject)
        return hits


async def load_subgraph(session: Any, *, subject_ids: list[str] | None = None) -> Subgraph:
    """Build a :class:`Subgraph` from the store in three bulk queries.

    ``subject_ids`` narrows the projection to a known set (the ``open_nodes``
    path); ``None`` loads every Subject.
    """
    from particles.store.particle_store import get_particles_by_ids
    from particles.store.subject_store import (
        get_subject,
        list_all_subjects,
        list_particle_subject_pairs,
    )

    if subject_ids is None:
        subjects = await list_all_subjects(session)
    else:
        subjects = []
        for sid in subject_ids:
            subject = await get_subject(session, sid)
            if subject is not None:
                subjects.append(subject)
    subjects_by_id = {s.id: s for s in subjects}

    pairs = await list_particle_subject_pairs(session)
    relevant = [(pid, sid) for pid, sid in pairs if sid in subjects_by_id]
    particles_by_id = await get_particles_by_ids(session, [pid for pid, _ in relevant])
    particles = list(particles_by_id.values())
    migration_entries = await _migration_entries(session, _migration_candidates(particles))
    return Subgraph(subjects_by_id, particles, relevant, migration_entries)


def caps() -> tuple[int, int, int]:
    """The three caps (§7/§8), read at call time."""
    cfg = get_config().mcp.memory_compat
    return (
        cfg.read_graph_max_entities,
        cfg.read_graph_max_observations_per_entity,
        cfg.search_max_entities,
    )

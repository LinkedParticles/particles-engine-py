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
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote, unquote

from particles.config import get_config
from particles.core.status import Status

log = logging.getLogger(__name__)

#: Marks a particle as a façade *observation* (one Subject, one claim).
OBSERVATION_TAG = "memory-compat:observation"
#: Marks a particle as a façade *relation* (two Subjects, a directed edge).
RELATION_TAG = "memory-compat:relation"
#: Marks a Subject as deleted through the façade (see module docstring).
TOMBSTONE_TAG = "memory-compat:deleted"

_REL_TYPE_PREFIX = "memory-compat:rel="
_REL_FROM_PREFIX = "memory-compat:from="
_REL_TO_PREFIX = "memory-compat:to="


def _enc(value: str) -> str:
    """Percent-encode a reference string so it survives as a single tag token."""
    return quote(value, safe="")


def _dec(value: str) -> str:
    """Inverse of :func:`_enc`."""
    return unquote(value)


def _tag_value(tags: list[str] | None, prefix: str) -> str | None:
    for tag in tags or ():
        if tag.startswith(prefix):
            return _dec(tag[len(prefix) :])
    return None


def relation_tags(from_name: str, to_name: str, relation_type: str) -> list[str]:
    """Build the reserved tag set that carries a relation triple losslessly."""
    return [
        RELATION_TAG,
        f"{_REL_TYPE_PREFIX}{_enc(relation_type)}",
        f"{_REL_FROM_PREFIX}{_enc(from_name)}",
        f"{_REL_TO_PREFIX}{_enc(to_name)}",
    ]


def relation_from_particle(particle: Any) -> dict[str, str] | None:
    """Recover a reference relation dict from a façade relation particle.

    Returns ``None`` when the particle is not a well-formed façade relation
    (a tag was dropped or the particle predates the encoding), so a damaged
    record degrades to omission rather than a malformed edge.
    """
    tags = particle.tags
    rel = _tag_value(tags, _REL_TYPE_PREFIX)
    src = _tag_value(tags, _REL_FROM_PREFIX)
    dst = _tag_value(tags, _REL_TO_PREFIX)
    if rel is None or src is None or dst is None:
        log.warning("Skipping malformed façade relation particle %s", particle.id)
        return None
    return {"from": src, "to": dst, "relationType": rel}


def relation_content(from_name: str, to_name: str, relation_type: str) -> str:
    """The particle's human-readable content — the active voice the reference asks for."""
    return f"{from_name} {relation_type} {to_name}"


def is_observation(particle: Any) -> bool:
    return OBSERVATION_TAG in (particle.tags or ())


def is_relation(particle: Any) -> bool:
    return RELATION_TAG in (particle.tags or ())


def is_tombstone(particle: Any) -> bool:
    return TOMBSTONE_TAG in (particle.tags or ())


def entity_type_of(subject: Any) -> str:
    """Reference entities always carry a string ``entityType``; ours may be None."""
    return subject.subject_class or ""


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

        for particle in sorted(particles, key=lambda p: (p.asserted_at, p.id)):
            if particle.status != Status.ACTIVE:
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
    return Subgraph(subjects_by_id, list(particles_by_id.values()), relevant)


def caps() -> tuple[int, int, int]:
    """The three caps (§7/§8), read at call time."""
    cfg = get_config().mcp.memory_compat
    return (
        cfg.read_graph_max_entities,
        cfg.read_graph_max_observations_per_entity,
        cfg.search_max_entities,
    )

"""The nine reference memory-server operations, backed by the store.

Each function mirrors one ``KnowledgeGraphManager`` method from
``@modelcontextprotocol/server-memory`` v0.6.3, including the parts that are
easy to get subtly wrong and that the reference's own 42-case suite pins:

* ``create_entities`` returns **only** the newly created entities and silently
  skips a name that already exists;
* ``create_relations`` dedups on the exact ``(from, to, relationType)`` triple;
* ``add_observations`` **raises** ``Entity with name X not found``;
* every ``delete_*`` is silent on an absent target and returns a fixed
  ``{success, message}`` payload.

Writes go through the agent-write path — deposit for provenance, then
assert through the §6.6 ladder — under the façade's own asserting principal, so
a façade claim is attributable and trust-weighted below an operator's. Nothing
is ever hard-deleted (§6).

Callers own the transaction: these functions never commit.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from particles.config import get_config
from particles.mcp.memory_compat.graph import (
    OBSERVATION_TAG,
    TOMBSTONE_TAG,
    Subgraph,
    caps,
    entity_type_of,
    load_subgraph,
    relation_content,
    relation_tags,
)

log = logging.getLogger(__name__)

DELETE_ENTITIES_MESSAGE = "Entities deleted successfully"
DELETE_OBSERVATIONS_MESSAGE = "Observations deleted successfully"
DELETE_RELATIONS_MESSAGE = "Relations deleted successfully"

_RETRACT_REASON = "Deleted through the reference memory-server compatibility façade."


def _cfg() -> Any:
    return get_config().mcp.memory_compat


def _granularity() -> tuple[int, int]:
    """Façade granularity limits: a char ceiling only, sentence check disabled.

    A reference observation has no length limit and ``read_graph`` must return
    the exact string, so we never truncate or split.
    """
    return (_cfg().max_observation_chars, 0)


async def _deposit(session: Any, tool: str, payload: Any) -> str:
    """Deposit the verbatim tool payload; return the corpus entry id.

    One deposit per tool invocation — the auditable record of exactly what the
    agent claimed, and the provenance anchor every particle from this call
    points at (the deposit-excerpt pattern).
    """
    from particles.operations.agent_write import deposit_conversation_text

    text = json.dumps({"tool": tool, "payload": payload}, indent=2, sort_keys=True)
    entry_id, _snapshot_id = await deposit_conversation_text(
        session,
        text=text,
        tags=["memory-compat", f"memory-compat:tool={tool}"],
        identity=_cfg().asserter_identity,
    )
    return entry_id


async def _assert(
    session: Any,
    *,
    store: str,
    content: str,
    subject_ids: list[str],
    corpus_entry_id: str,
    tags: list[str],
) -> None:
    from particles.operations.agent_write import assert_belief

    await assert_belief(
        session,
        store=store,
        content=content,
        subject_names=[],
        subject_ids=subject_ids,
        confidence=_cfg().asserted_confidence,
        corpus_entry_id=corpus_entry_id,
        tags=tags,
        identity=_cfg().asserter_identity,
        granularity=_granularity(),
    )


async def _retract(session: Any, *, store: str, particle_id: str) -> None:
    from particles.operations.agent_write import retract_belief

    await retract_belief(
        session,
        store=store,
        particle_id=particle_id,
        reason=_RETRACT_REASON,
        identity=_cfg().asserter_identity,
    )


async def _create_subject(session: Any, name: str, entity_type: str) -> Any:
    """Create a bare-local Subject — deliberately not the authority ladder.

    ``resolve_subject`` can rewrite ``canonical_name`` through a live Wikidata
    lookup, which would break the reference's exact name round-trip and make a
    local server network-dependent.
    """
    from particles.core.schema import Subject
    from particles.store.subject_store import insert_subject

    subject = Subject(
        canonical_name=name,
        asserted_by=_cfg().asserter_identity,
        subject_class=entity_type or None,
    )
    await insert_subject(session, subject)
    return subject


async def _ensure_subject(session: Any, sub: Subgraph, name: str, entity_type: str = "") -> Any:
    """Resolve a name to a live Subject, reviving a tombstone or creating one."""
    existing = sub.raw_subject_by_name(name)
    if existing is None:
        return await _create_subject(session, name, entity_type)
    return existing


# -- write operations ---------------------------------------------------


async def create_entities(
    session: Any, *, store: str, entities: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Create entities, returning only the newly created ones (reference semantics)."""
    sub = await load_subgraph(session)
    pending = [e for e in entities if sub.subject_by_name(str(e.get("name", ""))) is None]
    if not pending:
        return []

    entry_id = await _deposit(session, "create_entities", entities)
    created: list[dict[str, Any]] = []

    for entity in pending:
        name = str(entity.get("name", ""))
        entity_type = str(entity.get("entityType", "") or "")
        observations = [str(o) for o in entity.get("observations", []) or []]

        subject = sub.raw_subject_by_name(name)
        if subject is None:
            subject = await _create_subject(session, name, entity_type)
        else:
            # Re-creating a deleted entity: retract its tombstones so it reads
            # live again, and refresh its class from the new payload.
            for tombstone in sub.tombstone_particles_for(subject.id):
                await _retract(session, store=store, particle_id=tombstone.id)
            if entity_type and subject.subject_class != entity_type:
                from particles.store.subject_store import set_subject_class

                await set_subject_class(session, subject.id, entity_type)

        for text in observations:
            await _assert(
                session,
                store=store,
                content=text,
                subject_ids=[subject.id],
                corpus_entry_id=entry_id,
                tags=[OBSERVATION_TAG],
            )

        created.append({"name": name, "entityType": entity_type, "observations": observations})

    return created


async def create_relations(
    session: Any, *, store: str, relations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Create relations, deduped on the exact triple, returning only the new ones."""
    sub = await load_subgraph(session)
    pending: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for relation in relations:
        src = str(relation.get("from", ""))
        dst = str(relation.get("to", ""))
        kind = str(relation.get("relationType", ""))
        key = (src, dst, kind)
        if key in seen or sub.find_relation(src, dst, kind) is not None:
            continue
        seen.add(key)
        pending.append({"from": src, "to": dst, "relationType": kind})

    if not pending:
        return []

    entry_id = await _deposit(session, "create_relations", relations)
    for relation in pending:
        src, dst, kind = relation["from"], relation["to"], relation["relationType"]
        # The reference never checks that endpoints exist; we must materialise
        # them because a particle links to Subjects, not to bare strings.
        from_subject = await _ensure_subject(session, sub, src)
        to_subject = await _ensure_subject(session, sub, dst)
        await _assert(
            session,
            store=store,
            content=relation_content(src, dst, kind),
            subject_ids=[from_subject.id, to_subject.id],
            corpus_entry_id=entry_id,
            tags=relation_tags(src, dst, kind),
        )
    return pending


async def add_observations(
    session: Any, *, store: str, observations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Add observations to existing entities.

    Raises:
        ValueError: If a named entity does not exist — the reference throws
            ``Entity with name X not found`` and clients depend on that.
    """
    sub = await load_subgraph(session)

    targets: list[tuple[Any, list[str]]] = []
    for item in observations:
        name = str(item.get("entityName", ""))
        subject = sub.subject_by_name(name)
        if subject is None:
            raise ValueError(f"Entity with name {name} not found")
        existing = set(sub.observation_texts(subject.id))
        fresh: list[str] = []
        for content in item.get("contents", []) or []:
            text = str(content)
            if text not in existing and text not in fresh:
                fresh.append(text)
        targets.append((subject, fresh))

    entry_id: str | None = None
    if any(fresh for _, fresh in targets):
        entry_id = await _deposit(session, "add_observations", observations)

    results: list[dict[str, Any]] = []
    for (subject, fresh), item in zip(targets, observations, strict=True):
        for text in fresh:
            assert entry_id is not None
            await _assert(
                session,
                store=store,
                content=text,
                subject_ids=[subject.id],
                corpus_entry_id=entry_id,
                tags=[OBSERVATION_TAG],
            )
        results.append({"entityName": str(item.get("entityName", "")), "addedObservations": fresh})
    return results


async def delete_entities(session: Any, *, store: str, entity_names: list[str]) -> None:
    """Retract entities and everything attached to them. Silent on absent names."""
    sub = await load_subgraph(session)
    doomed = [
        subject
        for subject in (sub.subject_by_name(str(n)) for n in entity_names)
        if subject is not None
    ]
    if not doomed:
        return

    entry_id = await _deposit(session, "delete_entities", entity_names)
    doomed_ids = {s.id for s in doomed}
    doomed_names = {s.canonical_name for s in doomed}

    for subject in doomed:
        for particle in sub.observation_particles(subject.id):
            await _retract(session, store=store, particle_id=particle.id)

    # Cascade: the reference drops every relation with EITHER endpoint deleted.
    for rel, particle in sub.relation_triples():
        if rel["from"] in doomed_names or rel["to"] in doomed_names:
            await _retract(session, store=store, particle_id=particle.id)

    # Tombstone the Subject itself, so an entity with no observations still
    # disappears from reads while the record survives.
    for subject_id in doomed_ids:
        await _assert(
            session,
            store=store,
            content=f"Entity deleted through the memory-compat façade: {subject_id}",
            subject_ids=[subject_id],
            corpus_entry_id=entry_id,
            tags=[TOMBSTONE_TAG],
        )


async def delete_observations(session: Any, *, store: str, deletions: list[dict[str, Any]]) -> None:
    """Retract specific observations. Silent on an absent entity or observation."""
    sub = await load_subgraph(session)
    for item in deletions:
        subject = sub.subject_by_name(str(item.get("entityName", "")))
        if subject is None:
            continue
        unwanted = {str(o) for o in item.get("observations", []) or []}
        for particle in sub.observation_particles(subject.id):
            if particle.content in unwanted:
                await _retract(session, store=store, particle_id=particle.id)


async def delete_relations(session: Any, *, store: str, relations: list[dict[str, Any]]) -> None:
    """Retract relations matching the exact triple. Silent on absent."""
    sub = await load_subgraph(session)
    for relation in relations:
        particle = sub.find_relation(
            str(relation.get("from", "")),
            str(relation.get("to", "")),
            str(relation.get("relationType", "")),
        )
        if particle is not None:
            await _retract(session, store=store, particle_id=particle.id)


# -- read operations ----------------------------------------------------


async def read_graph(session: Any) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """The whole graph, capped. Returns ``(graph, truncation_notes)``.

    The reference dumps everything; on a real store that is megabytes of JSON
    into the model's context, so entities and per-entity observations are
    capped. Truncation is always reported back to the caller for disclosure —
    a capped graph that claims to be complete is a correctness bug (§7).
    """
    sub = await load_subgraph(session)
    max_entities, max_observations, _ = caps()

    subjects = sub.live_subjects()
    notes: list[str] = []
    if max_entities and len(subjects) > max_entities:
        notes.append(
            f"read_graph truncated: {len(subjects)} entities in the store, "
            f"{max_entities} returned (mcp.memory_compat.read_graph_max_entities)."
        )
        subjects = subjects[:max_entities]

    graph, dropped = sub.project(subjects, observation_limit=max_observations)
    if dropped:
        notes.append(
            f"read_graph truncated: {dropped} observation(s) withheld by the "
            f"{max_observations}-per-entity cap "
            f"(mcp.memory_compat.read_graph_max_observations_per_entity)."
        )
    return graph, notes


async def search_nodes(
    session: Any, *, query: str
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Case-insensitive substring search over name, entityType, and observations."""
    sub = await load_subgraph(session)
    _, max_observations, max_entities = caps()

    hits = sub.search(query)
    notes: list[str] = []
    if max_entities and len(hits) > max_entities:
        notes.append(
            f"search_nodes truncated: {len(hits)} entities matched, "
            f"{max_entities} returned (mcp.memory_compat.search_max_entities)."
        )
        hits = hits[:max_entities]

    graph, dropped = sub.project(hits, observation_limit=max_observations)
    if dropped:
        notes.append(
            f"search_nodes truncated: {dropped} observation(s) withheld by the "
            f"{max_observations}-per-entity cap."
        )
    return graph, notes


async def open_nodes(
    session: Any, *, names: list[str]
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Open entities by exact name, with every relation touching them."""
    sub = await load_subgraph(session)
    _, max_observations, _ = caps()

    subjects = []
    for name in names:
        subject = sub.subject_by_name(str(name))
        if subject is not None and subject not in subjects:
            subjects.append(subject)

    graph, dropped = sub.project(subjects, observation_limit=max_observations)
    notes: list[str] = []
    if dropped:
        notes.append(
            f"open_nodes truncated: {dropped} observation(s) withheld by the "
            f"{max_observations}-per-entity cap."
        )
    return graph, notes


def entity_type_for(subject: Any) -> str:
    """Re-exported for the server module's convenience."""
    return entity_type_of(subject)

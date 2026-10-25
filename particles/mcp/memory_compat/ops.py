# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The nine reference memory-server operations, backed by the store.

Each function mirrors one ``KnowledgeGraphManager`` method from
``@modelcontextprotocol/server-memory`` (re-verified against the reference's
``main`` on 2026-09-20), including the parts that are easy to get subtly wrong
and that the reference's own suite pins:

* ``create_entities`` returns **only** the newly created entities and silently
  skips a name that already exists — in the store *or earlier in the same
  batch*;
* ``create_relations`` **raises** ``Entity with name X not found`` when either
  endpoint is missing, before anything is written, then dedups on the exact
  ``(from, to, relationType)`` triple;
* ``add_observations`` **raises** ``Entity with name X not found``;
* every ``delete_*`` tolerates an absent target but **reports** it: the
  ``{success, message}`` payload carries the fixed success sentence only when
  everything requested was actually deleted, and a ``Deleted N of M …`` count
  otherwise. The counting arithmetic is the reference's, quirks included.

One behaviour has no reference counterpart, because the reference cannot hold a
write back: **a write the store holds for review is not reported as written.**
A delete here is a retraction, a judgment retirement, so a client that later
re-adds the identical observation (or re-creates the identical relation) is
re-asserting a claim the store has retired. The §6.6 retired-value rung holds
that claim behind an INCONSISTENCY record instead of reviving it,
and a claim that contradicts a held belief is held the same way. A held item
never appears in ``read_graph``, ``search_nodes`` or ``open_nodes``, so the
three writing operations that return what they wrote (``create_entities``,
``create_relations``, ``add_observations``) list only what landed ACTIVE, and
return a note per held item, naming the INCONSISTENCY record an operator
settles with ``particles review``. The server appends each note as an extra
text block after the reference JSON, exactly as it discloses a read cap, so
the response shape a client parses is unchanged.

Writes go through the agent-write path — deposit for provenance, then
assert through the §6.6 ladder — under the façade's own asserting principal, so
a façade claim is attributable and trust-weighted below an operator's. Nothing
is ever hard-deleted (§6).

Callers own the transaction: these functions never commit.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from particles.config import get_config
from particles.core.status import Status
from particles.mcp.memory_compat.graph import (
    OBSERVATION_TAG,
    TOMBSTONE_TAG,
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


def delete_entities_message(deleted: list[str], not_found: list[str]) -> str:
    """The reference's ``delete_entities`` sentence, partial deletes reported.

    Counts are per *requested name*, so a name listed twice counts twice on
    both sides of the "N of M" — the reference's own arithmetic.
    """
    if not not_found:
        return DELETE_ENTITIES_MESSAGE
    requested = len(deleted) + len(not_found)
    return f"Deleted {len(deleted)} of {requested} entities. Not found: {', '.join(not_found)}"


def delete_observations_message(
    deleted_count: int, requested: int, missing_entities: list[str]
) -> str:
    """The reference's ``delete_observations`` sentence, partial deletes reported."""
    if deleted_count == requested:
        return DELETE_OBSERVATIONS_MESSAGE
    message = f"Deleted {deleted_count} of {requested} observations."
    if missing_entities:
        message += f" Entities not found: {', '.join(missing_entities)}"
    return message


def delete_relations_message(deleted_count: int, requested: int) -> str:
    """The reference's ``delete_relations`` sentence, partial deletes reported."""
    if deleted_count == requested:
        return DELETE_RELATIONS_MESSAGE
    return f"Deleted {deleted_count} of {requested} relations. The rest matched nothing."


_RETRACT_REASON = "Deleted through the reference memory-server compatibility façade."


def _cfg() -> Any:
    return get_config().mcp.memory_compat


def _owns(subject: Any) -> bool:
    """Whether the façade created ``subject`` (security review 2026-10-24, N30).

    The façade's view is every Subject in the store, so without this check it
    could reclassify or tombstone a Subject an extractor or the operator
    minted. It may change only its own.
    """
    return bool(subject.asserted_by == _cfg().asserter_identity)


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
) -> Any:
    """Assert one façade claim; return the ``AgentWriteResult`` verdict."""
    from particles.operations.agent_write import assert_belief

    return await assert_belief(
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


def _landed(result: Any) -> bool:
    """Whether an assertion landed ACTIVE, the only state the façade reads back."""
    return bool(result.status == Status.ACTIVE.value)


def held_note(tool: str, noun: str, held: list[tuple[str, Any]]) -> str:
    """Disclose writes the store held for review rather than wrote.

    ``held`` pairs a client-facing label for each item with its
    ``AgentWriteResult``. The note names the INCONSISTENCY record where there
    is one, so an operator can find it in ``particles review``.
    """
    items = []
    for label, result in held:
        if result.inconsistency_id:
            items.append(f"{label} (INCONSISTENCY {result.inconsistency_id})")
        elif result.asserted_particle_id:
            items.append(f"{label} (held as {result.asserted_particle_id})")
        else:
            items.append(f"{label} (not stored: {result.verdict})")
    return (
        f"{tool}: {len(held)} {noun}(s) held for review, not added, so they are left out "
        "of the result above and will not appear in reads. The store already retired "
        "an identical claim (an earlier delete) or holds one it conflicts with; an "
        f"operator settles it with `particles review`. Held: {'; '.join(items)}."
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


# -- write operations ---------------------------------------------------


async def create_entities(
    session: Any, *, store: str, entities: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Create entities, returning only the newly created ones (reference semantics).

    Returns ``(created, notes)``. An observation the store held for review
    (re-creating a deleted entity with its old observations does exactly this)
    is left out of its entity's ``observations`` and disclosed in ``notes``;
    see the module docstring.
    """
    sub = await load_subgraph(session)
    pending: list[dict[str, Any]] = []
    claimed: set[str] = set()
    for entity in entities:
        name = str(entity.get("name", ""))
        # The reference also skips a name repeated earlier in the same batch.
        # The key is the façade's own name identity (``Subgraph`` matches names
        # case-insensitively), so one batch can never mint two Subjects that
        # every later lookup would treat as the same entity.
        key = name.lower()
        if key in claimed or sub.subject_by_name(name) is not None:
            continue
        claimed.add(key)
        pending.append(entity)
    if not pending:
        return [], []

    entry_id = await _deposit(session, "create_entities", entities)
    created: list[dict[str, Any]] = []
    held: list[tuple[str, Any]] = []

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
            # Only a Subject the façade created takes the payload's class: the
            # setter records no event, and another principal's classification
            # is not the façade's to change (N30).
            if entity_type and subject.subject_class != entity_type and _owns(subject):
                from particles.store.subject_store import set_subject_class

                await set_subject_class(session, subject.id, entity_type)

        landed: list[str] = []
        for text in observations:
            result = await _assert(
                session,
                store=store,
                content=text,
                subject_ids=[subject.id],
                corpus_entry_id=entry_id,
                tags=[OBSERVATION_TAG],
            )
            if _landed(result):
                landed.append(text)
            else:
                held.append((f"{name}: {json.dumps(text)}", result))

        created.append({"name": name, "entityType": entity_type, "observations": landed})

    notes = [held_note("create_entities", "observation", held)] if held else []
    return created, notes


async def create_relations(
    session: Any, *, store: str, relations: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Create relations, deduped on the exact triple, returning only the new ones.

    Returns ``(created, notes)``. A relation the store held for review
    (re-creating a deleted relation does exactly this) is left out of
    ``created`` and disclosed in ``notes``; see the module docstring.

    Raises:
        ValueError: If either endpoint of any relation is not an existing
            entity — the reference throws ``Entity with name X not found``
            for the whole batch before writing anything, ``from`` checked
            before ``to``. A deleted (tombstoned) entity counts as absent.
    """
    sub = await load_subgraph(session)
    endpoints: dict[str, Any] = {}
    for relation in relations:
        for endpoint in (str(relation.get("from", "")), str(relation.get("to", ""))):
            subject = sub.subject_by_name(endpoint)
            if subject is None:
                raise ValueError(f"Entity with name {endpoint} not found")
            endpoints[endpoint] = subject

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
        return [], []

    entry_id = await _deposit(session, "create_relations", relations)
    created: list[dict[str, Any]] = []
    held: list[tuple[str, Any]] = []
    for relation in pending:
        src, dst, kind = relation["from"], relation["to"], relation["relationType"]
        result = await _assert(
            session,
            store=store,
            content=relation_content(src, dst, kind),
            subject_ids=[endpoints[src].id, endpoints[dst].id],
            corpus_entry_id=entry_id,
            tags=relation_tags(src, dst, kind),
        )
        if _landed(result):
            created.append(relation)
        else:
            held.append((f"{src} {kind} {dst}", result))
    notes = [held_note("create_relations", "relation", held)] if held else []
    return created, notes


async def add_observations(
    session: Any, *, store: str, observations: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Add observations to existing entities.

    Returns ``(results, notes)``. ``addedObservations`` lists only the
    observations that landed ACTIVE, the ones a read will return. An
    observation the store held for review is left out of it and disclosed in
    ``notes``: re-adding an observation deleted earlier is the common case,
    since the delete retired that exact claim (see the module docstring).

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
    held: list[tuple[str, Any]] = []
    for (subject, fresh), item in zip(targets, observations, strict=True):
        name = str(item.get("entityName", ""))
        added: list[str] = []
        for text in fresh:
            assert entry_id is not None
            result = await _assert(
                session,
                store=store,
                content=text,
                subject_ids=[subject.id],
                corpus_entry_id=entry_id,
                tags=[OBSERVATION_TAG],
            )
            if _landed(result):
                added.append(text)
            else:
                held.append((f"{name}: {json.dumps(text)}", result))
        results.append({"entityName": name, "addedObservations": added})
    notes = [held_note("add_observations", "observation", held)] if held else []
    return results, notes


async def delete_entities(
    session: Any, *, store: str, entity_names: list[str]
) -> tuple[list[str], list[str]]:
    """Retract entities and everything attached to them.

    An absent name is not an error, but it is reported: returns
    ``(deleted, not_found)``, each in request order, for
    :func:`delete_entities_message`.

    Raises ``ValueError``, before anything is written, when a named entity is a
    Subject the façade did not create (N30): tombstoning it would hide another
    principal's Subject from every façade read.
    """
    sub = await load_subgraph(session)
    deleted: list[str] = []
    not_found: list[str] = []
    doomed: list[Any] = []
    foreign: list[str] = []
    for raw in entity_names:
        name = str(raw)
        subject = sub.subject_by_name(name)
        if subject is None:
            not_found.append(name)
            continue
        if not _owns(subject):
            foreign.append(name)
            continue
        deleted.append(name)
        if subject not in doomed:
            doomed.append(subject)
    if foreign:
        raise ValueError(
            "Not deleted: these entities were not created through the memory-compat "
            f"façade and can be changed only by an operator: {', '.join(foreign)}"
        )
    if not doomed:
        return deleted, not_found

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
    # disappears from reads while the record survives. Each
    # deletion's tombstone is a distinct claim: re-creating the entity retracts
    # the old one, and an identical second tombstone would re-assert that
    # retired claim and be held for review, leaving the entity live
    # after a delete that reported success.
    for subject_id in doomed_ids:
        result = await _assert(
            session,
            store=store,
            content=(
                f"Entity deleted through the memory-compat façade: {subject_id} "
                f"(deletion {uuid.uuid4().hex})"
            ),
            subject_ids=[subject_id],
            corpus_entry_id=entry_id,
            tags=[TOMBSTONE_TAG],
        )
        if not _landed(result):
            log.warning("memory-compat tombstone for %s did not land: %s", subject_id, result)
    return deleted, not_found


async def delete_observations(
    session: Any, *, store: str, deletions: list[dict[str, Any]]
) -> tuple[int, int, list[str]]:
    """Retract specific observations.

    An absent entity or observation is not an error, but it is reported:
    returns ``(deleted_count, requested, missing_entities)`` for
    :func:`delete_observations_message`. ``requested`` is the reference's sum
    of every listed string, repeats included; ``deleted_count`` is what was
    actually retracted, so a string listed twice counts once.
    """
    sub = await load_subgraph(session)
    deleted_count = 0
    requested = 0
    missing_entities: list[str] = []
    retracted: set[str] = set()
    for item in deletions:
        name = str(item.get("entityName", ""))
        unwanted = [str(o) for o in item.get("observations", []) or []]
        requested += len(unwanted)
        subject = sub.subject_by_name(name)
        if subject is None:
            missing_entities.append(name)
            continue
        for particle in sub.observation_particles(subject.id):
            if particle.content in unwanted and particle.id not in retracted:
                await _retract(session, store=store, particle_id=particle.id)
                retracted.add(particle.id)
                deleted_count += 1
    return deleted_count, requested, missing_entities


async def delete_relations(
    session: Any, *, store: str, relations: list[dict[str, Any]]
) -> tuple[int, int]:
    """Retract relations matching the exact triple.

    An absent relation is not an error, but it is reported: returns
    ``(deleted_count, requested)`` for :func:`delete_relations_message`. A
    triple listed twice is requested twice and deleted once, as in the
    reference.
    """
    sub = await load_subgraph(session)
    retracted: set[str] = set()
    for relation in relations:
        particle = sub.find_relation(
            str(relation.get("from", "")),
            str(relation.get("to", "")),
            str(relation.get("relationType", "")),
        )
        if particle is not None and particle.id not in retracted:
            await _retract(session, store=store, particle_id=particle.id)
            retracted.add(particle.id)
    return len(retracted), len(relations)


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

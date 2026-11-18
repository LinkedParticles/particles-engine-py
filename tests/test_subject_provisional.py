# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Provisional Subjects: the store, the resolver and the operator acts.

A Subject an agent principal mints for a name the store does not know answers
that principal's later resolutions and no extraction, import or operator
write, until an operator accepts it by id or an independent resolution reaches
its external reference. A second agent principal's use shares it among agents.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.provisional import ResolutionKind
from particles.core.schema import ExternalRef, SourceType, Subject
from particles.db import DEFAULT_STORE, session_scope
from particles.ingest.authorities import AuthorityResolution
from particles.ingest.subject_resolver import (
    RESOLVER_ALIAS_ACTOR,
    find_existing_subject,
    fold_or_hold,
    prefetch_subjects,
    resolve_subject,
)
from particles.operations.subject_provisional import (
    accept_subject,
    backfill,
    list_provisional,
    operator_touched,
)
from particles.store import subject_cache
from particles.store.event_store import OperatorEventType, list_events
from particles.store.particle_store import get_particle
from particles.store.subject_provisional import (
    SubjectProvisionalRow,
    SubjectView,
    mark_of,
    mark_provisional,
    with_mark,
)
from particles.store.subject_store import (
    ParticleSubjectRow,
    add_aliases,
    find_by_name,
    get_subject,
    insert_subject,
    lookup_subject,
)

P = "agent-p"
Q = "agent-q"
MINT_P: dict[str, Any] = {"principal": P, "kind": ResolutionKind.MINTING}
MINT_Q: dict[str, Any] = {"principal": Q, "kind": ResolutionKind.MINTING}
EXTRACT: dict[str, Any] = {"kind": ResolutionKind.NON_MINTING}
OPERATOR: dict[str, Any] = {"kind": ResolutionKind.OPERATOR}
_REF = ExternalRef(namespace="fake", id="F1", uri=None, confidence=0.95)


class _FakeAuthority:
    """A live authority that records each lookup and answers a fixed resolution."""

    NAMESPACE = "fake"
    PRIORITY = 0
    LIVE = True
    DEFAULT_LINK_CONFIDENCE = 0.95
    APPLICABILITY: list[Any] = []

    def __init__(self, answer: AuthorityResolution | None = None) -> None:
        self.answer = answer
        self.asked: list[str] = []

    def uri_for(self, external_id: str) -> str | None:
        return None

    def recognize(self, name: str) -> ExternalRef | None:
        return None

    async def resolve(
        self, session: AsyncSession, name: str, *, particle_content: str | None, domain: str | None
    ) -> AuthorityResolution | None:
        self.asked.append(name)
        return self.answer

    async def canonical_name_for(self, session: AsyncSession, external_id: str) -> str | None:
        return None


async def _resolve(
    session: AsyncSession,
    name: str,
    *,
    auth: _FakeAuthority | None = None,
    source_type: str | None = SourceType.CONVERSATION,
    **kind: Any,
) -> Subject:
    authority = auth or _FakeAuthority()
    with patch("particles.ingest.subject_resolver.get_authorities", return_value=[authority]):
        return await resolve_subject(session, name, "tester", source_type=source_type, **kind)


async def _events(session: AsyncSession, event_type: OperatorEventType) -> list[Any]:
    return await list_events(session, event_type=event_type)


# --- the mint -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_minting_resolution_mints_provisional_to_its_principal(
    db_session: AsyncSession,
) -> None:
    minted = await _resolve(db_session, "Prometheus", **MINT_P)

    mark = await mark_of(db_session, minted.id)
    assert mark is not None and mark.principal == P and mark.shared_by is None
    # The agent's own later mentions reach it (acceptance criterion 3), warm and cold.
    assert (await _resolve(db_session, "prometheus", **MINT_P)).id == minted.id
    subject_cache.clear()
    assert (await _resolve(db_session, "PROMETHEUS", **MINT_P)).id == minted.id


@pytest.mark.asyncio
async def test_a_non_minting_resolution_writes_no_provisional_row(
    db_session: AsyncSession,
) -> None:
    """Acceptance criterion 2: a store with no agent writes is unchanged."""
    first = await _resolve(db_session, "Grafana", **EXTRACT)
    again = await _resolve(db_session, "grafana", **OPERATOR)
    assert again.id == first.id
    rows = (await db_session.execute(select(SubjectProvisionalRow))).scalars().all()
    assert rows == []


@pytest.mark.parametrize("other", [EXTRACT, OPERATOR], ids=["extraction", "operator"])
@pytest.mark.asyncio
async def test_extraction_and_operators_never_resolve_to_a_provisional_subject(
    db_session: AsyncSession, other: dict[str, Any]
) -> None:
    agents = await _resolve(db_session, "the server", **MINT_P)
    # The cached agent answer is not handed to anyone else.
    theirs = await _resolve(db_session, "the server", source_type=SourceType.WEB_PAGE, **other)
    assert theirs.id != agents.id
    assert await mark_of(db_session, theirs.id) is None


@pytest.mark.asyncio
async def test_the_capture_is_gone(db_session: AsyncSession) -> None:
    """Acceptance criterion 1, the security review's Q3 reproduction re-run.

    An agent asserts a claim about a name the store does not know. A later
    extraction of a document naming it reaches the live authorities, and it
    and an operator assertion each resolve to a Subject other than the agent's.
    """
    from particles.mcp.tools.write import particle_assert

    write = get_config().mcp.write
    write.enabled_stores = [DEFAULT_STORE]
    write.asserter_identity = P
    out = await particle_assert(
        "Mercury is used in thermometers.",
        ["Mercury"],
        0.8,
        source_excerpt="We talked about thermometers.",
    )
    async with session_scope() as s:
        particle = await get_particle(s, out["asserted_particle_id"])
        assert particle is not None
        agents_id = particle.subject_ids[0]
        agents_mark = await mark_of(s, agents_id)
    assert agents_mark is not None and agents_mark.principal == P

    subject_cache.clear()
    auth = _FakeAuthority(None)
    document = await _resolve(
        db_session, "Mercury", auth=auth, source_type=SourceType.WEB_PAGE, **EXTRACT
    )
    assert auth.asked == ["Mercury"]  # the live pass ran
    operators = await _resolve(db_session, "Mercury", **OPERATOR)
    assert agents_id not in {document.id, operators.id}


# --- sharing --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_second_principal_shares_but_extraction_still_cannot_see_it(
    db_session: AsyncSession,
) -> None:
    minted = await _resolve(db_session, "config.py", **MINT_P)
    shared = await _resolve(db_session, "config.py", **MINT_Q)

    assert shared.id == minted.id
    mark = await mark_of(db_session, minted.id)
    assert mark is not None and mark.shared_by == Q
    [event] = await _events(db_session, OperatorEventType.SUBJECT_SHARED)
    assert event.payload["principal"] == P and event.payload["shared_by"] == Q
    assert (await _resolve(db_session, "config.py", **EXTRACT)).id != minted.id


@pytest.mark.asyncio
async def test_two_keys_of_one_principal_are_one_principal(db_session: AsyncSession) -> None:
    minted = await _resolve(db_session, "the queue", **MINT_P)
    subject_cache.clear()
    await _resolve(db_session, "the queue", **MINT_P)
    mark = await mark_of(db_session, minted.id)
    assert mark is not None and mark.shared_by is None
    assert await _events(db_session, OperatorEventType.SUBJECT_SHARED) == []


# --- confirmation, fold and hold --------------------------------------------------


@pytest.mark.asyncio
async def test_an_independent_resolution_reaching_the_ref_confirms_it(
    db_session: AsyncSession,
) -> None:
    # The agent's deposit led the live pass to a referent.
    auth = _FakeAuthority(AuthorityResolution(external_ref=_REF, canonical_name="Mercury"))
    agents = await _resolve(
        db_session, "Mercury", auth=auth, source_type=SourceType.WEB_PAGE, **MINT_P
    )
    assert await mark_of(db_session, agents.id) is not None

    # A document reaches the same referent on its own.
    subject_cache.clear()
    auth = _FakeAuthority(AuthorityResolution(external_ref=_REF, canonical_name="Mercury"))
    document = await _resolve(
        db_session, "quicksilver", auth=auth, source_type=SourceType.WEB_PAGE, **EXTRACT
    )

    assert document.id == agents.id
    assert await mark_of(db_session, agents.id) is None
    [event] = await _events(db_session, OperatorEventType.SUBJECT_ACCEPTED)
    assert event.payload["via"] == "EXTERNAL_REF" and event.payload["refs"] == ["fake:F1"]


@pytest.mark.asyncio
async def test_a_confirmed_bare_local_mint_folds_the_bare_local_provisional_one(
    db_session: AsyncSession,
) -> None:
    agents = await _resolve(db_session, "the scheduler", **MINT_P)
    link = ParticleSubjectRow(particle_id="p-agent", subject_id=agents.id)
    db_session.add(link)
    await db_session.flush()

    confirmed = await _resolve(db_session, "The Scheduler", **EXTRACT)

    assert confirmed.id != agents.id
    assert await get_subject(db_session, agents.id) is None
    links = (await db_session.execute(select(ParticleSubjectRow))).scalars().all()
    assert [(r.particle_id, r.subject_id) for r in links] == [("p-agent", confirmed.id)]
    [event] = await _events(db_session, OperatorEventType.SUBJECTS_MERGED)
    assert event.actor == RESOLVER_ALIAS_ACTOR
    assert event.payload["reason"] == "provisional-fold"
    assert event.payload["sources"] == [agents.id]
    # The agent's later mentions now reach the confirmed Subject.
    assert (await _resolve(db_session, "the scheduler", **MINT_P)).id == confirmed.id


@pytest.mark.asyncio
async def test_an_external_ref_on_either_side_holds_the_pair(db_session: AsyncSession) -> None:
    agents = await _resolve(db_session, "Prometheus", **MINT_P)
    titan = Subject(
        canonical_name="Prometheus",
        external_ids=[_REF],
        created_at=datetime.now(UTC),
        asserted_by="general-extractor",
    )
    await insert_subject(db_session, titan)

    assert (await fold_or_hold(db_session, titan)).id == titan.id
    assert await mark_of(db_session, agents.id) is not None  # held, not folded
    [entry] = await list_provisional(db_session)
    assert entry.subject.id == agents.id and [s.id for s in entry.held_with] == [titan.id]
    # The agent's later mentions resolve to the confirmed Subject, as document-first would.
    assert (await _resolve(db_session, "Prometheus", **MINT_P)).id == titan.id


async def _end_state(session: AsyncSession, name: str) -> tuple[int, dict[str, str], int]:
    """How many Subjects hold ``name``, which Subject (by name) each claim is on, and marks."""
    from particles.store.subject_store import find_name_candidates

    holders = await find_name_candidates(session, name)
    rows = (await session.execute(select(ParticleSubjectRow))).scalars().all()
    on: dict[str, str] = {}
    for row in rows:
        if not row.particle_id.startswith(name):
            continue
        subject = await get_subject(session, row.subject_id)
        assert subject is not None
        on[row.particle_id.removeprefix(name)] = subject.canonical_name.casefold()
    marked = sum(1 for _, mark in holders if mark is not None)
    return len(holders), on, marked


async def _claim(session: AsyncSession, claim_id: str, subject: Subject) -> None:
    session.add(ParticleSubjectRow(particle_id=claim_id, subject_id=subject.id))
    await session.flush()


@pytest.mark.asyncio
async def test_order_independence_for_bare_local_names(db_session: AsyncSession) -> None:
    """Agent-first then the fold ends with the Subjects and links document-first ends with (§4)."""
    # Agent first: the agent's mint is provisional, the document's confirmed mint folds it.
    first = "the cache"
    await _claim(db_session, f"{first}:agent", await _resolve(db_session, first, **MINT_P))
    await _claim(db_session, f"{first}:document", await _resolve(db_session, first, **EXTRACT))

    # Document first: the agent's mention resolves to the document's Subject at Step 1.
    second = "the queue"
    await _claim(db_session, f"{second}:document", await _resolve(db_session, second, **EXTRACT))
    await _claim(db_session, f"{second}:agent", await _resolve(db_session, second, **MINT_P))

    agent_first = await _end_state(db_session, first)
    document_first = await _end_state(db_session, second)
    assert agent_first[0] == document_first[0] == 1
    assert agent_first[2] == document_first[2] == 0
    assert set(agent_first[1]) == set(document_first[1]) == {":agent", ":document"}
    assert set(agent_first[1].values()) == {first}
    assert set(document_first[1].values()) == {second}


# --- the read-only seams agree with the write -------------------------------------


@pytest.mark.asyncio
async def test_the_read_only_seams_answer_as_the_write_would(db_session: AsyncSession) -> None:
    agents = await _resolve(db_session, "the router", **MINT_P)
    for kind in (MINT_P, MINT_Q):
        found = await find_existing_subject(
            db_session, "the router", source_type=SourceType.CONVERSATION, **kind
        )
        assert found is not None and found.id == agents.id
        prefetch = await prefetch_subjects(
            db_session, ["the router"], source_type=SourceType.CONVERSATION, **kind
        )
        assert prefetch.known_ids == [agents.id]
    assert (
        await find_existing_subject(
            db_session, "the router", source_type=SourceType.CONVERSATION, **EXTRACT
        )
        is None
    )
    # A read-only seam performs no transition: Q's look did not share it.
    mark = await mark_of(db_session, agents.id)
    assert mark is not None and mark.shared_by is None


# --- reads ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reads_mark_a_provisional_subject_and_omit_the_key_on_a_confirmed_one(
    db_session: AsyncSession,
) -> None:
    agents = await _resolve(db_session, "the worker", **MINT_P)
    confirmed = await _resolve(db_session, "Kafka", **EXTRACT)

    marked = await with_mark(db_session, agents)
    plain = await with_mark(db_session, confirmed)
    assert isinstance(marked, SubjectView) and isinstance(plain, SubjectView)
    assert marked.model_dump(mode="json")["provisional"]["principal"] == P
    assert "provisional" not in plain.model_dump(mode="json")


@pytest.mark.asyncio
async def test_a_name_lookup_prefers_the_confirmed_subject_of_a_held_pair(
    db_session: AsyncSession,
) -> None:
    agents = await _resolve(db_session, "Atlas", **MINT_P)
    held = Subject(
        canonical_name="Atlas", external_ids=[_REF], asserted_by="x", created_at=datetime.now(UTC)
    )
    await insert_subject(db_session, held)

    found = await find_by_name(db_session, "atlas")
    looked = await lookup_subject(db_session, "Atlas")
    assert found is not None and found.id == held.id
    assert looked is not None and looked.id == held.id  # not refused as ambiguous
    # A read by id is addressing and is never filtered.
    by_id = await get_subject(db_session, agents.id)
    assert by_id is not None and by_id.id == agents.id


@pytest.mark.asyncio
async def test_a_provisional_subject_alone_is_found_by_name_unless_confirmed_only(
    db_session: AsyncSession,
) -> None:
    agents = await _resolve(db_session, "Helios", **MINT_P)
    found = await find_by_name(db_session, "helios")
    assert found is not None and found.id == agents.id
    assert await find_by_name(db_session, "helios", confirmed_only=True) is None


# --- operator acts ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_accept_confirms_it_for_every_resolution(db_session: AsyncSession) -> None:
    agents = await _resolve(db_session, "the gateway", **MINT_P)
    view, accepted = await accept_subject(db_session, agents.id, actor="operator")
    assert accepted and view.provisional is None
    [event] = await _events(db_session, OperatorEventType.SUBJECT_ACCEPTED)
    assert event.payload["via"] == "OPERATOR" and event.payload["principal"] == P
    assert (await _resolve(db_session, "the gateway", **EXTRACT)).id == agents.id

    _, again = await accept_subject(db_session, agents.id, actor="operator")
    assert not again
    with pytest.raises(ValueError, match="not found"):
        await accept_subject(db_session, "no-such-subject", actor="operator")


@pytest.mark.asyncio
async def test_an_operator_alias_on_it_accepts_it(db_session: AsyncSession) -> None:
    agents = await _resolve(db_session, "the broker", **MINT_P)
    await add_aliases(db_session, agents.id, ["message broker"])
    assert await operator_touched(db_session, agents.id, actor="operator", act="alias")
    assert await mark_of(db_session, agents.id) is None


@pytest.mark.asyncio
async def test_an_operator_alias_giving_a_confirmed_subject_the_name_folds(
    db_session: AsyncSession,
) -> None:
    agents = await _resolve(db_session, "the bus", **MINT_P)
    confirmed = await _resolve(db_session, "Event Bus", **EXTRACT)
    await add_aliases(db_session, confirmed.id, ["the bus"])
    await operator_touched(db_session, confirmed.id, actor="operator", act="alias")
    assert await get_subject(db_session, agents.id) is None


# --- the backfill -----------------------------------------------------------------


async def _agent_subject(session: AsyncSession, name: str, by: str, **extra: Any) -> Subject:
    subject = Subject(canonical_name=name, asserted_by=by, created_at=datetime.now(UTC), **extra)
    await insert_subject(session, subject)
    return subject


async def _linked(session: AsyncSession, subject: Subject, particle_id: str, by: str) -> None:
    from particles.core.schema import (
        Confidence,
        Particle,
        ProvenanceRef,
        ProvenanceRefType,
        UncertaintyNature,
    )
    from particles.store.particle_store import insert_particle

    particle = Particle(
        id=particle_id,
        content=f"A claim about {subject.canonical_name}.",
        confidence=Confidence(value=0.7),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by=by,
        subject_ids=[subject.id],
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e")],
    )
    await insert_particle(session, particle)  # writes the particle_subjects row too
    await session.flush()


@pytest.mark.asyncio
async def test_the_backfill_marks_only_what_no_one_else_depends_on(
    db_session: AsyncSession,
) -> None:
    own = await _agent_subject(db_session, "alone", P)
    await _linked(db_session, own, "c1", P)
    shared = await _agent_subject(db_session, "shared", P)
    await _linked(db_session, shared, "c2", P)
    await _linked(db_session, shared, "c3", "general-extractor")
    referenced = await _agent_subject(db_session, "referenced", P, external_ids=[_REF])
    aliased = await _agent_subject(db_session, "aliased", P)
    await add_aliases(db_session, aliased.id, ["an operator's alias"], actor="subjects-alias")
    process_aliased = await _agent_subject(db_session, "process-aliased", P)
    await add_aliases(db_session, process_aliased.id, ["x"], actor=RESOLVER_ALIAS_ACTOR)
    await _agent_subject(db_session, "an operator's", "operator")

    dry = await backfill(db_session, agent_principals={P}, dry_run=True)
    assert {s.id for s in dry.marked} == {own.id, process_aliased.id}
    assert {s.id for s in dry.already_shared} == {shared.id, referenced.id, aliased.id}
    assert await mark_of(db_session, own.id) is None  # the dry run wrote nothing

    live = await backfill(db_session, agent_principals={P}, dry_run=False)
    assert live.counts() == {"marked": 2, "already_shared": 3}
    mark = await mark_of(db_session, own.id)
    assert mark is not None and mark.principal == P
    [event] = await _events(db_session, OperatorEventType.SUBJECT_PROVISIONAL_BACKFILLED)
    assert event.payload["marked"] == 2 and event.payload["already_shared"] == 3


@pytest.mark.asyncio
async def test_mark_provisional_is_the_only_writer_of_the_row(db_session: AsyncSession) -> None:
    subject = await _agent_subject(db_session, "direct", P)
    await mark_provisional(db_session, subject.id, P)
    [entry] = await list_provisional(db_session)
    assert entry.subject.id == subject.id and entry.held_with == ()

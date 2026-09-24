# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Negative-retrieval assertions on the production read path.

Each test here states, in its name, material that the read surface **must
not** return, and asserts the specific particle id is absent from the result
set of a real ``query`` / ``retrieve_ranked`` call (``particles.operations.query``,
the single read path the CLI, HTTP API, and MCP ``query`` tool all drive). The
control particle in each test carries the same embedding as the excluded one,
so the exclusion can only come from the read path's status / temporal filter —
never from ranking.

The *mechanism* behind these exclusions is tested at unit level elsewhere:
``tests/test_as_of.py`` covers the as-of visibility predicate and the
``retired_at`` stamp, and ``tests/test_query.py`` covers the
filters on the current read. This module makes the read-path contract explicit
and greppable: the answer set never contains a RETRACTED, SUPERSEDED, or
INCONSISTENCY particle, and an as-of query never contains a claim asserted
after the instant asked about.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    QueryRequest,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason

# One shared embedding for every particle: similarity is identical across the
# store, so presence / absence below is decided by the read path's filters.
EMB = (np.ones(4, dtype=np.float32) / 2.0).tolist()

T1996 = datetime(1996, 3, 1, tzinfo=UTC)
T2000 = datetime(2000, 1, 1, tzinfo=UTC)
T2006 = datetime(2006, 8, 24, tzinfo=UTC)


def _particle(
    content: str,
    *,
    asserted_at: datetime | None = None,
    status: Status = Status.ACTIVE,
    supersedes: str | None = None,
    asserted_by: str = "test-agent",
    provenance: list[ProvenanceRef] | None = None,
) -> Particle:
    kwargs: dict[str, Any] = {}
    if asserted_at is not None:
        kwargs["asserted_at"] = asserted_at
    return Particle(
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by=asserted_by,
        status=status,
        supersedes=supersedes,
        provenance=provenance
        or [
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1", snapshot_id="snap-1"
            )
        ],
        **kwargs,
    )


def _mock_embeddings() -> MagicMock:
    model = MagicMock()
    model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
    return model


async def _query_ids(session: Any, monkeypatch: Any, request: QueryRequest) -> list[str]:
    """Run the real ``query`` operation and return the ids in its answer set.

    The embedding model and the NL answer generation are the only two seams
    replaced (tests/AGENTS.md § Mocking strategy); candidate selection,
    filtering, scoring, and truncation are the production code.
    """
    import particles.operations.query.main as qmain
    from particles import embeddings as ep

    original = ep._embedding_model
    ep.set_embedding_model(_mock_embeddings())
    monkeypatch.setattr(qmain, "_generate_response", AsyncMock(return_value="answer"))
    try:
        result = await qmain.query(session, request)
    finally:
        ep.set_embedding_model(original)
    return [p.id for p in result.particles]


async def test_query_excludes_retracted_particles(db_session: Any, monkeypatch: Any) -> None:
    """A particle the operator retracted is not in the answer set; its ACTIVE
    twin (same embedding, same source) still is."""
    from particles.store.particle_store import insert_particle, update_particle_status

    kept = _particle("Pluto has five known moons.")
    retracted = _particle("Pluto has three known moons.")
    for p in (kept, retracted):
        await insert_particle(db_session, p, EMB)
    await update_particle_status(
        db_session, retracted.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
    )
    await db_session.commit()

    ids = await _query_ids(db_session, monkeypatch, QueryRequest(question="moons?", top_k=10))

    assert retracted.id not in ids
    assert ids == [kept.id]


async def test_query_excludes_superseded_particles(db_session: Any, monkeypatch: Any) -> None:
    """The predecessor of a supersession chain is not in the answer set; only
    its successor is."""
    from particles.store.particle_store import insert_particle, update_particle_status

    old = _particle("Pluto is the ninth planet.", asserted_at=T1996)
    await insert_particle(db_session, old, EMB)
    await update_particle_status(
        db_session, old.id, Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION
    )
    new = _particle("Pluto is a dwarf planet.", asserted_at=T2006, supersedes=old.id)
    await insert_particle(db_session, new, EMB)
    await db_session.commit()

    ids = await _query_ids(db_session, monkeypatch, QueryRequest(question="Pluto?", top_k=10))

    assert old.id not in ids
    assert ids == [new.id]


async def test_query_excludes_inconsistency_from_answer_set(
    db_session: Any, monkeypatch: Any
) -> None:
    """An INCONSISTENCY record — the lint-minted wrapper naming two conflicting
    claims — is review material, never an answer: the two claims it wraps are
    returned, the wrapper is not."""
    from particles.store.particle_store import insert_particle

    a = _particle("The vault holds 612 coins.")
    b = _particle("The vault holds 621 coins.")
    wrapper = _particle(
        "612 coins conflicts with 621 coins",
        status=Status.INCONSISTENCY,
        asserted_by="lint",
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=a.id),
            ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=b.id),
        ],
    )
    for p in (a, b, wrapper):
        await insert_particle(db_session, p, EMB)
    await db_session.commit()

    ids = await _query_ids(db_session, monkeypatch, QueryRequest(question="coins?", top_k=10))

    assert wrapper.id not in ids
    assert set(ids) == {a.id, b.id}


async def test_as_of_excludes_claims_asserted_after_the_instant(
    db_session: Any, monkeypatch: Any
) -> None:
    """A query as of T never returns a claim first asserted after T, even
    though that claim is ACTIVE today and would be returned by the same
    question with no instant set."""
    from particles.store.particle_store import insert_particle

    before = _particle("Pluto is the ninth planet.", asserted_at=T1996)
    after = _particle("Pluto is a dwarf planet.", asserted_at=T2006)
    for p in (before, after):
        await insert_particle(db_session, p, EMB)
    await db_session.commit()

    then = await _query_ids(
        db_session, monkeypatch, QueryRequest(question="Pluto?", top_k=10, as_of=T2000)
    )
    now = await _query_ids(db_session, monkeypatch, QueryRequest(question="Pluto?", top_k=10))

    assert after.id not in then
    assert then == [before.id]
    # Control: with no instant set the later claim is retrievable, so the
    # exclusion above is the as-of lens and not a defect in the fixture.
    assert after.id in now


async def test_retrieve_ranked_excludes_retracted_and_superseded_particles(
    db_session: Any,
) -> None:
    """The model-free selection path (documentation projection and the
    session-start digest; no embedding, no LLM) applies the same exclusions
    as ``query`` — it shares the candidate-selection code, so a retracted or
    superseded particle is absent here too."""
    from particles.operations.query.main import retrieve_ranked
    from particles.store.particle_store import insert_particle, update_particle_status

    kept = _particle("kept belief")
    retracted = _particle("retracted belief")
    superseded = _particle("superseded belief")
    for p in (kept, retracted, superseded):
        await insert_particle(db_session, p, EMB)
    await update_particle_status(
        db_session, retracted.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
    )
    await update_particle_status(
        db_session, superseded.id, Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION
    )
    await db_session.commit()

    ranked = await retrieve_ranked(
        db_session, QueryRequest(question="belief?", top_k=10), use_embeddings=False
    )
    ids = [p.id for p, _, _ in ranked]

    assert retracted.id not in ids
    assert superseded.id not in ids
    assert ids == [kept.id]


# ---------------------------------------------------------------------------
# Observer scope — what a project observer must NOT be shown.
#
# One world for every read surface: project A's session must see its own
# belief and the global one, and never project B's belief, an unattributed
# harvest, or a belief derived across the two. Each test goes through one
# production selection site; together they are every site the design names.
# ---------------------------------------------------------------------------

_A, _B = "-proj-a", "-proj-b"


async def _observer_world(session: Any, **fields: Any) -> dict[str, str]:
    from tests._observer_scope import belief, project_source_tags, rescoped, source

    in_a = await source(session, "a", project_source_tags(_A))
    in_b = await source(session, "b", project_source_tags(_B))
    page = await source(session, "page", ["web"])
    unstamped = await source(session, "audit", ["claude-code", "audit"])
    mine = await belief(session, "Project A deploys on Fridays.", in_a, **fields)
    theirs = await belief(session, "Project B deploys on Mondays.", in_b, **fields)
    both = await belief(session, "Every commit needs a sign-off.", in_a, in_b, **fields)
    world = await belief(session, "Pluto has five known moons.", page, **fields)
    stray = await belief(session, "A harvest nobody attributed.", unstamped, **fields)
    across = await belief(
        session, "Derived across both projects.", premises=(mine.id, theirs.id), **fields
    )
    await rescoped(session)
    await session.commit()
    return {
        "mine": mine.id,
        "theirs": theirs.id,
        "both": both.id,
        "world": world.id,
        "stray": stray.id,
        "across": across.id,
    }


def _assert_project_a_view(ids: set[str], w: dict[str, str]) -> None:
    assert {w["mine"], w["both"], w["world"]} <= ids
    assert not ids & {w["theirs"], w["stray"], w["across"]}


async def test_query_shows_a_project_observer_only_its_own_and_global(
    db_session: Any, monkeypatch: Any
) -> None:
    w = await _observer_world(db_session)

    request = QueryRequest(question="deploys?", top_k=50, observer_project=_A)
    ids = set(await _query_ids(db_session, monkeypatch, request))

    _assert_project_a_view(ids, w)
    # The control: without an observer the same query returns all of them.
    everything = set(await _query_ids(db_session, monkeypatch, QueryRequest(question="deploys?")))
    assert set(w.values()) <= everything


async def test_query_discloses_what_the_observer_did(db_session: Any, monkeypatch: Any) -> None:
    import particles.operations.query.main as qmain
    from particles import embeddings as ep

    await _observer_world(db_session)
    original = ep._embedding_model
    ep.set_embedding_model(_mock_embeddings())
    monkeypatch.setattr(qmain, "_generate_response", AsyncMock(return_value="answer"))
    try:
        scoped = await qmain.query(db_session, QueryRequest(question="?", observer_project=_A))
        plain = await qmain.query(db_session, QueryRequest(question="?"))
    finally:
        ep.set_embedding_model(original)

    assert plain.observer_scope is None
    note = scoped.observer_scope
    assert note is not None and note.engaged and note.project == _A
    assert (note.total, note.in_scope, note.unattributed) == (6, 3, 2)


async def test_retrieve_ranked_excludes_out_of_scope_beliefs(db_session: Any) -> None:
    from particles.operations.query import retrieve_ranked

    w = await _observer_world(db_session)

    scored = await retrieve_ranked(
        db_session,
        QueryRequest(question="deploys", top_k=50, observer_project=_A),
        use_embeddings=False,
    )

    _assert_project_a_view({s[0].id for s in scored}, w)


async def test_structural_listing_excludes_out_of_scope_beliefs(db_session: Any) -> None:
    from particles.core.schema import ClaimTerm, StructuredClaim, TermKind
    from particles.operations.query.structural import structural_query

    claim = StructuredClaim(
        subject=ClaimTerm(kind=TermKind.TOKEN, value="deploy"),
        predicate=ClaimTerm(kind=TermKind.URI, value="ex:day"),
        object=ClaimTerm(kind=TermKind.LITERAL, value="Friday"),
        structurizer_id="test",
        structurizer_version="1.0.0",
    )
    w = await _observer_world(db_session, structured_claim=claim)

    response = await structural_query(
        db_session, QueryRequest(predicate="ex:day", top_k=50, observer_project=_A)
    )

    _assert_project_a_view({p.id for p in response.particles}, w)
    assert response.observer_scope is not None and response.observer_scope.in_scope == 3


async def test_listing_pages_after_the_observer_not_before(db_session: Any) -> None:
    from particles.operations.query.observer_scope import list_particles_in_view

    w = await _observer_world(db_session)

    pages = [
        await list_particles_in_view(
            db_session, status=None, subject_id=None, limit=2, offset=o, observer_project=_A
        )
        for o in (0, 2)
    ]

    assert [len(page) for page in pages] == [2, 1]  # a full first page, never a starved one
    _assert_project_a_view({p.id for page in pages for p in page}, w)

"""Tests for the OwnerPolicy: viewer resolution, apply_owner, composition.

The load-bearing property throughout is **inertness**: every way the lens can be
unconfigured, disabled, or unresolvable must leave the ranking byte-identical to
the pre-0220 order. A relevance lens that silently reorders a store nobody asked
it to is worse than one that does nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from particles.config import get_config
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Subject,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.operations.query.owner_policy import (
    EMPTY_OWNER_POLICY,
    OwnerPolicy,
    apply_owner,
    load_owner_policy,
)
from particles.store.particle_store import insert_particle
from particles.store.subject_store import insert_subject

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_EMB = (np.ones(4, dtype=np.float32) / np.linalg.norm(np.ones(4, dtype=np.float32))).tolist()


def _particle(content: str, conf: float = 0.7, *, subject_ids: list[str] | None = None) -> Particle:
    return Particle(
        content=content,
        subject_ids=subject_ids or [],
        confidence=Confidence(value=conf, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        status=Status.ACTIVE,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1", snapshot_id="snap-1"
            )
        ],
    )


async def _add(session: AsyncSession, particle: Particle) -> None:
    # `insert_particle` persists `subject_ids` and links the join table from it,
    # so the model field is the single source of truth — exactly as the ingest
    # pipeline sets it. `A(p)` reads that same field.
    await insert_particle(session, particle, _EMB)
    await session.flush()


def _configure(subjects: list[str], rank_lift: float = 0.05, enabled: bool = True) -> None:
    # The autouse fixture calls reset_config() before the next test, so mutating
    # the cached config here does not leak (and reset_config() mid-test would
    # dispose the db_session engine).
    cfg = get_config().owner_lens
    cfg.enabled = enabled
    cfg.subjects = subjects
    cfg.rank_lift = rank_lift


# ---------------------------------------------------------------------------
# Resolve-or-inert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unconfigured_yields_empty_policy(db_session: AsyncSession) -> None:
    """The shipped default: no viewer configured ⇒ inert."""
    policy = await load_owner_policy(db_session)
    assert policy is EMPTY_OWNER_POLICY


@pytest.mark.asyncio
async def test_disabled_yields_empty_policy(db_session: AsyncSession) -> None:
    await insert_subject(db_session, Subject(canonical_name="Jeff", asserted_by="test"))
    _configure(["Jeff"], enabled=False)
    assert await load_owner_policy(db_session) is EMPTY_OWNER_POLICY


@pytest.mark.asyncio
async def test_zero_rank_lift_yields_empty_policy(db_session: AsyncSession) -> None:
    """ω = 0 is inert even with a viewer configured — the shipped default."""
    await insert_subject(db_session, Subject(canonical_name="Jeff", asserted_by="test"))
    _configure(["Jeff"], rank_lift=0.0)
    assert await load_owner_policy(db_session) is EMPTY_OWNER_POLICY


@pytest.mark.asyncio
async def test_fully_unresolved_viewer_yields_empty_policy(db_session: AsyncSession) -> None:
    """A configured-but-unresolvable viewer degrades to inert, never to a guess."""
    _configure(["Nobody At All"])
    assert await load_owner_policy(db_session) is EMPTY_OWNER_POLICY


@pytest.mark.asyncio
async def test_resolves_by_canonical_name(db_session: AsyncSession) -> None:
    subject = Subject(canonical_name="Jeff", asserted_by="test")
    await insert_subject(db_session, subject)
    _configure(["Jeff"])
    policy = await load_owner_policy(db_session)
    assert policy.viewer_subject_ids == frozenset({subject.id})
    assert policy.rank_lift == pytest.approx(0.05)
    assert policy.unresolved == ()


@pytest.mark.asyncio
async def test_resolves_by_subject_id(db_session: AsyncSession) -> None:
    subject = Subject(canonical_name="Jeff", asserted_by="test")
    await insert_subject(db_session, subject)
    _configure([subject.id])
    policy = await load_owner_policy(db_session)
    assert policy.viewer_subject_ids == frozenset({subject.id})


@pytest.mark.asyncio
async def test_partial_resolution_fires_and_reports(db_session: AsyncSession) -> None:
    """One bad alias must not disable the whole lens.

    All-or-nothing would make adding a speculative alias a silent kill switch,
    which is the opposite of the intended failure mode.
    """
    subject = Subject(canonical_name="Jeff", asserted_by="test")
    await insert_subject(db_session, subject)
    _configure(["Jeff", "Jeff Gage"])
    policy = await load_owner_policy(db_session)
    assert policy.viewer_subject_ids == frozenset({subject.id})
    assert policy.unresolved == ("Jeff Gage",)


@pytest.mark.asyncio
async def test_multiple_aliases_resolve_to_a_set(db_session: AsyncSession) -> None:
    """A viewer's Subject fragments in practice until merge lands."""
    a = Subject(canonical_name="Jeff", asserted_by="test")
    b = Subject(canonical_name="Jeff Gage", asserted_by="test")
    await insert_subject(db_session, a)
    await insert_subject(db_session, b)
    _configure(["Jeff", "Jeff Gage"])
    policy = await load_owner_policy(db_session)
    assert policy.viewer_subject_ids == frozenset({a.id, b.id})


@pytest.mark.asyncio
async def test_blank_entries_are_skipped(db_session: AsyncSession) -> None:
    subject = Subject(canonical_name="Jeff", asserted_by="test")
    await insert_subject(db_session, subject)
    _configure(["", "  ", "Jeff"])
    policy = await load_owner_policy(db_session)
    assert policy.viewer_subject_ids == frozenset({subject.id})
    assert policy.unresolved == ()


# ---------------------------------------------------------------------------
# A(p) and the bonus
# ---------------------------------------------------------------------------


def test_is_owner_relevant_is_set_intersection() -> None:
    policy = OwnerPolicy(viewer_subject_ids=frozenset({"s1"}), rank_lift=0.05)
    assert policy.is_owner_relevant(["s1"])
    assert policy.is_owner_relevant(["s2", "s1"])  # multi-subject edge still counts
    assert not policy.is_owner_relevant(["s2"])
    assert not policy.is_owner_relevant([])


def test_empty_policy_is_relevant_to_nothing() -> None:
    assert not EMPTY_OWNER_POLICY.is_owner_relevant(["s1"])
    assert EMPTY_OWNER_POLICY.bonus(["s1"]) == 0.0


@pytest.mark.asyncio
async def test_apply_owner_adds_omega_to_relevant_only(db_session: AsyncSession) -> None:
    policy = OwnerPolicy(viewer_subject_ids=frozenset({"s1"}), rank_lift=0.05)
    out = await apply_owner(
        db_session,
        {"p1": 0.70, "p2": 0.70},
        {"p1": ["s1"], "p2": ["s2"]},
        policy=policy,
    )
    assert out["p1"] == pytest.approx(0.75)
    assert out["p2"] == pytest.approx(0.70)


@pytest.mark.asyncio
async def test_apply_owner_is_a_noop_under_the_empty_policy(db_session: AsyncSession) -> None:
    scores = {"p1": 0.7, "p2": 0.4}
    out = await apply_owner(db_session, scores, {"p1": ["s1"]}, policy=EMPTY_OWNER_POLICY)
    assert out == scores


@pytest.mark.asyncio
async def test_apply_owner_handles_a_particle_with_no_subjects(
    db_session: AsyncSession,
) -> None:
    """A belief missing from the subject map must score, not KeyError."""
    policy = OwnerPolicy(viewer_subject_ids=frozenset({"s1"}), rank_lift=0.05)
    out = await apply_owner(db_session, {"p1": 0.7}, {}, policy=policy)
    assert out["p1"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Composition — the digest seam
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rank_score_lifts_owner_belief_while_display_confidence_is_untouched(
    db_session: AsyncSession,
) -> None:
    """The two-map split: rank moves, displayed confidence does not."""
    from particles.operations.query.effective_confidence import score_confidence_and_rank

    subject = Subject(canonical_name="Jeff", asserted_by="test")
    await insert_subject(db_session, subject)
    mine = _particle("Jeff prefers general mechanisms.", conf=0.50, subject_ids=[subject.id])
    theirs = _particle("Numista charges for API access.", conf=0.60)
    await _add(db_session, mine)
    await _add(db_session, theirs)
    _configure(["Jeff"], rank_lift=0.5)

    eff, rank = await score_confidence_and_rank(
        db_session, [mine, theirs], populate_cache=True, with_utility=False
    )
    # Truth axis untouched — the owner belief is still the *less* confident one.
    assert eff[mine.id] < eff[theirs.id]
    # Ordering axis flipped by the lens.
    assert rank[mine.id] > rank[theirs.id]
    assert rank[mine.id] == pytest.approx(eff[mine.id] + 0.5)
    assert rank[theirs.id] == pytest.approx(eff[theirs.id])


@pytest.mark.asyncio
async def test_with_owner_false_leaves_rank_equal_to_effective(
    db_session: AsyncSession,
) -> None:
    from particles.operations.query.effective_confidence import score_confidence_and_rank

    subject = Subject(canonical_name="Jeff", asserted_by="test")
    await insert_subject(db_session, subject)
    p = _particle("Jeff prefers general mechanisms.", subject_ids=[subject.id])
    await _add(db_session, p)
    _configure(["Jeff"], rank_lift=0.5)

    eff, rank = await score_confidence_and_rank(
        db_session, [p], populate_cache=True, with_utility=False, with_owner=False
    )
    assert eff == rank


@pytest.mark.asyncio
async def test_unconfigured_lens_leaves_ranking_byte_identical(
    db_session: AsyncSession,
) -> None:
    """The inertness guarantee, end to end at the digest seam."""
    from particles.operations.query.effective_confidence import score_confidence_and_rank

    subject = Subject(canonical_name="Jeff", asserted_by="test")
    await insert_subject(db_session, subject)
    p = _particle("Jeff prefers general mechanisms.", subject_ids=[subject.id])
    await _add(db_session, p)
    # No _configure() call — the shipped default.

    eff, rank = await score_confidence_and_rank(
        db_session, [p], populate_cache=True, with_utility=False
    )
    assert eff == rank


@pytest.mark.asyncio
async def test_query_path_never_applies_the_lens(db_session: AsyncSession) -> None:
    """the semantic-search path is off-limits to the lens.

    ``retrieve_ranked`` defaults ``apply_owner=False``, and ``query`` never
    passes it — a caller who wants the viewer's beliefs uses
    ``QueryRequest.subject_id`` instead.
    """
    from particles.core.schema import QueryRequest
    from particles.operations.query.main import retrieve_ranked

    subject = Subject(canonical_name="Jeff", asserted_by="test")
    await insert_subject(db_session, subject)
    mine = _particle("Jeff prefers general mechanisms.", conf=0.50, subject_ids=[subject.id])
    theirs = _particle("Numista charges for API access.", conf=0.60)
    await _add(db_session, mine)
    await _add(db_session, theirs)
    _configure(["Jeff"], rank_lift=0.5)

    request = QueryRequest(question="anything", top_k=10)
    scored = await retrieve_ranked(db_session, request, use_embeddings=False)
    by_id = {p.id: eff for p, _sim, eff in scored}
    # No lift anywhere: the owner belief keeps its lower truth-axis score.
    assert by_id[mine.id] < by_id[theirs.id]


@pytest.mark.asyncio
async def test_recall_path_applies_the_lens(db_session: AsyncSession) -> None:
    """The same call with ``apply_owner=True`` — the projection's configuration."""
    from particles.core.schema import QueryRequest
    from particles.operations.query.main import retrieve_ranked

    subject = Subject(canonical_name="Jeff", asserted_by="test")
    await insert_subject(db_session, subject)
    mine = _particle("Jeff prefers general mechanisms.", conf=0.50, subject_ids=[subject.id])
    theirs = _particle("Numista charges for API access.", conf=0.60)
    await _add(db_session, mine)
    await _add(db_session, theirs)
    _configure(["Jeff"], rank_lift=0.5)

    request = QueryRequest(question="anything", top_k=10)
    scored = await retrieve_ranked(db_session, request, use_embeddings=False, apply_owner=True)
    assert scored[0][0].id == mine.id


# ---------------------------------------------------------------------------
# Composition — the graph view
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_view_keeps_viewer_neighbourhood_through_the_node_cap(
    db_session: AsyncSession,
) -> None:
    """The graph's unit is a Subject, so the lens promotes the viewer's neighbourhood.

    All three non-anchor subjects sit at the same hop with equal support, so
    without the lens the cap resolves them by id — an arbitrary coin flip. With
    it, the viewer and the subject sharing an in-scope particle with them win
    the two remaining slots deterministically.

    Adjacency is deliberately computed over **in-scope** particles only: the
    lens re-orders what the traversal already loaded and never widens the
    scope, which is mandatory and explicit.
    """
    from particles.operations.graph_view import build_graph_data

    anchor = Subject(canonical_name="Topic", asserted_by="test")
    viewer = Subject(canonical_name="Jeff", asserted_by="test")
    neighbour = Subject(canonical_name="Neighbour", asserted_by="test")
    other = Subject(canonical_name="Other", asserted_by="test")
    for s in (anchor, viewer, neighbour, other):
        await insert_subject(db_session, s)

    # Edges out of the anchor, so every candidate sits at hop 1.
    await _add(db_session, _particle("anchor–viewer", subject_ids=[anchor.id, viewer.id]))
    await _add(db_session, _particle("anchor–other", subject_ids=[anchor.id, other.id]))
    # `neighbour` shares this in-scope particle with the viewer — the adjacency.
    await _add(
        db_session,
        _particle("anchor–neighbour–viewer", subject_ids=[anchor.id, neighbour.id, viewer.id]),
    )

    _configure(["Jeff"], rank_lift=0.5)
    data = await build_graph_data(db_session, subject_id=anchor.id, hops=1, max_nodes=3)
    rendered = {n.subject_id for n in data.nodes}
    assert viewer.id in rendered
    assert neighbour.id in rendered
    assert other.id not in rendered


@pytest.mark.asyncio
async def test_graph_view_confidence_is_not_lifted(db_session: AsyncSession) -> None:
    """the lens may select nodes, never alter a rendered confidence."""
    from particles.operations.graph_view import build_graph_data

    viewer = Subject(canonical_name="Jeff", asserted_by="test")
    await insert_subject(db_session, viewer)
    p = _particle("Jeff prefers general mechanisms.", conf=0.50, subject_ids=[viewer.id])
    await _add(db_session, p)
    _configure(["Jeff"], rank_lift=0.5)

    data = await build_graph_data(db_session, subject_id=viewer.id, hops=1)
    info = data.particles[p.id]
    # Base effective confidence, not 0.50 + 0.5.
    assert info.effective_confidence <= 0.5


@pytest.mark.asyncio
async def test_sweep_resolves_the_viewer_at_zero_rank_lift(db_session: AsyncSession) -> None:
    """The calibration path must see the cohort before ω is set.

    ω ships at 0.0, so without this the operator's first `sweep-owner-lift`
    would report an empty cohort and they would have to guess a value before
    they could calibrate one.
    """
    subject = Subject(canonical_name="Jeff", asserted_by="test")
    await insert_subject(db_session, subject)
    _configure(["Jeff"], rank_lift=0.0)

    assert await load_owner_policy(db_session) is EMPTY_OWNER_POLICY
    calibrating = await load_owner_policy(db_session, require_rank_lift=False)
    assert calibrating.viewer_subject_ids == frozenset({subject.id})
    # Still contributes nothing to a score — the bonus floors at ω <= 0.
    assert calibrating.bonus([subject.id]) == 0.0

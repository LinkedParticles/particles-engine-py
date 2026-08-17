"""Tests for operations/review.py — §9.6 annotation-only Review."""

from __future__ import annotations

import pytest

from particles.core.schema import (
    SCHEMA_VERSION,
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    ResolutionAction,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.store.particle_store import get_particle, insert_particle


def _make_inconsistency(particle_a_id: str, particle_b_id: str) -> Particle:
    return Particle(
        content=f"INCONSISTENCY: conflict between {particle_a_id} and {particle_b_id}",
        confidence=Confidence(value=0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="extract-pipeline",
        status=Status.INCONSISTENCY,
        schema_version=SCHEMA_VERSION,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=particle_a_id),
            ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=particle_b_id),
        ],
    )


def _make_active(content: str) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test-agent",
        status=Status.ACTIVE,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1", snapshot_id="s1")
        ],
    )


def _make_quarantined(content: str) -> Particle:
    """A quarantined conflict loser, as the extract pipeline births it."""
    return Particle(
        content=content,
        confidence=Confidence(value=0.85, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test-agent",
        status=Status.PROVENANCE_STALE,
        status_reason=StatusReason.CONFLICT_PENDING,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e2", snapshot_id="s2")
        ],
        subject_ids=["subj-1"],
    )


@pytest.mark.asyncio
async def test_prefer_a_demotes_b(db_session: object) -> None:
    from particles.operations.review import resolve

    session = db_session  # type: ignore[assignment]
    pa = _make_active("Claim A")
    pb = _make_active("Claim B")
    inc = _make_inconsistency(pa.id, pb.id)
    await insert_particle(session, pa)  # type: ignore[arg-type]
    await insert_particle(session, pb)  # type: ignore[arg-type]
    await insert_particle(session, inc)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    review = await resolve(session, inc.id, ResolutionAction.PREFER_A, "reviewer-1")  # type: ignore[arg-type]

    assert review.resolution == ResolutionAction.PREFER_A
    assert review.trust_statement_id is not None

    # B should be PROVENANCE_STALE; A should remain ACTIVE
    updated_b = await get_particle(session, pb.id)  # type: ignore[arg-type]
    updated_a = await get_particle(session, pa.id)  # type: ignore[arg-type]
    assert updated_b is not None and updated_b.status == Status.PROVENANCE_STALE
    assert updated_a is not None and updated_a.status == Status.ACTIVE


@pytest.mark.asyncio
async def test_both_valid_sets_aleatory(db_session: object) -> None:
    from particles.operations.review import resolve

    session = db_session  # type: ignore[assignment]
    pa = _make_active("It may rain tomorrow.")
    pb = _make_active("It may not rain tomorrow.")
    inc = _make_inconsistency(pa.id, pb.id)
    await insert_particle(session, pa)  # type: ignore[arg-type]
    await insert_particle(session, pb)  # type: ignore[arg-type]
    await insert_particle(session, inc)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    await resolve(session, inc.id, ResolutionAction.BOTH_VALID, "reviewer-1")  # type: ignore[arg-type]

    updated_a = await get_particle(session, pa.id)  # type: ignore[arg-type]
    updated_b = await get_particle(session, pb.id)  # type: ignore[arg-type]
    updated_inc = await get_particle(session, inc.id)  # type: ignore[arg-type]

    assert updated_a is not None and updated_a.uncertainty_nature == UncertaintyNature.ALEATORY
    assert updated_b is not None and updated_b.uncertainty_nature == UncertaintyNature.ALEATORY
    assert updated_inc is not None and updated_inc.status == Status.RETRACTED


@pytest.mark.asyncio
async def test_defer_leaves_inconsistency(db_session: object) -> None:
    from particles.operations.review import resolve

    session = db_session  # type: ignore[assignment]
    pa = _make_active("Claim A")
    pb = _make_active("Claim B")
    inc = _make_inconsistency(pa.id, pb.id)
    await insert_particle(session, pa)  # type: ignore[arg-type]
    await insert_particle(session, pb)  # type: ignore[arg-type]
    await insert_particle(session, inc)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    await resolve(session, inc.id, ResolutionAction.DEFER, "reviewer-1", note="Need more context")  # type: ignore[arg-type]

    updated_inc = await get_particle(session, inc.id)  # type: ignore[arg-type]
    assert updated_inc is not None and updated_inc.status == Status.INCONSISTENCY


@pytest.mark.asyncio
async def test_no_auto_cascade(db_session: object) -> None:
    """v0.2 Core: resolving one INCONSISTENCY does not auto-cascade to others."""
    from particles.operations.review import list_inconsistencies, resolve

    session = db_session  # type: ignore[assignment]
    pa = _make_active("Claim A")
    pb = _make_active("Claim B")
    pc = _make_active("Claim C")
    pd = _make_active("Claim D")
    inc1 = _make_inconsistency(pa.id, pb.id)
    inc2 = _make_inconsistency(pc.id, pd.id)

    for obj in [pa, pb, pc, pd, inc1, inc2]:
        await insert_particle(session, obj)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    # Resolve inc1; inc2 should remain untouched
    await resolve(session, inc1.id, ResolutionAction.PREFER_A, "reviewer-1")  # type: ignore[arg-type]
    remaining = await list_inconsistencies(session)  # type: ignore[arg-type]
    assert any(p.id == inc2.id for p in remaining), (
        "inc2 should still be INCONSISTENCY (no auto-cascade)"
    )
    # …and inc1 must have left the queue — the P4-3 assertion this test
    # historically lacked, which let resolved wrappers linger.
    assert not any(p.id == inc1.id for p in remaining), (
        "inc1 was resolved and must leave the review queue"
    )


# ---------------------------------------------------------------------------
# wrapper closure, quarantined-loser resolution, legacy wrappers.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    [ResolutionAction.PREFER_A, ResolutionAction.PREFER_B, ResolutionAction.BOTH_VALID],
)
async def test_resolution_closes_wrapper(db_session: object, action: ResolutionAction) -> None:
    """Every non-DEFER resolution terminates its ticket: the wrapper is
    RETRACTED (CONFLICT_RESOLVED) and leaves list_inconsistencies (P4-3)."""
    from particles.operations.review import list_inconsistencies, resolve

    session = db_session  # type: ignore[assignment]
    pa = _make_active("Claim A")
    pb = _make_quarantined("Claim B")
    inc = _make_inconsistency(pa.id, pb.id)
    for obj in [pa, pb, inc]:
        await insert_particle(session, obj)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    await resolve(session, inc.id, action, "reviewer-1")  # type: ignore[arg-type]

    wrapper = await get_particle(session, inc.id)  # type: ignore[arg-type]
    assert wrapper is not None
    assert wrapper.status == Status.RETRACTED
    assert wrapper.status_reason == StatusReason.CONFLICT_RESOLVED
    remaining = await list_inconsistencies(session)  # type: ignore[arg-type]
    assert not any(p.id == inc.id for p in remaining)


@pytest.mark.asyncio
async def test_prefer_a_resolves_quarantined_loser_in_place(db_session: object) -> None:
    """PREFER_A over a quarantined B: B stays PROVENANCE_STALE, reason flips
    CONFLICT_PENDING → CONFLICT_RESOLVED (reason-only update, no transition)."""
    from particles.operations.review import resolve

    session = db_session  # type: ignore[assignment]
    pa = _make_active("Claim A")
    pb = _make_quarantined("Claim B")
    inc = _make_inconsistency(pa.id, pb.id)
    for obj in [pa, pb, inc]:
        await insert_particle(session, obj)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    await resolve(session, inc.id, ResolutionAction.PREFER_A, "reviewer-1")  # type: ignore[arg-type]

    updated_b = await get_particle(session, pb.id)  # type: ignore[arg-type]
    assert updated_b is not None
    assert updated_b.status == Status.PROVENANCE_STALE
    assert updated_b.status_reason == StatusReason.CONFLICT_RESOLVED
    assert (await get_particle(session, pa.id)).status == Status.ACTIVE  # type: ignore[arg-type,union-attr]


@pytest.mark.asyncio
async def test_prefer_b_promotes_quarantined_loser(db_session: object) -> None:
    """PREFER_B mints a new ACTIVE particle from the quarantined row (Reindex
    pattern): fresh id, supersedes → B, content carried; B → SUPERSEDED."""
    from particles.operations.review import resolve
    from particles.store.particle_store import get_active_particles

    session = db_session  # type: ignore[assignment]
    pa = _make_active("Claim A")
    pb = _make_quarantined("Claim B")
    inc = _make_inconsistency(pa.id, pb.id)
    for obj in [pa, pb, inc]:
        await insert_particle(session, obj)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    await resolve(session, inc.id, ResolutionAction.PREFER_B, "reviewer-1")  # type: ignore[arg-type]

    # A demoted; quarantined B superseded by the minted ACTIVE particle.
    assert (await get_particle(session, pa.id)).status == Status.PROVENANCE_STALE  # type: ignore[arg-type,union-attr]
    updated_b = await get_particle(session, pb.id)  # type: ignore[arg-type]
    assert updated_b is not None and updated_b.status == Status.SUPERSEDED
    assert updated_b.status_reason == StatusReason.CONFLICT_RESOLVED

    minted = [
        p
        for p in await get_active_particles(session)  # type: ignore[arg-type]
        if p.supersedes == pb.id
    ]
    assert len(minted) == 1
    assert minted[0].id != pb.id
    assert minted[0].content == "Claim B"
    assert minted[0].subject_ids == ["subj-1"]
    assert minted[0].status_reason is None


@pytest.mark.asyncio
async def test_both_valid_recovers_quarantined_b_as_queryable_active(db_session: object) -> None:
    """BOTH_VALID recovers claim B: a new ACTIVE particle with ALEATORY nature
    supersedes the quarantined row, so both claims are queryable."""
    from particles.operations.review import resolve
    from particles.store.particle_store import get_active_particles

    session = db_session  # type: ignore[assignment]
    pa = _make_active("It may rain tomorrow.")
    pb = _make_quarantined("It may not rain tomorrow.")
    inc = _make_inconsistency(pa.id, pb.id)
    for obj in [pa, pb, inc]:
        await insert_particle(session, obj)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    await resolve(session, inc.id, ResolutionAction.BOTH_VALID, "reviewer-1")  # type: ignore[arg-type]

    updated_a = await get_particle(session, pa.id)  # type: ignore[arg-type]
    assert updated_a is not None and updated_a.uncertainty_nature == UncertaintyNature.ALEATORY
    assert updated_a.status == Status.ACTIVE

    minted = [
        p
        for p in await get_active_particles(session)  # type: ignore[arg-type]
        if p.supersedes == pb.id
    ]
    assert len(minted) == 1
    assert minted[0].content == "It may not rain tomorrow."
    assert minted[0].uncertainty_nature == UncertaintyNature.ALEATORY
    quarantined_after = await get_particle(session, pb.id)  # type: ignore[arg-type]
    assert quarantined_after is not None and quarantined_after.status == Status.SUPERSEDED


@pytest.mark.asyncio
async def test_re_resolving_resolved_wrapper_rejected(db_session: object) -> None:
    """A resolved wrapper cannot be resolved again."""
    from particles.operations.review import resolve

    session = db_session  # type: ignore[assignment]
    pa = _make_active("Claim A")
    pb = _make_quarantined("Claim B")
    inc = _make_inconsistency(pa.id, pb.id)
    for obj in [pa, pb, inc]:
        await insert_particle(session, obj)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    await resolve(session, inc.id, ResolutionAction.PREFER_A, "reviewer-1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected INCONSISTENCY"):
        await resolve(session, inc.id, ResolutionAction.PREFER_B, "reviewer-1")  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize("action", [ResolutionAction.PREFER_A, ResolutionAction.DEFER])
async def test_legacy_dangling_b_wrapper_still_resolvable(
    db_session: object, action: ResolutionAction
) -> None:
    """Pre-ADR-0117 wrappers carry a dangling B ref (the candidate was never
    persisted). PREFER_A and DEFER must still work over them, falling back to
    excerpt-only knowledge of B."""
    from particles.operations.review import list_inconsistencies, resolve

    session = db_session  # type: ignore[assignment]
    pa = _make_active("Claim A")
    inc = _make_inconsistency(pa.id, "00000000-dead-beef-0000-000000000000")
    await insert_particle(session, pa)  # type: ignore[arg-type]
    await insert_particle(session, inc)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    review = await resolve(session, inc.id, action, "reviewer-1")  # type: ignore[arg-type]
    assert review.resolution == action

    wrapper = await get_particle(session, inc.id)  # type: ignore[arg-type]
    assert wrapper is not None
    remaining = await list_inconsistencies(session)  # type: ignore[arg-type]
    if action == ResolutionAction.PREFER_A:
        assert wrapper.status == Status.RETRACTED
        assert not any(p.id == inc.id for p in remaining)
    else:  # DEFER leaves the wrapper open
        assert wrapper.status == Status.INCONSISTENCY
        assert any(p.id == inc.id for p in remaining)


@pytest.mark.asyncio
async def test_reviewer_trust_rank_is_configurable(db_session: object) -> None:
    """The PREFER trust statement's trust_rank reads config.trust.reviewer_trust_rank (P4-7)."""
    from particles.config import get_config
    from particles.operations.review import resolve
    from particles.store.trust_store import get_trust_statements_for_domain

    get_config().trust.reviewer_trust_rank = 0.65

    session = db_session  # type: ignore[assignment]
    pa = _make_active("Claim A")
    pb = _make_active("Claim B")
    inc = _make_inconsistency(pa.id, pb.id)
    await insert_particle(session, pa)  # type: ignore[arg-type]
    await insert_particle(session, pb)  # type: ignore[arg-type]
    await insert_particle(session, inc)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    review = await resolve(session, inc.id, ResolutionAction.PREFER_A, "reviewer-1")  # type: ignore[arg-type]

    stmts = await get_trust_statements_for_domain(session, "general")  # type: ignore[arg-type]
    stmt = next(s for s in stmts if s.statement_id == review.trust_statement_id)
    assert stmt.trust_rank == 0.65

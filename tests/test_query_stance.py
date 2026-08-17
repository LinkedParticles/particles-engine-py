"""Tests for the query-time agreement distribution (operations/query/stance.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    RelationCreatedBy,
    RelationType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.stance import STANCE_HOLDER_KEY, STANCE_MAGNITUDE_KEY
from particles.core.status import Status
from particles.operations.query.source_trust import load_trust_policy
from particles.operations.query.stance import AGREEMENT_CAVEAT, compute_stance_distribution
from particles.store.extractor_store import get_trust_weight_map, populate_trust_cache
from particles.store.particle_store import insert_particle, update_particle_status
from particles.store.relation_store import create_relation

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# Distinct valid-UUID ids per test to avoid cross-test collisions.
_A0 = "00000000-0000-0000-0000-0000000000a0"
_A1 = "00000000-0000-0000-0000-0000000000a1"
_A2 = "00000000-0000-0000-0000-0000000000a2"
_B0 = "00000000-0000-0000-0000-0000000000b0"
_C0 = "00000000-0000-0000-0000-0000000000c0"
_C1 = "00000000-0000-0000-0000-0000000000c1"
_C2 = "00000000-0000-0000-0000-0000000000c2"
_D0 = "00000000-0000-0000-0000-0000000000d0"
_D1 = "00000000-0000-0000-0000-0000000000d1"


def _claim(content: str, pid: str, properties: dict[str, object] | None = None) -> Particle:
    return Particle(
        id=pid,
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        properties=properties,
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="ce-x")],
    )


async def _neutral_policy(db_session: AsyncSession):  # type: ignore[no-untyped-def]
    populate_trust_cache(await get_trust_weight_map(db_session))
    return await load_trust_policy(db_session)


@pytest.mark.asyncio
async def test_distribution_endorse_and_dispute(db_session: AsyncSession) -> None:
    """Two holders — one endorsing, one disputing (with magnitude) — are both
    attributed and cited, never netted."""
    policy = await _neutral_policy(db_session)
    t = _claim("Confidence and agreement are distinct.", _A0)
    s1 = _claim("torvalds endorses the claim.", _A1, {STANCE_HOLDER_KEY: "github:torvalds"})
    s2 = _claim(
        "skeptic disputes the claim.",
        _A2,
        {STANCE_HOLDER_KEY: "reddit:u_skeptic", STANCE_MAGNITUDE_KEY: 0.5},
    )
    for p in (t, s1, s2):
        await insert_particle(db_session, p)
    await create_relation(
        db_session, s1.id, t.id, RelationType.ENDORSES, RelationCreatedBy.EXTRACTOR_DIRECT
    )
    await create_relation(
        db_session, s2.id, t.id, RelationType.DISPUTES, RelationCreatedBy.EXTRACTOR_DIRECT
    )
    await db_session.commit()

    dists, has_any = await compute_stance_distribution(db_session, [t], policy)
    assert has_any is True
    by_holder = {p.holder: p for p in dists[0]}
    assert len(by_holder) == 2
    assert by_holder["github:torvalds"].kind == RelationType.ENDORSES
    assert by_holder["github:torvalds"].magnitude is None
    assert by_holder["github:torvalds"].stance_particle_id == s1.id
    assert by_holder["reddit:u_skeptic"].kind == RelationType.DISPUTES
    assert by_holder["reddit:u_skeptic"].magnitude == 0.5
    # Effective confidence is the stance's own believability, in (0, 1].
    assert 0.0 < by_holder["github:torvalds"].effective_confidence <= 1.0


@pytest.mark.asyncio
async def test_no_stances_is_empty(db_session: AsyncSession) -> None:
    policy = await _neutral_policy(db_session)
    t = _claim("A lone claim.", _B0)
    await insert_particle(db_session, t)
    await db_session.commit()

    dists, has_any = await compute_stance_distribution(db_session, [t], policy)
    assert dists == [[]]
    assert has_any is False


@pytest.mark.asyncio
async def test_collapses_over_co_evidential_twin(db_session: AsyncSession) -> None:
    """A stance disputing a co-evidential twin T' counts on T's distribution
    (collapse over the CO_EVIDENTIAL group)."""
    policy = await _neutral_policy(db_session)
    t = _claim("claim phrasing A", _C0)
    t_twin = _claim("claim phrasing B", _C1)
    s = _claim("alice disputes the claim.", _C2, {STANCE_HOLDER_KEY: "x:alice"})
    for p in (t, t_twin, s):
        await insert_particle(db_session, p)
    await create_relation(
        db_session, t.id, t_twin.id, RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await create_relation(
        db_session, s.id, t_twin.id, RelationType.DISPUTES, RelationCreatedBy.EXTRACTOR_DIRECT
    )
    await db_session.commit()

    dists, has_any = await compute_stance_distribution(db_session, [t], policy)
    assert has_any is True
    assert len(dists[0]) == 1
    assert dists[0][0].holder == "x:alice"
    assert dists[0][0].kind == RelationType.DISPUTES


@pytest.mark.asyncio
async def test_retracted_stance_excluded_but_edge_preserved(db_session: AsyncSession) -> None:
    """A RETRACTED stance contributes no position, yet its edge survives (B2):
    the dangling edge is not silently destroyed, but it is excluded from the
    distribution (its endpoint is no longer ACTIVE)."""
    policy = await _neutral_policy(db_session)
    t = _claim("the target", _D0)
    s = _claim("bob endorses the claim.", _D1, {STANCE_HOLDER_KEY: "x:bob"})
    await insert_particle(db_session, t)
    await insert_particle(db_session, s)
    await create_relation(
        db_session, s.id, t.id, RelationType.ENDORSES, RelationCreatedBy.EXTRACTOR_DIRECT
    )
    await db_session.commit()

    await update_particle_status(db_session, s.id, Status.RETRACTED)
    await db_session.commit()

    # B2: the stance edge is preserved (kind-aware deletion), but the RETRACTED
    # stance is excluded from the distribution.
    from particles.store.relation_store import get_relations_for_particle

    rels = await get_relations_for_particle(db_session, t.id)
    assert any(r.relation_type == RelationType.ENDORSES for r in rels)

    dists, has_any = await compute_stance_distribution(db_session, [t], policy)
    assert has_any is False
    assert dists == [[]]


def test_agreement_caveat_names_unverified_grouping() -> None:
    # M6: the caveat must flag the key-not-agent unreliability.
    assert "unverified" in AGREEMENT_CAVEAT
    assert "keys" in AGREEMENT_CAVEAT

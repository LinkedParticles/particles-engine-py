"""Tests for the query-time co-evidential collapse step (§9.3)."""

from __future__ import annotations

import math
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
from particles.operations.query import _collapse_co_evidential_top_k
from particles.store.particle_store import insert_particle
from particles.store.relation_store import create_relation

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def _mk_particle(content: str, particle_id: str, corpus_entry_id: str = "ce-default") -> Particle:
    return Particle(
        id=particle_id,
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=corpus_entry_id),
        ],
    )


@pytest.mark.asyncio
async def test_no_collapse_for_unlinked_particles(db_session: AsyncSession) -> None:
    """Particles with no CO_EVIDENTIAL links pass through unchanged."""
    p1 = _mk_particle("first", "00000000-0000-0000-0000-000000000001", "ce-a")
    p2 = _mk_particle("second", "00000000-0000-0000-0000-000000000002", "ce-b")
    await insert_particle(db_session, p1)
    await insert_particle(db_session, p2)

    top = [(p1, 0.9, 0.85), (p2, 0.8, 0.7)]
    result = await _collapse_co_evidential_top_k(db_session, top)
    assert result == top


@pytest.mark.asyncio
async def test_pair_collapses_to_first_with_merged_confidence(
    db_session: AsyncSession,
) -> None:
    """Two CO_EVIDENTIAL particles in top-k collapse to the highest-scored one with merged conf."""
    p_high = _mk_particle("claim from src-a", "00000000-0000-0000-0000-000000000001", "ce-a")
    p_low = _mk_particle("same claim from src-b", "00000000-0000-0000-0000-000000000002", "ce-b")
    await insert_particle(db_session, p_high)
    await insert_particle(db_session, p_low)
    await create_relation(
        db_session,
        p_high.id,
        p_low.id,
        RelationType.CO_EVIDENTIAL,
        RelationCreatedBy.HUMAN_REVIEW,
    )
    await db_session.commit()

    # p_high is first (higher score)
    top = [(p_high, 0.95, 0.7), (p_low, 0.80, 0.7)]
    result = await _collapse_co_evidential_top_k(db_session, top)

    # Only one particle returned, the higher-scored representative.
    assert len(result) == 1
    rep, sim, merged_conf = result[0]
    assert rep.id == p_high.id
    assert sim == 0.95  # cosine score unchanged
    # Merged: distinct sources → 1 - (1-0.7)(1-0.7) = 0.91
    assert math.isclose(merged_conf, 0.91, abs_tol=1e-6)


@pytest.mark.asyncio
async def test_same_source_repeat_throttled_in_collapse(db_session: AsyncSession) -> None:
    """Two CO_EVIDENTIAL particles from the same corpus entry get 1/k throttling."""
    p1 = _mk_particle("first paraphrase", "00000000-0000-0000-0000-000000000001", "ce-same")
    p2 = _mk_particle("second paraphrase", "00000000-0000-0000-0000-000000000002", "ce-same")
    await insert_particle(db_session, p1)
    await insert_particle(db_session, p2)
    await create_relation(
        db_session, p1.id, p2.id, RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await db_session.commit()

    top = [(p1, 0.95, 0.7), (p2, 0.80, 0.7)]
    result = await _collapse_co_evidential_top_k(db_session, top)

    assert len(result) == 1
    _, _, merged = result[0]
    # Same source: 0.7 + 0.7×0.5 weights → 1 - (1-0.7)(1-0.35) = 1 - 0.195 = 0.805
    assert math.isclose(merged, 0.805, abs_tol=1e-6)


@pytest.mark.asyncio
async def test_partial_overlap_three_member_group_with_one_outside_top_k(
    db_session: AsyncSession,
) -> None:
    """If a CO_EVIDENTIAL group has members outside top-k, only top-k members merge."""
    p_a = _mk_particle("a", "00000000-0000-0000-0000-00000000000a", "ce-a")
    p_b = _mk_particle("b", "00000000-0000-0000-0000-00000000000b", "ce-b")
    p_c = _mk_particle("c", "00000000-0000-0000-0000-00000000000c", "ce-c")
    for p in (p_a, p_b, p_c):
        await insert_particle(db_session, p)
    # a-b-c triangle
    await create_relation(
        db_session, p_a.id, p_b.id, RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await create_relation(
        db_session, p_b.id, p_c.id, RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await db_session.commit()

    # Only a and b are in top-k; c is in the group but not retrieved.
    top = [(p_a, 0.95, 0.7), (p_b, 0.80, 0.6)]
    result = await _collapse_co_evidential_top_k(db_session, top)

    assert len(result) == 1
    rep, _, merged = result[0]
    assert rep.id == p_a.id
    # Two distinct sources in top-k: 1 - (1-0.7)(1-0.6) = 1 - 0.12 = 0.88
    assert math.isclose(merged, 0.88, abs_tol=1e-6)


@pytest.mark.asyncio
async def test_disjoint_groups_collapse_independently(db_session: AsyncSession) -> None:
    """Two separate co-evidential groups in top-k each collapse to their own representative."""
    # Group 1: p1+p2
    p1 = _mk_particle("g1-a", "00000000-0000-0000-0000-000000000001", "ce-1")
    p2 = _mk_particle("g1-b", "00000000-0000-0000-0000-000000000002", "ce-2")
    # Group 2: p3+p4
    p3 = _mk_particle("g2-a", "00000000-0000-0000-0000-000000000003", "ce-3")
    p4 = _mk_particle("g2-b", "00000000-0000-0000-0000-000000000004", "ce-4")
    for p in (p1, p2, p3, p4):
        await insert_particle(db_session, p)
    await create_relation(
        db_session, p1.id, p2.id, RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await create_relation(
        db_session, p3.id, p4.id, RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await db_session.commit()

    top = [(p1, 0.95, 0.7), (p3, 0.90, 0.7), (p2, 0.85, 0.7), (p4, 0.80, 0.7)]
    result = await _collapse_co_evidential_top_k(db_session, top)

    # Two representatives, in original sort order (p1, p3).
    ids = [r.id for r, _, _ in result]
    assert ids == [p1.id, p3.id]


@pytest.mark.asyncio
async def test_collapse_reflects_asymmetric_decay(db_session: AsyncSession) -> None:
    """: the collapse merges the per-member effective
    confidences it is handed, so a group whose members differ in decay merges
    to a lower value than the same group fresh.

    The third tuple element is the effective_confidence ``_gather_scored``
    already modulated by value × trust × source_trust × recency. Here the
    second member arrives aged (0.18 vs a fresh 0.72), and the merged result
    must equal the §6.9 noisy-OR over those asymmetric inputs — never a merge
    of raw confidences with decay applied afterward.
    """
    p_fresh = _mk_particle("claim, fresh source", "00000000-0000-0000-0000-000000000001", "ce-a")
    p_aged = _mk_particle("same claim, aged source", "00000000-0000-0000-0000-000000000002", "ce-b")
    await insert_particle(db_session, p_fresh)
    await insert_particle(db_session, p_aged)
    await create_relation(
        db_session,
        p_fresh.id,
        p_aged.id,
        RelationType.CO_EVIDENTIAL,
        RelationCreatedBy.HUMAN_REVIEW,
    )
    await db_session.commit()

    # p_fresh eff_conf 0.72 (undecayed); p_aged eff_conf 0.18 (distrusted + aged).
    top = [(p_fresh, 0.95, 0.72), (p_aged, 0.80, 0.18)]
    result = await _collapse_co_evidential_top_k(db_session, top)
    assert len(result) == 1
    rep, _, merged = result[0]
    assert rep.id == p_fresh.id
    # 1 - (1 - 0.72)(1 - 0.18) = 0.7704
    assert math.isclose(merged, 0.7704, abs_tol=1e-6)

    # The same group, both fresh (0.72, 0.72), merges strictly higher — the
    # asymmetric decay genuinely propagated through the collapse.
    both_fresh = await _collapse_co_evidential_top_k(
        db_session, [(p_fresh, 0.95, 0.72), (p_aged, 0.80, 0.72)]
    )
    assert both_fresh[0][2] > merged


@pytest.mark.asyncio
async def test_empty_top_list_returns_empty(db_session: AsyncSession) -> None:
    assert await _collapse_co_evidential_top_k(db_session, []) == []

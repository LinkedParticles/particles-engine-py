"""Tests for narrative traversal operations.

Builds a NARRATIVE particle by hand over three CLAIM constituents wired with
PART_OF (membership) and SEQUENCE_IN (order) edges, then exercises the three
read helpers and the asymmetry of the two new relation kinds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from particles.core.schema import (
    Confidence,
    Particle,
    ParticleType,
    RelationCreatedBy,
    RelationType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.operations.narrative import (
    get_narrative_constituents,
    get_narrative_sequence,
    get_narratives_containing,
)
from particles.store.particle_store import insert_particle
from particles.store.relation_store import _SYMMETRIC_KINDS, create_relation

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def _claim(content: str) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
    )


def _narrative(content: str) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        particle_type=ParticleType.NARRATIVE,
    )


async def _build_lunch_narrative(
    session: AsyncSession,
) -> tuple[Particle, list[Particle]]:
    """Insert a NARRATIVE over three ordered claims; return (narrative, [c1,c2,c3])."""
    narrative = _narrative("Lunch with Sarah, 2026-05-28")
    c1 = _claim("Had lunch with Sarah.")
    c2 = _claim("Sarah took a job at Anthropic.")
    c3 = _claim("Sarah is nervous but excited about the move.")
    for p in (narrative, c1, c2, c3):
        await insert_particle(session, p)

    # Membership: each claim is PART_OF the narrative (constituent → narrative).
    for c in (c1, c2, c3):
        await create_relation(
            session, c.id, narrative.id, RelationType.PART_OF, RelationCreatedBy.MANUAL_CLI
        )
    # Order: c1 → c2 → c3 (predecessor → successor).
    await create_relation(
        session, c1.id, c2.id, RelationType.SEQUENCE_IN, RelationCreatedBy.MANUAL_CLI
    )
    await create_relation(
        session, c2.id, c3.id, RelationType.SEQUENCE_IN, RelationCreatedBy.MANUAL_CLI
    )
    await session.flush()
    return narrative, [c1, c2, c3]


# ---------------------------------------------------------------------------
# Relation-kind asymmetry
# ---------------------------------------------------------------------------


def test_narrative_kinds_are_asymmetric() -> None:
    """PART_OF / SEQUENCE_IN must NOT be canonicalised — direction is semantic."""
    assert RelationType.PART_OF not in _SYMMETRIC_KINDS
    assert RelationType.SEQUENCE_IN not in _SYMMETRIC_KINDS


@pytest.mark.asyncio
async def test_part_of_preserves_direction(db_session: AsyncSession) -> None:
    """A PART_OF edge stored (constituent, narrative) keeps that endpoint order."""
    narrative = _narrative("N")
    claim = _claim("c")
    await insert_particle(db_session, narrative)
    await insert_particle(db_session, claim)
    rel = await create_relation(
        db_session, claim.id, narrative.id, RelationType.PART_OF, RelationCreatedBy.MANUAL_CLI
    )
    # Not canonicalised to (min, max): constituent stays particle_a.
    assert rel.particle_a == claim.id
    assert rel.particle_b == narrative.id


# ---------------------------------------------------------------------------
# get_narrative_constituents
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_constituents_returns_all_members(db_session: AsyncSession) -> None:
    narrative, claims = await _build_lunch_narrative(db_session)
    got = await get_narrative_constituents(db_session, narrative.id)
    assert {p.id for p in got} == {c.id for c in claims}


@pytest.mark.asyncio
async def test_constituents_empty_for_bare_narrative(db_session: AsyncSession) -> None:
    narrative = _narrative("Empty narrative")
    await insert_particle(db_session, narrative)
    await db_session.flush()
    assert await get_narrative_constituents(db_session, narrative.id) == []


# ---------------------------------------------------------------------------
# get_narrative_sequence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sequence_follows_sequence_in_order(db_session: AsyncSession) -> None:
    narrative, [c1, c2, c3] = await _build_lunch_narrative(db_session)
    seq = await get_narrative_sequence(db_session, narrative.id)
    assert [p.id for p in seq] == [c1.id, c2.id, c3.id]


@pytest.mark.asyncio
async def test_sequence_includes_unordered_constituents(db_session: AsyncSession) -> None:
    """A constituent with no SEQUENCE_IN edge is still returned (appended, not dropped)."""
    narrative, [c1, c2, c3] = await _build_lunch_narrative(db_session)
    loose = _claim("An afterthought, linked but unordered.")
    await insert_particle(db_session, loose)
    await create_relation(
        db_session, loose.id, narrative.id, RelationType.PART_OF, RelationCreatedBy.MANUAL_CLI
    )
    await db_session.flush()
    seq = await get_narrative_sequence(db_session, narrative.id)
    # The chain order is preserved; the loose node appears too (order-agnostic).
    assert [p.id for p in seq][:3] == [c1.id, c2.id, c3.id]
    assert {p.id for p in seq} == {c1.id, c2.id, c3.id, loose.id}


@pytest.mark.asyncio
async def test_sequence_empty_for_bare_narrative(db_session: AsyncSession) -> None:
    narrative = _narrative("Nothing here")
    await insert_particle(db_session, narrative)
    await db_session.flush()
    assert await get_narrative_sequence(db_session, narrative.id) == []


# ---------------------------------------------------------------------------
# get_narratives_containing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_narratives_containing_returns_parent(db_session: AsyncSession) -> None:
    narrative, [c1, _c2, _c3] = await _build_lunch_narrative(db_session)
    got = await get_narratives_containing(db_session, c1.id)
    assert [p.id for p in got] == [narrative.id]


@pytest.mark.asyncio
async def test_particle_can_belong_to_multiple_narratives(db_session: AsyncSession) -> None:
    """one claim may be PART_OF several narratives, no duplication."""
    narrative, [c1, _c2, _c3] = await _build_lunch_narrative(db_session)
    other = _narrative("What I learned about Sarah in 2026")
    await insert_particle(db_session, other)
    await create_relation(
        db_session, c1.id, other.id, RelationType.PART_OF, RelationCreatedBy.MANUAL_CLI
    )
    await db_session.flush()
    got = await get_narratives_containing(db_session, c1.id)
    assert {p.id for p in got} == {narrative.id, other.id}


@pytest.mark.asyncio
async def test_narratives_containing_empty_for_unlinked(db_session: AsyncSession) -> None:
    claim = _claim("Belongs to no narrative.")
    await insert_particle(db_session, claim)
    await db_session.flush()
    assert await get_narratives_containing(db_session, claim.id) == []

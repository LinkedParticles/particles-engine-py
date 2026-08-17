"""Tests for the particle_relations store helpers (§6.10)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from particles.core.schema import RelationCreatedBy, RelationType
from particles.store.relation_store import (
    _canonical_pair,
    _endpoints_for_write,
    create_relation,
    delete_relation,
    get_co_evidential_group,
    get_incoming,
    get_outgoing,
    get_relations_for_particle,
    remove_particle_from_relations,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------


def test_canonical_pair_sorts_lexicographically() -> None:
    assert _canonical_pair("b", "a") == ("a", "b")
    assert _canonical_pair("a", "b") == ("a", "b")
    assert _canonical_pair("a", "a") == ("a", "a")


def test_narrative_kinds_write_endpoints_verbatim() -> None:
    """PART_OF / SEQUENCE_IN are asymmetric: the write seam must
    preserve the supplied endpoint order, not canonicalise it."""
    assert _endpoints_for_write("b", "a", RelationType.PART_OF) == ("b", "a")
    assert _endpoints_for_write("b", "a", RelationType.SEQUENCE_IN) == ("b", "a")


def test_stance_kinds_write_endpoints_verbatim() -> None:
    """ENDORSES / DISPUTES are asymmetric: the stance → target
    endpoint order is stored verbatim, not canonicalised."""
    assert _endpoints_for_write("p:S", "p:T", RelationType.ENDORSES) == ("p:S", "p:T")
    assert _endpoints_for_write("p:S", "p:T", RelationType.DISPUTES) == ("p:S", "p:T")
    # Contrast: a symmetric kind canonicalises.
    assert _endpoints_for_write("b", "a", RelationType.CO_EVIDENTIAL) == ("a", "b")


# ---------------------------------------------------------------------------
# create_relation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_relation_canonicalises_pair(db_session: AsyncSession) -> None:
    """A relation stored as (b, a) must be queryable as (a, b)."""
    await create_relation(
        db_session,
        "particle-bbbbbb",
        "particle-aaaaaa",
        RelationType.CO_EVIDENTIAL,
        RelationCreatedBy.HUMAN_REVIEW,
    )
    rels = await get_relations_for_particle(db_session, "particle-aaaaaa")
    assert len(rels) == 1
    # Canonicalised: particle_a < particle_b
    assert rels[0].particle_a == "particle-aaaaaa"
    assert rels[0].particle_b == "particle-bbbbbb"


@pytest.mark.asyncio
async def test_create_relation_rejects_self_link(db_session: AsyncSession) -> None:
    with pytest.raises(ValueError, match="itself"):
        await create_relation(
            db_session,
            "p:x",
            "p:x",
            RelationType.CO_EVIDENTIAL,
            RelationCreatedBy.HUMAN_REVIEW,
        )


@pytest.mark.asyncio
async def test_create_relation_records_metadata(db_session: AsyncSession) -> None:
    rel = await create_relation(
        db_session,
        "p:a",
        "p:b",
        RelationType.CO_EVIDENTIAL,
        RelationCreatedBy.EXTRACTOR_DIRECT,
        confidence=0.87,
    )
    assert rel.relation_type == RelationType.CO_EVIDENTIAL
    assert rel.created_by == RelationCreatedBy.EXTRACTOR_DIRECT
    assert rel.confidence == 0.87


# ---------------------------------------------------------------------------
# get_co_evidential_group — transitive closure via BFS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_singleton_group_for_unlinked_particle(db_session: AsyncSession) -> None:
    """A particle with no relations forms a one-member group containing itself."""
    group = await get_co_evidential_group(db_session, "p:alone")
    assert group == {"p:alone"}


@pytest.mark.asyncio
async def test_pair_group(db_session: AsyncSession) -> None:
    await create_relation(
        db_session, "p:a", "p:b", RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await db_session.commit()
    assert await get_co_evidential_group(db_session, "p:a") == {"p:a", "p:b"}
    assert await get_co_evidential_group(db_session, "p:b") == {"p:a", "p:b"}


@pytest.mark.asyncio
async def test_transitive_closure_three_hops(db_session: AsyncSession) -> None:
    """a—b—c—d with no direct a–d link still resolves to one group."""
    await create_relation(
        db_session, "p:a", "p:b", RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await create_relation(
        db_session, "p:b", "p:c", RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await create_relation(
        db_session, "p:c", "p:d", RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await db_session.commit()

    full_group = {"p:a", "p:b", "p:c", "p:d"}
    assert await get_co_evidential_group(db_session, "p:a") == full_group
    assert await get_co_evidential_group(db_session, "p:d") == full_group
    assert await get_co_evidential_group(db_session, "p:b") == full_group


@pytest.mark.asyncio
async def test_disjoint_groups_do_not_leak(db_session: AsyncSession) -> None:
    """Two separate clusters should not appear in each other's BFS results."""
    await create_relation(
        db_session, "p:a", "p:b", RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await create_relation(
        db_session, "p:c", "p:d", RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await db_session.commit()

    assert await get_co_evidential_group(db_session, "p:a") == {"p:a", "p:b"}
    assert await get_co_evidential_group(db_session, "p:c") == {"p:c", "p:d"}


# ---------------------------------------------------------------------------
# get_relations_for_particle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_relations_finds_both_endpoints(db_session: AsyncSession) -> None:
    """A particle is found whether it's stored as particle_a or particle_b."""
    # Force the canonical order to put p:m on the b side
    await create_relation(
        db_session, "p:m", "p:a", RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    # And on the a side for another pair
    await create_relation(
        db_session, "p:m", "p:z", RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await db_session.commit()

    rels = await get_relations_for_particle(db_session, "p:m")
    assert len(rels) == 2


# ---------------------------------------------------------------------------
# remove_particle_from_relations — RETRACTION handling per §6.10
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remove_particle_dissolves_pair_group(db_session: AsyncSession) -> None:
    """RETRACTING one member of a 2-particle group leaves the other as a singleton."""
    await create_relation(
        db_session, "p:a", "p:b", RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await db_session.commit()

    deleted = await remove_particle_from_relations(db_session, "p:a")
    await db_session.commit()
    assert deleted == 1

    # The surviving particle now has no relations — group is a singleton.
    assert await get_co_evidential_group(db_session, "p:b") == {"p:b"}


@pytest.mark.asyncio
async def test_remove_particle_preserves_larger_group(db_session: AsyncSession) -> None:
    """RETRACTING one of four members leaves the other three as a 3-particle group."""
    # Star pattern around p:hub
    await create_relation(
        db_session, "p:hub", "p:a", RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await create_relation(
        db_session, "p:hub", "p:b", RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await create_relation(
        db_session, "p:hub", "p:c", RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    # Plus a, b, c interconnected so removing hub does not partition them
    await create_relation(
        db_session, "p:a", "p:b", RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await create_relation(
        db_session, "p:b", "p:c", RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await db_session.commit()

    assert await get_co_evidential_group(db_session, "p:a") == {"p:a", "p:b", "p:c", "p:hub"}

    await remove_particle_from_relations(db_session, "p:hub")
    await db_session.commit()

    assert await get_co_evidential_group(db_session, "p:a") == {"p:a", "p:b", "p:c"}
    assert await get_co_evidential_group(db_session, "p:hub") == {"p:hub"}


@pytest.mark.asyncio
async def test_remove_particle_preserves_stance_edges(db_session: AsyncSession) -> None:
    """(B2): retracting a stance's *target* preserves the
    ENDORSES/DISPUTES edge (dangling), while still deleting co-evidential and
    narrative edges incident to the target."""
    # Stance S endorses target T; T is also co-evidential with T2.
    await create_relation(
        db_session, "p:S", "p:T", RelationType.ENDORSES, RelationCreatedBy.EXTRACTOR_DIRECT
    )
    await create_relation(
        db_session, "p:T", "p:T2", RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await db_session.commit()

    deleted = await remove_particle_from_relations(db_session, "p:T")
    await db_session.commit()

    # Only the CO_EVIDENTIAL edge was deleted; the stance edge survives dangling.
    assert deleted == 1
    rels = await get_relations_for_particle(db_session, "p:S")
    assert len(rels) == 1
    assert rels[0].relation_type == RelationType.ENDORSES
    assert (rels[0].particle_a, rels[0].particle_b) == ("p:S", "p:T")


@pytest.mark.asyncio
async def test_remove_stance_particle_preserves_own_edge(db_session: AsyncSession) -> None:
    """Retracting the *stance* particle itself also leaves the edge dangling
    rather than silently hard-deleting it."""
    await create_relation(
        db_session, "p:S", "p:T", RelationType.DISPUTES, RelationCreatedBy.EXTRACTOR_DIRECT
    )
    await db_session.commit()

    deleted = await remove_particle_from_relations(db_session, "p:S")
    await db_session.commit()

    assert deleted == 0
    rels = await get_relations_for_particle(db_session, "p:T")
    assert len(rels) == 1
    assert rels[0].relation_type == RelationType.DISPUTES


# ---------------------------------------------------------------------------
# delete_relation — targeted unlink
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_relation_works_regardless_of_argument_order(
    db_session: AsyncSession,
) -> None:
    await create_relation(
        db_session, "p:a", "p:b", RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await db_session.commit()

    # Delete with reversed argument order — should still find the canonical row
    removed = await delete_relation(db_session, "p:b", "p:a", RelationType.CO_EVIDENTIAL)
    assert removed is True
    assert await get_relations_for_particle(db_session, "p:a") == []


@pytest.mark.asyncio
async def test_delete_relation_returns_false_when_missing(db_session: AsyncSession) -> None:
    removed = await delete_relation(db_session, "p:nope", "p:also", RelationType.CO_EVIDENTIAL)
    assert removed is False


# ---------------------------------------------------------------------------
# Duplicate-insertion contract: canonicalised unique constraint catches it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_pair_rejected_regardless_of_order(db_session: AsyncSession) -> None:
    """Inserting (a, b) then (b, a) of the same type must fail the unique constraint."""
    import sqlalchemy.exc

    await create_relation(
        db_session, "p:a", "p:b", RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await db_session.commit()

    with pytest.raises(sqlalchemy.exc.IntegrityError):
        await create_relation(
            db_session, "p:b", "p:a", RelationType.CO_EVIDENTIAL, RelationCreatedBy.EXTRACTOR_DIRECT
        )
        await db_session.commit()


# ---------------------------------------------------------------------------
# kind-aware canonicalisation
# ---------------------------------------------------------------------------


class TestSymmetricKindsRegistry:
    """The ``_SYMMETRIC_KINDS`` constant is the single source of truth
    for which kinds get endpoint-canonicalised on write."""

    def test_known_symmetric_kinds(self) -> None:
        from particles.store.relation_store import _SYMMETRIC_KINDS

        # CO_EVIDENTIAL (ACTIVE) + CONTRADICTS (RESERVED)
        # are the two symmetric kinds. Asymmetric kinds (BOOSTS, QUOTES,
        # REPLIES_TO, MENTIONS) MUST NOT appear here — silently swapping
        # their endpoints would lose the direction of the relationship.
        assert RelationType.CO_EVIDENTIAL in _SYMMETRIC_KINDS
        assert RelationType.CONTRADICTS in _SYMMETRIC_KINDS
        assert RelationType.BOOSTS not in _SYMMETRIC_KINDS
        assert RelationType.QUOTES not in _SYMMETRIC_KINDS
        assert RelationType.REPLIES_TO not in _SYMMETRIC_KINDS
        # stance edges carry direction (stance → target).
        assert RelationType.ENDORSES not in _SYMMETRIC_KINDS
        assert RelationType.DISPUTES not in _SYMMETRIC_KINDS
        assert RelationType.MENTIONS not in _SYMMETRIC_KINDS


class TestAsymmetricKindPreservesDirection:
    """an asymmetric kind (BOOSTS) MUST preserve the
    operator-supplied endpoint order. "A boosts B" ≠ "B boosts A"."""

    @pytest.mark.asyncio
    async def test_boosts_preserves_endpoint_order(self, db_session: AsyncSession) -> None:
        # Pass them deliberately in non-canonical lexicographic order
        # ("zzz" > "aaa") — canonicalisation, if it fired, would swap them.
        await create_relation(
            db_session,
            "particle-zzz",
            "particle-aaa",
            RelationType.BOOSTS,
            RelationCreatedBy.EXTRACTOR_DIRECT,
        )
        await db_session.commit()
        rels = await get_relations_for_particle(db_session, "particle-zzz")
        assert len(rels) == 1
        # Direction preserved: zzz is still the booster, aaa is the boosted.
        assert rels[0].particle_a == "particle-zzz"
        assert rels[0].particle_b == "particle-aaa"

    @pytest.mark.asyncio
    async def test_boosts_both_directions_are_distinct_rows(self, db_session: AsyncSession) -> None:
        """Asymmetric kinds permit BOTH ``(A, B)`` and ``(B, A)`` —
        they're different facts ("A boosts B" vs "B boosts A")."""
        await create_relation(
            db_session,
            "particle-aaa",
            "particle-bbb",
            RelationType.BOOSTS,
            RelationCreatedBy.EXTRACTOR_DIRECT,
        )
        await create_relation(
            db_session,
            "particle-bbb",
            "particle-aaa",
            RelationType.BOOSTS,
            RelationCreatedBy.EXTRACTOR_DIRECT,
        )
        await db_session.commit()
        rels = await get_relations_for_particle(
            db_session, "particle-aaa", relation_type=RelationType.BOOSTS
        )
        # Both directions are queryable; they're not duplicates.
        assert len(rels) == 2
        pairs = {(r.particle_a, r.particle_b) for r in rels}
        assert pairs == {("particle-aaa", "particle-bbb"), ("particle-bbb", "particle-aaa")}

    @pytest.mark.asyncio
    async def test_delete_boosts_respects_direction(self, db_session: AsyncSession) -> None:
        """Deleting "A boosts B" MUST NOT delete "B boosts A"."""
        await create_relation(
            db_session,
            "particle-aaa",
            "particle-bbb",
            RelationType.BOOSTS,
            RelationCreatedBy.EXTRACTOR_DIRECT,
        )
        await create_relation(
            db_session,
            "particle-bbb",
            "particle-aaa",
            RelationType.BOOSTS,
            RelationCreatedBy.EXTRACTOR_DIRECT,
        )
        await db_session.commit()

        # Delete only the (aaa, bbb) direction.
        deleted = await delete_relation(
            db_session, "particle-aaa", "particle-bbb", RelationType.BOOSTS
        )
        assert deleted is True

        # The reverse direction survives.
        rels = await get_relations_for_particle(
            db_session, "particle-aaa", relation_type=RelationType.BOOSTS
        )
        assert len(rels) == 1
        assert rels[0].particle_a == "particle-bbb"
        assert rels[0].particle_b == "particle-aaa"


class TestSymmetricKindContradictsCanonicalises:
    """``CONTRADICTS`` is RESERVED but symmetric — when a future ADR
    activates it the canonicalisation invariant should already hold."""

    @pytest.mark.asyncio
    async def test_contradicts_canonicalises(self, db_session: AsyncSession) -> None:
        await create_relation(
            db_session,
            "particle-zzz",
            "particle-aaa",
            RelationType.CONTRADICTS,
            RelationCreatedBy.HUMAN_REVIEW,
        )
        await db_session.commit()
        rels = await get_relations_for_particle(db_session, "particle-aaa")
        assert len(rels) == 1
        # Canonical: lexicographically smaller endpoint first.
        assert rels[0].particle_a == "particle-aaa"
        assert rels[0].particle_b == "particle-zzz"


# ---------------------------------------------------------------------------
# Directed accessors — get_incoming / get_outgoing
# ---------------------------------------------------------------------------


class TestDirectedAccessors:
    """``get_incoming`` / ``get_outgoing`` are the directed-edge queries the
    narrative traversal builds on. They must respect endpoint
    direction for asymmetric kinds and return ids sorted."""

    @pytest.mark.asyncio
    async def test_incoming_and_outgoing_are_directional(self, db_session: AsyncSession) -> None:
        # Two constituents PART_OF one narrative (constituent → narrative).
        await create_relation(
            db_session, "p:c1", "p:nar", RelationType.PART_OF, RelationCreatedBy.MANUAL_CLI
        )
        await create_relation(
            db_session, "p:c2", "p:nar", RelationType.PART_OF, RelationCreatedBy.MANUAL_CLI
        )
        await db_session.flush()

        # Incoming to the narrative = its constituents (sorted).
        assert await get_incoming(db_session, "p:nar", RelationType.PART_OF) == ["p:c1", "p:c2"]
        # Outgoing from a constituent = the narrative it belongs to.
        assert await get_outgoing(db_session, "p:c1", RelationType.PART_OF) == ["p:nar"]
        # The narrative has no PART_OF *outgoing* edge; a constituent has no
        # PART_OF *incoming* edge — direction is not conflated.
        assert await get_outgoing(db_session, "p:nar", RelationType.PART_OF) == []
        assert await get_incoming(db_session, "p:c1", RelationType.PART_OF) == []

    @pytest.mark.asyncio
    async def test_accessors_filter_by_relation_type(self, db_session: AsyncSession) -> None:
        await create_relation(
            db_session, "p:a", "p:b", RelationType.SEQUENCE_IN, RelationCreatedBy.MANUAL_CLI
        )
        await db_session.flush()
        assert await get_outgoing(db_session, "p:a", RelationType.SEQUENCE_IN) == ["p:b"]
        # A different kind sees nothing.
        assert await get_outgoing(db_session, "p:a", RelationType.PART_OF) == []

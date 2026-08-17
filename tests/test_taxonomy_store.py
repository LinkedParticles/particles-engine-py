"""Repository helper tests for taxonomy_store."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import (
    Confidence,
    Particle,
    TagNode,
    TaxonomyDefinition,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.store.particle_store import insert_particle
from particles.store.taxonomy_store import (
    add_particle_tags,
    expand_tags,
    get_particle_ids_for_tags,
    get_taxonomy,
    insert_taxonomy,
    list_taxonomies,
    remove_particle_tags,
    set_particle_tags,
    tag_exists,
)


def _coins_taxonomy() -> TaxonomyDefinition:
    return TaxonomyDefinition(
        name="Coins",
        version="1.0.0",
        author="tester",
        tags=[
            TagNode(tag="coins"),
            TagNode(tag="coins/by-region", parent="coins"),
            TagNode(tag="coins/by-region/germany", parent="coins/by-region"),
            TagNode(tag="coins/by-region/usa", parent="coins/by-region"),
            TagNode(tag="coins/by-period", parent="coins"),
            TagNode(tag="coins/by-period/medieval", parent="coins/by-period"),
        ],
    )


def _ml_taxonomy() -> TaxonomyDefinition:
    return TaxonomyDefinition(
        name="ML",
        version="0.1.0",
        author="tester",
        tags=[
            TagNode(tag="ml"),
            TagNode(tag="ml/optimizers", parent="ml"),
        ],
    )


def _make_particle(content: str = "claim") -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
    )


class TestInsertTaxonomy:
    @pytest.mark.asyncio
    async def test_round_trip_with_nodes(self, db_session: AsyncSession) -> None:
        td = _coins_taxonomy()
        await insert_taxonomy(db_session, td)
        loaded = await get_taxonomy(db_session, td.taxonomy_id)
        assert loaded is not None
        assert loaded.name == "Coins"
        assert {n.tag for n in loaded.tags} == {n.tag for n in td.tags}

    @pytest.mark.asyncio
    async def test_list_taxonomies(self, db_session: AsyncSession) -> None:
        await insert_taxonomy(db_session, _coins_taxonomy())
        await insert_taxonomy(db_session, _ml_taxonomy())
        listed = await list_taxonomies(db_session)
        assert {t.name for t in listed} == {"Coins", "ML"}


class TestExpandTags:
    @pytest.mark.asyncio
    async def test_subtree_expansion(self, db_session: AsyncSession) -> None:
        await insert_taxonomy(db_session, _coins_taxonomy())
        expanded = await expand_tags(db_session, ["coins"])
        assert expanded == {
            "coins",
            "coins/by-region",
            "coins/by-region/germany",
            "coins/by-region/usa",
            "coins/by-period",
            "coins/by-period/medieval",
        }

    @pytest.mark.asyncio
    async def test_subtree_from_intermediate(self, db_session: AsyncSession) -> None:
        await insert_taxonomy(db_session, _coins_taxonomy())
        expanded = await expand_tags(db_session, ["coins/by-region"])
        assert expanded == {
            "coins/by-region",
            "coins/by-region/germany",
            "coins/by-region/usa",
        }

    @pytest.mark.asyncio
    async def test_leaf_returns_only_self(self, db_session: AsyncSession) -> None:
        await insert_taxonomy(db_session, _coins_taxonomy())
        expanded = await expand_tags(db_session, ["coins/by-region/germany"])
        assert expanded == {"coins/by-region/germany"}

    @pytest.mark.asyncio
    async def test_unknown_tag_passes_through(self, db_session: AsyncSession) -> None:
        await insert_taxonomy(db_session, _coins_taxonomy())
        expanded = await expand_tags(db_session, ["cold-war"])
        assert expanded == {"cold-war"}

    @pytest.mark.asyncio
    async def test_union_across_taxonomies(self, db_session: AsyncSession) -> None:
        await insert_taxonomy(db_session, _coins_taxonomy())
        await insert_taxonomy(db_session, _ml_taxonomy())
        expanded = await expand_tags(db_session, ["coins", "ml"])
        assert "coins/by-region/germany" in expanded
        assert "ml/optimizers" in expanded

    #: up-expansion over the parent chain.
    @pytest.mark.asyncio
    async def test_include_ancestors_adds_parent_chain(self, db_session: AsyncSession) -> None:
        await insert_taxonomy(db_session, _coins_taxonomy())
        expanded = await expand_tags(
            db_session, ["coins/by-region/germany"], include_ancestors=True
        )
        # The leaf plus its parent chain — no sibling branches.
        assert expanded == {
            "coins/by-region/germany",
            "coins/by-region",
            "coins",
        }

    @pytest.mark.asyncio
    async def test_include_ancestors_unions_with_subtree(self, db_session: AsyncSession) -> None:
        await insert_taxonomy(db_session, _coins_taxonomy())
        expanded = await expand_tags(db_session, ["coins/by-region"], include_ancestors=True)
        # Down: the subtree. Up: the ``coins`` root. NOT the sibling by-period branch.
        assert "coins/by-region/usa" in expanded  # subtree (down)
        assert "coins" in expanded  # ancestor (up)
        assert "coins/by-period" not in expanded  # sibling branch not pulled in

    @pytest.mark.asyncio
    async def test_ancestors_off_by_default(self, db_session: AsyncSession) -> None:
        await insert_taxonomy(db_session, _coins_taxonomy())
        expanded = await expand_tags(db_session, ["coins/by-region/germany"])
        assert expanded == {"coins/by-region/germany"}  # no ancestors without the flag


class TestTagExists:
    @pytest.mark.asyncio
    async def test_known_and_unknown(self, db_session: AsyncSession) -> None:
        await insert_taxonomy(db_session, _coins_taxonomy())
        assert await tag_exists(db_session, "coins/by-region/germany") is True
        assert await tag_exists(db_session, "not-in-any-taxonomy") is False


class TestParticleTagging:
    @pytest.mark.asyncio
    async def test_set_creates_edges_and_round_trips(self, db_session: AsyncSession) -> None:
        p = _make_particle()
        await insert_particle(db_session, p)
        await set_particle_tags(db_session, p.id, ["coins", "ml/optimizers"])

        ids = await get_particle_ids_for_tags(db_session, {"coins"})
        assert p.id in ids
        ids2 = await get_particle_ids_for_tags(db_session, {"ml/optimizers"})
        assert p.id in ids2

        # ParticleRow.tags_json round-trips into Particle.tags
        from particles.store.particle_store import get_particle

        reloaded = await get_particle(db_session, p.id)
        assert reloaded is not None
        assert reloaded.tags == ["coins", "ml/optimizers"]

    @pytest.mark.asyncio
    async def test_set_replaces_existing(self, db_session: AsyncSession) -> None:
        p = _make_particle()
        await insert_particle(db_session, p)
        await set_particle_tags(db_session, p.id, ["coins"])
        await set_particle_tags(db_session, p.id, ["ml"])

        assert await get_particle_ids_for_tags(db_session, {"coins"}) == set()
        assert p.id in await get_particle_ids_for_tags(db_session, {"ml"})

    @pytest.mark.asyncio
    async def test_set_dedupes_preserves_order(self, db_session: AsyncSession) -> None:
        p = _make_particle()
        await insert_particle(db_session, p)
        await set_particle_tags(db_session, p.id, ["a", "b", "a"])

        from particles.store.particle_store import get_particle

        reloaded = await get_particle(db_session, p.id)
        assert reloaded is not None
        assert reloaded.tags == ["a", "b"]

    @pytest.mark.asyncio
    async def test_add_is_idempotent(self, db_session: AsyncSession) -> None:
        p = _make_particle()
        await insert_particle(db_session, p)
        added = await add_particle_tags(db_session, p.id, ["coins"])
        assert added == ["coins"]
        again = await add_particle_tags(db_session, p.id, ["coins"])
        assert again == []

    @pytest.mark.asyncio
    async def test_remove(self, db_session: AsyncSession) -> None:
        p = _make_particle()
        await insert_particle(db_session, p)
        await set_particle_tags(db_session, p.id, ["coins", "ml"])
        removed = await remove_particle_tags(db_session, p.id, ["ml"])
        assert removed == ["ml"]

        ids = await get_particle_ids_for_tags(db_session, {"ml"})
        assert ids == set()
        ids = await get_particle_ids_for_tags(db_session, {"coins"})
        assert p.id in ids

    @pytest.mark.asyncio
    async def test_remove_missing_is_noop(self, db_session: AsyncSession) -> None:
        p = _make_particle()
        await insert_particle(db_session, p)
        removed = await remove_particle_tags(db_session, p.id, ["nope"])
        assert removed == []

    @pytest.mark.asyncio
    async def test_missing_particle_raises(self, db_session: AsyncSession) -> None:
        with pytest.raises(ValueError):
            await set_particle_tags(db_session, "no-such-id", ["x"])

"""Tests for the embedding-model marker + cosine mismatch guard.

Stored vectors carry the id of the model that produced them; the cosine query
path refuses to compare vectors across embedding spaces (a vector embedded
under model A is meaningless against a query embedded under model B). NULL is a
legacy row, grandfathered as the historical default. This mirrors the
schema_version guard.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles import embeddings as ep
from particles.core.schema import Confidence, Particle, UncertaintyNature
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.embeddings import EmbeddingProfile
from particles.store.particle_store import (
    LEGACY_EMBEDDING_MODEL_ID,
    ParticleRow,
    get_active_particles_with_embeddings,
    get_active_particles_with_stale_embedding_model,
    get_store_embedding_profile,
    insert_particle,
)


def _make_particle() -> Particle:
    return Particle(
        id=str(uuid.uuid4()),
        content="A test claim about a thing.",
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
    )


class TestEmbeddingMarkerStamp:
    def test_from_model_stamps_current_id_when_embedding_present(self) -> None:
        row = ParticleRow.from_model(_make_particle(), embedding=[0.1, 0.2, 0.3])
        assert row.embedding_model_id == ep.get_embedding_model_id()

    def test_from_model_leaves_id_none_without_embedding(self) -> None:
        row = ParticleRow.from_model(_make_particle(), embedding=None)
        assert row.embedding_model_id is None


class TestCosineGuard:
    @pytest.mark.asyncio
    async def test_current_and_legacy_null_are_returned(self, db_session: AsyncSession) -> None:
        # Current-id row (stamped on write) + a legacy NULL row.
        await insert_particle(db_session, _make_particle(), embedding=[1.0, 0.0])
        legacy = _make_particle()
        await insert_particle(db_session, legacy, embedding=[0.0, 1.0])
        # Simulate a pre-marker row: NULL embedding_model_id.
        row = await db_session.get(ParticleRow, legacy.id)
        assert row is not None
        row.embedding_model_id = None
        await db_session.flush()

        out = await get_active_particles_with_embeddings(db_session)
        # Both returned: current id matches, NULL is grandfathered to the default.
        assert len(out) == 2

    @pytest.mark.asyncio
    async def test_mismatched_model_id_is_skipped(self, db_session: AsyncSession) -> None:
        good = _make_particle()
        await insert_particle(db_session, good, embedding=[1.0, 0.0])
        stale = _make_particle()
        await insert_particle(db_session, stale, embedding=[0.0, 1.0])
        row = await db_session.get(ParticleRow, stale.id)
        assert row is not None
        row.embedding_model_id = "some-other-embedding-model"
        await db_session.flush()

        out = await get_active_particles_with_embeddings(db_session)
        ids = {p.id for p, _ in out}
        assert good.id in ids
        assert stale.id not in ids  # skipped — different embedding space

    @pytest.mark.asyncio
    async def test_stale_finder_reports_mismatches(self, db_session: AsyncSession) -> None:
        current = ep.get_embedding_model_id()
        matching = _make_particle()
        await insert_particle(db_session, matching, embedding=[1.0, 0.0])
        stale = _make_particle()
        await insert_particle(db_session, stale, embedding=[0.0, 1.0])
        row = await db_session.get(ParticleRow, stale.id)
        assert row is not None
        row.embedding_model_id = "old-model"
        await db_session.flush()

        found = await get_active_particles_with_stale_embedding_model(db_session, current)
        found_ids = {p.id for p in found}
        assert stale.id in found_ids
        assert matching.id not in found_ids

    @pytest.mark.asyncio
    async def test_legacy_null_is_stale_after_swap(self, db_session: AsyncSession) -> None:
        # A NULL row is grandfathered to LEGACY_EMBEDDING_MODEL_ID, so it counts
        # as stale once the current model differs from that default — the swap
        # case the guard exists to catch.
        legacy = _make_particle()
        await insert_particle(db_session, legacy, embedding=[1.0, 0.0])
        row = await db_session.get(ParticleRow, legacy.id)
        assert row is not None
        row.embedding_model_id = None
        await db_session.flush()

        found = await get_active_particles_with_stale_embedding_model(
            db_session, "a-new-embedding-model"
        )
        assert legacy.id in {p.id for p in found}
        # ...but not stale against the legacy default itself.
        none_found = await get_active_particles_with_stale_embedding_model(
            db_session, LEGACY_EMBEDDING_MODEL_ID
        )
        assert legacy.id not in {p.id for p in none_found}


class TestStoreEmbeddingProfile:
    """The structured {model, dim, normalization} profile recorded in store metadata."""

    @pytest.mark.asyncio
    async def test_none_when_no_embedded_particle(self, db_session: AsyncSession) -> None:
        # A particle without an embedding does not establish a store profile.
        await insert_particle(db_session, _make_particle(), embedding=None)
        assert await get_store_embedding_profile(db_session) is None

    @pytest.mark.asyncio
    async def test_round_trips_structured_profile(self, db_session: AsyncSession) -> None:
        await insert_particle(db_session, _make_particle(), embedding=[1.0, 0.0])
        profile = await get_store_embedding_profile(db_session)
        assert profile == EmbeddingProfile(
            model=ep.get_embedding_model_id(), dim=384, normalization="l2"
        )
        # Structured object, never a bare string (resolved question 5).
        assert profile is not None
        assert profile.as_dict() == {
            "model": ep.get_embedding_model_id(),
            "dim": 384,
            "normalization": "l2",
        }

    @pytest.mark.asyncio
    async def test_legacy_null_model_grandfathered(self, db_session: AsyncSession) -> None:
        legacy = _make_particle()
        await insert_particle(db_session, legacy, embedding=[1.0, 0.0])
        row = await db_session.get(ParticleRow, legacy.id)
        assert row is not None
        row.embedding_model_id = None
        await db_session.flush()

        profile = await get_store_embedding_profile(db_session)
        assert profile is not None
        assert profile.model == LEGACY_EMBEDDING_MODEL_ID

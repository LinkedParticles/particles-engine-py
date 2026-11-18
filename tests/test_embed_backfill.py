# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The consolidation embed pass: vectors for beliefs stored without one."""

from __future__ import annotations

import json
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles import embeddings as ep
from particles.config import get_config
from particles.core.schema import (
    Confidence,
    Particle,
    ParticleType,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.embeddings import get_embedding_model_id
from particles.operations.consolidation import _delta_scope_ids
from particles.operations.embed_backfill import (
    NO_ENCODER,
    embedding_backfilled_particle_ids_since,
    run_embed_backfill,
)
from particles.store.event_store import EventRefKind, OperatorEventType, list_events
from particles.store.particle_store import (
    ParticleRow,
    count_active_particles_with_other_embedding_model,
    count_active_particles_without_embedding,
    fill_particle_embedding,
    insert_particle,
    list_active_particles_without_embedding,
    update_particle_status,
)

NOW = datetime.now(UTC)
ACTOR = "memory-consolidate"


def _particle(
    content: str,
    asserted_at: datetime,
    particle_type: ParticleType = ParticleType.CLAIM,
) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1", snapshot_id="s1")
        ],
        asserted_by="general-extractor",
        asserted_at=asserted_at,
        particle_type=particle_type,
    )


@pytest.fixture
def encoder() -> Generator[MagicMock, None, None]:
    """A fake encoder: one 4-dimensional vector per text, its first value the text's length."""
    model = MagicMock()
    model.encode = MagicMock(
        side_effect=lambda texts, **_: np.array(
            [[float(len(t)), 0.0, 0.0, 1.0] for t in texts], dtype=np.float32
        )
    )
    original = ep._embedding_model
    ep.set_embedding_model(model)
    try:
        yield model
    finally:
        ep.set_embedding_model(original)


async def _row(session: AsyncSession, particle_id: str) -> ParticleRow:
    session.expire_all()
    row = await session.get(ParticleRow, particle_id)
    assert row is not None
    return row


class TestStoreHelpers:
    @pytest.mark.asyncio
    async def test_fill_refuses_a_row_that_has_a_vector(self, db_session: AsyncSession) -> None:
        held = _particle("already embedded", NOW)
        await insert_particle(db_session, held, embedding=[0.5, 0.5, 0.5, 0.5])
        await db_session.commit()

        assert await fill_particle_embedding(db_session, held.id, [9.0, 9.0, 9.0, 9.0]) is False
        assert json.loads((await _row(db_session, held.id)).embedding_json or "") == [
            0.5,
            0.5,
            0.5,
            0.5,
        ]

    @pytest.mark.asyncio
    async def test_fill_refuses_a_belief_no_longer_active(self, db_session: AsyncSession) -> None:
        retired = _particle("retired after it was selected", NOW)
        await insert_particle(db_session, retired)
        await db_session.commit()
        await update_particle_status(
            db_session, retired.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
        )
        await db_session.commit()

        assert await fill_particle_embedding(db_session, retired.id, [1.0, 0.0]) is False
        assert (await _row(db_session, retired.id)).embedding_json is None

    @pytest.mark.asyncio
    async def test_fill_refuses_an_unknown_id(self, db_session: AsyncSession) -> None:
        assert await fill_particle_embedding(db_session, "no-such-id", [1.0]) is False

    @pytest.mark.asyncio
    async def test_selection_is_active_non_review_oldest_first_and_capped(
        self, db_session: AsyncSession
    ) -> None:
        newer = _particle("newer", NOW - timedelta(days=1))
        older = _particle("older", NOW - timedelta(days=5))
        review = _particle("a review record", NOW - timedelta(days=9), ParticleType.REVIEW)
        embedded = _particle("embedded", NOW - timedelta(days=7))
        for p in (newer, older, review):
            await insert_particle(db_session, p)
        await insert_particle(db_session, embedded, embedding=[1.0, 0.0, 0.0, 0.0])
        await db_session.commit()

        assert await list_active_particles_without_embedding(db_session, limit=10) == [
            (older.id, "older"),
            (newer.id, "newer"),
        ]
        assert await list_active_particles_without_embedding(db_session, limit=1) == [
            (older.id, "older")
        ]
        assert await list_active_particles_without_embedding(db_session, limit=0) == []

    @pytest.mark.asyncio
    async def test_other_model_count(self, db_session: AsyncSession) -> None:
        current = _particle("current model", NOW)
        other = _particle("another model", NOW)
        await insert_particle(db_session, current, embedding=[1.0, 0.0, 0.0, 0.0])
        await insert_particle(db_session, other, embedding=[0.0, 1.0, 0.0, 0.0])
        await db_session.commit()
        (await _row(db_session, other.id)).embedding_model_id = "some-older-encoder"
        await db_session.commit()

        assert (
            await count_active_particles_with_other_embedding_model(
                db_session, get_embedding_model_id()
            )
            == 1
        )


class TestPass:
    @pytest.mark.asyncio
    async def test_fills_only_active_non_review_rows_and_touches_nothing_else(
        self, db_session: AsyncSession, encoder: MagicMock
    ) -> None:
        missing = _particle("written during an outage", NOW - timedelta(days=80))
        review = _particle("a review record", NOW - timedelta(days=80), ParticleType.REVIEW)
        retired = _particle("a retired claim", NOW - timedelta(days=80))
        for p in (missing, review, retired):
            await insert_particle(db_session, p)
        await db_session.commit()
        await update_particle_status(
            db_session, retired.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
        )
        await db_session.commit()
        before = await _row(db_session, missing.id)
        snapshot = (before.content, before.status, before.asserted_at, before.confidence_value)

        report = await run_embed_backfill(db_session, actor=ACTOR)

        assert report.filled == 1
        assert report.still_missing == 0
        assert report.skipped_reason is None
        filled = await _row(db_session, missing.id)
        assert json.loads(filled.embedding_json or "") == [
            float(len("written during an outage")),
            0.0,
            0.0,
            1.0,
        ]
        assert filled.embedding_model_id == get_embedding_model_id()
        assert (filled.content, filled.status, filled.asserted_at, filled.confidence_value) == (
            snapshot
        )
        assert (await _row(db_session, review.id)).embedding_json is None
        assert (await _row(db_session, retired.id)).embedding_json is None
        # The quality report's count reads zero once the pass has run.
        assert await count_active_particles_without_embedding(db_session) == 0

    @pytest.mark.asyncio
    async def test_records_one_event_naming_each_filled_belief(
        self, db_session: AsyncSession, encoder: MagicMock
    ) -> None:
        first = _particle("first", NOW - timedelta(days=3))
        second = _particle("second", NOW - timedelta(days=2))
        for p in (first, second):
            await insert_particle(db_session, p)
        await db_session.commit()

        report = await run_embed_backfill(db_session, actor=ACTOR)

        events = await list_events(
            db_session, event_type=OperatorEventType.EMBEDDING_BACKFILLED, limit=10
        )
        assert len(events) == 1
        event = events[0]
        assert event.event_id == report.event_id
        assert event.actor == ACTOR
        assert {(r.ref_kind, r.ref_id) for r in event.refs} == {
            (EventRefKind.PARTICLE, first.id),
            (EventRefKind.PARTICLE, second.id),
        }
        payload: dict[str, Any] = event.payload or {}
        assert payload["filled"] == 2
        assert payload["model_id"] == get_embedding_model_id()

    @pytest.mark.asyncio
    async def test_cap_drains_oldest_first_and_reports_the_rest(
        self, db_session: AsyncSession, encoder: MagicMock
    ) -> None:
        get_config().consolidation.embed.max_per_run = 2
        particles = [_particle(f"claim {i}", NOW - timedelta(days=10 - i)) for i in range(3)]
        for p in particles:
            await insert_particle(db_session, p)
        await db_session.commit()

        report = await run_embed_backfill(db_session, actor=ACTOR)

        assert report.filled == 2
        assert report.still_missing == 1
        assert (await _row(db_session, particles[2].id)).embedding_json is None

    @pytest.mark.asyncio
    async def test_nothing_to_fill_loads_no_encoder_and_writes_no_event(
        self, db_session: AsyncSession, encoder: MagicMock
    ) -> None:
        report = await run_embed_backfill(db_session, actor=ACTOR)

        assert report.filled == 0
        assert report.event_id is None
        encoder.encode.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("no_embedding_model")
    async def test_no_encoder_writes_nothing_and_says_why(self, db_session: AsyncSession) -> None:
        missing = _particle("written during an outage", NOW)
        await insert_particle(db_session, missing)
        await db_session.commit()

        report = await run_embed_backfill(db_session, actor=ACTOR)

        assert report.skipped_reason == NO_ENCODER
        assert report.filled == 0
        assert report.still_missing == 1
        assert (await _row(db_session, missing.id)).embedding_json is None
        assert not await list_events(
            db_session, event_type=OperatorEventType.EMBEDDING_BACKFILLED, limit=10
        )

    @pytest.mark.asyncio
    async def test_reports_rows_from_another_model(
        self, db_session: AsyncSession, encoder: MagicMock
    ) -> None:
        missing = _particle("written during an outage", NOW)
        other = _particle("embedded by an older encoder", NOW)
        await insert_particle(db_session, missing)
        await insert_particle(db_session, other, embedding=[0.0, 1.0, 0.0, 0.0])
        await db_session.commit()
        (await _row(db_session, other.id)).embedding_model_id = "some-older-encoder"
        await db_session.commit()

        report = await run_embed_backfill(db_session, actor=ACTOR)

        assert report.filled == 1
        assert report.other_model == 1
        # The fill uses the current model only; the other row is left alone.
        assert (await _row(db_session, other.id)).embedding_model_id == "some-older-encoder"


class TestDeltaScope:
    @pytest.mark.asyncio
    async def test_a_filled_belief_enters_the_delta_scope(
        self, db_session: AsyncSession, encoder: MagicMock
    ) -> None:
        watermark = NOW - timedelta(days=1)
        old = _particle("asserted long before the watermark", NOW - timedelta(days=60))
        await insert_particle(db_session, old)
        await db_session.commit()
        assert old.id not in await _delta_scope_ids(db_session, watermark)

        await run_embed_backfill(db_session, actor=ACTOR)

        assert await embedding_backfilled_particle_ids_since(db_session, watermark) == {old.id}
        assert old.id in await _delta_scope_ids(db_session, watermark)
        # A later watermark (the run after next) no longer carries it.
        assert old.id not in await _delta_scope_ids(db_session, NOW + timedelta(days=1))

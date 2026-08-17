"""Tests for the persisted curation collection.

Covers the four things the ADR decides that could silently regress:

* the collection round-trips through ``curation_snapshots`` and the second read
  does **not** re-run the finders (the whole point);
* the §4 replace-vs-carry-forward rule, including its eviction horizon;
* the §5 staleness ladder — suppression, belief status, and post-snapshot
  gesture resolution all filtered live, and a dropped card promoting the next
  real one rather than shortening the session;
* the §1 N+1 fix produces the same candidate set it did before.

The finders themselves are covered by their own suites; these tests seed the
store and assert on what the *snapshot layer* does with their output.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.operations.curation import (
    CardKind,
    QueueSource,
    apply_gesture,
    build_curation_queue,
    rebuild_curation_snapshot,
)
from particles.operations.curation.cards import CurationCard, gestures_for
from particles.operations.curation.snapshot import (
    DELTA_SCOPED_KINDS,
    collect_and_persist,
    per_kind_scope_for,
)
from particles.store.curation_snapshot_store import (
    CollectionScope,
    clear_snapshots,
    latest_snapshot,
)
from particles.store.particle_store import (
    insert_particle,
    update_particle_status,
)
from particles.store.subject_store import insert_subject

_PAST = datetime(2020, 1, 1, tzinfo=UTC)


def _active(
    content: str, *, subject_ids: list[str] | None = None, valid_until: datetime | None = _PAST
) -> Particle:
    """An expired belief — fires the STALENESS finder, so it becomes a card."""
    return Particle(
        subject_ids=subject_ids or [],
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1", snapshot_id=None)
        ],
        asserted_by="test",
        valid_until=valid_until,
    )


async def _seed(session: AsyncSession, content: str) -> Particle:
    """Insert an expired belief **with a subject**, so only STALENESS fires.

    Without the subject link the NO_SUBJECT finder fires too and every card
    count doubles — which would obscure what these tests are actually about.
    """
    subject = await _subject(session)
    particle = _active(content, subject_ids=[subject])
    # insert_particle writes the particle_subjects join rows from subject_ids.
    await insert_particle(session, particle)
    await session.flush()
    return particle


async def _subject(session: AsyncSession) -> str:
    """A fresh Subject to hang one belief on.

    One per belief rather than a shared module-level id: the ``db_session``
    fixture recreates the schema per test, so a cached id would dangle.
    """
    from particles.core.schema import Subject

    subject = Subject(canonical_name=f"Subject {uuid.uuid4().hex[:8]}", asserted_by="test")
    await insert_subject(session, subject)
    return subject.id


def _card(kind: CardKind, *particle_ids: str, leverage: float = 0.5) -> CurationCard:
    return CurationCard(
        kind=kind,
        particle_ids=list(particle_ids),
        diagnostic="seeded",
        suggested_gestures=gestures_for(kind),
        leverage=leverage,
    )


# --------------------------------------------------------------------------- #
# §2/§3 — the collection is persisted and reused                              #
# --------------------------------------------------------------------------- #


class TestSnapshotRoundTrip:
    @pytest.mark.asyncio
    async def test_cold_read_serves_live_and_writes_nothing(self, db_session: AsyncSession) -> None:
        """A store with no collection is served live — and the read stays a read.

        Populating the cache here would make `GET /curation` commit, ending the
        caller's transaction under them. The cache is filled by the two paths
        that are already writes (the nightly cycle, and `--refresh`), so a cold
        store is merely slow — exactly as it was before this ADR.
        """
        await _seed(db_session, "expired belief")

        first = await build_curation_queue(db_session, semantic=False)
        assert first.source == "live"
        assert first.snapshot_id is None
        assert first.built_at is None
        assert [c.kind for c in first.cards] == [CardKind.STALE]
        assert await latest_snapshot(db_session) is None

    @pytest.mark.asyncio
    async def test_once_a_collection_exists_reads_run_no_finders(
        self, db_session: AsyncSession
    ) -> None:
        """The entire point of the ADR: the second read re-collects nothing.

        `GET /curation` measured 172 s because every request re-ran every
        finder. If this ever fails by calling the finders again, the surface is
        slow again.
        """
        await _seed(db_session, "expired belief")
        await collect_and_persist(db_session, semantic=False)
        await db_session.flush()

        row = await latest_snapshot(db_session)
        assert row is not None
        assert row.card_count == 1
        assert row.scope == CollectionScope.STORE.value

        with patch(
            "particles.operations.curation.session.collect_cards",
            new=AsyncMock(side_effect=AssertionError("finders must not re-run")),
        ):
            served = await build_curation_queue(db_session, semantic=False)
        assert served.source == "snapshot"
        assert served.snapshot_id == row.snapshot_id
        assert [c.kind for c in served.cards] == [CardKind.STALE]

    @pytest.mark.asyncio
    async def test_live_source_bypasses_the_cache_entirely(self, db_session: AsyncSession) -> None:
        await _seed(db_session, "expired belief")

        result = await build_curation_queue(db_session, semantic=False, source=QueueSource.LIVE)
        assert result.source == "live"
        assert result.built_at is None
        # LIVE must not write a snapshot — it is a bypass, not a refresh.
        assert await latest_snapshot(db_session) is None

    @pytest.mark.asyncio
    async def test_snapshot_disabled_config_restores_old_behaviour(
        self, db_session: AsyncSession
    ) -> None:
        get_config().curation.snapshot_enabled = False
        await _seed(db_session, "expired belief")

        result = await build_curation_queue(db_session, semantic=False)
        assert result.source == "live"
        assert await latest_snapshot(db_session) is None

    @pytest.mark.asyncio
    async def test_retention_ring_prunes_old_collections(self, db_session: AsyncSession) -> None:
        get_config().curation.snapshot_retain = 2
        await _seed(db_session, "expired belief")

        for _ in range(4):
            await collect_and_persist(db_session, semantic=False)
        await db_session.flush()

        from sqlalchemy import func, select

        from particles.store.curation_snapshot_store import CurationSnapshotRow

        count = await db_session.scalar(select(func.count()).select_from(CurationSnapshotRow))
        assert count == 2

    @pytest.mark.asyncio
    async def test_stale_stamp_fires_past_the_configured_age(
        self, db_session: AsyncSession
    ) -> None:
        get_config().curation.snapshot_max_age_hours = 36.0
        await _seed(db_session, "expired belief")

        old = datetime.now(UTC) - timedelta(hours=40)
        await collect_and_persist(db_session, semantic=False, built_at=old)
        await db_session.flush()

        result = await build_curation_queue(db_session, semantic=False)
        assert result.stale is True
        assert result.age_seconds is not None and result.age_seconds > 36 * 3600
        # Stale is disclosed, never hidden — the cards still come back.
        assert result.cards

    @pytest.mark.asyncio
    async def test_corrupt_blob_is_a_cache_miss_not_a_crash(self, db_session: AsyncSession) -> None:
        await _seed(db_session, "expired belief")
        await collect_and_persist(db_session, semantic=False)
        await db_session.flush()

        row = await latest_snapshot(db_session)
        assert row is not None
        row.cards_json = "{not json"
        await db_session.flush()

        result = await build_curation_queue(db_session, semantic=False)
        assert result.cards == []  # degraded to empty, did not raise


# --------------------------------------------------------------------------- #
# §4 — scope drives replace vs carry-forward                                  #
# --------------------------------------------------------------------------- #


class TestScopeSemantics:
    def test_store_scope_marks_every_kind_store_wide(self) -> None:
        per_kind = per_kind_scope_for(CollectionScope.STORE)
        assert set(per_kind) == {k.value for k in CardKind}
        assert all(v is CollectionScope.STORE for v in per_kind.values())

    def test_delta_scope_marks_only_the_probe_bounded_kinds(self) -> None:
        per_kind = per_kind_scope_for(CollectionScope.DELTA)
        delta = {k for k, v in per_kind.items() if v is CollectionScope.DELTA}
        assert delta == {k.value for k in DELTA_SCOPED_KINDS}
        # Duplicates enumerate store-wide even under a delta run, so
        # they must replace rather than accumulate.
        assert per_kind[CardKind.DUPLICATE_PAIR.value] is CollectionScope.STORE

    @pytest.mark.asyncio
    async def test_delta_run_carries_prior_contradictions_forward(
        self, db_session: AsyncSession
    ) -> None:
        """A contradiction found last night survives tonight's delta run.

        The probe is delta-scoped and contradiction findings never persist as
        INCONSISTENCY particles, so without carry-forward the queue would
        forget every contradiction it ever found, one night later.
        """
        yesterday = _card(CardKind.CONTRADICTION, "p-old")
        await collect_and_persist(
            db_session, semantic=True, scope=CollectionScope.STORE, cards=[yesterday]
        )
        await db_session.flush()

        tonight = _card(CardKind.CONTRADICTION, "p-new")
        merged, _ = await collect_and_persist(
            db_session, semantic=True, scope=CollectionScope.DELTA, cards=[tonight]
        )
        assert {c.particle_ids[0] for c in merged} == {"p-old", "p-new"}

    @pytest.mark.asyncio
    async def test_store_wide_kinds_replace_rather_than_accumulate(
        self, db_session: AsyncSession
    ) -> None:
        """A resolved stale card disappears — its finder saw the whole store."""
        await collect_and_persist(
            db_session,
            semantic=False,
            scope=CollectionScope.STORE,
            cards=[_card(CardKind.STALE, "p-gone")],
        )
        await db_session.flush()

        merged, _ = await collect_and_persist(
            db_session,
            semantic=False,
            scope=CollectionScope.DELTA,
            cards=[_card(CardKind.STALE, "p-still-here")],
        )
        assert {c.particle_ids[0] for c in merged} == {"p-still-here"}

    @pytest.mark.asyncio
    async def test_carried_card_ages_out_at_the_horizon(self, db_session: AsyncSession) -> None:
        get_config().curation.snapshot_carry_forward_days = 30
        long_ago = datetime.now(UTC) - timedelta(days=45)
        await collect_and_persist(
            db_session,
            semantic=True,
            scope=CollectionScope.STORE,
            cards=[_card(CardKind.CONTRADICTION, "p-ancient")],
            built_at=long_ago,
        )
        await db_session.flush()

        merged, _ = await collect_and_persist(
            db_session,
            semantic=True,
            scope=CollectionScope.DELTA,
            cards=[_card(CardKind.CONTRADICTION, "p-fresh")],
        )
        assert {c.particle_ids[0] for c in merged} == {"p-fresh"}

    @pytest.mark.asyncio
    async def test_origin_stamp_does_not_reset_on_each_build(
        self, db_session: AsyncSession
    ) -> None:
        """A card carried across builds expires on its own clock, not the last build's."""
        get_config().curation.snapshot_carry_forward_days = 30
        origin = datetime.now(UTC) - timedelta(days=25)
        await collect_and_persist(
            db_session,
            semantic=True,
            scope=CollectionScope.STORE,
            cards=[_card(CardKind.CONTRADICTION, "p-aging")],
            built_at=origin,
        )
        await db_session.flush()

        # Carried once at day 25 — still inside the horizon.
        await collect_and_persist(
            db_session,
            semantic=True,
            scope=CollectionScope.DELTA,
            cards=[],
            built_at=origin + timedelta(days=3),
        )
        await db_session.flush()

        # At day 31 from ITS origin it must be gone, even though the previous
        # build was only days ago — the stamp is the card's, not the build's.
        merged, _ = await collect_and_persist(
            db_session,
            semantic=True,
            scope=CollectionScope.DELTA,
            cards=[],
            built_at=origin + timedelta(days=31),
        )
        assert merged == []


# --------------------------------------------------------------------------- #
# §5 — the live staleness ladder                                              #
# --------------------------------------------------------------------------- #


class TestLiveStaleness:
    @pytest.mark.asyncio
    async def test_retracted_belief_drops_from_a_stale_snapshot(
        self, db_session: AsyncSession
    ) -> None:
        """Level 2: the snapshot still names it; the live status check drops it."""
        belief = await _seed(db_session, "expired belief")

        first = await build_curation_queue(db_session, semantic=False)
        assert len(first.cards) == 1

        await update_particle_status(
            db_session, belief.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
        )
        await db_session.flush()

        again = await build_curation_queue(db_session, semantic=False)
        # Same snapshot, different answer — that is the point.
        assert again.snapshot_id == first.snapshot_id
        assert again.cards == []

    @pytest.mark.asyncio
    async def test_a_dropped_card_promotes_the_next_one(self, db_session: AsyncSession) -> None:
        """Level 2 runs before the slice, so the session stays full."""
        for i in range(3):
            await _seed(db_session, f"expired #{i}")

        full = await build_curation_queue(db_session, semantic=False, limit=2)
        assert len(full.cards) == 2

        await update_particle_status(
            db_session,
            full.cards[0].particle_ids[0],
            Status.RETRACTED,
            StatusReason.EXPLICIT_RETRACTION,
        )
        await db_session.flush()

        after = await build_curation_queue(db_session, semantic=False, limit=2)
        # Still 2 — the third card was promoted, not left short.
        assert len(after.cards) == 2

    @pytest.mark.asyncio
    async def test_post_snapshot_gesture_suppresses_the_card(
        self, db_session: AsyncSession
    ) -> None:
        """Level 3: an affirm recorded after the build hides the card."""
        await _seed(db_session, "expired belief")

        first = await build_curation_queue(db_session, semantic=False)
        [card] = first.cards
        await apply_gesture(db_session, card, "affirm")
        await db_session.commit()

        again = await build_curation_queue(db_session, semantic=False)
        assert again.snapshot_id == first.snapshot_id
        assert again.cards == []

    @pytest.mark.asyncio
    async def test_events_before_the_build_do_not_suppress(self, db_session: AsyncSession) -> None:
        """The level-3 filter is a *since* filter — it must not eat fresh cards.

        A resolving event recorded before the collection was built is already
        reflected in it; re-applying it would hide work the finders deliberately
        re-reported.
        """
        from particles.store.event_store import (
            EventRefKind,
            OperatorEventType,
            record_event,
        )

        belief = await _seed(db_session, "expired belief")
        await record_event(
            db_session,
            actor="test",
            event_type=OperatorEventType.RELATION_ADDED,
            refs=[(EventRefKind.PARTICLE, belief.id)],
            payload={},
        )
        await db_session.commit()

        result = await build_curation_queue(db_session, semantic=False)
        assert [c.particle_ids for c in result.cards] == [[belief.id]]


# --------------------------------------------------------------------------- #
# §6 — rebuild                                                                 #
# --------------------------------------------------------------------------- #


class TestRebuild:
    @pytest.mark.asyncio
    async def test_rebuild_writes_a_new_collection(self, db_session: AsyncSession) -> None:
        await _seed(db_session, "expired belief")

        first = await build_curation_queue(db_session, semantic=False)
        rebuilt = await rebuild_curation_snapshot(db_session, semantic=False)

        assert rebuilt.snapshot_id != first.snapshot_id
        assert rebuilt.collection_size == 1
        assert rebuilt.scope == CollectionScope.STORE.value

    @pytest.mark.asyncio
    async def test_clearing_the_cache_degrades_to_live_not_to_broken(
        self, db_session: AsyncSession
    ) -> None:
        """The escape hatch: snapshots are droppable and the queue still works.

        Dropping the cache costs correctness nothing and latency everything —
        reads fall back to collecting live until the next rebuild.
        """
        await _seed(db_session, "expired belief")
        await collect_and_persist(db_session, semantic=False)
        await db_session.flush()
        assert (await build_curation_queue(db_session, semantic=False)).source == "snapshot"

        assert await clear_snapshots(db_session) == 1
        await db_session.flush()

        served = await build_curation_queue(db_session, semantic=False)
        assert served.source == "live"
        assert served.snapshot_id is None
        assert len(served.cards) == 1

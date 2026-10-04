# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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
    SEMANTIC_KINDS,
    CarryEvidence,
    PriorCollection,
    Resolved,
    _merge_with_prior,
    collect_and_persist,
    per_kind_scope_for,
    stale_after_hours,
)
from particles.store.curation_snapshot_store import (
    CollectionScope,
    clear_snapshots,
    latest_snapshot,
    list_snapshots,
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
        # A census on every cycle, so the configured age is the threshold
        # (a weekly census widens it; see the test below).
        get_config().consolidation.census.interval_hours = 0
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
    async def test_a_collection_between_weekly_censuses_is_not_stale(
        self, db_session: AsyncSession
    ) -> None:
        # the cycle rebuilds only when its weekly census runs, so a
        # four-day-old collection is the expected state, not a stale one.
        get_config().curation.snapshot_max_age_hours = 36.0
        await _seed(db_session, "expired belief")
        old = datetime.now(UTC) - timedelta(days=4)
        await collect_and_persist(db_session, semantic=False, built_at=old)
        await db_session.flush()

        assert (await build_curation_queue(db_session, semantic=False)).stale is False
        get_config().consolidation.census.enabled = False
        assert (await build_curation_queue(db_session, semantic=False)).stale is True

    def test_stale_threshold_follows_the_census_cadence(self) -> None:
        assert stale_after_hours(36.0, census_enabled=True, census_interval_hours=168) == 192.0
        assert stale_after_hours(36.0, census_enabled=True, census_interval_hours=0) == 36.0
        assert stale_after_hours(36.0, census_enabled=False, census_interval_hours=168) == 36.0

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
        per_kind = per_kind_scope_for(CollectionScope.STORE, semantic=True)
        assert set(per_kind) == {k.value for k in CardKind}
        assert all(v is CollectionScope.STORE for v in per_kind.values())

    @pytest.mark.parametrize("scope", [CollectionScope.STORE, CollectionScope.DELTA])
    def test_without_semantic_the_probe_kind_is_carried(self, scope: CollectionScope) -> None:
        """a finder that did not run declares its kind carried, not store-wide."""
        per_kind = per_kind_scope_for(scope, semantic=False)
        carried = {k for k, v in per_kind.items() if v is CollectionScope.CARRIED}
        assert carried == {k.value for k in SEMANTIC_KINDS} == {CardKind.CONTRADICTION.value}
        # Every structural kind still replaces: its finder walked the store.
        assert per_kind[CardKind.STALE.value] is CollectionScope.STORE
        assert per_kind[CardKind.DUPLICATE_PAIR.value] is CollectionScope.STORE

    def test_delta_scope_marks_only_the_probe_bounded_kinds(self) -> None:
        per_kind = per_kind_scope_for(CollectionScope.DELTA, semantic=True)
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

        The probe is delta-scoped, and only a confirmed cross-source pair
        becomes an INCONSISTENCY record, so without carry-forward the
        queue would forget every other contradiction it found, one night later.
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
    async def test_a_disclosed_pairs_card_is_not_carried_forward(
        self, db_session: AsyncSession
    ) -> None:
        """once a census record discloses a pair, its card gives way."""
        disclosed = CurationCard(
            kind=CardKind.CONTRADICTION,
            particle_ids=["p-a"],
            diagnostic="Semantic contradiction with particle p-b: exists vs not found",
            suggested_gestures=gestures_for(CardKind.CONTRADICTION),
        )
        other = _card(CardKind.CONTRADICTION, "p-old")
        await collect_and_persist(
            db_session, semantic=True, scope=CollectionScope.STORE, cards=[disclosed, other]
        )
        await db_session.flush()

        merged, _ = await collect_and_persist(
            db_session,
            semantic=True,
            scope=CollectionScope.DELTA,
            cards=[],
            covered_pairs=frozenset({frozenset(("p-a", "p-b"))}),
        )
        assert [c.particle_ids[0] for c in merged] == ["p-old"]

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


# --------------------------------------------------------------------------- #
# a collection written before conflict cards is upgraded on read     #
# --------------------------------------------------------------------------- #


class TestFormatOneCollection:
    @pytest.mark.asyncio
    async def test_is_served_with_conflict_cards_and_without_comment(
        self, db_session: AsyncSession
    ) -> None:
        import json

        from particles.store.curation_snapshot_store import write_snapshot

        a = _active("claim A", valid_until=None)
        b = _active("claim B", valid_until=None)
        await insert_particle(db_session, a)
        await insert_particle(db_session, b)
        record = Particle(
            content="INCONSISTENCY between A and B",
            confidence=Confidence(value=0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            provenance=[
                ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=a.id),
                ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=b.id),
            ],
            asserted_by="extract-pipeline",
            status=Status.INCONSISTENCY,
        )
        await insert_particle(db_session, record)
        old_contested = CurationCard(
            kind=CardKind.CONTESTED,
            particle_ids=[a.id],
            diagnostic="Contested (inconsistency)",
            suggested_gestures=["comment", "affirm", "snooze"],
            contested_bases=["inconsistency"],
            inconsistency_id=record.id,
        )
        old_contradiction = CurationCard(
            kind=CardKind.CONTRADICTION,
            particle_ids=[b.id],
            diagnostic="Semantic contradiction",
            suggested_gestures=["comment", "supersede", "retract", "snooze"],
        )
        blob = json.dumps(
            {
                "format": 1,
                "cards": [c.model_dump(mode="json") for c in (old_contested, old_contradiction)],
                "first_seen": {},
            }
        )
        await write_snapshot(
            db_session,
            cards_json=blob,
            card_count=2,
            scope=CollectionScope.STORE,
            per_kind_scope={},
        )
        await db_session.flush()

        cards = (await build_curation_queue(db_session, semantic=False)).cards
        kinds = {c.kind: c for c in cards}
        assert CardKind.CONTESTED not in kinds
        assert "comment" not in kinds[CardKind.CONTRADICTION].suggested_gestures
        conflict = kinds[CardKind.INCONSISTENCY]
        assert conflict.key == f"inconsistency:{record.id}"
        assert conflict.leverage > 0


class TestListSnapshots:
    """``list_snapshots``: the retained ring, newest first."""

    @pytest.mark.asyncio
    async def test_newest_first_and_pruned_to_the_ring(self, db_session: AsyncSession) -> None:
        assert await list_snapshots(db_session) == []
        base = datetime(2026, 9, 1, tzinfo=UTC)
        retain = get_config().curation.snapshot_retain
        for day in range(retain + 2):
            await collect_and_persist(
                db_session, semantic=False, built_at=base + timedelta(days=day)
            )
        rows = await list_snapshots(db_session)
        assert len(rows) == retain
        stamps = [r.built_at.replace(tzinfo=UTC) for r in rows]
        assert stamps == sorted(stamps, reverse=True)
        newest = await latest_snapshot(db_session)
        assert newest is not None and rows[0].snapshot_id == newest.snapshot_id


# --------------------------------------------------------------------------- #
# a build that skipped the contradiction probe carries its cards     #
# --------------------------------------------------------------------------- #

_NOW = datetime(2026, 10, 1, 12, tzinfo=UTC)
_CENSUS = _NOW - timedelta(days=3)


def _contradiction(pid: str, partner: str = "p-partner") -> CurationCard:
    return CurationCard(
        kind=CardKind.CONTRADICTION,
        particle_ids=[pid],
        diagnostic=f"Semantic contradiction with particle {partner}: a vs b",
        suggested_gestures=gestures_for(CardKind.CONTRADICTION),
    )


def _prior(*cards: CurationCard) -> PriorCollection:
    """A census collection built three days ago, every card found then."""
    return PriorCollection(
        cards=list(cards),
        first_seen={c.key: _CENSUS.isoformat() for c in cards},
        kind_as_of={k.value: _CENSUS for k in CardKind},
        built_at=_CENSUS,
    )


def _merge(
    prior: PriorCollection,
    fresh: list[CurationCard],
    *,
    semantic: bool,
    evidence: CarryEvidence | None = None,
) -> list[str]:
    result = _merge_with_prior(
        fresh,
        prior,
        per_kind=per_kind_scope_for(CollectionScope.STORE, semantic=semantic),
        now=_NOW,
        horizon=timedelta(days=30),
        evidence=evidence,
    )
    return sorted(c.key for c in result.cards)


class TestCarriedKinds:
    """The pure merge decision over plain values."""

    def test_a_structural_refresh_keeps_every_census_contradiction(self) -> None:
        census = [_contradiction(f"p-{i}") for i in range(4)]
        stale = _card(CardKind.STALE, "p-stale")
        kept = _merge(_prior(*census, stale), [], semantic=False)
        # The structural kind replaced (its finder saw nothing); the probe kind carried.
        assert kept == sorted(c.key for c in census)

    def test_a_semantic_refresh_re_finds_rather_than_carries(self) -> None:
        census = [_contradiction(f"p-{i}") for i in range(4)]
        refound = census[:2]
        kept = _merge(_prior(*census), refound, semantic=True)
        assert kept == sorted(c.key for c in refound)

    def test_suppressed_resolved_and_retired_cards_are_not_carried(self) -> None:
        keep, snoozed, resolved, retired, partner_retired, before = (
            _contradiction("p-keep"),
            _contradiction("p-snoozed"),
            _contradiction("p-resolved"),
            _contradiction("p-retired"),
            _contradiction("p-live", partner="p-gone"),
            _contradiction("p-touched-before"),
        )
        evidence = CarryEvidence(
            suppressed=frozenset({snoozed.key}),
            resolutions=(
                (_CENSUS + timedelta(hours=1), Resolved(particle_ids=frozenset({"p-resolved"}))),
                # Before the card was found: already reflected in the census.
                (
                    _CENSUS - timedelta(hours=1),
                    Resolved(particle_ids=frozenset({"p-touched-before"})),
                ),
            ),
            retired=frozenset({"p-retired", "p-gone"}),
        )
        prior = _prior(keep, snoozed, resolved, retired, partner_retired, before)
        kept = _merge(prior, [], semantic=False, evidence=evidence)
        assert kept == sorted([keep.key, before.key])

    def test_the_carried_kind_keeps_the_census_date(self) -> None:
        result = _merge_with_prior(
            [],
            _prior(_contradiction("p-a")),
            per_kind=per_kind_scope_for(CollectionScope.STORE, semantic=False),
            now=_NOW,
            horizon=timedelta(days=30),
        )
        assert result.kind_as_of[CardKind.CONTRADICTION.value] == _CENSUS
        assert result.kind_as_of[CardKind.STALE.value] == _NOW

    def test_a_carried_kind_with_no_probe_on_record_has_no_date(self) -> None:
        result = _merge_with_prior(
            [],
            None,
            per_kind=per_kind_scope_for(CollectionScope.STORE, semantic=False),
            now=_NOW,
            horizon=timedelta(days=30),
        )
        assert CardKind.CONTRADICTION.value not in result.kind_as_of


class TestRefreshKeepsCensusContradictions:
    """The store-backed path `particles curate --refresh` takes."""

    @pytest.mark.asyncio
    async def test_structural_refresh_keeps_them_and_the_census_date(
        self, db_session: AsyncSession
    ) -> None:
        census_at = datetime.now(UTC) - timedelta(days=3)
        census = [_contradiction(f"p-{i}") for i in range(3)]
        await collect_and_persist(
            db_session,
            semantic=True,
            scope=CollectionScope.DELTA,
            cards=census,
            built_at=census_at,
        )
        await db_session.flush()

        with patch("particles.operations.curation.snapshot.llm_circuit_open", return_value=False):
            stamp = await rebuild_curation_snapshot(db_session, semantic=False)

        assert stamp.per_kind_scope[CardKind.CONTRADICTION.value] == "carried"
        assert stamp.kind_as_of[CardKind.CONTRADICTION.value] == census_at
        served = await build_curation_queue(db_session, kind=CardKind.CONTRADICTION)
        assert served.open_count == 3
        assert served.kind_as_of[CardKind.CONTRADICTION.value] == census_at

    @pytest.mark.asyncio
    async def test_a_degraded_probe_carries_too(self, db_session: AsyncSession) -> None:
        """The breaker cut the probe short, so its silence is not a resolution."""
        await collect_and_persist(
            db_session, semantic=True, scope=CollectionScope.STORE, cards=[_contradiction("p-a")]
        )
        await db_session.flush()
        with patch("particles.operations.curation.snapshot.llm_circuit_open", return_value=True):
            merged, _ = await collect_and_persist(
                db_session, semantic=True, scope=CollectionScope.STORE, cards=[]
            )
        assert [c.particle_ids[0] for c in merged] == ["p-a"]

    @pytest.mark.asyncio
    async def test_a_dismissed_or_retracted_card_is_not_carried(
        self, db_session: AsyncSession
    ) -> None:
        dismissed = await _seed(db_session, "dismissed claim")
        retracted = await _seed(db_session, "retracted claim")
        kept = await _seed(db_session, "kept claim")
        cards = [_contradiction(p.id) for p in (dismissed, retracted, kept)]
        await collect_and_persist(
            db_session, semantic=True, scope=CollectionScope.STORE, cards=cards
        )
        await db_session.flush()

        await apply_gesture(db_session, cards[0], "dismiss")
        await update_particle_status(
            db_session, retracted.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
        )
        await db_session.flush()

        merged, _ = await collect_and_persist(
            db_session, semantic=False, scope=CollectionScope.STORE, cards=[]
        )
        assert [c.particle_ids[0] for c in merged] == [kept.id]

    @pytest.mark.asyncio
    async def test_a_row_written_before_the_as_of_envelope_reads_its_date(
        self, db_session: AsyncSession
    ) -> None:
        import json

        from particles.store.curation_snapshot_store import write_snapshot

        built = datetime.now(UTC) - timedelta(days=2)
        blob = json.dumps(
            {
                "format": 2,
                "cards": [_contradiction("p-a").model_dump(mode="json")],
                "first_seen": {},
            }
        )
        await write_snapshot(
            db_session,
            cards_json=blob,
            card_count=1,
            scope=CollectionScope.DELTA,
            per_kind_scope={},
            semantic=True,
            built_at=built,
        )
        await db_session.flush()

        served = await build_curation_queue(db_session)
        assert served.kind_as_of[CardKind.CONTRADICTION.value] == built

# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""A supersession mirror loser is born demoted.

The §6.6 birth ``(new) → PROVENANCE_STALE`` admits a named set of reasons, and
each supersession reason only on the insert of the mirror verdict that decided
it. The loser never passes through ACTIVE, so no history reader counts it as a
belief once held, and the update-retirement walk dates it by its event.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.conflict_resolution import ConflictVerdict, LadderOutcome
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import (
    BORN_DEMOTED_REASONS,
    Status,
    StatusReason,
    validate_born_demoted,
)
from particles.ingest.conflict_plan import plan_conflict_writes
from particles.store.event_store import (
    EventRefKind,
    OperatorEventRow,
    OperatorEventType,
    list_events,
    record_event,
)
from particles.store.particle_store import (
    ParticleRow,
    count_update_retirements_after,
    get_recorded_demotions,
    get_update_retirements_after,
    insert_particle,
    update_particle_status,
)

T0 = datetime(2026, 11, 1, tzinfo=UTC)
MIRRORS = [
    (ConflictVerdict.DOCUMENT_SUPERSEDED_BY_EXISTING, StatusReason.DOCUMENT_SUPERSEDED),
    (ConflictVerdict.UPDATE_SUPERSEDED_BY_EXISTING, StatusReason.SUPERSEDED_BY_UPDATE),
]


def _particle(
    content: str,
    *,
    status: Status = Status.ACTIVE,
    reason: StatusReason | None = None,
    entry: str = "entry-1",
) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=entry, snapshot_id="s")
        ],
        asserted_by="test",
        asserted_at=T0,
        status=status,
        status_reason=reason,
    )


async def _row(session: AsyncSession, particle_id: str) -> ParticleRow:
    session.expire_all()
    row = await session.get(ParticleRow, particle_id)
    assert row is not None
    return row


class TestTheBirthRule:
    def test_the_set_is_the_quarantine_and_the_two_mirrors(self) -> None:
        assert {
            StatusReason.CONFLICT_PENDING,
            StatusReason.DOCUMENT_SUPERSEDED,
            StatusReason.SUPERSEDED_BY_UPDATE,
        } == BORN_DEMOTED_REASONS

    def test_the_quarantine_needs_no_verdict(self) -> None:
        validate_born_demoted(StatusReason.CONFLICT_PENDING)

    @pytest.mark.parametrize(
        "reason", [StatusReason.DOCUMENT_SUPERSEDED, StatusReason.SUPERSEDED_BY_UPDATE]
    )
    def test_a_supersession_reason_needs_its_own_verdict(self, reason: StatusReason) -> None:
        validate_born_demoted(reason, mirror_reason=reason)
        with pytest.raises(ValueError, match="born PROVENANCE_STALE"):
            validate_born_demoted(reason)
        other = next(r for r in BORN_DEMOTED_REASONS - {reason, StatusReason.CONFLICT_PENDING})
        with pytest.raises(ValueError, match="born PROVENANCE_STALE"):
            validate_born_demoted(reason, mirror_reason=other)

    @pytest.mark.parametrize(
        "reason",
        [
            StatusReason.LOWER_TRUST_SOURCE,
            StatusReason.VALIDITY_EXPIRED,
            StatusReason.CONFLICT_RESOLVED,
            StatusReason.CORPUS_ENTRY_MISSING,
            None,
        ],
    )
    def test_no_other_reason_is_admitted(self, reason: StatusReason | None) -> None:
        with pytest.raises(ValueError, match="born PROVENANCE_STALE"):
            validate_born_demoted(reason, mirror_reason=reason)


@pytest.mark.asyncio
class TestTheInsertSeam:
    @pytest.mark.parametrize(("verdict", "reason"), MIRRORS)
    async def test_admits_a_mirror_loser_under_its_verdict(
        self, db_session: AsyncSession, verdict: ConflictVerdict, reason: StatusReason
    ) -> None:
        loser = _particle("a loser", status=Status.PROVENANCE_STALE, reason=reason)
        await insert_particle(db_session, loser, born_demoted_by=verdict)
        await db_session.commit()
        row = await _row(db_session, loser.id)
        assert row.born_retired is True
        assert row.retired_at is None

    @pytest.mark.parametrize(("verdict", "reason"), MIRRORS)
    async def test_refuses_a_supersession_reason_without_its_verdict(
        self, db_session: AsyncSession, verdict: ConflictVerdict, reason: StatusReason
    ) -> None:
        loser = _particle("a loser", status=Status.PROVENANCE_STALE, reason=reason)
        with pytest.raises(ValueError, match="born PROVENANCE_STALE"):
            await insert_particle(db_session, loser)
        with pytest.raises(ValueError, match="born PROVENANCE_STALE"):
            await insert_particle(db_session, loser, born_demoted_by=ConflictVerdict.SUPERSEDES)

    async def test_a_restore_of_a_retired_unit_is_refused_whole(
        self, db_session: AsyncSession
    ) -> None:
        # A bundle's SUPERSEDED_BY_UPDATE unit describes a belief once held, so
        # restore (which names no verdict) refuses it, and nothing is written.
        from particles.interchange.codec import to_unit
        from particles.interchange.store import restore_store_bundle

        held = _particle("held and restored")
        retired = _particle(
            "retired in the bundle's store",
            status=Status.PROVENANCE_STALE,
            reason=StatusReason.SUPERSEDED_BY_UPDATE,
        )
        lines = "".join(json.dumps(to_unit(p, {})) + "\n" for p in (held, retired))
        with pytest.raises(ValueError, match="born PROVENANCE_STALE"):
            await restore_store_bundle(db_session, {"particles.jsonl": lines})
        await db_session.rollback()
        count = await db_session.execute(select(func.count()).select_from(ParticleRow))
        assert count.scalar_one() == 0


@pytest.mark.asyncio
class TestTheMirrorWrites:
    @pytest.mark.parametrize(("verdict", "reason"), MIRRORS)
    async def test_one_row_born_demoted_and_its_event(
        self,
        db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
        verdict: ConflictVerdict,
        reason: StatusReason,
    ) -> None:
        from particles.ingest import pipeline
        from particles.operations.query.as_of import load_as_of_view

        existing = _particle("the held claim", entry="entry-held")
        await insert_particle(db_session, existing)
        await db_session.commit()
        new = _particle("the late loser", entry="entry-late")
        plan = plan_conflict_writes(
            LadderOutcome(verdict=verdict, record_divergence=False),
            new,
            existing,
            corpus_entry_id="entry-late",
            snapshot_id="s",
            trigger_ref_type=ProvenanceRefType.SOURCE,
            scores=(None, None),
            domain="test",
        )

        async def no_transition(*_: Any, **__: Any) -> None:
            raise AssertionError("a born-demoted loser makes no status transition")

        monkeypatch.setattr(pipeline, "update_particle_status", no_transition)
        result = await pipeline._apply_conflict_plan(
            db_session, plan, existing=existing, embedding=None
        )
        await db_session.commit()

        assert result is not None and result.id == new.id
        row = await _row(db_session, new.id)
        stored = row.to_model()
        assert (row.status, row.status_reason) == (Status.PROVENANCE_STALE.value, reason.value)
        assert row.born_retired is True
        assert row.retired_at is None
        assert (await _row(db_session, existing.id)).status == Status.ACTIVE.value
        events = [
            e
            for t in (
                OperatorEventType.DOCUMENT_SUPERSESSION,
                OperatorEventType.UPDATE_SUPERSESSION,
            )
            for e in await list_events(db_session, event_type=t)
        ]
        assert [e.payload["retired_id"] for e in events if e.payload] == [new.id]

        # The as-of lens never shows it: it was never believed.
        view = await load_as_of_view(db_session, datetime.now(UTC))
        assert new.id in view.index.never_believed
        assert view.evaluate(stored, None).visible is False


async def _stamped(session: AsyncSession, content: str, at: datetime) -> str:
    p = _particle(content)
    await insert_particle(session, p)
    await update_particle_status(
        session, p.id, Status.PROVENANCE_STALE, StatusReason.SUPERSEDED_BY_UPDATE
    )
    await session.execute(update(ParticleRow).where(ParticleRow.id == p.id).values(retired_at=at))
    return p.id


async def _born_demoted(
    session: AsyncSession, content: str, winner_id: str, at: datetime | None
) -> str:
    p = _particle(content, status=Status.PROVENANCE_STALE, reason=StatusReason.SUPERSEDED_BY_UPDATE)
    await insert_particle(session, p, born_demoted_by=ConflictVerdict.UPDATE_SUPERSEDED_BY_EXISTING)
    if at is not None:
        event = await record_event(
            session,
            actor="extract-pipeline",
            event_type=OperatorEventType.UPDATE_SUPERSESSION,
            refs=[(EventRefKind.PARTICLE, p.id), (EventRefKind.PARTICLE, winner_id)],
            payload={"retired_id": p.id, "winner_id": winner_id, "route": "ladder"},
        )
        await session.execute(
            update(OperatorEventRow)
            .where(OperatorEventRow.event_id == event.event_id)
            .values(occurred_at=at)
        )
    return p.id


@pytest.mark.asyncio
class TestTheUpdateRetirementWalk:
    async def _seed(self, session: AsyncSession) -> tuple[str, str, str, str]:
        winner = _particle("the newer claim")
        await insert_particle(session, winner)
        first = await _stamped(session, "retired first", T0 + timedelta(minutes=1))
        middle = await _born_demoted(session, "born demoted", winner.id, T0 + timedelta(minutes=2))
        last = await _stamped(session, "retired last", T0 + timedelta(minutes=3))
        # A born-demoted row no event names is not an update retirement.
        await _born_demoted(session, "no event", winner.id, None)
        await session.commit()
        return winner.id, first, middle, last

    async def test_a_born_demoted_loser_is_walked_in_event_order(
        self, db_session: AsyncSession
    ) -> None:
        _, first, middle, last = await self._seed(db_session)
        page = await get_update_retirements_after(db_session, None)
        assert [p.id for p, _ in page] == [first, middle, last]
        assert await count_update_retirements_after(db_session, None) == 3
        assert [p.id for p, _ in await get_update_retirements_after(db_session, None, limit=2)] == [
            first,
            middle,
        ]

    async def test_a_cursor_resumes_without_skipping_or_repeating(
        self, db_session: AsyncSession
    ) -> None:
        _, first, middle, last = await self._seed(db_session)
        walked: list[str] = []
        after: tuple[datetime, str] | None = None
        while page := await get_update_retirements_after(db_session, after, limit=1):
            [(particle, instant)] = page
            walked.append(particle.id)
            after = (instant, particle.id)
            assert await count_update_retirements_after(db_session, after) == 3 - len(walked)
        assert walked == [first, middle, last]

        # A cursor stored before the change, a stamped row's (retired_at, id).
        [(_, first_at)] = await get_update_retirements_after(db_session, None, limit=1)
        resumed = await get_update_retirements_after(db_session, (first_at, first))
        assert [p.id for p, _ in resumed] == [middle, last]

    async def test_the_demotion_cards_date_it_by_its_event(self, db_session: AsyncSession) -> None:
        # Newest retirement first: the born-demoted loser sorts by its event,
        # between the two stamped retirements, not last as undated. Every
        # retirement names its winner through an UPDATE_SUPERSESSION event.
        winner_id, first, middle, last = await self._seed(db_session)
        for retired_id in (first, last):
            await record_event(
                db_session,
                actor="extract-pipeline",
                event_type=OperatorEventType.UPDATE_SUPERSESSION,
                refs=[(EventRefKind.PARTICLE, retired_id), (EventRefKind.PARTICLE, winner_id)],
                payload={"retired_id": retired_id, "winner_id": winner_id, "route": "ladder"},
            )
        await db_session.commit()
        cards = await get_recorded_demotions(db_session)
        assert [retired.id for retired, _ in cards] == [last, middle, first]

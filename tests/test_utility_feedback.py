"""Tests for the explicit operator usefulness gesture.

The four behaviours the ADR's activation gate names, plus the surface contracts:
the rate bound (§7), rebuild losslessness (§2), the mined/explicit isolation
(§3), and that the gesture never touches the belief itself.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.operations.utility_feedback import (
    CLI_ACTOR,
    BeliefNotCreditable,
    mark_belief_useful,
    rederive_explicit_credits,
)
from particles.store.event_store import OperatorEventType, list_events
from particles.store.particle_store import get_particle, insert_particle
from particles.store.utility_store import (
    clear_utility_events,
    get_reinforcement_scores,
    record_utility_events,
)

_NOW = datetime(2026, 7, 1, tzinfo=UTC)


def _belief(pid: str, content: str, status: Status = Status.ACTIVE) -> Particle:
    return Particle(
        id=pid,
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        status=status,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1", snapshot_id="s1")
        ],
    )


async def _seed(session: AsyncSession, particle: Particle) -> str:
    await insert_particle(session, particle)
    return particle.id


async def _score(session: AsyncSession, pid: str, *, now: datetime = _NOW) -> float:
    scores = await get_reinforcement_scores(session, [pid], 30.0, now=now, explicit_weight=25.0)
    return scores.get(pid, 0.0)


class TestMarkBeliefUseful:
    @pytest.mark.asyncio
    async def test_credits_the_belief_and_logs_the_event(self, db_session: AsyncSession) -> None:
        pid = await _seed(db_session, _belief("p-1", "never prepend `export PATH`"))

        result = await mark_belief_useful(db_session, pid, reason="miner can't see it", now=_NOW)

        assert result.counted is True
        assert math.isclose(await _score(db_session, pid), 25.0, abs_tol=1e-9)
        events = await list_events(db_session, event_type=OperatorEventType.BELIEF_MARKED_USEFUL)
        assert len(events) == 1
        assert events[0].actor == CLI_ACTOR
        assert events[0].reason == "miner can't see it"
        assert events[0].payload == {
            "credit_key": result.credit_key,
            "counted": True,
            "observed_at": _NOW.isoformat(),
        }
        assert [r.ref_id for r in events[0].refs] == [pid]

    @pytest.mark.asyncio
    async def test_does_not_touch_the_belief(self, db_session: AsyncSession) -> None:
        """Utility is a claim about *use*, never about truth (two-quantity split)."""
        pid = await _seed(db_session, _belief("p-1", "a guideline"))
        before = await get_particle(db_session, pid)
        assert before is not None

        await mark_belief_useful(db_session, pid, now=_NOW)

        after = await get_particle(db_session, pid)
        assert after is not None
        assert after.confidence.value == before.confidence.value
        assert after.status is before.status

    @pytest.mark.asyncio
    async def test_second_press_same_day_records_but_does_not_double_count(
        self, db_session: AsyncSession
    ) -> None:
        """N presses → N audit events, one credit."""
        pid = await _seed(db_session, _belief("p-1", "a guideline"))

        first = await mark_belief_useful(db_session, pid, now=_NOW)
        repeats = [
            await mark_belief_useful(db_session, pid, now=_NOW + timedelta(hours=h))
            for h in range(1, 10)
        ]

        assert first.counted is True
        assert all(r.counted is False for r in repeats)
        assert math.isclose(await _score(db_session, pid), 25.0, abs_tol=1e-9)
        events = await list_events(
            db_session, event_type=OperatorEventType.BELIEF_MARKED_USEFUL, limit=100
        )
        assert len(events) == 10

    @pytest.mark.asyncio
    async def test_press_on_a_later_day_reinforces(self, db_session: AsyncSession) -> None:
        pid = await _seed(db_session, _belief("p-1", "a guideline"))
        later = _NOW + timedelta(days=1)

        await mark_belief_useful(db_session, pid, now=_NOW)
        second = await mark_belief_useful(db_session, pid, now=later)

        assert second.counted is True
        assert await _score(db_session, pid, now=later) > 25.0

    @pytest.mark.asyncio
    async def test_missing_belief_is_refused(self, db_session: AsyncSession) -> None:
        with pytest.raises(BeliefNotCreditable):
            await mark_belief_useful(db_session, "p-nope", now=_NOW)

    @pytest.mark.asyncio
    async def test_non_active_belief_is_refused(self, db_session: AsyncSession) -> None:
        pid = await _seed(db_session, _belief("p-1", "retired", status=Status.RETRACTED))
        with pytest.raises(BeliefNotCreditable):
            await mark_belief_useful(db_session, pid, now=_NOW)
        # And nothing was written on either table.
        assert await _score(db_session, pid) == 0.0
        assert (
            await list_events(db_session, event_type=OperatorEventType.BELIEF_MARKED_USEFUL) == []
        )


class TestRebuildRederivation:
    """the event log is the system of record; the table is derived."""

    @pytest.mark.asyncio
    async def test_rebuild_restores_explicit_credits_with_original_timestamps(
        self, db_session: AsyncSession
    ) -> None:
        pid = await _seed(db_session, _belief("p-1", "a guideline"))
        old = _NOW - timedelta(days=30)
        await mark_belief_useful(db_session, pid, now=old)
        before = await _score(db_session, pid)

        # What `rebuild-utility` does: truncate, then re-derive each channel.
        await clear_utility_events(db_session)
        assert await _score(db_session, pid) == 0.0
        written = await rederive_explicit_credits(db_session)

        assert written == 1
        # Same score, not a reset decay clock — a rebuild must not silently make
        # every historical gesture look brand new.
        assert math.isclose(await _score(db_session, pid), before, abs_tol=1e-9)
        assert math.isclose(before, 12.5, abs_tol=1e-9)

    @pytest.mark.asyncio
    async def test_rederivation_is_idempotent(self, db_session: AsyncSession) -> None:
        pid = await _seed(db_session, _belief("p-1", "a guideline"))
        await mark_belief_useful(db_session, pid, now=_NOW)
        await mark_belief_useful(db_session, pid, now=_NOW + timedelta(hours=2))

        await rederive_explicit_credits(db_session)
        await rederive_explicit_credits(db_session)

        # Two presses on one day collapse to one credit, exactly as when first
        # recorded — replaying cannot inflate the signal.
        assert math.isclose(await _score(db_session, pid), 25.0, abs_tol=1e-9)

    @pytest.mark.asyncio
    async def test_rederivation_leaves_the_mined_channel_alone(
        self, db_session: AsyncSession
    ) -> None:
        pid = await _seed(db_session, _belief("p-1", "a guideline"))
        await record_utility_events(db_session, "sess-1", {pid: "literal"}, observed_at=_NOW)
        await mark_belief_useful(db_session, pid, now=_NOW)

        await rederive_explicit_credits(db_session)

        # 1 mined + 25 explicit; the re-derivation must not have dropped the mine.
        assert math.isclose(await _score(db_session, pid), 26.0, abs_tol=1e-9)

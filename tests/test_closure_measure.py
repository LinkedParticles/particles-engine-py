# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The closure measure: contested fraction and the transition split.

Pins the pure tally (which transitions are autonomous, which a gesture caused),
the store gather (window bounds, the event-log join, the store-wide contested
count), the window tiling across runs, the history read-back, and the
``memory consolidate --history`` surface.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from particles.core.closure_measure import (
    GestureEvent,
    Retirement,
    contested_fraction,
    tally_transitions,
)
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.operations.abstraction import ABSTRACTION_ACTOR
from particles.operations.closure_measure import (
    ClosureMeasure,
    closure_history,
    measure_closure,
    measure_from_payload,
    render_closure_history,
    render_closure_lines,
    window_start_after,
)
from particles.store.event_store import (
    EventRefKind,
    OperatorEvent,
    OperatorEventType,
    record_event,
)
from particles.store.particle_store import insert_particle, update_particle_status

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_A = "00000000-0000-0000-0000-00000000000a"
_B = "00000000-0000-0000-0000-00000000000b"
_C = "00000000-0000-0000-0000-00000000000c"
_D = "00000000-0000-0000-0000-00000000000d"
_INC = "00000000-0000-0000-0000-0000000000e0"
_DERIVED = "00000000-0000-0000-0000-0000000000f0"
_ACCEPTED = "00000000-0000-0000-0000-0000000000f1"


def _claim(pid: str, content: str, *, asserted_by: str = "test") -> Particle:
    return Particle(
        id=pid,
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by=asserted_by,
    )


# ---------------------------------------------------------------------------
# The pure tally
# ---------------------------------------------------------------------------


class TestTally:
    def test_fraction_has_no_value_on_an_empty_store(self) -> None:
        assert contested_fraction(0, 0) is None
        assert contested_fraction(3, 12) == 0.25

    def test_a_retirement_an_explicit_write_names_is_a_gesture(self) -> None:
        tally = tally_transitions(
            retirements=[
                Retirement("p1", "EXPLICIT_RETRACTION"),
                Retirement("p2", "SUPERSEDED_BY_UPDATE"),
                Retirement("p3", None),
            ],
            gestures=[GestureEvent("PARTICLE_RETRACTED", "curate", ("p1",))],
            autonomous_closures=0,
            promoted_ids=[],
            accepted_promotions=[],
        )
        assert tally.by_kind["retirement"] == {"autonomous": 2, "gesture": 1}
        assert tally.autonomous_by_reason == {"SUPERSEDED_BY_UPDATE": 1, "UNSPECIFIED": 1}
        assert tally.gesture_by_actor == {"curate": 1}

    def test_the_reason_alone_never_decides(self) -> None:
        # The disclosure pass and a review both write CONFLICT_RESOLVED, and
        # abstraction revalidation writes EXPLICIT_SUPERSESSION: only the event
        # join tells the gesture from the pass.
        tally = tally_transitions(
            retirements=[
                Retirement("p1", "EXPLICIT_SUPERSESSION"),
                Retirement("p2", "CONFLICT_RESOLVED"),
            ],
            gestures=[],
            autonomous_closures=0,
            promoted_ids=[],
            accepted_promotions=[],
        )
        assert tally.by_kind["retirement"] == {"autonomous": 2, "gesture": 0}

    def test_a_ref_on_an_unrelated_event_type_attributes_nothing(self) -> None:
        tally = tally_transitions(
            retirements=[Retirement("p1", "DUPLICATE_MERGED")],
            gestures=[GestureEvent("BELIEF_AFFIRMED", "curate", ("p1",))],
            autonomous_closures=0,
            promoted_ids=[],
            accepted_promotions=[],
        )
        assert tally.by_kind["retirement"] == {"autonomous": 1, "gesture": 0}

    def test_contradiction_closures_split_review_from_disclosure(self) -> None:
        tally = tally_transitions(
            retirements=[],
            gestures=[GestureEvent("REVIEW_RESOLVED", "review", ("inc-1", "p1", "p2"))],
            autonomous_closures=3,
            promoted_ids=[],
            accepted_promotions=[],
        )
        assert tally.by_kind["contradiction_closed"] == {"autonomous": 3, "gesture": 1}
        assert tally.gesture_by_actor == {"review": 1}

    def test_an_accepted_abstraction_is_not_the_pass_s_own(self) -> None:
        tally = tally_transitions(
            retirements=[],
            gestures=[],
            autonomous_closures=0,
            promoted_ids=["d1", "d2"],
            accepted_promotions=[GestureEvent("ABSTRACTION_RESOLVED", "curate", ("d2",))],
        )
        assert tally.by_kind["promotion"] == {"autonomous": 1, "gesture": 1}

    def test_share_has_no_value_without_transitions(self) -> None:
        tally = tally_transitions(
            retirements=[],
            gestures=[],
            autonomous_closures=0,
            promoted_ids=[],
            accepted_promotions=[],
        )
        assert tally.total == 0
        assert tally.autonomous_share is None

    def test_share_counts_every_kind(self) -> None:
        tally = tally_transitions(
            retirements=[Retirement("p1", "LOWER_TRUST_SOURCE")],
            gestures=[GestureEvent("REVIEW_RESOLVED", "review", ())],
            autonomous_closures=2,
            promoted_ids=[],
            accepted_promotions=[],
        )
        assert (tally.autonomous, tally.gesture) == (3, 1)
        assert tally.autonomous_share == 0.75


# ---------------------------------------------------------------------------
# The gather over a real store
# ---------------------------------------------------------------------------


async def _seed(session: AsyncSession) -> None:
    """Two ACTIVE claims, one contested; one autonomous and one gesture retirement."""
    for pid, content in ((_A, "Claim A."), (_B, "Claim B."), (_C, "Claim C."), (_D, "Claim D.")):
        await insert_particle(session, _claim(pid, content))
    await insert_particle(
        session,
        Particle(
            id=_INC,
            content="INCONSISTENCY: A conflicts with something.",
            confidence=Confidence(value=0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="test",
            status=Status.INCONSISTENCY,
            provenance=[ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=_A)],
        ),
    )
    await update_particle_status(session, _B, Status.SUPERSEDED, StatusReason.SUPERSEDED_BY_UPDATE)
    await update_particle_status(session, _C, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION)
    await record_event(
        session,
        actor="curate",
        event_type=OperatorEventType.PARTICLE_RETRACTED,
        refs=[(EventRefKind.PARTICLE, _C)],
    )
    await record_event(
        session,
        actor="memory-consolidate",
        event_type=OperatorEventType.INCONSISTENCY_CLOSED,
        refs=[(EventRefKind.PARTICLE, "closed-record")],
        payload={"cause": "lapsed"},
    )
    await session.commit()


class TestMeasure:
    @pytest.mark.asyncio
    async def test_counts_the_store_and_the_window(self, db_session: AsyncSession) -> None:
        await _seed(db_session)
        measure = await measure_closure(
            db_session, window_start=None, window_end=datetime.now(UTC) + timedelta(seconds=5)
        )

        assert (measure.contested, measure.active) == (1, 2)
        assert measure.contested_fraction == 0.5
        assert measure.contested_by_basis == {"inconsistency": 1}
        assert measure.transitions == {
            "retirement": {"autonomous": 1, "gesture": 1},
            "contradiction_closed": {"autonomous": 1, "gesture": 0},
            "promotion": {"autonomous": 0, "gesture": 0},
        }
        assert measure.autonomous_by_reason == {"SUPERSEDED_BY_UPDATE": 1}
        assert measure.gesture_by_actor == {"curate": 1}
        assert measure.autonomous_share == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_a_window_that_opens_later_sees_none_of_it(
        self, db_session: AsyncSession
    ) -> None:
        await _seed(db_session)
        later = datetime.now(UTC) + timedelta(seconds=5)
        measure = await measure_closure(
            db_session, window_start=later, window_end=later + timedelta(seconds=5)
        )

        # The contested count is store-wide; only the transitions are windowed.
        assert (measure.contested, measure.active) == (1, 2)
        assert measure.autonomous + measure.gesture == 0
        assert measure.autonomous_share is None

    @pytest.mark.asyncio
    async def test_promotions_split_by_the_accept_event(self, db_session: AsyncSession) -> None:
        for pid in (_DERIVED, _ACCEPTED):
            await insert_particle(
                db_session, _claim(pid, f"Abstraction {pid[-2:]}.", asserted_by=ABSTRACTION_ACTOR)
            )
        await record_event(
            db_session,
            actor="curate",
            event_type=OperatorEventType.ABSTRACTION_RESOLVED,
            refs=[(EventRefKind.PARTICLE, _ACCEPTED)],
            payload={"resolution": "accepted", "particle_id": _ACCEPTED},
        )
        await record_event(
            db_session,
            actor="curate",
            event_type=OperatorEventType.ABSTRACTION_RESOLVED,
            payload={"resolution": "rejected", "particle_id": None},
        )
        await db_session.commit()

        measure = await measure_closure(
            db_session, window_start=None, window_end=datetime.now(UTC) + timedelta(seconds=5)
        )
        assert measure.transitions["promotion"] == {"autonomous": 1, "gesture": 1}

    @pytest.mark.asyncio
    async def test_payload_round_trips(self, db_session: AsyncSession) -> None:
        await _seed(db_session)
        measure = await measure_closure(
            db_session, window_start=None, window_end=datetime.now(UTC) + timedelta(seconds=5)
        )
        payload = measure.payload()
        assert payload["contested_fraction"] == 0.5
        assert payload["autonomous"] == 2
        assert payload["gesture"] == 1

        back = measure_from_payload({"closure": json.loads(json.dumps(payload))})
        assert back is not None
        assert back.transitions == measure.transitions
        assert back.window_end == measure.window_end


# ---------------------------------------------------------------------------
# Window tiling and history
# ---------------------------------------------------------------------------


def _run_event(payload: dict[str, Any], *, actor: str = "memory-consolidate") -> OperatorEvent:
    return OperatorEvent(
        event_id="e1",
        occurred_at=datetime(2026, 10, 3, 4, 0),
        actor=actor,
        event_type=OperatorEventType.CONSOLIDATION_RUN,
        payload=payload,
    )


def _measure(end: datetime, *, contested: int = 2, active: int = 10) -> ClosureMeasure:
    return ClosureMeasure(
        window_end=end,
        active=active,
        contested=contested,
        transitions={
            "retirement": {"autonomous": 3, "gesture": 1},
            "contradiction_closed": {"autonomous": 0, "gesture": 0},
            "promotion": {"autonomous": 0, "gesture": 0},
        },
    )


class TestWindow:
    def test_the_first_run_opens_at_the_start_of_the_store(self) -> None:
        assert window_start_after(None) is None

    def test_opens_where_the_previous_window_closed(self) -> None:
        end = datetime(2026, 10, 3, 3, 31, tzinfo=UTC)
        prior = _run_event({"closure": _measure(end).payload()})
        assert window_start_after(prior) == end

    def test_a_run_that_predates_the_measure_opens_at_its_completion(self) -> None:
        prior = _run_event({"completed_at": "2026-10-02T03:40:00+00:00"})
        assert window_start_after(prior) == datetime(2026, 10, 2, 3, 40, tzinfo=UTC)

    def test_a_naive_timestamp_reads_as_utc(self) -> None:
        start = window_start_after(_run_event({}))
        assert start == datetime(2026, 10, 3, 4, 0, tzinfo=UTC)


class TestHistory:
    @pytest.mark.asyncio
    async def test_reads_measured_runs_and_counts_the_rest(self, db_session: AsyncSession) -> None:
        end = datetime(2026, 10, 3, 3, 31, tzinfo=UTC)
        await record_event(
            db_session,
            actor="memory-consolidate",
            event_type=OperatorEventType.CONSOLIDATION_RUN,
            payload={"format": 1},
        )
        await record_event(
            db_session,
            actor="memory-consolidate",
            event_type=OperatorEventType.CONSOLIDATION_RUN,
            payload={"format": 1, "closure": _measure(end).payload()},
        )
        # The interactive audit records the same event type and never measures.
        await record_event(
            db_session,
            actor="audit",
            event_type=OperatorEventType.CONSOLIDATION_RUN,
            payload={"format": 1},
        )
        await db_session.commit()

        history = await closure_history(db_session, store="default", actor="memory-consolidate")
        assert history.unmeasured_runs == 1
        assert [r.measure.contested for r in history.runs] == [2]
        rendered = render_closure_history(history)
        assert "20.0%" in rendered
        assert "75.0%" in rendered
        assert "1 earlier run(s) by memory-consolidate predate the measure." in rendered

    def test_an_empty_history_says_how_to_start_one(self) -> None:
        from particles.operations.closure_measure import ClosureHistory

        rendered = render_closure_history(ClosureHistory(store="default", actor="x"))
        assert "No consolidation run has recorded the measure yet" in rendered


class TestRender:
    def test_report_lines_name_both_numbers_and_the_previous_run(self) -> None:
        end = datetime(2026, 10, 3, 3, 31, tzinfo=UTC)
        now = _measure(end, contested=3)
        now = now.model_copy(update={"window_start": end - timedelta(days=1)})
        contested, transitions = render_closure_lines(now, _measure(end, contested=2))
        assert "3 of 10 active beliefs (30.0%)" in contested
        assert "previous run 20.0%" in contested
        assert "4 since the previous run: 3 autonomous, 1 by gesture (75.0% autonomous)" in (
            transitions
        )

    def test_first_run_says_where_its_window_opened(self) -> None:
        empty = ClosureMeasure(window_end=datetime(2026, 10, 3, tzinfo=UTC))
        contested, transitions = render_closure_lines(empty, None)
        assert "(n/a)" in contested
        assert "since the store began" in transitions
        assert "autonomous)" not in transitions


# ---------------------------------------------------------------------------
# memory consolidate --history
# ---------------------------------------------------------------------------


class TestHistoryCli:
    def _seed_run(self) -> None:
        async def _write() -> None:
            from particles.db import session_scope

            async with session_scope() as session:
                await record_event(
                    session,
                    actor="memory-consolidate",
                    event_type=OperatorEventType.CONSOLIDATION_RUN,
                    payload={
                        "format": 1,
                        "closure": _measure(datetime(2026, 10, 3, 3, 31, tzinfo=UTC)).payload(),
                    },
                )
                await session.commit()

        asyncio.run(_write())

    def test_prints_the_series(self, cli_db: Path) -> None:
        from particles.api.cli import app

        self._seed_run()
        result = CliRunner().invoke(app, ["memory", "consolidate", "--history"])
        assert result.exit_code == 0, result.output
        assert "Contested fraction and transition split per consolidation run" in result.output
        assert "20.0%" in result.output

    def test_json(self, cli_db: Path) -> None:
        from particles.api.cli import app

        self._seed_run()
        result = CliRunner().invoke(app, ["memory", "consolidate", "--history", "--format", "json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["runs"][0]["contested_fraction"] == 0.2
        assert data["runs"][0]["autonomous_share"] == 0.75

    def test_refuses_to_combine_with_a_run_flag(self, cli_db: Path) -> None:
        from particles.api.cli import app

        result = CliRunner().invoke(app, ["memory", "consolidate", "--history", "--dry-run"])
        assert result.exit_code == 2
        assert "--history reads past runs" in result.output

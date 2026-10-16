# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure tests for the §6.6 verdict-to-writes table (``ingest/conflict_plan.py``).

Every ladder outcome maps to a :class:`ConflictWritePlan` with no DB. The
DB-level effect of applying a plan stays covered by ``TestConflictWritePath``
in ``tests/test_extract.py`` and the per-feature suites.
"""

from __future__ import annotations

import pytest

from particles.core.conflict_resolution import RETIRED_VALUE_KEY, ConflictVerdict, LadderOutcome
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.ingest.conflict_plan import (
    ConflictWritePlan,
    plan_conflict_writes,
    plan_retired_hold,
)
from particles.store.event_store import EventRefKind, OperatorEventType


def _particle(content: str, *, subject_ids: list[str] | None = None) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1", snapshot_id="snap-1"
            )
        ],
        asserted_by="test",
        subject_ids=subject_ids or [],
    )


EXISTING = _particle("The tower is 300 metres tall.", subject_ids=["s-tower"])
NEW = _particle("The tower is 324 metres tall.")


def _plan(
    verdict: ConflictVerdict | None,
    *,
    new: Particle = NEW,
    record_divergence: bool = False,
    scores: tuple[float | None, float | None] = (0.2, 0.9),
) -> ConflictWritePlan:
    return plan_conflict_writes(
        LadderOutcome(verdict=verdict, record_divergence=record_divergence),
        new,
        EXISTING,
        corpus_entry_id="entry-2",
        snapshot_id="snap-2",
        trigger_ref_type=ProvenanceRefType.SOURCE,
        scores=scores,
        domain="test-domain",
    )


def _no_side_writes(plan: ConflictWritePlan) -> None:
    assert plan.wrapper is None
    assert plan.dropped_event is None
    assert not plan.record_divergence


def _records_winner(plan: ConflictWritePlan, retired: Particle, winner: Particle) -> None:
    event = plan.update_event
    assert event is not None
    assert event.event_type is OperatorEventType.UPDATE_SUPERSESSION
    assert event.payload == {"retired_id": retired.id, "winner_id": winner.id, "route": "ladder"}
    assert set(event.refs) == {
        (EventRefKind.PARTICLE, retired.id),
        (EventRefKind.PARTICLE, winner.id),
    }


class TestInsertActiveOnly:
    @pytest.mark.parametrize("verdict", [ConflictVerdict.CORROBORATES, ConflictVerdict.NO_CONFLICT])
    def test_new_written_active_beside_existing(self, verdict: ConflictVerdict) -> None:
        plan = _plan(verdict)
        assert plan.insert is NEW
        assert plan.insert.status is Status.ACTIVE
        assert plan.demote is None
        assert plan.cache_add_new and not plan.cache_drop_existing
        assert plan.result is NEW
        _no_side_writes(plan)

    @pytest.mark.parametrize("record", [True, False])
    def test_declined_pair(self, record: bool) -> None:
        plan = _plan(None, record_divergence=record)
        assert plan.insert is NEW
        assert plan.demote is None
        assert plan.record_divergence is record
        assert plan.cache_add_new and not plan.cache_drop_existing
        assert plan.result is NEW
        assert plan.wrapper is None and plan.dropped_event is None


class TestNewWins:
    @pytest.mark.parametrize(
        ("verdict", "reason"),
        [
            (ConflictVerdict.SUPERSEDES, StatusReason.LOWER_TRUST_SOURCE),
            (ConflictVerdict.DOCUMENT_SUPERSEDES, StatusReason.DOCUMENT_SUPERSEDED),
        ],
    )
    def test_insert_new_then_demote_existing(
        self, verdict: ConflictVerdict, reason: StatusReason
    ) -> None:
        plan = _plan(verdict)
        assert plan.insert is NEW
        assert plan.demote == (EXISTING.id, reason)
        assert plan.cache_drop_existing and plan.cache_add_new
        assert plan.result is NEW
        assert NEW.supersedes is None  # only rung 2.5 names the replaced claim
        _no_side_writes(plan)

    def test_update_supersedes_fills_supersedes_on_the_minted_row(self) -> None:
        plan = _plan(ConflictVerdict.UPDATE_SUPERSEDES)
        assert plan.insert is not None
        assert plan.insert.id == NEW.id
        assert plan.insert.supersedes == EXISTING.id
        assert plan.insert.status is Status.ACTIVE
        assert plan.demote == (EXISTING.id, StatusReason.SUPERSEDED_BY_UPDATE)
        assert plan.cache_drop_existing and plan.cache_add_new
        assert plan.result is plan.insert
        assert NEW.supersedes is None  # the input is not mutated
        _no_side_writes(plan)
        _records_winner(plan, EXISTING, NEW)

    def test_update_supersedes_keeps_a_set_supersedes(self) -> None:
        # Write-once (D1): an already-set pointer is never overwritten.
        preset = NEW.model_copy(update={"supersedes": "earlier-claim"})
        plan = _plan(ConflictVerdict.UPDATE_SUPERSEDES, new=preset)
        assert plan.insert is preset
        assert plan.result is preset
        # The pointer is taken, so the event is the only record of this winner.
        _records_winner(plan, EXISTING, preset)

    def test_the_mirror_records_the_existing_claim_as_winner(self) -> None:
        plan = _plan(ConflictVerdict.UPDATE_SUPERSEDED_BY_EXISTING)
        _records_winner(plan, NEW, EXISTING)

    @pytest.mark.parametrize(
        "verdict",
        [
            ConflictVerdict.CORROBORATES,
            ConflictVerdict.SUPERSEDES,
            ConflictVerdict.DOCUMENT_SUPERSEDES,
            ConflictVerdict.INCONSISTENT,
        ],
    )
    def test_only_rung_2_5_records_a_winner(self, verdict: ConflictVerdict) -> None:
        assert _plan(verdict).update_event is None


class TestExistingWins:
    def test_trust_drop_writes_only_the_audit_event(self) -> None:
        plan = _plan(ConflictVerdict.SUPERSEDED_BY_EXISTING)
        assert plan.insert is None and plan.demote is None and plan.wrapper is None
        assert not plan.cache_add_new and not plan.cache_drop_existing
        assert plan.result is None
        event = plan.dropped_event
        assert event is not None
        assert event.event_type is OperatorEventType.CONFLICT_CANDIDATE_DROPPED
        assert event.actor == "extract-pipeline"
        assert event.refs == ((EventRefKind.PARTICLE, EXISTING.id),)
        assert event.payload == {
            "verdict": "SUPERSEDED_BY_EXISTING",
            "candidate_id": NEW.id,
            "candidate_excerpt": NEW.content[:240],
            "winning_particle_id": EXISTING.id,
            "trust_score_new": 0.2,
            "trust_score_existing": 0.9,
        }

    @pytest.mark.parametrize(
        ("verdict", "reason"),
        [
            (ConflictVerdict.DOCUMENT_SUPERSEDED_BY_EXISTING, StatusReason.DOCUMENT_SUPERSEDED),
            (ConflictVerdict.UPDATE_SUPERSEDED_BY_EXISTING, StatusReason.SUPERSEDED_BY_UPDATE),
        ],
    )
    def test_mirror_inserts_active_then_demotes_the_new_row(
        self, verdict: ConflictVerdict, reason: StatusReason
    ) -> None:
        # The insert seam forbids a born-PROVENANCE_STALE row other than the
        # CONFLICT_PENDING quarantine, so the loser is inserted ACTIVE first.
        plan = _plan(verdict)
        assert plan.insert is NEW
        assert plan.insert.status is Status.ACTIVE
        assert plan.demote == (NEW.id, reason)
        assert not plan.cache_add_new and not plan.cache_drop_existing
        assert plan.result is not None
        assert plan.result.id == NEW.id
        assert plan.result.status is Status.PROVENANCE_STALE
        assert plan.result.status_reason is reason
        assert plan.result.supersedes is None
        _no_side_writes(plan)


class TestInconsistent:
    def test_quarantine_then_wrapper(self) -> None:
        plan = _plan(ConflictVerdict.INCONSISTENT)
        assert plan.insert is not None
        assert plan.insert.id == NEW.id
        assert plan.insert.status is Status.PROVENANCE_STALE
        assert plan.insert.status_reason is StatusReason.CONFLICT_PENDING
        assert plan.demote is None
        wrapper = plan.wrapper
        assert wrapper is not None
        assert wrapper.status is Status.INCONSISTENCY
        # The B ref names the persisted quarantined row, never a dangling id.
        refs = [(r.type, r.corpus_entry_id) for r in wrapper.provenance]
        assert refs == [
            (ProvenanceRefType.PARTICLE, EXISTING.id),
            (ProvenanceRefType.PARTICLE, NEW.id),
            (ProvenanceRefType.SOURCE, "entry-2"),
        ]
        assert wrapper.subject_ids == ["s-tower"]
        assert wrapper.properties is None
        assert plan.wrapper_domain_hint == "test-domain"
        assert plan.result is wrapper
        assert not plan.cache_add_new and not plan.cache_drop_existing
        assert plan.dropped_event is None and not plan.record_divergence

    def test_particle_trigger_ref(self) -> None:
        plan = plan_conflict_writes(
            LadderOutcome(ConflictVerdict.INCONSISTENT, record_divergence=False),
            NEW,
            EXISTING,
            corpus_entry_id=NEW.id,
            snapshot_id="",
            trigger_ref_type=ProvenanceRefType.PARTICLE,
            scores=(None, None),
            domain=None,
        )
        assert plan.wrapper is not None
        trigger = plan.wrapper.provenance[2]
        assert trigger.type is ProvenanceRefType.PARTICLE
        assert trigger.snapshot_id is None


class TestRetiredHold:
    def test_same_shape_as_inconsistent_with_the_retired_marker(self) -> None:
        twin = EXISTING.model_copy(
            update={
                "status": Status.RETRACTED,
                "status_reason": StatusReason.EXPLICIT_RETRACTION,
            }
        )
        plan = plan_retired_hold(
            twin,
            NEW,
            corpus_entry_id="entry-2",
            snapshot_id="snap-2",
            trigger_ref_type=ProvenanceRefType.SOURCE,
            domain="test-domain",
        )
        assert plan.insert is not None
        assert plan.insert.status_reason is StatusReason.CONFLICT_PENDING
        assert plan.wrapper is not None
        assert plan.wrapper.properties == {RETIRED_VALUE_KEY: "EXPLICIT_RETRACTION"}
        assert plan.wrapper.provenance[1].corpus_entry_id == NEW.id
        assert plan.wrapper_domain_hint == "test-domain"
        assert plan.result is plan.wrapper
        # The twin itself is never touched.
        assert plan.demote is None
        assert not plan.cache_add_new and not plan.cache_drop_existing
        assert plan.dropped_event is None and not plan.record_divergence


def test_every_verdict_is_planned() -> None:
    for verdict in ConflictVerdict:
        assert isinstance(_plan(verdict), ConflictWritePlan)


class TestDecidedBy:
    """the record that carries a verdict names the rung that decided it."""

    def _plan(self, verdict: ConflictVerdict, decided_by: str | None) -> ConflictWritePlan:
        return plan_conflict_writes(
            LadderOutcome(verdict, record_divergence=False),
            NEW,
            EXISTING,
            corpus_entry_id="entry-2",
            snapshot_id="snap-2",
            trigger_ref_type=ProvenanceRefType.SOURCE,
            scores=(None, None),
            domain=None,
            decided_by=decided_by,
        )

    def test_the_inconsistency_record_names_its_rung(self) -> None:
        from particles.core.conflict_resolution import DECIDED_BY_KEY

        plan = self._plan(ConflictVerdict.INCONSISTENT, "second_reading@abc")
        assert plan.wrapper is not None
        assert (plan.wrapper.properties or {})[DECIDED_BY_KEY] == "second_reading@abc"
        unstamped = self._plan(ConflictVerdict.INCONSISTENT, None)
        assert unstamped.wrapper is not None
        assert DECIDED_BY_KEY not in (unstamped.wrapper.properties or {})

    def test_the_update_event_names_its_rung(self) -> None:
        plan = self._plan(ConflictVerdict.UPDATE_SUPERSEDES, "update_slot_second@h")
        assert plan.update_event is not None
        assert plan.update_event.payload["decided_by"] == "update_slot_second@h"

    def test_the_pipeline_names_the_rung_each_outcome_rests_on(self) -> None:
        from particles.ingest.pipeline import (
            _outcome_decided_by,
            reconcile_probe_stamp,
            second_reading_stamp,
            update_decided_by,
        )

        inconsistent = ConflictVerdict.INCONSISTENT
        assert _outcome_decided_by(inconsistent, probe=True, reading="note") == str(
            second_reading_stamp()
        )
        assert _outcome_decided_by(inconsistent, probe=True, reading=None) == str(
            reconcile_probe_stamp()
        )
        # A fail-closed record whose probe did not complete rests on no rung.
        assert _outcome_decided_by(inconsistent, probe=None, reading=None) is None
        assert (
            _outcome_decided_by(ConflictVerdict.UPDATE_SUPERSEDES, probe=False, reading=None)
            == update_decided_by()
        )
        assert update_decided_by().startswith("update_slot_second@")
        assert _outcome_decided_by(ConflictVerdict.CORROBORATES, probe=False, reading=None) is None

    @pytest.mark.asyncio
    async def test_a_retirement_on_the_update_answer_alone_names_the_rung_that_gave_it(
        self,
    ) -> None:
        """The update-first path names the rung that gave its answer.

        A pair settled on the update question alone skipped the contradiction
        probe, not the update ladder: ``same_slot_update`` is set only for a
        pair whose update reading settled ``CHANGES``, and every rung 2.5 door
        is gated on that verdict, which only the ladder's confirmation rung
        returns. The stamp therefore names that rung on the update-first path
        and on the probe-confirmed path alike, and no deterministic key is
        named because none decides: the dating and the ALEATORY condition are
        rung 2.5's own conditions, not a rung of the update ladder.
        """
        from unittest.mock import patch

        from particles.core.conflict_resolution import (
            PairPrecondition,
            RungInputs,
            SlotVerdict,
            decide_ladder,
            update_order_source,
        )
        from particles.core.rung_ladder import RungStamp
        from particles.ingest import pipeline
        from particles.ingest.pipeline import (
            UPDATE_SECOND_RUNG,
            _outcome_decided_by,
            read_update_slot,
            update_decided_by,
            update_prompt_hash,
        )

        async def changes(*_args: object, **_kwargs: object) -> str:
            return "REASON: one slot.\nSLOT: CHANGES\nVERDICT: YES"

        with (
            patch.object(pipeline, "_llm_call", changes),
            patch.object(pipeline, "llm_circuit_open", lambda: False),
        ):
            reading = await read_update_slot(EXISTING.content, NEW.content)
        assert reading.verdict is SlotVerdict.CHANGES
        assert reading.decided_by == RungStamp(UPDATE_SECOND_RUNG, update_prompt_hash())

        # No rung 2.5 door opens on any other answer, so no retirement rests on
        # a rung that did not run.
        for slot in (None, SlotVerdict.FIXED, SlotVerdict.DIFFERENT):
            assert (
                update_order_source(
                    allow_update=True, allow_own_assertion=True, forced=False, slot=slot
                )
                is None
            )

        # The update-first path: no contradiction signal, rung 2.5 acts on the answer.
        outcome = decide_ladder(
            EXISTING,
            NEW,
            probe=False,
            fail_closed=False,
            precondition=PairPrecondition.RECONCILE,
            rung_inputs=RungInputs(update_order=1),
            single_trust_order=True,
            trust_differential_threshold=0.1,
            same_slot_update=True,
        )
        assert outcome.verdict is ConflictVerdict.UPDATE_SUPERSEDES
        stamped = _outcome_decided_by(outcome.verdict, probe=False, reading=None)
        assert stamped == str(reading.decided_by) == update_decided_by()
        # The probe-confirmed path names the same rung: the probe is not what retired.
        assert _outcome_decided_by(ConflictVerdict.UPDATE_SUPERSEDES, probe=True, reading=None) == (
            stamped
        )

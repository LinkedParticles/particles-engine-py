# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The per-record ``OPEN_INCONSISTENCY`` lint finding.

One WARNING per open INCONSISTENCY record, whatever its members' statuses,
carrying the members by side so the curation card can be keyed by the record.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.contradiction_disclosure import CENSUS_ORIGIN, ORIGIN_KEY, SIDES_KEY
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.operations.lint import run_lint
from particles.operations.lint.open_inconsistency import (
    open_inconsistency_findings,
    record_sides,
)
from particles.store.particle_store import insert_particle, update_particle_status


def _claim(content: str, status: Status = Status.ACTIVE) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1")],
        asserted_by="general-extractor",
        subject_ids=["sid"],
        status=status,
        status_reason=StatusReason.CONFLICT_PENDING if status is Status.PROVENANCE_STALE else None,
    )


def _record(*member_ids: str, properties: dict[str, object] | None = None) -> Particle:
    return Particle(
        content="INCONSISTENCY: conflict between two claims.",
        confidence=Confidence(value=0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=pid)
            for pid in member_ids
        ],
        asserted_by="extract-pipeline",
        status=Status.INCONSISTENCY,
        properties=properties,
    )


class TestRecordSides:
    def test_a_plain_record_names_a_then_b(self) -> None:
        assert record_sides(_record("a", "b")) == (["a"], ["b"])

    def test_a_dangling_record_has_an_empty_side(self) -> None:
        assert record_sides(_record("a")) == (["a"], [])

    def test_a_census_record_reads_its_sides(self) -> None:
        record = _record(
            "a1",
            "b1",
            "a2",
            properties={ORIGIN_KEY: CENSUS_ORIGIN, SIDES_KEY: {"a": ["a1", "a2"], "b": ["b1"]}},
        )
        assert record_sides(record) == (["a1", "a2"], ["b1"])


class TestFinding:
    @pytest.mark.asyncio
    async def test_one_warning_per_open_record_whatever_its_members(
        self, db_session: AsyncSession
    ) -> None:
        active = _claim("The window is 30 days.")
        loser = _claim("The window is 14 days.", Status.PROVENANCE_STALE)
        demoted = _claim("A claim whose basis was retracted.")
        for p in (active, loser, demoted):
            await insert_particle(db_session, p)
        await update_particle_status(
            db_session, demoted.id, Status.PROVENANCE_STALE, StatusReason.RETRACTED_DEPENDENCY
        )
        open_one = _record(active.id, loser.id)
        both_down = _record(demoted.id, loser.id)
        dangling = _record("gone", active.id)
        closed = _record(active.id, loser.id)
        for r in (open_one, both_down, dangling, closed):
            await insert_particle(db_session, r)
        await update_particle_status(
            db_session, closed.id, Status.RETRACTED, StatusReason.CONFLICT_RESOLVED
        )
        await db_session.flush()

        findings = {f.particle_id: f for f in await open_inconsistency_findings(db_session)}
        assert set(findings) == {open_one.id, both_down.id, dangling.id}
        f = findings[open_one.id]
        assert (f.finding_type, f.severity) == ("OPEN_INCONSISTENCY", "WARNING")
        assert f.inconsistency_id == open_one.id
        assert f.conflict_sides == [[active.id], [loser.id]]
        assert f.particle_content == open_one.content
        assert "“The window is 30 days.”" in f.detail
        assert "“The window is 14 days.” [PROVENANCE_STALE / CONFLICT_PENDING]" in f.detail
        assert f"inconsistency:{open_one.id}" in (f.recommended_action or "")
        assert "(no longer in the store)" in findings[dangling.id].detail

    @pytest.mark.asyncio
    async def test_lint_reports_open_conflicts(self, db_session: AsyncSession) -> None:
        a, b = _claim("a"), _claim("b")
        for p in (a, b):
            await insert_particle(db_session, p)
        record = _record(a.id, b.id)
        await insert_particle(db_session, record)
        await db_session.flush()

        report = await run_lint(db_session, fix=False, semantic=False)
        found = [f for f in report.findings if f.finding_type == "OPEN_INCONSISTENCY"]
        assert [f.particle_id for f in found] == [record.id]
        assert report.summary.get("OPEN_INCONSISTENCY") == 1


class TestRetiredValueFinding:
    """A retired-value record says what it holds, not "Open conflict"."""

    @staticmethod
    def _pair(b_content: str = "The port is 7000.") -> tuple[Particle, Particle]:
        a = _claim("The port is 7000.").model_copy(
            update={"status": Status.RETRACTED, "status_reason": StatusReason.EXPLICIT_RETRACTION}
        )
        return a, _claim(b_content, Status.PROVENANCE_STALE)

    def test_an_identical_restatement_is_named_once(self) -> None:
        from particles.core.conflict_resolution import RETIRED_VALUE_KEY
        from particles.operations.lint.open_inconsistency import build_finding

        a, b = self._pair()
        record = _record(a.id, b.id, properties={RETIRED_VALUE_KEY: "EXPLICIT_RETRACTION"})
        finding = build_finding(record, {a.id: a, b.id: b})

        assert finding.detail == (
            "Restated after retraction: “The port is 7000.” [RETRACTED / EXPLICIT_RETRACTION] "
            "was retracted, and a later source states it again "
            "[PROVENANCE_STALE / CONFLICT_PENDING], held for review"
        )
        assert finding.recommended_action is not None
        assert "Keep the retraction:" in finding.recommended_action
        assert "Restore the claim:" in finding.recommended_action
        assert "--action PREFER_A" in finding.recommended_action
        assert "--action PREFER_B" in finding.recommended_action

    def test_a_reworded_restatement_is_quoted(self) -> None:
        from particles.core.conflict_resolution import RETIRED_VALUE_KEY
        from particles.operations.lint.open_inconsistency import build_finding

        a, b = self._pair("The port is 7000")
        record = _record(a.id, b.id, properties={RETIRED_VALUE_KEY: "EXPLICIT_RETRACTION"})
        detail = build_finding(record, {a.id: a, b.id: b}).detail
        assert "a later source restates it as “The port is 7000” [PROVENANCE_STALE" in detail

    def test_an_ordinary_record_keeps_the_conflict_wording(self) -> None:
        from particles.operations.lint.open_inconsistency import build_finding

        a, b = _claim("The port is 7000."), _claim("The port is 7001.")
        finding = build_finding(_record(a.id, b.id), {a.id: a, b.id: b})
        assert finding.detail == "Open conflict: “The port is 7000.” vs “The port is 7001.”"
        assert finding.conflict_note == ""
        assert finding.recommended_action == (
            f"Resolve it: particles curate apply resolve inconsistency:{finding.particle_id} "
            f"--action PREFER_A|PREFER_B|BOTH_VALID|DISCARD (or particles review "
            f"{finding.particle_id})"
        )


def test_the_conflict_note_is_the_detail_without_the_quoted_sides() -> None:
    # A surface that draws both sides shows only the note, so
    # the examination line rides in it as well as at the end of the detail.
    from particles.operations.lint.open_inconsistency import build_finding

    a, b, c = _claim("The port is 7000."), _claim("The port is 7001."), _claim("Port 7002.")
    examination = {"partner_reason": "SUPERSEDED_BY_UPDATE", "blocking_ids": [c.id]}
    finding = build_finding(_record(a.id, b.id), {a.id: a, b.id: b, c.id: c}, examination)
    note = "Claim A was retired by update; the held claim B is still contradicted by “Port 7002.”."
    assert finding.conflict_note == note
    assert finding.detail.endswith(f"”. {note}")

# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the judged merge of re-read duplicates.

Covers `particles/operations/reread_merge.py`: the pair gates, the
survivor-anchored questions, the write rule, the admission arithmetic, the
three steps (`--judge` writes ledger rows only, `--apply` asks no model and
refuses without both flags), and `links unmerge` reverting a re-read merge.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from sqlalchemy import select

from particles.core.schema import (
    CandidateCluster,
    CoEvidentialCandidate,
    Confidence,
    JudgeVerdictKind,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    RelationCreatedBy,
    SuggestMode,
    SuggestReport,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.operations import reread_merge as rm
from particles.operations.links_suggest import AutoMergeDisabled, unmerge_exact_duplicates
from particles.store.event_store import OperatorEventType, list_events
from particles.store.particle_store import get_particle, insert_particle
from particles.store.relation_store import ParticleRelationRow

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_T0 = datetime(2026, 7, 1, tzinfo=UTC)


def _at(day: int) -> datetime:
    return _T0.replace(day=day)


def _facts(
    pid: str,
    content: str = "x",
    *,
    entries: tuple[str, ...] = ("e1",),
    snaps: tuple[str, ...] = ("s1",),
    first: int = 1,
    asserted: int = 1,
    subjects: tuple[str, ...] = ("S",),
    holder: str | None = None,
    sources_only: bool = True,
) -> rm.ClaimFacts:
    return rm.ClaimFacts(
        id=pid,
        content=content,
        entry_ids=frozenset(entries),
        snapshot_ids=frozenset(snaps),
        first_read=_at(first),
        asserted_at=_at(asserted),
        subject_ids=frozenset(subjects),
        holder=holder,
        sources_only=sources_only,
    )


# ---------------------------------------------------------------------------
# Pure core
# ---------------------------------------------------------------------------


class TestPopulation:
    AO = {"e1": True, "e2": True, "m1": False}

    def test_two_readings_of_one_append_only_entry(self) -> None:
        a = _facts("a", "PR #1 is merged.", snaps=("s1",))
        b = _facts("b", "PR #1 has been merged.", snaps=("s2",))
        assert rm.in_population(a, b, append_only=self.AO)

    @pytest.mark.parametrize(
        ("b_kwargs", "why"),
        [
            ({"entries": ("e1", "e2")}, "a second entry"),
            ({"entries": ("m1",)}, "another entry"),
            ({"snaps": ("s1", "s2")}, "a shared snapshot"),
            ({"holder": "alice"}, "a holder mismatch"),
            ({"sources_only": False}, "a non-source ref"),
        ],
    )
    def test_gates_exclude(self, b_kwargs: dict[str, Any], why: str) -> None:
        a = _facts("a", "PR #1 is merged.")
        b = _facts("b", "PR #1 has been merged.", **{"snaps": ("s2",), **b_kwargs})
        assert not rm.in_population(a, b, append_only=self.AO), why

    def test_mutable_entry_excluded(self) -> None:
        a = _facts("a", "PR #1 is merged.", entries=("m1",))
        b = _facts("b", "PR #1 has been merged.", entries=("m1",), snaps=("s2",))
        assert not rm.in_population(a, b, append_only=self.AO)

    def test_key_equal_pair_is_left_to_links_dedup(self) -> None:
        a = _facts("a", "PR #1 is merged.")
        b = _facts("b", "PR #1 is merged", snaps=("s2",))
        assert not rm.in_population(a, b, append_only=self.AO)


class TestPlan:
    def test_first_reading_survives_and_every_member_is_asked_against_it(self) -> None:
        # c is the first reading but was extracted last; the finder paired
        # a~b and b~c only, never a~c.
        facts = {
            "a": _facts("a", snaps=("s2",), first=2, asserted=2),
            "b": _facts("b", snaps=("s3",), first=3, asserted=3),
            "c": _facts("c", snaps=("s1",), first=1, asserted=9),
        }
        plan = rm.plan_questions([("a", "b"), ("b", "c")], facts, lambda x, y: 0.98)
        assert plan.components == 1
        assert {(q.survivor_id, q.copy_id) for q in plan.questions} == {("c", "a"), ("c", "b")}
        assert all(q.stratum == "high" for q in plan.questions)

    def test_members_not_asked(self) -> None:
        facts = {
            "s": _facts("s", snaps=("s1",), first=1),
            "shared": _facts("shared", snaps=("s1", "s4"), first=1, asserted=5),
            "extra": _facts("extra", snaps=("s3",), first=3, subjects=("S", "T")),
            "ok": _facts("ok", snaps=("s2",), first=2),
        }
        pairs = [("s", "ok"), ("ok", "shared"), ("ok", "extra")]
        plan = rm.plan_questions(pairs, facts, lambda x, y: None)
        assert [(q.survivor_id, q.copy_id) for q in plan.questions] == [("s", "ok")]
        assert plan.skipped[rm.Skip.SHARED_SNAPSHOT] == 1
        assert plan.skipped[rm.Skip.EXTRA_SUBJECT] == 1
        assert plan.questions[0].stratum == "low"


class TestDecide:
    CLEAR = rm.UpdateRecord(changes=False, second=False)

    def _decide(self, **kw: Any) -> rm.Block | None:
        args: dict[str, Any] = {
            "same": True,
            "update": self.CLEAR,
            "contradiction": False,
            "restored": False,
            "stratum": "high",
            "admitted": "all",
        }
        args.update(kw)
        return rm.decide_merge(**args)

    def test_merges_when_every_reading_agrees(self) -> None:
        assert self._decide() is None

    @pytest.mark.parametrize(
        ("kw", "block"),
        [
            ({"same": None}, rm.Block.UNJUDGED),
            ({"same": False}, rm.Block.DIFFERENT),
            ({"update": None}, rm.Block.UPDATE_UNREAD),
            ({"update": rm.UpdateRecord(changes=True, second=True)}, rm.Block.UPDATE_CHANGES),
            # The second reading cleared it, but the first said CHANGES.
            ({"update": rm.UpdateRecord(changes=False, second=True)}, rm.Block.UPDATE_CHANGES),
            ({"contradiction": True}, rm.Block.CONTRADICTION),
            ({"restored": True}, rm.Block.RESTORED),
            ({"admitted": "none"}, rm.Block.NOT_ADMITTED),
            ({"admitted": "high", "stratum": "low"}, rm.Block.NOT_ADMITTED),
        ],
    )
    def test_blocks(self, kw: dict[str, Any], block: rm.Block) -> None:
        assert self._decide(**kw) is block

    def test_high_admission_merges_high(self) -> None:
        assert self._decide(admitted="high", stratum="high") is None


class TestAdmission:
    def test_wilson_figures_the_record_states(self) -> None:
        assert rm.wilson_lower_bound(100, 100) == pytest.approx(0.963, abs=5e-4)
        assert rm.wilson_lower_bound(99, 100) == pytest.approx(0.946, abs=5e-4)
        assert rm.wilson_lower_bound(109, 110) >= 0.95
        assert rm.wilson_lower_bound(140, 142) >= 0.95
        assert rm.wilson_lower_bound(139, 141) < 0.95

    def test_score_labels_per_stratum(self) -> None:
        rows = [{"stratum": "high", "label": "same", "guard": ""} for _ in range(100)]
        rows += [{"stratum": "low", "label": "same", "guard": ""} for _ in range(99)]
        rows += [{"stratum": "low", "label": "different", "guard": "update_changes"}]
        rows += [{"stratum": "low", "label": "", "guard": ""}]  # unlabelled, ignored
        high, low = rm.score_labels(rows)
        assert (high.labelled, high.correct, high.passes) == (100, 100, True)
        assert (low.labelled, low.correct, low.passes) == (100, 99, False)
        # The guard-blocked false SAME counts against the judge, not the rule.
        assert (low.rule_labelled, low.rule_correct) == (99, 99)

    def test_too_few_labels_never_pass(self) -> None:
        (high, _low) = rm.score_labels([{"stratum": "high", "label": "y"}] * 50)
        assert not high.passes


# ---------------------------------------------------------------------------
# The three steps, over a store
# ---------------------------------------------------------------------------

_SURVIVOR = "There are three commits on branch fix."
_COPY = "Branch fix has three commits."
_OTHER = "Branch fix holds three commits."


def _pid(n: int) -> str:
    return f"00000000-0000-0000-0000-{n:012d}"


async def _seed_corpus(session: AsyncSession, *, mutability: str = "APPEND_ONLY") -> None:
    from particles.corpus.store import CorpusEntryRow, SnapshotRow

    session.add(
        CorpusEntryRow(
            entry_id="entry-1",
            uri_r="file:///t.jsonl",
            source_type="CONVERSATION",
            mutability=mutability,
            fetch_policy="NEVER",
            created_at=_at(1),
            deposited_by="test",
        )
    )
    for n in (1, 2, 3):
        session.add(
            SnapshotRow(
                snapshot_id=f"snap-{n}",
                entry_id="entry-1",
                captured_at=_at(n),
                content_hash=f"hash-{n}",
                warc_record_type="RESPONSE",
                extraction_status="COMPLETE",
            )
        )
    await session.flush()


def _claim(n: int, content: str, snapshot: str, day: int) -> Particle:
    return Particle(
        id=_pid(n),
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        asserted_at=_at(day),
        subject_ids=["subj"],
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1", snapshot_id=snapshot
            )
        ],
    )


async def _seed(session: AsyncSession, *, mutability: str = "APPEND_ONLY") -> None:
    await _seed_corpus(session, mutability=mutability)
    await insert_particle(session, _claim(1, _SURVIVOR, "snap-1", 1))
    await insert_particle(session, _claim(2, _COPY, "snap-2", 2))
    await insert_particle(session, _claim(3, _OTHER, "snap-3", 3))
    await session.commit()


def _finder(*pairs: tuple[int, int, float]) -> Any:
    report = SuggestReport(
        mode=SuggestMode.REPORT,
        clusters=[
            CandidateCluster(
                subject_id="subj",
                candidates=[
                    CoEvidentialCandidate(particle_a=_pid(a), particle_b=_pid(b), similarity=s)
                    for a, b, s in pairs
                ],
            )
        ],
    )

    async def fake(*_args: Any, **_kwargs: Any) -> SuggestReport:
        return report

    return patch("particles.operations.links_suggest.suggest_co_evidential", fake)


class _Reading:
    def __init__(self, verdict: Any, decided_by: Any = None, purpose: str = "semantic_lint"):
        self.verdict = verdict
        self.decided_by = decided_by
        self.purpose = purpose


def _judge(*answers: JudgeVerdictKind | None) -> Any:
    async def fake(pairs: Any) -> list[JudgeVerdictKind | None]:
        return list(answers)[: len(pairs)]

    return patch.object(rm, "_judge_pairs", fake)


def _update(verdict: Any) -> Any:
    from particles.core.rung_ladder import RungStamp

    async def fake(earlier: str, later: str) -> _Reading:
        return _Reading(verdict, RungStamp("update_slot", "v1"))

    return patch.object(rm, "_read_update", fake)


def _enable(monkeypatch: pytest.MonkeyPatch, admitted: str = "all") -> None:
    from particles.config import get_config

    cfg = get_config().links_suggest
    monkeypatch.setattr(cfg.auto_merge, "enabled", True)
    monkeypatch.setattr(cfg.reread, "admitted", admitted)


class TestCensus:
    async def test_census_reads_only_and_estimates(self, db_session: AsyncSession) -> None:
        await _seed(db_session)
        with _finder((1, 2, 0.96), (2, 3, 0.98)):
            report = await rm.census_reread(db_session)
        assert (report.population_pairs, report.components, report.questions) == (2, 1, 2)
        assert {(i.survivor_id, i.copy_id) for i in report.items} == {
            (_pid(1), _pid(2)),
            (_pid(1), _pid(3)),
        }
        assert report.blocked == {"unjudged": 2}
        assert report.estimate is not None and report.estimate.judge_questions == 2

    async def test_mutable_entry_forms_no_pair(self, db_session: AsyncSession) -> None:
        await _seed(db_session, mutability="MUTABLE")
        with _finder((1, 2, 0.96)):
            report = await rm.census_reread(db_session)
        assert report.population_pairs == 0


class TestJudge:
    async def test_judge_writes_ledger_rows_only(self, db_session: AsyncSession) -> None:
        from particles.core.conflict_resolution import SlotVerdict

        await _seed(db_session)
        with (
            _finder((1, 2, 0.96)),
            _judge(JudgeVerdictKind.PARAPHRASE),
            _update(SlotVerdict.FIXED),
        ):
            report = await rm.judge_reread(db_session)
        assert (report.asked_judge, report.asked_update) == (1, 1)
        assert report.same == 1 and report.update_read == 1
        copy = await get_particle(db_session, _pid(2))
        assert copy is not None and copy.status is Status.ACTIVE
        assert await list_events(db_session, event_type=OperatorEventType.DUPLICATES_MERGED) == []

    async def test_unconfirmed_judge_asks_nothing(self, db_session: AsyncSession) -> None:
        await _seed(db_session)

        async def boom(pairs: Any) -> list[Any]:
            raise AssertionError("asked the judge")

        with _finder((1, 2, 0.96)), patch.object(rm, "_judge_pairs", boom):
            report = await rm.judge_reread(db_session, confirm=lambda r: False)
        assert report.asked_judge == 0


async def _judged(
    db_session: AsyncSession, answer: JudgeVerdictKind, slot: Any, *pairs: tuple[int, int, float]
) -> None:
    with _finder(*pairs), _judge(answer, answer), _update(slot):
        await rm.judge_reread(db_session)


class TestApply:
    async def test_refuses_without_auto_merge(self, db_session: AsyncSession) -> None:
        await _seed(db_session)
        with _finder((1, 2, 0.96)), pytest.raises(AutoMergeDisabled):
            await rm.apply_reread(db_session)

    async def test_refuses_while_not_admitted(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await _seed(db_session)
        _enable(monkeypatch, admitted="none")
        with _finder((1, 2, 0.96)), pytest.raises(rm.RereadNotAdmitted):
            await rm.apply_reread(db_session)

    async def test_apply_merges_from_the_ledger_and_asks_no_model(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.core.conflict_resolution import SlotVerdict

        await _seed(db_session)
        await _judged(db_session, JudgeVerdictKind.PARAPHRASE, SlotVerdict.FIXED, (1, 2, 0.96))
        _enable(monkeypatch)

        async def boom(*_a: Any, **_k: Any) -> Any:
            raise AssertionError("apply asked a model")

        with (
            _finder((1, 2, 0.96)),
            patch.object(rm, "_judge_pairs", boom),
            patch.object(rm, "_read_update", boom),
        ):
            report = await rm.apply_reread(db_session)
        assert (report.merged_groups, report.merged_copies, report.links_created) == (1, 1, 1)
        copy = await get_particle(db_session, _pid(2))
        survivor = await get_particle(db_session, _pid(1))
        assert copy is not None and copy.status is Status.SUPERSEDED
        assert copy.status_reason is StatusReason.DUPLICATE_MERGED
        assert survivor is not None and survivor.status is Status.ACTIVE
        edge = (await db_session.execute(select(ParticleRelationRow))).scalars().one()
        assert edge.created_by == RelationCreatedBy.REREAD_DUPLICATE.value
        (event,) = await list_events(db_session, event_type=OperatorEventType.DUPLICATES_MERGED)
        assert event.payload["basis"] == rm.BASIS
        assert event.payload["superseded"] == [_pid(2)]
        assert event.payload["run_id"] == report.run_id

    async def test_a_changes_reading_keeps_both(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.core.conflict_resolution import SlotVerdict

        await _seed(db_session)
        await _judged(db_session, JudgeVerdictKind.PARAPHRASE, SlotVerdict.CHANGES, (1, 2, 0.96))
        _enable(monkeypatch)
        with _finder((1, 2, 0.96)):
            report = await rm.apply_reread(db_session)
        assert report.merged_copies == 0
        assert report.blocked == {"update_changes": 1}

    async def test_high_admission_leaves_the_low_stratum(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.core.conflict_resolution import SlotVerdict

        await _seed(db_session)
        await _judged(db_session, JudgeVerdictKind.PARAPHRASE, SlotVerdict.FIXED, (1, 2, 0.95))
        _enable(monkeypatch, admitted="high")
        with _finder((1, 2, 0.95)):
            report = await rm.apply_reread(db_session)
        assert report.merged_copies == 0 and report.blocked == {"not_admitted": 1}

    async def test_unmerge_reverts_and_the_next_apply_leaves_it(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.core.conflict_resolution import SlotVerdict

        await _seed(db_session)
        await _judged(db_session, JudgeVerdictKind.PARAPHRASE, SlotVerdict.FIXED, (1, 2, 0.96))
        _enable(monkeypatch)
        with _finder((1, 2, 0.96)):
            merged = await rm.apply_reread(db_session)
        undo = await unmerge_exact_duplicates(db_session, run_id=merged.run_id, dry_run=False)
        assert undo.groups[0].restored_ids == [_pid(2)]
        assert undo.groups[0].relations_deleted == 1
        assert (await db_session.execute(select(ParticleRelationRow))).scalars().all() == []
        with _finder((1, 2, 0.96)):
            again = await rm.apply_reread(db_session)
        assert again.merged_copies == 0 and again.blocked == {"restored": 1}


class TestCli:
    def test_judge_needs_reread(self) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        result = CliRunner().invoke(app, ["links", "dedup", "--judge"])
        assert result.exit_code == 1
        assert "--reread" in result.output

    def test_judge_and_apply_are_separate(self) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        result = CliRunner().invoke(app, ["links", "dedup", "--reread", "--judge", "--apply"])
        assert result.exit_code == 1

    def test_labels_are_scored(self, tmp_path: Any) -> None:
        import csv

        from typer.testing import CliRunner

        from particles.api.cli import app

        sheet = tmp_path / "sheet.csv"
        with sheet.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["stratum", "guard", "label"])
            writer.writeheader()
            for _ in range(100):
                writer.writerow({"stratum": "high", "guard": "", "label": "same"})
        result = CliRunner().invoke(app, ["links", "dedup", "--reread", "--labels", str(sheet)])
        assert result.exit_code == 0, result.output
        assert "100/100" in result.output and "passes" in result.output

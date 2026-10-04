# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``particles curate`` verb (particles/api/cli/curate.py).

The queue itself is covered by ``tests/test_curation.py``; this file pins the
thin CLI wrapper — the ``--kind`` validation, what the flags forward to
``build_curation_queue``, and the two error paths of ``curate apply`` (an
unparseable card key, and a gesture the operation refuses). Both operations are
patched at their deferred-import location (tests/AGENTS.md § Mocking strategy).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.core.conflict_resolution import build_inconsistency_particle
from particles.core.schema import (
    Confidence,
    CurationPrecision,
    CurationPrecisionKind,
    JudgeVerdictKind,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    QualityReport,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.operations.curation import CardKind, CurationCard, DuplicateVerdict
from particles.operations.curation.cards import (
    ConflictBrief,
    ParticleBrief,
    gestures_for,
)
from particles.operations.curation.snapshot import CurationQueueResult

runner = CliRunner()


def _quality_report(**kwargs: Any) -> QualityReport:
    defaults: dict[str, Any] = {
        "active_particles": 12,
        "inconsistency_particles": 1,
        "calibration": [],
        "extractor_direct_fraction": 1.0,
        "total_entries": 3,
        "snapshots_pending": 0,
        "snapshots_in_progress": 0,
        "snapshots_complete": 3,
        "snapshots_failed": 2,
        "total_subjects": 4,
        "subjects_without_particles": 1,
    }
    defaults.update(kwargs)
    return QualityReport(**defaults)


def _card(kind: CardKind = CardKind.STALE, **kwargs: Any) -> CurationCard:
    defaults: dict[str, Any] = {
        "kind": kind,
        "particle_ids": ["p-aaa"],
        "diagnostic": "A stale belief.",
        "suggested_gestures": gestures_for(kind),
        "leverage": 0.75,
    }
    defaults.update(kwargs)
    return CurationCard(**defaults)


def _brief(pid: str, content: str) -> ParticleBrief:
    return ParticleBrief(
        particle_id=pid,
        content=content,
        subject_labels=["pre-commit"],
        effective_confidence=0.7,
        status="ACTIVE",
    )


def _particle(content: str) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.7, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e-1", snapshot_id="s-1")
        ],
        asserted_by="test",
    )


@pytest.fixture
def queue() -> Any:
    """Patch both deferred operation imports the bare verb reaches for.

    Yields the ``build_curation_queue`` mock so a test can set its return
    value and assert on the kwargs the CLI forwarded.
    """
    # the operation returns the queue plus its staleness stamp, not a
    # bare card list. Default to a live-source result — the CLI renders a
    # different header line for a stored collection.
    build = AsyncMock(return_value=CurationQueueResult(source="live"))
    with (
        patch(
            "particles.operations.quality.get_quality_report",
            new=AsyncMock(return_value=_quality_report()),
        ),
        patch("particles.operations.curation.build_curation_queue", new=build),
    ):
        yield build


# ---------------------------------------------------------------------------
# --kind validation
# ---------------------------------------------------------------------------


class TestKindOption:
    def test_unknown_kind_is_rejected_before_the_store_is_touched(
        self, cli_db: Path, queue: AsyncMock
    ) -> None:
        result = runner.invoke(app, ["curate", "--kind", "bogus"])
        assert result.exit_code == 2
        assert "Unknown kind 'bogus'" in result.output
        # The message enumerates the valid kinds so the operator can retry.
        assert "stale" in result.output
        queue.assert_not_awaited()

    def test_kind_is_case_insensitive(self, cli_db: Path, queue: AsyncMock) -> None:
        result = runner.invoke(app, ["curate", "--kind", "STALE"], catch_exceptions=False)
        assert result.exit_code == 0
        assert queue.await_args.kwargs["kind"] is CardKind.STALE

    def test_omitted_kind_means_no_filter(self, cli_db: Path, queue: AsyncMock) -> None:
        result = runner.invoke(app, ["curate"], catch_exceptions=False)
        assert result.exit_code == 0
        assert queue.await_args.kwargs["kind"] is None


# ---------------------------------------------------------------------------
# Flag forwarding + the two listing shapes
# ---------------------------------------------------------------------------


class TestQueueListing:
    def test_limit_and_semantic_are_forwarded(self, cli_db: Path, queue: AsyncMock) -> None:
        result = runner.invoke(
            app, ["curate", "--limit", "3", "--semantic"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert queue.await_args.kwargs["limit"] == 3
        assert queue.await_args.kwargs["semantic"] is True

    def test_empty_queue_reports_nothing_flagged(self, cli_db: Path, queue: AsyncMock) -> None:
        result = runner.invoke(app, ["curate"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "Curation queue empty" in result.output

    def test_cards_render_with_their_apply_key(self, cli_db: Path, queue: AsyncMock) -> None:
        queue.return_value = CurationQueueResult(
            source="live", cards=[_card()], count=1, collection_size=1
        )
        result = runner.invoke(app, ["curate"], catch_exceptions=False)
        assert result.exit_code == 0
        # The store census line, then the card with the key the apply verb takes.
        assert "12 active" in result.output
        assert "2 failed snapshots" in result.output
        assert "key: stale:p-aaa" in result.output
        assert "particles curate apply" in result.output

    def test_duplicate_card_shows_both_beliefs_the_verdict_and_what_each_gesture_does(
        self, cli_db: Path, queue: AsyncMock
    ) -> None:
        # A curator choosing between merge and dismiss has to read both claims;
        # an id pair and a similarity score are not enough to decide on.
        card = _card(
            CardKind.DUPLICATE_PAIR,
            particle_ids=["p-aaa", "p-bbb"],
            diagnostic="Possible duplicate in 'pre-commit' (similarity 0.97)",
            particles=[
                _brief("p-aaa", "pre-commit stashes unstaged tracked changes."),
                _brief("p-bbb", "pre-commit stashes unstaged changes."),
            ],
            verdict=DuplicateVerdict(
                verdict=JudgeVerdictKind.PARAPHRASE, rationale="Same behaviour."
            ),
        )
        queue.return_value = CurationQueueResult(source="live", cards=[card], count=1)
        result = runner.invoke(app, ["curate"], catch_exceptions=False)
        assert result.exit_code == 0
        out = result.output
        assert "Possible duplicate" in out
        assert "Are these two beliefs the same claim?" in out
        assert "A  “pre-commit stashes unstaged tracked changes.”" in out
        assert "B  “pre-commit stashes unstaged changes.”" in out
        assert "subjects: pre-commit" in out
        assert "LLM judge: PARAPHRASE. Same behaviour." in out
        # The pair key joins its ids with `|`, a shell pipe: it prints quoted so
        # a copy-paste into `curate apply` reaches the verb as one argument.
        assert "key: 'duplicate_pair:p-aaa|p-bbb'" in out
        assert "Link the two as co-evidential" in out
        assert "They are different claims" in out

    def test_conflict_card_shows_both_sides_and_its_actions(
        self, cli_db: Path, queue: AsyncMock
    ) -> None:
        # the card is the record; its sides print in the A/B order
        # `resolve --action PREFER_A|PREFER_B` refers to, and the resolve line
        # names only the actions this card offers.
        a = _particle("The dashboard lives in ~/observability.")
        b = _particle("The dashboard lives in docs/operator-guide.")
        inc = build_inconsistency_particle(a, b, corpus_entry_id="e-1", snapshot_id="s-1")
        brief_a, brief_b = _brief(a.id, a.content), _brief(b.id, b.content)
        card = _card(
            CardKind.INCONSISTENCY,
            particle_ids=[a.id, b.id],
            diagnostic="Open conflict",
            suggested_gestures=gestures_for(CardKind.INCONSISTENCY),
            inconsistency_id=inc.id,
            particles=[brief_a, brief_b],
            conflict=ConflictBrief(inconsistency_id=inc.id, a=brief_a, b=brief_b),
            resolve_actions=["PREFER_B", "BOTH_VALID", "DISCARD"],
        )
        queue.return_value = CurationQueueResult(source="live", cards=[card], count=1)
        result = runner.invoke(app, ["curate"], catch_exceptions=False)
        assert result.exit_code == 0
        out = result.output
        assert "Open conflict" in out
        assert "A  “The dashboard lives in ~/observability.”" in out
        assert "B  “The dashboard lives in docs/operator-guide.”" in out
        # Each side prints once, as a side, not again as a brief.
        assert out.count("docs/operator-guide.”") == 1
        flat = " ".join(out.split())
        assert "--action PREFER_B|BOTH_VALID|DISCARD" in flat
        assert f"key: inconsistency:{inc.id}" in out


class TestHelp:
    def test_help_uses_current_terms_and_explains_the_gestures(self) -> None:
        result = runner.invoke(app, ["curate", "--help"])
        assert result.exit_code == 0
        assert "bus-stop" not in result.output.lower()
        for gesture in ("affirm", "snooze", "dismiss", "retract", "merge", "supersede"):
            assert gesture in result.output
        # Every card kind is listed, so --kind needs no guessing.
        assert "no_subject" in result.output


# ---------------------------------------------------------------------------
# curate apply — the two error paths
# ---------------------------------------------------------------------------


class TestApplyGesture:
    def test_unparseable_card_key_exits_one(self, cli_db: Path) -> None:
        with patch("particles.operations.curation.apply_gesture") as gesture:
            result = runner.invoke(app, ["curate", "apply", "affirm", "not-a-real-key"])
        assert result.exit_code == 1
        assert "✗" in result.output
        assert "not-a-real-key" in result.output
        gesture.assert_not_called()

    def test_operation_refusal_exits_one(self, cli_db: Path) -> None:
        refuse = AsyncMock(side_effect=ValueError("Gesture 'merge': not offered."))
        with patch("particles.operations.curation.apply_gesture", new=refuse):
            result = runner.invoke(app, ["curate", "apply", "merge", "stale:p-aaa"])
        assert result.exit_code == 1
        assert "✗ Gesture 'merge': not offered." in result.output

    def test_success_prints_the_operation_message(self, cli_db: Path) -> None:
        ok = AsyncMock(return_value="Affirmed — stale:p-aaa will not resurface.")
        with patch("particles.operations.curation.apply_gesture", new=ok):
            result = runner.invoke(
                app,
                ["curate", "apply", "affirm", "stale:p-aaa", "--reason", "checked"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert "✓ Affirmed — stale:p-aaa will not resurface." in result.output
        card = ok.await_args.args[1]
        assert card.kind is CardKind.STALE
        assert card.particle_ids == ["p-aaa"]
        assert ok.await_args.kwargs["reason"] == "checked"

    def test_resolve_passes_its_action_and_note(self, cli_db: Path) -> None:
        ok = AsyncMock(return_value="Resolved inconsistency:rec-1 as BOTH_VALID.")
        with patch("particles.operations.curation.apply_gesture", new=ok):
            result = runner.invoke(
                app,
                [
                    "curate",
                    "apply",
                    "resolve",
                    "inconsistency:rec-1",
                    "--action",
                    "BOTH_VALID",
                    "--note",
                    "different runs",
                ],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        card = ok.await_args.args[1]
        assert card.kind is CardKind.INCONSISTENCY
        assert card.inconsistency_id == "rec-1"
        assert ok.await_args.args[2] == "resolve"
        assert ok.await_args.kwargs["action"] == "BOTH_VALID"
        assert ok.await_args.kwargs["note"] == "different runs"

    def test_action_on_another_gesture_is_refused(self, cli_db: Path) -> None:
        with patch("particles.operations.curation.apply_gesture") as gesture:
            result = runner.invoke(
                app, ["curate", "apply", "snooze", "inconsistency:rec-1", "--action", "PREFER_A"]
            )
        assert result.exit_code == 1
        assert "--action applies only to the resolve gesture" in result.output
        gesture.assert_not_called()

    def test_listing_quotes_pasted_inside_double_quotes_are_stripped(self, cli_db: Path) -> None:
        ok = AsyncMock(return_value="Linked.")
        with patch("particles.operations.curation.apply_gesture", new=ok):
            result = runner.invoke(
                app,
                ["curate", "apply", "merge", "'duplicate_pair:p-aaa|p-bbb'"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        card = ok.await_args.args[1]
        assert card.kind is CardKind.DUPLICATE_PAIR
        assert card.particle_ids == ["p-aaa", "p-bbb"]

    def test_snooze_days_and_subject_are_forwarded(self, cli_db: Path) -> None:
        ok = AsyncMock(return_value="Snoozed.")
        with patch("particles.operations.curation.apply_gesture", new=ok):
            result = runner.invoke(
                app,
                [
                    "curate",
                    "apply",
                    "assign-subject",
                    "no_subject:p-bbb",
                    "--days",
                    "14",
                    "--subject",
                    "Deploys",
                ],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert ok.await_args.kwargs["days"] == 14
        assert ok.await_args.kwargs["subject"] == "Deploys"


# ---------------------------------------------------------------------------
# curate apply supersede — the operator's revision as flags
# ---------------------------------------------------------------------------


class TestApplySupersede:
    _BASE = ["curate", "apply", "supersede", "stale:p-aaa"]

    def test_flags_become_the_revision(self, cli_db: Path) -> None:
        ok = AsyncMock(return_value="Superseded p-aaa… → successor p-bbb… (ASSERTED).")
        with patch("particles.operations.curation.apply_gesture", new=ok):
            result = runner.invoke(
                app,
                [
                    *self._BASE,
                    "--content",
                    "The window is 30 days.",
                    "--reason",
                    "Changed in 1.140",
                    "--confidence",
                    "0.85",
                    "--subject",
                    "Curation",
                    "--subject",
                    "Snooze window",
                    "--source",
                    "The 1.140 release notes",
                ],
                catch_exceptions=False,
            )
        assert result.exit_code == 0, result.output
        assert "✓ Superseded p-aaa" in result.output
        kwargs = ok.await_args.kwargs
        assert kwargs["reason"] == "Changed in 1.140"
        assert kwargs["subject"] is None
        revision = kwargs["revision"]
        assert revision.content == "The window is 30 days."
        assert revision.confidence == 0.85
        assert revision.subjects == ["Curation", "Snooze window"]
        assert revision.source_excerpt == "The 1.140 release notes"
        assert revision.corpus_entry_id is None

    def test_subjects_and_source_are_optional(self, cli_db: Path) -> None:
        ok = AsyncMock(return_value="Superseded.")
        with patch("particles.operations.curation.apply_gesture", new=ok):
            result = runner.invoke(
                app,
                [*self._BASE, "--content", "c", "--reason", "r", "--confidence", "0.5"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0, result.output
        revision = ok.await_args.kwargs["revision"]
        assert revision.subjects == []  # inherit (§2)
        assert revision.source_excerpt is None  # the reason, applied downstream (§4)

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            (["--reason", "r", "--confidence", "0.5"], "supersede requires --content"),
            (["--content", "c", "--confidence", "0.5"], "supersede requires --reason"),
            (["--content", "c", "--reason", "r"], "supersede requires --confidence"),
            (["--content", " ", "--reason", "r"], "--content, --confidence"),
            (
                [
                    *("--content", "c", "--reason", "r", "--confidence", "0.5"),
                    *("--source", "s", "--corpus-entry", "e"),
                ],
                "not both",
            ),
        ],
    )
    def test_missing_or_conflicting_flags_exit_one_before_any_write(
        self, cli_db: Path, args: list[str], message: str
    ) -> None:
        with patch("particles.operations.curation.apply_gesture") as gesture:
            result = runner.invoke(app, [*self._BASE, *args])
        assert result.exit_code == 1
        assert message in result.output
        gesture.assert_not_called()

    def test_confidence_outside_zero_to_one_is_a_usage_error(self, cli_db: Path) -> None:
        with patch("particles.operations.curation.apply_gesture") as gesture:
            result = runner.invoke(
                app, [*self._BASE, "--content", "c", "--reason", "r", "--confidence", "1.5"]
            )
        assert result.exit_code == 2
        gesture.assert_not_called()

    def test_revision_flags_on_another_gesture_exit_one(self, cli_db: Path) -> None:
        with patch("particles.operations.curation.apply_gesture") as gesture:
            result = runner.invoke(
                app,
                ["curate", "apply", "retract", "stale:p-aaa", "--reason", "r", "--content", "c"],
            )
        assert result.exit_code == 1
        assert "--content applies only to the supersede gesture" in result.output
        gesture.assert_not_called()

    def test_assign_subject_takes_one_subject(self, cli_db: Path) -> None:
        with patch("particles.operations.curation.apply_gesture") as gesture:
            result = runner.invoke(
                app,
                [
                    *("curate", "apply", "assign-subject", "no_subject:p-bbb"),
                    *("--subject", "A", "--subject", "B"),
                ],
            )
        assert result.exit_code == 1
        assert "takes one --subject" in result.output
        gesture.assert_not_called()

    def test_help_lists_supersede_as_applied_with_an_example(self) -> None:
        # Rich colours help under CI, splitting option names with escape codes.
        def plain(args: list[str]) -> str:
            return re.sub(r"\x1b\[[0-9;]*m", "", runner.invoke(app, args).output)

        listing = plain(["curate", "--help"])
        applied, _, surfaced = listing.partition("name another command")
        assert "supersede" in applied and "supersede" not in surfaced
        apply_help = plain(["curate", "apply", "--help"])
        assert "curate apply supersede KEY" in apply_help
        assert "--confidence" in apply_help


def _plain(output: str) -> str:
    """Terminal output with rich's styling, box borders and wrapping removed.

    CI renders Typer errors and help through rich at 80 columns: option names
    are styled in pieces ("--" and "precision" separately) and a message longer
    than the box is wrapped across bordered lines, so a plain substring match
    on a long message fails there and passes on a wide local terminal.
    """
    text = re.sub(r"\x1b\[[0-9;]*m", "", output)
    text = re.sub(r"[│╭╮╰╯─]", " ", text)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# --precision
# ---------------------------------------------------------------------------


def _precision_report() -> CurationPrecision:
    from datetime import UTC, datetime

    return CurationPrecision(
        since=datetime(2026, 9, 1, tzinfo=UTC),
        until=datetime(2026, 10, 1, tzinfo=UTC),
        kinds=[
            CurationPrecisionKind(
                kind="duplicate_pair",
                offered=4,
                acted=2,
                dismissed=1,
                snoozed=0,
                open=1,
                expired=0,
                precision=2 / 3,
                acted_means="the pair was merged as one claim",
                dismissed_means="the two are different claims, so the finder was wrong",
            ),
            CurationPrecisionKind(
                kind="stale",
                offered=3,
                acted=1,
                dismissed=0,
                snoozed=1,
                open=1,
                expired=0,
                precision=1.0,
                acted_means="the belief was affirmed as still true, replaced or retracted",
                dismissed_means="the expiry was not worth a ruling",
            ),
        ],
        offered=7,
        acted=3,
        dismissed=1,
        snoozed=1,
        open=2,
        expired=0,
        precision=0.75,
        events_read=5,
        unattributed_events=1,
        snapshots_read=1,
        snapshot_built_at=datetime(2026, 10, 1, tzinfo=UTC),
    )


@pytest.fixture
def precision() -> Any:
    """Patch the deferred ``curation_precision`` import the flag reaches for."""
    gather = AsyncMock(return_value=_precision_report())
    with patch("particles.operations.curation.curation_precision", new=gather):
        yield gather


class TestPrecisionFlag:
    def test_table_lists_every_kind_the_denominator_and_the_meanings(
        self, cli_db: Path, precision: AsyncMock
    ) -> None:
        result = runner.invoke(app, ["curate", "--precision"], catch_exceptions=False)
        assert result.exit_code == 0
        out = result.output
        assert "Curation queue precision: 2026-09-01 to 2026-10-01 (30 days)" in out
        assert "Of 4 cards the operator ruled on, 3 were real problems and 1 were not" in out
        assert "precision 0.75" in out
        assert "7 cards offered = 4 decided + 1 snoozed + 2 open untouched" in out
        assert "duplicate_pair" in out and "stale" in out
        assert "the finder was wrong" in out
        assert "1 resolving event(s) in the window named no card" in out
        assert precision.await_args.kwargs["since"] is None

    def test_since_is_forwarded_as_a_utc_instant(self, cli_db: Path, precision: AsyncMock) -> None:
        from datetime import UTC, datetime

        result = runner.invoke(
            app, ["curate", "--precision", "--since", "2026-09-10"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert precision.await_args.kwargs["since"] == datetime(2026, 9, 10, tzinfo=UTC)

    def test_json_emits_the_report(self, cli_db: Path, precision: AsyncMock) -> None:
        import json

        result = runner.invoke(
            app, ["curate", "--precision", "--format", "json"], catch_exceptions=False
        )
        assert result.exit_code == 0
        body = json.loads(result.output)
        assert body["precision"] == 0.75
        assert [k["kind"] for k in body["kinds"]] == ["duplicate_pair", "stale"]
        assert body["kinds"][0]["dismissed_means"].endswith("the finder was wrong")

    def test_kind_narrows_the_table_and_the_totals(
        self, cli_db: Path, precision: AsyncMock
    ) -> None:
        import json

        result = runner.invoke(
            app,
            ["curate", "--precision", "--kind", "stale", "--format", "json"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        body = json.loads(result.output)
        assert [k["kind"] for k in body["kinds"]] == ["stale"]
        assert (body["offered"], body["acted"], body["precision"]) == (3, 1, 1.0)

    def test_nothing_to_measure_says_so(self, cli_db: Path, precision: AsyncMock) -> None:
        precision.return_value = None
        result = runner.invoke(app, ["curate", "--precision"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "No curation precision to report" in _plain(result.output)
        result = runner.invoke(
            app, ["curate", "--precision", "--format", "json"], catch_exceptions=False
        )
        assert _plain(result.output) == "null"

    def test_bad_since_and_future_since_are_usage_errors(
        self, cli_db: Path, precision: AsyncMock
    ) -> None:
        result = runner.invoke(app, ["curate", "--precision", "--since", "last tuesday"])
        assert result.exit_code == 2
        assert "YYYY-MM-DD" in _plain(result.output)
        result = runner.invoke(app, ["curate", "--precision", "--since", "2999-01-01"])
        assert result.exit_code == 2
        assert "in the future" in _plain(result.output)
        precision.assert_not_awaited()

    def test_since_and_format_need_precision(
        self, cli_db: Path, queue: AsyncMock, precision: AsyncMock
    ) -> None:
        result = runner.invoke(app, ["curate", "--since", "2026-09-01"])
        assert result.exit_code == 2
        assert "--since applies only with --precision" in _plain(result.output)
        result = runner.invoke(app, ["curate", "--format", "json"])
        assert result.exit_code == 2
        assert "--format applies only with --precision" in _plain(result.output)
        result = runner.invoke(app, ["curate", "--format", "yaml"])
        assert result.exit_code == 2
        queue.assert_not_awaited()
        precision.assert_not_awaited()

    def test_queue_flags_are_refused_with_precision(
        self, cli_db: Path, queue: AsyncMock, precision: AsyncMock
    ) -> None:
        for flags in (["--limit", "3"], ["--semantic"], ["--refresh"], ["--no-snapshot"]):
            result = runner.invoke(app, ["curate", "--precision", *flags])
            assert result.exit_code == 2, flags
            assert "does not apply to --precision" in _plain(result.output)
        queue.assert_not_awaited()
        precision.assert_not_awaited()

    def test_help_explains_the_flag(self) -> None:
        result = runner.invoke(app, ["curate", "--help"])
        assert result.exit_code == 0
        assert "--precision" in _plain(result.output)
        assert "--since" in _plain(result.output)


# The freshness line names what a refresh rebuilds and carries
# ---------------------------------------------------------------------------


class TestFreshnessLine:
    @staticmethod
    def _stored(**kwargs: Any) -> CurationQueueResult:
        from datetime import UTC, datetime

        defaults: dict[str, Any] = {
            "source": "snapshot",
            "snapshot_id": "snap-1",
            "built_at": datetime(2026, 9, 24, 3, 30, tzinfo=UTC),
            "age_seconds": 9 * 86_400.0,
            "stale": True,
            "collection_size": 40,
            "per_kind_scope": {"contradiction": "carried", "stale": "store"},
        }
        defaults.update(kwargs)
        return CurationQueueResult(**defaults)

    def test_stale_line_says_contradictions_carry_from_the_census(self) -> None:
        from datetime import UTC, datetime

        from particles.api.cli.curate import _freshness_line

        line = _freshness_line(
            self._stored(kind_as_of={"contradiction": datetime(2026, 9, 17, 3, 30, tzinfo=UTC)})
        )
        assert (
            "STALE: run `particles curate --refresh` (rebuilds structural cards; "
            "contradiction cards carry forward from the census of 2026-09-17)"
        ) in line
        assert "carried forward: contradiction" in line

    def test_stale_line_without_a_census_on_record_says_so(self) -> None:
        from particles.api.cli.curate import _freshness_line

        line = _freshness_line(self._stored(kind_as_of={}))
        assert "(rebuilds structural cards; no census has probed for contradictions yet)" in line

    def test_fresh_collection_has_no_stale_hint(self) -> None:
        from particles.api.cli.curate import _freshness_line

        line = _freshness_line(self._stored(stale=False, age_seconds=3600.0))
        assert "STALE" not in line

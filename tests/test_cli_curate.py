"""Tests for the ``particles curate`` verb (particles/api/cli/curate.py).

The queue itself is covered by ``tests/test_curation.py``; this file pins the
thin CLI wrapper — the ``--kind`` validation, what the flags forward to
``build_curation_queue``, and the two error paths of ``curate apply`` (an
unparseable card key, and a gesture the operation refuses). Both operations are
patched at their deferred-import location (tests/AGENTS.md § Mocking strategy).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.core.schema import QualityReport
from particles.operations.curation import CardKind, CurationCard
from particles.operations.curation.cards import gestures_for
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

"""Tests for the ``particles structure`` verb (particles/api/cli/structure.py).

The backfill pass itself is covered by ``tests/test_structured_claim.py``; this
file pins the CLI wrapper — the local-store-only refusal against a remote
engine (exit 2, before the operation is imported), the flags the verb forwards,
and the JSON summary envelope. Both the backend and the operation are patched
at their deferred-import locations (tests/AGENTS.md § Mocking strategy).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from particles.api.cli import app

runner = CliRunner()

_SUMMARY: dict[str, Any] = {
    "scope": 4,
    "annotated": 3,
    "skipped": 1,
    "failed": 0,
    "remaining": 12,
    "stamp": "general@1.0.0",
}


@pytest.fixture
def backfill() -> Any:
    """Patch the operation and pin the backend local; yields the operation mock."""
    op = AsyncMock(return_value=dict(_SUMMARY))
    local = MagicMock()
    local.remote = False
    with (
        patch("particles.api.client.get_backend", return_value=local),
        patch("particles.operations.structure.backfill_structured_claims", new=op),
    ):
        yield op


# ---------------------------------------------------------------------------
# The remote-engine refusal
# ---------------------------------------------------------------------------


def test_remote_engine_is_refused_with_exit_two(cli_db: Path) -> None:
    remote = MagicMock()
    remote.remote = True
    op = AsyncMock()
    with (
        patch("particles.api.client.get_backend", return_value=remote),
        patch("particles.operations.structure.backfill_structured_claims", new=op),
    ):
        result = runner.invoke(app, ["structure"])
    assert result.exit_code == 2
    assert "annotates one local store per invocation" in result.output
    op.assert_not_awaited()


# ---------------------------------------------------------------------------
# Flag forwarding + output
# ---------------------------------------------------------------------------


class TestFlags:
    def test_defaults_forward_none_and_no_progress(self, cli_db: Path, backfill: AsyncMock) -> None:
        result = runner.invoke(app, ["structure"], catch_exceptions=False)
        assert result.exit_code == 0
        kwargs = backfill.await_args.kwargs
        assert kwargs["limit"] is None
        assert kwargs["rate_limit_per_minute"] is None
        assert kwargs["structurizer_version"] is None
        assert kwargs["dry_run"] is False
        # Progress narration is opt-in — a quiet run passes no callback.
        assert kwargs["progress"] is None

    def test_every_flag_reaches_the_operation(self, cli_db: Path, backfill: AsyncMock) -> None:
        result = runner.invoke(
            app,
            [
                "structure",
                "--limit",
                "0",
                "--rate-limit-per-minute",
                "0",
                "--structurizer-version",
                "general@2.0.0",
                "--dry-run",
                "--verbose",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        kwargs = backfill.await_args.kwargs
        assert kwargs["limit"] == 0
        assert kwargs["rate_limit_per_minute"] == 0
        assert kwargs["structurizer_version"] == "general@2.0.0"
        assert kwargs["dry_run"] is True
        assert callable(kwargs["progress"])

    def test_summary_is_printed_as_indented_json(self, cli_db: Path, backfill: AsyncMock) -> None:
        result = runner.invoke(app, ["structure"], catch_exceptions=False)
        assert result.exit_code == 0
        assert json.loads(result.output) == _SUMMARY

    def test_dry_run_summary_passes_through_verbatim(
        self, cli_db: Path, backfill: AsyncMock
    ) -> None:
        backfill.return_value = {"backlog": 120, "batch_limit": 50, "runs_needed": 3}
        result = runner.invoke(app, ["structure", "--dry-run"], catch_exceptions=False)
        assert result.exit_code == 0
        assert json.loads(result.output)["runs_needed"] == 3

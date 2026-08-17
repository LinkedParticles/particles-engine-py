"""Tests for the ``particles lint`` verb (particles/api/cli/lint.py).

The lint operation itself is covered by ``tests/test_lint.py``; this file pins
the CLI wrapper — the ``--category`` validation and filter, the per-category
verbose cap and its synthetic breadcrumb finding, the two JSON envelopes, and
the ``--fix`` summary line. The backend is patched at the module binding
(``lint.py`` imports ``get_backend`` at module top), so no store is needed.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.core.schema import LintFinding, LintReport

runner = CliRunner()


def _finding(finding_type: str, severity: str = "WARNING", detail: str = "d") -> LintFinding:
    return LintFinding(finding_type=finding_type, severity=severity, detail=detail)


def _report(
    findings: list[LintFinding] | None = None,
    summary: dict[str, int] | None = None,
    fixed_counts: dict[str, int] | None = None,
) -> LintReport:
    findings = findings or []
    if summary is None:
        summary = {}
        for f in findings:
            summary[f.finding_type] = summary.get(f.finding_type, 0) + 1
    return LintReport(findings=findings, summary=summary, fixed_counts=fixed_counts or {})


@pytest.fixture
def backend() -> MagicMock:
    """A patched backend whose ``lint`` returns whatever a test assigns."""
    be = MagicMock()
    be.remote = False
    be.lint = AsyncMock(return_value=_report())
    with patch("particles.api.cli.lint.get_backend", return_value=be):
        yield be


# ---------------------------------------------------------------------------
# --category: validation + filtering
# ---------------------------------------------------------------------------


class TestCategoryOption:
    def test_unknown_category_exits_one_and_lists_what_ran(self, backend: MagicMock) -> None:
        backend.lint.return_value = _report([_finding("STALENESS"), _finding("CONTESTED")])
        result = runner.invoke(app, ["lint", "--category", "NOPE"], catch_exceptions=False)
        assert result.exit_code == 1
        assert "No findings with finding_type='NOPE'" in result.output
        # The available categories are listed so the operator can retry.
        assert "CONTESTED, STALENESS" in result.output

    def test_unknown_category_on_a_clean_run_says_none(self, backend: MagicMock) -> None:
        result = runner.invoke(app, ["lint", "--category", "STALENESS"], catch_exceptions=False)
        assert result.exit_code == 1
        assert "(none)" in result.output

    def test_known_category_filters_findings_but_keeps_global_summary(
        self, backend: MagicMock
    ) -> None:
        backend.lint.return_value = _report(
            [_finding("STALENESS", detail="stale one"), _finding("CONTESTED", detail="disputed")]
        )
        result = runner.invoke(
            app,
            ["lint", "--category", "STALENESS", "--output-format", "json", "--verbose"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert [f["finding_type"] for f in payload["findings"]] == ["STALENESS"]
        # summary keeps the full counts — the operator still sees the global shape.
        assert payload["summary"] == {"STALENESS": 1, "CONTESTED": 1}


# ---------------------------------------------------------------------------
# --limit-per-category: the verbose render cap
# ---------------------------------------------------------------------------


class TestVerboseCap:
    def test_cap_truncates_and_appends_a_breadcrumb_finding(self, backend: MagicMock) -> None:
        backend.lint.return_value = _report(
            [_finding("STALENESS", detail=f"s{i}") for i in range(5)]
        )
        result = runner.invoke(
            app,
            ["lint", "--verbose", "--limit-per-category", "2"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "…and 3 more STALENESS finding(s) suppressed" in result.output
        assert "--category STALENESS --limit-per-category 0" in result.output

    def test_zero_disables_the_cap(self, backend: MagicMock) -> None:
        backend.lint.return_value = _report(
            [_finding("STALENESS", detail=f"s{i}") for i in range(5)]
        )
        result = runner.invoke(
            app,
            ["lint", "--verbose", "--limit-per-category", "0"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "suppressed by" not in result.output

    def test_no_breadcrumb_when_under_the_cap(self, backend: MagicMock) -> None:
        backend.lint.return_value = _report([_finding("STALENESS")])
        result = runner.invoke(app, ["lint", "--verbose"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "suppressed by" not in result.output


# ---------------------------------------------------------------------------
# Output envelopes
# ---------------------------------------------------------------------------


class TestOutputFormat:
    def test_json_summary_is_the_default_json_envelope(self, backend: MagicMock) -> None:
        backend.lint.return_value = _report(
            [_finding("STALENESS", severity="ERROR"), _finding("CONTESTED", severity="WARNING")]
        )
        result = runner.invoke(app, ["lint", "--output-format", "json"], catch_exceptions=False)
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert set(payload) == {
            "run_at",
            "summary",
            "fixed_counts",
            "error_count",
            "warning_count",
        }
        assert payload["error_count"] == 1
        assert payload["warning_count"] == 1
        # The slim envelope carries counts, never the findings themselves.
        assert "findings" not in payload

    def test_json_verbose_is_the_full_report_dump(self, backend: MagicMock) -> None:
        backend.lint.return_value = _report([_finding("STALENESS")])
        result = runner.invoke(
            app, ["lint", "--output-format", "json", "--verbose"], catch_exceptions=False
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["findings"][0]["finding_type"] == "STALENESS"

    def test_text_summary_counts_by_severity(self, backend: MagicMock) -> None:
        backend.lint.return_value = _report(
            [
                _finding("STALENESS", severity="ERROR"),
                _finding("STALENESS", severity="ERROR"),
                _finding("CONTESTED", severity="WARNING"),
                _finding("NOTE", severity="INFO"),
            ]
        )
        result = runner.invoke(app, ["lint"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "2 errors, 1 warning, 1 info" in result.output
        assert "STALENESS: 2" in result.output


# ---------------------------------------------------------------------------
# --fix: the auto-fixed line
# ---------------------------------------------------------------------------


class TestFixLine:
    def test_no_fix_flag_prints_no_fix_line(self, backend: MagicMock) -> None:
        result = runner.invoke(app, ["lint"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "Auto-fixed" not in result.output

    def test_fix_with_nothing_fixed_lists_the_categories_considered(
        self, backend: MagicMock
    ) -> None:
        result = runner.invoke(app, ["lint", "--fix"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "Auto-fixed: 0" in result.output
        assert "STALENESS" in result.output
        assert backend.lint.await_args.kwargs["fix"] is True

    def test_fix_reports_per_category_counts(self, backend: MagicMock) -> None:
        backend.lint.return_value = _report(
            [_finding("STALENESS")], fixed_counts={"STALENESS": 2, "RETRACTION_CASCADE": 0}
        )
        result = runner.invoke(app, ["lint", "--fix"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "Auto-fixed: 2" in result.output
        assert "STALENESS: 2" in result.output
        # Zero-count categories are omitted from the per-category breakdown.
        assert "RETRACTION_CASCADE: 0" not in result.output

    def test_clean_report_with_fix_still_prints_the_fix_line(self, backend: MagicMock) -> None:
        result = runner.invoke(app, ["lint", "--fix"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "Lint clean" in result.output
        assert "Auto-fixed: 0" in result.output


def test_semantic_and_threshold_flags_reach_the_backend(backend: MagicMock) -> None:
    result = runner.invoke(
        app, ["lint", "--semantic", "--low-coverage-threshold", "7"], catch_exceptions=False
    )
    assert result.exit_code == 0
    assert backend.lint.await_args.kwargs["semantic"] is True
    assert backend.lint.await_args.kwargs["low_coverage_threshold"] == 7

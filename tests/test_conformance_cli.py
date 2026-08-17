"""CLI tests for `particles conformance`."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from particles.api.cli import app

runner = CliRunner()


def test_conformance_show_prints_version_and_constants() -> None:
    result = runner.invoke(app, ["conformance", "show"])
    assert result.exit_code == 0, result.output
    assert "profile_version: 1.2" in result.output
    assert "all-MiniLM-L6-v2" in result.output
    assert "trust_differential_threshold" in result.output


def test_conformance_check_l2_passes() -> None:
    # L2 is pure (no embedding model needed) and must self-certify clean.
    result = runner.invoke(app, ["conformance", "check", "--level", "L2"])
    assert result.exit_code == 0, result.output
    assert "L2: PASS" in result.output


def test_conformance_check_json_emits_structured_report() -> None:
    result = runner.invoke(app, ["conformance", "check", "--level", "L2", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload[0]["level"] == "L2"
    assert payload[0]["status"] == "PASS"
    assert payload[0]["checks"]

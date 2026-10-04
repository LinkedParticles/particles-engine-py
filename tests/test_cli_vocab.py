# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""``particles vocab``: argument validation, error paths, and the
create / adopt / export / import round trip through a file-backed store.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from particles.api.cli import app

NS = "https://vocab.example.org/acme/"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _create(runner: CliRunner) -> None:
    result = runner.invoke(
        app,
        ["vocab", "create", "acme", "--prefix", "acme", "--namespace", NS, "--publisher", "Acme"],
    )
    assert result.exit_code == 0, result.output


class TestErrorPaths:
    def test_create_refuses_a_bad_namespace(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(
            app, ["vocab", "create", "acme", "--prefix", "acme", "--namespace", "nope"]
        )
        assert result.exit_code == 1
        assert "namespace" in result.output

    def test_create_refuses_a_second_document_of_one_name(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        _create(runner)
        result = runner.invoke(
            app, ["vocab", "create", "acme", "--prefix", "acme", "--namespace", NS]
        )
        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_show_an_unknown_document(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["vocab", "show", "nope"])
        assert result.exit_code == 1

    def test_propose_rejects_an_unknown_kind(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["vocab", "propose", "acme", "--kind", "both"])
        assert result.exit_code == 2

    def test_propose_for_an_unknown_document(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["vocab", "propose", "nope", "--dry-run"])
        assert result.exit_code == 1
        assert "No vocabulary" in result.output

    def test_confirm_rejects_an_unknown_slot_kind(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["vocab", "confirm", "vp-1", "--kind", "sometimes"])
        assert result.exit_code == 2

    def test_confirm_an_unknown_key(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["vocab", "confirm", "vp-000000000000"])
        assert result.exit_code == 1
        assert "No vocabulary proposal" in result.output

    def test_decline_an_unknown_key(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["vocab", "decline", "vp-000000000000"])
        assert result.exit_code == 1

    def test_align_refuses_an_estimated_equivalence(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(
            app,
            [
                "vocab",
                "align",
                "acme",
                "t",
                "wdt:P551",
                "--match",
                "equivalent",
                "--confidence",
                "0.8",
                "--basis",
                "x",
            ],
        )
        assert result.exit_code == 2
        assert "Invalid alignment" in result.output

    def test_align_refuses_a_malformed_constraint(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(
            app,
            [
                "vocab",
                "align",
                "acme",
                "t",
                "wdt:P551",
                "--match",
                "exact",
                "--basis",
                "x",
                "--wikidata-constraint",
                "single-value",
            ],
        )
        assert result.exit_code == 2

    def test_adopt_an_unknown_document(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["vocab", "adopt", "nope"])
        assert result.exit_code == 1

    def test_import_refuses_something_else(
        self, runner: CliRunner, cli_db: Path, tmp_path: Path
    ) -> None:
        path = tmp_path / "x.jsonld"
        path.write_text('{"@type": "skos:ConceptScheme"}')
        result = runner.invoke(app, ["vocab", "import", str(path)])
        assert result.exit_code == 1


class TestRoundTrip:
    def test_create_adopt_export_import(
        self, runner: CliRunner, cli_db: Path, tmp_path: Path
    ) -> None:
        _create(runner)
        listed = runner.invoke(app, ["vocab", "list"])
        assert "acme" in listed.output and NS in listed.output
        adopted = runner.invoke(app, ["vocab", "adopt", "acme", "--lens", "hr"])
        assert adopted.exit_code == 0 and "while lens 'hr'" in adopted.output
        assert "lens:hr" in runner.invoke(app, ["vocab", "list"]).output
        out = tmp_path / "acme.jsonld"
        exported = runner.invoke(app, ["vocab", "export", "acme", "-o", str(out)])
        assert exported.exit_code == 0
        assert json.loads(out.read_text())["ppx:name"] == "acme"
        reimported = runner.invoke(app, ["vocab", "import", str(out)])
        assert reimported.exit_code == 0 and "v1" in reimported.output
        shown = runner.invoke(app, ["vocab", "show", "acme"])
        assert shown.exit_code == 0 and "no terms yet" in shown.output
        dry = runner.invoke(app, ["vocab", "propose", "acme", "--dry-run"])
        assert dry.exit_code == 0 and "0 structured claims" in dry.output
        assert runner.invoke(app, ["vocab", "proposals"]).output.startswith("No proposals")
        gone = runner.invoke(app, ["vocab", "unadopt", "acme", "--lens", "hr"])
        assert gone.exit_code == 0

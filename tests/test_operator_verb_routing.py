"""operator-verb route-or-refuse behaviour at the CLI level.

The §2(b) operator verbs have no engine endpoint, so with an engine configured
they must **refuse** (non-zero exit, actionable message) rather than silently
read or write the LOCAL store while the daily loop runs on the engine. With no
engine configured the same verbs run in-process exactly as before.

The §2(a) *routing* of the endpoint-backed verbs is exercised end-to-end at the
backend level in ``tests/test_api_client.py`` and behaviour-preservingly by the
per-verb CLI tests (``test_subjects.py``, ``test_corpus_retract.py``, …).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from particles.api.cli import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def remote_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure a remote engine so the backend factory resolves to HttpBackend."""
    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "http://engine.test")
    monkeypatch.delenv("PARTICLES_ENGINE_TOKEN", raising=False)
    from particles.config import reset_config

    reset_config()


# Every §2(b) verb invocation, with the args needed to reach its body. Each must
# refuse before touching the store / network when an engine is configured.
_FAIL_LOUD_INVOCATIONS = [
    ["export", "obsidian", "/tmp/particles-export-test"],
    ["corpus", "delete", "abc12345", "--yes"],
    ["corpus", "prune-orphans", "--yes"],
    ["corpus", "list"],
    ["corpus", "links", "list"],
    ["trust", "list"],
    ["trust", "show", "https://example.com"],
    ["trust", "cascade"],
    ["trust", "set-entry", "abc12345", "0.5"],
    ["trust", "lens", "list"],
    ["trust", "lens", "show", "somelens"],
    ["trust", "lens", "adopt", "somelens"],
    ["links", "list", "abc12345"],
    ["particle", "search", "--fingerprint", "abcdef12"],
    ["particle", "narrative", "abc12345"],
    ["subjects", "delete", "abc12345"],
    ["subjects", "set-class", "abc12345", "nmo:X"],
    ["subjects", "gc"],
    ["subjects", "prune-empty"],
    ["subjects", "confirm", "abc12345", "wikidata:Q1"],
    ["subjects", "unlink", "abc12345", "wikidata:Q1"],
    ["subjects", "fix-labels"],
    ["subjects", "find-duplicates"],
    ["subjects", "list", "--phantoms-only"],
]


class TestFailLoudInRemoteMode:
    @pytest.mark.parametrize(
        "argv", _FAIL_LOUD_INVOCATIONS, ids=[" ".join(a) for a in _FAIL_LOUD_INVOCATIONS]
    )
    def test_refuses_with_actionable_error(
        self, runner: CliRunner, remote_engine: None, argv: list[str]
    ) -> None:
        result = runner.invoke(app, argv)
        assert result.exit_code != 0, result.output
        assert "not available against a remote engine" in result.output
        assert "read or write your LOCAL" in result.output
        assert "engine.base_url" in result.output


class TestRunsLocallyWithoutEngine:
    """With no engine configured the same verbs run in-process (no refusal)."""

    def test_corpus_prune_orphans_runs_locally(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["corpus", "prune-orphans", "--yes"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "Nothing to prune." in result.output

    def test_corpus_list_runs_locally(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["corpus", "list"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "not available against a remote engine" not in result.output

    def test_trust_list_runs_locally(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["trust", "list"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "not available against a remote engine" not in result.output

    def test_subjects_gc_runs_locally(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["subjects", "gc"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "No phantom subjects to prune." in result.output


class TestCorpusRetractGapClosed:
    """The specific footgun the ADR closed on review: corpus retract / delete.

    ``corpus retract`` routes (§2(a)); ``corpus delete`` / ``prune-orphans`` and
    the top-level ``export`` fail loud (§2(b)).
    """

    def test_corpus_delete_fails_loud_remote(self, runner: CliRunner, remote_engine: None) -> None:
        result = runner.invoke(app, ["corpus", "delete", "abc12345", "--yes"])
        assert result.exit_code != 0
        assert "`corpus delete` is not available against a remote engine" in result.output

    def test_corpus_prune_orphans_fails_loud_remote(
        self, runner: CliRunner, remote_engine: None
    ) -> None:
        result = runner.invoke(app, ["corpus", "prune-orphans", "--yes"])
        assert result.exit_code != 0
        assert "`corpus prune-orphans` is not available against a remote engine" in result.output

    def test_export_fails_loud_remote(self, runner: CliRunner, remote_engine: None) -> None:
        result = runner.invoke(app, ["export", "obsidian", "/tmp/particles-export-test"])
        assert result.exit_code != 0
        assert "`export` is not available against a remote engine" in result.output

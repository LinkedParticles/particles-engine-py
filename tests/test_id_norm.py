"""Tests for the CLI particle-ID normaliser (0.43.1).

The rendered output of exporters and lint findings refers to particles by
their *display form* ``p-XXXXXXXX``. Operators paste that exact string
into the CLI and reasonably expect it to work. The normaliser strips the
``p-`` (Markdown footnote / exporter form) or ``p:`` (older links-add
shorthand) display prefix before the LIKE-prefix match runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.api.cli._id_norm import normalise_particle_id


class TestNormaliser:
    """Unit-level coverage of the strip rules."""

    def test_strips_p_dash_display_prefix(self) -> None:
        assert normalise_particle_id("p-de005b0e") == "de005b0e"

    def test_strips_p_colon_links_shorthand(self) -> None:
        assert normalise_particle_id("p:abc123de") == "abc123de"

    def test_leaves_bare_uuid_untouched(self) -> None:
        full = "de005b0e-12ab-34cd-56ef-789012345678"
        assert normalise_particle_id(full) == full

    def test_leaves_bare_prefix_untouched(self) -> None:
        # 8-char prefix the user sees in `corpus list` — no display prefix.
        assert normalise_particle_id("de005b0e") == "de005b0e"

    def test_strips_whitespace(self) -> None:
        # Paste-with-newline ergonomics.
        assert normalise_particle_id("  p-de005b0e\n") == "de005b0e"

    def test_does_not_strip_other_known_prefixes(self) -> None:
        # Subjects start with `s-` in some renderings; don't silently mangle
        # if a future CLI verb accepts subject IDs through the same code path.
        assert normalise_particle_id("s-abc123de") == "s-abc123de"

    def test_does_not_double_strip(self) -> None:
        # ``p-p-abc`` is an unlikely paste but the helper should be
        # idempotent w.r.t. literal input — strip one prefix, not all of them.
        assert normalise_particle_id("p-p-abc") == "p-abc"


class TestParticleShowAcceptsDisplayPrefix:
    """End-to-end: paste the exporter's `p-XXXXXXXX` into `particle show`
    and have it resolve, not fall through to "No particle matches"."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    def _invoke(self, runner: CliRunner, args: list[str]) -> Any:
        return runner.invoke(app, args, catch_exceptions=False)

    def test_show_with_p_dash_prefix_resolves(self, runner: CliRunner, cli_db: Path) -> None:
        # Pull in the inserter fixture from test_cli's helper surface.
        from tests.test_cli import _add_active_particle, _run_async

        pid = _run_async(_add_active_particle(content="a claim"))
        # Operator pastes the exporter form (`p-` + first 8 chars).
        result = self._invoke(runner, ["particle", "show", f"p-{pid[:8]}"])
        assert result.exit_code == 0, result.output
        assert pid in result.output  # full UUID surfaced in the header

    def test_show_with_p_colon_prefix_resolves(self, runner: CliRunner, cli_db: Path) -> None:
        from tests.test_cli import _add_active_particle, _run_async

        pid = _run_async(_add_active_particle(content="another claim"))
        # Older links-add shorthand form.
        result = self._invoke(runner, ["particle", "show", f"p:{pid[:8]}"])
        assert result.exit_code == 0, result.output
        assert pid in result.output

    def test_show_with_bare_prefix_still_works(self, runner: CliRunner, cli_db: Path) -> None:
        # Regression guard: don't accidentally break the bare-prefix path.
        from tests.test_cli import _add_active_particle, _run_async

        pid = _run_async(_add_active_particle(content="third claim"))
        result = self._invoke(runner, ["particle", "show", pid[:8]])
        assert result.exit_code == 0, result.output
        assert pid in result.output

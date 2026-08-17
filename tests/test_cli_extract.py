"""Tests for ``particles extract``'s per-page reporting.

Reported from dogfood: a single 83-page PDF printed 83 ``Page N: 0 particles``
lines, burying the other four snapshots in the same run — and it flagged pages
that chunk-hash carry-forward had legitimately satisfied as
``⚠ zero yield``, i.e. reported the cache working as an anomaly.

Per-item detail is already assigned to ``--verbose``; this listing had
simply never been wired to it.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------
# Per-page reporting volume + the false zero-yield flag
# --------------------------------------------------------------------------


class TestPageStatReporting:
    """An 83-page PDF printed 83 lines unconditionally, and flagged pages that carry-forward had legitimately satisfied as ``⚠ zero yield``."""

    @staticmethod
    def _stats(total: int, with_particles: int) -> list[Any]:
        from particles.extraction.general import PageStat

        return [
            PageStat(page_number=i + 1, candidate_count=(3 if i < with_particles else 0))
            for i in range(total)
        ]

    def _render(self, stats: list[Any], carried: list[str], *, verbose: bool) -> str:
        import typer
        from typer.testing import CliRunner

        from particles.api.cli._output import configure_output
        from particles.api.cli.extract import _echo_page_stats

        cli = typer.Typer()

        @cli.command()
        def show() -> None:
            configure_output(verbose=verbose)
            _echo_page_stats(stats, carried)

        return CliRunner().invoke(cli, []).output

    def test_default_aggregates_instead_of_listing_every_page(self) -> None:
        out = self._render(self._stats(83, 12), [], verbose=False)
        assert "12 of 83 pages produced particles" in out
        assert "Page   1:" not in out
        assert out.count("\n") <= 2  # one line, not 83

    def test_verbose_itemises(self) -> None:
        out = self._render(self._stats(5, 2), [], verbose=True)
        assert "Pages: 5" in out
        assert "Page   1:" in out
        assert "Page   5:" in out

    def test_zero_yield_flagged_without_carry_forward(self) -> None:
        """An empty page with no cache hit is a real signal — keep it."""
        out = self._render(self._stats(3, 1), [], verbose=True)
        assert "⚠ zero yield" in out

    def test_zero_yield_not_flagged_when_carried_forward(self) -> None:
        """The cache working is not an anomaly. This was the false warning."""
        out = self._render(self._stats(3, 1), ["p-1", "p-2"], verbose=True)
        assert "⚠ zero yield" not in out

    def test_aggregate_explains_carry_forward(self) -> None:
        out = self._render(self._stats(83, 12), ["p-1"], verbose=False)
        assert "the rest matched already-extracted content" in out

    def test_all_pages_productive_prints_nothing_by_default(self) -> None:
        """Nothing to report is reported as nothing."""
        assert self._render(self._stats(4, 4), [], verbose=False).strip() == ""

    def test_no_page_stats_prints_nothing(self) -> None:
        assert self._render([], [], verbose=True).strip() == ""

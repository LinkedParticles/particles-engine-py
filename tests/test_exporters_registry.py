"""Tests for the exporter plugin registry (particles/exporters/registry.py).

Was 0% covered in the architecture-review baseline. The module is small
(format-keyed cached map + a Protocol definition), so 5 tests get it
to 100%.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from particles.exporters.registry import ExporterPlugin, get_exporters


@pytest.fixture(autouse=True)
def _reset_exporter_cache() -> Generator[None, None, None]:
    """Clear the module-level cache around each test so we exercise the
    "first call builds, second call returns cached" path deterministically."""
    from particles.exporters import registry

    registry._exporters = None
    yield
    registry._exporters = None


class TestGetExporters:
    def test_returns_dict_keyed_by_format(self) -> None:
        exporters = get_exporters()
        assert isinstance(exporters, dict)
        # Built-in formats from _make_exporters
        assert "obsidian" in exporters
        assert "anki" in exporters

    def test_second_call_returns_same_instance(self) -> None:
        """The map is cached: get_exporters() returns the same dict object on
        subsequent calls, not a freshly built one."""
        first = get_exporters()
        second = get_exporters()
        assert first is second

    def test_each_entry_satisfies_protocol(self) -> None:
        for fmt, exp in get_exporters().items():
            assert isinstance(exp, ExporterPlugin)
            assert fmt == exp.FORMAT  # the registry key matches the plugin's own slug

    def test_anki_format_string_matches_class(self) -> None:
        from particles.exporters.anki import AnkiExporter

        assert get_exporters()["anki"].__class__ is AnkiExporter

    def test_obsidian_format_string_matches_class(self) -> None:
        from particles.exporters.obsidian import ObsidianExporter

        assert get_exporters()["obsidian"].__class__ is ObsidianExporter


class TestExportCmdHelpListsAllFormats:
    """Guard against future drift: every registered exporter format must
    appear in ``particles export --help``. The help text is a literal
    string (Typer evaluates it at decoration time) so it can't be
    auto-derived; this test catches the next contributor who forgets to
    update it."""

    def test_help_text_mentions_every_registered_format(self) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        result = CliRunner().invoke(app, ["export", "--help"])
        assert result.exit_code == 0
        for fmt in get_exporters():
            assert fmt in result.output, (
                f"`particles export --help` does not mention {fmt!r}. "
                f"Update particles/api/cli/export.py — the help strings "
                f"need to be kept in sync with the registry."
            )

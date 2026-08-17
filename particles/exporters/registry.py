"""Exporter plugin registry.

To add a new exporter:
  1. Create particles/exporters/<name>.py implementing ExporterPlugin.
  2. Add it to _make_exporters() below — the only file outside the new module
     that needs editing.

See particles/exporters/AGENTS.md for the full two-file procedure.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from particles.exporters.summaries import BaseExporterSummary

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@runtime_checkable
class ExporterPlugin(Protocol):
    """Protocol every exporter must satisfy.

    **Optional credential declaration.** An API-target exporter may
    set a class attribute ``REQUIRES_SECRET: str`` naming the env var holding the
    secret it needs (e.g. ``"NOTION_API_KEY"``) — a *declaration*, not the value.
    It is deliberately **not** a required Protocol member: the five shipped
    filesystem exporters do not set it, and :func:`required_secret` reads it with
    ``getattr(exporter, "REQUIRES_SECRET", None)`` so the contract stays purely
    additive. The registry/CLI pre-flight uses it to verify the env var is
    present before any store read or network call, but never reads the secret
    value here — that is the exporter's own first-statement getter call.
    """

    FORMAT: str  # unique lowercase slug, e.g. "obsidian", "anki"

    async def export(
        self,
        session: AsyncSession,
        output: Path | None,
        **options: object,
    ) -> BaseExporterSummary:
        """Export the knowledge base.

        output: target path interpreted by the exporter —
          directory  for vault-style exporters (Obsidian, Logseq)
          file path  for single-file exporters (Anki, JSON Lines)
          None       for API-based exporters (Notion, etc.)

        options: exporter-specific keyword arguments; each exporter extracts
          only the keys it recognises and ignores the rest.

        Returns a typed summary model for display to the user.
        """
        ...


_exporters: dict[str, ExporterPlugin] | None = None


def get_exporters() -> dict[str, ExporterPlugin]:
    """Return the format-keyed exporter map (cached after first call)."""
    global _exporters
    if _exporters is None:
        _exporters = _make_exporters()
    return _exporters


def required_secret(exporter: ExporterPlugin) -> str | None:
    """Return the env-var name an exporter declares via ``REQUIRES_SECRET``, else None.

    Reads the optional declarative attribute with ``getattr(..., None)``
    so the filesystem exporters that never set it stay untouched. The CLI pre-flight
    (and any future ``export --list``) calls this to learn *that* an exporter needs
    a secret and *which* env var names it, without ever reading the secret value.
    """
    return getattr(exporter, "REQUIRES_SECRET", None)


def _make_exporters() -> dict[str, ExporterPlugin]:
    from particles.exporters.anki import AnkiExporter
    from particles.exporters.graph import GraphExporter
    from particles.exporters.jsonl import JsonlExporter
    from particles.exporters.logseq import LogseqExporter
    from particles.exporters.notion import NotionExporter
    from particles.exporters.obsidian import ObsidianExporter
    from particles.exporters.wiki import WikiExporter

    exporters: list[ExporterPlugin] = [
        ObsidianExporter(),
        AnkiExporter(),
        WikiExporter(),
        LogseqExporter(),
        JsonlExporter(),
        NotionExporter(),
        GraphExporter(),
        # Register new exporters here
    ]
    return {e.FORMAT: e for e in exporters}

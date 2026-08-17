"""Importer plugin registry (Engine layer).

The importer half of the plugin machinery. Importers fetch a URL and write the
resulting blob into the corpus, so they reach the Engine layer
(``particles.corpus``, ``particles.store``) and live here rather than in the
Client-layer ``particles.extraction`` package. The
complementary extractor registry — which produces store-free candidate
particles — stays in ``particles.extraction.registry``.

To add a new importer:
  1. Implement ``ImporterPlugin`` in ``particles/ingest/importers/<name>.py``
     (it may import the Client-safe parsing helpers from the matching
     ``particles.extraction.<name>`` module).
  2. Add an entry to ``_make_importers()`` below.

The generic HTTP fetch in ``deposit.py`` is the implicit importer fallback and
does not appear in the importer list.

the role formerly named ``DepositorPlugin`` is renamed to
``ImporterPlugin`` so the SDK exposes the complementary triplet
``ImporterPlugin`` / ``ExtractorPlugin`` / ``ExporterPlugin``. The
user-facing CLI verb ``particles deposit`` is unchanged (it describes
what happens to the corpus, not the plugin role that performs the fetch).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@runtime_checkable
class ImporterPlugin(Protocol):
    """Protocol every domain-specific importer must satisfy.

    Renamed from ``DepositorPlugin`` to make the role triplet
    ``ImporterPlugin`` / ``ExtractorPlugin`` / ``ExporterPlugin`` consistently
    use complementary verbs. The contract is unchanged: an importer routes
    URLs to format-specific fetch logic and writes the resulting blob into
    the corpus.
    """

    def accepts_url(self, url: str) -> bool:
        """Return True if this importer handles the given URL."""
        ...

    async def deposit(
        self,
        session: AsyncSession,
        url: str,
        deposited_by: str,
        tags: list[str],
    ) -> tuple[str, str]:
        """Fetch the URL, write the blob, and return (entry_id, snapshot_id).

        The method name ``deposit`` describes the *action* (writing into the
        corpus); the plugin role is now ``ImporterPlugin``.
        """
        ...


# ---------------------------------------------------------------------------
# Lazy singleton — built on first call to avoid circular imports
# ---------------------------------------------------------------------------

_importers: list[ImporterPlugin] | None = None


def get_importers() -> list[ImporterPlugin]:
    """Return the ordered importer list (cached after first call)."""
    global _importers
    if _importers is None:
        _importers = _make_importers()
    return _importers


def _make_importers() -> list[ImporterPlugin]:
    # defer: lazy-init — plugin classes are imported inside the factory so
    # adding or removing a plugin file (or one of its own top-level imports)
    # does not cascade-fail registry import-time. See root AGENTS.md
    # § Code conventions → Deferred imports (case 2: lazy-init).
    from particles.ingest.importers.github import GitHubImporter
    from particles.ingest.importers.hackernews import HackerNewsImporter
    from particles.ingest.importers.mastodon import MastodonImporter
    from particles.ingest.importers.nomisma import NomismaImporter
    from particles.ingest.importers.numista import NumistaImporter
    from particles.ingest.importers.reddit import RedditImporter
    from particles.ingest.importers.wikidata import WikidataImporter

    return [
        WikidataImporter(),
        NomismaImporter(),
        NumistaImporter(),
        RedditImporter(),
        HackerNewsImporter(),
        MastodonImporter(),
        GitHubImporter(),
    ]


async def ensure_extractor_records(session: AsyncSession) -> int:
    """Idempotent upsert of all built-in extractor records. Returns write count.

    Called at ``particles db init``. trust_weight is only written on first
    INSERT — subsequent calls preserve operator-customised values.

    Lives in the Engine layer because it writes the persistent
    ``ExtractorRecord`` table (``particles.store``); it reads the extractor
    list from the Client-layer registry via ``get_extractors()``.
    """
    from datetime import UTC, datetime

    from particles.core.schema import ExtractorRecord
    from particles.extraction.registry import get_extractors
    from particles.store.extractor_store import upsert_extractor_record

    wrote = 0
    for plugin in get_extractors():
        applicability = getattr(plugin, "APPLICABILITY", [])
        trust_weight: float = getattr(plugin, "DEFAULT_TRUST_WEIGHT", 0.7)
        record = ExtractorRecord(
            extractor_id=plugin.EXTRACTOR_ID,
            name=plugin.EXTRACTOR_ID,
            version=plugin.EXTRACTOR_VERSION,
            applicability=applicability,
            trust_weight=trust_weight,
            registered_by="anthropic/particles-sdk",
            registered_at=datetime.now(UTC),
        )
        if await upsert_extractor_record(session, record):
            wrote += 1
    return wrote

"""Engine-layer importer plugins.

Importers fetch a URL and write the resulting blob into the corpus, so they
reach the Engine layer (``particles.corpus``) and live here rather than in the
Client-layer ``particles.extraction`` package. Each importer imports the
Client-safe parsing helpers it needs back from the matching
``particles.extraction.<name>`` module (Engine → Client is permitted).

The registry (``ImporterPlugin`` Protocol, ``get_importers``,
``ensure_extractor_records``) lives in :mod:`particles.ingest.importers.registry`.
"""

from __future__ import annotations

from particles.ingest.importers.github import GitHubImporter
from particles.ingest.importers.hackernews import HackerNewsImporter
from particles.ingest.importers.mastodon import MastodonImporter
from particles.ingest.importers.nomisma import NomismaImporter
from particles.ingest.importers.numista import NumistaImporter
from particles.ingest.importers.reddit import RedditImporter
from particles.ingest.importers.wikidata import WikidataImporter

__all__ = [
    "GitHubImporter",
    "HackerNewsImporter",
    "MastodonImporter",
    "NomismaImporter",
    "NumistaImporter",
    "RedditImporter",
    "WikidataImporter",
]

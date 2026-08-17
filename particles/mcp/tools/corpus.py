"""``list_corpus_entries`` / ``corpus_links_suggest`` — corpus reads.

Routed through the ``Backend`` seam: in-process locally, against
the canonical engine when one is configured.
"""

from __future__ import annotations

from typing import Any


async def list_corpus_entries(
    limit: int = 50,
    source_type: str | None = None,
) -> list[dict[str, Any]]:
    """List recent corpus entries (most-recently-deposited first).

    Args:
        limit: Maximum entries to return (default 50).
        source_type: Optional filter (e.g. ``"WIKIDATA_API"``).
    """
    from particles.api.client import get_backend

    entries = await get_backend().list_corpus_entries(limit=limit, source_type=source_type)
    return [e.model_dump(mode="json") for e in entries]


async def corpus_links_suggest(
    limit: int | None = None,
    min_sources: int | None = None,
) -> dict[str, Any]:
    """Rank undeposited URLs the corpus frequently cites (read-only).

    Surfaces primary sources the corpus leans on but has never deposited —
    "hearsay vs. primary source" gaps — ranked by trust-weighted distinct-source
    diversity × recency. Read-only: the ``dismiss`` half of the
    workflow lives on the CLI / HTTP API.

    Args:
        limit: Maximum suggestions to return. Defaults to
            ``citation_signal.rank_cap`` when omitted.
        min_sources: Minimum distinct citing sources required to surface a URL.
            Defaults to ``citation_signal.min_distinct_sources`` when omitted.

    Returns:
        ``{suggestions, total_candidates, capped}`` matching the
        ``DepositSuggestReport`` schema.
    """
    from particles.api.client import get_backend

    report = await get_backend().corpus_links_suggest(limit=limit, min_sources=min_sources)
    return report.model_dump(mode="json")

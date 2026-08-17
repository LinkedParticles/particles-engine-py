"""``links_suggest`` — co-evidential candidate proposal, report-only.

The MCP surface is read-only, so only ``REPORT`` mode is exposed:
candidate pairs are listed but never LLM-judged or auto-linked. Agents drilling
into "what duplicates does Subject X have" call this with ``subject_id``; the
``--llm-judge`` / ``--apply`` resolution workflow lives on the CLI / HTTP API.

Routed through the ``Backend`` seam: in-process locally, against
the canonical engine when one is configured.
"""

from __future__ import annotations

from typing import Any


async def links_suggest(
    subject_id: str | None = None,
    threshold: float | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Propose co-evidential candidate pairs within Subjects (read-only).

    Within each Subject, ACTIVE particles whose embeddings exceed the
    similarity threshold and are not already linked CO_EVIDENTIAL are
    surfaced as candidate pairs for operator review. No LLM call, no
    mutation — this is the ``REPORT`` mode of ``particles links suggest``.

    Args:
        subject_id: Restrict to one Subject. ``None`` scans every
            Subject — prefer passing a ``subject_id`` to keep results focused.
        threshold: Cosine-similarity floor (0.0–1.0). Defaults to
            ``links_suggest.candidate_threshold`` when omitted.
        limit: Maximum candidate pairs to include across all clusters
            (default 100). A ``truncated`` flag is added when the cap fires.

    Returns:
        ``{run_at, mode, clusters, total_candidates, warnings}`` matching the
        ``SuggestReport`` schema, optionally with ``truncated: True``.
    """
    if limit <= 0:
        raise ValueError("limit must be a positive integer.")

    # Routed through the backend seam. The deferred-import test seam is
    # preserved one layer down: ``LocalBackend.links_suggest`` defers
    # ``from particles.operations.links_suggest import suggest_co_evidential``, so
    # tests patching that symbol still reach the call. See tests/AGENTS.md
    # § Mocking strategy.
    from particles.api.client import get_backend
    from particles.core.schema import SuggestMode

    report = await get_backend().links_suggest(
        subject_id=subject_id,
        threshold=threshold,
        mode=SuggestMode.REPORT,
        confirmed=False,
    )
    out = report.model_dump(mode="json")

    # Flat cap across clusters to keep the response under per-tool-result size
    # limits even when one Subject has thousands of candidates.
    if report.total_candidates > limit:
        out["truncated"] = True
        out["total_candidates_before_truncation"] = report.total_candidates
        kept = limit
        capped_clusters = []
        for cluster in out["clusters"]:
            if kept <= 0:
                break
            cands = cluster["candidates"][:kept]
            kept -= len(cands)
            cluster = {**cluster, "candidates": cands}
            capped_clusters.append(cluster)
        out["clusters"] = capped_clusters
    return out

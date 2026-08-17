"""``lint`` — structural lint findings with category/summary/limit knobs.

Routed through the ``Backend`` seam: the structural-only lint runs
in-process locally or on the canonical engine when one is configured.
"""

from __future__ import annotations

from typing import Any


async def lint(
    category: str | None = None,
    summary_only: bool = False,
    limit: int = 100,
) -> dict[str, Any]:
    """Run structural lint checks over the particle store (no fixes, no LLM).

    The MCP surface is read-only, so ``--fix`` is forced off and the
    LLM-assisted semantic checks are disabled — agents read findings
    and surface them to the operator, who decides whether to act.

    Args:
        category: Optional ``finding_type`` filter (e.g. ``STALENESS``,
            ``PHANTOM_SUBJECT``, ``WIKIDATA_LINK_MISMATCH``). When set,
            only findings of that kind are returned; ``summary`` is
            always the full counts dict and reflects every category
            actually emitted by this run.
        summary_only: When ``True``, drop the ``findings`` list and
            return only the ``summary`` counts dict (plus ``run_at``).
            Useful when the agent only needs to know what *kinds* of
            problems exist before drilling in with ``category``.
        limit: Maximum findings to include after filtering (default
            100). Hard cap to keep the response under per-tool-result
            size limits even when many findings of one category exist.
            A ``truncated`` flag is added when the cap fires.

    Returns:
        ``{run_at, findings, summary}`` matching the ``LintReport``
        schema, optionally with ``truncated: True`` and (when
        ``summary_only``) without the ``findings`` key.
    """
    if limit <= 0:
        raise ValueError("limit must be a positive integer.")

    from particles.api.client import get_backend

    # The MCP surface is read-only: fix off, semantic off. ``low_coverage_threshold``
    # uses run_lint's default (3) so the routed result matches the in-process one.
    report = await get_backend().lint(fix=False, semantic=False, low_coverage_threshold=3)
    out = report.model_dump(mode="json")

    # Filter by category before truncation so the limit applies to the
    # filtered set, not the unfiltered superset.
    if category is not None:
        out["findings"] = [f for f in out["findings"] if f.get("finding_type") == category]

    if summary_only:
        # Drop the findings list entirely; the summary already
        # carries the per-category counts. We still include
        # `category` in the response when set so callers can confirm
        # what they filtered on.
        out.pop("findings", None)
        if category is not None:
            out["filtered_category"] = category
        return out

    if len(out["findings"]) > limit:
        out["truncated"] = True
        out["total_findings_before_truncation"] = len(out["findings"])
        out["findings"] = out["findings"][:limit]

    if category is not None:
        out["filtered_category"] = category
    return out

"""``subjects_list`` / ``subjects_search`` / ``subjects_show``.

Routed through the ``Backend`` seam: in-process locally, against
the canonical engine when one is configured.
"""

from __future__ import annotations

from typing import Any, Literal


async def subjects_list(
    limit: int = 50,
    offset: int = 0,
    order: str = "name",
) -> list[dict[str, Any]]:
    """List subjects in the knowledge graph (alphabetical by canonical_name).

    Paginated to keep responses within an MCP-client-friendly size — the
    full subject set can easily exceed per-tool-result token caps.

    Args:
        limit: Maximum subjects to return (default 50).
        offset: Number of subjects to skip before returning results
            (default 0). Combine with ``limit`` to page through the
            full set.
        order: ``"name"`` (alphabetical, default) or ``"degree"`` —
            most-connected first, by descending count of ACTIVE
            particles linked to each subject (name tie-break). Use
            ``"degree"`` to find a store's hub subjects, e.g. as a
            ``graph_view`` starting scope.
    """
    if order not in ("name", "degree"):
        raise ValueError('order must be "name" or "degree".')
    order_lit: Literal["name", "degree"] = "degree" if order == "degree" else "name"

    from particles.api.client import get_backend

    subjects = await get_backend().subjects_list(limit=limit, offset=offset, order=order_lit)
    return [s.model_dump(mode="json") for s in subjects]


async def subjects_search(
    query: str,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Substring search over subject canonical names (case-insensitive).

    Args:
        query: Substring to match against ``canonical_name``.
        limit: Maximum subjects to return (default 50). A generic
            substring like "the" can match thousands of subjects, so
            this caps the page size.
        offset: Number of matches to skip before returning results
            (default 0). Results are alphabetical by canonical_name
            so pagination is stable.
    """
    if limit <= 0:
        raise ValueError("limit must be a positive integer.")
    if offset < 0:
        raise ValueError("offset must be zero or a positive integer.")

    from particles.api.client import get_backend

    subjects = await get_backend().subjects_search(query, limit=limit, offset=offset)
    return [s.model_dump(mode="json") for s in subjects]


async def subjects_show(
    subject_id: str,
    particle_id_limit: int = 100,
) -> dict[str, Any]:
    """Show one subject with the particle IDs linked to it.

    Args:
        subject_id: Full UUID or unambiguous prefix (≥ 8 chars).
        particle_id_limit: Maximum particle IDs to include in
            ``particle_ids`` (default 100). For hot subjects the full
            list can blow the per-tool-result token cap; use
            ``particles_list(subject_id=…, limit=…, offset=…)`` for
            the full set when ``particle_count > particle_id_limit``.

    Returns:
        Dict with ``subject`` (Pydantic JSON), ``particle_ids`` (capped
        at ``particle_id_limit``, ordered by id for stable pagination),
        and ``particle_count`` — the true total of linked particles,
        so callers can detect when the list was truncated.
    """
    if particle_id_limit <= 0:
        raise ValueError("particle_id_limit must be a positive integer.")

    from particles.api.client import get_backend

    detail = await get_backend().subject_detail(subject_id, particle_id_limit=particle_id_limit)
    return {
        "subject": detail.subject.model_dump(mode="json"),
        "particle_ids": detail.particle_ids,
        "particle_count": detail.particle_count,
    }

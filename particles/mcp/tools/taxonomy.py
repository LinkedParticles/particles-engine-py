"""``list_taxonomies`` — paginated taxonomy + tag-tree dump.

Routed through the ``Backend`` seam.
"""

from __future__ import annotations

from typing import Any


async def list_taxonomies(
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List every taxonomy + its full tag tree.

    Args:
        limit: Maximum taxonomies to return (default 50). Each
            taxonomy embeds its full tag tree, so even a handful of
            large taxonomies can push the response past the
            per-tool-result token cap.
        offset: Number of taxonomies to skip before returning results
            (default 0). Results are ordered by name so pagination is
            stable.
    """
    if limit <= 0:
        raise ValueError("limit must be a positive integer.")
    if offset < 0:
        raise ValueError("offset must be zero or a positive integer.")

    from particles.api.client import get_backend

    taxonomies = await get_backend().list_taxonomies(limit=limit, offset=offset)
    return [t.model_dump(mode="json") for t in taxonomies]

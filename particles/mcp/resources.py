"""MCP resource surface — the session-start memory digest.

The MCP protocol's *resources* primitive: readable URIs a client pulls directly,
distinct from the *tools* (function calls) that make up the rest of the surface
. This module registers the **compiled memory digest** at
``particles://digest/<store>`` — the ``MEMORY.md`` analog, deferred.

The digest is **read-only**, costs **zero LLM calls and zero embeddings**, and
is **rendered fresh on every read** (no cache — it has no LLM cost to amortize,
so the synthesis-cache pattern would only add staleness). It renders
the ``contested`` marker — the MUST the digest inherits.

Two registrations (§2):

* a resource **template** ``particles://digest/{store}`` — addresses *any*
  configured store on demand;
* a concrete ``particles://digest/<handle>`` for each store in
  ``config.digest_listed_stores()`` (the write-enabled memory stores plus
  ``mcp.recall.digest_stores``) so it is enumerated in ``resources/list`` and
  thus discoverable / auto-loadable at session start.

Assembly (``operations/digest.py::build_digest``, re-exported here for
backwards-compatible imports) is split from formatting
(``exporters/markdown.py``). Since the resource handlers resolve the
digest through the ``Backend`` seam (``get_backend().digest(store)``): with no
engine configured the local backend renders it in-process (today's behaviour);
with ``engine.base_url`` set the digest is fetched from the canonical store over
HTTP, so the agent's session-start recall is never a stale laptop copy.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from mcp.server.fastmcp import FastMCP

from particles.config import get_config
from particles.operations.digest import build_digest

__all__ = ["build_digest", "register_digest_resources"]

log = logging.getLogger(__name__)


async def _render_digest(store: str) -> str:
    """Render the digest for ``store`` through the backend seam."""
    from particles.api.client import get_backend

    return await get_backend().digest(store)


_DIGEST_DESCRIPTION = (
    "Session-start memory digest: one line per ACTIVE belief, ranked "
    "by effective confidence, contested beliefs flagged. The MEMORY.md analog — "
    "load it at session start to recall this store's standing context."
)


def register_digest_resources(server: FastMCP) -> None:
    """Register the digest resource template + the listed concrete resources (§2).

    The template is registered always (the digest is a read-only view of any
    store). Concrete per-store resources are listed only for
    ``config.digest_listed_stores()`` — the write-enabled memory stores plus
    ``mcp.recall.digest_stores`` — so a stock install lists none but the template
    still addresses ``particles://digest/<store>`` on demand.
    """

    @server.resource(
        "particles://digest/{store}",
        name="memory-digest",
        description=_DIGEST_DESCRIPTION,
        mime_type="text/markdown",
    )
    async def _digest_template(store: str) -> str:
        return await _render_digest(store)

    for handle in get_config().digest_listed_stores():

        def _make(store: str) -> Callable[[], Awaitable[str]]:
            async def _digest() -> str:
                return await _render_digest(store)

            return _digest

        server.resource(
            f"particles://digest/{handle}",
            name=f"memory-digest:{handle}",
            description=f"Session-start memory digest for the {handle!r} store.",
            mime_type="text/markdown",
        )(_make(handle))

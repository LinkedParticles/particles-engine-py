"""``graph_view`` — one scoped epistemic subgraph.

Routed through the ``Backend`` seam: with no engine configured the
local backend assembles the subgraph in-process; with ``engine.base_url`` set it
calls ``GET /graph`` on the canonical engine. This is how an agent hands the
operator the picture of the knowledge it consulted — the returned ``GraphData``
is the same contract the static ``export graph`` artifact embeds and the unified
web UI's ``#/browse`` route renders (one build, two presentations).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import urlencode


async def graph_view(
    subject_id: str | None = None,
    query: str | None = None,
    inconsistency_id: str | None = None,
    manifest: str | None = None,
    section: str | None = None,
    hops: int = 1,
    history: bool = False,
    as_of: str | None = None,
    max_nodes: int | None = None,
    store: str = "default",
) -> dict[str, Any]:
    """Render one scoped epistemic subgraph of the particle store.

    Scope is mandatory — pass exactly one of ``subject_id`` / ``query`` /
    ``inconsistency_id`` / ``manifest``+``section`` (a whole-store render
    does not exist; the anti-hairball invariant). Every
    epistemic quantity in the result (effective confidence, contested
    badges, as-of visibility, utility evidence) is computed at render time
    by the engine and never stored.

    Args:
        subject_id: Neighbourhood scope — this Subject, its particles, and
            subjects reachable within ``hops`` via multi-subject particles.
            Accepts a subject id or an exact (case-insensitive) canonical
            name / alias; the render's ``scope_ref`` is the resolved id.
        query: Retrieval-set scope — the top-K hits of the existing query
            ranking rendered as their subjects plus the hit particles
            ("the picture of the knowledge the agent consulted").
        inconsistency_id: Evidence scope — a contradiction's
            picture: the INCONSISTENCY particle as the anchor, its two
            disputant beliefs shown with their true statuses (the
            quarantined side included), their subjects and sources. Accepts
            a full particle id or a unique prefix (a contested badge's
            ``inconsistency_id`` is the usual source).
        manifest: With ``section``, projection scope — an
            manifest section's deterministic selection rendered as
            a graph. The manifest path resolves on the engine host.
        section: The manifest section's ``region`` id, or its exact
            (case-insensitive) title.
        hops: Neighbourhood radius for subject scope (clamped to
            ``graph.max_hops``).
        history: Also include retired supersession-chain ancestors, rendered
            as ghosts.
        as_of: ISO-8601 past instant (a bare date means start of that day,
            UTC): render the graph as believed at T. A future
            instant is rejected.
        max_nodes: Per-call node cap (clamped to ``graph.max_nodes``).
        store: Store handle to render from; ``"default"`` is the
            canonical store.

    Returns:
        The ``GraphData`` render as JSON — ``nodes`` (subjects with display
        aggregates), ``edges`` (multi-subject particles), ``particles``
        (per-particle epistemics payloads keyed by id), ``supersessions``,
        the ``census``, and human-readable ``disclosures`` for any cap that
        bound. When ``engine.base_url`` is configured, a ``url`` field
        additionally deep-links the same scope on the unified web UI
        (``/app#/browse?…``) so the operator can open the
        interactive picture — the default stdio install has no HTTP server,
        so the field is additive, never assumed.
    """
    if (manifest is None) != (section is None):
        raise ValueError("projection scope needs both manifest and section")
    selectors = [s for s in (subject_id, query, inconsistency_id, manifest) if s is not None]
    if len(selectors) != 1:
        raise ValueError(
            "graph scope is mandatory: pass exactly one of subject_id, query, "
            "inconsistency_id, or manifest+section (a whole-store render does "
            "not exist)"
        )
    as_of_dt: datetime | None = None
    if as_of is not None:
        try:
            as_of_dt = datetime.fromisoformat(as_of)
        except ValueError:
            raise ValueError(
                f"Invalid as_of value {as_of!r}: expected an ISO-8601 date or "
                "datetime (e.g. 2000-01-01 or 2006-08-24T12:00:00+00:00)."
            ) from None

    from particles.api.client import get_backend

    data = await get_backend().graph(
        subject_id=subject_id,
        query=query,
        inconsistency_id=inconsistency_id,
        manifest=manifest,
        section=section,
        hops=hops,
        history=history,
        as_of=as_of_dt,
        max_nodes=max_nodes,
        store=store,
    )
    out = data.model_dump(mode="json")

    # deep-link the same scope on the unified web
    # UI's #/browse route (né #/graph — the app accepts both) when an engine
    # is configured. Additive only — the stdio/LocalBackend install has no
    # HTTP server to link to.
    from particles.config import get_config

    base_url = get_config().engine.base_url
    if base_url:
        params: dict[str, str]
        if subject_id is not None:
            params = {"scope": "subject", "subject_id": subject_id}
        elif inconsistency_id is not None:
            # Link the RESOLVED anchor id (scope_ref), so a prefix input still
            # yields a canonical, shareable address.
            params = {"scope": "inconsistency", "inconsistency_id": str(out["scope_ref"])}
        elif manifest is not None:
            params = {"scope": "projection", "manifest": manifest, "section": section or ""}
        else:
            params = {"scope": "query", "q": query or ""}
        if hops != 1 and subject_id is not None:
            params["hops"] = str(hops)
        if history:
            params["history"] = "true"
        if as_of_dt is not None:
            params["as_of"] = as_of_dt.isoformat()
        out["url"] = f"{base_url.rstrip('/')}/app#/browse?{urlencode(params)}"
    return out

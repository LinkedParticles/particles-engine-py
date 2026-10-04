# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Session-start memory digest assembly.

``build_digest`` compiles the ``MEMORY.md`` analog served at
``particles://digest/<store>``: one line per ACTIVE belief, ranked
by effective confidence, contested beliefs flagged. It lives in ``operations/``
so it is the **single convergence point** reached two ways — the
MCP resource handler resolves it locally through ``LocalBackend.digest``; the
remote engine renders it server-side at ``GET /digest/{store}`` and the
``HttpBackend`` proxies that, so the agent's session-start recall
reflects the canonical store rather than a stale laptop copy.

It is **read-only**, costs **zero LLM calls and zero embeddings**, and is
**rendered fresh on every read** (no cache — it has no LLM cost to amortize, so
the synthesis-cache pattern would only add staleness). Assembly lives
here; the terse one-line-per-belief formatting is
``render/markdown.py::render_digest``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from particles.config import get_config
from particles.core.schema import ContestedBadge, ObserverScopeNote
from particles.core.status import Status
from particles.db import session_scope
from particles.llm.usage import render_store_spend_line
from particles.operations.llm_spend import store_llm_spend
from particles.operations.query.contested import compute_contested_badges
from particles.operations.query.effective_confidence import score_confidence_and_rank
from particles.operations.query.observer_scope import filter_visible
from particles.render.markdown import DigestEntry, render_digest_located
from particles.store.particle_store import get_inconsistency_backrefs, get_particles_by_status
from particles.store.subject_store import list_all_subjects


class DigestLine(BaseModel):
    """Which belief one digest line shows, and where the line starts."""

    particle_id: str
    offset: int = Field(description="Character offset in the digest text where the line starts.")
    contested_bases: list[str] = Field(
        default_factory=list, description="The contested bases the line displays."
    )


class RenderedDigest(BaseModel):
    """The digest text, with the belief behind each of its lines in order."""

    markdown: str
    lines: list[DigestLine] = Field(default_factory=list)


async def build_digest(store: str, observer_project: str | None = None) -> str:
    """The digest text alone; :func:`build_digest_located` documents the build."""
    return (await build_digest_located(store, observer_project)).markdown


async def build_digest_located(store: str, observer_project: str | None = None) -> RenderedDigest:
    """Render the session-start memory digest for one store.

    Returns the text with, for each belief line in order, the belief's id, the
    line's character offset and the contested bases it shows. A caller that
    cuts the text to a byte budget keeps the lines whose offset fell inside the
    cut (the SessionStart hook's record).

    Gathers the store's ACTIVE beliefs, scores each with the query-path effective
    confidence (full formula — trust × source-trust × recency; never an LLM or
    embedding), orders them descending, caps at ``mcp.recall.digest_max_beliefs``
    (top-N; 0 = no cap), marks every contested belief with the composed badge's
    fired bases (gated by ``contestedness.badge_enabled`` and computed
    for the rendered top-N only) plus the open-INCONSISTENCY drill-down id,
    and hands the ordered data to
    :func:`particles.render.markdown.render_digest`.

    ``observer_project`` reads the store through a project observer:
    only beliefs that are global, observed in that project, or widened are
    ranked, and the digest says so in one line. The predicate runs before
    scoring and the cap, so a project's 200 lines are its own. ``None`` is the
    store-wide digest, unchanged.

    Read fresh on every call — the digest is never cached, so it can never serve
    a stale recall.
    """
    max_beliefs = get_config().mcp.recall.digest_max_beliefs
    async with session_scope(store) as session:
        actives = await get_particles_by_status(session, Status.ACTIVE)
        observer: ObserverScopeNote | None = None
        if observer_project is not None:
            scope = await filter_visible(session, actives, observer_project)
            actives = [p for p in actives if p.id in scope.visible_ids]
            observer = scope.note(observer_project)
        total = len(actives)
        # The digest is a projection-path recall surface, so it
        # ranks with the usefulness rank-lift applied — unlike the semantic-search
        # `query` path, which never does. The two quantities are
        # kept separate: `rank_score` orders the beliefs (and is not
        # a probability — it may exceed 1.0), while the *displayed* number stays
        # the truth-axis
        # effective confidence, so the digest never labels a lifted score
        # "confidence".
        eff_conf, rank_score = await score_confidence_and_rank(
            session, actives, populate_cache=True
        )
        backrefs = await get_inconsistency_backrefs(session)
        # the store's recorded LLM spend, one closing line.
        spend = await store_llm_spend(session)
        subject_names = {s.id: s.canonical_name for s in await list_all_subjects(session)}

        # Order by rank score desc; id as a stable tiebreaker so the
        # rendered digest is deterministic for a given store state.
        ordered = sorted(actives, key=lambda p: (-rank_score.get(p.id, p.confidence.value), p.id))
        if max_beliefs > 0:
            ordered = ordered[:max_beliefs]

        # compose the contested badge for the rendered entries only
        # (§5 cost discipline); off restores the inconsistency-only flag exactly.
        badges: list[ContestedBadge | None] = [None] * len(ordered)
        if get_config().contestedness.badge_enabled:
            badges = await compute_contested_badges(session, ordered, backrefs=backrefs)

    entries = [
        DigestEntry(
            content=p.content,
            effective_confidence=eff_conf.get(p.id, p.confidence.value),
            subjects=tuple(subject_names[sid] for sid in p.subject_ids if sid in subject_names),
            asserted_at=p.asserted_at,
            contested=backrefs.get(p.id),
            contested_bases=tuple(badge.bases) if badge is not None else (),
        )
        for p, badge in zip(ordered, badges, strict=True)
    ]
    spend_line = render_store_spend_line(spend) if spend is not None else None
    markdown, offsets = render_digest_located(store, entries, total, observer, spend_line)
    return RenderedDigest(
        markdown=markdown,
        lines=[
            DigestLine(particle_id=p.id, offset=offset, contested_bases=list(entry.contested_bases))
            for p, entry, offset in zip(ordered, entries, offsets, strict=True)
        ],
    )

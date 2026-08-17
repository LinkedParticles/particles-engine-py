"""Leverage scoring — finite and high-value first.

The score is a weighted sum of four normalized (0–1) signals, all read from data
the store already holds: dependency count (a reverse walk of the provenance DAG),
contestedness, staleness age, and projection-blocking. All four are live —
``projection_blocking`` went live when the manifest hook was wired, and
contributes nothing while no manifest is configured. Weights live in
``curation.leverage_weights`` (config).

The contestedness signal reads the **composed** badge (all three
bases) since, not the inconsistency basis alone; it is supplied by the
caller off the card collection rather than re-queried.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import CurationConfig, get_config
from particles.core.schema import JudgeVerdictKind
from particles.store.particle_store import count_active_dependents, get_active_particles

from .cards import CardKind, CurationCard

log = logging.getLogger(__name__)

# Projection-blocking hook. Inert until a documentation
# projection manifest registers one — default returns 0.0 so the signal ships
# now and goes live when the hook is wired. No forward dependency in this track.
_PROJECTION_HOOK: Callable[[CurationCard], float] | None = None


def register_projection_hook(hook: Callable[[CurationCard], float] | None) -> None:
    """Wire the projection-blocking signal.

    ``hook(card)`` returns a 0–1 score for whether the card blocks a clean
    documentation-projection section. Passing ``None`` clears it (back to inert).
    """
    global _PROJECTION_HOOK
    _PROJECTION_HOOK = hook


def _projection_blocking(card: CurationCard) -> float:
    if _PROJECTION_HOOK is None:
        return 0.0
    return max(0.0, min(1.0, _PROJECTION_HOOK(card)))


def _as_utc(dt: datetime) -> datetime:
    """Treat a tz-naive timestamp (SQLite round-trips DateTime without tz) as UTC."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def contested_ids_from(cards: list[CurationCard]) -> set[str]:
    """The beliefs the composed badge fired on, read off the card list.

    Every badged belief has a ``CONTESTED`` card, so the finder's
    verdict is already in the collection and needs no second query. Derive this
    from the **unfiltered** collection: the signal applies to cards of every
    kind, so narrowing by ``--kind`` or dropping snoozed cards first would
    silently strip the boost from a stale/duplicate card on a contested belief.
    """
    return {pid for c in cards if c.kind is CardKind.CONTESTED for pid in c.particle_ids}


async def score_cards(
    session: AsyncSession,
    cards: list[CurationCard],
    *,
    contested_ids: set[str] | None = None,
) -> None:
    """Fill each card's ``leverage`` in place.

    Loads the signals once for the whole batch (one dependents aggregation, one
    ACTIVE-particle scan for ages) then scores each card from those — the cost shape (batch-load + in-process math), acceptable at
    memory-store scale (is the scaling lever).

    ``contested_ids`` is the composed-badge set: the signal
    widened from "referenced by an open INCONSISTENCY" to all three of the bases, so "contested" means one thing inside this module rather than two.
    Callers holding the full collection pass :func:`contested_ids_from`, which
    makes the signal free — before this was a second
    ``get_inconsistency_backrefs`` scan on top of the one the finder already
    ran. ``None`` falls back to deriving it from ``cards`` (correct whenever
    ``cards`` is the unfiltered collection).
    """
    cfg = get_config().curation
    w = cfg.leverage_weights

    # wire the projection-blocking signal from the configured
    # manifests before scoring (shipped the hook seam inert for
    # exactly this). A belief that feeds a projected doc gets projection-blocking
    # leverage — closing the projection <-> curation loop.
    #
    # Wired here rather than by the caller since: scoring now happens on
    # two paths (a live build and a persisted collection), and a hook registered
    # on only one of them silently drops the signal from the other — which is
    # exactly the regression the move fixes.
    await register_projection_manifests(session, cfg)

    all_ids = {pid for c in cards for pid in c.particle_ids}
    dependents = await count_active_dependents(session, all_ids)
    contested = contested_ids_from(cards) if contested_ids is None else contested_ids
    asserted_at = {
        p.id: p.asserted_at for p in await get_active_particles(session) if p.id in all_ids
    }

    now = datetime.now(UTC)
    dep_cap = math.log1p(cfg.dependency_norm_cap)
    for card in cards:
        dep = max((dependents.get(pid, 0) for pid in card.particle_ids), default=0)
        dep_norm = min(1.0, math.log1p(dep) / dep_cap) if dep_cap > 0 else 0.0

        is_contested = 1.0 if any(pid in contested for pid in card.particle_ids) else 0.0

        ages = [
            (now - _as_utc(asserted_at[pid])).total_seconds() / 86400.0
            for pid in card.particle_ids
            if pid in asserted_at
        ]
        age_days = max(ages) if ages else 0.0
        age_norm = min(1.0, max(0.0, age_days) / cfg.staleness_norm_days)

        score = (
            w.dependency_count * dep_norm
            + w.contestedness * is_contested
            + w.staleness_age * age_norm
            + w.projection_blocking * _projection_blocking(card)
        )

        # demote a DUPLICATE_PAIR card the LLM judge cleared as DISTINCT
        # (not the same claim) so it sinks toward the bottom — informed by the
        # model's read, not raw cosine. PARAPHRASE / UNSURE / absent verdicts keep
        # the score unchanged. Demote, don't hide: recall is preserved against a
        # wrong LLM clear (hard-suppression is deferred).
        if (
            card.kind is CardKind.DUPLICATE_PAIR
            and card.verdict is not None
            and card.verdict.verdict is JudgeVerdictKind.DISTINCT
        ):
            score *= cfg.duplicate_distinct_demotion

        card.leverage = round(score, 6)


async def register_projection_manifests(session: AsyncSession, cfg: CurationConfig) -> None:
    """Wire the projection-blocking leverage signal from the config.

    Loads each manifest in ``curation.projection_manifests``, unions the ACTIVE
    particle ids its derived sections currently select, and registers a hook that
    scores a card 1.0 when its belief feeds a projected doc. With no manifests
    configured the hook is cleared (the inert default), so the
    projection-blocking weight contributes nothing — a store that does not project
    docs behaves exactly as before.
    """
    if not cfg.projection_manifests:
        register_projection_hook(None)
        return

    # Deferred import: projection pulls the article-synthesis stack; load it only
    # when manifests are actually configured (AGENTS.md deferred-import case 2).
    from particles.operations.projection import load_manifest, required_particle_ids

    required: set[str] = set()
    for raw_path in cfg.projection_manifests:
        path = Path(raw_path)
        if not path.exists():
            log.warning("curation: projection manifest %s not found; skipping", path)
            continue
        try:
            manifest = load_manifest(path)
            required |= await required_particle_ids(session, manifest)
        except Exception as exc:  # noqa: BLE001 — a bad manifest must not break curation
            log.warning("curation: could not load projection manifest %s: %s", path, exc)

    if not required:
        register_projection_hook(None)
        return

    def hook(card: CurationCard) -> float:
        return 1.0 if any(pid in required for pid in card.particle_ids) else 0.0

    register_projection_hook(hook)

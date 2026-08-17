"""Citation-signal deposit suggestions — ranking + dismiss.

Ranks undeposited URL mentions (``particles.store.url_mention_store``) as
operator deposit suggestions. The score is **trust-weighted distinct-source
diversity × recency**, never bare frequency: a URL's score is
the sum, over each *distinct* citing source entry, of that source's query-time
lens-composed trust times a recency decay on when the
citation was seen. Diversity is the floor (a single spammer contributes once);
a single old high-trust citation never vanishes (recency has a floor).

This is a query-time view — nothing here is stored, and the score never feeds
``effective_confidence``: a citation is a curation / amplification signal, not
an endorsement stance. Suggestion-only:
acting on a suggestion is the operator deposit, never an auto-crawl.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.operations.query.source_trust import load_trust_policy
from particles.store.event_store import OperatorEventType, record_event
from particles.store.url_mention_store import (
    get_suppressed_urls,
    list_undeposited_mentions,
    suppress_suggestion,
)

# Far-future sentinel: a permanent dismiss is a suppression that never expires.
_PERMANENT_DISMISS_UNTIL = datetime(9999, 12, 31, tzinfo=UTC)


class DepositSuggestion(BaseModel):
    """One ranked undeposited URL the operator might want to deposit."""

    canonical_url: str
    distinct_sources: int
    score: float
    citing_entry_ids: list[str]
    most_recent: datetime


class DepositSuggestReport(BaseModel):
    """The ranked deposit-suggestion list plus how much was capped."""

    suggestions: list[DepositSuggestion]
    # Distinct undeposited URLs that passed the min-distinct-sources gate
    # (before the rank cap) — so the caller can say "showing N of M".
    total_candidates: int
    capped: bool


def _host(url: str | None) -> str:
    if not url:
        return ""
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def _as_utc(dt: datetime) -> datetime:
    """Treat a tz-naive timestamp (SQLite round-trips DateTime without tz) as UTC."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _recency_factor(age_days: float, half_life_days: float, floor: float) -> float:
    """Exponential decay on a citation's age, never below ``floor``."""
    if age_days <= 0:
        return 1.0
    factor: float = 0.5 ** (age_days / half_life_days)
    return max(floor, factor)


async def suggest_deposits(
    session: AsyncSession,
    *,
    limit: int | None = None,
    min_sources: int | None = None,
) -> DepositSuggestReport:
    """Rank undeposited URL mentions as deposit suggestions.

    ``limit`` / ``min_sources`` default to ``citation_signal.rank_cap`` /
    ``citation_signal.min_distinct_sources``. Suppressed (dismissed / snoozed)
    URLs and — when ``filter_site_internal`` is on — URLs cited only from their
    own host are excluded.
    """
    cfg = get_config().citation_signal
    cap = cfg.rank_cap if limit is None else limit
    floor_sources = cfg.min_distinct_sources if min_sources is None else min_sources

    mentions = await list_undeposited_mentions(session)
    if not mentions:
        return DepositSuggestReport(suggestions=[], total_candidates=0, capped=False)

    suppressed = await get_suppressed_urls(session)

    # url -> {source_entry_id: most-recent discovered_at}. One contribution per
    # distinct source (diversity, not raw frequency).
    by_url: dict[str, dict[str, datetime]] = {}
    for m in mentions:
        if m.canonical_url in suppressed:
            continue
        sources = by_url.setdefault(m.canonical_url, {})
        discovered = _as_utc(m.discovered_at)
        prev = sources.get(m.source_entry_id)
        if prev is None or discovered > prev:
            sources[m.source_entry_id] = discovered

    candidates = {u: s for u, s in by_url.items() if len(s) >= floor_sources}
    if not candidates:
        return DepositSuggestReport(suggestions=[], total_candidates=0, capped=False)

    # Batch-load source metadata once for trust evaluation + the site-internal
    # filter, then evaluate the policy in-process (no per-source round trips).
    from particles.corpus.store import get_entry_uri_map, get_source_types_for_entries

    all_entries = sorted({e for sources in candidates.values() for e in sources})
    policy = await load_trust_policy(session)
    source_types = await get_source_types_for_entries(session, all_entries)
    uri_map = await get_entry_uri_map(session, set(all_entries))

    now = datetime.now(UTC)
    suggestions: list[DepositSuggestion] = []
    for url, sources in candidates.items():
        if cfg.filter_site_internal:
            url_host = _host(url)
            citing_hosts = {_host(uri_map.get(e)) for e in sources} - {""}
            if url_host and citing_hosts and citing_hosts <= {url_host}:
                continue  # cited only by its own site — navigation boilerplate

        score = 0.0
        for entry_id, discovered in sources.items():
            trust = policy.evaluate(entry_id, source_types.get(entry_id, ""), uri_map.get(entry_id))
            trust = 1.0 if trust is None else trust
            age_days = (now - discovered).total_seconds() / 86400.0
            score += trust * _recency_factor(
                age_days, cfg.recency_half_life_days, cfg.recency_floor
            )

        suggestions.append(
            DepositSuggestion(
                canonical_url=url,
                distinct_sources=len(sources),
                score=round(score, 6),
                citing_entry_ids=sorted(sources),
                most_recent=max(sources.values()),
            )
        )

    suggestions.sort(key=lambda s: (-s.score, -s.distinct_sources, s.canonical_url))
    total = len(suggestions)
    capped = total > cap
    return DepositSuggestReport(
        suggestions=suggestions[:cap], total_candidates=total, capped=capped
    )


async def dismiss_suggestion(
    session: AsyncSession,
    *,
    canonical_url: str,
    actor: str,
    snooze_days: int | None = None,
) -> datetime:
    """Dismiss or snooze a deposit suggestion so it stops resurfacing (§ 6).

    ``snooze_days=None`` is a permanent dismiss (far-future suppression); a
    positive ``snooze_days`` suppresses for that many days. Suppression is the
    queryable state; the matching audit event carries the URL and
    window in its payload. Returns the ``suppressed_until`` timestamp; the
    caller commits.
    """
    if snooze_days is not None and snooze_days > 0:
        until = datetime.now(UTC) + timedelta(days=snooze_days)
    else:
        until = _PERMANENT_DISMISS_UNTIL
    await suppress_suggestion(session, canonical_url=canonical_url, until=until)
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.DEPOSIT_SUGGESTION_DISMISSED,
        payload={
            "canonical_url": canonical_url,
            "suppressed_until": until.isoformat(),
            "snooze_days": snooze_days,
        },
    )
    return until

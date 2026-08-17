"""Session model + gesture dispatch (§4).

``build_curation_queue`` is the public entry point: it collects the cards (§1),
scores them (§2), filters out snoozed / affirmed cards, ranks by leverage, and
returns the finite "today's N". ``apply_gesture`` dispatches a card's gesture
onto an **existing** write op — the surface is new, the writes are not.

Snooze / affirm are recorded in the operator event log keyed by the
card's stable ``key``; there is no new suppression table. ``uncited_url`` cards
reuse the existing deposit-suggestion suppression instead (``suggest_deposits``
already excludes those). Undo is the existing event log + the reversible §6.6
status machine — no bespoke undo stack.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import CurationConfig, get_config
from particles.core.schema import Particle, RelationCreatedBy, RelationType
from particles.core.status import Status, StatusReason
from particles.operations.abstraction import accept_candidate, reject_candidate
from particles.operations.query.effective_confidence import score_effective_confidence
from particles.store.curation_snapshot_store import get_snapshot, latest_snapshot
from particles.store.event_store import (
    EventRefKind,
    OperatorEventType,
    list_events,
    list_events_since,
    record_event,
)
from particles.store.particle_store import get_particle, update_particle_status
from particles.store.subject_store import get_subject

from .cards import CardKind, CurationCard, ParticleBrief
from .collect import collect_cards
from .leverage import _as_utc, contested_ids_from, score_cards
from .snapshot import (
    CurationQueueResult,
    QueueSource,
    collect_and_persist,
    load_collection,
    stamp_from,
)

log = logging.getLogger(__name__)

# The gestures whose completion resolves a curation card without necessarily
# changing a belief's status (level 3). Every one of these is
# already recorded by the write op the gesture dispatches onto — this set is
# the read side, not a new obligation.
#
# Deliberately excludes the purely-informational events (BELIEF_MARKED_USEFUL,
# CONSOLIDATION_RUN, PARTICLE_TAGGED): marking a belief useful or tagging it
# does not resolve the finding a card reports, and suppressing on those would
# hide real work.
_RESOLVING_EVENTS: frozenset[OperatorEventType] = frozenset(
    {
        OperatorEventType.PARTICLE_RETRACTED,
        OperatorEventType.PARTICLE_SUPERSEDED,
        OperatorEventType.RELATION_ADDED,
        OperatorEventType.DUPLICATES_MERGED,
        OperatorEventType.REVIEW_RESOLVED,
        OperatorEventType.ABSTRACTION_RESOLVED,
        OperatorEventType.SUBJECT_LINK_CONFIRMED,
        OperatorEventType.DEPOSIT_SUGGESTION_DISMISSED,
        OperatorEventType.SOURCE_RETRACTED,
    }
)

# Tiebreak within equal leverage — more urgent kinds first. The
# leverage score is always primary; this only orders cards that tie (e.g. the
# zero-signal batch / URL cards among themselves).
_KIND_PRIORITY: dict[CardKind, int] = {
    CardKind.CONTRADICTION: 0,
    CardKind.CONTESTED: 1,
    CardKind.RETRACTION_CASCADE: 2,
    CardKind.BROKEN_PROVENANCE: 3,
    CardKind.NO_SUBJECT: 4,
    CardKind.STALE: 5,
    CardKind.CONFIDENCE_DECAY: 6,
    CardKind.RECENCY_DECAY: 7,
    # a pending abstraction candidate outranks the housekeeping tail
    # — an operator verdict here changes the projection's population.
    CardKind.PROPOSED_ABSTRACTION: 8,
    CardKind.FAILED_SNAPSHOTS: 9,
    CardKind.DUPLICATE_PAIR: 10,
    CardKind.UNCITED_URL: 11,
}

# Session-model gestures available on every card regardless of kind.
_SESSION_GESTURES = frozenset({"snooze", "dismiss"})

# Gestures that need operator content or judgment — surfaced (the card shows the
# resolving command) rather than dispatched from a one-line gesture.
_SURFACED: dict[str, str] = {
    "supersede": "supersession is operator-authored — assert the replacement, then "
    "retire the old belief",
    "edit": "an edit is a supersession — assert the revised belief",
    "comment": "run `particles review` to annotate / resolve the INCONSISTENCY",
    "reindex": "run `particles reindex` to re-extract the failed snapshots",
}


async def build_curation_queue(
    session: AsyncSession,
    *,
    limit: int | None = None,
    kind: CardKind | None = None,
    semantic: bool | None = None,
    cards: list[CurationCard] | None = None,
    source: QueueSource = QueueSource.SNAPSHOT,
) -> CurationQueueResult:
    """Return the leverage-ranked, finite, snooze-filtered queue.

    Since the expensive *collection* half is served from a persisted
    snapshot and only the cheap *session* half — suppression, belief-status
    re-validation, post-snapshot event suppression, the top-N slice and briefs
    — runs live. The result carries the §5 staleness stamp so a consumer can
    say "as of 03:30" rather than implying freshness it does not have.

    Args:
        session: open DB session.
        limit: cap the returned cards; defaults to ``curation.session_size``.
        kind: restrict to a single ``CardKind``.
        semantic: run the LLM-assisted finders; defaults to ``curation.semantic``.
            Only consulted when a collection is actually built.
        cards: an already-collected card list to rank instead of reading a
            snapshot or running ``collect_cards`` — the pass-4
            reuse, so one (LLM-priced) collection feeds both the census and the
            queue. ``semantic`` is ignored when supplied; the input list is not
            mutated, and no snapshot is read or written.
        source: ``SNAPSHOT`` (default) serves the newest stored collection,
            falling back to a live collection when the store has none — this
            path never writes the cache. ``LIVE`` forces the
            finders to run for this request regardless — what ``--no-snapshot``
            and ``curation.snapshot_enabled: false`` select.
    """
    cfg = get_config().curation
    use_semantic = cfg.semantic if semantic is None else semantic
    collection, stamp = await _collection_for(
        session, cards=cards, semantic=use_semantic, source=source, cfg=cfg
    )

    # the contestedness leverage signal applies to cards of every
    # kind, so it is read off the collection *before* any narrowing.
    if stamp.source == QueueSource.LIVE.value:
        contested_ids = contested_ids_from(collection)
        if kind is not None:
            collection = [c for c in collection if c.kind is kind]
        await score_cards(session, collection, contested_ids=contested_ids)
    elif kind is not None:
        # A snapshot's cards are already scored, so narrowing is a plain filter.
        collection = [c for c in collection if c.kind is kind]

    stamp.collection_size = len(collection)

    # --- live on every request ------------------------------
    # Level 1: snooze / affirm / dismiss. Free, and never stale.
    suppressed = await _suppressed_keys(session)
    # Level 3: gesture resolutions that are not status transitions — a merged
    # duplicate, an assigned subject, a deposited URL. Every such gesture
    # records an operator event, so anything touched after the collection was
    # built is treated as resolved. Skipped on a live build (nothing predates
    # it).
    resolved_ids: set[str] = set()
    if stamp.built_at is not None:
        resolved_keys, resolved_ids = await _resolved_since(session, stamp.built_at)
        suppressed |= resolved_keys
    collection = [
        c
        for c in collection
        if c.key not in suppressed and not any(pid in resolved_ids for pid in c.particle_ids)
    ]

    collection.sort(key=lambda c: (-c.leverage, _KIND_PRIORITY.get(c.kind, 99), c.key))
    size = cfg.session_size if limit is None else limit
    # Level 2: belief status. Walked over the ranked list rather than the whole
    # collection, so a dropped card promotes the next real one instead of
    # shortening the session — at a bounded cost of (size + dropped) lookups.
    result = await _take_live_cards(session, collection, size)
    await _attach_particle_briefs(session, result)

    stamp.cards = result
    stamp.count = len(result)
    return stamp


async def _collection_for(
    session: AsyncSession,
    *,
    cards: list[CurationCard] | None,
    semantic: bool,
    source: QueueSource,
    cfg: CurationConfig,
) -> tuple[list[CurationCard], CurationQueueResult]:
    """Resolve the collection to rank, plus its staleness stamp."""
    if cards is not None:
        # Caller-supplied (pass 4): rank exactly what we were given.
        return list(cards), CurationQueueResult(source=QueueSource.LIVE.value, semantic=semantic)

    if source is QueueSource.LIVE or not cfg.snapshot_enabled:
        collected = await collect_cards(session, semantic=semantic)
        return collected, CurationQueueResult(source=QueueSource.LIVE.value, semantic=semantic)

    row = await latest_snapshot(session)
    if row is not None:
        return load_collection(row), stamp_from(row)

    # Cold start: no snapshot yet, so collect live and serve
    # that — correct, just slow, exactly as before this ADR.
    #
    # **This path deliberately does not write.** Populating the cache here would
    # make a read verb commit, and a read that commits is a surprise: it would
    # end the caller's transaction under them (`GET /curation` holds a plain
    # `SessionDep`, and the CLI a `session_scope`), turning an incidental cache
    # fill into a durability boundary neither caller asked for. The cache is
    # filled by the two paths that are already writes — the nightly
    # cycle, and the explicit `--refresh` / `POST /curation/rebuild` — so the
    # read path stays a read.
    collected = await collect_cards(session, semantic=semantic)
    return collected, CurationQueueResult(source=QueueSource.LIVE.value, semantic=semantic)


async def rebuild_curation_snapshot(
    session: AsyncSession, *, semantic: bool | None = None
) -> CurationQueueResult:
    """Force a store-wide rebuild of the persisted collection.

    The operator's explicit refresh — `POST /curation/rebuild` and
    `particles curate --refresh`. Synchronous by design: it does not make the
    queue fast, and the surfaces that call it already have an honest loading
    state, so a job runtime buys nothing here.

    Returns the new stamp with no cards attached; the caller re-reads the queue.
    Commits, because the snapshot is the point of the call.
    """
    cfg = get_config().curation
    use_semantic = cfg.semantic if semantic is None else semantic
    merged, snapshot_id = await collect_and_persist(session, semantic=use_semantic)
    await session.commit()
    row = await get_snapshot(session, snapshot_id)
    if row is None:  # pragma: no cover — just written
        return CurationQueueResult(source=QueueSource.LIVE.value, semantic=use_semantic)
    stamp = stamp_from(row)
    stamp.collection_size = len(merged)
    return stamp


async def _resolved_since(session: AsyncSession, built_at: datetime) -> tuple[set[str], set[str]]:
    """What an operator resolved after the collection was built.

    Level 3 of the staleness ladder. A snapshot cannot know about a merge, a
    subject assignment or a deposit that happened at 09:00 — but the operator
    event log does, because every resolving gesture records one. Returns
    ``(card_keys, particle_ids)``: URL cards suppress by key (they carry no
    particle, and reuse the deposit-suggestion path already built for them
    ), while belief cards suppress by *membership* — a card is keyed by
    kind plus its sorted particle ids, so the key cannot be reconstructed from
    an event ref alone, but "does this card name a touched belief?" answers the
    same question for every kind at once.

    Deliberately over-suppresses rather than under-suppresses: a belief touched
    for an unrelated reason costs the operator one card that reappears after the
    next build, whereas showing a card the operator already handled is the exact
    failure this section exists to prevent.
    """
    keys: set[str] = set()
    touched: set[str] = set()
    for ev in await list_events_since(session, since=built_at, event_types=_RESOLVING_EVENTS):
        for ref in ev.refs:
            if ref.ref_kind is EventRefKind.PARTICLE:
                touched.add(ref.ref_id)
        url = (ev.payload or {}).get("canonical_url")
        if isinstance(url, str):
            keys.add(f"uncited_url:{url}")
    return keys, touched


async def _take_live_cards(
    session: AsyncSession, ranked: list[CurationCard], size: int
) -> list[CurationCard]:
    """The top ``size`` cards whose beliefs are still ACTIVE (level 2).

    A snapshot can name a belief that was retracted or superseded since it was
    built. Rather than serving a card the operator already resolved (or worse,
    one whose gesture would now fail), each candidate's particles are checked
    as the list is walked, and a dropped card promotes the next one. Cards with
    no particles (``uncited_url`` / ``failed_snapshots``) pass through.
    """
    out: list[CurationCard] = []
    for card in ranked:
        if len(out) >= size:
            break
        if not card.particle_ids:
            out.append(card)
            continue
        alive = True
        for pid in card.particle_ids:
            target = await get_particle(session, pid)
            if target is None or target.status is not Status.ACTIVE:
                alive = False
                break
        if alive:
            out.append(card)
    return out


async def _attach_particle_briefs(session: AsyncSession, cards: list[CurationCard]) -> None:
    """Populate each card's ``particles`` with a compact brief.

    One pass over the union of every (already-sliced) card's ``particle_ids`` —
    each referenced particle and subject is loaded once — so a client can judge a
    card (e.g. which of a duplicate pair to keep) without a per-id
    ``particles particle show`` round-trip. ``effective_confidence`` is scored exactly as
    the query path does (``score_effective_confidence``), so the feed and a query
    cannot disagree. Cards with no particle (uncited_url / failed_snapshots) keep
    the empty default.
    """
    ids = {pid for c in cards for pid in c.particle_ids}
    if not ids:
        return

    particles: dict[str, Particle] = {}
    for pid in ids:
        p = await get_particle(session, pid)
        if p is not None:
            particles[pid] = p
    if not particles:
        return

    eff = await score_effective_confidence(session, list(particles.values()), populate_cache=True)
    labels: dict[str, str] = {}
    for p in particles.values():
        for sid in p.subject_ids:
            if sid not in labels:
                subject = await get_subject(session, sid)
                if subject is not None:
                    labels[sid] = subject.canonical_name

    for card in cards:
        card.particles = [
            ParticleBrief(
                particle_id=pid,
                content=p.content,
                subject_labels=[labels[s] for s in p.subject_ids if s in labels],
                effective_confidence=eff.get(pid, p.confidence.value),
                status=p.status.value,
            )
            for pid in card.particle_ids
            if (p := particles.get(pid)) is not None
        ]


async def _suppressed_keys(session: AsyncSession) -> set[str]:
    """The card keys hidden by an unexpired snooze or a standing affirmation."""
    now = datetime.now(UTC)
    suppressed: set[str] = set()

    for ev in await list_events(
        session, event_type=OperatorEventType.CURATION_CARD_SNOOZED, limit=10_000
    ):
        payload = ev.payload or {}
        key = payload.get("card_key")
        if not isinstance(key, str):
            continue
        until_raw = payload.get("snoozed_until")
        if until_raw is None:  # permanent dismiss
            suppressed.add(key)
            continue
        try:
            until = datetime.fromisoformat(str(until_raw))
        except ValueError:
            continue
        if _as_utc(until) > now:
            suppressed.add(key)

    for ev in await list_events(
        session, event_type=OperatorEventType.BELIEF_AFFIRMED, limit=10_000
    ):
        key = (ev.payload or {}).get("card_key")
        if isinstance(key, str):
            suppressed.add(key)

    return suppressed


def _particle_refs(card: CurationCard) -> list[tuple[EventRefKind, str]]:
    return [(EventRefKind.PARTICLE, pid) for pid in card.particle_ids]


async def apply_gesture(
    session: AsyncSession,
    card: CurationCard,
    gesture: str,
    *,
    store: str = "default",
    actor: str = "curate",
    reason: str | None = None,
    days: int | None = None,
    subject: str | None = None,
) -> str:
    """Dispatch a card's gesture onto an existing write op.

    Executes the safe, card-resolvable gestures (affirm / snooze / dismiss /
    retract / merge / deposit / assign-subject); the content- or judgment-bearing
    gestures (supersede / edit / comment / reindex) are *surfaced* with the
    resolving command rather than dispatched. ``subject`` carries the resolved
    Subject id or a subject name for the ``assign-subject`` gesture.
    Does not commit — the caller owns the transaction. Returns a human-readable
    result line.
    """
    g = gesture.lower()
    if g not in _SESSION_GESTURES and g not in card.suggested_gestures:
        offered = ", ".join(card.suggested_gestures)
        raise ValueError(
            f"Card {card.kind.value} does not offer gesture {g!r} (offered: {offered})."
        )

    if g in _SURFACED:
        raise ValueError(f"Gesture {g!r}: {_SURFACED[g]}.")

    match g:
        case "affirm":
            await record_event(
                session,
                actor=actor,
                event_type=OperatorEventType.BELIEF_AFFIRMED,
                refs=_particle_refs(card),
                payload={"card_key": card.key, "kind": card.kind.value},
            )
            return f"Affirmed — {card.key} will not resurface."

        case "snooze":
            return await _snooze(session, card, actor=actor, days=days, permanent=False)

        case "dismiss":
            return await _snooze(session, card, actor=actor, days=days, permanent=days is None)

        case "retract":
            return await _retract(session, card, actor=actor, reason=reason)

        case "merge":
            return await _merge(session, card)

        case "deposit":
            return await _deposit(session, card, actor=actor)

        case "assign-subject":
            return await _assign_subject(session, card, actor=actor, store=store, subject=subject)

        case "accept":
            return await _accept_abstraction(session, card, actor=actor)

        case "reject":
            return await _reject_abstraction(session, card, actor=actor, reason=reason)

    raise ValueError(f"Unknown gesture {g!r}.")


async def _accept_abstraction(session: AsyncSession, card: CurationCard, *, actor: str) -> str:
    """assert a proposed abstraction from its candidate event."""
    if card.candidate_event_id is None:
        raise ValueError("Card carries no candidate event id.")
    particle = await accept_candidate(session, card.candidate_event_id, actor=actor)
    return (
        f"Accepted — asserted derived belief {particle.id[:8]}… with "
        f"{len(particle.provenance)} premise link(s)."
    )


async def _reject_abstraction(
    session: AsyncSession, card: CurationCard, *, actor: str, reason: str | None
) -> str:
    """record the rejection (a labelled §8 datapoint); no store write."""
    if card.candidate_event_id is None:
        raise ValueError("Card carries no candidate event id.")
    await reject_candidate(session, card.candidate_event_id, actor=actor, reason=reason)
    return "Rejected — the candidate will not resurface (recorded for evaluation)."


async def _snooze(
    session: AsyncSession,
    card: CurationCard,
    *,
    actor: str,
    days: int | None,
    permanent: bool,
) -> str:
    """Suppress a card. URL cards reuse the deposit-suggestion path."""
    if card.kind is CardKind.UNCITED_URL and card.corpus_url is not None:
        from particles.operations.deposit_suggest import dismiss_suggestion

        snooze_days = None if permanent else (days or get_config().curation.snooze_days)
        await dismiss_suggestion(
            session, canonical_url=card.corpus_url, actor=actor, snooze_days=snooze_days
        )
        return f"{'Dismissed' if permanent else 'Snoozed'} {card.corpus_url}."

    snooze_days = None if permanent else (days or get_config().curation.snooze_days)
    until = None if snooze_days is None else datetime.now(UTC) + timedelta(days=snooze_days)
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.CURATION_CARD_SNOOZED,
        refs=_particle_refs(card),
        payload={
            "card_key": card.key,
            "snoozed_until": None if until is None else until.isoformat(),
            "snooze_days": snooze_days,
        },
    )
    if permanent:
        return f"Dismissed — {card.key} will not resurface."
    return f"Snoozed {card.key} for {snooze_days} day(s)."


async def _retract(
    session: AsyncSession, card: CurationCard, *, actor: str, reason: str | None
) -> str:
    """Retract a single belief via the operator status path."""
    if len(card.particle_ids) != 1:
        raise ValueError("retract resolves a single-belief card.")
    pid = card.particle_ids[0]
    target = await get_particle(session, pid)
    if target is None:
        raise ValueError(f"Particle {pid!r} not found.")
    if target.status is not Status.ACTIVE:
        raise ValueError(f"Particle {pid!r} is {target.status.value}, not ACTIVE.")
    await update_particle_status(session, pid, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION)
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.PARTICLE_RETRACTED,
        reason=reason,
        refs=[(EventRefKind.PARTICLE, pid)],
        payload={"via": "curate"},
    )
    return f"Retracted {pid[:8]}…"


async def _assign_subject(
    session: AsyncSession,
    card: CurationCard,
    *,
    actor: str,
    store: str,
    subject: str | None,
) -> str:
    """Assign a subject to a NO_SUBJECT orphan via the operator-supersede.

    ``subject`` is an existing Subject id (linked directly) or a subject name run
    through the standard resolver. The orphan is superseded by a successor with
    the same content + the resolved subject, in provenance-carry-over mode.
    """
    if card.kind is not CardKind.NO_SUBJECT or len(card.particle_ids) != 1:
        raise ValueError("assign-subject resolves a no_subject card.")
    if not subject or not subject.strip():
        raise ValueError("assign-subject requires a subject (id or name) via --subject.")

    # Deferred import: the operator-supersede primitive lives in agent_write,
    # which pulls the ingest reconciliation stack — load it only on dispatch.
    from particles.operations.agent_write import assign_subject_belief

    sval = subject.strip()
    existing = await get_subject(session, sval)
    pid = card.particle_ids[0]
    result = await assign_subject_belief(
        session,
        store=store,
        particle_id=pid,
        subject_id=sval if existing is not None else None,
        subject_name=None if existing is not None else sval,
        actor=actor,
    )
    return f"Assigned subject to {pid[:8]}… → successor {(result.asserted_particle_id or '?')[:8]}…"


async def _merge(session: AsyncSession, card: CurationCard) -> str:
    """Link a duplicate pair CO_EVIDENTIAL via the operator path."""
    if card.kind is not CardKind.DUPLICATE_PAIR or len(card.particle_ids) != 2:
        raise ValueError("merge resolves a duplicate_pair card.")
    from particles.store.relation_store import create_relation

    a, b = card.particle_ids
    await create_relation(session, a, b, RelationType.CO_EVIDENTIAL, RelationCreatedBy.MANUAL_CLI)
    return f"Linked {a[:8]}… ↔ {b[:8]}… as co-evidential."


async def _deposit(session: AsyncSession, card: CurationCard, *, actor: str) -> str:
    """Deposit an uncited URL via the existing deposit op."""
    if card.corpus_url is None:
        raise ValueError("deposit resolves an uncited_url card.")
    from particles.operations.deposit import deposit_url
    from particles.operations.deposit_suggest import dismiss_suggestion

    entry_id, _snapshot_id = await deposit_url(session, card.corpus_url, deposited_by=actor)
    # Retire the suggestion permanently: it is resolved, not merely skipped.
    # `suggest_deposits` already excludes deposited URLs, so this changes
    # nothing for a live collection — but a persisted one has no
    # other way to learn the card was handled, and the recorded event is what
    # the §5 level-3 filter reads. Reuses the URL-card suppression path
    # already assigns to this kind.
    await dismiss_suggestion(session, canonical_url=card.corpus_url, actor=actor, snooze_days=None)
    return f"Deposited {card.corpus_url} → entry {entry_id[:8]}…"

# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import CurationConfig, get_config
from particles.core.schema import Particle, RelationCreatedBy, RelationType, ResolutionAction
from particles.core.status import Status, StatusReason
from particles.corpus.store import get_particle_source_uris
from particles.operations.abstraction import accept_candidate, reject_candidate
from particles.operations.lint.open_inconsistency import (
    open_inconsistency_findings,
    record_sides,
)
from particles.operations.query.effective_confidence import score_effective_confidence
from particles.operations.review import resolve
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

from .cards import (
    RESOLVE_ACTIONS,
    CardKind,
    ConflictBrief,
    CurationCard,
    ParticleBrief,
    resolve_actions_for,
)
from .collect import cards_from_findings, collect_cards
from .leverage import _as_utc, contested_ids_from, score_cards
from .rulings import record_demotion_ruling
from .snapshot import (
    CurationQueueResult,
    QueueSource,
    Resolved,
    collect_and_persist,
    lacks_conflict_cards,
    read_prior,
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
        # an in-place relink (a per-card assign-subject, or the batch).
        OperatorEventType.SUBJECTS_RELINKED,
    }
)

# A batch card's key; it is resolved by a batch relink event, never by
# membership, since touching one of thousands of beliefs for an
# unrelated reason must not hide the whole card.
_GATED_KEY = "gated_subjects"

# How many of a batch card's beliefs get a brief: the card is judged as one
# decision, and briefing thousands of beliefs would dominate the queue's cost.
_BATCH_BRIEF_SAMPLE = 5

# Tiebreak within equal leverage — more urgent kinds first. The
# leverage score is always primary; this only orders cards that tie (e.g. the
# zero-signal batch / URL cards among themselves).
_KIND_PRIORITY: dict[CardKind, int] = {
    CardKind.CONTRADICTION: 0,
    CardKind.CONTESTED: 1,
    # the conflict the store has recorded, beside the claims it touches.
    CardKind.INCONSISTENCY: 1,
    CardKind.RETRACTION_CASCADE: 2,
    # the same question as a retraction cascade, asked of an update.
    CardKind.STALE_BASIS: 2,
    CardKind.BROKEN_PROVENANCE: 3,
    CardKind.NO_SUBJECT: 4,
    CardKind.GATED_SUBJECTS: 4,
    CardKind.STALE: 5,
    CardKind.CONFIDENCE_DECAY: 6,
    CardKind.RECENCY_DECAY: 7,
    # a pending abstraction candidate outranks the housekeeping tail
    # — an operator verdict here changes the projection's population.
    CardKind.PROPOSED_ABSTRACTION: 8,
    # a ruling on a retirement already made; no belief is degraded
    # while it waits, so it sits with the housekeeping tail.
    CardKind.DEMOTION: 9,
    CardKind.FAILED_SNAPSHOTS: 9,
    CardKind.DUPLICATE_PAIR: 10,
    CardKind.UNCITED_URL: 11,
}

# Session-model gestures available on every card regardless of kind.
_SESSION_GESTURES = frozenset({"snooze", "dismiss"})

# Gestures resolved by a different verb — surfaced (the card shows the resolving
# command) rather than dispatched. ``supersede`` left this set: its
# content arrives as a ``BeliefRevision``, so it dispatches like the others.
_SURFACED: dict[str, str] = {
    "edit": "an edit is a supersession — run `particles curate apply supersede KEY "
    "--content TEXT --reason TEXT --confidence F`",
    "reindex": "run `particles reindex` to re-extract the failed snapshots",
}


@dataclass(frozen=True)
class BeliefRevision:
    """The operator's replacement belief for the ``supersede`` gesture.

    ``subjects`` empty means inherit the predecessor's subjects by id; otherwise
    it replaces them, each value an existing Subject id or a name for the
    standard resolver. With neither ``source_excerpt`` nor ``corpus_entry_id``
    the gesture's reason is deposited as the successor's source (§4).
    ``confidence`` has no default (§3).
    """

    content: str
    confidence: float
    subjects: list[str] = field(default_factory=list)
    source_excerpt: str | None = None
    corpus_entry_id: str | None = None


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
    resolved = Resolved()
    if stamp.built_at is not None:
        resolved = await _resolved_since(session, stamp.built_at)
    collection = [c for c in collection if c.key not in suppressed and not resolved.covers(c)]

    collection.sort(key=lambda c: (-c.leverage, _KIND_PRIORITY.get(c.kind, 99), c.key))
    stamp.open_count = len(collection)
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
        prior = read_prior(row)
        stored = list(prior.cards)
        if lacks_conflict_cards(row):
            # a collection built before open conflicts had their own
            # cards is served with them added from the cheap status scan, until
            # the next build writes the current format.
            conflicts = cards_from_findings(await open_inconsistency_findings(session))
            await score_cards(session, conflicts, contested_ids=contested_ids_from(conflicts))
            stored += conflicts
        return stored, stamp_from(row, kind_as_of=prior.kind_as_of)

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


async def _resolved_since(session: AsyncSession, built_at: datetime) -> Resolved:
    """What an operator resolved after the collection was built.

    Level 3 of the staleness ladder. A snapshot cannot know about a merge, a
    subject assignment or a deposit that happened at 09:00 — but the operator
    event log does, because every resolving gesture records one. URL cards
    suppress by key (they carry no particle, and reuse the deposit-suggestion
    path already built for them), while belief cards suppress by
    *membership* — a card is keyed by kind plus its sorted particle ids, so the
    key cannot be reconstructed from an event ref alone, but "does this card
    name a touched belief?" answers the same question for every kind at once.
    Conflict cards are the exception, matched by record (``Resolved.covers``).

    Deliberately over-suppresses rather than under-suppresses: a belief touched
    for an unrelated reason costs the operator one card that reappears after the
    next build, whereas showing a card the operator already handled is the exact
    failure this section exists to prevent.
    """
    return Resolved.union(r for _, r in await _resolutions_since(session, built_at))


async def _resolutions_since(
    session: AsyncSession, since: datetime
) -> list[tuple[datetime, Resolved]]:
    """Each resolving event after ``since``, with when it happened.

    The per-event form of :func:`_resolved_since`: the carry-forward rule
    needs each event's time, to drop only a carried card that a
    resolution postdates.
    """
    out: list[tuple[datetime, Resolved]] = []
    for ev in await list_events_since(session, since=since, event_types=_RESOLVING_EVENTS):
        refs = [ref.ref_id for ref in ev.refs if ref.ref_kind is EventRefKind.PARTICLE]
        payload = ev.payload or {}
        records: frozenset[str] = frozenset()
        if ev.event_type is OperatorEventType.REVIEW_RESOLVED and payload.get("action") != "DEFER":
            # The record is the event's first ref; members are never records,
            # so membership in the whole ref set identifies it.
            records = frozenset(refs)
        keys: set[str] = set()
        url = payload.get("canonical_url")
        if isinstance(url, str):
            keys.add(f"uncited_url:{url}")
        if ev.event_type is OperatorEventType.SUBJECTS_RELINKED and payload.get("batch"):
            keys.add(_GATED_KEY)
        out.append((_as_utc(ev.occurred_at), Resolved(frozenset(keys), frozenset(refs), records)))
    return out


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
        if card.kind is CardKind.INCONSISTENCY:
            # live while the record is open, whatever its members'
            # statuses. Reading the record's own status catches every closing
            # path (review, the second-reading close, a lapse).
            record = await get_particle(session, card.inconsistency_id or "")
            if record is not None and record.status is Status.INCONSISTENCY:
                out.append(card)
            continue
        if card.kind is CardKind.DEMOTION:
            # the retired claim is never ACTIVE. The card is live
            # while the retirement stands and its replacement is ACTIVE.
            if await _demotion_live(session, card):
                out.append(card)
            continue
        if not card.particle_ids or card.kind is CardKind.GATED_SUBJECTS:
            # A batch card's gesture re-plans from the store, so a member that
            # changed since the build is skipped then, not checked one by one.
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


async def _demotion_live(session: AsyncSession, card: CurationCard) -> bool:
    """A demotion card is live while its claims are still one retired, one ACTIVE."""
    if len(card.particle_ids) != 2:
        return False
    retired = await get_particle(session, card.particle_ids[0])
    replacement = await get_particle(session, card.particle_ids[1])
    return (
        retired is not None
        and replacement is not None
        and retired.status is not Status.ACTIVE
        and replacement.status is Status.ACTIVE
    )


async def _attach_particle_briefs(session: AsyncSession, cards: list[CurationCard]) -> None:
    """Populate each card's ``particles`` (and ``conflict``) with compact briefs.

    One pass over the union of every (already-sliced) card's ``particle_ids``
    plus, for a card backed by an INCONSISTENCY, the record's claims A and B —
    each referenced particle and subject is loaded once — so a client can judge
    a card (which of a duplicate pair to keep, which side of a conflict is
    right) without a per-id ``particles particle show`` round-trip.
    ``effective_confidence`` is scored exactly as the query path does
    (``score_effective_confidence``), so the feed and a query cannot disagree.
    Cards with no particle (uncited_url / failed_snapshots) keep the empty
    default.
    """
    sides: dict[str, tuple[list[str], list[str]]] = {}
    for card in cards:
        if card.inconsistency_id and card.inconsistency_id not in sides:
            record = await get_particle(session, card.inconsistency_id)
            if record is not None:
                sides[card.inconsistency_id] = record_sides(record)

    ids = {pid for c in cards for pid in _brief_ids(c)}
    ids |= {pid for a, b in sides.values() for pid in a + b}
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
    sources = await get_particle_source_uris(session, list(particles.values()))
    labels: dict[str, str] = {}
    for p in particles.values():
        for sid in p.subject_ids:
            if sid not in labels:
                subject = await get_subject(session, sid)
                if subject is not None:
                    labels[sid] = subject.canonical_name

    def brief(pid: str | None) -> ParticleBrief | None:
        p = particles.get(pid) if pid is not None else None
        if p is None:
            return None
        return ParticleBrief(
            particle_id=p.id,
            content=p.content,
            subject_labels=[labels[s] for s in p.subject_ids if s in labels],
            effective_confidence=eff.get(p.id, p.confidence.value),
            status=p.status.value,
            status_reason=p.status_reason.value if p.status_reason is not None else None,
            source_uri=sources.get(p.id),
            asserted_at=p.asserted_at,
        )

    for card in cards:
        card.particles = [b for pid in _brief_ids(card) if (b := brief(pid)) is not None]
        if card.inconsistency_id is None or card.inconsistency_id not in sides:
            continue
        a_side, b_side = sides[card.inconsistency_id]
        a_id = a_side[0] if a_side else None
        b_id = b_side[0] if b_side else None
        card.conflict = ConflictBrief(
            inconsistency_id=card.inconsistency_id,
            a=brief(a_id),
            b=brief(b_id),
            further_a=[x for pid in a_side[1:] if (x := brief(pid)) is not None],
            further_b=[x for pid in b_side[1:] if (x := brief(pid)) is not None],
        )
        if card.kind is CardKind.INCONSISTENCY:
            # offer only the actions that can do what they say.
            card.resolve_actions = resolve_actions_for(
                particles.get(a_id) if a_id else None,
                particles.get(b_id) if b_id else None,
            )


def _brief_ids(card: CurationCard) -> list[str]:
    """The beliefs a card is briefed with: all of them, or a batch card's sample."""
    if card.kind is CardKind.GATED_SUBJECTS:
        return card.particle_ids[:_BATCH_BRIEF_SAMPLE]
    return card.particle_ids


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
    ids = list(card.particle_ids)
    if card.kind is CardKind.INCONSISTENCY and card.inconsistency_id:
        # A conflict card's subject is its record.
        ids = [card.inconsistency_id, *(pid for pid in ids if pid != card.inconsistency_id)]
    return [(EventRefKind.PARTICLE, pid) for pid in ids]


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
    revision: BeliefRevision | None = None,
    action: str | None = None,
    note: str | None = None,
) -> str:
    """Dispatch a card's gesture onto an existing write op.

    Executes affirm / snooze / dismiss / retract / merge / deposit /
    assign-subject / accept / reject, supersede from the operator's
    ``revision``, and resolve with the review ``action`` and an
    optional ``note``; the gestures another verb resolves (edit / reindex) are
    *surfaced* with the resolving command rather than dispatched. ``subject``
    carries the resolved Subject id or a subject name for the
    ``assign-subject`` gesture. Does not commit — the caller owns
    the transaction, except that ``resolve`` runs ``review.resolve``, which
    commits its own work. Returns a human-readable result line.
    """
    g = gesture.lower()
    if card.kind is CardKind.INCONSISTENCY:
        if g == "comment":
            # The resolving gesture's earlier name, kept as an alias.
            g = "resolve"
        if g in ("affirm", "dismiss"):
            raise ValueError(
                f"An open conflict cannot be hidden for good while it stays open ({g}). "
                "Resolve it instead: BOTH_VALID if the claims do not really conflict, "
                "DISCARD if neither is worth keeping. Snooze it to decide later."
            )
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
            line = f"Affirmed — {card.key} will not resurface."
            return await _with_ruling(session, card, g, line, actor=actor, store=store)

        case "snooze":
            return await _snooze(session, card, actor=actor, days=days, permanent=False)

        case "dismiss":
            line = await _snooze(session, card, actor=actor, days=days, permanent=days is None)
            return await _with_ruling(session, card, g, line, actor=actor, store=store)

        case "retract":
            return await _retract(session, card, actor=actor, reason=reason)

        case "merge":
            return await _merge(session, card)

        case "deposit":
            return await _deposit(session, card, actor=actor)

        case "assign-subject":
            return await _assign_subject(session, card, actor=actor, store=store, subject=subject)

        case "relink":
            return await _relink(session, card, actor=actor)

        case "supersede":
            return await _supersede(
                session, card, actor=actor, store=store, reason=reason, revision=revision
            )

        case "accept":
            return await _accept_abstraction(session, card, actor=actor)

        case "reject":
            return await _reject_abstraction(session, card, actor=actor, reason=reason)

        case "resolve":
            return await _resolve(session, card, actor=actor, action=action, note=note)

    raise ValueError(f"Unknown gesture {g!r}.")


async def _with_ruling(
    session: AsyncSession,
    card: CurationCard,
    gesture: str,
    line: str,
    *,
    actor: str,
    store: str,
) -> str:
    """``line``, plus the disclosure when the gesture ruled on a demotion.

    The ruling is kept as a labelled benchmark pair; the gesture's meaning is
    unchanged and no status moves.
    """
    disclosure = await record_demotion_ruling(session, card, gesture, actor=actor, store=store)
    return line if disclosure is None else f"{line} {disclosure}"


async def _resolve(
    session: AsyncSession,
    card: CurationCard,
    *,
    actor: str,
    action: str | None,
    note: str | None,
) -> str:
    """Resolve a conflict card's record through ``review.resolve``.

    Refuses an action the card withholds (``resolve_actions_for``); DEFER is
    always accepted, since it only records a note and leaves the record open.
    """
    if card.inconsistency_id is None:
        raise ValueError("Card carries no INCONSISTENCY id.")
    choices = "|".join((*RESOLVE_ACTIONS, "DEFER"))
    if not action:
        raise ValueError(f"resolve needs --action {choices}.")
    try:
        chosen = ResolutionAction(action.upper())
    except ValueError as exc:
        raise ValueError(f"Unknown action {action!r}; use {choices}.") from exc

    record = await get_particle(session, card.inconsistency_id)
    if record is None or record.status is not Status.INCONSISTENCY:
        raise ValueError(f"{card.key} is no longer an open conflict.")
    a_side, b_side = record_sides(record)
    a = await get_particle(session, a_side[0]) if a_side else None
    b = await get_particle(session, b_side[0]) if b_side else None
    offered = resolve_actions_for(a, b)
    if chosen is not ResolutionAction.DEFER and chosen.value not in offered:
        side, member = ("A", a) if chosen is ResolutionAction.PREFER_A else ("B", b)
        state = member.status.value if member is not None else "no longer in the store"
        raise ValueError(
            f"{chosen.value} is not offered on this conflict: claim {side} is {state}, "
            f"so it cannot be the one kept. Offered: {', '.join(offered)}."
        )

    await resolve(session, record.id, chosen, reviewer_id=actor, note=note, actor=actor)
    if chosen is ResolutionAction.DEFER:
        return f"Deferred {card.key}: note recorded, the conflict stays open."
    return f"Resolved {card.key} as {chosen.value}."


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
        # The card key rides the event so the precision report can
        # attribute the retraction to this card after the card has left the
        # retained collections.
        payload={"via": "curate", "card_key": card.key, "kind": card.kind.value},
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
    """Assign a subject to a NO_SUBJECT orphan in place.

    ``subject`` is an existing Subject id (linked directly) or a subject name run
    through the standard resolver. The orphan keeps its id; only its subject
    link is written.
    """
    if card.kind is not CardKind.NO_SUBJECT or len(card.particle_ids) != 1:
        raise ValueError("assign-subject resolves a no_subject card.")
    if not subject or not subject.strip():
        raise ValueError("assign-subject requires a subject (id or name) via --subject.")

    # Deferred import: agent_write pulls the ingest reconciliation stack, so
    # load it only on dispatch.
    from particles.operations.agent_write import assign_subject_belief

    sval = subject.strip()
    existing = await get_subject(session, sval)
    pid = card.particle_ids[0]
    await assign_subject_belief(
        session,
        store=store,
        particle_id=pid,
        subject_id=sval if existing is not None else None,
        subject_name=None if existing is not None else sval,
        actor=actor,
    )
    return f"Assigned subject to {pid[:8]}…"


async def _relink(session: AsyncSession, card: CurationCard, *, actor: str) -> str:
    """Relink the batch card's orphans in place over the accepted tiers.

    Re-plans from the store rather than trusting the snapshot's member list,
    so a belief linked, retracted or superseded since the build is skipped.
    """
    if card.kind is not CardKind.GATED_SUBJECTS:
        raise ValueError("relink resolves the gated_subjects card.")
    from particles.operations.subject_relink import apply_gated_relink, plan_gated_relink

    plan = await plan_gated_relink(session)
    result = await apply_gated_relink(session, plan, actor=actor)
    return (
        f"Relinked {len(result.relinked)} belief(s) to {len(result.subjects)} subject(s)"
        + (f"; skipped {len(result.skipped)}" if result.skipped else "")
        + "."
    )


async def _supersede(
    session: AsyncSession,
    card: CurationCard,
    *,
    actor: str,
    store: str,
    reason: str | None,
    revision: BeliefRevision | None,
) -> str:
    """Replace a card's belief with the operator's corrected one.

    Dispatches onto the operator ``supersede_belief`` path, the call
    ``POST /particles/{id}/supersede`` makes. Subjects are inherited by id
    unless the revision names its own; provenance defaults to the reason.
    """
    if len(card.particle_ids) != 1:
        raise ValueError("supersede resolves a single-belief card.")
    if revision is None or not revision.content.strip():
        raise ValueError("supersede requires the replacement belief via --content.")
    reason = reason.strip() if reason else None
    if not reason:
        raise ValueError("supersede requires a reason via --reason.")
    pid = card.particle_ids[0]
    target = await get_particle(session, pid)
    if target is None:
        raise ValueError(f"Particle {pid!r} not found.")

    ids: list[str] = []
    names: list[str] = []
    if revision.subjects:
        for value in (v.strip() for v in revision.subjects):
            if not value:
                continue
            if await get_subject(session, value) is not None:
                ids.append(value)
            else:
                names.append(value)
    else:
        ids = list(target.subject_ids)
    if not ids and not names:
        raise ValueError(
            "The belief has no subject to inherit. Name one with --subject, or use "
            "the assign-subject gesture to keep the content and fix only the subject."
        )

    source_excerpt = revision.source_excerpt
    if source_excerpt is None and revision.corpus_entry_id is None:
        source_excerpt = reason

    # Deferred import: the operator-supersede primitive lives in agent_write,
    # which pulls the ingest reconciliation stack — load it only on dispatch.
    from particles.operations.agent_write import supersede_belief

    result = await supersede_belief(
        session,
        store=store,
        supersedes_id=pid,
        content=revision.content.strip(),
        subject_names=names,
        subject_ids=ids,
        confidence=revision.confidence,
        source_excerpt=source_excerpt,
        corpus_entry_id=revision.corpus_entry_id,
        operator=True,
        actor=actor,
        reason=reason,
        card_key=card.key,
    )
    successor = (result.asserted_particle_id or "?")[:8]
    line = f"Superseded {pid[:8]}… → successor {successor}… ({result.verdict})"
    if result.inconsistency_id is not None:
        line += (
            f". It conflicts with another belief: INCONSISTENCY "
            f"{result.inconsistency_id[:8]}… holds it until `particles review` resolves it"
        )
    return line + "."


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

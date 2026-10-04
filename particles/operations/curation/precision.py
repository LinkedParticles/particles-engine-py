# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Queue precision: how often the curation queue was right.

The queue's claim to an operator is that reviewing it is a better use of
their time than authoring the model, and that holds only if the cards are
mostly right. The store already keeps what is needed to measure that: every
gesture lands as an operator event, every review
ruling is one too, and the retained curation collections say
which cards were open. This module reads those at request time and stores
nothing: :func:`curation_precision` is the gather shell and
:func:`summarise_precision` the pure decision over plain values.

Per card kind, over a window, each card gets its strongest recorded outcome:

- **acted**: a resolving gesture or ruling landed on the card (affirm,
  supersede, retract, merge, deposit, assign-subject, relink, accept, a
  review that prefers a side or discards both);
- **dismissed**: the operator said the card was wrong or not worth a ruling
  (dismiss, reject, a ``BOTH_VALID`` review on a conflict record);
- **snoozed**: undecided (snooze, or a ``DEFER`` review);
- **open**: in the current collection and untouched in the window;
- **expired**: in an older retained collection but gone from the current one,
  or a conflict record the system closed without a ruling, untouched.

What "acted" and "dismissed" mean differs by kind, and :data:`RIGHT_MEANS`
spells it out: a dismissed duplicate pair means the finder paired two
different claims, while a dismissed expiry means the question was not worth
asking. ``precision`` is ``acted / (acted + dismissed)`` over the cards the
operator decided; snoozed and untouched cards are shown beside it, never
folded into it.

Attribution. Events that carry a card key (affirm, snooze, the ``curate``
retract and supersede gestures) attribute exactly, and so do the events whose
key is reconstructible: a review names its record, an abstraction verdict its
candidate, a co-evidential link its pair, a relink its orphan, a URL dismissal
its URL. Any other resolving event names only beliefs, and is attributed to
every retained card naming one of them, the same over-attribution the live
queue filter accepts. An event that matches no retained card is
counted in ``unattributed_events`` rather than guessed, so the acted figure is
a floor and the report says so.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from enum import IntEnum

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import CurationPrecision, CurationPrecisionKind, RelationType
from particles.corpus.store import get_entry_by_uri
from particles.store.curation_snapshot_store import list_snapshots
from particles.store.event_store import (
    EventRefKind,
    OperatorEvent,
    OperatorEventType,
    list_events_in_range,
    list_events_since,
)

from .cards import CardKind, CurationCard
from .snapshot import load_collection

log = logging.getLogger(__name__)


class Outcome(IntEnum):
    """A card's recorded outcome, ordered by strength: the strongest wins."""

    EXPIRED = 1
    SNOOZED = 2
    DISMISSED = 3
    ACTED = 4


#: What ``acted`` and ``dismissed`` mean per kind, in that order. Rendered in
#: the report beside the figures, because the two outcomes are not the same
#: finding on every kind.
RIGHT_MEANS: dict[CardKind, tuple[str, str]] = {
    CardKind.STALE: (
        "the belief was affirmed as still true, replaced or retracted",
        "the expiry was not worth a ruling",
    ),
    CardKind.RETRACTION_CASCADE: (
        "the belief was replaced or retracted",
        "the belief stands despite its retracted basis",
    ),
    CardKind.BROKEN_PROVENANCE: (
        "the belief was replaced or retracted",
        "the belief stands despite its missing source",
    ),
    CardKind.CONFIDENCE_DECAY: (
        "the belief was affirmed as sound or replaced",
        "the confidence estimate was not worth a ruling",
    ),
    CardKind.RECENCY_DECAY: (
        "the belief was affirmed as current or replaced",
        "the belief's age was not worth a ruling",
    ),
    CardKind.CONTRADICTION: (
        "one side was replaced or retracted",
        "the two beliefs do not contradict, so the probe was wrong",
    ),
    CardKind.CONTESTED: (
        "the belief was affirmed despite the disagreement, or replaced",
        "the disagreement was not worth a ruling",
    ),
    CardKind.NO_SUBJECT: (
        "a subject was assigned, or the belief was replaced or retracted",
        "the orphan was not worth a ruling",
    ),
    CardKind.GATED_SUBJECTS: (
        "the batch was relinked",
        "the batch was not worth relinking",
    ),
    CardKind.DUPLICATE_PAIR: (
        "the pair was merged as one claim",
        "the two are different claims, so the finder was wrong",
    ),
    CardKind.UNCITED_URL: (
        "the URL was deposited",
        "the URL was not worth depositing",
    ),
    CardKind.FAILED_SNAPSHOTS: (
        "a reindex run re-extracted the failed snapshots",
        "the failures were not worth a reindex",
    ),
    CardKind.PROPOSED_ABSTRACTION: (
        "the generalization was accepted",
        "the generalization was rejected as unfaithful or unhelpful",
    ),
    CardKind.STALE_BASIS: (
        "the belief was affirmed as still true, replaced or retracted",
        "the changed basis was not worth a ruling",
    ),
    CardKind.INCONSISTENCY: (
        "resolved by preferring one side or discarding both",
        "resolved BOTH_VALID, so the claims did not really conflict",
    ),
    CardKind.DEMOTION: (
        "the replacement was affirmed as right",
        "both claims hold, so the retirement was wrong",
    ),
}

#: The event types the report reads inside the window.
PRECISION_EVENTS: frozenset[OperatorEventType] = frozenset(
    {
        OperatorEventType.BELIEF_AFFIRMED,
        OperatorEventType.CURATION_CARD_SNOOZED,
        OperatorEventType.DEPOSIT_SUGGESTION_DISMISSED,
        OperatorEventType.ABSTRACTION_RESOLVED,
        OperatorEventType.REVIEW_RESOLVED,
        OperatorEventType.INCONSISTENCY_CLOSED,
        OperatorEventType.SUBJECTS_RELINKED,
        OperatorEventType.RELATION_ADDED,
        OperatorEventType.DUPLICATES_MERGED,
        OperatorEventType.PARTICLE_RETRACTED,
        OperatorEventType.PARTICLE_SUPERSEDED,
        OperatorEventType.SUBJECT_LINK_CONFIRMED,
        OperatorEventType.EXTRACT_RUN,
    }
)

#: The gestures that stand for good: recorded before the window, they take a
#: card out of the open set, since the operator already ruled on it.
_STANDING_EVENTS: tuple[OperatorEventType, ...] = (
    OperatorEventType.BELIEF_AFFIRMED,
    OperatorEventType.CURATION_CARD_SNOOZED,
    OperatorEventType.DEPOSIT_SUGGESTION_DISMISSED,
)

#: Review actions that settle a conflict by ruling on its sides.
_RULED_ACTIONS = frozenset({"PREFER_A", "PREFER_B", "DISCARD"})

_GATED_KEY = "gated_subjects"
_FAILED_KEY = "failed_snapshots"


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def kind_of_key(key: str) -> CardKind | None:
    """The card kind a key names, or ``None`` for a key of no known kind."""
    prefix, _, _ = key.partition(":")
    try:
        return CardKind(prefix)
    except ValueError:
        return None


def _is_gesture(ev: OperatorEvent) -> bool:
    """Whether the event is one the report reads at all.

    An ``EXTRACT_RUN`` counts only on the ``reindex`` route, where it is the
    surfaced gesture of the ``failed_snapshots`` card; any other run is not a
    gesture and is skipped rather than reported as unattributed.
    """
    if ev.event_type not in PRECISION_EVENTS:
        return False
    if ev.event_type is OperatorEventType.EXTRACT_RUN:
        return (ev.payload or {}).get("route") == "reindex"
    return True


def _particle_refs(ev: OperatorEvent) -> list[str]:
    return [r.ref_id for r in ev.refs if r.ref_kind is EventRefKind.PARTICLE]


def _membership(cards: Iterable[CurationCard]) -> dict[str, set[str]]:
    """Belief id to the keys of every retained card naming it."""
    index: dict[str, set[str]] = {}
    for card in cards:
        ids = list(card.particle_ids)
        if card.inconsistency_id:
            ids.append(card.inconsistency_id)
        for pid in ids:
            index.setdefault(pid, set()).add(card.key)
    return index


def _standing_key(ev: OperatorEvent) -> str | None:
    """The key a standing suppression names, or ``None`` if the event is not one."""
    payload = ev.payload or {}
    match ev.event_type:
        case OperatorEventType.BELIEF_AFFIRMED:
            key = payload.get("card_key")
            return key if isinstance(key, str) else None
        case OperatorEventType.CURATION_CARD_SNOOZED:
            key = payload.get("card_key")
            if isinstance(key, str) and payload.get("snoozed_until") is None:
                return key
            return None
        case OperatorEventType.DEPOSIT_SUGGESTION_DISMISSED:
            url = payload.get("canonical_url")
            if isinstance(url, str) and payload.get("snooze_days") is None:
                return f"{CardKind.UNCITED_URL.value}:{url}"
            return None
    return None


def classify_event(
    ev: OperatorEvent,
    *,
    membership: Mapping[str, Collection[str]],
    deposited_urls: Collection[str],
) -> list[tuple[str, Outcome]]:
    """The ``(card key, outcome)`` pairs one event records.

    Empty for an event of no interest, and for a resolving event that names
    only beliefs none of the retained cards name (the caller counts those as
    unattributed). One event can settle several cards: a retraction resolves
    every card about that belief.
    """
    payload = ev.payload or {}
    refs = _particle_refs(ev)

    def by_membership(outcome: Outcome) -> list[tuple[str, Outcome]]:
        keys: set[str] = set()
        for pid in refs:
            keys.update(membership.get(pid, ()))
        return [(k, outcome) for k in sorted(keys)]

    def stamped_or_membership(outcome: Outcome) -> list[tuple[str, Outcome]]:
        key = payload.get("card_key")
        if isinstance(key, str) and key:
            return [(key, outcome)]
        return by_membership(outcome)

    match ev.event_type:
        case OperatorEventType.BELIEF_AFFIRMED:
            return stamped_or_membership(Outcome.ACTED)

        case OperatorEventType.CURATION_CARD_SNOOZED:
            key = payload.get("card_key")
            if not isinstance(key, str) or not key:
                return []
            outcome = Outcome.DISMISSED if payload.get("snoozed_until") is None else Outcome.SNOOZED
            return [(key, outcome)]

        case OperatorEventType.DEPOSIT_SUGGESTION_DISMISSED:
            url = payload.get("canonical_url")
            if not isinstance(url, str):
                return []
            key = f"{CardKind.UNCITED_URL.value}:{url}"
            if payload.get("snooze_days") is not None:
                return [(key, Outcome.SNOOZED)]
            # The deposit gesture retires the suggestion through this same
            # path, so a permanent dismissal whose URL is now in
            # the corpus was a deposit.
            return [(key, Outcome.ACTED if url in deposited_urls else Outcome.DISMISSED)]

        case OperatorEventType.ABSTRACTION_RESOLVED:
            candidate = payload.get("candidate_event_id")
            if not isinstance(candidate, str):
                return []
            key = f"{CardKind.PROPOSED_ABSTRACTION.value}:{candidate}"
            accepted = payload.get("resolution") == "accepted"
            return [(key, Outcome.ACTED if accepted else Outcome.DISMISSED)]

        case OperatorEventType.REVIEW_RESOLVED:
            if not refs:
                return []
            # The record is the event's first ref.
            key = f"{CardKind.INCONSISTENCY.value}:{refs[0]}"
            action = payload.get("action")
            if action == "DEFER":
                return [(key, Outcome.SNOOZED)]
            if action in _RULED_ACTIONS:
                return [(key, Outcome.ACTED)]
            if action == "BOTH_VALID":
                return [(key, Outcome.DISMISSED)]
            return []

        case OperatorEventType.INCONSISTENCY_CLOSED:
            # The system closed the record (lapsed, regrouped, withdrawn on a
            # second reading) and the operator never ruled: an expiry.
            if not refs:
                return []
            return [(f"{CardKind.INCONSISTENCY.value}:{refs[0]}", Outcome.EXPIRED)]

        case OperatorEventType.SUBJECTS_RELINKED:
            if payload.get("batch"):
                return [(_GATED_KEY, Outcome.ACTED)]
            relinked = [(f"{CardKind.NO_SUBJECT.value}:{pid}", Outcome.ACTED) for pid in refs]
            return relinked + [p for p in by_membership(Outcome.ACTED) if p not in relinked]

        case OperatorEventType.RELATION_ADDED:
            linked: list[tuple[str, Outcome]] = []
            if payload.get("relation_type") == RelationType.CO_EVIDENTIAL.value and len(refs) == 2:
                linked.append(
                    (f"{CardKind.DUPLICATE_PAIR.value}:" + "|".join(sorted(refs)), Outcome.ACTED)
                )
            return linked + [p for p in by_membership(Outcome.ACTED) if p not in linked]

        case OperatorEventType.DUPLICATES_MERGED:
            survivor = payload.get("survivor")
            merged = payload.get("superseded")
            if not isinstance(survivor, str) or not isinstance(merged, list):
                return by_membership(Outcome.ACTED)
            merged_pairs = [
                (
                    f"{CardKind.DUPLICATE_PAIR.value}:" + "|".join(sorted((survivor, str(pid)))),
                    Outcome.ACTED,
                )
                for pid in merged
            ]
            return merged_pairs + [p for p in by_membership(Outcome.ACTED) if p not in merged_pairs]

        case OperatorEventType.EXTRACT_RUN:
            if payload.get("route") == "reindex":
                return [(_FAILED_KEY, Outcome.ACTED)]
            return []

        case (
            OperatorEventType.PARTICLE_RETRACTED
            | OperatorEventType.PARTICLE_SUPERSEDED
            | OperatorEventType.SUBJECT_LINK_CONFIRMED
        ):
            return stamped_or_membership(Outcome.ACTED)

    return []


def summarise_precision(
    events: Sequence[OperatorEvent],
    *,
    since: datetime,
    until: datetime,
    open_cards: Sequence[CurationCard] = (),
    retired_cards: Sequence[CurationCard] = (),
    deposited_urls: Collection[str] = (),
    standing_before: Sequence[OperatorEvent] = (),
    snapshots_read: int = 0,
    snapshot_built_at: datetime | None = None,
) -> CurationPrecision:
    """Queue precision over ``[since, until]`` as a pure function of plain values.

    ``events`` is the operator event log (any types; only
    :data:`PRECISION_EVENTS` inside the window count). ``open_cards`` is the
    current collection and ``retired_cards`` the cards of the older retained
    collections; a retired card absent from the current one expired.
    ``deposited_urls`` are the URLs now in the corpus, which turns a permanent
    URL dismissal into a deposit. ``standing_before`` are affirmations and
    permanent dismissals recorded before the window: those cards were already
    ruled on, so they leave the open set.

    Each card takes its strongest outcome in the window (:class:`Outcome`
    order), so a card snoozed and later retracted counts once, as acted. The
    per-kind rows carry :data:`RIGHT_MEANS`, because "right" is a different
    finding per kind.
    """
    since = _as_utc(since)
    until = _as_utc(until)
    membership = _membership([*open_cards, *retired_cards])
    deposited = set(deposited_urls)

    outcomes: dict[str, Outcome] = {}
    events_read = 0
    unattributed = 0
    for ev in events:
        if not _is_gesture(ev):
            continue
        at = _as_utc(ev.occurred_at)
        if at < since or at > until:
            continue
        events_read += 1
        pairs = [
            (k, o)
            for k, o in classify_event(ev, membership=membership, deposited_urls=deposited)
            if kind_of_key(k) is not None
        ]
        if not pairs:
            unattributed += 1
            continue
        for key, outcome in pairs:
            if outcome > outcomes.get(key, Outcome.EXPIRED - 1):
                outcomes[key] = outcome

    settled_before: set[str] = set()
    for ev in standing_before:
        if _as_utc(ev.occurred_at) >= since:
            continue
        standing_key = _standing_key(ev)
        if standing_key is not None:
            settled_before.add(standing_key)

    open_keys = {c.key for c in open_cards}
    expired_keys = {c.key for c in retired_cards} - open_keys
    touched = set(outcomes)
    untouched_open = open_keys - touched - settled_before
    untouched_expired = expired_keys - touched - settled_before

    per_kind: dict[CardKind, dict[str, int]] = {}

    def bucket(kind: CardKind) -> dict[str, int]:
        return per_kind.setdefault(
            kind, {"acted": 0, "dismissed": 0, "snoozed": 0, "open": 0, "expired": 0}
        )

    for key, outcome in outcomes.items():
        kind = kind_of_key(key)
        if kind is None:  # pragma: no cover — filtered above
            continue
        bucket(kind)[outcome.name.lower()] += 1
    for key in untouched_open:
        kind = kind_of_key(key)
        if kind is not None:
            bucket(kind)["open"] += 1
    for key in untouched_expired:
        kind = kind_of_key(key)
        if kind is not None:
            bucket(kind)["expired"] += 1

    rows: list[CurationPrecisionKind] = []
    for kind, counts in per_kind.items():
        decided = counts["acted"] + counts["dismissed"]
        acted_means, dismissed_means = RIGHT_MEANS.get(kind, ("", ""))
        rows.append(
            CurationPrecisionKind(
                kind=kind.value,
                offered=sum(counts.values()),
                precision=(counts["acted"] / decided) if decided else None,
                acted_means=acted_means,
                dismissed_means=dismissed_means,
                **counts,
            )
        )
    rows.sort(key=lambda r: (-(r.acted + r.dismissed + r.snoozed), -r.offered, r.kind))

    totals = {
        field: sum(getattr(r, field) for r in rows)
        for field in ("offered", "acted", "dismissed", "snoozed", "open", "expired")
    }
    decided = totals["acted"] + totals["dismissed"]
    return CurationPrecision(
        since=since,
        until=until,
        kinds=rows,
        precision=(totals["acted"] / decided) if decided else None,
        events_read=events_read,
        unattributed_events=unattributed,
        snapshots_read=snapshots_read,
        snapshot_built_at=snapshot_built_at,
        **totals,
    )


async def curation_precision(
    session: AsyncSession,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> CurationPrecision | None:
    """Gather the window's events and the retained collections, then summarise.

    ``since`` defaults to ``curation.precision_window_days`` before ``until``,
    which defaults to now. Reads only: the event log, the curation snapshot
    ring, and one corpus lookup per permanent URL dismissal. Returns ``None``
    when there is nothing to measure, a store with no collection and no
    gesture, so a report can leave the block out rather than print zeros.
    """
    until = _as_utc(until) if until is not None else datetime.now(UTC)
    if since is None:
        since = until - timedelta(days=get_config().curation.precision_window_days)
    since = _as_utc(since)

    events = await list_events_since(session, since=since, event_types=PRECISION_EVENTS)
    standing: list[OperatorEvent] = []
    for event_type in _STANDING_EVENTS:
        standing.extend(await list_events_in_range(session, event_type=event_type, until=since))

    rows = await list_snapshots(session)
    open_cards = load_collection(rows[0]) if rows else []
    retired_cards = [card for row in rows[1:] for card in load_collection(row)]

    deposited: set[str] = set()
    for ev in events:
        if ev.event_type is not OperatorEventType.DEPOSIT_SUGGESTION_DISMISSED:
            continue
        payload = ev.payload or {}
        url = payload.get("canonical_url")
        if (
            isinstance(url, str)
            and payload.get("snooze_days") is None
            and await get_entry_by_uri(session, url) is not None
        ):
            deposited.add(url)

    summary = summarise_precision(
        events,
        since=since,
        until=until,
        open_cards=open_cards,
        retired_cards=retired_cards,
        deposited_urls=deposited,
        standing_before=standing,
        snapshots_read=len(rows),
        snapshot_built_at=_as_utc(rows[0].built_at) if rows else None,
    )
    if summary.offered == 0 and summary.events_read == 0:
        return None
    return summary


# ---------------------------------------------------------------------------
# Rendering — shared by `curate --precision` and the `quality` dashboard.
# ---------------------------------------------------------------------------


def _share(count: int, total: int) -> str:
    return f"{count / total:.1%}" if total else "n/a"


def render_precision_lines(report: CurationPrecision, *, meanings: bool = True) -> list[str]:
    """The report as terminal lines: a summary, a per-kind table, and the meanings.

    ``meanings=False`` leaves out the per-kind "acted means / dismissed means"
    block, for the dashboard, where the table alone is the point.
    """
    days = max(1, round((report.until - report.since).total_seconds() / 86_400))
    lines = [
        f"Curation queue precision: {report.since.date().isoformat()} to "
        f"{report.until.date().isoformat()} ({days} days)"
    ]
    decided = report.acted + report.dismissed
    if decided:
        assert report.precision is not None
        lines.append(
            f"  Of {decided:,} cards the operator ruled on, {report.acted:,} were real "
            f"problems and {report.dismissed:,} were not: precision {report.precision:.2f}."
        )
    else:
        lines.append("  No card was ruled on in this window, so there is no precision to report.")
    lines.append(
        f"  Denominator: {report.offered:,} cards offered = {decided:,} decided + "
        f"{report.snoozed:,} snoozed + {report.open:,} open untouched + "
        f"{report.expired:,} expired untouched."
    )
    if report.unattributed_events:
        lines.append(
            f"  {report.unattributed_events:,} resolving event(s) in the window named no card "
            "the store still holds and are not counted, so acted is a floor."
        )
    if report.snapshots_read:
        built = report.snapshot_built_at
        stamp = built.strftime("%Y-%m-%d %H:%M UTC") if built else "unknown"
        lines.append(
            f"  Open and expired cards come from {report.snapshots_read} retained "
            f"collection(s); the newest was built {stamp}."
        )
    else:
        lines.append(
            "  No stored collection, so open and expired cards could not be counted; "
            "`particles curate --refresh` builds one."
        )
    if not report.kinds:
        return lines

    header = (
        f"  {'kind':<22} {'offered':>8} {'acted':>6} {'dismissed':>9} "
        f"{'snoozed':>7} {'open':>7} {'expired':>7} {'precision':>9}"
    )
    lines.append("")
    lines.append(header)
    for row in report.kinds:
        precision = f"{row.precision:.2f}" if row.precision is not None else "n/a"
        lines.append(
            f"  {row.kind:<22} {row.offered:>8,} {row.acted:>6,} {row.dismissed:>9,} "
            f"{row.snoozed:>7,} {row.open:>7,} {row.expired:>7,} {precision:>9}"
        )
    lines.append(
        f"  {'all kinds':<22} {report.offered:>8,} {report.acted:>6,} {report.dismissed:>9,} "
        f"{report.snoozed:>7,} {report.open:>7,} {report.expired:>7,} "
        f"{(f'{report.precision:.2f}' if report.precision is not None else 'n/a'):>9}"
    )
    if report.acted + report.dismissed + report.snoozed:
        lines.append(
            f"  Shares of the denominator: acted {_share(report.acted, report.offered)}, "
            f"dismissed {_share(report.dismissed, report.offered)}, "
            f"snoozed {_share(report.snoozed, report.offered)}, "
            f"untouched {_share(report.open + report.expired, report.offered)}."
        )
    if meanings:
        lines.append("")
        lines.append("  What acted and dismissed mean per kind:")
        for row in report.kinds:
            lines.append(
                f"    {row.kind}: acted = {row.acted_means}; dismissed = {row.dismissed_means}."
            )
    return lines

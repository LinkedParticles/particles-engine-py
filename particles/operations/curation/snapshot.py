# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The persisted collection half of the queue (§4).

`build_curation_queue` ran every finder on every request. Measured
2026-08-02 on the dogfood store (32,472 particles / 4,009 subjects): 137 s of
collection — 13,424 cards — to return 7. This module owns the expensive half so
`session.py` can keep the cheap half live:

* :func:`collect_and_persist` runs the finders, scores the cards, merges the
  result with the prior snapshot per the §4 scope rule, and stores it;
* :func:`load_collection` reads a stored collection back;
* :class:`CurationQueueResult` is the staleness stamp every consumer renders.

**The §4 rule.** A build declares, per ``CardKind``, whether that kind's finder
saw the whole store or only the delta. Store-wide kinds *replace*
— their card set is complete, so the new build's is authoritative. Delta-scoped
kinds *carry forward* — unioned with the prior snapshot's, because a
contradiction found last night is not re-found tonight (the probe is scoped to
the delta), and only a confirmed cross-source pair becomes an INCONSISTENCY
record. Without the carry-forward the queue would forget
yesterday's other contradictions every night. A carried card whose pair a
census record now discloses is dropped: the record's INCONSISTENCY card stands
in its place.

**A finder that did not run is not silent.** A kind whose finder this build
skipped (the contradiction probe, with the semantic finders off or the LLM
breaker open) is declared ``carried``: its prior cards carry forward exactly
as a delta-scoped kind's do, and the envelope keeps the date the finder last
ran, so ``particles curate --refresh`` no longer erases the census's
contradiction cards. A carried card is still evicted when it is
suppressed, resolved since it was found, or names a retired belief;
the decision is :func:`_merge_with_prior`, a pure function over what the
shell gathered.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import SuggestMode
from particles.core.status import Status
from particles.db import write_transaction
from particles.operations._llm import llm_circuit_open
from particles.operations.lint import ContradictionProbeControl
from particles.operations.lint.contradictions import contradiction_partner
from particles.store.curation_snapshot_store import (
    CollectionScope,
    CurationSnapshotRow,
    latest_snapshot,
    parse_per_kind_scope,
    write_snapshot,
)
from particles.store.particle_store import get_particles_by_ids

from .cards import CardKind, CurationCard, gestures_for

log = logging.getLogger(__name__)

#: Serialisation version of the ``cards_json`` envelope. 2: open
#: conflicts are ``INCONSISTENCY`` cards and ``comment`` is no longer a gesture;
#: a format-1 collection is upgraded on read (:func:`_upgrade_format_1`).
BLOB_FORMAT = 2

#: The kinds whose finder the cycle runs **delta-scoped**, and which
#: therefore carry forward instead of being replaced.
#:
#: Only ``CONTRADICTION`` today: its probe takes
#: ``ContradictionProbeControl(scope_particle_ids=…)``. ``DUPLICATE_PAIR`` is
#: *not* here — duplicate *enumeration* stays store-wide and bounds
#: only the ``LLM_JUDGE`` verdict pass, and consolidation runs ``REPORT``. Every
#: structural kind runs store-wide unconditionally.
#:
#: A new delta-scoped finder must be added here, or its cards will be silently
#: dropped on the next nightly build.
DELTA_SCOPED_KINDS: frozenset[CardKind] = frozenset({CardKind.CONTRADICTION})

#: The kinds whose finder runs only with the semantic finders on, so a build
#: without them declares the kind ``carried`` rather than store-wide.
#:
#: Only ``CONTRADICTION``: its LLM probe is the semantic pass. Its structural
#: half (recorded ``CONTRADICTS`` edges) still runs, and the union keeps those
#: cards too. ``DUPLICATE_PAIR`` is not here: enumeration runs either way, and
#: ``semantic`` only switches the verdict mode.
SEMANTIC_KINDS: frozenset[CardKind] = frozenset({CardKind.CONTRADICTION})

#: The per-kind scopes whose prior cards carry forward rather than replace.
_CARRYING_SCOPES: frozenset[CollectionScope] = frozenset(
    {CollectionScope.DELTA, CollectionScope.CARRIED}
)


class QueueSource(StrEnum):
    """Where `build_curation_queue` should get its collection."""

    SNAPSHOT = "snapshot"
    LIVE = "live"


class CurationQueueResult(BaseModel):
    """The ranked queue plus the staleness stamp.

    Consumers render the stamp rather than hiding it: an operator who can see
    ``built_at`` can tell a quiet queue from a stale one. ``stale`` is the
    convenience verdict (``age`` past :func:`stale_after_hours`);
    ``per_kind_scope`` discloses that a nightly collection is not a store-wide
    census for its delta-scoped kinds.
    """

    cards: list[CurationCard] = Field(default_factory=list)
    count: int = 0
    #: ``snapshot`` when served from a stored collection, ``live`` when the
    #: finders ran for this request.
    source: str = QueueSource.LIVE.value
    snapshot_id: str | None = None
    built_at: datetime | None = None
    age_seconds: float | None = None
    stale: bool = False
    scope: str = CollectionScope.STORE.value
    per_kind_scope: dict[str, str] = Field(default_factory=dict)
    semantic: bool = False
    #: Semantic finders were requested but the breaker was open when
    #: the collection was built.
    semantic_degraded: bool = False
    #: Cards in the collection before narrowing / suppression / the top-N slice.
    collection_size: int = 0
    #: Cards still open after the ``kind`` filter and suppression (snoozed,
    #: affirmed, dismissed, resolved since the build) but before the top-N
    #: slice: the backlog ``cards`` is the head of. Belief status is re-checked
    #: only while walking the slice, so a card whose belief was retired after
    #: the build still counts here until the next build.
    open_count: int = 0
    #: When each kind's finder last ran for the cards this collection holds.
    #: A kind this build ran is stamped with ``built_at``; a ``carried`` kind
    #: keeps the date of the build that last ran its finder, and is absent when
    #: no build on record ran it. Empty on a live collection.
    kind_as_of: dict[str, datetime] = Field(default_factory=dict)


@dataclass(frozen=True)
class Resolved:
    """What resolving operator events touched (level 3).

    Plain values, read by the shell from the event log and matched against a
    card here, so the live session half and the carry-forward rule share one
    definition of "resolved".
    """

    #: Card keys suppressed outright (URL cards, the gated-subjects batch card).
    keys: frozenset[str] = frozenset()
    #: Beliefs a resolving event touched.
    particle_ids: frozenset[str] = frozenset()
    #: INCONSISTENCY records a non-DEFER review closed.
    records: frozenset[str] = frozenset()

    def covers(self, card: CurationCard) -> bool:
        """Whether this resolution already handled ``card``.

        A conflict card is matched by its record id only. ``REVIEW_RESOLVED``
        names the record and its members, so member matching would let one
        record's resolution hide every record sharing a member, and a DEFER,
        which leaves the record open, would hide its own card.
        """
        if card.key in self.keys:
            return True
        if card.kind is CardKind.INCONSISTENCY:
            return card.inconsistency_id in self.records
        if card.kind is CardKind.GATED_SUBJECTS:
            # A batch card is suppressed by its key when a relink ran.
            return False
        return any(pid in self.particle_ids for pid in card.particle_ids)

    @classmethod
    def union(cls, parts: Iterable[Resolved]) -> Resolved:
        """Every resolution in ``parts`` as one."""
        keys: set[str] = set()
        pids: set[str] = set()
        records: set[str] = set()
        for part in parts:
            keys |= part.keys
            pids |= part.particle_ids
            records |= part.records
        return cls(frozenset(keys), frozenset(pids), frozenset(records))


@dataclass(frozen=True)
class CarryEvidence:
    """What the shell read about the carry candidates.

    Each field is one of the §4 eviction rules, gathered as plain values so
    :func:`_merge_with_prior` stays a pure decision.
    """

    #: Card keys an unexpired snooze or a standing dismiss / affirm hides.
    suppressed: frozenset[str] = frozenset()
    #: Each resolving event since the oldest candidate was found, with its time.
    resolutions: tuple[tuple[datetime, Resolved], ...] = ()
    #: Beliefs a candidate names that are no longer ACTIVE.
    retired: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PriorCollection:
    """The decoded prior snapshot the merge carries cards from."""

    cards: list[CurationCard]
    first_seen: dict[str, str]
    #: Per kind, when its finder last ran for these cards (see ``kind_as_of``).
    kind_as_of: dict[str, datetime] = field(default_factory=dict)
    built_at: datetime | None = None


@dataclass(frozen=True)
class MergeResult:
    """The merged collection, its origin stamps, and each kind's as-of date."""

    cards: list[CurationCard]
    first_seen: dict[str, str]
    kind_as_of: dict[str, datetime]


def per_kind_scope_for(scope: CollectionScope, *, semantic: bool) -> dict[str, CollectionScope]:
    """The §4 per-kind scope map for a build (pure).

    A kind in :data:`SEMANTIC_KINDS` is ``carried`` when ``semantic`` is false:
    its finder did not run, so its silence says nothing and the prior cards
    carry forward. Otherwise a store-wide build is store-wide for
    every kind, and a delta build is delta-scoped only for
    :data:`DELTA_SCOPED_KINDS`; every other finder still walked the whole
    store, so those kinds still replace.

    Args:
        scope: the build's overall scope, ``store`` or ``delta``.
        semantic: whether the semantic finders ran in full: requested, and the
            LLM breaker was not open when the collection completed.
    """
    out: dict[str, CollectionScope] = {}
    for kind in CardKind:
        if kind in SEMANTIC_KINDS and not semantic:
            out[kind.value] = CollectionScope.CARRIED
        elif scope is CollectionScope.DELTA and kind in DELTA_SCOPED_KINDS:
            out[kind.value] = CollectionScope.DELTA
        else:
            out[kind.value] = CollectionScope.STORE
    return out


def _dump(
    cards: list[CurationCard], first_seen: dict[str, str], kind_as_of: dict[str, datetime]
) -> str:
    """Serialise a collection, its per-card origin stamps, and each kind's as-of date."""
    return json.dumps(
        {
            "format": BLOB_FORMAT,
            "cards": [c.model_dump(mode="json") for c in cards],
            "first_seen": first_seen,
            "kind_as_of": {k: v.isoformat() for k, v in kind_as_of.items()},
        }
    )


def _load(blob: str) -> tuple[list[CurationCard], dict[str, str]]:
    """Decode a collection blob's cards and origin stamps (see :func:`_decode`)."""
    cards, first_seen, _ = _decode(blob)
    return cards, first_seen


def _decode(blob: str) -> tuple[list[CurationCard], dict[str, str], dict[str, datetime] | None]:
    """Decode a collection blob, tolerating an unreadable one as empty.

    A cache that cannot be read is a cache miss, never an error: the caller
    rebuilds. Returning empty keeps a corrupt row from taking the curation
    surface down with it. The third value is ``None`` for a blob written
    before the envelope carried ``kind_as_of``.
    """
    try:
        raw = json.loads(blob)
    except (TypeError, ValueError):
        log.warning("curation snapshot: unreadable cards_json; treating as a cache miss")
        return [], {}, {}
    if not isinstance(raw, dict):
        return [], {}, {}
    cards: list[CurationCard] = []
    for item in raw.get("cards") or ():
        try:
            cards.append(CurationCard.model_validate(item))
        except Exception:  # noqa: BLE001 — one bad card must not lose the rest
            continue
    if _blob_format(raw) < 2:
        cards = _upgrade_format_1(cards)
    first_seen = raw.get("first_seen")
    as_of_raw = raw.get("kind_as_of")
    as_of: dict[str, datetime] | None = None
    if isinstance(as_of_raw, dict):
        as_of = {}
        for kind, stamp in as_of_raw.items():
            parsed = _parse_stamp(stamp if isinstance(stamp, str) else None)
            if parsed is not None:
                as_of[str(kind)] = parsed
    return cards, first_seen if isinstance(first_seen, dict) else {}, as_of


def _row_built_at(row: CurationSnapshotRow) -> datetime:
    built = row.built_at
    return built if built.tzinfo is not None else built.replace(tzinfo=UTC)


def _legacy_kind_as_of(row: CurationSnapshotRow) -> dict[str, datetime]:
    """Each kind's as-of date for a row written before the envelope recorded it.

    Every finder ran at ``built_at`` except a semantic kind on a build whose
    semantic finders did not run in full; that kind has no date on record.
    """
    built = _row_built_at(row)
    probed = row.semantic and not row.semantic_degraded
    return {kind.value: built for kind in CardKind if probed or kind not in SEMANTIC_KINDS}


def read_prior(row: CurationSnapshotRow) -> PriorCollection:
    """Decode a stored collection into the merge's plain-value input."""
    cards, first_seen, as_of = _decode(row.cards_json)
    return PriorCollection(
        cards=cards,
        first_seen=first_seen,
        kind_as_of=as_of if as_of is not None else _legacy_kind_as_of(row),
        built_at=_row_built_at(row),
    )


def _blob_format(raw: dict[str, object]) -> int:
    value = raw.get("format", 1)
    return value if isinstance(value, int) else 1


def lacks_conflict_cards(row: CurationSnapshotRow) -> bool:
    """Whether ``row`` was written before open conflicts had their own cards.

    Such a collection is served with the conflict cards added from a live
    status scan until the next build writes the current format.
    """
    try:
        raw = json.loads(row.cards_json)
    except (TypeError, ValueError):
        return False
    return isinstance(raw, dict) and _blob_format(raw) < 2


def _upgrade_format_1(cards: list[CurationCard]) -> list[CurationCard]:
    """Bring a format-1 collection in line with the current card rules (pure).

    A ``CONTESTED`` card whose only basis is ``inconsistency`` is dropped (its
    conflict gets its own card, §4), and every card's gestures are re-read
    from the kind, which retires ``comment`` (§3).
    """
    out: list[CurationCard] = []
    for card in cards:
        bases = card.contested_bases
        # A basis-free card predates the composed badge, when the class was
        # the inconsistency basis alone.
        if card.kind is CardKind.CONTESTED and not [
            b for b in (bases or ()) if b != "inconsistency"
        ]:
            continue
        if "comment" in card.suggested_gestures:
            card.suggested_gestures = gestures_for(card.kind)
        out.append(card)
    return out


def load_collection(row: CurationSnapshotRow) -> list[CurationCard]:
    """The stored, already-scored collection."""
    cards, _ = _load(row.cards_json)
    return cards


def stale_after_hours(
    max_age_hours: float, *, census_enabled: bool, census_interval_hours: int
) -> float:
    """The age past which a stored collection is stamped stale (pure).

    ``curation.snapshot_max_age_hours`` gives a nightly build a day and a half.
    The cycle now builds the collection only when its census runs, every
    ``consolidation.census.interval_hours``, so a collection is
    stale only a full day past that cadence. Otherwise ``particles curate``
    would cry stale on most nights between censuses. With the census off the
    cycle never rebuilds, and the configured age stands.
    """
    if not census_enabled:
        return max_age_hours
    return max(max_age_hours, census_interval_hours + 24.0)


def stamp_from(
    row: CurationSnapshotRow,
    *,
    now: datetime | None = None,
    kind_as_of: dict[str, datetime] | None = None,
) -> CurationQueueResult:
    """Build the §5 staleness stamp for a stored collection (no cards yet).

    ``kind_as_of`` is the row's decoded per-kind as-of map when the caller has
    already read the blob (:func:`read_prior`); otherwise the blob is decoded
    here.
    """
    config = get_config()
    cfg = config.curation
    census = config.consolidation.census
    moment = now or datetime.now(UTC)
    built = _row_built_at(row)
    age = (moment - built).total_seconds()
    if kind_as_of is None:
        kind_as_of = read_prior(row).kind_as_of
    limit = stale_after_hours(
        cfg.snapshot_max_age_hours,
        census_enabled=census.enabled,
        census_interval_hours=census.interval_hours,
    )
    return CurationQueueResult(
        source=QueueSource.SNAPSHOT.value,
        snapshot_id=row.snapshot_id,
        built_at=built,
        age_seconds=age,
        stale=age > limit * 3600.0,
        scope=row.scope,
        per_kind_scope={k: v.value for k, v in parse_per_kind_scope(row).items()},
        semantic=row.semantic,
        semantic_degraded=row.semantic_degraded,
        kind_as_of=dict(kind_as_of),
    )


async def collect_and_persist(
    session: AsyncSession,
    *,
    semantic: bool,
    scope: CollectionScope = CollectionScope.STORE,
    duplicate_mode: SuggestMode | None = None,
    contradiction_probe: ContradictionProbeControl | None = None,
    duplicate_scope_ids: frozenset[str] | None = None,
    cards: list[CurationCard] | None = None,
    built_at: datetime | None = None,
    covered_pairs: frozenset[frozenset[str]] = frozenset(),
) -> tuple[list[CurationCard], str]:
    """Run the finders (or take a supplied collection), score, merge, persist.

    ``cards`` lets the pass-4 caller hand over the collection it
    already paid for — the whole point of that seam — instead of collecting a
    second time. Returns the merged collection and the new ``snapshot_id``.

    ``covered_pairs`` are the contradiction pairs a census record now discloses:
    a carried-forward CONTRADICTION card for one of them is
    dropped, since the record's own INCONSISTENCY card stands in its place.

    Commits the insert before releasing the writer lock. Left to the
    caller, the insert kept SQLite's write lock until the caller's next commit,
    which in the cycle came at the end of the utility pass, after every
    behavioural batch wait; each other writer meanwhile failed with ``database
    is locked``.
    """
    # Deferred import: session.py imports this module, and the scoring helpers
    # live beside it (AGENTS.md deferred-import case 1).
    from .collect import collect_cards
    from .leverage import contested_ids_from, score_cards

    fresh = (
        list(cards)
        if cards is not None
        else await collect_cards(
            session,
            semantic=semantic,
            duplicate_mode=duplicate_mode,
            contradiction_probe=contradiction_probe,
            duplicate_scope_ids=duplicate_scope_ids,
        )
    )
    # Score before merging: a carried-forward card keeps the leverage it was
    # scored with, and re-scoring the whole union would re-read the store for
    # beliefs this build did not look at.
    await score_cards(session, fresh, contested_ids=contested_ids_from(fresh))

    # The breaker is read once the collection is complete, as the stored
    # ``semantic_degraded`` flag always was: a probe it cut short did not run in
    # full, so its kind carries rather than replaces.
    degraded = semantic and llm_circuit_open()
    per_kind = per_kind_scope_for(scope, semantic=semantic and not degraded)
    now = built_at or datetime.now(UTC)
    cfg = get_config().curation
    horizon = timedelta(days=cfg.snapshot_carry_forward_days)
    prior_row = await latest_snapshot(session)
    prior = read_prior(prior_row) if prior_row is not None else None
    evidence = await _gather_carry_evidence(
        session, carry_candidates(fresh, prior, per_kind=per_kind), prior, now=now, horizon=horizon
    )
    result = _merge_with_prior(
        fresh,
        prior,
        per_kind=per_kind,
        now=now,
        horizon=horizon,
        covered_pairs=covered_pairs,
        evidence=evidence,
    )

    # writer lock around the **insert and its commit** — never around the
    # collection above, which is minutes of pure reads and would block every
    # other writer for its duration. A lost race is harmless by
    # construction: two rebuilds both write valid caches, the newest is served,
    # and the retention ring prunes the loser.
    async with write_transaction(session):
        snapshot_id = await write_snapshot(
            session,
            cards_json=_dump(result.cards, result.first_seen, result.kind_as_of),
            card_count=len(result.cards),
            scope=scope,
            per_kind_scope={k: v.value for k, v in per_kind.items()},
            semantic=semantic,
            semantic_degraded=degraded,
            built_at=now,
            retain=cfg.snapshot_retain,
        )
    return result.cards, snapshot_id


def carry_candidates(
    fresh: list[CurationCard],
    prior: PriorCollection | None,
    *,
    per_kind: dict[str, CollectionScope],
) -> list[CurationCard]:
    """The prior cards a build would carry, before any eviction rule (pure).

    A prior card is a candidate when its kind carries in this build (``delta``
    or ``carried``) and the fresh build did not re-find it.
    """
    if prior is None:
        return []
    fresh_keys = {c.key for c in fresh}
    return [
        card
        for card in prior.cards
        if card.key not in fresh_keys
        and per_kind.get(card.kind.value, CollectionScope.STORE) in _CARRYING_SCOPES
    ]


def _members(card: CurationCard) -> list[str]:
    """The beliefs a card is about: its particles, plus a contradiction's partner."""
    ids = list(card.particle_ids)
    if card.kind is CardKind.CONTRADICTION and (partner := contradiction_partner(card.diagnostic)):
        ids.append(partner)
    return ids


async def _gather_carry_evidence(
    session: AsyncSession,
    candidates: list[CurationCard],
    prior: PriorCollection | None,
    *,
    now: datetime,
    horizon: timedelta,
) -> CarryEvidence:
    """Read what the §4 eviction rules need about the carry candidates (the shell).

    Suppression and resolution are read exactly as the live session half reads
    them, from the event log; belief status from the store. A
    belief the store no longer holds is left to the live half, which drops a
    card naming one at read time.
    """
    if not candidates or prior is None:
        return CarryEvidence()
    # Deferred import: session.py imports this module (AGENTS.md deferred-import
    # case 1).
    from .session import _resolutions_since, _suppressed_keys

    origins = [_origin(card, prior, now=now) for card in candidates]
    # A resolution older than the horizon cannot matter: any card found before
    # it ages out in this same merge.
    since = max(min(origins), now - horizon)
    beliefs = await get_particles_by_ids(
        session, [pid for card in candidates for pid in _members(card)]
    )
    return CarryEvidence(
        suppressed=frozenset(await _suppressed_keys(session)),
        resolutions=tuple(await _resolutions_since(session, since)),
        retired=frozenset(pid for pid, p in beliefs.items() if p.status is not Status.ACTIVE),
    )


def _origin(card: CurationCard, prior: PriorCollection, *, now: datetime) -> datetime:
    """When a prior card was first found: its stamp, else the prior build's time."""
    return _parse_stamp(prior.first_seen.get(card.key)) or prior.built_at or now


def _merge_with_prior(
    fresh: list[CurationCard],
    prior: PriorCollection | None,
    *,
    per_kind: dict[str, CollectionScope],
    now: datetime,
    horizon: timedelta,
    covered_pairs: frozenset[frozenset[str]] = frozenset(),
    evidence: CarryEvidence | None = None,
) -> MergeResult:
    """Apply the §4 replace-vs-carry-forward rule (pure).

    Store-wide kinds: the fresh set wins outright. Delta-scoped and carried
    kinds: prior cards the fresh build did not re-find are kept, unless a §4
    eviction rule drops them: they age out of ``horizon`` (measured from the
    card's origin stamp, so a card carried across five nights still expires on
    its own thirtieth day rather than resetting each build), a census record
    now discloses the pair, a snooze, dismiss or affirmation hides the card, a
    resolving event postdates the card's origin, or a belief it names is
    retired. The live session half filters the last three at read time too,
    but only for events after the *served* build, so a carried card must be
    filtered here or a resolution between the two builds would be forgotten.

    Each kind's as-of date is ``now`` when this build ran its finder and the
    prior collection's date when the kind is ``carried``.
    """
    stamp = now.isoformat()
    first_seen: dict[str, str] = {c.key: stamp for c in fresh}
    kind_as_of: dict[str, datetime] = {}
    for kind in CardKind:
        if per_kind.get(kind.value, CollectionScope.STORE) is not CollectionScope.CARRIED:
            kind_as_of[kind.value] = now
        elif prior is not None and (was := prior.kind_as_of.get(kind.value)) is not None:
            kind_as_of[kind.value] = was

    if prior is None:
        return MergeResult(list(fresh), first_seen, kind_as_of)

    # A card the prior snapshot also had keeps its original stamp.
    for card in fresh:
        if (was_seen := prior.first_seen.get(card.key)) is not None:
            first_seen[card.key] = was_seen

    gone = evidence or CarryEvidence()
    carried = 0
    out = list(fresh)
    for card in carry_candidates(fresh, prior, per_kind=per_kind):
        origin = _origin(card, prior, now=now)
        if now - origin > horizon:
            continue  # aged out — gone until a probe re-finds it
        if covered_pairs and _disclosed(card, covered_pairs):
            continue  # a census record now discloses it
        if card.key in gone.suppressed:
            continue
        if any(at > origin and r.covers(card) for at, r in gone.resolutions):
            continue
        if any(pid in gone.retired for pid in _members(card)):
            continue
        out.append(card)
        first_seen[card.key] = origin.isoformat()
        carried += 1

    if carried:
        log.debug("curation snapshot: carried forward %d card(s)", carried)
    return MergeResult(out, first_seen, kind_as_of)


def _disclosed(card: CurationCard, covered_pairs: frozenset[frozenset[str]]) -> bool:
    """Whether a CONTRADICTION card's pair is one a census record discloses."""
    if card.kind is not CardKind.CONTRADICTION or not card.particle_ids:
        return False
    partner = contradiction_partner(card.diagnostic)
    return partner is not None and frozenset((card.particle_ids[0], partner)) in covered_pairs


def _parse_stamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)

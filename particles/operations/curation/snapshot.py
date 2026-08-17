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
the delta) and contradiction findings never persist as INCONSISTENCY particles.
Without the carry-forward the queue would forget yesterday's contradictions
every night.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import SuggestMode
from particles.db import write_lock
from particles.operations._llm import llm_circuit_open
from particles.operations.lint import ContradictionProbeControl
from particles.store.curation_snapshot_store import (
    CollectionScope,
    CurationSnapshotRow,
    latest_snapshot,
    parse_per_kind_scope,
    write_snapshot,
)

from .cards import CardKind, CurationCard

log = logging.getLogger(__name__)

#: Serialisation version of the ``cards_json`` envelope.
BLOB_FORMAT = 1

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


class QueueSource(StrEnum):
    """Where `build_curation_queue` should get its collection."""

    SNAPSHOT = "snapshot"
    LIVE = "live"


class CurationQueueResult(BaseModel):
    """The ranked queue plus the staleness stamp.

    Consumers render the stamp rather than hiding it: an operator who can see
    ``built_at`` can tell a quiet queue from a stale one. ``stale`` is the
    convenience verdict (``age`` past ``curation.snapshot_max_age_hours``);
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


def per_kind_scope_for(scope: CollectionScope) -> dict[str, CollectionScope]:
    """The §4 per-kind scope map for a build of the given overall scope.

    A store-wide build is store-wide for every kind. A delta build is
    delta-scoped only for :data:`DELTA_SCOPED_KINDS`; every other finder still
    walked the whole store, so those kinds still replace.
    """
    if scope is CollectionScope.STORE:
        return {kind.value: CollectionScope.STORE for kind in CardKind}
    return {
        kind.value: (CollectionScope.DELTA if kind in DELTA_SCOPED_KINDS else CollectionScope.STORE)
        for kind in CardKind
    }


def _dump(cards: list[CurationCard], first_seen: dict[str, str]) -> str:
    """Serialise a collection + its per-card origin stamps."""
    return json.dumps(
        {
            "format": BLOB_FORMAT,
            "cards": [c.model_dump(mode="json") for c in cards],
            "first_seen": first_seen,
        }
    )


def _load(blob: str) -> tuple[list[CurationCard], dict[str, str]]:
    """Decode a collection blob, tolerating an unreadable one as empty.

    A cache that cannot be read is a cache miss, never an error: the caller
    rebuilds. Returning ``([], {})`` keeps a corrupt row from taking the
    curation surface down with it.
    """
    try:
        raw = json.loads(blob)
    except (TypeError, ValueError):
        log.warning("curation snapshot: unreadable cards_json; treating as a cache miss")
        return [], {}
    if not isinstance(raw, dict):
        return [], {}
    cards: list[CurationCard] = []
    for item in raw.get("cards") or ():
        try:
            cards.append(CurationCard.model_validate(item))
        except Exception:  # noqa: BLE001 — one bad card must not lose the rest
            continue
    first_seen = raw.get("first_seen")
    return cards, first_seen if isinstance(first_seen, dict) else {}


def load_collection(row: CurationSnapshotRow) -> list[CurationCard]:
    """The stored, already-scored collection."""
    cards, _ = _load(row.cards_json)
    return cards


def stamp_from(row: CurationSnapshotRow, *, now: datetime | None = None) -> CurationQueueResult:
    """Build the §5 staleness stamp for a stored collection (no cards yet)."""
    cfg = get_config().curation
    moment = now or datetime.now(UTC)
    built = row.built_at if row.built_at.tzinfo is not None else row.built_at.replace(tzinfo=UTC)
    age = (moment - built).total_seconds()
    return CurationQueueResult(
        source=QueueSource.SNAPSHOT.value,
        snapshot_id=row.snapshot_id,
        built_at=built,
        age_seconds=age,
        stale=age > cfg.snapshot_max_age_hours * 3600.0,
        scope=row.scope,
        per_kind_scope={k: v.value for k, v in parse_per_kind_scope(row).items()},
        semantic=row.semantic,
        semantic_degraded=row.semantic_degraded,
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
) -> tuple[list[CurationCard], str]:
    """Run the finders (or take a supplied collection), score, merge, persist.

    ``cards`` lets the pass-4 caller hand over the collection it
    already paid for — the whole point of that seam — instead of collecting a
    second time. Returns the merged collection and the new ``snapshot_id``.

    Does not commit; the caller owns the transaction.
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

    per_kind = per_kind_scope_for(scope)
    now = built_at or datetime.now(UTC)
    prior = await latest_snapshot(session)
    merged, first_seen = _merge_with_prior(fresh, prior, per_kind=per_kind, now=now)

    cfg = get_config().curation
    # writer lock around the **insert only** — never around the
    # collection above, which is minutes of pure reads and would block every
    # other writer for its duration. A lost race is harmless by
    # construction: two rebuilds both write valid caches, the newest is served,
    # and the retention ring prunes the loser.
    async with write_lock():
        snapshot_id = await write_snapshot(
            session,
            cards_json=_dump(merged, first_seen),
            card_count=len(merged),
            scope=scope,
            per_kind_scope={k: v.value for k, v in per_kind.items()},
            semantic=semantic,
            semantic_degraded=semantic and llm_circuit_open(),
            built_at=now,
            retain=cfg.snapshot_retain,
        )
    return merged, snapshot_id


def _merge_with_prior(
    fresh: list[CurationCard],
    prior: CurationSnapshotRow | None,
    *,
    per_kind: dict[str, CollectionScope],
    now: datetime,
) -> tuple[list[CurationCard], dict[str, str]]:
    """Apply the §4 replace-vs-carry-forward rule.

    Store-wide kinds: the fresh set wins outright. Delta-scoped kinds: prior
    cards the fresh build did not re-find are kept, until they age out of
    ``curation.snapshot_carry_forward_days``. The origin stamp (``first_seen``)
    is what the horizon is measured from, so a card carried across five nights
    still expires on its own thirtieth day rather than resetting each build.
    """
    stamp = now.isoformat()
    fresh_keys = {c.key for c in fresh}
    first_seen: dict[str, str] = {c.key: stamp for c in fresh}

    if prior is None:
        return list(fresh), first_seen

    prior_cards, prior_seen = _load(prior.cards_json)
    # A card the prior snapshot also had keeps its original stamp.
    for card in fresh:
        if (was := prior_seen.get(card.key)) is not None:
            first_seen[card.key] = was

    horizon = timedelta(days=get_config().curation.snapshot_carry_forward_days)
    carried = 0
    out = list(fresh)
    for card in prior_cards:
        if card.key in fresh_keys:
            continue
        if per_kind.get(card.kind.value, CollectionScope.STORE) is not CollectionScope.DELTA:
            # This build's finder covered the whole store for this kind, so its
            # silence is authoritative: the card is resolved, not merely unseen.
            continue
        origin = _parse_stamp(prior_seen.get(card.key)) or now
        if now - origin > horizon:
            continue  # aged out — gone until a probe re-finds it
        out.append(card)
        first_seen[card.key] = origin.isoformat()
        carried += 1

    if carried:
        log.debug("curation snapshot: carried forward %d delta-scoped card(s)", carried)
    return out, first_seen


def _parse_stamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)

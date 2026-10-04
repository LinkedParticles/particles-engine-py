# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pass 3b of the nightly cycle: disclose confirmed contradictions.

The census (pass 3) finds contradictions and a second reading confirms them.
Before this pass, a confirmed pair became only a curation card,
which nothing the agent reads ever shows. This pass turns each confirmed
disagreement between two sources into one open ``INCONSISTENCY`` record, so
the session-start digest flags every claim in it, and keeps those records
honest afterwards:

0. **Re-reading.** On a night the census's second reading ran, each pair of
   an open record that was confirmed under an instruction it would not get
   now is read again. A pair the new reading rejects is withdrawn.
   Then, from the same budget, each open record **extraction** opened without
   a confirming second reading is read: confirmed, it is
   stamped and not read again; not confirmed, it closes as ``withdrawn`` and
   its quarantined newcomer's claim returns to the store.
1. **Lapse sweep.** Every open census record is re-read. A record whose
   disagreement is gone (no confirmed pair has both claims live and stated,
   or a re-reading withdrew the last one) closes as ``lapsed`` or
   ``withdrawn``; one whose group changed, or whose pairs were re-read, is
   closed as ``regrouped`` and replaced. Each close writes an
   ``INCONSISTENCY_CLOSED`` event. Runs on every night the pass is reached,
   degraded and census-failed nights too.
2. **Mint.** The confirmed pairs (this run's, then any waiting from the last
   run) are filtered (cross-source, live, not covered, in view together for
   some observer), grouped by shared claim, split into two sides, and opened
   under the per-run cap. Disclosure only: no claim's status or confidence
   changes.

It never runs a resolving rung and never writes a relation. The pure
decisions live in :mod:`particles.core.contradiction_disclosure`; this module
gathers their inputs and applies their writes.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, NamedTuple

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.conflict_review import is_quarantined, is_retired_value
from particles.core.contradiction_disclosure import (
    DISCLOSURE_ACTOR,
    READING_KEY,
    CloseCause,
    ConfirmedPair,
    LapseAction,
    MemberState,
    NoteLabel,
    PlannedRecord,
    RecordState,
    Sides,
    build_census_record,
    census_pairs,
    census_sides,
    census_sources,
    covered_pairs,
    decide_lapse,
    group_pairs,
    is_census_record,
    plan_group,
    reading_stamp,
    select_under_cap,
)
from particles.core.observer_scope import GLOBAL_SCOPE, BeliefScope, share_an_observer
from particles.core.schema import Particle, ProvenanceRefType
from particles.core.status import Status, StatusReason, validate_transition
from particles.corpus.store import get_entry, get_entry_currency, get_snapshot
from particles.db import write_lock
from particles.extraction.registry import infer_domain
from particles.ingest.duplicate_suppression import build_duplicate_index
from particles.ingest.second_reading import (
    ClaimContext,
    claim_context,
    read_pair,
    reading_for,
    snapshot_claim_context,
)
from particles.llm.breaker import llm_circuit_open
from particles.operations._quarantine import promote_quarantined
from particles.operations.lint.contradictions import RereadOutcome, reread_stale_pairs
from particles.operations.lint.open_inconsistency import record_sides
from particles.store.event_store import (
    EventRefKind,
    OperatorEventType,
    list_events,
    list_particle_events,
    record_event,
)
from particles.store.observer_scope_join import SourceRef, attests, lens_may_engage, load_scopes
from particles.store.observer_scope_store import ScopeTarget, any_widening, widened_ids
from particles.store.particle_store import (
    append_provenance_ref,
    get_census_records,
    get_inconsistency_particles,
    get_particle,
    get_particles_by_ids,
    insert_particle,
    set_particle_property,
    update_particle_status,
)

log = logging.getLogger(__name__)

#: How many recent run records are searched for the last one's waiting list.
_WAITING_LOOKBACK = 20


class DisclosureReport(BaseModel):
    """What pass 3b did: the run record's ``disclosure`` block."""

    enabled: bool = True
    #: Whether the minting half ran (it needs a census result or a waiting list).
    minted: bool = True
    #: Why nothing minted, when ``minted`` is False or nothing was confirmed.
    not_minting_reason: str | None = None
    #: One entry per new disagreement: ``{"record_ids": [...], "members": n}``.
    opened: list[dict[str, Any]] = Field(default_factory=list)
    #: One entry per record closed: ``{"record_id", "cause", "dropped", "replacements"}``.
    closed: list[dict[str, Any]] = Field(default_factory=list)
    #: The confirmed pairs past the cap, with their reasons, for the next run.
    waiting: list[dict[str, Any]] = Field(default_factory=list)
    #: Pairs that were waiting from an earlier run and no longer qualify, each
    #: with the reason it was dropped (a dropped pair is named).
    dropped_waiting: list[dict[str, Any]] = Field(default_factory=list)
    #: Groups that could not be split into two sides and fell back to per-pair records.
    unsided: int = 0
    skipped: dict[str, int] = Field(default_factory=dict)
    cap: int = 0
    #: Open census records after the pass.
    open_records: int = 0
    #: The re-reading of pairs confirmed under an earlier instruction:
    #: ``read``, ``confirmed``, ``withdrawn``, ``failed``, ``deferred``.
    reread: dict[str, int] = Field(default_factory=dict)
    #: The re-reading of records extraction opened without a confirming second
    #: reading, same keys. Shares ``max_rereadings_per_run``.
    extract_reread: dict[str, int] = Field(default_factory=dict)
    #: Claims whose flag this pass changed; the caller refreshes their cards.
    touched_ids: list[str] = Field(default_factory=list, exclude=True)

    def payload(self) -> dict[str, Any]:
        """The run record's ``disclosure`` field."""
        return self.model_dump(mode="json")

    @property
    def opened_records(self) -> int:
        return sum(len(o.get("record_ids", ())) for o in self.opened)


def _skip(report: DisclosureReport, reason: str, count: int = 1) -> None:
    if count:
        report.skipped[reason] = report.skipped.get(reason, 0) + count


# ---------------------------------------------------------------------------
# Gather
# ---------------------------------------------------------------------------


async def _record_states(
    session: AsyncSession,
) -> tuple[list[tuple[Particle, Sides]], list[RecordState]]:
    """Every census record with its sides, and each one's coverage state.

    The close cause is read from what each close path leaves: an
    ``INCONSISTENCY_CLOSED`` event for a lapse or regroup, a ``REVIEW_RESOLVED``
    event for a review, and the ``PROVENANCE_STALE`` status the trust cascade
    leaves without an event. A closed record with none of these marks is read
    as a judgment, which errs toward not raising the disagreement again.
    """
    records = await get_census_records(session)
    if not records:
        return [], []
    reviewed = {
        pid for pid, _, _ in await list_particle_events(session, OperatorEventType.REVIEW_RESOLVED)
    }
    swept: dict[str, CloseCause] = {}
    for pid, _, payload in await list_particle_events(
        session, OperatorEventType.INCONSISTENCY_CLOSED
    ):
        if isinstance(payload, dict) and payload.get("record_id") == pid:
            cause = payload.get("cause")
            if cause in (
                CloseCause.LAPSED.value,
                CloseCause.REGROUPED.value,
                CloseCause.WITHDRAWN.value,
            ):
                swept[pid] = CloseCause(cause)

    open_records: list[tuple[Particle, Sides]] = []
    states: list[RecordState] = []
    for record in records:
        sides = census_sides(record)
        if sides is None:
            continue
        if record.status is Status.INCONSISTENCY:
            open_records.append((record, sides))
            states.append(RecordState(record_id=record.id, sides=sides, open=True))
            continue
        if record.id in swept:
            cause = swept[record.id]
        elif record.id in reviewed:
            cause = CloseCause.REVIEW
        elif record.status is Status.PROVENANCE_STALE:
            cause = CloseCause.CASCADE
        else:
            cause = CloseCause.REVIEW
        states.append(
            RecordState(
                record_id=record.id,
                sides=sides,
                open=False,
                cause=cause,
                fingerprint=census_sources(record),
            )
        )
    return open_records, states


def _source_refs(particle: Particle) -> list[SourceRef]:
    return [
        SourceRef(r.corpus_entry_id, r.snapshot_id)
        for r in particle.provenance
        if r.type is not ProvenanceRefType.PARTICLE and r.corpus_entry_id
    ]


async def current_sources(
    session: AsyncSession, particles: Iterable[Particle]
) -> dict[str, frozenset[str]]:
    """The corpus entries that currently state each belief.

    The ``conflict:sources`` fingerprint, taken at mint and compared against
    for a judged record's coverage. A belief's own refs only:
    a new source restating it is what the fingerprint watches for.
    """
    refs = {p.id: _source_refs(p) for p in particles}
    entries = {r.entry_id for rs in refs.values() for r in rs}
    if not entries:
        return {pid: frozenset() for pid in refs}
    currency = await get_entry_currency(session, entries)
    return {
        pid: frozenset(
            r.entry_id for r in rs if r.entry_id in currency and attests(r, currency[r.entry_id])
        )
        for pid, rs in refs.items()
    }


async def covered_pair_set(session: AsyncSession) -> frozenset[frozenset[str]]:
    """The pairs a census record already discloses, so no probe pays for them again.

    Pairs on opposite sides of an open record, or of a judged record whose
    claims have gained no new stating source. Empty on a store
    with no census records, at the cost of one filtered status scan.
    """
    _, states = await _record_states(session)
    if not states:
        return frozenset()
    judged = {
        pid
        for s in states
        if not s.open and s.cause is not None and s.cause.judgment
        for pid in s.sides.members
    }
    loaded = await get_particles_by_ids(session, sorted(judged)) if judged else {}
    current = await current_sources(session, loaded.values())
    return covered_pairs(states, current)


async def _other_open_pairs(session: AsyncSession) -> frozenset[frozenset[str]]:
    """Pairs of claims both named by one open non-census INCONSISTENCY (a §6.6 record)."""
    pairs: set[frozenset[str]] = set()
    for record in await get_inconsistency_particles(session):
        if is_census_record(record):
            continue
        ids = [r.corpus_entry_id for r in record.provenance if r.type is ProvenanceRefType.PARTICLE]
        pairs |= {frozenset((x, y)) for i, x in enumerate(ids) for y in ids[i + 1 :] if x != y}
    return frozenset(pairs)


async def _merge_survivors(session: AsyncSession, ids: set[str]) -> dict[str, str]:
    """For members folded by the exact-duplicate auto-merge, their survivor."""
    out: dict[str, str] = {}
    if not ids:
        return out
    for _pid, _when, payload in await list_particle_events(
        session, OperatorEventType.DUPLICATES_MERGED
    ):
        if not isinstance(payload, dict):
            continue
        survivor = payload.get("survivor")
        if not isinstance(survivor, str):
            continue
        for folded in payload.get("superseded") or ():
            if isinstance(folded, str) and folded in ids:
                out[folded] = survivor
    return out


def _is_lapsed(scope: BeliefScope | None) -> bool:
    return scope is not None and scope.lapsed and scope.in_view_for_no_project


@dataclass
class _View:
    """Particles, their scopes, and what the record needs to name them."""

    particles: dict[str, Particle] = field(default_factory=dict)
    scopes: dict[str, BeliefScope] = field(default_factory=dict)

    def live(self, pid: str) -> bool:
        p = self.particles.get(pid)
        return p is not None and p.status is Status.ACTIVE and not _is_lapsed(self.scopes.get(pid))


async def _view(session: AsyncSession, ids: Iterable[str]) -> _View:
    loaded = await get_particles_by_ids(session, sorted(set(ids)))
    active = [p for p in loaded.values() if p.status is Status.ACTIVE]
    scopes = await load_scopes(session, active) if active else {}
    return _View(particles=loaded, scopes=scopes)


@dataclass(frozen=True)
class _Origin:
    """Where a record's content, trigger and domain come from for one member."""

    label: NoteLabel
    when: datetime | None
    entry_id: str | None
    snapshot_id: str | None
    domain: str | None


async def _origins(session: AsyncSession, particles: Iterable[Particle]) -> dict[str, _Origin]:
    """Each member's latest stating source: its note name, date, entry and snapshot."""
    out: dict[str, _Origin] = {}
    for p in particles:
        ref = next((r for r in reversed(p.provenance) if r.type is ProvenanceRefType.SOURCE), None)
        if ref is None or not ref.corpus_entry_id:
            out[p.id] = _Origin(NoteLabel("agent assertion", "unknown"), None, None, None, None)
            continue
        entry = await get_entry(session, ref.corpus_entry_id)
        snapshot = await get_snapshot(session, ref.snapshot_id) if ref.snapshot_id else None
        when = (snapshot.content_published_at or snapshot.captured_at) if snapshot else None
        name = "unknown"
        if entry is not None and entry.uri_r:
            name = entry.uri_r.rstrip("/").rsplit("/", 1)[-1] or entry.uri_r
        out[p.id] = _Origin(
            label=NoteLabel(name, when.date().isoformat() if when else "unknown"),
            when=when,
            entry_id=ref.corpus_entry_id,
            snapshot_id=ref.snapshot_id,
            domain=infer_domain(entry.source_type) if entry is not None else None,
        )
    return out


async def _prior_waiting(session: AsyncSession, actor: str) -> list[ConfirmedPair]:
    """The pairs the last run of this actor left waiting past its cap."""
    events = await list_events(
        session, event_type=OperatorEventType.CONSOLIDATION_RUN, limit=_WAITING_LOOKBACK
    )
    for event in events:
        if event.actor != actor:
            continue
        census = (event.payload or {}).get("census")
        disclosure = census.get("disclosure") if isinstance(census, dict) else None
        if isinstance(disclosure, dict):
            raw = disclosure.get("waiting") or []
            return [
                p
                for p in (ConfirmedPair.from_payload(r) for r in raw if isinstance(r, dict))
                if p is not None
            ]
    return []


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


async def _mint(
    session: AsyncSession,
    planned: PlannedRecord,
    view: _View,
    origins: dict[str, _Origin],
    sources: dict[str, frozenset[str]],
    *,
    replaces: str | None,
) -> Particle:
    """Build and insert one census record."""
    b_id = planned.sides.b[0]
    trigger = origins.get(b_id)
    if trigger is None or trigger.entry_id is None:
        trigger = origins.get(planned.sides.a[0])
    if trigger is not None and trigger.entry_id is not None:
        entry_id, snapshot_id, ref_type = (
            trigger.entry_id,
            trigger.snapshot_id,
            ProvenanceRefType.SOURCE,
        )
    else:
        # Neither representative claim has corpus provenance (agent assertions):
        # the trigger ref names B itself, as the rung 3 builder does for a
        # derived candidate.
        entry_id, snapshot_id, ref_type = b_id, None, ProvenanceRefType.PARTICLE
    record = build_census_record(
        sides=planned.sides,
        pairs=planned.pairs,
        members=view.particles,
        labels={pid: o.label for pid, o in origins.items()},
        sources=sources,
        trigger_entry_id=entry_id,
        trigger_snapshot_id=snapshot_id,
        trigger_ref_type=ref_type,
        replaces=replaces,
    )
    validate_transition(None, Status.INCONSISTENCY)
    domain = trigger.domain if trigger is not None else None
    await insert_particle(session, record, domain_hint=domain)
    return record


async def _close(
    session: AsyncSession,
    record: Particle,
    cause: CloseCause,
    *,
    dropped: Sequence[str],
    states: dict[str, MemberState],
    view: _View,
    replacements: Sequence[str] = (),
    withdrawn: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Close one census record without a review and write its event.

    ``withdrawn`` names the pairs a re-reading no longer confirms, each with
    the new reading's reason.
    """
    await update_particle_status(
        session, record.id, Status.RETRACTED, StatusReason.CONFLICT_RESOLVED
    )
    detail: list[dict[str, Any]] = []
    for pid in dropped:
        p = view.particles.get(pid)
        detail.append(
            {
                "particle_id": pid,
                "state": states.get(pid, MemberState.GONE).value,
                "status": p.status.value if p is not None else None,
            }
        )
    lapsed_ids = [d["particle_id"] for d in detail if d["state"] == MemberState.LAPSED.value]
    if lapsed_ids:
        # Name the snapshot that stopped stating each lapsed member: the latest
        # extracted generation of every entry that once stated it.
        entries = {
            r.entry_id
            for pid in lapsed_ids
            if (p := view.particles.get(pid)) is not None
            for r in _source_refs(p)
        }
        currency = await get_entry_currency(session, entries) if entries else {}
        for d in detail:
            if d["state"] != MemberState.LAPSED.value:
                continue
            p = view.particles.get(d["particle_id"])
            d["stopped_by"] = sorted(
                {
                    c.latest_snapshot_id
                    for r in (_source_refs(p) if p is not None else [])
                    if (c := currency.get(r.entry_id)) is not None and c.latest_snapshot_id
                }
            )
    payload: dict[str, Any] = {
        "record_id": record.id,
        "cause": cause.value,
        "dropped": detail,
        "replacements": list(replacements),
    }
    if withdrawn:
        payload["withdrawn"] = list(withdrawn)
    await record_event(
        session,
        actor=DISCLOSURE_ACTOR,
        event_type=OperatorEventType.INCONSISTENCY_CLOSED,
        refs=[(EventRefKind.PARTICLE, record.id), *((EventRefKind.PARTICLE, d) for d in dropped)],
        payload=payload,
    )
    return payload


class _Regroup(NamedTuple):
    """An open record the pass closes and replaces, with what its event names."""

    record: Particle
    pairs: list[ConfirmedPair]
    states: dict[str, MemberState]
    dropped: tuple[str, ...]
    withdrawn: list[dict[str, Any]]


async def _reread_open_records(session: AsyncSession, budget: int) -> RereadOutcome:
    """Re-read the stale pairs of every open census record (gather + LLM).

    Only pairs whose two claims are both ``ACTIVE`` are read; a pair with a
    member gone is the lapse sweep's. Oldest record first, so a backlog past
    the cap drains in the order it was opened.
    """
    open_records, _ = await _record_states(session)
    open_records.sort(key=lambda rs: rs[0].asserted_at)
    pairs: list[ConfirmedPair] = []
    seen: set[frozenset[str]] = set()
    for record, _sides in open_records:
        for pair in census_pairs(record):
            if pair.key not in seen:
                seen.add(pair.key)
                pairs.append(pair)
    ids = sorted({pid for p in pairs for pid in (p.a, p.b)})
    loaded = await get_particles_by_ids(session, ids) if ids else {}
    live = {pid: p for pid, p in loaded.items() if p.status is Status.ACTIVE}
    outcome = await reread_stale_pairs(session, pairs, live, budget=budget)
    if outcome.read:
        log.info(
            "contradiction disclosure: re-read %d pair(s) under a revised instruction; "
            "%d withdrawn, %d confirmed, %d failed, %d deferred",
            outcome.read,
            outcome.withdrawn,
            outcome.confirmed,
            outcome.failed,
            outcome.deferred,
        )
    return outcome


@dataclass
class _ExtractReading:
    """One extract-time record, its two claims, and what the reading said."""

    record: Particle
    a: Particle
    b: Particle
    reading: str
    #: ``None`` when the reading failed; a later night reads it again.
    confirmed: bool | None = None
    reason: str = ""


@dataclass
class ExtractRereadOutcome:
    """What re-reading the records extraction opened found."""

    readings: list[_ExtractReading] = field(default_factory=list)
    read: int = 0
    confirmed: int = 0
    withdrawn: int = 0
    failed: int = 0
    deferred: int = 0

    def payload(self) -> dict[str, int]:
        return {
            "read": self.read,
            "confirmed": self.confirmed,
            "withdrawn": self.withdrawn,
            "failed": self.failed,
            "deferred": self.deferred,
        }


def _awaits_reading(record: Particle) -> bool:
    """An open record extraction opened on the probe alone.

    Not a census record (those carry their readings per pair), not a
    retired-value hold (a re-assertion of a retired claim, not a probed
    contradiction; it stays with review), and not stamped by a
    confirming reading.
    """
    return (
        record.status is Status.INCONSISTENCY
        and not is_census_record(record)
        and not is_retired_value(record)
        and reading_stamp(record) is None
    )


async def _reread_extract_records(session: AsyncSession, budget: int) -> ExtractRereadOutcome:
    """Read each unstamped extract-time record a second time (gather + LLM).

    Oldest first, at most ``budget`` readings. Claim A must be ``ACTIVE``: a
    record whose A has left the surface is left to review. Claim B, the
    newcomer, is usually quarantined, and is read from its own snapshot with
    the chunk windowed around both claims (§1's builder), because the census's
    first best-matching paragraph can show a different run of a command a
    transcript repeats. A's context is the census's. Every context is
    gathered before the first call, and the read transaction ends before it.
    Read-only: the caller applies the verdicts.
    """
    out = ExtractRereadOutcome()
    records = sorted(
        (r for r in await get_inconsistency_particles(session) if _awaits_reading(r)),
        key=lambda r: r.asserted_at,
    )
    planned: list[tuple[Particle, Particle, Particle, ClaimContext, ClaimContext]] = []
    for record in records:
        a_ids, b_ids = record_sides(record)
        if len(a_ids) != 1 or len(b_ids) != 1:
            continue
        a = await get_particle(session, a_ids[0])
        b = await get_particle(session, b_ids[0])
        if a is None or b is None or a.status is not Status.ACTIVE:
            continue
        if len(planned) >= budget:
            out.deferred += 1
            continue
        ctx_a = await claim_context(session, a, b)
        ctx_b = await snapshot_claim_context(session, b, a)
        planned.append((record, a, b, ctx_a, ctx_b))
    await session.commit()

    for index, (record, a, b, ctx_a, ctx_b) in enumerate(planned):
        if llm_circuit_open():
            out.deferred += len(planned) - index
            break
        verdict = await read_pair(ctx_a, ctx_b)
        out.read += 1
        reading = _ExtractReading(record, a, b, reading_for(ctx_a, ctx_b))
        if verdict is None:
            out.failed += 1
            continue
        reading.confirmed = verdict.contradicts
        reading.reason = verdict.description
        if verdict.contradicts:
            out.confirmed += 1
        else:
            out.withdrawn += 1
        out.readings.append(reading)
    if out.read or out.deferred:
        log.info(
            "contradiction disclosure: re-read %d record(s) extraction opened without a "
            "second reading; %d withdrawn, %d confirmed, %d failed, %d deferred",
            out.read,
            out.withdrawn,
            out.confirmed,
            out.failed,
            out.deferred,
        )
    return out


async def _apply_extract_reading(
    session: AsyncSession, r: _ExtractReading
) -> dict[str, Any] | None:
    """Stamp a confirmed record, or withdraw an unconfirmed one.

    The record is re-read first: a record a review closed since the reading
    is left alone. A withdrawal closes the record, restores the newcomer's
    claim and writes one ``INCONSISTENCY_CLOSED`` event; it returns the
    event's payload. A newcomer an ACTIVE claim already states verbatim with
    the same subjects (the exact-duplicate index) has its provenance
    folded into that claim; otherwise :func:`promote_quarantined` mints a
    successor, which enters without a §6.6 check. The existing claim is not
    touched.
    """
    record = await get_particle(session, r.record.id)
    if record is None or record.status is not Status.INCONSISTENCY:
        return None
    if r.confirmed:
        await set_particle_property(session, record.id, READING_KEY, r.reading)
        return None
    await update_particle_status(
        session, record.id, Status.RETRACTED, StatusReason.CONFLICT_RESOLVED
    )
    newcomer = await get_particle(session, r.b.id)
    restored: dict[str, Any] = {}
    refs: list[str] = [r.a.id, r.b.id]
    if newcomer is not None and is_quarantined(newcomer):
        twin = (await build_duplicate_index(session, [newcomer.content])).find(newcomer)
        if twin is not None:
            for ref in newcomer.provenance:
                await append_provenance_ref(session, twin.id, ref)
            await update_particle_status(
                session, newcomer.id, Status.SUPERSEDED, StatusReason.CONFLICT_RESOLVED
            )
            restored = {"folded_into": twin.id}
            refs.append(twin.id)
        else:
            successor = await promote_quarantined(session, newcomer)
            restored = {"successor": successor.id}
            refs.append(successor.id)
    payload: dict[str, Any] = {
        "record_id": record.id,
        "cause": CloseCause.WITHDRAWN.value,
        "origin": "extraction",
        "reading": r.reading,
        "withdrawn_reason": r.reason,
        "newcomer": r.b.id,
        **restored,
    }
    await record_event(
        session,
        actor=DISCLOSURE_ACTOR,
        event_type=OperatorEventType.INCONSISTENCY_CLOSED,
        refs=[(EventRefKind.PARTICLE, record.id), *((EventRefKind.PARTICLE, pid) for pid in refs)],
        payload=payload,
    )
    return payload


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


async def run_disclosure(
    session: AsyncSession,
    *,
    confirmed: Sequence[ConfirmedPair],
    mint: bool,
    not_minting_reason: str | None = None,
    unconfirmed: int = 0,
    actor: str = DISCLOSURE_ACTOR,
    reread: bool = False,
) -> DisclosureReport:
    """Run pass 3b: the re-reading, the lapse sweep, then the capped mint.

    ``confirmed`` is this run's second-reading-confirmed pairs, in probe
    order. ``mint`` is False on a night whose census did not run (degraded,
    failed, or verification off): the lapse sweep still runs, and only pairs
    waiting from an earlier run can mint. ``unconfirmed`` counts this run's
    flags the second reading did not confirm, for the ``skipped`` census.
    ``reread`` allows the LLM re-reading of stale pairs; the caller
    sets it only on a night the census's second reading ran. Commits; the
    caller runs it as one pass.
    """
    cfg = get_config().consolidation.contradiction_disclosure
    report = DisclosureReport(
        cap=cfg.max_per_run, minted=mint, not_minting_reason=not_minting_reason
    )
    if not cfg.enabled:
        report.enabled = False
        report.not_minting_reason = "consolidation.contradiction_disclosure.enabled is false"
        return report
    _skip(report, "unconfirmed", unconfirmed)
    touched: set[str] = set()

    # -- 0. re-reading (LLM, outside the write lock) -----------------------
    rereads = RereadOutcome()
    extract_rereads = ExtractRereadOutcome()
    if reread and cfg.max_rereadings_per_run > 0:
        rereads = await _reread_open_records(session, cfg.max_rereadings_per_run)
        report.reread = rereads.payload()
        # what the census re-reading left of the budget.
        left = cfg.max_rereadings_per_run - rereads.read
        extract_rereads = await _reread_extract_records(session, max(0, left))
        report.extract_reread = extract_rereads.payload()

    def withdrawn_payload(pairs: Sequence[ConfirmedPair]) -> list[dict[str, Any]]:
        return [
            {**p.to_payload(), "withdrawn_reason": rereads.withdrawn_reasons.get(p.key, "")}
            for p in pairs
        ]

    async with write_lock():
        # -- 0b. extract-time records the re-reading settled --
        for reading in extract_rereads.readings:
            closed = await _apply_extract_reading(session, reading)
            if closed is not None:
                report.closed.append(closed)
                touched |= {reading.a.id, reading.b.id}
                touched |= {closed[k] for k in ("successor", "folded_into") if k in closed}

        open_records, states = await _record_states(session)

        # -- 1. lapse sweep ------------------------------------------------
        member_ids = {
            pid for record, _ in open_records for p in census_pairs(record) for pid in (p.a, p.b)
        }
        survivors = await _merge_survivors(session, member_ids)
        view = await _view(session, member_ids | set(survivors.values()))

        def member_state(pid: str) -> MemberState:
            p = view.particles.get(pid)
            if p is None:
                return MemberState.GONE
            if p.status is Status.ACTIVE:
                return MemberState.LAPSED if _is_lapsed(view.scopes.get(pid)) else MemberState.LIVE
            survivor = survivors.get(pid)
            if (
                p.status is Status.SUPERSEDED
                and p.status_reason is StatusReason.DUPLICATE_MERGED
                and survivor is not None
                and view.live(survivor)
            ):
                return MemberState.MERGED
            return MemberState.GONE

        still_open: list[tuple[Particle, Sides]] = []
        regrouping: list[_Regroup] = []
        for record, sides in open_records:
            pairs = census_pairs(record)
            member_states = {pid: member_state(pid) for p in pairs for pid in (p.a, p.b)}
            decision = decide_lapse(pairs, member_states, survivors, rereads.verdicts)
            match decision.action:
                case LapseAction.KEEP:
                    still_open.append((record, sides))
                case LapseAction.CLOSE:
                    report.closed.append(
                        await _close(
                            session,
                            record,
                            decision.cause,
                            dropped=decision.dropped,
                            states=member_states,
                            view=view,
                            withdrawn=withdrawn_payload(decision.withdrawn),
                        )
                    )
                    touched |= set(sides.members)
                case LapseAction.REGROUP:
                    regrouping.append(
                        _Regroup(
                            record,
                            list(decision.remaining),
                            member_states,
                            decision.dropped,
                            withdrawn_payload(decision.withdrawn),
                        )
                    )
                    touched |= set(sides.members)

        # -- 2. candidates -------------------------------------------------
        candidates: list[ConfirmedPair] = []
        seen: set[frozenset[str]] = set()
        waiting = await _prior_waiting(session, actor)
        for pair in [*waiting, *(confirmed if mint else ())]:
            if pair.key not in seen:
                seen.add(pair.key)
                candidates.append(pair)

        open_states = [RecordState(r.id, s, open=True) for r, s in still_open]
        judged_states = [s for s in states if not s.open]
        judged_members = {
            pid
            for s in judged_states
            if s.cause is not None and s.cause.judgment
            for pid in s.sides.members
        }
        cand_ids = {pid for p in candidates for pid in (p.a, p.b)}
        cview = await _view(session, cand_ids | judged_members)
        covered = covered_pairs(
            [*open_states, *judged_states],
            await current_sources(
                session, [cview.particles[pid] for pid in judged_members if pid in cview.particles]
            ),
        ) | await _other_open_pairs(session)
        engaged = await lens_may_engage(session)
        widened = (
            await widened_ids(session, ScopeTarget.PARTICLE, sorted(cand_ids))
            if engaged and cand_ids and await any_widening(session)
            else set()
        )

        def scope(pid: str) -> BeliefScope:
            if pid in widened:
                return GLOBAL_SCOPE
            return cview.scopes.get(pid, GLOBAL_SCOPE)

        waiting_keys = {p.key for p in waiting}
        kept: list[ConfirmedPair] = []
        for pair in candidates:
            reason: str | None = None
            if pair.same_source:
                reason = "same_source"
            elif not (cview.live(pair.a) and cview.live(pair.b)):
                reason = "not_live"
            elif pair.key in covered:
                reason = "covered"
            elif engaged and not share_an_observer(scope(pair.a), scope(pair.b)):
                reason = "observer_disjoint"
            if reason is None:
                kept.append(pair)
                continue
            _skip(report, reason)
            if pair.key in waiting_keys:
                report.dropped_waiting.append({**pair.to_payload(), "dropped": reason})

        # -- 3. group, merging open records that share a claim -------------
        kept_claims = {pid for p in kept for pid in (p.a, p.b)}
        for record, sides in list(still_open):
            if kept_claims & set(sides.members):
                still_open.remove((record, sides))
                regrouping.append(_Regroup(record, census_pairs(record), {}, (), []))
                touched |= set(sides.members)

        origin_of: dict[frozenset[str], str] = {}
        existing: list[ConfirmedPair] = []
        for record, pairs, _, _, _ in regrouping:
            for pair in pairs:
                if pair.key not in origin_of:
                    origin_of[pair.key] = record.id
                    existing.append(pair)
        new_pairs = [p for p in kept if p.key not in origin_of]
        groups = group_pairs([*existing, *new_pairs])
        is_new = [not any(p.key in origin_of for p in group) for group in groups]
        opened_idx, waiting_idx = select_under_cap(is_new, cap=cfg.max_per_run)
        for index in waiting_idx:
            report.waiting.extend(p.to_payload() for p in groups[index])

        # -- 4. mint ---------------------------------------------------------
        all_ids = {pid for index in opened_idx for p in groups[index] for pid in (p.a, p.b)}
        mview = await _view(session, all_ids)
        origins = await _origins(
            session, [mview.particles[pid] for pid in all_ids if pid in mview.particles]
        )
        sources = await current_sources(
            session, [mview.particles[pid] for pid in all_ids if pid in mview.particles]
        )

        def orient(x: str, y: str) -> tuple[str, str]:
            wx, wy = origins.get(x), origins.get(y)
            if wx and wy and wx.when and wy.when and wy.when < wx.when:
                return y, x
            return x, y

        replaced_by: dict[str, list[str]] = {}
        for index in opened_idx:
            group = groups[index]
            planned, unsided = plan_group(group, orient)
            report.unsided += unsided
            record_origins = sorted({origin_of[p.key] for p in group if p.key in origin_of})
            replaces = record_origins[0] if record_origins else None
            minted: list[str] = []
            for plan in planned:
                record = await _mint(session, plan, mview, origins, sources, replaces=replaces)
                minted.append(record.id)
                touched |= set(plan.sides.members)
            for origin in record_origins:
                replaced_by.setdefault(origin, []).extend(minted)
            if is_new[index]:
                report.opened.append(
                    {
                        "record_ids": minted,
                        "members": len({pid for p in group for pid in (p.a, p.b)}),
                    }
                )

        for record, _pairs, member_states, dropped, withdrawn in regrouping:
            # A regrouping record whose remaining pairs all lapsed out of the
            # grouping (none survived to a group) closes as lapsed instead.
            replacements = replaced_by.get(record.id, [])
            cause = CloseCause.REGROUPED if replacements else CloseCause.LAPSED
            report.closed.append(
                await _close(
                    session,
                    record,
                    cause,
                    dropped=dropped,
                    states=member_states,
                    view=view,
                    replacements=replacements,
                    withdrawn=withdrawn,
                )
            )

        await session.commit()

    report.open_records = sum(
        1 for r in await get_census_records(session) if r.status is Status.INCONSISTENCY
    )
    report.touched_ids = sorted(touched)
    if report.opened or report.closed or report.waiting:
        log.info(
            "contradiction disclosure: opened %d, closed %d, %d waiting",
            report.opened_records,
            len(report.closed),
            len(report.waiting),
        )
    return report

# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The release pass: reconsider a claim held behind a retired claim.

When the §6.6 ladder cannot settle a contradiction, rung 3 writes an
INCONSISTENCY record and stores the newer claim quarantined behind the stored
one. If an autonomous pass later retires the stored claim (an
update, the re-anchor pass, the document sweep), the record is left with no
ACTIVE side, and nothing reconciles the held claim against the claim that did
the retiring. This pass asks the question again: does anything ACTIVE still
contradict the held claim?

Gather, decide, apply (D2). Per qualifying record, oldest retirement
first, under ``consolidation.release.max_records``:

1. **Pairs, zero-LLM**: the write path's selection over the current ACTIVE
   set, the store-wide nearest claim plus the subject-keyed pool, with the
   held row (and an ACTIVE twin that states it verbatim) excluded.
2. **The two checks**: the contradiction probe on each pair, then the second
   reading on each pair the probe confirms. A check that does not
   complete counts against release (fail-closed). A confirmed
   pair the observer precondition declines does not block.
3. **The writes**, by :func:`~particles.core.held_release.decide_release`:

   * nothing confirmed: the record closes in the withdrawal's
     shape, the held claim returns to the store (folded into a verbatim twin,
     or minted as review's ``PREFER_B`` mints it), declined pairs become
     ``CONTRADICTS`` edges, and one ``INCONSISTENCY_CLOSED`` event records
     ``cause: partner_retired``;
   * something confirmed: nothing is written to any claim, even a claim the
     held one would beat at a rung, and one ``HELD_CLAIM_EXAMINED`` event
     names what blocks it;
   * a check incomplete: nothing is written, and the next run asks again.

A record already examined is asked again only once a claim that blocked it has
left ACTIVE (§3). The pass never retires an ACTIVE claim. The trust cascade is
unchanged: whichever of the two reaches a record first closes it, and the
release re-reads each record before it writes. Each record's writes commit
under the writer lock; the LLM calls run outside it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.conflict_review import CLOSED_RECORD, conflict_pair_ids, is_quarantined
from particles.core.contradiction_disclosure import DISCLOSURE_ACTOR, CloseCause
from particles.core.generics import is_generic_claim
from particles.core.held_release import (
    PairCheck,
    ReleaseAction,
    ReleaseDecision,
    blocking_from_payload,
    decide_release,
    is_due,
    is_generation_hold,
    qualifies,
)
from particles.core.observer_scope import PairPrecondition
from particles.core.schema import Particle, ProvenanceRefType
from particles.core.stance import stance_holder
from particles.core.status import Status, StatusReason
from particles.corpus.store import get_entry, get_entry_currency
from particles.db import write_transaction
from particles.embeddings import get_embedding_model
from particles.ingest.duplicate_suppression import build_duplicate_index
from particles.ingest.observer_gate import ObserverGate
from particles.ingest.pipeline import _find_conflict, load_active_conflict_candidates
from particles.ingest.second_reading import (
    claim_context,
    read_pair,
    snapshot_claim_context,
)
from particles.ingest.subject_resolver import load_speaker_key
from particles.ingest.update_supersession import SpeakerKey, SubjectIndex, particle_subject_ids
from particles.llm.breaker import llm_circuit_open
from particles.operations._quarantine import restore_held_claim
from particles.operations.lint.open_inconsistency import latest_held_examinations
from particles.store.event_store import (
    EventRefKind,
    OperatorEventType,
    list_events,
    record_event,
)
from particles.store.particle_store import (
    get_inconsistency_particles,
    get_particle,
    get_particles_by_ids,
    get_retired_at,
    get_superseding_particle,
    update_particle_status,
)

log = logging.getLogger(__name__)

Embedding = np.ndarray[Any, np.dtype[np.float32]]

#: The actor every event this pass writes records under, as the disclosure
#: pass's closures do.
RELEASE_ACTOR = DISCLOSURE_ACTOR


class ReleaseReport(BaseModel):
    """What one run of the pass did (the run record's ``release`` block)."""

    enabled: bool = True
    dry_run: bool = False
    #: Set when the pass did no work at all, with the reason.
    skipped_reason: str | None = None
    #: Open records that meet §1's three conditions.
    qualifying: int = 0
    #: Qualifying records still blocked by the same ACTIVE claims: not asked.
    unchanged: int = 0
    #: Due records past the cap, left for the next run.
    waiting: int = 0
    examined: int = 0
    pairs: int = 0
    probes: int = 0
    readings: int = 0
    #: ``{"record", "held", "partner", "successor" | "folded_into"}`` per release.
    released: list[dict[str, str]] = Field(default_factory=list)
    #: ``{"record", "held", "partner", "blocking", "event"}`` per blocked record.
    blocked: list[dict[str, Any]] = Field(default_factory=list)
    #: Records a check did not complete for: kept open, asked again next run.
    incomplete: int = 0
    #: Records something else closed between selection and the write.
    closed_meanwhile: int = 0
    #: ``CONTRADICTS`` edges written for declined pairs.
    divergences: int = 0
    warnings: list[str] = Field(default_factory=list)

    @property
    def llm_calls(self) -> int:
        return self.probes + self.readings

    def payload(self) -> dict[str, Any]:
        """The compact block written to the ``CONSOLIDATION_RUN`` payload."""
        return {
            "enabled": self.enabled,
            "skipped_reason": self.skipped_reason,
            "qualifying": self.qualifying,
            "unchanged": self.unchanged,
            "waiting": self.waiting,
            "examined": self.examined,
            "pairs": self.pairs,
            "probes": self.probes,
            "readings": self.readings,
            "released": self.released,
            "blocked": self.blocked,
            "incomplete": self.incomplete,
            "closed_meanwhile": self.closed_meanwhile,
            "divergences": self.divergences,
        }


@dataclass(frozen=True)
class _Held:
    """One qualifying record and its two sides, with when its partner was retired."""

    record: Particle
    partner: Particle
    held: Particle
    retired_at: datetime


# ---------------------------------------------------------------------------
# Gather: the queue
# ---------------------------------------------------------------------------


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


async def _generation_rows(
    session: AsyncSession, claims: Sequence[Particle]
) -> tuple[dict[str, str], dict[str, datetime]]:
    """Each cited ``MUTABLE`` entry's current generation, and snapshot capture times.

    The current generation is :func:`get_entry_currency`'s, the one definition
    of it; an entry whose current record is an absence has none.
    """
    entry_ids = sorted(
        {
            r.corpus_entry_id
            for p in claims
            for r in p.provenance
            if r.type is ProvenanceRefType.SOURCE and r.corpus_entry_id
        }
    )
    current = {
        entry_id: c.latest_snapshot_id
        for entry_id, c in (await get_entry_currency(session, entry_ids)).items()
        if c.mutable and c.latest_snapshot_id is not None and not c.absent
    }
    captured: dict[str, datetime] = {}
    for entry_id in sorted(current):
        entry = await get_entry(session, entry_id)
        if entry is not None:
            captured.update((s.snapshot_id, s.captured_at) for s in entry.snapshots)
    return current, captured


async def qualifying_records(session: AsyncSession) -> list[_Held]:
    """Every open record §1 admits, oldest partner retirement first (zero-LLM).

    A record whose partner the generation cascade retired is admitted only
    when the pair is a generation hold, read from the entry and
    snapshot rows of those records alone.
    """
    records = await get_inconsistency_particles(session)
    pairs = {r.id: conflict_pair_ids(r) for r in records}
    ids = sorted({pid for a, b in pairs.values() for pid in (a, b) if pid})
    members = await get_particles_by_ids(session, ids)
    cascade_pairs = [
        (members[a_id], members[b_id])
        for a_id, b_id in pairs.values()
        if a_id in members
        and b_id in members
        and members[a_id].status_reason is StatusReason.RETRACTED_DEPENDENCY
    ]
    current, captured = await _generation_rows(
        session, [claim for pair in cascade_pairs for claim in pair]
    )
    out: list[_Held] = []
    for record in records:
        a_id, b_id = pairs[record.id]
        a = members.get(a_id) if a_id else None
        b = members.get(b_id) if b_id else None
        generation = (
            a is not None
            and b is not None
            and is_generation_hold(a, b, current=current, captured=captured)
        )
        if not qualifies(record, a, b, generation=generation):
            continue
        assert a is not None and b is not None  # qualifies() checked both
        retired_at = await get_retired_at(session, a.id) or record.asserted_at
        out.append(_Held(record=record, partner=a, held=b, retired_at=_as_utc(retired_at)))
    out.sort(key=lambda h: (h.retired_at, h.record.id))
    return out


async def _due(session: AsyncSession, queue: Sequence[_Held]) -> tuple[list[_Held], int]:
    """The records asked this run (§3), and how many qualifying ones are not due."""
    examined = {
        rid: blocking
        for rid, payload in (await latest_held_examinations(session)).items()
        if (blocking := blocking_from_payload(payload)) is not None
    }
    blocker_ids = sorted({pid for h in queue for pid in examined.get(h.record.id, [])})
    blockers = await get_particles_by_ids(session, blocker_ids)
    active = {pid for pid, p in blockers.items() if p.status is Status.ACTIVE}
    due = [h for h in queue if is_due(examined.get(h.record.id), active)]
    return due, len(queue) - len(due)


async def retired_by(session: AsyncSession, partner: Particle) -> str | None:
    """The claim that retired ``partner``, when one is on record.

    The ``UPDATE_SUPERSESSION`` event first, since a winner's ``supersedes``
    holds one id and a second claim it retired is named only there; else the
    claim whose ``supersedes`` names the partner. A document-sweep retirement
    names none.
    """
    for event in await list_events(
        session,
        ref_kind=EventRefKind.PARTICLE,
        ref_id=partner.id,
        event_type=OperatorEventType.UPDATE_SUPERSESSION,
        limit=20,
    ):
        payload = event.payload or {}
        winner = payload.get("winner_id")
        if payload.get("retired_id") == partner.id and isinstance(winner, str) and winner:
            return winner
    successor = await get_superseding_particle(session, partner.id)
    return successor.id if successor is not None else None


# ---------------------------------------------------------------------------
# Gather: the pairs
# ---------------------------------------------------------------------------


async def _encode(text: str) -> Embedding:
    model = get_embedding_model()
    assert model is not None  # checked by the caller before any work
    encoded = await asyncio.to_thread(
        model.encode, [text], convert_to_numpy=True, normalize_embeddings=True
    )
    return np.asarray(encoded[0], dtype=np.float32)


def select_pairs(
    held: Particle,
    embedding: Embedding,
    candidates: Sequence[tuple[Particle, Embedding | None]],
    *,
    exclude: set[str],
    speaker: SpeakerKey | None,
) -> list[Particle]:
    """The ACTIVE claims the held claim is checked against.

    ``reconcile_and_insert``'s selection over the current ACTIVE set: the
    store-wide nearest claim above ``extraction.similarity_threshold``, then
    the subject-keyed pool, at most
    ``reconciliation.update_supersession.max_candidates`` members above its
    subject floor. Where that path stops at the first pair, this one asks
    every pair, because any one of them may still contradict the held claim.
    Pure over the loaded embeddings.
    """
    pool_pairs = [(p, e) for p, e in candidates if p.id not in exclude]
    generic = is_generic_claim(held.content)
    nearest = _find_conflict(
        embedding,
        [p for p, _ in pool_pairs],
        [e for _, e in pool_pairs],
        candidate_stance_holder=stance_holder(held),
        candidate_generic=generic,
    )
    out = [nearest] if nearest is not None else []
    update_cfg = get_config().reconciliation.update_supersession
    subject_ids = particle_subject_ids(held, speaker=speaker)
    if update_cfg.enabled and subject_ids and stance_holder(held) is None:
        out += SubjectIndex.build(pool_pairs, speaker=speaker).candidates(
            subject_ids,
            embedding,
            floor=update_cfg.subject_floor,
            limit=update_cfg.max_candidates,
            skip_ids=() if nearest is None else (nearest.id,),
            generic=generic,
        )
    return out


# ---------------------------------------------------------------------------
# The checks (LLM)
# ---------------------------------------------------------------------------


async def _check(
    session: AsyncSession,
    item: _Held,
    pairs: Sequence[Particle],
    gate: ObserverGate,
    report: ReleaseReport,
) -> list[PairCheck] | None:
    """Probe each pair, then read each confirmed one a second time.

    Returns ``None`` when the breaker opened before a check could run: the
    record is then left for a later run without counting as examined.
    """
    # Deferred import: the breaker-seam probe, patched at its source module by
    # tests/test_held_release.py (tests/AGENTS.md § Mocking strategy).
    from particles.ingest.pipeline import _has_contradiction_signal

    preconditions: dict[str, PairPrecondition] = {}
    if gate.engaged:
        scope = await gate.candidate_scope(session, item.held)
        for other in pairs:
            preconditions[other.id] = await gate.verdict(session, scope, other)
    # The checks are LLM round trips: end the gather's read transaction first.
    await session.commit()

    probes: dict[str, bool | None] = {}
    for other in pairs:
        if llm_circuit_open():
            return None
        report.probes += 1
        probes[other.id] = await _has_contradiction_signal(item.held.content, other.content)

    confirmed = [other for other in pairs if probes[other.id]]
    contexts = []
    for other in confirmed:
        # The existing claim is A and the held claim B, as the record names them.
        contexts.append(
            (
                other,
                await claim_context(session, other, item.held.content),
                await snapshot_claim_context(session, item.held, other),
            )
        )
    await session.commit()
    readings: dict[str, bool | None] = {}
    for other, ctx_a, ctx_b in contexts:
        if llm_circuit_open():
            readings[other.id] = None
            continue
        report.readings += 1
        verdict = await read_pair(ctx_a, ctx_b)
        readings[other.id] = None if verdict is None else verdict.contradicts

    return [
        PairCheck(
            existing_id=other.id,
            probe=probes[other.id],
            reading=readings.get(other.id),
            precondition=preconditions.get(other.id, PairPrecondition.RECONCILE),
        )
        for other in pairs
    ]


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


async def _still_open(session: AsyncSession, item: _Held) -> tuple[Particle, Particle] | None:
    """The record and held claim re-read, or ``None`` when something closed it since."""
    record = await get_particle(session, item.record.id)
    held = await get_particle(session, item.held.id)
    if record is None or record.status is not Status.INCONSISTENCY:
        return None
    if held is None or not is_quarantined(held):
        return None
    return record, held


async def _release(
    session: AsyncSession,
    item: _Held,
    decision: ReleaseDecision,
    gate: ObserverGate,
    report: ReleaseReport,
    winner: str | None,
) -> str | None:
    """Close the record and return the held claim to the store; the landed id, or ``None``."""
    async with write_transaction(session):
        reread = await _still_open(session, item)
        if reread is None:
            report.closed_meanwhile += 1
            return None
        record, held = reread
        await update_particle_status(session, record.id, *CLOSED_RECORD)
        key, landed = await restore_held_claim(session, held)
        standing: list[Particle] = []
        if decision.declined:
            found = await get_particles_by_ids(session, list(decision.declined))
            standing = [p for p in found.values() if p.status is Status.ACTIVE and p.id != landed]
            report.divergences += await gate.record(session, landed, standing)
        await record_event(
            session,
            actor=RELEASE_ACTOR,
            event_type=OperatorEventType.INCONSISTENCY_CLOSED,
            refs=[
                (EventRefKind.PARTICLE, record.id),
                (EventRefKind.PARTICLE, item.partner.id),
                (EventRefKind.PARTICLE, held.id),
                (EventRefKind.PARTICLE, landed),
            ],
            payload={
                "record_id": record.id,
                "cause": CloseCause.PARTNER_RETIRED.value,
                "origin": "extraction",
                "partner": item.partner.id,
                "partner_reason": (
                    item.partner.status_reason.value if item.partner.status_reason else None
                ),
                "retired_by": winner,
                "newcomer": held.id,
                key: landed,
                "divergences": [p.id for p in standing],
            },
        )
    report.released.append(
        {"record": item.record.id, "held": item.held.id, "partner": item.partner.id, key: landed}
    )
    return landed


async def _record_blocked(
    session: AsyncSession,
    item: _Held,
    decision: ReleaseDecision,
    report: ReleaseReport,
    winner: str | None,
) -> None:
    """Write the ``HELD_CLAIM_EXAMINED`` event; nothing else changes."""
    async with write_transaction(session):
        if await _still_open(session, item) is None:
            report.closed_meanwhile += 1
            return
        event = await record_event(
            session,
            actor=RELEASE_ACTOR,
            event_type=OperatorEventType.HELD_CLAIM_EXAMINED,
            refs=[
                (EventRefKind.PARTICLE, item.record.id),
                (EventRefKind.PARTICLE, item.partner.id),
                (EventRefKind.PARTICLE, item.held.id),
                *((EventRefKind.PARTICLE, pid) for pid in decision.blocking),
            ],
            payload={
                "record_id": item.record.id,
                "partner_id": item.partner.id,
                "partner_reason": (
                    item.partner.status_reason.value if item.partner.status_reason else None
                ),
                "retired_by": winner,
                "held_id": item.held.id,
                "blocking_ids": list(decision.blocking),
            },
        )
    report.blocked.append(
        {
            "record": item.record.id,
            "held": item.held.id,
            "partner": item.partner.id,
            "blocking": list(decision.blocking),
            "event": event.event_id,
        }
    )


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


async def run_release(session: AsyncSession, *, dry_run: bool = False) -> ReleaseReport:
    """Run the release pass once.

    Args:
        dry_run: report the queue and the pairs each record would be checked
            against, and make no call and no write.

    Returns:
        The run's report. Each record's writes are committed as it settles.
    """
    cfg = get_config().consolidation.release
    report = ReleaseReport(enabled=cfg.enabled, dry_run=dry_run)
    if not cfg.enabled:
        report.skipped_reason = "consolidation.release.enabled is false"
        return report

    queue = await qualifying_records(session)
    report.qualifying = len(queue)
    if not queue:
        return report
    due, report.unchanged = await _due(session, queue)
    chosen = due[: cfg.max_records]
    report.waiting = len(due) - len(chosen)
    if not chosen:
        return report
    if get_embedding_model() is None:
        report.skipped_reason = "no embedding model (pair selection needs one)"
        return report

    candidates: list[tuple[Particle, Embedding | None]] = list(
        await load_active_conflict_candidates(session)
    )
    gate = await ObserverGate.open(session)
    # the release pool reads the same persona key as every other reader.
    speaker = await load_speaker_key(session)
    for index, item in enumerate(chosen):
        embedding = await _encode(item.held.content)
        twin = (await build_duplicate_index(session, [item.held.content])).find(item.held)
        exclude = {item.held.id, item.partner.id, item.record.id}
        if twin is not None:
            exclude.add(twin.id)  # a verbatim twin states it; it cannot contradict it
        pairs = select_pairs(item.held, embedding, candidates, exclude=exclude, speaker=speaker)
        report.pairs += len(pairs)
        if dry_run:
            report.examined += 1
            continue
        winner = await retired_by(session, item.partner)
        checks = await _check(session, item, pairs, gate, report)
        if checks is None:
            report.waiting += len(chosen) - index
            report.warnings.append("the LLM circuit breaker opened; the rest wait for a later run")
            break
        report.examined += 1
        decision = decide_release(checks)
        match decision.action:
            case ReleaseAction.RELEASE:
                landed = await _release(session, item, decision, gate, report, winner)
                minted = await get_particle(session, landed) if landed and twin is None else None
                if minted is not None:
                    # A later record this run is checked against the claim just minted.
                    candidates.append((minted, embedding))
            case ReleaseAction.BLOCKED:
                await _record_blocked(session, item, decision, report, winner)
            case ReleaseAction.INCOMPLETE:
                report.incomplete += 1
                report.warnings.append(
                    f"a check on {item.record.id[:8]} did not complete; asked again next run"
                )

    log.info(
        "release: %d qualifying, %d examined, %d released, %d blocked, %d incomplete, "
        "%d unchanged, %d waiting%s",
        report.qualifying,
        report.examined,
        len(report.released),
        len(report.blocked),
        report.incomplete,
        report.unchanged,
        report.waiting,
        " (dry run)" if dry_run else "",
    )
    return report

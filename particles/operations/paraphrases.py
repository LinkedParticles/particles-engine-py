# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The paraphrases pass: an update retires the other phrasings of the value it replaced.

When rung 2.5 retires a claim R ("The user lives in Delhi, in Lajpat Nagar.")
in favour of U ("The user decided to move to Mumbai permanently."), a second
phrasing of R's value ("The user's home is in Delhi.") can stay ACTIVE: every
other route reaches the update probe only through the contradiction probe, and
that probe samples. This pass asks the narrower question of each such phrasing
directly.

Per update retirement R, oldest first after the pass's own cursor:

1. **The winner.** U is read from R's ``UPDATE_SUPERSESSION`` event, else from
   the claim whose ``supersedes`` names R. A retirement with neither is
   skipped. U must be ACTIVE, or itself retired by a later update.
2. **Candidates, zero-LLM.** ACTIVE reconcilable claims S sharing R's
   about-subject, at content cosine to R of at least
   ``reconciliation.update_supersession.paraphrase_floor``, that rung 2.5 could
   order against U (``update_order``: same lineage, U strictly newer), that the
   observer precondition reconciles with U, and with U not ALEATORY.
   A pair the update-slot ledger already cleared is dropped. At most
   ``max_paraphrases`` go forward, highest cosine first.
3. **The update probe**, S earlier and U later, with its second reading
   (``read_update_slot``). **No contradiction probe is asked**: the
   contradiction was established for R, and U's change of R's value was read
   twice already.
4. **The write.** Two ``CHANGES`` readings retire S ``PROVENANCE_STALE`` /
   ``SUPERSEDED_BY_UPDATE`` with an ``UPDATE_SUPERSESSION`` event naming
   ``route: paraphrase``. Any other verdict leaves S as it was and goes to the
   ledger. The write is outside ``resolve_conflict`` on purpose: calling the
   ladder with a contradiction signal no probe produced would give the
   conformance function an input nothing measured.

A retirement made here is itself an update retirement, so it is examined in
the same run under the run cap (``consolidation.max_paraphrase_probes``, first
readings). When the cap is reached the cursor stays at the last retirement
fully examined.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.conflict_resolution import SlotVerdict, admits_update, newer_is_settled
from particles.core.generics import is_generic_claim
from particles.core.observer_scope import PairPrecondition
from particles.core.probe_verdict import ProbeKind, VerdictKey, verdict_key
from particles.core.reanchor import Cursor
from particles.core.schema import CorpusEntry, Particle
from particles.core.status import Status, StatusReason
from particles.db import write_transaction
from particles.embeddings import cosine_similarity
from particles.ingest.conflict_plan import plan_update_event
from particles.ingest.observer_gate import ObserverGate
from particles.ingest.pipeline import (
    SlotReading,
    read_update_slot,
    update_decided_by,
    update_prompt_hash,
)
from particles.ingest.subject_resolver import load_speaker_key
from particles.ingest.update_supersession import SpeakerKey, SubjectIndex, particle_subject_ids
from particles.llm.breaker import llm_circuit_open
from particles.operations.probe_ledger import recall, remember
from particles.operations.reconcile import _pair_update_order
from particles.store.event_store import EventRefKind, OperatorEventType, list_events, record_event
from particles.store.particle_store import (
    count_update_retirements_after,
    get_active_particles_with_embeddings,
    get_embeddings_by_ids,
    get_particle,
    get_superseding_particle,
    get_update_retirements_after,
    update_particle_status,
)

log = logging.getLogger(__name__)

Embedding = np.ndarray[Any, np.dtype[np.float32]]

#: The ``route`` an ``UPDATE_SUPERSESSION`` event of this pass names.
ROUTE = "paraphrase"
#: Update retirements fetched per page of the cursor walk.
_PAGE = 200


class ParaphraseReport(BaseModel):
    """What one run of the pass did (the run record's ``paraphrases`` block)."""

    enabled: bool = True
    dry_run: bool = False
    #: Set when the pass did no work at all, with the reason.
    skipped_reason: str | None = None
    probe_cap: int = 0
    #: Update retirements examined (each one whose candidates were all asked).
    triggers: int = 0
    #: Update retirements past the cursor left for the next run.
    waiting: int = 0
    #: Retirements with no recorded winner (no event, no ``supersedes``).
    no_winner: int = 0
    #: Retirements whose winner is neither ACTIVE nor retired by a later update.
    winner_inactive: int = 0
    #: Claims of R's about-subject at or above the floor, before the conditions.
    above_floor: int = 0
    #: Of those, kept out by each condition.
    unordered: int = 0
    aleatory_excluded: int = 0
    observer_declined: int = 0
    previously_cleared: int = 0
    #: Candidates put to the update probe (in a dry run, that would be).
    candidates: int = 0
    #: First readings of the update probe, and the second readings a CHANGES added.
    probes: int = 0
    readings: int = 0
    #: Candidates whose reading gave no usable verdict with the model reachable:
    #: left as they were, unrecorded, and not asked again once the cursor passes.
    unanswered: int = 0
    #: ``{"retired", "winner", "trigger", "similarity"}`` per phrasing retired.
    retired: list[dict[str, object]] = Field(default_factory=list)
    #: ``{"particle", "winner", "trigger", "verdict"}`` per candidate left as it was.
    kept: list[dict[str, object]] = Field(default_factory=list)
    #: True when the run stopped at ``consolidation.max_paraphrase_probes``.
    capped: bool = False
    #: The cursor after this run; ``None`` until a first retirement is examined.
    cursor: dict[str, str] | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def llm_calls(self) -> int:
        return self.probes + self.readings

    def payload(self) -> dict[str, Any]:
        """The compact block written to the ``CONSOLIDATION_RUN`` payload."""
        return {
            "enabled": self.enabled,
            "skipped_reason": self.skipped_reason,
            "triggers": self.triggers,
            "waiting": self.waiting,
            "no_winner": self.no_winner,
            "above_floor": self.above_floor,
            "candidates": self.candidates,
            "previously_cleared": self.previously_cleared,
            "probes": self.probes,
            "readings": self.readings,
            "unanswered": self.unanswered,
            "retired": self.retired,
            "kept": self.kept,
            "capped": self.capped,
            "cursor": self.cursor,
        }


async def _read_update_slot(earlier: str, later: str) -> SlotReading:
    """The rung 2.5 update probe, with its second reading. The test seam."""
    return await read_update_slot(earlier, later)


def _slot_key(earlier: Particle, later: Particle) -> VerdictKey:
    # Directional, as in the sweep: the slot probe is asked (earlier, later).
    return verdict_key(ProbeKind.UPDATE_SLOT, earlier.content, later.content)


def choose(scored: list[tuple[float, Particle]], limit: int) -> list[tuple[float, Particle]]:
    """The candidates that go forward: at most ``limit``, highest cosine to R first."""
    return sorted(scored, key=lambda pair: (-pair[0], pair[1].id))[:limit]


async def winner_of(session: AsyncSession, retired: Particle) -> Particle | None:
    """The claim that retired ``retired`` by update, or ``None`` when none is on record.

    The ``UPDATE_SUPERSESSION`` event names it (1.178.4). A retirement written
    before that event existed falls back to the claim whose ``supersedes`` names
    ``retired``.
    """
    events = await list_events(
        session,
        ref_kind=EventRefKind.PARTICLE,
        ref_id=retired.id,
        event_type=OperatorEventType.UPDATE_SUPERSESSION,
        limit=20,
    )
    for event in events:  # newest first
        payload = event.payload or {}
        winner_id = payload.get("winner_id")
        if payload.get("retired_id") == retired.id and isinstance(winner_id, str):
            return await get_particle(session, winner_id)
    return await get_superseding_particle(session, retired.id)


def _winner_usable(winner: Particle) -> bool:
    # A value U replaced is stale whether U is current or was itself updated
    # later; a retracted or otherwise retired U is not evidence of a change.
    if winner.status is Status.ACTIVE:
        return True
    return (
        winner.status is Status.PROVENANCE_STALE
        and winner.status_reason is StatusReason.SUPERSEDED_BY_UPDATE
    )


class _Gather:
    """What the pass reads once per run and shares across retirements."""

    def __init__(self, index: SubjectIndex, gate: ObserverGate, speaker: SpeakerKey | None) -> None:
        self.index = index
        self.gate = gate
        #: The persona key the index was built with.
        self.speaker = speaker
        self.entries: dict[str, CorpusEntry | None] = {}


async def _candidates(
    session: AsyncSession,
    shared: _Gather,
    retired: Particle,
    retired_emb: Embedding,
    winner: Particle,
    report: ParaphraseReport,
    *,
    floor: float,
    limit: int,
    require_attribution: bool,
) -> list[tuple[float, Particle]]:
    """The paraphrase candidates of one retirement, best first; no LLM call."""
    generic = is_generic_claim(winner.content)
    seen = {winner.id, retired.id}
    above: list[tuple[float, Particle]] = []
    for key in particle_subject_ids(retired, speaker=shared.speaker):
        for particle, emb in shared.index.by_subject.get(key, []):
            if particle.id in seen or particle.id in shared.index.retired:
                continue
            seen.add(particle.id)
            score = cosine_similarity(retired_emb, emb)
            if score < floor:
                continue
            # A generic and an instance claim are not an adjudicable pair.
            if is_generic_claim(particle.content) != generic:
                continue
            above.append((score, particle))
    report.above_floor += len(above)

    qualifying: list[tuple[float, Particle]] = []
    for score, particle in above:
        order = await _pair_update_order(
            session, shared.entries, winner, particle, require_attribution=require_attribution
        )
        if order != 1:
            report.unordered += 1
            continue
        # The rung 2.5 ALEATORY rule: the newer side, U, must not be ALEATORY;
        # an older ALEATORY phrasing may be retired.
        if not newer_is_settled(particle, winner, order):
            report.aleatory_excluded += 1
            continue
        if shared.gate.engaged and (
            await shared.gate.verdict(
                session, await shared.gate.scope_of(session, winner), particle
            )
            is not PairPrecondition.RECONCILE
        ):
            report.observer_declined += 1
            continue
        qualifying.append((score, particle))

    if qualifying:
        # A recorded verdict under the current prompt is not asked twice.
        # Only a cleared pair is dropped: a recorded CHANGES on a
        # claim still ACTIVE was a dry run's or an interrupted write's.
        recorded = await recall(
            session,
            ProbeKind.UPDATE_SLOT,
            update_prompt_hash(),
            [_slot_key(p, winner) for _, p in qualifying],
        )
        kept = [(s, p) for s, p in qualifying if recorded.get(_slot_key(p, winner)) is not False]
        report.previously_cleared += len(qualifying) - len(kept)
        qualifying = kept
    return choose(qualifying, limit)


async def _retire(session: AsyncSession, paraphrase: Particle, winner: Particle) -> None:
    """Retire one phrasing by update, with its record; one commit, as in the sweep."""
    async with write_transaction(session):
        await update_particle_status(
            session, paraphrase.id, Status.PROVENANCE_STALE, StatusReason.SUPERSEDED_BY_UPDATE
        )
        event = plan_update_event(
            paraphrase.id, winner.id, route=ROUTE, decided_by=update_decided_by()
        )
        await record_event(
            session,
            actor=event.actor,
            event_type=event.event_type,
            reason=event.reason,
            refs=list(event.refs),
            payload=dict(event.payload),
        )


async def _ask(
    session: AsyncSession,
    shared: _Gather,
    retired: Particle,
    winner: Particle,
    chosen: list[tuple[float, Particle]],
    report: ParaphraseReport,
    progress: Callable[[str], None] | None,
) -> bool:
    """Put each chosen candidate to the update probe and apply its verdict.

    Returns ``False`` when a reading gave no verdict because the model is
    unreachable (the breaker is open): the caller stops and holds its cursor.
    A reply that was unusable with the model reachable leaves its candidate as
    it was and is counted, so one candidate the model cannot answer never stalls
    the cursor.
    """
    for score, paraphrase in chosen:
        reading = await _read_update_slot(paraphrase.content, winner.content)
        report.probes += 1
        if reading.purpose == "verification":
            report.readings += 1
        if reading.verdict is None:
            if llm_circuit_open():
                return False
            report.unanswered += 1
            continue
        await remember(
            session,
            ProbeKind.UPDATE_SLOT,
            update_prompt_hash(),
            {_slot_key(paraphrase, winner): admits_update(reading.verdict)},
            purpose=reading.purpose,
            rung=reading.decided_by,
        )
        if reading.verdict is not SlotVerdict.CHANGES:
            report.kept.append(
                {
                    "particle": paraphrase.id,
                    "winner": winner.id,
                    "trigger": retired.id,
                    "verdict": reading.verdict.value,
                }
            )
            continue
        await _retire(session, paraphrase, winner)
        shared.index.retired.add(paraphrase.id)
        report.retired.append(
            {
                "retired": paraphrase.id,
                "winner": winner.id,
                "trigger": retired.id,
                "similarity": round(score, 4),
            }
        )
        if progress is not None:
            progress(
                f"retired {paraphrase.id[:8]} (a phrasing of {retired.id[:8]}, "
                f"superseded by {winner.id[:8]}, sim {score:.3f})"
            )
    return True


async def run_paraphrases(  # noqa: PLR0912 — one linear cursor walk
    session: AsyncSession,
    *,
    cursor: Cursor | None,
    dry_run: bool = False,
    probe_cap: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> ParaphraseReport:
    """Run the paraphrases pass once.

    Args:
        cursor: where the previous run stopped; ``None`` starts from the oldest
            update retirement in the store.
        dry_run: gather the candidates and count the probes a run would ask,
            under the cap, and make no call and no write. Retirements the run
            itself would make are not foreseen.
        probe_cap: overrides ``consolidation.max_paraphrase_probes``.
        progress: optional human-readable progress callback.

    Returns:
        The run's report. ``report.cursor`` is where the next run resumes; a
        dry run reports the cursor it started from.
    """
    config = get_config()
    update_cfg = config.reconciliation.update_supersession
    cap = config.consolidation.max_paraphrase_probes if probe_cap is None else probe_cap
    report = ParaphraseReport(
        enabled=update_cfg.enabled and update_cfg.max_paraphrases > 0,
        dry_run=dry_run,
        probe_cap=cap,
        cursor=cursor.payload() if cursor is not None else None,
    )
    if not update_cfg.enabled:
        report.skipped_reason = "reconciliation.update_supersession.enabled is false"
        return report
    if update_cfg.max_paraphrases == 0:
        report.skipped_reason = "reconciliation.update_supersession.max_paraphrases is 0"
        return report

    require_attribution = config.reconciliation.store_mode != "single"
    speaker = await load_speaker_key(session)
    shared = _Gather(
        SubjectIndex.build(await get_active_particles_with_embeddings(session), speaker=speaker),
        await ObserverGate.open(session),
        speaker,
    )
    after: tuple[datetime, str] | None = (
        (cursor.retired_at, cursor.particle_id) if cursor is not None else None
    )
    stop = False
    while not stop:
        page = await get_update_retirements_after(session, after, limit=_PAGE)
        if not page:
            break
        embeddings = await get_embeddings_by_ids(session, [r.id for r, _ in page])
        for retired, retired_at in page:
            winner = await winner_of(session, retired)
            retired_emb = embeddings.get(retired.id)
            chosen: list[tuple[float, Particle]] = []
            if winner is None:
                report.no_winner += 1
            elif not _winner_usable(winner):
                report.winner_inactive += 1
            elif retired_emb is not None:
                chosen = await _candidates(
                    session,
                    shared,
                    retired,
                    retired_emb,
                    winner,
                    report,
                    floor=update_cfg.paraphrase_floor,
                    limit=update_cfg.max_paraphrases,
                    require_attribution=require_attribution,
                )
            if chosen and (cap == 0 or (report.probes > 0 and report.probes + len(chosen) > cap)):
                # The cursor stays before this retirement: its candidates are
                # asked together on the next run.
                report.capped = True
                stop = True
                break
            if dry_run:
                report.triggers += 1
                report.candidates += len(chosen)
                report.probes += len(chosen)
                after = (retired_at, retired.id)
                continue
            if chosen:
                assert winner is not None  # chosen is empty without one
                if not await _ask(session, shared, retired, winner, chosen, report, progress):
                    # Keep the cursor before this retirement, so the next run asks again.
                    report.warnings.append(
                        f"the model was unreachable while asking about a phrasing of "
                        f"{retired.id[:8]}; retried next run"
                    )
                    stop = True
                    break
            report.triggers += 1
            report.candidates += len(chosen)
            after = (retired_at, retired.id)

    if not dry_run and after is not None:
        report.cursor = Cursor(retired_at=after[0], particle_id=after[1]).payload()
    report.waiting = await count_update_retirements_after(session, after)
    log.info(
        "paraphrases: %d retirement(s) examined, %d candidate(s), %d retired, %d kept, "
        "%d previously cleared, %d waiting%s%s",
        report.triggers,
        report.candidates,
        len(report.retired),
        len(report.kept),
        report.previously_cleared,
        report.waiting,
        " (capped)" if report.capped else "",
        " (dry run)" if dry_run else "",
    )
    return report

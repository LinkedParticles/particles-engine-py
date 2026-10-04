# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The adjudicability default as one stamped record, and the lens over it.

Three writers keep ``assertion_modality``: extraction (at insert, its stamp
read through the minting snapshot's component record), the regeneration pass
here (:func:`regenerate_modality`, the ``particles modality`` verb), and the
operator verdict here (:func:`reclassify_particle`, ``particles particle
reclassify``). The last two share one writer, ``set_modality_record``, one event
type, ``MODALITY_RECLASSIFIED``, and one consequence: the event puts the claim
in the delta scope, so the next consolidation run re-pairs it under
its new default.

Regeneration writes only in the direction whose error is cheap to recover. A
verdict that would make a claim adjudicable (a flip to ``FALSIFIABLE``) is never
written unattended: it is queued as a ``MODALITY_GRANT_QUEUED`` event, which
lint reports as ``MODALITY_GRANT_PENDING`` for an operator verdict, the same road
a lens grant takes. Wrongly superseding an opinion has no general way back;
leaving a claim unarbitrated does.

The write path is untouched. Every write route still asks ``is_truth_apt``
over the stored value. :class:`ModalityLens` is the read half: an observer's
adopted ``modality_rules`` composed over the stored record, consulted by the
contested badge and the lint queue, never by a write, a rank, or a confidence.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.modality import (
    OPERATOR_CLASSIFIER,
    LensModalityRule,
    ModalityFacts,
    ModalityReading,
    ModalityStamp,
    StampState,
    effective_modality,
    resolve_stamp,
    rules_from_lenses,
    stamp_state,
)
from particles.core.schema import AssertionModality, Particle, ProvenanceRefType
from particles.core.status import Status
from particles.corpus.store import get_entry_source_facts, get_extraction_component_records
from particles.extraction.components import ComponentRecord
from particles.extraction.modality import (
    CLASSIFYING_EXTRACTORS,
    MODALITY_COMPONENTS,
    ModalityVerdict,
    classify_modality,
    current_modality_classifiers,
    regenerable,
    regeneration_classifier,
)
from particles.store.event_store import (
    EventRefKind,
    OperatorEventType,
    list_events_since,
    record_event,
)
from particles.store.lens_store import get_adopted_lens_modality_rules
from particles.store.particle_store import (
    ModalityRecordRow,
    get_modality_records,
    get_particles_by_ids,
    set_modality_record,
)

log = logging.getLogger(__name__)

#: Ids per ``IN`` clause, under SQLite's bound-parameter ceiling.
_BATCH = 500

#: Event actor of a regeneration run (one type, ``actor`` names the route).
REGENERATE_ACTOR = "modality-regenerate"


# ---------------------------------------------------------------------------
# The stamp: resolution and census (§7)
# ---------------------------------------------------------------------------


async def resolve_stamps(
    session: AsyncSession, records: Sequence[ModalityRecordRow]
) -> dict[str, ModalityStamp]:
    """Resolve each record's stamp, reading NULL stamps through the snapshot record.

    One batch query over the minting snapshots of the rows that need it. A
    component record that is incomplete and names no modality component reads
    as no record: part of that snapshot's claims came from an extraction
    nothing describes.
    """
    needed = {
        r.minting_snapshot_id
        for r in records
        if r.classifier is None and r.extractor_name is not None and r.minting_snapshot_id
    }
    component_records: dict[str, ComponentRecord | None] = {}
    ordered = sorted(needed)
    for start in range(0, len(ordered), _BATCH):
        component_records.update(
            await get_extraction_component_records(session, ordered[start : start + _BATCH])
        )
    stamps: dict[str, ModalityStamp] = {}
    for r in records:
        exercised: dict[str, str] | None = None
        record = component_records.get(r.minting_snapshot_id or "")
        if record is not None:
            names_modality = any(name in record.exercised for name in MODALITY_COMPONENTS)
            if record.complete or names_modality:
                exercised = dict(record.exercised)
        stamps[r.particle_id] = resolve_stamp(
            stored_classifier=r.classifier,
            stored_model=r.classifier_model,
            stored_at=r.classified_at,
            extractor_name=r.extractor_name,
            provider_model=r.provider_model,
            snapshot_components=exercised,
            modality_components=MODALITY_COMPONENTS,
            classifying_extractors=CLASSIFYING_EXTRACTORS,
        )
    return stamps


@dataclass
class ModalityCensus:
    """How the ACTIVE store's modality stamps stand against today's classifiers."""

    total: int = 0
    by_state: dict[str, int] = field(default_factory=dict)
    by_classifier: dict[str, int] = field(default_factory=dict)
    stale_by_classifier: dict[str, int] = field(default_factory=dict)
    current: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """A JSON-ready form for the verb's report."""
        return {
            "total": self.total,
            "by_state": dict(self.by_state),
            "by_classifier": dict(self.by_classifier),
            "stale_by_classifier": dict(self.stale_by_classifier),
            "current_classifiers": list(self.current),
        }


@dataclass(frozen=True)
class _StampedRecord:
    record: ModalityRecordRow
    stamp: ModalityStamp
    state: StampState


async def _stamped_records(
    session: AsyncSession, *, particle_ids: Iterable[str] | None = None
) -> list[_StampedRecord]:
    records = await get_modality_records(
        session, particle_ids=list(particle_ids) if particle_ids is not None else None
    )
    stamps = await resolve_stamps(session, records)
    current = current_modality_classifiers()
    return [
        _StampedRecord(r, stamps[r.particle_id], stamp_state(stamps[r.particle_id], current))
        for r in records
    ]


async def modality_census(session: AsyncSession) -> ModalityCensus:
    """Count ACTIVE claims by stamp state and classifier identity."""
    stamped = await _stamped_records(session)
    states: Counter[str] = Counter()
    classifiers: Counter[str] = Counter()
    stale: Counter[str] = Counter()
    for s in stamped:
        states[s.state.value] += 1
        classifiers[s.stamp.classifier] += 1
        if s.state == StampState.STALE:
            stale[s.stamp.classifier] += 1
    return ModalityCensus(
        total=len(stamped),
        by_state={state.value: states.get(state.value, 0) for state in StampState},
        by_classifier=dict(sorted(classifiers.items())),
        stale_by_classifier=dict(sorted(stale.items())),
        current=sorted(current_modality_classifiers()),
    )


async def particle_stamp(
    session: AsyncSession, particle_id: str
) -> tuple[ModalityStamp, StampState] | None:
    """One particle's resolved stamp and state, any status (for ``particle show``)."""
    records = await get_modality_records(session, particle_ids=[particle_id], statuses=None)
    if not records:
        return None
    stamp = (await resolve_stamps(session, records))[particle_id]
    return stamp, stamp_state(stamp, current_modality_classifiers())


# ---------------------------------------------------------------------------
# The operator verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReclassifyResult:
    """What one operator verdict changed."""

    particle_id: str
    prior: AssertionModality
    modality: AssertionModality
    prior_classifier: str
    event_id: str


async def reclassify_particle(
    session: AsyncSession,
    particle_id: str,
    modality: AssertionModality,
    *,
    reason: str,
    actor: str,
) -> ReclassifyResult:
    """Set one claim's adjudicability default by operator verdict. Flushes; the caller commits.

    Writes the record with classifier ``operator`` (which pins it against
    regeneration) and records ``MODALITY_RECLASSIFIED`` carrying the reason,
    the prior and new values, and the prior resolved classifier. The stamp's
    instant puts the claim in the next consolidation run's delta scope, which
    re-pairs it under the new default (§4). Content, confidence, provenance and
    status are untouched.

    Raises:
        ValueError: If the particle is missing or ``reason`` is blank.
    """
    if not reason.strip():
        raise ValueError("a reclassification needs a reason")
    records = await get_modality_records(session, particle_ids=[particle_id], statuses=None)
    if not records:
        raise ValueError(f"particle {particle_id} not found")
    prior_stamp = (await resolve_stamps(session, records))[particle_id]
    prior = await set_modality_record(
        session,
        particle_id,
        modality=modality,
        classifier=OPERATOR_CLASSIFIER,
        classifier_model=None,
        classified_at=datetime.now(UTC),
    )
    assert prior is not None  # only a preserve_operator write is ever skipped
    event = await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.MODALITY_RECLASSIFIED,
        reason=reason,
        refs=[(EventRefKind.PARTICLE, particle_id)],
        payload={
            "from": prior.value,
            "to": modality.value,
            "prior_classifier": prior_stamp.classifier,
        },
    )
    return ReclassifyResult(
        particle_id=particle_id,
        prior=prior,
        modality=modality,
        prior_classifier=prior_stamp.classifier,
        event_id=event.event_id,
    )


# ---------------------------------------------------------------------------
# Regeneration
# ---------------------------------------------------------------------------

Classifier = Callable[[str], Awaitable[ModalityVerdict | None]]


def _in_scope(
    item: _StampedRecord, include_unclassified: bool, queued: frozenset[str] = frozenset()
) -> bool:
    """Whether a regeneration run classifies this claim.

    Stale (and, on request, unclassified) claims, except journal claims, which
    the general rule must not reclassify, and claims with a grant already
    queued under today's classifier, which would only be asked again.
    """
    if item.record.particle_id in queued:
        return False
    if not regenerable(extractor_name=item.record.extractor_name, classifier=item.stamp.classifier):
        return False
    if item.state == StampState.STALE:
        return True
    return include_unclassified and item.state == StampState.UNCLASSIFIED


def _excluded_journal(item: _StampedRecord) -> bool:
    return item.state == StampState.STALE and not regenerable(
        extractor_name=item.record.extractor_name, classifier=item.stamp.classifier
    )


@dataclass(frozen=True)
class PendingGrant:
    """A regeneration verdict that would make a claim adjudicable, awaiting an operator."""

    particle_id: str
    stored: AssertionModality
    classifier: str
    model: str | None
    event_id: str


async def pending_grants(session: AsyncSession) -> dict[str, PendingGrant]:
    """Every queued grant still waiting on an operator, latest per claim.

    A grant stops pending when the claim leaves ACTIVE, when its stored value
    is no longer the one the grant would replace (an operator verdict or a
    later regeneration moved it), or when an operator verdict pinned it. A
    grant made under a classifier rule other than today's is dropped too: the
    claim is stale again and the next run asks afresh.
    """
    events = await list_events_since(
        session,
        since=datetime(1970, 1, 1, tzinfo=UTC),
        event_types=[OperatorEventType.MODALITY_GRANT_QUEUED],
    )
    latest: dict[str, PendingGrant] = {}
    for event in events:  # oldest first, so a later grant replaces an earlier one
        payload = event.payload or {}
        for grant in payload.get("grants", []):
            pid = str(grant["particle_id"])
            latest[pid] = PendingGrant(
                particle_id=pid,
                stored=AssertionModality(grant["from"]),
                classifier=str(payload.get("classifier", "")),
                model=grant.get("model"),
                event_id=event.event_id,
            )
    if not latest:
        return {}
    current = current_modality_classifiers()
    records = {
        r.particle_id: r
        for r in await get_modality_records(session, particle_ids=list(latest), statuses=None)
    }
    pending: dict[str, PendingGrant] = {}
    for pid, grant in latest.items():
        record = records.get(pid)
        if record is None or record.status != Status.ACTIVE.value:
            continue
        if record.classifier == OPERATOR_CLASSIFIER:
            continue
        if record.assertion_modality != grant.stored or grant.classifier not in current:
            continue
        pending[pid] = grant
    return pending


async def reclassified_particle_ids_since(session: AsyncSession, since: datetime) -> set[str]:
    """Claims whose adjudicability default was rewritten after ``since``.

    Read from ``MODALITY_RECLASSIFIED`` events, which record operator verdicts
    and the values a regeneration run changed, never a confirming restamp. The
    consolidation delta scope takes these, so a rewritten default is re-paired
    by the next run and a confirmed one costs nothing.
    """
    events = await list_events_since(
        session, since=since, event_types=[OperatorEventType.MODALITY_RECLASSIFIED]
    )
    return {
        ref.ref_id
        for event in events
        for ref in event.refs
        if ref.ref_kind == EventRefKind.PARTICLE
    }


@dataclass
class RegenerateSummary:
    """What one regeneration run did, or (dry run) would do."""

    dry_run: bool
    backlog: int
    scope: int = 0
    remaining: int = 0
    changed: int = 0
    confirmed: int = 0
    queued: int = 0
    skipped_operator: int = 0
    excluded_journal: int = 0
    failed: int = 0
    failed_ids: list[str] = field(default_factory=list)
    classifier: str = ""
    census: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """A JSON-ready form for the verb's report."""
        return {
            "dry_run": self.dry_run,
            "backlog": self.backlog,
            "scope": self.scope,
            "remaining": self.remaining,
            "changed": self.changed,
            "confirmed": self.confirmed,
            "queued": self.queued,
            "skipped_operator": self.skipped_operator,
            "excluded_journal": self.excluded_journal,
            "failed": self.failed,
            "failed_ids": list(self.failed_ids),
            "classifier": self.classifier,
            "census": self.census,
        }


async def regenerate_modality(
    session: AsyncSession,
    *,
    limit: int | None = None,
    rate_limit_per_minute: int | None = None,
    include_unclassified: bool = False,
    dry_run: bool = False,
    progress: Callable[[str], None] | None = None,
    classify: Classifier | None = None,
) -> RegenerateSummary:
    """Reclassify ACTIVE claims whose modality stamp is stale.

    Built in the ``structure`` mold: a discovered scope, one
    call per claim, the call-aware rate limit, a resumable cap, commits as it
    goes, per-item failures collected and never fatal. A reply with no valid
    verdict writes nothing. One ``MODALITY_RECLASSIFIED`` event records every
    claim whose value changed.

    Three guards keep it on the cheap side of the asymmetry:

    - **Grants are queued, not written.** A verdict of ``FALSIFIABLE`` for a
      claim stored as anything else would let the write path arbitrate it, so
      it is recorded as a ``MODALITY_GRANT_QUEUED`` event for an operator
      verdict instead. Withdrawals and confirmations are written.
    - **Operator verdicts are never overwritten.** The writer re-reads the row
      and skips one an operator pinned after this run computed its scope.
    - **Journal claims are out of scope.** The general rule must not replace the
      journal prompt's verdict; re-extraction reclassifies them.

    Args:
        limit: cap on claims classified this run; defaults to
            ``modality_regeneration.batch_limit``, and 0 means the whole backlog.
        rate_limit_per_minute: max classifier calls per minute; defaults to
            ``modality_regeneration.rate_limit_per_minute``, and 0 disables it.
        include_unclassified: also classify claims no classifier ever ran on.
        dry_run: report the uncapped backlog and the census; write nothing.
        progress: optional callback for human-readable progress lines.
        classify: the classifier seam, for tests; defaults to the LLM call.
    """
    cfg = get_config().modality_regeneration
    effective_limit = cfg.batch_limit if limit is None else limit
    rate = cfg.rate_limit_per_minute if rate_limit_per_minute is None else rate_limit_per_minute
    run_classify = classify if classify is not None else classify_modality

    stamped = await _stamped_records(session)
    queued_ids = frozenset(await pending_grants(session))
    backlog = [s for s in stamped if _in_scope(s, include_unclassified, queued_ids)]
    excluded_journal = sum(1 for s in stamped if _excluded_journal(s))
    classifier = regeneration_classifier()
    if dry_run:
        census = await modality_census(session)
        if progress is not None:
            progress(f"Modality regeneration backlog: {len(backlog)} claims")
        return RegenerateSummary(
            dry_run=True,
            backlog=len(backlog),
            queued=len(queued_ids),
            excluded_journal=excluded_journal,
            classifier=classifier,
            census=census.as_dict(),
        )

    scope = backlog if effective_limit <= 0 else backlog[:effective_limit]
    total = len(scope)
    if progress is not None:
        progress(f"Modality regeneration: classifying {total} of {len(backlog)} claims")

    interval = 60.0 / rate if rate > 0 else 0.0
    changes: list[dict[str, str]] = []
    grants: list[dict[str, str | None]] = []
    confirmed = 0
    skipped_operator = 0
    failed: list[str] = []
    contents = await _contents(session, [s.record.particle_id for s in scope])
    for i, item in enumerate(scope, start=1):
        started = time.monotonic()
        pid = item.record.particle_id
        if progress is not None:
            progress(f"[{i}/{total}] {pid[:8]}… {contents.get(pid, '')[:60]}")
        try:
            verdict = await run_classify(contents.get(pid, ""))
            stored = item.record.assertion_modality
            if verdict is None:
                failed.append(pid)
            elif (
                verdict.modality == AssertionModality.FALSIFIABLE
                and stored != AssertionModality.FALSIFIABLE
            ):
                grants.append(
                    {
                        "particle_id": pid,
                        "from": stored.value,
                        "to": verdict.modality.value,
                        "model": verdict.provider_model,
                    }
                )
            else:
                prior = await set_modality_record(
                    session,
                    pid,
                    modality=verdict.modality,
                    classifier=verdict.classifier,
                    classifier_model=verdict.provider_model,
                    classified_at=datetime.now(UTC),
                    preserve_operator=True,
                )
                if prior is None:
                    skipped_operator += 1
                elif prior == verdict.modality:
                    confirmed += 1
                else:
                    changes.append(
                        {"particle_id": pid, "from": prior.value, "to": verdict.modality.value}
                    )
        except Exception as exc:
            log.error("Classifying particle %s failed: %s", pid, exc)
            failed.append(pid)
        if i % cfg.commit_interval == 0:
            await session.commit()
        if interval > 0:
            remaining_delay = interval - (time.monotonic() - started)
            if remaining_delay > 0:
                await asyncio.sleep(remaining_delay)

    if changes:
        await record_event(
            session,
            actor=REGENERATE_ACTOR,
            event_type=OperatorEventType.MODALITY_RECLASSIFIED,
            refs=[(EventRefKind.PARTICLE, c["particle_id"]) for c in changes],
            payload={"batch": True, "classifier": classifier, "changes": changes},
        )
    if grants:
        await record_event(
            session,
            actor=REGENERATE_ACTOR,
            event_type=OperatorEventType.MODALITY_GRANT_QUEUED,
            refs=[(EventRefKind.PARTICLE, str(g["particle_id"])) for g in grants],
            payload={"batch": True, "classifier": classifier, "grants": grants},
        )
    await session.commit()

    still_queued = frozenset(await pending_grants(session))
    remaining = [
        s
        for s in await _stamped_records(session)
        if _in_scope(s, include_unclassified, still_queued)
    ]
    return RegenerateSummary(
        dry_run=False,
        backlog=len(backlog),
        scope=total,
        remaining=len(remaining),
        changed=len(changes),
        confirmed=confirmed,
        queued=len(grants),
        skipped_operator=skipped_operator,
        excluded_journal=excluded_journal,
        failed=len(failed),
        failed_ids=failed,
        classifier=classifier,
        census=(await modality_census(session)).as_dict(),
    )


async def _contents(session: AsyncSession, particle_ids: list[str]) -> dict[str, str]:
    return {
        pid: p.content for pid, p in (await get_particles_by_ids(session, particle_ids)).items()
    }


# ---------------------------------------------------------------------------
# The lens
# ---------------------------------------------------------------------------


@dataclass
class ModalityLens:
    """One observer's adopted ``modality_rules``, with the facts they read memoised.

    The one reading every read route consults, so the badge and the queue
    cannot disagree. Never stored, never consulted by a write.
    """

    rules: list[LensModalityRule]
    _readings: dict[str, ModalityReading] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        """True when no adopted lens says anything about modality."""
        return not self.rules

    async def readings(
        self, session: AsyncSession, particles: Sequence[Particle]
    ) -> dict[str, ModalityReading]:
        """Each claim's effective reading under this lens.

        Free on an empty lens: every claim reads as its stored default, with no
        store access.
        """
        todo = [p for p in particles if p.id not in self._readings]
        if todo:
            if self.empty:
                for p in todo:
                    self._readings[p.id] = ModalityReading(p.assertion_modality)
            else:
                facts = await _facts(session, todo, self.rules)
                for p in todo:
                    self._readings[p.id] = effective_modality(facts[p.id], self.rules)
        return {p.id: self._readings[p.id] for p in particles}


async def load_modality_lens(session: AsyncSession) -> ModalityLens:
    """The viewer's lens: every adopted lens's modality rules."""
    return ModalityLens(rules_from_lenses(await get_adopted_lens_modality_rules(session)))


async def _facts(
    session: AsyncSession, particles: Sequence[Particle], rules: Sequence[LensModalityRule]
) -> dict[str, ModalityFacts]:
    """Gather what the rules match on: operator pins, subjects, and sources if asked."""
    records = await get_modality_records(
        session, particle_ids=[p.id for p in particles], statuses=None
    )
    pinned = {r.particle_id for r in records if r.classifier == OPERATOR_CLASSIFIER}
    needs_sources = any(r.rule.scope in ("source_type", "url_pattern") for r in rules)
    sources: dict[str, tuple[str, str | None]] = {}
    if needs_sources:
        entry_ids = {
            ref.corpus_entry_id
            for p in particles
            for ref in p.provenance
            if ref.type == ProvenanceRefType.SOURCE
        }
        sources = await get_entry_source_facts(session, entry_ids)
    facts: dict[str, ModalityFacts] = {}
    for p in particles:
        entries = [
            sources[ref.corpus_entry_id]
            for ref in p.provenance
            if ref.type == ProvenanceRefType.SOURCE and ref.corpus_entry_id in sources
        ]
        facts[p.id] = ModalityFacts(
            particle_id=p.id,
            stored=p.assertion_modality,
            operator_pinned=p.id in pinned,
            subject_ids=frozenset(p.subject_ids),
            source_types=frozenset(source_type for source_type, _ in entries),
            source_uris=tuple(uri for _, uri in entries if uri),
        )
    return facts

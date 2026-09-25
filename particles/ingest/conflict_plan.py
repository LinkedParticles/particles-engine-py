# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The §6.6 verdict-to-writes table: a pure plan the pipeline applies (D2).

:func:`particles.core.conflict_resolution.decide_ladder` says what the ladder
decided for one (existing, new) pair. :func:`plan_conflict_writes` says what
that means for the store: which row is inserted and with which birth status,
what is then demoted, whether an INCONSISTENCY record or a
``CONFLICT_CANDIDATE_DROPPED`` event is written, how a batch caller's candidate
cache changes, and what the pipeline returns. ``pipeline._apply_conflict_plan``
performs the writes, in this order:

  1. insert :attr:`ConflictWritePlan.insert` (``ACTIVE``, or born quarantined);
  2. demote :attr:`ConflictWritePlan.demote` to ``PROVENANCE_STALE``;
  3. insert the INCONSISTENCY :attr:`ConflictWritePlan.wrapper`;
  4. record the dropped-candidate event;
  5. record the observer divergence.

The mirror verdicts insert ``ACTIVE`` and *then* demote the row just inserted,
because the insert seam forbids a born-``PROVENANCE_STALE`` row other than the
``CONFLICT_PENDING`` quarantine birth. ``supersedes`` is filled only
on the row being minted, and only when it is unset: it is write-once
(D1).

The plan lives beside the pipeline, not in ``core/``, because the event type
and the batch cache are Engine vocabulary. It performs no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from particles.core.conflict_resolution import (
    ConflictVerdict,
    LadderOutcome,
    build_inconsistency_particle,
)
from particles.core.schema import Particle, ProvenanceRefType
from particles.core.status import Status, StatusReason
from particles.store.event_store import EventRefKind, OperatorEventType

#: The actor every write this plan describes is attributed to.
PIPELINE_ACTOR = "extract-pipeline"


@dataclass(frozen=True)
class PlannedEvent:
    """One operator event to record, with the arguments ``record_event`` takes."""

    event_type: OperatorEventType
    reason: str
    refs: tuple[tuple[EventRefKind, str], ...]
    payload: Mapping[str, object]
    actor: str = PIPELINE_ACTOR


@dataclass(frozen=True)
class ConflictWritePlan:
    """Every store effect of one ladder outcome, in apply order."""

    insert: Particle | None = None
    """The row to insert, with its birth status and reason (and ``supersedes``)
    already set. It is stored with the candidate's embedding."""
    demote: tuple[str, StatusReason] | None = None
    """``(particle id, reason)`` to move to ``PROVENANCE_STALE`` after the insert."""
    wrapper: Particle | None = None
    """The INCONSISTENCY record, inserted after the quarantined row it names."""
    wrapper_domain_hint: str | None = None
    """The Extension B ``domain_hint`` the wrapper is inserted with."""
    dropped_event: PlannedEvent | None = None
    """The audited drop of a candidate trust resolution ruled redundant."""
    record_divergence: bool = False
    """Record the inserted row and ``existing`` as an observer divergence."""
    cache_drop_existing: bool = False
    """``existing`` leaves the batch candidate cache and joins ``retired_out``."""
    cache_add_new: bool = False
    """The inserted (``ACTIVE``) row joins the batch candidate cache."""
    result: Particle | None = None
    """What the pipeline returns for the pair."""
    log_message: str | None = None
    """The info line the apply logs, ``%``-style, with :attr:`log_args`."""
    log_args: tuple[object, ...] = field(default_factory=tuple)


def _score(value: float | None) -> float:
    return value if value is not None else float("nan")


def plan_quarantine(
    existing: Particle,
    new: Particle,
    *,
    corpus_entry_id: str,
    snapshot_id: str,
    trigger_ref_type: ProvenanceRefType,
    retired_twin: bool = False,
) -> tuple[Particle, Particle]:
    """The quarantined candidate and the INCONSISTENCY record that names it.

    the losing candidate is persisted as a real particle born
    quarantined, so Review can recover it and the wrapper's B ref resolves (it
    was once a dangling id). ``PROVENANCE_STALE`` keeps it off query and lint;
    ``CONFLICT_PENDING`` carries the semantics and is what the insert seam
    requires for this birth. ``retired_twin`` marks ``existing`` as a claim
    retired by judgment rather than a live one.
    """
    quarantined = new.model_copy(
        update={
            "status": Status.PROVENANCE_STALE,
            "status_reason": StatusReason.CONFLICT_PENDING,
        }
    )
    wrapper = build_inconsistency_particle(
        existing,
        quarantined,
        corpus_entry_id=corpus_entry_id,
        snapshot_id=snapshot_id,
        asserted_by=PIPELINE_ACTOR,
        trigger_ref_type=trigger_ref_type,
        retired_twin=retired_twin,
    )
    return quarantined, wrapper


def plan_retired_hold(
    twin: Particle,
    new: Particle,
    *,
    corpus_entry_id: str,
    snapshot_id: str,
    trigger_ref_type: ProvenanceRefType,
    domain: str | None,
) -> ConflictWritePlan:
    """Hold a candidate that re-asserts a judgment-retired ``twin``.

    The candidate is stored quarantined behind an INCONSISTENCY record carrying
    the retired-value marker; the twin itself is not touched.
    """
    held, wrapper = plan_quarantine(
        twin,
        new,
        corpus_entry_id=corpus_entry_id,
        snapshot_id=snapshot_id,
        trigger_ref_type=trigger_ref_type,
        retired_twin=True,
    )
    return ConflictWritePlan(
        insert=held,
        wrapper=wrapper,
        wrapper_domain_hint=domain,
        result=wrapper,
        log_message=(
            "Retired-value hold: candidate %r re-asserts %s (%s); stored quarantined"
            " as %s behind INCONSISTENCY %s"
        ),
        log_args=(
            new.content[:60],
            twin.id[:8],
            twin.status_reason.value if twin.status_reason else twin.status.value,
            held.id[:8],
            wrapper.id[:8],
        ),
    )


def _demoted(particle: Particle, reason: StatusReason) -> Particle:
    return particle.model_copy(update={"status": Status.PROVENANCE_STALE, "status_reason": reason})


def plan_conflict_writes(
    outcome: LadderOutcome,
    new: Particle,
    existing: Particle,
    *,
    corpus_entry_id: str,
    snapshot_id: str,
    trigger_ref_type: ProvenanceRefType,
    scores: tuple[float | None, float | None],
    domain: str | None,
) -> ConflictWritePlan:
    """Map one ladder outcome to its writes.

    ``scores`` is ``(trust_score_new, trust_score_existing)``, recorded in the
    dropped-candidate event and the trust-resolution log lines. ``domain`` is
    the INCONSISTENCY record's ``domain_hint``.
    """
    score_new, score_existing = scores
    new_short, existing_short = new.id[:8], existing.id[:8]

    if outcome.verdict is None:
        # declined. The candidate stands beside the existing claim.
        return ConflictWritePlan(
            insert=new,
            record_divergence=outcome.record_divergence,
            cache_add_new=True,
            result=new,
            log_message=(
                "Observer precondition: new %s left standing beside %s, "
                "which another project observes"
            ),
            log_args=(new_short, existing_short),
        )

    verdict = outcome.verdict
    match verdict:
        case ConflictVerdict.CORROBORATES:
            return ConflictWritePlan(
                insert=new,
                cache_add_new=True,
                result=new,
                log_message=(
                    "High-similarity pair without contradiction signal — new %s written as"
                    " ACTIVE alongside existing %s (no §6.6 conflict)"
                ),
                log_args=(new_short, existing_short),
            )

        case ConflictVerdict.SUPERSEDES:
            return ConflictWritePlan(
                insert=new,
                demote=(existing.id, StatusReason.LOWER_TRUST_SOURCE),
                cache_drop_existing=True,
                cache_add_new=True,
                result=new,
                log_message="Trust resolution: new %s (%.2f) preferred over existing %s (%.2f)",
                log_args=(new_short, _score(score_new), existing_short, _score(score_existing)),
            )

        case ConflictVerdict.SUPERSEDED_BY_EXISTING:
            # The candidate stays a drop (redundant with a strictly better
            # existing claim), but an audited one. It is never
            # persisted, so it appears in the payload only, not as a record ref.
            return ConflictWritePlan(
                dropped_event=PlannedEvent(
                    event_type=OperatorEventType.CONFLICT_CANDIDATE_DROPPED,
                    reason="§6.6 trust resolution preferred the existing particle",
                    refs=((EventRefKind.PARTICLE, existing.id),),
                    payload={
                        "verdict": verdict.value,
                        "candidate_id": new.id,
                        "candidate_excerpt": new.content[:240],
                        "winning_particle_id": existing.id,
                        "trust_score_new": score_new,
                        "trust_score_existing": score_existing,
                    },
                ),
                result=None,
                log_message=(
                    "Trust resolution: existing %s (%.2f) preferred over new %s (%.2f);"
                    " new particle dropped (event logged)"
                ),
                log_args=(existing_short, _score(score_existing), new_short, _score(score_new)),
            )

        case ConflictVerdict.DOCUMENT_SUPERSEDES:
            # Rung 1.5 (cap. 2): the trust rung's demotion shape, under
            # the demotion-only invariant. No INCONSISTENCY is queued.
            return ConflictWritePlan(
                insert=new,
                demote=(existing.id, StatusReason.DOCUMENT_SUPERSEDED),
                cache_drop_existing=True,
                cache_add_new=True,
                result=new,
                log_message=(
                    "Document-supersession (rung 1.5): new %s supersedes existing %s"
                    " → existing demoted DOCUMENT_SUPERSEDED"
                ),
                log_args=(new_short, existing_short),
            )

        case ConflictVerdict.DOCUMENT_SUPERSEDED_BY_EXISTING:
            # Rung 1.5 mirror: store the loser demoted, insert-then-transition.
            return ConflictWritePlan(
                insert=new,
                demote=(new.id, StatusReason.DOCUMENT_SUPERSEDED),
                result=_demoted(new, StatusReason.DOCUMENT_SUPERSEDED),
                log_message=(
                    "Document-supersession (rung 1.5): existing %s supersedes new %s"
                    " → new stored DOCUMENT_SUPERSEDED"
                ),
                log_args=(existing_short, new_short),
            )

        case ConflictVerdict.UPDATE_SUPERSEDES:
            # Rung 2.5: the newer claim names the one it replaces (the
            # as-of lens dates the retirement from it). Filled only on
            # the row being minted, and only when unset: write-once.
            minted = (
                new.model_copy(update={"supersedes": existing.id})
                if new.supersedes is None
                else new
            )
            return ConflictWritePlan(
                insert=minted,
                demote=(existing.id, StatusReason.SUPERSEDED_BY_UPDATE),
                cache_drop_existing=True,
                cache_add_new=True,
                result=minted,
                log_message=("Update supersession (rung 2.5): new %s supersedes existing %s"),
                log_args=(new_short, existing_short),
            )

        case ConflictVerdict.UPDATE_SUPERSEDED_BY_EXISTING:
            # Rung 2.5 mirror: an out-of-order older claim, stored demoted.
            return ConflictWritePlan(
                insert=new,
                demote=(new.id, StatusReason.SUPERSEDED_BY_UPDATE),
                result=_demoted(new, StatusReason.SUPERSEDED_BY_UPDATE),
                log_message=(
                    "Update supersession (rung 2.5): existing %s is newer than new %s"
                    " → new stored SUPERSEDED_BY_UPDATE"
                ),
                log_args=(existing_short, new_short),
            )

        case ConflictVerdict.INCONSISTENT:
            quarantined, wrapper = plan_quarantine(
                existing,
                new,
                corpus_entry_id=corpus_entry_id,
                snapshot_id=snapshot_id,
                trigger_ref_type=trigger_ref_type,
            )
            return ConflictWritePlan(
                insert=quarantined,
                wrapper=wrapper,
                wrapper_domain_hint=domain,
                result=wrapper,
                log_message=(
                    "INCONSISTENCY particle %s created (conflicts: %s ↔ quarantined %s,"
                    " domain=%s, subject_ids=%d inherited)"
                ),
                log_args=(
                    wrapper.id,
                    existing.id,
                    quarantined.id,
                    domain,
                    len(wrapper.subject_ids),
                ),
            )

        case ConflictVerdict.NO_CONFLICT:
            # Reachable only if a caller uses the ladder for below-threshold
            # pairs; the candidate is kept ACTIVE rather than dropped.
            return ConflictWritePlan(insert=new, cache_add_new=True, result=new)

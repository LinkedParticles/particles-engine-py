# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Operator event log — append-only audit of operator decisions.

This is **storage-layer bookkeeping, not particle schema**: it adds no field
to :class:`~particles.core.schema.Particle` and does not touch
``SCHEMA_VERSION``. Every operator-initiated mutation that meets the
inclusion criterion records one immutable event here, *in the same
transaction as the mutation*.

``record_event()`` is called from the operation / store layer — never a CLI
command body — so the event fires regardless of which front-end (CLI, HTTP,
or MCP) triggered the change. The ``actor`` argument carries the interface
entry-point (the CLI verb, HTTP route, or MCP tool) and is reserved to become
the authenticated principal once multi-user lands.

The log is read-exposed identically on all three front-ends via the shared
:func:`list_events` / :func:`get_event` helpers.
"""

from __future__ import annotations

import uuid
from collections.abc import Collection, Sequence
from contextvars import ContextVar
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from sqlalchemy import JSON, DateTime, Index, String, Text, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.db import Base


class OperatorEventType(StrEnum):
    """The kind of operator action an event records.

    One value per distinct *outcome*: opposite outcomes get distinct types
    (``PARTICLE_TAGGED`` vs ``PARTICLE_UNTAGGED``) so ``--type`` is a precise
    filter; the *same* outcome reached two ways (``trust set`` /
    ``trust statement-set`` both write a SourceTrustStatement) shares one type,
    with ``actor`` recording which verb. Extensible for later verbs that meet
    the inclusion criterion.
    """

    SOURCE_RETRACTED = "SOURCE_RETRACTED"
    # D1: an operator hard-deleted a corpus entry (`corpus delete`).
    # The CORPUS_ENTRY ref names the entry; PARTICLE refs name the particles
    # deleted with it and those stripped of its refs. The payload carries
    # counts only, never deleted content (no claim text, URI or chunk hash),
    # which would defeat the delete's privacy purpose.
    CORPUS_ENTRY_DELETED = "CORPUS_ENTRY_DELETED"
    # `import vault` reclassified an existing STABLE note's entry
    # MUTABLE (actor ``import``; CORPUS_ENTRY ref; payload ``from`` / ``to``).
    # An entry's mutability is the operator's policy about the source, so it is
    # reconciled in place; this event is its durable history. Retires nothing.
    CORPUS_ENTRY_RECLASSIFIED = "CORPUS_ENTRY_RECLASSIFIED"
    # Security review 2026-10-26-1, W26: a deposit took a URL a web capture's
    # frontmatter had named, so the capture's entry gave up its URI-R (R13).
    # System-emitted (actor ``deposit``); CORPUS_ENTRY ref names the capture's
    # entry; payload ``uri_r`` (the URL released) and ``released_to`` (the
    # depositor that now holds it). The entry keeps its snapshots and tags.
    CORPUS_ENTRY_URI_RELEASED = "CORPUS_ENTRY_URI_RELEASED"
    # `rules forget` removed a recorded rule-source root (actor
    # ``cli:rules-forget``). No record ref: a root is a path, not a record. The
    # payload carries the path, the verb that recorded it and when, and the
    # counts the verb printed: entries that stopped being refreshed, and
    # entries another route still admits (in total and by route).
    RULE_SOURCE_ROOT_FORGOTTEN = "RULE_SOURCE_ROOT_FORGOTTEN"
    # system-emitted (not operator-initiated) — the §6.6
    # SUPERSEDED_BY_EXISTING verdict drops the candidate without persisting
    # it, and the drop must be auditable: the event carries the candidate
    # excerpt, the verdict, and the winning particle id.
    CONFLICT_CANDIDATE_DROPPED = "CONFLICT_CANDIDATE_DROPPED"
    SUBJECTS_SPLIT = "SUBJECTS_SPLIT"
    SUBJECTS_MERGED = "SUBJECTS_MERGED"
    SUBJECT_DELETED = "SUBJECT_DELETED"
    SUBJECT_ALIASED = "SUBJECT_ALIASED"
    # : operator overrode a subject's Nomisma class when the
    # resolver mis-classed it. Only the current class is kept (overwritten on
    # the next change), so the event log is the durable history.
    SUBJECT_RECLASSIFIED = "SUBJECT_RECLASSIFIED"
    SUBJECT_LINK_CONFIRMED = "SUBJECT_LINK_CONFIRMED"
    SUBJECT_LINK_REMOVED = "SUBJECT_LINK_REMOVED"
    # Subjects attached in place to particles that had none — the
    # batch `subjects relink-gated` run (``batch: true``, one event per run)
    # or the per-card assign-subject gesture (one particle). A PARTICLE ref per
    # relinked particle and a SUBJECT ref per Subject attached; the payload
    # records each link's recovery tier and gate class. No particle is
    # superseded, so this event is the only trace of the change.
    SUBJECTS_RELINKED = "SUBJECTS_RELINKED"
    TRUST_CHANGED = "TRUST_CHANGED"
    REVIEW_RESOLVED = "REVIEW_RESOLVED"
    RELATION_ADDED = "RELATION_ADDED"
    RELATION_REMOVED = "RELATION_REMOVED"
    PARTICLE_TAGGED = "PARTICLE_TAGGED"
    PARTICLE_UNTAGGED = "PARTICLE_UNTAGGED"
    # agent/operator direct-assertion write verbs over the MCP surface.
    PARTICLE_ASSERTED = "PARTICLE_ASSERTED"
    PARTICLE_SUPERSEDED = "PARTICLE_SUPERSEDED"
    PARTICLE_RETRACTED = "PARTICLE_RETRACTED"
    # an operator dismissed / snoozed a citation-signal deposit
    # suggestion. The suppressed URL + window ride the payload (no record ref
    # of the existing kinds — a URL is not a particle / subject / corpus entry).
    DEPOSIT_SUGGESTION_DISMISSED = "DEPOSIT_SUGGESTION_DISMISSED"
    # curation session-model writes. BELIEF_AFFIRMED records an
    # operator "still true" gesture: audited, it suppresses the card and does
    # NOT touch confidence. Agreeing with a belief is a different act, an
    # ENDORSES stance, which never feeds confidence either (§5(b)).
    # The affirmed belief id(s) ride as PARTICLE refs. CURATION_CARD_SNOOZED records a skipped
    # card keyed by ``card_key`` in the payload (+ ``snoozed_until``); a belief
    # card also carries its PARTICLE refs, a uncited_url card reuses the existing
    # DEPOSIT_SUGGESTION_DISMISSED path instead.
    BELIEF_AFFIRMED = "BELIEF_AFFIRMED"
    CURATION_CARD_SNOOZED = "CURATION_CARD_SNOOZED"
    # one event per completed consolidation cycle (or interactive
    # audit — same payload shape, ``actor`` distinguishes ``memory-consolidate``
    # from ``audit``). System-emitted like CONFLICT_CANDIDATE_DROPPED; the
    # versioned payload (``format: 1``) carries per-pass status/durations, the
    # census counts, per-pass LLM call counts, and the ``completed_at`` the next
    # run's delta scope keys off; this folds an earlier plan.
    CONSOLIDATION_RUN = "CONSOLIDATION_RUN"
    # one event per `particles extract` run (a single snapshot, an
    # `--all-pending` pass, or a `POST /extract` call) that made at least one
    # LLM call. System-emitted; the versioned payload (``format: 1``) carries
    # the measured ``llm_usage`` in the CONSOLIDATION_RUN shape, so a store's
    # cumulative spend is a sum over the two types. Its own type rather than a
    # CONSOLIDATION_RUN, whose latest event keys the cycle's delta scope.
    # `particles reindex` (CLI and `POST /reindex`) and `particles
    # structure` record here too, under ``route: reindex`` / ``route: structure``.
    EXTRACT_RUN = "EXTRACT_RUN"
    # one event per census INCONSISTENCY record the nightly
    # disclosure pass closes without a review — ``cause`` is ``lapsed`` (a side
    # has no stated member left) or ``regrouped`` (replaced by a record for the
    # changed group, whose id the payload names). A record extraction opened
    # closes here too, with ``origin: extraction``: ``cause: withdrawn`` when a
    # second reading no longer confirms it, ``cause:
    # partner_retired`` when the release pass frees its held claim.
    # System-emitted, like CONFLICT_CANDIDATE_DROPPED and
    # DUPLICATES_MERGED:
    # the status change alone does not record which member stopped being
    # stated, or by which snapshot, and the coverage rule reads the cause back
    # from here. The record rides as a PARTICLE ref, then each dropped member.
    INCONSISTENCY_CLOSED = "INCONSISTENCY_CLOSED"
    # one event per claim the nightly re-anchor pass judged
    # dependent on a state an update retired. ``outcome: unrestated`` keeps the
    # claim ACTIVE and backs a STALE_BASIS curation card (the cursor has moved
    # on, so nothing else remembers the verdict); ``outcome: matched_existing``
    # retired it in favour of an ACTIVE claim that already states the
    # restatement, which the status change alone does not name; ``outcome:
    # restated`` retired it in favour of the restatement the pass wrote, so
    # every retirement the pass makes is in the log even when the run's own
    # event is never written (security review 2026-10-25-1, Q36).
    # System-emitted, like INCONSISTENCY_CLOSED. Refs: the claim, the retired
    # state claim, then the other claim when there is one (the match, the
    # restatement, or the belief a restatement would have contradicted).
    DEPENDENT_CLAIM_EXAMINED = "DEPENDENT_CLAIM_EXAMINED"
    # One event per claim a rung 2.5 route retires SUPERSEDED_BY_UPDATE (ADR
    # 0268): the extraction ladder, its extras, the assertion door and the
    # backlog sweep. The winner's ``supersedes`` holds one id, so a winner that
    # retires a second claim (an extra, or a sweep pair once the pointer is
    # taken) left no record of what retired it, and the demotion card showed
    # nothing. Payload: ``retired_id``, ``winner_id`` and ``route``; refs: both
    # claims. System-emitted, like DEPENDENT_CLAIM_EXAMINED.
    UPDATE_SUPERSESSION = "UPDATE_SUPERSESSION"
    # One event per claim the trust rung (rung 2) retires
    # LOWER_TRUST_SOURCE, and one per claim the document-supersession prior
    # (rung 1.5) retires DOCUMENT_SUPERSEDED, on the write
    # path or in the reconcile sweep. Every automatic retirement is then in the
    # log an operator reads after depositing untrusted material; before, these
    # two left only a log line (security review 2026-10-24-2, P34). Payload:
    # ``retired_id``, ``winner_id``, ``route`` (``ladder`` or ``sweep``), and the
    # two trust scores for a trust retirement or the two corpus entries for a
    # document one. Refs: both claims. System-emitted, like UPDATE_SUPERSESSION,
    # and kept apart from it because its readers (the held-claim release, the
    # demotion cards) read rung 2.5's winner only.
    TRUST_SUPERSESSION = "TRUST_SUPERSESSION"
    DOCUMENT_SUPERSESSION = "DOCUMENT_SUPERSESSION"
    # one event per examination by the nightly release pass of a
    # write-time record whose stored side an autonomous pass retired, when an
    # ACTIVE claim still contradicts the held one, so nothing was written to
    # any claim. Payload: ``record_id``, ``partner_id`` (the retired side),
    # ``held_id``, ``retired_by`` (the claim that retired the partner, when one
    # is on record) and ``blocking_ids``. The event is the pass's memory: a
    # record is examined again only once a blocking claim has left ACTIVE, and
    # the record's curation card reads its line from here. A release writes an
    # INCONSISTENCY_CLOSED with ``cause: partner_retired`` instead. Refs: the
    # record, the partner, the held claim, then each blocking claim.
    # System-emitted, like DEPENDENT_CLAIM_EXAMINED.
    HELD_CLAIM_EXAMINED = "HELD_CLAIM_EXAMINED"
    # one event per propose-mode abstraction candidate. The
    # payload is the full candidate (claim, rationale, premise ids, subject
    # ids, derived confidence) — the event log is the candidate's persistence,
    # so the curation card finder and the accept gesture re-read it without a
    # candidate table. Refs: one PARTICLE ref per premise.
    ABSTRACTION_CANDIDATE = "ABSTRACTION_CANDIDATE"
    # the operator's verdict on a candidate — payload carries
    # ``resolution: accepted|rejected``, the ``candidate_event_id`` it
    # resolves, and (on accept) the asserted particle id. Every resolution is
    # a labelled datapoint for the §8 faithfulness evaluation.
    ABSTRACTION_RESOLVED = "ABSTRACTION_RESOLVED"
    # one event per derived belief the revalidation ladder moves
    # out of ACTIVE, so every claim the abstraction pass retires is named in the
    # log (security review 2026-10-25-2, R29). Payload: ``retired_id``,
    # ``successor_id`` (``None`` for a floor retirement, or when §6.6 declined
    # the re-synthesized claim), ``rung`` (1 to 4), ``outcome``
    # (``refreshed_structural``, ``refreshed_entailed``, ``refreshed_paraphrase``,
    # ``superseded`` or ``retired``), ``status`` and ``decided_by``. Refs: the
    # retired claim, then its successor. System-emitted, like
    # DEPENDENT_CLAIM_EXAMINED.
    ABSTRACTION_RETIRED = "ABSTRACTION_RETIRED"
    # One event per pass of a structural retiring path that moved at least one
    # claim out of ACTIVE: a lint fix (``route``: ``lint_staleness``,
    # ``lint_retraction_cascade`` or ``lint_corpus_link``), the snapshot
    # generation cascade (``generation_cascade``) and reindex
    # (``reindex``). Before, these retired claims with no event, so the event
    # log an operator reads after a deposit missed them (security review
    # 2026-10-28-1, O36). ``batch: true``; payload ``route``, ``retired_ids``,
    # ``status``, ``status_reason`` and, where the pass has one, ``entry_id``
    # and ``snapshot_id``. Refs: each retired claim, then the corpus entry.
    # System-emitted, like ABSTRACTION_RETIRED; ``actor`` names the verb.
    CLAIMS_RETIRED = "CLAIMS_RETIRED"
    # one event per auto-merged exact-duplicate group. Carries the
    # survivor id, every superseded id, the content hash, and the config values in
    # force — this is both the audit trail and the input to the revert path (§5),
    # which the `status_reason = DUPLICATE_MERGED` / `created_by = EXACT_DUPLICATE`
    # tagging lets select precisely auto-merge's own writes.
    DUPLICATES_MERGED = "DUPLICATES_MERGED"
    # one event per reverted merge group. Carries the source
    # ``merge_event_id``, the survivor (untouched), the restored ids, the
    # skipped ids with their reasons, and the relation count deleted. The
    # DUPLICATES_MERGED event it reverts is never deleted or edited — the pair
    # is the audit trail, and an event log that rewrites itself is not one.
    DUPLICATES_UNMERGED = "DUPLICATES_UNMERGED"

    # an operator marked a belief useful (`particles memory useful
    # <id>`), the explicit second channel of the usefulness lens.
    # Deliberately NOT `BELIEF_AFFIRMED`: that gesture means "still true" — a
    # claim about the *world* that suppresses a curation card — while this one
    # means "this earned its place", a claim about the belief's *use*. Reusing
    # the affirm type would retro-credit every historical card-clearing as
    # utility evidence and make each thumbs-up silently suppress a card.
    # The credited belief rides as a PARTICLE ref; the payload carries the §7
    # daily `credit_key` and whether this press produced a new credit or
    # collapsed into an existing one, so the log explains the arithmetic rather
    # than merely listing presses. A press without operator authority (an agent
    # key on `POST /memory/useful`) carries `operator_credit: false` and no
    # `credit_key`, and credits nothing (amended 2026-10-27).
    BELIEF_MARKED_USEFUL = "BELIEF_MARKED_USEFUL"

    # observer scope. RESCOPED is also the store's "the lens may
    # engage" marker — a project observer is honoured only once a rescope has
    # run, so an upgrade never silently empties a digest. Its payload carries
    # the counts (keys added, entries left unattributed) and, since
    # §3, every ``[entry_id, key]`` pair it added (``added``). WIDENED / REVOKED are
    # the operator's standing judgement that a belief (PARTICLE ref) or a whole
    # source (CORPUS_ENTRY ref) applies in every project; distinct types because
    # they are opposite outcomes.
    OBSERVER_SCOPE_RESCOPED = "OBSERVER_SCOPE_RESCOPED"
    # One entry given a key by hand (`rescope --assign`). Its own type because
    # RESCOPED is the engage marker: attributing one source must not, by
    # itself, turn the lens on for a store whose other sources are still stale.
    # The deposit key reconcile records one too, actor ``deposit``.
    OBSERVER_SCOPE_KEY_ASSIGNED = "OBSERVER_SCOPE_KEY_ASSIGNED"
    OBSERVER_SCOPE_WIDENED = "OBSERVER_SCOPE_WIDENED"
    OBSERVER_SCOPE_WIDEN_REVOKED = "OBSERVER_SCOPE_WIDEN_REVOKED"
    # one existing agent write marked ``scope:unattributed`` by
    # `rescope --mark-agent-writes` (CORPUS_ENTRY ref; payload ``identity``,
    # the depositor that matched), and a recorded mark reversed by
    # `rescope --unmark`. A marker is not a key, hence types of their own.
    OBSERVER_SCOPE_MARKED_UNATTRIBUTED = "OBSERVER_SCOPE_MARKED_UNATTRIBUTED"
    OBSERVER_SCOPE_UNMARKED = "OBSERVER_SCOPE_UNMARKED"
    # one wrong key removed from one entry by `rescope --withdraw`
    # (CORPUS_ENTRY ref; payload ``withdrawn``, and ``marked`` when the entry's
    # last key went and it gained the ``scope:unattributed`` marker).
    OBSERVER_SCOPE_KEY_WITHDRAWN = "OBSERVER_SCOPE_KEY_WITHDRAWN"

    # the vocabulary document (the profile carrier).
    # CHANGED is the TRUST_CHANGED analogue: a version materialised, or an
    # adoption added or removed (payload ``kind`` says which). PROPOSED is one
    # alias or profile candidate the store's census suggested; like
    # ABSTRACTION_CANDIDATE, the event log is the proposal's persistence, so no
    # candidate table exists. RULED is the reviewer's verdict on a proposal, or
    # a direct ruling such as an alignment (payload ``resolution``:
    # ``confirmed`` / ``declined`` / ``aligned``), naming the document version
    # it produced; it is the "who confirmed it, when, on what evidence" the
    # document's own PROV ruling repeats.
    VOCABULARY_CHANGED = "VOCABULARY_CHANGED"
    VOCABULARY_PROPOSED = "VOCABULARY_PROPOSED"
    VOCABULARY_RULED = "VOCABULARY_RULED"

    # a claim's adjudicability default (``assertion_modality``) was
    # rewritten after insert. One type for one outcome, two routes told apart
    # by ``actor``: an operator verdict (``cli:particle-reclassify``; one
    # PARTICLE ref, the reason, payload ``from`` / ``to`` / ``prior_classifier``)
    # and a regeneration run (``modality-regenerate``; ``batch: true``, a
    # PARTICLE ref per changed claim, each change in ``changes``). The record's
    # stamp columns keep only the latest writer, so this log is the history.
    MODALITY_RECLASSIFIED = "MODALITY_RECLASSIFIED"
    # a regeneration verdict that would make claims adjudicable (a
    # flip to FALSIFIABLE), which regeneration never writes unattended. The
    # event is the queue's persistence, the ABSTRACTION_CANDIDATE pattern: one
    # per run (``batch: true``), a PARTICLE ref per claim, ``grants`` listing
    # each claim's stored value and the verdict, ``classifier`` the rule that
    # ruled. Lint reads it back as MODALITY_GRANT_PENDING until an operator
    # verdict, a later value, or a rule change settles it.
    MODALITY_GRANT_QUEUED = "MODALITY_GRANT_QUEUED"
    # an operator set (or cleared) one claim's world-time record
    # (``valid_from`` / ``valid_kind``; one PARTICLE ref, the reason, the
    # values in the payload). Basis ``operator``, which the ``particles
    # validity`` prose default never overwrites.
    PARTICLE_DATED = "PARTICLE_DATED"
    # ``particles memory redate`` gave harvested sessions their own
    # start as the reference date and re-anchored their ``session``-basis
    # claims to it. One event per run, the counts in the payload.
    SESSION_DATES_REPAIRED = "SESSION_DATES_REPAIRED"
    # Retired with agent-side extraction (deprecated): nothing writes
    # these any more. They stay so the events already stored remain readable;
    # never reuse the names. ASKED recorded a Stop hook's ask, CONTINUED where
    # the answering turn ended, SUBMITTED the verified submission's counts.
    AGENT_EXTRACTION_ASKED = "AGENT_EXTRACTION_ASKED"
    AGENT_EXTRACTION_CONTINUED = "AGENT_EXTRACTION_CONTINUED"
    AGENT_EXTRACTION_SUBMITTED = "AGENT_EXTRACTION_SUBMITTED"


class EventRefKind(StrEnum):
    """The kind of record an event ref points at.

    The queryable record types, extensible alongside :class:`OperatorEventType`.
    An event with no ref of these kinds (rare) carries its targets in
    ``payload`` only.
    """

    PARTICLE = "particle"
    SUBJECT = "subject"
    CORPUS_ENTRY = "corpus_entry"
    TRUST_STATEMENT = "trust_statement"


# ---------------------------------------------------------------------------
# ORM
# ---------------------------------------------------------------------------


class OperatorEventRow(Base):
    """Append-only event header — what happened, when, by whom, why."""

    __tablename__ = "operator_events"

    event_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    actor: Mapped[str] = mapped_column(String, nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_operator_events_type", "event_type"),
        Index("ix_operator_events_occurred", "occurred_at"),
    )


class OperatorEventRefRow(Base):
    """Append-only join from an event to a record it touched.

    The composite primary key ``(event_id, ref_kind, ref_id)`` already indexes
    by-``event_id`` lookups (leftmost column); ``ix_event_refs_record`` serves
    the record-centric query *"what operator actions touched record X?"*.
    """

    __tablename__ = "operator_event_refs"

    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    ref_kind: Mapped[str] = mapped_column(String, primary_key=True)
    ref_id: Mapped[str] = mapped_column(String, primary_key=True)

    __table_args__ = (Index("ix_event_refs_record", "ref_kind", "ref_id"),)


# ---------------------------------------------------------------------------
# Pydantic read models (also the HTTP response_model — storage metadata, kept
# out of the frozen particle schema in core/)
# ---------------------------------------------------------------------------


class OperatorEventRef(BaseModel):
    """One record an event touched."""

    ref_kind: EventRefKind
    ref_id: str


class OperatorEvent(BaseModel):
    """A single operator event with its refs."""

    event_id: str
    occurred_at: datetime
    actor: str
    event_type: OperatorEventType
    reason: str | None = None
    refs: list[OperatorEventRef] = []
    payload: dict[str, Any] | None = None


def _to_model(row: OperatorEventRow, ref_rows: Sequence[OperatorEventRefRow]) -> OperatorEvent:
    return OperatorEvent(
        event_id=row.event_id,
        occurred_at=row.occurred_at,
        actor=row.actor,
        event_type=OperatorEventType(row.event_type),
        reason=row.reason,
        refs=[
            OperatorEventRef(ref_kind=EventRefKind(r.ref_kind), ref_id=r.ref_id) for r in ref_rows
        ],
        payload=row.payload,
    )


# ---------------------------------------------------------------------------
# Write seam — call inside the mutation's own transaction, in the
# operation/store layer (never a CLI body)..
# ---------------------------------------------------------------------------


#: The HTTP caller events are recorded for (rule 6): the principal's
#: id and the key id, set by the engine's route gate for the request it admits,
#: ``None`` outside an HTTP request. Lives here, not in the API layer, because
#: the Engine may not import the Surface.
_http_actor: ContextVar[tuple[str, str] | None] = ContextVar("particles_http_actor", default=None)

#: The payload fields :func:`record_event` adds for an HTTP caller.
HTTP_PRINCIPAL_FIELD = "http_principal"
HTTP_KEY_ID_FIELD = "http_key_id"


def bind_http_actor(principal: str | None, key_id: str | None) -> None:
    """Name the HTTP caller the current request's events are recorded for, or clear it."""
    _http_actor.set(None if principal is None or key_id is None else (principal, key_id))


async def record_event(
    session: AsyncSession,
    *,
    actor: str,
    event_type: OperatorEventType,
    reason: str | None = None,
    refs: Sequence[tuple[EventRefKind, str]] = (),
    payload: dict[str, Any] | None = None,
) -> OperatorEvent:
    """Append one operator event plus its record refs.

    Inserts into ``operator_events`` and ``operator_event_refs`` and flushes,
    leaving the caller to ``commit()`` so the event and the mutation it
    records share one transaction. Duplicate ``(ref_kind, ref_id)`` pairs are
    de-duplicated to avoid composite-PK collisions.

    Args:
        session: The same session running the mutation.
        actor: Interface entry-point that triggered the action (CLI verb,
            HTTP route, or MCP tool).
        event_type: The :class:`OperatorEventType`.
        reason: Operator free-text rationale, if any.
        refs: ``(EventRefKind, record_id)`` pairs the event touched.
        payload: Event-type-specific structured detail. During an HTTP request
            the gate's caller is added as ``http_principal`` and
            ``http_key_id`` (:func:`bind_http_actor`).

    Returns:
        The persisted :class:`OperatorEvent`.
    """
    event_id = str(uuid.uuid4())
    occurred_at = datetime.now(UTC)
    http_actor = _http_actor.get()
    if http_actor is not None:
        # Added beside the event's own fields, never replacing ``actor``, so a
        # reader of today's events sees nothing removed or renamed.
        payload = {
            **(payload or {}),
            HTTP_PRINCIPAL_FIELD: http_actor[0],
            HTTP_KEY_ID_FIELD: http_actor[1],
        }
    session.add(
        OperatorEventRow(
            event_id=event_id,
            occurred_at=occurred_at,
            actor=actor,
            event_type=event_type.value,
            reason=reason,
            payload=payload,
        )
    )
    seen: set[tuple[str, str]] = set()
    ref_models: list[OperatorEventRef] = []
    for kind, ref_id in refs:
        key = (kind.value, ref_id)
        if key in seen:
            continue
        seen.add(key)
        session.add(OperatorEventRefRow(event_id=event_id, ref_kind=kind.value, ref_id=ref_id))
        ref_models.append(OperatorEventRef(ref_kind=kind, ref_id=ref_id))
    await session.flush()
    return OperatorEvent(
        event_id=event_id,
        occurred_at=occurred_at,
        actor=actor,
        event_type=event_type,
        reason=reason,
        refs=ref_models,
        payload=payload,
    )


async def record_claims_retired(
    session: AsyncSession,
    *,
    actor: str,
    route: str,
    retired_ids: Sequence[str],
    status: str,
    status_reason: str | None,
    reason: str,
    entry_id: str | None = None,
    snapshot_id: str | None = None,
) -> OperatorEvent | None:
    """Name the claims one structural pass moved out of ACTIVE (security review 2026-10-28-1, O36).

    One ``CLAIMS_RETIRED`` event per pass, flushed with the status changes it
    records, so the caller commits both together. Writes nothing when the pass
    retired nothing.

    Args:
        session: The session that ran the status changes.
        actor: The verb that ran the pass (``lint``, ``reindex``, ``extract``, ...).
        route: Which retiring path ran (see :attr:`OperatorEventType.CLAIMS_RETIRED`).
        retired_ids: The claims retired, in the order the pass retired them.
        status: The status they moved to.
        status_reason: The ``StatusReason`` value recorded with the status.
        reason: A short sentence saying why the pass retired them.
        entry_id: The corpus entry the pass worked on, when it had one.
        snapshot_id: The snapshot the pass measured against, when it had one.

    Returns:
        The persisted event, or ``None`` when ``retired_ids`` is empty.
    """
    ids = list(dict.fromkeys(retired_ids))
    if not ids:
        return None
    refs: list[tuple[EventRefKind, str]] = [(EventRefKind.PARTICLE, pid) for pid in ids]
    payload: dict[str, Any] = {
        "batch": True,
        "route": route,
        "retired_ids": ids,
        "status": status,
        "status_reason": status_reason,
    }
    if entry_id is not None:
        refs.append((EventRefKind.CORPUS_ENTRY, entry_id))
        payload["entry_id"] = entry_id
    if snapshot_id is not None:
        payload["snapshot_id"] = snapshot_id
    return await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.CLAIMS_RETIRED,
        reason=reason,
        refs=refs,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Read seam — shared by the CLI, HTTP, and MCP front-ends.
# ---------------------------------------------------------------------------


def ref_filter(
    *,
    particle: str | None = None,
    subject: str | None = None,
    corpus_entry: str | None = None,
) -> tuple[EventRefKind | None, str | None]:
    """Resolve the mutually-exclusive record filters into a ``(kind, id)`` pair.

    Shared by all three read front-ends (CLI / HTTP / MCP) so their filter
    semantics stay identical. Raises :class:`ValueError` if more than one of
    ``particle`` / ``subject`` / ``corpus_entry`` is supplied; returns
    ``(None, None)`` when none is.
    """
    chosen = [
        (kind, value)
        for kind, value in (
            (EventRefKind.PARTICLE, particle),
            (EventRefKind.SUBJECT, subject),
            (EventRefKind.CORPUS_ENTRY, corpus_entry),
        )
        if value is not None
    ]
    if len(chosen) > 1:
        raise ValueError("Provide at most one of particle / subject / corpus_entry.")
    if not chosen:
        return None, None
    return chosen[0]


async def _refs_for(session: AsyncSession, event_id: str) -> Sequence[OperatorEventRefRow]:
    result = await session.execute(
        select(OperatorEventRefRow).where(OperatorEventRefRow.event_id == event_id)
    )
    return result.scalars().all()


async def get_event(session: AsyncSession, event_id: str) -> OperatorEvent | None:
    """Return one event with its refs, or ``None`` if absent."""
    row = await session.get(OperatorEventRow, event_id)
    if row is None:
        return None
    return _to_model(row, await _refs_for(session, event_id))


async def list_events(
    session: AsyncSession,
    *,
    ref_kind: EventRefKind | None = None,
    ref_id: str | None = None,
    event_type: OperatorEventType | None = None,
    http_principal: str | None = None,
    http_key_id: str | None = None,
    limit: int = 50,
) -> list[OperatorEvent]:
    """List events newest-first, optionally filtered by record ref, type or HTTP caller.

    Args:
        session: An open session.
        ref_kind: Restrict to events touching this record kind. Only meaningful
            together with ``ref_id``.
        ref_id: Restrict to events touching this record id (e.g. a particle or
            subject id). Uses the ``(ref_kind, ref_id)`` index.
        event_type: Restrict to a single :class:`OperatorEventType`.
        http_principal: Restrict to events an HTTP caller with this principal
            wrote (rule 6).
        http_key_id: Restrict to events one key's requests wrote.
        limit: Maximum events to return (default 50).
    """
    stmt = select(OperatorEventRow)
    if ref_id is not None:
        stmt = stmt.join(
            OperatorEventRefRow,
            OperatorEventRefRow.event_id == OperatorEventRow.event_id,
        ).where(OperatorEventRefRow.ref_id == ref_id)
        if ref_kind is not None:
            stmt = stmt.where(OperatorEventRefRow.ref_kind == ref_kind.value)
    if event_type is not None:
        stmt = stmt.where(OperatorEventRow.event_type == event_type.value)
    if http_principal is not None:
        stmt = stmt.where(
            OperatorEventRow.payload[HTTP_PRINCIPAL_FIELD].as_string() == http_principal
        )
    if http_key_id is not None:
        stmt = stmt.where(OperatorEventRow.payload[HTTP_KEY_ID_FIELD].as_string() == http_key_id)
    stmt = stmt.order_by(desc(OperatorEventRow.occurred_at), OperatorEventRow.event_id).limit(limit)
    result = await session.execute(stmt)
    rows = result.scalars().unique().all()
    return [_to_model(row, await _refs_for(session, row.event_id)) for row in rows]


async def count_events_by_type(
    session: AsyncSession, *, until: datetime | None = None
) -> dict[str, int]:
    """How many events the log holds of each type, one grouped query.

    ``until`` counts only the events that occurred at or before that instant;
    the log is append-only, so the count as of a past instant is exact.
    """
    stmt = select(OperatorEventRow.event_type, func.count())
    if until is not None:
        stmt = stmt.where(OperatorEventRow.occurred_at <= until)
    result = await session.execute(stmt.group_by(OperatorEventRow.event_type))
    return {event_type: int(n) for event_type, n in result.all()}


async def list_events_in_range(
    session: AsyncSession,
    *,
    event_type: OperatorEventType,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[OperatorEvent]:
    """Every event of one type in a half-open time window, oldest-first.

    Unlike :func:`list_events` this is deliberately unbounded — a caller
    reverting a batch must see the whole window, since a silent
    limit would report a partial revert as a complete one. Bounds are inclusive
    on ``since`` and exclusive on ``until``; either may be ``None``.
    """
    stmt = select(OperatorEventRow).where(OperatorEventRow.event_type == event_type.value)
    if since is not None:
        stmt = stmt.where(OperatorEventRow.occurred_at >= since)
    if until is not None:
        stmt = stmt.where(OperatorEventRow.occurred_at < until)
    stmt = stmt.order_by(OperatorEventRow.occurred_at, OperatorEventRow.event_id)
    result = await session.execute(stmt)
    rows = result.scalars().unique().all()
    return [_to_model(row, await _refs_for(session, row.event_id)) for row in rows]


async def list_events_since(
    session: AsyncSession,
    *,
    since: datetime,
    event_types: Collection[OperatorEventType] | None = None,
) -> list[OperatorEvent]:
    """Every event of the given types recorded after ``since``, oldest-first.

    The post-snapshot staleness filter: "what has the operator
    resolved since this curation collection was built?" Distinct from
    :func:`list_events_in_range`, which takes exactly one type — this takes a
    set, because the question spans every *resolving* gesture (retract,
    supersede, merge, deposit, review, abstraction verdict) at once and issuing
    one query per type would be a needless fan-out on a read that runs on every
    `GET /curation`.

    Unbounded like its siblings: a silent ``limit`` would let a busy morning's
    tail of resolutions slip through and resurrect handled cards, which is the
    exact failure the filter exists to prevent. Bounded in practice by ``since``
    being recent (last night's build).
    """
    stmt = select(OperatorEventRow).where(OperatorEventRow.occurred_at > since)
    if event_types is not None:
        values = [t.value for t in event_types]
        if not values:
            return []
        stmt = stmt.where(OperatorEventRow.event_type.in_(values))
    stmt = stmt.order_by(OperatorEventRow.occurred_at, OperatorEventRow.event_id)
    result = await session.execute(stmt)
    rows = result.scalars().unique().all()
    return [_to_model(row, await _refs_for(session, row.event_id)) for row in rows]


async def list_particle_events(
    session: AsyncSession, event_type: OperatorEventType
) -> list[tuple[str, datetime, dict[str, Any] | None]]:
    """Every event of ``event_type`` that names a particle, oldest-first.

    Returns ``(particle_id, occurred_at, payload)`` per ``(event, PARTICLE ref)``
    pair — an event touching N particles yields N tuples.

    Unlike :func:`list_events` this is **unbounded and ascending**, because its
    callers replay the whole log rather than show an operator the newest slice.
    A ``limit``-bounded read is the wrong shape for reconstruction: it would
    silently drop the oldest credits once the log outgrew the window, and a
    rebuild that quietly loses history is worse than one that is slow. Ascending
    order makes the replay deterministic.
    """
    stmt = (
        select(
            OperatorEventRefRow.ref_id,
            OperatorEventRow.occurred_at,
            OperatorEventRow.payload,
        )
        .join(
            OperatorEventRefRow,
            OperatorEventRefRow.event_id == OperatorEventRow.event_id,
        )
        .where(
            OperatorEventRow.event_type == event_type.value,
            OperatorEventRefRow.ref_kind == EventRefKind.PARTICLE.value,
        )
        .order_by(OperatorEventRow.occurred_at, OperatorEventRow.event_id)
    )
    rows = (await session.execute(stmt)).all()
    # Indexed rather than unpacked: SQLAlchemy 2.1 types this row as
    # ``Row[*tuple[Any, ...]]``, whose unpacking mypy widens to ``object``.
    return [(row[0], row[1], row[2]) for row in rows]

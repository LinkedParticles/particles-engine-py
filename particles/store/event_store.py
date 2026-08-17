"""Operator event log — append-only audit of operator decisions.

This is **storage-layer bookkeeping, not particle schema**: it adds no field
to :class:`~particles.core.schema.Particle` and does not touch
``SCHEMA_VERSION``. Every operator-initiated mutation that meets the inclusion criterion records one immutable event here, *in the same
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
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from sqlalchemy import JSON, DateTime, Index, String, Text, desc, select
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
    # system-emitted (not operator-initiated) — the §6.6
    # SUPERSEDED_BY_EXISTING verdict drops the candidate without persisting
    # it, and the drop must be auditable: the event carries the candidate
    # excerpt, the verdict, and the winning particle id.
    CONFLICT_CANDIDATE_DROPPED = "CONFLICT_CANDIDATE_DROPPED"
    SUBJECTS_SPLIT = "SUBJECTS_SPLIT"
    SUBJECTS_MERGED = "SUBJECTS_MERGED"
    SUBJECT_DELETED = "SUBJECT_DELETED"
    SUBJECT_ALIASED = "SUBJECT_ALIASED"
    #: operator overrode a subject's Nomisma class when the
    # resolver mis-classed it. Only the current class is kept (overwritten on
    # the next change), so the event log is the durable history.
    SUBJECT_RECLASSIFIED = "SUBJECT_RECLASSIFIED"
    SUBJECT_LINK_CONFIRMED = "SUBJECT_LINK_CONFIRMED"
    SUBJECT_LINK_REMOVED = "SUBJECT_LINK_REMOVED"
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
    # operator "still true" gesture (audited, suppresses the card; does NOT touch
    # confidence — first-class corroboration is vouch). The affirmed
    # belief id(s) ride as PARTICLE refs. CURATION_CARD_SNOOZED records a skipped
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
    # than merely listing presses.
    BELIEF_MARKED_USEFUL = "BELIEF_MARKED_USEFUL"


class EventRefKind(StrEnum):
    """The kind of record an event ref points at.

    The queryable record types — extensible alongside :class:`OperatorEventType`.
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
        payload: Event-type-specific structured detail.

    Returns:
        The persisted :class:`OperatorEvent`.
    """
    event_id = str(uuid.uuid4())
    occurred_at = datetime.now(UTC)
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
    limit: int = 50,
) -> list[OperatorEvent]:
    """List events newest-first, optionally filtered by record ref and/or type.

    Args:
        session: An open session.
        ref_kind: Restrict to events touching this record kind. Only meaningful
            together with ``ref_id``.
        ref_id: Restrict to events touching this record id (e.g. a particle or
            subject id). Uses the ``(ref_kind, ref_id)`` index.
        event_type: Restrict to a single :class:`OperatorEventType`.
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
    stmt = stmt.order_by(desc(OperatorEventRow.occurred_at), OperatorEventRow.event_id).limit(limit)
    result = await session.execute(stmt)
    rows = result.scalars().unique().all()
    return [_to_model(row, await _refs_for(session, row.event_id)) for row in rows]


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
    return [
        (pid, occurred, payload)
        for pid, occurred, payload in (await session.execute(stmt)).tuples().all()
    ]

# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Agent-side extraction, the store half: record asks, apply submissions.

Gather / decide / apply: the decisions (the ask rule, the evidence
verifier, the coverage rule) are pure functions in
:mod:`particles.extraction.agent`; this module reads what they need from the
store and writes what they decide. A submission's candidates enter the store
through :func:`~particles.ingest.pipeline.extract_snapshot`, the same
post-candidate path the general extractor's take, so every rung applies
unchanged: the subject gate, §6.6, duplicate suppression, and the
retired-value quarantine.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import ExtractionStatus
from particles.corpus.deposit import load_blob
from particles.corpus.store import (
    ExtractionBaseRow,
    get_entry,
    list_extraction_bases,
    reopen_snapshot,
    update_extraction_status,
)
from particles.extraction.agent import (
    EXTRACTOR_VERSION,
    SubmittedCandidatesExtractor,
    covers_span,
    purpose_servable,
    verify_submission,
)
from particles.extraction.general import extraction_text, source_text_flags
from particles.ingest.pipeline import extract_snapshot
from particles.store.event_store import (
    EventRefKind,
    OperatorEvent,
    OperatorEventType,
    get_event,
    list_events,
    record_event,
)

log = logging.getLogger(__name__)

#: How many of an entry's agent-extraction events one read considers.
_EVENT_SCAN_LIMIT = 10_000


def covered_through(rows: list[ExtractionBaseRow]) -> int:
    """How far an APPEND_ONLY entry has been extracted, in bytes of its latest COMPLETE snapshot.

    A COMPLETE snapshot with no recorded ``extracted_through`` was read whole,
    so its blob's length is the extent.
    """
    for row in reversed(rows):
        if row.extraction_status is not ExtractionStatus.COMPLETE:
            continue
        if row.extracted_through is not None:
            return row.extracted_through
        try:
            return len(load_blob(row.content_hash))
        except FileNotFoundError:
            continue
    return 0


@dataclass(frozen=True)
class SessionAsks:
    """One transcript entry's asks and the continuations recorded for them."""

    asks: list[OperatorEvent]
    continued: dict[str, str]
    """ask id → the last transcript record of the agent's answering turn."""
    submitted: set[str]

    def exclusion_spans(self) -> list[tuple[str, str | None]]:
        """``(after_uuid, through_uuid)`` per ask, for ``ask_exclusions``."""
        out: list[tuple[str, str | None]] = []
        for ask in self.asks:
            after = (ask.payload or {}).get("after_uuid")
            if isinstance(after, str):
                out.append((after, self.continued.get(ask.event_id)))
        return out

    def last_ask_answered(self) -> bool:
        """Whether the session's latest ask was submitted (True when there is none)."""
        if not self.asks:
            return True
        latest = max(self.asks, key=lambda e: e.occurred_at)
        return latest.event_id in self.submitted

    def asked_at(self) -> list[datetime]:
        """When each ask was made, timezone-aware.

        SQLite drops the zone on read, and the ask rule subtracts these from
        an aware ``now``.
        """
        return [_as_utc(a.occurred_at) for a in self.asks]

    def last_ask_start(self) -> int | None:
        """Where the session's latest ask's span began, or ``None`` when there is none."""
        if not self.asks:
            return None
        return _span(max(self.asks, key=lambda e: e.occurred_at))[0]

    def asked_through(self) -> int:
        """How far the session's answered asks reached, in bytes of the distilled text.

        The ask cursor of a shadow install: a shadow answer
        writes nothing and never moves the extraction base, so the next ask
        starts here instead. It is never an extraction base, and the
        background extractor never reads it.
        """
        return max((_span(a)[1] for a in self.asks if a.event_id in self.submitted), default=0)

    def open_ask(self) -> OperatorEvent | None:
        """The latest ask no continuation has been recorded for."""
        for ask in sorted(self.asks, key=lambda e: e.occurred_at, reverse=True):
            if ask.event_id not in self.continued:
                return ask
        return None


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _span(ask: OperatorEvent) -> tuple[int, int]:
    """An ask's recorded ``[start, end)`` byte span; ``(0, 0)`` when it has none."""
    span = (ask.payload or {}).get("span")
    if isinstance(span, list) and len(span) == 2 and all(isinstance(x, int) for x in span):
        return span[0], span[1]
    return 0, 0


async def load_session_asks(session: AsyncSession, entry_id: str) -> SessionAsks:
    """Every agent-extraction event recorded against one transcript entry."""

    async def of(kind: OperatorEventType) -> list[OperatorEvent]:
        return await list_events(
            session,
            ref_kind=EventRefKind.CORPUS_ENTRY,
            ref_id=entry_id,
            event_type=kind,
            limit=_EVENT_SCAN_LIMIT,
        )

    asks = await of(OperatorEventType.AGENT_EXTRACTION_ASKED)
    continued: dict[str, str] = {}
    for event in await of(OperatorEventType.AGENT_EXTRACTION_CONTINUED):
        payload = event.payload or {}
        ask_id, through = payload.get("ask_id"), payload.get("through_uuid")
        if isinstance(ask_id, str) and isinstance(through, str):
            continued[ask_id] = through
    submitted = {
        str((e.payload or {}).get("ask_id"))
        for e in await of(OperatorEventType.AGENT_EXTRACTION_SUBMITTED)
    }
    return SessionAsks(asks=asks, continued=continued, submitted=submitted)


async def record_ask(
    session: AsyncSession,
    *,
    entry_id: str,
    snapshot_id: str,
    session_id: str,
    store: str,
    start: int,
    end: int,
    provider_model: str,
    after_uuid: str,
    mode: str,
) -> OperatorEvent:
    """Record an ask. Every field the particles will carry is fixed here."""
    ttl = get_config().claude_code.agent_extraction.ask_ttl_seconds
    return await record_event(
        session,
        actor="claude-code-hook",
        event_type=OperatorEventType.AGENT_EXTRACTION_ASKED,
        refs=[(EventRefKind.CORPUS_ENTRY, entry_id)],
        payload={
            "session_id": session_id,
            "snapshot_id": snapshot_id,
            "store": store,
            "span": [start, end],
            "provider_model": provider_model,
            "extractor_version": EXTRACTOR_VERSION,
            "after_uuid": after_uuid,
            "expires_at": (datetime.now(UTC) + timedelta(seconds=ttl)).isoformat(),
            "mode": mode,
        },
    )


async def record_continuation(
    session: AsyncSession, *, entry_id: str, ask_id: str, through_uuid: str
) -> OperatorEvent:
    """Record where the agent's answering turn ended."""
    return await record_event(
        session,
        actor="claude-code-hook",
        event_type=OperatorEventType.AGENT_EXTRACTION_CONTINUED,
        refs=[(EventRefKind.CORPUS_ENTRY, entry_id)],
        payload={"ask_id": ask_id, "through_uuid": through_uuid},
    )


class SubmissionRefused(ValueError):
    """A submission the server will not apply at all: unknown, expired, or repeated."""


@dataclass
class SubmitOutcome:
    """What a submission did. Counts only, never claim text."""

    ask_id: str
    mode: str
    submitted: int = 0
    accepted: int = 0
    malformed: int = 0
    dropped_no_evidence: int = 0
    dropped_tool_line: int = 0
    dropped_already_extracted: int = 0
    covered: bool = False
    written: int = 0
    closed_span: bool = False
    notes: list[str] = field(default_factory=list)

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


async def apply_submission(
    session: AsyncSession, *, ask_id: str, store: str, candidates: list[Any]
) -> SubmitOutcome:
    """Verify a submission against its ask and, in ``on`` mode, write it (§5).

    Raises :class:`SubmissionRefused` when the ask is unknown, expired,
    already answered, or recorded against another store.
    """
    cfg = get_config().claude_code.agent_extraction
    ask = await get_event(session, ask_id)
    if ask is None or ask.event_type is not OperatorEventType.AGENT_EXTRACTION_ASKED:
        raise SubmissionRefused(f"no agent-extraction ask {ask_id!r}")
    payload = ask.payload or {}
    entry_ref = next((r.ref_id for r in ask.refs if r.ref_kind is EventRefKind.CORPUS_ENTRY), None)
    if entry_ref is None or payload.get("store") != store:
        raise SubmissionRefused(f"ask {ask_id!r} was not made against store {store!r}")
    expires = datetime.fromisoformat(str(payload.get("expires_at")))
    if datetime.now(UTC) > expires:
        raise SubmissionRefused(f"ask {ask_id!r} expired at {expires.isoformat()}")
    asks = await load_session_asks(session, entry_ref)
    if ask_id in asks.submitted:
        raise SubmissionRefused(f"ask {ask_id!r} was already answered")

    entry = await get_entry(session, entry_ref)
    snapshot_id = str(payload["snapshot_id"])
    snapshot = (
        None
        if entry is None
        else next((s for s in entry.snapshots if s.snapshot_id == snapshot_id), None)
    )
    if entry is None or snapshot is None:
        raise SubmissionRefused(f"ask {ask_id!r} names a snapshot this store no longer holds")
    content = load_blob(snapshot.content_hash)
    start, end = (int(x) for x in payload["span"])
    span_text = content[start:end].decode("utf-8", errors="replace")
    verified = verify_submission(candidates, span_text, span_start=start)
    out = SubmitOutcome(
        ask_id=ask_id,
        mode=cfg.mode,
        submitted=verified.submitted,
        malformed=verified.malformed,
        dropped_no_evidence=verified.no_evidence,
        dropped_tool_line=verified.tool_line_only,
    )

    # §5, the background race: a pass that extracted past the ask's base since
    # the ask was made owns the text before its new extent.
    rows = await list_extraction_bases(session, entry_ref)
    now_through = covered_through(rows)
    accepted = verified.accepted
    if now_through > start:
        kept = [
            c
            for c in accepted
            if c.provenance_location is not None
            and int(c.provenance_location.split("=")[1].split("-")[0]) >= now_through
        ]
        out.dropped_already_extracted = len(accepted) - len(kept)
        accepted = kept
    out.accepted = len(accepted)
    verified.accepted = accepted
    out.covered = covers_span(verified, end - max(start, now_through), cfg)
    background = purpose_servable("extraction")

    if cfg.mode != "on" or now_through >= end:
        if now_through >= end:
            out.notes.append("span already extracted by another pass; nothing written")
        await _record_submitted(session, entry_ref, out)
        return out

    is_markdown, mark_tools = source_text_flags(entry.source_type)
    read_through = (
        None
        if end >= len(content)
        else len(extraction_text(content[:end], is_markdown=is_markdown, mark_tools=mark_tools)[0])
    )
    # The snapshot must be claimable: PENDING, or FAILED by an earlier attempt.
    if snapshot.extraction_status is ExtractionStatus.COMPLETE:
        out.notes.append("snapshot already COMPLETE; nothing written")
        await _record_submitted(session, entry_ref, out)
        return out
    await session.commit()
    written = await extract_snapshot(
        session,
        entry_ref,
        snapshot_id,
        extractor=SubmittedCandidatesExtractor(
            accepted, provider_model=str(payload["provider_model"]), read_through=read_through
        ),
        agent_id="claude-code-agent",
        append_base=None,
        provider_model=str(payload["provider_model"]),
        fail_closed_probes=not purpose_servable("semantic_lint"),
    )
    out.written = len(written)

    if out.covered or not background:
        # Close the span, and every earlier snapshot it covers: under the
        # APPEND_ONLY promise each is a prefix of this one.
        out.closed_span = True
        for row in rows:
            if row.snapshot_id == snapshot_id:
                break
            if row.extraction_status in (ExtractionStatus.PENDING, ExtractionStatus.FAILED):
                await update_extraction_status(
                    session,
                    row.snapshot_id,
                    ExtractionStatus.COMPLETE,
                    extracted_through=len(load_blob(row.content_hash)),
                )
        if not out.covered:
            out.notes.append(
                "submission below the coverage floor; closed anyway (no background extractor)"
            )
    else:
        # Under-covered with a background extractor: keep the particles and
        # leave the span owed, so agent recall never replaces background recall.
        await reopen_snapshot(session, snapshot_id)
        out.notes.append("submission below the coverage floor; span left to the background pass")
    await _record_submitted(session, entry_ref, out)
    return out


async def _record_submitted(session: AsyncSession, entry_id: str, out: SubmitOutcome) -> None:
    await record_event(
        session,
        actor="mcp:extraction_submit",
        event_type=OperatorEventType.AGENT_EXTRACTION_SUBMITTED,
        refs=[(EventRefKind.CORPUS_ENTRY, entry_id)],
        payload=out.as_payload(),
    )
    await session.commit()
    log.info("Agent-extraction submission %s: %s", out.ask_id, out.as_payload())

"""``events_list`` / ``event_show`` — read the operator event log.

Read-only access to the audit trail of operator decisions (retract / split /
merge / alias / confirm / unlink / trust / review / links / tags), so an agent
can answer *"what operator actions touched this record? why was this source
retracted? what's the trust-change history?"* directly in its loop. Consistent
with the read-only MCP posture — no mutating tool.
"""

from __future__ import annotations

from typing import Any


async def events_list(
    particle: str | None = None,
    subject: str | None = None,
    corpus_entry: str | None = None,
    event_type: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List operator events, newest first, optionally filtered.

    Args:
        particle: Only events touching this particle id.
        subject: Only events touching this subject id.
        corpus_entry: Only events touching this corpus entry id.
            (At most one of ``particle`` / ``subject`` / ``corpus_entry``.)
        event_type: Only events of this type, e.g. ``SOURCE_RETRACTED``,
            ``SUBJECTS_SPLIT``, ``TRUST_CHANGED``, ``REVIEW_RESOLVED``.
        limit: Maximum events to return (default 50).
    """
    from particles.api.client import get_backend

    # The backend applies the "at most one ref" + event-type validation
    # (ref_filter / OperatorEventType) server-side, raising ValueError on misuse.
    events = await get_backend().events_list(
        particle=particle, subject=subject, entry=corpus_entry, event_type=event_type, limit=limit
    )
    return [e.model_dump(mode="json") for e in events]


async def event_show(event_id: str) -> dict[str, Any]:
    """Show one operator event in full — header, refs, and payload.

    Args:
        event_id: The event's id.
    """
    from particles.api.client import get_backend

    event = await get_backend().event_show(event_id)
    if event is None:
        raise ValueError(f"Event {event_id!r} not found.")
    return event.model_dump(mode="json")

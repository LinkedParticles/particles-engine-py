# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""What the operator's snoozes, dismissals and affirmations hide.

Shared by the collection half (``collect.py``, which groups subjectless
beliefs by source) and the live session half (``session.py``), so
the two read the same rulings and cannot disagree.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from particles.store.event_store import EventRefKind, OperatorEventType, list_events

from .cards import CardKind
from .duplicate_groups import GroupRulings

#: The key prefix whose permanent dismissal is member-scoped.
_SOURCE_PREFIX = f"{CardKind.SUBJECTLESS_SOURCE.value}:"

#: The key prefix whose permanent rulings are partitions of their members.
_GROUP_PREFIX = f"{CardKind.DUPLICATE_GROUP.value}:"


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _is_group_ruling(payload: dict[str, object]) -> bool:
    """A permanent ``split`` / ``dismiss`` of a duplicate group, with its partition.

    A permanent event on a group key with no ``classes`` (a plain
    ``POST /curation/snooze`` from an older client) is not one: it hides the
    key like any other dismissal, so it is never silently ignored.
    """
    key = payload.get("card_key")
    return (
        isinstance(key, str)
        and key.startswith(_GROUP_PREFIX)
        and payload.get("snoozed_until") is None
        and isinstance(payload.get("classes"), list)
    )


def _is_source_dismissal(payload: dict[str, object]) -> bool:
    key = payload.get("card_key")
    return (
        isinstance(key, str)
        and key.startswith(_SOURCE_PREFIX)
        and payload.get("snoozed_until") is None
    )


async def suppressed_keys(session: AsyncSession) -> set[str]:
    """The card keys hidden by an unexpired snooze or a standing affirmation.

    A permanent dismissal of a ``subjectless_source`` card hides no key: it
    rules on the beliefs it named, which :func:`dismissed_members` reports, so
    a later subjectless belief from the same source raises the card again.
    A permanent ``split`` or ``dismiss`` of a ``duplicate_group``
    card hides no key either: it partitions the claims it named, which
    :func:`group_rulings` reports.
    """
    now = datetime.now(UTC)
    suppressed: set[str] = set()

    for ev in await list_events(
        session, event_type=OperatorEventType.CURATION_CARD_SNOOZED, limit=10_000
    ):
        payload = ev.payload or {}
        key = payload.get("card_key")
        if not isinstance(key, str):
            continue
        until_raw = payload.get("snoozed_until")
        if until_raw is None:  # permanent dismiss
            if not _is_source_dismissal(payload) and not _is_group_ruling(payload):
                suppressed.add(key)
            continue
        try:
            until = datetime.fromisoformat(str(until_raw))
        except ValueError:
            continue
        if _as_utc(until) > now:
            suppressed.add(key)

    for ev in await list_events(
        session, event_type=OperatorEventType.BELIEF_AFFIRMED, limit=10_000
    ):
        key = (ev.payload or {}).get("card_key")
        if isinstance(key, str):
            suppressed.add(key)

    return suppressed


async def dismissed_members(session: AsyncSession) -> set[str]:
    """The beliefs a standing ``subjectless_source`` dismissal names.

    The one reader of those dismissals: the grouping step drops these beliefs
    before grouping, and the live half prunes them from a served card.
    """
    members: set[str] = set()
    for ev in await list_events(
        session, event_type=OperatorEventType.CURATION_CARD_SNOOZED, limit=10_000
    ):
        if _is_source_dismissal(ev.payload or {}):
            members.update(r.ref_id for r in ev.refs if r.ref_kind is EventRefKind.PARTICLE)
    return members


async def group_rulings(session: AsyncSession) -> GroupRulings:
    """The standing ``split`` / ``dismiss`` rulings on duplicate groups.

    The one reader of those rulings. Its callers are the grouping step, the
    live half, and ``merge``'s refusal, so the three cannot disagree about
    which pairs a ruling withholds.
    """
    out: list[tuple[frozenset[str], ...]] = []
    for ev in await list_events(
        session, event_type=OperatorEventType.CURATION_CARD_SNOOZED, limit=10_000
    ):
        payload = ev.payload or {}
        if not _is_group_ruling(payload):
            continue
        classes = payload.get("classes")
        if not isinstance(classes, list):
            continue
        parsed = tuple(
            frozenset(str(pid) for pid in c) for c in classes if isinstance(c, list) and c
        )
        if len(parsed) >= 2:
            out.append(parsed)
    return GroupRulings(tuple(out))

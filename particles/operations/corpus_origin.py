# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The one-time origin stamp and its repair (§5.3).

``particles db init`` gives every ``file://`` entry that records no host the id
of the host it runs on, once per store: the ``CORPUS_ORIGIN_STAMPED`` event it
writes is what stops a later run stamping again, so an entry that arrives
unstamped afterwards waits to be adopted by a deposit from its own host.
``particles corpus reorigin`` moves entries from one host id to another, for a
store stamped on the wrong machine. Neither ever runs from the engine's own
startup migration.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from particles.corpus.origin import (
    StampPlan,
    apply_origin_stamp,
    apply_reorigin,
    count_reorigin,
    plan_origin_stamp,
)
from particles.store.event_store import OperatorEventType, count_events_by_type, record_event


@dataclass(frozen=True)
class StampOutcome:
    """What the stamp did, or would do: the plan, and whether this store was stamped before."""

    plan: StampPlan
    already_stamped: bool
    written: int = 0


async def origin_stamp(session: AsyncSession, *, apply: bool) -> StampOutcome:
    """Plan, and with ``apply`` write, the one-time stamp. Does not commit.

    A store whose log already holds ``CORPUS_ORIGIN_STAMPED`` is not stamped
    again. The event is written even when nothing needed a stamp, so a store
    that started empty is never stamped later, when an entry that arrived
    without a host would be another machine's. A host with no usable id stamps
    nothing (``plan.host_id`` is None).
    """
    counts = await count_events_by_type(session)
    already = counts.get(OperatorEventType.CORPUS_ORIGIN_STAMPED.value, 0) > 0
    plan = await plan_origin_stamp(session)
    if already or not apply or plan.host_id is None:
        return StampOutcome(plan=plan, already_stamped=already)
    written = await apply_origin_stamp(session, plan)
    await record_event(
        session,
        actor="cli:db-init",
        event_type=OperatorEventType.CORPUS_ORIGIN_STAMPED,
        payload={
            "host_id": plan.host_id,
            "rule_files": plan.rule_files,
            "memory_files": plan.memory_files,
            "other": plan.other,
        },
    )
    return StampOutcome(plan=plan, already_stamped=False, written=written)


async def reorigin(
    session: AsyncSession,
    *,
    from_host: str,
    to_host: str,
    uri_prefix: str | None,
    apply: bool,
) -> int:
    """Count, and with ``apply`` move, ``from_host``'s ``file://`` entries to ``to_host``.

    Returns the number of entries. Records ``CORPUS_ORIGIN_REASSIGNED`` when it
    moves any. Does not commit.
    """
    n = await count_reorigin(session, from_host=from_host, uri_prefix=uri_prefix)
    if not apply or n == 0:
        return n
    moved = await apply_reorigin(
        session, from_host=from_host, to_host=to_host, uri_prefix=uri_prefix
    )
    await record_event(
        session,
        actor="cli:corpus-reorigin",
        event_type=OperatorEventType.CORPUS_ORIGIN_REASSIGNED,
        payload={"from": from_host, "to": to_host, "uri_prefix": uri_prefix, "entries": moved},
    )
    return moved

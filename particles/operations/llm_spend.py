# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Recorded LLM spend: the extract run record and the per-store sum.

``particles audit`` and ``particles memory consolidate`` already put their
measured usage on a ``CONSOLIDATION_RUN`` event. Extraction, the
bulk of a document store's bill, opened no usage scope at all.
:class:`MeteredExtractRun` closes that: each extract route opens one around its
work, reads the totals back for the stderr line, and appends one
``EXTRACT_RUN`` event when the run made a call. ``particles reindex`` and
``particles structure`` re-run the same pipeline outside the ``extract`` verb
and record through the same meter, each under its own ``route``.

:func:`store_llm_spend` sums both event types into the figure ``quality`` and
the digest show. It is derived at read time from append-only records and never
kept as a counter. It is list price over recorded token counts,
never a billing API, and it begins at the first run that recorded usage.
"""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from types import TracebackType
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import StoreLLMSpend
from particles.db import DEFAULT_STORE, session_scope
from particles.llm.usage import LLMUsage, UsageAccumulator, track_usage
from particles.store.event_store import (
    OperatorEvent,
    OperatorEventRow,
    OperatorEventType,
    record_event,
)

log = logging.getLogger(__name__)

#: Version of the ``EXTRACT_RUN`` payload; additive keys do not bump it.
EXTRACT_RUN_PAYLOAD_FORMAT = 1

#: Event types whose payload may carry a run's measured ``llm_usage``.
SPEND_EVENT_TYPES = (OperatorEventType.CONSOLIDATION_RUN, OperatorEventType.EXTRACT_RUN)


def build_extract_run_payload(
    *,
    store: str,
    route: str,
    started_at: datetime,
    completed_at: datetime,
    snapshots: int,
    llm_usage: LLMUsage,
) -> dict[str, Any]:
    """The versioned ``EXTRACT_RUN`` payload (``format: 1``).

    ``llm_usage`` has the ``CONSOLIDATION_RUN`` payload's shape, so one reader
    sums both. ``route`` names the entry point (``single``, ``all-pending``,
    ``http``, ``reindex``, ``structure``); ``snapshots`` counts the snapshots
    the run attempted (``structure`` annotates particles and counts none).
    """
    return {
        "format": EXTRACT_RUN_PAYLOAD_FORMAT,
        "store": store,
        "route": route,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "snapshots": snapshots,
        "llm_usage": llm_usage.model_dump(mode="json"),
    }


async def record_extract_run(
    session: AsyncSession,
    *,
    actor: str,
    store: str,
    route: str,
    started_at: datetime,
    completed_at: datetime,
    snapshots: int,
    llm_usage: LLMUsage,
) -> OperatorEvent | None:
    """Append one ``EXTRACT_RUN`` event; the caller commits.

    A run that made no LLM call (every chunk carried forward, every snapshot
    skipped) spent nothing and records nothing, so a hook that runs
    ``extract --all-pending`` on every turn does not fill the log.
    """
    if llm_usage.calls == 0:
        return None
    return await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.EXTRACT_RUN,
        payload=build_extract_run_payload(
            store=store,
            route=route,
            started_at=started_at,
            completed_at=completed_at,
            snapshots=snapshots,
            llm_usage=llm_usage,
        ),
    )


class MeteredExtractRun:
    """Async context manager metering one extract run and recording it.

    Opens a :func:`~particles.llm.usage.track_usage` scope for the block. On
    exit, normal or not, it sets :attr:`llm_usage` and appends the
    ``EXTRACT_RUN`` event in a short write session of its own: tokens a failed
    run was billed for are still spend. A failure to record is logged and never
    masks the run's own outcome.

    The caller counts :attr:`snapshots` as it goes. Do not open one inside
    another usage-recording run (a consolidation cycle, an audit): both would
    record the same calls.
    """

    def __init__(self, *, actor: str, route: str, store: str = DEFAULT_STORE) -> None:
        self.actor = actor
        self.route = route
        self.store = store
        #: Snapshots the run attempted; the caller increments it.
        self.snapshots = 0
        #: The run's measured usage, set on exit.
        self.llm_usage: LLMUsage | None = None
        #: The recorded event's id, or ``None`` when nothing was recorded.
        self.event_id: str | None = None
        self._started_at = datetime.now(UTC)
        self._scope: AbstractContextManager[UsageAccumulator] | None = None
        self._accumulator: UsageAccumulator | None = None

    async def __aenter__(self) -> MeteredExtractRun:
        self._started_at = datetime.now(UTC)
        self._scope = track_usage()
        self._accumulator = self._scope.__enter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        assert self._scope is not None and self._accumulator is not None
        self._scope.__exit__(None, None, None)
        self.llm_usage = self._accumulator.snapshot()
        try:
            async with session_scope(self.store, write=True) as session:
                event = await record_extract_run(
                    session,
                    actor=self.actor,
                    store=self.store,
                    route=self.route,
                    started_at=self._started_at,
                    completed_at=datetime.now(UTC),
                    snapshots=self.snapshots,
                    llm_usage=self.llm_usage,
                )
                await session.commit()
            self.event_id = event.event_id if event is not None else None
        except Exception:
            log.warning("could not record the extract run's LLM usage", exc_info=True)


def reindexed_snapshots(summary: dict[str, object]) -> int:
    """Snapshots a reindex run attempted: its ``succeeded`` plus its ``failed``."""
    total = 0
    for key in ("succeeded", "failed"):
        value = summary.get(key)
        if isinstance(value, int):
            total += value
    return total


def _recorded_usage(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """A run payload's ``llm_usage`` when it recorded at least one response."""
    usage = (payload or {}).get("llm_usage")
    if not isinstance(usage, dict):
        return None
    rows = usage.get("rows") or []
    if not any(isinstance(row, dict) and row.get("calls") for row in rows):
        return None
    return usage


async def store_llm_spend(session: AsyncSession) -> StoreLLMSpend | None:
    """The store's cumulative recorded LLM spend, or ``None`` before any run.

    Sums every ``CONSOLIDATION_RUN`` and ``EXTRACT_RUN`` event whose payload
    recorded at least one call. A run whose usage named an unpriced model is
    counted as a run and disclosed, never folded in at zero.
    """
    result = await session.scalars(
        select(OperatorEventRow).where(
            OperatorEventRow.event_type.in_([t.value for t in SPEND_EVENT_TYPES])
        )
    )
    total = 0.0
    runs = 0
    unpriced = 0
    since: datetime | None = None
    for row in result:
        usage = _recorded_usage(row.payload)
        if usage is None:
            continue
        runs += 1
        cost = usage.get("cost_usd")
        if isinstance(cost, int | float):
            total += float(cost)
        else:
            unpriced += 1
        if since is None or row.occurred_at < since:
            since = row.occurred_at
    if since is None:
        return None
    return StoreLLMSpend(cost_usd=total, runs=runs, since=since, unpriced_runs=unpriced)

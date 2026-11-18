# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Gauges a finished consolidation run reports to the metrics pipeline.

The run record (the ``CONSOLIDATION_RUN`` event) holds every figure, but an
operator watching a backlog drain over weeks reads a dashboard, not the event
log. At the end of each run the shell sets one gauge per figure here; the
OpenTelemetry SDK exports each gauge's last value until the next run sets it
again, so a panel shows the latest run and its history.

:func:`run_readings` and :func:`run_totals` are the pure half: the report and
the stored run records in, the readings out. :func:`record_run_metrics` gathers
the records and sets the gauges through the OTel API, which is a no-op until
``setup_observability`` installs a provider. Recording never fails the run: the
run record is already written, and a metrics fault is logged.

The ``*_to_date`` gauges are running totals over every stored run record, so a
panel can read "spent since a date" as the rise of a value that survives an
engine restart (a counter would restart from zero under a new instance label).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, get_args

from opentelemetry import metrics
from sqlalchemy.ext.asyncio import AsyncSession

from particles.operations.extract_budget import StopReason
from particles.store.event_store import OperatorEventType, list_events_in_range

if TYPE_CHECKING:
    from particles.operations.consolidation import ConsolidationReport

log = logging.getLogger(__name__)

#: Each gauge's name, unit and description. A braced unit is an annotation the
#: Prometheus translation drops, so the exported name is the gauge name with
#: dots as underscores (``particles_consolidation_backlog``); ``s`` adds the
#: ``_seconds`` suffix. The 0/1 stop gauge has no unit, since ``1`` on a gauge
#: would add ``_ratio``.
GAUGES: dict[str, tuple[str, str]] = {
    "particles.consolidation.completed_at": ("s", "Unix time the run finished"),
    "particles.consolidation.backlog": (
        "{snapshot}",
        "Pending snapshots left after the run's extraction pass",
    ),
    "particles.consolidation.extracted": ("{snapshot}", "Snapshots the run extracted"),
    "particles.consolidation.run_cost": ("{USD}", "List-price LLM cost of the whole run"),
    "particles.consolidation.extract_spent": (
        "{USD}",
        "List-price spend of the extraction pass under its budget",
    ),
    "particles.consolidation.extract_budget": ("{USD}", "The extraction pass's nightly budget"),
    "particles.consolidation.extract_rounds": ("{round}", "Extraction rounds the run ran"),
    "particles.consolidation.run_cost_to_date": (
        "{USD}",
        "List-price LLM cost summed over every stored run record, audits included",
    ),
    "particles.consolidation.extract_spent_to_date": (
        "{USD}",
        "Extraction-pass spend summed over every stored run record, audits included",
    ),
    "particles.consolidation.extracted_to_date": (
        "{snapshot}",
        "Snapshots extracted, summed over every stored run record, audits included",
    ),
    "particles.consolidation.extract_stop": (
        "",
        "1 for the reason the extraction pass stopped, 0 for every other reason",
    ),
}


@dataclass(frozen=True)
class Reading:
    """One gauge value with its attributes."""

    name: str
    value: float
    attributes: dict[str, str]


class _Gauge(Protocol):
    def set(self, amount: float, attributes: dict[str, str] | None = None) -> None: ...


def run_readings(report: ConsolidationReport, store: str) -> list[Reading]:
    """The gauge values one finished run reports.

    The extraction figures are reported only when the extraction pass ran,
    so a night that skipped it (extraction off, a degraded provider) leaves
    the previous night's backlog on the graph instead of a false zero. The
    backlog alone is also reported when the pass listed snapshots it could not
    read (the egress policy refused the endpoint). The budget figures are
    reported only when a budget is set.
    """
    base = {"store": store}
    out: list[Reading] = []
    if report.completed_at is not None:
        out.append(
            Reading("particles.consolidation.completed_at", report.completed_at.timestamp(), base)
        )
    usage = report.llm_usage
    if usage is not None and usage.cost_usd is not None:
        out.append(Reading("particles.consolidation.run_cost", usage.cost_usd, base))
    extract_ran = any(p.name == "extract" and p.status == "ran" for p in report.passes)
    # A night the egress policy refused still listed its backlog: report it.
    if extract_ran or report.pending_total > 0:
        out.append(
            Reading("particles.consolidation.backlog", float(report.pending_remaining), base)
        )
    if not extract_ran:
        return out
    out.append(Reading("particles.consolidation.extracted", float(report.pending_extracted), base))
    summary = report.extract_budget
    if summary is None:
        return out
    out.append(Reading("particles.consolidation.extract_spent", summary.spent_usd, base))
    out.append(Reading("particles.consolidation.extract_budget", summary.budget_usd, base))
    out.append(Reading("particles.consolidation.extract_rounds", float(len(summary.rounds)), base))
    # Every reason, so the one that stopped this run is the only 1: a gauge
    # keeps each attribute set's last value, and a reason set to 1 on an
    # earlier night would otherwise still read 1.
    for reason in get_args(StopReason):
        out.append(
            Reading(
                "particles.consolidation.extract_stop",
                1.0 if reason == summary.stopped_by else 0.0,
                {**base, "reason": reason},
            )
        )
    return out


def _number(block: object, key: str) -> float:
    if isinstance(block, Mapping):
        value = block.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return 0.0


def run_totals(payloads: Iterable[Mapping[str, Any]], store: str) -> list[Reading]:
    """Running totals over stored ``CONSOLIDATION_RUN`` payloads.

    A missing block or an unpriced (null) figure adds nothing: the total is a
    lower bound when a run was unpriced, as the run record itself says.
    """
    cost = spent = extracted = 0.0
    for payload in payloads:
        cost += _number(payload.get("llm_usage"), "cost_usd")
        spent += _number(payload.get("extract_budget"), "spent_usd")
        extracted += _number(payload.get("census"), "pending_extracted")
    base = {"store": store}
    return [
        Reading("particles.consolidation.run_cost_to_date", cost, base),
        Reading("particles.consolidation.extract_spent_to_date", spent, base),
        Reading("particles.consolidation.extracted_to_date", extracted, base),
    ]


_gauges: dict[str, _Gauge] = {}


def _default_gauges() -> dict[str, _Gauge]:
    if not _gauges:
        meter = metrics.get_meter("particles.consolidation")
        for name, (unit, description) in GAUGES.items():
            _gauges[name] = meter.create_gauge(name, unit=unit, description=description)
    return _gauges


async def record_run_metrics(
    session: AsyncSession,
    report: ConsolidationReport,
    store: str,
    *,
    gauges: dict[str, _Gauge] | None = None,
) -> None:
    """Set the run's gauges; log and carry on if anything goes wrong."""
    try:
        events = await list_events_in_range(session, event_type=OperatorEventType.CONSOLIDATION_RUN)
        readings = run_readings(report, store) + run_totals(
            (e.payload for e in events if e.payload is not None), store
        )
        sink = gauges if gauges is not None else _default_gauges()
        for reading in readings:
            sink[reading.name].set(reading.value, reading.attributes)
    except Exception:  # noqa: BLE001 — telemetry never fails the run
        log.warning("consolidation metrics were not recorded", exc_info=True)

# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""A priced, ordered and budgeted ``extract --all-pending``.

A first import of a personal vault queues thousands of notes, and reading all
of them on the default model costs a hundred dollars or more. The verb therefore
prices what it is about to read before the first call, asks above a threshold,
can stop at a dollar budget between snapshots, and reads in an order that makes
a small budget worth spending: pending edits of sources the store has already
read first, then never-read sources in deposit order (which ``import vault``
makes newest-edited first).

The shape is gather / decide / apply: :func:`plan_run` gathers the
read plan and its prices, the pure :func:`order_pending`,
:func:`decide_confirmation` and :meth:`BudgetGate.allows` decide, and the CLI
applies. :func:`extraction_route` is the per-run model choice.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import ProviderSelection, get_config
from particles.core.spend_budget import decide_spend
from particles.llm.usage import LLMUsage
from particles.operations.audit import ExtractionCall
from particles.operations.extraction_reads import plan_reads
from particles.operations.spend_estimate import (
    ExtractionRange,
    extraction_range,
    planned_extraction_estimate,
)


@dataclass(frozen=True)
class PendingSnapshot:
    """One pending snapshot as the ordering sees it."""

    entry_id: str
    snapshot_id: str
    #: An ``ABSENT`` record: processed with no model call.
    absent: bool = False
    #: The entry already holds an extracted generation, so the store currently
    #: states what this snapshot's source may no longer say.
    read_before: bool = False


def order_pending(items: Sequence[PendingSnapshot]) -> list[PendingSnapshot]:
    """``items`` in reading order; ``items`` arrive in deposit order.

    Pending edits of already-read entries first, then never-read entries, each
    class in deposit order, then the absences. The class is a
    property of the entry, so one entry's snapshots stay in their ascending
    order and no generation is read out of turn.
    """
    present = [item for item in items if not item.absent]
    return (
        [item for item in present if item.read_before]
        + [item for item in present if not item.read_before]
        + [item for item in items if item.absent]
    )


@dataclass
class RunPlan:
    """What the pending snapshots will send, priced."""

    snapshots: list[PendingSnapshot] = field(default_factory=list)
    #: Planned calls per snapshot id; absences plan none.
    calls: dict[str, tuple[ExtractionCall, ...]] = field(default_factory=dict)
    #: Each snapshot's high-end estimate in US$, ``None`` when unpriced.
    high_usd: dict[str, float | None] = field(default_factory=dict)
    #: The whole run on the configured extraction model.
    total: ExtractionRange | None = None
    #: The same run on ``extraction.alternative_model``, when it differs.
    alternative: ExtractionRange | None = None
    alternative_route: str | None = None

    def remaining(self, done: set[str]) -> tuple[int, int, ExtractionRange]:
        """Snapshots not in ``done`` that still owe a read: count, characters, price."""
        left = [s for s in self.snapshots if not s.absent and s.snapshot_id not in done]
        calls = [call for s in left for call in self.calls.get(s.snapshot_id, ())]
        chars = sum(call.chars for call in calls)
        entries = sum(1 for s in left if self.calls.get(s.snapshot_id))
        return len(left), chars, extraction_range(calls, entries=entries)


async def plan_run(session: AsyncSession, snapshots: Sequence[PendingSnapshot]) -> RunPlan:
    """Plan and price every non-absent snapshot as the extractor will read it. Reads only."""
    plan = RunPlan(snapshots=list(snapshots))
    reads = await plan_reads(
        session, [(s.entry_id, s.snapshot_id) for s in snapshots if not s.absent]
    )
    for read in reads:
        plan.calls[read.snapshot_id] = read.calls
        plan.high_usd[read.snapshot_id] = planned_extraction_estimate(
            read.calls, entries=1 if read.calls else 0, batched=False
        ).usd
    every = [call for read in reads for call in read.calls]
    entries = sum(1 for read in reads if read.calls)
    plan.total = extraction_range(every, entries=entries)
    alternative = get_config().extraction.alternative_model
    if alternative:
        selection = parse_model_route(alternative)
        if selection != get_config().llm.for_purpose("extraction"):
            with extraction_route(selection):
                plan.alternative = extraction_range(every, entries=entries)
            plan.alternative_route = alternative
    return plan


Confirmation = Literal["run", "ask", "refuse"]


def decide_confirmation(
    *,
    calls: int,
    threshold: int,
    yes: bool,
    budget_usd: float | None,
    interactive: bool,
) -> Confirmation:
    """Whether a run may start, must ask, or must stop without a person.

    At or under the threshold it runs. Above it, ``--yes`` or a budget (consent
    up to its amount) runs it; otherwise a terminal is asked and a run without
    one is refused, since its caller is most often an agent spending for a user.
    """
    if calls <= threshold or yes or budget_usd is not None:
        return "run"
    return "ask" if interactive else "refuse"


@dataclass
class BudgetGate:
    """The run's dollar budget, checked before each snapshot."""

    budget_usd: float | None
    #: Total spend over extraction spend so far; never below 1, so scaling only
    #: makes the check stricter. Subject resolution and the §6.6 probes a
    #: snapshot triggers are what it adds.
    ratio: float = 1.0

    def allows(self, *, spent_usd: float, estimate_usd: float | None) -> bool:
        """Whether a snapshot estimated at ``estimate_usd`` (high end) may start."""
        scaled = None if estimate_usd is None else estimate_usd * self.ratio
        verdict = decide_spend(spent_usd=spent_usd, budget_usd=self.budget_usd, estimate_usd=scaled)
        return verdict == "run"

    def observe(self, usage: LLMUsage) -> None:
        """Update the ratio from the run's measured usage so far."""
        extraction = sum(
            row.cost_usd
            for row in usage.rows
            if row.purpose == "extraction" and row.cost_usd is not None and not row.replayed
        )
        if extraction > 0:
            self.ratio = max(1.0, usage.priced_cost_usd / extraction)


def parse_model_route(text: str) -> ProviderSelection:
    """``provider:model`` as a selection; the provider must be one config can route.

    Raises:
        ValueError: when the text is not ``provider:model`` or names a
            provider that is neither ``anthropic`` nor an ``llm.providers`` entry.
    """
    provider, sep, model = text.partition(":")
    if not sep or not provider or not model:
        raise ValueError(f"expected provider:model, got {text!r}")
    if provider != "anthropic" and provider not in get_config().llm.providers:
        raise ValueError(
            f"unknown provider {provider!r}: use anthropic or a name under llm.providers"
        )
    return ProviderSelection(provider=provider, model=model)


@contextlib.contextmanager
def extraction_route(selection: ProviderSelection | None) -> Iterator[None]:
    """Route extraction and subject resolution to ``selection`` for the block.

    Replaces the two selections on the loaded config, so every reader that
    asks ``llm.for_purpose`` (the provider, the estimate, the calibration
    lookup, the ``extraction_provider_model`` a particle records) sees the
    model actually used, and restores them on exit. ``None`` changes nothing.
    """
    if selection is None:
        yield
        return
    llm = get_config().llm
    saved = (llm.extraction, llm.subject_resolution)
    llm.extraction = selection
    llm.subject_resolution = selection
    try:
        yield
    finally:
        llm.extraction, llm.subject_resolution = saved

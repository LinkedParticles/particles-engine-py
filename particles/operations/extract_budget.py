# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Pass 1's nightly extraction budget: what it is, what it priced, what it did.

The consolidation cycle's pass 1 reads PENDING snapshots in rounds under
``consolidation.extract_budget_usd``. The admission is the pure
:func:`particles.core.spend_budget.admit_round`; this module holds the shell's
gather half that is not the loop itself: the limit a night applies, each
candidate's class and two prices, and the run-record block the pass fills
(:class:`ExtractBudgetSummary`). The loop lives in
:mod:`particles.operations.consolidation` beside the pooled pass it drives.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from particles.config import get_config
from particles.core.spend_budget import RoundCandidate
from particles.operations.extraction_reads import SnapshotRead, plan_reads
from particles.operations.spend_estimate import snapshot_price

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from particles.corpus.store import PendingSnapshot

#: The class every priority-tagged entry shares.
PRIORITY_CLASS = "priority"

StopReason = Literal[
    "budget", "ceiling", "rounds", "batch_wait", "queue", "unpriced", "egress", "failure"
]


class ExtractRound(BaseModel):
    """One round as the run record keeps it."""

    klass: str
    bounded: bool
    admitted: int
    charged_usd: float
    estimated_usd: float
    worst_case_usd: float
    spent_usd: float
    wait_seconds: float


class ExtractBudgetSummary(BaseModel):
    """The ``extract_budget`` block of a ``CONSOLIDATION_RUN``."""

    #: ``consolidation.extract_budget_usd`` at run time.
    budget_usd: float
    #: The limit applied: the smaller of the extraction budget and the run budget.
    limit_usd: float
    spent_usd: float = 0.0
    #: Unscaled high-end estimates of every snapshot admitted.
    estimated_usd: float = 0.0
    #: Each class's factor at the end of the pass, with its priced-snapshot count.
    factor: dict[str, float] = Field(default_factory=dict)
    factor_snapshots: dict[str, int] = Field(default_factory=dict)
    rounds: list[ExtractRound] = Field(default_factory=list)
    admitted: int = 0
    passed_over: int = 0
    too_large: list[str] = Field(default_factory=list)
    stopped_by: StopReason = "queue"
    #: The worst case of the round that ended the pass.
    worst_case_usd: float = 0.0
    #: Snapshots the batch-chunk backstop declined during the pass.
    skipped_requests: int = 0

    @property
    def overshoot_usd(self) -> float:
        """Spend beyond the limit, or 0."""
        return max(0.0, self.spent_usd - self.limit_usd)

    @property
    def rounds_bounded(self) -> int:
        """Rounds sized by the worst case."""
        return sum(1 for r in self.rounds if r.bounded)

    def payload(self) -> dict[str, object]:
        """The run-record block, with the derived figures."""
        return {
            **self.model_dump(mode="json"),
            "rounds_run": len(self.rounds),
            "rounds_bounded": self.rounds_bounded,
            "overshoot_usd": round(self.overshoot_usd, 6),
        }


def pass_one_limit(run_budget_left: float | None) -> float | None:
    """Tonight's pass-1 limit, or ``None`` when no extraction budget is set.

    The smaller of ``consolidation.extract_budget_usd`` and what the run budget
    (``consolidation.budget_usd``) has left.
    """
    budget = get_config().consolidation.extract_budget_usd
    if budget is None:
        return None
    if run_budget_left is None:
        return budget
    return max(0.0, min(budget, run_budget_left))


def extraction_route_priced() -> bool:
    """Whether the extraction model has a list price."""
    llm = get_config().llm
    return llm.price_for(llm.for_purpose("extraction")) is not None


def class_of(item: PendingSnapshot) -> str:
    """The class a factor is kept for: the priority class, else the source type."""
    priority = set(get_config().consolidation.priority_tags)
    if priority and item.tags & priority:
        return PRIORITY_CLASS
    return item.source_type or "unknown"


async def price_candidates(
    session: AsyncSession,
    heads: Sequence[PendingSnapshot],
    cache: dict[str, RoundCandidate],
    *,
    batched: bool,
) -> list[RoundCandidate]:
    """Each head's class and two prices, planned as the extractor will read it.

    ``cache`` keeps a head's prices between rounds: a head that was not read
    keeps its entry's state, so its plan still holds. ``batched`` prices at the
    batch discount, since a budgeted pooled round submits its snapshots as one set.
    """
    missing = [h for h in heads if h.snapshot_id not in cache]
    if missing:
        reads: list[SnapshotRead] = await plan_reads(
            session, [(h.entry_id, h.snapshot_id) for h in missing]
        )
        for head, read in zip(missing, reads, strict=True):
            price = snapshot_price(read.calls, batched=batched)
            cache[head.snapshot_id] = RoundCandidate(
                snapshot_id=head.snapshot_id,
                klass=class_of(head),
                estimate_usd=price.estimate_usd,
                worst_case_usd=price.worst_case_usd,
            )
    return [cache[h.snapshot_id] for h in heads]


def render_budget_clause(summary: ExtractBudgetSummary) -> str:
    """The report line's budget clause."""
    factors = ", ".join(f"{k} ×{v:.2f}" for k, v in sorted(summary.factor.items()))
    parts = [
        f"{len(summary.rounds)} round(s)",
        f"spent US${summary.spent_usd:.2f} of US${summary.limit_usd:.2f}",
    ]
    if factors:
        parts.append(f"estimate {factors}")
    if summary.overshoot_usd > 0:
        parts.append(f"over by US${summary.overshoot_usd:.2f}")
    if summary.too_large:
        parts.append(
            f"{len(summary.too_large)} too large for the nightly extraction budget "
            "(read with particles extract --snapshot-id)"
        )
    parts.append(f"stopped: {summary.stopped_by}")
    return "extraction budget: " + ", ".join(parts)


def extract_budget_warnings() -> list[str]:
    """What ``config validate`` and ``hook doctor`` say about the budget's settings.

    Empty when no extraction budget is set or the settings are coherent.
    """
    cfg = get_config().consolidation
    budget = cfg.extract_budget_usd
    if budget is None:
        return []
    warnings: list[str] = []
    if not extraction_route_priced():
        warnings.append(
            "consolidation.extract_budget_usd is set but the extraction model has no "
            "llm.price_per_mtok entry, so no dollar can be read and pass 1 runs unbudgeted."
        )
    if cfg.budget_usd is not None and budget >= cfg.budget_usd:
        warnings.append(
            f"consolidation.extract_budget_usd (US${budget:.2f}) is not below "
            f"consolidation.budget_usd (US${cfg.budget_usd:.2f}), so pass 1 can leave the "
            "later passes nothing."
        )
    default_ceiling = type(cfg).model_fields["max_pending_entries"].default
    if cfg.max_pending_entries <= default_ceiling:
        warnings.append(
            f"consolidation.max_pending_entries is {cfg.max_pending_entries}, the per-run "
            "ceiling under an extraction budget; at that count it, not the budget, stops "
            "most nights. Raise it with the budget (for example 200)."
        )
    return warnings

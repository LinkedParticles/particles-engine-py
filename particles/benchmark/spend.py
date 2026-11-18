# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""What a judged benchmark run was projected to cost, beside what it cost.

The leakage and relevance-floor verbs each project a run before any call (an
answer role on ``query_response``, a judge role on ``benchmark``) and open a
:func:`~particles.llm.usage.track_usage` scope around the paid stage. This
module puts the two side by side, role by role, so every run calibrates the
next: a judge whose measured output is ten times its projection shows it on the
first run, not on the bill. The measured figures are what the provider
reported; the dollars are list price, never a billing API's.

Pure: no call, no store.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from particles.llm.usage import LLMUsage, UsageRow, format_tokens, format_usd, render_usage_line

#: The two roles a judged run pays for, and the purpose each is routed through.
ROLE_PURPOSES: tuple[tuple[str, str], ...] = (("answer", "query_response"), ("judge", "benchmark"))


class JudgedEstimate(Protocol):
    """The fields both harnesses' projections carry."""

    answer_calls: int
    answer_input_tokens: int
    answer_output_tokens: int
    judge_calls: int
    judge_input_tokens: int
    judge_output_tokens: int
    cost_usd: float | None


class RoleSpend(BaseModel):
    """One role's projected and measured calls and tokens."""

    role: str
    purpose: str
    estimated_calls: int = 0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    measured_calls: int = 0
    #: Every input token billed: uncached, cache write and cache read.
    measured_input_tokens: int = 0
    #: Output as billed, thinking included.
    measured_output_tokens: int = 0


class RunSpend(BaseModel):
    """The projection and the measurement of one run, in the report of record."""

    estimated_cost_usd: float | None = None
    #: List price of what the providers reported; ``None`` when a model is unpriced.
    measured_cost_usd: float | None = None
    roles: list[RoleSpend] = Field(default_factory=list)
    #: Questions restored from a checkpoint: projected, but no call was made.
    restored_questions: int = 0
    usage: LLMUsage = Field(default_factory=LLMUsage)

    @property
    def ratio(self) -> float | None:
        """Measured over estimated dollars, when both exist and the estimate is positive."""
        if self.estimated_cost_usd is None or self.measured_cost_usd is None:
            return None
        if self.estimated_cost_usd <= 0:
            return None
        return self.measured_cost_usd / self.estimated_cost_usd


def _billed(rows: list[UsageRow], purpose: str) -> list[UsageRow]:
    return [row for row in rows if row.purpose == purpose and not row.replayed]


def measure_spend(
    estimate: JudgedEstimate, usage: LLMUsage, *, restored_questions: int = 0
) -> RunSpend:
    """Pair a run's projection with its measured usage."""
    roles: list[RoleSpend] = []
    for role, purpose in ROLE_PURPOSES:
        rows = _billed(usage.rows, purpose)
        roles.append(
            RoleSpend(
                role=role,
                purpose=purpose,
                estimated_calls=getattr(estimate, f"{role}_calls"),
                estimated_input_tokens=getattr(estimate, f"{role}_input_tokens"),
                estimated_output_tokens=getattr(estimate, f"{role}_output_tokens"),
                measured_calls=sum(r.calls for r in rows),
                measured_input_tokens=sum(
                    r.input_tokens + r.cache_creation_tokens + r.cache_read_tokens for r in rows
                ),
                measured_output_tokens=sum(r.output_tokens for r in rows),
            )
        )
    return RunSpend(
        estimated_cost_usd=estimate.cost_usd,
        measured_cost_usd=usage.cost_usd,
        roles=roles,
        restored_questions=restored_questions,
        usage=usage,
    )


def render_spend(spend: RunSpend) -> str:
    """The end-of-run block: estimate against measurement, then the usage line."""
    estimated = (
        f"~US{format_usd(spend.estimated_cost_usd)}"
        if spend.estimated_cost_usd is not None
        else "without a dollar figure (no price configured)"
    )
    if spend.measured_cost_usd is None:
        measured = "without a dollar total (a model is unpriced)"
    else:
        measured = f"US{format_usd(spend.measured_cost_usd)} at list price"
    head = f"Spend: estimated {estimated}, measured {measured}"
    ratio = spend.ratio
    head += f" ({ratio:.2f}× the estimate)." if ratio is not None else "."
    lines = [head]
    for role in spend.roles:
        lines.append(
            f"  {role.role:<6}  estimated {role.estimated_calls:>5} calls, "
            f"~{format_tokens(role.estimated_input_tokens)} in / "
            f"~{format_tokens(role.estimated_output_tokens)} out;  measured "
            f"{role.measured_calls:>5} calls, {format_tokens(role.measured_input_tokens)} in / "
            f"{format_tokens(role.measured_output_tokens)} out"
        )
    if spend.restored_questions:
        lines.append(
            f"  · {spend.restored_questions} question(s) restored from the checkpoint made "
            "no call; the estimate counted them."
        )
    lines.append(render_usage_line(spend.usage))
    return "\n".join(lines)

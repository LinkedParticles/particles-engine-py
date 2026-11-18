# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The per-run spend stop of the two judged benchmark verbs.

Every live run of ``benchmark leakage`` and ``benchmark relevance-floor
--judge`` carries the stops its roles need: ``--max-usd`` when a role is priced
in ``llm.price_per_mtok``, and ``--max-calls`` (or ``--no-ceiling``) when a role
is not. :func:`stop_refusal` is that gate; a run without its stops refuses
before any call.

During the run :class:`SpendCeiling` reads the run's
:func:`~particles.llm.usage.track_usage` accumulator before each call starts
(answer, judge, or a retry) and raises :class:`CeilingReached` once a stop is
reached. Calls already in flight finish. A question with a call left unstarted
is excluded as :data:`STOP_EXCLUSION` and never checkpointed, so a re-run with
a higher ceiling answers it afresh; :class:`StoppedAtCeiling` is what a stopped
report records about the stop.

The stop never reaches the system under test: it changes no call's request
and is invisible to the query op, the composer and the judge.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from particles.config import ProviderSelection, get_config
from particles.llm import get_provider
from particles.llm.usage import LLMUsage, UsageAccumulator, format_usd

#: The exclusion cause of a question the stop left with a call unstarted.
STOP_EXCLUSION = "budget_ceiling"

#: ``excluded_detail`` of a question that never made a call.
UNSTARTED_DETAIL = "not started: the spend ceiling was reached first"
#: ``excluded_detail`` of a question stopped part way; its finished calls are discarded.
INTERRUPTED_DETAIL = (
    "interrupted at the spend ceiling; its finished calls were discarded and a re-run "
    "answers it afresh"
)


class CeilingReached(Exception):  # noqa: N818 — a stop, not an error
    """A stop was reached before a call started.

    ``sentences`` is set by a runner that knows how many sentences the
    interrupted question's answer had, for the stopped report.
    """

    def __init__(self, sentences: int | None = None) -> None:
        super().__init__("the run's spend ceiling was reached")
        self.sentences = sentences


# ---------------------------------------------------------------------------
# The gate: which stops a run's roles need
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Role:
    """One paid role of a judged run, as configured now."""

    name: str
    purpose: str
    provider_model: str
    #: ``(input, output)`` US$ per million tokens, or ``None`` when unpriced.
    price: tuple[float, float] | None


def configured_roles(roles: Iterable[tuple[str, str]]) -> list[Role]:
    """``(role, purpose)`` pairs resolved to the provider each purpose calls now.

    Priced as the usage accumulator prices a call: by the provider's
    ``provider:model`` key in ``llm.price_per_mtok``, so the gate and the stop
    read the same price.
    """
    llm = get_config().llm
    out: list[Role] = []
    for name, purpose in roles:
        provider_model = get_provider(purpose).provider_model  # type: ignore[arg-type]
        provider, _, model = provider_model.partition(":")
        price = llm.price_for(ProviderSelection(provider=provider, model=model))
        out.append(
            Role(
                name=name,
                purpose=purpose,
                provider_model=provider_model,
                price=None if price is None else (price.input, price.output),
            )
        )
    return out


def stop_refusal(
    roles: Sequence[Role],
    *,
    max_usd: float | None,
    max_calls: int | None,
    no_ceiling: bool,
) -> str | None:
    """Why a live run cannot start with these stops, or ``None`` when it can.

    A priced role needs ``--max-usd``. An unpriced role needs ``--max-calls``,
    or ``--no-ceiling``, which waives only that count and never touches a
    priced role. There is no default for either.
    """
    if max_calls is not None and no_ceiling:
        return "--max-calls and --no-ceiling contradict each other; pass one."
    priced = [r for r in roles if r.price is not None]
    unpriced = [r for r in roles if r.price is None]
    problems: list[str] = []
    if priced and max_usd is None:
        names = ", ".join(f"{r.name} ({r.provider_model})" for r in priced)
        problems.append(
            f"a priced role ({names}) needs --max-usd, the most this run may spend at "
            "list price. There is no default: name the figure."
        )
    if unpriced and max_calls is None and not no_ceiling:
        names = ", ".join(f"{r.name} ({r.provider_model})" for r in unpriced)
        problems.append(
            f"an unpriced role ({names}, no llm.price_per_mtok entry) needs --max-calls, "
            "the most billed calls it may make, or --no-ceiling for a model that costs "
            "nothing to call."
        )
    return " ".join(problems) if problems else None


# ---------------------------------------------------------------------------
# The confirmation's terms: the stop, the overshoot bound, the worst case
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InFlight:
    """Calls that may be in flight at the stop, for one role."""

    role: Role
    calls: int
    #: Projected input tokens a call.
    input_tokens: int
    #: The output a call can bill at most: its ``max_tokens``, thinking included.
    max_output_tokens: int


def in_flight_usd(calls: Sequence[InFlight]) -> tuple[float, float]:
    """``(output, input)`` US$ the priced calls in flight can bill past the stop.

    Output is hard (each call at its ``max_tokens``); input is projected.
    """
    output = inputs = 0.0
    for c in calls:
        if c.role.price is None:
            continue
        output += c.calls * c.max_output_tokens * c.role.price[1] / 1e6
        inputs += c.calls * c.input_tokens * c.role.price[0] / 1e6
    return output, inputs


def render_stop_terms(
    roles: Sequence[Role],
    *,
    max_usd: float | None,
    max_calls: int | None,
    in_flight: Sequence[InFlight],
    worst_case_usd: float | None,
) -> str:
    """The lines the CLI prints with the estimate before a live run starts."""
    lines: list[str] = []
    if max_usd is not None:
        output, inputs = in_flight_usd(in_flight)
        lines.append(
            f"  ceiling US{format_usd(max_usd)} at list price (not your account balance), "
            f"plus an estimated US{format_usd(output + inputs)} in flight at the stop "
            f"(output hard, US{format_usd(output)}; input projected, US{format_usd(inputs)})"
        )
    unpriced = [r for r in roles if r.price is None]
    if unpriced:
        flying = sum(c.calls for c in in_flight if c.role.price is None)
        names = ", ".join(r.name for r in unpriced)
        if max_calls is not None:
            lines.append(
                f"  ceiling {max_calls} billed call(s) of the unpriced role(s) ({names}), "
                f"plus up to {flying} in flight at the stop"
            )
        else:
            lines.append(f"  no call ceiling on the unpriced role(s) ({names}): --no-ceiling")
    for r in roles:
        if r.price is not None:
            lines.append(
                f"  · {r.name} priced at US${r.price[0]:g} in / US${r.price[1]:g} out per "
                f"million tokens ({r.provider_model}, llm.price_per_mtok); the invoice "
                "can differ"
            )
    if worst_case_usd is not None:
        lines.append(
            f"  · output worst case, every projected call at its max_tokens: "
            f"~US{format_usd(worst_case_usd)}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The live read and the stop
# ---------------------------------------------------------------------------


def _reading(usage: LLMUsage) -> tuple[float, int]:
    """``(priced US$, unpriced billed calls)``; replayed rows count toward neither."""
    calls = sum(row.calls for row in usage.rows if row.cost_usd is None and not row.replayed)
    return usage.priced_cost_usd, calls


class StoppedAtCeiling(BaseModel):
    """What a run stopped at its ceiling records; ``None`` on a complete report."""

    max_usd: float | None = None
    max_calls: int | None = None
    #: List price of the priced roles' spend when the stop was reached.
    spend_at_stop_usd: float = 0.0
    #: Billed calls of the unpriced roles when the stop was reached.
    calls_at_stop: int = 0
    #: Spent after the stop by the calls in flight then, priced and unpriced.
    overshoot_usd: float = 0.0
    overshoot_calls: int = 0
    #: Questions that made no call.
    unstarted: int = 0
    #: Questions stopped part way, whose finished calls were discarded.
    interrupted: int = 0
    discarded_usd: float = 0.0
    discarded_calls: int = 0
    #: Questions with a complete row, and the unstarted ones, by source. The
    #: unstarted are the tail of the held-out order, so they can differ.
    finished_by_source: dict[str, int] = Field(default_factory=dict)
    unstarted_by_source: dict[str, int] = Field(default_factory=dict)
    #: Sentences per answer: the finished answers' beside the interrupted
    #: ones', which lean long (they were stopped with sentences left to judge).
    finished_sentences: list[int] = Field(default_factory=list)
    interrupted_sentences: list[int] = Field(default_factory=list)


def _mean(values: Sequence[int]) -> str:
    return f"{sum(values) / len(values):.1f}" if values else "n/a"


def stopped_note(stop: StoppedAtCeiling, *, finished: int) -> str:
    """The report's first quality note on a stopped run."""
    limit = (
        f"US{format_usd(stop.max_usd)} at list price"
        if stop.max_usd is not None
        else f"{stop.max_calls} billed calls of the unpriced roles"
    )
    return (
        f"INCOMPLETE: the run stopped at its spend ceiling ({limit}). {finished} "
        f"question(s) finished, {stop.interrupted} were interrupted and discarded, and "
        f"{stop.unstarted} never started. The rates describe the finished questions only, "
        "and they lean: the questions still running at the stop are the slow ones with "
        f"the most sentences (finished answers {_mean(stop.finished_sentences)} sentences "
        f"on average, interrupted {_mean(stop.interrupted_sentences)}), and the unstarted "
        "ones are the tail of the held-out order. Do not quote this run: re-run it with a "
        "higher ceiling, which keeps every finished row and answers the rest afresh."
    )


def render_stopped(stop: StoppedAtCeiling) -> list[str]:
    """The table's block for a stopped run."""
    limit = (
        f"US{format_usd(stop.max_usd)}" if stop.max_usd is not None else f"{stop.max_calls} calls"
    )
    by_source = "  ".join(f"{k} {v}" for k, v in sorted(stop.unstarted_by_source.items()))
    return [
        "",
        "STOPPED AT THE SPEND CEILING — INCOMPLETE",
        f"  ceiling        {limit}   spend at the stop US{format_usd(stop.spend_at_stop_usd)}"
        f" ({stop.calls_at_stop} unpriced call(s))",
        f"  overshoot      US{format_usd(stop.overshoot_usd)}, {stop.overshoot_calls} call(s) "
        "in flight at the stop",
        f"  questions      {stop.unstarted} unstarted [{by_source or 'none'}], "
        f"{stop.interrupted} interrupted (US{format_usd(stop.discarded_usd)} over "
        f"{stop.discarded_calls} call(s) discarded)",
        f"  sentences      finished answers {_mean(stop.finished_sentences)} on average, "
        f"interrupted {_mean(stop.interrupted_sentences)}",
    ]


@dataclass
class _Tally:
    unstarted: int = 0
    interrupted: int = 0
    discarded_usd: float = 0.0
    discarded_calls: int = 0
    unstarted_by_source: dict[str, int] = field(default_factory=dict)
    interrupted_sentences: list[int] = field(default_factory=list)


class SpendCeiling:
    """The stop over one run's usage accumulator.

    :meth:`check` is called before each call starts. Once a stop is reached it
    stays reached: every later check raises, so no further call starts.
    """

    def __init__(
        self,
        usage: UsageAccumulator,
        *,
        max_usd: float | None = None,
        max_calls: int | None = None,
    ) -> None:
        self._usage = usage
        self.max_usd = max_usd
        self.max_calls = max_calls
        self._stopped = False
        self._at_stop: tuple[float, int] = (0.0, 0)
        self._all_calls_at_stop = 0
        self._tally = _Tally()
        self.report: StoppedAtCeiling | None = None

    @property
    def stopped(self) -> bool:
        """Whether a stop has been reached."""
        return self._stopped

    def check(self) -> None:
        """Raise :class:`CeilingReached` when a stop is reached; otherwise return."""
        if self._stopped:
            raise CeilingReached
        usage = self._usage.snapshot()
        usd, calls = _reading(usage)
        over_usd = self.max_usd is not None and usd >= self.max_usd
        over_calls = self.max_calls is not None and calls >= self.max_calls
        if over_usd or over_calls:
            self._stopped = True
            self._at_stop = (usd, calls)
            self._all_calls_at_stop = usage.calls
            raise CeilingReached

    def record_stopped_question(
        self, source: str, *, started: bool, usage: LLMUsage, sentences: int | None
    ) -> None:
        """Count one question the stop left incomplete, and what it discards."""
        tally = self._tally
        if not started:
            tally.unstarted += 1
            tally.unstarted_by_source[source] = tally.unstarted_by_source.get(source, 0) + 1
            return
        tally.interrupted += 1
        tally.discarded_usd += usage.priced_cost_usd
        tally.discarded_calls += usage.calls
        if sentences is not None:
            tally.interrupted_sentences.append(sentences)

    def settle(self, *, finished_sources: Iterable[str], finished_sentences: Iterable[int]) -> None:
        """Build :attr:`report` once every call has finished; a no-op when not stopped."""
        if not self._stopped:
            return
        usage = self._usage.snapshot()
        usd, _ = _reading(usage)
        at_usd, at_calls = self._at_stop
        by_source: dict[str, int] = {}
        for source in finished_sources:
            by_source[source] = by_source.get(source, 0) + 1
        tally = self._tally
        self.report = StoppedAtCeiling(
            max_usd=self.max_usd,
            max_calls=self.max_calls,
            spend_at_stop_usd=at_usd,
            calls_at_stop=at_calls,
            overshoot_usd=max(0.0, usd - at_usd),
            overshoot_calls=max(0, usage.calls - self._all_calls_at_stop),
            unstarted=tally.unstarted,
            interrupted=tally.interrupted,
            discarded_usd=tally.discarded_usd,
            discarded_calls=tally.discarded_calls,
            finished_by_source=dict(sorted(by_source.items())),
            unstarted_by_source=dict(sorted(tally.unstarted_by_source.items())),
            finished_sentences=list(finished_sentences),
            interrupted_sentences=list(tally.interrupted_sentences),
        )


class QuestionGuard:
    """One question's view of the stop: whether it has started a call yet."""

    def __init__(self, ceiling: SpendCeiling | None) -> None:
        self._ceiling = ceiling
        self.started = False

    def before_call(self) -> None:
        """Check the stop, then mark the question started; raises :class:`CeilingReached`."""
        if self._ceiling is not None:
            self._ceiling.check()
        self.started = True


def worst_case_usd(calls: Sequence[InFlight]) -> float | None:
    """Every projected call at its ``max_tokens``; ``None`` when a role is unpriced."""
    if any(c.role.price is None for c in calls):
        return None
    output, inputs = in_flight_usd(calls)
    return output + inputs

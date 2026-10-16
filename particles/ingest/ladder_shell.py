# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The shell around the rung ladder: availability, the async driver, the run tally.

The ladder itself is pure (:mod:`particles.core.rung_ladder`): it says which
rung runs next and which verdict stands. This module is what it needs from
the world:

* :func:`current_availability` reads the egress policy and the
  completion circuit breaker as they stand now, so a ``network``
  rung abstains under a refusing posture and a ``model`` rung while the
  breaker is open;
* :func:`drive` runs a ladder on one input, calling each rung's own
  ``decide`` or the shell's binding for an I/O rung, and asks for the
  availability afresh before each rung, since a breaker can open between two;
* :func:`rung_tally` opens a run-scoped accumulator (a ``ContextVar``, the
  :func:`particles.llm.usage.track_usage` pattern) that every ladder reports
  its verdicts to, so a consolidation run can say which rungs decided how much.
  With no scope open, reporting is a no-op.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextvars import ContextVar
from typing import Any, TypeVar

from particles.core.rung_ladder import (
    Availability,
    Ladder,
    QuestionKind,
    RungStamp,
    Settled,
)
from particles.egress import current_policy
from particles.llm.breaker import llm_circuit_open

__all__ = [
    "RungTally",
    "current_availability",
    "drive",
    "report",
    "report_settled",
    "rung_tally",
]

I = TypeVar("I")  # noqa: E741 - the ladder's input type, as in core.rung_ladder
V = TypeVar("V")

#: The tally key for "no rung decided".
NO_RUNG = "none"


def current_availability() -> Availability:
    """The egress policy and the breaker as they stand now."""
    policy = current_policy()
    return Availability(egress_allows=policy.allows, breaker_open=llm_circuit_open())


async def drive(
    ladder: Ladder[I, Any],
    item: I,
    bindings: Mapping[str, Callable[[I], Awaitable[Any]]],
    *,
    availability: Callable[[], Availability] = current_availability,
) -> Settled[Any]:
    """Run ``ladder`` on ``item`` to its settled verdict, and report it to the tally.

    ``bindings`` maps an I/O rung's name to the coroutine that answers it; a
    rung with its own ``decide`` needs none. Raises ``KeyError`` for an
    offered rung with neither.
    """
    run = ladder.start(item)
    while (rung := run.next_rung(availability())) is not None:
        if rung.decide is not None:
            verdict = rung.decide(item)
        else:
            verdict = await bindings[rung.name](item)
        run.record(rung, verdict)
    settled = run.settled()
    report_settled(ladder.kind, settled)
    return settled


class RungTally:
    """Per-ladder, per-rung verdict counts for one run.

    Three sections per question kind: ``rungs`` counts each verdict every
    non-shadow rung returned (the load each rung carried), ``settled`` the
    ladder's final verdicts by the rung that decided them (``none`` when every
    rung abstained), and ``shadow`` the verdicts of shadow rungs, recorded and
    never returned.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, dict[str, dict[str, dict[str, int]]]] = {}

    def add(
        self,
        kind: QuestionKind,
        section: str,
        stamp: RungStamp | None,
        verdict: str,
        n: int = 1,
    ) -> None:
        """Count ``n`` verdicts of one rung (``None``: no rung) in one section."""
        if n <= 0:
            return
        key = str(stamp) if stamp is not None else NO_RUNG
        with self._lock:
            cell = (
                self._counts.setdefault(kind.value, {}).setdefault(section, {}).setdefault(key, {})
            )
            cell[verdict] = cell.get(verdict, 0) + n

    def payload(self) -> dict[str, Any]:
        """The run record's ``rungs`` block: a plain, sorted copy."""
        with self._lock:
            return {
                kind: {
                    section: {
                        rung: dict(sorted(verdicts.items()))
                        for rung, verdicts in sorted(rungs.items())
                    }
                    for section, rungs in sorted(sections.items())
                }
                for kind, sections in sorted(self._counts.items())
            }


_TALLY: ContextVar[RungTally | None] = ContextVar("particles_rung_tally", default=None)


@contextlib.contextmanager
def rung_tally() -> Iterator[RungTally]:
    """Open a tally for the block; every ladder inside reports to it."""
    tally = RungTally()
    token = _TALLY.set(tally)
    try:
        yield tally
    finally:
        _TALLY.reset(token)


def report(
    kind: QuestionKind, section: str, stamp: RungStamp | None, verdict: str, n: int = 1
) -> None:
    """Count verdicts in the open tally; a no-op when none is open."""
    tally = _TALLY.get()
    if tally is not None:
        tally.add(kind, section, stamp, verdict, n)


def report_settled(kind: QuestionKind, settled: Settled[Any]) -> None:
    """Count one ladder run: each rung that ran, the outcome, and any shadow verdicts."""
    tally = _TALLY.get()
    if tally is None:
        return
    for stamp, verdict in settled.trail:
        tally.add(kind, "rungs", stamp, str(verdict))
    tally.add(kind, "settled", settled.decided_by, str(settled.verdict))
    for stamp, verdict in settled.shadow:
        tally.add(kind, "shadow", stamp, str(verdict))

# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Operator-ruled demotions as labelled pairs for the live update checks.

The synthetic worlds measure the update checks on templated sentences. An
operator's ruling on a ``demotion`` curation card is the same judgment made on
their own store: a "coexist" ruling says the older claim should have stayed,
a "replacement" ruling says the retirement was right. The curation queue keeps
each ruling in ``<benchmark.runs_dir>/demotion-rulings.jsonl``
(:mod:`particles.operations.curation.rulings`); this module re-asks the two
checks the update sweep asks of each pair (conflict, then same slot)
and scores the answer against the ruling:

* a "coexist" pair the checks would retire is a **false positive**;
* a "replacement" pair the checks would keep is a **miss**;
* a pair a check could not answer is **undecided**, counted apart.

The result is its own section of the report, never pooled into the world
metrics. Only the ``probe`` and ``live`` arms score it: the ``oracle`` arm's
scripted check answers from a world's value pools and has no ground truth for
a real claim.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

from particles.benchmark.rot.schema import RealPairOutcome, RealPairResult, RealPairsReport
from particles.config import get_config
from particles.operations.curation.rulings import DemotionRuling, load_rulings, rulings_path
from particles.operations.reconcile import update_checks

__all__ = ["UpdateChecks", "default_rulings_path", "score_real_pairs", "score_rulings_file"]

#: ``(older, newer) -> (contradiction, same_slot)``, the update sweep's two checks.
UpdateChecks = Callable[[str, str], Awaitable[tuple[bool | None, bool | None]]]


def default_rulings_path() -> Path:
    """The rulings file the curation queue writes (``benchmark.runs_dir``)."""
    return rulings_path()


def _decide(contradiction: bool | None, same_slot: bool | None) -> bool | None:
    """Whether the sweep would retire the older claim; ``None`` when it cannot say.

    A NO from the first check decides "keep" on its own, as in the sweep, so an
    unanswered second check matters only after a YES.
    """
    if contradiction is False or same_slot is False:
        return False
    if contradiction is True and same_slot is True:
        return True
    return None


async def score_real_pairs(
    rulings: Sequence[DemotionRuling],
    *,
    source: str,
    checks: UpdateChecks = update_checks,
    notes: Sequence[str] = (),
) -> RealPairsReport:
    """Ask the update checks about each ruled pair and score them against the rulings."""
    record_text = get_config().benchmark.record_claim_text
    report = RealPairsReport(source=source, pairs=len(rulings), notes=list(notes))
    for r in rulings:
        contradiction, same_slot = await checks(r.retired.content, r.replacement.content)
        retires = _decide(contradiction, same_slot)
        if retires is None:
            outcome = RealPairOutcome.UNDECIDED
            report.undecided += 1
        elif r.ruling == "coexist":
            report.false_positive.add(retires)
            outcome = RealPairOutcome.FALSE_POSITIVE if retires else RealPairOutcome.AGREE
        else:
            report.miss.add(not retires)
            outcome = RealPairOutcome.AGREE if retires else RealPairOutcome.MISS
        report.results.append(
            RealPairResult(
                retired_hash=r.retired.content_hash,
                replacement_hash=r.replacement.content_hash,
                reason=r.reason,
                ruling=r.ruling,
                retired_text=r.retired.content if record_text else None,
                replacement_text=r.replacement.content if record_text else None,
                contradiction=contradiction,
                same_slot=same_slot,
                retires=retires,
                outcome=outcome,
            )
        )
    return report


async def score_rulings_file(
    path: Path, *, checks: UpdateChecks = update_checks
) -> RealPairsReport | None:
    """Load ``path`` and score it; ``None`` when the file is absent or holds no ruling."""
    rulings, notes = load_rulings(path)
    if not rulings:
        return None
    return await score_real_pairs(rulings, source=str(path), checks=checks, notes=notes)

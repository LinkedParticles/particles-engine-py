"""Pure memory-benchmark metrics.

Functions over per-question retrieval hits and judged answers. No I/O, no SDK
seams — the runner accumulates the inputs and calls these to roll up the
report. Two deliberately separate families:

* **retrieval-stage** — :func:`recall_at_k` / :func:`precision_at_k` over the
  provenance-mapped session ids of the top-k retrieved particles vs. the
  dataset's labeled evidence sessions;
* **end-to-end QA** — :func:`qa_accuracy` / :func:`accuracy_by_type` over the
  judge's per-question verdicts.

Nothing here (or anywhere in the harness) merges the two into one number.

A question with **no labeled evidence sessions** (the LongMemEval abstention
variants) is *unscoreable* at the retrieval stage, not vacuously perfect:
:func:`recall_at_k` and :func:`precision_at_k` return ``None`` on an empty
evidence set, and the runner excludes such questions from every retrieval
aggregate while disclosing the excluded count
(``RetrievalStageMetrics.abstention_questions``). Returning ``None`` — rather
than keeping a vacuous ``1.0`` and filtering upstream — makes silent
re-blending the hardest failure to reintroduce: an accumulator that averages
the value fails ``mypy --strict`` (or raises at runtime) instead of quietly
inflating the mean. The remaining vacuous-denominator conventions
(``qa_accuracy([]) == 1.0``; ``precision_at_k`` with evidence but nothing
retrieved) match the content / modality / polarity siblings and are always
rendered beside their support counts.
"""

from __future__ import annotations

from collections.abc import Sequence
from collections.abc import Set as AbstractSet


def recall_at_k(
    hit_session_ids: AbstractSet[str], evidence_session_ids: AbstractSet[str]
) -> float | None:
    """Evidence-session recall: labeled evidence sessions hit / labeled.

    A session is "hit" when at least one top-k particle's provenance chain
    lands on it. Returns ``None`` when no evidence session is labeled (the
    abstention variants label none): retrieval is unscoreable there, and
    ``None`` — never a vacuous ``1.0`` — keeps the question out of every
    mean by construction (see the module docstring).
    """
    if not evidence_session_ids:
        return None
    return len(hit_session_ids & evidence_session_ids) / len(evidence_session_ids)


def precision_at_k(
    retrieved_session_ids: Sequence[str | None],
    evidence_session_ids: AbstractSet[str],
) -> float | None:
    """Retrieval precision: retrieved particles from evidence sessions / retrieved.

    ``retrieved_session_ids`` carries one entry per top-k particle — the
    session id its provenance chain resolved to, or ``None`` when the chain
    was broken (a broken chain can never count as an evidence hit). Returns
    ``None`` when no evidence session is labeled (abstention variants):
    against an empty evidence set every retrieved particle would count as a
    false positive, deterministically deflating the mean, so the question is
    unscoreable rather than scored 0. Returns ``1.0`` when evidence exists
    but nothing was retrieved (no false positives are possible); read
    together with the retrieved-particle count.
    """
    if not evidence_session_ids:
        return None
    if not retrieved_session_ids:
        return 1.0
    hits = sum(
        1 for sid in retrieved_session_ids if sid is not None and sid in evidence_session_ids
    )
    return hits / len(retrieved_session_ids)


def qa_accuracy(verdicts: Sequence[bool]) -> float:
    """Fraction of judged answers marked correct.

    Returns ``1.0`` on an empty verdict list (vacuous-denominator convention);
    the report always shows the question count next to the rate.
    """
    if not verdicts:
        return 1.0
    return sum(1 for v in verdicts if v) / len(verdicts)


def accuracy_by_type(pairs: Sequence[tuple[str, bool]]) -> dict[str, float]:
    """Per-question-type accuracy over ``(question_type, correct)`` pairs."""
    totals: dict[str, int] = {}
    correct: dict[str, int] = {}
    for qtype, verdict in pairs:
        totals[qtype] = totals.get(qtype, 0) + 1
        if verdict:
            correct[qtype] = correct.get(qtype, 0) + 1
    return {qtype: correct.get(qtype, 0) / total for qtype, total in totals.items()}


def mean_by_type(pairs: Sequence[tuple[str, float]]) -> dict[str, float]:
    """Per-question-type mean over ``(question_type, value)`` pairs."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for qtype, value in pairs:
        totals[qtype] = totals.get(qtype, 0.0) + value
        counts[qtype] = counts.get(qtype, 0) + 1
    return {qtype: totals[qtype] / counts[qtype] for qtype in totals}


def parse_judge_verdict(reply: str) -> bool:
    """Parse the LLM judge's yes/no reply into a correctness verdict.

    The judge prompt instructs a bare ``yes`` / ``no``; be tolerant of case,
    punctuation, and a short trailing rationale. Anything that does not start
    with an affirmative token counts as incorrect (fail-closed: an unparseable
    judge reply never inflates accuracy).
    """
    token = reply.strip().lower()
    for prefix in ("yes", "correct", "true"):
        if token == prefix or token.startswith(prefix + " ") or token.startswith(prefix + "."):
            return True
        if token.startswith(prefix + ","):
            return True
    return False

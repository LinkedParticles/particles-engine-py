# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure arithmetic for the relevance-floor benchmark.

Everything here is a function of recorded rows and a list of floors — no
store, encoder, or LLM — so a saved report can be re-swept over a different
floor list for free. The gate is reproduced exactly as the query op applies it:
a question is refused when ``max_similarity < floor``, strictly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from particles.benchmark.rot.schema import Rate

from .schema import FloorRow, HeldOutQuestion, QuestionResult, RefusalRow

_QUANTILES = (0, 10, 25, 50, 75, 90, 100)


def refused_at(max_similarity: float, floor: float) -> bool:
    """The verdict at ``floor`` — strict, so a floor of 0.0 never refuses."""
    return max_similarity < floor


def refusal_curve(results: Sequence[QuestionResult], floors: Sequence[float]) -> list[RefusalRow]:
    """Share of scoreable questions refused at each floor, overall and per source.

    Unjudged: this is what the gate *does*, and needs no label. Rows without a
    ``max_similarity`` (no encoder, empty result) are outside every denominator.
    """
    rows: list[RefusalRow] = []
    for floor in floors:
        row = RefusalRow(floor=floor)
        for result in results:
            if result.max_similarity is None:
                continue
            refused = refused_at(result.max_similarity, floor)
            row.refused.add(refused)
            row.refused_by_source.setdefault(result.source.value, Rate()).add(refused)
        rows.append(row)
    return rows


def floor_sweep(results: Sequence[QuestionResult], floors: Sequence[float]) -> list[FloorRow]:
    """The judged 2×2 table at each floor, under both conditionings.

    Only rows carrying both a cosine and a judged label count; an unjudged or
    excluded row joins no denominator.
    """
    judged = [
        (r.max_similarity, r.answerable)
        for r in results
        if r.max_similarity is not None and r.answerable is not None
    ]
    rows: list[FloorRow] = []
    for floor in floors:
        row = FloorRow(floor=floor)
        for max_similarity, answerable in judged:
            refused = refused_at(max_similarity, floor)
            if answerable:
                row.answerable_refused.add(refused)
            else:
                row.unanswerable_passed.add(not refused)
            if refused:
                row.refused_were_answerable.add(answerable)
            else:
                row.passed_were_unanswerable.add(not answerable)
        rows.append(row)
    return rows


def answerable_rate(results: Sequence[QuestionResult]) -> Rate:
    """Share of judged questions labelled answerable — the sweep's base rate."""
    rate = Rate()
    for result in results:
        if result.answerable is not None:
            rate.add(result.answerable)
    return rate


def similarity_quantiles(values: Sequence[float]) -> dict[str, float]:
    """Nearest-rank quantiles (p0 … p100) of the recorded cosines; empty in, empty out."""
    if not values:
        return {}
    ordered = sorted(values)
    last = len(ordered) - 1
    return {f"p{q}": ordered[round(q / 100 * last)] for q in _QUANTILES}


def sample_questions(
    questions: Sequence[HeldOutQuestion], limit: int | None, seed: int
) -> list[HeldOutQuestion]:
    """A deterministic sample of ``limit`` questions, stratified by source.

    Each source keeps its proportional share (largest remainder, and at least
    one question while the budget allows) so a small sample cannot silently
    drop the rare explicit-query sources. Within a source the order is a seeded
    hash of the question id, so a larger limit is a superset of a smaller one.
    """
    if limit is None or limit >= len(questions):
        return list(questions)
    by_source: dict[str, list[HeldOutQuestion]] = {}
    for question in questions:
        by_source.setdefault(question.source.value, []).append(question)
    for members in by_source.values():
        members.sort(key=lambda q: hashlib.sha256(f"{seed}:{q.question_id}".encode()).hexdigest())
    total = len(questions)
    exact = {s: limit * len(m) / total for s, m in by_source.items()}
    quota = {s: max(int(share), 1) for s, share in exact.items()}
    # Hand out (or claw back) the rounding difference by largest remainder.
    order = sorted(by_source, key=lambda s: (exact[s] - int(exact[s]), s), reverse=True)
    while sum(quota.values()) < limit:
        for source in order:
            if sum(quota.values()) < limit and quota[source] < len(by_source[source]):
                quota[source] += 1
    while sum(quota.values()) > limit:
        largest = max(quota, key=lambda s: (quota[s], s))
        quota[largest] -= 1
    sampled = [q for s in sorted(by_source) for q in by_source[s][: quota[s]]]
    return sorted(sampled, key=lambda q: q.question_id)

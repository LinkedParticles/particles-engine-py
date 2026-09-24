# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Hit classification and metric accumulation for the memory-rot benchmark.

Pure functions, no I/O. The classifier is the one judgement in the harness,
so it is deterministic and never an LLM: substring matching over a
value-disjoint world (:func:`particles.benchmark.rot.generator.check_value_invariants`),
refined by two pinned marker lexicons —

* a **history** marker ("previously", "used to", "no longer", …) turns a
  superseded value from ``STALE`` (served as current) into ``HISTORY`` (honest
  history — RotBench's round-4 correction, built in from the start);
* an **attribution** marker ("according to", "search result", "unverified",
  …) turns a poison value from ``POISON_ASSERTED`` into ``POISON_ATTRIBUTED``.

Bump :data:`SCORER_VERSION` on any change to either lexicon or to the class
precedence — it is on the run tuple, and reports scored under different
versions are not comparable.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from particles.benchmark.rot.generator import value_pattern
from particles.benchmark.rot.schema import (
    FIRST_HIT_MISS,
    FloorSweepRow,
    HitClass,
    ProbeResult,
    RotMetrics,
    SlotState,
)

#: The scorer version a run records. Bump on any lexicon or precedence change,
#: and add the new lexicons as a new entry in :data:`_LEXICONS` — never edit a
#: shipped version in place (the judge-protocol rule): reports scored
#: under different versions are not comparable, and ``rescore`` must be able to
#: reproduce any of them.
SCORER_VERSION = 2

#: The strict subset used when a particle carries the *current* value alone:
#: only unambiguous past framing demotes a current mention to ``HISTORY`` (the
#: reverted-value case), so a tense-neutral paraphrase ("the user was vegan")
#: never costs currency credit it has earned. Unchanged across versions; the
#: oracle contradiction probe reads it too.
STRONG_HISTORY_MARKERS: re.Pattern[str] = re.compile(
    r"\b(previously|formerly|used to|no longer|back when|former)\b",
    re.IGNORECASE,
)

_HISTORY_V1 = (
    r"previously|formerly|used to|no longer|any ?more|moved (?:away )?from|"
    r"switched (?:away )?from|left|former|old|until|back when|before|prior|"
    r"earlier|replaced|sold|cancell?ed|stopped|gave up|quit|was|were|"
    r"wrapped up|transferred from|went from|closed"
)
_ATTRIBUTION_V1 = (
    r"according to|search(?:ed)?|tool|profile (?:page|snippet)|snippet|"
    r"listing|directory|web ?page|website|online|claim(?:s|ed)?|reported(?:ly)?|"
    r"says|said|states|stated|unverified|user-submitted|allegedly|lookup|looked up|"
    r"found"
)

#: version → (history markers, attribution markers).
#:
#: v2, from the recorded hit texts of the first ``live`` run: a
#: public-profile phrasing ("The user's public profile lists the phone as
#: Zenfone.") read as asserted poison, and a completed-project mention ("The
#: user's project Zephyr has been completed.") read as stale. Past-tense
#: narration that names a value in passing ("meal-prepped every Sunday while
#: eating paleo") is deliberately *not* chased — no bounded lexicon catches it
#: without also swallowing real stale assertions; the recorded texts keep it
#: auditable.
_LEXICONS: dict[int, tuple[re.Pattern[str], re.Pattern[str]]] = {
    1: (
        re.compile(rf"\b({_HISTORY_V1})\b", re.IGNORECASE),
        re.compile(rf"\b({_ATTRIBUTION_V1})\b", re.IGNORECASE),
    ),
    2: (
        re.compile(rf"\b({_HISTORY_V1}|completed|finished)\b", re.IGNORECASE),
        re.compile(rf"\b({_ATTRIBUTION_V1}|public(?:ly)?|profile|lists|listed)\b", re.IGNORECASE),
    ),
}

#: The current version's lexicons, for callers that name them directly.
HISTORY_MARKERS, ATTRIBUTION_MARKERS = _LEXICONS[SCORER_VERSION]


def _has_any(text: str, values: Iterable[str]) -> bool:
    return any(value_pattern(v).search(text) for v in values)


def classify(text: str, state: SlotState, *, scorer_version: int = SCORER_VERSION) -> HitClass:
    """Classify one particle's text against one slot's truth at the checkpoint.

    ``scorer_version`` selects the marker lexicons (:data:`_LEXICONS`); the
    default is the current version, and ``rescore`` passes an older one only to
    reproduce a report as it was scored.

    Precedence:

    1. carries the current value and another of the slot's values → ``MIXED``
       (an honest transition, or an INCONSISTENCY wrapper naming both);
    2. carries the current value alone → ``CURRENT``, unless a *strong*
       history marker frames it as past (a reverted value's old mention) →
       ``HISTORY``;
    3. carries a superseded value → ``HISTORY`` with a history marker, else
       ``STALE`` (a superseded value served as current);
    4. carries a poison value → ``POISON_ATTRIBUTED`` with an attribution
       marker, else ``POISON_ASSERTED``;
    5. otherwise ``NONE`` (not value-bearing).
    """
    if scorer_version not in _LEXICONS:
        raise ValueError(f"unknown scorer version {scorer_version}; known: {sorted(_LEXICONS)}")
    history, attribution = _LEXICONS[scorer_version]
    has_current = state.current is not None and _has_any(text, [state.current])
    has_stale = _has_any(text, state.superseded)
    has_poison = _has_any(text, state.poison)
    if has_current:
        if has_stale or has_poison:
            return HitClass.MIXED
        return HitClass.HISTORY if STRONG_HISTORY_MARKERS.search(text) else HitClass.CURRENT
    if has_stale:
        return HitClass.HISTORY if history.search(text) else HitClass.STALE
    if has_poison:
        return HitClass.POISON_ATTRIBUTED if attribution.search(text) else HitClass.POISON_ASSERTED
    return HitClass.NONE


def first_value_bearing(result: ProbeResult) -> tuple[str, bool]:
    """``(class, contested)`` of the first value-bearing hit, or ``("miss", False)``."""
    for hit in result.hits:
        if hit.hit_class is not HitClass.NONE:
            return hit.hit_class.value, hit.contested
    return FIRST_HIT_MISS, False


def accumulate(metrics: RotMetrics, result: ProbeResult) -> None:
    """Add one probe to a metrics accumulator (negative probes are skipped).

    Denominators (the exclude-and-disclose rule): currency counts
    every non-negative probe; supersession only probes whose slot has a
    superseded value by the checkpoint; poison only probes whose slot has had a
    poison value injected by the checkpoint.
    """
    if result.negative:
        return
    classes = {h.hit_class for h in result.hits}
    first = result.first_hit

    metrics.recall_current_at_k.add(bool(classes & {HitClass.CURRENT, HitClass.MIXED}))
    metrics.current_first.add(first == HitClass.CURRENT.value)
    metrics.first_hit_counts[first] = metrics.first_hit_counts.get(first, 0) + 1

    if result.superseded:
        stale_first = first == HitClass.STALE.value
        metrics.stale_over_current.add(stale_first)
        metrics.stale_over_current_unflagged.add(stale_first and not result.first_hit_contested)
        metrics.stale_retained_at_k.add(HitClass.STALE in classes)

    if result.poison:
        metrics.poison_leak_at_k.add(HitClass.POISON_ASSERTED in classes)
        metrics.poison_first.add(first == HitClass.POISON_ASSERTED.value)
        metrics.poison_surfaced_at_k.add(
            bool(classes & {HitClass.POISON_ASSERTED, HitClass.POISON_ATTRIBUTED})
        )


def metrics_for(results: Iterable[ProbeResult]) -> RotMetrics:
    """Metrics over a set of probe results."""
    out = RotMetrics()
    for r in results:
        accumulate(out, r)
    return out


def merge_metrics(parts: Iterable[RotMetrics]) -> RotMetrics:
    """Pool several accumulators (numerators and denominators summed)."""
    out = RotMetrics()
    for m in parts:
        for name in RotMetrics.model_fields:
            if name == "first_hit_counts":
                for k, v in m.first_hit_counts.items():
                    out.first_hit_counts[k] = out.first_hit_counts.get(k, 0) + v
                continue
            setattr(out, name, getattr(out, name).merged(getattr(m, name)))
    return out


def breakdowns(
    results: list[ProbeResult],
) -> tuple[dict[int, RotMetrics], dict[str, RotMetrics], dict[str, RotMetrics]]:
    """Per-checkpoint, per-phrasing, and per-poison-channel metrics.

    Phrasing rows cover probes on changed slots (grouped by the form of the
    slot's latest update); channel rows cover poison-eligible probes, grouped
    by the channel that delivered the slot's poison.
    """
    by_cp: dict[int, RotMetrics] = {}
    by_phrasing: dict[str, RotMetrics] = {}
    by_channel: dict[str, RotMetrics] = {}
    for r in results:
        if r.negative:
            continue
        accumulate(by_cp.setdefault(r.checkpoint, RotMetrics()), r)
        if r.last_phrasing is not None:
            accumulate(by_phrasing.setdefault(r.last_phrasing.value, RotMetrics()), r)
        for ch in sorted({c.value for c in r.poison_channels}):
            accumulate(by_channel.setdefault(ch, RotMetrics()), r)
    return by_cp, by_phrasing, by_channel


def floor_sweep(results: list[ProbeResult], floors: list[float]) -> list[FloorSweepRow]:
    """The offline relevance-floor sweep over recorded top-1 cosines.

    ``answerable_refused``: of the probes whose current value was in top-k, the
    share whose top cosine sat below the floor — the product would have
    refused a question its store could answer. ``unanswerable_passed``: of the
    negative probes, the share whose top cosine cleared the floor. Both are
    retrieval-grounded proxies on synthetic data, not a judged measurement.
    """
    rows: list[FloorSweepRow] = []
    for floor in floors:
        row = FloorSweepRow(floor=floor)
        for r in results:
            if r.top_cosine is None:
                continue
            if r.negative:
                row.unanswerable_passed.add(r.top_cosine >= floor)
            elif any(h.hit_class in (HitClass.CURRENT, HitClass.MIXED) for h in r.hits):
                row.answerable_refused.add(r.top_cosine < floor)
        rows.append(row)
    return rows

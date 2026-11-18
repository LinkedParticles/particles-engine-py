# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Door-text scoring for the rot benchmark's door arm.

Pure functions, no I/O. Only the door text is scored: the text a client of
the memory's default door literally receives. It is split into segments at
line breaks and sentence ends (:func:`segment`, versioned by
:data:`DOOR_SEGMENTER_VERSION`), and each segment is classified by the
classifier against the probed slot. A memory's own ordering decides which
segment comes first, because that is what the model reads first.

No feature is credited. A memory that retired a value, attributed a poisoned
one or abstained scores for it only when the door text shows it.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Iterable, Sequence

from particles.benchmark.rot.door_schema import (
    DoorBuild,
    DoorMemoryResult,
    DoorMetrics,
    DoorProbeResult,
    DoorReport,
    DoorSegment,
    MetricRange,
    PredictionResult,
    PriorScoring,
    Stat,
)
from particles.benchmark.rot.generator import generate_world, slot_state
from particles.benchmark.rot.schema import FIRST_HIT_MISS, HitClass, RotWorld
from particles.benchmark.rot.scoring import SCORER_VERSION, classify

#: Bump on any change to how a door text is cut into segments. It joins the
#: run tuple, and ``rescore`` can re-cut a saved report under any version.
DOOR_SEGMENTER_VERSION = 1
_SEGMENTERS = (1,)

#: The context budgets ``recall_current@B`` is measured at, in characters.
BUDGETS_CHARS: tuple[int, ...] = (2000, 8000)

#: A cut inside a line: a JSON-escaped line break (the two characters ``\n``,
#: which a door that returns JSON uses for every line break in a string), or
#: whitespace after a sentence end.
_CUT = re.compile(r"\\n|(?<=[.!?])\s+")

#: Metric name → which way is better. The order is the render order, and the
#: last group is the axes where Particles is expected to lose.
METRIC_DIRECTIONS: dict[str, str] = {
    "current_in_door": "up",
    "current_first": "up",
    "recall_current_at_2k": "up",
    "recall_current_at_8k": "up",
    "stale_over_current": "down",
    "stale_in_door": "down",
    "poison_first": "down",
    "poison_leak": "down",
    "poison_surfaced": "down",
    "negative_silent": "up",
    "negative_chars": "down",
    "chars_per_door": "down",
    "write_ms": "down",
    "door_ms": "down",
    "build_minutes": "down",
    "build_usd": "down",
}

#: The axes where Particles is expected to lose.
EXPECTED_LOSSES: tuple[str, ...] = (
    "recall_current_at_2k",
    "chars_per_door",
    "negative_silent",
    "write_ms",
    "build_usd",
)

_PREDICTION_ORDER = {"trail": 0, "tie": 1, "beat": 2}


def segment(text: str, *, version: int = DOOR_SEGMENTER_VERSION) -> list[tuple[int, str]]:
    """``(offset, text)`` of each non-blank segment of a door text, in order.

    Version 1 cuts at every line break and at every JSON-escaped one (``\\n``
    inside a string), then inside a line after ``.``, ``!`` or ``?`` followed
    by whitespace. The offset is the segment's first
    non-blank character in the original text, so a budget is measured against
    what the model actually reads.

    Raises:
        ValueError: On an unknown segmenter version.
    """
    if version not in _SEGMENTERS:
        raise ValueError(f"unknown door segmenter version {version}; known: {list(_SEGMENTERS)}")
    out: list[tuple[int, str]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        line_start = pos
        pos += len(line)
        body = line.rstrip("\r\n")
        cut = 0
        pieces: list[tuple[int, str]] = []
        for m in _CUT.finditer(body):
            pieces.append((cut, body[cut : m.start()]))
            cut = m.end()
        pieces.append((cut, body[cut:]))
        for offset, piece in pieces:
            stripped = piece.strip()
            if stripped:
                lead = len(piece) - len(piece.lstrip())
                out.append((line_start + offset + lead, stripped))
    return out


def world_value_pattern(world: RotWorld) -> re.Pattern[str]:
    """One matcher for every value of every slot in play (the negative-probe test)."""
    values = sorted({v for pool in world.pools.values() for v in pool}, key=len, reverse=True)
    body = "|".join(re.escape(v) for v in values)
    return re.compile(rf"(?<![\w-])(?:{body})(?![\w-])", re.IGNORECASE)


def score_probe(
    result: DoorProbeResult,
    world: RotWorld,
    *,
    scorer_version: int = SCORER_VERSION,
    segmenter_version: int = DOOR_SEGMENTER_VERSION,
    any_value: re.Pattern[str] | None = None,
) -> None:
    """Fill a probe's truth, segments and first class from its recorded door text."""
    result.segments = []
    result.first_class = FIRST_HIT_MISS
    result.first_offset = None
    result.silent = None
    if result.error is not None:
        return
    if result.negative:
        pattern = any_value if any_value is not None else world_value_pattern(world)
        result.silent = pattern.search(result.door_text) is None
        return
    state = slot_state(world, result.slot, result.checkpoint)
    result.current = state.current
    result.superseded = state.superseded
    result.poison = state.poison
    result.poison_channels = state.poison_channels
    result.last_phrasing = state.last_phrasing
    for offset, text in segment(result.door_text, version=segmenter_version):
        cls = classify(text, state, scorer_version=scorer_version)
        if cls is not HitClass.NONE:
            result.segments.append(DoorSegment(start=offset, text=text, hit_class=cls))
    if result.segments:
        result.first_class = result.segments[0].hit_class.value
        result.first_offset = result.segments[0].start


def stat(values: Sequence[float]) -> Stat:
    """Median and 90th percentile (nearest rank) of a sample."""
    if not values:
        return Stat()
    ordered = sorted(values)
    p90 = ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]
    return Stat(n=len(ordered), median=float(statistics.median(ordered)), p90=float(p90))


def door_metrics(probes: Iterable[DoorProbeResult], write_ms: Sequence[float] = ()) -> DoorMetrics:
    """The metric table over a set of scored probes.

    Denominators follow the rot harness: currency counts every non-negative probe,
    supersession only probes whose slot has a superseded value, source trust
    only probes whose slot has been poisoned. A probe whose door call failed
    leaves every denominator and is counted in ``errors``.
    """
    m = DoorMetrics()
    chars: list[float] = []
    ms: list[float] = []
    neg_chars: list[float] = []
    for r in probes:
        if r.error is not None:
            m.errors += 1
            continue
        chars.append(float(r.chars))
        ms.append(r.door_ms)
        if r.negative:
            m.negative_silent.add(bool(r.silent))
            neg_chars.append(float(r.chars))
            continue
        classes = {s.hit_class for s in r.segments}
        first = r.first_class
        current_starts = [
            s.start for s in r.segments if s.hit_class in (HitClass.CURRENT, HitClass.MIXED)
        ]
        m.current_in_door.add(bool(current_starts))
        m.current_first.add(first == HitClass.CURRENT.value)
        m.recall_current_at_2k.add(any(s < BUDGETS_CHARS[0] for s in current_starts))
        m.recall_current_at_8k.add(any(s < BUDGETS_CHARS[1] for s in current_starts))
        m.first_class_counts[first] = m.first_class_counts.get(first, 0) + 1
        if r.superseded:
            m.stale_over_current.add(first == HitClass.STALE.value)
            m.stale_in_door.add(HitClass.STALE in classes)
        if r.poison:
            m.poison_first.add(first == HitClass.POISON_ASSERTED.value)
            m.poison_leak.add(HitClass.POISON_ASSERTED in classes)
            m.poison_surfaced.add(
                bool(classes & {HitClass.POISON_ASSERTED, HitClass.POISON_ATTRIBUTED})
            )
    m.chars_per_door = stat(chars)
    m.door_ms = stat(ms)
    m.negative_chars = stat(neg_chars)
    m.write_ms = stat(write_ms)
    return m


def channel_metrics(probes: Sequence[DoorProbeResult]) -> dict[str, DoorMetrics]:
    """Per-poison-channel metrics over the poison-eligible probes."""
    out: dict[str, list[DoorProbeResult]] = {}
    for r in probes:
        if r.negative or r.error is not None:
            continue
        for ch in sorted({c.value for c in r.poison_channels}):
            out.setdefault(ch, []).append(r)
    return {ch: door_metrics(rs) for ch, rs in out.items()}


def score_build(
    build: DoorBuild,
    world: RotWorld,
    *,
    scorer_version: int = SCORER_VERSION,
    segmenter_version: int = DOOR_SEGMENTER_VERSION,
) -> None:
    """Score every probe of a build and fill its metrics."""
    pattern = world_value_pattern(world)
    for r in build.probes:
        score_probe(
            r,
            world,
            scorer_version=scorer_version,
            segmenter_version=segmenter_version,
            any_value=pattern,
        )
    build.metrics = door_metrics(build.probes, build.write_ms_samples)
    build.by_channel = channel_metrics(build.probes)


def metric_values(metrics: DoorMetrics, build: DoorBuild | None = None) -> dict[str, float | None]:
    """Each metric's headline number: a rate's value, a sample's median."""
    out: dict[str, float | None] = {}
    for name in METRIC_DIRECTIONS:
        if name == "build_minutes":
            out[name] = build.build_minutes if build is not None else None
        elif name == "build_usd":
            out[name] = build.usd if build is not None else None
        else:
            field = getattr(metrics, name)
            out[name] = field.value if hasattr(field, "numerator") else field.median
    return out


def finalize_memory(memory: DoorMemoryResult) -> None:
    """Pool a memory's builds and compute each metric's mean and range."""
    probes = [p for b in memory.builds for p in b.probes]
    samples = [x for b in memory.builds for x in b.write_ms_samples]
    memory.metrics = door_metrics(probes, samples)
    per_build = [metric_values(b.metrics, b) for b in memory.builds]
    memory.ranges = {}
    for name in METRIC_DIRECTIONS:
        vals = [v[name] for v in per_build if v[name] is not None]
        nums = [float(x) for x in vals if x is not None]
        memory.ranges[name] = MetricRange(
            mean=statistics.fmean(nums) if nums else None,
            lo=min(nums) if nums else None,
            hi=max(nums) if nums else None,
            builds=len(nums),
        )


def _observed(particles: MetricRange, others: list[MetricRange], direction: str) -> str | None:
    """``beat`` / ``trail`` / ``tie`` of Particles against the median competitor."""
    means = sorted((r for r in others if r.mean is not None), key=lambda r: r.mean or 0.0)
    if particles.mean is None or not means:
        return None
    median = means[(len(means) - 1) // 2]
    assert median.mean is not None
    p_lo = particles.lo if particles.lo is not None else particles.mean
    p_hi = particles.hi if particles.hi is not None else particles.mean
    m_lo = median.lo if median.lo is not None else median.mean
    m_hi = median.hi if median.hi is not None else median.mean
    if p_lo <= m_hi and m_lo <= p_hi:
        # Overlapping ranges are never ranked (adebench's rule).
        return "tie"
    better = particles.mean > median.mean if direction == "up" else particles.mean < median.mean
    return "beat" if better else "trail"


def evaluate_predictions(
    predictions: dict[str, str], memories: Sequence[DoorMemoryResult]
) -> list[PredictionResult]:
    """Each pre-registered prediction beside the run's observation.

    Particles' headline row is compared with the median of the competitors'
    headline rows (baselines and variants excluded), on each metric's mean;
    overlapping ranges read as a tie. An answer metric is read off the
    headline reader's ``door`` reading.
    """
    # Deferred import: cycle break, since the answer stage scores with this module.
    from particles.benchmark.rot.answer import ANSWER_METRIC_DIRECTIONS, HEADLINE_READER, answer_key

    headline = answer_key(HEADLINE_READER, "door")

    def ranges(m: DoorMemoryResult, metric: str) -> MetricRange:
        if metric in ANSWER_METRIC_DIRECTIONS:
            return m.answer_ranges.get(headline, {}).get(metric, MetricRange())
        return m.ranges.get(metric, MetricRange())

    particles = next((m for m in memories if m.name == "particles" and m.role == "headline"), None)
    others = [m for m in memories if m.role == "headline" and m.name != "particles"]
    out: list[PredictionResult] = []
    for metric, predicted in predictions.items():
        direction = METRIC_DIRECTIONS.get(metric) or ANSWER_METRIC_DIRECTIONS.get(metric)
        observed = None
        if particles is not None and direction is not None:
            observed = _observed(
                ranges(particles, metric), [ranges(m, metric) for m in others], direction
            )
        if observed is None or predicted not in _PREDICTION_ORDER:
            out.append(
                PredictionResult(
                    metric=metric,
                    predicted=predicted,
                    observed=observed or "n/a",
                    verdict="not observed",
                )
            )
            continue
        delta = _PREDICTION_ORDER[observed] - _PREDICTION_ORDER[predicted]
        verdict = (
            "as predicted"
            if delta == 0
            else ("better than predicted" if delta > 0 else "worse than predicted")
        )
        out.append(
            PredictionResult(metric=metric, predicted=predicted, observed=observed, verdict=verdict)
        )
    return out


def _label(m: DoorMemoryResult) -> str:
    return m.name if m.variant_note is None else f"{m.name} ({m.variant_note})"


def rescore_door_report(
    report: DoorReport,
    *,
    scorer_version: int = SCORER_VERSION,
    segmenter_version: int = DOOR_SEGMENTER_VERSION,
) -> DoorReport:
    """Re-cut and re-classify a saved report's door texts and replies; nothing is called.

    Recorded replies are re-scored under the current reply scorer.
    The headline figures the report carried are kept in ``prior_scorings`` and
    printed beside the new ones, so a scorer or segmenter change can never
    move a ranking without showing that it did.
    """
    # Deferred import: cycle break, since the answer stage scores with this module.
    from particles.benchmark.rot.answer import (
        HEADLINE_READER,
        REPLY_SCORER_VERSION,
        answer_key,
        finalize_answers,
        score_answers,
    )

    out = report.model_copy(deep=True)
    s = out.selection
    headline = answer_key(HEADLINE_READER, "door")
    prior = PriorScoring(
        scorer_version=s.scorer_version,
        door_segmenter_version=s.door_segmenter_version,
        figures={_label(m): {k: r.mean for k, r in m.ranges.items()} for m in out.memories},
        reply_scorer_version=s.answer.reply_scorer_version if s.answer else None,
        answer_figures={
            _label(m): {k: r.mean for k, r in m.answer_ranges[headline].items()}
            for m in out.memories
            if headline in m.answer_ranges
        },
    )
    worlds: dict[int, RotWorld] = {}
    for memory in out.memories:
        for build in memory.builds:
            if not build.probes:
                continue
            if build.seed not in worlds:
                worlds[build.seed] = generate_world(build.seed, days=s.days, profile=s.profile)
            score_build(
                build,
                worlds[build.seed],
                scorer_version=scorer_version,
                segmenter_version=segmenter_version,
            )
            if s.answer is not None:
                score_answers(
                    build,
                    worlds[build.seed],
                    strip_version=s.answer.answer_strip_version,
                    scorer_version=scorer_version,
                    segmenter_version=segmenter_version,
                )
        scored = memory.model_copy(update={"builds": [b for b in memory.builds if b.probes]})
        finalize_memory(scored)
        memory.metrics, memory.ranges = scored.metrics, scored.ranges
        if s.answer is not None:
            finalize_answers(memory)
    out.prior_scorings = [*out.prior_scorings, prior]
    s.scorer_version = scorer_version
    s.door_segmenter_version = segmenter_version
    if s.answer is not None:
        s.answer.reply_scorer_version = REPLY_SCORER_VERSION
    out.prediction_results = evaluate_predictions(out.predictions, out.memories)
    return out

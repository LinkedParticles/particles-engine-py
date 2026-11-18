# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The door arm's answer stage: what a reader answers from a door.

Pure functions, no I/O. One pinned reader is handed a probe's question, the
checkpoint's date and a door's text, and answers in one sentence; this module
holds the pinned prompt (:func:`reader_prompt`), the readings derived from a
recorded door (:func:`derive_reading`), the deterministic reply scorer
(:func:`score_reply`, versioned by :data:`REPLY_SCORER_VERSION`) and the
answer family's metrics (:func:`answer_metrics`). The calls themselves, their
cache and the estimate live in :mod:`.answer_runner`.

There is no LLM judge: a reply is cut by the door segmenter and classified by
the classifier, then given one verdict by the first rule that
matches.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections.abc import Iterable
from datetime import date, timedelta
from enum import StrEnum

from particles.benchmark.rot.door import (
    DOOR_SEGMENTER_VERSION,
    segment,
    stat,
    world_value_pattern,
)
from particles.benchmark.rot.door_schema import (
    AnswerMetrics,
    DoorBuild,
    DoorMemoryResult,
    DoorProbeResult,
    DoorSegment,
    MetricRange,
    ReplyResult,
)
from particles.benchmark.rot.generator import slot_state, value_pattern
from particles.benchmark.rot.schema import HitClass, RotWorld
from particles.benchmark.rot.scoring import SCORER_VERSION, classify

#: Bump on any change to the reader prompt; a new version is a new cache key.
READER_PROMPT_VERSION = 1

_PROMPTS: dict[int, str] = {
    1: (
        "Today is {date}.\n\n"
        "Below is the text a memory system returned for a question about the user.\n\n"
        "Question: {question}\n\n"
        "<memory>\n{text}\n</memory>\n\n"
        "Using only that text, answer the question in one sentence with the user's "
        "current value, copying the value as the text writes it. If the text does not "
        "contain the answer, reply exactly: NOT IN MEMORY"
    ),
}

#: Bump on any change to which fields the ``hits_only`` reading removes.
ANSWER_STRIP_VERSION = 1
_STRIP_FIELDS: dict[int, tuple[str, ...]] = {1: ("answer", "answer_attribution")}

#: Bump on any change to the verdict rules; ``rescore`` re-applies the current one.
REPLY_SCORER_VERSION = 1
_REPLY_SCORERS = (1,)

#: The headline reading of every memory, then Particles' two diagnostic ones.
READINGS: tuple[str, ...] = ("door", "hits_only", "answer_only")
DIAGNOSTIC_READINGS: tuple[str, ...] = ("hits_only", "answer_only")
HEADLINE_READER = "headline"
SENSITIVITY_READER = "sensitivity"

#: The reply the prompt asks for when the text does not hold the answer.
NOT_IN_MEMORY = "NOT IN MEMORY"
_REFUSAL = re.compile(r"\s*not in memory\s*\.?\s*", re.IGNORECASE)


class ReplyVerdict(StrEnum):
    """One reply's verdict."""

    CURRENT = "current"
    HEDGED = "hedged"
    STALE = "stale"
    POISON = "poison"
    ABSTAIN = "abstain"
    CONFABULATED = "confabulated"
    MISS = "miss"


#: The probe sets a verdict distribution is counted over.
PROBE_SETS: tuple[str, ...] = ("currency", "supersession", "poison", "negative")

#: Answer metric → which way is better. ``answer_poison_disclosed`` is
#: reported, not ranked, so it has no direction and takes no prediction.
ANSWER_METRIC_DIRECTIONS: dict[str, str] = {
    "answer_current": "up",
    "answer_stale": "down",
    "answer_hedged": "down",
    "answer_poison": "down",
    "answer_abstain": "up",
    "answer_false_abstain": "down",
    "reader_missed": "down",
    "reader_tokens_in": "down",
    "reader_overflow": "down",
    "reader_usd": "down",
}
#: The answer metrics a freeze must predict: the headline reader never
#: overflows, so ``reader_overflow`` belongs to the sensitivity row only.
PREDICTED_ANSWER_METRICS: tuple[str, ...] = tuple(
    m for m in ANSWER_METRIC_DIRECTIONS if m != "reader_overflow"
)
#: The answer axes where Particles is expected to lose.
ANSWER_EXPECTED_LOSSES: tuple[str, ...] = ("reader_tokens_in", "reader_usd")
#: Each answer metric beside the door-text metric of the same family, so a
#: reader-dependent number is never quoted alone.
DOOR_COUNTERPART: dict[str, str] = {
    "answer_current": "current_first",
    "answer_stale": "stale_over_current",
    "answer_hedged": "stale_in_door",
    "answer_poison": "poison_leak",
    "answer_abstain": "negative_silent",
    "reader_tokens_in": "chars_per_door",
}


def prompt_template(version: int = READER_PROMPT_VERSION) -> str:
    """The reader prompt's template at ``version``.

    Raises:
        ValueError: On an unknown version.
    """
    if version not in _PROMPTS:
        raise ValueError(f"unknown reader prompt version {version}; known: {sorted(_PROMPTS)}")
    return _PROMPTS[version]


def prompt_sha256(version: int = READER_PROMPT_VERSION) -> str:
    """SHA-256 of the prompt template, hashed into the run tuple and the cache key."""
    return hashlib.sha256(prompt_template(version).encode("utf-8")).hexdigest()


def reader_prompt(
    question: str, on: str, text: str, *, version: int = READER_PROMPT_VERSION
) -> str:
    """The prompt the reader is handed: the date, the question, the text uncut."""
    return prompt_template(version).format(date=on, question=question, text=text)


def strip_fields(version: int = ANSWER_STRIP_VERSION) -> tuple[str, ...]:
    """The composed-answer fields the ``hits_only`` reading removes at ``version``.

    Raises:
        ValueError: On an unknown version.
    """
    if version not in _STRIP_FIELDS:
        raise ValueError(f"unknown answer strip version {version}; known: {sorted(_STRIP_FIELDS)}")
    return _STRIP_FIELDS[version]


def probe_date(build: DoorBuild, probe: DoorProbeResult) -> str:
    """The checkpoint's date in the build's own epoch."""
    return (date.fromisoformat(build.epoch) + timedelta(days=probe.checkpoint)).isoformat()


def derive_reading(
    door_text: str, reading: str, *, strip_version: int = ANSWER_STRIP_VERSION
) -> tuple[str | None, str | None]:
    """``(text, why_not)`` of one reading of a recorded door.

    ``door`` is the text as delivered. ``hits_only`` parses the door's JSON,
    removes the composed-answer fields and re-serialises it as the MCP tool
    does; ``answer_only`` is the ``answer`` string alone. A door that does not
    parse, or carries no ``answer``, has no diagnostic reading, and
    ``why_not`` says so.

    Raises:
        ValueError: On an unknown reading.
    """
    if reading == "door":
        return door_text, None
    if reading not in DIAGNOSTIC_READINGS:
        raise ValueError(f"unknown reading {reading!r}; known: {list(READINGS)}")
    try:
        data = json.loads(door_text)
    except ValueError:
        return None, "the door text is not JSON"
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
        return None, "the door carries no answer field"
    if reading == "answer_only":
        return str(data["answer"]), None
    # Deferred import: the child's renderer is the one place the tool's
    # serialisation is written down, and it pulls in pydantic_core lazily.
    from particles.benchmark.rot.door_child import render_tool_result

    stripped = {k: v for k, v in data.items() if k not in strip_fields(strip_version)}
    return render_tool_result(stripped), None


def is_refusal(reply: str) -> bool:
    """The reply is ``NOT IN MEMORY``, case and a trailing period ignored."""
    return _REFUSAL.fullmatch(reply) is not None


def _value_segments(
    text: str, probe: DoorProbeResult, world: RotWorld, scorer_version: int, segmenter_version: int
) -> list[DoorSegment]:
    state = slot_state(world, probe.slot, probe.checkpoint)
    out: list[DoorSegment] = []
    for offset, piece in segment(text, version=segmenter_version):
        cls = classify(piece, state, scorer_version=scorer_version)
        if cls is not HitClass.NONE:
            out.append(DoorSegment(start=offset, text=piece, hit_class=cls))
    return out


def score_reply(  # noqa: PLR0911, PLR0912 — the verdict table is one rule per branch
    reply: ReplyResult,
    probe: DoorProbeResult,
    world: RotWorld,
    *,
    strip_version: int = ANSWER_STRIP_VERSION,
    reply_scorer_version: int = REPLY_SCORER_VERSION,
    scorer_version: int = SCORER_VERSION,
    segmenter_version: int = DOOR_SEGMENTER_VERSION,
    any_value: re.Pattern[str] | None = None,
) -> None:
    """Fill a reply's segments and verdict, first matching rule wins.

    Negative probes: ``abstain`` (the exact refusal, no value of the world),
    ``confabulated`` (any value of the world), ``miss``. Answerable probes, in
    order: ``poison`` asserted; ``hedged`` (a stale or mixed segment and the
    current value somewhere); ``stale``; ``current`` (history and attributed
    poison allowed beside it, the latter counted as a disclosure); ``poison``
    adopted with its source (attributed poison and no current value);
    ``abstain`` (a false abstention); ``miss``.

    Raises:
        ValueError: On an unknown reply scorer version.
    """
    if reply_scorer_version not in _REPLY_SCORERS:
        raise ValueError(
            f"unknown reply scorer version {reply_scorer_version}; known: {list(_REPLY_SCORERS)}"
        )
    reply.verdict = None
    reply.poison_kind = None
    reply.disclosed = False
    reply.segments = []
    reply.current_in_text = None
    if reply.reply is None or reply.error is not None or reply.skipped or reply.overflow:
        return
    text = reply.reply
    if probe.negative:
        pattern = any_value if any_value is not None else world_value_pattern(world)
        carries = pattern.search(text) is not None
        if is_refusal(text) and not carries:
            reply.verdict = ReplyVerdict.ABSTAIN.value
        elif carries:
            reply.verdict = ReplyVerdict.CONFABULATED.value
        else:
            reply.verdict = ReplyVerdict.MISS.value
        return
    handed, _ = derive_reading(probe.door_text, reply.reading, strip_version=strip_version)
    if reply.reading == "door":
        reply.current_in_text = any(
            s.hit_class in (HitClass.CURRENT, HitClass.MIXED) for s in probe.segments
        )
    elif handed is not None:
        reply.current_in_text = any(
            s.hit_class in (HitClass.CURRENT, HitClass.MIXED)
            for s in _value_segments(handed, probe, world, scorer_version, segmenter_version)
        )
    reply.segments = _value_segments(text, probe, world, scorer_version, segmenter_version)
    classes = {s.hit_class for s in reply.segments}
    state = slot_state(world, probe.slot, probe.checkpoint)
    has_current = (
        state.current is not None and value_pattern(state.current).search(text) is not None
    )
    if HitClass.POISON_ASSERTED in classes:
        reply.verdict, reply.poison_kind = ReplyVerdict.POISON.value, "asserted"
    elif classes & {HitClass.STALE, HitClass.MIXED} and has_current:
        reply.verdict = ReplyVerdict.HEDGED.value
    elif HitClass.STALE in classes:
        reply.verdict = ReplyVerdict.STALE.value
    elif HitClass.CURRENT in classes:
        reply.verdict = ReplyVerdict.CURRENT.value
        reply.disclosed = HitClass.POISON_ATTRIBUTED in classes
    elif HitClass.POISON_ATTRIBUTED in classes:
        reply.verdict, reply.poison_kind = ReplyVerdict.POISON.value, "adopted"
    elif is_refusal(text):
        reply.verdict = ReplyVerdict.ABSTAIN.value
    else:
        reply.verdict = ReplyVerdict.MISS.value


def answer_key(reader: str, reading: str) -> str:
    """The key an answer family is filed under: ``<reader>/<reading>``."""
    return f"{reader}/{reading}"


def answer_metrics(
    probes: Iterable[DoorProbeResult], *, reader: str, reading: str
) -> AnswerMetrics:
    """The answer family over one reader's replies to one reading."""
    m = AnswerMetrics()
    tokens: list[float] = []
    usd: float | None = 0.0
    for probe in probes:
        if probe.error is not None:
            continue
        reply = next(
            (r for r in probe.replies if r.reader == reader and r.reading == reading), None
        )
        if reply is None:
            continue
        if reply.skipped:
            m.skipped += 1
            continue
        m.reader_overflow.add(reply.overflow)
        if reply.overflow:
            continue
        if reply.error is not None or reply.verdict is None:
            m.errors += 1
            continue
        m.cached += int(reply.cached)
        if reply.input_tokens is not None:
            tokens.append(float(reply.input_tokens))
        usd = None if usd is None or reply.usd is None else usd + reply.usd
        verdict = reply.verdict
        sets = ["negative"] if probe.negative else ["currency"]
        if probe.negative:
            m.answer_abstain.add(verdict == ReplyVerdict.ABSTAIN)
        else:
            m.answer_current.add(verdict == ReplyVerdict.CURRENT)
            m.answer_false_abstain.add(verdict == ReplyVerdict.ABSTAIN)
            if reply.current_in_text:
                m.reader_missed.add(verdict != ReplyVerdict.CURRENT)
            if probe.superseded:
                sets.append("supersession")
                m.answer_stale.add(verdict == ReplyVerdict.STALE)
                m.answer_hedged.add(verdict == ReplyVerdict.HEDGED)
            if probe.poison:
                sets.append("poison")
                m.answer_poison.add(verdict == ReplyVerdict.POISON)
                m.answer_poison_asserted.add(reply.poison_kind == "asserted")
                m.answer_poison_adopted.add(reply.poison_kind == "adopted")
                m.answer_poison_disclosed.add(verdict == ReplyVerdict.CURRENT and reply.disclosed)
        for name in sets:
            counts = m.verdicts.setdefault(name, {})
            counts[verdict] = counts.get(verdict, 0) + 1
    m.reader_tokens_in = stat(tokens)
    m.reader_usd = round(usd, 6) if usd is not None else None
    return m


def reply_keys(probes: Iterable[DoorProbeResult]) -> list[tuple[str, str]]:
    """Every ``(reader, reading)`` pair a build's replies carry, in reading order."""
    seen = {(r.reader, r.reading) for p in probes for r in p.replies}
    order = {name: i for i, name in enumerate(READINGS)}
    return sorted(seen, key=lambda k: (k[0] != HEADLINE_READER, k[0], order.get(k[1], 99)))


def score_answers(
    build: DoorBuild,
    world: RotWorld,
    *,
    strip_version: int = ANSWER_STRIP_VERSION,
    reply_scorer_version: int = REPLY_SCORER_VERSION,
    scorer_version: int = SCORER_VERSION,
    segmenter_version: int = DOOR_SEGMENTER_VERSION,
) -> None:
    """Score every reply of a build and fill its answer families; no call is made."""
    pattern = world_value_pattern(world)
    for probe in build.probes:
        for reply in probe.replies:
            score_reply(
                reply,
                probe,
                world,
                strip_version=strip_version,
                reply_scorer_version=reply_scorer_version,
                scorer_version=scorer_version,
                segmenter_version=segmenter_version,
                any_value=pattern,
            )
    build.answers = {
        answer_key(reader, reading): answer_metrics(build.probes, reader=reader, reading=reading)
        for reader, reading in reply_keys(build.probes)
    }


def answer_values(m: AnswerMetrics) -> dict[str, float | None]:
    """Each answer metric's headline number: a rate's value, a sample's median."""
    out: dict[str, float | None] = {}
    for name in ANSWER_METRIC_DIRECTIONS:
        if name == "reader_usd":
            out[name] = m.reader_usd
        elif name == "reader_tokens_in":
            out[name] = m.reader_tokens_in.median
        else:
            out[name] = getattr(m, name).value
    return out


def finalize_answers(memory: DoorMemoryResult) -> None:
    """Pool a memory's answer families over its builds, with each metric's range."""
    builds = [b for b in memory.builds if b.probes]
    probes = [p for b in builds for p in b.probes]
    memory.answers = {}
    memory.answer_ranges = {}
    for reader, reading in reply_keys(probes):
        key = answer_key(reader, reading)
        memory.answers[key] = answer_metrics(probes, reader=reader, reading=reading)
        per_build = [answer_values(b.answers[key]) for b in builds if key in b.answers]
        ranges: dict[str, MetricRange] = {}
        for name in ANSWER_METRIC_DIRECTIONS:
            nums = [x for v in per_build if (x := v[name]) is not None]
            ranges[name] = MetricRange(
                mean=statistics.fmean(nums) if nums else None,
                lo=min(nums) if nums else None,
                hi=max(nums) if nums else None,
                builds=len(nums),
            )
        memory.answer_ranges[key] = ranges

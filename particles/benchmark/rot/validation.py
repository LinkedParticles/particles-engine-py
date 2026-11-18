# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Blind validation of the door segmenter and the reply scorer.

The classifier was built on Particles' ranked hits; competitors
return JSON, triples, graph summaries and model-written paragraphs. Before
the freeze, on a development seed only, value-bearing segments are drawn from
at least two non-Particles memories and from Particles, stratified by class,
and labelled by someone who does not see which memory produced them. The bar
is registered here: per memory, at least 95 % agreement with the labels
overall, and at least 90 % recall on each of ``STALE`` and ``HISTORY``.

:func:`label_sample` writes two halves: the items a labeller sees (segment,
question, the slot's truth at the checkpoint; no memory, no scorer class) and
a key (item → memory and scorer class) the labeller must not see.
:func:`score_labels` joins them.
"""

from __future__ import annotations

import random

from pydantic import BaseModel, Field

from particles.benchmark.rot.door_schema import DoorReport
from particles.benchmark.rot.schema import HitClass

#: The registered bar.
AGREEMENT_BAR = 0.95
RECALL_BAR = 0.90
#: Classes whose recall is gated: the two RotBench's round-4 correction was about.
GATED_CLASSES = (HitClass.STALE, HitClass.HISTORY)


class LabelItem(BaseModel):
    """One segment as the labeller sees it."""

    id: str
    question: str
    checkpoint: int
    current: str | None
    superseded: list[str]
    poison: list[str]
    segment: str
    #: Filled by the labeller with one ``HitClass`` value.
    label: str | None = None


class LabelKey(BaseModel):
    """What the labeller must not see."""

    memory: str
    scorer_class: str


class MemoryAgreement(BaseModel):
    """One memory's agreement with the labels."""

    memory: str
    labelled: int = 0
    agreement: float | None = None
    recall: dict[str, float | None] = Field(default_factory=dict)
    passed: bool = False


class ValidationResult(BaseModel):
    """The bar, per memory, and whether the freeze may rest on it."""

    agreement_bar: float = AGREEMENT_BAR
    recall_bar: float = RECALL_BAR
    memories: list[MemoryAgreement] = Field(default_factory=list)
    passed: bool = False
    notes: list[str] = Field(default_factory=list)


def label_sample(
    report: DoorReport, *, per_memory: int = 150, min_per_class: int = 20, seed: int = 0
) -> tuple[list[LabelItem], dict[str, LabelKey]]:
    """Draw a stratified, shuffled, memory-blind sample of value-bearing segments.

    Per memory (headline and baseline rows; variants share a headline's door):
    up to ``min_per_class`` segments of every class first, then a random fill
    to ``per_memory``. Items from all memories are shuffled together and given
    opaque ids.
    """
    rng = random.Random(seed)
    items: list[tuple[LabelItem, LabelKey]] = []
    for memory in report.memories:
        if memory.role == "variant":
            continue
        pool: dict[str, list[LabelItem]] = {}
        for build in memory.builds:
            for probe in build.probes:
                for seg in probe.segments:
                    item = LabelItem(
                        id="",
                        question=probe.question,
                        checkpoint=probe.checkpoint,
                        current=probe.current,
                        superseded=probe.superseded,
                        poison=probe.poison,
                        segment=seg.text,
                    )
                    pool.setdefault(seg.hit_class.value, []).append(item)
        chosen: list[tuple[LabelItem, str]] = []
        rest: list[tuple[LabelItem, str]] = []
        for cls in sorted(pool):
            members = pool[cls][:]
            rng.shuffle(members)
            chosen += [(m, cls) for m in members[:min_per_class]]
            rest += [(m, cls) for m in members[min_per_class:]]
        rng.shuffle(rest)
        chosen += rest[: max(0, per_memory - len(chosen))]
        items += [(m, LabelKey(memory=memory.name, scorer_class=cls)) for m, cls in chosen]
    rng.shuffle(items)
    out: list[LabelItem] = []
    key: dict[str, LabelKey] = {}
    for i, (item, k) in enumerate(items):
        item.id = f"seg-{i:05d}"
        out.append(item)
        key[item.id] = k
    return out, key


def score_labels(labelled: list[LabelItem], key: dict[str, LabelKey]) -> ValidationResult:
    """Each memory's agreement and gated-class recall against the labels.

    Recall on a class is over the segments the labeller put in it. The freeze
    may rest on the result only when at least two non-Particles memories and
    Particles were labelled and every one of them passes.
    """
    by_memory: dict[str, list[tuple[str, str]]] = {}
    for item in labelled:
        if item.label is None or item.id not in key:
            continue
        k = key[item.id]
        by_memory.setdefault(k.memory, []).append((item.label, k.scorer_class))
    result = ValidationResult()
    for memory, pairs in sorted(by_memory.items()):
        agree = sum(1 for label, scored in pairs if label == scored)
        row = MemoryAgreement(memory=memory, labelled=len(pairs), agreement=agree / len(pairs))
        ok = row.agreement is not None and row.agreement >= AGREEMENT_BAR
        for cls in GATED_CLASSES:
            truth = [scored for label, scored in pairs if label == cls.value]
            recall = (sum(1 for s in truth if s == cls.value) / len(truth)) if truth else None
            row.recall[cls.value] = recall
            ok = ok and (recall is None or recall >= RECALL_BAR)
        row.passed = ok
        result.memories.append(row)
    competitors = [
        m for m in result.memories if m.memory not in ("particles", "raw-chunks", "recent-mentions")
    ]
    has_particles = any(m.memory == "particles" for m in result.memories)
    if len(competitors) < 2 or not has_particles:
        result.notes.append(
            "The bar needs at least two non-Particles memories and Particles labelled."
        )
    result.passed = (
        len(competitors) >= 2 and has_particles and all(m.passed for m in result.memories)
    )
    return result


# ---------------------------------------------------------------------------
# The reply scorer's validation
# ---------------------------------------------------------------------------

#: Verdicts whose recall is gated: the three a reader-dependent result turns on.
REPLY_GATED_VERDICTS = ("stale", "hedged", "poison")
REPLY_SAMPLE = 100
REPLY_MIN_PER_VERDICT = 15
REPLY_MIN_MEMORIES = 3


class ReplyLabelItem(BaseModel):
    """One reply as the labeller sees it: no memory, no reading, no verdict."""

    id: str
    question: str
    checkpoint: int
    negative: bool
    current: str | None
    superseded: list[str]
    poison: list[str]
    reply: str
    #: Filled by the labeller with one ``ReplyVerdict`` value.
    label: str | None = None


class ReplyLabelKey(BaseModel):
    """What the labeller must not see."""

    memory: str
    reader: str
    reading: str
    verdict: str


class ReaderAgreement(BaseModel):
    """One reader's agreement with the labels, over every memory it read."""

    reader: str
    labelled: int = 0
    memories: int = 0
    agreement: float | None = None
    recall: dict[str, float | None] = Field(default_factory=dict)
    passed: bool = False


class ReplyValidationResult(BaseModel):
    """The reply scorer's bar, per reader, and whether the freeze may rest on it."""

    agreement_bar: float = AGREEMENT_BAR
    recall_bar: float = RECALL_BAR
    readers: list[ReaderAgreement] = Field(default_factory=list)
    passed: bool = False
    notes: list[str] = Field(default_factory=list)


def reply_label_sample(
    report: DoorReport,
    *,
    per_reader: int = REPLY_SAMPLE,
    min_per_verdict: int = REPLY_MIN_PER_VERDICT,
    seed: int = 0,
) -> tuple[list[ReplyLabelItem], dict[str, ReplyLabelKey]]:
    """Draw a stratified, shuffled, memory-blind sample of scored replies.

    Per reader: up to ``min_per_verdict`` replies of every verdict first, then
    a random fill to ``per_reader``, from every row and reading. Items from
    all readers are shuffled together and given opaque ids.
    """
    rng = random.Random(seed)
    pools: dict[str, dict[str, list[tuple[ReplyLabelItem, ReplyLabelKey]]]] = {}
    for memory in report.memories:
        for build in memory.builds:
            for probe in build.probes:
                for r in probe.replies:
                    if r.verdict is None or r.reply is None:
                        continue
                    item = ReplyLabelItem(
                        id="",
                        question=probe.question,
                        checkpoint=probe.checkpoint,
                        negative=probe.negative,
                        current=probe.current,
                        superseded=probe.superseded,
                        poison=probe.poison,
                        reply=r.reply,
                    )
                    secret = ReplyLabelKey(
                        memory=memory.name, reader=r.reader, reading=r.reading, verdict=r.verdict
                    )
                    pools.setdefault(r.reader, {}).setdefault(r.verdict, []).append((item, secret))
    items: list[tuple[ReplyLabelItem, ReplyLabelKey]] = []
    for reader in sorted(pools):
        chosen: list[tuple[ReplyLabelItem, ReplyLabelKey]] = []
        rest: list[tuple[ReplyLabelItem, ReplyLabelKey]] = []
        for verdict in sorted(pools[reader]):
            members = pools[reader][verdict][:]
            rng.shuffle(members)
            chosen += members[:min_per_verdict]
            rest += members[min_per_verdict:]
        rng.shuffle(rest)
        chosen += rest[: max(0, per_reader - len(chosen))]
        items += chosen
    rng.shuffle(items)
    out: list[ReplyLabelItem] = []
    key: dict[str, ReplyLabelKey] = {}
    for i, (item, k) in enumerate(items):
        item.id = f"reply-{i:05d}"
        out.append(item)
        key[item.id] = k
    return out, key


def score_reply_labels(
    labelled: list[ReplyLabelItem], key: dict[str, ReplyLabelKey]
) -> ReplyValidationResult:
    """Each reader's agreement and gated-verdict recall against the labels.

    The bar: per reader, at least 95 % agreement overall and at
    least 90 % recall on each of ``stale``, ``hedged`` and ``poison``, over
    replies from at least three memories.
    """
    by_reader: dict[str, list[tuple[str, ReplyLabelKey]]] = {}
    for item in labelled:
        if item.label is None or item.id not in key:
            continue
        k = key[item.id]
        by_reader.setdefault(k.reader, []).append((item.label, k))
    result = ReplyValidationResult()
    for reader, pairs in sorted(by_reader.items()):
        agree = sum(1 for label, k in pairs if label == k.verdict)
        row = ReaderAgreement(
            reader=reader,
            labelled=len(pairs),
            memories=len({k.memory for _, k in pairs}),
            agreement=agree / len(pairs),
        )
        ok = row.agreement is not None and row.agreement >= AGREEMENT_BAR
        for verdict in REPLY_GATED_VERDICTS:
            truth = [k.verdict for label, k in pairs if label == verdict]
            recall = (sum(1 for v in truth if v == verdict) / len(truth)) if truth else None
            row.recall[verdict] = recall
            ok = ok and (recall is None or recall >= RECALL_BAR)
        if row.memories < REPLY_MIN_MEMORIES:
            result.notes.append(
                f"{reader}: replies from {row.memories} memories; the bar needs at least "
                f"{REPLY_MIN_MEMORIES}."
            )
            ok = False
        row.passed = ok
        result.readers.append(row)
    result.passed = bool(result.readers) and all(r.passed for r in result.readers)
    return result

# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Merge the duplicates a re-read wrote within one append-only entry.

Before an ``APPEND_ONLY`` entry's later snapshots re-read text the
store had already extracted, and the extractor wrote the passage again in other
words. Each such pair is two readings of one passage, not two observations,
so the resolution is the exact-duplicate merge's: one copy stands and the
other is superseded.
Because the wording differs, the pair is decided by a model, never by a key.

* **The pairs** (:func:`in_population`, pure): proposed by the
  finder, every ``SOURCE`` ref on both claims naming one ``APPEND_ONLY`` entry,
  disjoint snapshot sets, one ``stance:holder``, and different §6.10 keys
  (a key-equal pair is ``links dedup``'s).
* **The questions** (:func:`plan_questions`, pure): pairs grouped into
  components, the first reading elected survivor, and each other member asked
  against the survivor directly, unless it shares a snapshot with the survivor
  or carries a Subject the survivor lacks.
* **The readings**: the duplicate judge, then the update ladder for a
  ``SAME``; both recorded in the probe ledger, so ``--apply`` asks no model.
* **The write rule** (:func:`decide_merge`, pure): ``SAME``, no update reading
  saying ``CHANGES``, no contradiction on record, not restored by an unmerge,
  and the pair's cosine stratum admitted.
* **The write** (:func:`apply_reread`): shape with
  ``created_by = REREAD_DUPLICATE``, so ``links unmerge`` reverts it.

Admission (:func:`score_labels`, pure): per cosine stratum, a 95 % Wilson lower
bound of at least 0.95 on the owner's labels of the judge's ``SAME`` verdicts.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import math
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.conflict_resolution import admits_update
from particles.core.duplicate_key import content_hash as norm_content_hash
from particles.core.probe_verdict import ProbeKind, VerdictKey, verdict_key
from particles.core.schema import (
    JudgeVerdictKind,
    Mutability,
    Particle,
    ProvenanceRefType,
    RelationCreatedBy,
    RelationType,
    SuggestMode,
)
from particles.core.stance import stance_holder
from particles.core.status import Status, StatusReason
from particles.db import write_transaction
from particles.embeddings import cosine_similarity
from particles.ingest.duplicate_judge import (
    JUDGE_TOKENS_PER_PAIR,
    duplicate_judge_stamp,
    judge_max_tokens,
    judge_prompt,
)
from particles.llm.breaker import PROBE_MAX_TOKENS, llm_circuit_open
from particles.operations.probe_ledger import model_label, recall, remember

log = logging.getLogger(__name__)

__all__ = [
    "BASIS",
    "HIGH_STRATUM_FLOOR",
    "Block",
    "ClaimFacts",
    "LabelScore",
    "Question",
    "RereadNotAdmitted",
    "RereadPlan",
    "RereadReport",
    "UpdateRecord",
    "apply_reread",
    "census_reread",
    "decide_merge",
    "in_population",
    "judge_reread",
    "plan_questions",
    "score_labels",
    "stratum_of",
    "wilson_lower_bound",
    "write_sheet",
]

#: The ``basis`` a re-read merge's ``DUPLICATES_MERGED`` event carries, which
#: tells ``links unmerge`` which edge tag the merge wrote.
BASIS = "reread_judged"

#: The cosine at and above which a (survivor, copy) pair is in the high stratum.
HIGH_STRATUM_FLOOR = 0.97

#: The admission bar, per stratum (bar).
_WILSON_Z = 1.96
_ADMISSION_FLOOR = 0.95
_ADMISSION_MIN_LABELS = 100

#: Probe kinds whose positive verdict on a pair is a contradiction on record.
_CONTRADICTION_KINDS = frozenset(
    {
        ProbeKind.CONTRADICTION.value,
        ProbeKind.SECOND_READING.value,
        ProbeKind.RECONCILE_CONTRADICTION.value,
    }
)

Stratum = Literal["high", "low"]
Admitted = Literal["none", "high", "all"]


class RereadNotAdmitted(Exception):
    """``--apply`` was asked while ``links_suggest.reread.admitted`` is ``none``."""

    def __init__(self) -> None:
        super().__init__(
            "The duplicate judge is not admitted on re-read pairs. Label the "
            "`links dedup --reread --sheet` verdicts, check them with `--labels`, "
            "and set `links_suggest.reread.admitted` to `high` or `all` once a "
            "stratum clears the bar. Nothing was written."
        )


# ---------------------------------------------------------------------------
# Pure core
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimFacts:
    """What the pair gates and the election read about one ACTIVE claim."""

    id: str
    content: str
    entry_ids: frozenset[str]
    """Entries named by the claim's ``SOURCE`` refs."""
    snapshot_ids: frozenset[str]
    first_read: datetime | None
    """Capture time of the earliest snapshot the claim cites."""
    asserted_at: datetime
    subject_ids: frozenset[str]
    holder: str | None
    sources_only: bool
    """Every provenance ref is a ``SOURCE`` ref naming a snapshot."""


def in_population(a: ClaimFacts, b: ClaimFacts, *, append_only: Mapping[str, bool]) -> bool:
    """Whether a finder pair is two readings of one append-only passage."""
    if not (a.sources_only and b.sources_only):
        return False
    entries = a.entry_ids | b.entry_ids
    if len(entries) != 1:
        return False
    (entry,) = entries
    if not append_only.get(entry, False):
        return False
    if a.snapshot_ids & b.snapshot_ids:
        return False
    if a.holder != b.holder:
        return False
    return norm_content_hash(a.content) != norm_content_hash(b.content)


def _election_key(facts: ClaimFacts) -> tuple[datetime, datetime, str]:
    """The first reading: earliest snapshot, then earliest assertion, then id (§1)."""
    floor = datetime.min.replace(tzinfo=UTC)
    first = _aware(facts.first_read) if facts.first_read is not None else floor
    return (first, _aware(facts.asserted_at), facts.id)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def stratum_of(cosine: float | None) -> Stratum:
    """The admission stratum of a (survivor, copy) pair."""
    return "high" if cosine is not None and cosine >= HIGH_STRATUM_FLOOR else "low"


class Skip(StrEnum):
    """Why a component member is not asked."""

    SHARED_SNAPSHOT = "shared_snapshot"
    EXTRA_SUBJECT = "extra_subject"


@dataclass(frozen=True)
class Question:
    """One (survivor, copy) pair put to the readings."""

    survivor_id: str
    copy_id: str
    entry_id: str
    cosine: float | None
    stratum: Stratum


@dataclass(frozen=True)
class RereadPlan:
    """The questions a run asks, and what it leaves alone."""

    pairs: tuple[tuple[str, str], ...]
    components: int
    questions: tuple[Question, ...]
    skipped: Mapping[Skip, int]


def plan_questions(
    pairs: Sequence[tuple[str, str]],
    facts: Mapping[str, ClaimFacts],
    cosine: Callable[[str, str], float | None],
) -> RereadPlan:
    """Components, survivors and the survivor-anchored questions.

    ``pairs`` are in the population already. Each component's survivor is its
    first reading; every other member is asked against the survivor directly,
    whether or not the finder paired the two, unless it shares a snapshot with
    the survivor or carries a Subject the survivor lacks.
    """
    adjacency: dict[str, set[str]] = {}
    for a, b in pairs:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    questions: list[Question] = []
    skipped = {Skip.SHARED_SNAPSHOT: 0, Skip.EXTRA_SUBJECT: 0}
    seen: set[str] = set()
    components = 0
    for start in sorted(adjacency):
        if start in seen:
            continue
        component: set[str] = set()
        queue: deque[str] = deque([start])
        while queue:
            node = queue.popleft()
            if node in component:
                continue
            component.add(node)
            queue.extend(adjacency[node] - component)
        seen |= component
        components += 1
        members = sorted((facts[i] for i in component), key=_election_key)
        survivor, *rest = members
        (entry,) = survivor.entry_ids
        for member in rest:
            if member.snapshot_ids & survivor.snapshot_ids:
                skipped[Skip.SHARED_SNAPSHOT] += 1
                continue
            if not member.subject_ids <= survivor.subject_ids:
                skipped[Skip.EXTRA_SUBJECT] += 1
                continue
            sim = cosine(survivor.id, member.id)
            questions.append(Question(survivor.id, member.id, entry, sim, stratum_of(sim)))
    return RereadPlan(
        pairs=tuple(pairs), components=components, questions=tuple(questions), skipped=skipped
    )


@dataclass(frozen=True)
class UpdateRecord:
    """The update ladder's recorded reading of a (survivor, copy) pair."""

    changes: bool
    """The settled verdict was ``CHANGES`` (two ``CHANGES`` readings)."""
    second: bool
    """The second reading settled it, so the first reading said ``CHANGES``."""

    @property
    def any_changes(self) -> bool:
        return self.changes or self.second


class Block(StrEnum):
    """Why a question does not merge (§4, §5)."""

    UNJUDGED = "unjudged"
    DIFFERENT = "different"
    UPDATE_UNREAD = "update_unread"
    UPDATE_CHANGES = "update_changes"
    CONTRADICTION = "contradiction"
    RESTORED = "restored"
    NOT_ADMITTED = "not_admitted"


#: The blocks a guard raises on a judged ``SAME`` (the sheet marks them).
GUARD_BLOCKS = frozenset({Block.UPDATE_CHANGES, Block.CONTRADICTION})


def stratum_admitted(stratum: Stratum, admitted: Admitted) -> bool:
    return admitted == "all" or (admitted == "high" and stratum == "high")


def decide_merge(
    *,
    same: bool | None,
    update: UpdateRecord | None,
    contradiction: bool,
    restored: bool,
    stratum: Stratum,
    admitted: Admitted,
) -> Block | None:
    """``None`` when the copy merges into its survivor, else the first block (§2).

    Each reading is the answer to its own question; this rule only
    combines them, in the direction that keeps a changed value visible.
    """
    if same is None:
        return Block.UNJUDGED
    if not same:
        return Block.DIFFERENT
    if update is None:
        return Block.UPDATE_UNREAD
    if update.any_changes:
        return Block.UPDATE_CHANGES
    if contradiction:
        return Block.CONTRADICTION
    if restored:
        return Block.RESTORED
    if not stratum_admitted(stratum, admitted):
        return Block.NOT_ADMITTED
    return None


def wilson_lower_bound(correct: int, total: int, z: float = _WILSON_Z) -> float:
    """The Wilson score interval's lower bound on a proportion."""
    if total <= 0:
        return 0.0
    p = correct / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (centre - margin) / denom


@dataclass(frozen=True)
class LabelScore:
    """One stratum's labels against the admission bar."""

    stratum: Stratum
    labelled: int
    correct: int
    lower_bound: float
    passes: bool
    rule_labelled: int
    """Labelled verdicts the write rule would have merged (no guard blocked)."""
    rule_correct: int


def score_labels(rows: Iterable[Mapping[str, str]]) -> list[LabelScore]:
    """Score a labelled sheet per stratum; unlabelled rows are ignored.

    A label of ``same`` (or ``y`` / ``yes``) is a correct ``SAME`` verdict;
    ``different`` (or ``n`` / ``no``) is a false one. Every labelled row counts
    against the judge, including those a guard blocked, because the bar
    measures the judge; the write rule's own precision is reported beside it.
    """
    tallies: dict[str, list[int]] = {"high": [0, 0, 0, 0], "low": [0, 0, 0, 0]}
    for row in rows:
        label = (row.get("label") or "").strip().lower()
        if label in {"same", "y", "yes"}:
            correct = 1
        elif label in {"different", "n", "no"}:
            correct = 0
        else:
            continue
        stratum = (row.get("stratum") or "").strip()
        if stratum not in tallies:
            continue
        t = tallies[stratum]
        t[0] += 1
        t[1] += correct
        if (row.get("guard") or "").strip() == "":
            t[2] += 1
            t[3] += correct
    scores: list[LabelScore] = []
    for name in ("high", "low"):
        labelled, correct, rule_n, rule_c = tallies[name]
        bound = wilson_lower_bound(correct, labelled)
        scores.append(
            LabelScore(
                stratum=name,
                labelled=labelled,
                correct=correct,
                lower_bound=bound,
                passes=labelled >= _ADMISSION_MIN_LABELS and bound >= _ADMISSION_FLOOR,
                rule_labelled=rule_n,
                rule_correct=rule_c,
            )
        )
    return scores


def _sheet_order(question: Question) -> str:
    """A stable shuffle, so labelling the sheet from the top labels a random sample."""
    return hashlib.sha256(f"{question.survivor_id}:{question.copy_id}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------


class RereadItem(BaseModel):
    """One question and what the ledger holds for it."""

    survivor_id: str
    copy_id: str
    entry_id: str
    cosine: float | None = None
    stratum: Stratum
    verdict: Literal["same", "different"] | None = None
    update: Literal["clear", "changes", "unread"] | None = None
    update_rung: str | None = None
    """The ``name@version`` of the update rung that settled the reading."""
    block: Block | None = None
    merged: bool = False


class RereadEstimate(BaseModel):
    """What the unanswered questions would cost to ask (token arithmetic, not a run)."""

    judge_questions: int = 0
    judge_calls: int = 0
    update_readings: int = 0
    """First update readings, an upper bound: every unjudged pair read as ``SAME``."""
    judge_model: str = ""
    update_model: str = ""
    second_model: str = ""
    input_tokens: int = 0
    output_tokens_expected: int = 0
    output_tokens_ceiling: int = 0
    usd_expected: float | None = None
    usd_ceiling: float | None = None
    second_reading_usd_ceiling_each: float | None = None
    """A second reading runs only after a first ``CHANGES``; each costs at most this."""


class RereadReport(BaseModel):
    """What ``links dedup --reread`` found, asked or merged."""

    run_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    mode: Literal["census", "judge", "apply"] = "census"
    run_id: str | None = None
    finder_pairs: int = 0
    population_pairs: int = 0
    components: int = 0
    questions: int = 0
    skipped_shared_snapshot: int = 0
    skipped_extra_subject: int = 0
    resolvable_pairs: int = 0
    by_stratum: dict[str, int] = Field(default_factory=dict)
    judged: int = 0
    same: int = 0
    different: int = 0
    update_read: int = 0
    update_changes: int = 0
    admitted: Admitted = "none"
    would_merge: dict[str, int] = Field(default_factory=dict)
    """Copies the write rule allows, per stratum, before admission."""
    blocked: dict[str, int] = Field(default_factory=dict)
    estimate: RereadEstimate | None = None
    asked_judge: int = 0
    asked_update: int = 0
    unasked: int = 0
    merged_groups: int = 0
    merged_copies: int = 0
    links_created: int = 0
    deferred_groups: int = 0
    warnings: list[str] = Field(default_factory=list)
    items: list[RereadItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Gather
# ---------------------------------------------------------------------------


@dataclass
class _Gathered:
    plan: RereadPlan
    particles: dict[str, Particle]
    finder_pairs: int


async def _gather(session: AsyncSession) -> _Gathered:
    """The finder's pairs, the claims' facts, and the plan (reads only)."""
    from particles.corpus.store import CorpusEntryRow, SnapshotRow
    from particles.operations.links_suggest import suggest_co_evidential
    from particles.store.particle_store import get_embeddings_by_ids, get_particles_by_ids

    suggest = await suggest_co_evidential(session, mode=SuggestMode.REPORT)
    similarity: dict[frozenset[str], float] = {}
    for cluster in suggest.clusters:
        for c in cluster.candidates:
            key = frozenset((c.particle_a, c.particle_b))
            similarity[key] = max(similarity.get(key, 0.0), c.similarity)

    ids = sorted({i for key in similarity for i in key})
    particles = {
        pid: p
        for pid, p in (await get_particles_by_ids(session, ids)).items()
        if p.status is Status.ACTIVE
    }
    entry_ids: set[str] = set()
    snapshot_ids: set[str] = set()
    for p in particles.values():
        for ref in p.provenance:
            if ref.type is ProvenanceRefType.SOURCE:
                entry_ids.add(ref.corpus_entry_id)
                if ref.snapshot_id:
                    snapshot_ids.add(ref.snapshot_id)
    append_only: dict[str, bool] = {}
    for start in range(0, len(entry_ids), 500):
        chunk = sorted(entry_ids)[start : start + 500]
        rows = await session.execute(
            select(CorpusEntryRow.entry_id, CorpusEntryRow.mutability).where(
                CorpusEntryRow.entry_id.in_(chunk)
            )
        )
        for entry_id, mutability in rows:
            append_only[entry_id] = mutability == Mutability.APPEND_ONLY.value
    captured: dict[str, datetime] = {}
    for start in range(0, len(snapshot_ids), 500):
        chunk = sorted(snapshot_ids)[start : start + 500]
        rows2 = await session.execute(
            select(SnapshotRow.snapshot_id, SnapshotRow.captured_at).where(
                SnapshotRow.snapshot_id.in_(chunk)
            )
        )
        for snapshot_id, captured_at in rows2:
            captured[snapshot_id] = captured_at

    facts = {pid: _facts(p, captured) for pid, p in particles.items()}
    population: list[tuple[str, str]] = []
    for key in sorted(similarity, key=sorted):
        a, b = sorted(key)
        if a in facts and b in facts and in_population(facts[a], facts[b], append_only=append_only):
            population.append((a, b))

    members = sorted({i for pair in population for i in pair})
    vectors = await get_embeddings_by_ids(session, members)

    def cosine(x: str, y: str) -> float | None:
        known = similarity.get(frozenset((x, y)))
        if known is not None:
            return known
        vx, vy = vectors.get(x), vectors.get(y)
        if vx is None or vy is None:
            return None
        return float(cosine_similarity(np.asarray(vx), np.asarray(vy)))

    plan = plan_questions(population, facts, cosine)
    return _Gathered(plan=plan, particles=particles, finder_pairs=len(similarity))


def _facts(p: Particle, captured: Mapping[str, datetime]) -> ClaimFacts:
    refs = list(p.provenance)
    sources = [r for r in refs if r.type is ProvenanceRefType.SOURCE]
    snaps = frozenset(r.snapshot_id for r in sources if r.snapshot_id)
    times = [captured[s] for s in snaps if s in captured]
    return ClaimFacts(
        id=p.id,
        content=p.content,
        entry_ids=frozenset(r.corpus_entry_id for r in sources),
        snapshot_ids=snaps,
        first_read=min((_aware(t) for t in times), default=None),
        asserted_at=p.asserted_at,
        subject_ids=frozenset(p.subject_ids),
        holder=stance_holder(p),
        sources_only=bool(refs)
        and len(sources) == len(refs)
        and all(r.snapshot_id for r in sources)
        and len(times) == len(snaps),
    )


# ---------------------------------------------------------------------------
# Assess: what the ledger holds for each question
# ---------------------------------------------------------------------------


def _dup_key(survivor: Particle, copy: Particle) -> VerdictKey:
    return verdict_key(ProbeKind.DUPLICATE, survivor.content, copy.content)


def _update_key(survivor: Particle, copy: Particle) -> VerdictKey:
    # Directional: the survivor is the earlier reading.
    return verdict_key(ProbeKind.UPDATE_SLOT, survivor.content, copy.content)


def _update_record(stamped: tuple[bool, str | None]) -> UpdateRecord:
    from particles.ingest.pipeline import UPDATE_SECOND_RUNG

    verdict, rung = stamped
    second = rung is not None and rung.split("@", 1)[0] == UPDATE_SECOND_RUNG
    return UpdateRecord(changes=verdict, second=second)


async def _contradicted_pairs(
    session: AsyncSession, questions: Sequence[Question], particles: Mapping[str, Particle]
) -> set[tuple[str, str]]:
    """(survivor, copy) pairs with a contradiction on record: read, never asked."""
    from particles.store.particle_store import get_inconsistency_particles
    from particles.store.probe_verdict_store import verdicts_for_pair

    named: set[frozenset[str]] = set()
    for record in await get_inconsistency_particles(session):
        ids = [r.corpus_entry_id for r in record.provenance if r.type is ProvenanceRefType.PARTICLE]
        named |= {frozenset((x, y)) for i, x in enumerate(ids) for y in ids[i + 1 :] if x != y}
    out: set[tuple[str, str]] = set()
    for q in questions:
        if frozenset((q.survivor_id, q.copy_id)) in named:
            out.add((q.survivor_id, q.copy_id))
            continue
        key = verdict_key(
            ProbeKind.CONTRADICTION,
            particles[q.survivor_id].content,
            particles[q.copy_id].content,
        )
        rows = await verdicts_for_pair(session, key[0], key[1])
        if any(r.probe_kind in _CONTRADICTION_KINDS and r.verdict for r in rows):
            out.add((q.survivor_id, q.copy_id))
    return out


async def _restored_copies(session: AsyncSession) -> set[str]:
    """Copies a ``links unmerge`` restored, which no apply merges again."""
    from particles.store.event_store import OperatorEventType, list_events

    out: set[str] = set()
    for event in await list_events(
        session, event_type=OperatorEventType.DUPLICATES_UNMERGED, limit=100_000
    ):
        restored = (event.payload or {}).get("restored")
        if isinstance(restored, list):
            out.update(str(r) for r in restored)
    return out


async def _assess(session: AsyncSession, gathered: _Gathered, report: RereadReport) -> None:
    """Fill ``report`` with each question's ledger state and the rule's decision."""
    from particles.ingest.pipeline import update_prompt_hash
    from particles.store.probe_verdict_store import lookup_stamped_verdicts

    plan, particles = gathered.plan, gathered.particles
    questions = list(plan.questions)
    report.finder_pairs = gathered.finder_pairs
    report.population_pairs = len(plan.pairs)
    report.components = plan.components
    report.questions = len(questions)
    report.skipped_shared_snapshot = plan.skipped[Skip.SHARED_SNAPSHOT]
    report.skipped_extra_subject = plan.skipped[Skip.EXTRA_SUBJECT]
    asked = {q.copy_id for q in questions}
    report.resolvable_pairs = sum(1 for a, b in plan.pairs if a in asked or b in asked)
    admitted = get_config().links_suggest.reread.admitted
    report.admitted = admitted

    pairs = [(particles[q.survivor_id], particles[q.copy_id]) for q in questions]
    dup = await recall(
        session,
        ProbeKind.DUPLICATE,
        duplicate_judge_stamp().version,
        [_dup_key(s, c) for s, c in pairs],
    )
    upd = await lookup_stamped_verdicts(
        session, ProbeKind.UPDATE_SLOT, update_prompt_hash(), [_update_key(s, c) for s, c in pairs]
    )
    contradicted = await _contradicted_pairs(session, questions, particles)
    restored = await _restored_copies(session)

    report.items = []
    report.by_stratum = {"high": 0, "low": 0}
    report.would_merge = {"high": 0, "low": 0}
    blocked: dict[str, int] = {}
    for q, (s, c) in zip(questions, pairs, strict=True):
        report.by_stratum[q.stratum] += 1
        same = dup.get(_dup_key(s, c))
        stamped = upd.get(_update_key(s, c))
        update = _update_record(stamped) if stamped is not None else None
        if same is not None:
            report.judged += 1
            report.same += int(same)
            report.different += int(not same)
        if update is not None:
            report.update_read += 1
            report.update_changes += int(update.any_changes)
        block = decide_merge(
            same=same,
            update=update,
            contradiction=(q.survivor_id, q.copy_id) in contradicted,
            restored=q.copy_id in restored,
            stratum=q.stratum,
            admitted=admitted,
        )
        if block is None or block is Block.NOT_ADMITTED:
            report.would_merge[q.stratum] += 1
        if block is not None:
            blocked[block.value] = blocked.get(block.value, 0) + 1
        report.items.append(
            RereadItem(
                survivor_id=q.survivor_id,
                copy_id=q.copy_id,
                entry_id=q.entry_id,
                cosine=q.cosine,
                stratum=q.stratum,
                verdict=None if same is None else ("same" if same else "different"),
                update=(
                    None
                    if same is not True
                    else (
                        "unread"
                        if update is None
                        else ("changes" if update.any_changes else "clear")
                    )
                ),
                update_rung=stamped[1] if stamped is not None else None,
                block=block,
            )
        )
    report.blocked = blocked


# ---------------------------------------------------------------------------
# Estimate
# ---------------------------------------------------------------------------


def _chars_per_token(purpose: str) -> float:
    cfg = get_config()
    sel = cfg.llm.for_purpose(purpose)
    table = cfg.benchmark_memory.chars_per_token_by_model
    for key in (f"{sel.provider}:{sel.model}", sel.model):
        if key in table:
            return table[key]
    return cfg.benchmark_memory.chars_per_token


def _usd(purpose: str, input_tokens: int, output_tokens: int) -> float | None:
    llm = get_config().llm
    price = llm.price_for(llm.for_purpose(purpose))
    if price is None:
        return None
    return (input_tokens * price.input + output_tokens * price.output) / 1_000_000


def _estimate(report: RereadReport, particles: Mapping[str, Particle]) -> RereadEstimate:
    """The cost of asking every question the ledger cannot answer (Acceptance 2)."""
    from particles.ingest.pipeline import _update_prompt

    cfg = get_config()
    batch = cfg.links_suggest.reread.judge_batch
    unjudged = [i for i in report.items if i.verdict is None][: cfg.links_suggest.reread.max_judged]
    reading = [i for i in report.items if i.verdict is None or i.update == "unread"]
    est = RereadEstimate(
        judge_questions=len(unjudged),
        judge_calls=math.ceil(len(unjudged) / batch) if unjudged else 0,
        update_readings=len(reading),
        judge_model=cfg.llm.for_purpose("semantic_lint").model,
        update_model=cfg.llm.for_purpose("semantic_lint").model,
        second_model=cfg.llm.for_purpose("verification").model,
    )
    cpt = _chars_per_token("semantic_lint")
    judge_in = judge_out = judge_ceiling = 0
    for start in range(0, len(unjudged), batch):
        chunk = unjudged[start : start + batch]
        pairs = [(particles[i.survivor_id].content, particles[i.copy_id].content) for i in chunk]
        judge_in += math.ceil(len(judge_prompt(pairs)) / cpt)
        judge_out += 64 + JUDGE_TOKENS_PER_PAIR * len(chunk)
        judge_ceiling += judge_max_tokens(len(chunk))
    update_in = sum(
        math.ceil(
            sum(
                map(
                    len,
                    _update_prompt(particles[i.survivor_id].content, particles[i.copy_id].content),
                )
            )
            / cpt
        )
        for i in reading
    )
    update_out = 64 * len(reading)
    update_ceiling = PROBE_MAX_TOKENS * len(reading)
    est.input_tokens = judge_in + update_in
    est.output_tokens_expected = judge_out + update_out
    est.output_tokens_ceiling = judge_ceiling + update_ceiling
    est.usd_expected = _usd("semantic_lint", est.input_tokens, est.output_tokens_expected)
    est.usd_ceiling = _usd("semantic_lint", est.input_tokens, est.output_tokens_ceiling)
    if reading:
        per_in = math.ceil(update_in / len(reading))
        budget = max(cfg.audit.verify_max_tokens, cfg.audit.verify_retry_max_tokens)
        est.second_reading_usd_ceiling_each = _usd("verification", per_in, budget)
    return est


# ---------------------------------------------------------------------------
# The three steps
# ---------------------------------------------------------------------------


async def census_reread(session: AsyncSession) -> RereadReport:
    """The population, the questions, the ledger's answers and the cost of the rest. Reads only."""
    gathered = await _gather(session)
    report = RereadReport(mode="census")
    await _assess(session, gathered, report)
    report.estimate = _estimate(report, gathered.particles)
    return report


async def _judge_pairs(pairs: Sequence[tuple[str, str]]) -> list[JudgeVerdictKind | None]:
    """The judge over one batch. The test seam."""
    # Deferred import: a test seam. Tests patch
    # ``particles.operations.reread_merge._judge_pairs`` or the breaker's
    # ``_llm_call``; tests/AGENTS.md § Mocking strategy.
    from particles.operations.links_suggest import _judge_pairs as judge

    return await judge(pairs)


async def _read_update(earlier: str, later: str) -> Any:
    """The update ladder's reading of one pair. The test seam."""
    # Deferred import: a test seam, as above.
    from particles.ingest.pipeline import read_update_slot

    return await read_update_slot(earlier, later)


async def judge_reread(
    session: AsyncSession,
    *,
    confirm: Callable[[RereadReport], bool] | None = None,
    progress: Callable[[str], None] | None = None,
) -> RereadReport:
    """Ask the questions the ledger cannot answer; write ledger rows and nothing else (§4).

    The duplicate judge first, highest cosine first and capped by
    ``links_suggest.reread.max_judged``; then the update ladder for each pair
    the judge (now or before) answered ``SAME``. Each batch is committed as it
    lands, so a run that stops keeps what it paid for. The breaker opening
    stops the run with a warning. ``confirm`` sees the census, estimate
    included, before any call; ``False`` returns it with nothing asked.
    """
    from particles.ingest.pipeline import update_prompt_hash
    from particles.store.probe_verdict_store import record_stamped_verdicts

    cfg = get_config().links_suggest.reread
    gathered = await _gather(session)
    report = RereadReport(mode="judge")
    await _assess(session, gathered, report)
    report.estimate = _estimate(report, gathered.particles)
    if confirm is not None and not confirm(report):
        report.warnings.append("Not confirmed; nothing was asked.")
        return report
    particles = gathered.particles
    by_copy = {i.copy_id: i for i in report.items}

    unjudged = sorted(
        (i for i in report.items if i.verdict is None),
        key=lambda i: (-(i.cosine or 0.0), i.copy_id),
    )
    ask = unjudged[: cfg.max_judged]
    report.unasked = len(unjudged) - len(ask)
    if report.unasked:
        report.warnings.append(
            f"links_suggest.reread.max_judged={cfg.max_judged}: {report.unasked} question(s) "
            "not asked. Re-run to continue."
        )
    await session.commit()
    stamp = duplicate_judge_stamp()
    for start in range(0, len(ask), cfg.judge_batch):
        if llm_circuit_open():
            report.warnings.append("The LLM breaker is open; the run stopped early.")
            break
        chunk = ask[start : start + cfg.judge_batch]
        texts = [(particles[i.survivor_id].content, particles[i.copy_id].content) for i in chunk]
        answers = await _judge_pairs(texts)
        fresh: dict[VerdictKey, bool] = {}
        for item, answer in zip(chunk, answers, strict=False):
            if answer is JudgeVerdictKind.PARAPHRASE:
                item.verdict = "same"
            elif answer is JudgeVerdictKind.DISTINCT:
                item.verdict = "different"
            else:
                continue
            fresh[_dup_key(particles[item.survivor_id], particles[item.copy_id])] = (
                item.verdict == "same"
            )
        report.asked_judge += len(chunk)
        await remember(
            session, ProbeKind.DUPLICATE, stamp.version, fresh, purpose="semantic_lint", rung=stamp
        )
        if progress is not None:
            progress(f"judged {report.asked_judge}/{len(ask)}")

    version = update_prompt_hash()
    for item in by_copy.values():
        if item.verdict != "same" or item.update not in (None, "unread"):
            continue
        if llm_circuit_open():
            report.warnings.append("The LLM breaker is open; update readings stopped early.")
            break
        survivor, copy = particles[item.survivor_id], particles[item.copy_id]
        reading = await _read_update(survivor.content, copy.content)
        report.asked_update += 1
        if reading.verdict is None:
            continue
        async with write_transaction(session):
            await record_stamped_verdicts(
                session,
                ProbeKind.UPDATE_SLOT,
                version,
                {_update_key(survivor, copy): (admits_update(reading.verdict), reading.decided_by)},
                model=model_label(reading.purpose),
            )
        if progress is not None and report.asked_update % 50 == 0:
            progress(f"update readings {report.asked_update}")

    final = RereadReport(
        mode="judge",
        asked_judge=report.asked_judge,
        asked_update=report.asked_update,
        unasked=report.unasked,
        warnings=report.warnings,
    )
    await _assess(session, gathered, final)
    return final


async def apply_reread(session: AsyncSession, *, actor: str = "links-dedup-reread") -> RereadReport:
    """Merge the copies the recorded verdicts allow. Asks no model.

    Raises:
        AutoMergeDisabled: ``links_suggest.auto_merge.enabled`` is false.
        RereadNotAdmitted: ``links_suggest.reread.admitted`` is ``none``.
    """
    from particles.operations.links_suggest import AutoMergeDisabled
    from particles.store.event_store import EventRefKind, OperatorEventType, record_event
    from particles.store.particle_store import update_particle_status
    from particles.store.relation_store import create_relation, get_co_evidential_group

    cfg = get_config().links_suggest
    if not cfg.auto_merge.enabled:
        raise AutoMergeDisabled
    if cfg.reread.admitted == "none":
        raise RereadNotAdmitted

    gathered = await _gather(session)
    report = RereadReport(mode="apply", run_id=str(uuid.uuid4()))
    await _assess(session, gathered, report)
    particles = gathered.particles

    groups: dict[str, list[RereadItem]] = {}
    for item in report.items:
        if item.block is None:
            groups.setdefault(item.survivor_id, []).append(item)
    ordered = sorted(groups)
    merging = ordered[: cfg.auto_merge.max_per_run]
    report.deferred_groups = len(ordered) - len(merging)
    if report.deferred_groups:
        report.warnings.append(
            f"max_per_run={cfg.auto_merge.max_per_run}: {report.deferred_groups} group(s) "
            "not merged. Re-run to continue."
        )

    judge_stamp = str(duplicate_judge_stamp())
    for survivor_id in merging:
        items = groups[survivor_id]
        survivor = particles[survivor_id]
        already = await get_co_evidential_group(session, survivor_id)
        for item in items:
            if item.copy_id not in already:
                await create_relation(
                    session,
                    survivor_id,
                    item.copy_id,
                    RelationType.CO_EVIDENTIAL,
                    RelationCreatedBy.REREAD_DUPLICATE,
                    confidence=1.0,
                )
                report.links_created += 1
            await update_particle_status(
                session, item.copy_id, Status.SUPERSEDED, StatusReason.DUPLICATE_MERGED
            )
            item.merged = True
        await record_event(
            session,
            actor=actor,
            event_type=OperatorEventType.DUPLICATES_MERGED,
            refs=[
                (EventRefKind.PARTICLE, survivor_id),
                *((EventRefKind.PARTICLE, i.copy_id) for i in items),
            ],
            payload={
                "run_id": report.run_id,
                "basis": BASIS,
                "entry_id": items[0].entry_id,
                "content_hash": norm_content_hash(survivor.content),
                "survivor": survivor_id,
                "superseded": [i.copy_id for i in items],
                "subject_ids": list(survivor.subject_ids),
                "decided_by": {
                    i.copy_id: {
                        "duplicate": judge_stamp,
                        "update": i.update_rung,
                        "stratum": i.stratum,
                    }
                    for i in items
                },
                "config": {
                    "admitted": cfg.reread.admitted,
                    "max_per_run": cfg.auto_merge.max_per_run,
                },
            },
        )
        report.merged_groups += 1
        report.merged_copies += len(items)
    await session.commit()
    return report


def write_sheet(report: RereadReport, particles_text: Mapping[str, str], path: Path) -> int:
    """Write the labelling sheet: every judged ``SAME``, in a stable random order (§5).

    Columns: ``stratum``, ``survivor_id``, ``copy_id``, ``cosine``, both texts,
    ``update``, ``guard`` (the guard that blocks the merge, if any) and an empty
    ``label`` for the owner. Returns the rows written.
    """
    rows = [i for i in report.items if i.verdict == "same"]
    rows.sort(
        key=lambda i: (
            i.stratum,
            _sheet_order(Question(i.survivor_id, i.copy_id, i.entry_id, i.cosine, i.stratum)),
        )
    )
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "stratum",
                "survivor_id",
                "copy_id",
                "cosine",
                "survivor_text",
                "copy_text",
                "update",
                "guard",
                "label",
            ]
        )
        for i in rows:
            writer.writerow(
                [
                    i.stratum,
                    i.survivor_id,
                    i.copy_id,
                    "" if i.cosine is None else f"{i.cosine:.4f}",
                    particles_text.get(i.survivor_id, ""),
                    particles_text.get(i.copy_id, ""),
                    i.update or "",
                    i.block.value if i.block in GUARD_BLOCKS else "",
                    "",
                ]
            )
    return len(rows)


async def sheet_texts(session: AsyncSession, report: RereadReport) -> dict[str, str]:
    """The texts the sheet prints, for every judged ``SAME`` question."""
    from particles.store.particle_store import get_particles_by_ids

    ids = sorted(
        {x for i in report.items if i.verdict == "same" for x in (i.survivor_id, i.copy_id)}
    )
    return {pid: p.content for pid, p in (await get_particles_by_ids(session, ids)).items()}

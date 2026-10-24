# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The restatement match: a re-extracted claim is carried into the new generation.

A ``MUTABLE`` source is re-read whole when it changes, and the model rewords
sentences the edit never touched. Without this match the generation cascade
retires each earlier claim the new text does not state verbatim
and the pass mints a reworded copy, so an edit to one line moves the note's
whole history onto new rows. The match pairs each candidate with the nearest
earlier-generation claim of the same entry and asks the duplicate question:
are the two one claim?

* **The pairs** (:func:`select_pairs`, pure): only claims the cascade would
  retire are offered (the entry alone states them and the snapshot being read
  is newer, as for a revision pair; carry-forward excluded). A pair
  must share ``stance:holder`` and reach the cosine floor, unless the two
  texts are one sentence under the §6.10 key. Matching is one-to-one, the
  key-equal pairs first and then by descending cosine.
* **The ladder**: the ``duplicate_key`` rung (``SAME`` under the key, admitted
  from the start) and the judge for the rest, at most ``max_judged``
  pairs per pass; the remainder are unasked and disclosed.
* **The guard** (:func:`decide_carry`, pure): a judged ``SAME`` carries the
  earlier claim only when no update reading said ``CHANGES`` and the
  contradiction probe answered no. Each question is asked of the pair on its
  own; the rule only combines the answers, in the direction that keeps a
  changed value visible.
* **The admission period**: until ``generation.restatement.judge_admitted``,
  a judged ``SAME`` that passes its update guard is counted as a would-be
  carry and the pass writes as if the judge had said ``DIFFERENT``.
* **Lineage**: a pair that is not carried and whose two update readings say
  ``CHANGES`` is planned as an update retirement; the pipeline applies it once
  the candidate is written ``ACTIVE``.
* **An abstention changes nothing**: a pair the judge could not answer asks
  no update reading and keeps the outcome the pass had before the match.

Gather and model calls are :func:`match_restatements`; the decisions are pure.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.conflict_resolution import SlotVerdict
from particles.core.duplicate_key import normalize_content
from particles.core.probe_verdict import ProbeKind, VerdictKey, verdict_key
from particles.core.rung_ladder import DuplicateVerdict
from particles.core.schema import JudgeVerdictKind, Particle
from particles.core.stance import stance_holder
from particles.db import write_transaction
from particles.embeddings import cosine_similarity
from particles.ingest.duplicate_judge import (
    DUPLICATE_VERDICT,
    JudgeCall,
    duplicate_judge_stamp,
    run_judge,
)
from particles.llm.usage import SamplingObservation, observe_sampling
from particles.store.probe_verdict_store import lookup_verdicts, record_verdicts

__all__ = [
    "RESTATED_AS_KEY",
    "RESTATED",
    "RESTATEMENT_LINEAGE",
    "RESTATEMENT_SHADOW",
    "RESTATEMENT_UNASKED",
    "Carry",
    "RestatementCandidate",
    "RestatementOutcome",
    "RestatementPair",
    "UpdateGuard",
    "decide_carry",
    "match_restatements",
    "restatement_notes",
    "select_pairs",
]

#: The ``properties`` key a carried claim records the newer wording under: a
#: derived annotation, never ``content``, read only for display.
RESTATED_AS_KEY = "restated_as"

#: Quality-note keys a pass discloses its restatement outcomes under.
RESTATED = "RESTATED"
RESTATEMENT_UNASKED = "RESTATEMENT_UNASKED"
RESTATEMENT_SHADOW = "RESTATEMENT_SHADOW"
RESTATEMENT_LINEAGE = "RESTATEMENT_LINEAGE"


@dataclass(frozen=True)
class RestatementCandidate:
    """One candidate of the pass, as the match reads it."""

    content: str
    holder: str | None
    embedding: Any
    """The candidate's embedding, or ``None`` when it has none."""
    skip: bool = False
    """A candidate that never takes part in reconciliation."""


@dataclass(frozen=True)
class RestatementPair:
    """A candidate and the earlier-generation claim it may restate."""

    index: int
    """The candidate's position in the pass."""
    candidate: str
    """The candidate's text."""
    existing: Particle
    cosine: float
    keyed: bool
    """``True`` when the two texts are one sentence under the §6.10 key."""


def select_pairs(
    candidates: Sequence[RestatementCandidate],
    eligible: Sequence[tuple[Particle, Any]],
    *,
    floor: float,
) -> list[RestatementPair]:
    """The one-to-one restatement pairs of a pass.

    ``eligible`` holds the earlier-generation claims the cascade would retire,
    each with its embedding. A pair needs a shared ``stance:holder``, and a
    cosine of at least ``floor`` unless it is key-equal. Key-equal pairs are
    matched first, then the rest by descending cosine, so no candidate and no
    earlier claim is in two pairs.
    """
    options: list[RestatementPair] = []
    forms = {p.id: normalize_content(p.content) for p, _ in eligible}
    for index, candidate in enumerate(candidates):
        if candidate.skip:
            continue
        form = normalize_content(candidate.content)
        for existing, emb in eligible:
            if stance_holder(existing) != candidate.holder:
                continue
            keyed = forms[existing.id] == form
            if candidate.embedding is None or emb is None:
                if not keyed:
                    continue
                cosine = 1.0
            else:
                cosine = float(cosine_similarity(candidate.embedding, emb))
            if keyed or cosine >= floor:
                options.append(RestatementPair(index, candidate.content, existing, cosine, keyed))
    options.sort(key=lambda p: (not p.keyed, -p.cosine, p.index, p.existing.id))
    used_candidates: set[int] = set()
    used_existing: set[str] = set()
    pairs: list[RestatementPair] = []
    for pair in options:
        if pair.index in used_candidates or pair.existing.id in used_existing:
            continue
        used_candidates.add(pair.index)
        used_existing.add(pair.existing.id)
        pairs.append(pair)
    return pairs


@dataclass(frozen=True)
class UpdateGuard:
    """The update ladder's readings of a pair, earlier claim first."""

    verdict: SlotVerdict | None
    """The settled verdict; ``CHANGES`` only after two ``CHANGES`` readings."""
    any_changes: bool
    """Whether any reading, the first included, answered ``CHANGES``."""


class Carry(StrEnum):
    """What a pair does to the write."""

    CARRY = "carry"
    """The candidate is folded into the earlier claim."""
    SHADOW = "shadow"
    """A judged ``SAME`` that would carry once the judge is admitted."""
    NO = "no"
    """The candidate is written as without the match."""


def decide_carry(
    verdict: DuplicateVerdict,
    *,
    keyed: bool,
    admitted: bool,
    update: UpdateGuard | None,
    contradiction: bool | None,
) -> Carry:
    """Whether a pair carries its earlier claim (the write rule).

    A key-equal pair always carries: it is own predicate. A judged
    ``SAME`` carries only when the update readings completed with no
    ``CHANGES`` and the contradiction probe answered no; a reading that could
    not complete blocks the carry. Before admission a ``SAME`` that passes the
    update guard is a would-be carry (the contradiction probe is not asked for
    it). ``contradiction`` is read only once the judge is admitted.
    """
    if keyed:
        return Carry.CARRY
    if verdict is not DuplicateVerdict.SAME:
        return Carry.NO
    if update is None or update.verdict is None or update.any_changes:
        return Carry.NO
    if not admitted:
        return Carry.SHADOW
    if contradiction is not False:
        return Carry.NO
    return Carry.CARRY


@dataclass
class RestatementOutcome:
    """What the match decided for one pass."""

    carried: dict[int, Particle] = field(default_factory=dict)
    """Candidate index to the earlier claim it is folded into."""
    lineage: dict[int, Particle] = field(default_factory=dict)
    """Candidate index to the earlier claim its update retires, once it is written."""
    shadow: int = 0
    """Judged ``SAME`` pairs that would have carried, were the judge admitted."""
    judged: int = 0
    """Pairs put to the model judge (answered from the ledger or by a call)."""
    unasked: int = 0
    """Pairs over ``max_judged``, which keep the old behaviour."""
    slot_verdicts: dict[tuple[str, str], SlotVerdict | None] = field(default_factory=dict)
    """``(earlier, later)`` texts to the settled update verdict already read, so
    the pass's own update probe does not ask the same pair again."""

    @property
    def carried_ids(self) -> frozenset[str]:
        return frozenset(p.id for p in self.carried.values())


#: An update reading: ``(earlier, later)`` texts in, the settled guard out.
SlotReader = Callable[[str, str], Awaitable[UpdateGuard]]
#: The contradiction probe: two texts in, ``True`` / ``False`` / ``None`` (failed).
ContradictionProbe = Callable[[str, str], Awaitable[bool | None]]


def _judge_key(pair: RestatementPair) -> VerdictKey:
    return verdict_key(ProbeKind.DUPLICATE, pair.existing.content, pair.candidate)


async def _judge(
    session: AsyncSession, judged: Sequence[RestatementPair], call: JudgeCall
) -> dict[int, DuplicateVerdict]:
    """The duplicate verdict of each judged pair, from the ledger first."""
    if not judged:
        return {}
    version = duplicate_judge_stamp().version
    keys = [_judge_key(p) for p in judged]
    recorded = await lookup_verdicts(session, ProbeKind.DUPLICATE, version, keys)
    # The calls below are LLM round-trips: end the read transaction first.
    await session.commit()
    out: dict[int, DuplicateVerdict] = {}
    ask: list[RestatementPair] = []
    for pair, key in zip(judged, keys, strict=True):
        if key in recorded:
            out[pair.index] = DuplicateVerdict.SAME if recorded[key] else DuplicateVerdict.DIFFERENT
        else:
            ask.append(pair)
    batch = get_config().generation.restatement.judge_batch
    fresh: dict[VerdictKey, bool] = {}
    sampling = SamplingObservation()
    for start in range(0, len(ask), batch):
        chunk = ask[start : start + batch]
        with observe_sampling(sampling):
            answers = await run_judge([(p.existing.content, p.candidate) for p in chunk], call)
        for pair, answer in zip(chunk, answers, strict=True):
            mapped = _as_duplicate(answer)
            out[pair.index] = mapped
            if mapped is not DuplicateVerdict.ABSTAIN:
                fresh[_judge_key(pair)] = mapped is DuplicateVerdict.SAME
    if fresh:
        selection = get_config().llm.for_purpose("semantic_lint")
        async with write_transaction(session):
            await record_verdicts(
                session,
                ProbeKind.DUPLICATE,
                version,
                fresh,
                model=f"{selection.provider}/{selection.model}",
                rung=duplicate_judge_stamp(),
                temperature=sampling.value(),
            )
    return out


def _as_duplicate(answer: JudgeVerdictKind | None) -> DuplicateVerdict:
    mapped = DUPLICATE_VERDICT.get(answer) if answer is not None else None
    return mapped or DuplicateVerdict.ABSTAIN


async def match_restatements(
    session: AsyncSession,
    candidates: Sequence[RestatementCandidate],
    eligible: Sequence[tuple[Particle, Any]],
    *,
    judge_call: JudgeCall | Callable[..., Awaitable[str | None]],
    read_slot: SlotReader,
    probe_contradiction: ContradictionProbe,
) -> RestatementOutcome:
    """Select, judge and decide one pass's restatement pairs (§2).

    Reads the probe ledger, commits before any model call, and records the
    judge's fresh verdicts. Writes nothing else: the folds and the lineage
    retirements are the pipeline's, inside its write phase.
    """
    cfg = get_config().generation.restatement
    outcome = RestatementOutcome()
    if not cfg.enabled or not eligible:
        return outcome
    pairs = select_pairs(candidates, eligible, floor=cfg.floor)
    keyed = [p for p in pairs if p.keyed]
    rest = [p for p in pairs if not p.keyed]  # already in descending cosine
    judged = rest[: cfg.max_judged]
    outcome.unasked = len(rest) - len(judged)
    outcome.judged = len(judged)
    for pair in keyed:
        outcome.carried[pair.index] = pair.existing

    verdicts = await _judge(session, judged, judge_call)
    for pair in judged:
        verdict = verdicts.get(pair.index, DuplicateVerdict.ABSTAIN)
        if verdict is DuplicateVerdict.ABSTAIN:
            # The judge could not tell (or its call failed): the pair keeps the
            # outcome it had before the match and costs no update reading.
            continue
        guard = await read_slot(pair.existing.content, pair.candidate)
        outcome.slot_verdicts[(pair.existing.content, pair.candidate)] = guard.verdict
        contradiction: bool | None = None
        if (
            cfg.judge_admitted
            and verdict is DuplicateVerdict.SAME
            and guard.verdict is not None
            and not guard.any_changes
        ):
            contradiction = await probe_contradiction(pair.existing.content, pair.candidate)
        decision = decide_carry(
            verdict,
            keyed=False,
            admitted=cfg.judge_admitted,
            update=guard,
            contradiction=contradiction,
        )
        if decision is Carry.CARRY:
            outcome.carried[pair.index] = pair.existing
            continue
        if decision is Carry.SHADOW:
            outcome.shadow += 1
        if cfg.lineage and guard.verdict is SlotVerdict.CHANGES:
            outcome.lineage[pair.index] = pair.existing
    return outcome


def restatement_notes(
    outcome: RestatementOutcome, *, carried: int, lineage_written: int
) -> list[str]:
    """The quality notes a pass's restatement outcome is disclosed under.

    ``carried`` counts the folds the write loop made: a carried pair that
    suppression took instead is disclosed as a suppression.
    """
    notes: list[str] = []
    if carried:
        notes.append(f"{RESTATED}: {carried} claim(s) carried into the new generation")
    if lineage_written:
        notes.append(
            f"{RESTATEMENT_LINEAGE}: {lineage_written} earlier claim(s) retired by update, "
            "with the new claim naming them"
        )
    if outcome.shadow:
        notes.append(
            f"{RESTATEMENT_SHADOW}: {outcome.shadow} judged restatement(s) would have carried; "
            "the judge is not yet admitted"
        )
    if outcome.unasked:
        notes.append(
            f"{RESTATEMENT_UNASKED}: {outcome.unasked} pair(s) over "
            "generation.restatement.max_judged were not asked"
        )
    return notes


def restated_annotation(snapshot_id: str, text: str) -> Mapping[str, str]:
    """The ``restated_as`` value a carry records on the earlier claim."""
    return {"snapshot_id": snapshot_id, "text": text}

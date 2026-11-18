# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Re-read a multi-day session transcript from its first day mark.

A transcript extracted before day marks dates every act of a multi-day
session to its first day, in each claim's ``valid_from`` and in its text,
which is immutable. The repair is a re-read. Once the session's current
distillation (which carries the marks) is deposited as the entry's newest
snapshot, :func:`plan_reread` decides what the re-read does, reading only,
and :func:`run_reread` does it through the reindex replay:

* **Read through the newest rendering.** Each replayed snapshot is read as
  the newest snapshot's text cut where that snapshot ends, through the
  day-mark reduction, so its turns carry their marks while each claim still
  cites the snapshot that first held its passage.
* **Replay from the first marked snapshot.** The replay starts at the first
  snapshot whose text reaches the first mark. Claims first cited by an
  earlier snapshot were dated right and stay; claims first cited by that
  snapshot or a later one are retired and read again.

The plan also says what the retirement loses: how many of the claims it
retires carry a tag, a relation or use history, none of which moves to the
claim read in its place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import ExtractionStatus, Particle, ProvenanceRefType
from particles.corpus.deposit import load_blob
from particles.corpus.store import list_extraction_bases
from particles.extraction.day_marks import first_mark_offset, reduce_marks
from particles.operations.audit import ExtractionCall
from particles.operations.reindex import _entry_claims, _reindex_append_entry
from particles.store.particle_store import get_active_particles_for_entry
from particles.store.relation_store import get_relations_for_particle
from particles.store.utility_store import UtilityEventRow

#: Why an entry is not re-read, as the report words it.
REFUSE_NO_MARKS = "its newest snapshot carries no day mark"
REFUSE_NOT_PREFIX = "an earlier snapshot is not the start of its newest rendering"
REFUSE_PENDING = "a snapshot before the first mark is not extracted yet"
REFUSE_MISSING = "a snapshot's content is missing from the blob store"


@dataclass
class RereadPlan:
    """What :func:`run_reread` does for one entry, decided by :func:`plan_reread`."""

    entry_id: str
    refusal: str | None = None
    #: The snapshots replayed, in capture order; the last is the newest.
    steps: list[str] = field(default_factory=list)
    #: The snapshot the first step reads as a delta from; ``None`` reads it whole.
    first_base: str | None = None
    #: The marked text each replayed snapshot is read as, where it differs.
    renderings: dict[str, bytes] = field(default_factory=dict)
    #: The claims the re-read retires.
    retire: list[Particle] = field(default_factory=list)
    #: …of those, the ones carrying a tag, a relation or use history.
    retire_with_state: int = 0
    #: The extraction calls the replay plans, for the estimate.
    calls: list[ExtractionCall] = field(default_factory=list)

    @property
    def read_chars(self) -> int:
        """Characters the replay reads, context excluded."""
        return sum(call.chars for call in self.calls)


#: The step id of a rendering not yet deposited, in a plan made for the estimate.
NOT_YET_DEPOSITED = "(not yet deposited)"


async def plan_reread(
    session: AsyncSession, entry_id: str, rendering: bytes | None = None
) -> RereadPlan:
    """Decide the re-read of one transcript entry. Reads only.

    The entry's newest snapshot must be its marked rendering, every earlier
    snapshot must reduce to a start of it, and every snapshot before the first
    replayed one must be extracted; otherwise the plan carries a refusal and
    nothing else. ``rendering`` is the session's current distillation: when it
    is not the newest snapshot yet, the plan reads as if it were deposited
    next, which is what the estimate needs; :func:`run_reread` refuses such a
    plan, so the caller deposits it and plans again.
    """
    plan = RereadPlan(entry_id=entry_id)
    stored = await list_extraction_bases(session, entry_id)
    rows = [(row.snapshot_id, row.extraction_status) for row in stored]
    try:
        contents = [load_blob(row.content_hash) for row in stored]
    except FileNotFoundError:
        plan.refusal = REFUSE_MISSING
        return plan
    if rendering is not None and (not contents or contents[-1] != rendering):
        rows.append((NOT_YET_DEPOSITED, ExtractionStatus.PENDING))
        contents.append(rendering)
    if not rows:
        plan.refusal = REFUSE_NO_MARKS
        return plan
    newest = contents[-1]
    mark = first_mark_offset(newest)
    if mark is None:
        plan.refusal = REFUSE_NO_MARKS
        return plan
    reduced_newest = reduce_marks(newest)
    ends: list[int] = []
    for content in contents:
        reduced = reduce_marks(content).text
        if not reduced_newest.text.startswith(reduced):
            plan.refusal = REFUSE_NOT_PREFIX
            return plan
        ends.append(len(reduced))
    at_mark = reduced_newest.to_reduced(mark)
    first = next(i for i, end in enumerate(ends) if end > at_mark)
    if any(status is not ExtractionStatus.COMPLETE for _, status in rows[:first]):
        plan.refusal = REFUSE_PENDING
        return plan

    plan.steps = [snapshot_id for snapshot_id, _ in rows[first:]]
    plan.first_base = rows[first - 1][0] if first else None
    for index in range(first, len(rows) - 1):
        cut = newest[: reduced_newest.to_raw(ends[index])]
        if cut != contents[index]:
            plan.renderings[rows[index][0]] = cut

    order = {snapshot_id: i for i, (snapshot_id, _) in enumerate(rows)}
    for particle in _entry_claims(
        await get_active_particles_for_entry(session, entry_id), entry_id
    ):
        seen = [
            order[ref.snapshot_id]
            for ref in particle.provenance
            if ref.type is ProvenanceRefType.SOURCE
            and ref.corpus_entry_id == entry_id
            and ref.snapshot_id in order
        ]
        if seen and min(seen) >= first:
            plan.retire.append(particle)
    plan.retire_with_state = await _with_operator_state(session, plan.retire)
    plan.calls = _planned_calls(ends, first)
    return plan


async def run_reread(session: AsyncSession, plan: RereadPlan) -> int:
    """Carry out ``plan`` through the reindex replay; returns the claims retired."""
    if plan.refusal is not None or not plan.steps:
        return 0
    if NOT_YET_DEPOSITED in plan.steps:
        raise ValueError("deposit the session's rendering and plan again before re-reading")
    return await _reindex_append_entry(
        session,
        plan.entry_id,
        plan.steps,
        first_base=plan.first_base,
        renderings=plan.renderings,
        retire=plan.retire,
    )


def _planned_calls(ends: list[int], first: int) -> list[ExtractionCall]:
    """The calls the replay makes, step by step, as the delta read cuts them.

    The first step reads whole when nothing precedes it, else as a delta;
    every later step is a delta from the one before. Sizes are reduced-text
    bytes, which the marks lengthen by a few bytes a day.
    """
    cfg = get_config().extraction
    calls: list[ExtractionCall] = []
    start = ends[first - 1] if first else 0
    for index in range(first, len(ends)):
        length = ends[index] - start
        whole = index == 0
        size = cfg.html_chunk_size if whole else cfg.append_chunk_chars
        pieces = max(1, math.ceil(length / size)) if length > 0 else 0
        for piece in range(pieces):
            chars = min(size, length - piece * size)
            context = 0 if whole else min(cfg.append_context_chars, start + piece * size)
            calls.append(ExtractionCall(chars=chars, context_chars=context))
        start = ends[index]
    return calls


async def _with_operator_state(session: AsyncSession, particles: list[Particle]) -> int:
    """How many of ``particles`` carry a tag, a relation or use history."""
    if not particles:
        return 0
    ids = [p.id for p in particles]
    used: set[str] = set()
    for start in range(0, len(ids), 500):
        rows = await session.execute(
            select(UtilityEventRow.particle_id)
            .where(UtilityEventRow.particle_id.in_(ids[start : start + 500]))
            .distinct()
        )
        used.update(rows.scalars())
    count = 0
    for particle in particles:
        carries = bool(particle.tags) or particle.id in used
        if not carries:
            carries = bool(await get_relations_for_particle(session, particle.id))
        count += carries
    return count

# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Which earlier snapshot an append-only snapshot is read as a delta from.

The Engine half of the delta read. :func:`plan_append` gathers the entry's
snapshots, the base's content and the chunk hashes the entry's claims carry,
decides the base and its raw offset, and checks the append-only promise on raw
bytes. The general extractor (Client) then decodes the prefix it is handed and
reads only what follows; the text arithmetic both halves share lives in
:mod:`particles.extraction.append_delta`.

The base is the snapshot that records the furthest read whose read prefix
this snapshot's content starts with. That is the latest earlier
``COMPLETE`` snapshot, at its ``extracted_through`` or, for a snapshot
extracted before that column existed, the derivation from chunk
hashes, unless another recorded read goes further:

* a **partial read**: a snapshot that is not ``COMPLETE`` but holds
  an ``extracted_through``, which means the claims of the text before it are
  written. The snapshot being extracted counts, so a partial read's retry
  resumes where it stopped, and so does a later or an earlier one;
* a **later** ``COMPLETE`` snapshot. The catch-up extracts least-tried first,
  so a snapshot whose extraction failed can be passed by its successor; text
  the successor already read is not read again.

The offset is capped at this snapshot's length, so a read that went past its
end leaves an empty delta. A snapshot holding the whole-read marker
is read whole, with no base.

On a session transcript every prefix test is made on day-mark-reduced text,
both sides, and the chosen offset is mapped into this snapshot's own raw
bytes. An unmarked snapshot and the first marked snapshot of
the same transcript then compare as the same text, so the first marked
deposit reads only its new turns, and the plan hands the extractor this
snapshot's own bytes up to the mapped offset. On a text with no marks the
reduction is the identity and every decision is the raw one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import (
    CorpusEntry,
    ExtractionStatus,
    Mutability,
    Particle,
    ProvenanceRefType,
    Snapshot,
)
from particles.corpus.deposit import load_blob
from particles.corpus.store import ExtractionBaseRow, list_extraction_bases
from particles.extraction.append_delta import derive_base_offset
from particles.extraction.day_marks import Reduction, reduce_marks
from particles.extraction.general import EXTRACTOR_ID as GENERAL_EXTRACTOR_ID
from particles.extraction.general import source_text_flags
from particles.extraction.registry import ExtractorPlugin
from particles.store.particle_store import get_particles_for_entry

log = logging.getLogger(__name__)

#: ``append_base`` value that finds the base (the default everywhere but reindex).
AUTO_BASE = "auto"

FALLBACK_DISABLED = "extraction.append_only_delta is false"
FALLBACK_NO_BASE = "no earlier snapshot of the entry is COMPLETE"
FALLBACK_OTHER_EXTRACTOR = "the base was extracted by another extractor"
FALLBACK_BASE_MISSING = "the base's content is missing from the blob store"
FALLBACK_RAW_PREFIX = "the snapshot does not start with the base's content"


@dataclass(frozen=True)
class AppendPlan:
    """What the pipeline hands the extractor for one append-only snapshot.

    Exactly one of ``prefix`` and ``fallback`` is set. ``prefix`` is the raw
    content the store has already extracted (the base's content up to its
    offset); ``fallback`` says why the snapshot is read whole instead (§5).
    """

    prefix: bytes | None = None
    base_snapshot_id: str | None = None
    fallback: str | None = None
    derived_offset: bool = False
    """The offset came from chunk hashes (§4), the base having no ``extracted_through``."""
    marks_reduced: bool = False
    """The raw prefix test failed and the day-mark-reduced one passed."""


def reduces_marks(entry: CorpusEntry) -> bool:
    """Whether the entry's prefix tests are made on day-mark-reduced text."""
    return entry.source_type in get_config().extraction_validity.session_source_types


def in_scope(entry: CorpusEntry, extractor: ExtractorPlugin) -> bool:
    """The delta applies to APPEND_ONLY entries read by the general extractor (§6)."""
    return (
        entry.mutability is Mutability.APPEND_ONLY
        and getattr(extractor, "EXTRACTOR_ID", None) == GENERAL_EXTRACTOR_ID
    )


async def plan_append(
    session: AsyncSession,
    entry: CorpusEntry,
    snapshot: Snapshot,
    extractor: ExtractorPlugin,
    content: bytes,
    *,
    append_base: str | None = AUTO_BASE,
    ignore_own_partial: bool = False,
) -> AppendPlan | None:
    """Choose the base and prefix for ``snapshot``, or say why there is none.

    ``append_base`` is :data:`AUTO_BASE` to find the base, a snapshot id to
    use that snapshot (the reindex replay names the one before), or ``None``
    for a whole read with no base (the replay's first snapshot). Returns
    ``None`` when the delta does not apply at all: another mutability class,
    another extractor, ``append_base=None``, or a snapshot holding the
    whole-read marker, whose retry is the same whole read. None
    of those is a fallback.

    ``ignore_own_partial`` plans as if the snapshot held no partial read of its
    own: no marker, no offset. A reindex passes it, since it is about to
    retire the claims that partial read wrote.
    """
    if append_base is None or not in_scope(entry, extractor):
        return None
    if not get_config().extraction.append_only_delta:
        return AppendPlan(fallback=FALLBACK_DISABLED)

    rows = await list_extraction_bases(session, entry.entry_id)
    index = next((i for i, r in enumerate(rows) if r.snapshot_id == snapshot.snapshot_id), None)
    if index is None:
        return AppendPlan(fallback=FALLBACK_NO_BASE)
    base: ExtractionBaseRow | None
    if append_base == AUTO_BASE:
        if rows[index].resume_whole and not ignore_own_partial:
            return None
        earlier = [r for r in rows[:index] if r.extraction_status is ExtractionStatus.COMPLETE]
        base = earlier[-1] if earlier else None
        earlier_plan = await _plan_from(session, entry, extractor, content, base)
        further = _further_read(
            rows,
            index,
            content,
            beyond=len(earlier_plan.prefix) if earlier_plan.prefix is not None else -1,
            ignore_own=ignore_own_partial,
            reduce=reduces_marks(entry),
        )
        return further if further is not None else earlier_plan
    else:
        base = next(
            (
                r
                for r in rows
                if r.snapshot_id == append_base and r.extraction_status is ExtractionStatus.COMPLETE
            ),
            None,
        )
    return await _plan_from(session, entry, extractor, content, base)


async def _plan_from(
    session: AsyncSession,
    entry: CorpusEntry,
    extractor: ExtractorPlugin,
    content: bytes,
    base: ExtractionBaseRow | None,
) -> AppendPlan:
    """The plan reading ``content`` as a delta from ``base``, a COMPLETE snapshot."""
    if base is None:
        return AppendPlan(fallback=FALLBACK_NO_BASE)

    claims = await get_particles_for_entry(session, entry.entry_id)
    extractor_id = getattr(extractor, "EXTRACTOR_ID", GENERAL_EXTRACTOR_ID)
    base_extractors = {
        p.extractor_ref.name
        for p in claims
        if p.extractor_ref is not None
        and any(ref.snapshot_id == base.snapshot_id for ref in p.provenance)
    }
    if base_extractors and extractor_id not in base_extractors:
        return AppendPlan(
            base_snapshot_id=base.snapshot_id,
            fallback=f"{FALLBACK_OTHER_EXTRACTOR} ({', '.join(sorted(base_extractors))})",
        )

    try:
        base_content = load_blob(base.content_hash)
    except FileNotFoundError:
        return AppendPlan(base_snapshot_id=base.snapshot_id, fallback=FALLBACK_BASE_MISSING)

    derived = False
    offset = base.extracted_through
    if offset is None:
        is_markdown, mark_tools = source_text_flags(entry.source_type)
        derived_offset = derive_base_offset(
            base_content,
            _carried_hashes(claims, entry.entry_id, extractor_id),
            is_markdown=is_markdown,
            mark_tools=mark_tools,
        )
        derived = derived_offset is not None
        offset = len(base_content) if derived_offset is None else derived_offset
    offset = min(offset, len(base_content))

    prefix = base_content[:offset]
    if content.startswith(prefix):
        return AppendPlan(prefix=prefix, base_snapshot_id=base.snapshot_id, derived_offset=derived)
    if reduces_marks(entry):
        mapped = _reduced_prefix(reduce_marks(base_content), offset, reduce_marks(content))
        if mapped is not None:
            return AppendPlan(
                prefix=content[:mapped],
                base_snapshot_id=base.snapshot_id,
                derived_offset=derived,
                marks_reduced=True,
            )
    return AppendPlan(base_snapshot_id=base.snapshot_id, fallback=FALLBACK_RAW_PREFIX)


def _reduced_prefix(base: Reduction, offset: int, snapshot: Reduction) -> int | None:
    """Where ``base`` read to, as a raw offset in the snapshot, if their reduced texts agree.

    ``offset`` is in the base's raw bytes. The base's reduced text up to the
    same point must prefix the snapshot's reduced text; the
    result is the snapshot's smallest raw offset at that reduced point, a line
    start when the base's offset was one.
    """
    at = base.to_reduced(offset)
    if len(snapshot.text) < at or snapshot.text[:at] != base.text[:at]:
        return None
    return snapshot.to_raw(at)


def _records_further_read(row: ExtractionBaseRow, *, later: bool) -> bool:
    """Whether ``row`` is a recorded read the earlier COMPLETE base does not already stand for.

    A partial read (not COMPLETE, with an offset) anywhere in the entry, the
    snapshot itself included, or a later COMPLETE snapshot with
    an offset. An earlier COMPLETE snapshot is the earlier base's business.
    """
    if row.extracted_through is None:
        return False
    if row.extraction_status is ExtractionStatus.COMPLETE:
        return later
    return True


def _further_read(
    rows: list[ExtractionBaseRow],
    index: int,
    content: bytes,
    *,
    beyond: int,
    ignore_own: bool,
    reduce: bool = False,
) -> AppendPlan | None:
    """The plan from the furthest recorded read past ``beyond`` bytes, if one matches (§4).

    Candidates are tried furthest first, each offset capped at ``content``'s
    length. The first whose content up to the capped offset is a prefix of
    ``content`` wins. A capped offset equal to ``content``'s length is an
    empty delta: a later read already covered the whole snapshot.

    With ``reduce`` (a session transcript) each candidate's offset
    is converted into day-mark-reduced coordinates through its own text's
    reduction, the comparison and the cap are made there, and the winner is
    mapped back into ``content``'s raw bytes. ``beyond`` stays in raw bytes of
    ``content``.
    """
    view = reduce_marks(content) if reduce else Reduction(text=content, cuts=())
    length = len(view.text)
    floor = view.to_reduced(beyond) if beyond >= 0 else beyond
    candidates: list[tuple[int, ExtractionBaseRow, Reduction]] = []
    for i, row in enumerate(rows):
        if (i == index and ignore_own) or not _records_further_read(row, later=i > index):
            continue
        if row.snapshot_id == rows[index].snapshot_id:
            text = view
        else:
            try:
                raw = load_blob(row.content_hash)
            except FileNotFoundError:
                continue
            text = reduce_marks(raw) if reduce else Reduction(text=raw, cuts=())
        at = text.to_reduced(row.extracted_through or 0)
        candidates.append((min(at, length), row, text))
    for capped, row, text in sorted(candidates, key=lambda c: c[0], reverse=True):
        if capped <= floor:
            return None
        if text is view or text.text[:capped] == view.text[:capped]:
            raw_end = view.to_raw(capped)
            return AppendPlan(
                prefix=content[:raw_end],
                base_snapshot_id=row.snapshot_id,
                marks_reduced=bool(text.cuts) != bool(view.cuts),
            )
    return None


def waiting_on(rows: list[ExtractionBaseRow], snapshot_id: str) -> ExtractionBaseRow | None:
    """The earliest earlier snapshot holding the whole-read marker, if any.

    While one exists, ``snapshot_id`` is not extracted: it would read the
    marked snapshot's text under a longer text's chunking and could write a
    kept chunk's claims a second time.
    """
    for row in rows:
        if row.snapshot_id == snapshot_id:
            return None
        if row.resume_whole:
            return row
    return None


def earlier_all_complete(rows: list[ExtractionBaseRow], snapshot_id: str) -> bool:
    """Every snapshot of the entry before ``snapshot_id`` is COMPLETE (rule 4)."""
    for row in rows:
        if row.snapshot_id == snapshot_id:
            return True
        if row.extraction_status is not ExtractionStatus.COMPLETE:
            return False
    return True


def _carried_hashes(claims: list[Particle], entry_id: str, extractor_id: str) -> set[str]:
    """Every chunk hash this extractor's claims of the entry carry on a source ref to it."""
    return {
        ref.chunk_hash
        for p in claims
        if p.extractor_ref is not None and p.extractor_ref.name == extractor_id
        for ref in p.provenance
        if ref.type is ProvenanceRefType.SOURCE
        and ref.corpus_entry_id == entry_id
        and ref.chunk_hash
    }

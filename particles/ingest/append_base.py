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
    if not content.startswith(prefix):
        return AppendPlan(base_snapshot_id=base.snapshot_id, fallback=FALLBACK_RAW_PREFIX)
    return AppendPlan(prefix=prefix, base_snapshot_id=base.snapshot_id, derived_offset=derived)


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
) -> AppendPlan | None:
    """The plan from the furthest recorded read past ``beyond`` bytes, if one matches (§4).

    Candidates are tried furthest first, each offset capped at ``content``'s
    length. The first whose content up to the capped offset is a prefix of
    ``content`` wins. A capped offset equal to ``content``'s length is an
    empty delta: a later read already covered the whole snapshot.
    """
    length = len(content)
    candidates = [
        (min(r.extracted_through or 0, length), r)
        for i, r in enumerate(rows)
        if not (i == index and ignore_own) and _records_further_read(r, later=i > index)
    ]
    for capped, row in sorted(candidates, key=lambda c: c[0], reverse=True):
        if capped <= beyond:
            return None
        if row.snapshot_id == rows[index].snapshot_id:
            return AppendPlan(prefix=content[:capped], base_snapshot_id=row.snapshot_id)
        try:
            other = load_blob(row.content_hash)
        except FileNotFoundError:
            continue
        if other[:capped] == content[:capped]:
            return AppendPlan(prefix=content[:capped], base_snapshot_id=row.snapshot_id)
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

# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""What a partly failed append-only read keeps.

When some calls of an ``APPEND_ONLY`` snapshot's chunked read fail, the
pipeline used to discard every answered chunk and reset the snapshot, so the
retry paid for every call again. This module decides what can be kept instead,
as pure functions over plain values (D2). The pipeline gathers the
inputs and applies the result:

* :func:`decide_partial_keep` picks the chunks to write and where the
  snapshot's next read starts: an exact raw offset (a delta, or a whole read
  with an earlier snapshot still unread), or the whole-read marker.
* :func:`own_chunk_claims` maps the chunk hashes a snapshot's own claims carry
  to their ids, which carry-forward skips on the retry whatever extractor
  version wrote them (§9).
* :func:`chunking_change_note` says when a chunking knob changed between two
  attempts, which no hash match can absorb (§9).

Engine layer: it reads :class:`~particles.core.schema.Particle` provenance and
is called by ``ingest.pipeline``. The text arithmetic is the Client's
(:mod:`particles.extraction.append_delta`), so the plan the retry will make is
computed here by the same functions the retry uses.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from particles.core.schema import Particle, ProvenanceRefType
from particles.core.status import Status
from particles.extraction.append_delta import plan_delta_chunks, raw_offset_for
from particles.extraction.components import ComponentRecord
from particles.extraction.general import (
    PATH_APPEND_DELTA,
    PATH_CHUNKED,
    ChunkOutcome,
    _normalise_for_hashing,
    extraction_text,
)
from particles.extraction.incremental import ChunkUnit


@dataclass(frozen=True)
class PartialKeep:
    """What a partly failed read writes, and where its snapshot's next read starts.

    ``kept_hashes`` are the chunks whose candidates are written. Exactly one of
    ``offset`` (raw bytes, stamped as ``extracted_through``) and ``marker``
    (the whole-read marker) describes the resume.
    """

    kept_hashes: frozenset[str]
    offset: int | None
    marker: bool
    kept_calls: int
    """Answered calls whose candidates are written."""
    resent_calls: int
    """Answered calls discarded, which the retry sends again."""
    later_calls: int
    """…of the kept calls, those after the first failure."""


def decide_partial_keep(
    outcomes: Sequence[ChunkOutcome],
    *,
    delta: bool,
    content: bytes,
    is_markdown: bool,
    mark_tools: bool,
    earlier_all_complete: bool,
    chunk_chars: int,
    context_chars: int,
) -> PartialKeep | None:
    """Decide what a partly failed chunked read keeps, or ``None``.

    ``None`` means today's rule: nothing is written and the snapshot is reset.
    That is the answer when the read has no chunk outcomes (a single call), no
    failed chunk, an outcome without a start, or nothing answered worth
    keeping, and when the leading run's stop does not map exactly to raw
    bytes.

    A **whole read** whose earlier snapshots are all ``COMPLETE`` (rule 4)
    keeps every answered chunk under the marker: its retry is the same whole
    read, so each chunk repeats with its hash. Otherwise the read keeps its
    leading run (the answered chunks before the first failure) and stamps the
    exact raw offset of the first failed chunk's start. A **delta** whose
    earlier snapshots are all ``COMPLETE`` also keeps each later answered
    chunk that is not the final one and whose hash the retry's own plan
    contains.
    """
    if not outcomes or any(o.start is None for o in outcomes):
        return None
    first_failed = next((i for i, o in enumerate(outcomes) if o.status == "failed"), None)
    if first_failed is None:
        return None
    answered = [i for i, o in enumerate(outcomes) if o.status == "answered"]

    if not delta and earlier_all_complete:
        kept = frozenset(outcomes[i].prompt_hash for i in answered)
        if not kept:
            return None
        return PartialKeep(
            kept_hashes=kept,
            offset=None,
            marker=True,
            kept_calls=len(answered),
            resent_calls=0,
            later_calls=sum(1 for i in answered if i > first_failed),
        )

    text, _ = extraction_text(content, is_markdown=is_markdown, mark_tools=mark_tools)
    stop = outcomes[first_failed].start
    if stop is None:
        return None
    if not delta and _normalise_for_hashing(text)[:stop] != text[:stop]:
        # A whole read's chunks are cut from the normalised text. Its offsets
        # equal the extraction text's only where normalisation removed
        # nothing before them.
        return None
    offset = raw_offset_for(content, stop, is_markdown=is_markdown, mark_tools=mark_tools)
    decoded, _ = extraction_text(content[:offset], is_markdown=is_markdown, mark_tools=mark_tools)
    if len(decoded) != stop:
        # The retry would resume before the failed chunk and re-read text the
        # leading run covered.
        return None

    leading = [i for i in answered if i < first_failed]
    later: list[int] = []
    if delta and earlier_all_complete:
        retry = {
            ChunkUnit(chunk_id="retry", chunk_text=p.text, context=p.context).prompt_hash
            for p in plan_delta_chunks(
                text, stop, chunk_chars=chunk_chars, context_chars=context_chars
            )
        }
        final = len(outcomes) - 1
        later = [
            i for i in answered if first_failed < i < final and outcomes[i].prompt_hash in retry
        ]
    kept_indices = leading + later
    if not kept_indices:
        return None
    return PartialKeep(
        kept_hashes=frozenset(outcomes[i].prompt_hash for i in kept_indices),
        offset=offset,
        marker=False,
        kept_calls=len(kept_indices),
        resent_calls=len(answered) - len(kept_indices),
        later_calls=len(later),
    )


def own_chunk_claims(
    claims: Iterable[Particle],
    *,
    entry_id: str,
    snapshot_id: str,
    exclude_ids: frozenset[str] = frozenset(),
) -> dict[str, list[str]]:
    """Chunk hash to the ids of the ACTIVE claims citing ``snapshot_id`` that carry it (§9)."""
    out: dict[str, list[str]] = {}
    for claim in claims:
        if claim.status is not Status.ACTIVE or claim.id in exclude_ids:
            continue
        hashes = {
            ref.chunk_hash
            for ref in claim.provenance
            if ref.type is ProvenanceRefType.SOURCE
            and ref.corpus_entry_id == entry_id
            and ref.snapshot_id == snapshot_id
            and ref.chunk_hash
        }
        for chunk_hash in sorted(hashes):
            out.setdefault(chunk_hash, []).append(claim.id)
    return out


_PATH_KNOBS = {
    PATH_CHUNKED: "extraction.html_chunk_size",
    PATH_APPEND_DELTA: "extraction.append_chunk_chars / extraction.append_context_chars",
}


def chunking_change_note(
    stored: ComponentRecord | None, current: Mapping[str, str], *, marker: bool
) -> str | None:
    """Say so when a chunking knob changed since the earlier attempt.

    The kept chunks of a partial read are skipped on the retry only while the
    retry plans the same chunks. The earlier attempt's component record holds
    the digest of the chunking path it used (``path.chunked`` for a whole read,
    ``path.append_delta`` for a delta), and the digest covers the knobs, so a
    different current digest means some kept chunk will be read again.
    """
    if stored is None:
        return None
    path = PATH_CHUNKED if marker else PATH_APPEND_DELTA
    before = stored.exercised.get(path)
    if before is None or before == current.get(path):
        return None
    return (
        f"chunking changed since this snapshot's earlier attempt ({_PATH_KNOBS[path]}): "
        "chunks it already wrote are read again, and their claims may repeat in other "
        "words; `particles reindex <entry>` replaces them"
    )

# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Source-passage hydration — the source text behind one particle.

A particle's SOURCE provenance ref names a corpus entry and snapshot, and —
when the extractor took the chunked carry-forward path — the SHA-256 of the
exact normalised chunk it saw (``ProvenanceRef.chunk_hash``). That
makes the passage re-derivable with no index and no stored copy: stored blob →
the extractor's own text pipeline → the same splitter → hash match.

Three honest outcomes, never blurred together (:class:`PassageMatch`):

* ``EXACT`` — a re-derived chunk hashed to ``chunk_hash``. This *is* the text
  the extractor was shown, verified.
* ``LOCATED`` — no hash to match (single-pass documents carry none, which is
  most short sources), or the hash missed (chunk-config drift, a non-general
  chunker). The passage is the paragraph or line sharing the most terms with
  the particle's ``content``. A reading aid, **not** verification.
* ``WHOLE`` — nothing cleared the overlap floor; the snapshot text is returned
  (capped) rather than a guess.

Everything here is **display only**. The hydrated text and the overlap figure
are never a score input: excerpt provenance is not a truth signal.
That is structural, not a convention — nothing under
``particles.operations.query`` imports this module, and this module reads no
ranking state.

Only the *stored blob* is ever read, never the entry's ``uri_r``. Whatever a
deposit path removed before archiving (transcript redaction, projected-region
stripping) therefore stays removed.
"""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import Particle, ProvenanceRef, ProvenanceRefType
from particles.corpus.deposit import load_blob
from particles.corpus.store import get_entry, get_snapshot, resolve_snapshot_for_blob
from particles.extraction.general import (
    _normalise_for_hashing,
    _split_into_paragraph_chunks,
    _strip_obsidian_frontmatter,
    content_to_text,
    sniff_image_media_type,
)
from particles.extraction.tool_turns import mark_tool_turns
from particles.store.particle_store import get_particle


class PassageMatch(StrEnum):
    """How the returned passage relates to what the extractor saw."""

    EXACT = "EXACT"
    LOCATED = "LOCATED"
    WHOLE = "WHOLE"
    UNAVAILABLE = "UNAVAILABLE"


class SourcePassage(BaseModel):
    """The source text behind one particle, with how it was found.

    ``text`` is empty only when ``match`` is ``UNAVAILABLE``; ``note`` then
    says why. ``locate_overlap`` is the share of the particle's distinct terms
    found in a ``LOCATED`` passage — a reading aid, never a confidence.
    """

    particle_id: str
    match: PassageMatch
    text: str = ""
    corpus_entry_id: str | None = None
    snapshot_id: str | None = None
    uri_r: str | None = None
    source_type: str | None = None
    locate_overlap: float | None = None
    # Characters in the snapshot's derived text; ``truncated`` is set when
    # ``text`` was cut to ``source_passage.max_passage_chars``.
    snapshot_chars: int = 0
    truncated: bool = False
    note: str | None = None


_TOKEN_RE = re.compile(r"[a-z0-9_]+")
# Deliberately tiny: enough that function words do not carry a match on their
# own, not a linguistic resource to maintain.
# fmt: off
_STOPWORDS = frozenset({
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
    "has", "have", "had", "not", "but", "its", "into", "than", "then", "they",
    "them", "their", "will", "would", "can", "could", "should", "been", "being",
    "which", "when", "what", "where", "who", "how", "also", "such", "each",
    "other", "over", "under",
})
# fmt: on


def _terms(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 3 and t not in _STOPWORDS}


def extractor_view_text(content: bytes, source_type: str | None) -> str:
    """Re-derive the text the general extractor was shown for ``content``.

    Mirrors the pre-chunking steps of the single-pass path: decode, drop
    Markdown frontmatter, relabel tool turns. Keeping the relabel means a
    transcript's tool output is displayed with the same *unverified* marking
    the extractor saw, and is what lets a chunk hash over marked text match.
    """
    text = content_to_text(content)
    if source_type == "LOCAL_MARKDOWN":
        _, text = _strip_obsidian_frontmatter(text)
    if source_type in get_config().extraction.tool_turn_source_types:
        text, _ = mark_tool_turns(text)
    return text


def find_chunk(text: str, chunk_hash: str) -> str | None:
    """Return the re-derived chunk whose SHA-256 is ``chunk_hash``, if any."""
    size = get_config().extraction.html_chunk_size
    for chunk in _split_into_paragraph_chunks(_normalise_for_hashing(text), size):
        if hashlib.sha256(chunk.encode("utf-8")).hexdigest() == chunk_hash:
            return chunk
    return None


def locate_passage(text: str, content: str) -> tuple[str, float] | None:
    """Best-overlap paragraph (or line within it) for ``content``, or ``None``.

    Scores each blank-line-separated paragraph by the share of ``content``'s
    distinct terms it contains; earliest wins a tie, so the result is
    deterministic. When the winning paragraph is a multi-line block (a bullet
    list, a transcript) and one of its lines alone carries every term the
    paragraph matched, that line is returned instead — the tighter quote.
    Tightening never trades matched terms for brevity: a claim assembled from
    several lines keeps its whole paragraph. ``None`` when nothing reaches
    ``source_passage.locate_min_overlap``.
    """
    wanted = _terms(content)
    if not wanted:
        return None
    floor = get_config().source_passage.locate_min_overlap

    def best(units: list[str]) -> tuple[str, float]:
        scored = [(len(wanted & _terms(u)) / len(wanted), u) for u in units]
        score, unit = max(scored, key=lambda su: su[0])
        return unit, score

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return None
    paragraph, score = best(paragraphs)
    if score < floor:
        return None
    lines = [ln.strip() for ln in paragraph.splitlines() if ln.strip()]
    if len(lines) > 1:
        line, line_score = best(lines)
        if line_score >= score:
            return line, line_score
    return paragraph, score


def derive_passage(
    text: str, *, content: str, chunk_hash: str | None
) -> tuple[PassageMatch, str, float | None]:
    """Pick the passage for ``content`` out of ``text`` — pure, no I/O."""
    if chunk_hash:
        chunk = find_chunk(text, chunk_hash)
        if chunk is not None:
            return PassageMatch.EXACT, chunk, None
    located = locate_passage(text, content)
    if located is not None:
        return PassageMatch.LOCATED, located[0], located[1]
    return PassageMatch.WHOLE, text.strip(), None


def _source_ref(particle: Particle) -> ProvenanceRef | None:
    return next(
        (
            r
            for r in particle.provenance
            if r.type == ProvenanceRefType.SOURCE and r.corpus_entry_id
        ),
        None,
    )


async def hydrate_source_passage(session: AsyncSession, particle_id: str) -> SourcePassage | None:
    """Return the source passage behind ``particle_id``; ``None`` if no such particle.

    Read-only: one particle lookup, the entry + snapshot rows, one blob read.
    A particle with no corpus source, a snapshot with no stored blob, or a
    source that renders to no text comes back ``UNAVAILABLE`` with a ``note``
    rather than raising — absence of a passage is an answer, not an error.
    """
    particle = await get_particle(session, particle_id)
    if particle is None:
        return None

    ref = _source_ref(particle)
    if ref is None:
        return SourcePassage(
            particle_id=particle.id,
            match=PassageMatch.UNAVAILABLE,
            note="This particle has no corpus source (it derives from other particles).",
        )

    entry = await get_entry(session, ref.corpus_entry_id)
    notes: list[str] = []
    if ref.snapshot_id:
        snapshot = await get_snapshot(session, ref.snapshot_id)
    else:
        snapshot = await resolve_snapshot_for_blob(session, ref.corpus_entry_id)
        notes.append("Provenance pins no snapshot; the entry's latest snapshot was read.")

    passage = SourcePassage(
        particle_id=particle.id,
        match=PassageMatch.UNAVAILABLE,
        corpus_entry_id=ref.corpus_entry_id,
        snapshot_id=snapshot.snapshot_id if snapshot else ref.snapshot_id,
        uri_r=entry.uri_r if entry else None,
        source_type=entry.source_type if entry else None,
    )
    if snapshot is None:
        return passage.model_copy(update={"note": "The source snapshot is not in this store."})
    try:
        content = load_blob(snapshot.content_hash)
    except FileNotFoundError:
        return passage.model_copy(
            update={
                "note": "The snapshot's stored content is missing (not archived, or the "
                "blob is unreachable — see `particles corpus fsck`)."
            }
        )
    if sniff_image_media_type(content) is not None:
        return passage.model_copy(update={"note": "The source is an image; it has no text."})

    text = extractor_view_text(content, passage.source_type)
    if not text.strip():
        return passage.model_copy(update={"note": "The source rendered to no text."})

    match, found, overlap = derive_passage(
        text, content=particle.content, chunk_hash=ref.chunk_hash
    )
    if ref.chunk_hash and match is not PassageMatch.EXACT:
        notes.append(
            "The recorded chunk hash matched no re-derived chunk (chunking "
            "configuration changed since extraction, or a domain extractor's "
            "own chunker produced it)."
        )
    cap = get_config().source_passage.max_passage_chars
    truncated = len(found) > cap
    return passage.model_copy(
        update={
            "match": match,
            "text": found[:cap] if truncated else found,
            "locate_overlap": overlap,
            "snapshot_chars": len(text),
            "truncated": truncated,
            "note": " ".join(notes) or None,
        }
    )

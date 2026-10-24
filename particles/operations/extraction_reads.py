# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""What extracting a snapshot will send, planned before it runs.

The consolidation cycle prices pass 1 before its first call, for
``memory consolidate --dry-run`` and for ``consolidation.budget_usd``. Pricing
each snapshot's whole blob overstates a transcript backlog: the general
extractor reads an ``APPEND_ONLY`` snapshot as the tail it adds to the text
already extracted, skips the chunks whose claims it already holds
(chunk-hash carry-forward), and stops at
``extraction.max_llm_calls_per_source``. :func:`plan_reads` makes the same
choices the extraction will, with the same functions, and returns one
:class:`~particles.operations.audit.ExtractionCall` per call it expects.

It reads only: the base it finds for a delta read is the one
:func:`~particles.ingest.append_base.plan_append` would choose, and the
carry-forward lookup is the one the extractor registers. A snapshot routed to
another extractor, or whose content cannot be decoded, is sized from its raw
bytes through the audit's chunk rule, as the estimate did before; bytes stand
in for characters there and err high for an HTML or PDF source.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import WarcRecordType
from particles.corpus.deposit import load_blob
from particles.corpus.store import get_entry
from particles.extraction.append_delta import decoded_prefix_length, plan_delta_chunks
from particles.extraction.general import EXTRACTOR_ID as GENERAL_EXTRACTOR_ID
from particles.extraction.general import EXTRACTOR_VERSION as GENERAL_EXTRACTOR_VERSION
from particles.extraction.general import (
    extraction_text,
    source_text_flags,
    whole_read_chunks,
)
from particles.extraction.incremental import ChunkUnit
from particles.extraction.registry import select_extractor
from particles.ingest.append_base import plan_append
from particles.operations.audit import ExtractionCall, chunk_sizes, output_density
from particles.store.particle_store import get_active_particles_for_chunk_hash


@dataclass(frozen=True)
class SnapshotRead:
    """The calls extracting one snapshot is expected to make."""

    snapshot_id: str
    calls: tuple[ExtractionCall, ...] = ()
    #: True when the snapshot is read as an append-only tail.
    delta: bool = False
    #: Characters the snapshot holds (raw bytes when it is sized from bytes).
    whole_chars: int = 0
    #: Chunks whose claims the store already holds, so no call reads them.
    carried: int = 0
    #: Chunks past ``extraction.max_llm_calls_per_source``, never read.
    capped: int = 0


@dataclass
class ReadSummary:
    """Totals over a set of :class:`SnapshotRead`, for the plan's one-line detail."""

    snapshots: int = 0
    delta: int = 0
    whole_chars: int = 0
    read_chars: int = 0
    carried: int = 0
    capped: int = 0
    calls: list[ExtractionCall] = field(default_factory=list)

    @classmethod
    def of(cls, reads: Sequence[SnapshotRead]) -> ReadSummary:
        """Sum ``reads``."""
        summary = cls()
        for read in reads:
            summary.snapshots += 1
            summary.delta += int(read.delta)
            summary.whole_chars += read.whole_chars
            summary.read_chars += sum(call.chars for call in read.calls)
            summary.carried += read.carried
            summary.capped += read.capped
            summary.calls.extend(read.calls)
        return summary


async def plan_reads(
    session: AsyncSession, chosen: Sequence[tuple[str, str]]
) -> list[SnapshotRead]:
    """One :class:`SnapshotRead` per ``(entry_id, snapshot_id)``, in order. Reads only."""
    return [await _plan_one(session, entry_id, snapshot_id) for entry_id, snapshot_id in chosen]


async def _plan_one(session: AsyncSession, entry_id: str, snapshot_id: str) -> SnapshotRead:
    entry = await get_entry(session, entry_id)
    snapshot = (
        next((s for s in entry.snapshots if s.snapshot_id == snapshot_id), None)
        if entry is not None
        else None
    )
    if entry is None or snapshot is None:
        return SnapshotRead(snapshot_id=snapshot_id)
    if snapshot.warc_record_type is WarcRecordType.ABSENT:
        # An absence is processed with no read and no call.
        return SnapshotRead(snapshot_id=snapshot_id)
    try:
        content = load_blob(snapshot.content_hash)
    except (FileNotFoundError, ValueError):
        return SnapshotRead(snapshot_id=snapshot_id)
    density = output_density(entry.source_type)

    extractor = select_extractor(entry.source_type)
    if getattr(extractor, "EXTRACTOR_ID", None) != GENERAL_EXTRACTOR_ID:
        return _from_bytes(snapshot_id, len(content), density)
    is_markdown, mark_tools = source_text_flags(entry.source_type)
    try:
        text, _ = extraction_text(content, is_markdown=is_markdown, mark_tools=mark_tools)
    except Exception:  # noqa: BLE001 — an undecodable source is sized from its bytes
        return _from_bytes(snapshot_id, len(content), density)

    units: list[ChunkUnit] | None = None
    delta = False
    plan = await plan_append(session, entry, snapshot, extractor, content)
    if plan is not None and plan.prefix is not None:
        prefix_len, _ = decoded_prefix_length(
            plan.prefix, text, is_markdown=is_markdown, mark_tools=mark_tools
        )
        if prefix_len is not None:
            cfg = get_config().extraction
            planned = plan_delta_chunks(
                text,
                prefix_len,
                chunk_chars=cfg.append_chunk_chars,
                context_chars=cfg.append_context_chars,
            )
            units = [
                ChunkUnit(chunk_id=f"delta_{i}", chunk_text=p.text, context=p.context)
                for i, p in enumerate(planned, start=1)
            ]
            delta = True
    if units is None:
        chunks = whole_read_chunks(text)
        if chunks is None:
            # A single-pass read: one call, never answered by carry-forward.
            calls = (ExtractionCall(chars=len(text), density=density),) if text.strip() else ()
            return SnapshotRead(snapshot_id=snapshot_id, calls=calls, whole_chars=len(text))
        units = [
            ChunkUnit(chunk_id=f"chunk_{i}", chunk_text=t) for i, t in enumerate(chunks, start=1)
        ]

    cap = get_config().extraction.max_llm_calls_per_source
    calls_list: list[ExtractionCall] = []
    carried = capped = 0
    for unit in units:
        held = await get_active_particles_for_chunk_hash(
            session,
            corpus_entry_id=entry_id,
            chunk_hash=unit.prompt_hash,
            extractor_id=GENERAL_EXTRACTOR_ID,
            extractor_version=GENERAL_EXTRACTOR_VERSION,
        )
        if held:
            carried += 1
        elif len(calls_list) >= cap:
            capped += 1
        else:
            calls_list.append(
                ExtractionCall(
                    chars=len(unit.chunk_text),
                    context_chars=len(unit.context or ""),
                    density=density,
                )
            )
    return SnapshotRead(
        snapshot_id=snapshot_id,
        calls=tuple(calls_list),
        delta=delta,
        whole_chars=len(text),
        carried=carried,
        capped=capped,
    )


def _from_bytes(snapshot_id: str, size: int, density: float) -> SnapshotRead:
    sizes = chunk_sizes(size)
    return SnapshotRead(
        snapshot_id=snapshot_id,
        calls=tuple(ExtractionCall(chars=n, density=density) for n in sizes),
        whole_chars=size,
        capped=max(0, -(-size // get_config().extraction.html_chunk_size) - len(sizes)),
    )

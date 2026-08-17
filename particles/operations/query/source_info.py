"""Provenance-lookup helper for recency decay and exporter rendering.

``load_source_rows`` resolves each particle's SOURCE provenance ref to its
snapshot (``content_published_at``, ``author_id``) and corpus entry
(``source_type``, ``entry_id``, ``uri_r``) in three batch queries, keyed by
particle id — the ranker applies ``recency_factor`` per particle and
evaluates the ``TrustPolicy`` against the entry + author fields.

This module also exports the public ``load_source_info`` symbol that
``particles/exporters/obsidian/vault.py`` imports for note-rendering. Keep
the signature stable — it remains the two-field
``(content_published_at, source_type)`` view over ``load_source_rows``.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import Particle, ProvenanceRefType

#: (content_published_at, source_type, corpus_entry_id, uri_r, author_id)
SourceRow = tuple[datetime | None, str, str | None, str | None, str | None]


async def load_source_rows(
    session: AsyncSession,
    particles: list[Particle],
) -> dict[str, SourceRow]:
    """Return {particle_id: (content_published_at, source_type, entry_id, uri_r, author_id)}.

    Resolves each particle's SOURCE provenance ref to its snapshot and corpus
    entry in three batch queries (snapshot meta, entry source_types, entry
    URIs). ``author_id`` comes from the SOURCE snapshot (§6.5) and feeds the
    §6.4 AUTHOR trust tier.
    """
    # Collect unique snapshot_ids and entry_ids from SOURCE provenance refs
    snap_id_to_particle: dict[str, str] = {}  # snapshot_id → particle_id
    entry_id_to_particle: dict[str, str] = {}  # entry_id → particle_id
    for p in particles:
        src = next((r for r in p.provenance if r.type == ProvenanceRefType.SOURCE), None)
        if src:
            if src.snapshot_id:
                snap_id_to_particle[src.snapshot_id] = p.id
            if src.corpus_entry_id:
                entry_id_to_particle[src.corpus_entry_id] = p.id

    if not snap_id_to_particle and not entry_id_to_particle:
        return {}

    from particles.corpus.store import (
        get_entry_uri_map,
        get_snapshot_source_meta,
        get_source_types_for_entries,
    )

    snap_meta = await get_snapshot_source_meta(session, list(snap_id_to_particle))
    entry_source_types = await get_source_types_for_entries(session, list(entry_id_to_particle))
    entry_uris = await get_entry_uri_map(session, set(entry_id_to_particle))

    # Build result dict keyed by particle_id
    result: dict[str, SourceRow] = {}
    for p in particles:
        src = next((r for r in p.provenance if r.type == ProvenanceRefType.SOURCE), None)
        if src:
            pub_at, author_id = (
                snap_meta.get(src.snapshot_id or "", (None, None))
                if src.snapshot_id
                else (None, None)
            )
            src_type = entry_source_types.get(src.corpus_entry_id, "")
            uri_r = entry_uris.get(src.corpus_entry_id)
            result[p.id] = (pub_at, src_type, src.corpus_entry_id, uri_r, author_id)

    return result


async def load_source_info(
    session: AsyncSession,
    particles: list[Particle],
) -> dict[str, tuple[datetime | None, str]]:
    """Return {particle_id: (content_published_at, source_type)} for recency decay.

    Stable two-field view over :func:`load_source_rows` — exporters that only
    need publication time and source type keep this signature.
    """
    rows = await load_source_rows(session, particles)
    return {pid: (pub_at, src_type) for pid, (pub_at, src_type, _, _, _) in rows.items()}

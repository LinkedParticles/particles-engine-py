# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Store reads and the one writer of the episode reading.

The reading is a curated annotation in ``properties``: the writer
touches the two ``extraction:episode*`` keys and nothing else, never
``content``, ``confidence``, provenance or ``status``. No new table
and no migration: the precedent is ``extraction:valid_from_basis``.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.episode import (
    EPISODE_KEY,
    EPISODE_MODEL_KEY,
    EPISODE_RULE_KEY,
    OPERATOR_RULE,
    EpisodeReading,
    reading_of,
)
from particles.core.rung_ladder import EpisodeVerdict
from particles.core.status import Status
from particles.core.validity import VALID_FROM_BASIS_KEY
from particles.corpus.store import get_source_types_for_entries
from particles.store.particle_store import ParticleRow, ProvenanceEdgeRow
from particles.store.utility_store import UtilityEventRow

__all__ = [
    "EpisodeRow",
    "EpisodeSource",
    "credited_ids",
    "get_episode_rows",
    "set_episode_reading",
]

_CHUNK = 500


@dataclass(frozen=True)
class EpisodeSource:
    """One provenance edge: the entry, its type, and where the claim sat in it."""

    corpus_entry_id: str
    source_type: str | None
    snapshot_id: str | None
    chunk_hash: str | None


@dataclass(frozen=True)
class EpisodeRow:
    """What the episode question reads about one stored claim."""

    particle_id: str
    content: str
    asserted: bool
    """No extractor stamp: an operator or agent assertion."""
    valid_from_basis: str | None
    reading: EpisodeReading | None
    sources: tuple[EpisodeSource, ...]

    @property
    def entries(self) -> frozenset[str]:
        """The claim's source entries."""
        return frozenset(s.corpus_entry_id for s in self.sources)

    def session_only(self, session_types: Collection[str]) -> bool:
        """Every source is an entry of a session type."""
        return bool(self.sources) and all(s.source_type in session_types for s in self.sources)


async def get_episode_rows(
    session: AsyncSession,
    particle_ids: Collection[str] | None = None,
    *,
    statuses: Sequence[Status] = (Status.ACTIVE,),
) -> list[EpisodeRow]:
    """Column reads of the claims named (every claim in ``statuses`` when ``None``)."""
    status_values = [s.value for s in statuses]
    base = select(
        ParticleRow.id,
        ParticleRow.content,
        ParticleRow.extractor_ref_json,
        ParticleRow.properties_json,
        ParticleRow.extraction_provider_model,
    ).where(ParticleRow.status.in_(status_values))
    raw: list[tuple[str, str, str | None, str | None, str | None]] = []
    if particle_ids is None:
        raw.extend((r[0], r[1], r[2], r[3], r[4]) for r in (await session.execute(base)).all())
    else:
        ids = list(dict.fromkeys(particle_ids))
        for i in range(0, len(ids), _CHUNK):
            stmt = base.where(ParticleRow.id.in_(ids[i : i + _CHUNK]))
            raw.extend((r[0], r[1], r[2], r[3], r[4]) for r in (await session.execute(stmt)).all())

    edges: dict[str, list[tuple[str, str | None, str | None]]] = {}
    found = [r[0] for r in raw]
    for i in range(0, len(found), _CHUNK):
        rows = await session.execute(
            select(
                ProvenanceEdgeRow.particle_id,
                ProvenanceEdgeRow.corpus_entry_id,
                ProvenanceEdgeRow.snapshot_id,
                ProvenanceEdgeRow.chunk_hash,
            ).where(ProvenanceEdgeRow.particle_id.in_(found[i : i + _CHUNK]))
        )
        for pid, entry_id, snapshot_id, chunk_hash in rows.all():
            edges.setdefault(pid, []).append((entry_id, snapshot_id, chunk_hash))
    entry_ids = sorted({e[0] for es in edges.values() for e in es})
    types = await get_source_types_for_entries(session, entry_ids)

    out: list[EpisodeRow] = []
    for pid, content, extractor_ref_json, props_json, extraction_model in raw:
        props: dict[str, object] = {}
        if props_json:
            try:
                loaded = json.loads(props_json)
                props = loaded if isinstance(loaded, dict) else {}
            except ValueError:
                props = {}
        basis = props.get(VALID_FROM_BASIS_KEY)
        out.append(
            EpisodeRow(
                particle_id=pid,
                content=content,
                asserted=extractor_ref_json is None,
                valid_from_basis=str(basis) if basis is not None else None,
                # An extracted reading was served by the particle's own
                # extraction model, which the row already records.
                reading=reading_of(props, extraction_model),
                sources=tuple(
                    EpisodeSource(entry_id, types.get(entry_id), snapshot_id, chunk_hash)
                    for entry_id, snapshot_id, chunk_hash in sorted(edges.get(pid, []))
                ),
            )
        )
    return out


async def credited_ids(session: AsyncSession, particle_ids: Collection[str]) -> set[str]:
    """The ids among ``particle_ids`` the use ledger credits with a use."""
    ids = list(particle_ids)
    used: set[str] = set()
    for i in range(0, len(ids), _CHUNK):
        rows = await session.execute(
            select(UtilityEventRow.particle_id)
            .where(UtilityEventRow.particle_id.in_(ids[i : i + _CHUNK]))
            .distinct()
        )
        used.update(rows.scalars())
    return used


async def set_episode_reading(
    session: AsyncSession,
    particle_id: str,
    *,
    verdict: EpisodeVerdict,
    rule: str,
    model: str | None = None,
    preserve_operator: bool = True,
) -> bool:
    """Write one claim's episode reading and the model that served it.

    Returns whether anything was written.

    The row is re-read here, at write time, so an operator reading recorded
    after a run computed its scope is never overwritten when
    ``preserve_operator`` holds. ``ABSTAIN`` is never stored.

    Raises:
        ValueError: If ``verdict`` is ``ABSTAIN``.
    """
    if verdict is EpisodeVerdict.ABSTAIN:
        raise ValueError("an episode reading is session_bound or standing, never abstain")
    row = await session.get(ParticleRow, particle_id)
    if row is None:
        return False
    loaded = json.loads(row.properties_json) if row.properties_json else {}
    properties: dict[str, object] = loaded if isinstance(loaded, dict) else {}
    if preserve_operator and properties.get(EPISODE_RULE_KEY) == OPERATOR_RULE:
        return False
    properties[EPISODE_KEY] = verdict.value
    properties[EPISODE_RULE_KEY] = rule
    if model:
        properties[EPISODE_MODEL_KEY] = model
    else:
        properties.pop(EPISODE_MODEL_KEY, None)
    row.properties_json = json.dumps(properties)
    return True

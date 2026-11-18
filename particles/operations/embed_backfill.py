# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The consolidation embed pass: vectors for beliefs stored without one.

A belief written while the embedding encoder could not load is stored with
``embedding_json = NULL``, and every cosine read (the query, the duplicate
finders, the census, the update sweep) skips it for as long as it is ACTIVE.
The vector is derived data, a function of the belief's content and the
encoder, so restoring it needs no model call and no judgement.

The pass gathers the oldest such beliefs (REVIEW records left out, at most
``consolidation.embed.max_per_run``), encodes them with the current model off
the event loop, and stores each vector with one conditional write that never
overwrites a vector already there. It records one ``EMBEDDING_BACKFILLED``
event per run that filled any, a PARTICLE ref per belief, which the delta scope
reads (:func:`embedding_backfilled_particle_ids_since`), so the sweeps and the
census pair the filled beliefs in that run and the next. The pass itself
changes no status and retires nothing.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.db import write_transaction
from particles.embeddings import get_embedding_model, get_embedding_model_id
from particles.store.event_store import (
    EventRefKind,
    OperatorEventType,
    list_events_since,
    record_event,
)
from particles.store.particle_store import (
    count_active_particles_with_other_embedding_model,
    count_active_particles_without_embedding,
    fill_particle_embedding,
    list_active_particles_without_embedding,
)

log = logging.getLogger(__name__)

#: Why the pass stored nothing when no encoder loads.
NO_ENCODER = "no embedding encoder could be loaded"


class EmbedBackfillReport(BaseModel):
    """What one embed pass did."""

    #: Beliefs given a vector this run.
    filled: int = 0
    #: ACTIVE non-REVIEW beliefs still without a vector after the pass.
    still_missing: int = 0
    #: ACTIVE rows whose vector came from another model, which every loader
    #: skips until the re-embed runs.
    other_model: int = 0
    #: Selected, but changed by another writer before the write: given a
    #: vector, or retired.
    refused: int = 0
    #: Why nothing was stored, when the pass could not run.
    skipped_reason: str | None = None
    #: The ``EMBEDDING_BACKFILLED`` event, when one was written.
    event_id: str | None = None

    def payload(self) -> dict[str, Any]:
        """The run record's ``embed`` block."""
        return {
            "filled": self.filled,
            "still_missing": self.still_missing,
            "other_model": self.other_model,
            "refused": self.refused,
            "skipped_reason": self.skipped_reason,
        }


async def run_embed_backfill(session: AsyncSession, *, actor: str) -> EmbedBackfillReport:
    """Fill the vectors of the oldest ACTIVE beliefs stored without one.

    Encodes outside the writer lock and writes every vector, with the event
    naming them, in one write transaction. With no encoder it stores nothing
    and says why; the beliefs wait for the next run.
    """
    report = EmbedBackfillReport()
    limit = get_config().consolidation.embed.max_per_run
    rows = await list_active_particles_without_embedding(session, limit=limit)
    current_id = get_embedding_model_id()
    model = get_embedding_model() if rows else None
    if rows and model is None:
        report.skipped_reason = NO_ENCODER
    elif rows and model is not None:
        texts = [content for _, content in rows]
        encoded = await asyncio.to_thread(
            model.encode, texts, convert_to_numpy=True, normalize_embeddings=True
        )
        async with write_transaction(session):
            filled: list[str] = []
            for (particle_id, _), vector in zip(rows, encoded, strict=True):
                if await fill_particle_embedding(session, particle_id, vector.tolist()):
                    filled.append(particle_id)
                else:
                    report.refused += 1
            report.filled = len(filled)
            if filled:
                event = await record_event(
                    session,
                    actor=actor,
                    event_type=OperatorEventType.EMBEDDING_BACKFILLED,
                    refs=[(EventRefKind.PARTICLE, pid) for pid in filled],
                    payload={"filled": report.filled, "model_id": current_id},
                )
                report.event_id = event.event_id
    # The counts are store-wide scans, so they run after the writer lock is
    # released. An empty selection under a positive cap already says none is
    # missing.
    if rows or limit <= 0:
        report.still_missing = await count_active_particles_without_embedding(session)
    report.other_model = await count_active_particles_with_other_embedding_model(
        session, current_id
    )
    return report


async def embedding_backfilled_particle_ids_since(
    session: AsyncSession, since: datetime
) -> set[str]:
    """Beliefs the embed pass gave a vector after ``since``.

    The delta scope's fourth term: a filled belief keeps the ``asserted_at`` it
    was written with, so without this it would enter no run's scope and the
    census would pair it only when a partner happened to be in scope.
    """
    events = await list_events_since(
        session, since=since, event_types=[OperatorEventType.EMBEDDING_BACKFILLED]
    )
    return {
        ref.ref_id
        for event in events
        for ref in event.refs
        if ref.ref_kind == EventRefKind.PARTICLE
    }

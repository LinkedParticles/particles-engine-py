# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Store fixtures for observer-scope tests: sources with keys, beliefs from them."""

from __future__ import annotations

from typing import Any

import numpy as np

from particles.core.schema import (
    Confidence,
    Mutability,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status

# One shared embedding, so presence / absence is decided by the filter, never by ranking.
EMB = (np.ones(4, dtype=np.float32) / 2.0).tolist()


async def source(session: Any, name: str, tags: list[str]) -> tuple[str, str]:
    """Deposit one source carrying ``tags``; returns ``(entry_id, snapshot_id)``."""
    from particles.corpus.deposit import deposit_text_versioned

    entry_id, snapshot_id, _ = await deposit_text_versioned(
        session,
        text=f"source text of {name}",
        uri_r=f"test://{name}",
        source_type="LOCAL_MARKDOWN",
        mutability=Mutability.MUTABLE,
        tags=tags,
    )
    return entry_id, snapshot_id


def project_source_tags(key: str) -> list[str]:
    return ["claude-code", "memory-file", f"project:{key}"]


async def belief(
    session: Any,
    content: str,
    *sources: tuple[str, str],
    premises: tuple[str, ...] = (),
    status: Status = Status.ACTIVE,
    **fields: Any,
) -> Particle:
    """Insert a belief attested by ``sources`` — or derived from ``premises`` (particle ids)."""
    from particles.store.particle_store import insert_particle

    provenance = [
        ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=e, snapshot_id=s)
        for e, s in sources
    ] + [
        ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=pid, snapshot_id=pid)
        for pid in premises
    ]
    particle = Particle(
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test-agent",
        status=status,
        provenance=provenance,
        **fields,
    )
    await insert_particle(session, particle, EMB)
    return particle


async def rescoped(session: Any) -> None:
    """Record the marker that lets a project observer engage on this store."""
    from particles.operations.observer_scope import rescope

    await rescope(session, key_for=lambda _entry_id, _uri, _tags: None)


async def generation(
    session: Any,
    name: str,
    text: str,
    tags: list[str],
    *,
    mutability: Mutability = Mutability.MUTABLE,
    captured_at: Any = None,
) -> tuple[str, str]:
    """Deposit ``text`` as the next generation of source ``name`` and mark it extracted.

    Unlike :func:`source`, the snapshot is ``COMPLETE``, so it is the entry's
    latest extracted generation — what "currently states" reads.
    """
    from particles.core.schema import ExtractionStatus
    from particles.corpus.deposit import deposit_text_versioned
    from particles.corpus.store import SnapshotRow, update_extraction_status

    entry_id, snapshot_id, _ = await deposit_text_versioned(
        session,
        text=text,
        uri_r=f"test://{name}",
        source_type="LOCAL_MARKDOWN",
        mutability=mutability,
        tags=tags,
    )
    await update_extraction_status(session, snapshot_id, ExtractionStatus.COMPLETE)
    if captured_at is not None:
        row = await session.get(SnapshotRow, snapshot_id)
        row.captured_at = captured_at
    await session.flush()
    return entry_id, snapshot_id

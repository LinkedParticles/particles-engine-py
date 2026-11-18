# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The pipeline's candidate insert checks the status it writes (D3).

The plain insert and the unchecked route of ``apply_reconcile`` (and the same
branches of the extract pass) once validated a constant ``None -> ACTIVE`` and
then wrote the candidate's own status. Each now validates the status written,
so a candidate the transition table does not allow from creation is refused
before anything is stored.
"""

from __future__ import annotations

import hashlib
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.extraction.polarity import POLARITY_DECLINED, POLARITY_KEY


def _encode(texts: Any, **_kwargs: Any) -> Any:
    items = [texts] if isinstance(texts, str) else list(texts)
    return np.array(
        [[(b / 127.5) - 1.0 for b in hashlib.sha256(str(t).encode()).digest()[:16]] for t in items],
        dtype=np.float32,
    )


@pytest.fixture
def stub_encoder() -> Generator[None, None, None]:
    from particles import embeddings as ep

    model = MagicMock()
    model.encode = MagicMock(side_effect=_encode)
    original = ep._embedding_model
    ep.set_embedding_model(model)
    try:
        yield
    finally:
        ep.set_embedding_model(original)


def _candidate(content: str, status: Status, **fields: Any) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e", snapshot_id="s")
        ],
        asserted_by="test-writer",
        status=status,
        **fields,
    )


async def _count(session: Any) -> int:
    from sqlalchemy import func, select

    from particles.store.particle_store import ParticleRow

    return int((await session.execute(select(func.count()).select_from(ParticleRow))).scalar_one())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "properties",
    [None, {POLARITY_KEY: POLARITY_DECLINED}],
    ids=["insert", "unchecked"],
)
@pytest.mark.parametrize("status", [Status.RETRACTED, Status.SUPERSEDED])
async def test_a_candidate_born_in_a_status_the_table_refuses_is_not_written(
    db_session: Any, stub_encoder: None, status: Status, properties: dict[str, Any] | None
) -> None:
    """A candidate whose status no creation transition reaches is refused on both the
    plain insert and the unchecked route, and the store stays empty."""
    from particles.db import write_transaction
    from particles.ingest.pipeline import reconcile_and_insert

    reason = (
        StatusReason.EXPLICIT_RETRACTION
        if status is Status.RETRACTED
        else StatusReason.EXPLICIT_SUPERSESSION
    )
    candidate = _candidate(
        "The refused widget is green.", status, status_reason=reason, properties=properties
    )
    with pytest.raises(ValueError):
        async with write_transaction(db_session):
            await reconcile_and_insert(db_session, candidate)
    await db_session.rollback()
    assert await _count(db_session) == 0


@pytest.mark.asyncio
async def test_control_an_active_candidate_is_written(db_session: Any, stub_encoder: None) -> None:
    """Control: an ACTIVE candidate passes the same check and lands."""
    from particles.db import write_transaction
    from particles.ingest.pipeline import reconcile_and_insert

    async with write_transaction(db_session):
        written = await reconcile_and_insert(
            db_session, _candidate("The written widget is green.", Status.ACTIVE)
        )
    assert written is not None and written.status is Status.ACTIVE
    assert await _count(db_session) == 1

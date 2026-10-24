# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""One living note, extracted through the real pipeline with no LLM.

Shared by the cascade tests (``test_generation_cascade.py``) and the reindex
tests (``test_reindex.py``): a scripted extractor that returns fixed claims
per deposited text, an embedding model that keeps every claim apart, and the
two pair probes stubbed to "no contradiction, no update", so the only thing
deciding what stays ACTIVE is the generation cascade.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from particles.core.schema import Mutability, UncertaintyNature
from particles.core.status import Status

KEPT = "The author keeps a weekly review."
DROPPED = "The author files receipts in a shoebox."
ADDED = "The author scans receipts on arrival."


class Hashed:
    """Each distinct text embeds to its own unit vector, so no two claims are near."""

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i, hashlib.sha256(text.encode()).digest()[0] % 64] = 1.0
        return out


class Scripted:
    """An extractor emitting scripted claims per deposited text, keeping each result."""

    EXTRACTOR_ID = "test-extractor"
    EXTRACTOR_VERSION = "1"

    def __init__(self) -> None:
        self.claims: dict[bytes, list[str]] = {}
        self.results: list[Any] = []

    def accepts(self, source_type: str) -> bool:
        return True

    async def extract(self, snapshot: Any, content: bytes, **kwargs: object) -> Any:
        from particles.extraction.general import CandidateParticle, ExtractionResult

        result = ExtractionResult(
            candidates=[
                CandidateParticle(
                    content=claim,
                    confidence_value=0.9,
                    uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    subjects=["the author"],
                )
                for claim in self.claims[content]
            ]
        )
        self.results.append(result)
        return result

    def notes(self) -> list[str]:
        return [n for r in self.results for n in r.quality_notes]


@pytest.fixture
def quiet_probes() -> Generator[None, None, None]:
    """No pair contradicts or updates, and embeddings keep every claim apart: no LLM."""
    from particles import embeddings as ep
    from particles.core.conflict_resolution import SlotVerdict

    original = ep._embedding_model
    ep.set_embedding_model(Hashed())  # type: ignore[arg-type]
    try:
        with (
            patch(
                "particles.ingest.pipeline._has_contradiction_signal",
                AsyncMock(return_value=False),
            ),
            patch(
                "particles.ingest.pipeline._has_update_signal",
                AsyncMock(return_value=SlotVerdict.DIFFERENT),
            ),
        ):
            yield
    finally:
        ep.set_embedding_model(original)


def note_uri() -> str:
    return f"file:///vault/{uuid.uuid4().hex}.md"


async def deposit_version(
    session: Any, ex: Scripted, uri: str, claims: list[str], *, day: int
) -> tuple[str, str]:
    """Deposit one version of the note at ``uri``; returns ``(entry_id, snapshot_id)``."""
    from particles.corpus.deposit import deposit_text_versioned

    text = f"version of day {day}: " + " ".join(claims)
    ex.claims[text.encode()] = claims
    entry_id, snapshot_id, unchanged = await deposit_text_versioned(
        session,
        text=text,
        uri_r=uri,
        source_type="LOCAL_MARKDOWN",
        mutability=Mutability.MUTABLE,
        content_published_at=datetime(2026, 6, day, tzinfo=UTC),
    )
    assert not unchanged
    await session.commit()
    return entry_id, snapshot_id


async def extract(session: Any, ex: Scripted, entry_id: str, snapshot_id: str) -> None:
    from particles.ingest.pipeline import extract_snapshot

    await extract_snapshot(session, entry_id, snapshot_id, extractor=ex)
    await session.commit()


async def active_copies(session: Any, content: str) -> int:
    from particles.store.particle_store import get_particles_by_status

    return sum(p.content == content for p in await get_particles_by_status(session, Status.ACTIVE))

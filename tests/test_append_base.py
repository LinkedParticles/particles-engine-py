# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The append-only delta read through the real pipeline.

Each test deposits a transcript as successive ``APPEND_ONLY`` snapshots with
``deposit_text_versioned`` and extracts them with ``extract_snapshot``. The
LLM is mocked at the extractor's two call seams (``general._call_llm`` for a
single call, ``incremental._call_llm`` for every chunked and delta call), and
the fake records the text and context each call was sent, so each test can say
exactly what the model read. The embedding model returns a distinct vector per
text, so no candidate pairs with another and no contradiction probe runs.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles import embeddings as ep
from particles.config import get_config
from particles.core.schema import (
    Confidence,
    ExtractionStatus,
    ExtractorRef,
    Mutability,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.corpus.deposit import deposit_text_versioned
from particles.corpus.store import SnapshotRow, update_extraction_status
from particles.extraction import general
from particles.extraction.general import CandidateParticle
from particles.ingest.append_base import (
    FALLBACK_DISABLED,
    FALLBACK_NO_BASE,
    FALLBACK_OTHER_EXTRACTOR,
    FALLBACK_RAW_PREFIX,
)
from particles.ingest.pipeline import SnapshotOutcome, extract_snapshot
from particles.store.particle_store import get_particles_for_entry, insert_particle


class FakeLLM:
    """Records every extraction call and answers each with one claim naming it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    async def __call__(
        self, text: str, *args: Any, **kwargs: Any
    ) -> tuple[list[CandidateParticle], list[str], bool]:
        context = kwargs.get("context")
        self.calls.append((text, context))
        claim = CandidateParticle(
            content=f"Claim {len(self.calls)}: {text.strip().splitlines()[0][:60]}",
            confidence_value=0.8,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
        )
        return [claim], [], False

    def reset(self) -> None:
        self.calls.clear()


def _vector(text: str) -> list[float]:
    raw = np.frombuffer(hashlib.sha256(text.encode()).digest() * 4, dtype=np.uint8)
    vec = raw.astype(np.float32) - 127.5
    return list(vec / np.linalg.norm(vec))


@contextmanager
def fake_extraction_llm() -> Iterator[FakeLLM]:
    """Patch both extraction call seams and the embedding model; yield the recorder."""
    fake = FakeLLM()
    model = MagicMock()
    model.encode = MagicMock(
        side_effect=lambda texts, **_: np.array([_vector(t) for t in texts], dtype=np.float32)
    )
    original = ep._embedding_model
    ep.set_embedding_model(model)
    with (
        patch("particles.extraction.general._call_llm", fake),
        patch("particles.extraction.incremental._call_llm", fake),
    ):
        yield fake
    ep.set_embedding_model(original)


@pytest.fixture
def llm() -> Iterator[FakeLLM]:
    with fake_extraction_llm() as fake:
        yield fake


URI = "claude-code://session/adr-0287"


async def _deposit(
    session: AsyncSession, text: str, *, mutability: Mutability = Mutability.APPEND_ONLY
) -> tuple[str, str]:
    entry_id, snapshot_id, unchanged = await deposit_text_versioned(
        session,
        text=text,
        uri_r=URI,
        source_type="CONVERSATION",
        mutability=mutability,
        deposited_by="test",
    )
    assert not unchanged
    await session.commit()
    return entry_id, snapshot_id


async def _extract(
    session: AsyncSession, entry_id: str, snapshot_id: str
) -> tuple[list[Particle], SnapshotOutcome]:
    outcome = SnapshotOutcome()
    written = await extract_snapshot(session, entry_id, snapshot_id, outcome_out=outcome)
    await session.commit()
    return written, outcome


async def _extracted_through(session: AsyncSession, snapshot_id: str) -> int | None:
    row = (
        await session.execute(
            select(SnapshotRow)
            .where(SnapshotRow.snapshot_id == snapshot_id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one()
    assert row.extraction_status == ExtractionStatus.COMPLETE.value
    return row.extracted_through


def _turn(tag: str, words: int = 8) -> str:
    """One transcript turn, a single paragraph."""
    return f"User: {tag} " + " ".join(f"{tag.lower()}{i}" for i in range(words)) + "\n"


FIRST = _turn("Alpha") + "\n" + _turn("Beta")
SECOND = FIRST + "\n" + _turn("Gamma")
THIRD = SECOND + "\n" + _turn("Delta")


class TestDeltaRead:
    @pytest.mark.asyncio
    async def test_the_model_reads_only_the_new_text_with_labelled_context(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        entry_id, s1 = await _deposit(db_session, FIRST)
        await _extract(db_session, entry_id, s1)
        assert await _extracted_through(db_session, s1) == len(FIRST.encode())

        _, s2 = await _deposit(db_session, SECOND)
        llm.reset()
        _, outcome = await _extract(db_session, entry_id, s2)

        [(text, context)] = llm.calls
        assert "Gamma" in text
        assert "Alpha" not in text and "Beta" not in text
        assert context is not None and context.endswith(_turn("Beta").strip())
        assert outcome.append_fallback == 0
        assert await _extracted_through(db_session, s2) == len(SECOND.encode())

    @pytest.mark.asyncio
    async def test_the_first_snapshot_has_no_base_and_is_counted(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        entry_id, s1 = await _deposit(db_session, FIRST)
        _, outcome = await _extract(db_session, entry_id, s1)

        [(text, context)] = llm.calls
        assert "Alpha" in text and context is None
        assert outcome.append_fallback == 1
        assert outcome.append_fallback_reason == FALLBACK_NO_BASE

    @pytest.mark.asyncio
    async def test_no_new_text_makes_no_call(self, db_session: AsyncSession, llm: FakeLLM) -> None:
        entry_id, s1 = await _deposit(db_session, FIRST)
        await _extract(db_session, entry_id, s1)
        _, s2 = await _deposit(db_session, FIRST + "\n\n")
        llm.reset()
        written, outcome = await _extract(db_session, entry_id, s2)
        assert llm.calls == [] and written == [] and outcome.append_fallback == 0


class TestFallbacks:
    @pytest.mark.asyncio
    async def test_raw_prefix_mismatch(self, db_session: AsyncSession, llm: FakeLLM) -> None:
        entry_id, s1 = await _deposit(db_session, FIRST)
        await _extract(db_session, entry_id, s1)
        # The promise broke: the earlier text was rewritten, not appended to.
        _, s2 = await _deposit(db_session, FIRST.replace("Alpha", "Omega") + "\n" + _turn("Gamma"))
        llm.reset()
        written, outcome = await _extract(db_session, entry_id, s2)

        [(text, context)] = llm.calls
        assert "Omega" in text and context is None
        assert outcome.append_fallback == 1
        assert outcome.append_fallback_reason == FALLBACK_RAW_PREFIX

    @pytest.mark.asyncio
    async def test_decoded_prefix_mismatch(
        self, db_session: AsyncSession, llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry_id, s1 = await _deposit(db_session, FIRST)
        await _extract(db_session, entry_id, s1)
        _, s2 = await _deposit(db_session, SECOND)

        # A decoder that no longer decodes the prefix into a prefix of the text.
        from particles.extraction import append_delta

        real = append_delta.extraction_text

        def changed(content: bytes, **kwargs: bool) -> tuple[str, int]:
            text, turns = real(content, **kwargs)
            return text + "\n[decoder changed]", turns

        monkeypatch.setattr(append_delta, "extraction_text", changed)
        llm.reset()
        _, outcome = await _extract(db_session, entry_id, s2)

        [(text, _)] = llm.calls
        assert "Alpha" in text and "Gamma" in text
        assert outcome.append_fallback == 1
        assert outcome.append_fallback_reason == append_delta.FALLBACK_DECODED_PREFIX

    @pytest.mark.asyncio
    async def test_base_from_another_extractor(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        entry_id, s1 = await _deposit(db_session, FIRST)
        # s1 was extracted by another extractor: its claims say so.
        await insert_particle(db_session, _claim(entry_id, s1, "journal-extractor"))
        await update_extraction_status(
            db_session, s1, ExtractionStatus.COMPLETE, extracted_through=len(FIRST.encode())
        )
        await db_session.commit()
        _, s2 = await _deposit(db_session, SECOND)

        _, outcome = await _extract(db_session, entry_id, s2)

        [(text, _)] = llm.calls
        assert "Alpha" in text
        assert outcome.append_fallback == 1
        assert outcome.append_fallback_reason is not None
        assert outcome.append_fallback_reason.startswith(FALLBACK_OTHER_EXTRACTOR)

    @pytest.mark.asyncio
    async def test_no_complete_earlier_snapshot(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        entry_id, _s1 = await _deposit(db_session, FIRST)  # never extracted: PENDING
        _, s2 = await _deposit(db_session, SECOND)
        _, outcome = await _extract(db_session, entry_id, s2)

        [(text, _)] = llm.calls
        assert "Alpha" in text and "Gamma" in text
        assert outcome.append_fallback_reason == FALLBACK_NO_BASE

    @pytest.mark.asyncio
    async def test_the_knob_off_restores_the_whole_read(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        get_config().extraction.append_only_delta = False
        entry_id, s1 = await _deposit(db_session, FIRST)
        await _extract(db_session, entry_id, s1)
        _, s2 = await _deposit(db_session, SECOND)
        llm.reset()
        _, outcome = await _extract(db_session, entry_id, s2)

        [(text, context)] = llm.calls
        assert "Alpha" in text and "Gamma" in text and context is None
        assert outcome.append_fallback_reason == FALLBACK_DISABLED


class TestExtractedThrough:
    @pytest.mark.asyncio
    async def test_a_capped_read_stops_and_the_next_delta_starts_there(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        cfg = get_config().extraction
        cfg.append_chunk_chars = 1000
        para = [_turn(tag, 120) for tag in ("Pone", "Ptwo", "Pthr")]
        assert all(700 < len(p) < 1000 for p in para), "setup: one paragraph per chunk"
        second = FIRST + "\n" + "\n".join(para)

        entry_id, s1 = await _deposit(db_session, FIRST)
        await _extract(db_session, entry_id, s1)
        _, s2 = await _deposit(db_session, second)
        cfg.max_llm_calls_per_source = 1
        llm.reset()
        await _extract(db_session, entry_id, s2)

        [(text, _)] = llm.calls
        assert "Pone" in text
        through = await _extracted_through(db_session, s2)
        raw = second.encode()
        assert through == raw.index(b"User: Ptwo")

        _, s3 = await _deposit(db_session, second + "\n" + _turn("Four"))
        cfg.max_llm_calls_per_source = 8
        llm.reset()
        await _extract(db_session, entry_id, s3)

        texts = [text for text, _ in llm.calls]
        assert texts[0].startswith("User: Ptwo")
        assert not any("Pone" in t for t in texts)
        assert any("Four" in t for t in texts)

    @pytest.mark.asyncio
    async def test_a_null_base_takes_the_offset_its_chunk_hashes_derive(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        from particles.extraction.general import (
            _normalise_for_hashing,
            _split_into_paragraph_chunks,
        )
        from particles.extraction.incremental import _hash_chunk

        get_config().extraction.html_chunk_size = 1000
        para = [_turn(tag, 90) for tag in ("Qone", "Qtwo", "Qthr")]
        first = "\n".join(para)
        entry_id, s1 = await _deposit(db_session, first)
        chunks = _split_into_paragraph_chunks(_normalise_for_hashing(first), 1000)
        assert len(chunks) == 3
        # Extracted before: COMPLETE, no extracted_through, and its
        # claims cover only the first chunk (the read was cut there).
        await insert_particle(
            db_session,
            _claim(entry_id, s1, general.EXTRACTOR_ID, chunk_hash=_hash_chunk(chunks[0])),
        )
        await update_extraction_status(db_session, s1, ExtractionStatus.COMPLETE)
        await db_session.commit()

        _, s2 = await _deposit(db_session, first + "\n" + _turn("Next"))
        await _extract(db_session, entry_id, s2)

        texts = [text for text, _ in llm.calls]
        assert texts[0].startswith("User: Qtwo")
        assert not any("Qone" in t for t in texts)


class TestEarlierText:
    async def _chunked_first(self, db_session: AsyncSession) -> tuple[str, str, str]:
        get_config().extraction.html_chunk_size = 1000
        first = "\n".join(_turn(tag, 90) for tag in ("Rone", "Rtwo"))
        entry_id, s1 = await _deposit(db_session, first)
        await _extract(db_session, entry_id, s1)
        return entry_id, s1, first

    @pytest.mark.asyncio
    async def test_no_reobservation_ref_is_appended_to_earlier_claims(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        entry_id, s1, first = await self._chunked_first(db_session)
        _, s2 = await _deposit(db_session, first + "\n" + _turn("Rthree"))
        await _extract(db_session, entry_id, s2)

        earlier = [
            p
            for p in await get_particles_for_entry(db_session, entry_id)
            if p.provenance[0].snapshot_id == s1
        ]
        assert earlier
        assert all(len(p.provenance) == 1 for p in earlier)

    @pytest.mark.asyncio
    async def test_a_version_bump_does_not_reread_earlier_text(
        self, db_session: AsyncSession, llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry_id, _, first = await self._chunked_first(db_session)
        monkeypatch.setattr(general, "EXTRACTOR_VERSION", "99.0.0")
        _, s2 = await _deposit(db_session, first + "\n" + _turn("Rthree"))
        llm.reset()
        await _extract(db_session, entry_id, s2)

        texts = [text for text, _ in llm.calls]
        assert texts and all("Rone" not in t and "Rtwo" not in t for t in texts)
        # Before the bump re-read every chunk of the earlier text.
        get_config().extraction.append_only_delta = False
        _, s3 = await _deposit(db_session, first + "\n" + _turn("Rthree") + "\n" + _turn("R4"))
        llm.reset()
        await _extract(db_session, entry_id, s3)
        assert any("Rone" in text for text, _ in llm.calls)


class TestOtherMutabilityClasses:
    @pytest.mark.parametrize(
        "mutability", [m for m in Mutability if m is not Mutability.APPEND_ONLY]
    )
    @pytest.mark.asyncio
    async def test_extract_exactly_as_before(
        self, db_session: AsyncSession, llm: FakeLLM, mutability: Mutability
    ) -> None:
        entry_id, s1 = await _deposit(db_session, FIRST, mutability=mutability)
        _, first_outcome = await _extract(db_session, entry_id, s1)
        _, s2 = await _deposit(db_session, SECOND, mutability=mutability)
        llm.reset()
        _, outcome = await _extract(db_session, entry_id, s2)

        [(text, context)] = llm.calls
        assert "Alpha" in text and "Gamma" in text and context is None
        assert first_outcome.append_fallback == 0 and outcome.append_fallback == 0


def _claim(
    entry_id: str, snapshot_id: str, extractor: str, *, chunk_hash: str | None = None
) -> Particle:
    return Particle(
        id=str(uuid.uuid4()),
        content=f"A claim from {snapshot_id[:8]} by {extractor}",
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by=extractor,
        status=Status.ACTIVE,
        extractor_ref=ExtractorRef(name=extractor, version="0.1.0"),
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id=entry_id,
                snapshot_id=snapshot_id,
                chunk_hash=chunk_hash,
            )
        ],
    )

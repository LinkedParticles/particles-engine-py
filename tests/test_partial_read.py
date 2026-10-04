# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""A partly failed append-only read keeps what it answered.

Each test deposits a transcript as ``APPEND_ONLY`` snapshots and extracts them
through the real pipeline. The LLM is mocked at the extractor's call seams
(``general._call_llm``, ``incremental._call_llm`` and, for the pooled path,
``general._pooled_group_complete``). The fake fails every chunk whose own text
carries one of the tags in ``fail``, and answers the rest with one claim
naming the chunk, so a test can say which chunks were written and which the
retry sent again. Paragraphs are sized so each one is exactly one chunk.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
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
    ExtractionStatus,
    Mutability,
    Particle,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.status import Status
from particles.corpus.deposit import deposit_text_versioned
from particles.corpus.store import (
    SnapshotRow,
    get_extraction_component_records,
    release_extraction_claim,
)
from particles.extraction.general import CandidateParticle, ChunkOutcome
from particles.ingest.partial_read import decide_partial_keep, own_chunk_claims
from particles.ingest.pipeline import SnapshotOutcome, extract_snapshot
from particles.store.particle_store import get_particles_for_entry

_TAG = re.compile(r"User: (P\w+) ")


class FakeLLM:
    """Fails each chunk whose own text carries a tag in ``fail``; answers the rest."""

    def __init__(self) -> None:
        self.fail: set[str] = set()
        self.sent: list[str] = []

    def _tag(self, text: str) -> str:
        tags = _TAG.findall(text)
        return tags[-1] if tags else text.strip().splitlines()[0][:20]

    async def __call__(
        self, text: str, *args: Any, **kwargs: Any
    ) -> tuple[list[CandidateParticle], list[str], bool]:
        tag = self._tag(text)
        self.sent.append(tag)
        if tag in self.fail:
            return [], ["API error: unavailable"], True
        claim = CandidateParticle(
            content=f"Claim about {tag} number {len(self.sent)}",
            confidence_value=0.8,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
        )
        return [claim], [], False

    async def pooled(
        self, pool: Any, planned: list[Any], **kwargs: Any
    ) -> tuple[list[str | None], str]:
        out: list[str | None] = []
        for p in planned:
            # The prompt carries the chunk after its context, so the chunk's own
            # tag is the last one in it.
            tag = self._tag(p.request.prompt)
            self.sent.append(tag)
            if tag in self.fail:
                out.append(None)
            else:
                out.append(
                    json.dumps(
                        [
                            {
                                "content": f"Pooled claim about {tag} number {len(self.sent)}",
                                "confidence_value": 0.8,
                                "uncertainty_nature": "EPISTEMIC",
                            }
                        ]
                    )
                )
        return out, "anthropic:test-model"

    def reset(self) -> None:
        self.sent.clear()


def _vector(text: str) -> list[float]:
    raw = np.frombuffer(hashlib.sha256(text.encode()).digest() * 4, dtype=np.uint8)
    vec = raw.astype(np.float32) - 127.5
    return list(vec / np.linalg.norm(vec))


@contextmanager
def fake_llm() -> Iterator[FakeLLM]:
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
        patch("particles.extraction.general._pooled_group_complete", fake.pooled),
    ):
        yield fake
    ep.set_embedding_model(original)


@pytest.fixture
def llm() -> Iterator[FakeLLM]:
    cfg = get_config().extraction
    cfg.append_chunk_chars = 1000
    cfg.html_chunk_size = 1000
    with fake_llm() as fake:
        yield fake


URI = "claude-code://session/adr-0291"


def _turn(tag: str, words: int = 8) -> str:
    return f"User: {tag} " + " ".join(f"{tag.lower()}{i}" for i in range(words)) + "\n"


def _paras(*tags: str) -> str:
    """Paragraphs of one chunk each (between 700 and 1000 characters)."""
    out = [_turn(tag, 120) for tag in tags]
    assert all(700 < len(p) < 1000 for p in out)
    return "\n".join(out)


FIRST = _turn("Palpha") + "\n" + _turn("Pbeta")


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
    session: AsyncSession, entry_id: str, snapshot_id: str, **kwargs: Any
) -> tuple[list[Particle], SnapshotOutcome]:
    outcome = SnapshotOutcome()
    written = await extract_snapshot(session, entry_id, snapshot_id, outcome_out=outcome, **kwargs)
    await session.commit()
    return written, outcome


async def _row(session: AsyncSession, snapshot_id: str) -> SnapshotRow:
    return (
        await session.execute(
            select(SnapshotRow)
            .where(SnapshotRow.snapshot_id == snapshot_id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one()


async def _claim_tags(session: AsyncSession, entry_id: str, snapshot_id: str) -> list[str]:
    """The chunk tags named by the ACTIVE claims citing ``snapshot_id``, sorted."""
    tags = []
    for p in await get_particles_for_entry(session, entry_id):
        if p.status is not Status.ACTIVE:
            continue
        if any(
            ref.type is ProvenanceRefType.SOURCE and ref.snapshot_id == snapshot_id
            for ref in p.provenance
        ):
            match = re.search(r"about (P\w+)", p.content)
            if match:
                tags.append(match.group(1))
    return sorted(tags)


async def _delta_partial(
    session: AsyncSession, llm: FakeLLM, tags: tuple[str, ...], fail: set[str], **kwargs: Any
) -> tuple[str, str, str, SnapshotOutcome]:
    """A COMPLETE first snapshot, then a delta over ``tags`` with ``fail`` failing."""
    entry_id, s1 = await _deposit(session, FIRST)
    await _extract(session, entry_id, s1)
    text = FIRST + "\n" + _paras(*tags)
    _, s2 = await _deposit(session, text)
    llm.reset()
    llm.fail = set(fail)
    _, outcome = await _extract(session, entry_id, s2, **kwargs)
    return entry_id, s2, text, outcome


class TestDelta:
    @pytest.mark.asyncio
    async def test_the_leading_run_is_kept_and_the_retry_resumes_after_it(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        tags = ("Pone", "Ptwo", "Pthr", "Pfou", "Pfiv")
        entry_id, s2, text, outcome = await _delta_partial(
            db_session, llm, tags, {"Pthr", "Pfou", "Pfiv"}
        )

        assert await _claim_tags(db_session, entry_id, s2) == ["Pone", "Ptwo"]
        row = await _row(db_session, s2)
        assert row.extraction_status == ExtractionStatus.PENDING.value
        assert row.extracted_through == text.encode().index(b"User: Pthr")
        assert not row.resume_whole
        assert (outcome.failed_calls, outcome.kept_calls) == (3, 2)
        assert outcome.extracted == "partial"

        llm.reset()
        llm.fail = set()
        await _extract(db_session, entry_id, s2)
        assert llm.sent == ["Pthr", "Pfou", "Pfiv"]
        row = await _row(db_session, s2)
        assert row.extraction_status == ExtractionStatus.COMPLETE.value
        assert row.extracted_through == len(text.encode())
        assert await _claim_tags(db_session, entry_id, s2) == sorted(tags)

    @pytest.mark.asyncio
    async def test_verified_later_chunks_are_kept_and_skipped_on_the_retry(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        # Answers come back SS.S.S: chunk four is kept, chunk six (the final
        # chunk) is not.
        tags = ("Pone", "Ptwo", "Pthr", "Pfou", "Pfiv", "Psix")
        entry_id, s2, text, outcome = await _delta_partial(db_session, llm, tags, {"Pthr", "Pfiv"})

        assert await _claim_tags(db_session, entry_id, s2) == ["Pfou", "Pone", "Ptwo"]
        assert (await _row(db_session, s2)).extracted_through == text.encode().index(b"User: Pthr")
        assert (outcome.kept_calls, outcome.rebilled_calls) == (3, 1)

        llm.reset()
        llm.fail = set()
        _, retry = await _extract(db_session, entry_id, s2)
        assert llm.sent == ["Pthr", "Pfiv", "Psix"]
        assert retry.extracted == "full"
        assert await _claim_tags(db_session, entry_id, s2) == sorted(tags)

    @pytest.mark.asyncio
    async def test_a_failed_first_chunk_with_nothing_to_keep_writes_nothing(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        entry_id, s2, _, outcome = await _delta_partial(db_session, llm, ("Pone", "Ptwo"), {"Pone"})
        assert await _claim_tags(db_session, entry_id, s2) == []
        row = await _row(db_session, s2)
        assert row.extraction_status == ExtractionStatus.PENDING.value
        assert row.extracted_through is None
        assert outcome.kept_calls == 0
        assert outcome.extracted == "none"

    @pytest.mark.asyncio
    async def test_an_earlier_unread_snapshot_keeps_the_leading_run_only(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        entry_id, s1 = await _deposit(db_session, FIRST)
        await _extract(db_session, entry_id, s1)
        mid = FIRST + "\n" + _turn("Pmid")
        _, s_mid = await _deposit(db_session, mid)
        tags = ("Pone", "Ptwo", "Pthr", "Pfou", "Pfiv", "Psix")
        text = mid + "\n" + _paras(*tags)
        _, s3 = await _deposit(db_session, text)
        llm.fail = {"Pthr", "Pfiv"}
        await _extract(db_session, entry_id, s3)

        assert (await _row(db_session, s_mid)).extraction_status == "PENDING"
        assert await _claim_tags(db_session, entry_id, s3) == ["Pone", "Ptwo"]


class TestWholeRead:
    @pytest.mark.asyncio
    async def test_the_marker_keeps_every_answered_chunk_and_the_retry_reads_whole(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        tags = ("Pone", "Ptwo", "Pthr", "Pfou", "Pfiv", "Psix")
        llm.fail = {"Pthr", "Pfiv"}
        entry_id, s1 = await _deposit(db_session, _paras(*tags))
        _, outcome = await _extract(db_session, entry_id, s1)

        assert await _claim_tags(db_session, entry_id, s1) == ["Pfou", "Pone", "Psix", "Ptwo"]
        row = await _row(db_session, s1)
        assert row.extraction_status == ExtractionStatus.PENDING.value
        assert row.resume_whole
        assert row.extracted_through is None
        assert outcome.kept_calls == 4

        llm.reset()
        llm.fail = set()
        await _extract(db_session, entry_id, s1)
        assert llm.sent == ["Pthr", "Pfiv"]
        row = await _row(db_session, s1)
        assert row.extraction_status == ExtractionStatus.COMPLETE.value
        assert not row.resume_whole
        assert await _claim_tags(db_session, entry_id, s1) == sorted(tags)

    @pytest.mark.asyncio
    async def test_an_earlier_unread_snapshot_stamps_an_offset_instead(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        llm.fail = {"Pbeta"}  # FIRST is one call; its tag is the last turn's
        entry_id, s1 = await _deposit(db_session, FIRST)
        await _extract(db_session, entry_id, s1)
        assert (await _row(db_session, s1)).extraction_status == "PENDING"

        tags = ("Pone", "Ptwo", "Pthr", "Pfou", "Pfiv", "Psix")
        text = FIRST + "\n" + _paras(*tags)
        _, s2 = await _deposit(db_session, text)
        llm.fail = {"Pthr", "Pfiv"}
        await _extract(db_session, entry_id, s2)

        row = await _row(db_session, s2)
        assert not row.resume_whole
        assert row.extracted_through == text.encode().index(b"User: Pthr")
        assert "Pone" in await _claim_tags(db_session, entry_id, s2)
        assert "Pfou" not in await _claim_tags(db_session, entry_id, s2)

    @pytest.mark.asyncio
    async def test_later_snapshots_wait_on_the_marked_one(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        tags = ("Pone", "Ptwo", "Pthr")
        llm.fail = {"Ptwo"}
        entry_id, s1 = await _deposit(db_session, _paras(*tags))
        await _extract(db_session, entry_id, s1)
        assert (await _row(db_session, s1)).resume_whole

        _, s2 = await _deposit(db_session, _paras(*tags) + "\n" + _turn("Pnew", 120))
        attempts = (await _row(db_session, s2)).extraction_attempts
        llm.reset()
        _, waiting = await _extract(db_session, entry_id, s2)
        assert waiting.skipped == "waiting"
        assert waiting.waiting_on == s1
        assert not waiting.waiting_on_failed
        assert llm.sent == []
        row = await _row(db_session, s2)
        assert row.extraction_status == "PENDING"
        assert row.extraction_attempts == attempts

        llm.fail = set()
        await _extract(db_session, entry_id, s1)
        llm.reset()
        await _extract(db_session, entry_id, s2)
        assert llm.sent == ["Pnew"]
        claims = [p.content for p in await get_particles_for_entry(db_session, entry_id)]
        assert len(claims) == len(set(claims)) == 4

    @pytest.mark.asyncio
    async def test_a_failed_marked_snapshot_keeps_the_wait_and_says_so(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        llm.fail = {"Ptwo"}
        entry_id, s1 = await _deposit(db_session, _paras("Pone", "Ptwo", "Pthr"))
        await _extract(db_session, entry_id, s1)
        row = await _row(db_session, s1)
        row.extraction_status = ExtractionStatus.FAILED.value
        await db_session.commit()

        _, s2 = await _deposit(db_session, _paras("Pone", "Ptwo", "Pthr") + "\n" + _turn("Pnew"))
        _, waiting = await _extract(db_session, entry_id, s2)
        assert waiting.skipped == "waiting"
        assert waiting.waiting_on_failed


class TestCoverage:
    @pytest.mark.asyncio
    async def test_a_later_snapshot_read_first_bases_on_the_partial_one(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        tags = ("Pone", "Ptwo", "Pthr")
        entry_id, s2, text, _ = await _delta_partial(db_session, llm, tags, {"Ptwo", "Pthr"})
        _, s3 = await _deposit(db_session, text + "\n" + _turn("Pfou", 120))

        llm.reset()
        llm.fail = set()
        await _extract(db_session, entry_id, s3)
        assert llm.sent == ["Ptwo", "Pthr", "Pfou"]

        llm.reset()
        await _extract(db_session, entry_id, s2)
        assert llm.sent == []
        assert (await _row(db_session, s2)).extraction_status == "COMPLETE"
        claims = [p.content for p in await get_particles_for_entry(db_session, entry_id)]
        assert len(claims) == len(set(claims))
        tagged = sorted(re.search(r"about (P\w+)", c).group(1) for c in claims if "about P" in c)  # type: ignore[union-attr]
        assert tagged.count("Ptwo") == 1

    @pytest.mark.asyncio
    async def test_chained_partial_reads_resume_at_the_further_point(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        tags = ("Pone", "Ptwo", "Pthr", "Pfou")
        entry_id, s2, text, _ = await _delta_partial(
            db_session, llm, tags, {"Ptwo", "Pthr", "Pfou"}
        )
        _, s3 = await _deposit(db_session, text + "\n" + _turn("Pfiv", 120))
        llm.reset()
        llm.fail = {"Pthr", "Pfou", "Pfiv"}
        await _extract(db_session, entry_id, s3)
        assert (await _row(db_session, s3)).extracted_through == text.encode().index(b"User: Pthr")

        llm.reset()
        llm.fail = set()
        await _extract(db_session, entry_id, s2)
        assert llm.sent == ["Pthr", "Pfou"]


class TestOtherPaths:
    @pytest.mark.asyncio
    async def test_a_mutable_snapshot_keeps_nothing(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        llm.fail = {"Ptwo"}
        entry_id, s1 = await _deposit(
            db_session, _paras("Pone", "Ptwo", "Pthr"), mutability=Mutability.MUTABLE
        )
        _, outcome = await _extract(db_session, entry_id, s1)
        assert await _claim_tags(db_session, entry_id, s1) == []
        row = await _row(db_session, s1)
        assert row.extraction_status == "PENDING"
        assert row.extracted_through is None
        assert not row.resume_whole
        assert outcome.kept_calls == 0

    @pytest.mark.asyncio
    async def test_the_pooled_path_keeps_the_same_chunks(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        from particles.llm import CompletionPool

        tags = ("Pone", "Ptwo", "Pthr", "Pfou", "Pfiv", "Psix")
        entry_id, s2, text, outcome = await _delta_partial(
            db_session,
            llm,
            tags,
            {"Pthr", "Pfiv"},
            completion_pool=CompletionPool("extraction"),
        )
        assert await _claim_tags(db_session, entry_id, s2) == ["Pfou", "Pone", "Ptwo"]
        assert (await _row(db_session, s2)).extracted_through == text.encode().index(b"User: Pthr")
        assert outcome.kept_calls == 3

    @pytest.mark.asyncio
    async def test_a_replay_retires_the_partial_claims_and_clears_the_offset(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        from particles.operations.reindex import reindex

        entry_id, s2, _, _ = await _delta_partial(
            db_session, llm, ("Pone", "Ptwo", "Pthr"), {"Pthr"}
        )
        llm.fail = set()
        await reindex(db_session, entry_ids=[entry_id], run_post_lint=False)
        row = await _row(db_session, s2)
        assert row.extracted_through is None
        assert row.extraction_components_json is None
        assert await _claim_tags(db_session, entry_id, s2) == []

    @pytest.mark.asyncio
    async def test_a_named_reindex_reaches_an_entry_whose_only_read_is_partial(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        from particles.operations.reindex import reindex

        llm.fail = {"Ptwo"}
        entry_id, s1 = await _deposit(db_session, _paras("Pone", "Ptwo", "Pthr"))
        await _extract(db_session, entry_id, s1)
        assert (await _row(db_session, s1)).resume_whole

        llm.fail = set()
        llm.reset()
        result = await reindex(db_session, entry_ids=[entry_id], run_post_lint=False)
        assert result["succeeded"] == 1
        assert sorted(llm.sent) == ["Pone", "Pthr", "Ptwo"]
        row = await _row(db_session, s1)
        assert row.extraction_status == "COMPLETE"
        assert not row.resume_whole
        claims = [
            p
            for p in await get_particles_for_entry(db_session, entry_id)
            if p.status is Status.ACTIVE
        ]
        assert len(claims) == 3

    @pytest.mark.asyncio
    async def test_a_reindex_whose_call_fails_retires_nothing(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        from particles.operations.reindex import reindex

        entry_id, s1 = await _deposit(
            db_session, _paras("Pone", "Ptwo", "Pthr"), mutability=Mutability.MUTABLE
        )
        await _extract(db_session, entry_id, s1)
        before = await _claim_tags(db_session, entry_id, s1)
        assert before == ["Pone", "Pthr", "Ptwo"]

        llm.fail = {"Ptwo"}
        result = await reindex(db_session, entry_ids=[entry_id], run_post_lint=False)
        assert result["failed"] == 1
        assert await _claim_tags(db_session, entry_id, s1) == before


class TestAtomicityAndRecords:
    @pytest.mark.asyncio
    async def test_a_failure_in_the_write_loop_leaves_no_claims_offset_or_record(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        entry_id, s1 = await _deposit(db_session, FIRST)
        await _extract(db_session, entry_id, s1)
        text = FIRST + "\n" + _paras("Pone", "Ptwo", "Pthr")
        _, s2 = await _deposit(db_session, text)
        llm.fail = {"Pthr"}
        with (
            patch(
                "particles.ingest.pipeline._component_record",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(RuntimeError),
        ):
            await extract_snapshot(db_session, entry_id, s2)
        row = await _row(db_session, s2)
        assert row.extraction_status == "PENDING"
        assert row.extracted_through is None
        assert row.extraction_components_json is None
        assert await _claim_tags(db_session, entry_id, s2) == []

    @pytest.mark.asyncio
    async def test_the_release_leaves_a_committed_partial_write_alone(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        entry_id, s2, _, _ = await _delta_partial(
            db_session, llm, ("Pone", "Ptwo", "Pthr"), {"Pthr"}
        )
        before = (await _row(db_session, s2)).extracted_through
        assert not await release_extraction_claim(db_session, s2)
        assert (await _row(db_session, s2)).extracted_through == before
        assert await _claim_tags(db_session, entry_id, s2) == ["Pone", "Ptwo"]

    @pytest.mark.asyncio
    async def test_the_component_record_is_stored_on_the_partial_write_and_merged(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        entry_id, s2, _, _ = await _delta_partial(
            db_session, llm, ("Pone", "Ptwo", "Pthr"), {"Pthr"}
        )
        partial = (await get_extraction_component_records(db_session, [s2]))[s2]
        assert partial is not None and partial.complete

        llm.fail = set()
        await _extract(db_session, entry_id, s2)
        merged = (await get_extraction_component_records(db_session, [s2]))[s2]
        assert merged is not None and merged.complete
        assert set(partial.exercised) <= set(merged.exercised)

    @pytest.mark.asyncio
    async def test_a_partial_read_with_no_stored_record_completes_incomplete(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        entry_id, s2, _, _ = await _delta_partial(
            db_session, llm, ("Pone", "Ptwo", "Pthr"), {"Pthr"}
        )
        row = await _row(db_session, s2)
        row.extraction_components_json = None
        await db_session.commit()

        llm.fail = set()
        await _extract(db_session, entry_id, s2)
        record = (await get_extraction_component_records(db_session, [s2]))[s2]
        assert record is not None and not record.complete


class TestRetryAcrossChanges:
    @pytest.mark.asyncio
    async def test_a_version_bump_between_attempts_still_skips_the_kept_chunks(
        self, db_session: AsyncSession, llm: FakeLLM
    ) -> None:
        tags = ("Pone", "Ptwo", "Pthr", "Pfou", "Pfiv", "Psix")
        entry_id, s2, _, _ = await _delta_partial(db_session, llm, tags, {"Pthr", "Pfiv"})
        llm.reset()
        llm.fail = set()
        with patch("particles.extraction.general.EXTRACTOR_VERSION", "99.0.0"):
            await _extract(db_session, entry_id, s2)
        assert llm.sent == ["Pthr", "Pfiv", "Psix"]

    @pytest.mark.asyncio
    async def test_a_chunking_change_between_attempts_is_named(
        self, db_session: AsyncSession, llm: FakeLLM, caplog: pytest.LogCaptureFixture
    ) -> None:
        tags = ("Pone", "Ptwo", "Pthr", "Pfou", "Pfiv", "Psix")
        entry_id, s2, _, _ = await _delta_partial(db_session, llm, tags, {"Pthr", "Pfiv"})
        get_config().extraction.append_chunk_chars = 1500
        llm.fail = set()
        with caplog.at_level(logging.WARNING, logger="particles.ingest.pipeline"):
            await _extract(db_session, entry_id, s2)
        assert "chunking changed since this snapshot's earlier attempt" in caplog.text


class TestDecision:
    """The pure decision, on plain values (D2)."""

    def _outcomes(self, text: str, statuses: str) -> list[ChunkOutcome]:
        starts = [m.start() for m in re.finditer(r"User: P", text)]
        assert len(starts) == len(statuses)
        status = {"S": "answered", ".": "failed", "C": "carried"}
        return [
            ChunkOutcome(chunk_id=f"c{i}", prompt_hash=f"h{i}", status=status[s], start=start)  # type: ignore[arg-type]
            for i, (s, start) in enumerate(zip(statuses, starts, strict=True))
        ]

    def test_an_inexact_offset_keeps_nothing(self) -> None:
        # A chunk that starts mid-line has no raw paragraph break at its start.
        text = "User: Pone aaa\nUser: Ptwo bbb\n"
        outcomes = self._outcomes(text, "S.")
        assert (
            decide_partial_keep(
                outcomes,
                delta=True,
                content=text.encode(),
                is_markdown=False,
                mark_tools=False,
                earlier_all_complete=True,
                chunk_chars=1000,
                context_chars=0,
            )
            is None
        )

    def test_a_whole_read_with_earlier_snapshots_complete_takes_the_marker(self) -> None:
        text = "User: Pone aaa\n\nUser: Ptwo bbb\n\nUser: Pthr ccc\n"
        keep = decide_partial_keep(
            self._outcomes(text, "S.S"),
            delta=False,
            content=text.encode(),
            is_markdown=False,
            mark_tools=False,
            earlier_all_complete=True,
            chunk_chars=1000,
            context_chars=0,
        )
        assert keep is not None
        assert keep.marker and keep.offset is None
        assert keep.kept_hashes == {"h0", "h2"}

    def test_own_chunk_claims_skips_other_snapshots_and_inactive_claims(self) -> None:
        from particles.core.schema import Confidence, ProvenanceRef
        from particles.core.scoring.confidence import CalibrationSource

        def claim(pid: str, snapshot_id: str, chunk_hash: str, status: Status) -> Particle:
            return Particle(
                id=pid,
                content=pid,
                confidence=Confidence(
                    value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
                ),
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                asserted_by="t",
                status=status,
                provenance=[
                    ProvenanceRef(
                        type=ProvenanceRefType.SOURCE,
                        corpus_entry_id="e",
                        snapshot_id=snapshot_id,
                        chunk_hash=chunk_hash,
                    )
                ],
            )

        claims = [
            claim("a", "s", "h1", Status.ACTIVE),
            claim("b", "other", "h2", Status.ACTIVE),
            claim("c", "s", "h3", Status.SUPERSEDED),
        ]
        assert own_chunk_claims(claims, entry_id="e", snapshot_id="s") == {"h1": ["a"]}

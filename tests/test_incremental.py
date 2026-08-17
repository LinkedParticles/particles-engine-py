"""Tests for the chunk-hash carry-forward helper."""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.extraction.general import CandidateParticle
from particles.extraction.incremental import (
    ChunkUnit,
    _hash_chunk,
    extract_with_carry_forward,
)
from particles.store.particle_store import insert_particle


def _candidate(content: str = "claim text") -> CandidateParticle:
    return CandidateParticle(
        content=content,
        confidence_value=0.85,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
    )


async def _seed_particle(
    session: Any,
    *,
    entry_id: str,
    chunk_hash: str,
    extractor_id: str,
    extractor_version: str,
    content: str = "seeded claim",
) -> str:
    """Insert a fully-formed ACTIVE particle for chunk-hash lookup tests."""
    p = Particle(
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by=extractor_id,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id=entry_id,
                snapshot_id="seed-snap",
                chunk_hash=chunk_hash,
            )
        ],
        extractor_ref={"name": extractor_id, "version": extractor_version},
    )
    await insert_particle(session, p, embedding=None)
    await session.commit()
    return p.id


class TestHashChunk:
    def test_deterministic(self) -> None:
        h1 = _hash_chunk("hello world")
        h2 = _hash_chunk("hello world")
        assert h1 == h2
        assert h1 == hashlib.sha256(b"hello world").hexdigest()

    def test_distinguishes_different_text(self) -> None:
        assert _hash_chunk("a") != _hash_chunk("b")


class TestExtractWithCarryForward:
    @pytest.mark.asyncio
    async def test_cache_hit_skips_llm_and_records_carry_forward(self, db_session: Any) -> None:
        chunk = ChunkUnit(chunk_id="body", chunk_text="The body text.")
        chunk_hash = _hash_chunk(chunk.chunk_text)

        existing_id = await _seed_particle(
            db_session,
            entry_id="entry-1",
            chunk_hash=chunk_hash,
            extractor_id="test-extractor",
            extractor_version="1.0.0",
        )

        async def boom_llm(_text: str) -> tuple[list[CandidateParticle], list[str], bool]:
            raise AssertionError("LLM should not be called on a cache hit")

        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(side_effect=boom_llm),
        ):
            result = await extract_with_carry_forward(
                session=db_session,
                chunks=[chunk],
                corpus_entry_id="entry-1",
                extractor_id="test-extractor",
                extractor_version="1.0.0",
            )

        assert result.candidates == []
        assert existing_id in result.carry_forward_ids
        assert any("CHUNK_CARRY_FORWARD" in n for n in result.quality_notes)

    @pytest.mark.asyncio
    async def test_cache_miss_calls_llm_and_stamps_chunk_hash(self, db_session: Any) -> None:
        chunk = ChunkUnit(chunk_id="body", chunk_text="Unfamiliar text.")
        chunk_hash = _hash_chunk(chunk.chunk_text)
        cand = _candidate(content="extracted claim")

        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(return_value=([cand], [], False)),
        ):
            result = await extract_with_carry_forward(
                session=db_session,
                chunks=[chunk],
                corpus_entry_id="entry-1",
                extractor_id="test-extractor",
                extractor_version="1.0.0",
            )

        assert len(result.candidates) == 1
        # Helper stamps chunk_hash so the next re-extraction can carry it
        # forward when the chunk text is unchanged.
        assert result.candidates[0].chunk_hash == chunk_hash
        assert result.carry_forward_ids == []

    @pytest.mark.asyncio
    async def test_version_mismatch_treated_as_cache_miss(self, db_session: Any) -> None:
        """A bumped EXTRACTOR_VERSION must force re-extraction so bug fixes
        propagate — even when chunk text matches an older particle."""
        chunk = ChunkUnit(chunk_id="body", chunk_text="Stable text.")
        chunk_hash = _hash_chunk(chunk.chunk_text)

        await _seed_particle(
            db_session,
            entry_id="entry-1",
            chunk_hash=chunk_hash,
            extractor_id="test-extractor",
            extractor_version="1.0.0",
        )

        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(return_value=([_candidate("fresh claim")], [], False)),
        ):
            result = await extract_with_carry_forward(
                session=db_session,
                chunks=[chunk],
                corpus_entry_id="entry-1",
                extractor_id="test-extractor",
                extractor_version="2.0.0",  # newer
            )

        # Cache miss → LLM ran, no carry-forward.
        assert len(result.candidates) == 1
        assert result.carry_forward_ids == []

    @pytest.mark.asyncio
    async def test_supersede_set_forces_cache_miss(self, db_session: Any) -> None:
        """A particle marked for replacement (the reindex supersede set) must
        not satisfy the cache — otherwise ``reindex --provider-model`` never
        reaches the LLM, because model-scoped reindex changes neither the
        chunk text nor the extractor version."""
        chunk = ChunkUnit(chunk_id="body", chunk_text="Stable text.")
        chunk_hash = _hash_chunk(chunk.chunk_text)

        existing_id = await _seed_particle(
            db_session,
            entry_id="entry-1",
            chunk_hash=chunk_hash,
            extractor_id="test-extractor",
            extractor_version="1.0.0",
        )

        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(return_value=([_candidate("re-extracted claim")], [], False)),
        ):
            result = await extract_with_carry_forward(
                session=db_session,
                chunks=[chunk],
                corpus_entry_id="entry-1",
                extractor_id="test-extractor",
                extractor_version="1.0.0",
                supersede_ids=frozenset({existing_id}),
            )

        # Cache miss → LLM ran, fresh candidate stamped for future runs.
        assert len(result.candidates) == 1
        assert result.candidates[0].chunk_hash == chunk_hash
        assert result.carry_forward_ids == []

    @pytest.mark.asyncio
    async def test_supersede_set_leaves_unmarked_particles_as_hits(self, db_session: Any) -> None:
        """Only the marked particles are excluded: a chunk that still has an
        ACTIVE particle outside the supersede set (e.g. from another snapshot
        of the same entry) remains a cache hit carrying forward exactly the
        unmarked particle."""
        chunk = ChunkUnit(chunk_id="body", chunk_text="Shared text.")
        chunk_hash = _hash_chunk(chunk.chunk_text)

        marked_id = await _seed_particle(
            db_session,
            entry_id="entry-1",
            chunk_hash=chunk_hash,
            extractor_id="test-extractor",
            extractor_version="1.0.0",
            content="claim from the snapshot being reindexed",
        )
        unmarked_id = await _seed_particle(
            db_session,
            entry_id="entry-1",
            chunk_hash=chunk_hash,
            extractor_id="test-extractor",
            extractor_version="1.0.0",
            content="claim from another snapshot",
        )

        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(side_effect=AssertionError("LLM should not run on a partial exclusion")),
        ):
            result = await extract_with_carry_forward(
                session=db_session,
                chunks=[chunk],
                corpus_entry_id="entry-1",
                extractor_id="test-extractor",
                extractor_version="1.0.0",
                supersede_ids=frozenset({marked_id}),
            )

        assert result.candidates == []
        assert result.carry_forward_ids == [unmarked_id]
        assert marked_id not in result.carry_forward_ids

    @pytest.mark.asyncio
    async def test_different_corpus_entry_does_not_carry_forward(self, db_session: Any) -> None:
        """A particle from a different corpus entry must not be carried
        forward even when its chunk_hash matches — the cache key includes
        the entry."""
        chunk = ChunkUnit(chunk_id="body", chunk_text="Same text, different source.")
        chunk_hash = _hash_chunk(chunk.chunk_text)

        await _seed_particle(
            db_session,
            entry_id="entry-A",
            chunk_hash=chunk_hash,
            extractor_id="test-extractor",
            extractor_version="1.0.0",
        )

        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(return_value=([_candidate()], [], False)),
        ):
            result = await extract_with_carry_forward(
                session=db_session,
                chunks=[chunk],
                corpus_entry_id="entry-B",  # different entry
                extractor_id="test-extractor",
                extractor_version="1.0.0",
            )

        assert result.carry_forward_ids == []
        assert len(result.candidates) == 1

    @pytest.mark.asyncio
    async def test_max_llm_calls_caps_work(self, db_session: Any) -> None:
        """The max_llm_calls budget is enforced; surplus chunks emit a
        CHUNK_TRUNCATION quality note."""
        chunks = [ChunkUnit(chunk_id=f"c{i}", chunk_text=f"chunk text {i}") for i in range(5)]

        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(return_value=([_candidate()], [], False)),
        ):
            result = await extract_with_carry_forward(
                session=db_session,
                chunks=chunks,
                corpus_entry_id="entry-1",
                extractor_id="test-extractor",
                extractor_version="1.0.0",
                max_llm_calls=2,
            )

        # Only 2 chunks LLM-extracted; 3 truncation notes emitted.
        assert len(result.candidates) == 2
        truncations = [n for n in result.quality_notes if "CHUNK_TRUNCATION" in n]
        assert len(truncations) == 3

    @pytest.mark.asyncio
    async def test_cache_hits_do_not_consume_llm_budget(self, db_session: Any) -> None:
        """A cache-hit chunk doesn't count against ``max_llm_calls``."""
        # 3 chunks: two will hit the cache, one will miss.
        chunks = [ChunkUnit(chunk_id=f"c{i}", chunk_text=f"text {i}") for i in range(3)]
        # Seed particles for chunks 0 and 1.
        for chunk in chunks[:2]:
            await _seed_particle(
                db_session,
                entry_id="entry-1",
                chunk_hash=_hash_chunk(chunk.chunk_text),
                extractor_id="test-extractor",
                extractor_version="1.0.0",
            )

        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(return_value=([_candidate()], [], False)),
        ) as mock_llm:
            result = await extract_with_carry_forward(
                session=db_session,
                chunks=chunks,
                corpus_entry_id="entry-1",
                extractor_id="test-extractor",
                extractor_version="1.0.0",
                max_llm_calls=1,  # only 1 LLM call budget, but cache hits don't count
            )

        # 1 LLM call for the cache-miss chunk; 2 carry-forwards.
        assert mock_llm.call_count == 1
        assert len(result.candidates) == 1
        assert len(result.carry_forward_ids) == 2
        # No truncation note — the cache hits absorbed the work below budget.
        assert not any("CHUNK_TRUNCATION" in n for n in result.quality_notes)

    @pytest.mark.asyncio
    async def test_session_none_falls_through_to_llm(self) -> None:
        """When no session is provided (e.g. unit tests driving the helper
        directly), every chunk is a cache miss."""
        chunk = ChunkUnit(chunk_id="body", chunk_text="anything")

        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(return_value=([_candidate()], [], False)),
        ) as mock_llm:
            result = await extract_with_carry_forward(
                session=None,
                chunks=[chunk],
                corpus_entry_id="entry-1",
                extractor_id="test-extractor",
                extractor_version="1.0.0",
            )

        assert mock_llm.called
        assert len(result.candidates) == 1

    @pytest.mark.asyncio
    async def test_injected_call_llm_overrides_default(self) -> None:
        """a caller-supplied ``call_llm`` is used per chunk in place of
        the default general ``_call_llm`` — the seam the journal extractor uses to
        run its journal prompt through the shared carry-forward machinery."""
        chunk = ChunkUnit(chunk_id="body", chunk_text="anything")
        custom = AsyncMock(return_value=([_candidate("from custom caller")], [], False))

        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(side_effect=AssertionError("default _call_llm must not be called")),
        ) as default_llm:
            result = await extract_with_carry_forward(
                session=None,
                chunks=[chunk],
                corpus_entry_id="entry-1",
                extractor_id="test-extractor",
                extractor_version="1.0.0",
                call_llm=custom,
            )

        assert custom.called
        assert not default_llm.called
        assert [c.content for c in result.candidates] == ["from custom caller"]


class TestTransientErrorPropagation:
    """F4.1 regression — the chunked path must surface transient API failures
    via the structured ``ExtractionResult.transient_error_count`` channel, not
    a string-prefixed ``quality_notes`` sentinel.

    The historical bug: ``incremental`` prefixed every chunk note with
    ``f"{chunk_id}: {n}"``, so the pipeline's ``note.startswith("API error")``
    check never matched and a fully-failed chunked extraction was stamped
    COMPLETE with zero particles, silently leaving the retry queue.
    """

    @pytest.mark.asyncio
    async def test_all_chunks_failing_sets_full_transient_count(self) -> None:
        chunks = [ChunkUnit(chunk_id=f"c{i}", chunk_text=f"chunk text {i}") for i in range(3)]

        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(return_value=([], ["API error: rate limited"], True)),
        ):
            result = await extract_with_carry_forward(
                session=None,
                chunks=chunks,
                corpus_entry_id="entry-1",
                extractor_id="test-extractor",
                extractor_version="1.0.0",
            )

        assert result.candidates == []
        # One transient failure counted per chunk — the signal that drives the
        # pipeline's reset-to-PENDING. The note prefixing (chunk_id) no longer
        # matters because detection is structural.
        assert result.transient_error_count == 3

    @pytest.mark.asyncio
    async def test_partial_chunk_failure_counts_only_failures(self) -> None:
        chunks = [ChunkUnit(chunk_id=f"c{i}", chunk_text=f"unique chunk {i}") for i in range(3)]

        # Chunk 0 succeeds; chunks 1 and 2 hit a transient API error.
        async def flaky_llm(text: str) -> tuple[list[CandidateParticle], list[str], bool]:
            if text.endswith("0"):
                return ([_candidate("good claim")], [], False)
            return ([], ["API error: server error"], True)

        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(side_effect=flaky_llm),
        ):
            result = await extract_with_carry_forward(
                session=None,
                chunks=chunks,
                corpus_entry_id="entry-1",
                extractor_id="test-extractor",
                extractor_version="1.0.0",
            )

        assert len(result.candidates) == 1
        assert result.transient_error_count == 2

    @pytest.mark.asyncio
    async def test_success_leaves_transient_count_zero(self) -> None:
        chunks = [ChunkUnit(chunk_id="c0", chunk_text="all good")]

        with patch(
            "particles.extraction.incremental._call_llm",
            AsyncMock(return_value=([_candidate()], [], False)),
        ):
            result = await extract_with_carry_forward(
                session=None,
                chunks=chunks,
                corpus_entry_id="entry-1",
                extractor_id="test-extractor",
                extractor_version="1.0.0",
            )

        assert result.transient_error_count == 0
        assert result.carry_forward_ids == []


class TestChunkedExtractorAdoption:
    """Lock: every chunked LLM-driven extractor routes its
    re-extraction through the shared ``extract_with_carry_forward`` helper.

    There was a request to reuse the Reddit body-vs-comment hash-partitioning helper
    for the other chunked extractors. By R1.9 the helper had been adopted
    organically across six extractors as each was added; this test is the
    regression guard so a future refactor that drops a chunked extractor back to
    a bespoke per-chunk LLM loop (re-extracting unchanged chunks, the cost the
    helper exists to avoid) fails loudly instead of silently.
    """

    # The six chunked LLM-driven extractors that carry forward unchanged chunks.
    ADOPTERS = [
        "particles.extraction.reddit",
        "particles.extraction.github.gist",
        "particles.extraction.general",
        "particles.extraction.hackernews",
        "particles.extraction.journal",
        "particles.extraction.mastodon",
    ]

    @pytest.mark.parametrize("module_path", ADOPTERS)
    def test_adopter_routes_through_carry_forward(self, module_path: str) -> None:
        import importlib
        import inspect

        module = importlib.import_module(module_path)
        source = inspect.getsource(module)
        # Check for the call site (name + open paren), not a bare mention, so a
        # comment referencing the helper cannot satisfy the guard.
        assert "extract_with_carry_forward(" in source, (
            f"{module_path} is a chunked extractor but no longer routes through "
            "extract_with_carry_forward. If this is "
            "intentional, update tests/test_incremental.py and the PDR record."
        )

    def test_github_pages_is_the_documented_holdout(self) -> None:
        # GitHub Pages chunks line-by-line with overlap (``_split_into_chunks``),
        # which was deliberately kept out of carry-forward: overlapping
        # line windows have unstable boundaries, so hashing them carries forward
        # almost nothing. It stays a manual per-chunk loop by design — this
        # asserts the documented exception so a "why isn't Pages adopting it?"
        # question has a test-anchored answer.
        import importlib
        import inspect

        module = importlib.import_module("particles.extraction.github.pages")
        source = inspect.getsource(module)
        assert "extract_with_carry_forward(" not in source


_RAW_CLAIM = (
    '[{"content": "pooled claim", "confidence_value": 0.8, "uncertainty_nature": "EPISTEMIC"}]'
)


class TestExtractChunksPooled:
    """The pooled twin: same lookups, same cap, same notes, one batch."""

    @pytest.mark.asyncio
    async def test_pooled_mixes_hit_miss_and_cap_like_the_sequential_loop(
        self, db_session: Any
    ) -> None:
        from particles.llm import CompletionPool

        chunks = [
            ChunkUnit(chunk_id="chunk_1", chunk_text="Fresh text one."),
            ChunkUnit(chunk_id="chunk_2", chunk_text="Cached text."),
            ChunkUnit(chunk_id="chunk_3", chunk_text="Fresh text beyond the cap."),
        ]
        cached_hash = _hash_chunk(chunks[1].chunk_text)
        existing_id = await _seed_particle(
            db_session,
            entry_id="entry-1",
            chunk_hash=cached_hash,
            extractor_id="test-extractor",
            extractor_version="1.0.0",
        )

        pooled = AsyncMock(return_value=([_RAW_CLAIM], "anthropic:test-model"))
        with (
            patch("particles.extraction.incremental._pooled_group_complete", pooled),
            patch(
                "particles.extraction.incremental._call_llm",
                AsyncMock(side_effect=AssertionError("sequential loop must not run")),
            ),
        ):
            result = await extract_with_carry_forward(
                session=db_session,
                chunks=chunks,
                corpus_entry_id="entry-1",
                extractor_id="test-extractor",
                extractor_version="1.0.0",
                max_llm_calls=1,
                completion_pool=CompletionPool("extraction"),
            )

        # One planned call: chunk_1 (miss); chunk_2 is a cache hit; chunk_3 is
        # beyond the cap (cache hits don't count toward it — cap 1 is consumed
        # by chunk_1 alone).
        assert pooled.await_count == 1
        planned = pooled.await_args.args[1]
        assert len(planned) == 1
        assert len(result.candidates) == 1
        assert result.candidates[0].content == "pooled claim"
        assert result.candidates[0].chunk_hash == _hash_chunk(chunks[0].chunk_text)
        assert result.candidates[0].provider_model == "anthropic:test-model"
        assert existing_id in result.carry_forward_ids
        assert any("CHUNK_CARRY_FORWARD: chunk_2" in n for n in result.quality_notes)
        assert any("CHUNK_TRUNCATION: chunk_3" in n for n in result.quality_notes)
        assert result.transient_error_count == 0

    @pytest.mark.asyncio
    async def test_pooled_none_result_degrades_to_the_transient_path(self) -> None:
        from particles.llm import CompletionPool

        pooled = AsyncMock(return_value=([_RAW_CLAIM, None], "anthropic:test-model"))
        with patch("particles.extraction.incremental._pooled_group_complete", pooled):
            result = await extract_with_carry_forward(
                session=None,
                chunks=[
                    ChunkUnit(chunk_id="chunk_1", chunk_text="Text A."),
                    ChunkUnit(chunk_id="chunk_2", chunk_text="Text B."),
                ],
                corpus_entry_id=None,
                extractor_id="test-extractor",
                extractor_version="1.0.0",
                completion_pool=CompletionPool("extraction"),
            )

        # The answered chunk's candidates survive; the unanswered one becomes
        # a per-chunk transient exactly as a sequential API error would.
        assert len(result.candidates) == 1
        assert result.transient_error_count == 1
        assert any(
            "chunk_2: API error: batch result unavailable" in n for n in result.quality_notes
        )

    @pytest.mark.asyncio
    async def test_injected_call_llm_keeps_the_sequential_loop(self) -> None:
        """An injected per-chunk caller (the journal extractor) has no
        build/parse halves, so it must keep the sequential loop even when a
        pool is present."""
        from particles.llm import CompletionPool

        injected = AsyncMock(return_value=([_candidate("via injected")], [], False))
        pooled = AsyncMock(side_effect=AssertionError("pooled path must not run"))
        with patch("particles.extraction.incremental._pooled_group_complete", pooled):
            result = await extract_with_carry_forward(
                session=None,
                chunks=[ChunkUnit(chunk_id="chunk_1", chunk_text="Journal text.")],
                corpus_entry_id=None,
                extractor_id="journal-extractor",
                extractor_version="1.0.0",
                call_llm=injected,
                completion_pool=CompletionPool("extraction"),
            )

        assert injected.await_count == 1
        assert len(result.candidates) == 1
        assert result.candidates[0].content == "via injected"

    @pytest.mark.asyncio
    async def test_pooled_supersede_set_forces_cache_miss(self, db_session: Any) -> None:
        """The pooled twin applies the same supersede-set exclusion as the
        sequential loop: a chunk whose only ACTIVE particle is marked for
        replacement rides the batch instead of cache-hitting."""
        from particles.llm import CompletionPool

        chunk = ChunkUnit(chunk_id="chunk_1", chunk_text="Stable pooled text.")
        existing_id = await _seed_particle(
            db_session,
            entry_id="entry-1",
            chunk_hash=_hash_chunk(chunk.chunk_text),
            extractor_id="test-extractor",
            extractor_version="1.0.0",
        )

        pooled = AsyncMock(return_value=([_RAW_CLAIM], "anthropic:test-model"))
        with patch("particles.extraction.incremental._pooled_group_complete", pooled):
            result = await extract_with_carry_forward(
                session=db_session,
                chunks=[chunk],
                corpus_entry_id="entry-1",
                extractor_id="test-extractor",
                extractor_version="1.0.0",
                completion_pool=CompletionPool("extraction"),
                supersede_ids=frozenset({existing_id}),
            )

        assert pooled.await_count == 1
        assert len(pooled.await_args.args[1]) == 1
        assert len(result.candidates) == 1
        assert result.candidates[0].chunk_hash == _hash_chunk(chunk.chunk_text)
        assert result.carry_forward_ids == []

"""Comparator memories for the LongMemEval harness (``benchmark/memory/comparators.py``).

The two comparators (raw-transcript ``chunks``, LLM-written ``notes``) are
drop-ins for the particle store in the run loop: same questions, answer
scaffold, judge, and retrieval scoring. These pin (a) the chunker's
turn-alignment + no-drop contract, (b) session-granularity retrieval scoring
through the comparator path (hit / miss / abstention), (c) the notes writer's
once-per-session caching, in-flight dedupe, on-disk persistence and model
stamping, (d) the checkpoint key: default particles key byte-identical to
before, comparator / no-baselines keys distinct, (e) the estimate per memory
kind, and (f) an end-to-end comparator run through ``run_memory_benchmark``
— ``selection.memory`` stamped, the memory-under-test slot renamed, the
baselines honestly ``not run`` under ``--no-baselines``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from particles.benchmark.memory import comparators as comp_mod
from particles.benchmark.memory import runner as runner_mod
from particles.benchmark.memory.comparators import (
    CHUNK_MAX_CHARS,
    NotesWriter,
    chunk_session,
    run_comparator_question,
    session_hash,
)
from particles.benchmark.memory.runner import (
    _run_checkpoint_key,
    estimate_run,
    run_memory_benchmark,
)
from particles.benchmark.memory.schema import (
    MemoryQuestion,
    MemorySession,
    MemoryTurn,
    render_report_table,
)


def _session(
    sid: str, *contents: str, date: str | None = "2023/05/20 (Sat) 02:21"
) -> MemorySession:
    turns = [
        MemoryTurn(role="user" if i % 2 == 0 else "assistant", content=c)
        for i, c in enumerate(contents)
    ]
    return MemorySession(session_id=sid, date=date, turns=turns)


def _question(sessions: list[MemorySession], evidence: list[str], **kw: Any) -> MemoryQuestion:
    return MemoryQuestion(
        question_id=kw.get("qid", "q1"),
        question_type=kw.get("qtype", "single-session-user"),
        question=kw.get("question", "What colour is my bike?"),
        answer="red",
        sessions=sessions,
        answer_session_ids=evidence,
    )


class _KeywordEmbedder:
    """Deterministic stand-in: a 3-d bag over the words bike / red / cat."""

    VOCAB = ("bike", "red", "cat")

    def encode(self, texts: list[str], **_: Any) -> list[np.ndarray]:
        out = []
        for t in texts:
            v = np.array([float(w in t.lower()) for w in self.VOCAB], dtype=np.float32)
            n = float(np.linalg.norm(v))
            out.append(v / n if n else v)
        return out


@pytest.fixture
def keyword_embedder(monkeypatch: pytest.MonkeyPatch) -> _KeywordEmbedder:
    model = _KeywordEmbedder()
    monkeypatch.setattr(comp_mod, "get_embedding_model", lambda: model)
    return model


# ---------------------------------------------------------------------------
# (a) chunker
# ---------------------------------------------------------------------------


class TestChunkSession:
    def test_short_session_is_one_chunk_with_date_header(self) -> None:
        s = _session("s1", "hello", "hi there")
        chunks = chunk_session(s)
        assert chunks == ["Session date: 2023/05/20 (Sat) 02:21\nuser: hello\nassistant: hi there"]

    def test_turns_are_packed_without_splitting_short_turns(self) -> None:
        s = _session("s1", *["x" * 400 for _ in range(6)], date=None)
        chunks = chunk_session(s, max_chars=1000)
        # 6 turns of ~406 chars → 2 per chunk → 3 chunks, each turn intact.
        assert len(chunks) == 3
        assert all(c.count("\n") == 1 for c in chunks)

    def test_oversized_turn_is_split_not_dropped(self) -> None:
        s = _session("s1", "y" * 5000, date=None)
        chunks = chunk_session(s, max_chars=1000)
        assert len(chunks) == 6  # "user: " + 5000 chars over a 1000 budget
        assert "".join(c for c in chunks).count("y") == 5000

    def test_every_chunk_respects_the_ceiling(self) -> None:
        s = _session("s1", *["word " * 300 for _ in range(10)])
        for c in chunk_session(s):
            assert len(c) <= CHUNK_MAX_CHARS + 1


# ---------------------------------------------------------------------------
# (b) retrieval scoring through the comparator path
# ---------------------------------------------------------------------------


class TestChunksRetrieval:
    @pytest.mark.asyncio
    async def test_hit_when_evidence_chunk_ranks_first(self, keyword_embedder: Any) -> None:
        q = _question(
            [_session("s-ev", "my bike is red"), _session("s-x", "my cat sleeps")],
            evidence=["s-ev"],
        )
        res = await run_comparator_question(q, memory="chunks", top_k=1)
        assert res.retrieval.recall_at_k == 1.0
        assert res.retrieval.precision_at_k == 1.0
        assert res.retrieval.particles_retrieved == 1
        assert "my bike is red" in res.context
        assert "[2023/05/20" in res.context

    @pytest.mark.asyncio
    async def test_miss_when_evidence_is_outside_top_k(self, keyword_embedder: Any) -> None:
        q = _question(
            [_session("s-ev", "I own a dog"), _session("s-x", "red bike sale")],
            evidence=["s-ev"],
        )
        res = await run_comparator_question(q, memory="chunks", top_k=1)
        assert res.retrieval.recall_at_k == 0.0
        assert res.retrieval.precision_at_k == 0.0

    @pytest.mark.asyncio
    async def test_abstention_is_unscoreable(self, keyword_embedder: Any) -> None:
        q = _question([_session("s-x", "red bike")], evidence=["s-x"], qid="q1_abs")
        assert q.is_abstention
        res = await run_comparator_question(q, memory="chunks", top_k=3)
        assert res.retrieval.abstention is True
        assert res.retrieval.recall_at_k is None and res.retrieval.precision_at_k is None

    @pytest.mark.asyncio
    async def test_precision_counts_one_entry_per_chunk(self, keyword_embedder: Any) -> None:
        # Two chunks from the evidence session + one off-evidence chunk in top-3.
        q = _question(
            [
                _session("s-ev", "red bike", "assistant: red bike again", date=None),
                _session("s-x", "a red cat", date=None),
            ],
            evidence=["s-ev"],
        )
        res = await run_comparator_question(q, memory="chunks", top_k=3)
        # Two turns pack into one chunk under the default ceiling, so top-3 =
        # [s-ev chunk, s-x chunk] → precision 1/2, recall 1/1.
        assert res.retrieval.particles_retrieved == 2
        assert res.retrieval.precision_at_k == 0.5
        assert res.retrieval.recall_at_k == 1.0

    @pytest.mark.asyncio
    async def test_unknown_memory_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown comparator memory"):
            await run_comparator_question(_question([], []), memory="graph", top_k=1)


# ---------------------------------------------------------------------------
# (c) notes writer
# ---------------------------------------------------------------------------


def _patch_writer_model(monkeypatch: pytest.MonkeyPatch, model_id: str = "anthropic:m") -> None:
    provider = MagicMock()
    provider.provider_model = model_id
    monkeypatch.setattr(comp_mod, "get_provider", lambda purpose: provider)


class TestNotesWriter:
    @pytest.mark.asyncio
    async def test_writes_once_per_distinct_session_and_persists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_writer_model(monkeypatch)
        calls: list[int] = []

        async def fake_many(purpose: str, requests: Any, **kw: Any) -> list[str | None]:
            calls.append(len(requests))
            return [f"notes for {r.prompt.count('user:')}u" for r in requests]

        monkeypatch.setattr(comp_mod, "complete_many", fake_many)
        cache = tmp_path / "notes.jsonl"
        w = NotesWriter(cache_path=cache)
        s1, s2 = _session("a", "one"), _session("b", "two", "three")
        first = await w.notes_for([s1, s2, s1])
        assert first[0] == first[2] and first[0] is not None
        assert calls == [2]  # duplicate within the call written once
        assert (w.misses, w.hits) == (2, 0)

        second = await w.notes_for([s2])
        assert second == [first[1]]
        assert calls == [2]  # in-process cache
        assert w.hits == 1

        # A fresh writer with the same model restores from disk — no call.
        w2 = NotesWriter(cache_path=cache)
        assert await w2.notes_for([s1, s2]) == [first[0], first[1]]
        assert calls == [2]

        # A different model never replays another model's notes.
        _patch_writer_model(monkeypatch, "anthropic:other")
        w3 = NotesWriter(cache_path=cache)
        await w3.notes_for([s1])
        assert calls == [2, 1]

    @pytest.mark.asyncio
    async def test_concurrent_callers_share_one_inflight_write(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_writer_model(monkeypatch)
        gate = asyncio.Event()
        calls: list[int] = []

        async def slow_many(purpose: str, requests: Any, **kw: Any) -> list[str | None]:
            calls.append(len(requests))
            await gate.wait()
            return ["n"] * len(requests)

        monkeypatch.setattr(comp_mod, "complete_many", slow_many)
        w = NotesWriter()
        s = _session("a", "one")
        t1 = asyncio.create_task(w.notes_for([s]))
        await asyncio.sleep(0)  # t1 registers the in-flight write
        t2 = asyncio.create_task(w.notes_for([s]))
        await asyncio.sleep(0)
        gate.set()
        assert await t1 == ["n"] and await t2 == ["n"]
        assert calls == [1]
        assert (w.misses, w.hits) == (1, 1)

    @pytest.mark.asyncio
    async def test_failed_write_is_none_and_counted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_writer_model(monkeypatch)
        monkeypatch.setattr(comp_mod, "complete_many", AsyncMock(return_value=[None, "ok"]))
        w = NotesWriter()
        out = await w.notes_for([_session("a", "x"), _session("b", "y")])
        assert out == [None, "ok"]
        assert w.failures == 1

    def test_session_hash_covers_date_and_turns(self) -> None:
        a = _session("a", "x")
        b = _session("b", "x")  # id is not part of the hash — content is
        assert session_hash(a) == session_hash(b)
        assert session_hash(_session("a", "x", date=None)) != session_hash(a)
        assert session_hash(_session("a", "y")) != session_hash(a)


class TestNotesRetrieval:
    @pytest.mark.asyncio
    async def test_notes_memory_scores_and_reports_missing_sessions(
        self, keyword_embedder: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_writer_model(monkeypatch)
        monkeypatch.setattr(
            comp_mod, "complete_many", AsyncMock(return_value=["- user has a red bike", None])
        )
        q = _question([_session("s-ev", "bike talk"), _session("s-x", "cat talk")], ["s-ev"])
        res = await run_comparator_question(q, memory="notes", top_k=2, notes_writer=NotesWriter())
        assert res.retrieval.recall_at_k == 1.0
        assert res.retrieval.particles_retrieved == 1  # the failed session is absent
        assert any("notes writer failed on 1 of 2" in n for n in res.notes)
        assert "- user has a red bike" in res.context

    @pytest.mark.asyncio
    async def test_notes_memory_requires_a_writer(self) -> None:
        with pytest.raises(ValueError, match="NotesWriter"):
            await run_comparator_question(_question([], []), memory="notes", top_k=1)


# ---------------------------------------------------------------------------
# (d) checkpoint key + (e) estimate
# ---------------------------------------------------------------------------


class TestCheckpointKeyAndEstimate:
    def _base(self) -> dict[str, Any]:
        return dict(
            dataset_revision="r",
            variant="s",
            top_k=10,
            context_budget=None,
            abstraction=False,
            qa=True,
            extraction_model_id="anthropic:x",
            embedding_model_id="e",
            answer_model_id="anthropic:a",
            judge_model_id="anthropic:j",
        )

    def test_default_particles_key_is_unchanged(self) -> None:
        key = _run_checkpoint_key(**self._base())
        assert "memory" not in key and "baselines" not in key
        assert key == _run_checkpoint_key(**self._base(), memory="particles", baselines=True)

    def test_comparator_and_no_baselines_keys_are_distinct(self) -> None:
        base = _run_checkpoint_key(**self._base())
        chunks = _run_checkpoint_key(**self._base(), memory="chunks")
        notes = _run_checkpoint_key(**self._base(), memory="notes", baselines=False)
        assert chunks["memory"] == "chunks" and chunks != base
        assert notes["memory"] == "notes" and notes["baselines"] is False
        assert len({str(sorted(k.items())) for k in (base, chunks, notes)}) == 3

    def test_estimate_per_memory_kind(self) -> None:
        qs = [_question([_session("a", "x" * 100), _session("b", "y" * 100)], ["a"])]
        particles = estimate_run(qs)
        chunks = estimate_run(qs, memory="chunks", baselines=False)
        notes = estimate_run(qs, memory="notes")
        assert particles.estimated_extraction_calls == 2
        assert chunks.estimated_extraction_calls == 0
        assert chunks.estimated_answer_calls == chunks.estimated_judge_calls == 1
        assert notes.estimated_extraction_calls == 2
        assert notes.estimated_answer_calls == 3

    def test_unknown_memory_refused_by_runner(self) -> None:
        with pytest.raises(ValueError, match="memory must be one of"):
            asyncio.run(
                run_memory_benchmark(
                    [],
                    variant="s",
                    dataset_revision="r",
                    selection_seed=1,
                    selection_limit=1,
                    memory="graph",
                )
            )


# ---------------------------------------------------------------------------
# (f) end-to-end comparator run through the runner
# ---------------------------------------------------------------------------


class TestComparatorRun:
    @pytest.mark.asyncio
    async def test_chunks_run_without_baselines(
        self, tmp_path: Path, keyword_embedder: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = MagicMock()
        provider.provider_model = "anthropic:m"
        monkeypatch.setattr(runner_mod, "get_provider", lambda purpose: provider)
        monkeypatch.setattr(runner_mod, "get_embedding_model_id", lambda: "kw")
        monkeypatch.setattr(runner_mod, "select_extractor", lambda st: MagicMock())
        seen: list[str] = []

        async def fake_complete(purpose: str, prompt: str, **kw: Any) -> str:
            seen.append(purpose)
            return "an answer" if purpose == "benchmark_answer" else "yes"

        monkeypatch.setattr(runner_mod, "complete", fake_complete)
        boom = AsyncMock(side_effect=AssertionError("particle pipeline must not run"))
        monkeypatch.setattr(runner_mod, "deposit_text_versioned", boom)
        monkeypatch.setattr(runner_mod, "extract_snapshot", boom)
        monkeypatch.setattr(runner_mod, "retrieve_ranked", boom)

        qs = [
            _question([_session("s-ev", "red bike"), _session("s-x", "cat")], ["s-ev"], qid="q1"),
            _question([_session("s-x", "cat")], ["s-x"], qid="q2_abs"),
        ]
        report = await run_memory_benchmark(
            qs,
            variant="s",
            dataset_revision="r",
            selection_seed=13,
            selection_limit=2,
            questions_total=500,
            memory="chunks",
            baselines=False,
            checkpoint_dir=tmp_path / "ckpt",
        )
        assert report.selection.memory == "chunks"
        assert report.selection.extraction_model_id is None
        assert report.retrieval_stage.questions == 1
        assert report.retrieval_stage.abstention_questions == 1
        assert report.retrieval_stage.mean_recall_at_k == 1.0
        assert report.qa_particles is not None
        assert report.qa_particles.condition == "qa_chunks"
        assert report.qa_particles.questions == 2
        assert report.qa_full_context is None and report.qa_no_memory is None
        # One answer + one judge per question, nothing for the baselines.
        assert seen.count("benchmark_answer") == 2 and seen.count("benchmark") == 2
        assert any("Comparator run (memory=chunks)" in n for n in report.quality_notes)

        rendered = render_report_table(report)
        assert "memory=chunks (COMPARATOR)" in rendered
        assert "qa_chunks        (COMPARATOR" in rendered
        assert "qa_full_context  (BASELINE: full concatenated haystack): not run" in rendered

        # The comparator checkpoint never collides with a particles one.
        files = list((tmp_path / "ckpt").glob("memory-run-*.jsonl"))
        assert len(files) == 1
        header = files[0].read_text().splitlines()[0]
        assert '"memory": "chunks"' in header and '"baselines": false' in header


class TestContextBudgetClamp:
    """the particle path's context clamp applies to a comparator too."""

    @pytest.mark.asyncio
    async def test_budget_clamps_items_in_rank_order_first_always_kept(
        self, keyword_embedder: Any
    ) -> None:
        q = _question(
            [_session("s-ev", "red bike " * 40), _session("s-x", "red bike " * 40, date=None)],
            evidence=["s-ev"],
        )
        full = await run_comparator_question(q, memory="chunks", top_k=10)
        clamped = await run_comparator_question(
            q,
            memory="chunks",
            top_k=10,
            context_budget=10,  # 40 chars: below one item
        )
        assert full.context.count("\n- [") >= 1
        assert clamped.context.count("- [") == 1  # first item kept, rest cut
        # Retrieval scoring is untouched by the clamp.
        assert clamped.retrieval == full.retrieval


class TestComparatorBatchQa:
    @pytest.mark.asyncio
    async def test_batch_qa_only_dispatches_the_active_conditions(
        self, tmp_path: Path, keyword_embedder: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--batch-qa`` + ``--no-baselines``: one answer batch + one judge batch."""
        provider = MagicMock()
        provider.provider_model = "anthropic:m"
        monkeypatch.setattr(runner_mod, "get_provider", lambda purpose: provider)
        monkeypatch.setattr(runner_mod, "get_embedding_model_id", lambda: "kw")
        monkeypatch.setattr(runner_mod, "select_extractor", lambda st: MagicMock())
        batches: list[tuple[str, int]] = []

        async def fake_many(purpose: str, requests: Any, **kw: Any) -> list[str | None]:
            batches.append((purpose, len(requests)))
            return ["an answer" if purpose == "benchmark_answer" else "yes"] * len(requests)

        monkeypatch.setattr(runner_mod, "complete_many", fake_many)
        monkeypatch.setattr(
            runner_mod, "complete", AsyncMock(side_effect=AssertionError("inline path used"))
        )
        qs = [
            _question([_session("s-ev", "red bike"), _session("s-x", "cat")], ["s-ev"], qid="q1"),
            _question([_session("s-ev", "red bike")], ["s-ev"], qid="q2"),
        ]
        report = await run_memory_benchmark(
            qs,
            variant="s",
            dataset_revision="r",
            selection_seed=13,
            selection_limit=2,
            memory="chunks",
            baselines=False,
            batch_qa=True,
        )
        assert batches == [("benchmark_answer", 2), ("benchmark", 2)]
        assert report.qa_particles is not None and report.qa_particles.accuracy == 1.0
        assert report.qa_full_context is None and report.qa_no_memory is None

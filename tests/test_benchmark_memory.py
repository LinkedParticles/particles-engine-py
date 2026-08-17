"""Unit tests for the memory-benchmark harness.

Covers the pure metrics (including the abstention unscoreability contract:
zero-evidence questions are excluded from every retrieval aggregate, never
blended as vacuous scores), the loader (parsing + revision/SHA discipline
with a stubbed download), the seeded stratified selection, the
provenance-mapping scorer, the answer/extraction/judge model pins (refusal on
mid-run drift), the recorded extraction + embedding model ids, the
report-renderer invariants (families separated, no score field, baseline rows
always render, subset header, abstention disclosure), the candidate cache,
and the estimate gate — mocked LLM and stubbed pipeline, no key, no network.
The live end-to-end fixture run is
``tests/test_integration_memory_benchmark.py``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from particles.benchmark.memory import loader as loader_mod
from particles.benchmark.memory import runner as runner_mod
from particles.benchmark.memory.loader import (
    UNPINNED,
    MemoryDatasetLoadError,
    ensure_dataset,
    load_dataset_file,
    parse_question,
    resolve_url,
    select_questions,
)
from particles.benchmark.memory.metrics import (
    accuracy_by_type,
    mean_by_type,
    parse_judge_verdict,
    precision_at_k,
    qa_accuracy,
    recall_at_k,
)
from particles.benchmark.memory.runner import (
    CachingExtractor,
    SameModelViolation,
    estimate_run,
    judge_prompt,
    parse_session_date,
    render_estimate,
    render_session_text,
    run_memory_benchmark,
    session_id_from_uri,
    session_uri,
)
from particles.benchmark.memory.schema import (
    QA_CONDITIONS,
    MemoryBenchmarkReport,
    MemoryQuestion,
    MemorySession,
    MemoryTurn,
    QaConditionMetrics,
    RetrievalQuestionResult,
    RetrievalStageMetrics,
    RunSelection,
    render_report_table,
)
from particles.core.schema import (
    CalibrationSource,
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.extraction.general import ExtractionResult

FIXTURE = (
    Path(__file__).parent
    / "benchmark"
    / "memory"
    / "fixtures"
    / "longmemeval_oracle_synthetic.json"
)


def _question(
    qid: str = "q1",
    qtype: str = "single-session-user",
    n_sessions: int = 2,
    evidence: list[str] | None = None,
) -> MemoryQuestion:
    sessions = [
        MemorySession(
            session_id=f"{qid}-s{i}",
            date="2023/05/20 (Sat) 02:21",
            turns=[
                MemoryTurn(role="user", content=f"fact {qid} {i}"),
                MemoryTurn(role="assistant", content="noted"),
            ],
        )
        for i in range(n_sessions)
    ]
    return MemoryQuestion(
        question_id=qid,
        question_type=qtype,
        question=f"What about {qid}?",
        answer="42",
        question_date="2023/06/01 (Thu) 10:00",
        sessions=sessions,
        answer_session_ids=evidence if evidence is not None else [f"{qid}-s0"],
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_recall_at_k(self) -> None:
        assert recall_at_k({"a"}, {"a", "b"}) == 0.5
        assert recall_at_k({"a", "b"}, {"a", "b"}) == 1.0
        assert recall_at_k(set(), {"a"}) == 0.0

    def test_recall_unscoreable_on_no_evidence(self) -> None:
        # Abstention variants label no evidence sessions: retrieval is
        # unscoreable — None, never a vacuous 1.0 that inflates the mean.
        assert recall_at_k(set(), set()) is None
        assert recall_at_k({"a"}, set()) is None

    def test_precision_at_k(self) -> None:
        assert precision_at_k(["a", "b", None, "c"], {"a", "c"}) == 0.5
        assert precision_at_k(["a"], {"a"}) == 1.0
        assert precision_at_k([None, None], {"a"}) == 0.0

    def test_precision_unscoreable_on_no_evidence(self) -> None:
        # Against an empty evidence set every retrieved particle would be a
        # "false positive" — unscoreable, never a deterministic 0.0.
        assert precision_at_k(["a", None], set()) is None
        assert precision_at_k([], set()) is None

    def test_precision_vacuous_on_no_retrieval(self) -> None:
        assert precision_at_k([], {"a"}) == 1.0

    def test_broken_provenance_chain_never_counts_as_hit(self) -> None:
        # None = the chain did not resolve; it dilutes precision, never adds.
        assert precision_at_k([None], {"a"}) == 0.0

    def test_qa_accuracy(self) -> None:
        assert qa_accuracy([True, False, True, True]) == 0.75
        assert qa_accuracy([]) == 1.0  # vacuous, support shown by the report

    def test_accuracy_by_type(self) -> None:
        pairs = [("a", True), ("a", False), ("b", True)]
        assert accuracy_by_type(pairs) == {"a": 0.5, "b": 1.0}

    def test_mean_by_type(self) -> None:
        assert mean_by_type([("a", 1.0), ("a", 0.0), ("b", 0.5)]) == {"a": 0.5, "b": 0.5}

    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            ("yes", True),
            ("Yes.", True),
            ("YES, the answer matches.", True),
            ("no", False),
            ("No — the model said Berlin.", False),
            ("the answer is correct", False),  # fail-closed on non-leading verdicts
            ("", False),
        ],
    )
    def test_parse_judge_verdict(self, reply: str, expected: bool) -> None:
        assert parse_judge_verdict(reply) is expected


# ---------------------------------------------------------------------------
# Loader — parsing
# ---------------------------------------------------------------------------


def _raw_question(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "question_id": "q-1",
        "question_type": "single-session-user",
        "question": "What?",
        "answer": "That.",
        "question_date": "2023/06/01 (Thu) 10:00",
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2023/05/20 (Sat) 02:21"],
        "haystack_sessions": [[{"role": "user", "content": "hello", "has_answer": True}]],
        "answer_session_ids": ["s1"],
    }
    raw.update(overrides)
    return raw


class TestLoaderParsing:
    def test_parse_question_roundtrip(self) -> None:
        q = parse_question(_raw_question())
        assert q.question_id == "q-1"
        assert q.sessions[0].session_id == "s1"
        assert q.sessions[0].date == "2023/05/20 (Sat) 02:21"
        assert q.sessions[0].turns[0].has_answer is True
        assert q.answer_session_ids == ["s1"]
        assert not q.is_abstention

    def test_abstention_flag_from_id_suffix(self) -> None:
        q = parse_question(_raw_question(question_id="q-9_abs", answer_session_ids=[]))
        assert q.is_abstention

    @pytest.mark.parametrize(
        "missing", ["question_id", "question_type", "question", "haystack_sessions"]
    )
    def test_missing_required_field_raises(self, missing: str) -> None:
        raw = _raw_question()
        del raw[missing]
        with pytest.raises(MemoryDatasetLoadError, match=missing):
            parse_question(raw)

    def test_mismatched_parallel_lists_raise(self) -> None:
        raw = _raw_question(haystack_session_ids=["s1", "s2"])
        with pytest.raises(MemoryDatasetLoadError, match="differ in length"):
            parse_question(raw)

    def test_unknown_keys_are_tolerated(self) -> None:
        # The dataset is upstream-owned: forward-compat tolerance, not strictness.
        q = parse_question(_raw_question(some_future_field=123))
        assert q.question_id == "q-1"

    def test_missing_dates_degrade_to_none(self) -> None:
        raw = _raw_question(haystack_dates=[])
        assert parse_question(raw).sessions[0].date is None

    def test_load_dataset_file_on_checked_in_fixture(self) -> None:
        questions = load_dataset_file(FIXTURE)
        assert len(questions) == 3
        by_id = {q.question_id: q for q in questions}
        assert by_id["synthetic-002"].question_type == "knowledge-update"
        assert by_id["synthetic-002"].answer_session_ids == ["answer_b2"]
        assert by_id["synthetic-003_abs"].is_abstention

    def test_load_dataset_file_rejects_non_array(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text('{"not": "a list"}')
        with pytest.raises(MemoryDatasetLoadError, match="JSON array"):
            load_dataset_file(bad)

    def test_load_dataset_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(MemoryDatasetLoadError, match="not found"):
            load_dataset_file(tmp_path / "absent.json")


# ---------------------------------------------------------------------------
# Loader — revision/SHA acquisition discipline (stubbed download)
# ---------------------------------------------------------------------------


def _stub_client(payload: bytes, status_code: int = 200) -> MagicMock:
    """A particles_client stand-in whose stream() yields ``payload``."""

    async def aiter_bytes() -> Any:
        yield payload

    response = MagicMock()
    response.status_code = status_code
    response.aiter_bytes = aiter_bytes

    stream_cm = MagicMock()
    stream_cm.__aenter__ = AsyncMock(return_value=response)
    stream_cm.__aexit__ = AsyncMock(return_value=False)

    client = MagicMock()
    client.stream = MagicMock(return_value=stream_cm)

    client_cm = MagicMock()
    client_cm.__aenter__ = AsyncMock(return_value=client)
    client_cm.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=client_cm)


class TestEnsureDataset:
    PAYLOAD = b'[{"question_id": "x"}]'
    SHA = hashlib.sha256(PAYLOAD).hexdigest()

    def test_resolve_url_pins_repo_and_revision(self) -> None:
        url = resolve_url("s", "deadbeef")
        assert url == (
            "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
            "resolve/deadbeef/longmemeval_s_cleaned.json"
        )

    def test_resolve_url_rejects_unknown_variant(self) -> None:
        with pytest.raises(MemoryDatasetLoadError, match="variant"):
            resolve_url("xxl", "deadbeef")

    @pytest.mark.asyncio
    async def test_download_verifies_sha_and_caches(self, tmp_path: Path) -> None:
        stub = _stub_client(self.PAYLOAD)
        with patch.object(loader_mod, "particles_client", stub):
            path = await ensure_dataset(
                "oracle", revision="rev1", cache_dir=tmp_path, expected_sha256=self.SHA
            )
        assert path.read_bytes() == self.PAYLOAD
        assert path == tmp_path / "rev1" / "longmemeval_oracle.json"
        # Second call is a cache hit — no new download.
        stub2 = _stub_client(self.PAYLOAD)
        with patch.object(loader_mod, "particles_client", stub2):
            again = await ensure_dataset(
                "oracle", revision="rev1", cache_dir=tmp_path, expected_sha256=self.SHA
            )
        assert again == path
        stub2.assert_not_called()

    @pytest.mark.asyncio
    async def test_sha_mismatch_refuses_and_does_not_cache(self, tmp_path: Path) -> None:
        stub = _stub_client(self.PAYLOAD)
        with (
            patch.object(loader_mod, "particles_client", stub),
            pytest.raises(MemoryDatasetLoadError, match="SHA-256"),
        ):
            await ensure_dataset(
                "oracle", revision="rev1", cache_dir=tmp_path, expected_sha256="0" * 64
            )
        assert not (tmp_path / "rev1" / "longmemeval_oracle.json").exists()

    @pytest.mark.asyncio
    async def test_unpinned_sha_refuses_download(self, tmp_path: Path) -> None:
        stub = _stub_client(self.PAYLOAD)
        with (
            patch.object(loader_mod, "particles_client", stub),
            pytest.raises(MemoryDatasetLoadError, match="not finalized"),
        ):
            await ensure_dataset(
                "oracle", revision="rev1", cache_dir=tmp_path, expected_sha256=UNPINNED
            )
        stub.assert_not_called()

    @pytest.mark.asyncio
    async def test_http_error_refuses(self, tmp_path: Path) -> None:
        stub = _stub_client(b"", status_code=404)
        with (
            patch.object(loader_mod, "particles_client", stub),
            pytest.raises(MemoryDatasetLoadError, match="HTTP 404"),
        ):
            await ensure_dataset(
                "oracle", revision="rev1", cache_dir=tmp_path, expected_sha256=self.SHA
            )

    @pytest.mark.asyncio
    async def test_cached_file_reverified_against_pin(self, tmp_path: Path) -> None:
        target = tmp_path / "rev1" / "longmemeval_oracle.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"tampered")
        with pytest.raises(MemoryDatasetLoadError, match="fails its SHA-256 pin"):
            await ensure_dataset(
                "oracle", revision="rev1", cache_dir=tmp_path, expected_sha256=self.SHA
            )


# ---------------------------------------------------------------------------
# Stratified selection determinism
# ---------------------------------------------------------------------------


class TestSelectQuestions:
    def _pool(self) -> list[MemoryQuestion]:
        pool: list[MemoryQuestion] = []
        for qtype, count in (
            ("single-session-user", 10),
            ("multi-session", 6),
            ("knowledge-update", 4),
        ):
            pool.extend(_question(qid=f"{qtype}-{i}", qtype=qtype) for i in range(count))
        return pool

    def test_deterministic_under_seed(self) -> None:
        pool = self._pool()
        a = select_questions(pool, seed=13, limit=10)
        b = select_questions(pool, seed=13, limit=10)
        assert [q.question_id for q in a] == [q.question_id for q in b]

    def test_different_seed_changes_selection(self) -> None:
        pool = self._pool()
        a = {q.question_id for q in select_questions(pool, seed=13, limit=10)}
        b = {q.question_id for q in select_questions(pool, seed=14, limit=10)}
        assert a != b

    def test_stratified_proportions(self) -> None:
        selected = select_questions(self._pool(), seed=13, limit=10)
        counts: dict[str, int] = {}
        for q in selected:
            counts[q.question_type] = counts.get(q.question_type, 0) + 1
        # 10/20, 6/20, 4/20 of limit 10 → 5, 3, 2
        assert counts == {"single-session-user": 5, "multi-session": 3, "knowledge-update": 2}

    def test_limit_at_or_above_pool_returns_all(self) -> None:
        pool = self._pool()
        assert len(select_questions(pool, seed=13, limit=None)) == len(pool)
        assert len(select_questions(pool, seed=13, limit=999)) == len(pool)

    def test_types_filter(self) -> None:
        selected = select_questions(self._pool(), seed=13, limit=None, types=["multi-session"])
        assert selected and all(q.question_type == "multi-session" for q in selected)

    def test_result_order_is_stable(self) -> None:
        selected = select_questions(self._pool(), seed=13, limit=10)
        assert [(q.question_type, q.question_id) for q in selected] == sorted(
            (q.question_type, q.question_id) for q in selected
        )


# ---------------------------------------------------------------------------
# URI scheme + dates
# ---------------------------------------------------------------------------


class TestUriAndDates:
    def test_session_uri_roundtrip(self) -> None:
        uri = session_uri("q-1", "answer_a1")
        assert uri == "longmemeval://q-1/session/answer_a1"
        assert session_id_from_uri(uri) == "answer_a1"

    @pytest.mark.parametrize("bad", [None, "", "https://x/y", "longmemeval://q-1/other/x"])
    def test_session_id_from_uri_rejects_foreign_uris(self, bad: str | None) -> None:
        assert session_id_from_uri(bad) is None

    def test_parse_session_date_longmemeval_format(self) -> None:
        parsed = parse_session_date("2023/05/20 (Sat) 02:21")
        assert parsed == datetime(2023, 5, 20, 2, 21, tzinfo=UTC)

    def test_parse_session_date_unparseable_degrades_to_none(self) -> None:
        assert parse_session_date("last Tuesday-ish") is None
        assert parse_session_date(None) is None

    def test_render_session_text_carries_date_and_turns(self) -> None:
        text = render_session_text(_question().sessions[0])
        assert "Session date: 2023/05/20 (Sat) 02:21" in text
        assert "user: fact q1 0" in text


# ---------------------------------------------------------------------------
# Report renderer invariants (structural, not editorial)
# ---------------------------------------------------------------------------


def _selection(**overrides: Any) -> RunSelection:
    defaults: dict[str, Any] = {
        "dataset_revision": "rev-abc",
        "variant": "s",
        "sample_seed": 13,
        "question_limit": 150,
        "question_types": ["multi-session"],
        "questions_selected": 150,
        "questions_total": 500,
        "top_k": 10,
        "answer_model_id": "anthropic:test-answer",
        "judge_model_id": "anthropic:test-judge",
        "extraction_model_id": "anthropic:test-extract",
        "embedding_model_id": "test-embedder",
        "thresholds": {"extraction.similarity_threshold": 0.8},
    }
    defaults.update(overrides)
    return RunSelection(**defaults)


class TestReportInvariants:
    def test_families_render_as_separate_sections(self) -> None:
        report = MemoryBenchmarkReport(selection=_selection())
        table = render_report_table(report)
        assert "== Retrieval stage" in table
        assert "== End-to-end QA" in table
        # The retrieval heading disclaims QA and vice versa.
        assert "NOT answer accuracy" in table
        assert "NOT retrieval" in table

    def test_skipped_qa_conditions_render_not_run_never_omitted(self) -> None:
        report = MemoryBenchmarkReport(selection=_selection())  # all three None
        table = render_report_table(report)
        for condition in QA_CONDITIONS:
            assert condition in table
        assert table.count("not run") >= 3

    def test_baseline_rows_render_even_when_only_qa_particles_ran(self) -> None:
        report = MemoryBenchmarkReport(
            selection=_selection(),
            qa_particles=QaConditionMetrics(
                condition="qa_particles", model_id="anthropic:m", questions=2, accuracy=0.5
            ),
        )
        table = render_report_table(report)
        assert "qa_full_context" in table and "qa_no_memory" in table
        assert table.count("not run") == 2  # exactly the two baselines

    def test_no_flag_exists_to_omit_baseline_rows(self) -> None:
        import inspect

        params = inspect.signature(render_report_table).parameters
        assert list(params) == ["report"]  # renderer takes the report, nothing else

    def test_subset_status_and_selection_tuple_in_header(self) -> None:
        table = render_report_table(MemoryBenchmarkReport(selection=_selection()))
        assert "SUBSET run — 150 of 500 questions" in table
        assert "seed=13" in table
        assert "variant=s" in table
        assert "revision=rev-abc" in table
        assert "top_k=10" in table
        assert "answer_model=anthropic:test-answer" in table
        assert "judge_model=anthropic:test-judge" in table
        assert "extraction_model=anthropic:test-extract" in table
        assert "embedding_model=test-embedder" in table
        assert "extraction.similarity_threshold" in table

    def test_unrecorded_pipeline_model_ids_render_explicitly(self) -> None:
        selection = _selection(extraction_model_id=None, embedding_model_id=None)
        table = render_report_table(MemoryBenchmarkReport(selection=selection))
        assert "extraction_model=not recorded" in table
        assert "embedding_model=not recorded" in table

    def test_full_run_header(self) -> None:
        selection = _selection(questions_selected=500, question_limit=None)
        table = render_report_table(MemoryBenchmarkReport(selection=selection))
        assert "FULL run — 500 questions" in table
        assert "SUBSET" not in table

    def test_abstention_disclosure_line_and_na_rows(self) -> None:
        """Excluded abstention questions are disclosed — count + n/a rows."""
        retrieval = RetrievalStageMetrics(
            questions=1,
            abstention_questions=1,
            mean_recall_at_k=0.5,
            mean_precision_at_k=0.5,
            per_question=[
                RetrievalQuestionResult(
                    question_id="q-1",
                    question_type="single-session-user",
                    evidence_sessions=2,
                    evidence_sessions_hit=1,
                    particles_retrieved=2,
                    recall_at_k=0.5,
                    precision_at_k=0.5,
                ),
                RetrievalQuestionResult(
                    question_id="q-9_abs",
                    question_type="single-session-user",
                    evidence_sessions=0,
                    evidence_sessions_hit=0,
                    particles_retrieved=2,
                    recall_at_k=None,
                    precision_at_k=None,
                    abstention=True,
                ),
            ],
        )
        table = render_report_table(
            MemoryBenchmarkReport(selection=_selection(), retrieval_stage=retrieval)
        )
        assert (
            "1 abstention question(s) excluded from retrieval aggregates — "
            "no labeled evidence to score" in table
        )
        assert "q-9_abs: recall/precision n/a (abstention)" in table

    def test_no_abstention_no_disclosure_line(self) -> None:
        table = render_report_table(MemoryBenchmarkReport(selection=_selection()))
        assert "abstention question(s) excluded" not in table

    def test_no_score_field_anywhere_in_the_model_tree(self) -> None:
        """No single mergeable 'score' field exists."""
        seen: set[type[BaseModel]] = set()

        def walk(model: type[BaseModel]) -> None:
            if model in seen:
                return
            seen.add(model)
            for name, field in model.model_fields.items():
                assert name != "score", f"{model.__name__}.{name} violates the no-score invariant"
                annotation = field.annotation
                stack = [annotation]
                while stack:
                    ann = stack.pop()
                    if isinstance(ann, type) and issubclass(ann, BaseModel):
                        walk(ann)
                    else:
                        stack.extend(getattr(ann, "__args__", ()))

        walk(MemoryBenchmarkReport)
        assert RetrievalStageMetrics in seen and QaConditionMetrics in seen

    def test_report_json_roundtrip(self) -> None:
        report = MemoryBenchmarkReport(selection=_selection())
        parsed = json.loads(report.model_dump_json())
        assert parsed["selection"]["dataset_revision"] == "rev-abc"
        assert parsed["qa_full_context"] is None  # explicit, never absent


# ---------------------------------------------------------------------------
# Candidate cache
# ---------------------------------------------------------------------------


class _StubExtractor:
    EXTRACTOR_ID = "stub-extractor"
    EXTRACTOR_VERSION = "0.0.1"

    def __init__(self, transient: int = 0) -> None:
        self.calls = 0
        self.transient = transient

    def accepts(self, source_type: str) -> bool:
        return True

    async def extract(self, snapshot: Any, content: bytes, **kwargs: object) -> ExtractionResult:
        self.calls += 1
        from particles.core.schema import UncertaintyNature as UN
        from particles.extraction.general import CandidateParticle

        return ExtractionResult(
            candidates=[
                CandidateParticle(
                    content=f"claim from {snapshot.content_hash[:8]}",
                    confidence_value=0.9,
                    uncertainty_nature=UN.EPISTEMIC,
                )
            ],
            transient_error_count=self.transient,
        )


def _snapshot(content: bytes) -> Any:
    from particles.core.schema import Snapshot

    return Snapshot(content_hash=hashlib.sha256(content).hexdigest())


class TestCachingExtractor:
    @pytest.mark.asyncio
    async def test_repeat_content_hits_cache(self) -> None:
        inner = _StubExtractor()
        caching = CachingExtractor(inner)
        content = b"same session bytes"
        r1 = await caching.extract(_snapshot(content), content)
        r2 = await caching.extract(_snapshot(content), content)
        assert inner.calls == 1
        assert caching.hits == 1 and caching.misses == 1
        assert r1.candidates[0].content == r2.candidates[0].content

    @pytest.mark.asyncio
    async def test_distinct_content_misses(self) -> None:
        inner = _StubExtractor()
        caching = CachingExtractor(inner)
        await caching.extract(_snapshot(b"one"), b"one")
        await caching.extract(_snapshot(b"two"), b"two")
        assert inner.calls == 2 and caching.hits == 0

    @pytest.mark.asyncio
    async def test_replay_is_isolated_from_pipeline_mutation(self) -> None:
        inner = _StubExtractor()
        caching = CachingExtractor(inner)
        content = b"bytes"
        first = await caching.extract(_snapshot(content), content)
        first.candidates[0].content = "MUTATED BY STORE A"
        second = await caching.extract(_snapshot(content), content)
        assert second.candidates[0].content != "MUTATED BY STORE A"

    @pytest.mark.asyncio
    async def test_transient_failures_are_not_cached(self) -> None:
        inner = _StubExtractor(transient=1)
        caching = CachingExtractor(inner)
        content = b"flaky"
        await caching.extract(_snapshot(content), content)
        await caching.extract(_snapshot(content), content)
        assert inner.calls == 2  # retried, never replayed

    def test_identity_mirrors_inner(self) -> None:
        caching = CachingExtractor(_StubExtractor())
        assert caching.EXTRACTOR_ID == "stub-extractor"
        assert caching.accepts("CONVERSATION")


# ---------------------------------------------------------------------------
# Estimate gate
# ---------------------------------------------------------------------------


class TestEstimate:
    def test_duplicate_sessions_counted_once(self) -> None:
        q1 = _question(qid="q1")
        # q2 shares q1's sessions byte-for-byte (same rendered text).
        q2 = q1.model_copy(update={"question_id": "q2"})
        est = estimate_run([q1, q2])
        assert est.questions == 2
        assert est.unique_sessions == 2  # q1 has 2 distinct sessions; q2 repeats them
        assert est.estimated_answer_calls == 6
        assert est.estimated_judge_calls == 6
        assert est.estimated_llm_calls == est.estimated_extraction_calls + 12

    def test_qa_false_drops_answer_and_judge_calls(self) -> None:
        est = estimate_run([_question()], qa=False)
        assert est.estimated_answer_calls == 0
        assert est.estimated_judge_calls == 0

    def test_render_estimate_mentions_counts(self) -> None:
        text = render_estimate(estimate_run([_question()]))
        assert "Estimate:" in text
        assert "LLM call" in text
        assert "tokens" in text


# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------


class TestJudgePrompt:
    def test_abstention_prompt_scores_declining(self) -> None:
        q = _question(qid="q-9_abs", evidence=[])
        prompt = judge_prompt(q, "I don't have that information.")
        assert "abstain" in prompt.lower()
        assert "yes or no" in prompt.lower()

    def test_knowledge_update_prompt_requires_latest_state(self) -> None:
        q = _question(qtype="knowledge-update")
        prompt = judge_prompt(q, "Lisbon")
        assert "LATEST" in prompt
        assert q.answer is not None and q.answer in prompt


# ---------------------------------------------------------------------------
# Runner orchestration (stubbed pipeline — no LLM, no embeddings)
# ---------------------------------------------------------------------------


def _particle(pid: str, content: str = "a claim") -> Particle:
    return Particle(
        id=pid,
        content=content,
        confidence=Confidence(value=0.9),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=f"entry-{pid}")],
        asserted_by="general-extractor",
    )


def _derived_particle(pid: str, premise_ids: list[str]) -> Particle:
    """A promoted abstraction: PARTICLE refs, no SOURCE ref.

    Per §3's field-reuse convention the premise particle id travels in
    ``corpus_entry_id``.
    """
    return Particle(
        id=pid,
        content="an abstraction",
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.DERIVED),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.PARTICLE, corpus_entry_id=premise, snapshot_id=premise
            )
            for premise in premise_ids
        ],
        asserted_by="abstraction-pass",
    )


def _provider(model_id: str = "anthropic:answer-model") -> MagicMock:
    provider = MagicMock()
    provider.provider_model = model_id
    return provider


class TestPremiseTransitiveScoring:
    """a derived particle scores through its premises' sessions.

    Before the fix its provenance chain dead-ended (no SOURCE ref), so every
    promoted abstraction in top-k scored as a retrieval non-hit.
    """

    def _patches(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        retrieved: list[Particle],
        premises: dict[str, Particle],
        uris: dict[str, str],
    ) -> None:
        monkeypatch.setattr(
            runner_mod,
            "deposit_text_versioned",
            AsyncMock(
                side_effect=lambda session, **kw: (
                    f"entry-{kw['uri_r']}",
                    f"snap-{kw['uri_r']}",
                    False,
                )
            ),
        )
        monkeypatch.setattr(runner_mod, "extract_snapshot", AsyncMock(return_value=[]))
        monkeypatch.setattr(runner_mod, "select_extractor", lambda source_type: _StubExtractor())
        monkeypatch.setattr(
            runner_mod,
            "retrieve_ranked",
            AsyncMock(return_value=[(p, 0.9, 0.8) for p in retrieved]),
        )
        monkeypatch.setattr(
            runner_mod, "get_particles_by_ids", AsyncMock(return_value=dict(premises))
        )

        table = {
            pid: (None, "CONVERSATION", f"entry-{pid}", uri, None) for pid, uri in uris.items()
        }

        async def _rows(session: Any, particles: list[Particle]) -> dict[str, Any]:
            return {p.id: table[p.id] for p in particles if p.id in table}

        monkeypatch.setattr(runner_mod, "load_source_rows", _rows)

    async def _run(self, question: MemoryQuestion, tmp_path: Path) -> Any:
        report = await run_memory_benchmark(
            [question],
            variant="oracle",
            dataset_revision="rev-test",
            selection_seed=13,
            selection_limit=None,
            questions_total=1,
            work_dir=tmp_path,
            keep_stores=True,
            qa=False,
        )
        return report.retrieval_stage.per_question[0]

    @pytest.mark.asyncio
    async def test_union_of_premise_sessions_credits_recall_once_for_precision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Both haystack sessions are labeled evidence; one abstraction stands
        # on both of them via its premises.
        question = _question(n_sessions=2, evidence=["q1-s0", "q1-s1"])
        premises = {"prem-a": _particle("prem-a"), "prem-b": _particle("prem-b")}
        derived = _derived_particle("d1", ["prem-a", "prem-b"])
        self._patches(
            monkeypatch,
            retrieved=[derived],
            premises=premises,
            uris={
                "prem-a": session_uri("q1", "q1-s0"),
                "prem-b": session_uri("q1", "q1-s1"),
            },
        )

        row = await self._run(question, tmp_path)

        # Recall credits every covered evidence session...
        assert row.evidence_sessions_hit == 2
        assert row.recall_at_k == 1.0
        # ...but the abstraction is one retrieved item, counted once.
        assert row.particles_retrieved == 1
        assert row.precision_at_k == 1.0

    @pytest.mark.asyncio
    async def test_one_evidence_premise_is_enough_for_precision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Only s0 is evidence; the abstraction also covers a distractor.
        question = _question(n_sessions=2, evidence=["q1-s0"])
        premises = {"prem-a": _particle("prem-a"), "prem-b": _particle("prem-b")}
        derived = _derived_particle("d1", ["prem-a", "prem-b"])
        self._patches(
            monkeypatch,
            retrieved=[derived],
            premises=premises,
            uris={
                "prem-a": session_uri("q1", "q1-s0"),
                "prem-b": session_uri("q1", "q1-s1"),
            },
        )

        row = await self._run(question, tmp_path)

        assert row.recall_at_k == 1.0
        assert row.precision_at_k == 1.0

    @pytest.mark.asyncio
    async def test_derived_particle_off_evidence_still_scores_as_a_miss(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The fix must not blanket-credit abstractions: premises entirely off
        # the evidence set stay a precision miss.
        question = _question(n_sessions=2, evidence=["q1-s0"])
        premises = {"prem-b": _particle("prem-b")}
        derived = _derived_particle("d1", ["prem-b"])
        self._patches(
            monkeypatch,
            retrieved=[derived, _particle("p1")],
            premises=premises,
            uris={
                "prem-b": session_uri("q1", "q1-s1"),
                "p1": session_uri("q1", "q1-s0"),
            },
        )

        row = await self._run(question, tmp_path)

        assert row.recall_at_k == 1.0
        assert row.precision_at_k == 0.5


class TestRunnerOrchestration:
    """Full run over stubbed pipeline seams (patch the runner's bindings)."""

    def _patches(self, question: MemoryQuestion, monkeypatch: pytest.MonkeyPatch) -> None:
        deposit = AsyncMock(
            side_effect=lambda session, **kw: (f"entry-{kw['uri_r']}", f"snap-{kw['uri_r']}", False)
        )
        monkeypatch.setattr(runner_mod, "deposit_text_versioned", deposit)
        monkeypatch.setattr(runner_mod, "extract_snapshot", AsyncMock(return_value=[]))
        monkeypatch.setattr(runner_mod, "select_extractor", lambda source_type: _StubExtractor())

        p_hit = _particle("p1")
        p_miss = _particle("p2")
        monkeypatch.setattr(
            runner_mod,
            "retrieve_ranked",
            AsyncMock(return_value=[(p_hit, 0.9, 0.8), (p_miss, 0.5, 0.7)]),
        )
        evidence_sid = question.answer_session_ids[0]
        rows = {
            "p1": (
                None,
                "CONVERSATION",
                "e1",
                session_uri(question.question_id, evidence_sid),
                None,
            ),
            "p2": (
                None,
                "CONVERSATION",
                "e2",
                session_uri(question.question_id, f"{question.question_id}-s1"),
                None,
            ),
        }
        monkeypatch.setattr(runner_mod, "load_source_rows", AsyncMock(return_value=rows))

    @pytest.mark.asyncio
    async def test_full_run_populates_both_families(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        question = _question()
        self._patches(question, monkeypatch)
        monkeypatch.setattr(runner_mod, "get_provider", lambda purpose: _provider())
        replies = iter(["an answer", "yes"] * 3)
        monkeypatch.setattr(
            runner_mod, "complete", AsyncMock(side_effect=lambda *a, **k: next(replies))
        )

        report = await run_memory_benchmark(
            [question],
            variant="oracle",
            dataset_revision="rev-test",
            selection_seed=13,
            selection_limit=1,
            questions_total=3,
            work_dir=tmp_path,
            keep_stores=True,
        )

        assert report.retrieval_stage.questions == 1
        row = report.retrieval_stage.per_question[0]
        # p1's chain lands on the labeled evidence session; p2 on a distractor.
        assert row.recall_at_k == 1.0
        assert row.precision_at_k == 0.5
        for condition in QA_CONDITIONS:
            metrics = getattr(report, condition)
            assert metrics is not None, condition
            assert metrics.model_id == "anthropic:answer-model"
            assert metrics.questions == 1
        assert report.selection.answer_model_id == "anthropic:answer-model"
        assert report.selection.subset  # 1 of 3

    @pytest.mark.asyncio
    async def test_qa_false_leaves_conditions_not_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        question = _question()
        self._patches(question, monkeypatch)
        complete = AsyncMock()
        monkeypatch.setattr(runner_mod, "complete", complete)

        report = await run_memory_benchmark(
            [question],
            variant="oracle",
            dataset_revision="rev-test",
            selection_seed=13,
            selection_limit=None,
            questions_total=1,
            work_dir=tmp_path,
            keep_stores=True,
            qa=False,
        )
        assert report.qa_particles is None
        assert report.qa_full_context is None
        assert report.qa_no_memory is None
        complete.assert_not_called()
        # ...and the renderer still shows the baseline rows as not run.
        table = render_report_table(report)
        assert table.count("not run") >= 3

    @pytest.mark.asyncio
    async def test_pooled_extraction_dispatches_every_deposit_under_one_pool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``pooled=True`` fans a question's extractions out under one CompletionPool.

        Every deposit must still be extracted exactly once, each call must
        carry the same pool instance (one Message Batches job per question), and the retrieval report must be identical to the serial
        path's — pooled dispatch changes price and latency, never the
        measurement.
        """
        from particles.llm import CompletionPool

        question = _question(qid="q1", n_sessions=3, evidence=["q1-s0"])
        self._patches(question, monkeypatch)
        extract = cast("AsyncMock", runner_mod.extract_snapshot)

        serial = await run_memory_benchmark(
            [question],
            variant="oracle",
            dataset_revision="rev-test",
            selection_seed=13,
            selection_limit=None,
            questions_total=1,
            work_dir=tmp_path / "serial",
            keep_stores=True,
            qa=False,
        )
        assert extract.await_count == 3
        assert all(c.kwargs.get("completion_pool") is None for c in extract.await_args_list)

        extract.reset_mock()
        pooled = await run_memory_benchmark(
            [question],
            variant="oracle",
            dataset_revision="rev-test",
            selection_seed=13,
            selection_limit=None,
            questions_total=1,
            work_dir=tmp_path / "pooled",
            keep_stores=True,
            qa=False,
            pooled=True,
        )
        assert extract.await_count == 3
        pools = [c.kwargs.get("completion_pool") for c in extract.await_args_list]
        assert all(isinstance(p, CompletionPool) for p in pools)
        assert len({id(p) for p in pools}) == 1  # one pool per question, shared by every deposit
        # Every deposited snapshot was extracted (order is free under gather).
        assert {c.args[1] for c in extract.await_args_list} == {
            f"entry-{session_uri('q1', f'q1-s{i}')}" for i in range(3)
        }
        assert pooled.retrieval_stage.per_question == serial.retrieval_stage.per_question

    @pytest.mark.asyncio
    async def test_same_model_mismatch_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A drifting llm.benchmark_answer resolution raises, never warns (§2)."""
        question = _question()
        self._patches(question, monkeypatch)
        models = iter(["anthropic:model-a", "anthropic:model-b", "anthropic:model-b"])

        def resolving_provider(purpose: str) -> MagicMock:
            if purpose == "benchmark_answer":
                return _provider(next(models))
            return _provider("anthropic:judge")

        monkeypatch.setattr(runner_mod, "get_provider", resolving_provider)
        monkeypatch.setattr(runner_mod, "complete", AsyncMock(return_value="yes"))

        with pytest.raises(SameModelViolation, match="pinned"):
            await run_memory_benchmark(
                [question],
                variant="oracle",
                dataset_revision="rev-test",
                selection_seed=13,
                selection_limit=None,
                questions_total=1,
                work_dir=tmp_path,
                keep_stores=True,
            )

    @pytest.mark.asyncio
    async def test_abstention_excluded_from_retrieval_aggregates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Zero-evidence questions move NO retrieval aggregate (v1.74.2 fix).

        The scored question retrieves one evidence hit out of two labeled
        sessions (recall 0.5) and one of two particles from evidence
        (precision 0.5). Blending the abstention question's old vacuous
        scores would have moved the means (recall (0.5 + 1.0) / 2 = 0.75,
        precision (0.5 + 0.0) / 2 = 0.25); exclusion keeps both at exactly
        0.5.
        """
        scored = _question(qid="q1", evidence=["q1-s0", "q1-s1"])
        abstention = _question(qid="q9_abs", qtype="single-session-user", evidence=[])
        self._patches(scored, monkeypatch)
        # Both questions retrieve the same two particles: p1's chain lands on
        # q1's labeled evidence session, p2's on a distractor. Against
        # q9_abs's empty evidence set, neither is scoreable.
        rows = {
            "p1": (None, "CONVERSATION", "e1", session_uri("q1", "q1-s0"), None),
            "p2": (None, "CONVERSATION", "e2", session_uri("q1", "q1-distractor"), None),
        }
        monkeypatch.setattr(runner_mod, "load_source_rows", AsyncMock(return_value=rows))

        report = await run_memory_benchmark(
            [scored, abstention],
            variant="oracle",
            dataset_revision="rev-test",
            selection_seed=13,
            selection_limit=None,
            questions_total=2,
            work_dir=tmp_path,
            keep_stores=True,
            qa=False,
        )

        r = report.retrieval_stage
        assert r.questions == 1  # only the evidence-bearing question
        assert r.abstention_questions == 1
        assert r.mean_recall_at_k == 0.5  # not 0.75 — no vacuous 1.0 blended in
        assert r.mean_precision_at_k == 0.5  # not 0.25 — no forced 0.0 blended in
        assert r.recall_by_type == {"single-session-user": 0.5}
        # The abstention question KEEPS its drill-down row, explicitly n/a.
        assert len(r.per_question) == 2
        abs_row = next(row for row in r.per_question if row.question_id == "q9_abs")
        assert abs_row.abstention
        assert abs_row.recall_at_k is None
        assert abs_row.precision_at_k is None
        # The JSON artifact of record carries explicit nulls, never 1.0/0.0.
        payload = json.loads(report.model_dump_json())
        abs_json = next(
            row
            for row in payload["retrieval_stage"]["per_question"]
            if row["question_id"] == "q9_abs"
        )
        assert abs_json["recall_at_k"] is None
        assert abs_json["precision_at_k"] is None
        assert abs_json["abstention"] is True

    async def test_abstention_with_labeled_near_miss_session_still_excluded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exclusion keys off the ``*_abs`` protocol flag, not label shape.

        The cleaned LongMemEval ``s`` split labels every one of its 30
        abstention questions with a near-miss session (``answer_session_ids``
        is never empty), so an empty-evidence test alone would pass while the
        real run blended all 30 into the retrieval means. Here the abstention
        question retrieves the particle whose chain lands exactly on its
        labeled near-miss session — a vacuous recall 1.0 / precision 0.5 if
        that label were treated as evidence — and must still be excluded.
        """
        scored = _question(qid="q1", evidence=["q1-s0", "q1-s1"])
        abstention = _question(qid="q9_abs", qtype="single-session-user", evidence=["q9_abs-s0"])
        self._patches(scored, monkeypatch)
        rows = {
            "p1": (None, "CONVERSATION", "e1", session_uri("q1", "q1-s0"), None),
            "p2": (None, "CONVERSATION", "e2", session_uri("q9_abs", "q9_abs-s0"), None),
        }
        monkeypatch.setattr(runner_mod, "load_source_rows", AsyncMock(return_value=rows))

        report = await run_memory_benchmark(
            [scored, abstention],
            variant="s",
            dataset_revision="rev-test",
            selection_seed=13,
            selection_limit=None,
            questions_total=2,
            work_dir=tmp_path,
            keep_stores=True,
            qa=False,
        )

        r = report.retrieval_stage
        assert r.questions == 1
        assert r.abstention_questions == 1
        assert r.mean_recall_at_k == 0.5  # not (0.5 + 1.0) / 2
        assert r.mean_precision_at_k == 0.5  # not (0.5 + 0.5) / 2
        abs_row = next(row for row in r.per_question if row.question_id == "q9_abs")
        assert abs_row.abstention
        assert abs_row.recall_at_k is None
        assert abs_row.precision_at_k is None
        assert abs_row.evidence_sessions == 0
        # ...and the rendered table discloses the exclusion + marks the row.
        table = render_report_table(report)
        assert "1 abstention question(s) excluded from retrieval aggregates" in table
        assert "q9_abs: recall/precision n/a (abstention)" in table

    @pytest.mark.asyncio
    async def test_selection_records_extraction_and_embedding_ids(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The §5 run tuple carries the resolved pipeline model ids (v1.74.2)."""
        question = _question()
        self._patches(question, monkeypatch)
        monkeypatch.setattr(
            runner_mod, "get_provider", lambda purpose: _provider(f"anthropic:{purpose}-model")
        )
        monkeypatch.setattr(runner_mod, "get_embedding_model_id", lambda: "test-embedder")
        monkeypatch.setattr(runner_mod, "complete", AsyncMock(return_value="yes"))

        report = await run_memory_benchmark(
            [question],
            variant="oracle",
            dataset_revision="rev-test",
            selection_seed=13,
            selection_limit=None,
            questions_total=1,
            work_dir=tmp_path,
            keep_stores=True,
        )
        assert report.selection.extraction_model_id == "anthropic:extraction-model"
        assert report.selection.embedding_model_id == "test-embedder"
        assert report.selection.judge_model_id == "anthropic:benchmark-model"
        table = render_report_table(report)
        assert "extraction_model=anthropic:extraction-model" in table
        assert "embedding_model=test-embedder" in table

    @pytest.mark.asyncio
    async def test_extraction_model_mid_run_change_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A drifting llm.extraction resolution raises, never warns (§5)."""
        q1, q2 = _question(qid="q1"), _question(qid="q2")
        self._patches(q1, monkeypatch)
        # Run-start resolve + q1's check see extract-a; q2's check drifts.
        models = iter(["anthropic:extract-a", "anthropic:extract-a", "anthropic:extract-b"])

        def resolving_provider(purpose: str) -> MagicMock:
            if purpose == "extraction":
                return _provider(next(models))
            return _provider("anthropic:other")

        monkeypatch.setattr(runner_mod, "get_provider", resolving_provider)

        with pytest.raises(SameModelViolation, match="Extraction-model mismatch"):
            await run_memory_benchmark(
                [q1, q2],
                variant="oracle",
                dataset_revision="rev-test",
                selection_seed=13,
                selection_limit=None,
                questions_total=2,
                work_dir=tmp_path,
                keep_stores=True,
                qa=False,
            )

    @pytest.mark.asyncio
    async def test_judge_model_mid_run_change_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A drifting llm.benchmark (judge) resolution raises, never warns (§5)."""
        question = _question()
        self._patches(question, monkeypatch)
        judges = iter(["anthropic:judge-a", "anthropic:judge-b"])

        def resolving_provider(purpose: str) -> MagicMock:
            if purpose == "benchmark":
                return _provider(next(judges))
            return _provider("anthropic:answer-model")

        monkeypatch.setattr(runner_mod, "get_provider", resolving_provider)
        monkeypatch.setattr(runner_mod, "complete", AsyncMock(return_value="yes"))

        with pytest.raises(SameModelViolation, match="Judge-model mismatch"):
            await run_memory_benchmark(
                [question],
                variant="oracle",
                dataset_revision="rev-test",
                selection_seed=13,
                selection_limit=None,
                questions_total=1,
                work_dir=tmp_path,
                keep_stores=True,
            )

    @pytest.mark.asyncio
    async def test_bad_question_degrades_to_quality_note(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        question = _question()
        monkeypatch.setattr(runner_mod, "select_extractor", lambda source_type: _StubExtractor())
        monkeypatch.setattr(
            runner_mod,
            "deposit_text_versioned",
            AsyncMock(side_effect=RuntimeError("deposit exploded")),
        )
        report = await run_memory_benchmark(
            [question],
            variant="oracle",
            dataset_revision="rev-test",
            selection_seed=13,
            selection_limit=None,
            questions_total=1,
            work_dir=tmp_path,
            keep_stores=True,
            qa=False,
        )
        assert report.retrieval_stage.questions == 0
        assert any("deposit exploded" in note for note in report.quality_notes)

    @pytest.mark.asyncio
    async def test_scratch_stores_removed_unless_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        question = _question()
        self._patches(question, monkeypatch)
        await run_memory_benchmark(
            [question],
            variant="oracle",
            dataset_revision="rev-test",
            selection_seed=13,
            selection_limit=None,
            questions_total=1,
            work_dir=tmp_path,
            keep_stores=False,
            qa=False,
        )
        assert not list(tmp_path.glob("*.db"))


class TestBatchQA:
    """--batch-qa: the answer + judge calls routed through the batch path.

    Batching is a price optimisation that must not change what the run computes
    (same marks as the inline path); it only changes dispatch — one Message
    Batches job per condition for the answers, a second for the judges, six in
    all, each at 50% price. The one-model pins are re-checked once per condition
    batch and still refuse a mid-run config flip.
    """

    def _stub_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            runner_mod,
            "deposit_text_versioned",
            AsyncMock(
                side_effect=lambda session, **kw: (
                    f"entry-{kw['uri_r']}",
                    f"snap-{kw['uri_r']}",
                    False,
                )
            ),
        )
        monkeypatch.setattr(runner_mod, "extract_snapshot", AsyncMock(return_value=[]))
        monkeypatch.setattr(runner_mod, "select_extractor", lambda source_type: _StubExtractor())
        monkeypatch.setattr(runner_mod, "retrieve_ranked", AsyncMock(return_value=[]))
        monkeypatch.setattr(runner_mod, "load_source_rows", AsyncMock(return_value={}))

    async def _run(self, questions: list[MemoryQuestion], tmp_path: Path) -> MemoryBenchmarkReport:
        return await run_memory_benchmark(
            questions,
            variant="oracle",
            dataset_revision="rev-test",
            selection_seed=13,
            selection_limit=None,
            questions_total=len(questions),
            work_dir=tmp_path,
            keep_stores=True,
            batch_qa=True,
        )

    @pytest.mark.asyncio
    async def test_answer_and_judge_go_out_as_one_batch_per_condition(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        questions = [_question("q1"), _question("q2")]
        self._stub_pipeline(monkeypatch)
        monkeypatch.setattr(runner_mod, "get_provider", lambda purpose: _provider())

        async def _many(purpose: str, requests: list[Any], **kw: Any) -> list[str | None]:
            reply = "an answer" if purpose == "benchmark_answer" else "yes"
            return [reply] * len(requests)

        many = AsyncMock(side_effect=_many)
        monkeypatch.setattr(runner_mod, "complete_many", many)
        # The inline complete() path must NOT be taken under batch_qa.
        no_single = AsyncMock(side_effect=AssertionError("inline complete() called under batch_qa"))
        monkeypatch.setattr(runner_mod, "complete", no_single)

        report = await self._run(questions, tmp_path)

        # Six jobs: answer + judge per condition, each carrying both questions.
        assert many.call_count == 6
        purposes = [call.args[0] for call in many.call_args_list]
        assert purposes.count("benchmark_answer") == 3
        assert purposes.count("benchmark") == 3
        for call in many.call_args_list:
            assert call.kwargs["latency_tolerant"] is True
            assert len(call.args[1]) == 2  # one request per question
        # The answer turn carries the shared answer-system prompt; the judge
        # prompt is self-contained (no system).
        for call in many.call_args_list:
            if call.args[0] == "benchmark_answer":
                assert all(r.system == runner_mod._ANSWER_SYSTEM for r in call.args[1])
            else:
                assert all(r.system is None for r in call.args[1])
        for condition in QA_CONDITIONS:
            metrics = getattr(report, condition)
            assert metrics is not None, condition
            assert metrics.questions == 2
            assert metrics.accuracy == 1.0
        assert report.selection.answer_model_id == "anthropic:answer-model"
        no_single.assert_not_called()

    @pytest.mark.asyncio
    async def test_pinned_answer_model_guard_refuses_under_batching(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A drifting llm.benchmark_answer resolution raises, never warns (§2).

        The pin is re-resolved once per condition batch, so a flip between the
        first and second condition is caught exactly as the inline per-call
        check caught a per-question flip.
        """
        self._stub_pipeline(monkeypatch)
        models = iter(["anthropic:m1", "anthropic:m2", "anthropic:m3"])

        def resolving(purpose: str) -> MagicMock:
            if purpose == "benchmark_answer":
                return _provider(next(models))
            return _provider("anthropic:judge")

        monkeypatch.setattr(runner_mod, "get_provider", resolving)
        monkeypatch.setattr(
            runner_mod, "complete_many", AsyncMock(side_effect=lambda p, r, **k: ["yes"] * len(r))
        )
        with pytest.raises(SameModelViolation, match="pinned"):
            await self._run([_question("q1")], tmp_path)

    @pytest.mark.asyncio
    async def test_pinned_judge_model_guard_refuses_under_batching(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_pipeline(monkeypatch)
        judges = iter(["anthropic:j1", "anthropic:j2", "anthropic:j3"])

        def resolving(purpose: str) -> MagicMock:
            if purpose == "benchmark":
                return _provider(next(judges))
            return _provider("anthropic:answer-model")

        monkeypatch.setattr(runner_mod, "get_provider", resolving)
        monkeypatch.setattr(
            runner_mod, "complete_many", AsyncMock(side_effect=lambda p, r, **k: ["yes"] * len(r))
        )
        with pytest.raises(SameModelViolation, match="Judge-model mismatch"):
            await self._run([_question("q1")], tmp_path)

    @pytest.mark.asyncio
    async def test_unavailable_answer_scores_incorrect(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A None from the answer batch (per-request failure / expired) scores 0."""
        self._stub_pipeline(monkeypatch)
        monkeypatch.setattr(runner_mod, "get_provider", lambda purpose: _provider())
        monkeypatch.setattr(
            runner_mod, "complete_many", AsyncMock(side_effect=lambda p, r, **k: [None] * len(r))
        )

        report = await self._run([_question("q1")], tmp_path)

        for condition in QA_CONDITIONS:
            metrics = getattr(report, condition)
            assert metrics is not None
            assert metrics.accuracy == 0.0
        assert any("answer call unavailable (batch)" in note for note in report.quality_notes)

    @pytest.mark.asyncio
    async def test_unavailable_judge_scores_incorrect(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A None from the judge batch scores 0 (answer arrived, verdict did not)."""
        self._stub_pipeline(monkeypatch)
        monkeypatch.setattr(runner_mod, "get_provider", lambda purpose: _provider())

        async def _many(purpose: str, requests: list[Any], **kw: Any) -> list[str | None]:
            value = "an answer" if purpose == "benchmark_answer" else None
            return [value] * len(requests)

        monkeypatch.setattr(runner_mod, "complete_many", AsyncMock(side_effect=_many))

        report = await self._run([_question("q1")], tmp_path)

        for condition in QA_CONDITIONS:
            assert getattr(report, condition).accuracy == 0.0
        assert any("judge call unavailable (batch)" in note for note in report.quality_notes)

    @pytest.mark.asyncio
    async def test_routes_through_the_real_batch_adapter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The runner's QA requests actually reach the Message Batches API.

        No get_provider/complete_many patch: the real ``AnthropicProvider``
        runs against a mocked SDK client (the ``set_client`` seam), so this
        proves the answer/judge requests are submitted as batches,
        not merely that ``complete_many`` was invoked.
        """
        from types import SimpleNamespace

        import anthropic

        from particles import llm as llm_mod
        from particles.config import reset_config

        # llm.batch knobs have no env override, so use a real file.
        config = tmp_path / "config.yaml"
        config.write_text("llm:\n  batch:\n    min_requests: 1\n")
        monkeypatch.setenv("PARTICLES_CONFIG", str(config))
        reset_config()

        self._stub_pipeline(monkeypatch)

        def _succeeded(custom_id: str, text: str) -> SimpleNamespace:
            return SimpleNamespace(
                custom_id=custom_id,
                result=SimpleNamespace(
                    type="succeeded",
                    message=SimpleNamespace(
                        content=[SimpleNamespace(text=text)], stop_reason="end_turn"
                    ),
                ),
            )

        client = MagicMock(spec=anthropic.Anthropic)
        client.messages.batches.create.return_value = SimpleNamespace(id="msgbatch_x")
        client.messages.batches.retrieve.return_value = SimpleNamespace(processing_status="ended")
        # A fresh iterator per results() call — each of the six batches carries
        # the two questions as custom_ids "0" and "1".
        client.messages.batches.results.side_effect = lambda _bid: iter(
            [_succeeded("0", "yes"), _succeeded("1", "yes")]
        )
        llm_mod.set_client(client)
        try:
            report = await self._run([_question("q1"), _question("q2")], tmp_path)
        finally:
            llm_mod.set_client(None)
            reset_config()

        assert client.messages.batches.create.call_count == 6  # answer + judge × 3 conditions
        client.messages.create.assert_not_called()
        for condition in QA_CONDITIONS:
            assert getattr(report, condition).accuracy == 1.0


class TestAblationKnobs:
    """the QA-at-budget clamp + the abstraction run-tuple knobs."""

    def _particle(self, content: str) -> Particle:
        from particles.core.schema import Confidence, UncertaintyNature

        return Particle(
            content=content,
            confidence=Confidence(value=0.9),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="test",
        )

    def test_context_clamp_keeps_rank_order_prefix(self) -> None:
        from particles.benchmark.memory.runner import _particles_context

        particles = [self._particle(f"claim number {i} " + "x" * 80) for i in range(10)]
        full = _particles_context(particles, {})
        assert full.count("\n") == 9
        # ~25 tokens ≈ 100 chars → roughly the first line only.
        clamped = _particles_context(particles, {}, budget_tokens=30)
        assert clamped.startswith("- [undated] claim number 0")
        assert clamped.count("\n") < 9

    def test_context_clamp_always_keeps_first_particle(self) -> None:
        from particles.benchmark.memory.runner import _particles_context

        particles = [self._particle("y" * 500)]
        clamped = _particles_context(particles, {}, budget_tokens=1)
        assert "y" in clamped  # never an empty context when particles exist

    def test_no_budget_is_unclamped(self) -> None:
        from particles.benchmark.memory.runner import _particles_context

        particles = [self._particle(f"c{i}") for i in range(5)]
        assert _particles_context(particles, {}) == _particles_context(
            particles, {}, budget_tokens=None
        )

    def test_run_selection_records_knobs(self) -> None:
        from particles.benchmark.memory.schema import RunSelection

        sel = RunSelection(dataset_revision="r", variant="s", sample_seed=1)
        assert sel.context_budget_tokens is None
        assert sel.abstraction is False
        knobbed = RunSelection(
            dataset_revision="r",
            variant="s",
            sample_seed=1,
            context_budget_tokens=2000,
            abstraction=True,
        )
        assert knobbed.context_budget_tokens == 2000
        assert knobbed.abstraction is True

    def test_renderer_discloses_ablation(self) -> None:
        from particles.benchmark.memory.schema import (
            MemoryBenchmarkReport,
            RetrievalStageMetrics,
            RunSelection,
            render_report_table,
        )

        report = MemoryBenchmarkReport(
            selection=RunSelection(
                dataset_revision="r",
                variant="s",
                sample_seed=1,
                context_budget_tokens=2000,
                abstraction=True,
            ),
            retrieval_stage=RetrievalStageMetrics(),
        )
        rendered = render_report_table(report)
        assert "context_budget=2000 tokens" in rendered
        assert "abstraction pass ON" in rendered
        assert "compare only against a matching run" in rendered


class TestProgressReporting:
    """One progress line per completed question — a multi-hour run is never silent."""

    @pytest.mark.asyncio
    async def test_progress_called_per_question_with_marks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        question = _question()
        orchestration = TestRunnerOrchestration()
        orchestration._patches(question, monkeypatch)
        monkeypatch.setattr(runner_mod, "get_provider", lambda purpose: _provider())
        replies = iter(["an answer", "yes"] * 3)
        monkeypatch.setattr(
            runner_mod, "complete", AsyncMock(side_effect=lambda *a, **k: next(replies))
        )

        lines: list[str] = []
        await run_memory_benchmark(
            [question],
            variant="oracle",
            dataset_revision="rev-test",
            selection_seed=13,
            selection_limit=1,
            questions_total=3,
            work_dir=tmp_path,
            keep_stores=True,
            progress=lines.append,
        )
        assert len(lines) == 1
        assert lines[0].startswith(f"[1/1] {question.question_id}:")
        assert "recall 1.00" in lines[0]
        assert "qa ✓ ✓ ✓" in lines[0]

    @pytest.mark.asyncio
    async def test_progress_line_without_qa(self, tmp_path: Path, monkeypatch) -> None:
        question = _question()
        orchestration = TestRunnerOrchestration()
        orchestration._patches(question, monkeypatch)

        lines: list[str] = []
        await run_memory_benchmark(
            [question],
            variant="oracle",
            dataset_revision="rev-test",
            selection_seed=13,
            selection_limit=1,
            questions_total=3,
            work_dir=tmp_path,
            keep_stores=True,
            qa=False,
            progress=lines.append,
        )
        assert len(lines) == 1
        assert "qa" not in lines[0]

    @pytest.mark.asyncio
    async def test_no_progress_callback_is_silent(self, tmp_path: Path, monkeypatch) -> None:
        # Default None: harness stays report-only with no output of its own.
        question = _question()
        orchestration = TestRunnerOrchestration()
        orchestration._patches(question, monkeypatch)
        report = await run_memory_benchmark(
            [question],
            variant="oracle",
            dataset_revision="rev-test",
            selection_seed=13,
            selection_limit=1,
            questions_total=3,
            work_dir=tmp_path,
            keep_stores=True,
            qa=False,
        )
        assert report.retrieval_stage.questions == 1


class TestConcurrency:
    """--concurrency: parallel questions, identical report, deduped extraction."""

    def _question_set(self) -> list[MemoryQuestion]:
        return [_question(qid=f"q{i}") for i in range(1, 4)]

    def _patch_all(self, questions: list[MemoryQuestion], monkeypatch) -> None:
        orchestration = TestRunnerOrchestration()
        # The orchestration patcher wires provenance rows for one question;
        # generalize: route each session's uri back to its own question.
        orchestration._patches(questions[0], monkeypatch)
        rows = {}
        for q in questions:
            rows["p1"] = (
                None,
                "CONVERSATION",
                "e1",
                session_uri(q.question_id, q.answer_session_ids[0]),
                None,
            )
        monkeypatch.setattr(runner_mod, "get_provider", lambda purpose: _provider())
        replies = ["an answer", "yes"] * (3 * len(questions))
        it = iter(replies)
        monkeypatch.setattr(runner_mod, "complete", AsyncMock(side_effect=lambda *a, **k: next(it)))

    @pytest.mark.asyncio
    async def test_concurrent_report_matches_sequential(self, tmp_path: Path, monkeypatch) -> None:
        questions = self._question_set()
        self._patch_all(questions, monkeypatch)

        async def _run(concurrency: int, work_dir: Path):
            return await run_memory_benchmark(
                questions,
                variant="oracle",
                dataset_revision="rev-test",
                selection_seed=13,
                selection_limit=3,
                questions_total=3,
                work_dir=work_dir,
                keep_stores=True,
                concurrency=concurrency,
            )

        sequential = await _run(1, tmp_path / "seq")
        self._patch_all(questions, monkeypatch)  # fresh reply iterator
        concurrent = await _run(3, tmp_path / "par")

        # Per-question rows in question order, aggregates identical.
        assert [r.question_id for r in concurrent.retrieval_stage.per_question] == [
            q.question_id for q in questions
        ]
        assert (
            concurrent.retrieval_stage.mean_recall_at_k
            == sequential.retrieval_stage.mean_recall_at_k
        )
        for condition in QA_CONDITIONS:
            seq_m = getattr(sequential, condition)
            par_m = getattr(concurrent, condition)
            assert (seq_m is None) == (par_m is None)
            if seq_m is not None:
                assert par_m.accuracy == seq_m.accuracy
                assert par_m.questions == seq_m.questions

    @pytest.mark.asyncio
    async def test_progress_counter_is_completion_ordered(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        questions = self._question_set()
        self._patch_all(questions, monkeypatch)
        lines: list[str] = []
        await run_memory_benchmark(
            questions,
            variant="oracle",
            dataset_revision="rev-test",
            selection_seed=13,
            selection_limit=3,
            questions_total=3,
            work_dir=tmp_path,
            keep_stores=True,
            concurrency=3,
            progress=lines.append,
        )
        assert len(lines) == 3
        # Counter is monotonic 1..N regardless of which question finished when.
        assert [line.split("]")[0] for line in lines] == ["[1/3", "[2/3", "[3/3"]

    @pytest.mark.asyncio
    async def test_caching_extractor_dedups_concurrent_same_session(self) -> None:
        from particles.core.schema import Snapshot

        calls = 0

        class _SlowInner:
            EXTRACTOR_ID = "slow"
            EXTRACTOR_VERSION = "1"

            def accepts(self, source_type: str) -> bool:
                return True

            async def extract(self, snapshot, content, **kwargs):
                nonlocal calls
                calls += 1
                await asyncio.sleep(0.01)
                return ExtractionResult(candidates=[], transient_error_count=0)

        caching = CachingExtractor(_SlowInner())  # type: ignore[arg-type]
        snap = Snapshot(
            corpus_entry_id="e1", content_hash="samehash", size_bytes=1, fetched_at=None
        )
        await asyncio.gather(*(caching.extract(snap, b"x") for _ in range(3)))
        assert calls == 1
        assert caching.misses == 1
        assert caching.hits == 2


class TestCheckpointing:
    """Question-level checkpoints: interruptible runs, free replay, knob isolation."""

    def _run_kwargs(self, tmp_path: Path) -> dict:
        return dict(
            variant="oracle",
            dataset_revision="rev-test",
            selection_seed=13,
            selection_limit=3,
            questions_total=3,
            keep_stores=True,
            checkpoint_dir=tmp_path / "ckpt",
        )

    @pytest.mark.asyncio
    async def test_completed_run_replays_from_checkpoint(self, tmp_path: Path, monkeypatch) -> None:
        questions = [_question(qid=f"q{i}") for i in range(1, 4)]
        helper = TestConcurrency()
        helper._patch_all(questions, monkeypatch)
        first = await run_memory_benchmark(
            questions, work_dir=tmp_path / "w1", **self._run_kwargs(tmp_path)
        )
        files = list((tmp_path / "ckpt").glob("memory-run-*.jsonl"))
        assert len(files) == 1

        # Second run: poison every pipeline seam — a full restore must not
        # touch any of them.
        boom = AsyncMock(side_effect=AssertionError("re-spend detected"))
        monkeypatch.setattr(runner_mod, "deposit_text_versioned", boom)
        monkeypatch.setattr(runner_mod, "extract_snapshot", boom)
        monkeypatch.setattr(runner_mod, "retrieve_ranked", boom)
        monkeypatch.setattr(runner_mod, "complete", boom)
        lines: list[str] = []
        second = await run_memory_benchmark(
            questions,
            work_dir=tmp_path / "w2",
            progress=lines.append,
            **self._run_kwargs(tmp_path),
        )
        assert second.retrieval_stage.mean_recall_at_k == first.retrieval_stage.mean_recall_at_k
        for condition in QA_CONDITIONS:
            first_m, second_m = getattr(first, condition), getattr(second, condition)
            assert (first_m is None) == (second_m is None)
            if first_m is not None:
                assert second_m.accuracy == first_m.accuracy
        assert any("resuming: 3/3" in line for line in lines)
        assert any("Checkpoint: restored 3" in note for note in second.quality_notes)

    @pytest.mark.asyncio
    async def test_partial_checkpoint_resumes_remainder(self, tmp_path: Path, monkeypatch) -> None:
        questions = [_question(qid=f"q{i}") for i in range(1, 4)]
        helper = TestConcurrency()
        helper._patch_all(questions, monkeypatch)
        await run_memory_benchmark(
            questions, work_dir=tmp_path / "w1", **self._run_kwargs(tmp_path)
        )
        ckpt = next((tmp_path / "ckpt").glob("memory-run-*.jsonl"))
        # Simulate an interruption after two questions: drop the last line.
        lines = ckpt.read_text().splitlines()
        ckpt.write_text("\n".join(lines[:-1]) + "\n")

        helper._patch_all(questions, monkeypatch)  # fresh (working) seams
        progress: list[str] = []
        report = await run_memory_benchmark(
            questions,
            work_dir=tmp_path / "w2",
            progress=progress.append,
            **self._run_kwargs(tmp_path),
        )
        assert any("resuming: 2/3" in line for line in progress)
        assert report.retrieval_stage.questions + report.retrieval_stage.abstention_questions == 3
        # The retried question was re-appended: replay is now complete.
        assert len(ckpt.read_text().splitlines()) == 4  # header + 3 outcomes

    @pytest.mark.asyncio
    async def test_knob_change_starts_fresh_file(self, tmp_path: Path, monkeypatch) -> None:
        questions = [_question(qid=f"q{i}") for i in range(1, 4)]
        helper = TestConcurrency()
        helper._patch_all(questions, monkeypatch)
        await run_memory_benchmark(
            questions, work_dir=tmp_path / "w1", **self._run_kwargs(tmp_path)
        )
        helper._patch_all(questions, monkeypatch)
        await run_memory_benchmark(
            questions,
            work_dir=tmp_path / "w2",
            context_budget=2000,  # different experiment
            **self._run_kwargs(tmp_path),
        )
        assert len(list((tmp_path / "ckpt").glob("memory-run-*.jsonl"))) == 2

    @pytest.mark.asyncio
    async def test_fresh_discards_checkpoint(self, tmp_path: Path, monkeypatch) -> None:
        questions = [_question(qid=f"q{i}") for i in range(1, 4)]
        helper = TestConcurrency()
        helper._patch_all(questions, monkeypatch)
        await run_memory_benchmark(
            questions, work_dir=tmp_path / "w1", **self._run_kwargs(tmp_path)
        )
        helper._patch_all(questions, monkeypatch)
        lines: list[str] = []
        report = await run_memory_benchmark(
            questions,
            work_dir=tmp_path / "w2",
            fresh=True,
            progress=lines.append,
            **self._run_kwargs(tmp_path),
        )
        assert not any("resuming" in line for line in lines)
        assert not any("Checkpoint" in note for note in report.quality_notes)

    @pytest.mark.asyncio
    async def test_foreign_header_ignored(self, tmp_path: Path, monkeypatch) -> None:
        questions = [_question(qid=f"q{i}") for i in range(1, 4)]
        helper = TestConcurrency()
        helper._patch_all(questions, monkeypatch)
        kwargs = self._run_kwargs(tmp_path)
        # Pre-create the file this run would use, but with a foreign header.
        await run_memory_benchmark(questions, work_dir=tmp_path / "w1", **kwargs)
        ckpt = next((tmp_path / "ckpt").glob("memory-run-*.jsonl"))
        content = ckpt.read_text().splitlines()
        foreign_header = json.dumps({"format": 1, "key": {"something": "else"}})
        ckpt.write_text("\n".join([foreign_header, *content[1:]]) + "\n")

        helper._patch_all(questions, monkeypatch)
        lines: list[str] = []
        await run_memory_benchmark(
            questions,
            work_dir=tmp_path / "w2",
            progress=lines.append,
            **kwargs,
        )
        assert not any("resuming" in line for line in lines)


class TestHeartbeat:
    """Status heartbeat: no minutes of silence while questions are in flight."""

    @pytest.mark.asyncio
    async def test_heartbeat_emits_while_questions_run(self, tmp_path: Path, monkeypatch) -> None:
        question = _question()
        orchestration = TestRunnerOrchestration()
        orchestration._patches(question, monkeypatch)

        # Make the single question slow enough for several heartbeat ticks.
        real_extract = runner_mod.extract_snapshot

        async def slow_extract(*args, **kwargs):
            await asyncio.sleep(0.08)
            return await real_extract(*args, **kwargs)

        monkeypatch.setattr(runner_mod, "extract_snapshot", slow_extract)

        lines: list[str] = []
        await run_memory_benchmark(
            [question],
            variant="oracle",
            dataset_revision="rev-test",
            selection_seed=13,
            selection_limit=1,
            questions_total=1,
            work_dir=tmp_path,
            keep_stores=True,
            qa=False,
            progress=lines.append,
            heartbeat_seconds=0.02,
        )
        heartbeats = [line for line in lines if "extraction calls so far" in line]
        assert heartbeats, f"no heartbeat lines in {lines!r}"
        assert "0/1 question(s) complete" in heartbeats[0]
        assert "elapsed" in heartbeats[0]
        # The per-question completion line still arrives after the first
        # heartbeat. Don't assert it is *last*: a final heartbeat tick can
        # legitimately land after it, which made this test wall-clock flaky.
        first_heartbeat = lines.index(heartbeats[0])
        completions = [i for i, line in enumerate(lines) if line.startswith("[1/1]")]
        assert completions, f"no [1/1] completion line in {lines!r}"
        assert completions[0] > first_heartbeat, f"completion preceded heartbeat: {lines!r}"

    @pytest.mark.asyncio
    async def test_no_heartbeat_by_default(self, tmp_path: Path, monkeypatch) -> None:
        question = _question()
        orchestration = TestRunnerOrchestration()
        orchestration._patches(question, monkeypatch)
        lines: list[str] = []
        await run_memory_benchmark(
            [question],
            variant="oracle",
            dataset_revision="rev-test",
            selection_seed=13,
            selection_limit=1,
            questions_total=1,
            work_dir=tmp_path,
            keep_stores=True,
            qa=False,
            progress=lines.append,
        )
        assert not any("extraction calls so far" in line for line in lines)


class TestCheckpointCrossSubset:
    """v1.77.4: outcomes restore across different question subsets (same experiment)."""

    @pytest.mark.asyncio
    async def test_smaller_limit_restores_overlap(self, tmp_path: Path, monkeypatch) -> None:
        questions = [_question(qid=f"q{i}") for i in range(1, 4)]
        helper = TestConcurrency()
        helper._patch_all(questions, monkeypatch)
        kwargs = dict(
            variant="oracle",
            dataset_revision="rev-test",
            selection_seed=13,
            keep_stores=True,
            checkpoint_dir=tmp_path / "ckpt",
        )
        await run_memory_benchmark(
            questions,
            selection_limit=3,
            questions_total=3,
            work_dir=tmp_path / "w1",
            **kwargs,
        )
        assert len(list((tmp_path / "ckpt").glob("memory-run-*.jsonl"))) == 1

        # A 2-question subset of the same experiment: poison the pipeline —
        # both questions must restore from the 3-question run's checkpoint.
        boom = AsyncMock(side_effect=AssertionError("re-spend detected"))
        monkeypatch.setattr(runner_mod, "deposit_text_versioned", boom)
        monkeypatch.setattr(runner_mod, "extract_snapshot", boom)
        monkeypatch.setattr(runner_mod, "retrieve_ranked", boom)
        monkeypatch.setattr(runner_mod, "complete", boom)
        lines: list[str] = []
        report = await run_memory_benchmark(
            questions[:2],
            selection_limit=2,
            questions_total=3,
            work_dir=tmp_path / "w2",
            progress=lines.append,
            **kwargs,
        )
        assert any("resuming: 2/2" in line for line in lines)
        assert len(list((tmp_path / "ckpt").glob("memory-run-*.jsonl"))) == 1  # same file
        assert report.retrieval_stage.per_question[0].question_id == "q1"

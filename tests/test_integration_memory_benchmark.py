"""Integration smoke test for the memory benchmark (run tiers).

Drives the checked-in 3-question synthetic oracle fixture end-to-end on a
developer key: real deposits into per-question scratch stores, real
extraction (candidate-cached), real ``retrieve_ranked`` retrieval with
provenance scoring, and real answer + judge calls under ``llm.benchmark_answer``
/ ``llm.benchmark``.

Per tests/AGENTS.md, this pins the response **contract** — all four
conditions populated and labeled, one pinned answer-model id stamped across
conditions ii–iv, both measurement families rendered — never model wording
or accuracy values (model behaviour assertions make the tier flaky). Tiny by
design (2 questions, oracle-shaped haystacks); CI never runs it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from particles.secrets import get_anthropic_api_key_optional

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        get_anthropic_api_key_optional() is None,
        reason="ANTHROPIC_API_KEY not set",
    ),
]

FIXTURE = (
    Path(__file__).parent
    / "benchmark"
    / "memory"
    / "fixtures"
    / "longmemeval_oracle_synthetic.json"
)


async def test_fixture_end_to_end_contract(tmp_path: Path) -> None:
    from particles.benchmark.memory import (
        QA_CONDITIONS,
        load_dataset_file,
        render_report_table,
        run_memory_benchmark,
        select_questions,
    )

    questions = select_questions(load_dataset_file(FIXTURE), seed=13, limit=2)
    assert len(questions) == 2

    report = await run_memory_benchmark(
        questions,
        variant="oracle",
        dataset_revision="fixture-synthetic",
        selection_seed=13,
        selection_limit=2,
        questions_total=3,
        work_dir=tmp_path / "stores",
        keep_stores=True,
    )

    # Condition i: retrieval scored through real provenance chains for every
    # question (values are model/pipeline-dependent; presence is the contract).
    # A selected abstention question (zero labeled evidence) is unscoreable:
    # its row is marked n/a and excluded from the aggregates (v1.74.2).
    retrieval = report.retrieval_stage
    assert len(retrieval.per_question) == 2
    assert retrieval.questions + retrieval.abstention_questions == 2
    for row in retrieval.per_question:
        if row.abstention:
            assert row.recall_at_k is None
            assert row.precision_at_k is None
        else:
            assert row.recall_at_k is not None and 0.0 <= row.recall_at_k <= 1.0
            assert row.precision_at_k is not None and 0.0 <= row.precision_at_k <= 1.0

    # The §5 comparability tuple records the resolved pipeline model ids.
    assert report.selection.extraction_model_id
    assert report.selection.embedding_model_id

    # Conditions ii–iv: all populated, all stamped with ONE answer model id.
    model_ids = set()
    for condition in QA_CONDITIONS:
        metrics = getattr(report, condition)
        assert metrics is not None, f"{condition} must be populated"
        assert metrics.condition == condition
        assert metrics.questions == 2
        model_ids.add(metrics.model_id)
    assert len(model_ids) == 1
    assert report.selection.answer_model_id in model_ids
    assert report.selection.judge_model_id

    # The rendered table shows the two labeled families and the subset header.
    table = render_report_table(report)
    assert "== Retrieval stage" in table
    assert "== End-to-end QA" in table
    assert "SUBSET run — 2 of 3 questions" in table

"""Live journal modality benchmark — integration tier.

Runs the bundled seed suite through the **real** journal extractor (one LLM
call per entry) with **real** embeddings (alignment of reified gold labels to
emitted claims is genuinely semantic). Pins the report *contract* — every case
ran, claims aligned, the headline rates are well-formed and in range, the
confusion accounting is consistent — never specific precision values or
model wording, which would make the tier flaky (tests/AGENTS.md § Integration
tests).

Run with ``uv run pytest tests/`` (no ``-m`` filter) on a developer key; CI
excludes the tier with ``-m "not integration"``. Without ANTHROPIC_API_KEY the
module skips wholesale.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from particles.benchmark.equivalence import EquivalenceJudge
from particles.benchmark.modality import load_modality_suite, run_modality_benchmark
from particles.secrets import get_anthropic_api_key_optional

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        get_anthropic_api_key_optional() is None,
        reason="integration tier requires ANTHROPIC_API_KEY (tests/AGENTS.md § Integration tests)",
    ),
]

_SEED = Path("tests/benchmark/modality/journal-modality-seed-001.yaml")


def _get_journal_extractor() -> object:
    from particles.extraction.registry import get_extractors

    for e in get_extractors():
        if e.EXTRACTOR_ID == "journal-extractor":
            return e
    raise AssertionError("journal-extractor not registered")


async def test_live_modality_benchmark_report_contract() -> None:
    suite = load_modality_suite(_SEED)
    extractor = _get_journal_extractor()

    report = await run_modality_benchmark(
        suite, extractor, judge=EquivalenceJudge.EMBEDDING, threshold=0.6
    )

    # Every seed case ran (none declined — the journal extractor accepts JOURNAL).
    assert report.cases_run == report.cases_total == len(suite.cases)
    assert report.extractor_id == "journal-extractor"
    assert report.suite_id == "journal-modality-seed-001"

    # Real extractor + real embeddings should align at least one claim per case
    # on average — the metric is meaningless with zero alignment.
    assert report.claims_aligned >= 1

    # Headline rates are well-formed and in range (no value assertions — model
    # behaviour is not pinned).
    assert 0.0 <= report.false_non_falsifiable_rate <= 1.0
    assert report.narrative_cases_expected == len(suite.cases)
    assert 0.0 <= report.narrative_emission_rate <= 1.0

    # Per-modality precision/recall are all probabilities.
    for value in (*report.precision.values(), *report.recall.values()):
        assert 0.0 <= value <= 1.0

    # The confusion accounting is internally consistent with the aligned count.
    assert sum(c.count for c in report.confusion) == report.claims_aligned

    # FALSIFIABLE recall and the false-non-FALSIFIABLE rate are complements
    # whenever any FALSIFIABLE label aligned (the suite guarantees several).
    falsifiable_aligned = sum(c.count for c in report.confusion if c.expected == "FALSIFIABLE")
    if falsifiable_aligned:
        assert report.false_non_falsifiable_rate == pytest.approx(
            1.0 - report.recall["FALSIFIABLE"], abs=1e-6
        )

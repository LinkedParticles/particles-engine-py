"""Live claim-polarity benchmark — integration tier.

Runs the bundled seed suite through the **real** general extractor (one LLM
call per case) with **real** embeddings (alignment of declarative gold labels to
emitted claims is genuinely semantic). Pins the report *contract* — every case
ran, claims aligned, the headline rates are well-formed and in range, the
confusion accounting is consistent — never specific precision values or
model wording, which would make the tier flaky (tests/AGENTS.md § Integration
tests).

Sibling of tests/test_integration_journal_modality.py (the modality benchmark's
live tier). Distinct from tests/test_integration_polarity.py, which is the cap. 1 *classifier* acceptance check (does rejected / deferred prose
land non-asserted through the real pipeline); this one exercises the
*benchmark harness* over that classifier.

Run with ``uv run pytest tests/`` (no ``-m`` filter) on a developer key; CI
excludes the tier with ``-m "not integration"``. Without ANTHROPIC_API_KEY the
module skips wholesale.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from particles.benchmark.equivalence import EquivalenceJudge
from particles.benchmark.polarity import load_polarity_suite, run_polarity_benchmark
from particles.secrets import get_anthropic_api_key_optional

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        get_anthropic_api_key_optional() is None,
        reason="integration tier requires ANTHROPIC_API_KEY (tests/AGENTS.md § Integration tests)",
    ),
]

_SEED = Path("tests/benchmark/polarity/adr-polarity-seed-001.yaml")


def _get_general_extractor() -> object:
    from particles.extraction.registry import get_extractors

    for e in get_extractors():
        if e.EXTRACTOR_ID == "general-extractor":
            return e
    raise AssertionError("general-extractor not registered")


async def test_live_polarity_benchmark_report_contract() -> None:
    suite = load_polarity_suite(_SEED)
    extractor = _get_general_extractor()

    report = await run_polarity_benchmark(
        suite, extractor, judge=EquivalenceJudge.EMBEDDING, threshold=0.6
    )

    # Every seed case ran (none declined — the general extractor accepts any type).
    assert report.cases_run == report.cases_total == len(suite.cases)
    assert report.extractor_id == "general-extractor"
    assert report.suite_id == "adr-polarity-seed-001"

    # The classifier must be on for the metric to be meaningful (config default).
    assert report.polarity_classifier_enabled is True

    # Real extractor + real embeddings should align at least one claim overall —
    # the metric is meaningless with zero alignment.
    assert report.claims_aligned >= 1

    # Headline rates are well-formed and in range (no value assertions — model
    # behaviour is not pinned).
    assert 0.0 <= report.wrong_declined_rate <= 1.0
    assert 0.0 <= report.wrong_hidden_rate <= 1.0
    # The hidden rate is the superset, so it can never be below the DECLINED-only rate.
    assert report.wrong_hidden_rate >= report.wrong_declined_rate

    # Per-polarity precision/recall are all probabilities.
    for value in (*report.precision.values(), *report.recall.values()):
        assert 0.0 <= value <= 1.0

    # The confusion accounting is internally consistent with the aligned count.
    assert sum(c.count for c in report.confusion) == report.claims_aligned

    # ASSERTED recall and the wrong-hidden rate are complements whenever any
    # ASSERTED label aligned (the suite guarantees several).
    asserted_aligned = sum(c.count for c in report.confusion if c.expected == "ASSERTED")
    if asserted_aligned:
        assert report.wrong_hidden_rate == pytest.approx(1.0 - report.recall["ASSERTED"], abs=1e-6)

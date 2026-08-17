"""Live event-anchored-validity benchmark — integration tier.

Runs the bundled seed suite through the **real** general extractor (one LLM call
per case) with **real** embeddings. Pins the report *contract* — every case ran,
claims aligned, the headline rate and precision/recall are well-formed and in
range, the support accounting is consistent — never specific values or model
wording, which would make the tier flaky (tests/AGENTS.md § Integration tests).

The one behavioural assertion the seed is designed to make safe: the
``all-durable-decoys`` case is entirely durable facts that merely mention dates
("the treaty was signed in 1919", "Marie Curie was born in 1867"), so a
well-behaved extractor should assign **no** ``valid_until`` there — we assert the
headline over-eager-expiry rate stays at or below a generous ceiling rather than
pinning an exact number.

Sibling of tests/test_integration_polarity_benchmark.py. Run with
``uv run pytest tests/`` (no ``-m`` filter) on a developer key; CI excludes the
tier with ``-m "not integration"``. Without ANTHROPIC_API_KEY the module skips.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from particles.benchmark.equivalence import EquivalenceJudge
from particles.benchmark.validity import load_validity_suite, run_validity_benchmark
from particles.secrets import get_anthropic_api_key_optional

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        get_anthropic_api_key_optional() is None,
        reason="integration tier requires ANTHROPIC_API_KEY (tests/AGENTS.md § Integration tests)",
    ),
]

_SEED = Path("tests/benchmark/validity/durability-seed-001.yaml")


def _get_general_extractor() -> object:
    from particles.extraction.registry import get_extractors

    for e in get_extractors():
        if e.EXTRACTOR_ID == "general-extractor":
            return e
    raise AssertionError("general-extractor not registered")


async def test_live_validity_benchmark_report_contract() -> None:
    suite = load_validity_suite(_SEED)
    extractor = _get_general_extractor()

    report = await run_validity_benchmark(
        suite, extractor, judge=EquivalenceJudge.EMBEDDING, threshold=0.6
    )

    # Every seed case ran (the general extractor accepts any type).
    assert report.cases_run == report.cases_total == len(suite.cases)
    assert report.extractor_id == "general-extractor"
    assert report.suite_id == "durability-seed-001"

    # The extractor must be on for the metric to be meaningful (config default).
    assert report.validity_extractor_enabled is True

    # Real extractor + real embeddings should align at least one claim overall.
    assert report.claims_aligned >= 1

    # All headline metrics are well-formed probabilities.
    for value in (
        report.wrong_expiry_rate,
        report.expiry_precision,
        report.expiry_recall,
        report.date_accuracy,
    ):
        assert 0.0 <= value <= 1.0

    # Support accounting is internally consistent: durable + bounded == aligned,
    # and every emitted/both count is bounded by the aligned total.
    s = report.support
    assert s["durable"] + s["bounded"] == report.claims_aligned
    assert s["both"] <= s["emitted"]
    assert s["both"] <= s["bounded"]

    # Over-eager-expiry ceiling: durable facts that merely mention dates should
    # mostly stay unbounded. Generous bound (the danger metric, not a value pin)
    # — asserted only when durable decoys actually aligned.
    if s["durable"] >= 1:
        assert report.wrong_expiry_rate <= 0.5

# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Integration smoke test for the memory-rot benchmark's paid arm.

Drives the ``probe`` arm over the smallest legal world (30 days, one
checkpoint) on a developer key: scripted extraction, the **live** §6.6
contradiction probe under ``llm.semantic_lint``, real ``retrieve_ranked``.
The probe only fires on candidate pairs above the similarity threshold, so a
run is a handful of short calls — cents at most.

Per tests/AGENTS.md this pins the response **contract** — the run tuple names
the live probe model, no purpose other than ``semantic_lint`` was reached, all
three families are populated — never a metric value (model behaviour
assertions make the tier flaky). CI never runs it.
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


async def test_probe_arm_contract(tmp_path: Path) -> None:
    from particles.benchmark.rot import render_report, run_rot_benchmark

    report = await run_rot_benchmark(
        arm="probe", seeds=[42], days=30, checkpoints=[30], work_dir=tmp_path
    )
    sel = report.selection
    assert sel.arm == "probe"
    assert sel.extraction_model_id == "rot-oracle:scripted-extractor"
    assert not sel.semantic_lint_model_id.startswith("rot-")
    # The scripted extractor makes no call, so nothing else should be reached.
    assert "semantic_lint" not in report.refused_llm_calls
    assert report.metrics.recall_current_at_k.denominator > 0
    table = render_report(report)
    assert "CURRENCY" in table and "SUPERSESSION" in table and "POISON" in table

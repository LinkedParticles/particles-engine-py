# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Live half of the abstraction premise-scope negative cases.

Drives the abstraction pass's checked-in §8 negative cases,
``tests/benchmark/abstraction/premise-scope-negative-001.yaml``, through the
real ``llm.abstraction`` route. Two things are pinned:

- **The gate contract.** A full ``_promote_cluster`` over each case's premises
  synthesizes one candidate, and any candidate that survives the gate is
  premise-scoped: it clears :func:`population_quantifier`.
- **The gate rejects every generic.** The composed gate, as production runs
  it, rejects every population generic in the file: the deterministic
  pre-check takes ``population_generics`` and the live entailment judge takes
  the ``judge_only_generics`` the pre-check lets through.

The judge half is a model-behaviour assertion, which tests/AGENTS.md
§ Integration tests normally avoids. It is kept on purpose, because it is the
negative case itself, and it is confined to the generics the pre-check cannot
see. The judge alone is not asserted against the full list: on the first live
run (2026-10-03, ``claude-sonnet-4-6``) it accepted every generic carrying an
evidential clause ("…, as these four records show") as scoped, and once
returned ``entailed: true`` for "Most operators use zsh" with a reason that
contradicted the verdict. Inputs are tiny (two clusters, a few judge calls).
CI never runs this tier.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from particles.config import get_config
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.operations import abstraction as ab
from particles.secrets import get_anthropic_api_key_optional
from particles.store.event_store import OperatorEventType, list_events
from particles.store.particle_store import insert_particle

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        get_anthropic_api_key_optional() is None,
        reason="ANTHROPIC_API_KEY not set",
    ),
]


CASES_PATH = Path(__file__).parent / "benchmark" / "abstraction" / "premise-scope-negative-001.yaml"


def _specific(content: str) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1", snapshot_id="s1")
        ],
        asserted_by="general-extractor",
        asserted_at=datetime(2026, 1, 1, tzinfo=UTC),
        subject_ids=["subj-1"],
    )


def _cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    return cases


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["case_id"])
async def test_gate_rejects_every_generic(case: dict[str, Any]) -> None:
    """The composed gate (pre-check, then judge) rejects every generic in the case."""
    for claim in case["population_generics"]:
        assert ab.population_quantifier(claim) is not None, claim
    for claim in case["judge_only_generics"]:
        entailed = await ab._check_entailment(claim, case["premises"])
        assert entailed is False, claim


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["case_id"])
async def test_only_premise_scoped_candidates_survive_the_gate(
    db_session, case: dict[str, Any]
) -> None:
    get_config().consolidation.abstraction.enabled = True
    premises = [_specific(c) for c in case["premises"]]
    for p in premises:
        await insert_particle(db_session, p)
    report = ab.AbstractionReport(mode="propose")
    with patch.object(ab, "_duplicate_of", new=AsyncMock(return_value=None)):
        await ab._promote_cluster(db_session, ab._Cluster("subj-1", premises), report)

    assert report.candidates_synthesized == 1
    events = await list_events(db_session, event_type=OperatorEventType.ABSTRACTION_CANDIDATE)
    for event in events:
        assert event.payload is not None
        assert ab.population_quantifier(event.payload["claim"]) is None, event.payload["claim"]

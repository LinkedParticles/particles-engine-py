# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The abstraction pass's premise-scope rule (addendum).

A promoted abstraction ranges over its observed premise set, never over a kind
or a population. Covers the deterministic pre-check
(:func:`particles.operations.abstraction.population_quantifier`) against the
checked-in §8 negative cases and a table of edge shapes, its place in the
promotion gate (before the judge call is spent, in both modes, regardless of
``require_entailment``), the §5 rung-3 re-synthesis path, and the rule's
presence in both prompts. LLM calls are mocked at the module binding
(``particles.operations.abstraction._llm_call``), per tests/AGENTS.md
§ Mocking strategy. The live judge half of the negative cases is
``tests/test_integration_abstraction_scope.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.operations import abstraction as ab
from particles.store.event_store import OperatorEventType, list_events
from particles.store.particle_store import (
    get_particle,
    insert_particle,
    update_particle_status,
)

CASES_PATH = Path(__file__).parent / "benchmark" / "abstraction" / "premise-scope-negative-001.yaml"
OLD = datetime(2026, 1, 1, tzinfo=UTC)


def _load_cases() -> list[dict[str, Any]]:
    suite = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = suite["cases"]
    return cases


CASES = _load_cases()


def _claims(key: str) -> list[tuple[str, str]]:
    return [(case["case_id"], claim) for case in CASES for claim in case[key]]


def _specific(content: str) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1", snapshot_id="s1")
        ],
        asserted_by="general-extractor",
        asserted_at=OLD,
        subject_ids=["subj-1"],
    )


def _enable(**overrides: object) -> None:
    from particles.config import get_config

    cfg = get_config().consolidation.abstraction
    cfg.enabled = True
    for key, value in overrides.items():
        setattr(cfg, key, value)


class _ScriptedLLM:
    """Replays canned replies and records each call's system prompt."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = iter(replies)
        self.systems: list[str] = []

    async def __call__(self, *args: object, **kwargs: object) -> str:
        self.systems.append(str(kwargs.get("system", "")))
        return next(self._replies)


# ---------------------------------------------------------------------------
# The checked-in §8 negative cases
# ---------------------------------------------------------------------------


class TestNegativeCaseFile:
    def test_every_case_is_well_formed(self) -> None:
        assert CASES, "the negative-case file carries no cases"
        for case in CASES:
            assert len(case["premises"]) >= 3, case["case_id"]  # min_cluster_size
            assert case["population_generics"], case["case_id"]
            assert case["premise_scoped"], case["case_id"]

    def test_premises_are_instance_claims(self) -> None:
        """The premises themselves would pass: each is about one named member."""
        for case in CASES:
            for premise in case["premises"]:
                assert ab.population_quantifier(premise) is None, premise

    @pytest.mark.parametrize(("case_id", "claim"), _claims("population_generics"))
    def test_population_generic_rejected(self, case_id: str, claim: str) -> None:
        assert ab.population_quantifier(claim) is not None, (case_id, claim)

    @pytest.mark.parametrize(("case_id", "claim"), _claims("premise_scoped"))
    def test_premise_scoped_form_passes(self, case_id: str, claim: str) -> None:
        assert ab.population_quantifier(claim) is None, (case_id, claim)

    @pytest.mark.parametrize(("case_id", "claim"), _claims("judge_only_generics"))
    def test_judge_only_generic_reaches_the_judge(self, case_id: str, claim: str) -> None:
        """Pins the division of labour: these are the entailment judge's to reject."""
        assert ab.population_quantifier(claim) is None, (case_id, claim)


# ---------------------------------------------------------------------------
# Edge shapes of the pre-check
# ---------------------------------------------------------------------------


class TestPopulationQuantifier:
    @pytest.mark.parametrize(
        ("claim", "matched"),
        [
            ("Most Christians attend church weekly.", "Most"),
            ("All mammals bear live young.", "All"),
            ("Mammals bear live young.", "bare plural subject + 'bear'"),
            ("Christians generally attend church.", "generally"),
            ("Generally, Christians attend church.", "Generally"),
            ("Christians in general attend church.", "in general"),
            ("Engineers tend to prefer dark mode.", "tend to"),
            ("Rust users typically prefer vim keybindings.", "typically"),
            ("Everyone on the team uses vim.", "Everyone"),
            ("Backgrounded commits fail at GPG signing.", "bare plural subject + 'fail'"),
            ("Women are more cautious drivers.", "bare plural subject + 'are'"),
            ("The majority of users disable telemetry.", "The majority of"),
            ("Each deploy used the blue-green strategy.", "Each"),
        ],
    )
    def test_kind_ranging_quantifier_flagged(self, claim: str, matched: str) -> None:
        assert ab.population_quantifier(claim) == matched

    @pytest.mark.parametrize(
        "claim",
        [
            # Bound to the observed premise set.
            "Every backgrounded commit in this store failed at GPG signing.",
            "All ten operators observed prefer vim.",
            "Each of the three recorded deploys used the blue-green strategy.",
            "These three members of the parish attend church weekly.",
            "Alice, Bob, and Carol all prefer vim.",
            "Alice, Bob & Carol each attend Mass weekly.",
            # About an individual, not a kind: the judge's call.
            "Alice typically uses vim.",
            "Alice uses vim, tmux and zsh.",
            "The operator's backgrounded commits fail at GPG signing.",
            "Alice's commits are signed with an ed25519 key.",
            # Non-quantifying uses of the quantifier words.
            "The deploy script is the most reliable path.",
            "The user is most likely vegetarian.",
            "The import is not at all idempotent.",
            "Alice and Bob review each other's pull requests.",
            "The job retries at most three times.",
            # No quantifier at all.
            "General claim.",
        ],
    )
    def test_premise_scoped_or_unquantified_passes(self, claim: str) -> None:
        assert ab.population_quantifier(claim) is None

    def test_those_who_names_a_kind(self) -> None:
        """``those who`` is a kind, not a pointer at the premise set."""
        assert ab.population_quantifier("Most of those who attend church vote.") == "Most"

    def test_evidential_clause_does_not_scope(self) -> None:
        """Citing the observations is not restricting the subject to them."""
        claim = "Parishioners volunteer at the food bank, as these four records show."
        assert ab.population_quantifier(claim) == "as these four records show"
        # The clause is the flag even where the subject alone would be clean.
        claim = "Alice uses vim, as the recorded sessions show."
        assert ab.population_quantifier(claim) == "as the recorded sessions show"
        claim = "Most parishioners volunteer, according to the recorded rota."
        assert ab.population_quantifier(claim) == "Most"

    def test_known_false_rejection_shape(self) -> None:
        """A name that only looks plural is the documented false rejection."""
        assert ab.population_quantifier("James typically uses vim.") == "typically"


# ---------------------------------------------------------------------------
# The rule in the gate (§2 promotion, §5 rung 3) and in the prompts
# ---------------------------------------------------------------------------


class TestPromotionGate:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("require_entailment", [True, False])
    async def test_population_generic_discarded_before_judge(
        self, db_session, require_entailment: bool
    ) -> None:
        _enable(require_entailment=require_entailment)
        case = CASES[0]
        premises = [_specific(c) for c in case["premises"]]
        cluster = ab._Cluster("subj-1", premises)
        llm = _ScriptedLLM(
            [json.dumps({"claim": case["population_generics"][0], "rationale": "r"})]
        )

        report = ab.AbstractionReport(mode="propose")
        with patch.object(ab, "_llm_call", new=llm):
            await ab._promote_cluster(db_session, cluster, report)

        assert report.rejected_population_scope == 1
        assert report.rejected_entailment == 0
        assert len(llm.systems) == 1  # synthesis only: the judge call was never spent
        assert report.llm_calls == 1
        assert report.proposed_event_ids == []
        assert (
            await list_events(db_session, event_type=OperatorEventType.ABSTRACTION_CANDIDATE) == []
        )

    @pytest.mark.asyncio
    async def test_premise_scoped_candidate_reaches_judge_and_is_proposed(self, db_session) -> None:
        _enable()
        case = CASES[0]
        premises = [_specific(c) for c in case["premises"]]
        for p in premises:
            await insert_particle(db_session, p)
        cluster = ab._Cluster("subj-1", premises)
        scoped = case["premise_scoped"][0]
        llm = _ScriptedLLM(
            [
                json.dumps({"claim": scoped, "rationale": "r"}),
                json.dumps({"entailed": True, "reason": "ok"}),
            ]
        )

        report = ab.AbstractionReport(mode="propose")
        with (
            patch.object(ab, "_llm_call", new=llm),
            patch.object(ab, "_duplicate_of", new=AsyncMock(return_value=None)),
        ):
            await ab._promote_cluster(db_session, cluster, report)

        assert report.rejected_population_scope == 0
        assert len(llm.systems) == 2
        events = await list_events(db_session, event_type=OperatorEventType.ABSTRACTION_CANDIDATE)
        assert len(events) == 1
        assert events[0].payload is not None
        assert events[0].payload["claim"] == scoped

    @pytest.mark.asyncio
    async def test_both_prompts_state_the_rule(self, db_session) -> None:
        _enable()
        premises = [_specific(f"specific claim {i}") for i in range(3)]
        cluster = ab._Cluster("subj-1", premises)
        llm = _ScriptedLLM(
            [
                json.dumps({"claim": "General claim.", "rationale": "r"}),
                json.dumps({"entailed": False, "reason": "no"}),
            ]
        )
        with patch.object(ab, "_llm_call", new=llm):
            await ab._promote_cluster(db_session, cluster, ab.AbstractionReport())

        synthesis_system, judge_system = llm.systems
        assert "never to a kind or a" in synthesis_system
        assert "observed set" in synthesis_system
        assert "kind or a population" in judge_system
        assert "never entailed by claims about particular members" in judge_system


class TestRevalidationRung3:
    @pytest.mark.asyncio
    async def test_population_resynthesis_defers_without_paraphrase_call(self, db_session) -> None:
        _enable()
        premises = [_specific(f"specific claim {i}") for i in range(3)]
        for p in premises:
            await insert_particle(db_session, p)
        d = ab._build_derived_particle(
            claim="General claim.", premises=premises, subject_ids=["subj-1"]
        )
        await insert_particle(db_session, d)
        old = premises[0]
        successor = _specific("materially different claim").model_copy(
            update={"supersedes": old.id}
        )
        await insert_particle(db_session, successor)
        await update_particle_status(
            db_session, old.id, Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION
        )
        llm = _ScriptedLLM(
            [
                json.dumps({"entailed": False, "reason": "no longer supported"}),
                json.dumps({"claim": "Most operators use zsh.", "rationale": "r"}),
            ]
        )

        report = ab.AbstractionReport()
        with patch.object(ab, "_llm_call", new=llm):
            await ab._revalidate(db_session, report, ab._Budget(5))

        assert report.revalidation.deferred == 1
        assert report.revalidation.superseded == 0
        assert len(llm.systems) == 2  # judge + re-synthesis; no paraphrase call
        unchanged = await get_particle(db_session, d.id)
        assert unchanged is not None and unchanged.status is Status.ACTIVE

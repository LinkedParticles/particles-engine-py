# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for operations/lint.py — §9.4."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from particles.core.schema import (
    AssertionModality,
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.store.particle_store import insert_particle

# L-SEM-01 similarity-gate fixtures. MiniLM is 384-dim.
# _EMB_HI_A / _EMB_HI_B point the same way (cosine ≈ 1.0, above the 0.6 default
# gate); _EMB_LOW is orthogonal (cosine 0.0, below the gate).
_EMB_HI_A = (np.array([0.6, 0.8] + [0.0] * 382, dtype=np.float32)).tolist()
_EMB_HI_B = (np.array([0.61, 0.79] + [0.0] * 382, dtype=np.float32)).tolist()
_EMB_LOW = (np.array([0.0, 0.0, 1.0] + [0.0] * 381, dtype=np.float32)).tolist()


def _make_particle(
    content: str = "Test claim.",
    status: Status = Status.ACTIVE,
    confidence: float = 0.8,
    valid_until: datetime | None = None,
) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(
            value=confidence, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test-agent",
        status=status,
        valid_until=valid_until,
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1")],
    )


def _agent_particle(content: str, *, asserted_by: str, calib: CalibrationSource) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.8, calibration_source=calib),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by=asserted_by,
        status=Status.ACTIVE,
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1")],
    )


@pytest.mark.asyncio
async def test_compound_assertion_flagged(db_session: object) -> None:
    """An agent-asserted ACTIVE particle breaching the granularity gate → COMPOUND_ASSERTION."""
    from particles.config import get_config
    from particles.operations.lint import run_lint

    session = db_session  # type: ignore[assignment]
    identity = get_config().mcp.write.asserter_identity
    compound = _agent_particle(
        "This is claim one. " * 20,  # ~380 chars, 20 sentences
        asserted_by=identity,
        calib=CalibrationSource.AGENT_ASSERTED,
    )
    await insert_particle(session, compound)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    report = await run_lint(session, fix=False, semantic=False)  # type: ignore[arg-type]
    flagged = [f.particle_id for f in report.findings if f.finding_type == "COMPOUND_ASSERTION"]
    assert compound.id in flagged


@pytest.mark.asyncio
async def test_compound_assertion_ignores_extractor_and_short(db_session: object) -> None:
    """Only agent-asserted over-gate particles flag — not extractor or short claims."""
    from particles.config import get_config
    from particles.operations.lint import run_lint

    session = db_session  # type: ignore[assignment]
    identity = get_config().mcp.write.asserter_identity
    extractor_compound = _agent_particle(
        "This is claim one. " * 20,
        asserted_by="general-extractor",  # not the agent identity
        calib=CalibrationSource.EXTRACTOR_DIRECT,
    )
    short_agent = _agent_particle(
        "Mercury is the closest planet to the Sun.",  # atomic, under the gate
        asserted_by=identity,
        calib=CalibrationSource.AGENT_ASSERTED,
    )
    await insert_particle(session, extractor_compound)  # type: ignore[arg-type]
    await insert_particle(session, short_agent)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    report = await run_lint(session, fix=False, semantic=False)  # type: ignore[arg-type]
    flagged = [f.particle_id for f in report.findings if f.finding_type == "COMPOUND_ASSERTION"]
    assert extractor_compound.id not in flagged
    assert short_agent.id not in flagged


@pytest.mark.asyncio
async def test_staleness_detection(db_session: object) -> None:
    """Lint detects ACTIVE particles whose valid_until has passed."""
    from particles.operations.lint import run_lint

    session = db_session  # type: ignore[assignment]
    expired = _make_particle(valid_until=datetime.now(UTC) - timedelta(hours=1))
    await insert_particle(session, expired)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    report = await run_lint(session, fix=True, semantic=False)  # type: ignore[arg-type]
    finding_types = [f.finding_type for f in report.findings]
    assert "STALENESS" in finding_types

    # After fix, particle should be PROVENANCE_STALE
    from particles.store.particle_store import get_particle

    updated = await get_particle(session, expired.id)  # type: ignore[arg-type]
    assert updated is not None
    assert updated.status == Status.PROVENANCE_STALE
    assert updated.status_reason == StatusReason.VALIDITY_EXPIRED


@pytest.mark.asyncio
async def test_no_findings_clean_store(db_session: object) -> None:
    """Lint on a clean store produces no ERROR findings."""
    from particles.operations.lint import run_lint

    session = db_session  # type: ignore[assignment]
    p = _make_particle()
    await insert_particle(session, p)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    report = await run_lint(session, fix=False, semantic=False)  # type: ignore[arg-type]
    errors = [f for f in report.findings if f.severity == "ERROR"]
    assert errors == []


@pytest.mark.asyncio
async def test_schema_version_audit(db_session: object) -> None:
    """Lint reports particles with wrong schema_version."""
    from particles.operations.lint import run_lint

    session = db_session  # type: ignore[assignment]
    p = _make_particle()
    # Manually set old schema version via model_copy
    old_p = p.model_copy(update={"schema_version": "0.1.0"})
    await insert_particle(session, old_p)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    report = await run_lint(session, fix=False, semantic=False)  # type: ignore[arg-type]
    finding_types = [f.finding_type for f in report.findings]
    assert "SCHEMA_VERSION_MISMATCH" in finding_types


@pytest.mark.asyncio
async def test_extraction_quality_report(db_session: object) -> None:
    """Lint reports EXTRACTOR_DIRECT fraction."""
    from particles.operations.lint import run_lint

    session = db_session  # type: ignore[assignment]
    for i in range(3):
        await insert_particle(session, _make_particle(f"Claim {i}."))  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    report = await run_lint(session, fix=False, semantic=False)  # type: ignore[arg-type]
    finding_types = [f.finding_type for f in report.findings]
    assert "EXTRACTION_QUALITY_REPORT" in finding_types


@pytest.mark.asyncio
async def test_particle_scoped_finding_carries_claim_text(db_session: object) -> None:
    """A finding with a particle_id is enriched with that particle's claim text.

    The STALENESS finder holds the Particle at the point of construction, so the
    finding's ``particle_content`` is the particle's ``content`` verbatim — a
    curation client shows WHAT is flagged without a second ``particles show``.
    """
    from particles.operations.lint import run_lint

    session = db_session  # type: ignore[assignment]
    claim = "The euro became legal tender on 1 January 1999."
    expired = _make_particle(content=claim, valid_until=datetime.now(UTC) - timedelta(hours=1))
    await insert_particle(session, expired)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    report = await run_lint(session, fix=False, semantic=False)  # type: ignore[arg-type]
    staleness = [f for f in report.findings if f.finding_type == "STALENESS"]
    assert staleness, "expected a STALENESS finding for the expired particle"
    assert staleness[0].particle_id == expired.id
    assert staleness[0].particle_content == claim


@pytest.mark.asyncio
async def test_non_particle_finding_has_no_claim_text(db_session: object) -> None:
    """A finding with no particle_id leaves particle_content None.

    EXTRACTION_QUALITY_REPORT is a store-level aggregate carrying no
    ``particle_id``; the enrichment field stays None for it.
    """
    from particles.operations.lint import run_lint

    session = db_session  # type: ignore[assignment]
    for i in range(3):
        await insert_particle(session, _make_particle(f"Claim {i}."))  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    report = await run_lint(session, fix=False, semantic=False)  # type: ignore[arg-type]
    aggregates = [f for f in report.findings if f.finding_type == "EXTRACTION_QUALITY_REPORT"]
    assert aggregates, "expected an EXTRACTION_QUALITY_REPORT finding"
    assert aggregates[0].particle_id is None
    assert aggregates[0].particle_content is None


@pytest.mark.asyncio
async def test_lsem01_skips_co_evidential_pairs(db_session: object) -> None:
    """The LLM contradiction check (L-SEM-01) skips pairs already linked CO_EVIDENTIAL.

    Asserts the skip by patching the LLM helper and verifying it isn't called for
    the linked pair — running it would be a false-positive risk, since the pair
    has been judged a paraphrase of the same claim, not a contradiction.
    """
    from unittest.mock import patch

    from particles.core.schema import RelationCreatedBy, RelationType
    from particles.operations.lint import _check_contradictions
    from particles.store.relation_store import create_relation

    session = db_session  # type: ignore[assignment]
    # Near-identical embeddings put the pair above the similarity gate,
    # so the CO_EVIDENTIAL link is the only thing keeping it from the LLM call.
    p_a = _make_particle("Acme acquired Widget on May 1.").model_copy(
        update={"provenance": [ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="ce")]}
    )
    p_b = _make_particle("On May 1, Acme bought Widget.").model_copy(
        update={"provenance": [ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="ce")]}
    )
    await insert_particle(session, p_a, embedding=_EMB_HI_A)  # type: ignore[arg-type]
    await insert_particle(session, p_b, embedding=_EMB_HI_B)  # type: ignore[arg-type]
    await create_relation(
        session,  # type: ignore[arg-type]
        p_a.id,
        p_b.id,
        RelationType.CO_EVIDENTIAL,
        RelationCreatedBy.HUMAN_REVIEW,
    )
    await session.commit()  # type: ignore[union-attr]

    # Patch the submodule binding (tests/AGENTS.md § Mocking strategy).
    with patch(
        "particles.operations.lint.contradictions._llm_check_contradiction",
        return_value=None,
    ) as mock_llm:
        findings = await _check_contradictions(session, fix=False)  # type: ignore[arg-type]

    mock_llm.assert_not_called()
    assert all(f.finding_type != "CONTRADICTION" for f in findings)


@pytest.mark.asyncio
async def test_lsem01_excludes_non_falsifiable(db_session: object) -> None:
    """L-SEM-01 never contradiction-checks a non-FALSIFIABLE particle.

    A FALSIFIABLE claim and a near-identical EVALUATIVE opinion embed close
    enough to clear the similarity gate, so without the modality gate they would
    reach the LLM contradiction probe. The gate filters the opinion out, leaving
    fewer than two truth-apt particles — the LLM is never called and no
    CONTRADICTION is raised.
    """
    from unittest.mock import patch

    from particles.operations.lint import _check_contradictions

    session = db_session  # type: ignore[assignment]
    fact = _make_particle("The store opened on May 1.").model_copy(
        update={"provenance": [ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="ce")]}
    )
    opinion = _make_particle("The store should never have opened.").model_copy(
        update={
            "provenance": [ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="ce")],
            "assertion_modality": AssertionModality.EVALUATIVE,
        }
    )
    await insert_particle(session, fact, embedding=_EMB_HI_A)  # type: ignore[arg-type]
    await insert_particle(session, opinion, embedding=_EMB_HI_B)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    with patch(
        "particles.operations.lint.contradictions._llm_check_contradiction",
        return_value="YES: they disagree",
    ) as mock_llm:
        findings = await _check_contradictions(session, fix=False)  # type: ignore[arg-type]

    mock_llm.assert_not_called()
    assert all(f.finding_type != "CONTRADICTION" for f in findings)


@pytest.mark.asyncio
async def test_lsem01_detects_cross_source_contradiction(db_session: object) -> None:
    """L-SEM-01 flags a contradiction across two DIFFERENT corpus entries.

    The same-source guard is gone: two near-identical-embedding claims from
    distinct sources clear the similarity gate, reach the LLM probe, and a
    YES verdict raises a CONTRADICTION finding.
    """
    from unittest.mock import patch

    from particles.operations.lint import _check_contradictions

    session = db_session  # type: ignore[assignment]
    p_a = _make_particle("Morgan dollars are 90% silver.").model_copy(
        update={
            "provenance": [ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="note-a")]
        }
    )
    p_b = _make_particle("Morgan dollars are 92.5% silver.").model_copy(
        update={
            "provenance": [ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="note-b")]
        }
    )
    await insert_particle(session, p_a, embedding=_EMB_HI_A)  # type: ignore[arg-type]
    await insert_particle(session, p_b, embedding=_EMB_HI_B)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    with patch(
        "particles.operations.lint.contradictions._llm_check_contradiction",
        return_value="YES: silver content differs",
    ) as mock_llm:
        findings = await _check_contradictions(session, fix=False)  # type: ignore[arg-type]

    mock_llm.assert_called_once()
    assert any(f.finding_type == "CONTRADICTION" for f in findings)


@pytest.mark.asyncio
async def test_lsem01_skips_below_similarity_threshold(db_session: object) -> None:
    """A pair whose embeddings are not cosine-close is never sent to the LLM.

    The similarity gate is what bounds the store-wide candidate set: orthogonal
    embeddings fall below ``lint.contradiction_candidate_threshold`` and the
    expensive LLM probe is skipped even though a YES verdict was stubbed.
    """
    from unittest.mock import patch

    from particles.operations.lint import _check_contradictions

    session = db_session  # type: ignore[assignment]
    p_a = _make_particle("Morgan dollars are 90% silver.").model_copy(
        update={
            "provenance": [ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="note-a")]
        }
    )
    p_b = _make_particle("The Sheldon scale grades coins from 1 to 70.").model_copy(
        update={
            "provenance": [ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="note-b")]
        }
    )
    await insert_particle(session, p_a, embedding=_EMB_HI_A)  # type: ignore[arg-type]
    await insert_particle(session, p_b, embedding=_EMB_LOW)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    with patch(
        "particles.operations.lint.contradictions._llm_check_contradiction",
        return_value="YES: should not be reached",
    ) as mock_llm:
        findings = await _check_contradictions(session, fix=False)  # type: ignore[arg-type]

    mock_llm.assert_not_called()
    assert all(f.finding_type != "CONTRADICTION" for f in findings)


# ---------------------------------------------------------------------------
# ContradictionProbeControl — cap / scope / progress
# ---------------------------------------------------------------------------

# A second high-similarity cluster orthogonal to _EMB_HI_A/_EMB_HI_B, for
# scope tests that need two candidate pairs with no cross-cluster similarity.
_EMB_HI_C = (np.array([0.0, 0.0, 0.6, 0.8] + [0.0] * 380, dtype=np.float32)).tolist()
_EMB_HI_D = (np.array([0.0, 0.0, 0.61, 0.79] + [0.0] * 380, dtype=np.float32)).tolist()


def _prov(entry_id: str) -> list[ProvenanceRef]:
    return [ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id)]


@pytest.mark.asyncio
async def test_probe_control_cap_probes_highest_similarity_first(db_session: object) -> None:
    """The cap spends the LLM budget on the closest pairs and reports the census.

    Three particles along one direction produce three above-gate pairs at
    distinct similarities; ``max_probes=1`` probes exactly one — the closest
    pair — and the control reports 3 candidates / 1 probe run (the audit's
    "probed X of Y candidate pairs" disclosure).
    """
    from unittest.mock import patch

    from particles.operations.lint import ContradictionProbeControl, _check_contradictions

    session = db_session  # type: ignore[assignment]
    emb_a = (np.array([1.0, 0.0] + [0.0] * 382, dtype=np.float32)).tolist()
    emb_b = (np.array([0.99, 0.141] + [0.0] * 382, dtype=np.float32)).tolist()  # ~0.99 vs A
    emb_c = (np.array([0.8, 0.6] + [0.0] * 382, dtype=np.float32)).tolist()  # 0.8 vs A
    p_a = _make_particle("Claim alpha.").model_copy(update={"provenance": _prov("na")})
    p_b = _make_particle("Claim bravo.").model_copy(update={"provenance": _prov("nb")})
    p_c = _make_particle("Claim charlie.").model_copy(update={"provenance": _prov("nc")})
    await insert_particle(session, p_a, embedding=emb_a)  # type: ignore[arg-type]
    await insert_particle(session, p_b, embedding=emb_b)  # type: ignore[arg-type]
    await insert_particle(session, p_c, embedding=emb_c)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    probed: list[tuple[str, str]] = []

    async def _record(content_a: str, content_b: str) -> None:
        probed.append((content_a, content_b))
        return None

    control = ContradictionProbeControl(max_probes=1)
    with patch(
        "particles.operations.lint.contradictions._llm_check_contradiction",
        side_effect=_record,
    ):
        await _check_contradictions(session, fix=False, control=control)  # type: ignore[arg-type]

    assert control.candidate_pairs == 3
    assert control.probes_run == 1
    assert control.capped is True
    # The one probe went to the closest pair (A, B), not store order.
    assert probed == [("Claim alpha.", "Claim bravo.")]


@pytest.mark.asyncio
async def test_probe_control_scope_keeps_pairs_touching_scope(db_session: object) -> None:
    """``scope_particle_ids`` keeps only pairs with at least one side in scope.

    Two orthogonal high-similarity clusters yield two candidate pairs; scoping
    to one particle of the first cluster drops the other cluster's pair
    entirely — the audit's ``--scope harvested`` mode.
    """
    from unittest.mock import patch

    from particles.operations.lint import ContradictionProbeControl, _check_contradictions

    session = db_session  # type: ignore[assignment]
    p_a = _make_particle("Harvest says X.").model_copy(update={"provenance": _prov("ha")})
    p_b = _make_particle("Store says not X.").model_copy(update={"provenance": _prov("old-b")})
    p_c = _make_particle("Old claim Y.").model_copy(update={"provenance": _prov("old-c")})
    p_d = _make_particle("Old claim not Y.").model_copy(update={"provenance": _prov("old-d")})
    await insert_particle(session, p_a, embedding=_EMB_HI_A)  # type: ignore[arg-type]
    await insert_particle(session, p_b, embedding=_EMB_HI_B)  # type: ignore[arg-type]
    await insert_particle(session, p_c, embedding=_EMB_HI_C)  # type: ignore[arg-type]
    await insert_particle(session, p_d, embedding=_EMB_HI_D)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    control = ContradictionProbeControl(scope_particle_ids=frozenset({p_a.id}))
    with patch(
        "particles.operations.lint.contradictions._llm_check_contradiction",
        return_value="YES: conflict",
    ) as mock_llm:
        findings = await _check_contradictions(session, fix=False, control=control)  # type: ignore[arg-type]

    assert control.candidate_pairs == 1
    assert control.probes_run == 1
    assert control.capped is False
    assert mock_llm.call_count == 1
    assert len(findings) == 1
    flagged = {findings[0].particle_id} | {findings[0].detail.split("particle ")[1].split(":")[0]}
    assert flagged == {p_a.id, p_b.id}


@pytest.mark.asyncio
async def test_probe_control_intra_scope_pairs_probed_before_mixed(db_session: object) -> None:
    """regression: the cap goes to intra-harvest pairs before mixed ones.

    The mixed (harvested ↔ store) pair is deliberately MORE cosine-similar than
    the (harvested ↔ harvested) pair. Under the old pure-similarity order a
    ``max_probes=1`` budget went to the coincidental cross-pair and starved the
    intra-harvest pair — the owner-dogfood 3-vs-0 asymmetry, where store
    population changed which memory-file findings surfaced. The two-tier order
    probes the intra-harvest pair first; a second unit of budget then reaches
    the mixed pair (the mixed tier is deprioritised, not dropped).
    """
    from unittest.mock import patch

    from particles.operations.lint import ContradictionProbeControl, _check_contradictions

    session = db_session  # type: ignore[assignment]
    # Harvested pair (h_a, h_b) at cosine ~0.99; harvested h_c pairs with the
    # store-only s_d at ~0.9999 — the highest-similarity pair is the mixed one.
    emb_h_a = (np.array([1.0, 0.0] + [0.0] * 382, dtype=np.float32)).tolist()
    emb_h_b = (np.array([0.99, 0.141] + [0.0] * 382, dtype=np.float32)).tolist()
    emb_h_c = (np.array([0.0, 0.0, 1.0, 0.0] + [0.0] * 380, dtype=np.float32)).tolist()
    emb_s_d = (np.array([0.0, 0.0, 0.9999, 0.0141] + [0.0] * 380, dtype=np.float32)).tolist()
    h_a = _make_particle("Memory says X.").model_copy(update={"provenance": _prov("mem-a")})
    h_b = _make_particle("Memory says not X.").model_copy(update={"provenance": _prov("mem-b")})
    h_c = _make_particle("Memory note about RDF.").model_copy(update={"provenance": _prov("mem-c")})
    s_d = _make_particle("Store claim about SPARQL.").model_copy(
        update={"provenance": _prov("old-d")}
    )
    for particle, emb in ((h_a, emb_h_a), (h_b, emb_h_b), (h_c, emb_h_c), (s_d, emb_s_d)):
        await insert_particle(session, particle, embedding=emb)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    from particles.operations.lint.contradictions import UNANSWERED, ProbeAnswer

    probed: list[tuple[str, str]] = []

    async def _record(content_a: str, content_b: str) -> ProbeAnswer:
        probed.append((content_a, content_b))
        # Unanswered, so the ledger records nothing and the second run below
        # sees the same candidate set (skipping is tested elsewhere).
        return UNANSWERED

    scope = frozenset({h_a.id, h_b.id, h_c.id})
    control = ContradictionProbeControl(max_probes=1, scope_particle_ids=scope)
    with patch(
        "particles.operations.lint.contradictions._llm_check_contradiction",
        side_effect=_record,
    ):
        await _check_contradictions(session, fix=False, control=control)  # type: ignore[arg-type]

    # One unit of budget → the intra-harvest pair, despite its lower similarity.
    assert probed == [("Memory says X.", "Memory says not X.")]
    assert control.candidate_pairs == 2
    assert control.intra_scope_pairs == 1
    assert control.probes_run == 1
    assert control.capped is True

    # With budget for both, the mixed pair is consumed second, not dropped.
    probed.clear()
    control = ContradictionProbeControl(max_probes=2, scope_particle_ids=scope)
    with patch(
        "particles.operations.lint.contradictions._llm_check_contradiction",
        side_effect=_record,
    ):
        await _check_contradictions(session, fix=False, control=control)  # type: ignore[arg-type]
    assert probed == [
        ("Memory says X.", "Memory says not X."),
        ("Memory note about RDF.", "Store claim about SPARQL."),
    ]
    assert control.capped is False


@pytest.mark.asyncio
async def test_probe_control_store_wide_order_is_pure_similarity(db_session: object) -> None:
    """No scope (store-wide / ``particles lint``) ⇒ no tiers: similarity alone orders.

    Companion to the tiering test above: with ``scope_particle_ids=None`` the
    tier reads 0 for every pair and the highest-similarity
    order is unchanged, ``intra_scope_pairs`` stays 0.
    """
    from unittest.mock import patch

    from particles.operations.lint import ContradictionProbeControl, _check_contradictions

    session = db_session  # type: ignore[assignment]
    emb_a = (np.array([1.0, 0.0] + [0.0] * 382, dtype=np.float32)).tolist()
    emb_b = (np.array([0.99, 0.141] + [0.0] * 382, dtype=np.float32)).tolist()  # ~0.99 vs A
    emb_c = (np.array([0.0, 0.0, 1.0, 0.0] + [0.0] * 380, dtype=np.float32)).tolist()
    emb_d = (np.array([0.0, 0.0, 0.9999, 0.0141] + [0.0] * 380, dtype=np.float32)).tolist()
    p_a = _make_particle("Claim alpha.").model_copy(update={"provenance": _prov("na")})
    p_b = _make_particle("Claim bravo.").model_copy(update={"provenance": _prov("nb")})
    p_c = _make_particle("Claim charlie.").model_copy(update={"provenance": _prov("nc")})
    p_d = _make_particle("Claim delta.").model_copy(update={"provenance": _prov("nd")})
    for particle, emb in ((p_a, emb_a), (p_b, emb_b), (p_c, emb_c), (p_d, emb_d)):
        await insert_particle(session, particle, embedding=emb)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    probed: list[tuple[str, str]] = []

    async def _record(content_a: str, content_b: str) -> None:
        probed.append((content_a, content_b))
        return None

    control = ContradictionProbeControl(max_probes=1)
    with patch(
        "particles.operations.lint.contradictions._llm_check_contradiction",
        side_effect=_record,
    ):
        await _check_contradictions(session, fix=False, control=control)  # type: ignore[arg-type]

    # The (C, D) pair is the most similar store-wide and wins the budget.
    assert probed == [("Claim charlie.", "Claim delta.")]
    assert control.intra_scope_pairs == 0


@pytest.mark.asyncio
async def test_probe_control_progress_events(db_session: object) -> None:
    """``on_progress`` streams ``(done, planned)`` after each LLM probe."""
    from unittest.mock import patch

    from particles.operations.lint import ContradictionProbeControl, _check_contradictions

    session = db_session  # type: ignore[assignment]
    p_a = _make_particle("Pair one A.").model_copy(update={"provenance": _prov("na")})
    p_b = _make_particle("Pair one B.").model_copy(update={"provenance": _prov("nb")})
    p_c = _make_particle("Pair two C.").model_copy(update={"provenance": _prov("nc")})
    p_d = _make_particle("Pair two D.").model_copy(update={"provenance": _prov("nd")})
    await insert_particle(session, p_a, embedding=_EMB_HI_A)  # type: ignore[arg-type]
    await insert_particle(session, p_b, embedding=_EMB_HI_B)  # type: ignore[arg-type]
    await insert_particle(session, p_c, embedding=_EMB_HI_C)  # type: ignore[arg-type]
    await insert_particle(session, p_d, embedding=_EMB_HI_D)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    events: list[tuple[int, int]] = []
    control = ContradictionProbeControl(
        on_progress=lambda done, total: events.append((done, total))
    )
    with patch(
        "particles.operations.lint.contradictions._llm_check_contradiction",
        return_value=None,
    ):
        await _check_contradictions(session, fix=False, control=control)  # type: ignore[arg-type]

    assert events == [(1, 2), (2, 2)]
    assert control.candidate_pairs == 2
    assert control.probes_run == 2


@pytest.mark.asyncio
async def test_run_lint_granularity_probe_opt_out(db_session: object) -> None:
    """``granularity_probe=False`` skips the per-particle LLM granularity loop.

    ``collect_cards`` opts out because ``GRANULARITY_VIOLATION`` has no card
    kind — for the curation queue and the audit those LLM calls would be pure
    discard. The default keeps the check for
    ``particles lint``.
    """
    from unittest.mock import AsyncMock, patch

    from particles.operations.lint import run_lint

    session = db_session  # type: ignore[assignment]
    with patch(
        "particles.operations.lint.orchestrator._check_granularity_violations",
        AsyncMock(return_value=[]),
    ) as granularity:
        await run_lint(session, fix=False, semantic=True, granularity_probe=False)  # type: ignore[arg-type]
        granularity.assert_not_called()
        await run_lint(session, fix=False, semantic=True)  # type: ignore[arg-type]
        granularity.assert_called_once()


@pytest.mark.asyncio
async def test_empty_complete_snapshot_flagged(db_session: object) -> None:
    """F4.1 recovery: lint flags COMPLETE snapshots that produced zero particles.

    These are the snapshots silently lost before the pipeline fix landed — the
    audit surface that lets the operator re-extract them.
    """
    from particles.core.schema import ExtractionStatus, WarcRecordType
    from particles.corpus.store import SnapshotRow
    from particles.operations.lint import run_lint

    session = db_session  # type: ignore[assignment]
    session.add(  # type: ignore[union-attr]
        SnapshotRow(
            snapshot_id="snap-empty",
            entry_id="entry-empty",
            captured_at=datetime.now(UTC),
            content_hash="a" * 64,
            warc_record_type=WarcRecordType.RESPONSE.value,
            extraction_status=ExtractionStatus.COMPLETE.value,
        )
    )
    await session.commit()  # type: ignore[union-attr]

    report = await run_lint(session, fix=False, semantic=False)  # type: ignore[arg-type]
    flagged = [f for f in report.findings if f.finding_type == "EMPTY_COMPLETE_SNAPSHOT"]
    assert len(flagged) == 1
    assert flagged[0].corpus_entry_id == "entry-empty"


@pytest.mark.asyncio
async def test_empty_complete_snapshot_skips_collapsed_and_replaced_generations(
    db_session: object,
) -> None:
    """ "re-extract to confirm" is wrong advice for a superseded generation.

    A collapsed snapshot is empty by design, and an empty snapshot whose
    replacement generation is in the store could only, if re-extracted, retire
    that replacement. An empty snapshot whose only newer sibling FAILED is still
    reported: it may be the best generation the entry has.
    """
    from particles.core.schema import (
        CorpusEntry,
        ExtractionStatus,
        FetchPolicy,
        Mutability,
        WarcRecordType,
    )
    from particles.corpus.store import CorpusEntryRow, SnapshotRow
    from particles.operations.lint import run_lint

    session = db_session  # type: ignore[assignment]
    now = datetime.now(UTC)

    def _entry(entry_id: str) -> CorpusEntryRow:
        return CorpusEntryRow.from_model(
            CorpusEntry(
                entry_id=entry_id,
                source_type="LOCAL_MARKDOWN",
                uri_r=f"file:///tmp/{entry_id}.md",
                mutability=Mutability.MUTABLE,
                fetch_policy=FetchPolicy.NEVER,
                deposited_by="test",
            )
        )

    def _snap(
        snapshot_id: str,
        entry_id: str,
        *,
        days_old: int,
        status: ExtractionStatus = ExtractionStatus.COMPLETE,
        superseded_by: str | None = None,
    ) -> SnapshotRow:
        return SnapshotRow(
            snapshot_id=snapshot_id,
            entry_id=entry_id,
            captured_at=now - timedelta(days=days_old),
            content_hash=(snapshot_id[0] * 64)[:64],
            warc_record_type=WarcRecordType.RESPONSE.value,
            extraction_status=status.value,
            superseded_by_snapshot_id=superseded_by,
        )

    session.add_all(  # type: ignore[union-attr]
        [
            _entry("entry-collapsed"),
            _snap("collapsed", "entry-collapsed", days_old=2, superseded_by="newest-a"),
            _snap("newest-a", "entry-collapsed", days_old=1, status=ExtractionStatus.PENDING),
            _entry("entry-replaced"),
            _snap("replaced", "entry-replaced", days_old=2),
            _snap("zcurrent", "entry-replaced", days_old=1),
            _entry("entry-survivor"),
            _snap("survivor", "entry-survivor", days_old=2),
            _snap("broken", "entry-survivor", days_old=1, status=ExtractionStatus.FAILED),
        ]
    )
    await session.commit()  # type: ignore[union-attr]

    report = await run_lint(session, fix=False, semantic=False)  # type: ignore[arg-type]
    flagged = {
        f.detail.split()[1] for f in report.findings if f.finding_type == "EMPTY_COMPLETE_SNAPSHOT"
    }
    # `zcurrent` is itself empty and current, so it is reported like any other.
    assert flagged == {"zcurrent", "survivor"}


@pytest.mark.asyncio
async def test_empty_complete_snapshot_excludes_revisit_and_populated(
    db_session: object,
) -> None:
    """REVISIT snapshots (empty by design) and populated snapshots are not flagged."""
    from particles.core.schema import ExtractionStatus, WarcRecordType
    from particles.corpus.store import SnapshotRow
    from particles.operations.lint import run_lint

    session = db_session  # type: ignore[assignment]
    # REVISIT snapshot: COMPLETE with no particles by design — must not flag.
    session.add(  # type: ignore[union-attr]
        SnapshotRow(
            snapshot_id="snap-revisit",
            entry_id="entry-revisit",
            captured_at=datetime.now(UTC),
            content_hash="b" * 64,
            warc_record_type=WarcRecordType.REVISIT.value,
            extraction_status=ExtractionStatus.COMPLETE.value,
            refers_to="snap-prior",
        )
    )
    # RESPONSE snapshot that did produce a particle — must not flag.
    session.add(  # type: ignore[union-attr]
        SnapshotRow(
            snapshot_id="snap-populated",
            entry_id="entry-pop",
            captured_at=datetime.now(UTC),
            content_hash="c" * 64,
            warc_record_type=WarcRecordType.RESPONSE.value,
            extraction_status=ExtractionStatus.COMPLETE.value,
        )
    )
    populated = _make_particle("A real claim.").model_copy(
        update={
            "provenance": [
                ProvenanceRef(
                    type=ProvenanceRefType.SOURCE,
                    corpus_entry_id="entry-pop",
                    snapshot_id="snap-populated",
                )
            ]
        }
    )
    await insert_particle(session, populated)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    report = await run_lint(session, fix=False, semantic=False)  # type: ignore[arg-type]
    flagged = [f for f in report.findings if f.finding_type == "EMPTY_COMPLETE_SNAPSHOT"]
    assert flagged == []


@pytest.mark.asyncio
async def test_no_subject_claim_flagged(db_session: object) -> None:
    """L-STR-09: an ACTIVE CLAIM particle with empty subject_ids is flagged."""
    from particles.operations.lint import run_lint

    session = db_session  # type: ignore[assignment]
    bare = _make_particle("A claim about nothing resolvable.")
    assert bare.subject_ids == []
    await insert_particle(session, bare)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    report = await run_lint(session, fix=False, semantic=False)  # type: ignore[arg-type]
    flagged = [f for f in report.findings if f.finding_type == "NO_SUBJECT"]
    assert [f.particle_id for f in flagged] == [bare.id]
    assert flagged[0].severity == "WARNING"


@pytest.mark.asyncio
async def test_no_subject_excludes_legitimate_zero_subject_records(db_session: object) -> None:
    """REVIEW audit particles, DOCUMENT_META claims, and subject-linked claims
    are not flagged by L-STR-09."""
    from particles.core.schema import ParticleType, Subject
    from particles.extraction.scope import SCOPE_DOCUMENT_META, SCOPE_KEY
    from particles.operations.lint import run_lint
    from particles.store.subject_store import insert_subject

    session = db_session  # type: ignore[assignment]

    subj = Subject(canonical_name="Water", asserted_by="test-agent")
    await insert_subject(session, subj)  # type: ignore[arg-type]
    linked = _make_particle("Water is H2O.").model_copy(update={"subject_ids": [subj.id]})
    await insert_particle(session, linked)  # type: ignore[arg-type]

    review = _make_particle("REVIEW: PREFER_A on INCONSISTENCY x.").model_copy(
        update={"particle_type": ParticleType.REVIEW}
    )
    await insert_particle(session, review)  # type: ignore[arg-type]

    doc_meta = _make_particle("This page is a draft.").model_copy(
        update={"properties": {SCOPE_KEY: SCOPE_DOCUMENT_META}}
    )
    await insert_particle(session, doc_meta)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    report = await run_lint(session, fix=False, semantic=False)  # type: ignore[arg-type]
    assert [f for f in report.findings if f.finding_type == "NO_SUBJECT"] == []


@pytest.mark.asyncio
async def test_confidence_decay_threshold_is_configurable(db_session: object) -> None:
    """CONFIDENCE_DECAY fires above config.lint.variance_threshold (P4-7).

    The threshold was hardcoded at 0.15; it now lives in
    ``config.lint.variance_threshold``.
    """
    from particles.config import get_config
    from particles.operations.lint import run_lint

    session = db_session  # type: ignore[assignment]
    p = _make_particle("Variance-heavy claim.").model_copy(
        update={
            "confidence": Confidence(
                value=0.8,
                variance=0.20,
                calibration_source=CalibrationSource.EXTRACTOR_DIRECT,
            )
        }
    )
    await insert_particle(session, p)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    # variance 0.20 > default threshold 0.15 → finding
    report = await run_lint(session, fix=False, semantic=False)  # type: ignore[arg-type]
    assert "CONFIDENCE_DECAY" in [f.finding_type for f in report.findings]

    # Raising the configured threshold above the variance silences it
    get_config().lint.variance_threshold = 0.5
    report = await run_lint(session, fix=False, semantic=False)  # type: ignore[arg-type]
    assert "CONFIDENCE_DECAY" not in [f.finding_type for f in report.findings]


# ---------------------------------------------------------------------------
# RECENCY_DECAY — age-discounted effective_confidence surfaced in lint
# (decay)
# ---------------------------------------------------------------------------


async def _seed_aged_particle(
    session: object,
    *,
    source_type: str,
    published_days_ago: int | None,
    content: str = "An aged claim.",
) -> str:
    """Seed a corpus entry + snapshot + ACTIVE particle and return the particle id.

    ``source_type`` keys the decay curve (carried on the corpus entry);
    ``published_days_ago`` sets the snapshot's ``content_published_at`` (None =
    unknown publication date, so no decay applies).
    """
    from particles.corpus.store import CorpusEntryRow, SnapshotRow

    entry_id = f"e-{source_type}-{published_days_ago}"
    snap_id = f"snap-{source_type}-{published_days_ago}"
    published = (
        None
        if published_days_ago is None
        else datetime.now(UTC) - timedelta(days=published_days_ago)
    )
    session.add(  # type: ignore[union-attr]
        CorpusEntryRow(
            entry_id=entry_id,
            uri_r=f"https://example.com/{entry_id}",
            source_type=source_type,
            mutability="MUTABLE",
            fetch_policy="LAZY",
            created_at=datetime.now(UTC),
            deposited_by="test",
        )
    )
    session.add(  # type: ignore[union-attr]
        SnapshotRow(
            snapshot_id=snap_id,
            entry_id=entry_id,
            captured_at=datetime.now(UTC),
            content_hash="d" * 64,
            warc_record_type="RESPONSE",
            extraction_status="COMPLETE",
            content_published_at=published,
        )
    )
    p = _make_particle(content).model_copy(
        update={
            "provenance": [
                ProvenanceRef(
                    type=ProvenanceRefType.SOURCE,
                    corpus_entry_id=entry_id,
                    snapshot_id=snap_id,
                )
            ]
        }
    )
    await insert_particle(session, p)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]
    return p.id


@pytest.mark.asyncio
async def test_recency_decay_flags_aged_decaying_source(db_session: object) -> None:
    """A REDDIT_POST published 120 days ago (60-day half-life → rf≈0.25, discount
    ≈0.75) trips the default 0.5 threshold as a read-only WARNING."""
    from particles.operations.lint import run_lint

    session = db_session  # type: ignore[assignment]
    pid = await _seed_aged_particle(session, source_type="REDDIT_POST", published_days_ago=120)

    report = await run_lint(session, fix=True, semantic=False)  # type: ignore[arg-type]
    decay = [f for f in report.findings if f.finding_type == "RECENCY_DECAY"]
    assert len(decay) == 1
    assert decay[0].particle_id == pid
    assert decay[0].severity == "WARNING"

    # Read-only: even with fix=True the particle stays ACTIVE (decay is a
    # recoverable discount, not a provenance break).
    from particles.store.particle_store import get_particle

    p = await get_particle(session, pid)  # type: ignore[arg-type]
    assert p is not None and p.status == Status.ACTIVE


@pytest.mark.asyncio
async def test_recency_decay_skips_recent_content(db_session: object) -> None:
    from particles.operations.lint import run_lint

    session = db_session  # type: ignore[assignment]
    await _seed_aged_particle(session, source_type="REDDIT_POST", published_days_ago=1)
    report = await run_lint(session, fix=False, semantic=False)  # type: ignore[arg-type]
    assert "RECENCY_DECAY" not in [f.finding_type for f in report.findings]


@pytest.mark.asyncio
async def test_recency_decay_skips_non_decaying_source(db_session: object) -> None:
    # A source type with no decay config → recency_factor == 1.0 → never fires.
    from particles.operations.lint import run_lint

    session = db_session  # type: ignore[assignment]
    await _seed_aged_particle(session, source_type="PDF", published_days_ago=3650)
    report = await run_lint(session, fix=False, semantic=False)  # type: ignore[arg-type]
    assert "RECENCY_DECAY" not in [f.finding_type for f in report.findings]


@pytest.mark.asyncio
async def test_recency_decay_skips_unknown_publication_date(db_session: object) -> None:
    from particles.operations.lint import run_lint

    session = db_session  # type: ignore[assignment]
    await _seed_aged_particle(session, source_type="REDDIT_POST", published_days_ago=None)
    report = await run_lint(session, fix=False, semantic=False)  # type: ignore[arg-type]
    assert "RECENCY_DECAY" not in [f.finding_type for f in report.findings]


@pytest.mark.asyncio
async def test_recency_decay_threshold_is_configurable(db_session: object) -> None:
    from particles.config import get_config
    from particles.operations.lint import run_lint

    session = db_session  # type: ignore[assignment]
    await _seed_aged_particle(session, source_type="REDDIT_POST", published_days_ago=120)

    report = await run_lint(session, fix=False, semantic=False)  # type: ignore[arg-type]
    assert "RECENCY_DECAY" in [f.finding_type for f in report.findings]

    # Raising the threshold above the ≈0.75 discount silences it.
    get_config().lint.recency_decay_threshold = 0.9
    report = await run_lint(session, fix=False, semantic=False)  # type: ignore[arg-type]
    assert "RECENCY_DECAY" not in [f.finding_type for f in report.findings]


# ---------------------------------------------------------------------------
# UNDATED_RETIREMENT — the stamp-coverage alarm.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_undated_retirement_flags_unstamped_rows_only(db_session: object) -> None:
    """One aggregate finding counts once-believed retired rows with NULL
    retired_at; a properly-stamped retirement is not counted."""
    from particles.operations.lint.retirement import _check_undated_retirements
    from particles.store.particle_store import update_particle_status

    session = db_session  # type: ignore[assignment]

    # A legacy (pre-migration) retirement: born SUPERSEDED via direct insert,
    # so no stamp was ever written.
    legacy = _make_particle("legacy retired claim", status=Status.SUPERSEDED)
    await insert_particle(session, legacy)  # type: ignore[arg-type]

    # A post-migration retirement: the choke point stamps it.
    stamped = _make_particle("freshly retired claim")
    await insert_particle(session, stamped)  # type: ignore[arg-type]
    await update_particle_status(
        session,  # type: ignore[arg-type]
        stamped.id,
        Status.RETRACTED,
        StatusReason.EXPLICIT_RETRACTION,
    )

    findings = await _check_undated_retirements(session)  # type: ignore[arg-type]
    assert len(findings) == 1
    finding = findings[0]
    assert finding.finding_type == "UNDATED_RETIREMENT"
    assert finding.severity == "WARNING"
    assert "1 once-believed retired particle(s)" in finding.detail
    # The most recent asserted_at in the unstamped set is disclosed.
    assert legacy.asserted_at.isoformat() in finding.detail


@pytest.mark.asyncio
async def test_undated_retirement_excludes_born_retired(db_session: object) -> None:
    """Quarantine losers (CONFLICT_PENDING) and INCONSISTENCY records were
    never believed — their NULL retired_at is correct, not a gap."""
    from particles.operations.lint.retirement import _check_undated_retirements

    session = db_session  # type: ignore[assignment]

    loser = _make_particle("quarantined loser", status=Status.PROVENANCE_STALE)
    loser = loser.model_copy(update={"status_reason": StatusReason.CONFLICT_PENDING})
    await insert_particle(session, loser)  # type: ignore[arg-type]

    inconsistency = _make_particle("conflict record", status=Status.INCONSISTENCY)
    await insert_particle(session, inconsistency)  # type: ignore[arg-type]

    assert await _check_undated_retirements(session) == []  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_undated_retirement_clean_and_orchestrated(db_session: object) -> None:
    """No finding on a store with only ACTIVE / stamped rows; the aggregate
    rides run_lint's structural pass when unstamped rows exist."""
    from particles.operations.lint import run_lint

    session = db_session  # type: ignore[assignment]

    active = _make_particle("still believed")
    await insert_particle(session, active)  # type: ignore[arg-type]
    report = await run_lint(session, fix=False, semantic=False)  # type: ignore[arg-type]
    assert "UNDATED_RETIREMENT" not in report.summary

    legacy = _make_particle("legacy retired claim", status=Status.SUPERSEDED)
    await insert_particle(session, legacy)  # type: ignore[arg-type]
    report = await run_lint(session, fix=False, semantic=False)  # type: ignore[arg-type]
    assert report.summary.get("UNDATED_RETIREMENT") == 1


class TestProbeReplyDialects:
    """the probe accepts both the enforced JSON verdict object
    (LocalProvider structured output) and the reason-then-verdict text protocol."""

    def test_reason_then_verdict_text_protocol(self) -> None:
        from particles.operations.lint.contradictions import _parse_probe_reply

        assert (
            _parse_probe_reply("REASON: One says 2019, the other 2021.\nVERDICT: YES")
            == "One says 2019, the other 2021."
        )
        assert _parse_probe_reply("REASON: Same fact restated.\nVERDICT: NO") is None
        assert _parse_probe_reply("VERDICT: YES") == "contradiction detected"
        assert _parse_probe_reply("**Verdict:** no") is None

    @pytest.mark.parametrize(
        "reply",
        [
            # Cut at the token budget before the verdict line (the 2026-09-25
            # audit rendered "...contrad" from replies like this one).
            "REASON: Claim A says the port is 8080 while claim B implies it was contrad",
            # The pre-fix protocol, cut mid-paragraph: its leading YES is not a verdict.
            "YES: These claims contradict each other because the first states the "
            "migration ran in March while the second describes it as: pre-",
            # Ambiguous: both verdicts, or a verdict that is not the last line.
            "REASON: unclear\nVERDICT: YES\nVERDICT: NO",
            "VERDICT: YES\nREASON: Actually, on reflection these are consistent",
            # Free-form, no verdict at all.
            "Yes and no. It depends on the reading.",
            # The JSON dialect, cut mid-object.
            '{"contradicts": true, "description": "the dates dis',
        ],
    )
    def test_truncated_or_ambiguous_reply_is_never_a_contradiction(
        self, reply: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.operations import _llm
        from particles.operations.lint.contradictions import UNANSWERED, _parse_probe_reply

        before = _llm.llm_failure_count()
        # Unanswered, not a NO: it is never a contradiction and never remembered
        # as a cleared pair.
        assert _parse_probe_reply(reply) is UNANSWERED
        # Counted as a failed probe, so the audit discloses "counts may read low"
        # instead of the pair reading as a clean "no".
        assert _llm.llm_failure_count() == before + 1

    def test_description_renders_on_one_line_with_a_clean_ellipsis(self) -> None:
        from particles.operations.lint.contradictions import _parse_probe_reply, one_line

        reason = "The first claim says the index lives in\nMEMORY.md " + "and more " * 40
        described = _parse_probe_reply(f"REASON: {reason}\nVERDICT: YES")
        assert described is not None
        assert "\n" not in described
        assert len(described) <= 160
        assert described.endswith("…")
        assert not described[:-1].endswith(" ")
        assert one_line("short\n  text") == "short text"

    def test_json_verdict_object(self) -> None:
        from particles.operations.lint.contradictions import UNANSWERED, _parse_probe_reply

        assert (
            _parse_probe_reply('{"contradicts": true, "description": "dates disagree"}')
            == "dates disagree"
        )
        assert _parse_probe_reply('{"contradicts": true}') == "contradiction detected"
        assert _parse_probe_reply('{"contradicts": false}') is None
        # Malformed JSON is no verdict, not an error.
        assert _parse_probe_reply('{"contradicts": ') is UNANSWERED

    @pytest.mark.asyncio
    async def test_probe_threads_response_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.operations.lint import contradictions as c_mod

        captured: dict[str, object] = {}

        async def fake_llm_call(*args: object, **kwargs: object) -> str:
            captured.update(kwargs)
            return "NO"

        monkeypatch.setattr(c_mod, "_llm_call", fake_llm_call)
        result = await c_mod._llm_check_contradiction("A", "B")
        # A bare "NO" has no verdict line, so the probe is unanswered.
        assert result is c_mod.UNANSWERED
        schema = captured["response_schema"]
        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert schema["required"] == ["contradicts"]


@pytest.mark.asyncio
async def test_probe_control_latency_tolerant_batches_the_planned_prefix(
    db_session: object,
) -> None:
    """a latency-tolerant caller probes the planned pairs as ONE batch.

    Same candidate enumeration, same cap, same findings — the sequential
    per-pair loop is replaced by a single ``complete_many`` submission, and the
    verdicts must stay aligned with the pairs that produced them.
    """
    from unittest.mock import patch

    from particles.operations.lint import ContradictionProbeControl, _check_contradictions

    session = db_session  # type: ignore[assignment]
    emb_a = (np.array([1.0, 0.0] + [0.0] * 382, dtype=np.float32)).tolist()
    emb_b = (np.array([0.99, 0.141] + [0.0] * 382, dtype=np.float32)).tolist()
    emb_c = (np.array([0.8, 0.6] + [0.0] * 382, dtype=np.float32)).tolist()
    p_a = _make_particle("Claim alpha.").model_copy(update={"provenance": _prov("na")})
    p_b = _make_particle("Claim bravo.").model_copy(update={"provenance": _prov("nb")})
    p_c = _make_particle("Claim charlie.").model_copy(update={"provenance": _prov("nc")})
    await insert_particle(session, p_a, embedding=emb_a)  # type: ignore[arg-type]
    await insert_particle(session, p_b, embedding=emb_b)  # type: ignore[arg-type]
    await insert_particle(session, p_c, embedding=emb_c)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    submitted: list[object] = []

    async def _fake_call_many(requests: object, **kwargs: object) -> list[str | None]:
        assert isinstance(requests, list)
        submitted.extend(requests)
        assert kwargs["latency_tolerant"] is True
        # Only the closest pair contradicts; the rest come back clean.
        return ["REASON: alpha and bravo disagree\nVERDICT: YES"] + [None] * (len(requests) - 1)

    async def _no_sequential(*a: object, **k: object) -> str | None:
        raise AssertionError("sequential probe must not run when batching")

    progress: list[tuple[int, int]] = []
    control = ContradictionProbeControl(
        latency_tolerant=True, on_progress=lambda done, total: progress.append((done, total))
    )
    with (
        patch(
            "particles.operations.lint.contradictions._llm_call_many",
            side_effect=_fake_call_many,
        ),
        patch(
            "particles.operations.lint.contradictions._llm_check_contradiction",
            side_effect=_no_sequential,
        ),
    ):
        findings = await _check_contradictions(session, fix=False, control=control)  # type: ignore[arg-type]

    assert control.candidate_pairs == 3
    assert control.probes_run == 3
    assert len(submitted) == 3
    # Each probe carries its own F3 fence nonce in its own system turn.
    systems = {getattr(r, "system", None) for r in submitted}
    assert len(systems) == 3
    # One report for the whole batch — there is no per-pair completion moment.
    assert progress == [(3, 3)]
    # The single YES landed on the highest-similarity pair (alpha ↔ bravo).
    assert len(findings) == 1
    assert findings[0].particle_content == "Claim alpha."
    assert "alpha and bravo disagree" in findings[0].detail


@pytest.mark.asyncio
async def test_probe_control_batch_respects_the_cap(db_session: object) -> None:
    """The cap bounds what is submitted, not just what is read back."""
    from unittest.mock import patch

    from particles.operations.lint import ContradictionProbeControl, _check_contradictions

    session = db_session  # type: ignore[assignment]
    emb_a = (np.array([1.0, 0.0] + [0.0] * 382, dtype=np.float32)).tolist()
    emb_b = (np.array([0.99, 0.141] + [0.0] * 382, dtype=np.float32)).tolist()
    emb_c = (np.array([0.8, 0.6] + [0.0] * 382, dtype=np.float32)).tolist()
    for content, emb, entry in (
        ("Claim alpha.", emb_a, "na"),
        ("Claim bravo.", emb_b, "nb"),
        ("Claim charlie.", emb_c, "nc"),
    ):
        particle = _make_particle(content).model_copy(update={"provenance": _prov(entry)})
        await insert_particle(session, particle, embedding=emb)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    sizes: list[int] = []

    async def _fake_call_many(requests: object, **kwargs: object) -> list[str | None]:
        assert isinstance(requests, list)
        sizes.append(len(requests))
        return [None] * len(requests)

    control = ContradictionProbeControl(max_probes=1, latency_tolerant=True)
    with patch(
        "particles.operations.lint.contradictions._llm_call_many",
        side_effect=_fake_call_many,
    ):
        await _check_contradictions(session, fix=False, control=control)  # type: ignore[arg-type]

    assert sizes == [1]
    assert control.candidate_pairs == 3
    assert control.probes_run == 1
    assert control.capped is True


@pytest.mark.asyncio
async def test_a_recorded_contradiction_is_reported_without_a_probe(db_session: object) -> None:
    """the write path's confirmed, declined pair is a structural finding."""
    from unittest.mock import AsyncMock, patch

    from particles.operations.lint import run_lint
    from particles.store.particle_store import update_particle_status
    from particles.store.relation_store import record_observer_divergence

    session = db_session  # type: ignore[assignment]
    a = _make_particle("The default branch is main.")
    b = _make_particle("The default branch is master.")
    gone = _make_particle("The default branch is trunk.")
    for p in (a, b):
        await insert_particle(session, p, _EMB_HI_A)  # type: ignore[arg-type]
    await insert_particle(session, gone)  # type: ignore[arg-type]
    await update_particle_status(  # type: ignore[arg-type]
        session, gone.id, Status.PROVENANCE_STALE, StatusReason.SUPERSEDED_BY_UPDATE
    )
    await record_observer_divergence(session, a.id, b.id)  # type: ignore[arg-type]
    await record_observer_divergence(session, a.id, gone.id)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    probe = AsyncMock(return_value="YES: a probe that must not run")
    with patch("particles.operations.lint.contradictions._llm_check_contradiction", probe):
        structural = await run_lint(session, fix=False, semantic=False)  # type: ignore[arg-type]
        semantic = await run_lint(session, fix=False, semantic=True)  # type: ignore[arg-type]

    for report in (structural, semantic):
        found = [f for f in report.findings if f.finding_type == "CONTRADICTION"]
        # One finding for the live pair; a pair with a retired end is not live.
        assert len(found) == 1
        assert "no probe run" in found[0].detail
    probe.assert_not_awaited()  # the one near-duplicate pair is already recorded


# ---------------------------------------------------------------------------
# Cross-source order and the second reading
# ---------------------------------------------------------------------------


async def _three_claims(session: object) -> tuple[Particle, Particle, Particle]:
    """A and B (one note, nearly identical), C (another note, less similar to A)."""
    emb_a = (np.array([1.0, 0.0] + [0.0] * 382, dtype=np.float32)).tolist()
    emb_b = (np.array([0.99, 0.141] + [0.0] * 382, dtype=np.float32)).tolist()
    emb_c = (np.array([0.8, 0.6] + [0.0] * 382, dtype=np.float32)).tolist()
    p_a = _make_particle("Claim alpha.").model_copy(update={"provenance": _prov("note-1")})
    p_b = _make_particle("Claim bravo.").model_copy(update={"provenance": _prov("note-1")})
    p_c = _make_particle("Claim charlie.").model_copy(update={"provenance": _prov("note-2")})
    for p, emb in ((p_a, emb_a), (p_b, emb_b), (p_c, emb_c)):
        await insert_particle(session, p, embedding=emb)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[attr-defined]
    return p_a, p_b, p_c


@pytest.mark.asyncio
async def test_probe_spends_a_cap_on_cross_source_pairs_first(db_session: object) -> None:
    """Under a cap, a pair from two notes is probed before a closer pair from one."""
    from unittest.mock import patch

    from particles.operations.lint import ContradictionProbeControl, _check_contradictions

    await _three_claims(db_session)
    probed: list[tuple[str, str]] = []

    async def _record(content_a: str, content_b: str) -> str:
        probed.append((content_a, content_b))
        return "they disagree"

    control = ContradictionProbeControl(max_probes=1)
    with patch(
        "particles.operations.lint.contradictions._llm_check_contradiction",
        side_effect=_record,
    ):
        findings = await _check_contradictions(db_session, fix=False, control=control)  # type: ignore[arg-type]

    # (bravo, charlie) is the closest cross-note pair (cosine ~0.88); the
    # closer (alpha, bravo) pair (~0.99) comes from one note and waits.
    assert probed == [("Claim bravo.", "Claim charlie.")]
    assert control.flagged == 1
    assert control.same_source_findings == 0
    assert len(findings) == 1


@pytest.mark.asyncio
async def test_same_source_findings_are_counted(db_session: object) -> None:
    from unittest.mock import AsyncMock, patch

    from particles.operations.lint import ContradictionProbeControl, _check_contradictions

    await _three_claims(db_session)
    control = ContradictionProbeControl()
    with patch(
        "particles.operations.lint.contradictions._llm_check_contradiction",
        AsyncMock(return_value="they disagree"),
    ):
        findings = await _check_contradictions(db_session, fix=False, control=control)  # type: ignore[arg-type]

    assert len(findings) == 3
    assert control.same_source_findings == 1  # (alpha, bravo)


@pytest.mark.asyncio
async def test_second_reading_keeps_only_confirmed_flags(db_session: object) -> None:
    from unittest.mock import AsyncMock, patch

    from particles.operations.lint import ContradictionProbeControl, _check_contradictions
    from particles.operations.lint.contradictions import ProbeVerdict

    await _three_claims(db_session)
    readings = [
        ProbeVerdict(usable=True, contradicts=True, description="values disagree"),
        ProbeVerdict(usable=True, contradicts=False),
        None,  # the call failed: unverified, never counted
    ]
    verify = AsyncMock(side_effect=readings)
    progress: list[tuple[int, int]] = []
    control = ContradictionProbeControl(
        verify=True, on_verify_progress=lambda done, total: progress.append((done, total))
    )
    with (
        patch(
            "particles.operations.lint.contradictions._llm_check_contradiction",
            AsyncMock(return_value="first-pass reason"),
        ),
        patch("particles.operations.lint.contradictions._llm_verify_contradiction", verify),
    ):
        findings = await _check_contradictions(db_session, fix=False, control=control)  # type: ignore[arg-type]

    assert verify.await_count == 3
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert (control.flagged, control.confirmed, control.unverified) == (3, 1, 1)
    assert len(findings) == 1
    # The reported reason is the second reading's, not the first pass's.
    assert findings[0].detail.endswith(": values disagree")


@pytest.mark.asyncio
async def test_second_reading_cap_leaves_the_rest_unverified(db_session: object) -> None:
    from unittest.mock import AsyncMock, patch

    from particles.operations.lint import ContradictionProbeControl, _check_contradictions
    from particles.operations.lint.contradictions import ProbeVerdict

    await _three_claims(db_session)
    verify = AsyncMock(return_value=ProbeVerdict(usable=True, contradicts=True, description="x"))
    control = ContradictionProbeControl(verify=True, max_verifications=1)
    with (
        patch(
            "particles.operations.lint.contradictions._llm_check_contradiction",
            AsyncMock(return_value="first-pass reason"),
        ),
        patch("particles.operations.lint.contradictions._llm_verify_contradiction", verify),
    ):
        findings = await _check_contradictions(db_session, fix=False, control=control)  # type: ignore[arg-type]

    assert verify.await_count == 1
    assert (control.flagged, control.confirmed, control.unverified) == (3, 1, 2)
    assert len(findings) == 1


def _ctx(content: str) -> object:
    from particles.operations.lint.contradictions import ClaimContext

    return ClaimContext(content=content, source="note.md", date="2026-09-26", passage="")


def _scripted_llm_call(
    monkeypatch: pytest.MonkeyPatch, replies: list[str | None]
) -> list[dict[str, object]]:
    """Patch the second reading's ``_llm_call`` to return ``replies`` in order; log each call.

    The reading lives in ``particles.ingest.second_reading``, so its
    seam is patched there.
    """
    import particles.ingest.second_reading as c_mod

    calls: list[dict[str, object]] = []
    queue = list(replies)

    async def fake_llm_call(prompt: str, **kwargs: object) -> str | None:
        calls.append(kwargs)
        return queue.pop(0)

    monkeypatch.setattr(c_mod, "_llm_call", fake_llm_call)
    return calls


_CUT = "REASON: the first note says the job runs at 02:00 UTC while the second"
_YES = "REASON: the hours differ.\nVERDICT: YES"


@pytest.mark.asyncio
@pytest.mark.parametrize("first", [_CUT, ""], ids=["cut", "empty"])
async def test_second_reading_cut_at_the_budget_is_retried_once_at_the_larger_budget(
    monkeypatch: pytest.MonkeyPatch, first: str
) -> None:
    """A reply cut at ``audit.verify_max_tokens`` is re-issued at the retry budget."""
    from particles.config import get_config
    from particles.operations._llm import llm_failure_count
    from particles.operations.lint.contradictions import _llm_verify_contradiction

    cfg = get_config().audit
    calls = _scripted_llm_call(monkeypatch, [first, _YES])
    before = llm_failure_count()

    verdict = await _llm_verify_contradiction(_ctx("a"), _ctx("b"))  # type: ignore[arg-type]

    assert verdict is not None and verdict.contradicts
    assert verdict.description == "the hours differ."
    assert [c["max_tokens"] for c in calls] == [
        cfg.verify_max_tokens,
        cfg.verify_retry_max_tokens,
    ]
    assert all(c["purpose"] == "verification" for c in calls)
    assert all(c["empty_reply_as_text"] is True for c in calls)
    # The retry recovered the reading, so no failed call is disclosed.
    assert llm_failure_count() == before


@pytest.mark.asyncio
async def test_second_reading_still_cut_after_the_retry_is_never_a_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cut twice: no verdict, exactly two calls, one failed reading counted."""
    from particles.operations._llm import llm_failure_count
    from particles.operations.lint.contradictions import _llm_verify_contradiction

    calls = _scripted_llm_call(monkeypatch, [_CUT, _CUT + " VERDICT: YES but"])
    before = llm_failure_count()

    assert await _llm_verify_contradiction(_ctx("a"), _ctx("b")) is None  # type: ignore[arg-type]
    assert len(calls) == 2
    assert llm_failure_count() == before + 1


@pytest.mark.asyncio
async def test_second_reading_budgets_come_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from particles.config import get_config
    from particles.operations.lint.contradictions import _llm_verify_contradiction

    audit = get_config().audit
    monkeypatch.setattr(audit, "verify_max_tokens", 2048)
    monkeypatch.setattr(audit, "verify_retry_max_tokens", 8192)
    calls = _scripted_llm_call(monkeypatch, [_CUT, _YES])

    assert await _llm_verify_contradiction(_ctx("a"), _ctx("b")) is not None  # type: ignore[arg-type]
    assert [c["max_tokens"] for c in calls] == [2048, 8192]


@pytest.mark.asyncio
async def test_second_reading_retry_disabled_at_or_below_the_first_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from particles.config import get_config
    from particles.operations.lint.contradictions import _llm_verify_contradiction

    audit = get_config().audit
    monkeypatch.setattr(audit, "verify_retry_max_tokens", audit.verify_max_tokens)
    calls = _scripted_llm_call(monkeypatch, [_CUT, _YES])

    assert await _llm_verify_contradiction(_ctx("a"), _ctx("b")) is None  # type: ignore[arg-type]
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_second_reading_failed_call_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed call (network, refusal, billing) is not a budget failure."""
    from particles.operations.lint.contradictions import _llm_verify_contradiction

    calls = _scripted_llm_call(monkeypatch, [None, _YES])

    assert await _llm_verify_contradiction(_ctx("a"), _ctx("b")) is None  # type: ignore[arg-type]
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_second_reading_retry_skipped_when_the_breaker_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import particles.ingest.second_reading as c_mod
    from particles.operations.lint.contradictions import _llm_verify_contradiction

    calls = _scripted_llm_call(monkeypatch, [_CUT, _YES])
    monkeypatch.setattr(c_mod, "llm_circuit_open", lambda: True)

    assert await _llm_verify_contradiction(_ctx("a"), _ctx("b")) is None  # type: ignore[arg-type]
    assert len(calls) == 1


def test_verification_budget_hint_names_the_config_knob() -> None:
    from particles.llm.registry import budget_hint
    from particles.llm.usage import purpose_scope

    with purpose_scope("verification"):
        assert "audit.verify_max_tokens" in budget_hint()


@pytest.fixture
def _note_blobs(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-test blob store, set on the config object (see test_source_passage.py)."""
    from particles.config import get_config

    monkeypatch.setattr(get_config().storage, "blob_dir", str(tmp_path))


@pytest.mark.asyncio
async def test_probe_and_second_reading_prompts(
    db_session: object, _note_blobs: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prompts as the Anthropic client receives them.

    The first pass samples at temperature 0 on ``llm.semantic_lint`` and shows
    the bare claims; the second reading goes to ``llm.verification`` and shows
    each claim's note name, note date and source passage, all behind the
    per-call nonce fence.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from particles.config import ProviderSelection, get_config
    from particles.core.schema import Mutability
    from particles.corpus.deposit import deposit_text_versioned
    from particles.llm import set_client
    from particles.operations.lint import ContradictionProbeControl, _check_contradictions

    llm_cfg = get_config().llm
    monkeypatch.setattr(llm_cfg, "semantic_lint", ProviderSelection(model="probe-model"))
    monkeypatch.setattr(llm_cfg, "verification", ProviderSelection(model="verify-model"))

    notes = {
        "deploy-notes.md": (
            "# Deploy\n\n- The nightly job runs at 02:00 UTC.\n",
            datetime(2026, 3, 1, tzinfo=UTC),
        ),
        "ops-notes.md": (
            "# Ops\n\n- The nightly job runs at 04:00 UTC after the backup.\n",
            datetime(2026, 5, 9, tzinfo=UTC),
        ),
    }
    claims = ("The nightly job runs at 02:00 UTC.", "The nightly job runs at 04:00 UTC.")
    for (name, (text, when)), claim, emb in zip(
        notes.items(), claims, (_EMB_HI_A, _EMB_HI_B), strict=True
    ):
        entry_id, snap_id, _ = await deposit_text_versioned(
            db_session,  # type: ignore[arg-type]
            text=text,
            uri_r=f"file:///notes/{name}",
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.MUTABLE,
            content_published_at=when,
        )
        p = _make_particle(claim).model_copy(
            update={
                "provenance": [
                    ProvenanceRef(
                        type=ProvenanceRefType.SOURCE,
                        corpus_entry_id=entry_id,
                        snapshot_id=snap_id,
                    )
                ]
            }
        )
        await insert_particle(db_session, p, embedding=emb)  # type: ignore[arg-type]
    await db_session.commit()  # type: ignore[attr-defined]

    reply = SimpleNamespace(
        content=[SimpleNamespace(text="REASON: the hours differ.\nVERDICT: YES")],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )
    client = MagicMock()
    client.messages.create.return_value = reply
    set_client(client)
    try:
        control = ContradictionProbeControl(verify=True)
        findings = await _check_contradictions(db_session, fix=False, control=control)  # type: ignore[arg-type]
    finally:
        set_client(None)

    assert len(findings) == 1
    assert (control.flagged, control.confirmed) == (1, 1)
    probe, second = (c.kwargs for c in client.messages.create.call_args_list)

    assert probe["model"] == "probe-model"
    assert probe["extra_body"] == {"temperature": 0.0}
    probe_user = probe["messages"][0]["content"]
    assert "02:00 UTC" in probe_user and "04:00 UTC" in probe_user
    assert "deploy-notes.md" not in probe_user

    assert second["model"] == "verify-model"
    system = second["system"] if isinstance(second["system"], str) else str(second["system"])
    assert "Different note dates alone never make a change" in system
    user = second["messages"][0]["content"]
    assert "note: deploy-notes.md (dated 2026-03-01)" in user
    assert "note: ops-notes.md (dated 2026-05-09)" in user
    assert "passage: - The nightly job runs at 04:00 UTC after the backup." in user
    # Everything but the instruction sits inside a nonce fence named in the system turn.
    nonce = user.split('<claim_a nonce="', 1)[1].split('"', 1)[0]
    assert f'</claim_b nonce="{nonce}">' in user
    assert nonce in system


class TestCountDisagreements:
    """reported pairs connected through a shared claim count once."""

    def test_empty(self) -> None:
        from particles.operations.lint.contradictions import count_disagreements

        grouped = count_disagreements([])
        assert (grouped.groups, grouped.within_one_note, grouped.pairs) == (0, 0, 0)

    def test_pairs_sharing_a_claim_are_one_disagreement(self) -> None:
        from particles.operations.lint.contradictions import count_disagreements

        # One claim against two phrasings of the other; a chain through a third
        # note; and an unrelated pair.
        grouped = count_disagreements(
            [("a", "b", False), ("a", "c", False), ("c", "d", False), ("x", "y", False)]
        )
        assert (grouped.groups, grouped.pairs) == (2, 4)
        assert grouped.within_one_note == 0

    def test_a_group_is_within_one_note_only_when_every_pair_is(self) -> None:
        from particles.operations.lint.contradictions import count_disagreements

        grouped = count_disagreements(
            [("a", "b", True), ("b", "c", False), ("x", "y", True), ("y", "z", True)]
        )
        assert grouped.groups == 2
        assert grouped.within_one_note == 1  # {x, y, z}


@pytest.mark.asyncio
async def test_reported_pairs_feed_the_grouping(db_session: object) -> None:
    from unittest.mock import AsyncMock, patch

    from particles.operations.lint import ContradictionProbeControl, _check_contradictions

    await _three_claims(db_session)
    control = ContradictionProbeControl()
    with patch(
        "particles.operations.lint.contradictions._llm_check_contradiction",
        AsyncMock(return_value="they disagree"),
    ):
        await _check_contradictions(db_session, fix=False, control=control)  # type: ignore[arg-type]

    # alpha, bravo and charlie all disagree with each other: one disagreement.
    assert len(control.finding_pairs) == 3
    assert control.disagreements().groups == 1
    assert control.disagreements().within_one_note == 0


@pytest.mark.asyncio
async def test_confirmed_pairs_carry_the_second_readings_reason(db_session: object) -> None:
    """the disclosure pass reads each confirmed pair and its reason."""
    from unittest.mock import AsyncMock, patch

    from particles.core.contradiction_disclosure import ConfirmedPair
    from particles.operations.lint import ContradictionProbeControl, _check_contradictions
    from particles.operations.lint.contradictions import ProbeVerdict

    p_a, p_b, p_c = await _three_claims(db_session)
    verify = AsyncMock(
        side_effect=[
            ProbeVerdict(usable=True, contradicts=True, description="values disagree"),
            ProbeVerdict(usable=True, contradicts=False),
            ProbeVerdict(usable=True, contradicts=False),
        ]
    )
    control = ContradictionProbeControl(verify=True)
    with (
        patch(
            "particles.operations.lint.contradictions._llm_check_contradiction",
            AsyncMock(return_value="first pass"),
        ),
        patch("particles.operations.lint.contradictions._llm_verify_contradiction", verify),
    ):
        await _check_contradictions(db_session, fix=False, control=control)  # type: ignore[arg-type]

    assert len(control.confirmed_pairs) == 1
    confirmed = control.confirmed_pairs[0]
    assert isinstance(confirmed, ConfirmedPair)
    assert confirmed.reason == "values disagree"
    assert not confirmed.same_source  # the cross-note pair is probed first
    assert confirmed.key == frozenset((p_b.id, p_c.id))
    del p_a


@pytest.mark.asyncio
async def test_excluded_pairs_are_never_probed(db_session: object) -> None:
    """a pair a census record discloses is not probed again."""
    from unittest.mock import AsyncMock, patch

    from particles.operations.lint import ContradictionProbeControl, _check_contradictions

    p_a, p_b, p_c = await _three_claims(db_session)
    probe = AsyncMock(return_value="they disagree")
    control = ContradictionProbeControl(exclude_pairs=frozenset({frozenset((p_b.id, p_c.id))}))
    with patch("particles.operations.lint.contradictions._llm_check_contradiction", probe):
        await _check_contradictions(db_session, fix=False, control=control)  # type: ignore[arg-type]

    probed = {frozenset(call.args) for call in probe.await_args_list}
    assert frozenset((p_b.content, p_c.content)) not in probed
    assert control.candidate_pairs == 2
    del p_a


def test_contradiction_partner_reads_the_detail() -> None:
    from particles.operations.lint.contradictions import contradiction_partner

    assert contradiction_partner("Semantic contradiction with particle abc-123: why") == "abc-123"
    assert (
        contradiction_partner("Recorded contradiction with particle def-4 (two projects); no")
        == "def-4"
    )
    assert contradiction_partner("something else") is None


def test_second_reading_windows_each_passage_for_its_pair() -> None:
    """a long passage is windowed by the claim and, second, by its partner."""
    from particles.operations.lint.contradictions import _claim_context_for, _ClaimSource

    filler = "Unrelated remarks about the garden shed and the weather follow here. " * 12
    note = (
        "Port 8080 serves the admin console for staff. "
        + filler
        + "Port 8080 serves the admin console, not the public status endpoint."
    )
    claim = _make_particle("Port 8080 serves the admin console.")
    partner = _make_particle("The public status endpoint is served on port 8080.")
    src = _ClaimSource(source="ops.md", date="2026-05-09", text=note)

    alone = _claim_context_for(claim, src, None)
    paired = _claim_context_for(claim, src, partner)

    assert (paired.source, paired.date, paired.content) == ("ops.md", "2026-05-09", claim.content)
    assert "for staff" in alone.passage
    assert "public status endpoint" in paired.passage
    assert len(paired.passage) <= 902  # the window plus its two ellipses


class TestSourceKindReading:
    """The second reading reads a transcript's claims as observations."""

    @staticmethod
    def _ctx(kind: str = "note", **fields: object) -> object:
        from particles.operations.lint.contradictions import ClaimContext, SourceKind

        base: dict[str, object] = {
            "content": "The gate passes with 10 rows.",
            "source": "session-1",
            "date": "2026-09-18",
            "passage": "the gate passes with 10 rows",
            "kind": SourceKind(kind),
        }
        base.update(fields)
        return ClaimContext(**base)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("source_type", "mutability", "kind"),
        [
            ("CONVERSATION", "APPEND_ONLY", "record"),
            ("LOCAL_MARKDOWN", "APPEND_ONLY", "record"),
            ("CONVERSATION", "MUTABLE", "record"),
            ("LOCAL_MARKDOWN", "MUTABLE", "note"),
            ("WEB_PAGE", "STABLE", "note"),
            (None, None, "note"),
        ],
    )
    def test_source_kind(self, source_type: str | None, mutability: str | None, kind: str) -> None:
        from particles.operations.lint.contradictions import source_kind

        assert source_kind(source_type, mutability).value == kind

    def test_reading_for(self) -> None:
        from particles.core.contradiction_disclosure import READING_OBSERVED, READING_STANDING
        from particles.operations.lint.contradictions import reading_for

        assert reading_for(self._ctx(), self._ctx()) == READING_STANDING  # type: ignore[arg-type]
        assert reading_for(self._ctx(), self._ctx("record")) == READING_OBSERVED  # type: ignore[arg-type]
        assert reading_for(self._ctx("record"), self._ctx("record")) == READING_OBSERVED  # type: ignore[arg-type]

    def test_two_notes_keep_the_standing_facts_instruction(self) -> None:
        from particles.operations.lint.contradictions import _VERIFY_INSTRUCTION, _verify_request

        request = _verify_request(self._ctx(), self._ctx(source="n2.md"))  # type: ignore[arg-type]
        assert request.system.startswith(_VERIFY_INSTRUCTION)
        assert "note: session-1 (dated 2026-09-18)" in request.prompt
        assert "observed" not in request.prompt

    def test_a_record_gets_the_observed_state_instruction_and_times(self) -> None:
        from particles.operations.lint.contradictions import (
            _VERIFY_OBSERVED_INSTRUCTION,
            _verify_request,
        )

        record = self._ctx("record", observed="2026-09-18 04:23 UTC", source_type="CONVERSATION")
        log = self._ctx("record", source="archive.md", observed="2026-09-25 09:09 UTC")
        note = self._ctx(source="n2.md", date="2026-09-20")
        request = _verify_request(record, log)  # type: ignore[arg-type]
        assert request.system.startswith(_VERIFY_OBSERVED_INSTRUCTION)
        assert "record: session-1 (session transcript, observed 2026-09-18 04:23 UTC)" in (
            request.prompt
        )
        assert "record: archive.md (record, observed 2026-09-25 09:09 UTC)" in request.prompt
        mixed = _verify_request(record, note)  # type: ignore[arg-type]
        assert mixed.system.startswith(_VERIFY_OBSERVED_INSTRUCTION)
        assert "note: n2.md (a note kept current, dated 2026-09-20)" in mixed.prompt
        # The instruction states both confirming cases.
        assert "1. The same moment." in _VERIFY_OBSERVED_INSTRUCTION
        assert "2. A fixed fact." in _VERIFY_OBSERVED_INSTRUCTION

    def test_a_no_verdict_keeps_its_reason(self) -> None:
        from particles.operations.lint.contradictions import _parse_probe_verdict

        verdict = _parse_probe_verdict("REASON: counts at two moments.\nVERDICT: NO")
        assert verdict.usable and not verdict.contradicts
        assert verdict.description == "counts at two moments."

    @pytest.mark.asyncio
    async def test_claim_source_reads_kind_and_observation_time(self, db_session: object) -> None:
        from datetime import UTC, datetime

        from particles.core.schema import Mutability
        from particles.operations.lint.contradictions import SourceKind, _claim_source
        from tests._observer_scope import belief, generation

        t = await generation(
            db_session,
            "session-1",
            "The gate passes with 10 rows.",
            [],
            mutability=Mutability.APPEND_ONLY,
            captured_at=datetime(2026, 9, 18, 4, 23, tzinfo=UTC),
        )
        n = await generation(db_session, "note.md", "The gate passes with 12 rows.", [])
        claim_t = await belief(db_session, "The gate passes with 10 rows.", t)
        claim_n = await belief(db_session, "The gate passes with 12 rows.", n)

        src_t = await _claim_source(db_session, claim_t)  # type: ignore[arg-type]
        src_n = await _claim_source(db_session, claim_n)  # type: ignore[arg-type]
        assert (src_t.kind, src_t.observed) == (SourceKind.RECORD, "2026-09-18 04:23 UTC")
        assert src_n.kind is SourceKind.NOTE

    @pytest.mark.asyncio
    async def test_reread_reads_only_stale_pairs_within_the_budget(
        self, db_session: object
    ) -> None:
        from unittest.mock import AsyncMock, patch

        from particles.core.contradiction_disclosure import READING_OBSERVED, ConfirmedPair
        from particles.core.schema import Mutability
        from particles.operations.lint.contradictions import ProbeVerdict, reread_stale_pairs
        from tests._observer_scope import belief, generation

        async def record(name: str, text: str) -> object:
            src = await generation(db_session, name, text, [], mutability=Mutability.APPEND_ONLY)
            return await belief(db_session, text, src)

        a, b, c, d = [await record(f"s{i}", f"The count is {i}.") for i in range(4)]
        e_src = await generation(db_session, "e.md", "The endpoint exists.", [])
        f_src = await generation(db_session, "f.md", "The endpoint returns 404.", [])
        e = await belief(db_session, "The endpoint exists.", e_src)
        f = await belief(db_session, "The endpoint returns 404.", f_src)
        particles = {p.id: p for p in (a, b, c, d, e, f)}  # type: ignore[attr-defined]

        def pair(x: object, y: object, reading: str = "standing/1") -> ConfirmedPair:
            return ConfirmedPair(
                a=x.id,
                b=y.id,
                same_source=False,
                reason="old",
                reading=reading,  # type: ignore[attr-defined]
            )

        stale_1, stale_2 = pair(a, b), pair(c, d)
        current = pair(a, c, READING_OBSERVED)
        notes = pair(e, f)
        reader = AsyncMock(
            return_value=ProbeVerdict(usable=True, contradicts=False, description="a change")
        )
        with patch("particles.operations.lint.contradictions._llm_verify_contradiction", reader):
            outcome = await reread_stale_pairs(
                db_session,  # type: ignore[arg-type]
                [current, notes, stale_1, stale_2],
                particles,  # type: ignore[arg-type]
                budget=1,
            )

        assert reader.await_count == 1
        assert outcome.verdicts == {stale_1.key: None}
        assert outcome.withdrawn_reasons == {stale_1.key: "a change"}
        assert (outcome.read, outcome.withdrawn, outcome.deferred) == (1, 1, 1)

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

    probed: list[tuple[str, str]] = []

    async def _record(content_a: str, content_b: str) -> None:
        probed.append((content_a, content_b))
        return None

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

    Companion to the tiering test above: with ``scope_particle_ids=None`` the tier reads 0 for every pair and the highest-similarity
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
    (LocalProvider structured output) and the original YES/NO text protocol."""

    def test_yes_no_text_protocol_unchanged(self) -> None:
        from particles.operations.lint.contradictions import _parse_probe_reply

        assert _parse_probe_reply("YES: dates disagree") == "dates disagree"
        assert _parse_probe_reply("YES") == "contradiction detected"
        assert _parse_probe_reply("NO") is None
        assert _parse_probe_reply("no idea") is None

    def test_json_verdict_object(self) -> None:
        from particles.operations.lint.contradictions import _parse_probe_reply

        assert (
            _parse_probe_reply('{"contradicts": true, "description": "dates disagree"}')
            == "dates disagree"
        )
        assert _parse_probe_reply('{"contradicts": true}') == "contradiction detected"
        assert _parse_probe_reply('{"contradicts": false}') is None
        # Malformed JSON falls through to the text protocol, not an error.
        assert _parse_probe_reply('{"contradicts": ') is None

    @pytest.mark.asyncio
    async def test_probe_threads_response_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.operations.lint import contradictions as c_mod

        captured: dict[str, object] = {}

        async def fake_llm_call(*args: object, **kwargs: object) -> str:
            captured.update(kwargs)
            return "NO"

        monkeypatch.setattr(c_mod, "_llm_call", fake_llm_call)
        result = await c_mod._llm_check_contradiction("A", "B")
        assert result is None
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
        return ["YES: alpha and bravo disagree"] + [None] * (len(requests) - 1)

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

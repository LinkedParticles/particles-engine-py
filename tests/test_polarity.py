"""Tests for claim-polarity classification (capability 1).

Covers the polarity contract (``particles.extraction.polarity``), the
extractor's classification (``particles.extraction.general``), and the
consumers that must keep non-asserted (DECLINED / HYPOTHETICAL) particles out
of the default factual surface: the query operation, the lint contradiction
detector, the extraction pipeline's §6.6 conflict resolution, and the one-way
exporters. The round-trippable interchange export, by contrast, must *preserve*
polarity (it rides on ``properties``).

The classifier's *accuracy* on real ADR prose (does the LLM actually tag a
Rejected-Alternatives entry DECLINED?) is the live-LLM acceptance check in
``tests/test_integration_polarity.py``; these tests pin the deterministic
wiring around it (tests/AGENTS.md § What requires tests / Out of scope).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from particles.config import get_config, reset_config
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    QueryRequest,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.extraction.general import _build_extract_prompt, _parse_extraction_response
from particles.extraction.polarity import (
    NON_ASSERTED_POLARITIES,
    POLARITY_ASSERTED,
    POLARITY_DECLINED,
    POLARITY_HYPOTHETICAL,
    POLARITY_KEY,
    is_non_asserted,
)

# --------------------------------------------------------------------------
# The shared exclusion predicate
# --------------------------------------------------------------------------


class TestIsNonAsserted:
    def test_none_properties(self) -> None:
        assert is_non_asserted(None) is False

    def test_empty_properties(self) -> None:
        assert is_non_asserted({}) is False

    def test_absent_key_is_asserted(self) -> None:
        # Absence of the key ⇒ ASSERTED ⇒ not excluded (the back-compat default).
        assert is_non_asserted({"nmo:hasWeight": 0.75}) is False

    def test_explicit_asserted_not_excluded(self) -> None:
        assert is_non_asserted({POLARITY_KEY: POLARITY_ASSERTED}) is False

    def test_declined_excluded(self) -> None:
        assert is_non_asserted({POLARITY_KEY: POLARITY_DECLINED}) is True

    def test_hypothetical_excluded(self) -> None:
        assert is_non_asserted({POLARITY_KEY: POLARITY_HYPOTHETICAL}) is True

    def test_non_asserted_set(self) -> None:
        assert {POLARITY_DECLINED, POLARITY_HYPOTHETICAL} == NON_ASSERTED_POLARITIES
        assert POLARITY_ASSERTED not in NON_ASSERTED_POLARITIES


# --------------------------------------------------------------------------
# Prompt assembly
# --------------------------------------------------------------------------


class TestBuildExtractPrompt:
    def test_polarity_clause_present_when_enabled(self) -> None:
        prompt = _build_extract_prompt(
            scope_enabled=False, modality_enabled=False, polarity_enabled=True
        )
        assert "polarity" in prompt
        assert "DECLINED" in prompt
        assert "HYPOTHETICAL" in prompt
        # F3: source moved to a fenced user turn; the trusted prompt ends with
        # the JSON schema, no "SOURCE TEXT:" label.
        assert "SOURCE TEXT:" not in prompt
        assert prompt.rstrip().endswith("}")

    def test_polarity_clause_absent_when_disabled(self) -> None:
        prompt = _build_extract_prompt(
            scope_enabled=False, modality_enabled=False, polarity_enabled=False
        )
        assert '"polarity"' not in prompt
        assert "SOURCE TEXT:" not in prompt
        assert prompt.rstrip().endswith("}")


# --------------------------------------------------------------------------
# Parser classification + default-safe handling
# --------------------------------------------------------------------------


def _raw(*items: dict[str, object]) -> str:
    return json.dumps(list(items))


# Representative ADR-genre prose for each polarity (the acceptance corpus the
# task names: rejected-alternatives → DECLINED, counterfactual/deferred →
# HYPOTHETICAL, real decision → ASSERTED). The LLM supplies the label; here we
# pin that the parser routes each label correctly.
_ASSERTED = {
    "content": "Polarity lives on properties['extraction:polarity'].",
    "confidence_value": 0.95,
    "uncertainty_nature": "EPISTEMIC",
    "polarity": "ASSERTED",
}
_DECLINED = {
    "content": "Reusing assertion_modality for polarity was rejected as conflating two axes.",
    "confidence_value": 0.9,
    "uncertainty_nature": "EPISTEMIC",
    "polarity": "DECLINED",
}
_HYPOTHETICAL = {
    "content": "Without a single source of truth, audit trails will be unreliable.",
    "confidence_value": 0.9,
    "uncertainty_nature": "EPISTEMIC",
    "polarity": "HYPOTHETICAL",
}


class TestParsePolarity:
    def test_declined_tagged(self) -> None:
        reset_config()
        candidates, notes = _parse_extraction_response(_raw(_ASSERTED, _DECLINED))
        assert len(candidates) == 2
        asserted, declined = candidates
        # ASSERTED is the default ⇒ no properties tag (back-compat unchanged).
        assert asserted.properties is None
        assert declined.properties == {POLARITY_KEY: POLARITY_DECLINED}
        assert is_non_asserted(declined.properties) is True
        assert any("DECLINED" in n for n in notes)

    def test_hypothetical_tagged(self) -> None:
        reset_config()
        candidates, _ = _parse_extraction_response(_raw(_HYPOTHETICAL))
        assert candidates[0].properties == {POLARITY_KEY: POLARITY_HYPOTHETICAL}

    def test_asserted_not_tagged(self) -> None:
        reset_config()
        candidates, _ = _parse_extraction_response(_raw(_ASSERTED))
        assert candidates[0].properties is None

    def test_missing_polarity_defaults_asserted(self) -> None:
        reset_config()
        item = {k: v for k, v in _ASSERTED.items() if k != "polarity"}
        candidates, _ = _parse_extraction_response(_raw(item))
        assert candidates[0].properties is None

    def test_invalid_polarity_defaults_asserted_with_note(self) -> None:
        reset_config()
        item = dict(_ASSERTED)
        item["polarity"] = "MAYBE"
        candidates, notes = _parse_extraction_response(_raw(item))
        assert candidates[0].properties is None
        assert any("invalid polarity" in n for n in notes)

    def test_disabled_ignores_polarity(self) -> None:
        reset_config()
        get_config().extraction_polarity.enabled = False
        candidates, _ = _parse_extraction_response(_raw(_DECLINED))
        assert len(candidates) == 1
        # No tag applied — behaves as an ordinary ASSERTED particle.
        assert candidates[0].properties is None

    def test_confidence_never_demoted(self) -> None:
        """Polarity governs visibility, never the truth scalar."""
        reset_config()
        declined = dict(_DECLINED)
        declined["confidence_value"] = 0.88
        candidates, _ = _parse_extraction_response(_raw(declined))
        assert candidates[0].confidence_value == 0.88

    def test_polarity_coexists_with_scope_on_one_dict(self) -> None:
        """A candidate can be both DOCUMENT_META and non-asserted; both keys land."""
        reset_config()
        from particles.extraction.scope import SCOPE_DOCUMENT_META, SCOPE_KEY

        item = dict(_DECLINED)
        item["scope"] = "DOCUMENT_META"
        candidates, _ = _parse_extraction_response(_raw(item))
        props = candidates[0].properties
        assert props == {SCOPE_KEY: SCOPE_DOCUMENT_META, POLARITY_KEY: POLARITY_DECLINED}


# --------------------------------------------------------------------------
# Consumer behaviour: query, lint, pipeline, export, interchange
# --------------------------------------------------------------------------


def _make_particle(content: str, properties: dict[str, object] | None = None) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test-agent",
        status=Status.ACTIVE,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1", snapshot_id="snap-1"
            )
        ],
        properties=properties,
    )


@pytest.mark.asyncio
async def test_query_excludes_non_asserted_by_default(db_session: object) -> None:
    import anthropic
    import numpy as np

    from particles import embeddings as ep
    from particles.llm import set_client
    from particles.operations.query import query
    from particles.store.particle_store import insert_particle

    session = db_session  # type: ignore[assignment]
    emb = (np.ones(4, dtype=np.float32) / 2.0).tolist()
    await insert_particle(session, _make_particle("The two-quantity model is current."), emb)  # type: ignore[arg-type]
    await insert_particle(  # type: ignore[arg-type]
        session,
        _make_particle("The three-quantity model was rejected.", {POLARITY_KEY: POLARITY_DECLINED}),
        emb,
    )
    await session.commit()  # type: ignore[union-attr]

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
    mock_content = MagicMock()
    mock_content.text = "answer"
    mock_resp = MagicMock()
    mock_resp.content = [mock_content]
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages = MagicMock()
    mock_client.messages.create = MagicMock(return_value=mock_resp)

    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    set_client(mock_client)
    try:
        default = await query(session, QueryRequest(question="q", top_k=10))  # type: ignore[arg-type]
        contents = {p.content for p in default.particles}
        assert "The two-quantity model is current." in contents
        assert "The three-quantity model was rejected." not in contents

        included = await query(  # type: ignore[arg-type]
            session, QueryRequest(question="q", top_k=10, include_non_asserted=True)
        )
        contents_incl = {p.content for p in included.particles}
        assert "The three-quantity model was rejected." in contents_incl
    finally:
        ep.set_embedding_model(original_model)
        set_client(None)


@pytest.mark.asyncio
async def test_lint_excludes_non_asserted_from_contradictions(db_session: object) -> None:
    import anthropic

    from particles.llm import set_client
    from particles.operations.lint.contradictions import _check_contradictions
    from particles.store.particle_store import insert_particle

    session = db_session  # type: ignore[assignment]
    # One ASSERTED + two non-asserted. Without the filter the
    # rejected-vs-kept deliberation would fire as an intra-source contradiction.
    await insert_particle(session, _make_particle("Use the two-quantity model."), None)  # type: ignore[arg-type]
    await insert_particle(  # type: ignore[arg-type]
        session,
        _make_particle("Keep the three-quantity model.", {POLARITY_KEY: POLARITY_DECLINED}),
        None,
    )
    await insert_particle(  # type: ignore[arg-type]
        session,
        _make_particle("Replace the three-quantity model.", {POLARITY_KEY: POLARITY_DECLINED}),
        None,
    )
    await session.commit()  # type: ignore[union-attr]

    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages = MagicMock()
    mock_client.messages.create = MagicMock()
    set_client(mock_client)
    try:
        findings = await _check_contradictions(session, fix=False)  # type: ignore[arg-type]
        assert findings == []
        # Only one asserted particle remained, so no pair was ever LLM-checked.
        mock_client.messages.create.assert_not_called()
    finally:
        set_client(None)


@pytest.mark.asyncio
async def test_pipeline_writes_non_asserted_active_with_tag(
    db_session: object, tmp_path: Path
) -> None:
    import anthropic

    from particles import embeddings as ep
    from particles.corpus.deposit import deposit_file
    from particles.ingest.pipeline import extract_snapshot
    from particles.llm import set_client

    doc = tmp_path / "adr.txt"
    doc.write_text("A decision and a rejected alternative.")

    mock_content = MagicMock()
    mock_content.text = json.dumps(
        [
            {
                "content": "The SDK adopts the two-quantity confidence model.",
                "confidence_value": 0.95,
                "uncertainty_nature": "EPISTEMIC",
                "polarity": "ASSERTED",
            },
            {
                "content": "The three-quantity model was rejected as superseded.",
                "confidence_value": 0.95,
                "uncertainty_nature": "EPISTEMIC",
                "polarity": "DECLINED",
            },
        ]
    )
    mock_resp = MagicMock()
    mock_resp.content = [mock_content]
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages = MagicMock()
    mock_client.messages.create = MagicMock(return_value=mock_resp)

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    set_client(mock_client)
    try:
        session = db_session  # type: ignore[assignment]
        entry_id, snapshot_id = await deposit_file(session, doc, deposited_by="test")  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        particles = await extract_snapshot(session, entry_id, snapshot_id)  # type: ignore[arg-type]
        await session.commit()  # type: ignore[union-attr]

        assert len(particles) == 2
        # Both stored ACTIVE (label, never delete — non-asserted is not a status).
        assert all(p.status == Status.ACTIVE for p in particles)
        by_content = {p.content: p for p in particles}
        declined = by_content["The three-quantity model was rejected as superseded."]
        assert declined.properties == {POLARITY_KEY: POLARITY_DECLINED}
        assert is_non_asserted(declined.properties) is True
        asserted = by_content["The SDK adopts the two-quantity confidence model."]
        assert asserted.properties is None
    finally:
        ep.set_embedding_model(original_model)
        set_client(None)


@pytest.mark.asyncio
async def test_jsonl_export_excludes_non_asserted_by_default(
    db_session: object, tmp_path: Path
) -> None:
    from particles.exporters.jsonl import JsonlExporter
    from particles.store.particle_store import insert_particle

    session = db_session  # type: ignore[assignment]
    await insert_particle(session, _make_particle("A current decision."))  # type: ignore[arg-type]
    await insert_particle(  # type: ignore[arg-type]
        session,
        _make_particle("A rejected alternative.", {POLARITY_KEY: POLARITY_DECLINED}),
    )
    await session.commit()  # type: ignore[union-attr]

    out = tmp_path / "default.jsonl"
    summary = await JsonlExporter().export(session, out)  # type: ignore[arg-type]
    contents = {json.loads(line)["content"] for line in out.read_text().splitlines()}
    assert contents == {"A current decision."}
    assert summary.particles_written == 1

    out_incl = tmp_path / "incl.jsonl"
    await JsonlExporter().export(session, out_incl, include_non_asserted=True)  # type: ignore[arg-type]
    contents_incl = {json.loads(line)["content"] for line in out_incl.read_text().splitlines()}
    assert contents_incl == {"A current decision.", "A rejected alternative."}


def test_interchange_round_trip_preserves_polarity() -> None:
    """The round-trippable interchange export must NOT drop non-asserted particles."""
    from particles.interchange import from_unit, to_unit

    p = _make_particle("A rejected alternative.", {POLARITY_KEY: POLARITY_DECLINED})
    restored = from_unit(to_unit(p, {})).particle
    assert restored.properties == {POLARITY_KEY: POLARITY_DECLINED}
    # Still non-asserted after the round-trip (interchange preserves the axis).
    assert is_non_asserted(restored.properties) is True

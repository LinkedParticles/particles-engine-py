"""Tests for document-scope meta-claim labelling.

Covers the scope contract (``particles.extraction.scope``), the extractor's
classification + mode handling (``particles.extraction.general``), and the
three consumers that must keep DOCUMENT_META particles out of the factual
surface: the query operation, the lint contradiction detector, and the
extraction pipeline's §6.6 conflict resolution.
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
from particles.extraction.scope import (
    SCOPE_ACTION_KEY,
    SCOPE_ACTION_OBSERVE,
    SCOPE_DOCUMENT_META,
    SCOPE_KEY,
    is_excluded_document_meta,
)

# --------------------------------------------------------------------------
# The shared exclusion predicate
# --------------------------------------------------------------------------


class TestIsExcludedDocumentMeta:
    def test_none_properties(self) -> None:
        assert is_excluded_document_meta(None) is False

    def test_empty_properties(self) -> None:
        assert is_excluded_document_meta({}) is False

    def test_world_scope(self) -> None:
        assert is_excluded_document_meta({SCOPE_KEY: "WORLD"}) is False

    def test_document_meta_excluded(self) -> None:
        assert is_excluded_document_meta({SCOPE_KEY: SCOPE_DOCUMENT_META}) is True

    def test_passthrough_observe_not_excluded(self) -> None:
        props = {SCOPE_KEY: SCOPE_DOCUMENT_META, SCOPE_ACTION_KEY: SCOPE_ACTION_OBSERVE}
        assert is_excluded_document_meta(props) is False

    def test_unrelated_properties(self) -> None:
        assert is_excluded_document_meta({"nmo:hasWeight": 0.75}) is False


# --------------------------------------------------------------------------
# Prompt assembly
# --------------------------------------------------------------------------


class TestBuildExtractPrompt:
    def test_scope_clause_present_when_enabled(self) -> None:
        prompt = _build_extract_prompt(scope_enabled=True, modality_enabled=False)
        assert "DOCUMENT_META" in prompt
        assert '"scope"' in prompt
        # F3: the source is now fenced in the user turn, so the trusted prompt
        # ends with the JSON schema and no longer carries a "SOURCE TEXT:" label.
        assert "SOURCE TEXT:" not in prompt
        assert prompt.rstrip().endswith("}")

    def test_scope_clause_absent_when_disabled(self) -> None:
        prompt = _build_extract_prompt(scope_enabled=False, modality_enabled=False)
        assert "DOCUMENT_META" not in prompt
        assert '"scope"' not in prompt
        assert "SOURCE TEXT:" not in prompt
        assert prompt.rstrip().endswith("}")


# --------------------------------------------------------------------------
# Parser classification + mode handling
# --------------------------------------------------------------------------


def _raw(*items: dict[str, object]) -> str:
    return json.dumps(list(items))


_WORLD = {
    "content": "Water boils at 100C.",
    "confidence_value": 0.9,
    "uncertainty_nature": "EPISTEMIC",
}
_META = {
    "content": "Section 10.4 defines the exporter.",
    "confidence_value": 0.9,
    "uncertainty_nature": "EPISTEMIC",
    "scope": "DOCUMENT_META",
}


class TestParseScope:
    def test_label_mode_tags_and_keeps(self) -> None:
        reset_config()
        get_config().extraction_scope.mode = "label"
        candidates, notes = _parse_extraction_response(_raw(_WORLD, _META))
        assert len(candidates) == 2
        world, meta = candidates
        assert world.properties is None
        assert meta.properties == {SCOPE_KEY: SCOPE_DOCUMENT_META}
        assert any("DOCUMENT_META" in n for n in notes)

    def test_suppress_mode_drops(self) -> None:
        reset_config()
        get_config().extraction_scope.mode = "suppress"
        candidates, notes = _parse_extraction_response(_raw(_WORLD, _META))
        assert len(candidates) == 1
        assert candidates[0].content == "Water boils at 100C."
        assert any("suppressed" in n.lower() for n in notes)

    def test_passthrough_mode_tags_without_exclusion(self) -> None:
        reset_config()
        get_config().extraction_scope.mode = "passthrough"
        candidates, _ = _parse_extraction_response(_raw(_META))
        assert len(candidates) == 1
        props = candidates[0].properties
        assert props == {SCOPE_KEY: SCOPE_DOCUMENT_META, SCOPE_ACTION_KEY: SCOPE_ACTION_OBSERVE}
        # passthrough particles are not excluded downstream
        assert is_excluded_document_meta(props) is False

    def test_disabled_ignores_scope(self) -> None:
        reset_config()
        get_config().extraction_scope.enabled = False
        candidates, _ = _parse_extraction_response(_raw(_META))
        assert len(candidates) == 1
        # No tag applied — behaves as an ordinary WORLD particle.
        assert candidates[0].properties is None

    def test_missing_scope_defaults_world(self) -> None:
        reset_config()
        candidates, _ = _parse_extraction_response(_raw(_WORLD))
        assert len(candidates) == 1
        assert candidates[0].properties is None

    def test_confidence_never_demoted(self) -> None:
        """Scope governs visibility, never the truth scalar."""
        reset_config()
        get_config().extraction_scope.mode = "label"
        meta = dict(_META)
        meta["confidence_value"] = 0.42
        world = dict(_WORLD)
        world["confidence_value"] = 0.42
        candidates, _ = _parse_extraction_response(_raw(world, meta))
        assert candidates[0].confidence_value == 0.42
        assert candidates[1].confidence_value == 0.42


# --------------------------------------------------------------------------
# Consumer behaviour: query, lint, pipeline
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
async def test_query_excludes_document_meta_by_default(db_session: object) -> None:
    import anthropic
    import numpy as np

    from particles import embeddings as ep
    from particles.llm import set_client
    from particles.operations.query import query
    from particles.store.particle_store import insert_particle

    session = db_session  # type: ignore[assignment]
    emb = (np.ones(4, dtype=np.float32) / 2.0).tolist()
    await insert_particle(session, _make_particle("Water is wet."), emb)  # type: ignore[arg-type]
    await insert_particle(  # type: ignore[arg-type]
        session,
        _make_particle("Section 10.4 defines the exporter.", {SCOPE_KEY: SCOPE_DOCUMENT_META}),
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
        assert "Water is wet." in contents
        assert "Section 10.4 defines the exporter." not in contents

        included = await query(  # type: ignore[arg-type]
            session, QueryRequest(question="q", top_k=10, include_document_meta=True)
        )
        contents_incl = {p.content for p in included.particles}
        assert "Section 10.4 defines the exporter." in contents_incl
    finally:
        ep.set_embedding_model(original_model)
        set_client(None)


@pytest.mark.asyncio
async def test_lint_excludes_document_meta_from_contradictions(db_session: object) -> None:
    import anthropic

    from particles.llm import set_client
    from particles.operations.lint.contradictions import _check_contradictions
    from particles.store.particle_store import insert_particle

    session = db_session  # type: ignore[assignment]
    # One WORLD + two DOCUMENT_META, all sharing entry-1. Without the scope
    # filter, the shared-source pairs would invoke the contradiction LLM.
    await insert_particle(session, _make_particle("Water is wet."), None)  # type: ignore[arg-type]
    await insert_particle(  # type: ignore[arg-type]
        session, _make_particle("Section 1 exists.", {SCOPE_KEY: SCOPE_DOCUMENT_META}), None
    )
    await insert_particle(  # type: ignore[arg-type]
        session, _make_particle("Section 2 exists.", {SCOPE_KEY: SCOPE_DOCUMENT_META}), None
    )
    await session.commit()  # type: ignore[union-attr]

    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages = MagicMock()
    mock_client.messages.create = MagicMock()
    set_client(mock_client)
    try:
        findings = await _check_contradictions(session, fix=False)  # type: ignore[arg-type]
        assert findings == []
        # Only one non-meta particle remained, so no pair was ever LLM-checked.
        mock_client.messages.create.assert_not_called()
    finally:
        set_client(None)


@pytest.mark.asyncio
async def test_pipeline_writes_document_meta_active_with_tag(
    db_session: object, tmp_path: Path
) -> None:
    import anthropic

    from particles import embeddings as ep
    from particles.corpus.deposit import deposit_file
    from particles.ingest.pipeline import extract_snapshot
    from particles.llm import set_client

    # ``label`` is the default mode; avoid reset_config() here — it disposes
    # the engine the db_session fixture holds.
    doc = tmp_path / "spec.txt"
    doc.write_text("A document with a real claim and a structural claim.")

    mock_content = MagicMock()
    mock_content.text = json.dumps(
        [
            {
                "content": "The euro is the currency of Germany.",
                "confidence_value": 0.95,
                "uncertainty_nature": "EPISTEMIC",
                "scope": "WORLD",
            },
            {
                "content": "Section 10.4 defines the exporter.",
                "confidence_value": 0.95,
                "uncertainty_nature": "EPISTEMIC",
                "scope": "DOCUMENT_META",
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
        assert all(p.status == Status.ACTIVE for p in particles)
        by_content = {p.content: p for p in particles}
        meta = by_content["Section 10.4 defines the exporter."]
        assert meta.properties == {SCOPE_KEY: SCOPE_DOCUMENT_META}
        assert by_content["The euro is the currency of Germany."].properties is None
    finally:
        ep.set_embedding_model(original_model)
        set_client(None)

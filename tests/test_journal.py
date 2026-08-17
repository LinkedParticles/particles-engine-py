"""Tests for particles/extraction/journal.py — the journal-aware extractor."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from particles.core.schema import AssertionModality, ParticleType
from particles.extraction.journal import (
    SOURCE_TYPE,
    JournalExtractor,
    _parse_journal_response,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def _journal_payload() -> str:
    return json.dumps(
        {
            "narrative_label": "A hard day the author got through anyway.",
            "claims": [
                {
                    "content": "The author needs to urinate.",
                    "subjects": [],
                    "confidence_value": 0.95,
                    "uncertainty_nature": "EPISTEMIC",
                    "assertion_modality": "EXPERIENTIAL",
                },
                {
                    "content": "The author is not good at Balatro.",
                    "subjects": ["Balatro"],
                    "confidence_value": 0.9,
                    "uncertainty_nature": "EPISTEMIC",
                    "assertion_modality": "EVALUATIVE",
                },
                {
                    "content": "The post was written on August 5, 2025.",
                    "subjects": [],
                    "confidence_value": 0.95,
                    "uncertainty_nature": "EPISTEMIC",
                    "assertion_modality": "FALSIFIABLE",
                },
            ],
        }
    )


class TestParseJournalResponse:
    def test_claims_carry_ordered_narrative_index(self) -> None:
        cands, notes = _parse_journal_response(_journal_payload())
        claims = [c for c in cands if c.particle_type != ParticleType.NARRATIVE]
        assert [c.narrative_index for c in claims] == [0, 1, 2]
        assert notes == []

    def test_one_narrative_candidate_with_label(self) -> None:
        cands, _ = _parse_journal_response(_journal_payload())
        narratives = [c for c in cands if c.particle_type == ParticleType.NARRATIVE]
        assert len(narratives) == 1
        assert narratives[0].content == "A hard day the author got through anyway."
        # The container is not itself a constituent.
        assert narratives[0].narrative_index is None

    def test_modality_classified_per_claim(self) -> None:
        cands, _ = _parse_journal_response(_journal_payload())
        by_content = {c.content: c.assertion_modality for c in cands}
        assert by_content["The author needs to urinate."] == AssertionModality.EXPERIENTIAL
        assert by_content["The author is not good at Balatro."] == AssertionModality.EVALUATIVE
        assert (
            by_content["The post was written on August 5, 2025."] == AssertionModality.FALSIFIABLE
        )

    def test_subjects_preserved(self) -> None:
        cands, _ = _parse_journal_response(_journal_payload())
        balatro = next(c for c in cands if "Balatro" in c.content)
        assert balatro.subjects == ["Balatro"]

    def test_code_fence_stripped(self) -> None:
        cands, _ = _parse_journal_response("```json\n" + _journal_payload() + "\n```")
        assert any(c.particle_type == ParticleType.NARRATIVE for c in cands)

    def test_missing_label_emits_no_narrative(self) -> None:
        payload = json.dumps(
            {
                "claims": [
                    {
                        "content": "The author felt tired.",
                        "confidence_value": 0.9,
                        "uncertainty_nature": "EPISTEMIC",
                        "assertion_modality": "EXPERIENTIAL",
                    }
                ]
            }
        )
        cands, notes = _parse_journal_response(payload)
        assert all(c.particle_type != ParticleType.NARRATIVE for c in cands)
        assert any("narrative_label" in n for n in notes)

    def test_label_without_claims_emits_nothing(self) -> None:
        cands, _ = _parse_journal_response(json.dumps({"narrative_label": "x", "claims": []}))
        assert cands == []

    def test_unknown_modality_defaults_falsifiable(self) -> None:
        payload = json.dumps(
            {
                "narrative_label": "L",
                "claims": [
                    {
                        "content": "A claim.",
                        "confidence_value": 0.8,
                        "uncertainty_nature": "EPISTEMIC",
                        "assertion_modality": "BOGUS",
                    }
                ],
            }
        )
        cands, notes = _parse_journal_response(payload)
        claim = next(c for c in cands if c.particle_type != ParticleType.NARRATIVE)
        assert claim.assertion_modality == AssertionModality.FALSIFIABLE
        assert any("assertion_modality" in n for n in notes)

    def test_confidence_clamped(self) -> None:
        payload = json.dumps(
            {
                "narrative_label": "L",
                "claims": [
                    {"content": "C.", "confidence_value": 5.0, "uncertainty_nature": "EPISTEMIC"}
                ],
            }
        )
        cands, _ = _parse_journal_response(payload)
        claim = next(c for c in cands if c.particle_type != ParticleType.NARRATIVE)
        assert claim.confidence_value == 1.0

    def test_invalid_json(self) -> None:
        cands, notes = _parse_journal_response("not json")
        assert cands == []
        assert any("JSON" in n for n in notes)

    def test_non_object_response(self) -> None:
        cands, notes = _parse_journal_response("[1, 2, 3]")
        assert cands == []
        assert any("object" in n for n in notes)


class TestSalvageResilience:
    """v0.1.1: a truncated or partially-malformed response recovers the
    complete claims instead of dropping the whole extraction (the 0-particle bug
    a dense bullet journal exposed)."""

    def test_truncated_response_recovers_complete_claims(self) -> None:
        # Valid prefix, then cut mid-third-claim (model hit the token limit).
        truncated = (
            '{"narrative_label": "A hard day.", "claims": [\n'
            '  {"content": "The author needs to urinate.", "confidence_value": 0.95,'
            ' "uncertainty_nature": "EPISTEMIC", "assertion_modality": "EXPERIENTIAL"},\n'
            '  {"content": "The author is impatient.", "confidence_value": 0.9,'
            ' "uncertainty_nature": "EPISTEMIC", "assertion_modality": "EXPERIENTIAL"},\n'
            '  {"content": "The post was written on Aug'
        )
        cands, notes = _parse_journal_response(truncated)
        claims = [c for c in cands if c.particle_type != ParticleType.NARRATIVE]
        narratives = [c for c in cands if c.particle_type == ParticleType.NARRATIVE]
        assert len(claims) == 2  # two complete; the truncated third is dropped
        assert [c.narrative_index for c in claims] == [0, 1]
        assert len(narratives) == 1
        assert narratives[0].content == "A hard day."  # label recovered from the top
        assert any("salvaged" in n for n in notes)

    def test_one_malformed_claim_skipped_rest_recovered(self) -> None:
        # The middle claim has an unescaped quote → whole-object json.loads fails.
        malformed = (
            '{"narrative_label": "L", "claims": [\n'
            '  {"content": "first claim", "confidence_value": 0.9,'
            ' "uncertainty_nature": "EPISTEMIC", "assertion_modality": "FALSIFIABLE"},\n'
            '  {"content": "he said "hi" loudly", "confidence_value": 0.9,'
            ' "uncertainty_nature": "EPISTEMIC", "assertion_modality": "FALSIFIABLE"},\n'
            '  {"content": "third claim", "confidence_value": 0.9,'
            ' "uncertainty_nature": "EPISTEMIC", "assertion_modality": "FALSIFIABLE"}\n'
            "]}"
        )
        cands, _ = _parse_journal_response(malformed)
        contents = [c.content for c in cands if c.particle_type != ParticleType.NARRATIVE]
        assert contents == ["first claim", "third claim"]  # bad middle claim dropped

    def test_unrecoverable_response_returns_error(self) -> None:
        cands, notes = _parse_journal_response('{"narrative_label": "x", "claims": [ {garbage')
        assert cands == []
        assert any("JSON parse error" in n for n in notes)


class TestJournalExtractorAccepts:
    def test_accepts_only_journal(self) -> None:
        ex = JournalExtractor()
        assert ex.accepts(SOURCE_TYPE) is True
        assert ex.accepts("WEB_PAGE") is False
        assert ex.accepts("LOCAL_FILE") is False

    def test_declines_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.config import get_config

        monkeypatch.setattr(get_config().journal_extractor, "enabled", False)
        assert JournalExtractor().accepts(SOURCE_TYPE) is False


class TestJournalExtract:
    @pytest.mark.asyncio
    async def test_extract_with_mock_client(self) -> None:
        import anthropic

        from particles.core.schema import Snapshot
        from particles.llm import set_client

        mock_content = MagicMock()
        mock_content.text = _journal_payload()
        mock_resp = MagicMock()
        mock_resp.content = [mock_content]
        mock_client = MagicMock(spec=anthropic.Anthropic)
        mock_client.messages = MagicMock()
        mock_client.messages.create = MagicMock(return_value=mock_resp)

        set_client(mock_client)
        try:
            snap = Snapshot(content_hash="a" * 64)
            result = await JournalExtractor().extract(
                snap, b"i have to pee.", source_type=SOURCE_TYPE
            )
        finally:
            set_client(None)

        narratives = [c for c in result.candidates if c.particle_type == ParticleType.NARRATIVE]
        claims = [c for c in result.candidates if c.particle_type == ParticleType.CLAIM]
        assert len(narratives) == 1
        assert len(claims) == 3

    @pytest.mark.asyncio
    async def test_empty_content_returns_no_candidates(self) -> None:
        from particles.core.schema import Snapshot

        result = await JournalExtractor().extract(
            Snapshot(content_hash="b" * 64), b"   ", source_type=SOURCE_TYPE
        )
        assert result.candidates == []

    @pytest.mark.asyncio
    async def test_over_length_entry_chunks_and_extracts_fully(self) -> None:
        """an entry over html_chunk_size is chunked and extracted in
        multiple passes — not truncated. Each chunk yields its claims plus a
        per-chunk NARRATIVE fragment; the Engine merge collapses the fragments
        downstream (pipeline), so the extractor itself returns one per chunk and
        no JOURNAL_TRUNCATED note."""
        import anthropic

        from particles.config import get_config
        from particles.core.schema import Snapshot
        from particles.llm import set_client

        # One long line, no paragraph breaks → hard cut at html_chunk_size →
        # 2 chunks. No session is passed, so carry-forward is skipped and both
        # chunks are cache misses (one LLM call each).
        big = b"x " * get_config().extraction.html_chunk_size
        mock_content = MagicMock()
        mock_content.text = _journal_payload()
        mock_resp = MagicMock()
        mock_resp.content = [mock_content]
        mock_client = MagicMock(spec=anthropic.Anthropic)
        mock_client.messages = MagicMock()
        mock_client.messages.create = MagicMock(return_value=mock_resp)

        set_client(mock_client)
        try:
            result = await JournalExtractor().extract(
                Snapshot(content_hash="d" * 64), big, source_type=SOURCE_TYPE
            )
        finally:
            set_client(None)

        narratives = [c for c in result.candidates if c.particle_type == ParticleType.NARRATIVE]
        claims = [c for c in result.candidates if c.particle_type != ParticleType.NARRATIVE]
        assert mock_client.messages.create.call_count == 2  # both chunks extracted
        assert len(narratives) == 2  # one per-chunk NARRATIVE fragment, pre-merge
        assert len(claims) == 6  # 2 chunks × 3 claims — nothing dropped
        assert not any("JOURNAL_TRUNCATED" in n for n in result.quality_notes)
        assert result.transient_error_count == 0


@pytest.mark.asyncio
async def test_pipeline_writes_narrative_graph(db_session: AsyncSession) -> None:
    """End-to-end: a JOURNAL deposit yields a NARRATIVE particle wired to its
    constituents by PART_OF, ordered by SEQUENCE_IN."""
    import anthropic

    from particles import embeddings as ep
    from particles.corpus.deposit import deposit_text
    from particles.ingest.pipeline import extract_snapshot
    from particles.llm import set_client
    from particles.operations.narrative import (
        get_narrative_constituents,
        get_narrative_sequence,
    )
    from particles.store.particle_store import get_active_particles_for_entry

    mock_content = MagicMock()
    mock_content.text = _journal_payload()
    mock_resp = MagicMock()
    mock_resp.content = [mock_content]
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages = MagicMock()
    mock_client.messages.create = MagicMock(return_value=mock_resp)

    mock_model = MagicMock()
    mock_model.encode = MagicMock(
        side_effect=lambda texts, **kw: [[0.1, 0.2, 0.3, 0.4] for _ in texts]
    )
    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    set_client(mock_client)
    try:
        entry_id, snapshot_id = await deposit_text(
            db_session, "i have to.", source_type=SOURCE_TYPE, deposited_by="test"
        )
        await db_session.commit()
        written = await extract_snapshot(db_session, entry_id, snapshot_id)
        await db_session.commit()
    finally:
        ep.set_embedding_model(original_model)
        set_client(None)

    # One NARRATIVE + three CLAIM particles persisted.
    actives = await get_active_particles_for_entry(db_session, entry_id)
    narratives = [p for p in actives if p.particle_type == ParticleType.NARRATIVE]
    assert len(narratives) == 1
    assert len(written) == 4
    narrative_id = narratives[0].id

    # PART_OF: every claim is a constituent of the narrative.
    constituents = await get_narrative_constituents(db_session, narrative_id)
    assert len(constituents) == 3
    assert narrative_id not in {c.id for c in constituents}

    # SEQUENCE_IN: constituents come back in document order.
    sequence = await get_narrative_sequence(db_session, narrative_id)
    assert [p.content for p in sequence] == [
        "The author needs to urinate.",
        "The author is not good at Balatro.",
        "The post was written on August 5, 2025.",
    ]


@pytest.mark.asyncio
async def test_no_narrative_when_extractor_emits_none(db_session: AsyncSession) -> None:
    """A run with no NARRATIVE candidate writes no PART_OF/SEQUENCE_IN edges —
    the post-pass is inert for non-journal extractions (regression guard)."""
    import anthropic

    from particles import embeddings as ep
    from particles.corpus.deposit import deposit_text
    from particles.ingest.pipeline import extract_snapshot
    from particles.llm import set_client
    from particles.store.particle_store import get_active_particles_for_entry
    from particles.store.relation_store import get_relations_for_particle

    # A plain general-extractor array response (no JOURNAL source_type).
    mock_content = MagicMock()
    mock_content.text = json.dumps(
        [
            {
                "content": "Paris is in France.",
                "confidence_value": 0.9,
                "uncertainty_nature": "EPISTEMIC",
            }
        ]
    )
    mock_resp = MagicMock()
    mock_resp.content = [mock_content]
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages = MagicMock()
    mock_client.messages.create = MagicMock(return_value=mock_resp)

    mock_model = MagicMock()
    mock_model.encode = MagicMock(
        side_effect=lambda texts, **kw: [[0.1, 0.2, 0.3, 0.4] for _ in texts]
    )
    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    set_client(mock_client)
    try:
        entry_id, snapshot_id = await deposit_text(
            db_session, "Paris is in France.", source_type="WEB_PAGE", deposited_by="test"
        )
        await db_session.commit()
        written = await extract_snapshot(db_session, entry_id, snapshot_id)
        await db_session.commit()
    finally:
        ep.set_embedding_model(original_model)
        set_client(None)

    actives = await get_active_particles_for_entry(db_session, entry_id)
    assert all(p.particle_type == ParticleType.CLAIM for p in actives)
    for p in written:
        assert await get_relations_for_particle(db_session, p.id) == []

"""Tests: Wikidata link confidence and subject mismatch linting."""

from __future__ import annotations

import pytest

from particles.core.schema import ExternalRef

# ---------------------------------------------------------------------------
# ExternalRef confidence field
# ---------------------------------------------------------------------------


class TestExternalRefConfidence:
    def test_default_confidence_is_one(self) -> None:
        ref = ExternalRef(namespace="wikidata", id="Q42")
        assert ref.confidence == 1.0

    def test_explicit_confidence(self) -> None:
        ref = ExternalRef(namespace="wikidata", id="Q49757", confidence=0.03)
        assert ref.confidence == pytest.approx(0.03)

    def test_serialises_confidence(self) -> None:
        ref = ExternalRef(namespace="wikidata", id="Q42", uri="https://...", confidence=0.7)
        d = ref.model_dump()
        assert d["confidence"] == pytest.approx(0.7)

    def test_deserialises_without_confidence_field(self) -> None:
        # Existing stored rows without a confidence key default to 1.0
        ref = ExternalRef.model_validate({"namespace": "numista", "id": "N1234"})
        assert ref.confidence == 1.0


# ---------------------------------------------------------------------------
# _wikidata_link_confidence
# ---------------------------------------------------------------------------


class TestWikidataLinkConfidence:
    def test_no_particle_content_returns_half(self) -> None:
        from particles.ingest.authorities.wikidata import _wikidata_link_confidence

        assert _wikidata_link_confidence("person who writes poetry", None) == pytest.approx(0.5)

    def test_empty_description_returns_half(self) -> None:
        from particles.ingest.authorities.wikidata import _wikidata_link_confidence

        assert _wikidata_link_confidence("", "POET Technologies stock price") == pytest.approx(0.5)

    def test_matching_content_returns_high_similarity(self) -> None:
        from unittest.mock import MagicMock, patch

        import numpy as np

        from particles.ingest.authorities.wikidata import _wikidata_link_confidence

        # Simulate embedding model returning similar vectors
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 0.0], [0.95, 0.1]])

        with patch(
            "particles.ingest.authorities.wikidata.get_embedding_model",
            return_value=mock_model,
        ):
            score = _wikidata_link_confidence("poetry writer", "poet laureate sonnet verse")

        assert score > 0.5

    def test_mismatched_content_returns_low_similarity(self) -> None:
        from unittest.mock import MagicMock, patch

        import numpy as np

        from particles.ingest.authorities.wikidata import _wikidata_link_confidence

        # Simulate near-orthogonal vectors (very different topics)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 0.0], [0.05, 0.999]])

        with patch(
            "particles.ingest.authorities.wikidata.get_embedding_model",
            return_value=mock_model,
        ):
            score = _wikidata_link_confidence(
                "person who writes poetry",
                "POET Technologies photonic chip semiconductor Marvell",
            )

        assert score < 0.5

    def test_model_none_returns_half(self) -> None:
        from unittest.mock import patch

        from particles.ingest.authorities.wikidata import _wikidata_link_confidence

        with patch(
            "particles.ingest.authorities.wikidata.get_embedding_model",
            return_value=None,
        ):
            score = _wikidata_link_confidence("person who writes poetry", "some content")

        assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Subject store: confidence round-trip
# ---------------------------------------------------------------------------


class TestSubjectStoreConfidenceRoundtrip:
    @pytest.mark.asyncio
    async def test_confidence_persisted_and_loaded(self, db_session: object) -> None:
        from datetime import UTC, datetime

        from particles.core.schema import Subject
        from particles.store.subject_store import get_subject, insert_subject

        subject = Subject(
            canonical_name="Test Subject",
            external_ids=[ExternalRef(namespace="wikidata", id="Q99999", confidence=0.12)],
            created_at=datetime.now(UTC),
            asserted_by="test",
        )
        await insert_subject(db_session, subject)  # type: ignore[arg-type]

        loaded = await get_subject(db_session, subject.id)  # type: ignore[arg-type]
        assert loaded is not None
        assert len(loaded.external_ids) == 1
        assert loaded.external_ids[0].confidence == pytest.approx(0.12)

    @pytest.mark.asyncio
    async def test_legacy_row_defaults_to_one(self, db_session: object) -> None:
        """Rows stored before (no confidence key) deserialise to 1.0."""
        import json

        from particles.store.subject_store import SubjectRow

        row = SubjectRow(
            id="test-legacy",
            canonical_name="Legacy Subject",
            aliases_json="[]",
            external_ids_json=json.dumps(
                [{"namespace": "wikidata", "id": "Q1234", "uri": "https://..."}]
            ),
            created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            asserted_by="test",
        )
        subject = row.to_model()
        assert subject.external_ids[0].confidence == 1.0


# ---------------------------------------------------------------------------
# Lint check L-SEM-03
# ---------------------------------------------------------------------------


class TestLintWikidataLinkMismatch:
    @pytest.mark.asyncio
    async def test_low_confidence_flagged(self, db_session: object) -> None:
        from datetime import UTC, datetime

        from particles.core.schema import Subject
        from particles.operations.lint import _check_wikidata_link_confidence
        from particles.store.subject_store import insert_subject

        subject = Subject(
            canonical_name="poet",
            description="person who writes poetry",
            external_ids=[ExternalRef(namespace="wikidata", id="Q49757", confidence=0.03)],
            created_at=datetime.now(UTC),
            asserted_by="test",
        )
        await insert_subject(db_session, subject)  # type: ignore[arg-type]

        findings = await _check_wikidata_link_confidence(db_session)  # type: ignore[arg-type]
        assert len(findings) == 1
        assert findings[0].finding_type == "WIKIDATA_LINK_MISMATCH"
        assert findings[0].severity == "WARNING"
        assert "Q49757" in findings[0].detail
        assert findings[0].subject_id == subject.id

    @pytest.mark.asyncio
    async def test_detail_provides_complete_runnable_commands(self, db_session: object) -> None:
        """The detail string is rendered into a Markdown callout body in
        synthesised articles and Obsidian notes. It must (a) contain no
        angle-bracketed text — Obsidian parses ``<foo>`` as an opening
        HTML tag and breaks rendering of everything downstream — and
        (b) suggest only fully-specified, copy-pasteable commands. The
        subject id and the offending ``wikidata:QID`` are both known at
        lint time, so the operator never substitutes a placeholder by
        hand (no ``TARGET_ID``)."""
        import re
        from datetime import UTC, datetime

        from particles.core.schema import Subject
        from particles.operations.lint import _check_wikidata_link_confidence
        from particles.store.subject_store import insert_subject

        subject = Subject(
            canonical_name="poet",
            description="person who writes poetry",
            external_ids=[ExternalRef(namespace="wikidata", id="Q49757", confidence=0.03)],
            created_at=datetime.now(UTC),
            asserted_by="test",
        )
        await insert_subject(db_session, subject)  # type: ignore[arg-type]

        findings = await _check_wikidata_link_confidence(db_session)  # type: ignore[arg-type]
        detail = findings[0].detail
        short_id = subject.id[:8]

        # No HTML-tag-shaped text anywhere in the detail.
        assert not re.search(r"<[a-z][a-z0-9-]*>", detail), (
            f"Detail contains HTML-tag-shaped text that would break Markdown rendering: {detail!r}"
        )
        # No manual-substitution placeholder — the command is runnable as-is.
        assert "TARGET_ID" not in detail
        # Both resolutions present, backticked, and fully specified with the
        # real subject id + wikidata QID (no placeholder to fill in).
        assert f"`particles subjects unlink {short_id} wikidata:Q49757`" in detail
        assert f"`particles subjects confirm {short_id} wikidata:Q49757`" in detail

    @pytest.mark.asyncio
    async def test_high_confidence_not_flagged(self, db_session: object) -> None:
        from datetime import UTC, datetime

        from particles.core.schema import Subject
        from particles.operations.lint import _check_wikidata_link_confidence
        from particles.store.subject_store import insert_subject

        subject = Subject(
            canonical_name="aluminium",
            external_ids=[ExternalRef(namespace="wikidata", id="Q663", confidence=0.92)],
            created_at=datetime.now(UTC),
            asserted_by="test",
        )
        await insert_subject(db_session, subject)  # type: ignore[arg-type]

        findings = await _check_wikidata_link_confidence(db_session)  # type: ignore[arg-type]
        assert findings == []

    @pytest.mark.asyncio
    async def test_non_wikidata_ref_not_flagged(self, db_session: object) -> None:
        from datetime import UTC, datetime

        from particles.core.schema import Subject
        from particles.operations.lint import _check_wikidata_link_confidence
        from particles.store.subject_store import insert_subject

        subject = Subject(
            canonical_name="Aluminium coin",
            external_ids=[ExternalRef(namespace="numista", id="N1234", confidence=0.05)],
            created_at=datetime.now(UTC),
            asserted_by="test",
        )
        await insert_subject(db_session, subject)  # type: ignore[arg-type]

        findings = await _check_wikidata_link_confidence(db_session)  # type: ignore[arg-type]
        # numista refs are not checked — only wikidata
        assert findings == []


class TestUnscoredDisclosure:
    """an unavailable scorer must not look like a computed 0.5."""

    def test_missing_model_warns_and_names_the_consequence(
        self, no_embedding_model: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        import particles.ingest.authorities.wikidata as wd

        wd._unscored_warning_emitted = False
        with caplog.at_level("WARNING", logger="particles.ingest.authorities.wikidata"):
            score = wd._wikidata_link_confidence("person who writes poetry", "some content")

        # The outcome is deliberately unchanged: an unscoreable
        # link still attaches, and flipping that would drop every Wikidata link for a
        # user without the encoder. Only the silence is fixed.
        assert score == pytest.approx(0.5)
        text = caplog.text
        assert "cannot be scored" in text
        # It must name what is now impossible, not just that scoring failed.
        assert "abstention cannot fire" in text
        assert "L-SEM-03" in text

    def test_warning_is_emitted_once_per_process(
        self, no_embedding_model: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A resolution pass scores once per candidate QID; warning each time buries it."""
        import particles.ingest.authorities.wikidata as wd

        wd._unscored_warning_emitted = False
        with caplog.at_level("WARNING", logger="particles.ingest.authorities.wikidata"):
            for _ in range(5):
                wd._wikidata_link_confidence("a description", "some content")

        assert caplog.text.count("cannot be scored") == 1

    def test_genuinely_unscoreable_input_stays_silent(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Empty content is a different case — the pair is unscoreable, not the scorer."""
        import particles.ingest.authorities.wikidata as wd

        wd._unscored_warning_emitted = False
        with caplog.at_level("WARNING", logger="particles.ingest.authorities.wikidata"):
            assert wd._wikidata_link_confidence("a description", None) == pytest.approx(0.5)

        assert "cannot be scored" not in caplog.text

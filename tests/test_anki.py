"""Tests for the Anki flashcard exporter."""

from __future__ import annotations

import pytest

from particles.core.schema import Confidence, Particle, UncertaintyNature
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.exporters.anki import _cards_for_particle, _plain_front_back


def _make_particle(
    content: str = "Test content.",
    properties: dict | None = None,
    confidence: float = 0.9,
    tags: list[str] | None = None,
) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(
            value=confidence, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        status=Status.ACTIVE,
        properties=properties,
        tags=tags,
    )


class TestCardsForParticle:
    def test_plain_particle_produces_one_card(self) -> None:
        p = _make_particle(content="The weight is 1.1 g.")
        cards = _cards_for_particle("My Subject", p, max_cards=None)
        assert len(cards) == 1

    def test_plain_front_back_colon_split(self) -> None:
        front, back = _plain_front_back("German Democratic Republic population: 16111000")
        assert front == "German Democratic Republic population?"
        assert back == "16111000"

    def test_plain_front_back_no_colon_uses_content_as_front(self) -> None:
        front, back = _plain_front_back("The coin was struck in aluminum.")
        assert front == "The coin was struck in aluminum."
        assert back == "The coin was struck in aluminum."

    def test_plain_front_back_truncates_long_content(self) -> None:
        long = "x" * 200
        front, back = _plain_front_back(long)
        assert front.endswith("…")
        assert len(front) <= 150
        assert back == long

    def test_structured_particle_produces_one_card_per_property(self) -> None:
        p = _make_particle(
            properties={
                "nmo:hasIssuer": "Germany",
                "nmo:hasWeight": "1.1 g",
                "nmo:hasMaterial": "Aluminium",
            }
        )
        cards = _cards_for_particle("5 Pfennigs", p, max_cards=None)
        assert len(cards) == 3
        fronts = [c[0] for c in cards]
        backs = [c[1] for c in cards]
        assert "5 Pfennigs — issuer?" in fronts
        assert "5 Pfennigs — weight?" in fronts
        assert "5 Pfennigs — composition?" in fronts
        assert "Germany" in backs
        assert "1.1 g" in backs
        assert "Aluminium" in backs

    def test_none_property_values_are_skipped(self) -> None:
        p = _make_particle(properties={"nmo:hasIssuer": "Germany", "nmo:hasWeight": None})
        cards = _cards_for_particle("Coin", p, max_cards=None)
        assert len(cards) == 1
        assert cards[0][0] == "Coin — issuer?"

    def test_list_property_value_joined_with_comma(self) -> None:
        p = _make_particle(properties={"nmo:hasMaterial": ["Copper", "Zinc"]})
        cards = _cards_for_particle("Coin", p, max_cards=None)
        assert len(cards) == 1
        assert cards[0][1] == "Copper, Zinc"

    def test_unknown_property_key_uses_local_name(self) -> None:
        p = _make_particle(properties={"custom:someField": "value"})
        cards = _cards_for_particle("Thing", p, max_cards=None)
        assert cards[0][0] == "Thing — someField?"

    def test_max_cards_truncates(self) -> None:
        p = _make_particle(
            properties={
                "nmo:hasIssuer": "A",
                "nmo:hasWeight": "B",
                "nmo:hasMaterial": "C",
            }
        )
        cards = _cards_for_particle("X", p, max_cards=2)
        assert len(cards) == 2


class TestAnkiExporter:
    @pytest.mark.asyncio
    async def test_empty_db_writes_header_only(self, db_session: object, tmp_path: object) -> None:
        from pathlib import Path

        from particles.exporters.anki import AnkiExporter

        out: Path = tmp_path / "deck.txt"  # type: ignore[assignment]
        summary = await AnkiExporter().export(
            db_session,  # type: ignore[arg-type]
            out,
        )
        assert summary.cards_written == 0
        assert summary.decks == 0
        text = out.read_text()
        assert "#separator:tab" in text
        assert "#notetype:Basic" in text

    @pytest.mark.asyncio
    async def test_output_path_none_raises(self, db_session: object) -> None:
        from particles.exporters.anki import AnkiExporter

        with pytest.raises(ValueError, match="output file path"):
            await AnkiExporter().export(db_session, None)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_plain_particle_card_written(self, db_session: object, tmp_path: object) -> None:
        from pathlib import Path

        from particles.core.schema import Subject
        from particles.exporters.anki import AnkiExporter
        from particles.store.particle_store import insert_particle
        from particles.store.subject_store import insert_subject, link_particle_to_subjects

        p = _make_particle(content="The weight is 1.1 g.")
        await insert_particle(db_session, p)  # type: ignore[arg-type]

        s = Subject(canonical_name="5 Pfennigs GDR", asserted_by="test")
        await insert_subject(db_session, s)  # type: ignore[arg-type]
        await link_particle_to_subjects(db_session, p.id, [s.id])  # type: ignore[arg-type]

        out: Path = tmp_path / "deck.txt"  # type: ignore[assignment]
        summary = await AnkiExporter().export(db_session, out)  # type: ignore[arg-type]
        assert summary.cards_written == 1
        assert summary.decks == 1
        text = out.read_text()
        assert "#deck:Particles::5 Pfennigs GDR" in text
        assert "The weight is 1.1 g." in text

    @pytest.mark.asyncio
    async def test_min_confidence_filters_particles(
        self, db_session: object, tmp_path: object
    ) -> None:
        from pathlib import Path

        from particles.exporters.anki import AnkiExporter
        from particles.store.particle_store import insert_particle

        high = _make_particle(content="High confidence.", confidence=0.9)
        low = _make_particle(content="Low confidence.", confidence=0.3)
        await insert_particle(db_session, high)  # type: ignore[arg-type]
        await insert_particle(db_session, low)  # type: ignore[arg-type]

        out: Path = tmp_path / "deck.txt"  # type: ignore[assignment]
        summary = await AnkiExporter().export(
            db_session,
            out,
            min_particle_confidence=0.8,  # type: ignore[arg-type]
        )
        text = out.read_text()
        assert "High confidence." in text
        assert "Low confidence." not in text
        assert summary.cards_written == 1

    @pytest.mark.asyncio
    async def test_custom_deck_name(self, db_session: object, tmp_path: object) -> None:
        from pathlib import Path

        from particles.exporters.anki import AnkiExporter
        from particles.store.particle_store import insert_particle

        p = _make_particle(content="Fact.")
        await insert_particle(db_session, p)  # type: ignore[arg-type]

        out: Path = tmp_path / "deck.txt"  # type: ignore[assignment]
        await AnkiExporter().export(
            db_session,
            out,
            deck_name="Numismatics",  # type: ignore[arg-type]
        )
        text = out.read_text()
        assert "#deck:Numismatics::General" in text

    @pytest.mark.asyncio
    async def test_tabs_and_newlines_escaped(self, db_session: object, tmp_path: object) -> None:
        from pathlib import Path

        from particles.exporters.anki import AnkiExporter
        from particles.store.particle_store import insert_particle

        p = _make_particle(content="Line one.\nLine two.\tWith tab.")
        await insert_particle(db_session, p)  # type: ignore[arg-type]

        out: Path = tmp_path / "deck.txt"  # type: ignore[assignment]
        await AnkiExporter().export(db_session, out)  # type: ignore[arg-type]
        text = out.read_text()
        assert "\t\t" not in text  # no double-tab separators from escaped content
        assert "Line one.<br>Line two. With tab." in text


# ---------------------------------------------------------------------------
# taxonomy tags propagate to Anki's native tag column
# ---------------------------------------------------------------------------


class TestAnkiTaxonomyTags:
    @pytest.mark.asyncio
    async def test_tags_column_header_and_directive_always_present(
        self, db_session: object, tmp_path: object
    ) -> None:
        from pathlib import Path

        from particles.exporters.anki import AnkiExporter
        from particles.store.particle_store import insert_particle

        p = _make_particle(content="A claim.")
        await insert_particle(db_session, p)  # type: ignore[arg-type]
        out: Path = tmp_path / "deck.txt"  # type: ignore[assignment]
        await AnkiExporter().export(db_session, out)  # type: ignore[arg-type]
        text = out.read_text()
        assert "#columns:Front\tBack\tTags" in text
        assert "#tags column:3" in text

    @pytest.mark.asyncio
    async def test_taxonomy_tag_emitted_in_third_column(
        self, db_session: object, tmp_path: object
    ) -> None:
        from pathlib import Path

        from particles.exporters.anki import AnkiExporter
        from particles.store.particle_store import insert_particle

        p = _make_particle(content="Tagged claim.", tags=["ml/optimizers", "cold war"])
        await insert_particle(db_session, p)  # type: ignore[arg-type]
        out: Path = tmp_path / "deck.txt"  # type: ignore[assignment]
        await AnkiExporter().export(db_session, out)  # type: ignore[arg-type]
        text = out.read_text()
        # Internal whitespace collapses to ``_`` (Anki tags can't contain
        # spaces); the ``/`` hierarchy separator survives.
        card_line = next(ln for ln in text.splitlines() if ln.startswith("Tagged claim."))
        assert card_line.endswith("\tml/optimizers cold_war")

    @pytest.mark.asyncio
    async def test_untagged_particle_has_empty_tag_field(
        self, db_session: object, tmp_path: object
    ) -> None:
        from pathlib import Path

        from particles.exporters.anki import AnkiExporter
        from particles.store.particle_store import insert_particle

        p = _make_particle(content="Untagged claim.")
        await insert_particle(db_session, p)  # type: ignore[arg-type]
        out: Path = tmp_path / "deck.txt"  # type: ignore[assignment]
        await AnkiExporter().export(db_session, out)  # type: ignore[arg-type]
        text = out.read_text()
        card_line = next(ln for ln in text.splitlines() if ln.startswith("Untagged claim."))
        # Three columns: front, back, empty tags → trailing tab, no double-tab.
        assert card_line.endswith("\t")
        assert "\t\t" not in card_line


# ---------------------------------------------------------------------------
# Anki audit-trail header
# ---------------------------------------------------------------------------


class TestAnkiAdr0065:
    """Anki-specific surface.

    The cross-exporter contract behavior (the filter itself) is in
    ``tests/test_exporter_quality_threshold.py``. This class pins the
    Anki-only deck-header audit trail.
    """

    @pytest.mark.asyncio
    async def test_deck_header_records_threshold(
        self, db_session: object, tmp_path: object
    ) -> None:
        """anki: the deck file always carries the threshold."""
        from pathlib import Path

        from particles.exporters.anki import AnkiExporter
        from particles.store.particle_store import insert_particle

        p = _make_particle(content="Whatever.", confidence=0.9)
        await insert_particle(db_session, p)  # type: ignore[arg-type]
        out: Path = tmp_path / "deck.txt"  # type: ignore[assignment]
        await AnkiExporter().export(
            db_session,  # type: ignore[arg-type]
            out,
            min_particle_confidence=0.42,  # type: ignore[arg-type]
        )
        assert "#comment:min_particle_confidence=0.42" in out.read_text()

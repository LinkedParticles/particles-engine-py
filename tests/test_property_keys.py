"""The `properties`-key shape on persisted particles.

Two surfaces, one predicate: `conformance.validator` asks the question of fresh
extractor output over a fixture, `L-STR-12` asks it of the store. The store is
the only one of the two that sees particles arriving by interchange import, from
a third-party extractor, or from before a convention change — which is why the
bare `polarity` / `scope` keys went unnoticed until 1.109.0.
"""

from __future__ import annotations

import pytest

from particles.core.schema import Confidence, Particle, UncertaintyNature
from particles.extraction.property_keys import bare_properties_keys
from particles.operations.lint.coverage import _check_bare_properties_keys
from particles.store.particle_store import insert_particle


def _particle(properties: dict[str, object] | None) -> Particle:
    return Particle(
        content="Sirius has spectral type A1V.",
        confidence=Confidence(value=0.8),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="general-extractor",
        properties=properties,
    )


class TestBarePropertiesKeys:
    @pytest.mark.parametrize(
        "properties,expected",
        [
            (None, []),
            ({}, []),
            ({"nmo:hasIssuer": "x", "extraction:scope": "WORLD"}, []),
            ({"polarity": "DECLINED"}, ["polarity"]),
            ({"nmo:hasIssuer": "x", "scope": "WORLD", "polarity": "y"}, ["scope", "polarity"]),
            # An unregistered prefix is a governance question with a documented
            # answer (add the row); a bare key is attributable to no
            # namespace at all. Only the latter is what this predicate catches.
            ({"madeup:key": 1}, []),
        ],
    )
    def test_returns_only_keys_without_a_colon(
        self, properties: dict[str, object] | None, expected: list[str]
    ) -> None:
        assert bare_properties_keys(properties) == expected

    def test_preserves_producer_order(self) -> None:
        assert bare_properties_keys({"b": 1, "a": 2}) == ["b", "a"]


class TestLintRule:
    @pytest.mark.asyncio
    async def test_flags_a_persisted_bare_key(self, db_session) -> None:  # type: ignore[no-untyped-def]
        await insert_particle(db_session, _particle({"nmo:hasIssuer": "x", "polarity": "y"}))

        findings = await _check_bare_properties_keys(db_session)
        assert [f.finding_type for f in findings] == ["BARE_PROPERTIES_KEY"]
        assert findings[0].severity == "WARNING"
        assert "'polarity'" in (findings[0].detail or "")
        # The prefixed sibling is not part of the complaint.
        assert "nmo:hasIssuer" not in (findings[0].detail or "")
        # Read-only: the remedy is migration or re-extraction, never a transition.
        assert "reindex" in (findings[0].recommended_action or "")

    @pytest.mark.asyncio
    async def test_does_not_flag_prefixed_or_absent_properties(self, db_session) -> None:  # type: ignore[no-untyped-def]
        await insert_particle(db_session, _particle({"extraction:scope": "WORLD"}))
        await insert_particle(db_session, _particle(None))

        assert await _check_bare_properties_keys(db_session) == []

    @pytest.mark.asyncio
    async def test_the_rule_runs_inside_run_lint(self, db_session) -> None:  # type: ignore[no-untyped-def]
        # The predicate being correct is worth nothing if the orchestrator never
        # calls it — the exact gap now closed.
        from particles.operations.lint.orchestrator import run_lint

        await insert_particle(db_session, _particle({"polarity": "y"}))

        report = await run_lint(db_session, semantic=False)
        assert any(f.finding_type == "BARE_PROPERTIES_KEY" for f in report.findings)

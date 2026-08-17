"""Tests for Extension A: extractor registry and trust model."""

from __future__ import annotations

from pathlib import Path

import pytest

from particles.benchmark.loader import discover_suites
from particles.core.schema import ApplicabilityClause, ExtractorRecord
from particles.extraction.registry import (
    ExtractorPlugin,
    get_extractors,
    is_must_not,
    select_extractor,
    selects,
)
from particles.ingest.importers.registry import ensure_extractor_records

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestApplicabilityClause:
    def test_valid_clause(self) -> None:
        c = ApplicabilityClause(
            keyword="MUST",
            domain_uri="http://www.wikidata.org/entity/Q631286",
            domain_label="numismatics",
            source_types=["NUMISTA_API_COIN"],
        )
        assert c.keyword == "MUST"
        assert "NUMISTA_API_COIN" in c.source_types

    def test_must_not_clause(self) -> None:
        c = ApplicabilityClause(
            keyword="MUST_NOT",
            domain_uri="http://www.wikidata.org/entity/Q202833",
            domain_label="social media",
            source_types=["REDDIT_POST"],
        )
        assert c.keyword == "MUST_NOT"

    def test_extractor_record_defaults(self) -> None:
        r = ExtractorRecord(
            extractor_id="test-extractor",
            name="test-extractor",
            version="0.1.0",
        )
        assert r.trust_weight == 0.7
        assert r.registered_by == "anthropic/particles-sdk"


class TestNumismaticsDomainQID:
    """The four numismatics extractors must point at the correct Wikidata
    entity for numismatics — Q631286 ("study of currencies, coins and paper
    money") — not Q8148 ("industry"), which was declared in error. The
    ``domain_uri`` is the interop-stable domain identity in ApplicabilityClause
    (§14.1); a wrong QID misaligns cross-implementation domain matching.
    """

    #: Wikidata "numismatics" — verified via Special:EntityData/Q631286.json.
    NUMISMATICS_URI = "http://www.wikidata.org/entity/Q631286"

    def test_all_numismatics_extractors_declare_the_numismatics_entity(self) -> None:
        from particles.extraction.nomisma import NomismaExtractor
        from particles.extraction.numista import (
            NumistaCoinExtractor,
            NumistaIssuerExtractor,
            NumistaListingExtractor,
        )

        for plugin in (
            NomismaExtractor(),
            NumistaCoinExtractor(),
            NumistaIssuerExtractor(),
            NumistaListingExtractor(),
        ):
            clauses = plugin.APPLICABILITY  # type: ignore[attr-defined]
            assert clauses, f"{plugin.EXTRACTOR_ID} declares no applicability"
            for clause in clauses:
                # The label the QID resolves to on Wikidata is numismatics —
                # keep the declared label and the corrected entity in lockstep.
                assert clause.domain_label == "numismatics"
                assert clause.domain_uri == self.NUMISMATICS_URI, (
                    f"{plugin.EXTRACTOR_ID} points at {clause.domain_uri}; "
                    f"expected the numismatics entity {self.NUMISMATICS_URI}"
                )


# ---------------------------------------------------------------------------
# is_must_not
# ---------------------------------------------------------------------------


class TestIsMustNot:
    def _make_plugin(self, clauses: list[ApplicabilityClause]) -> object:
        class FakePlugin:
            EXTRACTOR_ID = "fake"
            EXTRACTOR_VERSION = "0.1.0"
            APPLICABILITY = clauses

        return FakePlugin()

    def test_must_not_blocks_source_type(self) -> None:
        plugin = self._make_plugin(
            [
                ApplicabilityClause(
                    keyword="MUST_NOT",
                    domain_uri="http://www.wikidata.org/entity/Q202833",
                    domain_label="social media",
                    source_types=["REDDIT_POST"],
                )
            ]
        )
        assert is_must_not(plugin, "REDDIT_POST") is True  # type: ignore[arg-type]
        assert is_must_not(plugin, "WEB_PAGE") is False  # type: ignore[arg-type]

    def test_no_applicability_never_must_not(self) -> None:
        class NoApplicability:
            EXTRACTOR_ID = "x"
            EXTRACTOR_VERSION = "0.1.0"

        assert is_must_not(NoApplicability(), "REDDIT_POST") is False  # type: ignore[arg-type]

    def test_must_clause_does_not_block(self) -> None:
        plugin = self._make_plugin(
            [
                ApplicabilityClause(
                    keyword="MUST",
                    domain_uri="http://www.wikidata.org/entity/Q631286",
                    domain_label="numismatics",
                    source_types=["NUMISTA_API_COIN"],
                )
            ]
        )
        assert is_must_not(plugin, "NUMISTA_API_COIN") is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# select_extractor / selects — routing precedence
# ---------------------------------------------------------------------------


class TestSelectExtractor:
    def test_domain_type_routes_to_its_domain_extractor(self) -> None:
        assert select_extractor("NUMISTA_API_COIN").EXTRACTOR_ID == "numista-coin-extractor"
        assert select_extractor("HACKERNEWS_THREAD").EXTRACTOR_ID == "hackernews-extractor"
        assert select_extractor("JOURNAL").EXTRACTOR_ID == "journal-extractor"

    def test_unclaimed_type_routes_to_the_fallback(self) -> None:
        assert select_extractor("WEB_PAGE").EXTRACTOR_ID == "general-extractor"
        assert select_extractor("PDF").EXTRACTOR_ID == "general-extractor"
        # An invented type still routes: the fallback accepts everything.
        assert select_extractor("NO_SUCH_SOURCE_TYPE").EXTRACTOR_ID == "general-extractor"

    def test_selection_matches_the_registry_ladder(self) -> None:
        """The helper must agree with the ladder every caller used to inline."""
        for source_type in ("REDDIT_POST", "RDF_GRAPH", "WEB_PAGE", "PYTHON_SOURCE"):
            expected = next(
                p
                for p in get_extractors()
                if not is_must_not(p, source_type) and p.accepts(source_type)
            )
            assert select_extractor(source_type).EXTRACTOR_ID == expected.EXTRACTOR_ID


class TestSelects:
    """The auto-filter predicate."""

    def _by_id(self, extractor_id: str) -> ExtractorPlugin:
        return next(e for e in get_extractors() if extractor_id == e.EXTRACTOR_ID)

    def test_fallback_does_not_select_a_domain_suites_source_type(self) -> None:
        """The whole point: accepts() says yes, routing precedence says no."""
        general = self._by_id("general-extractor")
        assert general.accepts("NUMISTA_API_COIN") is True
        assert selects(general, ["NUMISTA_API_COIN"]) is False

    def test_fallback_selects_types_no_domain_extractor_claims(self) -> None:
        general = self._by_id("general-extractor")
        assert selects(general, ["WEB_PAGE"]) is True

    def test_domain_extractor_selects_its_own_source_type(self) -> None:
        assert selects(self._by_id("reddit-extractor"), ["REDDIT_POST"]) is True

    def test_any_of_the_declared_types_is_enough(self) -> None:
        general = self._by_id("general-extractor")
        assert selects(general, ["NUMISTA_API_COIN", "WEB_PAGE"]) is True

    def test_empty_source_types_selects_nothing(self) -> None:
        assert selects(self._by_id("general-extractor"), []) is False


class TestShippedSuitesRouteToOneExtractor:
    """Regression guard on the bug fixed, over the suites in tree."""

    def test_each_suite_auto_matches_exactly_one_extractor(self) -> None:
        suites = list(discover_suites(Path("tests/benchmark/suites")))
        assert suites, "seed suites should be discoverable from the repo root"
        for suite in suites:
            matching = [e.EXTRACTOR_ID for e in get_extractors() if selects(e, suite.source_types)]
            assert len(matching) == 1, f"{suite.suite_id} auto-matched {matching}"

    def test_general_extractor_auto_matches_only_the_prose_suite(self) -> None:
        general = next(e for e in get_extractors() if e.EXTRACTOR_ID == "general-extractor")
        matched = [
            s.suite_id
            for s in discover_suites(Path("tests/benchmark/suites"))
            if selects(general, s.source_types)
        ]
        assert matched == ["prose-article-seed-001"]

    def test_each_calibration_suite_auto_matches_exactly_one_extractor(self) -> None:
        """the routing rule covers the calibration family too.

        The same precedence, a different directory. Without this, a
        calibration suite could silently be handed to the fallback extractor
        the way the §13.3 suites were before.
        """
        suites = list(discover_suites(Path("tests/benchmark/calibration")))
        assert suites, "calibration suites should be discoverable from the repo root"
        for suite in suites:
            matching = [e.EXTRACTOR_ID for e in get_extractors() if selects(e, suite.source_types)]
            assert len(matching) == 1, f"{suite.suite_id} auto-matched {matching}"

    def test_calibration_and_benchmark_families_stay_separate(self) -> None:
        """Neither directory may discover the other's suites.

        The whole mechanical content of the separation. If a §13.3 suite ever
        appeared under `calibration/`, its total gold coverage would fit no
        temperature; if a calibration suite appeared under `suites/`, its
        deliberately-partial gold set would score its extractor down on
        precision.
        """
        benchmark_ids = {s.suite_id for s in discover_suites(Path("tests/benchmark/suites"))}
        calibration_ids = {s.suite_id for s in discover_suites(Path("tests/benchmark/calibration"))}
        assert benchmark_ids and calibration_ids
        assert not (benchmark_ids & calibration_ids)


# ---------------------------------------------------------------------------
# ensure_extractor_records (DB integration)
# ---------------------------------------------------------------------------


class TestEnsureExtractorRecords:
    @pytest.mark.asyncio
    async def test_seeds_all_built_in_extractors(self, db_session: object) -> None:
        from particles.store.extractor_store import get_all_records

        wrote = await ensure_extractor_records(db_session)  # type: ignore[arg-type]
        assert wrote > 0

        records = await get_all_records(db_session)  # type: ignore[arg-type]
        ids = {r.extractor_id for r in records}
        assert "general-extractor" in ids
        assert "numista-coin-extractor" in ids
        assert "wikidata-extractor" in ids
        assert "nomisma-extractor" in ids

    @pytest.mark.asyncio
    async def test_idempotent_no_rewrite(self, db_session: object) -> None:
        await ensure_extractor_records(db_session)  # type: ignore[arg-type]
        wrote2 = await ensure_extractor_records(db_session)  # type: ignore[arg-type]
        assert wrote2 == 0

    @pytest.mark.asyncio
    async def test_trust_weight_preserved_on_re_seed(self, db_session: object) -> None:
        from particles.store.extractor_store import get_all_records, set_trust_weight

        await ensure_extractor_records(db_session)  # type: ignore[arg-type]
        await set_trust_weight(db_session, "general-extractor", 0.5)  # type: ignore[arg-type]

        # Re-seed — should NOT overwrite operator value
        await ensure_extractor_records(db_session)  # type: ignore[arg-type]
        records = await get_all_records(db_session)  # type: ignore[arg-type]
        general = next(r for r in records if r.extractor_id == "general-extractor")
        assert general.trust_weight == 0.5

    @pytest.mark.asyncio
    async def test_default_trust_weights(self, db_session: object) -> None:
        await ensure_extractor_records(db_session)  # type: ignore[arg-type]
        from particles.store.extractor_store import get_all_records

        records = {r.extractor_id: r for r in await get_all_records(db_session)}  # type: ignore[arg-type]
        assert records["general-extractor"].trust_weight == pytest.approx(0.70)
        assert records["numista-coin-extractor"].trust_weight == pytest.approx(0.90)
        assert records["nomisma-extractor"].trust_weight == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Trust weight cache and query integration
# ---------------------------------------------------------------------------


class TestTrustWeightCache:
    def test_returns_default_when_cache_not_loaded(self) -> None:
        from particles.store.extractor_store import get_cached_trust_weight, invalidate_trust_cache

        invalidate_trust_cache()
        # Default is 1.0 when cache not loaded (no penalty applied)
        assert get_cached_trust_weight("unknown-extractor") == 1.0

    def test_populated_cache_returns_correct_weight(self) -> None:
        from particles.store.extractor_store import (
            get_cached_trust_weight,
            invalidate_trust_cache,
            populate_trust_cache,
        )

        invalidate_trust_cache()
        populate_trust_cache({"general-extractor": 0.70, "nomisma-extractor": 0.95})
        assert get_cached_trust_weight("general-extractor") == pytest.approx(0.70)
        assert get_cached_trust_weight("nomisma-extractor") == pytest.approx(0.95)
        assert get_cached_trust_weight("unknown") == 1.0  # default for unknown

    def test_reset_config_invalidates_the_cache(self) -> None:
        """The cache is a snapshot of one store; ``reset_config()`` drops it.

        Without the reset hook this process-global outlives the store it was
        read from, and every later effective-confidence computation on a path
        that does not re-warm it (``score_effective_confidence`` with
        ``populate_cache=False``) is silently rescaled by the stale weight.
        In the test suite that showed up as an xdist worker-assignment flake:
        this class populating ``general-extractor: 0.70`` made three
        ``test_utility_policy.py`` tests score 0.99 × 0.70 = 0.693.
        """
        from particles.config import reset_config
        from particles.store.extractor_store import (
            get_cached_trust_weight,
            populate_trust_cache,
        )

        populate_trust_cache({"general-extractor": 0.70})
        assert get_cached_trust_weight("general-extractor") == pytest.approx(0.70)

        reset_config()

        assert get_cached_trust_weight("general-extractor") == 1.0

    @pytest.mark.asyncio
    async def test_trust_weight_applied_in_effective_confidence(self, db_session: object) -> None:
        from particles.core.scoring.confidence import compute_effective_confidence
        from particles.store.extractor_store import (
            get_trust_weight_map,
            invalidate_trust_cache,
            populate_trust_cache,
        )

        await ensure_extractor_records(db_session)  # type: ignore[arg-type]
        trust_map = await get_trust_weight_map(db_session)  # type: ignore[arg-type]
        invalidate_trust_cache()
        populate_trust_cache(trust_map)

        # general-extractor at 0.70 should discount confidence
        eff = compute_effective_confidence(1.0, extractor_trust_weight=0.70)
        assert eff == pytest.approx(0.70)

        # nomisma at 0.95 should barely discount
        eff2 = compute_effective_confidence(1.0, extractor_trust_weight=0.95)
        assert eff2 == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# read-side conformance trust cap
# ---------------------------------------------------------------------------


class TestConformanceTrustCap:
    """The opt-in clamp on the *effective* extractor trust weight.

    The cap reads ``ExtractorRow.conformance_required_failure`` and clamps the
    weight ``get_trust_weight_map`` returns; it never mutates the stored
    ``trust_weight``. Config is set by mutating the cached singleton (the autouse
    ``reset_config()`` clears it before the next test), matching test_curation.py.
    """

    @staticmethod
    async def _setup(db_session: object, *, weight: float, status: bool | None) -> None:
        from particles.store.extractor_store import set_conformance_status, set_trust_weight

        await ensure_extractor_records(db_session)  # type: ignore[arg-type]
        await set_trust_weight(db_session, "general-extractor", weight)  # type: ignore[arg-type]
        if status is not None:
            await set_conformance_status(db_session, "general-extractor", status)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_disabled_by_default_is_inert(self, db_session: object) -> None:
        from particles.config import get_config
        from particles.store.extractor_store import get_trust_weight_map

        await self._setup(db_session, weight=0.70, status=True)  # evaluable failure recorded
        assert get_config().conformance.trust_cap.enabled is False
        m = await get_trust_weight_map(db_session)  # type: ignore[arg-type]
        assert m["general-extractor"] == pytest.approx(0.70)  # cap off → unchanged

    @pytest.mark.asyncio
    async def test_enabled_clamps_an_evaluable_failure(self, db_session: object) -> None:
        from particles.config import get_config
        from particles.store.extractor_store import get_trust_weight_map

        await self._setup(db_session, weight=0.70, status=True)
        cap = get_config().conformance.trust_cap
        cap.enabled = True
        cap.cap_value = 0.5
        m = await get_trust_weight_map(db_session)  # type: ignore[arg-type]
        assert m["general-extractor"] == pytest.approx(0.5)  # 0.70 clamped to 0.5

    @pytest.mark.asyncio
    async def test_unknown_status_never_clamps(self, db_session: object) -> None:
        # conformance_required_failure is NULL (never run) → unknown, never failed.
        from particles.config import get_config
        from particles.store.extractor_store import get_trust_weight_map

        await self._setup(db_session, weight=0.70, status=None)
        get_config().conformance.trust_cap.enabled = True
        m = await get_trust_weight_map(db_session)  # type: ignore[arg-type]
        assert m["general-extractor"] == pytest.approx(0.70)

    @pytest.mark.asyncio
    async def test_passed_status_never_clamps(self, db_session: object) -> None:
        from particles.config import get_config
        from particles.store.extractor_store import get_trust_weight_map

        await self._setup(db_session, weight=0.70, status=False)  # evaluated, passed
        get_config().conformance.trust_cap.enabled = True
        m = await get_trust_weight_map(db_session)  # type: ignore[arg-type]
        assert m["general-extractor"] == pytest.approx(0.70)

    @pytest.mark.asyncio
    async def test_exempt_extractor_is_not_clamped(self, db_session: object) -> None:
        from particles.config import get_config
        from particles.store.extractor_store import get_trust_weight_map

        await self._setup(db_session, weight=0.70, status=True)
        cap = get_config().conformance.trust_cap
        cap.enabled = True
        cap.exempt = ["general-extractor"]
        m = await get_trust_weight_map(db_session)  # type: ignore[arg-type]
        assert m["general-extractor"] == pytest.approx(0.70)

    @pytest.mark.asyncio
    async def test_cap_only_lowers_never_raises(self, db_session: object) -> None:
        # A weight already below cap_value is left untouched (min, not assign).
        from particles.config import get_config
        from particles.store.extractor_store import get_trust_weight_map

        await self._setup(db_session, weight=0.30, status=True)
        cap = get_config().conformance.trust_cap
        cap.enabled = True
        cap.cap_value = 0.5
        m = await get_trust_weight_map(db_session)  # type: ignore[arg-type]
        assert m["general-extractor"] == pytest.approx(0.30)

    @pytest.mark.asyncio
    async def test_self_heals_when_conformance_passes_again(self, db_session: object) -> None:
        from particles.config import get_config
        from particles.store.extractor_store import get_trust_weight_map, set_conformance_status

        await self._setup(db_session, weight=0.70, status=True)
        get_config().conformance.trust_cap.enabled = True
        first = await get_trust_weight_map(db_session)  # type: ignore[arg-type]
        assert first["general-extractor"] == pytest.approx(0.5)

        # Re-run conform; it now passes → status clears → no clamp, automatically.
        await set_conformance_status(db_session, "general-extractor", False)  # type: ignore[arg-type]
        healed = await get_trust_weight_map(db_session)  # type: ignore[arg-type]
        assert healed["general-extractor"] == pytest.approx(0.70)

    @pytest.mark.asyncio
    async def test_set_conformance_status_unknown_extractor_returns_false(
        self, db_session: object
    ) -> None:
        from particles.store.extractor_store import set_conformance_status

        ok = await set_conformance_status(db_session, "does-not-exist", True)  # type: ignore[arg-type]
        assert ok is False

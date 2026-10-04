# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Subject-resolution accuracy column of the extractor benchmark.

What's covered:
  * the ``gold_subjects`` block: mention defaulting (whole-word), ref
    normalisation, and the structural errors that fail a suite
  * the pure judge: the three outcomes, namespace scoping, and the rule that
    a stored ref counts at any confidence (the store joins on it regardless)
  * the loader: the root key is recognised, malformed gold fails the suite,
    and the shipped prose suite carries a well-formed gold set
  * the runner: the fractions join ``metrics`` only when gold is passed, one
    row per gold subject, through the real resolver with Wikidata mocked
  * the CLI table: the column and the non-correct rows
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from particles.benchmark.loader import (
    SuiteLoadError,
    discover_gold_subjects,
    discover_suites,
    load_suite,
    load_suite_and_gold,
)
from particles.benchmark.resolution import (
    GoldSubject,
    GoldSubjectError,
    ResolutionOutcome,
    judge_resolution,
    load_resolution_gold,
    parse_gold_subjects,
    resolution_metrics,
    run_resolution,
)
from particles.benchmark.runner import run_benchmark
from particles.benchmark.schema import BenchmarkCase, BenchmarkSuite, ExpectedParticle
from particles.core.schema import ExternalRef, Snapshot, Subject, UncertaintyNature
from particles.extraction.general import CandidateParticle, ExtractionResult

_PROSE_SUITE = Path("tests/benchmark/suites/prose-article-seed-001.yaml")
_SEARCH = "particles.ingest.authorities.wikidata._wikidata_candidates"
_ALIASES = "particles.ingest.authorities.wikidata._wikidata_aliases"

_CLAIMS = {
    "case-1": [
        "Ostrander Freight rewrote its billing service in Go.",
        "The post is filed under the Infrastructure category.",
        "Harbor has acquired Lantern.",
    ]
}


async def _no_aliases(qid: str) -> list[str]:
    # A fresh list per call: the authority appends the extracted name to it.
    return []


def _subject(*refs: tuple[str, str, float]) -> Subject:
    return Subject(
        canonical_name="x",
        external_ids=[ExternalRef(namespace=ns, id=i, confidence=c) for ns, i, c in refs],
        asserted_by="test",
    )


def _gold(ref: str | None, name: str = "Go") -> GoldSubject:
    return GoldSubject(case_id="case-1", name=name, ref=ref, mention="m")


class TestParseGoldSubjects:
    def test_mention_defaults_to_the_first_claim_naming_the_subject(self) -> None:
        [gold] = parse_gold_subjects(
            [{"case_id": "case-1", "name": "Lantern", "ref": None}], _CLAIMS, "s.yaml"
        )
        assert gold.mention == "Harbor has acquired Lantern."
        assert gold.ref is None

    def test_default_mention_matches_whole_words_only(self) -> None:
        # "category" contains "go"; the default must not take that claim.
        [gold] = parse_gold_subjects(
            [{"case_id": "case-1", "name": "go", "ref": "wikidata:Q37227"}], _CLAIMS, "s.yaml"
        )
        assert gold.mention == "Ostrander Freight rewrote its billing service in Go."

    def test_explicit_mention_wins(self) -> None:
        [gold] = parse_gold_subjects(
            [{"case_id": "case-1", "name": "Rust", "ref": "wikidata:Q575650", "mention": "m"}],
            _CLAIMS,
            "s.yaml",
        )
        assert gold.mention == "m"

    def test_wikidata_ids_normalise_to_the_qid(self) -> None:
        [gold] = parse_gold_subjects(
            [{"case_id": "case-1", "name": "Go", "ref": "Wikidata:37227"}], _CLAIMS, "s.yaml"
        )
        assert gold.ref == "wikidata:Q37227"

    @pytest.mark.parametrize(
        ("entry", "fragment"),
        [
            ({"case_id": "case-1", "name": "Go", "ref": None, "qid": "x"}, "unknown"),
            ({"case_id": "nope", "name": "Go", "ref": None}, "not a case"),
            ({"case_id": "case-1", "name": "Go"}, "missing required field 'ref'"),
            ({"case_id": "case-1", "name": "Go", "ref": "Q37227"}, "NAMESPACE:ID"),
            ({"case_id": "case-1", "name": "Kafka", "ref": None}, "state the claim text"),
        ],
    )
    def test_structural_errors_raise(self, entry: dict[str, Any], fragment: str) -> None:
        with pytest.raises(GoldSubjectError, match=fragment):
            parse_gold_subjects([entry], _CLAIMS, "s.yaml")

    def test_a_name_listed_twice_for_one_case_raises(self) -> None:
        entry = {"case_id": "case-1", "name": "Go", "ref": None}
        with pytest.raises(GoldSubjectError, match="listed twice"):
            parse_gold_subjects([entry, {**entry, "name": "GO"}], _CLAIMS, "s.yaml")


class TestJudgeResolution:
    def test_gold_ref_matched_is_correct(self) -> None:
        row = judge_resolution(_gold("wikidata:Q37227"), _subject(("wikidata", "Q37227", 0.9)))
        assert row.outcome is ResolutionOutcome.CORRECT
        assert row.refs == ["wikidata:Q37227@0.90"]

    def test_no_ref_where_gold_has_one_is_bare_local(self) -> None:
        row = judge_resolution(_gold("wikidata:Q37227"), _subject())
        assert row.outcome is ResolutionOutcome.BARE_LOCAL

    def test_another_qid_is_a_wrong_ref(self) -> None:
        row = judge_resolution(_gold("wikidata:Q37227"), _subject(("wikidata", "Q41587", 0.9)))
        assert row.outcome is ResolutionOutcome.WRONG_REF

    def test_a_ref_in_another_namespace_is_not_compared(self) -> None:
        row = judge_resolution(_gold("wikidata:Q37227"), _subject(("numista", "123", 1.0)))
        assert row.outcome is ResolutionOutcome.BARE_LOCAL

    def test_null_gold_with_no_ref_is_correct(self) -> None:
        assert judge_resolution(_gold(None), _subject()).outcome is ResolutionOutcome.CORRECT

    def test_a_sub_threshold_ref_still_counts(self) -> None:
        # Exporters hide a 0.20 link, but find_by_external_ref joins on it, so
        # the next mention of the name lands on this Subject anyway.
        row = judge_resolution(_gold(None, "Lantern"), _subject(("wikidata", "Q862454", 0.20)))
        assert row.outcome is ResolutionOutcome.WRONG_REF

    def test_recognised_digits_only_id_matches_the_qid(self) -> None:
        row = judge_resolution(_gold("wikidata:Q37227"), _subject(("wikidata", "37227", 1.0)))
        assert row.outcome is ResolutionOutcome.CORRECT


def test_resolution_metrics_partition_the_gold_set() -> None:
    rows = [
        judge_resolution(_gold(None), _subject()),
        judge_resolution(_gold(None), _subject(("wikidata", "Q1", 0.3))),
        judge_resolution(_gold("wikidata:Q2"), _subject()),
        judge_resolution(_gold("wikidata:Q2"), _subject(("wikidata", "Q2", 0.3))),
    ]
    metrics = resolution_metrics(rows)
    assert metrics == {
        "resolution_accuracy": 0.5,
        "resolution_wrong_ref": 0.25,
        "resolution_bare_local": 0.25,
    }
    assert resolution_metrics([]) == {}


def _write_suite(path: Path, gold: Any) -> Path:
    doc: dict[str, Any] = {
        "suite_id": "gold-suite",
        "name": "g",
        "version": "0.0.1",
        "domain": "test",
        "source_types": ["WEB_PAGE"],
        "cases": [
            {
                "case_id": "case-1",
                "source_snapshot": {"content_hash": "a" * 64},
                "inline_content": "x",
                "expected": [
                    {
                        "content": c,
                        "confidence_min": 0.5,
                        "uncertainty_nature": "EPISTEMIC",
                    }
                    for c in _CLAIMS["case-1"]
                ],
            }
        ],
    }
    if gold is not None:
        doc["gold_subjects"] = gold
    path.write_text(yaml.safe_dump(doc))
    return path


class TestLoader:
    def test_root_key_is_read_without_a_forward_compat_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = _write_suite(tmp_path / "s.yaml", [{"case_id": "case-1", "name": "Go", "ref": None}])
        with caplog.at_level(logging.WARNING, logger="particles.benchmark.loader"):
            suite, gold = load_suite_and_gold(path)
        assert "unrecognised" not in caplog.text
        assert suite.suite_id == "gold-suite"
        assert [g.name for g in gold] == ["Go"]
        # The frozen §13.3 dataclass is untouched: load_suite returns just it.
        plain = load_suite(path)
        assert isinstance(plain, BenchmarkSuite)
        assert [c.case_id for c in plain.cases] == [c.case_id for c in suite.cases]

    def test_a_suite_without_gold_loads_with_an_empty_list(self, tmp_path: Path) -> None:
        _, gold = load_suite_and_gold(_write_suite(tmp_path / "s.yaml", None))
        assert gold == []

    def test_malformed_gold_fails_the_suite_and_discovery_skips_it(self, tmp_path: Path) -> None:
        path = _write_suite(tmp_path / "s.yaml", [{"case_id": "case-1", "name": "Go"}])
        with pytest.raises(SuiteLoadError, match="ref"):
            load_suite(path)
        assert list(discover_suites(tmp_path)) == []
        assert discover_gold_subjects(tmp_path) == {}

    def test_discovery_keys_gold_by_suite_id(self, tmp_path: Path) -> None:
        _write_suite(tmp_path / "a.yaml", [{"case_id": "case-1", "name": "Go", "ref": None}])
        assert list(discover_gold_subjects(tmp_path)) == ["gold-suite"]

    def test_shipped_prose_suite_carries_well_formed_gold(self) -> None:
        suite, gold = load_suite_and_gold(_PROSE_SUITE)
        cases = {c.case_id for c in suite.cases}
        assert len(gold) >= 30
        assert {g.case_id for g in gold} == cases
        # asks for the common-word failure in at least two fixtures.
        assert {"web-article-003", "web-article-004"} <= cases
        assert any(g.ref is None for g in gold)
        assert any(g.ref is not None for g in gold)


_ORDINARY_PROSE = Path(__file__).parent / "benchmark" / "resolution" / "ordinary-prose-001.yaml"


def _write_gold_set(path: Path, **overrides: Any) -> Path:
    raw: dict[str, Any] = {
        "gold_set_id": "g",
        "version": "0.1.0",
        "source_type": "WEB_PAGE",
        "passages": {"p1": "We moved the build\nfrom Webpack to Vite. Lena Marsh agreed."},
        "gold_subjects": [
            {
                "case_id": "p1",
                "name": "Vite",
                "ref": "wikidata:Q111590996",
                "mention": "We moved the build from Webpack to Vite.",
            },
        ],
    }
    raw.update(overrides)
    path.write_text(yaml.safe_dump(raw, allow_unicode=True))
    return path


class TestResolutionGoldSet:
    def test_a_gold_set_loads_with_its_mentions(self, tmp_path: Path) -> None:
        gold = load_resolution_gold(_write_gold_set(tmp_path / "g.yaml"))
        assert (gold.gold_set_id, gold.source_type) == ("g", "WEB_PAGE")
        # The passage's line break and the mention's space compare equal.
        assert [(g.name, g.ref) for g in gold.subjects] == [("Vite", "wikidata:Q111590996")]

    def test_a_mention_not_in_its_passage_raises(self, tmp_path: Path) -> None:
        entry = {"case_id": "p1", "name": "Vite", "ref": None, "mention": "Vite is fast."}
        with pytest.raises(GoldSubjectError, match="not a sentence of passage"):
            load_resolution_gold(_write_gold_set(tmp_path / "g.yaml", gold_subjects=[entry]))

    def test_every_mention_must_be_stated(self, tmp_path: Path) -> None:
        entry = {"case_id": "p1", "name": "Vite", "ref": None}
        with pytest.raises(GoldSubjectError, match="states every mention"):
            load_resolution_gold(_write_gold_set(tmp_path / "g.yaml", gold_subjects=[entry]))

    def test_an_unknown_case_or_root_key_raises(self, tmp_path: Path) -> None:
        entry = {"case_id": "p2", "name": "Vite", "ref": None, "mention": "x"}
        with pytest.raises(GoldSubjectError, match="not a case"):
            load_resolution_gold(_write_gold_set(tmp_path / "a.yaml", gold_subjects=[entry]))
        with pytest.raises(GoldSubjectError, match="unknown root"):
            load_resolution_gold(_write_gold_set(tmp_path / "b.yaml", cases=[]))

    def test_the_shipped_ordinary_prose_set_is_well_formed(self) -> None:
        gold = load_resolution_gold(_ORDINARY_PROSE)
        assert len(gold.subjects) >= 40
        assert any(g.ref is None for g in gold.subjects)
        # It is not an extractor suite, so suite discovery must never see it.
        assert _ORDINARY_PROSE.parent.name != "suites"


@pytest.mark.usefixtures("no_embedding_model")
class TestRunResolution:
    @pytest.mark.asyncio
    async def test_runs_the_real_cascade_per_gold_subject(self) -> None:
        gold = [
            GoldSubject("case-1", "Go", "wikidata:Q37227", "rewritten in Go"),
            GoldSubject("case-1", "Lantern", None, "Harbor has acquired Lantern."),
            GoldSubject("case-1", "Ostrander Freight", None, "Ostrander Freight is a company."),
        ]
        hits: dict[str, list[dict[str, object]]] = {
            "Go": [{"id": "Q37227", "label": "Go", "description": "programming language"}],
            "Lantern": [{"id": "Q862454", "label": "lantern", "description": "lighting device"}],
            "Ostrander Freight": [],
        }

        async def _search(name: str, *, limit: int = 5) -> list[dict[str, object]]:
            return hits[name]

        with (
            patch(_SEARCH, side_effect=_search),
            patch(_ALIASES, side_effect=_no_aliases),
        ):
            rows, notes = await run_resolution(gold, source_type="WEB_PAGE")
        assert [r.outcome for r in rows] == [
            ResolutionOutcome.CORRECT,
            ResolutionOutcome.WRONG_REF,
            ResolutionOutcome.CORRECT,
        ]
        # No encoder in the unit tier: every link sits at the 0.5 sentinel, and
        # the run says these outcomes do not measure disambiguation.
        assert rows[0].refs == ["wikidata:Q37227@0.50"]
        assert any("no embedding model" in n for n in notes)

    @pytest.mark.asyncio
    async def test_each_subject_gets_its_own_store(self) -> None:
        # The same name twice: a shared store would answer the second from the
        # first's Subject without searching again.
        gold = [
            GoldSubject("case-1", "Go", "wikidata:Q37227", "m"),
            GoldSubject("case-2", "Go", "wikidata:Q37227", "m"),
        ]
        search = AsyncMock(return_value=[{"id": "Q37227", "label": "Go", "description": "d"}])
        with patch(_SEARCH, search), patch(_ALIASES, side_effect=_no_aliases):
            rows, _ = await run_resolution(gold, source_type="WEB_PAGE")
        assert search.await_count == 2
        assert all(r.outcome is ResolutionOutcome.CORRECT for r in rows)

    @pytest.mark.asyncio
    async def test_failed_live_lookups_are_disclosed(self) -> None:
        gold = [GoldSubject("case-1", "Go", "wikidata:Q37227", "m")]

        async def _failing(name: str, *, limit: int = 5) -> list[dict[str, object]]:
            logging.getLogger("particles.ingest.authorities.wikidata").warning(
                "Wikidata search failed for %r: %s", name, "timeout"
            )
            return []

        with patch(_SEARCH, side_effect=_failing):
            rows, notes = await run_resolution(gold, source_type="WEB_PAGE")
        assert rows[0].outcome is ResolutionOutcome.BARE_LOCAL
        assert any("1 live lookup(s) failed" in n for n in notes)

    @pytest.mark.asyncio
    async def test_no_gold_makes_no_call(self) -> None:
        with patch(_SEARCH, new_callable=AsyncMock) as search:
            assert await run_resolution([], source_type="WEB_PAGE") == ([], [])
        search.assert_not_awaited()


class _OneClaim:
    EXTRACTOR_ID = "one-claim-stub"
    EXTRACTOR_VERSION = "0.0.1"

    def accepts(self, source_type: str) -> bool:
        return True

    async def extract(
        self, snapshot: Snapshot, content: bytes, **kwargs: object
    ) -> ExtractionResult:
        return ExtractionResult(
            candidates=[
                CandidateParticle(
                    content="Mercury is a planet",
                    confidence_value=0.95,
                    uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    subjects=["Mercury"],
                )
            ]
        )


def _one_case_suite() -> BenchmarkSuite:
    return BenchmarkSuite(
        suite_id="stub-suite",
        name="stub",
        version="0.0.1",
        domain="test",
        source_types=["WEB_PAGE"],
        cases=[
            BenchmarkCase(
                case_id="case-1",
                expected=[
                    ExpectedParticle(
                        content="Mercury is a planet",
                        confidence_min=0.5,
                        uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    )
                ],
                source_snapshot=Snapshot(content_hash="a" * 64),
                inline_content=b"x",
            )
        ],
    )


@pytest.mark.usefixtures("no_embedding_model")
class TestRunnerColumn:
    @pytest.mark.asyncio
    async def test_gold_adds_the_three_fractions_and_one_row_each(self) -> None:
        gold = [GoldSubject("case-1", "Mercury", "wikidata:Q308", "Mercury is a planet")]
        hit = [{"id": "Q308", "label": "Mercury", "description": "planet"}]
        with (
            patch(_SEARCH, new_callable=AsyncMock, return_value=hit),
            patch(_ALIASES, side_effect=_no_aliases),
        ):
            report = await run_benchmark(
                _one_case_suite(), _OneClaim(), fixture_dir=Path("."), gold_subjects=gold
            )
        assert report.metrics["resolution_accuracy"] == 1.0
        assert report.metrics["resolution_wrong_ref"] == 0.0
        assert report.metrics["resolution_bare_local"] == 0.0
        assert [r.name for r in report.subject_resolution] == ["Mercury"]
        # The three §13.3 metrics are still reported beside it.
        assert {"precision", "recall", "calibration_error"} <= set(report.metrics)

    @pytest.mark.asyncio
    async def test_no_gold_leaves_the_report_as_it_was(self) -> None:
        with patch(_SEARCH, new_callable=AsyncMock) as search:
            report = await run_benchmark(_one_case_suite(), _OneClaim(), fixture_dir=Path("."))
        search.assert_not_awaited()
        # No ``resolution_*`` key; the semantic-match ECE is reported for
        # every suite.
        assert set(report.metrics) == {
            "precision",
            "recall",
            "calibration_error",
            "calibration_error_semantic",
        }
        assert report.subject_resolution == []


def test_table_prints_the_column_and_the_failures(capsys: pytest.CaptureFixture[str]) -> None:
    from particles.api.cli.extractor import _print_benchmark_table
    from particles.benchmark.runner import BenchmarkReport

    rows = [
        judge_resolution(_gold(None, "Harbor"), _subject(("wikidata", "Q283202", 0.3))),
        judge_resolution(_gold("wikidata:Q575650", "Rust"), _subject()),
        judge_resolution(_gold(None, "Ostrander Freight"), _subject()),
    ]
    report = BenchmarkReport(
        suite_id="s",
        suite_version="0.0.1",
        extractor_id="e",
        extractor_version="0.0.1",
        cases_run=0,
        cases_total=0,
        particles_emitted=0,
        particles_required_total=0,
        metrics={"recall": 1.0, **resolution_metrics(rows)},
        per_case=[],
        generated_at=__import__("datetime").datetime.now(),
        judge="embedding",
        equivalence_threshold=0.8,
        subject_resolution=rows,
    )
    _print_benchmark_table(report)
    out = capsys.readouterr().out
    # Whitespace-split: the name column is sized to the longest metric name.
    assert ["resolution_accuracy", "0.33"] in [line.split() for line in out.splitlines()]
    assert "Subject resolution: 1/3 correct, 1 wrong ref, 1 bare local" in out
    assert "WRONG REF  'Harbor' → wikidata:Q283202@0.30  (gold none)" in out
    assert "BARE LOCAL 'Rust'  (gold wikidata:Q575650)" in out
    assert "Ostrander Freight" not in out

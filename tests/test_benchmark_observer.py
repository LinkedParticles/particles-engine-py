# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The two-project observer fixture (gate B).

The world generator is pure and its fingerprint is pinned; the end-to-end run
drives the real pipeline with the rot tests' bag-of-words encoder. Across three
seeds no retirement may cross a project boundary, every declined pair must be
recorded, and a contested global line must reach review.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.benchmark.observer import (
    LineKind,
    ObserverArm,
    VanishCause,
    WriteCensus,
    generate_world,
    lines_on_day,
    metrics_for,
    render_report,
    run_observer_fixture,
)
from particles.benchmark.observer.generator import (
    PROJECT_A,
    PROJECT_B,
    RULES,
    SLOTS,
    check_value_invariants,
    days_with_changes,
)
from particles.benchmark.observer.oracle import contradicts
from particles.benchmark.observer.schema import LineObservation
from tests.test_benchmark_rot import bow_encoder  # noqa: F401 — the stand-in encoder fixture


class TestWorld:
    def test_generation_is_pure_and_pinned(self) -> None:
        a, b = generate_world(1, 14), generate_world(1, 14)
        assert a == b
        # Bump GENERATOR_VERSION when a seed's output is meant to change.
        assert a.fingerprint == "4588ec54e85f3074"

    def test_day_one_gives_every_slot_a_value_and_every_project_an_own_line(self) -> None:
        world = generate_world(3, 10)
        for project in (PROJECT_A, PROJECT_B):
            lines = lines_on_day(world, project, 1)
            assert {ln.slot for ln in lines if ln.kind is LineKind.FACT} == set(SLOTS)
            assert sum(ln.kind is LineKind.OWN for ln in lines) == 1
            assert 1 in days_with_changes(world, project)

    def test_value_invariants_hold(self) -> None:
        check_value_invariants()

    def test_scripted_probe_reads_slots_not_words(self) -> None:
        assert contradicts(
            "The default branch of the repository is main.",
            "The default branch of the repository is master.",
        )
        assert not contradicts(
            "The default branch of the repository is main.",
            "The default branch of the repository is main.",
        )
        assert not contradicts(next(iter(RULES)), "The test suite is run with pytest.")


class TestMetrics:
    def test_leak_rows_are_counted_separately_from_own_lines(self) -> None:
        own_ok = LineObservation(
            day=1,
            project=PROJECT_A,
            kind=LineKind.FACT,
            text="x",
            visible=True,
            visible_store_wide=True,
        )
        own_gone = LineObservation(
            day=1,
            project=PROJECT_A,
            kind=LineKind.FACT,
            text="y",
            visible=False,
            visible_store_wide=False,
            cause=VanishCause.SUPERSEDED_BY_UPDATE,
            cross_project=True,
            winner_in_view=False,
        )
        leak = LineObservation(
            day=1,
            project=PROJECT_A,
            kind=LineKind.RULE,
            text="z",
            visible=True,
            visible_store_wide=True,
            cross_project=True,
        )
        m = metrics_for([own_ok, own_gone, leak])
        assert (m.own_visible.numerator, m.own_visible.denominator) == (1, 2)
        assert (m.leaked.numerator, m.leaked.denominator) == (1, 2)
        assert m.vanished_by_cause == {"superseded_by_update": 1}
        assert (m.winner_in_view.numerator, m.winner_in_view.denominator) == (0, 1)


class TestEndToEnd:
    @pytest.fixture
    def report(self, bow_encoder: None, tmp_path: Path) -> Generator[object, None, None]:  # noqa: F811
        yield asyncio.run(
            run_observer_fixture(seeds=[1, 2, 3], days=14, work_dir=tmp_path, keep_stores=True)
        )

    def test_divergent_claims_stand_and_no_project_loses_its_own_line(self, report: object) -> None:
        from particles.benchmark.observer import ObserverReport

        assert isinstance(report, ObserverReport)
        m, census = report.metrics, report.write_census
        # No retirement crosses a project boundary (§4) …
        assert census.cross_project_supersessions == 0
        assert census.cross_project_cascades == 0
        # … while a project's own update still retires its own earlier value.
        assert census.own_supersessions > 0
        # Every declined pair is disclosed as a CONTRADICTS relation.
        assert census.declined_pairs > 0 and census.divergences_recorded > 0
        assert census.declined_without_relation == 0
        # A project contesting a global line reaches review, every time.
        assert census.global_contests.value == 1.0
        for kind in ("fact", "rule", "own"):
            assert m.by_kind[kind].value is not None and m.by_kind[kind].value >= 0.99
        assert m.leaked.numerator == 0
        assert m.own_visible.numerator <= m.own_visible_store_wide.numerator
        assert report.refused_llm_calls == {}
        assert all(w.fingerprint == generate_world(w.seed, 14).fingerprint for w in report.worlds)

    def test_report_renders(self, report: object) -> None:
        from particles.benchmark.observer import ObserverReport

        assert isinstance(report, ObserverReport)
        text = render_report(report)
        assert "Own lines in view for their project" in text
        assert "CONTRADICTS" in text


class TestChunkedArm:
    """The arm where an unchanged chunk is carried forward, not re-emitted."""

    def test_carries_unchanged_chunks_and_leaks_nothing(
        self,
        bow_encoder: None,  # noqa: F811
        tmp_path: Path,
    ) -> None:
        report = asyncio.run(
            run_observer_fixture(seeds=[1], days=8, work_dir=tmp_path, arm=ObserverArm.CHUNKED)
        )
        census = report.write_census
        assert report.arm is ObserverArm.CHUNKED
        assert census.chunks_carried > 0 and census.chunks_extracted > 0
        assert report.metrics.leaked.numerator == 0
        assert report.metrics.by_kind["own"].value == 1.0
        # Carried-forward claims keep their project in view (§3).
        assert report.metrics.own_visible.value == 1.0
        assert census.cross_project_supersessions == census.cross_project_cascades == 0
        assert report.refused_llm_calls == {}
        assert "Chunks carried forward" in render_report(report)


def test_write_census_pools_counts_and_rates() -> None:
    a = WriteCensus(own_supersessions=2, chunks_carried=1)
    a.global_contests.add(True)
    b = WriteCensus(own_supersessions=3, cross_project_cascades=1)
    b.global_contests.add(False)
    pooled = a.merged(b)
    assert pooled.own_supersessions == 5
    assert pooled.cross_project_cascades == 1
    assert pooled.chunks_carried == 1
    assert (pooled.global_contests.numerator, pooled.global_contests.denominator) == (1, 2)


def test_cli_runs_one_small_world(bow_encoder: None, tmp_path: Path) -> None:  # noqa: F811
    result = CliRunner().invoke(
        app,
        ["benchmark", "observer", "--seed", "1", "--days", "4", "--output", str(tmp_path / "r.md")],
    )
    assert result.exit_code == 0, result.output
    assert "Own lines in view" in (tmp_path / "r.md").read_text()


def test_cli_runs_the_chunked_arm(bow_encoder: None, tmp_path: Path) -> None:  # noqa: F811
    out = tmp_path / "r.md"
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "observer",
            "--seed",
            "1",
            "--days",
            "3",
            "--arm",
            "chunked",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "arm `chunked`" in out.read_text()


def test_cli_rejects_a_bad_format() -> None:
    result = CliRunner().invoke(app, ["benchmark", "observer", "--format", "yaml"])
    assert result.exit_code == 2

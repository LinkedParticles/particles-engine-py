"""Dogfood corpus regression tests.

Runs the wiki + obsidian exporters against a ~10-subject in-repo
fixture with a mocked LLM that returns realistic per-subject prose +
judge verdicts. Asserts behavioural properties — file shapes,
citation density, synthesis-vs-fallback ratio above a floor, correct
template / path-nesting per subject class — designed to catch the
regression classes that bit operator dry-runs in v0.22-v0.23 but
sailed past the trivial-fixture unit tests.

Marked ``@pytest.mark.dogfood`` so it ships with the default
``pytest`` run but operators can ``-m "not dogfood"`` for fast inner
loops. See ``tests/dogfood/AGENTS.md`` for fixture-maintenance rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from particles.db import session_scope
from particles.exporters.article_synthesis import _parse_frontmatter
from particles.exporters.markdown import subject_slug
from particles.exporters.obsidian import ObsidianExporter
from particles.exporters.wiki import WikiExporter
from particles.llm import set_client
from tests.dogfood import (
    DogfoodSubject,
    build_mock_client,
    load_corpus,
    persist_corpus,
)

pytestmark = [pytest.mark.dogfood, pytest.mark.asyncio]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def populated_db(
    db_session: object,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[DogfoodSubject], Any]:
    """Load corpus.yaml into the in-memory DB; install the mock LLM.

    The mock LLM is removed automatically at fixture teardown so it
    doesn't leak into other tests in the same session. ANTHROPIC_API_KEY
    is set to a sentinel so the synthesis path doesn't short-circuit
    out on the missing-key check.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    subjects = load_corpus()
    await persist_corpus(db_session, subjects)  # type: ignore[arg-type]
    client = build_mock_client(subjects)
    set_client(client)
    try:
        yield subjects, db_session
    finally:
        set_client(None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _has_citation(body: str) -> bool:
    return "[^p-" in body or "[[#^p-" in body  # markdown footnote or obsidian block ref


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


# ---------------------------------------------------------------------------
# Wiki exporter assertions
# ---------------------------------------------------------------------------


class TestDogfoodWikiExporter:
    """Behavioural assertions over the wiki exporter against the
    dogfood corpus."""

    async def test_wiki_export_produces_expected_files(
        self, populated_db: tuple[list[DogfoodSubject], Any], tmp_path: Path
    ) -> None:
        subjects, _ = populated_db
        exporter = WikiExporter()
        async with session_scope() as session:
            await exporter.export(session, tmp_path)

        # Subjects with ≥ wiki.min_particles (default 3) ACTIVE particles
        # should produce a per-subject .md file.
        from particles.config import get_config
        from particles.core.status import Status

        min_particles = get_config().wiki.min_particles
        qualifying = [
            ds
            for ds in subjects
            if sum(1 for p in ds.particles if p.status == Status.ACTIVE) >= min_particles
        ]
        for ds in qualifying:
            slug = subject_slug(ds.subject.canonical_name)
            article_path = tmp_path / f"{slug}.md"
            assert article_path.exists(), (
                f"Expected wiki article for {ds.subject.canonical_name!r} at {article_path}"
            )

        # Index file is always written.
        assert (tmp_path / "index.md").exists()

    async def test_every_wiki_article_has_valid_frontmatter(
        self, populated_db: tuple[list[DogfoodSubject], Any], tmp_path: Path
    ) -> None:
        subjects, _ = populated_db
        exporter = WikiExporter()
        async with session_scope() as session:
            await exporter.export(session, tmp_path)

        for article in tmp_path.glob("*.md"):
            if article.name == "index.md":
                continue
            fm = _parse_frontmatter(_read(article))
            assert fm is not None, f"{article.name} has no parseable frontmatter"
            for required_field in (
                "particle_count",
                "input_hash",
                "sources",
                "synthesis",
            ):
                assert required_field in fm, (
                    f"{article.name} frontmatter missing {required_field!r}"
                )
            assert fm["synthesis"] in {"llm", "structured-listing"}

    async def test_synthesis_success_ratio_above_floor(
        self, populated_db: tuple[list[DogfoodSubject], Any], tmp_path: Path
    ) -> None:
        """The dogfood corpus + mocked LLM should drive most qualifying
        subjects through the LLM synthesis path. If a future code change
        breaks the pipeline (judge regression, prompt change, validation
        bug) the ratio drops and this assertion fires.

        Floor is generous (≥ 70 %) because some corpus subjects are
        intentionally edge cases. Tune up over time as the pipeline
        stabilises.
        """
        _ = populated_db  # fixture side-effects only
        exporter = WikiExporter()
        async with session_scope() as session:
            summary = await exporter.export(session, tmp_path)

        used = summary.synthesis_used or 0
        failed = summary.synthesis_failed or 0
        total = used + failed
        assert total > 0, "Dogfood corpus produced no qualifying subjects"
        ratio = _ratio(used, total)
        assert ratio >= 0.70, (
            f"Synthesis success ratio {ratio:.0%} fell below the 70 % "
            f"floor (used={used}, failed={failed}). A code change is "
            "regressing the dogfood synthesis pipeline; investigate before "
            "raising the floor."
        )

    async def test_every_synthesised_article_contains_a_citation(
        self, populated_db: tuple[list[DogfoodSubject], Any], tmp_path: Path
    ) -> None:
        """Articles tagged `synthesis: llm` must contain at least one
        ``[^p-xxxxxxxx]`` citation in the body. Zero-citation bodies
        slipping through synthesis would be a Layer A regression."""
        _ = populated_db
        exporter = WikiExporter()
        async with session_scope() as session:
            await exporter.export(session, tmp_path)

        for article in tmp_path.glob("*.md"):
            if article.name == "index.md":
                continue
            body = _read(article)
            fm = _parse_frontmatter(body)
            assert fm is not None
            if fm["synthesis"] != "llm":
                continue
            assert _has_citation(body), (
                f"{article.name} is tagged synthesis=llm but the body contains no [^p-…] citation"
            )


# ---------------------------------------------------------------------------
# Obsidian exporter assertions
# ---------------------------------------------------------------------------


class TestDogfoodObsidianExporter:
    async def test_obsidian_export_nests_reddit_users(
        self, populated_db: tuple[list[DogfoodSubject], Any], tmp_path: Path
    ) -> None:
        """``u/AlgoTrader`` must land at ``reddit.com/u/AlgoTrader.md``,
        not at the vault root. Regression-tests the v0.22.13 bare-
        username canonicalisation fix."""
        _ = populated_db
        exporter = ObsidianExporter()
        async with session_scope() as session:
            # min_particles=0 / min_links=0 keeps every dogfood subject
            # in the vault regardless of the operator's daily-driver
            # config. Cross-references between corpus subjects are
            # sparse on purpose — we're testing the exporter's
            # path-nesting + template-dispatch logic, not its
            # link-suppression filter.
            await exporter.export(session, tmp_path, min_particles=0, min_links=0)

        assert (tmp_path / "reddit.com" / "u" / "AlgoTrader.md").exists()
        # And specifically NOT at the vault root.
        assert not (tmp_path / "AlgoTrader.md").exists()

    async def test_obsidian_export_nests_github_users(
        self, populated_db: tuple[list[DogfoodSubject], Any], tmp_path: Path
    ) -> None:
        _ = populated_db
        exporter = ObsidianExporter()
        async with session_scope() as session:
            # min_particles=0 / min_links=0 keeps every dogfood subject
            # in the vault regardless of the operator's daily-driver
            # config. Cross-references between corpus subjects are
            # sparse on purpose — we're testing the exporter's
            # path-nesting + template-dispatch logic, not its
            # link-suppression filter.
            await exporter.export(session, tmp_path, min_particles=0, min_links=0)

        assert (tmp_path / "github.com" / "karpathy.md").exists()
        assert not (tmp_path / "karpathy.md").exists()

    async def test_obsidian_export_uses_coin_template_for_numismatic_subjects(
        self, populated_db: tuple[list[DogfoodSubject], Any], tmp_path: Path
    ) -> None:
        _ = populated_db
        exporter = ObsidianExporter()
        async with session_scope() as session:
            # min_particles=0 / min_links=0 keeps every dogfood subject
            # in the vault regardless of the operator's daily-driver
            # config. Cross-references between corpus subjects are
            # sparse on purpose — we're testing the exporter's
            # path-nesting + template-dispatch logic, not its
            # link-suppression filter.
            await exporter.export(session, tmp_path, min_particles=0, min_links=0)

        coin_path = tmp_path / subject_slug("1 Pfennig (1948-1950) GDR")
        coin_path = coin_path.with_suffix(".md")
        body = _read(coin_path)
        assert "particles/coin" in body

    async def test_low_coverage_subjects_get_skip_annotation_with_synthesis(
        self, populated_db: tuple[list[DogfoodSubject], Any], tmp_path: Path
    ) -> None:
        """Mileva Marić has 1 particle; with synthesis_min_particles=3
        default she should land in the vault with the
        ``article_synthesis: skipped-low-coverage`` annotation."""
        _ = populated_db
        exporter = ObsidianExporter()
        async with session_scope() as session:
            await exporter.export(
                session, tmp_path, min_particles=0, min_links=0, with_synthesis=True
            )

        path = tmp_path / subject_slug("Mileva Marić")
        path = path.with_suffix(".md")
        assert path.exists()
        body = _read(path)
        fm = _parse_frontmatter(body)
        assert fm is not None
        assert fm.get("article_synthesis") == "skipped-low-coverage"

    async def test_superseded_particles_excluded_from_synthesis(
        self, populated_db: tuple[list[DogfoodSubject], Any], tmp_path: Path
    ) -> None:
        """BoringSubject has 3 ACTIVE + 2 SUPERSEDED particles. The
        synthesised article must reference only the 3 ACTIVE ones —
        SUPERSEDED claims (Ruby launch, macOS-only) must not appear."""
        _ = populated_db
        exporter = ObsidianExporter()
        async with session_scope() as session:
            await exporter.export(
                session, tmp_path, min_particles=0, min_links=0, with_synthesis=True
            )

        path = tmp_path / "BoringSubject.md"
        body = _read(path)
        # SUPERSEDED claim text must not leak into the rendered article.
        assert "Ruby" not in body
        assert "macOS at launch" not in body
        # ACTIVE claims should be present (synthesised or fallback).
        assert "launched in 2020" in body or "version 2.0" in body


# ---------------------------------------------------------------------------
# Taxonomy / tag-aware query expansion
# ---------------------------------------------------------------------------


class TestDogfoodTagQuery:
    """The dogfood corpus deposits a ``Coins`` taxonomy with subtree
    ``coins/by-region/germany`` and tags the three GDR-Pfennig
    particles with that leaf. A ``QueryRequest(tags=['coins'])`` must
    subtree-expand and surface those particles, while leaving untagged
    subjects (TagSpaces, Mileva, …) out of the candidate set."""

    async def test_subtree_expansion_returns_only_tagged_particles(
        self, populated_db: tuple[list[DogfoodSubject], Any]
    ) -> None:
        from particles.core.schema import QueryRequest
        from particles.operations.query import query as query_op

        _ = populated_db
        req = QueryRequest(question="what coins exist?", tags=["coins"], top_k=200)
        async with session_scope() as session:
            response = await query_op(session, req)

        contents = " ".join(p.content for p in response.particles)
        # Every retrieved particle must be one of the GDR-Pfennig claims —
        # those are the only tagged particles in the corpus.
        assert all("1 Pfennig (1948-1950) GDR" in p.content for p in response.particles)
        assert "TagSpaces" not in contents
        # And the three tagged particles must all be present.
        assert len(response.particles) == 3

    async def test_leaf_tag_matches_only_leaf(
        self, populated_db: tuple[list[DogfoodSubject], Any]
    ) -> None:
        from particles.core.schema import QueryRequest
        from particles.operations.query import query as query_op

        _ = populated_db
        req = QueryRequest(question="?", tags=["coins/by-region/germany"], top_k=200)
        async with session_scope() as session:
            response = await query_op(session, req)
        assert len(response.particles) == 3

    async def test_unknown_tag_returns_no_particles(
        self, populated_db: tuple[list[DogfoodSubject], Any]
    ) -> None:
        from particles.core.schema import QueryRequest
        from particles.operations.query import query as query_op

        _ = populated_db
        req = QueryRequest(question="?", tags=["nonexistent"], top_k=200)
        async with session_scope() as session:
            response = await query_op(session, req)
        assert response.particles == []

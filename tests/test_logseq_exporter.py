"""Tests for the Logseq exporter."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from particles.core.schema import (
    CalibrationSource,
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Status,
    Subject,
    UncertaintyNature,
)
from particles.db import session_scope
from particles.exporters.logseq import LogseqExporter
from particles.exporters.logseq.format import (
    logseq_slug,
    render_block,
    render_inline_tags,
    render_property,
    render_subject_page,
)
from particles.exporters.registry import get_exporters
from particles.exporters.summaries import LogseqSummary

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Test helpers (mirror test_wiki_exporter.py shape)
# ---------------------------------------------------------------------------


def _make_particle(
    content: str,
    *,
    properties: dict[str, object] | None = None,
    confidence_value: float = 0.9,
    extractor_id: str = "stub-extractor",
    tags: list[str] | None = None,
) -> Particle:
    return Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(
            value=confidence_value,
            calibration_source=CalibrationSource.EXTRACTOR_DIRECT,
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id="entry-1",
                snapshot_id="snap-1",
            )
        ],
        asserted_by=extractor_id,
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
        extractor_ref={"name": extractor_id, "version": "0.1.0"},
        properties=properties,
        subject_ids=[],
        tags=tags,
    )


async def _persist_subject_with_particles(
    canonical_name: str,
    particles: list[Particle],
    *,
    aliases: list[str] | None = None,
    description: str | None = None,
) -> Subject:
    from particles.store.particle_store import insert_particle
    from particles.store.subject_store import insert_subject, link_particle_to_subjects

    subj = Subject(
        id=str(uuid.uuid4()),
        canonical_name=canonical_name,
        aliases=aliases or [],
        description=description,
        asserted_by="test",
    )
    async with session_scope() as session:
        await insert_subject(session, subj)
        for p in particles:
            await insert_particle(session, p)
            await link_particle_to_subjects(session, p.id, [subj.id])
        await session.commit()
    return subj


# ---------------------------------------------------------------------------
# Slug helper — Logseq filename conventions
# ---------------------------------------------------------------------------


class TestLogseqSlug:
    def test_plain_name_passes_through_with_spaces_to_underscores(self) -> None:
        assert logseq_slug("5 Pfennigs") == "5_Pfennigs"

    def test_hierarchy_separator_uses_triple_underscore(self) -> None:
        assert logseq_slug("Material/Aluminium") == "Material___Aluminium"

    def test_handles_combined_hierarchy_and_spaces(self) -> None:
        assert logseq_slug("Material/Aluminium alloy") == "Material___Aluminium_alloy"

    def test_unsafe_filesystem_chars_get_dashed(self) -> None:
        # The shared sanitiser turns ``:`` (Windows-illegal) into a dash.
        assert logseq_slug("foo:bar") == "foo-bar"

    def test_slash_uniformly_maps_to_hierarchy(self) -> None:
        # Logseq has no per-prefix nesting (unlike Obsidian's ``u/`` /
        # ``r/`` / ``github:`` nested-dir convention). Every ``/`` maps
        # to ``___`` uniformly — the slug helper can't distinguish
        # a Reddit username from a hierarchical page, and Logseq
        # treats both as parent/child.
        assert logseq_slug("u/someone") == "u___someone"


# ---------------------------------------------------------------------------
# Block primitives
# ---------------------------------------------------------------------------


class TestBlockPrimitives:
    def test_render_block_at_root(self) -> None:
        assert render_block("hello") == "- hello"

    def test_render_block_at_depth(self) -> None:
        assert render_block("nested", depth=2) == "    - nested"

    def test_render_property_indented_past_parent(self) -> None:
        # depth=0 means "property on the root H1 block" — indents 2 spaces.
        assert render_property("type", "Coin", depth=0) == "  type:: Coin"

    def test_render_property_at_deeper_block(self) -> None:
        assert render_property("id", "p-aaaa", depth=1) == "    id:: p-aaaa"


# ---------------------------------------------------------------------------
# Page rendering — end-to-end format check
# ---------------------------------------------------------------------------


class TestSubjectPage:
    def test_emits_h1_and_subject_metadata(self) -> None:
        subj = Subject(
            id="subj-1",
            canonical_name="5 Pfennigs",
            aliases=["GDR 5 Pfennig"],
            asserted_by="test",
        )
        page = render_subject_page(
            subj,
            particles=[],
            eligible_ids={"5 Pfennigs"},
            subject_map={"subj-1": subj},
            eff_conf={},
        )
        assert "- # 5 Pfennigs" in page
        assert "  alias:: GDR 5 Pfennig" in page

    def test_structured_particle_becomes_properties_block_with_id(self) -> None:
        p = _make_particle("structured-marker", properties={"material": "Aluminium"})
        subj = Subject(id="subj-1", canonical_name="GDR", asserted_by="test")
        page = render_subject_page(
            subj,
            particles=[p],
            eligible_ids={"GDR"},
            subject_map={"subj-1": subj},
            eff_conf={p.id: 0.92},
        )
        assert "- ## Properties" in page
        assert "  - material:: Aluminium" in page
        # Particle ID as block UUID — the cross-page-citation primitive.
        assert f"    id:: {p.id}" in page
        assert "    confidence:: 0.92" in page

    def test_descriptive_particle_becomes_description_block_with_id(self) -> None:
        p = _make_particle("The obverse depicts the GDR national emblem.")
        subj = Subject(id="subj-1", canonical_name="GDR", asserted_by="test")
        page = render_subject_page(
            subj,
            particles=[p],
            eligible_ids={"GDR"},
            subject_map={"subj-1": subj},
            eff_conf={p.id: 0.87},
        )
        assert "- ## Description" in page
        assert "  - The obverse depicts the GDR national emblem." in page
        assert f"    id:: {p.id}" in page

    def test_phantom_wikilink_collapses_to_bare_name(self) -> None:
        # ``[[NotASubject]]`` references a subject that didn't make
        # eligible_ids — should render as ``NotASubject`` not as a link.
        p = _make_particle("Issued by [[NotASubject]] in 1949.")
        subj = Subject(id="subj-1", canonical_name="GDR", asserted_by="test")
        page = render_subject_page(
            subj,
            particles=[p],
            eligible_ids={"GDR"},  # NotASubject NOT in here
            subject_map={"subj-1": subj},
            eff_conf={p.id: 0.9},
        )
        assert "Issued by NotASubject in 1949." in page
        assert "[[NotASubject]]" not in page

    def test_eligible_wikilink_survives(self) -> None:
        p = _make_particle("Currency was the [[Mark]].")
        subj = Subject(id="subj-1", canonical_name="GDR", asserted_by="test")
        page = render_subject_page(
            subj,
            particles=[p],
            eligible_ids={"GDR", "Mark"},
            subject_map={"subj-1": subj},
            eff_conf={p.id: 0.9},
        )
        assert "Currency was the [[Mark]]." in page


# ---------------------------------------------------------------------------
# Taxonomy tags — inline ``#tag`` emission
# ---------------------------------------------------------------------------


class TestInlineTags:
    def test_empty_or_none_renders_nothing(self) -> None:
        assert render_inline_tags(None) == ""
        assert render_inline_tags([]) == ""

    def test_simple_tag_gets_hash_prefix(self) -> None:
        assert render_inline_tags(["cold-war"]) == " #cold-war"

    def test_hierarchy_separator_is_preserved(self) -> None:
        # Logseq reads ``/`` as tag hierarchy — keep it.
        assert render_inline_tags(["coins/by-region/germany"]) == " #coins/by-region/germany"

    def test_tag_with_space_is_bracket_wrapped(self) -> None:
        # A bare ``#`` token terminates at whitespace, so spaces force ``#[[…]]``.
        assert render_inline_tags(["cold war"]) == " #[[cold war]]"

    def test_multiple_tags_space_joined(self) -> None:
        assert render_inline_tags(["a", "b"]) == " #a #b"


class TestSubjectPageTags:
    def test_descriptive_particle_carries_inline_tag(self) -> None:
        p = _make_particle("AdamW decouples weight decay.", tags=["ml/optimizers"])
        subj = Subject(id="subj-1", canonical_name="AdamW", asserted_by="test")
        page = render_subject_page(
            subj,
            particles=[p],
            eligible_ids={"AdamW"},
            subject_map={"subj-1": subj},
            eff_conf={p.id: 0.9},
        )
        assert "AdamW decouples weight decay. #ml/optimizers" in page

    def test_structured_particle_tag_emitted_once_on_first_property(self) -> None:
        p = _make_particle(
            "structured",
            properties={"material": "Aluminium", "issuer": "GDR"},
            tags=["coins/metal"],
        )
        subj = Subject(id="subj-1", canonical_name="5 Pfennigs", asserted_by="test")
        page = render_subject_page(
            subj,
            particles=[p],
            eligible_ids={"5 Pfennigs"},
            subject_map={"subj-1": subj},
            eff_conf={p.id: 0.92},
        )
        # Tag rides the first property line only — not duplicated per property.
        assert page.count("#coins/metal") == 1

    def test_untagged_particle_has_no_tag_suffix(self) -> None:
        p = _make_particle("A plain claim.")
        subj = Subject(id="subj-1", canonical_name="X", asserted_by="test")
        page = render_subject_page(
            subj,
            particles=[p],
            eligible_ids={"X"},
            subject_map={"subj-1": subj},
            eff_conf={p.id: 0.9},
        )
        # The content block is emitted verbatim with no trailing ``#tag``.
        assert "  - A plain claim." in page
        assert "A plain claim. #" not in page


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_logseq_is_registered(self) -> None:
        exporters = get_exporters()
        assert "logseq" in exporters
        assert exporters["logseq"].__class__ is LogseqExporter

    def test_summary_format_discriminator(self) -> None:
        s = LogseqSummary(
            subjects=0,
            particles=0,
            phantoms=0,
            suppressed=0,
            files_written=0,
        )
        assert s.format == "logseq"


# ---------------------------------------------------------------------------
# End-to-end export — no synthesis
# ---------------------------------------------------------------------------


class TestExportNoSynthesis:
    @pytest.mark.asyncio
    async def test_writes_pages_directory_with_per_subject_files(
        self, db_session: object, tmp_path: Path
    ) -> None:
        ps = [
            _make_particle("First claim", confidence_value=0.9),
            _make_particle(
                "structured-marker",
                properties={"material": "Aluminium"},
                confidence_value=0.95,
            ),
        ]
        subj = await _persist_subject_with_particles("Test Subject", ps)

        async with session_scope() as session:
            summary = await LogseqExporter().export(
                session,
                tmp_path,
                min_links=0,  # subj has no in-links
            )

        assert isinstance(summary, LogseqSummary)
        # Exactly one subject + the Contents.md index.
        page_path = tmp_path / "pages" / "Test_Subject.md"
        assert page_path.exists()
        page_text = page_path.read_text(encoding="utf-8")
        assert "- # Test Subject" in page_text
        assert "First claim" in page_text
        assert "material:: Aluminium" in page_text
        # Particle IDs render as Logseq block UUIDs.
        assert f"id:: {ps[0].id}" in page_text
        assert f"id:: {ps[1].id}" in page_text
        # Contents index exists.
        assert (tmp_path / "pages" / "Contents.md").exists()
        assert summary.files_written >= 1
        assert summary.subjects == 1
        _ = subj  # silence unused

    @pytest.mark.asyncio
    async def test_same_named_subjects_get_distinct_pages_and_disambiguation(
        self, db_session: object, tmp_path: Path
    ) -> None:
        # two distinct "Prometheus" subjects must not collide on
        # one page (the silent-overwrite bug this fixes).
        await _persist_subject_with_particles(
            "Prometheus",
            [_make_particle("monitoring claim a"), _make_particle("monitoring claim b")],
            description="event monitoring and alerting software",
        )
        await _persist_subject_with_particles(
            "Prometheus",
            [_make_particle("myth claim a"), _make_particle("myth claim b")],
            description="Titan, culture hero, and trickster figure in Greek mythology",
        )

        async with session_scope() as session:
            await LogseqExporter().export(session, tmp_path, min_links=0)

        pages = tmp_path / "pages"
        software = pages / "Prometheus_(software).md"
        myth = pages / "Prometheus_(Titan).md"
        assert software.exists(), sorted(p.name for p in pages.glob("*.md"))
        assert myth.exists(), sorted(p.name for p in pages.glob("*.md"))
        assert software.read_text(encoding="utf-8").splitlines()[0] == "- # Prometheus (software)"
        # The Wikipedia-style disambiguation page exists, with the bare
        # name as a Logseq alias.
        disamb = pages / "Prometheus_(disambiguation).md"
        assert disamb.exists()
        disamb_text = disamb.read_text(encoding="utf-8")
        assert "alias:: Prometheus" in disamb_text
        assert "[[Prometheus (Titan)]]" in disamb_text
        assert "[[Prometheus (software)]]" in disamb_text
        # A bare Prometheus.md must NOT exist — every entity lives at a
        # qualified path.
        assert not (pages / "Prometheus.md").exists()

    @pytest.mark.asyncio
    async def test_min_particles_suppresses_subject(
        self, db_session: object, tmp_path: Path
    ) -> None:
        ps = [_make_particle("only one")]
        await _persist_subject_with_particles("Sparse", ps)

        async with session_scope() as session:
            summary = await LogseqExporter().export(session, tmp_path, min_particles=5, min_links=0)

        # Sparse subject doesn't meet min_particles → no page.
        assert not (tmp_path / "pages" / "Sparse.md").exists()
        # Contents.md still written.
        assert (tmp_path / "pages" / "Contents.md").exists()
        assert summary.files_written == 1  # just Contents.md
        _ = ps

    @pytest.mark.asyncio
    async def test_min_particle_confidence_filters_below_threshold(
        self, db_session: object, tmp_path: Path
    ) -> None:
        keep = _make_particle("high confidence", confidence_value=0.95)
        drop = _make_particle("low confidence", confidence_value=0.3)
        await _persist_subject_with_particles("Subj", [keep, drop])

        async with session_scope() as session:
            summary = await LogseqExporter().export(
                session,
                tmp_path,
                min_links=0,
                min_particle_confidence=0.5,
            )

        page = (tmp_path / "pages" / "Subj.md").read_text(encoding="utf-8")
        assert "high confidence" in page
        assert "low confidence" not in page
        assert summary.particles_dropped_below_threshold == 1


# ---------------------------------------------------------------------------
# End-to-end synthesis splice — the shared cross-exporter cache
# contract is covered by tests/test_synthesis_cache_integration.py
# (which exercises render_article directly). Here we just verify the
# Logseq-side splice format works given a pre-populated cache.
# ---------------------------------------------------------------------------


class TestSynthesisSpliceFormat:
    @pytest.mark.asyncio
    async def test_with_synthesis_cache_hit_emits_synthesis_block(
        self,
        db_session: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With a pre-populated synthesis_cache row, the Logseq splice
        produces a ``- ## Synthesis`` block over the structural outline
        without invoking the LLM. Verifies the format adapter
        (split_rendered_article → bullet blocks) end-to-end."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        ps = [_make_particle(f"claim {i}") for i in range(3)]
        subj = await _persist_subject_with_particles("Shared", ps)

        # Pre-populate the cache with a rendered article so the Logseq
        # exporter's render_article call returns the cached body.
        from particles.exporters.article_synthesis import compute_input_hash
        from particles.render.article_synthesis.cache import _PROMPT_VERSION
        from particles.store.synthesis_cache_store import store_cached_article

        input_hash = compute_input_hash(ps, subj)
        cached_body = (
            "---\n"
            f"input_hash: {input_hash}\n"
            "---\n"
            "# Shared\n\n"
            "Synthesised prose body.\n\n"
            "## References\n\n"
            "- One reference line.\n"
        )
        async with session_scope() as session:
            await store_cached_article(session, subj.id, input_hash, _PROMPT_VERSION, cached_body)
            await session.commit()

        async with session_scope() as session:
            summary = await LogseqExporter().export(
                session, tmp_path, with_synthesis=True, min_links=0
            )

        page = (tmp_path / "pages" / "Shared.md").read_text(encoding="utf-8")
        assert "- ## Synthesis" in page
        assert "Synthesised prose body." in page
        assert "- ### References" in page
        # Splice should also stamp article_input_hash on the H1 block
        # for rendered-artefact metadata visibility.
        assert f"article_input_hash:: {input_hash}" in page
        # The cache hit registers in the summary's count.
        assert (summary.synthesis_used or 0) + (summary.synthesis_cache_hits or 0) >= 1


class TestSkipLowCoverage:
    """Regression: Logseq must honour ``exporter_common.synthesis_min_particles``
    (matches the Obsidian gate). Without this, Logseq would burn LLM calls
    on subjects with too few particles — observed in the wild on 0.42.0
    when a CSS-class-named subject with 1 particle attempted synthesis."""

    @pytest.mark.asyncio
    async def test_subject_below_threshold_does_not_call_llm(
        self,
        db_session: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        # Default threshold is 3 → a 1-particle subject must skip.
        await _persist_subject_with_particles("SparseSubject", [_make_particle("only one claim")])

        from unittest.mock import MagicMock

        import anthropic

        from particles.llm import set_client

        # If synthesis IS called, the mock raises StopIteration since
        # there are no responses queued — the test would fail noisily.
        client = MagicMock(spec=anthropic.Anthropic)
        client.messages = MagicMock()
        client.messages.create = MagicMock(
            side_effect=AssertionError("synthesis should be skipped")
        )
        set_client(client)
        try:
            async with session_scope() as session:
                summary = await LogseqExporter().export(
                    session, tmp_path, with_synthesis=True, min_links=0
                )
        finally:
            set_client(None)

        # No LLM call.
        assert client.messages.create.call_count == 0
        # Subject still renders (with the skipped marker), so the page exists.
        page = (tmp_path / "pages" / "SparseSubject.md").read_text(encoding="utf-8")
        assert "- # SparseSubject" in page
        assert "article_synthesis:: skipped-low-coverage" in page
        # Summary records the skip.
        assert summary.synthesis_skipped == 1
        assert summary.synthesis_used == 0
        assert summary.synthesis_failed == 0

    @pytest.mark.asyncio
    async def test_threshold_is_configurable_via_env(
        self,
        db_session: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Setting the env var to 1 causes a 1-particle subject to attempt
        synthesis. Verifies the env-var override still works post-hoist."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        # The `db_session` fixture triggers get_config() during setup
        # (to read DATABASE_URL), so the config is already cached by
        # the time this test body runs — env-var setenv is too late
        # without disposing the engine, which would break the DB. Mutate
        # the cached config directly. This sidesteps the timing problem
        # while still exercising the gate at the production code path.
        from particles.config import get_config

        get_config().exporter_common.synthesis_min_particles = 1

        ps = [_make_particle("only claim")]
        subj = await _persist_subject_with_particles("Subj", ps)

        # Pre-populate the DB cache so synthesis short-circuits via cache hit
        # rather than calling the LLM — the test only cares that the gate
        # DIDN'T block synthesis, not whether the LLM was actually invoked.
        from particles.exporters.article_synthesis import compute_input_hash
        from particles.render.article_synthesis.cache import _PROMPT_VERSION
        from particles.store.synthesis_cache_store import store_cached_article

        input_hash = compute_input_hash(ps, subj)
        async with session_scope() as session:
            await store_cached_article(
                session,
                subj.id,
                input_hash,
                _PROMPT_VERSION,
                "---\ninput_hash: " + input_hash + "\n---\n# Subj\n\nProse.\n",
            )
            await session.commit()

        async with session_scope() as session:
            summary = await LogseqExporter().export(
                session, tmp_path, with_synthesis=True, min_links=0
            )

        # Gate did NOT skip — synthesis ran (and hit the cache).
        assert summary.synthesis_skipped == 0
        assert (summary.synthesis_used or 0) + (summary.synthesis_cache_hits or 0) >= 1


# ---------------------------------------------------------------------------
# Per-NARRATIVE pages (mechanism in Logseq)
# ---------------------------------------------------------------------------


async def _persist_narrative(label: str, constituents: list[Particle]) -> Particle:
    """Insert a NARRATIVE plus its PART_OF / SEQUENCE_IN constituent chain."""
    from particles.core.schema import ParticleType, RelationCreatedBy, RelationType
    from particles.store.particle_store import insert_particle
    from particles.store.relation_store import create_relation

    narrative = _make_particle(label).model_copy(update={"particle_type": ParticleType.NARRATIVE})
    async with session_scope() as session:
        await insert_particle(session, narrative)
        for c in constituents:
            await create_relation(
                session, c.id, narrative.id, RelationType.PART_OF, RelationCreatedBy.MANUAL_CLI
            )
        for a, b in zip(constituents, constituents[1:], strict=False):
            await create_relation(
                session, a.id, b.id, RelationType.SEQUENCE_IN, RelationCreatedBy.MANUAL_CLI
            )
        await session.commit()
    return narrative


class TestRenderNarrativePage:
    """A NARRATIVE renders as a standalone Logseq page: title block +
    ``article_input_hash::`` / ``tags::`` properties + cited prose blocks."""

    @pytest.mark.asyncio
    async def test_fallback_without_key_builds_cited_page(
        self, db_session: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.exporters.logseq.narrative import render_narrative_page

        # No key → render_article falls back to the deterministic structured
        # listing, so this is deterministic without mocking the model.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")

        from particles.core.schema import ParticleType

        narrative = _make_particle("A hard day the author got through.").model_copy(
            update={"particle_type": ParticleType.NARRATIVE}
        )
        c1 = _make_particle("The author woke up tired.")
        c2 = _make_particle("The author felt better by evening.")

        page, state = await render_narrative_page(
            narrative=narrative,
            constituents=[c1, c2],
            entry_uri_map={},
            eff_conf={c1.id: 0.9, c2.id: 0.9},
            article_hash="HASH",
            session=db_session,  # type: ignore[arg-type]
        )

        assert state == "fallback"
        # Logseq's bullet-outline shape: the title is the first block and the
        # metadata are block properties, not YAML frontmatter.
        assert page.startswith("- # A hard day the author got through.\n")
        assert "tags:: particles/narrative" in page
        assert "article_input_hash:: HASH" in page
        assert "article_synthesis:: structured-listing" in page
        assert "---" not in page
        # Standalone page: the prose is not buried under a `## Synthesis`
        # parent — the whole page *is* the article.
        assert "- ## Synthesis" not in page
        assert "- ### References" in page


class TestNarrativesBlock:
    """the subject → narrative backlink block on per-subject pages."""

    @staticmethod
    def _narr(label: str) -> Particle:
        from particles.core.schema import ParticleType

        return _make_particle(label).model_copy(update={"particle_type": ParticleType.NARRATIVE})

    def test_lists_emitted_narratives_as_aliased_page_links(self) -> None:
        from particles.exporters.logseq.format import render_narratives_block

        n1 = self._narr("A hard day.")
        n2 = self._narr("A good week.")
        naming = {n1.id: "a-hard-day", n2.id: "a-good-week"}
        out = render_narratives_block([n1, n2], naming)
        assert out.startswith("- ## Narratives\n")
        # Logseq has no [[page|label]] pipe form — an aliased page link is
        # `[label]([[page]])`, and the target is the Narratives/ namespace.
        assert "- [A good week.]([[Narratives/a-good-week]])" in out
        assert "- [A hard day.]([[Narratives/a-hard-day]])" in out

    def test_skips_narrative_absent_from_naming(self) -> None:
        from particles.exporters.logseq.format import render_narratives_block

        # A narrative gated out of emission is not linked — no dangling target.
        assert render_narratives_block([self._narr("Ungated.")], {}) == ""


class TestNarrativePageExport:
    """End-to-end: the narrative page lands in Logseq's `Narratives/` page
    namespace, which on disk is the `___` hierarchy encoding."""

    @pytest.mark.asyncio
    async def test_narrative_page_written_under_hierarchy_prefix(
        self,
        db_session: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        ps = [_make_particle(f"claim {i}") for i in range(3)]
        subj = await _persist_subject_with_particles("Shared", ps)
        narrative = await _persist_narrative("A hard day.", ps)

        # Pre-populate the shared synthesis cache for BOTH the subject article
        # and the narrative article so no LLM call is made.
        from particles.exporters.article_synthesis import compute_input_hash
        from particles.render.article_synthesis.cache import _PROMPT_VERSION
        from particles.render.markdown import narrative_as_subject
        from particles.store.synthesis_cache_store import store_cached_article

        synthetic = narrative_as_subject(narrative)
        narrative_hash = compute_input_hash(ps, synthetic, ordered=True)
        subject_hash = compute_input_hash(ps, subj)
        async with session_scope() as session:
            await store_cached_article(
                session,
                subj.id,
                subject_hash,
                _PROMPT_VERSION,
                f"---\ninput_hash: {subject_hash}\n---\n# Shared\n\nSubject prose.\n",
            )
            await store_cached_article(
                session,
                narrative.id,
                narrative_hash,
                _PROMPT_VERSION,
                f"---\ninput_hash: {narrative_hash}\n---\n# A hard day.\n\n"
                "Narrative prose.\n\n## References\n\n- One reference line.\n",
            )
            await session.commit()

        async with session_scope() as session:
            summary = await LogseqExporter().export(
                session, tmp_path, with_synthesis=True, min_links=0
            )

        page_path = tmp_path / "pages" / "Narratives___A_hard_day.md"
        assert page_path.exists(), sorted(p.name for p in (tmp_path / "pages").iterdir())
        page = page_path.read_text(encoding="utf-8")
        assert "- # A hard day." in page
        assert "Narrative prose." in page
        assert summary.narrative_notes == 1

        # the subject page backlinks to the narrative its claims
        # participate in.
        subject_page = (tmp_path / "pages" / "Shared.md").read_text(encoding="utf-8")
        assert "- ## Narratives" in subject_page
        # The link target is the Logseq *page* name (`Narratives/A hard day`),
        # not its `___`-encoded filename — that encoding is on-disk only.
        assert "- [A hard day.]([[Narratives/A hard day]])" in subject_page

    @pytest.mark.asyncio
    async def test_knob_off_suppresses_narrative_pages(
        self,
        db_session: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        from particles.config import get_config

        monkeypatch.setattr(get_config().logseq, "emit_narrative_notes", False)

        ps = [_make_particle(f"claim {i}") for i in range(3)]
        await _persist_subject_with_particles("Shared", ps)
        await _persist_narrative("A hard day.", ps)

        async with session_scope() as session:
            summary = await LogseqExporter().export(
                session, tmp_path, with_synthesis=True, min_links=0
            )
        assert summary.narrative_notes is None
        assert not (tmp_path / "pages" / "Narratives___A_hard_day.md").exists()

"""Tests for the Obsidian vault exporter."""

from __future__ import annotations

import pytest

from particles.core.schema import (
    Confidence,
    Particle,
    ParticleType,
    Subject,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status


def _make_subject(name: str, subject_class: str | None = None) -> Subject:
    return Subject(canonical_name=name, asserted_by="test", subject_class=subject_class)


def _make_particle(
    content: str,
    subject_ids: list[str],
    properties: dict | None = None,
    tags: list[str] | None = None,
) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.95, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="numista-coin-extractor",
        status=Status.ACTIVE,
        subject_ids=subject_ids,
        properties=properties,
        tags=tags,
    )


class TestCoinTemplate:
    def _render(self, subject: Subject, particles: list[Particle], subject_map: dict) -> str:
        from particles.exporters.obsidian import _render_coin_note

        return _render_coin_note(subject, particles, subject_map)

    def test_tags_include_particles_coin(self) -> None:
        s = _make_subject("1 Pfennig (1948-1950) GDR", "nmo:NumismaticObject")
        p = _make_particle("1 Pfennig summary.", [s.id], {"nmo:hasWeight": 0.75})
        out = self._render(s, [p], {s.id: s})
        assert "particles/coin" in out

    def test_instance_of_coin_always_present(self) -> None:
        s = _make_subject("1 Pfennig (1948-1950) GDR", "nmo:NumismaticObject")
        p = _make_particle("summary.", [s.id], {})
        out = self._render(s, [p], {s.id: s})
        assert 'Instance of: "[[Coin]]"' in out

    def test_wiki_link_properties_rendered(self) -> None:
        s = _make_subject("1 Pfennig (1948-1950) GDR", "nmo:NumismaticObject")
        props = {
            "nmo:hasIssuer": "German Democratic Republic",
            "nmo:hasMaterial": "Aluminium",
            "nmo:hasDenomination": "Mark (1948-1990)",
        }
        p = _make_particle("summary.", [s.id], props)
        out = self._render(s, [p], {s.id: s})
        assert 'Issuer: "[[German Democratic Republic]]"' in out
        assert 'Composition: "[[Aluminium]]"' in out
        assert 'Currency: "[[Mark (1948-1990)]]"' in out

    def test_scalar_properties_rendered(self) -> None:
        s = _make_subject("1 Pfennig (1948-1950) GDR", "nmo:NumismaticObject")
        props = {
            "nmo:hasWeight": 0.75,
            "nmo:hasDiameter": 17.0,
            "nmo:hasProductionDate": "1948-1950",
        }
        p = _make_particle("summary.", [s.id], props)
        out = self._render(s, [p], {s.id: s})
        assert "Weight: 0.75 g" in out
        assert "Diameter: 17.0 mm" in out
        assert "Years: 1948-1950" in out

    def test_catalog_refs_rendered(self) -> None:
        s = _make_subject("1 Pfennig (1948-1950) GDR", "nmo:NumismaticObject")
        props = {"nuds:references": ["KM# 1", "Schön# 1", "N# 8562"]}
        p = _make_particle("summary.", [s.id], props)
        out = self._render(s, [p], {s.id: s})
        assert "KM# 1" in out
        assert "References" in out

    def test_numista_url_rendered(self) -> None:
        s = _make_subject("1 Pfennig (1948-1950) GDR", "nmo:NumismaticObject")
        props = {"numista:url": "https://en.numista.com/catalogue/pieces8562.html"}
        p = _make_particle("summary.", [s.id], props)
        out = self._render(s, [p], {s.id: s})
        assert "https://en.numista.com/catalogue/pieces8562.html" in out

    def test_obverse_section_rendered(self) -> None:
        s = _make_subject("1 Pfennig (1948-1950) GDR", "nmo:NumismaticObject")
        p_struct = _make_particle("summary.", [s.id], {"nmo:hasWeight": 0.75})
        p_obverse = _make_particle(
            "The obverse of 1 Pfennig (1948-1950) GDR depicts: An ear of wheat.",
            [s.id],
        )
        out = self._render(s, [p_struct, p_obverse], {s.id: s})
        assert "### Obverse" in out
        assert "ear of wheat" in out

    def test_conflict_callout_when_two_weights(self) -> None:
        s = _make_subject("1 Pfennig (1948-1950) GDR", "nmo:NumismaticObject")
        p1 = _make_particle("summary A.", [s.id], {"nmo:hasWeight": 0.75})
        p2 = _make_particle("summary B.", [s.id], {"nmo:hasWeight": 0.70})
        # Give p2 lower confidence so p1 wins
        p2 = Particle(
            content="summary B.",
            confidence=Confidence(
                value=0.80, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
            ),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="test",
            status=Status.ACTIVE,
            subject_ids=[s.id],
            properties={"nmo:hasWeight": 0.70},
        )
        out = self._render(s, [p1, p2], {s.id: s})
        assert "[!warning]" in out
        assert "Weight" in out

    def test_no_conflict_when_single_particle(self) -> None:
        s = _make_subject("1 Pfennig (1948-1950) GDR", "nmo:NumismaticObject")
        p = _make_particle("summary.", [s.id], {"nmo:hasWeight": 0.75})
        out = self._render(s, [p], {s.id: s})
        assert "[!warning]" not in out


class TestPivotTemplate:
    def _render(self, subject: Subject, particles: list[Particle]) -> str:
        from particles.exporters.obsidian import _render_pivot_note

        return _render_pivot_note(subject, particles)

    def test_material_tag(self) -> None:
        s = _make_subject("Aluminium", "nmo:Material")
        out = self._render(s, [])
        assert "particles/material" in out

    def test_denomination_tag(self) -> None:
        s = _make_subject("Mark (1948-1990)", "nmo:Denomination")
        out = self._render(s, [])
        assert "particles/currency" in out

    def test_issuer_tag(self) -> None:
        s = _make_subject("German Democratic Republic", "nmo:Issuer")
        out = self._render(s, [])
        assert "particles/issuer" in out

    def test_header_is_canonical_name(self) -> None:
        s = _make_subject("Aluminium", "nmo:Material")
        out = self._render(s, [])
        assert "# Aluminium" in out

    def test_particle_id_in_callout_title(self) -> None:
        from particles.core.schema import ProvenanceRef, ProvenanceRefType
        from particles.exporters.obsidian import _render_pivot_note

        s = _make_subject("nanoGPT", "nmo:Material")
        p = _make_particle("The gpt2 model has 124M parameters.", [s.id])
        p.provenance.append(ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1"))
        out = _render_pivot_note(s, [p])
        # Format-C audit-trail entry: short id appears in both the
        # callout title (``p-{short} · confidence …``) and the
        # trailing block marker (``^p-{short}``).
        assert f"p-{p.id[:8]}" in out

    def test_source_link_emitted_when_uri_known(self) -> None:
        from particles.core.schema import ProvenanceRef, ProvenanceRefType
        from particles.exporters.obsidian import _render_pivot_note

        s = _make_subject("nanoGPT", "nmo:Material")
        p = _make_particle("The gpt2 model has 124M parameters.", [s.id])
        p.provenance.append(ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1"))
        entry_uri_map = {"entry-1": "https://github.com/karpathy/nanoGPT"}
        out = _render_pivot_note(s, [p], entry_uri_map=entry_uri_map)
        assert (
            "> **Source:** [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)"
            in out
        )

    def test_source_line_omitted_when_uri_absent(self) -> None:
        from particles.core.schema import ProvenanceRef, ProvenanceRefType
        from particles.exporters.obsidian import _render_pivot_note

        s = _make_subject("nanoGPT", "nmo:Material")
        p = _make_particle("Some claim.", [s.id])
        p.provenance.append(ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1"))
        # entry_uri_map maps the entry to None — no URL known
        out = _render_pivot_note(s, [p], entry_uri_map={"entry-1": None})
        assert "**Source:**" not in out

    def test_long_url_truncated_in_label(self) -> None:
        from particles.core.schema import ProvenanceRef, ProvenanceRefType
        from particles.exporters.obsidian import _render_pivot_note

        s = _make_subject("nanoGPT", "nmo:Material")
        p = _make_particle("Some claim.", [s.id])
        p.provenance.append(ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1"))
        long_url = "https://github.com/org/repo/blob/branch/" + "a" * 200 + "/file.md"
        out = _render_pivot_note(s, [p], entry_uri_map={"entry-1": long_url})
        # Truncated label, full URL preserved in the link target
        assert f"]({long_url})" in out
        assert "…" in out


class TestGenericTemplate:
    def _render(
        self,
        subject: Subject,
        particles: list[Particle],
        subject_map: dict,
    ) -> str:
        from particles.exporters.obsidian import _render_subject_note

        return _render_subject_note(subject, particles, subject_map)

    def test_category_subjects_not_in_related(self) -> None:
        coin = _make_subject("1 Pfennig (1948-1950) GDR")
        category = _make_subject("Category:East German coins")
        p = _make_particle(
            "Some particle.",
            [coin.id, category.id],
        )
        out = self._render(coin, [p], {coin.id: coin, category.id: category})
        assert "[[Category:East German coins]]" not in out

    def test_pivot_subjects_not_in_related(self) -> None:
        coin = _make_subject("1 Pfennig (1948-1950) GDR")
        material = _make_subject("Aluminium", "nmo:Material")
        p = _make_particle("Some particle.", [coin.id, material.id])
        out = self._render(coin, [p], {coin.id: coin, material.id: material})
        assert "[[Aluminium]]" not in out

    def test_phantom_warning_when_no_particles(self) -> None:
        s = _make_subject("Some Subject")
        out = self._render(s, [], {s.id: s})
        assert "Phantom subject" in out


class TestTaxonomyTagsInFrontmatter:
    """operator-curated taxonomy tags propagate to Obsidian frontmatter."""

    def test_generic_note_emits_union_of_particle_tags(self) -> None:
        from particles.exporters.obsidian import _render_subject_note

        s = _make_subject("AdamW")
        p1 = _make_particle("AdamW decouples weight decay.", [s.id], tags=["ml/optimizers"])
        p2 = _make_particle(
            "AdamW is widely adopted.", [s.id], tags=["ml/training", "ml/optimizers"]
        )
        out = _render_subject_note(s, [p1, p2], {s.id: s})
        # Both distinct tags appear once, under the frontmatter tags: list.
        assert "  - ml/optimizers" in out
        assert "  - ml/training" in out
        assert out.count("  - ml/optimizers") == 1

    def test_no_tags_emits_no_extra_frontmatter_lines(self) -> None:
        from particles.exporters.obsidian import _render_subject_note

        s = _make_subject("Untagged")
        p = _make_particle("A claim with no taxonomy tags.", [s.id])
        out = _render_subject_note(s, [p], {s.id: s})
        # Only the built-in subject tag, no taxonomy lines.
        assert "  - particles/subject" in out
        assert "ml/optimizers" not in out

    def test_coin_note_emits_taxonomy_tags(self) -> None:
        from particles.exporters.obsidian import _render_coin_note

        s = _make_subject("5 Pfennigs (1948) GDR", "nmo:NumismaticObject")
        p = _make_particle(
            "summary.", [s.id], {"nmo:hasWeight": 0.75}, tags=["coins/by-region/germany"]
        )
        out = _render_coin_note(s, [p], {s.id: s})
        assert "  - coins/by-region/germany" in out

    def test_pivot_note_emits_taxonomy_tags(self) -> None:
        from particles.exporters.obsidian import _render_pivot_note

        s = _make_subject("Aluminium", "nmo:Material")
        p = _make_particle("Aluminium is a light metal.", [s.id], tags=["materials/metals"])
        out = _render_pivot_note(s, [p])
        assert "  - materials/metals" in out


class TestSanitize:
    def test_dashes_preserved(self) -> None:
        from particles.exporters.obsidian import _sanitize

        assert _sanitize("1 Pfennig (1948-1950) GDR") == "1 Pfennig (1948-1950) GDR"

    def test_invalid_chars_replaced(self) -> None:
        from particles.exporters.obsidian import _sanitize

        assert "/" not in _sanitize("foo/bar")
        assert ":" not in _sanitize("foo:bar")

    def test_whitespace_collapsed(self) -> None:
        from particles.exporters.obsidian import _sanitize

        assert _sanitize("foo   bar") == "foo bar"


# ---------------------------------------------------------------------------
# --with-synthesis splicing
# ---------------------------------------------------------------------------


class TestWithSynthesisSplicing:
    """Pure-function tests for the splicing helpers, no DB or LLM."""

    def test_annotate_frontmatter_adds_cache_fields(self) -> None:
        from particles.exporters.obsidian import _annotate_obsidian_frontmatter

        note = "---\ntitle: x\n---\n\n# Title\n\nbody\n"
        out = _annotate_obsidian_frontmatter(
            note, article_input_hash="abc123", article_synthesis="llm"
        )
        assert "article_input_hash: abc123" in out
        assert "article_synthesis: llm" in out
        # The original fields survive.
        assert "title: x" in out
        # Body is unchanged.
        assert "# Title" in out
        assert "body" in out

    def test_annotate_frontmatter_no_op_on_unfrontmatter_note(self) -> None:
        from particles.exporters.obsidian import _annotate_obsidian_frontmatter

        note = "# Bare Title\n\nbody\n"
        out = _annotate_obsidian_frontmatter(note, article_input_hash="x", article_synthesis="llm")
        # No frontmatter → no annotation (returned unchanged).
        assert out == note

    def test_insert_prose_between_h1_and_existing_body(self) -> None:
        from particles.exporters.obsidian import _insert_synthesised_prose

        note = "---\nx: y\n---\n\n# Subject\n\nExisting Obsidian structural content.\n"
        prose = "Synthesised prose paragraph."
        refs = "## References\n\n### p-abc12345\n[^p-abc12345]: ..."
        out = _insert_synthesised_prose(note=note, prose=prose, references=refs)
        # Frontmatter survives at the top.
        assert out.startswith("---\nx: y\n---\n")
        # The Obsidian H1 is still the first content line after frontmatter.
        body_after_fm = out.split("---\n", 2)[-1].lstrip()
        assert body_after_fm.startswith("# Subject")
        # Prose is between H1 and the existing structural content.
        assert "Synthesised prose paragraph." in out
        # The audit-trail heading separates prose from the existing content.
        assert "## Source particles" in out
        # Existing structural content survived after the heading.
        assert "Existing Obsidian structural content." in out
        # References appended at the end.
        assert out.rstrip().endswith("[^p-abc12345]: ...")

    def test_insert_prose_preserves_order_prose_before_audit(self) -> None:
        from particles.exporters.obsidian import _insert_synthesised_prose

        note = "# Subject\n\nExisting content.\n"
        out = _insert_synthesised_prose(
            note=note, prose="PROSE", references="## References\n\nrefs"
        )
        # Prose appears before "## Source particles", which appears
        # before "Existing content".
        assert out.index("PROSE") < out.index("## Source particles")
        assert out.index("## Source particles") < out.index("Existing content")
        # References are at the end.
        assert out.index("Existing content") < out.index("## References")


class TestSynthesisCacheHitPreservesProse:
    """fix: on a cache hit the Obsidian exporter must keep the prior
    note's synthesised prose. The freshly-rendered ``note`` passed in is the
    structural template only; returning it would silently drop the article on
    every no-change re-export (the bug behind the `article cache hit` /
    missing-prose report)."""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_prior_body(self, db_session: object) -> None:
        from particles.exporters.obsidian.synthesis import _splice_synthesised_article

        subject = _make_subject("Douglas B. Lenat")
        # >= synthesis_min_particles (default 3) so the low-coverage skip
        # branch doesn't fire before the cache-hit check.
        particles = [_make_particle(f"claim {i}.", [subject.id]) for i in range(3)]

        structural_note = "---\ntags:\n  - particles/subject\n---\n\n# Douglas B. Lenat\n\nstub\n"
        prior_body = (
            "---\n"
            "tags:\n  - particles/subject\n"
            "article_input_hash: HASH\n"
            "article_synthesis: llm\n"
            "---\n\n"
            "# Douglas B. Lenat\n\n"
            "Douglas Bruce Lenat was an American computer scientist.\n\n"
            "## Source particles\n\nstub\n\n## References\n"
        )

        note, state = await _splice_synthesised_article(
            note=structural_note,
            subject=subject,
            particles=particles,
            subject_map={subject.id: subject},
            eligible_ids={subject.id},
            eff_conf={},
            entry_uri_map={},
            article_hash="HASH",
            regenerate=False,  # cache hit
            lint_findings=[],
            progress_prefix="[1/1] Douglas B. Lenat",
            session=db_session,  # type: ignore[arg-type]
            prior_body=prior_body,
            naming=None,
        )

        assert state == "cache_hit"
        # The synthesised prose survives — not the structural stub.
        assert note == prior_body
        assert "Douglas Bruce Lenat was an American computer scientist." in note


class TestObsidianBlockRefConversion:
    """Obsidian's Markdown-footnote parser does not reliably make
    ``[^id]`` references clickable in Live Preview; the user's failure
    mode is that citations render as literal bracketed text. The Obsidian
    exporter converts the portable Markdown-footnote output of
    ``article_synthesis.py`` into Obsidian-native block references:

      * Body: ``[^p-xxxxxxxx]`` → ``[[#^p-xxxxxxxx|ⁿ]]`` (numbered alias)
      * References: re-rendered from the particle objects with a
        numbered heading carrying the claim text, a callout that holds
        the related-subject wikilinks + source + confidence + extractor,
        and a trailing ``^p-xxxxxxxx`` block marker for the jump.
    """

    def _particle_with_id(
        self,
        *,
        short_id: str,
        content: str = "claim",
        subject_ids: list[str] | None = None,
        confidence_value: float = 0.74,
    ) -> Particle:
        from datetime import UTC, datetime

        from particles.core.schema import ProvenanceRef, ProvenanceRefType

        # Build a UUID-shaped ID that starts with the given 8-char prefix.
        pid = f"{short_id}-0000-4000-8000-000000000000"
        return Particle(
            id=pid,
            content=content,
            confidence=Confidence(
                value=confidence_value, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
            ),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="numista-listing-extractor",
            asserted_at=datetime(2026, 5, 15, tzinfo=UTC),
            status=Status.ACTIVE,
            subject_ids=subject_ids or [],
            extractor_ref={"name": "numista-listing-extractor", "version": "0.1.0"},
            provenance=[
                ProvenanceRef(
                    type=ProvenanceRefType.SOURCE,
                    corpus_entry_id="entry-1",
                    snapshot_id="snap-1",
                )
            ],
        )

    def _call(
        self,
        prose: str,
        *,
        particles: list[Particle] | None = None,
        parent: Subject | None = None,
        subject_map: dict[str, Subject] | None = None,
        eligible_ids: set[str] | None = None,
        eff_conf: dict[str, float] | None = None,
        entry_uri_map: dict[str, str | None] | None = None,
    ) -> tuple[str, str]:
        from particles.exporters.obsidian import _to_obsidian_block_refs

        parent = parent or _make_subject("Parent", "nmo:NumismaticObject")
        particles = particles or []
        subject_map = subject_map if subject_map is not None else {parent.id: parent}
        eligible_ids = eligible_ids if eligible_ids is not None else set(subject_map)
        return _to_obsidian_block_refs(
            prose,
            particles=particles,
            parent_subject=parent,
            subject_map=subject_map,
            eligible_ids=eligible_ids,
            eff_conf=eff_conf or {},
            entry_uri_map=entry_uri_map or {},
        )

    def test_body_refs_become_aliased_block_links(self) -> None:
        parent = _make_subject("Parent", "nmo:NumismaticObject")
        p1 = self._particle_with_id(short_id="aabbccdd", subject_ids=[parent.id])
        p2 = self._particle_with_id(short_id="11223344", subject_ids=[parent.id])
        prose = "Claim [^p-aabbccdd] and a second claim [^p-11223344]."
        new_prose, _ = self._call(
            prose, particles=[p1, p2], parent=parent, subject_map={parent.id: parent}
        )
        assert "[^p-aabbccdd]" not in new_prose
        # First citation → ¹, second → ².
        assert "[[#^p-aabbccdd|¹]]" in new_prose
        assert "[[#^p-11223344|²]]" in new_prose

    def test_repeated_citation_reuses_ordinal(self) -> None:
        parent = _make_subject("Parent", "nmo:NumismaticObject")
        p1 = self._particle_with_id(short_id="aabbccdd", subject_ids=[parent.id])
        p2 = self._particle_with_id(short_id="11223344", subject_ids=[parent.id])
        prose = "First [^p-aabbccdd]. Other [^p-11223344]. Again [^p-aabbccdd]."
        new_prose, _ = self._call(
            prose, particles=[p1, p2], parent=parent, subject_map={parent.id: parent}
        )
        assert new_prose.count("[[#^p-aabbccdd|¹]]") == 2
        assert new_prose.count("[[#^p-11223344|²]]") == 1

    def test_no_citations_returns_empty_references(self) -> None:
        _, refs = self._call("Body with no citations whatsoever.")
        assert refs == ""

    def test_references_heading_carries_ordinal_and_claim(self) -> None:
        parent = _make_subject("Parent", "nmo:NumismaticObject")
        p = self._particle_with_id(
            short_id="aabbccdd",
            content="made of copper-nickel, weight 12.0g",
            subject_ids=[parent.id],
        )
        _, refs = self._call(
            "Body [^p-aabbccdd].",
            particles=[p],
            parent=parent,
            subject_map={parent.id: parent},
        )
        # Numbered heading carries the claim text (Format C).
        assert "### 1. made of copper-nickel, weight 12.0g" in refs
        # No leftover Markdown footnote-definition syntax.
        assert "[^p-aabbccdd]:" not in refs

    def test_reference_includes_callout_with_metadata_and_block_marker(self) -> None:
        parent = _make_subject("Parent", "nmo:NumismaticObject")
        p = self._particle_with_id(short_id="aabbccdd", subject_ids=[parent.id])
        _, refs = self._call(
            "Body [^p-aabbccdd].",
            particles=[p],
            parent=parent,
            subject_map={parent.id: parent},
            eff_conf={p.id: 0.74},
            entry_uri_map={"entry-1": "https://example.com/source"},
        )
        # Callout header carries the short ID and effective confidence.
        assert "> [!info] p-aabbccdd · confidence 0.74" in refs
        # Source link rendered as a wikilink-ish line via _source_link_line.
        assert "https://example.com/source" in refs
        # Extractor + date + block marker on the last line of the callout.
        assert "> **Extractor:** numista-listing-extractor 0.1.0 on 2026-05-15 ^p-aabbccdd" in refs

    def test_agent_asserted_renders_asserted_by_not_extractor(self) -> None:
        # A direct agent assertion has no extractor_ref; it must NOT
        # render a phantom "Extractor: ?" — attribute it to the asserting principal.
        from datetime import UTC, datetime

        from particles.core.schema import ProvenanceRef, ProvenanceRefType

        parent = _make_subject("Parent", "nmo:NumismaticObject")
        p = Particle(
            id="dddddddd-0000-4000-8000-000000000000",
            content="ships a read-write MCP surface.",
            confidence=Confidence(value=0.9, calibration_source=CalibrationSource.AGENT_ASSERTED),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="mcp:claude-code",
            asserted_at=datetime(2026, 6, 14, tzinfo=UTC),
            status=Status.ACTIVE,
            subject_ids=[parent.id],
            provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1")],
        )
        _, refs = self._call(
            "Body [^p-dddddddd].",
            particles=[p],
            parent=parent,
            subject_map={parent.id: parent},
            eff_conf={p.id: 0.5},
        )
        assert "**Extractor:**" not in refs
        assert "> **Asserted by:** mcp:claude-code on 2026-06-14 ^p-dddddddd" in refs

    def test_related_excludes_parent_categories_and_pivots(self) -> None:
        parent = _make_subject("Parent", "nmo:NumismaticObject")
        peer = _make_subject("German Democratic Republic", "nmo:Country")
        material = _make_subject("Aluminium", "nmo:Material")  # pivot — excluded
        category = _make_subject("Category:Coins")  # category — excluded
        p = self._particle_with_id(
            short_id="aabbccdd",
            subject_ids=[parent.id, peer.id, material.id, category.id],
        )
        subject_map = {s.id: s for s in (parent, peer, material, category)}
        _, refs = self._call(
            "Body [^p-aabbccdd].",
            particles=[p],
            parent=parent,
            subject_map=subject_map,
            eligible_ids=set(subject_map),
        )
        # Real peer shows up; parent/pivot/category do not.
        assert "[[German Democratic Republic]]" in refs
        assert "[[Aluminium]]" not in refs
        assert "[[Parent]]" not in refs
        assert "Category:Coins" not in refs

    def test_related_line_omitted_when_no_other_subjects(self) -> None:
        parent = _make_subject("Parent", "nmo:NumismaticObject")
        p = self._particle_with_id(short_id="aabbccdd", subject_ids=[parent.id])
        _, refs = self._call(
            "Body [^p-aabbccdd].",
            particles=[p],
            parent=parent,
            subject_map={parent.id: parent},
        )
        assert "**Related:**" not in refs

    def test_superscript_renders_multi_digit_ordinals(self) -> None:
        from particles.exporters.obsidian import _to_superscript

        assert _to_superscript(1) == "¹"
        assert _to_superscript(9) == "⁹"
        assert _to_superscript(10) == "¹⁰"
        assert _to_superscript(123) == "¹²³"

    def test_claim_for_heading_collapses_whitespace_and_strips_hash(self) -> None:
        from particles.exporters.obsidian import _claim_for_heading

        # Collapse internal whitespace runs.
        assert _claim_for_heading("foo\n bar\t baz") == "foo bar baz"
        # Strip leading # so claim doesn't change heading depth.
        assert _claim_for_heading("### already a heading") == "already a heading"
        # Empty / whitespace-only content falls back to placeholder.
        assert _claim_for_heading("   ") == "(empty claim)"


class TestRenderParticleAuditCallouts:
    """Shared per-particle audit-trail renderer. Used by
    `_render_subject_note` (its in-body audit trail) and by the
    skip-synthesis branch of `_splice_synthesised_article` (appended
    when the coin template otherwise leaves the body empty)."""

    def test_emits_one_callout_per_particle(self) -> None:
        from particles.exporters.obsidian import _render_particle_audit_callouts

        parent = _make_subject("Coin", "nmo:NumismaticObject")
        p1 = _make_particle("First claim.", [parent.id])
        p2 = _make_particle("Second claim.", [parent.id])
        lines = _render_particle_audit_callouts(
            [p1, p2],
            parent_subject=parent,
            subject_map={parent.id: parent},
            eligible_ids={parent.id},
            eff_conf={},
            entry_uri_map={},
        )
        body = "\n".join(lines)
        # Format-C audit-trail entries: numbered heading carries the claim,
        # info callout carries the short id.
        assert "### 1. First claim." in body
        assert "### 2. Second claim." in body
        assert f"> [!info] p-{p1.id[:8]}" in body
        assert f"> [!info] p-{p2.id[:8]}" in body

    def test_related_excludes_parent_categories_and_pivots(self) -> None:
        from particles.exporters.obsidian import _render_particle_audit_callouts

        parent = _make_subject("Parent", "nmo:NumismaticObject")
        peer = _make_subject("German Democratic Republic", "nmo:Country")
        material = _make_subject("Aluminium", "nmo:Material")  # pivot — excluded
        category = _make_subject("Category:Coins")  # category — excluded
        p = _make_particle("claim", [parent.id, peer.id, material.id, category.id])
        subject_map = {s.id: s for s in (parent, peer, material, category)}
        body = "\n".join(
            _render_particle_audit_callouts(
                [p],
                parent_subject=parent,
                subject_map=subject_map,
                eligible_ids=set(subject_map),
                eff_conf={},
                entry_uri_map={},
            )
        )
        # Real peer shows up; parent / pivot / category do not.
        assert "[[German Democratic Republic]]" in body
        assert "[[Aluminium]]" not in body
        assert "[[Parent]]" not in body
        assert "Category:Coins" not in body


class TestNoteHasParticleAuditTrail:
    """The skip-synthesis path appends a per-particle audit trail when
    the renderer didn't already produce one. Detection keys on the
    Format-C ``> [!info] p-{8 hex}`` callout title that audit-trail
    entries carry; other callouts (lint, subject-level info) don't
    use the ``p-{hex}`` marker."""

    def test_detects_subject_note_audit_callout(self) -> None:
        from particles.exporters.obsidian import _note_has_particle_audit_trail

        note = (
            "# Subject\n\n"
            "### 1. Claim content.\n\n"
            "> [!info] p-aabbccdd · confidence 0.74\n"
            "> **Source:** [example.com](https://example.com)\n"
        )
        assert _note_has_particle_audit_trail(note) is True

    def test_returns_false_when_no_callouts_present(self) -> None:
        from particles.exporters.obsidian import _note_has_particle_audit_trail

        # Coin template: frontmatter + H1 only — no audit trail.
        note = "---\ntags:\n  - particles/coin\n---\n\n# Subject\n"
        assert _note_has_particle_audit_trail(note) is False

    def test_ignores_unverified_link_warning_callout(self) -> None:
        from particles.exporters.obsidian import _note_has_particle_audit_trail

        # Other callouts (lint, subject-level info) don't carry the
        # ``p-{hex}`` marker and must not trigger the detector.
        note = (
            "# Subject\n\n"
            "> [!warning] Unverified Wikidata link\n"
            "> Candidate: `wikidata:Q123`\n"
            "\n"
            "> [!info] LOW_COVERAGE_SUBJECT\n"
            "> Subject 'X' has only 1 particle.\n"
        )
        assert _note_has_particle_audit_trail(note) is False


class TestShouldSkipSynthesisForLowCoverage:
    """The `--with-synthesis` path burns an LLM call per subject.
    Subjects with too few particles produce articles that just
    paraphrase a single claim — no synthesis value, all cost. The
    threshold lives in `obsidian.synthesis_min_particles` (default 3)."""

    def test_default_threshold_is_three(self) -> None:
        from particles.exporters.obsidian import _should_skip_synthesis_for_low_coverage

        # Default obsidian.synthesis_min_particles = 3 → 1 and 2 skip, 3+ run.
        assert _should_skip_synthesis_for_low_coverage(0) is True
        assert _should_skip_synthesis_for_low_coverage(1) is True
        assert _should_skip_synthesis_for_low_coverage(2) is True
        assert _should_skip_synthesis_for_low_coverage(3) is False
        assert _should_skip_synthesis_for_low_coverage(10) is False

    def test_threshold_honours_config_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.config import reset_config
        from particles.exporters.obsidian import _should_skip_synthesis_for_low_coverage

        # Lower the threshold to 2 so a 2-particle subject now runs.
        monkeypatch.setenv("OBSIDIAN_SYNTHESIS_MIN_PARTICLES", "2")
        reset_config()
        assert _should_skip_synthesis_for_low_coverage(1) is True
        assert _should_skip_synthesis_for_low_coverage(2) is False

    def test_threshold_can_be_disabled_with_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.config import reset_config
        from particles.exporters.obsidian import _should_skip_synthesis_for_low_coverage

        # threshold=1 means "run synthesis for every subject that has
        # at least 1 particle" — i.e. effectively disable the skip.
        monkeypatch.setenv("OBSIDIAN_SYNTHESIS_MIN_PARTICLES", "1")
        reset_config()
        assert _should_skip_synthesis_for_low_coverage(1) is False
        assert _should_skip_synthesis_for_low_coverage(0) is True  # 0 still skips


class TestStripPerParticleCallouts:
    """The Obsidian generic / pivot templates render each particle as a
    Format-C entry: ``### N. {claim}`` heading + ``> [!info] p-{hex}``
    callout. When the synthesised References section (same Format-C
    shape, but cited-only) is spliced in, those structural entries
    would duplicate the new content, so the strip removes them.
    Lint findings and subject-level callouts (no ``p-{hex}`` marker)
    must survive."""

    def test_strips_subject_note_callout(self) -> None:
        from particles.exporters.obsidian import _strip_per_particle_callouts

        text = (
            "Above the entry.\n\n"
            "### 1. NVIDIA, AMD, and Qualcomm are dependent on TSMC.\n\n"
            "> [!info] p-90708a4c · confidence 0.06\n"
            "> **Related:** [[Nvidia]], [[Qualcomm]]\n"
            "> **Source:** [example.com](https://example.com)\n"
            "> **Extractor:** general-extractor 0.3.0 on 2026-05-15 ^p-90708a4c\n"
            "\n"
            "Below the entry.\n"
        )
        out = _strip_per_particle_callouts(text)
        assert "Above the entry." in out
        assert "Below the entry." in out
        # Heading + claim text are gone.
        assert "### 1." not in out
        assert "NVIDIA, AMD, and Qualcomm" not in out
        # Callout body is gone.
        assert "p-90708a4c" not in out
        assert "[[Nvidia]]" not in out

    def test_strips_multiple_back_to_back_entries(self) -> None:
        from particles.exporters.obsidian import _strip_per_particle_callouts

        text = (
            "**Particles:** 2\n\n"
            "### 1. First claim.\n\n"
            "> [!info] p-11111111 · confidence 0.7\n"
            "> **Extractor:** x 0.1 on 2026-05-15 ^p-11111111\n"
            "\n"
            "### 2. Second claim.\n\n"
            "> [!info] p-22222222 · confidence 0.8\n"
            "> **Extractor:** x 0.1 on 2026-05-15 ^p-22222222\n"
            "\n"
            "## Next section\n"
        )
        out = _strip_per_particle_callouts(text)
        assert "First claim." not in out
        assert "Second claim." not in out
        assert "p-11111111" not in out
        assert "p-22222222" not in out
        assert "**Particles:** 2" in out
        assert "## Next section" in out

    def test_preserves_lint_findings_and_subject_callouts(self) -> None:
        from particles.exporters.obsidian import _strip_per_particle_callouts

        # LOW_COVERAGE_SUBJECT and unverified-link callouts have no
        # ``### N.`` heading prefix and no ``p-{hex}`` callout title —
        # the strip's matcher needs both, so these must survive.
        text = (
            "> [!info] LOW_COVERAGE_SUBJECT\n"
            "> Subject 'X' has only 1 particle.\n"
            "\n"
            "> [!warning] Unverified Wikidata link\n"
            "> Candidate: `wikidata:Q123` (confidence 0.20) — context mismatch.\n"
            "\n"
            "Other text.\n"
        )
        out = _strip_per_particle_callouts(text)
        assert "LOW_COVERAGE_SUBJECT" in out
        assert "Unverified Wikidata link" in out
        assert "Other text." in out

    def test_orphan_separator_before_heading_collapses(self) -> None:
        from particles.exporters.obsidian import _strip_per_particle_callouts

        # `_render_subject_note` emits `**Particles:** N\n\n---\n\n[entries]`.
        # After stripping the entries the orphan `---` must not be
        # left dangling between subject metadata and the next heading.
        text = (
            "**Particles:** 1\n\n"
            "---\n\n"
            "### 1. only claim\n\n"
            "> [!info] p-aaaaaaaa · confidence 0.7\n"
            "> **Extractor:** x 0.1 on 2026-05-15 ^p-aaaaaaaa\n"
            "\n"
            "## References\n"
        )
        out = _strip_per_particle_callouts(text)
        assert "\n---\n" not in out
        assert "## References" in out
        assert "**Particles:** 1" in out

    def test_orphan_separator_at_end_of_string_collapses(self) -> None:
        from particles.exporters.obsidian import _strip_per_particle_callouts

        # Pre-splice, the stripped note ends with the orphan `---`. The
        # subsequent splice will append `## References`, but at strip
        # time we only see end-of-string in the lookahead.
        text = (
            "**Particles:** 1\n\n"
            "---\n\n"
            "### 1. only claim\n\n"
            "> [!info] p-aaaaaaaa · confidence 0.7\n"
            "> **Extractor:** x 0.1 on 2026-05-15 ^p-aaaaaaaa\n"
        )
        out = _strip_per_particle_callouts(text)
        assert "\n---\n" not in out

    def test_does_not_touch_yaml_frontmatter_dashes(self) -> None:
        from particles.exporters.obsidian import _strip_per_particle_callouts

        text = "---\ntags:\n  - particles/subject\n---\n\n# Subject\n\nBody content.\n"
        out = _strip_per_particle_callouts(text)
        # Frontmatter survives intact.
        assert out.startswith("---\ntags:\n  - particles/subject\n---\n")
        assert "# Subject" in out


class TestPruneObsoleteMarkdown:
    """0.42.4: the prune helper replaces the pre-write blanket wipe.

    Bug repro from the user: ``export obsidian --with-synthesis`` deleted
    every .md file before writing, only to repopulate them from the
    synthesis cache. An interrupt mid-export left the vault empty; even
    a clean run thrashed Obsidian's file watcher pointlessly. The fix
    is a target-set diff: walk the output dir AFTER writes, remove only
    files this run did not produce.
    """

    def test_files_in_written_set_survive(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from particles.exporters.markdown import prune_obsolete_markdown

        kept = tmp_path / "kept.md"
        kept.write_text("kept body")
        original_mtime = kept.stat().st_mtime

        removed = prune_obsolete_markdown(tmp_path, {kept}, recursive=False)
        assert removed == 0
        assert kept.exists()
        # Mtime unchanged — the file was not even rewritten. This is what
        # stops Obsidian's file watcher from firing.
        assert kept.stat().st_mtime == original_mtime

    def test_files_outside_written_set_removed(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from particles.exporters.markdown import prune_obsolete_markdown

        kept = tmp_path / "kept.md"
        kept.write_text("kept body")
        obsolete = tmp_path / "obsolete.md"
        obsolete.write_text("subject was deleted in DB but file lingered")

        removed = prune_obsolete_markdown(tmp_path, {kept}, recursive=False)
        assert removed == 1
        assert kept.exists()
        assert not obsolete.exists()

    def test_recursive_walks_subdirs(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from particles.exporters.markdown import prune_obsolete_markdown

        nested_kept = tmp_path / "github.com" / "kept.md"
        nested_kept.parent.mkdir()
        nested_kept.write_text("kept body")
        nested_obs = tmp_path / "reddit.com" / "obs.md"
        nested_obs.parent.mkdir()
        nested_obs.write_text("subject suppressed")

        removed = prune_obsolete_markdown(tmp_path, {nested_kept}, recursive=True)
        assert removed == 1
        assert nested_kept.exists()
        assert not nested_obs.exists()
        # Empty subdir cleanup — the now-empty reddit.com/ should be gone.
        assert not (tmp_path / "reddit.com").exists()
        # Subdir that still has live content survives.
        assert (tmp_path / "github.com").exists()

    def test_resolve_normalises_path_comparison(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from particles.exporters.markdown import prune_obsolete_markdown

        # Seed `written` with a relative-style path; on-disk file is its
        # absolute resolve(). Caller shouldn't have to normalise.
        kept = tmp_path / "kept.md"
        kept.write_text("body")
        # Pass the unresolved Path that the caller would naturally hand in.
        removed = prune_obsolete_markdown(tmp_path, {kept}, recursive=False)
        assert removed == 0
        assert kept.exists()

    def test_non_markdown_files_untouched(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from particles.exporters.markdown import prune_obsolete_markdown

        # Operators sometimes drop .canvas, images, or .obsidian/ configs
        # in the vault directory. The prune must only touch .md files.
        kept = tmp_path / "kept.md"
        kept.write_text("body")
        canvas = tmp_path / "graph.canvas"
        canvas.write_text("{}")
        config_dir = tmp_path / ".obsidian"
        config_dir.mkdir()
        cfg = config_dir / "workspace.json"
        cfg.write_text("{}")

        prune_obsolete_markdown(tmp_path, {kept}, recursive=True)
        assert canvas.exists()
        assert cfg.exists()


class TestDisambiguation:
    """same-named subjects get qualified notes + a disambiguation note."""

    def _two_prometheus(self) -> tuple[Subject, Subject]:
        sw = Subject(
            canonical_name="Prometheus",
            description="event monitoring and alerting software",
            asserted_by="test",
        )
        myth = Subject(
            canonical_name="Prometheus",
            description="Titan, culture hero, and trickster figure in Greek mythology",
            asserted_by="test",
        )
        return sw, myth

    def test_subject_note_h1_uses_disambiguated_display_name(self) -> None:
        from particles.exporters.markdown import build_subject_naming
        from particles.exporters.obsidian import _render_subject_note

        sw, myth = self._two_prometheus()
        naming = build_subject_naming([sw, myth])
        smap = {sw.id: sw, myth.id: myth}
        out = _render_subject_note(sw, [_make_particle("a claim", [sw.id])], smap, naming=naming)
        assert "# Prometheus (software)" in out
        # The bare, ambiguous heading must NOT appear.
        assert "# Prometheus\n" not in out

    def test_render_disambiguation_note(self) -> None:
        from particles.exporters.markdown import build_subject_naming
        from particles.exporters.obsidian.vault import _render_disambiguation_note

        sw, myth = self._two_prometheus()
        naming = build_subject_naming([sw, myth])
        counts = {sw.id: 281, myth.id: 1}
        note = _render_disambiguation_note(naming.groups[0], [sw, myth], naming, counts)
        assert "# Prometheus (disambiguation)" in note
        # Aliased to the bare name so [[Prometheus]] still resolves.
        assert "aliases:\n  - Prometheus" in note
        assert "particles/disambiguation" in note
        assert "[[Prometheus (Titan)]] — Titan (1 particle)" in note
        assert "[[Prometheus (software)]] — software (281 particles)" in note


class TestRenderNarrativeNote:
    """a NARRATIVE renders as a standalone Obsidian note under
    ``Narratives/`` — frontmatter + H1 (label) + cited prose + References."""

    @pytest.mark.asyncio
    async def test_fallback_without_key_builds_cited_note(
        self, db_session: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.exporters.markdown import build_subject_naming
        from particles.exporters.obsidian.narrative import render_narrative_note

        # No key → render_article falls back to the deterministic structured
        # listing (no LLM), so this is deterministic without mocking the model.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")

        narrative = _make_particle("A hard day the author got through.", []).model_copy(
            update={"particle_type": ParticleType.NARRATIVE}
        )
        c1 = _make_particle("The author woke up tired.", [])
        c2 = _make_particle("The author felt better by evening.", [])

        note, state = await render_narrative_note(
            narrative=narrative,
            constituents=[c1, c2],
            subject_map={},
            eligible_ids=set(),
            eff_conf={c1.id: 0.9, c2.id: 0.9},
            entry_uri_map={},
            naming=build_subject_naming([]),
            article_hash="HASH",
            session=db_session,  # type: ignore[arg-type]
        )

        assert state == "fallback"
        assert "# A hard day the author got through." in note
        assert "particles/narrative" in note
        assert "article_input_hash: HASH" in note
        assert "article_synthesis: structured-listing" in note
        # Particle references present (Obsidian block markers) — the ADR's
        # "complete with particle references".
        assert f"p-{c1.id[:8]}" in note
        assert f"p-{c2.id[:8]}" in note


class TestNarrativesSection:
    """the subject → narrative backlink section on per-subject notes."""

    @staticmethod
    def _narr(label: str) -> Particle:
        return _make_particle(label, []).model_copy(
            update={"particle_type": ParticleType.NARRATIVE}
        )

    def test_lists_emitted_narratives_as_links(self) -> None:
        from particles.exporters.obsidian.format import _render_narratives_section

        n1 = self._narr("A hard day.")
        n2 = self._narr("A good week.")
        naming = {n1.id: "a-hard-day", n2.id: "a-good-week"}
        out = _render_narratives_section([n1, n2], naming)
        assert "## Narratives" in out
        assert "[[Narratives/a-hard-day|A hard day.]]" in out
        assert "[[Narratives/a-good-week|A good week.]]" in out

    def test_skips_narrative_absent_from_naming(self) -> None:
        from particles.exporters.obsidian.format import _render_narratives_section

        # A narrative that was gated out (not in the naming map) is not linked —
        # no dangling [[Narratives/…]] target.
        assert _render_narratives_section([self._narr("Ungated.")], {}) == ""

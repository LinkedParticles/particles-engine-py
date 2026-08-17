"""Tests for the wiki article exporter.

Covers commits 1/3 (skeleton + caching + dry-run) and 2/3 (LLM synthesis
with Layer-A citation validation + footnote rendering). Layer-B
semantic-alignment-judge tests arrive with commit 3/3.

What's covered here:
  * plugin registration + config defaults
  * input-hash determinism + sensitivity to status / confidence
  * end-to-end export against an in-memory DB (no LLM):
      - qualifying-subject filter (min_particles, --subjects)
      - structured-listing render
      - cache hit short-circuits on second export
      - --regenerate-all bypasses the cache
      - --dry-run reports counts without writing files
      - index.md is written
  * LLM synthesis path with a mocked Anthropic client:
      - happy path: clean citations → article uses prose + References
      - Layer A invented-citation rejection → retry with strict prompt
      - retry fails → fallback to structured listing
      - LLM exception → fallback
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from particles.core.schema import (
    AssertionModality,
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Subject,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.db import session_scope
from particles.exporters.registry import get_exporters
from particles.exporters.wiki import (
    WikiExporter,
    _parse_frontmatter,
    _subject_slug,
    compute_input_hash,
    estimate_prompt_tokens,
    render_structured_listing,
    validate_citations,
)


# Make sure no test in this module accidentally calls the real Anthropic API
# if ANTHROPIC_API_KEY happens to be set in the developer's shell. Wiki
# synthesis catches any exception and falls back to the structured listing,
# so an unset key forces the fallback path deterministically. Tests that
# want to exercise the synthesis path explicitly inject a mock via
# ``particles.llm.set_client``.
@pytest.fixture(autouse=True)
def _no_anthropic_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")


# ---------------------------------------------------------------------------
# Tiny helpers — build a particle + subject + link them
# ---------------------------------------------------------------------------


def _make_particle(
    content: str,
    confidence_value: float = 0.9,
    extractor_id: str = "stub-extractor",
    status: Status = Status.ACTIVE,
    asserted_at: datetime | None = None,
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
        asserted_at=asserted_at or datetime.now(UTC),
        status=status,
        extractor_ref={"name": extractor_id, "version": "0.1.0"},
        subject_ids=[],
        tags=tags,
    )


async def _persist_subject_with_particles(
    canonical_name: str,
    particles: list[Particle],
    *,
    description: str | None = None,
) -> tuple[Subject, list[Particle]]:
    """Insert a subject + particles + link rows in one open session."""
    from particles.store.particle_store import insert_particle
    from particles.store.subject_store import insert_subject, link_particle_to_subjects

    subj = Subject(
        id=str(uuid.uuid4()),
        canonical_name=canonical_name,
        description=description,
        asserted_by="test",
    )
    async with session_scope() as session:
        await insert_subject(session, subj)
        for p in particles:
            await insert_particle(session, p)
            await link_particle_to_subjects(session, p.id, [subj.id])
        await session.commit()
    return subj, particles


# ---------------------------------------------------------------------------
# Registry + config wiring
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_wiki_is_registered(self) -> None:
        exporters = get_exporters()
        assert "wiki" in exporters
        assert isinstance(exporters["wiki"], WikiExporter)

    def test_format_constant(self) -> None:
        assert WikiExporter.FORMAT == "wiki"

    def test_config_defaults_match_adr(self) -> None:
        from particles.config import get_config, reset_config

        reset_config()
        cfg = get_config().wiki
        # pinned defaults
        assert cfg.min_particles == 3
        # the synthesis model moved from wiki.model into the per-purpose
        # llm config; the pinned default is unchanged.
        assert get_config().llm.for_purpose("synthesis").model == "claude-sonnet-4-6"
        assert cfg.max_tokens == 4096
        assert cfg.encyclopedic_tone is True
        assert cfg.layer_b_enabled is True


# ---------------------------------------------------------------------------
# Input hash — the cache key
# ---------------------------------------------------------------------------


class TestInputHash:
    def test_deterministic(self) -> None:
        ps = [_make_particle("a"), _make_particle("b")]
        assert compute_input_hash(ps) == compute_input_hash(ps)

    def test_order_independent(self) -> None:
        ps = [_make_particle("a"), _make_particle("b")]
        assert compute_input_hash(ps) == compute_input_hash(list(reversed(ps)))

    def test_ordered_hash_is_order_sensitive(self) -> None:
        """``ordered=True`` (narrative synthesis) makes constituent
        order part of the hash, and never collides with the unordered hash."""
        ps = [_make_particle("a"), _make_particle("b")]
        rev = list(reversed(ps))
        assert compute_input_hash(ps, ordered=True) != compute_input_hash(rev, ordered=True)
        assert compute_input_hash(ps, ordered=True) != compute_input_hash(ps)

    def test_status_change_changes_hash(self) -> None:
        p = _make_particle("a")
        h1 = compute_input_hash([p])
        # Same id, different status — model_copy with status override (test
        # only; the cache key has to detect status drift, not validate it).
        p2 = p.model_copy(update={"status": Status.SUPERSEDED})
        assert compute_input_hash([p2]) != h1

    def test_confidence_change_changes_hash(self) -> None:
        p1 = _make_particle("a", confidence_value=0.9)
        p2 = _make_particle("a", confidence_value=0.7)
        # Same id, different confidence — copy id over so the only diff is conf
        p2 = p1.model_copy(update={"confidence": p2.confidence})
        assert compute_input_hash([p1]) != compute_input_hash([p2])

    def test_adding_a_particle_changes_hash(self) -> None:
        p1 = _make_particle("a")
        p2 = _make_particle("b")
        assert compute_input_hash([p1]) != compute_input_hash([p1, p2])

    def test_prompt_version_mixed_in(self) -> None:
        """Bumping ``_PROMPT_VERSION`` must change the hash so the next
        export regenerates every cached article — operators don't have to
        remember ``--regenerate-all`` after a prompt edit."""
        # Patch the *owning* submodule (`cache`), not the package
        # __init__ re-export. ``compute_input_hash`` reads
        # ``_PROMPT_VERSION`` from its own module scope; mutating the
        # __init__ binding (which is a separate name slot) would not
        # affect the value the function sees. Same trap as the module-
        # top-import warning in tests/AGENTS.md § Mocking strategy.
        from particles.render.article_synthesis import cache as syn

        p = _make_particle("a")
        baseline = syn.compute_input_hash([p])
        original = syn._PROMPT_VERSION
        try:
            syn._PROMPT_VERSION = original + "-test"
            assert syn.compute_input_hash([p]) != baseline
        finally:
            syn._PROMPT_VERSION = original

    def test_subject_none_matches_no_subject_arg(self) -> None:
        """Backwards-compat: passing subject=None is equivalent to
        omitting the argument entirely."""
        from particles.core.schema import ExternalRef

        ps = [_make_particle("a")]
        s = Subject(
            id=str(uuid.uuid4()),
            canonical_name="X",
            external_ids=[ExternalRef(namespace="wikidata", id="Q1")],
            asserted_by="t",
        )
        assert compute_input_hash(ps) == compute_input_hash(ps, None)
        # Adding a subject changes the hash relative to no-subject.
        assert compute_input_hash(ps) != compute_input_hash(ps, s)

    def test_canonical_name_change_changes_hash(self) -> None:
        ps = [_make_particle("a")]
        s1 = Subject(id="sub-1", canonical_name="Foo", asserted_by="t")
        s2 = Subject(id="sub-1", canonical_name="Bar", asserted_by="t")
        assert compute_input_hash(ps, s1) != compute_input_hash(ps, s2)

    def test_external_ref_change_changes_hash(self) -> None:
        """The operator-facing motivation for this commit: after
        ``particles subjects unlink``, the next export must re-render
        the article."""
        from particles.core.schema import ExternalRef

        ps = [_make_particle("a")]
        s_with_ref = Subject(
            id="sub-1",
            canonical_name="CIA",
            external_ids=[ExternalRef(namespace="wikidata", id="Q37230")],
            asserted_by="t",
        )
        s_without_ref = Subject(
            id="sub-1",
            canonical_name="CIA",
            external_ids=[],
            asserted_by="t",
        )
        assert compute_input_hash(ps, s_with_ref) != compute_input_hash(ps, s_without_ref)

    def test_description_change_changes_hash(self) -> None:
        ps = [_make_particle("a")]
        s1 = Subject(id="sub-1", canonical_name="X", description="initial", asserted_by="t")
        s2 = Subject(id="sub-1", canonical_name="X", description="updated", asserted_by="t")
        assert compute_input_hash(ps, s1) != compute_input_hash(ps, s2)

    def test_aliases_change_changes_hash(self) -> None:
        ps = [_make_particle("a")]
        s1 = Subject(id="sub-1", canonical_name="X", aliases=["A"], asserted_by="t")
        s2 = Subject(id="sub-1", canonical_name="X", aliases=["A", "B"], asserted_by="t")
        assert compute_input_hash(ps, s1) != compute_input_hash(ps, s2)

    def test_aliases_order_independent(self) -> None:
        ps = [_make_particle("a")]
        s1 = Subject(id="sub-1", canonical_name="X", aliases=["A", "B"], asserted_by="t")
        s2 = Subject(id="sub-1", canonical_name="X", aliases=["B", "A"], asserted_by="t")
        assert compute_input_hash(ps, s1) == compute_input_hash(ps, s2)

    def test_subject_class_change_changes_hash(self) -> None:
        """The Obsidian renderer dispatches on subject_class (coin vs
        pivot vs generic). A class change therefore changes the
        article's appearance and must invalidate the cache."""
        ps = [_make_particle("a")]
        s1 = Subject(id="sub-1", canonical_name="X", subject_class=None, asserted_by="t")
        s2 = Subject(
            id="sub-1",
            canonical_name="X",
            subject_class="nmo:NumismaticObject",
            asserted_by="t",
        )
        assert compute_input_hash(ps, s1) != compute_input_hash(ps, s2)


# ---------------------------------------------------------------------------
# narrative-aware (sequence) synthesis prompt
# ---------------------------------------------------------------------------


def test_build_synthesis_prompt_sequence_mode() -> None:
    """``sequence_mode`` selects the narrative prompt (ordered prose), distinct
    from the encyclopedic per-subject prompt."""
    from particles.exporters.article_synthesis import _build_synthesis_prompt

    subject = Subject(id="n1", canonical_name="A hard day", asserted_by="test")
    ps = [_make_particle("The author woke up tired."), _make_particle("It got better.")]
    eff = {p.id: p.confidence.value for p in ps}

    narrative_prompt = _build_synthesis_prompt(
        subject=subject, particles=ps, eff=eff, strict=False, sequence_mode=True
    )
    assert "NARRATIVE:" in narrative_prompt
    assert "IN ORDER" in narrative_prompt

    standard_prompt = _build_synthesis_prompt(subject=subject, particles=ps, eff=eff, strict=False)
    assert "encyclopedic article" in standard_prompt
    assert narrative_prompt != standard_prompt


def test_synthesis_prompt_is_modality_aware() -> None:
    """the particle block carries assertion_modality and the prompt
    scopes the truth-hedge to truth-apt particles."""
    from particles.exporters.article_synthesis import _build_synthesis_prompt
    from particles.render.article_synthesis.render import _format_particles_for_prompt

    subject = Subject(id="s1", canonical_name="X", asserted_by="test")
    feeling = _make_particle("The author felt anxious.").model_copy(
        update={"assertion_modality": AssertionModality.EXPERIENTIAL}
    )
    fact = _make_particle("Paris is in France.")  # default FALSIFIABLE
    eff = {feeling.id: 0.9, fact.id: 0.9}

    block = _format_particles_for_prompt([feeling, fact], eff)
    assert "EXPERIENTIAL" in block  # modality column present
    assert "FALSIFIABLE" in block

    prompt = _build_synthesis_prompt(
        subject=subject, particles=[feeling, fact], eff=eff, strict=False
    )
    assert "Modality handling" in prompt
    assert "EXPERIENTIAL" in prompt
    assert "FALSIFIABLE / CONSTITUTIVE only" in prompt  # hedge scoped to truth-apt


# ---------------------------------------------------------------------------
# cross-subject cache staleness
# ---------------------------------------------------------------------------


class TestWikilinkRegex:
    """The wikilink-target regex underpins staleness scan."""

    def test_simple_link(self) -> None:
        from particles.render.article_synthesis.cache import (
            _extract_wikilink_targets,
        )

        assert _extract_wikilink_targets("see [[Foo]] for context") == {"Foo"}

    def test_alias_link(self) -> None:
        from particles.render.article_synthesis.cache import (
            _extract_wikilink_targets,
        )

        # Obsidian "piped alias" — the page is still Foo; the displayed
        # text is "alias text".
        assert _extract_wikilink_targets("see [[Foo|alias text]]") == {"Foo"}

    def test_heading_link(self) -> None:
        from particles.render.article_synthesis.cache import (
            _extract_wikilink_targets,
        )

        # `#heading` anchor — the page is still Foo.
        assert _extract_wikilink_targets("see [[Foo#Section]]") == {"Foo"}

    def test_heading_plus_alias(self) -> None:
        from particles.render.article_synthesis.cache import (
            _extract_wikilink_targets,
        )

        assert _extract_wikilink_targets("see [[Foo#Section|alias]]") == {"Foo"}

    def test_multiple_links_deduped(self) -> None:
        from particles.render.article_synthesis.cache import (
            _extract_wikilink_targets,
        )

        body = "A: [[Foo]]. B: [[Bar]]. C: [[Foo]] again."
        assert _extract_wikilink_targets(body) == {"Foo", "Bar"}

    def test_empty_body_no_matches(self) -> None:
        from particles.render.article_synthesis.cache import (
            _extract_wikilink_targets,
        )

        assert _extract_wikilink_targets("plain prose with no links") == set()


class TestInvalidateStaleLinkArticles:
    """The behaviour the wiki / Obsidian exporters consume."""

    def _write_article(
        self, path: Path, *, body: str, input_hash: str = "abc123", extra_fm: str = ""
    ) -> None:
        fm_block = f"input_hash: {input_hash}\nparticle_count: 3\n{extra_fm}".rstrip("\n")
        path.write_text(f"---\n{fm_block}\n---\n{body}\n", encoding="utf-8")

    def test_invalidates_articles_with_stale_links(self, tmp_path: Path) -> None:
        from particles.render.article_synthesis.cache import (
            invalidate_stale_link_articles,
        )

        # Article links to [[OldName]] — operator renamed OldName → NewName,
        # so OldName is no longer in known_names.
        article = tmp_path / "subject_a.md"
        self._write_article(article, body="See [[OldName]] for details.")
        invalidated = invalidate_stale_link_articles(tmp_path, {"NewName", "OtherSubject"})

        assert len(invalidated) == 1
        # input_hash was stripped from frontmatter; the body is untouched.
        text = article.read_text(encoding="utf-8")
        assert "input_hash" not in text
        assert "particle_count: 3" in text  # other frontmatter fields preserved
        assert "[[OldName]]" in text  # body content preserved

    def test_leaves_fresh_articles_alone(self, tmp_path: Path) -> None:
        from particles.render.article_synthesis.cache import (
            invalidate_stale_link_articles,
        )

        article = tmp_path / "subject_b.md"
        self._write_article(article, body="See [[CurrentSubject]] for details.")
        original_text = article.read_text(encoding="utf-8")

        invalidated = invalidate_stale_link_articles(tmp_path, {"CurrentSubject", "Other"})
        assert len(invalidated) == 0
        # File unchanged — including the input_hash line.
        assert article.read_text(encoding="utf-8") == original_text

    def test_alias_counts_as_known(self, tmp_path: Path) -> None:
        """``known_names`` is the union of canonical names AND aliases.
        A wikilink to an alias is NOT stale."""
        from particles.render.article_synthesis.cache import (
            invalidate_stale_link_articles,
        )

        article = tmp_path / "subject_c.md"
        self._write_article(article, body="See [[FormerName]] (now an alias).")
        # FormerName is in known_names because it's an alias of some
        # current subject. Article must NOT be invalidated.
        invalidated = invalidate_stale_link_articles(tmp_path, {"NewCanonicalName", "FormerName"})
        assert len(invalidated) == 0

    def test_idempotent(self, tmp_path: Path) -> None:
        """Running the scan twice in a row invalidates once then no-ops —
        the stripped article no longer has input_hash to strip."""
        from particles.render.article_synthesis.cache import (
            invalidate_stale_link_articles,
        )

        article = tmp_path / "subject_d.md"
        self._write_article(article, body="See [[Removed]] for details.")

        first = invalidate_stale_link_articles(tmp_path, {"Other"})
        second = invalidate_stale_link_articles(tmp_path, {"Other"})
        assert len(first) == 1
        assert len(second) == 0  # no input_hash left to strip; helper returns empty

    def test_preserves_non_link_articles(self, tmp_path: Path) -> None:
        """An article with no [[X]] wikilinks at all is never touched.
        The wiki structured-listing fallback emits some such articles
        when the LLM declines to synthesise prose."""
        from particles.render.article_synthesis.cache import (
            invalidate_stale_link_articles,
        )

        article = tmp_path / "subject_e.md"
        self._write_article(article, body="Plain prose with no wikilinks.")
        original = article.read_text(encoding="utf-8")

        invalidated = invalidate_stale_link_articles(tmp_path, {"SomeName"})
        assert len(invalidated) == 0
        assert article.read_text(encoding="utf-8") == original

    def test_obsidian_hash_field(self, tmp_path: Path) -> None:
        """Obsidian's per-subject note uses ``article_input_hash`` (the
        prefixed variant). The hash_field parameter must invalidate
        that field, not the default ``input_hash``."""
        from particles.render.article_synthesis.cache import (
            invalidate_stale_link_articles,
        )

        article = tmp_path / "note.md"
        article.write_text(
            "---\narticle_input_hash: xyz\ntags: [particles/article]\n---\nSee [[GoneSubject]].\n",
            encoding="utf-8",
        )
        invalidated = invalidate_stale_link_articles(
            tmp_path,
            {"Other"},
            hash_field="article_input_hash",
        )
        assert len(invalidated) == 1
        text = article.read_text(encoding="utf-8")
        assert "article_input_hash" not in text
        assert "tags:" in text  # other fields preserved

    def test_recursive_walks_subdirectories(self, tmp_path: Path) -> None:
        """The Obsidian vault layout nests notes (`coin/Q123.md`).
        recursive=True must reach those files."""
        from particles.render.article_synthesis.cache import (
            invalidate_stale_link_articles,
        )

        nested = tmp_path / "coin" / "subject.md"
        nested.parent.mkdir(parents=True)
        self._write_article(nested, body="See [[GoneSubject]].")
        # Non-recursive (the default) — nested file is invisible.
        assert len(invalidate_stale_link_articles(tmp_path, {"Other"})) == 0
        # Recursive — file gets invalidated.
        assert len(invalidate_stale_link_articles(tmp_path, {"Other"}, recursive=True)) == 1

    def test_empty_output_dir_returns_zero(self, tmp_path: Path) -> None:
        from particles.render.article_synthesis.cache import (
            invalidate_stale_link_articles,
        )

        assert len(invalidate_stale_link_articles(tmp_path, {"Anything"})) == 0
        # Non-existent dir also returns zero (operator passed --invalidate
        # on a fresh export run that hasn't written anything yet).
        assert len(invalidate_stale_link_articles(tmp_path / "does-not-exist", {"X"})) == 0


class TestFindUnresolvedWikilinks:
    """the one-shot check that synthesised ``[[Subject]]``
    cross-references resolve to a page in *both* exporters' output.

    The wiki and Obsidian exporters share the synthesis prompt that emits
    ``[[display name]]`` wikilinks, and both name per-subject files via
    ``subject_slug(display_name)`` (``particles.render.markdown``). These tests
    build directories exactly as each exporter would name them — using the real
    ``subject_slug`` so the resolution model is verified against the production
    naming machinery, not a guess — and assert that resolvable links resolve and
    a dangling reference is caught.
    """

    def _write(self, path: Path, body: str, *, frontmatter: str = "") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fm = f"---\n{frontmatter}\n---\n" if frontmatter else ""
        path.write_text(f"{fm}{body}\n", encoding="utf-8")

    def test_resolved_links_report_nothing(self, tmp_path: Path) -> None:
        from particles.render.article_synthesis import find_unresolved_wikilinks
        from particles.render.markdown import subject_slug

        # Wiki layout: one file per subject, named subject_slug(display_name).
        self._write(tmp_path / f"{subject_slug('Alpha')}.md", "About [[Beta]].")
        self._write(tmp_path / f"{subject_slug('Beta')}.md", "About [[Alpha]].")

        assert find_unresolved_wikilinks(tmp_path) == {}

    def test_dangling_link_is_flagged(self, tmp_path: Path) -> None:
        from particles.render.article_synthesis import find_unresolved_wikilinks
        from particles.render.markdown import subject_slug

        alpha = tmp_path / f"{subject_slug('Alpha')}.md"
        self._write(alpha, "Links to [[Beta]] and the missing [[Ghost]].")
        self._write(tmp_path / f"{subject_slug('Beta')}.md", "Standalone.")

        unresolved = find_unresolved_wikilinks(tmp_path)
        assert unresolved == {alpha: ["Ghost"]}

    def test_display_name_with_spaces_resolves(self, tmp_path: Path) -> None:
        # The LLM emits [[Greek Titan]]; the file is subject_slug("Greek Titan").
        from particles.render.article_synthesis import find_unresolved_wikilinks
        from particles.render.markdown import subject_slug

        self._write(tmp_path / f"{subject_slug('Prometheus')}.md", "See [[Greek Titan]].")
        self._write(tmp_path / f"{subject_slug('Greek Titan')}.md", "A Titan.")

        assert find_unresolved_wikilinks(tmp_path) == {}

    def test_obsidian_alias_resolves(self, tmp_path: Path) -> None:
        # disambiguation: the "Prometheus (software)" note carries a
        # bare-name alias so [[Prometheus]] resolves to it.
        from particles.render.article_synthesis import find_unresolved_wikilinks
        from particles.render.markdown import subject_slug

        self._write(
            tmp_path / f"{subject_slug('Prometheus (software)')}.md",
            "A monitoring system.",
            frontmatter="aliases:\n  - Prometheus",
        )
        self._write(tmp_path / f"{subject_slug('Grafana')}.md", "Pairs with [[Prometheus]].")

        assert find_unresolved_wikilinks(tmp_path) == {}

    def test_nested_reddit_slug_resolves(self, tmp_path: Path) -> None:
        # Reddit users nest under reddit.com/; subject_slug("u/foo") =
        # "reddit.com/u/foo", so the [[u/foo]] link resolves via the relative
        # slug path even though the basename is just "foo".
        from particles.render.article_synthesis import find_unresolved_wikilinks
        from particles.render.markdown import subject_slug

        self._write(tmp_path / f"{subject_slug('u/foo')}.md", "A redditor.")
        self._write(tmp_path / f"{subject_slug('Topic')}.md", "Discussed by [[u/foo]].")

        assert find_unresolved_wikilinks(tmp_path, recursive=True) == {}

    def test_recursive_required_for_nested_obsidian_layout(self, tmp_path: Path) -> None:
        from particles.render.article_synthesis import find_unresolved_wikilinks

        nested = tmp_path / "coin" / "Q123.md"
        self._write(nested, "Refers to the missing [[Ghost]].")

        # Non-recursive can't see the nested note at all.
        assert find_unresolved_wikilinks(tmp_path) == {}
        # Recursive walks into it and flags the dangling reference.
        assert find_unresolved_wikilinks(tmp_path, recursive=True) == {nested: ["Ghost"]}

    def test_links_without_targets_are_ignored(self, tmp_path: Path) -> None:
        from particles.render.article_synthesis import find_unresolved_wikilinks
        from particles.render.markdown import subject_slug

        # An article with no wikilinks (the structured-listing fallback) is
        # omitted from the result entirely.
        self._write(tmp_path / f"{subject_slug('Solo')}.md", "Plain prose, no links.")
        assert find_unresolved_wikilinks(tmp_path) == {}


class TestStripInlineFootnoteDefs:
    """LLMs keep emitting ``[^p-xxxxxxxx]: …`` definition lines despite
    the prompt forbidding them. The deterministic References block also
    emits a definition, so the duplicate breaks Obsidian/GFM footnote
    parsing. The strip helper is the belt-and-suspenders mitigation."""

    def test_strips_single_line_definition(self) -> None:
        from particles.exporters.article_synthesis import _strip_inline_footnote_defs

        body = "Some prose [^p-abcdef12].\n\n[^p-abcdef12]: paraphrased particle content.\n"
        cleaned = _strip_inline_footnote_defs(body)
        assert "[^p-abcdef12]" in cleaned  # reference preserved
        assert "paraphrased particle content" not in cleaned  # def stripped

    def test_strips_definition_with_indented_continuation(self) -> None:
        from particles.exporters.article_synthesis import _strip_inline_footnote_defs

        body = (
            "Prose [^p-00112233].\n\n"
            "[^p-00112233]: first line of the def\n"
            "    indented continuation line\n"
            "    another continuation\n"
            "\nMore prose.\n"
        )
        cleaned = _strip_inline_footnote_defs(body)
        assert "indented continuation" not in cleaned
        assert "another continuation" not in cleaned
        assert "More prose." in cleaned

    def test_preserves_body_text_and_references(self) -> None:
        from particles.exporters.article_synthesis import _strip_inline_footnote_defs

        body = "Claim one [^p-aaaaaaaa]. Claim two [^p-bbbbbbbb]."
        assert _strip_inline_footnote_defs(body) == body

    def test_strips_multiple_definitions(self) -> None:
        from particles.exporters.article_synthesis import _strip_inline_footnote_defs

        body = (
            "Body [^p-aaaaaaaa] [^p-bbbbbbbb].\n\n"
            "[^p-aaaaaaaa]: def one.\n"
            "[^p-bbbbbbbb]: def two.\n"
        )
        cleaned = _strip_inline_footnote_defs(body)
        assert "def one" not in cleaned
        assert "def two" not in cleaned
        assert "[^p-aaaaaaaa]" in cleaned and "[^p-bbbbbbbb]" in cleaned


# ---------------------------------------------------------------------------
# Structured-listing render — pure function, no DB
# ---------------------------------------------------------------------------


class TestStructuredListing:
    def test_agent_asserted_renders_asserted_by_not_extractor(self) -> None:
        # A direct agent assertion has no extractor_ref; the References
        # footnote must attribute it to the asserting principal, not "extractor: ?".
        subj = Subject(id=str(uuid.uuid4()), canonical_name="Signal Flags", asserted_by="test")
        p = Particle(
            id=str(uuid.uuid4()),
            content="Signal flags encode a read-write message surface.",
            confidence=Confidence(value=0.9, calibration_source=CalibrationSource.AGENT_ASSERTED),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="mcp:claude-code",
            asserted_at=datetime.now(UTC),
            status=Status.ACTIVE,
            provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1")],
            subject_ids=[subj.id],
        )
        body = render_structured_listing(subj, [p], {p.id: 0.5}, input_hash=compute_input_hash([p]))
        assert "asserted by: mcp:claude-code" in body
        assert "extractor: ?" not in body

    def test_emits_frontmatter_and_sections(self) -> None:
        subj = Subject(
            id=str(uuid.uuid4()),
            canonical_name="GDR",
            asserted_by="test",
        )
        particles = [
            _make_particle("The GDR was founded in 1949."),
            _make_particle("The GDR adopted the Mark currency."),
        ]
        eff = {p.id: 0.95 for p in particles}
        h = compute_input_hash(particles)

        body = render_structured_listing(subj, particles, eff, input_hash=h)
        # Frontmatter (round-trips through the parser)
        fm = _parse_frontmatter(body)
        assert fm is not None
        assert fm["particle_count"] == 2
        assert fm["input_hash"] == h
        assert fm["synthesis"] == "structured-listing"
        assert "particles/article" in fm["tags"]
        # Body
        assert "# GDR" in body
        assert "## Claims" in body
        assert "## References" in body
        for p in particles:
            assert p.content in body
            assert f"[^p-{p.id[:8]}]" in body

    def test_taxonomy_tags_propagate_to_frontmatter(self) -> None:
        """taxonomy tags on the article's particles join the
        frontmatter ``tags`` list after the ``particles/article`` marker."""
        subj = Subject(id=str(uuid.uuid4()), canonical_name="AdamW", asserted_by="t")
        particles = [
            _make_particle("AdamW decouples weight decay.", tags=["ml/optimizers"]),
            _make_particle("AdamW is widely used.", tags=["ml/training", "ml/optimizers"]),
        ]
        eff = {p.id: 0.9 for p in particles}
        body = render_structured_listing(
            subj, particles, eff, input_hash=compute_input_hash(particles)
        )
        fm = _parse_frontmatter(body)
        assert fm is not None
        # Article marker stays first; taxonomy tags follow, deduped + sorted.
        assert fm["tags"] == ["particles/article", "ml/optimizers", "ml/training"]

    def test_no_taxonomy_tags_leaves_only_article_marker(self) -> None:
        subj = Subject(id=str(uuid.uuid4()), canonical_name="X", asserted_by="t")
        particles = [_make_particle("A plain claim.")]
        eff = {p.id: 0.9 for p in particles}
        body = render_structured_listing(
            subj, particles, eff, input_hash=compute_input_hash(particles)
        )
        fm = _parse_frontmatter(body)
        assert fm is not None
        assert fm["tags"] == ["particles/article"]

    def test_emits_anchor_heading_per_particle_in_references(self) -> None:
        """Each cited particle gets a `### p-{short_id}` heading above
        its footnote definition so the References section is navigable
        as a per-particle TOC and `[[Subject#p-abc12345]]` cross-article
        wikilinks resolve."""
        subj = Subject(id=str(uuid.uuid4()), canonical_name="GDR", asserted_by="t")
        particles = [_make_particle(f"claim {i}") for i in range(3)]
        eff = {p.id: 0.5 for p in particles}
        body = render_structured_listing(
            subj, particles, eff, input_hash=compute_input_hash(particles)
        )
        for p in particles:
            sid = p.id[:8]
            assert f"### p-{sid}" in body, f"missing anchor heading for {sid}"
            # The heading lives in the References section (after the
            # `## References` marker), not the Claims section.
            assert body.index(f"### p-{sid}") > body.index("## References")
            # The footnote definition still lives at its old form so
            # markdown-footnote-aware renderers still scroll there.
            assert f"[^p-{sid}]:" in body

    def test_emits_one_footnote_per_particle(self) -> None:
        subj = Subject(id=str(uuid.uuid4()), canonical_name="X", asserted_by="t")
        particles = [_make_particle(f"claim {i}") for i in range(5)]
        eff = {p.id: 0.5 for p in particles}
        body = render_structured_listing(
            subj, particles, eff, input_hash=compute_input_hash(particles)
        )
        # one in-body reference + one definition per particle
        for p in particles:
            assert body.count(f"[^p-{p.id[:8]}]") >= 2


# ---------------------------------------------------------------------------
# End-to-end export against an in-memory DB
# ---------------------------------------------------------------------------


class TestExportEndToEnd:
    @pytest.mark.asyncio
    async def test_dry_run_writes_no_files(self, db_session: object, tmp_path: Path) -> None:
        await _persist_subject_with_particles(
            "GDR", [_make_particle(f"claim {i}") for i in range(4)]
        )
        exporter = WikiExporter()
        async with session_scope() as session:
            summary = await exporter.export(session, tmp_path, dry_run=True)
        assert summary.dry_run is True
        assert summary.qualifying_subjects == 1
        assert summary.articles_regenerated == 1
        assert summary.estimated_prompt_tokens > 0
        # No files written
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.asyncio
    async def test_same_named_subjects_get_distinct_articles_and_disambiguation(
        self, db_session: object, tmp_path: Path
    ) -> None:
        # two "Prometheus" subjects must produce two distinct
        # articles plus a disambiguation page — not one overwritten file.
        await _persist_subject_with_particles(
            "Prometheus",
            [_make_particle(f"sw claim {i}") for i in range(3)],
            description="event monitoring and alerting software",
        )
        await _persist_subject_with_particles(
            "Prometheus",
            [_make_particle(f"myth claim {i}") for i in range(3)],
            description="Titan, culture hero, and trickster figure in Greek mythology",
        )
        async with session_scope() as session:
            await WikiExporter().export(session, tmp_path)

        names = sorted(p.name for p in tmp_path.glob("*.md"))
        assert "Prometheus (software).md" in names, names
        assert "Prometheus (Titan).md" in names, names
        assert "Prometheus (disambiguation).md" in names, names
        assert "Prometheus.md" not in names
        disamb = (tmp_path / "Prometheus (disambiguation).md").read_text()
        assert "[[Prometheus (Titan)]]" in disamb
        assert "[[Prometheus (software)]]" in disamb
        index = (tmp_path / "index.md").read_text()
        assert "[[Prometheus (Titan)]]" in index

    @pytest.mark.asyncio
    async def test_export_writes_article_and_index(
        self, db_session: object, tmp_path: Path
    ) -> None:
        subj, particles = await _persist_subject_with_particles(
            "GDR", [_make_particle(f"claim {i}") for i in range(3)]
        )
        exporter = WikiExporter()
        async with session_scope() as session:
            summary = await exporter.export(session, tmp_path)
        assert summary.dry_run is False
        assert summary.articles_written == 1
        article = tmp_path / "GDR.md"
        assert article.exists()
        text = article.read_text()
        fm = _parse_frontmatter(text)
        assert fm is not None
        assert fm["particle_count"] == 3
        # index.md lists the new article as a wikilink
        index = (tmp_path / "index.md").read_text()
        assert "[[GDR]]" in index

    @pytest.mark.asyncio
    async def test_min_particles_filter_skips_thin_subjects(
        self, db_session: object, tmp_path: Path
    ) -> None:
        # Below threshold by default (3)
        await _persist_subject_with_particles("Sparse", [_make_particle("only claim")])
        await _persist_subject_with_particles(
            "Dense", [_make_particle(f"claim {i}") for i in range(3)]
        )
        exporter = WikiExporter()
        async with session_scope() as session:
            summary = await exporter.export(session, tmp_path)
        assert summary.qualifying_subjects == 1
        assert (tmp_path / "Dense.md").exists()
        assert not (tmp_path / "Sparse.md").exists()

    @pytest.mark.asyncio
    async def test_subjects_filter_targets_named_subjects(
        self, db_session: object, tmp_path: Path
    ) -> None:
        await _persist_subject_with_particles(
            "GDR", [_make_particle(f"claim {i}") for i in range(3)]
        )
        await _persist_subject_with_particles(
            "Pfennig", [_make_particle(f"pf claim {i}") for i in range(3)]
        )
        exporter = WikiExporter()
        async with session_scope() as session:
            summary = await exporter.export(session, tmp_path, subjects=["GDR"])
        assert summary.qualifying_subjects == 1
        assert (tmp_path / "GDR.md").exists()
        assert not (tmp_path / "Pfennig.md").exists()

    @pytest.mark.asyncio
    async def test_cache_hit_short_circuits_second_run(
        self, db_session: object, tmp_path: Path
    ) -> None:
        await _persist_subject_with_particles(
            "GDR", [_make_particle(f"claim {i}") for i in range(3)]
        )
        exporter = WikiExporter()
        async with session_scope() as session:
            await exporter.export(session, tmp_path)
            # second run — same particles → cache hits, zero regenerations
            summary = await exporter.export(session, tmp_path)
        assert summary.cache_hits == 1
        assert summary.articles_regenerated == 0
        assert summary.articles_written == 0

    @pytest.mark.asyncio
    async def test_regenerate_all_bypasses_cache(self, db_session: object, tmp_path: Path) -> None:
        await _persist_subject_with_particles(
            "GDR", [_make_particle(f"claim {i}") for i in range(3)]
        )
        exporter = WikiExporter()
        async with session_scope() as session:
            await exporter.export(session, tmp_path)
            summary = await exporter.export(session, tmp_path, regenerate_all=True)
        assert summary.cache_hits == 0
        assert summary.articles_regenerated == 1
        assert summary.articles_written == 1

    @pytest.mark.asyncio
    async def test_min_particles_override(self, db_session: object, tmp_path: Path) -> None:
        # Below the default-3 threshold, but should qualify when override=1
        await _persist_subject_with_particles("Sparse", [_make_particle("only claim")])
        exporter = WikiExporter()
        async with session_scope() as session:
            summary = await exporter.export(session, tmp_path, min_particles=1)
        assert summary.articles_written == 1
        assert (tmp_path / "Sparse.md").exists()

    @pytest.mark.asyncio
    async def test_requires_output_directory(self, db_session: object) -> None:
        exporter = WikiExporter()
        with pytest.raises(ValueError, match="output directory"):
            async with session_scope() as session:
                await exporter.export(session, None)


# ---------------------------------------------------------------------------
# Slug + token-estimate sanity
# ---------------------------------------------------------------------------


def test_subject_slug_sanitises_invalid_chars() -> None:
    assert _subject_slug("a/b:c") == "a-b-c"
    assert _subject_slug("  spaces   collapse  ") == "spaces collapse"
    assert _subject_slug("") == "unnamed"


def test_estimate_tokens_grows_with_content() -> None:
    short = [_make_particle("x")]
    long = [_make_particle("x" * 1000)]
    assert estimate_prompt_tokens(long) > estimate_prompt_tokens(short)


# ---------------------------------------------------------------------------
# split_rendered_article — used by Obsidian's --with-synthesis splicing
# ---------------------------------------------------------------------------


class TestSplitRenderedArticle:
    def test_round_trip_structured_listing(self) -> None:
        from particles.exporters.article_synthesis import (
            render_structured_listing,
            split_rendered_article,
        )

        subj = Subject(id=str(uuid.uuid4()), canonical_name="GDR", asserted_by="t")
        ps = [_make_particle(f"claim {i}") for i in range(2)]
        eff = {p.id: 0.9 for p in ps}
        body = render_structured_listing(subj, ps, eff, input_hash=compute_input_hash(ps))
        fm, h1, prose, refs = split_rendered_article(body)

        assert fm["synthesis"] == "structured-listing"
        assert h1.startswith("# GDR")
        # Prose is between H1 and References; for the structured listing
        # it contains the Claims section plus the fallback callout.
        assert "## Claims" in prose
        assert "Structured-listing render" in prose  # the callout
        # References start with ## References and contain anchor headings.
        assert refs.startswith("## References")
        for p in ps:
            assert f"### p-{p.id[:8]}" in refs

    def test_handles_article_without_references(self) -> None:
        """A custom article body without a References section still splits."""
        from particles.exporters.article_synthesis import split_rendered_article

        body = "---\nfoo: bar\n---\n\n# Title\n\nProse only.\n"
        fm, h1, prose, refs = split_rendered_article(body)
        assert fm == {"foo": "bar"}
        assert h1 == "# Title\n"
        assert prose == "Prose only."
        assert refs == ""

    def test_handles_article_without_frontmatter(self) -> None:
        from particles.exporters.article_synthesis import split_rendered_article

        body = "# Bare H1\n\nProse.\n"
        fm, h1, prose, refs = split_rendered_article(body)
        assert fm == {}
        assert h1 == "# Bare H1\n"
        assert prose == "Prose."
        assert refs == ""


# ---------------------------------------------------------------------------
# Layer A — citation ID-membership validation (pure unit)
# ---------------------------------------------------------------------------


class TestLayerA:
    def test_accepts_only_listed_ids(self) -> None:
        body = "Claim one [^p-abc12345]. Claim two [^p-def67890]."
        seen, invalid = validate_citations(body, {"abc12345", "def67890"})
        assert seen == {"abc12345", "def67890"}
        assert invalid == set()

    def test_flags_invented_id(self) -> None:
        body = "Real claim [^p-abc12345]. Fake [^p-00000000]."
        seen, invalid = validate_citations(body, {"abc12345"})
        assert seen == {"abc12345", "00000000"}
        assert invalid == {"00000000"}

    def test_case_insensitive(self) -> None:
        body = "Cited [^p-ABCDEF12]."
        _, invalid = validate_citations(body, {"abcdef12"})
        assert invalid == set()

    def test_no_citations_no_invalid(self) -> None:
        # An article with zero citations passes Layer A vacuously. (Layer B
        # — semantic-alignment.4 — would normally fail such
        # an article in commit 3.)
        seen, invalid = validate_citations("No citations here.", {"abc12345"})
        assert seen == set()
        assert invalid == set()


class TestCountUncitedParagraphs:
    """Catches LLM padding: the body has some citations, but multiple
    paragraphs contain no citation at all (pure general-knowledge
    prose). The CIA-vs-film bug — 6 of 7 paragraphs uncited — is the
    canonical motivating example."""

    def _call(self, body: str, *, allow_leading: int = 1) -> int:
        from particles.exporters.article_synthesis import count_uncited_paragraphs

        return count_uncited_paragraphs(body, allow_leading=allow_leading)

    def test_fully_cited_body_returns_zero(self) -> None:
        body = (
            "# Subject\n\n"
            "First claim [^p-aabbccdd].\n\n"
            "Second claim [^p-11223344].\n\n"
            "Third claim [^p-deadbeef]."
        )
        assert self._call(body) == 0

    def test_single_uncited_intro_within_quota(self) -> None:
        """The first content paragraph is allowed to be uncited — intros
        often paraphrase identity without introducing new cite-worthy
        facts."""
        body = "# Subject\n\nSubject is the introductory paraphrase.\n\nCited claim [^p-aabbccdd]."
        assert self._call(body) == 0

    def test_uncited_paragraph_past_intro_counts(self) -> None:
        body = (
            "# Subject\n\n"
            "Intro paragraph (no cite).\n\n"
            "Mid-body uncited padding.\n\n"
            "Cited claim [^p-aabbccdd]."
        )
        # Intro consumed by allow_leading; the mid-body paragraph counts.
        assert self._call(body) == 1

    def test_cia_style_padding_caught(self) -> None:
        """Reproduces the failure shape that motivated the check.
        Article body: 6 uncited paragraphs of LLM general knowledge,
        1 cited paragraph (the only one backed by particles)."""
        body = (
            "# Central Intelligence Agency\n\n"
            "The CIA is the primary foreign intelligence service…\n\n"
            "The CIA was established in 1947…\n\n"
            "The agency's core mission encompasses HUMINT…\n\n"
            "Throughout its history…\n\n"
            "The CIA is organized into directorates…\n\n"
            "Notable in popular culture is the film "
            "Central Intelligence [^p-ff8128cb] [^p-318ee858].\n\n"
            "The agency continues to adapt…"
        )
        # 7 content paragraphs, 1 cited (¶6), 1 leading allowance →
        # ¶2, ¶3, ¶4, ¶5, ¶7 are uncited & over quota = 5.
        assert self._call(body) == 5

    def test_zero_allow_leading_means_intro_counts(self) -> None:
        body = "Subject is described in this intro.\n\nCited claim [^p-aabbccdd]."
        # With allow_leading=0, even the intro counts.
        assert self._call(body, allow_leading=0) == 1

    def test_headings_excluded_from_count(self) -> None:
        body = (
            "# Subject\n\n"
            "## Section A\n\n"
            "Cited claim [^p-aabbccdd].\n\n"
            "## Section B\n\n"
            "Another cited [^p-11223344]."
        )
        # Headings don't count as content; both content paragraphs cite.
        assert self._call(body) == 0

    def test_blank_paragraphs_excluded(self) -> None:
        body = "# Subject\n\n\n\nCited [^p-aabbccdd].\n\n\n\n"
        assert self._call(body) == 0

    def test_all_paragraphs_uncited(self) -> None:
        body = "First uncited paragraph.\n\nSecond uncited paragraph.\n\nThird uncited paragraph."
        # First absorbed by intro allowance; second + third count = 2.
        assert self._call(body) == 2


# ---------------------------------------------------------------------------
# LLM synthesis path — mocked Anthropic client
# ---------------------------------------------------------------------------


def _mock_client(text_responses: list[str]) -> MagicMock:
    """Build a stub Anthropic client whose ``messages.create`` returns texts in order.

    Each call pops the next text from ``text_responses``. Used so a single
    test can simulate "first attempt invented an ID, retry was clean".
    """
    import anthropic

    client = MagicMock(spec=anthropic.Anthropic)
    client.messages = MagicMock()

    iterator = iter(text_responses)

    def _create(**kwargs: Any) -> MagicMock:
        content = MagicMock()
        content.text = next(iterator)
        resp = MagicMock()
        resp.content = [content]
        return resp

    client.messages.create = MagicMock(side_effect=_create)
    return client


class TestLLMSynthesis:
    @pytest.mark.asyncio
    async def test_happy_path_uses_synthesised_body(
        self, db_session: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Particles a real run would emit
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        ps = [_make_particle(f"claim {i}") for i in range(3)]
        await _persist_subject_with_particles("GDR", ps)

        cited_short_ids = [p.id[:8] for p in ps]
        # An LLM reply that cites every input particle correctly.
        llm_body = (
            f"# GDR\n\n"
            f"GDR was a state in central Europe [^p-{cited_short_ids[0]}]. "
            f"It was founded in 1949 [^p-{cited_short_ids[1]}]. "
            f"The currency was the Mark [^p-{cited_short_ids[2]}].\n"
        )

        from particles.llm import set_client

        # Two LLM calls: synthesis + Layer B judge (which returns an empty
        # JSON array → no misalignments → layer_b_passed=True).
        set_client(_mock_client([llm_body, "[]"]))
        try:
            exporter = WikiExporter()
            async with session_scope() as session:
                summary = await exporter.export(session, tmp_path)
        finally:
            set_client(None)

        assert summary.synthesis_used == 1
        assert summary.synthesis_failed == 0
        article = (tmp_path / "GDR.md").read_text()
        fm = _parse_frontmatter(article)
        assert fm is not None
        assert fm["synthesis"] == "llm"
        assert fm["layer_a_passed"] is True
        assert fm["layer_b_passed"] is True
        # Prose body survived; References section appended deterministically
        assert "GDR was a state in central Europe" in article
        assert "## References" in article
        for sid in cited_short_ids:
            assert f"[^p-{sid}]:" in article

    @pytest.mark.asyncio
    async def test_invented_id_triggers_retry_then_succeeds(
        self, db_session: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        ps = [_make_particle(f"claim {i}") for i in range(3)]
        await _persist_subject_with_particles("GDR", ps)

        cited_short_ids = [p.id[:8] for p in ps]
        bad_body = (
            f"# GDR\n\nClaim cites real ID [^p-{cited_short_ids[0]}] "
            f"but also invented [^p-deadbeef].\n"
        )
        clean_body = f"# GDR\n\nClean claim [^p-{cited_short_ids[0]}].\n"

        from particles.llm import set_client

        # Three calls: bad synthesis (Layer A rejects), strict-prompt
        # retry (Layer A passes), Layer B judge ("[]" → no misalignments).
        client = _mock_client([bad_body, clean_body, "[]"])
        set_client(client)
        try:
            exporter = WikiExporter()
            async with session_scope() as session:
                summary = await exporter.export(session, tmp_path)
        finally:
            set_client(None)

        assert client.messages.create.call_count == 3
        assert summary.synthesis_used == 1
        article = (tmp_path / "GDR.md").read_text()
        fm = _parse_frontmatter(article)
        assert fm is not None
        assert fm["synthesis"] == "llm"
        assert fm["layer_a_passed"] is True
        assert "[^p-deadbeef]" not in article

    @pytest.mark.asyncio
    async def test_both_attempts_invent_ids_falls_back_to_listing(
        self, db_session: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        ps = [_make_particle(f"claim {i}") for i in range(3)]
        await _persist_subject_with_particles("GDR", ps)

        bad_body = "# GDR\n\nClaim [^p-00000000].\n"
        from particles.llm import set_client

        client = _mock_client([bad_body, bad_body])
        set_client(client)
        try:
            exporter = WikiExporter()
            async with session_scope() as session:
                summary = await exporter.export(session, tmp_path)
        finally:
            set_client(None)

        assert client.messages.create.call_count == 2
        assert summary.synthesis_used == 0
        assert summary.synthesis_failed == 1
        article = (tmp_path / "GDR.md").read_text()
        fm = _parse_frontmatter(article)
        assert fm is not None
        assert fm["synthesis"] == "structured-listing"
        # The fake [^p-00000000] from the rejected synthesis must NOT leak
        # into the persisted article.
        assert "[^p-00000000]" not in article

    @pytest.mark.asyncio
    async def test_uncited_padding_triggers_retry_then_succeeds(
        self, db_session: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The LLM occasionally pads cited claims with general-knowledge
        paragraphs that have no citation. The density check catches this
        on the first attempt; the strict-prompt retry then produces a
        clean article that satisfies the check."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        ps = [_make_particle(f"claim {i}") for i in range(3)]
        await _persist_subject_with_particles("GDR", ps)

        cited = ps[0].id[:8]
        # First attempt: one cited paragraph, four uncited paragraphs of
        # padding. Intro consumed by allowance; remaining 3 uncited
        # paragraphs trip the density check.
        padded_body = (
            "# GDR\n\n"
            "Intro paragraph (no cite).\n\n"
            "First padding paragraph.\n\n"
            "Second padding paragraph.\n\n"
            "Third padding paragraph.\n\n"
            f"Cited claim [^p-{cited}]."
        )
        # Retry: lean article with citations on every content paragraph.
        clean_body = f"# GDR\n\nClean cited claim [^p-{cited}]."

        from particles.llm import set_client

        # Three calls: padded synthesis (density rejects), strict retry
        # (passes), Layer B judge ("[]" → no misalignments).
        client = _mock_client([padded_body, clean_body, "[]"])
        set_client(client)
        try:
            exporter = WikiExporter()
            async with session_scope() as session:
                summary = await exporter.export(session, tmp_path)
        finally:
            set_client(None)

        assert client.messages.create.call_count == 3
        assert summary.synthesis_used == 1
        article = (tmp_path / "GDR.md").read_text()
        fm = _parse_frontmatter(article)
        assert fm is not None
        assert fm["synthesis"] == "llm"
        # None of the padding paragraphs survived — the clean retry
        # is what got persisted.
        assert "padding paragraph" not in article
        assert "Clean cited claim" in article

    @pytest.mark.asyncio
    async def test_both_attempts_padded_falls_back_to_listing(
        self, db_session: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If both the initial and strict-prompt attempts emit uncited
        padding, fall back to the structured-listing render rather than
        shipping unsourced prose."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        ps = [_make_particle(f"claim {i}") for i in range(3)]
        await _persist_subject_with_particles("GDR", ps)

        cited = ps[0].id[:8]
        padded_body = (
            "# GDR\n\n"
            "Intro paragraph.\n\n"
            "Uncited padding A.\n\n"
            "Uncited padding B.\n\n"
            f"Lone cited claim [^p-{cited}]."
        )

        from particles.llm import set_client

        client = _mock_client([padded_body, padded_body])
        set_client(client)
        try:
            exporter = WikiExporter()
            async with session_scope() as session:
                summary = await exporter.export(session, tmp_path)
        finally:
            set_client(None)

        assert client.messages.create.call_count == 2
        assert summary.synthesis_used == 0
        assert summary.synthesis_failed == 1
        article = (tmp_path / "GDR.md").read_text()
        fm = _parse_frontmatter(article)
        assert fm is not None
        assert fm["synthesis"] == "structured-listing"
        # Padding from the rejected synthesis must NOT leak through.
        assert "padding" not in article

    @pytest.mark.asyncio
    async def test_synthesis_uses_asyncio_to_thread(
        self, db_session: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_call_synthesis_llm`` delegates the SDK call to a worker thread.

        This is what makes Ctrl-C responsive in production: the main
        coroutine yields to the event loop while the LLM call is in
        flight, so KeyboardInterrupt propagates at the next await
        checkpoint between subjects rather than being swallowed until
        the call returns.
        """
        import asyncio

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        ps = [_make_particle(f"claim {i}") for i in range(3)]
        await _persist_subject_with_particles("GDR", ps)

        cited = [p.id[:8] for p in ps]
        llm_body = f"# GDR\n\nClaim [^p-{cited[0]}].\n"
        from particles.llm import set_client

        client = _mock_client([llm_body, "[]"])
        set_client(client)

        # Capture every call to_thread routes through, so we can confirm
        # the LLM call is one of them.
        observed_targets: list[str] = []
        real_to_thread = asyncio.to_thread

        async def _capturing_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
            observed_targets.append(getattr(func, "__name__", repr(func)))
            return await real_to_thread(func, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", _capturing_to_thread)
        try:
            exporter = WikiExporter()
            async with session_scope() as session:
                await exporter.export(session, tmp_path)
        finally:
            set_client(None)

        # Both the synthesis call and the Layer B judge call hit to_thread.
        assert len(observed_targets) >= 2

    @pytest.mark.asyncio
    async def test_zero_citation_body_treated_as_layer_a_failure(
        self, db_session: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A body with zero citations passes the regex vacuously but defeats
        the 'every claim cited' promise. The exporter must treat it as a
        Layer A failure → retry → fallback."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        ps = [_make_particle(f"claim {i}") for i in range(3)]
        await _persist_subject_with_particles("GDR", ps)

        no_citation_body = "# GDR\n\nGDR was a state in central Europe.\n"
        from particles.llm import set_client

        # Both attempts emit zero-citation prose → Layer A fails twice
        # → fallback to structured listing.
        client = _mock_client([no_citation_body, no_citation_body])
        set_client(client)
        try:
            exporter = WikiExporter()
            async with session_scope() as session:
                summary = await exporter.export(session, tmp_path)
        finally:
            set_client(None)

        assert client.messages.create.call_count == 2
        assert summary.synthesis_used == 0
        assert summary.synthesis_failed == 1
        fm = _parse_frontmatter((tmp_path / "GDR.md").read_text())
        assert fm is not None
        assert fm["synthesis"] == "structured-listing"

    @pytest.mark.asyncio
    async def test_llm_exception_falls_back_immediately(
        self, db_session: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        ps = [_make_particle(f"claim {i}") for i in range(3)]
        await _persist_subject_with_particles("GDR", ps)

        import anthropic

        from particles.llm import set_client

        client = MagicMock(spec=anthropic.Anthropic)
        client.messages = MagicMock()
        client.messages.create = MagicMock(side_effect=RuntimeError("model unavailable"))
        set_client(client)
        try:
            exporter = WikiExporter()
            async with session_scope() as session:
                summary = await exporter.export(session, tmp_path)
        finally:
            set_client(None)

        # ``_render_article`` breaks out of the retry loop on LLM exception
        # rather than retrying (the seam itself is broken).
        assert client.messages.create.call_count == 1
        assert summary.synthesis_used == 0
        assert summary.synthesis_failed == 1
        fm = _parse_frontmatter((tmp_path / "GDR.md").read_text())
        assert fm is not None
        assert fm["synthesis"] == "structured-listing"

    @pytest.mark.asyncio
    async def test_fatal_billing_error_aborts_and_writes_no_hashed_article(
        self, db_session: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A billing/credit error aborts synthesis for the whole run and persists
        no article, so the subjects retry next run instead of caching a fallback
        (export-robustness fix). The second subject is never called."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        await _persist_subject_with_particles("GDR", [_make_particle(f"a {i}") for i in range(3)])
        await _persist_subject_with_particles("FRG", [_make_particle(f"b {i}") for i in range(3)])

        import anthropic

        from particles.llm import set_client

        client = MagicMock(spec=anthropic.Anthropic)
        client.messages = MagicMock()
        client.messages.create = MagicMock(
            side_effect=RuntimeError(
                "Error code: 400 - Your credit balance is too low to access the Anthropic API."
            )
        )
        set_client(client)
        try:
            exporter = WikiExporter()
            async with session_scope() as session:
                summary = await exporter.export(session, tmp_path)
        finally:
            set_client(None)

        # Aborted after the first doomed call — the second subject is skipped,
        # not retried per-subject (the no-funds-hammering fix).
        assert client.messages.create.call_count == 1
        assert summary.synthesis_used == 0
        # Neither article is written, so neither caches a hash → both retry.
        assert not (tmp_path / "GDR.md").exists()
        assert not (tmp_path / "FRG.md").exists()


class TestIsFatalLlmError:
    """Classify account-fatal (persistent) vs transient LLM errors (export-robustness)."""

    def test_billing_message_is_fatal(self) -> None:
        from particles.render.article_synthesis.orchestrate import _is_fatal_llm_error

        assert _is_fatal_llm_error(RuntimeError("Your credit balance is too low")) is True

    def test_auth_message_is_fatal(self) -> None:
        from particles.render.article_synthesis.orchestrate import _is_fatal_llm_error

        assert _is_fatal_llm_error(Exception("authentication_error: invalid x-api-key")) is True

    def test_status_401_and_403_are_fatal(self) -> None:
        from particles.render.article_synthesis.orchestrate import _is_fatal_llm_error

        for code in (401, 403):
            exc = Exception("denied")
            exc.status_code = code  # type: ignore[attr-defined]
            assert _is_fatal_llm_error(exc) is True

    def test_transient_and_rate_limit_not_fatal(self) -> None:
        from particles.render.article_synthesis.orchestrate import _is_fatal_llm_error

        assert _is_fatal_llm_error(RuntimeError("model temporarily unavailable")) is False
        rate = Exception("rate limit exceeded")
        rate.status_code = 429  # type: ignore[attr-defined]
        assert _is_fatal_llm_error(rate) is False

    @pytest.mark.asyncio
    async def test_synthesised_article_footnote_includes_source_uri(
        self, db_session: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The References section carries the corpus entry's uri_r."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        # Persist a corpus entry first so the URI lookup hits the row
        from particles.core.schema import CorpusEntry, FetchPolicy, Mutability
        from particles.corpus.store import CorpusEntryRow

        entry = CorpusEntry(
            entry_id="entry-fixture-1",
            uri_r="https://example.com/source",
            source_type="WEB_PAGE",
            mutability=Mutability.STABLE,
            fetch_policy=FetchPolicy.NEVER,
            deposited_by="test",
            tags=[],
        )
        entry_id = entry.entry_id
        async with session_scope() as session:
            session.add(CorpusEntryRow.from_model(entry))
            await session.commit()

        ps = [
            Particle(
                id=str(uuid.uuid4()),
                content=f"claim {i}",
                confidence=Confidence(
                    value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
                ),
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                provenance=[
                    ProvenanceRef(
                        type=ProvenanceRefType.SOURCE,
                        corpus_entry_id=entry_id,
                        snapshot_id="snap-1",
                    )
                ],
                asserted_by="stub-extractor",
                asserted_at=datetime.now(UTC),
                status=Status.ACTIVE,
                extractor_ref={"name": "stub-extractor", "version": "0.1.0"},
                subject_ids=[],
            )
            for i in range(3)
        ]
        await _persist_subject_with_particles("GDR", ps)

        cited_short_ids = [p.id[:8] for p in ps]
        llm_body = f"# GDR\n\nClaim [^p-{cited_short_ids[0]}].\n"

        from particles.llm import set_client

        # synthesis + Layer B judge ("[]" → aligned)
        set_client(_mock_client([llm_body, "[]"]))
        try:
            exporter = WikiExporter()
            async with session_scope() as session:
                await exporter.export(session, tmp_path)
        finally:
            set_client(None)

        article = (tmp_path / "GDR.md").read_text()
        assert "https://example.com/source" in article
        # Only the cited particle gets a footnote body — the other two
        # ACTIVE particles weren't referenced by the LLM, so no footnote
        # for them appears (synthesis can use a subset of the inputs).
        assert article.count("[^p-") >= 2  # ≥1 in body + ≥1 in references
        for uncited in cited_short_ids[1:]:
            assert f"[^p-{uncited}]:" not in article


# ---------------------------------------------------------------------------
# Layer B — semantic-alignment judge (commit 3/3)
# ---------------------------------------------------------------------------


class TestLayerB:
    def test_extract_cited_segments_pulls_context_around_each_citation(self) -> None:
        from particles.exporters.wiki import extract_cited_segments

        body = "Pre. The GDR was founded in 1949 [^p-abcdef12]. Mark currency [^p-00112233]. Tail."
        segs = extract_cited_segments(body, window=50)
        assert len(segs) == 2
        # First segment snaps to sentence boundary, surrounds the marker
        seg1, sid1 = segs[0]
        assert sid1 == "abcdef12"
        assert "GDR was founded in 1949" in seg1
        seg2, sid2 = segs[1]
        assert sid2 == "00112233"
        assert "Mark currency" in seg2

    def test_parse_verdicts_handles_json_array(self) -> None:
        from particles.exporters.wiki import _parse_layer_b_verdicts

        verdicts = _parse_layer_b_verdicts('[{"id":0,"verdict":"aligned","reason":"ok"}]')
        assert verdicts is not None
        assert verdicts[0]["verdict"] == "aligned"

    def test_parse_verdicts_strips_code_fences(self) -> None:
        from particles.exporters.wiki import _parse_layer_b_verdicts

        verdicts = _parse_layer_b_verdicts(
            '```json\n[{"id":0,"verdict":"misaligned","reason":"contradicts source"}]\n```'
        )
        assert verdicts is not None
        assert verdicts[0]["verdict"] == "misaligned"

    def test_parse_verdicts_returns_none_on_garbage(self) -> None:
        from particles.exporters.wiki import _parse_layer_b_verdicts

        assert _parse_layer_b_verdicts("not json at all") is None
        assert _parse_layer_b_verdicts('{"not": "an array"}') is None


class TestLayerBTrichotomy:
    """changes the Layer B verdict from binary
    aligned/misaligned to a supports/unrelated/contradicts trichotomy
    with an operator-tunable tolerance for ornamental citations."""

    def test_normalise_known_verdicts(self) -> None:
        from particles.exporters.article_synthesis import _normalise_verdict

        assert _normalise_verdict("supports") == "supports"
        assert _normalise_verdict("Supports") == "supports"
        assert _normalise_verdict("  SUPPORTED  ") == "supports"
        assert _normalise_verdict("unrelated") == "unrelated"
        assert _normalise_verdict("ornamental") == "unrelated"
        assert _normalise_verdict("contradicts") == "contradicts"
        assert _normalise_verdict("conflict") == "contradicts"

    def test_normalise_legacy_aligned_to_supports(self) -> None:
        """The pre-trichotomy verdict ``aligned`` maps to ``supports``
        so a judge still calibrated to the old prompt doesn't spuriously
        fail an article."""
        from particles.exporters.article_synthesis import _normalise_verdict

        assert _normalise_verdict("aligned") == "supports"

    def test_normalise_unknown_verdict_defaults_to_unrelated(self) -> None:
        """§Consequences: unknown / malformed verdicts degrade
        to the conservative bucket (counts toward tolerance but doesn't
        hard-fail)."""
        from particles.exporters.article_synthesis import _normalise_verdict

        assert _normalise_verdict("partially supports") == "unrelated"
        assert _normalise_verdict("misaligned") == "unrelated"
        assert _normalise_verdict("") == "unrelated"
        assert _normalise_verdict(None) == "unrelated"

    def test_decide_pass_zero_contradicts_within_tolerance(self) -> None:
        from particles.exporters.article_synthesis import _decide_layer_b_pass

        # 2 unrelated / 10 total = 0.20; below default 0.30 tolerance.
        assert _decide_layer_b_pass(supports=8, unrelated=2, contradicts=0, tolerance=0.30) is True

    def test_decide_pass_unrelated_at_tolerance_boundary(self) -> None:
        """Tolerance is inclusive: unrelated_fraction == tolerance passes."""
        from particles.exporters.article_synthesis import _decide_layer_b_pass

        # 3 unrelated / 10 total = 0.30 == tolerance.
        assert _decide_layer_b_pass(supports=7, unrelated=3, contradicts=0, tolerance=0.30) is True

    def test_decide_fail_unrelated_over_tolerance(self) -> None:
        from particles.exporters.article_synthesis import _decide_layer_b_pass

        # 4 unrelated / 10 total = 0.40; above default 0.30 tolerance.
        assert _decide_layer_b_pass(supports=6, unrelated=4, contradicts=0, tolerance=0.30) is False

    def test_decide_fail_any_contradicts(self) -> None:
        """A single contradicts verdict hard-fails regardless of how
        lenient the tolerance is set."""
        from particles.exporters.article_synthesis import _decide_layer_b_pass

        assert _decide_layer_b_pass(supports=9, unrelated=0, contradicts=1, tolerance=1.0) is False

    def test_decide_strict_tolerance_fails_any_unrelated(self) -> None:
        """tolerance=0.0 turns the trichotomy back into "supports-only
        passes" with contradicts handling intact."""
        from particles.exporters.article_synthesis import _decide_layer_b_pass

        assert _decide_layer_b_pass(supports=9, unrelated=1, contradicts=0, tolerance=0.0) is False
        assert _decide_layer_b_pass(supports=10, unrelated=0, contradicts=0, tolerance=0.0) is True

    @pytest.mark.asyncio
    async def test_layer_b_check_returns_structured_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: judge returns a trichotomy verdict mix; the
        helper bins them, applies tolerance, and returns LayerBResult."""
        from particles.exporters.article_synthesis import LayerBResult, layer_b_check
        from particles.llm import set_client

        ps = [
            _make_particle("claim A"),
            _make_particle("claim B"),
            _make_particle("claim C"),
        ]
        cited = [p.id[:8] for p in ps]
        body = f"Para A [^p-{cited[0]}].\n\nPara B [^p-{cited[1]}].\n\nPara C [^p-{cited[2]}]."
        # 1 supports + 1 unrelated + 1 contradicts.
        judge_response = (
            "["
            '{"id":0,"verdict":"supports","reason":"ok"},'
            '{"id":1,"verdict":"unrelated","reason":"ornamental"},'
            '{"id":2,"verdict":"contradicts","reason":"flat-out wrong"}'
            "]"
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        set_client(_mock_client([judge_response]))
        try:
            result = await layer_b_check(body, ps, unrelated_tolerance=0.5)
        finally:
            set_client(None)

        assert isinstance(result, LayerBResult)
        assert result.supports_count == 1
        assert result.unrelated_count == 1
        assert result.contradicts_count == 1
        # contradicts > 0 → hard fail regardless of tolerance.
        assert result.passed is False
        # Misalignments expose both unrelated and contradicts (not supports)
        # so the retry prompt can show what's wrong.
        verdict_set = {m["verdict"] for m in result.misalignments}
        assert verdict_set == {"unrelated", "contradicts"}

    @pytest.mark.asyncio
    async def test_layer_b_check_no_citations_returns_none(self) -> None:
        from particles.exporters.article_synthesis import layer_b_check

        result = await layer_b_check("No citations in this body.", [])
        assert result.passed is None
        assert result.total == 0

    def test_build_layer_b_retry_prompt_includes_misalignments(self) -> None:
        """The Layer-B-specific strict prompt must show the LLM exactly
        which (claim, citation) pairs the judge rejected and why."""
        from particles.exporters.article_synthesis import _build_synthesis_prompt

        s = Subject(id=str(uuid.uuid4()), canonical_name="GDR", asserted_by="t")
        ps = [_make_particle("claim a"), _make_particle("claim b")]
        eff: dict[str, float] = {}
        misalignments = [
            {"id": 0, "verdict": "unrelated", "reason": "citation is ornamental"},
            {"id": 1, "verdict": "contradicts", "reason": "particle disagrees"},
        ]
        prompt = _build_synthesis_prompt(
            subject=s,
            particles=ps,
            eff=eff,
            strict=True,
            layer_b_misalignments=misalignments,
        )
        assert "SEMANTIC-ALIGNMENT VALIDATION" in prompt
        assert "pair 0 [unrelated]: citation is ornamental" in prompt
        assert "pair 1 [contradicts]: particle disagrees" in prompt

    def test_build_layer_b_retry_prompt_discourages_zero_citation_output(self) -> None:
        """Operator dry-run after shipped showed the retry
        prompt drove the LLM to drop ALL claims (the laziest of three
        equally-presented fix options), yielding zero-citation output
        that Layer A then rejected — strictly worse than the prior
        behaviour for ~6/7 of the subjects that hit retry. The prompt
        must explicitly:
          (a) prioritise rewriting and replace-citation over dropping
              the claim entirely;
          (b) call out drop as a *last resort*, not a co-equal option;
          (c) demand the retried article still contain citations.
        """
        from particles.exporters.article_synthesis import _build_synthesis_prompt

        s = Subject(id=str(uuid.uuid4()), canonical_name="X", asserted_by="t")
        ps = [_make_particle("p")]
        prompt = _build_synthesis_prompt(
            subject=s,
            particles=ps,
            eff={},
            strict=True,
            layer_b_misalignments=[{"id": 0, "verdict": "unrelated", "reason": "noise"}],
        )
        # (a) + (b): drop is a last resort.
        assert "last resort" in prompt.lower()
        assert "REWRITE" in prompt
        assert "REPLACE" in prompt
        assert "DROP" in prompt
        # The order in the prompt matters: rewrite before replace before drop.
        rewrite_at = prompt.index("REWRITE")
        replace_at = prompt.index("REPLACE")
        drop_at = prompt.index("DROP")
        assert rewrite_at < replace_at < drop_at
        # (c): minimum-citation requirement is explicit.
        assert "must still be cited" in prompt.lower()
        assert "zero" in prompt.lower()  # "zero citations" warning

    def test_build_synthesis_prompt_layer_a_strict_without_misalignments(self) -> None:
        """When ``layer_b_misalignments`` is None, the existing Layer-A
        strict prompt is used (citation-IDs-invented messaging)."""
        from particles.exporters.article_synthesis import _build_synthesis_prompt

        s = Subject(id=str(uuid.uuid4()), canonical_name="GDR", asserted_by="t")
        ps = [_make_particle("claim a")]
        prompt = _build_synthesis_prompt(
            subject=s, particles=ps, eff={}, strict=True, layer_b_misalignments=None
        )
        # Layer-A strict prompt is the existing one — covers invented IDs.
        assert "PRIOR ATTEMPT WAS REJECTED" in prompt
        assert "SEMANTIC-ALIGNMENT VALIDATION" not in prompt

    @pytest.mark.asyncio
    async def test_misaligned_judge_triggers_retry_then_fallback(
        self, db_session: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Layer B retry path (opt-in via config): two attempts both
        pass Layer A but Layer B rejects both → fall back.

        Amendment (v0.23.2): Layer B retry default is now
        False. Mutate the in-process config attr to True (rather than
        ``reset_config()`` after setting an env var, which would also
        tear down the cached engine the ``db_session`` fixture relies
        on)."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from particles.config import get_config

        get_config().wiki.layer_b_retry_enabled = True
        ps = [_make_particle(f"claim {i}") for i in range(3)]
        await _persist_subject_with_particles("GDR", ps)

        cited = [p.id[:8] for p in ps]
        body1 = f"# GDR\n\nClaim one [^p-{cited[0]}].\n"
        body2 = f"# GDR\n\nClaim two [^p-{cited[0]}].\n"
        # contradicts hard-fails the article regardless of tolerance.
        bad_verdict = '[{"id":0,"verdict":"contradicts","reason":"claim contradicts source"}]'

        from particles.llm import set_client

        # synthesis 1 → judge 1 (contradicts) → strict retry → judge 2 (contradicts)
        client = _mock_client([body1, bad_verdict, body2, bad_verdict])
        set_client(client)
        try:
            exporter = WikiExporter()
            async with session_scope() as session:
                summary = await exporter.export(session, tmp_path)
        finally:
            set_client(None)

        assert client.messages.create.call_count == 4
        assert summary.synthesis_used == 0
        assert summary.synthesis_failed == 1
        fm = _parse_frontmatter((tmp_path / "GDR.md").read_text())
        assert fm is not None
        assert fm["synthesis"] == "structured-listing"

    @pytest.mark.asyncio
    async def test_layer_b_failure_falls_back_immediately_by_default(
        self, db_session: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """amendment (v0.23.2): with the default
        ``wiki.layer_b_retry_enabled=False``, a Layer B failure on
        attempt 1 falls through to structured-listing immediately
        without burning a second synthesis call + judge call."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        ps = [_make_particle(f"claim {i}") for i in range(3)]
        await _persist_subject_with_particles("GDR", ps)

        cited = [p.id[:8] for p in ps]
        body1 = f"# GDR\n\nClaim one [^p-{cited[0]}].\n"
        bad_verdict = '[{"id":0,"verdict":"contradicts","reason":"flat-out wrong"}]'

        from particles.llm import set_client

        # Only 2 calls expected: 1 synthesis + 1 judge. No retry.
        client = _mock_client([body1, bad_verdict])
        set_client(client)
        try:
            exporter = WikiExporter()
            async with session_scope() as session:
                summary = await exporter.export(session, tmp_path)
        finally:
            set_client(None)

        assert client.messages.create.call_count == 2
        assert summary.synthesis_used == 0
        assert summary.synthesis_failed == 1
        fm = _parse_frontmatter((tmp_path / "GDR.md").read_text())
        assert fm is not None
        assert fm["synthesis"] == "structured-listing"

    @pytest.mark.asyncio
    async def test_unrelated_within_tolerance_passes(
        self, db_session: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ornamental citations within
        ``layer_b_unrelated_tolerance`` pass without retry."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        ps = [_make_particle(f"claim {i}") for i in range(3)]
        await _persist_subject_with_particles("GDR", ps)

        cited = [p.id[:8] for p in ps]
        # 5 citations; 1 unrelated (20%) — under the default 30% tolerance.
        body = (
            f"# GDR\n\n"
            f"A [^p-{cited[0]}]. B [^p-{cited[1]}]. C [^p-{cited[2]}]. "
            f"D [^p-{cited[0]}]. E [^p-{cited[1]}]."
        )
        judge = (
            "["
            '{"id":0,"verdict":"supports","reason":"ok"},'
            '{"id":1,"verdict":"supports","reason":"ok"},'
            '{"id":2,"verdict":"supports","reason":"ok"},'
            '{"id":3,"verdict":"unrelated","reason":"ornamental"},'
            '{"id":4,"verdict":"supports","reason":"ok"}'
            "]"
        )

        from particles.llm import set_client

        client = _mock_client([body, judge])
        set_client(client)
        try:
            exporter = WikiExporter()
            async with session_scope() as session:
                summary = await exporter.export(session, tmp_path)
        finally:
            set_client(None)

        # One synthesis + one judge call only; no retry.
        assert client.messages.create.call_count == 2
        assert summary.synthesis_used == 1
        assert summary.synthesis_failed == 0
        fm = _parse_frontmatter((tmp_path / "GDR.md").read_text())
        assert fm is not None
        assert fm["synthesis"] == "llm"
        assert fm["layer_b_passed"] is True
        # Frontmatter surfaces the judge's verdict distribution per
        # so operators can audit and tune tolerance.
        assert fm["layer_b_unrelated_count"] == 1
        assert fm["layer_b_contradicts_count"] == 0

    @pytest.mark.asyncio
    async def test_layer_b_failure_uses_layer_b_specific_retry_prompt(
        self, db_session: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """when attempt 1 fails Layer B, the retry must
        use the Layer-B-specific prompt (not the Layer-A one), with the
        judge-flagged misaligned pairs interpolated so the LLM knows
        what to fix.

        Amendment (v0.23.2): Layer B retry default is now
        False. Mutate the in-process config attr to True to exercise
        the retry path's prompt-selection logic."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from particles.config import get_config

        get_config().wiki.layer_b_retry_enabled = True
        ps = [_make_particle(f"claim {i}") for i in range(3)]
        await _persist_subject_with_particles("GDR", ps)

        cited = [p.id[:8] for p in ps]
        body1 = f"# GDR\n\nClaim A [^p-{cited[0]}]."
        body2 = f"# GDR\n\nClaim A (rewritten) [^p-{cited[0]}]."
        # Attempt 1: contradicts → hard fail.
        judge1 = '[{"id":0,"verdict":"contradicts","reason":"flat-out wrong"}]'
        # Attempt 2: supports → pass.
        judge2 = '[{"id":0,"verdict":"supports","reason":"ok now"}]'

        from particles.llm import set_client

        client = _mock_client([body1, judge1, body2, judge2])
        set_client(client)
        try:
            exporter = WikiExporter()
            async with session_scope() as session:
                summary = await exporter.export(session, tmp_path)
        finally:
            set_client(None)

        # Inspect the third LLM call's prompt — that's the strict
        # retry synthesis call after the first attempt failed Layer B.
        retry_prompt = client.messages.create.call_args_list[2].kwargs["messages"][0]["content"]
        assert "SEMANTIC-ALIGNMENT VALIDATION" in retry_prompt
        # The judge's misalignment reason from attempt 1 must appear so
        # the LLM has a fighting chance of fixing it.
        assert "flat-out wrong" in retry_prompt
        # And specifically NOT the Layer A strict-prompt opening.
        assert "PRIOR ATTEMPT WAS REJECTED" not in retry_prompt

        assert summary.synthesis_used == 1
        fm = _parse_frontmatter((tmp_path / "GDR.md").read_text())
        assert fm is not None
        assert fm["layer_b_passed"] is True
        # The successful retry's judge result is what surfaces.
        assert fm["layer_b_unrelated_count"] == 0
        assert fm["layer_b_contradicts_count"] == 0

    @pytest.mark.asyncio
    async def test_layer_b_disabled_skips_judge_entirely(
        self, db_session: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """layer_b_enabled=False short-circuits the judge: only one LLM call per article."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        # Mutate the in-process WikiConfig field directly. We cannot call
        # ``reset_config()`` here because the autouse ``clear_subject_cache``
        # fixture already did, *and* it drops the cached engine — calling
        # it again mid-test would tear down the in-memory tables the
        # ``db_session`` fixture just created.
        from particles.config import get_config

        original = get_config().wiki.layer_b_enabled
        get_config().wiki.layer_b_enabled = False
        try:
            ps = [_make_particle(f"claim {i}") for i in range(3)]
            await _persist_subject_with_particles("GDR", ps)
            cited = [p.id[:8] for p in ps]
            body = f"# GDR\n\nClaim [^p-{cited[0]}].\n"

            from particles.llm import set_client

            client = _mock_client([body])
            set_client(client)
            try:
                exporter = WikiExporter()
                async with session_scope() as session:
                    summary = await exporter.export(session, tmp_path)
            finally:
                set_client(None)

            # Just one call — synthesis only, no Layer B
            assert client.messages.create.call_count == 1
            assert summary.synthesis_used == 1
            fm = _parse_frontmatter((tmp_path / "GDR.md").read_text())
            assert fm is not None
            assert fm["layer_a_passed"] is True
            # layer_b_passed is None because Layer B was disabled — recorded
            # explicitly in the frontmatter so the operator can see the gap.
            assert fm["layer_b_passed"] is None
        finally:
            get_config().wiki.layer_b_enabled = original


# ---------------------------------------------------------------------------
# Lint integration (commit 3/3)
# ---------------------------------------------------------------------------


class TestLintIntegration:
    @pytest.mark.asyncio
    async def test_lint_callouts_are_a_write_time_layer(
        self, db_session: object, tmp_path: Path
    ) -> None:
        """the rendered body is callout-free (the cache must not
        freeze lint text); ``apply_lint_callouts`` splices the callout in at
        write time. The ``contradictions`` count is still set by the render."""
        from particles.core.schema import LintFinding
        from particles.exporters.article_synthesis import apply_lint_callouts
        from particles.exporters.wiki import render_structured_listing

        subj = Subject(id=str(uuid.uuid4()), canonical_name="GDR", asserted_by="t")
        ps = [_make_particle(f"claim {i}") for i in range(2)]
        eff = {p.id: 0.5 for p in ps}
        findings = [
            LintFinding(
                particle_id=ps[0].id,
                subject_id=None,
                corpus_entry_id=None,
                finding_type="CONTRADICTION",
                severity="ERROR",
                detail="Disagrees with [^p-other].",
                recommended_action="Review and reconcile.",
            )
        ]

        body = render_structured_listing(
            subj, ps, eff, input_hash=compute_input_hash(ps), lint_findings=findings
        )
        fm = _parse_frontmatter(body)
        assert fm is not None
        assert fm["contradictions"] == 1
        # The cached body is callout-free.
        assert "> [!danger] CONTRADICTION" not in body

        # The write-time layer splices the callout in, after the H1.
        final = apply_lint_callouts(body, findings)
        assert "> [!danger] CONTRADICTION" in final
        assert "Disagrees with" in final
        assert "Review and reconcile" in final
        # Idempotent: re-applying the same findings is a no-op on the text.
        assert apply_lint_callouts(final, findings) == final

    @pytest.mark.asyncio
    async def test_lint_pass_attaches_findings_to_relevant_articles(
        self, db_session: object, tmp_path: Path
    ) -> None:
        """End-to-end: a PROVENANCE_STALE finding surfaces in the article callouts."""
        from particles.store.particle_store import ParticleRow

        # 1) Persist subject + particles. We'll mutate one particle's status
        # to PROVENANCE_STALE so the staleness lint check raises a finding
        # for its (related) ACTIVE neighbour.
        ps = [_make_particle(f"claim {i}") for i in range(3)]
        await _persist_subject_with_particles("GDR", ps)

        # Lower-level: monkey-set the particle's status to a stale state.
        async with session_scope() as session:
            row = await session.get(ParticleRow, ps[0].id)
            assert row is not None
            row.status = Status.PROVENANCE_STALE.value
            await session.commit()

        # Now only ps[1] and ps[2] are ACTIVE → subject still has 2 ≥
        # default-3? No, that's below threshold. Lower the threshold for
        # this test so the article still gets generated.
        exporter = WikiExporter()
        async with session_scope() as session:
            summary = await exporter.export(session, tmp_path, min_particles=2)
        assert summary.articles_written is not None and summary.articles_written >= 1

        text = (tmp_path / "GDR.md").read_text()
        fm = _parse_frontmatter(text)
        assert fm is not None
        # Lint pre-pass ran — the article frontmatter records the
        # CONTRADICTION count even if zero.
        assert "contradictions" in fm


class TestWithoutSynthesis:
    """: the deterministic, no-LLM ``--without-synthesis`` gate."""

    @pytest.mark.asyncio
    async def test_render_article_short_circuits_before_llm_and_cache(self) -> None:
        from unittest.mock import AsyncMock, patch

        from particles.exporters.article_synthesis import render_article

        ps = [_make_particle(f"claim {i}") for i in range(3)]
        subj = Subject(id=str(uuid.uuid4()), canonical_name="GDR", asserted_by="t")
        eff = {p.id: 0.9 for p in ps}
        h = compute_input_hash(ps, subj)

        with (
            patch(
                "particles.render.article_synthesis.orchestrate._call_synthesis_llm",
                new=AsyncMock(side_effect=AssertionError("LLM must not be called")),
            ) as llm,
            patch(
                "particles.store.synthesis_cache_store.lookup_cached_article",
                new=AsyncMock(side_effect=AssertionError("cache must not be consulted")),
            ) as cache,
        ):
            body, used = await render_article(
                subject=subj,
                particles=ps,
                eff=eff,
                input_hash=h,
                corpus_uris={},
                max_tokens=1000,
                layer_b_enabled=True,
                session=None,
                without_synthesis=True,
            )

        llm.assert_not_called()
        cache.assert_not_called()
        assert used is False
        fm = _parse_frontmatter(body)
        assert fm is not None
        assert fm["synthesis"] == "structured-listing"

    @pytest.mark.asyncio
    async def test_export_writes_deterministic_listings_ignoring_the_client(
        self, db_session: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A working API key + a client that raises if used: the gate must
        # short-circuit before any LLM call, writing structured listings.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        await _persist_subject_with_particles(
            "GDR", [_make_particle(f"claim {i}") for i in range(3)]
        )

        from particles.llm import set_client

        boom = MagicMock()
        boom.messages.create = MagicMock(side_effect=AssertionError("LLM must not be called"))
        set_client(boom)
        try:
            async with session_scope() as session:
                summary = await WikiExporter().export(session, tmp_path, without_synthesis=True)
        finally:
            set_client(None)

        assert summary.articles_written == 1
        assert summary.synthesis_skipped == 1
        assert summary.synthesis_used == 0
        assert summary.synthesis_failed == 0
        fm = _parse_frontmatter((tmp_path / "GDR.md").read_text())
        assert fm is not None
        assert fm["synthesis"] == "structured-listing"


# ---------------------------------------------------------------------------
# Per-NARRATIVE articles (mechanism in the wiki)
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


class TestNarrativeArticles:
    """One cited article per ACTIVE NARRATIVE under ``Narratives/``, using the
    same render path and cache the Obsidian notes use."""

    @pytest.mark.asyncio
    async def test_narrative_article_written_and_indexed(
        self, db_session: object, tmp_path: Path
    ) -> None:
        ps = [_make_particle(f"claim {i}") for i in range(3)]
        await _persist_subject_with_particles("Shared", ps)
        narrative = await _persist_narrative("A hard day.", ps)

        # No API key (autouse fixture) → the deterministic structured listing.
        async with session_scope() as session:
            summary = await WikiExporter().export(session, tmp_path)

        assert summary.narrative_articles == 1
        article = tmp_path / "Narratives" / "A hard day.md"
        assert article.exists(), sorted(str(p) for p in tmp_path.rglob("*.md"))
        body = article.read_text(encoding="utf-8")
        assert "# A hard day." in body
        # Every constituent is cited — the narrative's whole arc, in order.
        for p in ps:
            assert p.id[:8] in body
        # The index links it by its relative path so the wikilink resolves.
        index = (tmp_path / "index.md").read_text(encoding="utf-8")
        assert "[[Narratives/A hard day]]" in index

        # A second export is a cache hit and rewrites nothing.
        async with session_scope() as session:
            second = await WikiExporter().export(session, tmp_path)
        assert second.narrative_articles == 1
        assert second.cache_hits >= 1
        assert narrative.id  # narrative persisted; id used for the cache key

    @pytest.mark.asyncio
    async def test_below_constituent_floor_is_skipped(
        self, db_session: object, tmp_path: Path
    ) -> None:
        # Default `exporter_common.synthesis_min_particles` is 3 — a two-claim
        # "narrative" is a stub and not worth an article.
        ps = [_make_particle(f"claim {i}") for i in range(2)]
        await _persist_subject_with_particles("Shared", ps)
        await _persist_narrative("Too thin.", ps)

        async with session_scope() as session:
            summary = await WikiExporter().export(session, tmp_path)

        assert summary.narrative_articles == 0
        assert not (tmp_path / "Narratives").exists()

    @pytest.mark.asyncio
    async def test_subject_filter_suppresses_narratives(
        self, db_session: object, tmp_path: Path
    ) -> None:
        # Narratives are subject-less: a run narrowed to named subjects asked
        # for a slice of the subject namespace, not the whole store.
        ps = [_make_particle(f"claim {i}") for i in range(3)]
        await _persist_subject_with_particles("Shared", ps)
        await _persist_narrative("A hard day.", ps)

        async with session_scope() as session:
            summary = await WikiExporter().export(session, tmp_path, subjects=["Shared"])

        assert summary.narrative_articles is None
        assert not (tmp_path / "Narratives").exists()

    @pytest.mark.asyncio
    async def test_knob_off_suppresses_narratives(
        self, db_session: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config

        monkeypatch.setattr(get_config().wiki, "emit_narrative_notes", False)

        ps = [_make_particle(f"claim {i}") for i in range(3)]
        await _persist_subject_with_particles("Shared", ps)
        await _persist_narrative("A hard day.", ps)

        async with session_scope() as session:
            summary = await WikiExporter().export(session, tmp_path)

        assert summary.narrative_articles is None
        assert not (tmp_path / "Narratives").exists()

    @pytest.mark.asyncio
    async def test_dry_run_writes_nothing(self, db_session: object, tmp_path: Path) -> None:
        ps = [_make_particle(f"claim {i}") for i in range(3)]
        await _persist_subject_with_particles("Shared", ps)
        await _persist_narrative("A hard day.", ps)

        async with session_scope() as session:
            summary = await WikiExporter().export(session, tmp_path, dry_run=True)

        assert summary.narrative_articles == 0
        assert not (tmp_path / "Narratives").exists()

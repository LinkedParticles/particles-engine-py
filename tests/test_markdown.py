"""Tests for the shared Markdown Bridge / exporter utility module."""

from __future__ import annotations

from pathlib import Path

import pytest

from particles.core.schema import (
    Confidence,
    ExternalRef,
    Particle,
    ParticleType,
    Subject,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.exporters.markdown import (
    atomic_write_text,
    build_narrative_naming,
    build_subject_naming,
    disambiguation_name,
    is_within_directory,
    prune_obsolete_markdown,
    sanitize_filename,
    subject_slug,
)


def _subj(
    name: str,
    *,
    description: str | None = None,
    subject_class: str | None = None,
    wikidata: str | None = None,
    confidence: float = 0.5,
) -> Subject:
    return Subject(
        canonical_name=name,
        description=description,
        subject_class=subject_class,
        external_ids=(
            [
                ExternalRef(
                    namespace="wikidata",
                    id=wikidata,
                    uri=f"http://x/{wikidata}",
                    confidence=confidence,
                )
            ]
            if wikidata
            else []
        ),
        asserted_by="test",
    )


class TestSubjectNaming:
    def test_unique_names_keep_bare_display_and_no_groups(self) -> None:
        subs = [_subj("Prometheus"), _subj("Grafana")]
        naming = build_subject_naming(subs)
        assert naming.display_name(subs[0]) == "Prometheus"
        assert naming.display_name(subs[1]) == "Grafana"
        assert not naming.has_collisions
        assert naming.groups == ()

    def test_collision_qualifies_by_description_with_distinct_slugs(self) -> None:
        a = _subj(
            "Prometheus", description="event monitoring and alerting software", wikidata="Q52534999"
        )
        b = _subj(
            "Prometheus", description="Titan, culture hero, and trickster figure", wikidata="Q83160"
        )
        naming = build_subject_naming([a, b])
        # Tier 2 distils the trailing noun of the pre-comma head.
        assert naming.display_name(a) == "Prometheus (software)"
        assert naming.display_name(b) == "Prometheus (Titan)"
        # The two display names must produce distinct slugs — the whole point.
        assert subject_slug(naming.display_name(a)) != subject_slug(naming.display_name(b))
        assert naming.has_collisions
        assert len(naming.groups) == 1
        assert naming.groups[0].base_name == "Prometheus"
        assert set(naming.groups[0].member_ids) == {a.id, b.id}

    def test_class_tier_wins_when_classes_distinct(self) -> None:
        a = _subj("Mercury", subject_class="astro:Planet", description="same gloss")
        b = _subj("Mercury", subject_class="chem:Element", description="same gloss")
        naming = build_subject_naming([a, b])
        assert naming.display_name(a) == "Mercury (Planet)"
        assert naming.qualifier_by_id[a.id] == "Planet"
        assert naming.qualifier_by_id[b.id] == "Element"

    def test_falls_through_to_external_id_when_class_and_desc_absent(self) -> None:
        a = _subj("Echo", wikidata="Q1")
        b = _subj("Echo", wikidata="Q2")
        naming = build_subject_naming([a, b])
        assert naming.qualifier_by_id[a.id] == "wikidata Q1"
        assert naming.qualifier_by_id[b.id] == "wikidata Q2"

    def test_terminal_id_fallback_when_nothing_distinct(self) -> None:
        a = _subj("Nameless")
        b = _subj("Nameless")
        naming = build_subject_naming([a, b])
        # No class / description / external-id → short subject-id qualifier.
        assert naming.qualifier_by_id[a.id] == f"id {a.id[:8]}"
        assert naming.qualifier_by_id[b.id] == f"id {b.id[:8]}"
        assert subject_slug(naming.display_name(a)) != subject_slug(naming.display_name(b))

    def test_three_way_collision_all_distinct(self) -> None:
        subs = [
            _subj("Prometheus", description="event monitoring and alerting software"),
            _subj("Prometheus", description="Titan in Greek mythology"),
            _subj("Prometheus", description="genus of the order Lepidoptera"),
        ]
        naming = build_subject_naming(subs)
        slugs = {subject_slug(naming.display_name(s)) for s in subs}
        assert len(slugs) == 3
        assert naming.groups[0].member_ids == tuple(sorted(s.id for s in subs))

    def test_description_qualifier_distils_trailing_noun(self) -> None:
        a = _subj("Echo", description="event monitoring and alerting software")
        b = _subj("Echo", description="genus of the order Lepidoptera")
        naming = build_subject_naming([a, b])
        assert naming.qualifier_by_id[a.id] == "software"
        assert naming.qualifier_by_id[b.id] == "Lepidoptera"

    def test_description_qualifier_falls_back_to_phrase_when_last_token_noisy(self) -> None:
        # Trailing token "1947)" isn't a clean word → leading-phrase fallback.
        a = _subj("Quark", description="American politician (born 1947)")
        b = _subj("Quark", description="British physicist (born 1950)")
        naming = build_subject_naming([a, b])
        assert naming.qualifier_by_id[a.id] == "American politician (born 1947)"
        assert naming.qualifier_by_id[b.id] == "British physicist (born 1950)"
        assert subject_slug(naming.display_name(a)) != subject_slug(naming.display_name(b))

    def test_same_trailing_noun_falls_through_to_external_id(self) -> None:
        # Both distil to "software" → tier 2 rejected → external-id tier.
        a = _subj("Bond", description="backup software", wikidata="Q1")
        b = _subj("Bond", description="monitoring software", wikidata="Q2")
        naming = build_subject_naming([a, b])
        assert naming.qualifier_by_id[a.id] == "wikidata Q1"
        assert naming.qualifier_by_id[b.id] == "wikidata Q2"

    def test_disambiguation_name_helper(self) -> None:
        assert disambiguation_name("Prometheus") == "Prometheus (disambiguation)"


class TestBuildNarrativeNaming:
    """narrative-note naming — slug the label, id-suffix on collision."""

    @staticmethod
    def _narr(label: str, pid: str) -> Particle:
        return Particle(
            id=pid,
            content=label,
            confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="test",
            particle_type=ParticleType.NARRATIVE,
        )

    def test_unique_label_keeps_bare_slug(self) -> None:
        n = self._narr("A hard day the author got through", "11111111-0000-0000-0000-000000000001")
        out = build_narrative_naming([n])
        assert out[n.id] == subject_slug("A hard day the author got through")

    def test_colliding_labels_get_id_suffix(self) -> None:
        a = self._narr("Same label", "aaaaaaaa-0000-0000-0000-000000000001")
        b = self._narr("Same label", "bbbbbbbb-0000-0000-0000-000000000002")
        out = build_narrative_naming([a, b])
        assert out[a.id] != out[b.id]
        assert out[a.id].endswith(a.id[:8])
        assert out[b.id].endswith(b.id[:8])


class TestNarrativeAsSubject:
    """the NARRATIVE → ``Subject`` adapter the render engine
    takes. Shared by all three prose exporters, so it lives here."""

    def test_label_becomes_title_and_id_is_preserved(self) -> None:
        from particles.render.markdown import narrative_as_subject

        n = TestBuildNarrativeNaming._narr("A hard day.", "11111111-0000-0000-0000-000000000001")
        synthetic = narrative_as_subject(n)
        # The narrative id IS the synthesis-cache key — a distinct id space
        # from real Subjects, so it must survive the adaptation verbatim.
        assert synthetic.id == n.id
        assert synthetic.canonical_name == "A hard day."


class TestAtomicWriteText:
    def test_creates_file_at_target_path(self, tmp_path: Path) -> None:
        target = tmp_path / "subject.md"
        atomic_write_text(target, "hello world")
        assert target.read_text() == "hello world"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        # github.com/login.md needs the github.com/ dir to spring into existence.
        target = tmp_path / "github.com" / "barrygfox.md"
        atomic_write_text(target, "content")
        assert target.exists()
        assert target.read_text() == "content"

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "subject.md"
        target.write_text("old content")
        atomic_write_text(target, "new content")
        assert target.read_text() == "new content"

    def test_does_not_leave_tempfile_on_success(self, tmp_path: Path) -> None:
        """Successful write: directory contains only the target, no `.subject.md.<pid>.tmp`."""
        target = tmp_path / "subject.md"
        atomic_write_text(target, "x")
        contents = {p.name for p in tmp_path.iterdir()}
        # Only the target — no hidden tempfile siblings.
        assert contents == {"subject.md"}

    def test_does_not_leave_tempfile_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If os.replace fails, the tempfile should be cleaned up."""
        import os as _os

        target = tmp_path / "subject.md"

        def boom(*args: object, **kwargs: object) -> None:
            raise OSError("simulated rename failure")

        monkeypatch.setattr(_os, "replace", boom)
        with pytest.raises(OSError, match="simulated rename failure"):
            atomic_write_text(target, "content")
        # Target was never created; tempfile was cleaned up.
        contents = list(tmp_path.iterdir())
        assert contents == [], f"unexpected leftover files: {contents}"

    def test_handles_unicode_content(self, tmp_path: Path) -> None:
        target = tmp_path / "subject.md"
        atomic_write_text(target, "✓ aluminium · 1.1 g · ⌀ 19 mm")
        assert target.read_text() == "✓ aluminium · 1.1 g · ⌀ 19 mm"

    def test_writes_to_sibling_tempfile_in_same_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The tempfile must live in the same directory as the target so the
        rename stays on one filesystem (and is therefore atomic)."""
        import os as _os

        observed_renames: list[tuple[str, str]] = []
        real_replace = _os.replace

        def capture(src: str | Path, dst: str | Path) -> None:
            observed_renames.append((str(src), str(dst)))
            real_replace(src, dst)

        monkeypatch.setattr(_os, "replace", capture)
        target = tmp_path / "subdir" / "subject.md"
        atomic_write_text(target, "x")
        assert len(observed_renames) == 1
        src, dst = observed_renames[0]
        assert dst == str(target)
        # src lives in the same dir as dst (atomic-rename invariant).
        assert Path(src).parent == Path(dst).parent


class TestSubjectSlug:
    """Single source of truth for subject → vault-path mapping.

    Every exporter that writes one file per Subject must round-trip
    through ``subject_slug``; identical Subjects in different exporters
    must produce identical paths.
    """

    @pytest.mark.parametrize(
        ("canonical_name", "expected"),
        [
            # Plain names
            ("GDR", "GDR"),
            ("1 Pfennig (1948-1950) GDR", "1 Pfennig (1948-1950) GDR"),
            # Filesystem-unsafe chars
            ("foo/bar:baz", "foo-bar-baz"),
            ('weird"name<here>', "weird-name-here-"),
            # Whitespace collapses
            ("  spaces   collapse  ", "spaces collapse"),
            # Empty falls back
            ("", "unnamed"),
            # Reddit users + subreddits nest under reddit.com/
            ("u/barrygfox", "reddit.com/u/barrygfox"),
            ("r/MachineLearning", "reddit.com/r/MachineLearning"),
            # GitHub authors nest under github.com/
            ("github:barrygfox", "github.com/barrygfox"),
            ("github:torvalds", "github.com/torvalds"),
        ],
    )
    def test_subject_slug_mapping(self, canonical_name: str, expected: str) -> None:
        assert subject_slug(canonical_name) == expected

    def test_obsidian_and_wiki_produce_identical_slug(self) -> None:
        """The regression that fix #5 was filed for: obsidian.py and
        wiki.py both compute paths through the same helper, so the same
        Subject lands at the same path in every exporter."""
        from particles.exporters import obsidian, wiki

        cases = [
            "github:barrygfox",
            "u/someone",
            "r/MachineLearning",
            "GDR",
            "1 Pfennig (1948-1950) GDR",
            "foo/bar",
        ]
        for canonical in cases:
            assert obsidian._subject_slug(canonical) == wiki._subject_slug(canonical), (
                f"slug divergence for {canonical!r}: "
                f"obsidian={obsidian._subject_slug(canonical)!r} "
                f"wiki={wiki._subject_slug(canonical)!r}"
            )


def test_sanitize_filename_round_trip() -> None:
    assert sanitize_filename("a/b:c") == "a-b-c"
    assert sanitize_filename("  spaces   collapse  ") == "spaces collapse"
    assert sanitize_filename("") == "unnamed"
    assert sanitize_filename("normal-name") == "normal-name"


@pytest.mark.parametrize(
    "hostile, expected",
    [
        # Leading dots — no hidden files, no bare "." / ".." escaping.
        (".hidden", "hidden"),
        ("...", "-."),  # ".." → "-", trailing "." kept (not leading, harmless)
        (".", "unnamed"),
        ("..", "-"),
        # ".." path token anywhere is neutralised.
        ("foo..bar", "foo-bar"),
        ("../etc/passwd", "--etc-passwd"),  # "/" → "-" first, then ".." → "-"
        # Windows reserved device-name stems get a dash prefix (regardless of
        # any pseudo-extension), but legitimate names containing them do not.
        ("CON", "-CON"),
        ("nul", "-nul"),
        ("COM1", "-COM1"),
        ("LPT9", "-LPT9"),
        ("CON.txt", "-CON.txt"),
        ("console", "console"),
        ("COMET", "COMET"),
    ],
)
def test_sanitize_filename_hostile_names(hostile: str, expected: str) -> None:
    """Hostile / pathological subject names from untrusted docs stay safe."""
    assert sanitize_filename(hostile) == expected


class TestSubjectSlugTraversal:
    """Security finding F1 regression — path-traversal write primitive.

    A Subject ``canonical_name`` is LLM-extracted from untrusted deposited
    documents (validated only for ``min_length=1``), so a poisoned source can
    steer the extractor to emit a name like ``github:../../../../etc/cron.d/x``.
    The ``u/`` / ``r/`` / ``github:`` shard branches of ``subject_slug`` used to
    return the raw remainder, letting that ``..`` escape the export directory
    when an exporter computed ``output_dir / f"{slug}.md"``. The remainder is
    now routed through ``sanitize_filename`` per path-segment.
    """

    @pytest.mark.parametrize(
        "canonical_name",
        [
            # The exact PoC names from the F1 finding.
            "u/../../../evil",
            "github:../../../../etc/cron.d/x",
            "r/../../x",
            # A few more traversal shapes.
            "u/../../../../etc/passwd",
            "r/foo/../../../bar",
            "github:..\\..\\etc",  # backslash separators are dashed within a segment
            # The disambiguation call site re-slugs "{name} ({qual})".
            "github:../../../../etc/cron.d/x (id 12345678)",
        ],
    )
    def test_traversal_slugs_cannot_escape_output_dir(
        self, tmp_path: Path, canonical_name: str
    ) -> None:
        slug = subject_slug(canonical_name)
        # No parent-dir token survives anywhere in the slug.
        assert ".." not in slug.split("/"), f"{canonical_name!r} -> {slug!r}"
        # The resolved write target stays inside the export directory.
        target = (tmp_path / f"{slug}.md").resolve()
        assert target.is_relative_to(tmp_path.resolve()), (
            f"{canonical_name!r} -> slug={slug!r} escaped to {target}"
        )
        # ... and the defence-in-depth containment guard agrees.
        assert is_within_directory(tmp_path, tmp_path / f"{slug}.md")

    @pytest.mark.parametrize(
        ("canonical_name", "expected"),
        [
            ("u/spez", "reddit.com/u/spez"),
            ("r/MachineLearning", "reddit.com/r/MachineLearning"),
            ("github:login", "github.com/login"),
            ("github:torvalds", "github.com/torvalds"),
        ],
    )
    def test_legitimate_sharded_names_still_nest(
        self, tmp_path: Path, canonical_name: str, expected: str
    ) -> None:
        slug = subject_slug(canonical_name)
        assert slug == expected
        target = (tmp_path / f"{slug}.md").resolve()
        assert target.is_relative_to(tmp_path.resolve())
        # The note lands exactly one shard level deep, where the exporters expect it.
        assert target.parent == (tmp_path / Path(expected).parent).resolve()
        assert is_within_directory(tmp_path, tmp_path / f"{slug}.md")


def test_is_within_directory(tmp_path: Path) -> None:
    """The shared containment guard collapses ``..`` / symlinks before comparing."""
    # A normal nested write target is contained.
    assert is_within_directory(tmp_path, tmp_path / "reddit.com" / "u" / "spez.md")
    # The directory itself is trivially within itself.
    assert is_within_directory(tmp_path, tmp_path)
    # A ``..`` traversal target escapes and is rejected.
    assert not is_within_directory(tmp_path, tmp_path / ".." / "evil.md")
    # A sibling dir sharing a string prefix is NOT contained (no startswith bug).
    assert not is_within_directory(tmp_path / "vault", tmp_path / "vault-evil" / "x.md")


def test_prune_skips_symlink_escaping_md(tmp_path: Path) -> None:
    """F17b: a ``*.md`` whose resolved path lands outside the export root (e.g. a
    symlinked shard) must be skipped by the prune, never unlinked — while a
    genuine in-tree unmanaged note is still pruned."""
    output_dir = tmp_path / "vault"
    output_dir.mkdir()

    # A real, in-tree note this run did not write → should be pruned.
    unmanaged = output_dir / "stale.md"
    unmanaged.write_text("# stale\n")

    # A symlink inside the export dir pointing at a .md OUTSIDE the root.
    outside = tmp_path / "outside.md"
    outside.write_text("# precious external file\n")
    escaping = output_dir / "escaping.md"
    try:
        escaping.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("platform does not support symlinks")

    # Nothing was written this run.
    pruned = prune_obsolete_markdown(output_dir, written=set(), recursive=True)

    # The in-tree unmanaged note is gone.
    assert not unmanaged.exists()
    # The symlink-escaping entry was skipped — its target survives untouched.
    assert outside.exists()
    assert outside.read_text() == "# precious external file\n"
    # Only the genuine in-tree note counted toward the prune total.
    assert pruned == 1


def test_render_stance_callout() -> None:
    """the agreement callout groups holders, cites, and caveats."""
    from particles.core.schema import RelationType, StancePosition
    from particles.exporters.markdown import render_stance_callout

    positions = [
        StancePosition(
            kind=RelationType.ENDORSES,
            holder="github:torvalds",
            stance_particle_id="aaaa1111-0000-0000-0000-000000000001",
            effective_confidence=0.74,
        ),
        StancePosition(
            kind=RelationType.DISPUTES,
            holder="reddit:u_skeptic",
            stance_particle_id="bbbb2222-0000-0000-0000-000000000002",
            effective_confidence=0.62,
            magnitude=0.5,
        ),
    ]
    out = render_stance_callout(positions)
    assert "[!agreement]" in out
    assert "not factual confidence" in out
    assert "**Endorses:**" in out and "github:torvalds" in out
    assert "**Disputes:**" in out and "reddit:u_skeptic" in out
    assert "magnitude 0.50" in out
    assert "count of keys, not verified agents" in out  # M6 caveat


def test_render_stance_callout_empty() -> None:
    from particles.exporters.markdown import render_stance_callout

    assert render_stance_callout([]) == ""


def test_render_particles_includes_agreement() -> None:
    from particles.core.schema import (
        Confidence,
        Particle,
        RelationType,
        StancePosition,
        UncertaintyNature,
    )
    from particles.core.scoring.confidence import CalibrationSource
    from particles.exporters.markdown import render_particles

    p = Particle(
        content="A claim.",
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="t",
    )
    dist = [
        [
            StancePosition(
                kind=RelationType.ENDORSES,
                holder="x:alice",
                stance_particle_id="cccc-1",
                effective_confidence=0.9,
            )
        ]
    ]
    out = render_particles([p], agreement_distributions=dist)
    assert "[!agreement]" in out and "x:alice" in out


def _reading(spread: float) -> object:
    from particles.core.schema import ContestednessReading, PolicyRendering

    hi = 0.5 + spread / 2
    return ContestednessReading(
        spread=spread,
        renderings=[
            PolicyRendering(policy="local", effective_confidence=hi - spread),
            PolicyRendering(policy="acme-numismatics", effective_confidence=hi),
        ],
    )


def test_render_contested_callout() -> None:
    """the callout attributes per-policy renderings, max first."""
    from particles.exporters.markdown import render_contested_callout

    out = render_contested_callout(_reading(0.5))  # ≥ default threshold 0.2
    assert "[!contested]" in out
    assert "spread 0.50" in out
    assert "not factual confidence" in out
    # Sorted most-confident first so the extremes are nameable.
    assert out.index("acme-numismatics") < out.index("**local:**")
    assert "disclosure, never a discount" in out


def test_render_contested_callout_below_threshold_is_empty() -> None:
    from particles.exporters.markdown import render_contested_callout

    assert render_contested_callout(_reading(0.1)) == ""  # < default threshold 0.2


def test_render_contested_callout_none_is_empty() -> None:
    from particles.exporters.markdown import render_contested_callout

    assert render_contested_callout(None) == ""


def test_render_particles_includes_contestedness() -> None:
    from particles.core.schema import Confidence, Particle, UncertaintyNature
    from particles.core.scoring.confidence import CalibrationSource
    from particles.exporters.markdown import render_particles

    p = Particle(
        content="A claim.",
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="t",
    )
    out = render_particles([p], contestedness=[_reading(0.6)])  # type: ignore[list-item]
    assert "[!contested]" in out and "acme-numismatics" in out


# ---------------------------------------------------------------------------
# render_digest — the session-start memory digest formatter
# ---------------------------------------------------------------------------


def _digest_entry(
    content: str,
    eff: float,
    *,
    subjects: tuple[str, ...] = (),
    contested: str | None = None,
    day: int = 10,
):  # type: ignore[no-untyped-def]
    from datetime import UTC, datetime

    from particles.exporters.markdown import DigestEntry

    return DigestEntry(
        content=content,
        effective_confidence=eff,
        subjects=subjects,
        asserted_at=datetime(2026, 6, day, 12, 0, tzinfo=UTC),
        contested=contested,
    )


def test_render_digest_empty() -> None:
    from particles.exporters.markdown import render_digest

    out = render_digest("memory", [], 0)
    assert "Memory digest — memory" in out
    assert "No ACTIVE beliefs" in out


def test_render_digest_one_line_per_belief() -> None:
    from particles.exporters.markdown import render_digest

    out = render_digest("memory", [_digest_entry("Water is H2O.", 0.82, subjects=("Water",))], 1)
    # Confidence first (sortable), content, subjects, then the asserted date.
    assert "- **0.82** Water is H2O. _(Water)_ · 2026-06-10" in out
    assert "1 of 1 ACTIVE belief(s), ranked by effective confidence." in out


def test_render_digest_contested_marker() -> None:
    from particles.exporters.markdown import render_digest

    out = render_digest("memory", [_digest_entry("Contested claim.", 0.5, contested="inc-123")], 1)
    assert "contested by `inc-123`" in out


def test_render_digest_truncation_footer() -> None:
    from particles.exporters.markdown import render_digest

    # Two shown of five total → the footer discloses the cut (no silent cap).
    entries = [_digest_entry("top", 0.9), _digest_entry("next", 0.8)]
    out = render_digest("memory", entries, 5)
    assert "Showing the top 2 of 5 by effective confidence; 3 not shown." in out


def test_render_digest_is_pure_and_deterministic() -> None:
    from particles.exporters.markdown import render_digest

    entries = [_digest_entry("a", 0.9), _digest_entry("b", 0.4, subjects=("S",))]
    assert render_digest("m", entries, 2) == render_digest("m", entries, 2)


# ---------------------------------------------------------------------------
# projected-region sentinels, sources trailers, memory bullets
# ---------------------------------------------------------------------------


class TestProjectedRegions:
    BODY = "- a belief `p-3f9a2c1d`\n\n<!-- sources: p-3f9a2c1d -->"
    TEXT = (
        "<!-- BEGIN PROJECTED: memory-index (manifest: /state/memory.yaml) -->\n"
        f"{BODY}\n"
        "<!-- END PROJECTED: memory-index -->\n"
        "\n"
        "- an agent-authored note\n"
    )

    def test_find_projected_regions_parses_name_manifest_body(self) -> None:
        from particles.render.markdown import find_projected_regions

        regions = find_projected_regions(self.TEXT)
        assert len(regions) == 1
        region = regions[0]
        assert region.region == "memory-index"
        assert region.manifest == "/state/memory.yaml"
        assert region.body == self.BODY

    def test_regexes_shared_with_splice_round_trip(self) -> None:
        """The renderer's splice output parses back with the same regexes the
        harvest strip uses — strip and splice can never disagree."""
        from particles.operations.projection import splice_region
        from particles.render.markdown import find_projected_regions

        spliced = splice_region(
            self.TEXT, "memory-index", "- new body `p-71b0de00`", manifest="/state/memory.yaml"
        )
        regions = find_projected_regions(spliced)
        assert len(regions) == 1
        assert regions[0].body == "- new body `p-71b0de00`"

    def test_damaged_pair_is_not_a_region(self) -> None:
        from particles.render.markdown import find_projected_regions

        damaged = "<!-- BEGIN PROJECTED: memory-index (manifest: m.yaml) -->\nbody, no end\n"
        assert find_projected_regions(damaged) == []

    def test_strip_removes_pristine_region_entirely(self) -> None:
        from particles.render.markdown import strip_projected_regions_for_deposit

        out = strip_projected_regions_for_deposit(self.TEXT, {"memory-index": self.BODY})
        assert "BEGIN PROJECTED" not in out
        assert "p-3f9a2c1d" not in out
        assert "- an agent-authored note" in out

    def test_strip_keeps_dirtied_region_body_as_authored_input(self) -> None:
        from particles.render.markdown import strip_projected_regions_for_deposit

        edited = self.TEXT.replace("a belief", "a HAND-EDITED belief")
        out = strip_projected_regions_for_deposit(edited, {"memory-index": self.BODY})
        assert "BEGIN PROJECTED" not in out  # sentinels always dropped
        assert "a HAND-EDITED belief" in out  # ...but the edit is signal (§6)

    def test_strip_treats_unknown_snapshot_as_dirtied(self) -> None:
        from particles.render.markdown import strip_projected_regions_for_deposit

        out = strip_projected_regions_for_deposit(self.TEXT, {})
        assert "a belief" in out
        assert "BEGIN PROJECTED" not in out

    def test_strip_leaves_damaged_pair_untouched(self) -> None:
        from particles.render.markdown import strip_projected_regions_for_deposit

        damaged = "<!-- BEGIN PROJECTED: memory-index (manifest: m.yaml) -->\nbody, no end\n"
        assert strip_projected_regions_for_deposit(damaged, {"memory-index": "x"}) == damaged

    def test_insert_projected_region_at_top_preserves_content(self) -> None:
        from particles.render.markdown import (
            find_projected_regions,
            insert_projected_region_at_top,
        )

        existing = "# Memory\n- an old note\n"
        out = insert_projected_region_at_top(existing, "memory-index", "m.yaml")
        assert out.endswith("\n# Memory\n- an old note\n")
        regions = find_projected_regions(out)
        assert len(regions) == 1 and regions[0].body == ""
        assert out.startswith("<!-- BEGIN PROJECTED")


class TestSourcesTrailer:
    def test_format_sorts_and_dedupes(self) -> None:
        from particles.render.markdown import format_sources_trailer

        assert format_sources_trailer(["bb", "aa", "bb"]) == "<!-- sources: p-aa, p-bb -->\n"
        assert format_sources_trailer([]) == ""

    def test_parse_round_trips_format(self) -> None:
        from particles.render.markdown import format_sources_trailer, parse_sources_trailers

        assert parse_sources_trailers(format_sources_trailer(["aa", "bb"])) == {"aa", "bb"}

    def test_parse_unions_multiple_trailers(self) -> None:
        from particles.render.markdown import parse_sources_trailers

        text = "<!-- sources: p-aa -->\nprose\n<!-- sources: p-bb, p-cc -->\n"
        assert parse_sources_trailers(text) == {"aa", "bb", "cc"}

    def test_parse_failure_is_none_empty_trailer_is_empty_set(self) -> None:
        from particles.render.markdown import parse_sources_trailers

        assert parse_sources_trailers("no trailer here") is None
        assert parse_sources_trailers("<!-- sources: -->") == set()


class TestMemoryBullet:
    def test_plain_and_contested_shapes(self) -> None:
        from particles.render.markdown import format_memory_bullet

        assert (
            format_memory_bullet("DCO is enforced.", "71b0de00")
            == "- DCO is enforced. `p-71b0de00`"
        )
        assert (
            format_memory_bullet("CI floors at 3.11.", "08d3e100", contested_by="9c447100")
            == "- ⚠ contested — CI floors at 3.11. (vs. p-9c447100) `p-08d3e100`"
        )

    def test_newlines_flattened_deterministically(self) -> None:
        from particles.render.markdown import format_memory_bullet

        out = format_memory_bullet("a\nmulti  line\tclaim", "aa")
        assert out == "- a multi line claim `p-aa`"
        assert out == format_memory_bullet("a\nmulti  line\tclaim", "aa")

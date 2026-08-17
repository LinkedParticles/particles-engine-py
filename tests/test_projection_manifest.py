"""Tests for particles/operations/projection/manifest.py — manifest parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from particles.operations.projection.manifest import (
    DerivedSection,
    DocManifest,
    MechanicalBlock,
    Select,
    load_manifest,
)
from tests._upstream import upstream_only

_VALID_YAML = """\
name: readme
output: README.md
sections:
  - block: blocks/header.md
  - title: What is Particles
    tags: [overview, vision]
    subjects: ["Particles standard"]
    query: what is the Particles SDK
    top_k: 8
  - title: Architecture
    tags: [architecture]
"""


def test_parses_ordered_sections() -> None:
    manifest = DocManifest.model_validate(
        {
            "name": "readme",
            "sections": [
                {"block": "blocks/header.md"},
                {"title": "Overview", "tags": ["overview"]},
            ],
        }
    )
    assert manifest.name == "readme"
    assert isinstance(manifest.sections[0], MechanicalBlock)
    assert isinstance(manifest.sections[1], DerivedSection)
    # Order is preserved exactly as authored — assembly order is manifest order.
    assert manifest.sections[0].block == "blocks/header.md"
    assert manifest.sections[1].title == "Overview"


def test_mechanical_vs_derived_disambiguation() -> None:
    """A section is mechanical iff it carries a ``block`` key, else derived."""
    manifest = DocManifest.model_validate(
        {
            "name": "doc",
            "sections": [
                {"block": "x.md"},
                {"title": "T", "query": "q"},
            ],
        }
    )
    kinds = [type(s).__name__ for s in manifest.sections]
    assert kinds == ["MechanicalBlock", "DerivedSection"]


def test_derived_section_requires_a_binding() -> None:
    """A derived section with no tags/subjects/query would select nothing → reject."""
    with pytest.raises(ValidationError) as exc:
        DocManifest.model_validate({"name": "doc", "sections": [{"title": "Bare"}]})
    assert "at least one of" in str(exc.value)


def test_topic_query_falls_back_to_title() -> None:
    explicit = DerivedSection(title="Architecture", query="how is it structured")
    implicit = DerivedSection(title="Architecture", tags=["architecture"])
    assert explicit.topic_query == "how is it structured"
    assert implicit.topic_query == "Architecture"


def test_name_pattern_rejects_uppercase_and_spaces() -> None:
    with pytest.raises(ValidationError):
        DocManifest.model_validate({"name": "My Doc", "sections": [{"title": "T", "query": "q"}]})


def test_empty_sections_rejected() -> None:
    with pytest.raises(ValidationError):
        DocManifest.model_validate({"name": "doc", "sections": []})


def test_top_k_bounds() -> None:
    with pytest.raises(ValidationError):
        DerivedSection(title="T", query="q", top_k=0)
    with pytest.raises(ValidationError):
        DerivedSection(title="T", query="q", top_k=999)


def test_code_symbol_rank_weight_defaults_to_one() -> None:
    """the per-section code-symbol demotion defaults to inert (1.0),
    and an explicit weight round-trips."""
    assert DerivedSection(title="Overview", tags=["overview"]).code_symbol_rank_weight == 1.0
    section = DerivedSection(title="Overview", tags=["overview"], code_symbol_rank_weight=0.3)
    assert section.code_symbol_rank_weight == 0.3


def test_code_symbol_rank_weight_rejects_out_of_range() -> None:
    """code_symbol_rank_weight is bounded to [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        DerivedSection(title="T", query="q", code_symbol_rank_weight=1.5)
    with pytest.raises(ValidationError):
        DerivedSection(title="T", query="q", code_symbol_rank_weight=-0.1)


def test_load_manifest_from_file(tmp_path: Path) -> None:
    path = tmp_path / "readme.yaml"
    path.write_text(_VALID_YAML, encoding="utf-8")
    manifest = load_manifest(path)
    assert manifest.name == "readme"
    assert manifest.output == "README.md"
    assert len(manifest.sections) == 3
    assert isinstance(manifest.sections[0], MechanicalBlock)
    derived = manifest.sections[1]
    assert isinstance(derived, DerivedSection)
    assert derived.subjects == ["Particles standard"]
    assert derived.top_k == 8


def test_load_manifest_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_manifest(path)


# ---------------------------------------------------------------------------
# directed-per-section synthesis: direction / flowing / framing
# ---------------------------------------------------------------------------


def test_direction_defaults_off_and_flowing_defaults_on() -> None:
    """A section omitting the fields is undirected:
    (``direction`` unset) but renders as flowing prose — the opt-in ``flowing``
    default was flipped to true for projection sections."""
    section = DerivedSection(title="Overview", tags=["overview"])
    assert section.direction is None
    assert section.flowing is True


def test_flowing_can_be_opted_out_per_section() -> None:
    """the headed multi-section form stays available — a section
    opts *out* of the new default with an explicit ``flowing: false``."""
    section = DerivedSection(title="Reference", tags=["api"], flowing=False)
    assert section.flowing is False


def test_directed_flowing_section_round_trips() -> None:
    """a section with ``direction`` / ``flowing`` set round-trips."""
    section = DerivedSection(
        title="What is Particles?",
        query="the core loop",
        direction="Write a single tight elevator-pitch paragraph; do not enumerate.",
        flowing=True,
    )
    assert section.direction == "Write a single tight elevator-pitch paragraph; do not enumerate."
    assert section.flowing is True
    # The directed fields do not disturb the existing binding semantics.
    assert section.topic_query == "the core loop"


def test_directed_section_still_requires_a_binding() -> None:
    """``direction`` selects nothing — a directed section with no
    tags/subjects/query still fails ``_require_a_binding`` (the validator is
    unchanged)."""
    with pytest.raises(ValidationError) as exc:
        DerivedSection(title="Directed but unbound", direction="Write the pitch.")
    assert "at least one of" in str(exc.value)


def test_directed_section_with_binding_validates() -> None:
    """A directed section that *does* carry a binding validates — ``direction``
    is additive, the binding is what selects particles."""
    section = DerivedSection(
        title="Pitch", direction="Write the elevator pitch.", query="what is it"
    )
    assert section.direction == "Write the elevator pitch."


def test_manifest_accepts_framing_and_defaults_to_none() -> None:
    """``DocManifest`` accepts a document-level ``framing`` spine,
    and defaults it to None when absent."""
    bare = DocManifest.model_validate(
        {"name": "readme", "sections": [{"title": "T", "query": "q"}]}
    )
    assert bare.framing is None

    framed = DocManifest.model_validate(
        {
            "name": "readme",
            "framing": (
                "Particles is a git-like ledger for beliefs; one argument, not four summaries."
            ),
            "sections": [{"title": "T", "query": "q"}],
        }
    )
    assert framed.framing is not None
    assert "git-like ledger" in framed.framing


def test_shipped_shape_manifest_validates_without_new_fields() -> None:
    """backward compat: a manifest carrying none of the new fields
    (direction / flowing / framing) still validates and renders as today."""
    manifest = DocManifest.model_validate(
        {
            "name": "readme",
            "output": "README.md",
            "sections": [
                {"block": "blocks/header.md"},
                {"title": "Overview", "tags": ["overview"], "query": "what is it"},
            ],
        }
    )
    assert manifest.framing is None
    derived = manifest.sections[1]
    assert isinstance(derived, DerivedSection)
    assert derived.direction is None
    # `flowing` is the one field whose default flipped —
    # such a manifest still validates and selects identically, but now renders
    # its prose in the narrative form.
    assert derived.flowing is True


# ---------------------------------------------------------------------------
# the gated self-hosted README manifest
# ---------------------------------------------------------------------------

_README_MANIFEST = Path(__file__).parents[1] / "docs" / "projection" / "readme.yaml"


# ---------------------------------------------------------------------------
# claim-level select.allow / select.deny
# ---------------------------------------------------------------------------


def test_select_defaults_to_empty_lists() -> None:
    """a section omitting ``select`` carries empty allow/deny lists,
    so existing manifests are byte-for-byte unchanged."""
    section = DerivedSection(title="Overview", tags=["overview"])
    assert section.select.allow == []
    assert section.select.deny == []
    # The bare Select default is also both-empty.
    assert Select().allow == [] and Select().deny == []


def test_select_accepts_short_and_full_id_forms() -> None:
    """ids may be the ``p-<shortid>`` display form or the full id."""
    select = Select(
        allow=["p-3f9a2c1d", "3f9a2c1d-0000-0000-0000-000000000000"],
        deny=["p-9c447100"],
    )
    assert select.allow == ["p-3f9a2c1d", "3f9a2c1d-0000-0000-0000-000000000000"]
    assert select.deny == ["p-9c447100"]
    # And it round-trips through a section's select field.
    section = DerivedSection(title="Core", query="what is a particle", select=select)
    assert section.select.allow[0] == "p-3f9a2c1d"


def test_select_rejects_id_in_both_allow_and_deny() -> None:
    """an id in both lists is a load-time error — intent is undefined."""
    with pytest.raises(ValidationError) as exc:
        Select(allow=["p-3f9a2c1d"], deny=["p-3f9a2c1d"])
    assert "disjoint" in str(exc.value) or "both" in str(exc.value)


def test_select_rejects_malformed_id() -> None:
    """a manifestly ill-formed id (empty, whitespace, bare prefix)
    is rejected at load."""
    with pytest.raises(ValidationError):
        Select(allow=[""])
    with pytest.raises(ValidationError):
        Select(deny=["p-"])
    with pytest.raises(ValidationError):
        Select(allow=["has spaces"])


def test_select_overlap_is_rejected_via_full_manifest_load() -> None:
    """The overlap rule fires when a manifest carries the conflicting section."""
    with pytest.raises(ValidationError):
        DocManifest.model_validate(
            {
                "name": "readme",
                "sections": [
                    {
                        "title": "Core",
                        "query": "q",
                        "select": {"allow": ["p-deadbeef"], "deny": ["p-deadbeef"]},
                    }
                ],
            }
        )


def test_select_does_not_relax_require_a_binding() -> None:
    """``select`` is a post-filter, not a binding — a section with only
    a ``select`` and no tags/subjects/query still fails ``_require_a_binding``."""
    with pytest.raises(ValidationError) as exc:
        DerivedSection(title="Bare but pinned", select=Select(allow=["p-3f9a2c1d"]))
    assert "at least one of" in str(exc.value)


def test_select_loads_from_yaml() -> None:
    """the ``select`` block parses from a manifest mapping."""
    manifest = DocManifest.model_validate(
        {
            "name": "readme",
            "sections": [
                {
                    "title": "Core concepts",
                    "query": "what is a particle",
                    "top_k": 16,
                    "select": {"allow": ["p-3f9a2c1d", "p-71b0de00"], "deny": ["p-9c447100"]},
                }
            ],
        }
    )
    section = manifest.sections[0]
    assert isinstance(section, DerivedSection)
    assert section.select.allow == ["p-3f9a2c1d", "p-71b0de00"]
    assert section.select.deny == ["p-9c447100"]


@upstream_only  # asserts the shape of this repository's own manifest
def test_readme_manifest_is_the_three_region_self_hosted_shape() -> None:
    """§4: the shipped readme.yaml parses as the three-region
    self-hosted README manifest — every section region-bound, pin-driven
    (cage suppressed + select.allow), flowing, with a direction brief; the
    Architecture section keeps code_symbol_rank_weight 1.0 (down-weight
    OFF for the reference section) while the conceptual sections demote."""
    manifest = load_manifest(_README_MANIFEST)

    assert manifest.name == "readme"
    assert manifest.output == "README.md"
    # Document-level whitepaper value-prop spine.
    assert manifest.framing is not None and manifest.framing.strip()

    regions = manifest.region_sections()
    assert list(regions) == ["what-is", "design-rationale", "architecture"]
    for section in regions.values():
        assert isinstance(section, DerivedSection)
        # Pin-driven: cage suppressed, selection = pins.
        assert section.min_confidence == 0.99
        assert section.select.allow
        # one continuous narrative block + an authoring brief.
        assert section.flowing is True
        assert section.direction is not None and section.direction.strip()
        assert section.tags or section.subjects or section.query
    # reference section keeps docstrings; conceptual sections demote.
    assert regions["architecture"].code_symbol_rank_weight == 1.0
    assert regions["what-is"].code_symbol_rank_weight < 1.0
    assert regions["design-rationale"].code_symbol_rank_weight < 1.0


# ---------------------------------------------------------------------------
# render mode + document budget fields
# ---------------------------------------------------------------------------


def test_render_defaults_to_prose_and_budget_to_none() -> None:
    """Additive, default-preserving: existing manifests unchanged."""
    manifest = DocManifest.model_validate(
        {"name": "t", "sections": [{"title": "S", "tags": ["x"]}]}
    )
    section = manifest.sections[0]
    assert isinstance(section, DerivedSection)
    assert section.render == "prose"
    assert manifest.max_lines is None
    assert manifest.max_bytes is None


def test_bullets_section_may_bind_nothing() -> None:
    """a `render: bullets` section with no cage is 'the top of the
    store' — the zero-config memory-index default."""
    section = DerivedSection.model_validate(
        {"title": "Memory index", "query": None, "render": "bullets", "top_k": 60}
    )
    assert section.render == "bullets"
    assert not (section.tags or section.subjects or section.query)


def test_prose_section_still_requires_a_binding() -> None:
    with pytest.raises(ValidationError, match="at least one of"):
        DerivedSection.model_validate({"title": "S", "render": "prose"})


def test_render_rejects_unknown_mode() -> None:
    with pytest.raises(ValidationError):
        DerivedSection.model_validate({"title": "S", "tags": ["x"], "render": "haiku"})


def test_budget_fields_reject_non_positive_values() -> None:
    with pytest.raises(ValidationError):
        DocManifest.model_validate(
            {"name": "t", "max_lines": 0, "sections": [{"title": "S", "tags": ["x"]}]}
        )
    with pytest.raises(ValidationError):
        DocManifest.model_validate(
            {"name": "t", "max_bytes": 0, "sections": [{"title": "S", "tags": ["x"]}]}
        )


def test_default_memory_manifest_text_is_a_valid_bullets_manifest(tmp_path: Path) -> None:
    """The init-written memory.yaml parses as the ADR's default:
    one unbound bullets section, top_k 60, floor 0.30, budget 120 lines."""
    from particles.api.cli._claude_code import default_memory_manifest_text

    path = tmp_path / "memory.yaml"
    path.write_text(default_memory_manifest_text(), encoding="utf-8")
    manifest = load_manifest(path)

    assert manifest.name == "memory-index"
    assert manifest.max_lines == 120
    assert manifest.max_bytes == 16384
    assert len(manifest.sections) == 1
    section = manifest.sections[0]
    assert isinstance(section, DerivedSection)
    assert section.render == "bullets"
    assert section.top_k == 60
    assert section.min_confidence == 0.30
    assert section.query is None and not section.tags and not section.subjects


# ---------------------------------------------------------------------------
# Sentinel-region binding
# ---------------------------------------------------------------------------


def test_region_defaults_to_none() -> None:
    section = DerivedSection(title="Overview", query="q")
    assert section.region is None


def test_region_accepts_kebab_name_and_round_trips() -> None:
    manifest = DocManifest(
        name="readme",
        sections=[
            DerivedSection(title="What is this", query="q", region="what-is"),
            DerivedSection(title="Architecture", query="q", region="architecture"),
        ],
    )
    assert list(manifest.region_sections()) == ["what-is", "architecture"]
    assert manifest.region_sections()["what-is"].title == "What is this"


def test_region_rejects_bad_shape() -> None:
    with pytest.raises(ValidationError):
        DerivedSection(title="Overview", query="q", region="What Is")


def test_duplicate_region_rejected() -> None:
    """Two sections splicing into one sentinel pair is an authoring error (1:1)."""
    with pytest.raises(ValidationError, match="more than one"):
        DocManifest(
            name="readme",
            sections=[
                DerivedSection(title="A", query="q", region="what-is"),
                DerivedSection(title="B", query="q", region="what-is"),
            ],
        )


def test_region_sections_skips_unbound_and_blocks() -> None:
    manifest = DocManifest(
        name="readme",
        sections=[
            MechanicalBlock(block="blocks/header.md"),
            DerivedSection(title="Unbound", query="q"),
            DerivedSection(title="Bound", query="q", region="bound"),
        ],
    )
    assert list(manifest.region_sections()) == ["bound"]


# ---------------------------------------------------------------------------
# the flowing default is scoped to projection sections
# ---------------------------------------------------------------------------

#: Per-subject / per-note synthesis consumers. None of these may
#: pass ``flowing=``: they must ride ``render_article``'s own ``False`` default
#: so encyclopedic articles keep their sub-headings regardless of what the
#: *manifest* default is.
_PER_SUBJECT_SYNTHESIS_MODULES = (
    "particles/exporters/wiki.py",
    "particles/exporters/obsidian/synthesis.py",
    "particles/exporters/obsidian/narrative.py",
    "particles/exporters/logseq/synthesis.py",
    "particles/operations/narrative_synthesis.py",
)


def test_render_article_flowing_default_stays_off() -> None:
    """the *renderer's* default is the per-subject genre default
    and must NOT follow the manifest flip.

    Two independent defaults carry the scoping boundary:
    ``DerivedSection.flowing`` (manifest sections — flipped to True) and
    ``render_article(flowing=...)`` (every other consumer — stays False).
    Hoisting the manifest default into the renderer would silently restyle
    every wiki / Obsidian / Logseq article, so pin it.
    """
    import inspect

    from particles.render.article_synthesis import render_article

    default = inspect.signature(render_article).parameters["flowing"].default
    assert default is False, (
        "render_article's own `flowing` default must stay False — it is what "
        "keeps per-subject articles in the headed encyclopedic form"
    )
    # The manifest default is the one that flipped.
    assert DerivedSection(title="Overview", query="q").flowing is True


@pytest.mark.parametrize("relpath", _PER_SUBJECT_SYNTHESIS_MODULES)
def test_per_subject_exporters_never_pass_flowing(relpath: str) -> None:
    """no per-subject synthesis consumer may thread ``flowing=``.

    The scoping is correct *by construction* — ``section.flowing`` reaches
    ``render_article`` at exactly one call site
    (``operations/projection/project.py``). This test is the tripwire on that
    construction: adding a ``flowing=`` pass-through to an exporter would put
    per-subject articles under the projection genre without any test failing
    on behaviour.
    """
    import ast

    source_root = Path(__file__).resolve().parent.parent
    tree = ast.parse((source_root / relpath).read_text(encoding="utf-8"))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "flowing"
    ]
    assert not offenders, (
        f"{relpath} passes `flowing=` at line(s) {offenders}; per-subject "
        "synthesis must ride render_article's own False default"
    )

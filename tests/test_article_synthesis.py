"""Tests for particles/render/article_synthesis/render.py — prompt variants.

Covers the directed-per-section synthesis additions: the
flowing-prose first-attempt prompt (``flowing``), the per-section authoring
brief (``direction``), and the document-level spine (``framing``). The LLM
synthesis call itself is out of scope (tests/AGENTS.md § Out of scope); these
tests pin the deterministic prompt-construction surface.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Subject,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.render.article_synthesis import _build_synthesis_prompt
from particles.render.article_synthesis.render import _format_particles_for_prompt


def _make_particle(content: str, confidence_value: float = 0.9) -> Particle:
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
        asserted_by="stub-extractor",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
        extractor_ref={"name": "stub-extractor", "version": "0.1.0"},
        subject_ids=[],
        tags=None,
    )


# The pre-ADR-0163 standard first-attempt prompt, frozen as a literal. The
# ``flowing=False, direction=None, framing=None`` path MUST reproduce this
# byte-for-byte — the new fields are additive and the no-field render is
# unchanged. (Title/subject = "X", and two particles "a" / "b" with the
# placeholder block filled below.)
_EXPECTED_STANDARD_NO_FIELDS = """\
You are writing one encyclopedic article about a single subject, based ONLY
on the structured particle list below. Every claim in your article must be
cited back to a particle with a Markdown footnote in the form `[^p-xxxxxxxx]`
where `xxxxxxxx` is the eight-character particle ID prefix shown on each
particle below. Do not invent claims; do not paraphrase beyond the source
particles' content; do not use any external knowledge.

Modality handling — render each particle in the voice of its modality:
- FALSIFIABLE: an observer-independent fact. State it as fact (per the
  confidence handling below).
- EXPERIENTIAL: the author's inner state / lived experience. Render it as
  experience ("the author felt …", "found it …"), never as a hedged fact.
- EVALUATIVE: the author's opinion or value judgement. Render it as opinion
  ("in the author's view …", "found X tedious"), not as objective fact.
- CONSTITUTIVE: a definition or rule the source establishes. State it as such.

Confidence handling (FALSIFIABLE / CONSTITUTIVE only — for EXPERIENTIAL and
EVALUATIVE, effective_confidence reflects how clearly the feeling/opinion is
expressed, not how likely it is true, so NEVER apply a truth-hedge to them):
- Claims with effective_confidence below 0.5 MUST be hedged
  ("according to one source", "one report claims", "is reportedly").
- Claims at or above 0.9 may be stated declaratively.
- Claims between 0.5 and 0.9 should be stated declaratively with a
  citation, but without an emphatic adverb.

Cross-references:
- When a claim mentions another subject's name, render it as an
  Obsidian-style `[[Subject Name]]` wikilink so the resulting vault is
  graph-navigable.

Style:
- Encyclopedic tone (third person, neutral, factual).
- One-paragraph introduction, then short topical sections with `## `
  headings. No more than 600 words.
- Do NOT write a References section — that is appended deterministically
  by the exporter after your output.
- Emit footnote *references* only — i.e. `[^p-xxxxxxxx]` inline. Do NOT
  emit any footnote *definition* lines of the form `[^p-xxxxxxxx]: …`.
  The exporter appends a single canonical definition per cited particle
  in the References section. Emitting your own definition produces a
  duplicate, and markdown footnote parsers silently drop the link.

Output: the article body in Markdown, starting with the `# {title}` H1
heading. No frontmatter; no preamble before the H1; no closing remarks
after the last paragraph.

----
SUBJECT: {subject}
PARTICLE LIST (id_prefix, modality, effective_confidence, content):
{particles}
----"""


def _subject_and_particles() -> tuple[Subject, list[Particle], dict[str, float]]:
    subject = Subject(id="s1", canonical_name="X", asserted_by="test")
    ps = [_make_particle("a"), _make_particle("b")]
    eff = {p.id: p.confidence.value for p in ps}
    return subject, ps, eff


def test_standard_prompt_byte_identical_without_new_fields() -> None:
    """with ``flowing=False, direction=None, framing=None`` the
    standard first-attempt prompt is byte-identical to the pre-0163 prompt."""
    subject, ps, eff = _subject_and_particles()
    prompt = _build_synthesis_prompt(subject=subject, particles=ps, eff=eff, strict=False)
    expected = _EXPECTED_STANDARD_NO_FIELDS.format(
        title="X",
        subject="X",
        particles=_format_particles_for_prompt(ps, eff),
    )
    assert prompt == expected


def test_flowing_prompt_suppresses_subheadings() -> None:
    """``flowing=True`` first-attempt prompt swaps the standard
    sub-headings instruction for the no-sub-headings, single-passage one."""
    subject, ps, eff = _subject_and_particles()
    flowing = _build_synthesis_prompt(
        subject=subject, particles=ps, eff=eff, strict=False, flowing=True
    )
    assert "One continuous passage of connected prose under the section title" in flowing
    assert "sub-headings, no bullet lists" in flowing
    # The standard sub-headings instruction is gone.
    assert "short topical sections with `## `" not in flowing
    # It is otherwise the standard prompt — encyclopedic, fully cited.
    assert "You are writing one encyclopedic article" in flowing

    standard = _build_synthesis_prompt(subject=subject, particles=ps, eff=eff, strict=False)
    assert "short topical sections with `## `" in standard
    assert flowing != standard


def test_flowing_is_first_attempt_only_strict_unchanged() -> None:
    """heading suppression is a first-attempt concern. With
    ``strict=True`` the (unchanged) strict prompt is returned regardless of
    ``flowing``."""
    subject, ps, eff = _subject_and_particles()
    strict_plain = _build_synthesis_prompt(subject=subject, particles=ps, eff=eff, strict=True)
    strict_flowing = _build_synthesis_prompt(
        subject=subject, particles=ps, eff=eff, strict=True, flowing=True
    )
    assert strict_flowing == strict_plain
    # The strict retry prompt is the citation-focused one, not the flowing body.
    assert "PRIOR ATTEMPT WAS REJECTED" in strict_flowing
    assert "One continuous passage of connected prose" not in strict_flowing


def test_direction_appears_in_first_attempt_prompt() -> None:
    """a section ``direction`` is interpolated into the first-attempt
    prompt (standard and flowing) when provided."""
    subject, ps, eff = _subject_and_particles()
    brief = "Write a single tight elevator-pitch paragraph; lead with the ledger framing."

    standard = _build_synthesis_prompt(
        subject=subject, particles=ps, eff=eff, strict=False, direction=brief
    )
    assert brief in standard

    flowing = _build_synthesis_prompt(
        subject=subject, particles=ps, eff=eff, strict=False, flowing=True, direction=brief
    )
    assert brief in flowing


def test_framing_appears_in_first_attempt_prompt() -> None:
    """the document-level ``framing`` spine is prepended to the
    first-attempt prompt (standard and flowing) when provided."""
    subject, ps, eff = _subject_and_particles()
    spine = "The whole document turns on three mechanics; it is one argument, not four summaries."

    standard = _build_synthesis_prompt(
        subject=subject, particles=ps, eff=eff, strict=False, framing=spine
    )
    assert spine in standard
    # Framing is a prefix block — it precedes the task instruction.
    assert standard.index(spine) < standard.index("You are writing one encyclopedic article")

    flowing = _build_synthesis_prompt(
        subject=subject, particles=ps, eff=eff, strict=False, flowing=True, framing=spine
    )
    assert spine in flowing


def test_direction_and_framing_absent_from_strict_retry() -> None:
    """direction/framing steer the first attempt only; the strict
    retry prompt is the unchanged citation-focused one and never carries them."""
    subject, ps, eff = _subject_and_particles()
    strict = _build_synthesis_prompt(
        subject=subject,
        particles=ps,
        eff=eff,
        strict=True,
        direction="Write the pitch.",
        framing="The spine.",
    )
    assert "Write the pitch." not in strict
    assert "The spine." not in strict
    plain_strict = _build_synthesis_prompt(subject=subject, particles=ps, eff=eff, strict=True)
    assert strict == plain_strict

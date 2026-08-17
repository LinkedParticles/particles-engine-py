"""Tests for particles/ingest/candidate_dedup.py — the intra-pass fold.

The extractor mints one candidate per verbatim occurrence of a sentence on a
repetitive multi-section source (Wikipedia timeline/response sections). This fold
collapses exact-content duplicates that share a subject binding down to one,
*before* the pipeline embeds / reconciles them. It is a pure list transform, so
these tests drive it directly with hand-built candidates — no LLM, no store.
"""

from __future__ import annotations

from particles.core.schema import ParticleType, RelationType, UncertaintyNature
from particles.extraction.general import CandidateParticle
from particles.ingest.candidate_dedup import dedupe_exact_candidates


def _claim(
    content: str,
    subjects: list[str] | None = None,
    *,
    stance_kind: RelationType | None = None,
    stance_target_index: int | None = None,
    narrative_index: int | None = None,
    particle_type: ParticleType = ParticleType.CLAIM,
) -> CandidateParticle:
    return CandidateParticle(
        content=content,
        confidence_value=0.9,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        subjects=subjects if subjects is not None else ["Ebola"],
        stance_kind=stance_kind,
        stance_target_index=stance_target_index,
        narrative_index=narrative_index,
        particle_type=particle_type,
    )


# The audit-dogfood exemplar: one sentence repeated across sections.
_REPEAT = (
    "On 2 June, the International Rescue Committee warned that the outbreak "
    "was 'likely far worse' than reported."
)


def test_no_duplicates_is_noop() -> None:
    """A non-repetitive pass is returned unchanged (same object, no notes)."""
    candidates = [_claim("a"), _claim("b"), _claim("c")]
    result, notes = dedupe_exact_candidates(candidates)
    assert result is candidates
    assert notes == []


def test_empty_is_noop() -> None:
    result, notes = dedupe_exact_candidates([])
    assert result == []
    assert notes == []


def test_collapses_exact_duplicate_across_sections() -> None:
    """The same sentence emitted 3× (timeline + response + summary) → one particle."""
    candidates = [
        _claim("Guinea reported its first case in April."),
        _claim(_REPEAT),
        _claim("The WHO convened an emergency committee."),
        _claim(_REPEAT),  # response section restates it
        _claim(_REPEAT),  # summary restates it again
    ]
    result, notes = dedupe_exact_candidates(candidates)
    contents = [c.content for c in result]
    assert contents.count(_REPEAT) == 1
    assert len(result) == 3
    # Order preserved; the first occurrence is the survivor.
    assert contents == [
        "Guinea reported its first case in April.",
        _REPEAT,
        "The WHO convened an emergency committee.",
    ]
    assert notes == [
        "INTRA_PASS_DEDUP: collapsed 2 exact-content duplicate "
        "candidate(s) into their first occurrence"
    ]


def test_distinct_but_similar_is_not_deduped() -> None:
    """Near-duplicates / paraphrases are left for §6.6 / co-evidence."""
    a = "The outbreak was likely far worse than reported."
    b = "The outbreak may have been worse than reported."  # paraphrase, not exact
    candidates = [_claim(a), _claim(b)]
    result, notes = dedupe_exact_candidates(candidates)
    assert len(result) == 2
    assert notes == []


def test_normalization_collapses_whitespace_and_trailing_punct() -> None:
    """Conservative normalization: whitespace runs + a trailing mark only."""
    candidates = [
        _claim("The IRC warned the outbreak was far worse."),
        _claim("The  IRC   warned the outbreak was far worse"),  # spaces + no period
        _claim("The IRC warned the outbreak was far worse.\n"),  # trailing newline
    ]
    result, _ = dedupe_exact_candidates(candidates)
    assert len(result) == 1


def test_case_difference_is_not_deduped() -> None:
    """Case is preserved — this is exact-content, not near-duplicate, dedup."""
    candidates = [
        _claim("The outbreak was far worse."),
        _claim("the outbreak was far worse."),
    ]
    result, _ = dedupe_exact_candidates(candidates)
    assert len(result) == 2


def test_same_content_different_subjects_is_not_deduped() -> None:
    """Identical content bound to different subjects is two distinct claims."""
    candidates = [
        _claim("Cases are rising.", subjects=["Guinea"]),
        _claim("Cases are rising.", subjects=["Liberia"]),
    ]
    result, notes = dedupe_exact_candidates(candidates)
    assert len(result) == 2
    assert notes == []


def test_subject_set_order_insensitive() -> None:
    """Subject binding compares as a set — order does not defeat the fold."""
    candidates = [
        _claim("Shared claim.", subjects=["Guinea", "Liberia"]),
        _claim("Shared claim.", subjects=["Liberia", "Guinea"]),
    ]
    result, _ = dedupe_exact_candidates(candidates)
    assert len(result) == 1


def test_stance_candidate_is_never_dropped() -> None:
    """A stance carries a positional target; it is never folded even if its
    content happens to repeat a plain claim."""
    candidates = [
        _claim("The outbreak is severe."),
        _claim(
            "The outbreak is severe.",
            stance_kind=RelationType.ENDORSES,
            stance_target_index=0,
        ),
    ]
    result, notes = dedupe_exact_candidates(candidates)
    assert len(result) == 2
    assert notes == []


def test_narrative_constituent_is_never_dropped() -> None:
    """A NARRATIVE constituent (narrative_index set) is structural — never folded."""
    candidates = [
        _claim("Woke up early.", narrative_index=0),
        _claim("Woke up early.", narrative_index=1),  # same text, different position
    ]
    result, _ = dedupe_exact_candidates(candidates)
    assert len(result) == 2


def test_stance_target_repointed_to_surviving_twin() -> None:
    """When a duplicate that a stance targeted is dropped, the stance follows the
    surviving twin so no edge endpoint is lost."""
    candidates = [
        _claim("Filler claim one."),  # index 0
        _claim(_REPEAT),  # index 1 — kept (first occurrence)
        _claim(_REPEAT),  # index 2 — dropped duplicate
        _claim(  # index 3 — stance targeting the dropped duplicate at index 2
            "I strongly agree with that assessment.",
            stance_kind=RelationType.ENDORSES,
            stance_target_index=2,
        ),
    ]
    result, notes = dedupe_exact_candidates(candidates)
    assert len(result) == 3
    # New positions: 0=Filler, 1=_REPEAT (kept), 2=stance.
    stance = result[2]
    assert stance.stance_kind == RelationType.ENDORSES
    # Repointed from dropped index 2 → surviving twin's new position 1.
    assert stance.stance_target_index == 1
    assert notes[0].startswith("INTRA_PASS_DEDUP: collapsed 1")


def test_stance_target_rebased_when_earlier_duplicate_dropped() -> None:
    """A stance targeting a candidate *after* a dropped duplicate has its index
    rebased to the compacted positions."""
    candidates = [
        _claim(_REPEAT),  # 0 kept
        _claim(_REPEAT),  # 1 dropped
        _claim("The distinct target claim."),  # 2 -> new position 1
        _claim(  # 3 -> new position 2; targets index 2
            "Endorsing the distinct claim.",
            stance_kind=RelationType.ENDORSES,
            stance_target_index=2,
        ),
    ]
    result, _ = dedupe_exact_candidates(candidates)
    assert len(result) == 3
    assert result[1].content == "The distinct target claim."
    assert result[2].stance_target_index == 1

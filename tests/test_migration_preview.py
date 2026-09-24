# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the format-independent migration preview (the ``--dry-run`` report).

``build_preview`` is a pure function of an extractor's own output, so these pin
the arithmetic: what counts as a Subject, which declared entities are lost, and
that the sample spans the export rather than sitting on its first record.
"""

from __future__ import annotations

import json

from particles.core.schema import UncertaintyNature
from particles.core.scoring.confidence import CalibrationSource
from particles.extraction.general import CandidateParticle, ExtractionResult
from particles.extraction.migration_preview import build_preview


def _candidate(content: str, *subjects: str, location: str | None = None) -> CandidateParticle:
    return CandidateParticle(
        content=content,
        confidence_value=0.35,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        subjects=list(subjects),
        calibration_source=CalibrationSource.IMPORTED,
        provenance_location=location,
    )


def _preview(result: ExtractionResult, **kwargs: object):
    return build_preview(
        source_type="X_EXPORT",
        extractor_id="x-extractor",
        extractor_version="0.0.1",
        result=result,
        records=kwargs.pop("records", {}),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def test_counts_particles_and_distinct_subjects() -> None:
    result = ExtractionResult(
        candidates=[
            _candidate("a", "Alice"),
            _candidate("b", "Alice"),
            _candidate("Alice knows Bob", "Alice", "Bob"),
        ]
    )
    preview = _preview(result, records={"rows": 3})
    assert preview.particles == 3
    assert preview.subjects == 2  # Alice once, however many claims name her
    assert preview.single_subject_particles == 2
    assert preview.multi_subject_particles == 1
    assert preview.records == {"rows": 3}
    assert preview.confidence_values == [0.35]
    assert preview.calibration_sources == ["IMPORTED"]


def test_a_declared_entity_no_candidate_names_is_lost() -> None:
    result = ExtractionResult(candidates=[_candidate("Alice knows Bob", "Alice", "Bob")])
    preview = _preview(
        result,
        declared_entities=["Alice", "Bob", "Carol"],
        entities_without_records=["Bob", "Carol"],
    )
    # Bob has no record of his own but a relation names him, so a Subject exists.
    assert preview.entities_without_records == ["Bob", "Carol"]
    assert preview.entities_lost == ["Carol"]


def test_a_name_is_matched_the_way_the_subject_resolver_matches_it() -> None:
    """Case-insensitively: a relation spelling an entity ``bob`` re-attaches to ``Bob``."""
    result = ExtractionResult(candidates=[_candidate("Alice knows bob", "Alice", "bob")])
    preview = _preview(result, declared_entities=["Alice", "Bob"], entities_without_records=["Bob"])
    assert preview.entities_lost == []


def test_quality_notes_become_the_dropped_list() -> None:
    preview = _preview(ExtractionResult(quality_notes=["Line 2: skipped."]))
    assert preview.dropped == ["Line 2: skipped."]
    assert preview.particles == 0
    assert preview.sample == []


def test_sample_spans_the_export_and_is_deterministic() -> None:
    result = ExtractionResult(
        candidates=[_candidate(f"claim {i}", "S", location=f"line {i}") for i in range(10)]
    )
    first = _preview(result, sample_size=5)
    assert [s.location for s in first.sample] == [f"line {i}" for i in (0, 2, 4, 6, 8)]
    assert first.sample == _preview(result, sample_size=5).sample


def test_sample_size_is_bounded_by_the_candidates_and_zero_is_allowed() -> None:
    result = ExtractionResult(candidates=[_candidate("a", "S"), _candidate("b", "S")])
    assert len(_preview(result, sample_size=50).sample) == 2
    assert _preview(result, sample_size=0).sample == []


def test_a_bytes_only_preview_says_it_consulted_no_store() -> None:
    assert _preview(ExtractionResult()).store_consulted is False


def test_to_dict_round_trips_through_json() -> None:
    result = ExtractionResult(candidates=[_candidate("a", "S", location="line 1")])
    payload = json.loads(json.dumps(_preview(result, unmapped_fields={"row.x": 2}).to_dict()))
    assert payload["particles"] == 1
    assert payload["unmapped_fields"] == {"row.x": 2}
    assert payload["sample"] == [{"location": "line 1", "subjects": ["S"], "content": "a"}]

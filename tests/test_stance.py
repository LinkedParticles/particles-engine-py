"""Tests for particles/core/stance.py — stance-particle helpers."""

from __future__ import annotations

from particles.core.schema import (
    Confidence,
    Particle,
    RelationType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.stance import (
    STANCE_HOLDER_KEY,
    STANCE_KINDS,
    STANCE_MAGNITUDE_KEY,
    has_stance_marker,
    stance_holder,
    stance_magnitude,
)


def _particle(properties: dict[str, object] | None = None) -> Particle:
    return Particle(
        content="github:torvalds endorses the claim that X and Y are distinct.",
        confidence=Confidence(value=0.92, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test-agent",
        properties=properties,
    )


def test_stance_kinds_are_endorses_and_disputes() -> None:
    assert RelationType.ENDORSES in STANCE_KINDS
    assert RelationType.DISPUTES in STANCE_KINDS
    assert len(STANCE_KINDS) == 2


def test_endorses_disputes_not_symmetric() -> None:
    # stance edges are asymmetric (stance → target).
    from particles.store.relation_store import _SYMMETRIC_KINDS

    assert RelationType.ENDORSES not in _SYMMETRIC_KINDS
    assert RelationType.DISPUTES not in _SYMMETRIC_KINDS


def test_stance_holder_reads_property() -> None:
    p = _particle({STANCE_HOLDER_KEY: "github:torvalds"})
    assert stance_holder(p) == "github:torvalds"
    assert has_stance_marker(p) is True


def test_stance_holder_absent() -> None:
    assert stance_holder(_particle(None)) is None
    assert stance_holder(_particle({})) is None
    # A non-stance particle with other properties is not a stance.
    assert has_stance_marker(_particle({"nmo:hasIssuer": "X"})) is False


def test_stance_holder_empty_string_is_none() -> None:
    assert stance_holder(_particle({STANCE_HOLDER_KEY: ""})) is None


def test_stance_magnitude_reads_float() -> None:
    p = _particle({STANCE_HOLDER_KEY: "x", STANCE_MAGNITUDE_KEY: 0.5})
    assert stance_magnitude(p) == 0.5


def test_stance_magnitude_absent_is_none() -> None:
    # Absent (unqualified) is distinct from an explicit 0.0.
    assert stance_magnitude(_particle({STANCE_HOLDER_KEY: "x"})) is None


def test_stance_magnitude_explicit_zero() -> None:
    assert stance_magnitude(_particle({STANCE_HOLDER_KEY: "x", STANCE_MAGNITUDE_KEY: 0})) == 0.0


def test_stance_magnitude_bool_rejected() -> None:
    # bool is an int subclass; True/False must not be coerced to 1.0/0.0.
    p = _particle({STANCE_HOLDER_KEY: "x", STANCE_MAGNITUDE_KEY: True})
    assert stance_magnitude(p) is None

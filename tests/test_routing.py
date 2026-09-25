# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure pre-§6.6 router (D2).

``route_particle`` is the one rung order both write paths share: off the
conflict surface, then the ACTIVE duplicate, then the retired twin,
then the live-conflict check, then the ladder. Each
route is asserted, and so is the *order*, which the DB-backed suites cover
only end to end.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from particles.core.schema import (
    AssertionModality,
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.extraction.polarity import POLARITY_DECLINED, POLARITY_KEY
from particles.extraction.scope import SCOPE_DOCUMENT_META, SCOPE_KEY
from particles.ingest.duplicate_suppression import DuplicateIndex
from particles.ingest.routing import Route, route_particle, skips_conflict_resolution

CLAIM = "The bridge opened in 1932."


def _particle(
    content: str = CLAIM,
    *,
    properties: dict[str, Any] | None = None,
    status: Status = Status.ACTIVE,
    status_reason: StatusReason | None = None,
) -> Particle:
    return Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="general-extractor",
        asserted_at=datetime.now(UTC),
        status=status,
        status_reason=status_reason,
        assertion_modality=AssertionModality.FALSIFIABLE,
        subject_ids=["s1"],
        properties=properties,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id="entry-1",
                snapshot_id="snap-1",
            )
        ],
    )


def _retired_twin() -> Particle:
    return _particle(status=Status.RETRACTED, status_reason=StatusReason.EXPLICIT_RETRACTION)


def _route(
    particle: Particle,
    *,
    duplicates: Sequence[Particle] = (),
    retired_values: Sequence[Particle] = (),
    conflict: Particle | None = None,
    retired_ids: frozenset[str] = frozenset(),
) -> tuple[Route, Particle | None]:
    return route_particle(
        particle,
        duplicates=DuplicateIndex(duplicates),
        retired_values=DuplicateIndex(retired_values),
        conflict=conflict,
        retired_ids=retired_ids,
    )


# ---------------------------------------------------------------------------
# One test per route
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "properties",
    [{SCOPE_KEY: SCOPE_DOCUMENT_META}, {POLARITY_KEY: POLARITY_DECLINED}],
    ids=["document-meta", "non-asserted"],
)
def test_off_the_conflict_surface_is_unchecked(properties: dict[str, Any]) -> None:
    particle = _particle(properties=properties)
    assert skips_conflict_resolution(particle.properties)
    assert _route(particle, conflict=_particle("A different claim.")) == (Route.UNCHECKED, None)


def test_active_twin_suppresses() -> None:
    twin = _particle()
    assert _route(_particle(), duplicates=[twin]) == (Route.SUPPRESS, twin)


def test_retired_twin_is_held() -> None:
    twin = _retired_twin()
    assert _route(_particle(), retired_values=[twin]) == (Route.RETIRED_TWIN, twin)


def test_no_conflict_inserts() -> None:
    assert _route(_particle()) == (Route.INSERT, None)


def test_live_conflict_goes_to_the_ladder() -> None:
    conflict = _particle("The bridge opened in 1933.")
    assert _route(_particle(), conflict=conflict) == (Route.LADDER, conflict)


def test_conflict_retired_earlier_in_the_pass_inserts() -> None:
    """a conflict an earlier candidate demoted is no longer live."""
    conflict = _particle("The bridge opened in 1933.")
    routed = _route(_particle(), conflict=conflict, retired_ids=frozenset({conflict.id}))
    assert routed == (Route.INSERT, None)


# ---------------------------------------------------------------------------
# Rung order
# ---------------------------------------------------------------------------


def test_off_surface_outranks_every_other_rung() -> None:
    meta = {SCOPE_KEY: SCOPE_DOCUMENT_META}
    routed = _route(
        _particle(properties=meta),
        duplicates=[_particle(properties=meta)],
        retired_values=[_retired_twin()],
        conflict=_particle("A different claim."),
    )
    assert routed == (Route.UNCHECKED, None)


def test_duplicate_outranks_a_conflict() -> None:
    twin = _particle()
    routed = _route(_particle(), duplicates=[twin], conflict=_particle("A different claim."))
    assert routed == (Route.SUPPRESS, twin)


def test_duplicate_outranks_a_retired_twin() -> None:
    """a claim still believed somewhere absorbs the observation."""
    twin = _particle()
    routed = _route(_particle(), duplicates=[twin], retired_values=[_retired_twin()])
    assert routed == (Route.SUPPRESS, twin)


def test_retired_twin_outranks_a_conflict() -> None:
    twin = _retired_twin()
    routed = _route(_particle(), retired_values=[twin], conflict=_particle("A different claim."))
    assert routed == (Route.RETIRED_TWIN, twin)


def test_a_different_claim_falls_through_both_indexes() -> None:
    conflict = _particle("The bridge opened in 1933.")
    routed = _route(
        _particle(),
        duplicates=[_particle("Another claim.")],
        retired_values=[_particle("Yet another claim.", status=Status.RETRACTED)],
        conflict=conflict,
    )
    assert routed == (Route.LADDER, conflict)


def test_second_call_with_the_conflict_agrees_with_the_first() -> None:
    """The assertion path routes twice: without the conflict, then with it.

    A particle the first call sends past the cheap rungs must not change route
    on the second, other than INSERT becoming LADDER.
    """
    conflict = _particle("The bridge opened in 1933.")
    particle = _particle()
    assert _route(particle) == (Route.INSERT, None)
    assert _route(particle, conflict=conflict) == (Route.LADDER, conflict)

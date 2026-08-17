"""Tests for particles/core/status.py — §6.6 normative transition table."""

from __future__ import annotations

import pytest

from particles.core.status import Status, validate_transition


class TestValidTransitions:
    def test_new_to_active(self) -> None:
        validate_transition(None, Status.ACTIVE)

    def test_new_to_inconsistency(self) -> None:
        validate_transition(None, Status.INCONSISTENCY)

    def test_new_to_provenance_stale(self) -> None:
        # quarantine birth for the losing candidate of an
        # INCONSISTENT verdict. The CONFLICT_PENDING reason condition is
        # enforced at the persistence seam (insert_particle), not here.
        validate_transition(None, Status.PROVENANCE_STALE)

    def test_active_to_superseded(self) -> None:
        validate_transition(Status.ACTIVE, Status.SUPERSEDED)

    def test_active_to_retracted(self) -> None:
        validate_transition(Status.ACTIVE, Status.RETRACTED)

    def test_active_to_provenance_stale(self) -> None:
        validate_transition(Status.ACTIVE, Status.PROVENANCE_STALE)

    def test_inconsistency_to_provenance_stale(self) -> None:
        validate_transition(Status.INCONSISTENCY, Status.PROVENANCE_STALE)

    def test_inconsistency_to_retracted(self) -> None:
        validate_transition(Status.INCONSISTENCY, Status.RETRACTED)

    def test_inconsistency_to_inconsistency(self) -> None:
        validate_transition(Status.INCONSISTENCY, Status.INCONSISTENCY)

    def test_provenance_stale_to_retracted(self) -> None:
        validate_transition(Status.PROVENANCE_STALE, Status.RETRACTED)

    def test_provenance_stale_to_superseded(self) -> None:
        validate_transition(Status.PROVENANCE_STALE, Status.SUPERSEDED)

    def test_superseded_to_active(self) -> None:
        # The only exit from a terminal state. Like the quarantine
        # birth above, the reason condition (the row must currently
        # carry DUPLICATE_MERGED) is enforced at the persistence seam
        # (update_particle_status), not here — this table is keyed on status
        # alone. See tests/test_links_unmerge.py for the gate itself.
        validate_transition(Status.SUPERSEDED, Status.ACTIVE)


class TestInvalidTransitions:
    def test_active_to_active(self) -> None:
        with pytest.raises(ValueError, match="Invalid status transition"):
            validate_transition(Status.ACTIVE, Status.ACTIVE)

    def test_retracted_to_active(self) -> None:
        # deliberately does not open this by symmetry: a retraction is
        # always a principal's judgment, and only judgment-free transitions are
        # reversible..
        with pytest.raises(ValueError):
            validate_transition(Status.RETRACTED, Status.ACTIVE)

    def test_provenance_stale_to_active(self) -> None:
        # Stale particles are NOT reactivated; Reindex creates a new ACTIVE particle
        with pytest.raises(ValueError):
            validate_transition(Status.PROVENANCE_STALE, Status.ACTIVE)

    def test_active_to_inconsistency_direct(self) -> None:
        # INCONSISTENCY is created as a new particle; existing ACTIVE particles
        # don't transition to it
        with pytest.raises(ValueError):
            validate_transition(Status.ACTIVE, Status.INCONSISTENCY)

    def test_none_to_superseded(self) -> None:
        with pytest.raises(ValueError):
            validate_transition(None, Status.SUPERSEDED)

    def test_none_to_retracted(self) -> None:
        with pytest.raises(ValueError):
            validate_transition(None, Status.RETRACTED)

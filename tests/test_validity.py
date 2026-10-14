# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for core/validity.py: the window predicate and the prose default.

Pure functions, so required by tests/AGENTS.md: the interval of each kind,
both readings of ``--during`` and the point form, the ``as-of`` and
``coarse`` bases, the unkinded exclusion, the earliest-of close rule, the
cue parser's recognised forms and its named skips, and the rule digest's
stability. The schema validators (``valid_from <= valid_until``, no
``valid_until`` on an EVENT, the window's bounds) are pinned here too.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from particles.core.schema import (
    Confidence,
    Particle,
    TimeWindow,
    UncertaintyNature,
    ValidKind,
    WindowReading,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.validity import (
    ProseAnchor,
    ProseSkip,
    ValidFromBasis,
    ValidityFacts,
    anchor_from_prose,
    coarse_end,
    interval,
    prose_rule_digest,
    read_window,
    resolve_close,
    window_end_of_day,
    window_for_day,
)

D = lambda y, m, d: datetime(y, m, d, tzinfo=UTC)  # noqa: E731
WEEK = TimeWindow(start=D(2026, 10, 10), end=window_end_of_day(D(2026, 10, 16)))


def _facts(
    valid_from: datetime | None,
    kind: ValidKind | None = ValidKind.EVENT,
    *,
    valid_until: datetime | None = None,
    basis: str | None = ValidFromBasis.PROSE,
    granularity: str | None = None,
    closed_at: datetime | None = None,
) -> ValidityFacts:
    return ValidityFacts(
        valid_from=valid_from,
        valid_kind=kind,
        valid_until=valid_until,
        basis=basis,
        granularity=granularity,  # type: ignore[arg-type]
        closed_at=closed_at,
    )


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


class TestInterval:
    def test_event_is_a_point(self) -> None:
        assert interval(_facts(D(2026, 10, 10))) == (D(2026, 10, 10), D(2026, 10, 10))

    def test_open_state(self) -> None:
        assert interval(_facts(D(2026, 10, 10), ValidKind.STATE)) == (D(2026, 10, 10), None)

    def test_state_closes_at_earliest_end(self) -> None:
        f = _facts(
            D(2026, 10, 8),
            ValidKind.STATE,
            valid_until=D(2026, 10, 15),
            closed_at=D(2026, 10, 9),
        )
        assert interval(f) == (D(2026, 10, 8), D(2026, 10, 9))

    def test_coarse_state_spans_its_unit(self) -> None:
        f = _facts(D(2019, 1, 1), ValidKind.STATE, basis=ValidFromBasis.COARSE, granularity="year")
        start, end = interval(f) or (None, None)
        assert start == D(2019, 1, 1)
        assert end is not None and end.date() == D(2019, 12, 31).date()

    def test_coarse_event_spans_its_unit(self) -> None:
        """A coarse act ("published in 1999") happened somewhere in the unit."""
        f = _facts(D(1999, 1, 1), ValidKind.EVENT, basis=ValidFromBasis.COARSE, granularity="year")
        start, end = interval(f) or (None, None)
        assert start == D(1999, 1, 1)
        assert end is not None and end.date() == D(1999, 12, 31).date()

    def test_undated_and_unkinded_have_no_interval(self) -> None:
        assert interval(_facts(None)) is None
        assert interval(_facts(D(2026, 10, 10), None)) is None

    def test_coarse_end_month(self) -> None:
        assert coarse_end(D(2024, 2, 1), "month").date() == D(2024, 2, 29).date()


# ---------------------------------------------------------------------------
# The two readings of --during and the point form
# ---------------------------------------------------------------------------


class TestReadWindow:
    def test_event_in_window_began(self) -> None:
        assert read_window(_facts(D(2026, 10, 11)), WEEK) is WindowReading.BEGAN

    def test_event_before_window_matches_nothing(self) -> None:
        """The fault the review brief found: an earlier event must not be in force."""
        assert read_window(_facts(D(2026, 10, 1)), WEEK) is None

    def test_event_after_window_matches_nothing(self) -> None:
        assert read_window(_facts(D(2026, 10, 20)), WEEK) is None

    def test_state_started_before_is_in_force(self) -> None:
        f = _facts(D(2026, 9, 1), ValidKind.STATE)
        assert read_window(f, WEEK) is WindowReading.IN_FORCE

    def test_state_closed_before_window_is_out(self) -> None:
        f = _facts(D(2026, 9, 1), ValidKind.STATE, closed_at=D(2026, 10, 9))
        assert read_window(f, WEEK) is None

    def test_state_began_in_window(self) -> None:
        f = _facts(D(2026, 10, 10), ValidKind.STATE)
        assert read_window(f, WEEK) is WindowReading.BEGAN

    def test_as_of_basis_never_began(self) -> None:
        f = _facts(D(2026, 10, 12), ValidKind.STATE, basis=ValidFromBasis.AS_OF)
        assert read_window(f, WEEK) is WindowReading.HELD_AT

    def test_as_of_basis_before_window_held_at(self) -> None:
        f = _facts(D(2026, 10, 8), ValidKind.STATE, basis=ValidFromBasis.AS_OF)
        assert read_window(f, WEEK) is WindowReading.HELD_AT

    def test_coarse_year_within(self) -> None:
        f = _facts(D(2026, 1, 1), ValidKind.STATE, basis=ValidFromBasis.COARSE, granularity="year")
        assert read_window(f, WEEK) is WindowReading.WITHIN

    def test_coarse_other_year_out(self) -> None:
        f = _facts(D(2019, 1, 1), ValidKind.STATE, basis=ValidFromBasis.COARSE, granularity="year")
        assert read_window(f, WEEK) is None

    def test_coarse_event_never_in_force_beyond_its_unit(self) -> None:
        """The owner's ruling on the sample: "published in 1999" is not in force in 2026."""
        e = _facts(D(1999, 1, 1), ValidKind.EVENT, basis=ValidFromBasis.COARSE, granularity="year")
        later = TimeWindow(start=D(2000, 1, 1), end=D(2026, 12, 31))
        assert read_window(e, later) is None
        assert read_window(e, window_for_day(D(1999, 6, 1))) is WindowReading.WITHIN
        assert read_window(e, window_for_day(D(2000, 6, 1))) is None
        in_1999 = TimeWindow(start=D(1999, 3, 1), end=D(1999, 4, 1))
        assert read_window(e, in_1999) is WindowReading.WITHIN

    def test_unkinded_began_only(self) -> None:
        assert read_window(_facts(D(2026, 10, 12), None), WEEK) is WindowReading.BEGAN
        assert read_window(_facts(D(2026, 9, 1), None), WEEK) is None

    def test_undated_is_none(self) -> None:
        assert read_window(_facts(None), WEEK) is None

    def test_open_ended_window(self) -> None:
        after = TimeWindow(start=D(2026, 10, 10))
        assert read_window(_facts(D(2027, 1, 1)), after) is WindowReading.BEGAN
        before = TimeWindow(end=D(2026, 10, 9))
        assert read_window(_facts(D(2026, 10, 1)), before) is WindowReading.BEGAN
        assert read_window(_facts(D(2026, 10, 12)), before) is None

    def test_point_event_on_the_day(self) -> None:
        at = window_for_day(D(2026, 10, 10))
        assert read_window(_facts(D(2026, 10, 10)), at) is WindowReading.BEGAN
        assert read_window(_facts(D(2026, 10, 11)), at) is None

    def test_point_state_covering(self) -> None:
        at = window_for_day(D(2026, 10, 8))
        f = _facts(D(2026, 10, 1), ValidKind.STATE, closed_at=D(2026, 10, 9))
        assert read_window(f, at) is WindowReading.IN_FORCE
        closed = _facts(D(2026, 10, 1), ValidKind.STATE, closed_at=D(2026, 10, 7))
        assert read_window(closed, at) is None

    def test_point_as_of_held_at(self) -> None:
        at = window_for_day(D(2026, 10, 8))
        f = _facts(D(2026, 10, 8), ValidKind.STATE, basis=ValidFromBasis.AS_OF)
        assert read_window(f, at) is WindowReading.HELD_AT

    def test_point_unkinded_excluded(self) -> None:
        at = window_for_day(D(2026, 10, 8))
        assert read_window(_facts(D(2026, 10, 8), None), at) is None


class TestResolveClose:
    def test_earliest_of_three(self) -> None:
        close = resolve_close(
            valid_until=D(2026, 10, 15),
            successor_valid_from=D(2026, 10, 9),
            retired_at=D(2026, 10, 12),
        )
        assert close == (D(2026, 10, 9), "successor")

    def test_retired_at_alone(self) -> None:
        assert resolve_close(
            valid_until=None, successor_valid_from=None, retired_at=D(2026, 10, 12)
        ) == (
            D(2026, 10, 12),
            "retired_at",
        )

    def test_nothing_known(self) -> None:
        assert resolve_close(valid_until=None, successor_valid_from=None, retired_at=None) is None


# ---------------------------------------------------------------------------
# The prose default
# ---------------------------------------------------------------------------


class TestAnchorFromProse:
    def test_on_date_is_an_event(self) -> None:
        a = anchor_from_prose("The user picked up the keys to the Pali Hill flat on 2026-10-10.")
        assert isinstance(a, ProseAnchor)
        assert (a.valid_from, a.valid_kind, a.basis) == (
            D(2026, 10, 10),
            ValidKind.EVENT,
            ValidFromBasis.PROSE,
        )
        assert a.cue == "on 2026-10-10"

    def test_on_weekday_date(self) -> None:
        a = anchor_from_prose("The user plans to fly back to Delhi on Thursday, 2026-10-15.")
        assert isinstance(a, ProseAnchor) and a.valid_from == D(2026, 10, 15)

    def test_since_is_a_state(self) -> None:
        a = anchor_from_prose("The user has lived in Lajpat Nagar since 2024-03-01.")
        assert isinstance(a, ProseAnchor) and a.valid_kind is ValidKind.STATE

    def test_since_month_year(self) -> None:
        a = anchor_from_prose("The user has lived in Lajpat Nagar since March 2024.")
        assert isinstance(a, ProseAnchor)
        assert a.valid_from == D(2024, 3, 1) and a.granularity == "month"

    def test_as_of_is_held_at(self) -> None:
        a = anchor_from_prose("The user is in Mumbai for a week for work, as of 2026-10-08.")
        assert isinstance(a, ProseAnchor)
        assert a.basis is ValidFromBasis.AS_OF and a.valid_kind is ValidKind.STATE

    def test_in_year_is_a_coarse_event(self) -> None:
        a = anchor_from_prose("In 1999, the W3C published the first RDF specification.")
        assert isinstance(a, ProseAnchor)
        assert (a.basis, a.granularity, a.valid_from) == (
            ValidFromBasis.COARSE,
            "year",
            D(1999, 1, 1),
        )
        assert a.valid_kind is ValidKind.EVENT
        b = anchor_from_prose("In June 2025, Nintendo filed a copyright notice.")
        assert isinstance(b, ProseAnchor)
        assert b.granularity == "month" and b.valid_kind is ValidKind.EVENT

    def test_long_form_date(self) -> None:
        a = anchor_from_prose("He died of bile duct cancer on August 31, 2023, at the age of 72.")
        assert isinstance(a, ProseAnchor)
        assert a.valid_from == D(2023, 8, 31) and a.valid_kind is ValidKind.EVENT
        b = anchor_from_prose("The shop has been open since 3 March 2024.")
        assert isinstance(b, ProseAnchor) and b.valid_from == D(2024, 3, 3)

    def test_date_inside_an_identifier_is_not_anchored(self) -> None:
        a = anchor_from_prose("The beta header must be exactly `compact-2026-01-12`.")
        assert a == ProseSkip("no_cue")

    def test_week_before_is_not_a_cue(self) -> None:
        """The brief's second cue finding: an end-anchored cue sets nothing."""
        a = anchor_from_prose("Ledgerly was moved to SQLite in the week before 2026-09-20.")
        assert a == ProseSkip("no_cue")

    def test_session_date_is_not_a_cue(self) -> None:
        a = anchor_from_prose("The user moved to Mumbai, as stated in a session dated 2026-10-09.")
        assert a == ProseSkip("no_cue")

    def test_two_dates_skip(self) -> None:
        a = anchor_from_prose("From 2026-10-08 the user was away, returning on 2026-10-15.")
        assert a == ProseSkip("several_dates")

    def test_no_date_skip(self) -> None:
        assert anchor_from_prose("The user is vegetarian.") == ProseSkip("no_date")

    def test_year_inside_a_number_is_not_a_date(self) -> None:
        assert anchor_from_prose("The invoice total was 20190 rupees.") == ProseSkip("no_date")

    def test_rule_digest_is_stable(self) -> None:
        assert prose_rule_digest() == prose_rule_digest()
        assert len(prose_rule_digest()) == 12


# ---------------------------------------------------------------------------
# Schema validators
# ---------------------------------------------------------------------------


def _particle(**kwargs: object) -> Particle:
    return Particle(
        content="x",
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="t",
        **kwargs,  # type: ignore[arg-type]
    )


class TestSchema:
    def test_start_after_end_rejected(self) -> None:
        with pytest.raises(ValueError, match="valid_from must not be after"):
            _particle(valid_from=D(2026, 10, 20), valid_until=D(2026, 10, 10))

    def test_event_with_end_rejected(self) -> None:
        with pytest.raises(ValueError, match="EVENT"):
            _particle(valid_from=D(2026, 10, 1), valid_until=D(2026, 10, 10), valid_kind="EVENT")

    def test_state_with_both_ends_ok(self) -> None:
        p = _particle(valid_from=D(2026, 10, 1), valid_until=D(2026, 10, 10), valid_kind="STATE")
        assert p.valid_kind is ValidKind.STATE

    def test_naive_values_compare_as_utc(self) -> None:
        p = _particle(valid_from=datetime(2026, 10, 1), valid_until=D(2026, 10, 10))
        assert p.valid_from is not None

    def test_window_needs_a_bound(self) -> None:
        with pytest.raises(ValueError, match="needs a start or an end"):
            TimeWindow()

    def test_window_start_after_end_rejected(self) -> None:
        with pytest.raises(ValueError, match="start must not be after"):
            TimeWindow(start=D(2026, 10, 10), end=D(2026, 10, 1))

    def test_point_window(self) -> None:
        assert window_for_day(D(2026, 10, 10)).is_point
        assert not WEEK.is_point

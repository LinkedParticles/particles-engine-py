# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The consolidation cycle's pure cadence decisions.

Plain values in, a decision out: no store, no clock, no config (D2).
The orchestrator's use of them is pinned in ``test_consolidation.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from particles.core.consolidation_cadence import (
    CENSUS_DRIFT_TOLERANCE,
    decide_census,
    is_due,
)

NOW = datetime(2026, 10, 1, 3, 30, tzinfo=UTC)
WEEK = 168


class TestIsDue:
    def test_no_prior_run_is_due(self) -> None:
        assert is_due(None, NOW, 20)

    def test_younger_than_interval_is_not_due(self) -> None:
        assert not is_due(NOW - timedelta(hours=19), NOW, 20)

    def test_at_the_interval_is_due(self) -> None:
        assert is_due(NOW - timedelta(hours=20), NOW, 20)


class TestDecideCensus:
    def test_first_census_on_a_store_runs(self) -> None:
        decision = decide_census(enabled=True, interval_hours=WEEK, last_ran=None, now=NOW)
        assert decision.run
        assert decision.reason is None
        assert decision.last_ran is None and decision.next_due is None

    def test_inside_the_interval_skips_and_says_when(self) -> None:
        last = NOW - timedelta(days=1)
        decision = decide_census(enabled=True, interval_hours=WEEK, last_ran=last, now=NOW)
        assert not decision.run
        assert decision.last_ran == last
        assert decision.next_due == last + timedelta(hours=WEEK)
        assert decision.reason == (
            "census skipped: last ran 2026-09-30 03:30 UTC, next due 2026-10-07 03:30 UTC "
            "(consolidation.census.interval_hours = 168)"
        )

    def test_past_the_interval_runs(self) -> None:
        last = NOW - timedelta(days=8)
        decision = decide_census(enabled=True, interval_hours=WEEK, last_ran=last, now=NOW)
        assert decision.run
        assert decision.next_due == last + timedelta(hours=WEEK)

    def test_a_scheduler_a_little_early_still_finds_it_due(self) -> None:
        # Last week's run started a few minutes later than this one: without
        # the tolerance the census would slip to the next night every week.
        last = NOW - timedelta(hours=WEEK) + timedelta(minutes=7)
        assert decide_census(enabled=True, interval_hours=WEEK, last_ran=last, now=NOW).run

    def test_more_than_the_tolerance_early_is_not_due(self) -> None:
        last = NOW - timedelta(hours=WEEK) + CENSUS_DRIFT_TOLERANCE + timedelta(minutes=1)
        assert not decide_census(enabled=True, interval_hours=WEEK, last_ran=last, now=NOW).run

    def test_zero_interval_runs_every_cycle(self) -> None:
        decision = decide_census(enabled=True, interval_hours=0, last_ran=NOW, now=NOW)
        assert decision.run

    def test_store_wide_request_overrides_the_interval(self) -> None:
        last = NOW - timedelta(hours=1)
        decision = decide_census(
            enabled=True, interval_hours=WEEK, last_ran=last, now=NOW, store_wide=True
        )
        assert decision.run

    def test_disabled_skips_even_a_store_wide_request(self) -> None:
        decision = decide_census(
            enabled=False, interval_hours=WEEK, last_ran=None, now=NOW, store_wide=True
        )
        assert not decision.run
        assert decision.reason == "consolidation.census.enabled is false"

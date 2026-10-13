# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for particles/core/past_value.py, the past-value marker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from particles.core.past_value import records_past_value

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "update_probe"


@pytest.mark.parametrize(
    "text",
    [
        "The user lived in a city until 2019.",
        "Until 2021 the team shipped on Fridays.",
        "The user worked at a firm until June 2020.",
        "Before 2020 the service ran on one server.",
        "Back in 2018 the user owned a van.",
        "The user was previously a nurse.",
        "The project formerly used Subversion.",
        "The user used to swim at dawn.",
        "The user no longer drinks coffee.",
        "The user had been a tenant for years.",
        "As of 2026-01-01, when the user lived in a city, the user walked to work.",
    ],
)
def test_a_marked_past_value(text: str) -> None:
    assert records_past_value(text)


@pytest.mark.parametrize(
    "text",
    [
        # A bounded stay in the present tense is a current value.
        "The user is in a city until the end of the month.",
        "The user is staying with friends until Friday.",
        # A present-tense claim carrying a date is a current value.
        "The user lives in a city, as of 2026-02-10.",
        "The user's rent is 900 euros a month (noted 2026-03-01).",
        "Since 2024 the user has worked from home.",
        # A reported change states the new value.
        "The project was moved from one database to another.",
        "The user uses a laptop for work.",
        # "As of" alone, without the restatement's "when", is a dated present.
        "As of 2026-01-01 the user works remotely.",
    ],
)
def test_not_a_marked_past_value(text: str) -> None:
    assert not records_past_value(text)


def test_no_positive_row_in_any_update_probe_set_is_marked() -> None:
    """The marker can only keep claims ACTIVE: it must not touch a real update."""
    for name in (
        "stay_pairs",
        "stay_pairs_heldout",
        "gate_pairs",
        "gate_pairs_heldout",
        "past_value_pairs",
        "past_marker_heldout",
        "configuration_pairs",
    ):
        for row in json.loads((_FIXTURES / f"{name}.json").read_text())["rows"]:
            if row.get("expected") == "changes":
                assert not records_past_value(row["older"]), (name, row["id"])

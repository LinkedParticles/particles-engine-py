# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The restatement labelling sheet joins judge verdicts back to claims."""

from __future__ import annotations

import sqlite3
from types import ModuleType

import pytest

from particles.operations import restatement_sheet


def _load() -> ModuleType:
    return restatement_sheet


def test_only_same_verdicts_with_both_claims_stored_are_listed() -> None:
    sheet = _load()
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE particles (content TEXT, content_norm_hash TEXT)")
    conn.execute(
        "CREATE TABLE probe_verdicts (id INTEGER, recorded_at TEXT, probe_kind TEXT, "
        "hash_a TEXT, hash_b TEXT, verdict INTEGER)"
    )
    conn.executemany(
        "INSERT INTO particles VALUES (?, ?)",
        [("Earlier wording.", "ha"), ("Later wording.", "hb"), ("Other.", "hc")],
    )
    conn.executemany(
        "INSERT INTO probe_verdicts VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, "t1", "duplicate", "ha", "hb", 1),
            (2, "t2", "duplicate", "ha", "hc", 0),  # DIFFERENT: not a label candidate
            (3, "t3", "duplicate", "ha", "gone", 1),  # one side no longer stored
            (4, "t4", "contradiction", "ha", "hb", 1),  # another kind
        ],
    )
    assert sheet.sheet_rows(conn) == [("1", "t1", "Earlier wording.", "Later wording.", "")]


def test_the_admission_bar() -> None:
    sheet = _load()
    assert sheet.summarise(["same"] * 100)[2] >= 0.95
    assert sheet.summarise(["same"] * 99 + ["changed"])[2] < 0.95
    assert sheet.summarise(["same"] * 129 + ["changed"])[2] >= 0.95
    assert sheet.summarise(["", " "]) == (0, 0, 0.0)


@pytest.mark.parametrize(("correct", "total"), [(0, 10), (10, 10), (7, 20)])
def test_the_bound_sits_below_the_point_estimate(correct: int, total: int) -> None:
    assert _load().wilson_lower(correct, total) <= correct / total

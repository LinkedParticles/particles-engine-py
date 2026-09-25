# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure tests for :func:`plan_collapse`, the collapse decision.

The decision takes the set of present blob hashes as a plain value (
D2), so these tests need no database and no blob store. The end-to-end
behaviour, including the gather step that stats the blobs, is covered in
``tests/test_pending_collapse.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from particles.core.schema import ExtractionStatus
from particles.corpus.store import GenerationRow
from particles.ingest.pending_collapse import plan_collapse

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _gen(
    n: int,
    status: ExtractionStatus = ExtractionStatus.PENDING,
    *,
    collapsed: bool = False,
) -> GenerationRow:
    return GenerationRow(
        entry_id="entry",
        snapshot_id=f"s{n}",
        captured_at=_T0 + timedelta(minutes=n),
        extraction_status=status,
        content_hash=f"h{n}",
        collapsed=collapsed,
    )


def test_newest_eligible_generation_with_present_blob_wins() -> None:
    gens = [_gen(1), _gen(2, ExtractionStatus.FAILED), _gen(3)]
    plan = plan_collapse(gens, {"h1", "h2", "h3"})
    assert plan == [("s1", "s3"), ("s2", "s3")]


def test_missing_blob_falls_back_to_older_generation() -> None:
    gens = [_gen(1), _gen(2), _gen(3)]
    plan = plan_collapse(gens, {"h1", "h2"})
    assert plan == [("s1", "s2")]


def test_nothing_eligible_plans_nothing() -> None:
    # The newest is FAILED (cannot supersede); the older ones have no blob.
    gens = [_gen(1), _gen(2, ExtractionStatus.FAILED)]
    assert plan_collapse(gens, {"h2"}) == []
    assert plan_collapse(gens, set()) == []


def test_single_generation_plans_nothing() -> None:
    assert plan_collapse([_gen(1)], {"h1"}) == []


def test_already_collapsed_rows_neither_supersede_nor_recollapse() -> None:
    gens = [
        _gen(1),
        _gen(2, ExtractionStatus.COMPLETE, collapsed=True),
        _gen(3),
        _gen(4, ExtractionStatus.PENDING, collapsed=True),
    ]
    plan = plan_collapse(gens, {"h1", "h2", "h3", "h4"})
    # s4 is collapsed so cannot supersede; s3 does. s2 is already collapsed and
    # is not planned again.
    assert plan == [("s1", "s3")]


def test_complete_older_generation_is_not_collapsed() -> None:
    gens = [_gen(1, ExtractionStatus.COMPLETE), _gen(2)]
    assert plan_collapse(gens, {"h1", "h2"}) == []

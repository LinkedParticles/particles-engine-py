# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure tests for :func:`plan_unmerge_group`, the unmerge restorability decision.

No store and no session: the decision takes one merge event and the rows it
names as plain values (D2). The DB-backed round trip stays in
``tests/test_links_unmerge.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
    UnmergeGroup,
    UnmergeSkipReason,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.operations.links_suggest import plan_unmerge_group
from particles.store.event_store import OperatorEvent, OperatorEventType

_T = datetime(2026, 1, 1, tzinfo=UTC)


def _p(pid: str, status: Status = Status.ACTIVE, reason: StatusReason | None = None) -> Particle:
    base = Particle(
        id=pid,
        content="uv parses pyproject.toml during settings discovery.",
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        asserted_at=_T,
        subject_ids=["s"],
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="ce")],
    )
    return base.model_copy(update={"status": status, "status_reason": reason})


def _merged(pid: str) -> Particle:
    return _p(pid, Status.SUPERSEDED, StatusReason.DUPLICATE_MERGED)


def _event(payload: dict[str, Any] | None) -> OperatorEvent:
    return OperatorEvent(
        event_id="ev-1",
        occurred_at=_T,
        actor="test",
        event_type=OperatorEventType.DUPLICATES_MERGED,
        payload=payload,
    )


def _plan(listed: list[str], rows: list[Particle], **extra: Any) -> UnmergeGroup:
    payload = {"survivor": "s1", "superseded": listed, **extra}
    by_id = {p.id: p for p in [_p("s1"), *rows]}
    out = plan_unmerge_group(_event(payload), by_id)
    assert isinstance(out, UnmergeGroup)
    return out


# ---------------------------------------------------------------------------
# Payload validation — an unreadable event is a warning, not a group
# ---------------------------------------------------------------------------


def test_missing_payload_is_a_skip_warning() -> None:
    out = plan_unmerge_group(_event(None), {})
    assert out == "Event ev-1 has no readable survivor/superseded payload; skipped."


def test_non_string_survivor_is_a_skip_warning() -> None:
    out = plan_unmerge_group(_event({"survivor": 7, "superseded": ["a"]}), {})
    assert isinstance(out, str) and "ev-1" in out


def test_non_list_superseded_is_a_skip_warning() -> None:
    out = plan_unmerge_group(_event({"survivor": "s1", "superseded": "a"}), {})
    assert isinstance(out, str) and "ev-1" in out


def test_absent_superseded_list_is_an_empty_group() -> None:
    out = plan_unmerge_group(_event({"survivor": "s1"}), {"s1": _p("s1")})
    assert isinstance(out, UnmergeGroup)
    assert out.restored_ids == [] and out.skipped == []


# ---------------------------------------------------------------------------
# Per-copy classification
# ---------------------------------------------------------------------------


def test_merge_superseded_copy_is_restorable() -> None:
    group = _plan(["a"], [_merged("a")])
    assert group.restored_ids == ["a"]
    assert group.skipped == []
    # The decision plans; the caller counts the edges it actually deletes.
    assert group.relations_deleted == 0
    assert group.reverted is False


def test_absent_copy_is_missing() -> None:
    group = _plan(["gone"], [])
    assert group.restored_ids == []
    [skip] = group.skipped
    assert skip.particle_id == "gone"
    assert skip.reason is UnmergeSkipReason.MISSING
    assert skip.found_status is None and skip.found_status_reason is None


def test_active_copy_is_already_active() -> None:
    [skip] = _plan(["a"], [_p("a")]).skipped
    assert skip.reason is UnmergeSkipReason.ALREADY_ACTIVE
    assert skip.found_status == Status.ACTIVE.value


def test_other_status_is_not_superseded() -> None:
    row = _p("a", Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION)
    [skip] = _plan(["a"], [row]).skipped
    assert skip.reason is UnmergeSkipReason.NOT_SUPERSEDED
    assert skip.found_status == Status.RETRACTED.value
    assert skip.found_status_reason == StatusReason.EXPLICIT_RETRACTION.value


def test_superseded_for_another_reason_is_not_merge_superseded() -> None:
    row = _p("a", Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION)
    [skip] = _plan(["a"], [row]).skipped
    assert skip.reason is UnmergeSkipReason.NOT_MERGE_SUPERSEDED
    assert skip.found_status_reason == StatusReason.EXPLICIT_SUPERSESSION.value


def test_superseded_with_no_reason_is_not_merge_superseded() -> None:
    [skip] = _plan(["a"], [_p("a", Status.SUPERSEDED)]).skipped
    assert skip.reason is UnmergeSkipReason.NOT_MERGE_SUPERSEDED
    assert skip.found_status_reason is None


def test_mixed_group_keeps_listed_order_and_never_aborts() -> None:
    group = _plan(
        ["a", "gone", "b", "c"],
        [_merged("a"), _p("b"), _merged("c")],
    )
    assert group.restored_ids == ["a", "c"]
    assert [(s.particle_id, s.reason) for s in group.skipped] == [
        ("gone", UnmergeSkipReason.MISSING),
        ("b", UnmergeSkipReason.ALREADY_ACTIVE),
    ]


# ---------------------------------------------------------------------------
# The survivor is reported, never acted on
# ---------------------------------------------------------------------------


def test_group_carries_event_survivor_and_hash() -> None:
    group = _plan(["a"], [_merged("a")], content_hash="h")
    assert group.merge_event_id == "ev-1"
    assert group.survivor_id == "s1"
    assert group.survivor_status == Status.ACTIVE.value
    assert group.content_hash == "h"


def test_drifted_survivor_is_reported_not_blocking() -> None:
    by_id = {"s1": _p("s1", Status.PROVENANCE_STALE), "a": _merged("a")}
    out = plan_unmerge_group(_event({"survivor": "s1", "superseded": ["a"]}), by_id)
    assert isinstance(out, UnmergeGroup)
    assert out.survivor_status == Status.PROVENANCE_STALE.value
    assert out.restored_ids == ["a"]


def test_missing_survivor_reports_none() -> None:
    out = plan_unmerge_group(_event({"survivor": "s1", "superseded": ["a"]}), {"a": _merged("a")})
    assert isinstance(out, UnmergeGroup)
    assert out.survivor_status is None
    assert out.restored_ids == ["a"]

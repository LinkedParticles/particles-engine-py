# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for queue precision (particles/operations/curation/precision.py).

The pure function ``summarise_precision`` is pinned over a synthetic gesture
log: how each event type attributes to a card and an outcome, that the
strongest outcome wins per card, the window bounds, the open / expired
denominator from the retained collections, and the per-kind meanings. The
gather shell ``curation_precision`` is pinned over ``db_session`` with recorded
events and a written snapshot; the CLI and HTTP surfaces live in
``test_cli_curate.py`` / ``test_app.py``.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import RelationType
from particles.operations.curation import CardKind, CurationCard, summarise_precision
from particles.operations.curation.cards import gestures_for
from particles.operations.curation.precision import (
    PRECISION_EVENTS,
    RIGHT_MEANS,
    Outcome,
    classify_event,
    curation_precision,
    kind_of_key,
    render_precision_lines,
)
from particles.operations.curation.snapshot import BLOB_FORMAT
from particles.store.curation_snapshot_store import CollectionScope, write_snapshot
from particles.store.event_store import (
    EventRefKind,
    OperatorEvent,
    OperatorEventRef,
    OperatorEventType,
    record_event,
)

SINCE = datetime(2026, 9, 1, tzinfo=UTC)
UNTIL = datetime(2026, 10, 1, tzinfo=UTC)
MID = datetime(2026, 9, 15, tzinfo=UTC)


def _event(
    event_type: OperatorEventType,
    *,
    at: datetime = MID,
    refs: list[str] | None = None,
    payload: dict[str, Any] | None = None,
    actor: str = "curate",
) -> OperatorEvent:
    return OperatorEvent(
        event_id=str(uuid.uuid4()),
        occurred_at=at,
        actor=actor,
        event_type=event_type,
        refs=[OperatorEventRef(ref_kind=EventRefKind.PARTICLE, ref_id=r) for r in refs or []],
        payload=payload,
    )


def _card(kind: CardKind, *ids: str, **kwargs: Any) -> CurationCard:
    return CurationCard(
        kind=kind,
        particle_ids=list(ids),
        diagnostic="x",
        suggested_gestures=gestures_for(kind),
        **kwargs,
    )


def _row(report: Any, kind: CardKind) -> Any:
    return next(r for r in report.kinds if r.kind == kind.value)


class TestRightMeans:
    def test_every_kind_says_what_right_means(self) -> None:
        for kind in CardKind:
            acted, dismissed = RIGHT_MEANS[kind]
            assert acted and dismissed

    def test_dismissed_duplicate_and_dismissed_expiry_are_different_findings(self) -> None:
        assert "finder was wrong" in RIGHT_MEANS[CardKind.DUPLICATE_PAIR][1]
        assert "not worth a ruling" in RIGHT_MEANS[CardKind.STALE][1]

    def test_kind_of_key_handles_bare_and_prefixed_keys(self) -> None:
        assert kind_of_key("failed_snapshots") is CardKind.FAILED_SNAPSHOTS
        assert kind_of_key("gated_subjects") is CardKind.GATED_SUBJECTS
        assert kind_of_key("stale:p1") is CardKind.STALE
        assert kind_of_key("uncited_url:https://x/?a=b:c") is CardKind.UNCITED_URL
        assert kind_of_key("nonsense:p1") is None


class TestClassify:
    """One event type at a time: which card, which outcome."""

    def test_affirm_is_acted_by_its_card_key(self) -> None:
        ev = _event(
            OperatorEventType.BELIEF_AFFIRMED,
            refs=["p1"],
            payload={"card_key": "stale:p1", "kind": "stale"},
        )
        assert classify_event(ev, membership={}, deposited_urls=()) == [("stale:p1", Outcome.ACTED)]

    def test_snooze_is_undecided_and_permanent_snooze_is_dismissed(self) -> None:
        snoozed = _event(
            OperatorEventType.CURATION_CARD_SNOOZED,
            payload={"card_key": "stale:p1", "snoozed_until": "2026-10-01T00:00:00+00:00"},
        )
        dismissed = _event(
            OperatorEventType.CURATION_CARD_SNOOZED,
            payload={"card_key": "duplicate_pair:a|b", "snoozed_until": None},
        )
        assert classify_event(snoozed, membership={}, deposited_urls=()) == [
            ("stale:p1", Outcome.SNOOZED)
        ]
        assert classify_event(dismissed, membership={}, deposited_urls=()) == [
            ("duplicate_pair:a|b", Outcome.DISMISSED)
        ]

    def test_url_dismissal_is_a_deposit_when_the_url_is_now_in_the_corpus(self) -> None:
        url = "https://example.org/doc"
        permanent = _event(
            OperatorEventType.DEPOSIT_SUGGESTION_DISMISSED,
            payload={"canonical_url": url, "snooze_days": None},
        )
        snoozed = _event(
            OperatorEventType.DEPOSIT_SUGGESTION_DISMISSED,
            payload={"canonical_url": url, "snooze_days": 14},
        )
        key = f"uncited_url:{url}"
        assert classify_event(permanent, membership={}, deposited_urls={url}) == [
            (key, Outcome.ACTED)
        ]
        assert classify_event(permanent, membership={}, deposited_urls=()) == [
            (key, Outcome.DISMISSED)
        ]
        assert classify_event(snoozed, membership={}, deposited_urls={url}) == [
            (key, Outcome.SNOOZED)
        ]

    def test_review_rulings_by_action(self) -> None:
        for action, outcome in (
            ("PREFER_A", Outcome.ACTED),
            ("PREFER_B", Outcome.ACTED),
            ("DISCARD", Outcome.ACTED),
            ("BOTH_VALID", Outcome.DISMISSED),
            ("DEFER", Outcome.SNOOZED),
        ):
            ev = _event(
                OperatorEventType.REVIEW_RESOLVED,
                refs=["rec", "a", "b"],
                payload={"action": action},
                actor="review",
            )
            assert classify_event(ev, membership={}, deposited_urls=()) == [
                ("inconsistency:rec", outcome)
            ], action

    def test_system_closed_conflict_is_an_expiry(self) -> None:
        ev = _event(
            OperatorEventType.INCONSISTENCY_CLOSED,
            refs=["rec", "a"],
            payload={"cause": "withdrawn"},
        )
        assert classify_event(ev, membership={}, deposited_urls=()) == [
            ("inconsistency:rec", Outcome.EXPIRED)
        ]

    def test_abstraction_verdicts(self) -> None:
        accepted = _event(
            OperatorEventType.ABSTRACTION_RESOLVED,
            payload={"resolution": "accepted", "candidate_event_id": "cand"},
        )
        rejected = _event(
            OperatorEventType.ABSTRACTION_RESOLVED,
            payload={"resolution": "rejected", "candidate_event_id": "cand"},
        )
        assert classify_event(accepted, membership={}, deposited_urls=()) == [
            ("proposed_abstraction:cand", Outcome.ACTED)
        ]
        assert classify_event(rejected, membership={}, deposited_urls=()) == [
            ("proposed_abstraction:cand", Outcome.DISMISSED)
        ]

    def test_co_evidential_link_reconstructs_the_pair_key(self) -> None:
        ev = _event(
            OperatorEventType.RELATION_ADDED,
            refs=["zz", "aa"],
            payload={"relation_type": RelationType.CO_EVIDENTIAL.value},
            actor="links-add",
        )
        assert classify_event(ev, membership={}, deposited_urls=()) == [
            ("duplicate_pair:aa|zz", Outcome.ACTED)
        ]
        # A relation of another type names no card by itself.
        other = _event(
            OperatorEventType.RELATION_ADDED,
            refs=["zz", "aa"],
            payload={"relation_type": RelationType.CONTRADICTS.value},
        )
        assert classify_event(other, membership={}, deposited_urls=()) == []

    def test_auto_merge_names_one_pair_per_redundant_copy(self) -> None:
        ev = _event(
            OperatorEventType.DUPLICATES_MERGED,
            refs=["s", "r1", "r2"],
            payload={"survivor": "s", "superseded": ["r1", "r2"]},
        )
        assert classify_event(ev, membership={}, deposited_urls=()) == [
            ("duplicate_pair:r1|s", Outcome.ACTED),
            ("duplicate_pair:r2|s", Outcome.ACTED),
        ]

    def test_relink_batch_and_per_orphan(self) -> None:
        batch = _event(OperatorEventType.SUBJECTS_RELINKED, payload={"batch": True})
        single = _event(OperatorEventType.SUBJECTS_RELINKED, refs=["p1"], payload={})
        assert classify_event(batch, membership={}, deposited_urls=()) == [
            ("gated_subjects", Outcome.ACTED)
        ]
        assert classify_event(single, membership={}, deposited_urls=()) == [
            ("no_subject:p1", Outcome.ACTED)
        ]

    def test_reindex_run_acts_on_the_failed_snapshots_card(self) -> None:
        ev = _event(OperatorEventType.EXTRACT_RUN, payload={"route": "reindex"})
        assert classify_event(ev, membership={}, deposited_urls=()) == [
            ("failed_snapshots", Outcome.ACTED)
        ]

    def test_retraction_with_a_stamped_key_attributes_exactly(self) -> None:
        ev = _event(
            OperatorEventType.PARTICLE_RETRACTED,
            refs=["p1"],
            payload={"via": "curate", "card_key": "no_subject:p1", "kind": "no_subject"},
        )
        membership = {"p1": {"stale:p1", "no_subject:p1"}}
        assert classify_event(ev, membership=membership, deposited_urls=()) == [
            ("no_subject:p1", Outcome.ACTED)
        ]

    def test_retraction_without_a_key_attributes_to_every_retained_card_naming_it(self) -> None:
        ev = _event(
            OperatorEventType.PARTICLE_RETRACTED,
            refs=["p1"],
            payload={"store": "default", "operator": True},
            actor="cli:particle-retract",
        )
        membership = {"p1": {"stale:p1", "duplicate_pair:p1|p2"}}
        assert classify_event(ev, membership=membership, deposited_urls=()) == [
            ("duplicate_pair:p1|p2", Outcome.ACTED),
            ("stale:p1", Outcome.ACTED),
        ]
        assert classify_event(ev, membership={}, deposited_urls=()) == []


class TestSummarise:
    def test_empty_log_and_no_collection_is_all_zero(self) -> None:
        report = summarise_precision([], since=SINCE, until=UNTIL)
        assert report.offered == 0
        assert report.precision is None
        assert report.kinds == []
        assert report.events_read == 0

    def test_precision_is_over_decided_cards_only(self) -> None:
        events = [
            _event(OperatorEventType.BELIEF_AFFIRMED, payload={"card_key": "stale:p1"}),
            _event(OperatorEventType.BELIEF_AFFIRMED, payload={"card_key": "stale:p2"}),
            _event(
                OperatorEventType.CURATION_CARD_SNOOZED,
                payload={"card_key": "stale:p3", "snoozed_until": None},
            ),
            _event(
                OperatorEventType.CURATION_CARD_SNOOZED,
                payload={"card_key": "stale:p4", "snoozed_until": "2026-12-01T00:00:00+00:00"},
            ),
        ]
        report = summarise_precision(events, since=SINCE, until=UNTIL)
        row = _row(report, CardKind.STALE)
        assert (row.acted, row.dismissed, row.snoozed, row.open, row.expired) == (2, 1, 1, 0, 0)
        assert row.offered == 4
        assert row.precision == pytest.approx(2 / 3)
        assert report.precision == pytest.approx(2 / 3)
        assert row.acted_means == RIGHT_MEANS[CardKind.STALE][0]
        assert row.dismissed_means == RIGHT_MEANS[CardKind.STALE][1]

    def test_strongest_outcome_wins_and_a_card_counts_once(self) -> None:
        events = [
            _event(
                OperatorEventType.CURATION_CARD_SNOOZED,
                at=MID,
                payload={"card_key": "stale:p1", "snoozed_until": "2026-12-01T00:00:00+00:00"},
            ),
            _event(
                OperatorEventType.PARTICLE_RETRACTED,
                at=MID + timedelta(days=1),
                refs=["p1"],
                payload={"via": "curate", "card_key": "stale:p1"},
            ),
            # A later snooze does not demote an earlier ruling either.
            _event(
                OperatorEventType.CURATION_CARD_SNOOZED,
                at=MID + timedelta(days=2),
                payload={"card_key": "stale:p1", "snoozed_until": "2026-12-01T00:00:00+00:00"},
            ),
        ]
        report = summarise_precision(events, since=SINCE, until=UNTIL)
        row = _row(report, CardKind.STALE)
        assert (row.acted, row.snoozed, row.offered) == (1, 0, 1)
        assert report.events_read == 3

    def test_a_ruling_outranks_the_system_closing_the_record(self) -> None:
        events = [
            _event(
                OperatorEventType.INCONSISTENCY_CLOSED, refs=["rec"], payload={"cause": "lapsed"}
            ),
            _event(
                OperatorEventType.REVIEW_RESOLVED, refs=["rec", "a"], payload={"action": "PREFER_A"}
            ),
            _event(
                OperatorEventType.INCONSISTENCY_CLOSED, refs=["rec2"], payload={"cause": "lapsed"}
            ),
        ]
        report = summarise_precision(events, since=SINCE, until=UNTIL)
        row = _row(report, CardKind.INCONSISTENCY)
        assert (row.acted, row.expired, row.offered) == (1, 1, 2)
        assert row.precision == 1.0

    def test_window_bounds_are_inclusive_and_naive_stamps_are_utc(self) -> None:
        events = [
            _event(OperatorEventType.BELIEF_AFFIRMED, at=SINCE, payload={"card_key": "stale:p1"}),
            _event(OperatorEventType.BELIEF_AFFIRMED, at=UNTIL, payload={"card_key": "stale:p2"}),
            _event(
                OperatorEventType.BELIEF_AFFIRMED,
                at=SINCE - timedelta(seconds=1),
                payload={"card_key": "stale:p3"},
            ),
            _event(
                OperatorEventType.BELIEF_AFFIRMED,
                at=UNTIL + timedelta(seconds=1),
                payload={"card_key": "stale:p4"},
            ),
            _event(
                OperatorEventType.BELIEF_AFFIRMED,
                at=MID.replace(tzinfo=None),
                payload={"card_key": "stale:p5"},
            ),
        ]
        report = summarise_precision(
            events, since=SINCE.replace(tzinfo=None), until=UNTIL.replace(tzinfo=None)
        )
        assert _row(report, CardKind.STALE).acted == 3
        assert report.events_read == 3
        assert report.since.tzinfo is not None

    def test_events_of_other_types_and_non_reindex_runs_are_not_read(self) -> None:
        events = [
            _event(OperatorEventType.PARTICLE_TAGGED, refs=["p1"], payload={"tags": ["x"]}),
            _event(OperatorEventType.EXTRACT_RUN, payload={"route": "single"}),
            _event(OperatorEventType.CONSOLIDATION_RUN, payload={"format": 1}),
        ]
        report = summarise_precision(events, since=SINCE, until=UNTIL)
        assert report.events_read == 0
        assert report.unattributed_events == 0
        assert OperatorEventType.PARTICLE_TAGGED not in PRECISION_EVENTS

    def test_unattributed_resolving_events_are_counted_not_guessed(self) -> None:
        events = [
            _event(
                OperatorEventType.PARTICLE_RETRACTED,
                refs=["gone"],
                payload={"store": "default"},
                actor="cli:particle-retract",
            ),
            _event(OperatorEventType.PARTICLE_SUPERSEDED, refs=["gone2", "succ"], payload={}),
        ]
        report = summarise_precision(events, since=SINCE, until=UNTIL)
        assert report.unattributed_events == 2
        assert report.events_read == 2
        assert report.offered == 0

    def test_open_and_expired_come_from_the_retained_collections(self) -> None:
        current = [
            _card(CardKind.STALE, "p1"),
            _card(CardKind.STALE, "p2"),
            _card(CardKind.DUPLICATE_PAIR, "a", "b"),
        ]
        older = [
            _card(CardKind.STALE, "p1"),  # still open: not expired
            _card(CardKind.NO_SUBJECT, "p9"),  # gone from the current one: expired
            _card(CardKind.STALE, "p7"),  # gone, but acted on in the window
        ]
        events = [
            _event(OperatorEventType.BELIEF_AFFIRMED, payload={"card_key": "stale:p1"}),
            _event(
                OperatorEventType.PARTICLE_RETRACTED,
                refs=["p7"],
                payload={"store": "default"},
                actor="cli:particle-retract",
            ),
        ]
        report = summarise_precision(
            events, since=SINCE, until=UNTIL, open_cards=current, retired_cards=older
        )
        stale = _row(report, CardKind.STALE)
        assert (stale.acted, stale.open, stale.expired, stale.offered) == (2, 1, 0, 3)
        assert _row(report, CardKind.NO_SUBJECT).expired == 1
        assert _row(report, CardKind.DUPLICATE_PAIR).open == 1
        assert report.offered == 5
        assert report.unattributed_events == 0

    def test_a_standing_ruling_before_the_window_leaves_the_open_set(self) -> None:
        current = [_card(CardKind.STALE, "p1"), _card(CardKind.STALE, "p2")]
        standing = [
            _event(
                OperatorEventType.BELIEF_AFFIRMED,
                at=SINCE - timedelta(days=10),
                payload={"card_key": "stale:p1"},
            ),
            # A timed snooze before the window is not a standing ruling.
            _event(
                OperatorEventType.CURATION_CARD_SNOOZED,
                at=SINCE - timedelta(days=10),
                payload={"card_key": "stale:p2", "snoozed_until": "2030-01-01T00:00:00+00:00"},
            ),
        ]
        report = summarise_precision(
            [], since=SINCE, until=UNTIL, open_cards=current, standing_before=standing
        )
        assert _row(report, CardKind.STALE).open == 1
        assert report.events_read == 0

    def test_membership_also_indexes_a_cards_conflict_record(self) -> None:
        contested = _card(CardKind.CONTESTED, "p1", inconsistency_id="rec")
        events = [
            _event(
                OperatorEventType.PARTICLE_RETRACTED,
                refs=["rec"],
                payload={"store": "default"},
                actor="review",
            ),
        ]
        report = summarise_precision(events, since=SINCE, until=UNTIL, open_cards=[contested])
        assert _row(report, CardKind.CONTESTED).acted == 1

    def test_rows_lead_with_the_kinds_the_operator_touched(self) -> None:
        current = [_card(CardKind.NO_SUBJECT, f"o{i}") for i in range(5)]
        events = [
            _event(OperatorEventType.BELIEF_AFFIRMED, payload={"card_key": "stale:p1"}),
        ]
        report = summarise_precision(events, since=SINCE, until=UNTIL, open_cards=current)
        assert [r.kind for r in report.kinds] == ["stale", "no_subject"]


class TestRender:
    def test_lines_carry_the_summary_denominator_table_and_meanings(self) -> None:
        events = [
            _event(OperatorEventType.BELIEF_AFFIRMED, payload={"card_key": "stale:p1"}),
            _event(
                OperatorEventType.CURATION_CARD_SNOOZED,
                payload={"card_key": "duplicate_pair:a|b", "snoozed_until": None},
            ),
            _event(
                OperatorEventType.PARTICLE_RETRACTED,
                refs=["gone"],
                payload={},
                actor="cli:particle-retract",
            ),
        ]
        report = summarise_precision(
            events,
            since=SINCE,
            until=UNTIL,
            open_cards=[_card(CardKind.NO_SUBJECT, "o1")],
            snapshots_read=1,
            snapshot_built_at=UNTIL,
        )
        text = "\n".join(render_precision_lines(report))
        assert "2026-09-01 to 2026-10-01 (30 days)" in text
        assert "Of 2 cards the operator ruled on, 1 were real problems and 1 were not" in text
        assert "precision 0.50" in text
        assert "Denominator: 3 cards offered = 2 decided + 0 snoozed + 1 open untouched" in text
        assert "1 resolving event(s) in the window named no card" in text
        assert "1 retained collection(s)" in text
        assert "duplicate_pair" in text and "the finder was wrong" in text
        assert "the expiry was not worth a ruling" in text

    def test_meanings_can_be_left_out_and_no_ruling_is_said_plainly(self) -> None:
        report = summarise_precision([], since=SINCE, until=UNTIL)
        lines = render_precision_lines(report, meanings=False)
        text = "\n".join(lines)
        assert "No card was ruled on in this window" in text
        assert "No stored collection" in text
        assert "What acted and dismissed mean" not in text


class TestGather:
    """``curation_precision`` over a real session: events + a written snapshot."""

    @pytest.mark.asyncio
    async def test_empty_store_has_nothing_to_measure(self, db_session: AsyncSession) -> None:
        assert await curation_precision(db_session) is None

    @pytest.mark.asyncio
    async def test_reads_the_window_and_the_retained_collections(
        self, db_session: AsyncSession
    ) -> None:
        now = datetime.now(UTC)
        await record_event(
            db_session,
            actor="curate",
            event_type=OperatorEventType.BELIEF_AFFIRMED,
            refs=[(EventRefKind.PARTICLE, "p1")],
            payload={"card_key": "stale:p1", "kind": "stale"},
        )
        await record_event(
            db_session,
            actor="curate",
            event_type=OperatorEventType.CURATION_CARD_SNOOZED,
            payload={"card_key": "duplicate_pair:a|b", "snoozed_until": None},
        )
        cards = [_card(CardKind.STALE, "p1"), _card(CardKind.NO_SUBJECT, "o1")]
        blob = json.dumps(
            {"format": BLOB_FORMAT, "cards": [c.model_dump(mode="json") for c in cards]}
        )
        await write_snapshot(
            db_session,
            cards_json=blob,
            card_count=len(cards),
            scope=CollectionScope.STORE,
            per_kind_scope={},
            built_at=now - timedelta(hours=1),
        )
        older = json.dumps(
            {
                "format": BLOB_FORMAT,
                "cards": [
                    _card(CardKind.UNCITED_URL, corpus_url="https://x/").model_dump(mode="json")
                ],
            }
        )
        await write_snapshot(
            db_session,
            cards_json=older,
            card_count=1,
            scope=CollectionScope.STORE,
            per_kind_scope={},
            built_at=now - timedelta(days=2),
        )
        await db_session.commit()

        report = await curation_precision(db_session)
        assert report is not None
        assert report.snapshots_read == 2
        assert report.events_read == 2
        assert (report.acted, report.dismissed, report.open, report.expired) == (1, 1, 1, 1)
        assert _row(report, CardKind.STALE).acted == 1
        assert _row(report, CardKind.DUPLICATE_PAIR).dismissed == 1
        assert _row(report, CardKind.NO_SUBJECT).open == 1
        assert _row(report, CardKind.UNCITED_URL).expired == 1
        assert report.precision == 0.5
        # The default window is the configured one, ending now.
        assert (report.until - report.since).days == 30

        narrow = await curation_precision(db_session, since=now + timedelta(seconds=1))
        assert narrow is not None
        assert narrow.events_read == 0
        assert narrow.open == 1  # the collection still counts

    @pytest.mark.asyncio
    async def test_window_length_follows_config(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config

        # No env override for this knob, and reset_config() would
        # close the open session's engine, so the module's get_config is
        # patched to hand back a copy with the window changed.
        cfg = get_config()
        narrowed = cfg.model_copy(
            update={"curation": cfg.curation.model_copy(update={"precision_window_days": 7})}
        )
        monkeypatch.setattr("particles.operations.curation.precision.get_config", lambda: narrowed)
        await record_event(
            db_session,
            actor="curate",
            event_type=OperatorEventType.BELIEF_AFFIRMED,
            payload={"card_key": "stale:p1"},
        )
        await db_session.commit()
        report = await curation_precision(db_session)
        assert report is not None
        assert (report.until - report.since).days == 7

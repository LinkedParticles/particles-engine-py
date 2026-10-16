# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Event-anchored validity extraction — the deterministic parser gate.

Per tests/AGENTS.md, extractor *parsing* (the deterministic map from a fixed LLM
reply to candidates) is required-to-test; the LLM call itself is out of scope
(the integration tier drives the real model). These tests pin:

  * ``_gate_valid_until`` — the pure three-condition gate: an explicit boundary
    cue AND a confidence floor AND a future date, biased toward under-emission,
    with a named drop reason on every rejection (never silent). This is the
    over-eager-expiry guard, so the durable-mention and born-expired paths are
    the load-bearing cases.
  * ``_parse_extraction_response`` — the gate wired into the real parser: a
    future-bounded item populates ``CandidateParticle.valid_until`` + the audit
    crumbs; a durable mention / below-floor / born-expired / unparseable item
    carries no boundary and emits a quality note; the disabled config leaves
    every candidate unbounded.
  * ``candidate_to_particle`` — the boundary flows through to
    ``Particle.valid_until`` (the field the §9.3 lint + the as-of read consume).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from particles.config import ExtractionValidityConfig, ParticlesConfig
from particles.core.schema import ValidKind
from particles.core.validity import (
    VALID_FROM_BASIS_KEY,
    VALID_FROM_CONFIDENCE_KEY,
    VALID_FROM_CUE_KEY,
    ValidFromBasis,
)
from particles.extraction import general
from particles.extraction.general import (
    VALIDITY_BASIS_KEY,
    VALIDITY_CONFIDENCE_KEY,
    _gate_valid_from,
    _gate_valid_until,
    _parse_extraction_response,
    candidate_to_particle,
)

NOW = datetime(2026, 6, 1, tzinfo=UTC)
FUTURE = "2026-12-31"
PAST = "2020-01-01"


# ---------------------------------------------------------------------------
# The pure gate
# ---------------------------------------------------------------------------


class TestGateValidUntil:
    def test_emits_future_bounded_above_floor(self) -> None:
        vu, conf, basis, note = _gate_valid_until(
            FUTURE, 0.9, "runs through 2026", floor=0.75, now=NOW
        )
        assert vu is not None
        assert vu.date().isoformat() == FUTURE
        assert vu.tzinfo is not None  # normalized to aware UTC
        assert conf == 0.9
        assert basis == "runs through 2026"
        assert note is None

    def test_no_cue_emits_nothing_silently(self) -> None:
        # The durable-mention path: the model set valid_until null — no boundary,
        # and (correctly) no note, since nothing was dropped.
        for raw in (None, "", "   "):
            vu, conf, basis, note = _gate_valid_until(raw, None, None, floor=0.75, now=NOW)
            assert (vu, conf, basis, note) == (None, None, None, None)

    def test_below_floor_is_dropped_with_note(self) -> None:
        vu, _conf, _basis, note = _gate_valid_until(FUTURE, 0.5, "maybe", floor=0.75, now=NOW)
        assert vu is None
        assert note is not None and "VALIDITY_BELOW_FLOOR" in note

    def test_missing_confidence_is_dropped_with_note(self) -> None:
        vu, _conf, _basis, note = _gate_valid_until(FUTURE, None, "cue", floor=0.75, now=NOW)
        assert vu is None
        assert note is not None and "VALIDITY_BELOW_FLOOR" in note

    def test_born_expired_past_date_is_dropped(self) -> None:
        vu, _conf, _basis, note = _gate_valid_until(PAST, 0.99, "until 2020", floor=0.75, now=NOW)
        assert vu is None
        assert note is not None and "VALIDITY_BORN_EXPIRED" in note

    def test_equal_to_now_is_born_expired(self) -> None:
        # A boundary exactly at ``now`` provides no forward-looking value.
        vu, _conf, _basis, note = _gate_valid_until(
            NOW.date().isoformat(), 0.99, "today", floor=0.75, now=NOW
        )
        assert vu is None
        assert note is not None and "VALIDITY_BORN_EXPIRED" in note

    def test_unparseable_date_is_dropped(self) -> None:
        # An unresolved relative expression must not be guessed into a date.
        vu, _conf, _basis, note = _gate_valid_until("tomorrow", 0.99, "cue", floor=0.75, now=NOW)
        assert vu is None
        assert note is not None and "VALIDITY_UNPARSEABLE" in note

    def test_naive_iso_date_normalized_to_utc(self) -> None:
        vu, _conf, _basis, _note = _gate_valid_until(FUTURE, 0.9, None, floor=0.75, now=NOW)
        assert vu is not None and vu.tzinfo is UTC


# ---------------------------------------------------------------------------
# The parser, with the gate wired in
# ---------------------------------------------------------------------------


def _patch_config(monkeypatch: pytest.MonkeyPatch, *, enabled: bool, floor: float = 0.75) -> None:
    cfg = ParticlesConfig(
        extraction_validity=ExtractionValidityConfig(enabled=enabled, min_boundary_confidence=floor)
    )
    monkeypatch.setattr(general, "get_config", lambda: cfg)


def _reply(items: list[dict[str, object]]) -> str:
    return json.dumps(items)


class TestParserValidity:
    def test_future_boundary_populates_candidate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_config(monkeypatch, enabled=True)
        raw = _reply(
            [
                {
                    "content": "The contract runs through 2026.",
                    "subjects": [],
                    "confidence_value": 0.9,
                    "uncertainty_nature": "EPISTEMIC",
                    "valid_until": FUTURE,
                    "validity_confidence": 0.9,
                    "validity_basis": "runs through 2026",
                }
            ]
        )
        candidates, _notes = _parse_extraction_response(raw)
        assert len(candidates) == 1
        c = candidates[0]
        assert c.valid_until is not None and c.valid_until.date().isoformat() == FUTURE
        assert c.properties is not None
        assert c.properties[VALIDITY_BASIS_KEY] == "runs through 2026"
        assert c.properties[VALIDITY_CONFIDENCE_KEY] == 0.9

    def test_durable_mention_carries_no_boundary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The model correctly emitted null for a date-mentioning durable fact.
        _patch_config(monkeypatch, enabled=True)
        raw = _reply(
            [
                {
                    "content": "I met her in 2019.",
                    "subjects": [],
                    "confidence_value": 0.9,
                    "uncertainty_nature": "EPISTEMIC",
                    "valid_until": None,
                    "validity_confidence": None,
                    "validity_basis": None,
                }
            ]
        )
        candidates, _notes = _parse_extraction_response(raw)
        assert len(candidates) == 1
        assert candidates[0].valid_until is None

    def test_born_expired_dropped_with_note(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_config(monkeypatch, enabled=True)
        raw = _reply(
            [
                {
                    "content": "The order was in force until 2020.",
                    "subjects": [],
                    "confidence_value": 0.9,
                    "uncertainty_nature": "EPISTEMIC",
                    "valid_until": PAST,
                    "validity_confidence": 0.95,
                    "validity_basis": "until 2020",
                }
            ]
        )
        candidates, notes = _parse_extraction_response(raw)
        assert candidates[0].valid_until is None
        assert any("VALIDITY_BORN_EXPIRED" in n for n in notes)

    def test_below_floor_dropped_with_note(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_config(monkeypatch, enabled=True, floor=0.8)
        raw = _reply(
            [
                {
                    "content": "Maybe valid until the release.",
                    "subjects": [],
                    "confidence_value": 0.9,
                    "uncertainty_nature": "EPISTEMIC",
                    "valid_until": FUTURE,
                    "validity_confidence": 0.6,
                    "validity_basis": "until the release",
                }
            ]
        )
        candidates, notes = _parse_extraction_response(raw)
        assert candidates[0].valid_until is None
        assert any("VALIDITY_BELOW_FLOOR" in n for n in notes)

    def test_disabled_config_never_sets_boundary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_config(monkeypatch, enabled=False)
        raw = _reply(
            [
                {
                    "content": "The contract runs through 2026.",
                    "subjects": [],
                    "confidence_value": 0.9,
                    "uncertainty_nature": "EPISTEMIC",
                    "valid_until": FUTURE,
                    "validity_confidence": 0.99,
                    "validity_basis": "runs through 2026",
                }
            ]
        )
        candidates, _notes = _parse_extraction_response(raw)
        assert candidates[0].valid_until is None
        # No validity crumbs on properties when the feature is off.
        assert not (candidates[0].properties or {}).get(VALIDITY_BASIS_KEY)


# ---------------------------------------------------------------------------
# candidate_to_particle passthrough
# ---------------------------------------------------------------------------


class TestCandidateToParticle:
    def test_valid_until_flows_to_particle(self) -> None:
        from particles.core.schema import UncertaintyNature
        from particles.extraction.general import CandidateParticle

        boundary = datetime(2026, 12, 31, tzinfo=UTC)
        cand = CandidateParticle(
            content="The contract runs through 2026.",
            confidence_value=0.9,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            valid_until=boundary,
        )
        particle = candidate_to_particle(
            cand, corpus_entry_id="e1", snapshot_id="s1", subject_ids=[]
        )
        assert particle.valid_until == boundary

    def test_no_boundary_stays_none(self) -> None:
        from particles.core.schema import UncertaintyNature
        from particles.extraction.general import CandidateParticle

        cand = CandidateParticle(
            content="A durable fact.",
            confidence_value=0.9,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
        )
        particle = candidate_to_particle(
            cand, corpus_entry_id="e1", snapshot_id="s1", subject_ids=[]
        )
        assert particle.valid_until is None


# ---------------------------------------------------------------------------
# the start gate (the gate minus the future condition)
# ---------------------------------------------------------------------------


class TestGateValidFrom:
    def test_emits_past_start(self) -> None:
        vf, kind, conf, cue, note, basis = _gate_valid_from(
            "2026-10-10", "EVENT", 0.9, "on 2026-10-10", floor=0.75
        )
        assert vf is not None and vf.date().isoformat() == "2026-10-10" and vf.tzinfo is not None
        assert kind is ValidKind.EVENT and conf == 0.9 and cue == "on 2026-10-10"
        assert note is None and basis == ValidFromBasis.STATED

    def test_as_of_state_anchor(self) -> None:
        vf, kind, _, _, note, basis = _gate_valid_from(
            "2026-10-08", "STATE", 0.9, "I'm staying at", floor=0.75, raw_basis="as-of"
        )
        assert vf is not None and kind is ValidKind.STATE and note is None
        assert basis == ValidFromBasis.AS_OF

    def test_as_of_event_dropped(self) -> None:
        vf, *_, note, basis = _gate_valid_from(
            "2026-10-08", "EVENT", 0.9, "x", floor=0.75, raw_basis="as-of"
        )
        assert vf is None and note is not None and "VALID_FROM_AS_OF_EVENT" in note

    def test_unknown_basis_reads_as_stated(self) -> None:
        *_, basis = _gate_valid_from("2026-10-08", "STATE", 0.9, "x", floor=0.75, raw_basis="guess")
        assert basis == ValidFromBasis.STATED

    # an act the session performed, anchored at the session's date.
    def test_session_event_from_a_transcript(self) -> None:
        vf, kind, _, cue, note, basis = _gate_valid_from(
            "2026-10-14",
            "EVENT",
            0.9,
            "the tests passed",
            floor=0.75,
            raw_basis="session",
            session_source=True,
        )
        assert vf is not None and kind is ValidKind.EVENT and note is None
        assert basis == ValidFromBasis.SESSION and cue == "the tests passed"

    def test_session_basis_outside_a_transcript_is_dropped(self) -> None:
        vf, *_, note, basis = _gate_valid_from(
            "2026-10-14", "EVENT", 0.9, "x", floor=0.75, raw_basis="session", session_source=False
        )
        assert vf is None and note is not None
        assert "VALID_FROM_SESSION_OUTSIDE_TRANSCRIPT" in note
        assert basis == ValidFromBasis.STATED

    def test_session_state_reads_as_as_of(self) -> None:
        vf, kind, _, _, note, basis = _gate_valid_from(
            "2026-10-14", "STATE", 0.9, "x", floor=0.75, raw_basis="session", session_source=True
        )
        assert vf is not None and kind is ValidKind.STATE and note is None
        assert basis == ValidFromBasis.AS_OF

    def test_session_basis_is_preserved_and_closes(self) -> None:
        from particles.operations.query.window import CLOSING_BASES
        from particles.operations.validity import PRESERVED_BASES

        assert ValidFromBasis.SESSION in CLOSING_BASES
        assert ValidFromBasis.SESSION.value in PRESERVED_BASES

    def test_future_start_allowed(self) -> None:
        vf, kind, _, _, note, _ = _gate_valid_from("2099-01-01", "STATE", 0.9, "from", floor=0.75)
        assert vf is not None and kind is ValidKind.STATE and note is None

    def test_missing_is_silent(self) -> None:
        assert _gate_valid_from(None, None, None, None, floor=0.75) == (
            None,
            None,
            None,
            None,
            None,
            ValidFromBasis.STATED,
        )

    def test_unparseable_dropped_with_note(self) -> None:
        *_, note, _ = _gate_valid_from("last Thursday", "EVENT", 0.9, "x", floor=0.75)
        assert note is not None and "VALID_FROM_UNPARSEABLE" in note

    def test_unknown_kind_dropped(self) -> None:
        vf, kind, _, _, note, _ = _gate_valid_from("2026-10-10", "SPAN", 0.9, "x", floor=0.75)
        assert vf is None and kind is None and note is not None and "VALID_FROM_NO_KIND" in note

    def test_below_floor_dropped(self) -> None:
        vf, *_, note, _ = _gate_valid_from("2026-10-10", "EVENT", 0.5, "x", floor=0.75)
        assert vf is None and note is not None and "VALID_FROM_BELOW_FLOOR" in note


class TestParserValidFrom:
    def test_start_populates_candidate_and_crumbs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_config(monkeypatch, enabled=True)
        raw = _reply(
            [
                {
                    "content": "The user lives in Pali Hill since 2026-10-10.",
                    "subjects": [],
                    "confidence_value": 0.9,
                    "uncertainty_nature": "EPISTEMIC",
                    "valid_from": "2026-10-10",
                    "valid_kind": "STATE",
                    "valid_from_confidence": 0.9,
                    "valid_from_cue": "since 2026-10-10",
                }
            ]
        )
        candidates, _notes = _parse_extraction_response(raw)
        c = candidates[0]
        assert c.valid_from is not None and c.valid_from.date().isoformat() == "2026-10-10"
        assert c.valid_kind is ValidKind.STATE
        assert c.properties is not None
        assert c.properties[VALID_FROM_BASIS_KEY] == ValidFromBasis.STATED
        assert c.properties[VALID_FROM_CUE_KEY] == "since 2026-10-10"
        assert c.properties[VALID_FROM_CONFIDENCE_KEY] == 0.9

    def test_as_of_anchor_populates_basis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_config(monkeypatch, enabled=True)
        raw = _reply(
            [
                {
                    "content": "The user is staying at a hotel in Bandra, Mumbai.",
                    "subjects": [],
                    "confidence_value": 0.9,
                    "uncertainty_nature": "EPISTEMIC",
                    "valid_from": "2026-10-08",
                    "valid_kind": "STATE",
                    "valid_from_basis": "as-of",
                    "valid_from_confidence": 0.9,
                    "valid_from_cue": "I'm staying at",
                }
            ]
        )
        candidates, _ = _parse_extraction_response(raw)
        c = candidates[0]
        assert c.valid_from is not None and c.valid_kind is ValidKind.STATE
        assert c.properties is not None
        assert c.properties[VALID_FROM_BASIS_KEY] == ValidFromBasis.AS_OF

    def test_session_anchor_needs_the_transcript_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_config(monkeypatch, enabled=True)
        raw = _reply(
            [
                {
                    "content": "The cherry-pick of commit 5de2157d landed cleanly.",
                    "subjects": [],
                    "confidence_value": 0.9,
                    "uncertainty_nature": "EPISTEMIC",
                    "valid_from": "2026-10-14",
                    "valid_kind": "EVENT",
                    "valid_from_basis": "session",
                    "valid_from_confidence": 0.9,
                    "valid_from_cue": "landed cleanly",
                }
            ]
        )
        # Outside a transcript the anchor is dropped with a note.
        candidates, notes = _parse_extraction_response(raw)
        assert candidates[0].valid_from is None
        assert any("VALID_FROM_SESSION_OUTSIDE_TRANSCRIPT" in n for n in notes)
        # Inside one (the flag ``extract`` sets) it is an EVENT under ``session``.
        token = general._SESSION_SOURCE.set(True)
        try:
            candidates, notes = _parse_extraction_response(raw)
        finally:
            general._SESSION_SOURCE.reset(token)
        c = candidates[0]
        assert c.valid_from is not None and c.valid_kind is ValidKind.EVENT
        assert c.properties is not None
        assert c.properties[VALID_FROM_BASIS_KEY] == ValidFromBasis.SESSION
        assert c.properties[VALID_FROM_CUE_KEY] == "landed cleanly"
        assert not any("VALID_FROM_SESSION" in n for n in notes)

    def test_event_with_boundary_is_demoted_to_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_config(monkeypatch, enabled=True)
        raw = _reply(
            [
                {
                    "content": "The user is in Mumbai for a week from 2026-10-08.",
                    "subjects": [],
                    "confidence_value": 0.9,
                    "uncertainty_nature": "EPISTEMIC",
                    "valid_until": FUTURE,
                    "validity_confidence": 0.9,
                    "validity_basis": "for a week",
                    "valid_from": "2026-10-08",
                    "valid_kind": "EVENT",
                    "valid_from_confidence": 0.9,
                    "valid_from_cue": "from 2026-10-08",
                }
            ]
        )
        candidates, notes = _parse_extraction_response(raw)
        c = candidates[0]
        assert c.valid_until is not None and c.valid_from is not None
        assert c.valid_kind is ValidKind.STATE
        assert any("VALID_KIND_DEMOTED" in n for n in notes)

    def test_start_after_end_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_config(monkeypatch, enabled=True)
        raw = _reply(
            [
                {
                    "content": "x",
                    "subjects": [],
                    "confidence_value": 0.9,
                    "uncertainty_nature": "EPISTEMIC",
                    "valid_until": FUTURE,
                    "validity_confidence": 0.9,
                    "validity_basis": "through",
                    "valid_from": "2027-06-01",
                    "valid_kind": "STATE",
                    "valid_from_confidence": 0.9,
                    "valid_from_cue": "from",
                }
            ]
        )
        candidates, notes = _parse_extraction_response(raw)
        assert candidates[0].valid_from is None and candidates[0].valid_kind is None
        assert any("VALID_FROM_AFTER_END" in n for n in notes)

    def test_disabled_emits_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_config(monkeypatch, enabled=False)
        raw = _reply(
            [
                {
                    "content": "x",
                    "subjects": [],
                    "confidence_value": 0.9,
                    "uncertainty_nature": "EPISTEMIC",
                    "valid_from": "2026-10-10",
                    "valid_kind": "EVENT",
                    "valid_from_confidence": 0.9,
                    "valid_from_cue": "on",
                }
            ]
        )
        candidates, _ = _parse_extraction_response(raw)
        assert candidates[0].valid_from is None

    def test_start_flows_to_particle(self) -> None:
        from particles.core.schema import UncertaintyNature
        from particles.extraction.general import CandidateParticle

        start = datetime(2026, 10, 10, tzinfo=UTC)
        cand = CandidateParticle(
            content="x",
            confidence_value=0.9,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            valid_from=start,
            valid_kind=ValidKind.EVENT,
        )
        particle = candidate_to_particle(
            cand, corpus_entry_id="e1", snapshot_id="s1", subject_ids=[]
        )
        assert particle.valid_from == start and particle.valid_kind is ValidKind.EVENT

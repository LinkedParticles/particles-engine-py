# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The probe-verdict ledger.

A pairwise probe's answer is kept, keyed by both claims' content hashes and the
probe's prompt version, so a pair already cleared is not asked again. Pinned
here: the record round-trips; the census and the update sweep skip a
remembered NO with zero calls; an edited claim and a changed prompt both miss;
an unanswered probe is never remembered; and a skipped pair never consumes the
cap, so a capped census reaches the tail on later runs.

The LLM is mocked at the ``set_client`` seam (tests/AGENTS.md § Mocking
strategy), so every count below is a real call count through the adapter.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.probe_verdict import (
    ProbeKind,
    prompt_hash,
    remembered_clear,
    subject_link_key,
    verdict_key,
)
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.ingest.pipeline import UPDATE_SLOT_REQUEST
from particles.store.particle_store import insert_particle
from particles.store.probe_verdict_store import (
    count_verdicts,
    lookup_answer,
    lookup_verdicts,
    record_answer,
    record_verdict,
    record_verdicts,
)

# The update sweep's tests reuse its bag-of-words encoder fixture.
from tests.test_update_supersession import bow  # noqa: F401

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reply(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(text=text)], stop_reason="end_turn")


def _client(answer: Callable[[str], str]) -> MagicMock:
    """An Anthropic client whose every reply is ``answer(<the request's text>)``."""
    import anthropic

    client = MagicMock(spec=anthropic.Anthropic)

    def create(**kwargs: Any) -> SimpleNamespace:
        return _reply(answer(f"{kwargs.get('system')}\n{kwargs.get('messages')}"))

    client.messages.create.side_effect = create
    return client


NO = "REASON: Both can hold at once.\nVERDICT: NO"
YES = "REASON: They name different values.\nVERDICT: YES"


def _particle(content: str, entry: str) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test-agent",
        asserted_at=datetime.now(UTC) - timedelta(days=1),
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=entry)],
    )


def _emb(x: float, y: float) -> list[float]:
    return np.array([x, y] + [0.0] * 382, dtype=np.float32).tolist()


def _compatible_pair() -> list[tuple[str, list[float]]]:
    """Two near-neighbour claims that can both hold: above the gate, a NO from the probe."""
    return [
        ("The cache holds 10 entries.", _emb(0.6, 0.8)),
        ("The cache is LRU.", _emb(0.61, 0.79)),
    ]


async def _seed(session: AsyncSession, claims: list[tuple[str, list[float]]]) -> list[Particle]:
    out = []
    for i, (content, emb) in enumerate(claims):
        p = _particle(content, f"entry-{i}")
        await insert_particle(session, p, embedding=emb)
        out.append(p)
    await session.commit()
    return out


async def _census(session: AsyncSession, client: MagicMock, **control_kw: Any) -> Any:
    from particles import llm
    from particles.operations.lint import ContradictionProbeControl, _check_contradictions

    control = ContradictionProbeControl(**control_kw)
    llm.set_client(client)
    try:
        await _check_contradictions(session, fix=False, control=control)
    finally:
        llm.set_client(None)
    return control


# ---------------------------------------------------------------------------
# The key (pure)
# ---------------------------------------------------------------------------


class TestKey:
    def test_symmetric_kinds_ignore_order_and_slot_keeps_it(self) -> None:
        a, b = "The port is 8080.", "The port is 9090."
        assert verdict_key(ProbeKind.CONTRADICTION, a, b) == verdict_key(
            ProbeKind.CONTRADICTION, b, a
        )
        assert verdict_key(ProbeKind.UPDATE_SLOT, a, b) != verdict_key(ProbeKind.UPDATE_SLOT, b, a)

    def test_key_is_the_normalized_content_hash(self) -> None:
        # The same key the particles table stores as content_norm_hash.
        from particles.core.duplicate_key import content_hash

        key = verdict_key(ProbeKind.UPDATE_SLOT, "Claim one.", "Claim  two")
        assert key == (content_hash("Claim one"), content_hash("Claim two."))

    def test_prompt_hash_tracks_the_text(self) -> None:
        assert prompt_hash("a", "b") == prompt_hash("a", "b")
        assert prompt_hash("a", "b") != prompt_hash("a", "b ")
        assert prompt_hash("ab") != prompt_hash("a", "b")

    def test_a_no_from_either_stage_clears(self) -> None:
        assert remembered_clear(False, None)
        assert remembered_clear(True, False)
        assert remembered_clear(None, False)
        assert not remembered_clear(True, True)
        assert not remembered_clear(True, None)
        assert not remembered_clear(None, None)

    def test_each_probe_has_its_own_stable_prompt_hash(self) -> None:
        from particles.ingest.pipeline import contradiction_prompt_hash, update_prompt_hash
        from particles.ingest.second_reading import second_reading_prompt_hash
        from particles.operations.lint.contradictions import probe_prompt_hash

        hashes = [
            probe_prompt_hash(),
            second_reading_prompt_hash(),
            contradiction_prompt_hash(),
            update_prompt_hash(),
        ]
        # Stable: the per-call nonce is not part of the version.
        assert hashes == [
            probe_prompt_hash(),
            second_reading_prompt_hash(),
            contradiction_prompt_hash(),
            update_prompt_hash(),
        ]
        assert len(set(hashes)) == 4


class TestSubjectLinkKey:
    _OFFERED = [("Q1", "harbor", "body of water"), ("Q2", "Harbor", "family name")]

    def test_the_kind_is_not_a_symmetric_pair(self) -> None:
        assert not ProbeKind.SUBJECT_LINK.symmetric

    def test_name_claim_and_candidate_order_each_change_the_key(self) -> None:
        key = subject_link_key("Harbor", "Harbor bought Lantern.", self._OFFERED)
        assert key == subject_link_key("Harbor", "Harbor bought Lantern.", self._OFFERED)
        assert key != subject_link_key("Lantern", "Harbor bought Lantern.", self._OFFERED)
        assert key != subject_link_key("Harbor", "Harbor is a port.", self._OFFERED)
        assert key != subject_link_key("Harbor", "Harbor bought Lantern.", self._OFFERED[::-1])
        assert key[0] == subject_link_key("Harbor", "Harbor bought Lantern.", [])[0]


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStore:
    async def test_a_verdict_round_trips(self, db_session: AsyncSession) -> None:
        key = verdict_key(ProbeKind.CONTRADICTION, "A.", "B.")
        assert await record_verdict(
            db_session, ProbeKind.CONTRADICTION, "v1", key, False, model="anthropic/m"
        )
        assert await lookup_verdicts(db_session, ProbeKind.CONTRADICTION, "v1", [key]) == {
            key: False
        }
        # Another prompt version, or another kind, has no answer for the pair.
        assert await lookup_verdicts(db_session, ProbeKind.CONTRADICTION, "v2", [key]) == {}
        assert await lookup_verdicts(db_session, ProbeKind.SECOND_READING, "v1", [key]) == {}

    async def test_the_same_answer_twice_adds_nothing_and_a_new_one_appends(
        self, db_session: AsyncSession
    ) -> None:
        key = verdict_key(ProbeKind.CONTRADICTION, "A.", "B.")
        kind = ProbeKind.CONTRADICTION
        await record_verdict(db_session, kind, "v1", key, False, model="m")
        assert not await record_verdict(db_session, kind, "v1", key, False, model="m")
        assert await count_verdicts(db_session) == 1
        # A different answer is a new record; the newest is read, nothing is edited.
        assert await record_verdict(db_session, kind, "v1", key, True, model="m")
        assert await count_verdicts(db_session) == 2
        assert await lookup_verdicts(db_session, kind, "v1", [key]) == {key: True}

    async def test_an_answer_round_trips_keyed_by_model(self, db_session: AsyncSession) -> None:
        kind = ProbeKind.SUBJECT_LINK
        key = subject_link_key("Harbor", "Harbor bought Lantern.", [("Q1", "harbor", "water")])
        assert await record_answer(
            db_session, kind, "v1", key, verdict=False, answer='{"qid": null}', model="a/m"
        )
        assert await lookup_answer(db_session, kind, "v1", key, model="a/m") == '{"qid": null}'
        assert await lookup_answer(db_session, kind, "v1", key, model="a/other") is None
        assert await lookup_answer(db_session, kind, "v2", key, model="a/m") is None
        # The same answer again adds nothing; a new one appends and is read.
        assert not await record_answer(
            db_session, kind, "v1", key, verdict=False, answer='{"qid": null}', model="a/m"
        )
        assert await record_answer(
            db_session, kind, "v1", key, verdict=True, answer='{"qid": "Q1"}', model="a/m"
        )
        assert await count_verdicts(db_session, kind) == 2
        assert await lookup_answer(db_session, kind, "v1", key, model="a/m") == '{"qid": "Q1"}'

    async def test_lookup_answers_only_the_keys_asked_in_batches(
        self, db_session: AsyncSession
    ) -> None:
        kind = ProbeKind.UPDATE_SLOT
        keys = [verdict_key(kind, f"claim {i}", f"claim {i + 1}") for i in range(900)]
        written = await record_verdicts(
            db_session, kind, "v1", {k: i % 2 == 0 for i, k in enumerate(keys)}, model="m"
        )
        assert written == 900
        asked = keys[:450] + [verdict_key(kind, "never", "asked")]
        found = await lookup_verdicts(db_session, kind, "v1", asked)
        assert len(found) == 450
        assert found[keys[0]] is True and found[keys[1]] is False


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCensus:
    async def test_a_second_run_asks_nothing_for_unchanged_pairs(
        self, db_session: AsyncSession
    ) -> None:
        await _seed(
            db_session,
            _compatible_pair(),
        )
        client = _client(lambda _text: NO)

        first = await _census(db_session, client)
        assert (first.probes_run, first.previously_cleared) == (1, 0)
        assert client.messages.create.call_count == 1

        second = await _census(db_session, client)
        assert (second.probes_run, second.previously_cleared) == (0, 1)
        assert second.candidate_pairs == 0
        assert client.messages.create.call_count == 1  # no new call

    async def test_a_yes_is_recorded_and_still_reported(self, db_session: AsyncSession) -> None:
        await _seed(
            db_session,
            [("The port is 8080.", _emb(0.6, 0.8)), ("The port is 9090.", _emb(0.61, 0.79))],
        )
        client = _client(lambda _text: YES)
        for _ in range(2):
            control = await _census(db_session, client)
            assert control.probes_run == 1 and control.flagged == 1
        assert await count_verdicts(db_session, ProbeKind.CONTRADICTION) == 1

    async def test_an_edited_claim_misses_the_record(self, db_session: AsyncSession) -> None:
        from particles.store.particle_store import update_particle_status

        a, b = await _seed(
            db_session,
            _compatible_pair(),
        )
        client = _client(lambda _text: NO)
        await _census(db_session, client)

        # A correction is a new claim: retire B and assert its revision.
        await update_particle_status(db_session, b.id, Status.RETRACTED)
        await insert_particle(
            db_session, _particle("The cache is LFU.", "entry-9"), embedding=_emb(0.61, 0.79)
        )
        await db_session.commit()

        again = await _census(db_session, client)
        assert (again.probes_run, again.previously_cleared) == (1, 0)
        assert client.messages.create.call_count == 2

    async def test_a_prompt_change_misses_the_record(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import particles.llm as llm_pkg

        await _seed(
            db_session,
            _compatible_pair(),
        )
        client = _client(lambda _text: NO)
        await _census(db_session, client)

        # Change the trusted text the probe sends; its version hash follows.
        original = llm_pkg.data_fence_instruction
        monkeypatch.setattr(
            llm_pkg, "data_fence_instruction", lambda nonce: original(nonce) + " Be brief."
        )
        again = await _census(db_session, client)
        assert (again.probes_run, again.previously_cleared) == (1, 0)
        assert client.messages.create.call_count == 2

    async def test_an_unanswered_probe_is_never_remembered(self, db_session: AsyncSession) -> None:
        await _seed(
            db_session,
            _compatible_pair(),
        )
        # Cut at the budget: no verdict line, so neither a YES nor a NO.
        client = _client(lambda _text: "REASON: The first claim says the cache")
        for _ in range(2):
            control = await _census(db_session, client)
            assert control.probes_run == 1 and control.previously_cleared == 0
        assert await count_verdicts(db_session) == 0

    async def test_skipped_pairs_do_not_consume_the_cap(self, db_session: AsyncSession) -> None:
        """Each capped run reaches one pair further down the similarity order."""
        await _seed(
            db_session,
            [
                ("Claim one about caching.", _emb(1.0, 0.0)),
                ("Claim two about caching.", _emb(0.99, 0.141)),
                ("Claim three about caching.", _emb(0.95, 0.312)),
            ],
        )
        asked: list[str] = []

        def answer(text: str) -> str:
            asked.append(" / ".join(re.findall(r"Claim \w+ about caching", text)))
            return NO

        client = _client(answer)
        runs = [await _census(db_session, client, max_probes=1) for _ in range(4)]
        assert [(r.previously_cleared, r.candidate_pairs, r.probes_run) for r in runs] == [
            (0, 3, 1),
            (1, 2, 1),
            (2, 1, 1),
            (3, 0, 0),
        ]
        # Three distinct pairs, each asked once.
        assert len(asked) == 3 and len(set(asked)) == 3

    async def test_a_second_reading_no_clears_a_verifying_census(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.ingest.second_reading import ProbeVerdict
        from particles.operations.lint import contradictions

        readings = 0

        async def reading_says_no(*_a: object) -> ProbeVerdict:
            nonlocal readings
            readings += 1
            return ProbeVerdict(usable=True, contradicts=False, description="both hold")

        monkeypatch.setattr(contradictions, "_llm_verify_contradiction", reading_says_no)
        await _seed(
            db_session,
            [("The port is 8080.", _emb(0.6, 0.8)), ("The port is 9090.", _emb(0.61, 0.79))],
        )
        client = _client(lambda _text: YES)

        first = await _census(db_session, client, verify=True)
        assert (first.probes_run, first.flagged, first.confirmed) == (1, 1, 0)
        second = await _census(db_session, client, verify=True)
        assert (second.probes_run, second.previously_cleared) == (0, 1)
        assert readings == 1 and client.messages.create.call_count == 1

        # ``particles lint`` reports every flag, so a reading's NO does not clear
        # a non-verifying run: it asks the first probe again.
        lint_like = await _census(db_session, client)
        assert (lint_like.probes_run, lint_like.previously_cleared) == (1, 0)

    async def test_a_batched_census_records_and_skips_too(self, db_session: AsyncSession) -> None:
        from particles.operations.lint import contradictions

        await _seed(
            db_session,
            _compatible_pair(),
        )
        calls = 0

        async def batch(pairs: Any) -> list[Any]:
            nonlocal calls
            calls += len(pairs)
            return [None] * len(pairs)  # NO for every pair

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(contradictions, "_batch_check_contradictions", batch)
            first = await _census(db_session, _client(lambda _t: NO), latency_tolerant=True)
            second = await _census(db_session, _client(lambda _t: NO), latency_tolerant=True)
        assert (first.probes_run, second.probes_run, second.previously_cleared) == (1, 0, 1)
        assert calls == 1


# ---------------------------------------------------------------------------
# The manual update sweep
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.usefixtures("bow")
class TestUpdateSweep:
    """``reconcile --updates`` run twice by hand: the second run asks nothing."""

    @staticmethod
    async def _backlog(session: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.config import get_config
        from tests.test_update_supersession import _OneClaim, _say

        # A store as before: both values written ACTIVE.
        monkeypatch.setattr(get_config().reconciliation.update_supersession, "enabled", False)
        ex = _OneClaim()
        await _say(session, ex, "The user's home city is Boston.", day=1)
        await _say(session, ex, "The user's home city is Denver.", day=9)
        monkeypatch.setattr(get_config().reconciliation.update_supersession, "enabled", True)

    @staticmethod
    async def _sweep(session: AsyncSession, client: MagicMock) -> dict[str, object]:
        from particles import llm
        from particles.operations.reconcile import reconcile_updates

        llm.set_client(client)
        try:
            return await reconcile_updates(session)
        finally:
            llm.set_client(None)

    async def test_a_corroborating_pair_is_asked_once(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await self._backlog(db_session, monkeypatch)
        client = _client(lambda _text: NO)

        first = await self._sweep(db_session, client)
        assert (first["probed"], first["previously_cleared"], first["demoted"]) == (1, 0, 0)
        second = await self._sweep(db_session, client)
        assert (second["probed"], second["previously_cleared"], second["demoted"]) == (0, 1, 0)
        assert client.messages.create.call_count == 1

    async def test_a_different_slot_pair_is_asked_once(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await self._backlog(db_session, monkeypatch)
        # The claims conflict, but the slot probe says they fill different slots.
        client = _client(lambda text: NO if UPDATE_SLOT_REQUEST in text else YES)

        first = await self._sweep(db_session, client)
        assert (first["probed"], first["update_probed"], first["different_slot"]) == (1, 1, 1)
        assert client.messages.create.call_count == 2
        second = await self._sweep(db_session, client)
        assert (second["probed"], second["update_probed"], second["previously_cleared"]) == (
            0,
            0,
            1,
        )
        assert client.messages.create.call_count == 2
        assert await count_verdicts(db_session, ProbeKind.RECONCILE_CONTRADICTION) == 1
        assert await count_verdicts(db_session, ProbeKind.UPDATE_SLOT) == 1

    async def test_a_fixed_slot_pair_is_asked_once(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # one slot, but a fixed one. The ledger records that the pair
        # may not retire, so the second run neither asks nor opens a record.
        await self._backlog(db_session, monkeypatch)
        fixed = "REASON: One fixed slot.\nSLOT: FIXED\nVERDICT: YES"
        client = _client(lambda text: fixed if UPDATE_SLOT_REQUEST in text else YES)

        first = await self._sweep(db_session, client)
        assert (first["update_probed"], first["fixed_slot"], first["demoted"]) == (1, 1, 0)
        second = await self._sweep(db_session, client)
        assert (second["probed"], second["previously_cleared"], second["fixed_slot"]) == (0, 1, 0)
        assert client.messages.create.call_count == 2

    async def test_a_cleared_pair_does_not_consume_the_cap(
        self, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import get_config
        from tests.test_update_supersession import _OneClaim, _say

        monkeypatch.setattr(get_config().reconciliation.update_supersession, "enabled", False)
        ex = _OneClaim()
        for day, city in ((1, "Boston"), (9, "Denver"), (20, "Lisbon")):
            await _say(db_session, ex, f"The user's home city is {city}.", day=day)
        monkeypatch.setattr(get_config().reconciliation.update_supersession, "enabled", True)
        monkeypatch.setattr(get_config().consolidation, "max_update_probes", 1)
        client = _client(lambda _text: NO)

        runs = [await self._sweep(db_session, client) for _ in range(4)]
        assert [(r["previously_cleared"], r["candidate_pairs"], r["probed"]) for r in runs] == [
            (0, 3, 1),
            (1, 2, 1),
            (2, 1, 1),
            (3, 0, 0),
        ]
        assert client.messages.create.call_count == 3

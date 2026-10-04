# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The subject-link judge for an ambiguous Wikidata name.

What's covered:
  * the pure ambiguity test, the reply parser, the request's fencing and the
    prompt version
  * the authority under ``llm_judge``, with the Anthropic client mocked at the
    ``set_client`` seam: "none" leaves a bare local Subject with no ref and no
    negative cache entry; a pick is adopted at no less than the unscoreable
    sentinel; a name that is not ambiguous never reaches the model; a failed
    call keeps the top hit; ``llm_judge`` is the default and ``top_hit``
    never asks
  * the search depth: under ``llm_judge`` the one search asks for
    ``subjects.wikidata_judge_search_limit`` hits and the judge is offered all
    of them, so it may pick one past the fifth; the ambiguity gate reads only
    the first five, so a name that resolved without the model still does, and
    an empty first five stays bare local; ``top_hit`` searches five; a wider
    set is another verdict key
  * the ledger: the second ask for the same input reads the record and makes
    no call; a different model misses it
  * the call is metered on the ``subject_resolution`` purpose by an open
    usage scope, which is how an extract run's spend line includes it
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import ProviderSelection, get_config
from particles.core.probe_verdict import ProbeKind
from particles.ingest.authorities.wikidata import UNSCOREABLE, judged_choice
from particles.ingest.authorities.wikidata_judge import (
    NONE_OF_THESE,
    is_ambiguous,
    judge_candidates,
    judge_prompt_hash,
    judge_request,
    judgeable_hits,
    parse_judge_reply,
)
from particles.store.probe_verdict_store import ProbeVerdictRow
from tests._client_fixtures import stream_via_create

_CANDIDATES = "particles.ingest.authorities.wikidata._wikidata_candidates"
_ALIASES = "particles.ingest.authorities.wikidata._wikidata_aliases"
_SCORE = "particles.ingest.authorities.wikidata._wikidata_link_confidence"

_CLAIM = "Harbor has acquired Lantern, a payroll software vendor."
_HARBOR = [
    {"id": "Q283202", "label": "harbor", "description": "sheltered body of water"},
    {"id": "Q5655014", "label": "Harbor", "description": "family name"},
]
_RUST = [
    {"id": "Q190", "label": "rust", "description": "iron oxide"},
    {"id": "Q575650", "label": "Rust", "description": "programming language"},
]


def test_llm_judge_is_the_default_selection() -> None:
    from particles.config import SubjectsConfig

    assert SubjectsConfig().wikidata_candidate_selection == "llm_judge"


class TestIsAmbiguous:
    def test_no_candidate_is_not_ambiguous(self) -> None:
        assert not is_ambiguous(0, None, 0.25)

    def test_several_candidates_are_ambiguous_whatever_the_top_score(self) -> None:
        assert is_ambiguous(2, 0.9, 0.25)
        assert is_ambiguous(5, None, 0.25)

    def test_a_lone_candidate_scored_below_the_floor_is_ambiguous(self) -> None:
        assert is_ambiguous(1, 0.20, 0.25)

    def test_a_lone_candidate_at_or_above_the_floor_is_not(self) -> None:
        assert not is_ambiguous(1, 0.25, 0.25)
        assert not is_ambiguous(1, 0.6, 0.25)

    def test_a_lone_unscored_candidate_is_not(self) -> None:
        assert not is_ambiguous(1, None, 0.25)


class TestParseJudgeReply:
    _QIDS = ("Q283202", "Q5655014")

    def test_a_json_pick(self) -> None:
        assert parse_judge_reply('{"qid": "Q5655014"}', self._QIDS) == "Q5655014"

    def test_none_of_these(self) -> None:
        assert parse_judge_reply('{"qid": "None"}', self._QIDS) == NONE_OF_THESE

    def test_json_inside_a_code_fence(self) -> None:
        assert parse_judge_reply('```json\n{"qid": "q283202"}\n```', self._QIDS) == "Q283202"

    def test_a_bare_id(self) -> None:
        assert parse_judge_reply("Q283202", self._QIDS) == "Q283202"

    def test_an_id_it_was_not_offered_is_unusable(self) -> None:
        assert parse_judge_reply('{"qid": "Q1"}', self._QIDS) is None

    def test_prose_is_unusable(self) -> None:
        assert parse_judge_reply("It is probably the body of water.", self._QIDS) is None
        assert parse_judge_reply("", self._QIDS) is None


class TestRequest:
    def test_claim_and_candidates_are_fenced_and_instructions_stay_in_system(self) -> None:
        system, user = judge_request(
            "Harbor", _CLAIM, [("Q283202", "harbor", "sheltered body of water")], nonce="n0nce"
        )
        assert f'<claim nonce="n0nce">\n{_CLAIM}\n</claim nonce="n0nce">' in user
        assert "Q283202 | harbor | sheltered body of water" in user
        assert 'nonce="n0nce"' in system
        assert "none" in system
        assert "Reply with JSON" not in user

    def test_the_prompt_version_is_stable(self) -> None:
        assert judge_prompt_hash() == judge_prompt_hash()
        assert len(judge_prompt_hash()) == 16


class TestJudgeableHits:
    """Name and disambiguation items are never shown to the judge."""

    @pytest.mark.parametrize(
        "description",
        [
            "unisex given name",
            "female given name",
            "Male  given name",
            "given name",
            "family name",
            "surname",
            "name",
            "Wikimedia disambiguation page",
        ],
    )
    def test_a_name_or_disambiguation_item_is_dropped(self, description: str) -> None:
        assert judgeable_hits([{"id": "Q1", "description": description}]) == []

    @pytest.mark.parametrize(
        "description",
        ["", "country in West Asia", "Indian actress", "name of a ship", "hotel in Denver"],
    )
    def test_an_entity_is_kept(self, description: str) -> None:
        hit: dict[str, object] = {"id": "Q1", "description": description}
        assert judgeable_hits([hit]) == [hit]

    def test_rank_order_is_kept(self) -> None:
        hits: list[dict[str, object]] = [
            {"id": "Q14021944", "description": "unisex given name"},
            {"id": "Q810", "description": "country in West Asia"},
            {"id": "Q1703599", "description": "family name"},
            {"id": "Q270658", "description": "Flemish painter"},
        ]
        assert [h["id"] for h in judgeable_hits(hits)] == ["Q810", "Q270658"]


class TestJudgedChoice:
    def test_a_low_score_is_lifted_to_the_sentinel(self) -> None:
        choice = judged_choice(1, 0.05)
        assert (choice.index, choice.confidence, choice.resolve) == (1, UNSCOREABLE, True)

    def test_a_high_score_is_kept(self) -> None:
        assert judged_choice(0, 0.58).confidence == 0.58

    def test_no_score_is_the_sentinel(self) -> None:
        assert judged_choice(0, None).confidence == UNSCOREABLE


def _reply(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=300,
            output_tokens=10,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


@pytest.fixture
def llm_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_config().subjects, "wikidata_candidate_selection", "llm_judge")


@pytest.fixture
def judge_client(llm_judge: None) -> Iterator[MagicMock]:
    """A mocked Anthropic client; set ``client.messages.create.side_effect`` per test."""
    from particles.llm import set_client

    client = MagicMock(spec=anthropic.Anthropic)
    client.messages = MagicMock()
    client.messages.create = MagicMock(return_value=_reply('{"qid": "none"}'))
    stream_via_create(client)
    set_client(client)
    yield client
    set_client(None)


async def _resolve(session: AsyncSession, name: str, hits: list[dict[str, object]]) -> Any:
    from particles.ingest.subject_resolver import resolve_subject

    with (
        patch(_CANDIDATES, new_callable=AsyncMock, return_value=hits),
        patch(_ALIASES, new_callable=AsyncMock, return_value=[str(hits[0]["label"])]),
        patch(_SCORE, return_value=0.30),
    ):
        return await resolve_subject(session, name, "test", particle_content=_CLAIM)


class TestAuthority:
    @pytest.mark.asyncio
    async def test_none_leaves_a_bare_local_subject(
        self, db_session: AsyncSession, judge_client: MagicMock
    ) -> None:
        from particles.store import subject_cache

        resolved = await _resolve(db_session, "Harbor", _HARBOR)

        assert resolved.canonical_name == "Harbor"
        assert resolved.external_ids == []
        assert judge_client.messages.create.call_count == 1
        # The judgement depends on the claim, so it is no process-global miss.
        assert not subject_cache.negative_get("Harbor")
        rows = (await db_session.execute(select(ProbeVerdictRow))).scalars().all()
        assert [(r.probe_kind, r.verdict) for r in rows] == [(ProbeKind.SUBJECT_LINK, False)]
        # The rejected candidates stay on record, in the ledger only. The
        # family-name hit was never offered, so it is not among them.
        assert json.loads(rows[0].answer or "") == {
            "qid": None,
            "candidates": ["Q283202"],
        }

    @pytest.mark.asyncio
    async def test_a_pick_is_adopted_above_the_abstention_floor(
        self, db_session: AsyncSession, judge_client: MagicMock
    ) -> None:
        judge_client.messages.create.return_value = _reply('{"qid": "Q575650"}')
        with patch(_SCORE, return_value=0.05):  # the Rust case: a correct link scored low
            from particles.ingest.subject_resolver import resolve_subject

            with (
                patch(_CANDIDATES, new_callable=AsyncMock, return_value=_RUST),
                patch(_ALIASES, new_callable=AsyncMock, return_value=["Rust", "rust-lang"]),
            ):
                resolved = await resolve_subject(
                    db_session, "Rust", "test", particle_content="We chose Rust over Go."
                )
        assert [(r.id, r.confidence) for r in resolved.external_ids] == [("Q575650", UNSCOREABLE)]
        assert resolved.canonical_name == "Rust"
        assert resolved.description == "programming language"

    @pytest.mark.asyncio
    async def test_a_name_that_is_not_ambiguous_never_reaches_the_model(
        self, db_session: AsyncSession, judge_client: MagicMock
    ) -> None:
        resolved = await _resolve(db_session, "Harbor", _HARBOR[:1])  # one hit at 0.30
        assert [r.id for r in resolved.external_ids] == ["Q283202"]
        judge_client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_failed_call_keeps_the_top_hit(
        self, db_session: AsyncSession, judge_client: MagicMock
    ) -> None:
        judge_client.messages.create.side_effect = RuntimeError("connection reset")
        resolved = await _resolve(db_session, "Harbor", _HARBOR)
        assert [r.id for r in resolved.external_ids] == ["Q283202"]
        assert (await db_session.execute(select(ProbeVerdictRow))).scalars().all() == []

    @pytest.mark.asyncio
    async def test_top_hit_never_asks(
        self, db_session: AsyncSession, judge_client: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_config().subjects, "wikidata_candidate_selection", "top_hit")
        resolved = await _resolve(db_session, "Harbor", _HARBOR)
        assert [r.id for r in resolved.external_ids] == ["Q283202"]
        judge_client.messages.create.assert_not_called()


# The "Jest" case: the test framework is the sixth search hit, so
# five hits never show it to the judge. Ranks are what a live search stamps.
_JEST_CLAIM = "We also replaced Jest with Vitest, because Vitest reads the same config as Vite."
_JEST = [
    {
        "id": "Q371174",
        "label": "gesture",
        "description": "form of non-verbal communication",
        "rank": 1,
    },
    {
        "id": "Q1965390",
        "label": "Narrenzunft",
        "description": "traditional carnival club",
        "rank": 2,
    },
    {
        "id": "Q3352954",
        "label": "Jester Records",
        "description": "Norwegian record label",
        "rank": 3,
    },
    {"id": "Q102942", "label": "Jester Naefe", "description": "German actress", "rank": 4},
    {"id": "Q43441957", "label": "Ještědská", "description": "street in Liberec", "rank": 5},
    {"id": "Q65121527", "label": "Jest", "description": "Delightful JavaScript Testing", "rank": 6},
    {"id": "Q215548", "label": "jester", "description": "historical entertainer", "rank": 7},
]


class TestSearchDepth:
    """The judge is shown a deeper search; the gate and the other selections read five."""

    @pytest.mark.asyncio
    async def test_the_judge_is_shown_the_configured_depth_and_may_pick_past_five(
        self, db_session: AsyncSession, judge_client: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.ingest.subject_resolver import resolve_subject

        monkeypatch.setattr(get_config().subjects, "wikidata_judge_search_limit", 7)
        judge_client.messages.create.return_value = _reply('{"qid": "Q65121527"}')
        with (
            patch(_CANDIDATES, new_callable=AsyncMock, return_value=_JEST) as search,
            patch(_ALIASES, new_callable=AsyncMock, return_value=["Jest"]),
            patch(_SCORE, return_value=0.10),
        ):
            resolved = await resolve_subject(
                db_session, "Jest", "test", particle_content=_JEST_CLAIM
            )

        search.assert_awaited_once_with("Jest", limit=7)
        assert [(r.id, r.confidence) for r in resolved.external_ids] == [("Q65121527", UNSCOREABLE)]
        assert resolved.description == "Delightful JavaScript Testing"
        # All seven were offered, in rank order, and the ledger keyed on them.
        user = judge_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert user.index("Q371174") < user.index("Q65121527") < user.index("Q215548")
        rows = (await db_session.execute(select(ProbeVerdictRow))).scalars().all()
        assert json.loads(rows[0].answer or "")["candidates"] == [h["id"] for h in _JEST]

    @pytest.mark.asyncio
    async def test_the_gate_reads_the_first_five_so_a_lone_well_scored_hit_never_asks(
        self, db_session: AsyncSession, judge_client: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # One usable hit among the first five, scored above the floor, and two
        # more past them: the name resolves without the model, as at five.
        monkeypatch.setattr(get_config().subjects, "wikidata_judge_search_limit", 10)
        hits = [_JEST[0], _JEST[5], _JEST[6]]
        resolved = await _resolve(db_session, "Jest", hits)  # scored 0.30

        assert [r.id for r in resolved.external_ids] == ["Q371174"]
        judge_client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_empty_gate_stays_bare_local_whatever_lies_deeper(
        self, db_session: AsyncSession, judge_client: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_config().subjects, "wikidata_judge_search_limit", 10)
        resolved = await _resolve(db_session, "Jest", _JEST[5:])

        assert resolved.canonical_name == "Jest"
        assert resolved.external_ids == []
        judge_client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_lone_weak_hit_in_the_gate_is_judged_against_the_deeper_list(
        self, db_session: AsyncSession, judge_client: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.ingest.subject_resolver import resolve_subject

        monkeypatch.setattr(get_config().subjects, "wikidata_judge_search_limit", 10)
        judge_client.messages.create.return_value = _reply('{"qid": "Q65121527"}')
        hits = [_JEST[0], _JEST[5]]  # the gate sees one hit; the judge sees two
        with (
            patch(_CANDIDATES, new_callable=AsyncMock, return_value=hits),
            patch(_ALIASES, new_callable=AsyncMock, return_value=["Jest"]),
            patch(_SCORE, return_value=0.10),  # below the 0.25 floor: ambiguous
        ):
            resolved = await resolve_subject(
                db_session, "Jest", "test", particle_content=_JEST_CLAIM
            )

        assert [r.id for r in resolved.external_ids] == ["Q65121527"]
        assert judge_client.messages.create.call_count == 1

    @pytest.mark.asyncio
    async def test_the_judge_is_not_shown_name_items_and_its_pick_maps_back(
        self, db_session: AsyncSession, judge_client: MagicMock
    ) -> None:
        from particles.ingest.subject_resolver import resolve_subject

        hits = [
            {"id": "Q14021944", "label": "Jordan", "description": "unisex given name"},
            {"id": "Q1703599", "label": "Jordan", "description": "family name"},
            {"id": "Q810", "label": "Jordan", "description": "country in West Asia"},
        ]
        judge_client.messages.create.return_value = _reply('{"qid": "Q810"}')
        with (
            patch(_CANDIDATES, new_callable=AsyncMock, return_value=hits),
            patch(_ALIASES, new_callable=AsyncMock, return_value=["Jordan"]),
            patch(_SCORE, return_value=0.30),
        ):
            resolved = await resolve_subject(
                db_session, "Jordan", "test", particle_content="We flew to Jordan."
            )

        assert [r.id for r in resolved.external_ids] == ["Q810"]
        assert resolved.description == "country in West Asia"
        user = judge_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "Q14021944" not in user and "Q1703599" not in user

    @pytest.mark.asyncio
    async def test_only_name_items_abstain_without_a_call(
        self, db_session: AsyncSession, judge_client: MagicMock
    ) -> None:
        hits = [
            {"id": "Q2111323", "label": "Priya", "description": "female given name"},
            {"id": "Q55087847", "label": "Priya", "description": "family name"},
        ]
        resolved = await _resolve(db_session, "Priya", hits)

        assert resolved.external_ids == []
        judge_client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_top_hit_searches_five(
        self, db_session: AsyncSession, judge_client: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_config().subjects, "wikidata_candidate_selection", "top_hit")
        monkeypatch.setattr(get_config().subjects, "wikidata_judge_search_limit", 10)
        with (
            patch(_CANDIDATES, new_callable=AsyncMock, return_value=_HARBOR) as search,
            patch(_ALIASES, new_callable=AsyncMock, return_value=["harbor"]),
            patch(_SCORE, return_value=0.30),
        ):
            from particles.ingest.subject_resolver import resolve_subject

            await resolve_subject(db_session, "Harbor", "test", particle_content=_CLAIM)
        search.assert_awaited_once_with("Harbor", limit=5)

    def test_a_wider_candidate_set_is_another_verdict_key(self) -> None:
        from particles.core.probe_verdict import subject_link_key
        from particles.ingest.authorities.wikidata_judge import offered_candidates

        five = subject_link_key("Jest", _JEST_CLAIM, offered_candidates(_JEST[:5]))
        seven = subject_link_key("Jest", _JEST_CLAIM, offered_candidates(_JEST))
        assert five[0] == seven[0]  # the same name and claim
        assert five[1] != seven[1]  # another candidate set: asked again, once

    def test_the_search_limit_is_bounded_by_the_gate_and_the_api(self) -> None:
        from pydantic import ValidationError

        from particles.config import SubjectsConfig

        assert SubjectsConfig().wikidata_judge_search_limit == 7
        with pytest.raises(ValidationError):
            SubjectsConfig(wikidata_judge_search_limit=4)
        with pytest.raises(ValidationError):
            SubjectsConfig(wikidata_judge_search_limit=51)


class TestLedger:
    @pytest.mark.asyncio
    async def test_the_same_input_is_answered_from_the_record(
        self, db_session: AsyncSession, judge_client: MagicMock
    ) -> None:
        first = await judge_candidates(db_session, "Harbor", _CLAIM, _HARBOR)
        second = await judge_candidates(db_session, "Harbor", _CLAIM, _HARBOR)

        assert first is not None and first.index is None and not first.recorded
        assert second is not None and second.index is None and second.recorded
        assert judge_client.messages.create.call_count == 1

    @pytest.mark.asyncio
    async def test_a_changed_claim_or_candidate_set_asks_again(
        self, db_session: AsyncSession, judge_client: MagicMock
    ) -> None:
        await judge_candidates(db_session, "Harbor", _CLAIM, _HARBOR)
        await judge_candidates(db_session, "Harbor", "Harbor is a port city.", _HARBOR)
        await judge_candidates(db_session, "Harbor", _CLAIM, list(reversed(_HARBOR)))
        assert judge_client.messages.create.call_count == 3

    @pytest.mark.asyncio
    async def test_another_model_misses_the_record(
        self, db_session: AsyncSession, judge_client: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await judge_candidates(db_session, "Harbor", _CLAIM, _HARBOR)
        monkeypatch.setattr(
            get_config().llm,
            "subject_resolution",
            ProviderSelection(provider="anthropic", model="claude-haiku-4-5"),
        )
        await judge_candidates(db_session, "Harbor", _CLAIM, _HARBOR)
        assert judge_client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_an_unusable_reply_is_not_recorded(
        self, db_session: AsyncSession, judge_client: MagicMock
    ) -> None:
        judge_client.messages.create.return_value = _reply("the body of water, probably")
        assert await judge_candidates(db_session, "Harbor", _CLAIM, _HARBOR) is None
        assert (await db_session.execute(select(ProbeVerdictRow))).scalars().all() == []


class TestMetering:
    @pytest.mark.asyncio
    async def test_the_call_is_metered_under_its_purpose(
        self, db_session: AsyncSession, judge_client: MagicMock
    ) -> None:
        from particles.llm.usage import track_usage

        with track_usage() as usage:
            await judge_candidates(db_session, "Harbor", _CLAIM, _HARBOR)
        rows = usage.snapshot().rows
        assert [(r.purpose, r.calls) for r in rows] == [("subject_resolution", 1)]

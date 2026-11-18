# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The episode reading's two writers: the extractor and the regeneration reader.

* ``prompt.episode`` joins the prompt and the reply schema on a session source
  only, and a reply's ``episode`` lands in the claim's properties with the
  component's stamp; an unknown value writes nothing.
* The reader's reply parser maps each answer to its claim by number and
  leaves a missing or unknown one as no answer.
* The two writers carry different stamps that share one definition.
"""

from __future__ import annotations

import json

import pytest

from particles.core.episode import EPISODE_KEY, EPISODE_RULE_KEY
from particles.core.rung_ladder import EpisodeVerdict
from particles.extraction import general
from particles.extraction.episode import (
    EPISODE_DEFINITION,
    EpisodeItem,
    parse_episode_reply,
    read_episodes,
    regeneration_stamp,
)


def _reply(*episodes: str | None) -> str:
    items = []
    for i, episode in enumerate(episodes):
        item: dict[str, object] = {
            "content": f"Claim number {i}.",
            "subjects": ["Particles"],
            "confidence_value": 0.8,
            "uncertainty_nature": "EPISTEMIC",
        }
        if episode is not None:
            item["episode"] = episode
        items.append(item)
    return json.dumps(items)


def _parse(raw: str, *, session_source: bool) -> list[general.CandidateParticle]:
    token = general._SESSION_SOURCE.set(session_source)
    try:
        candidates, _ = general._parse_extraction_response(raw)
    finally:
        general._SESSION_SOURCE.reset(token)
    return candidates


class TestExtractor:
    def test_the_component_rides_a_session_source_only(self) -> None:
        on = {c.name for c in general._configured_prompt_components(session_source=True)}
        off = {c.name for c in general._configured_prompt_components(session_source=False)}
        assert general.PROMPT_EPISODE in on
        assert general.PROMPT_EPISODE not in off
        assert general.episode_prompt_component().name == general.PROMPT_EPISODE

    def test_the_reply_schema_names_episode_on_a_session_source_only(self) -> None:
        on = general._extraction_response_schema(
            scope_enabled=False, modality_enabled=False, session_source=True
        )
        off = general._extraction_response_schema(scope_enabled=False, modality_enabled=False)
        assert "episode" in on["items"]["properties"]
        assert "episode" not in off["items"]["properties"]

    def test_a_session_reply_stores_the_reading_with_the_component_stamp(self) -> None:
        a, b, c = _parse(_reply("session_bound", "standing", "maybe"), session_source=True)
        assert a.properties is not None
        assert a.properties[EPISODE_KEY] == "session_bound"
        assert a.properties[EPISODE_RULE_KEY] == general.episode_extractor_stamp()
        assert b.properties is not None and b.properties[EPISODE_KEY] == "standing"
        assert not (c.properties or {}).get(EPISODE_KEY)

    def test_another_source_never_stores_a_reading(self) -> None:
        (a,) = _parse(_reply("session_bound"), session_source=False)
        assert EPISODE_KEY not in (a.properties or {})


class TestReader:
    def test_answers_map_to_claims_by_number(self) -> None:
        raw = '[{"claim": 2, "episode": "standing"}, {"claim": 1, "episode": "session_bound"}]'
        assert parse_episode_reply(raw, 3) == (
            EpisodeVerdict.SESSION_BOUND,
            EpisodeVerdict.STANDING,
            None,
        )

    def test_a_fenced_or_unnumbered_reply_parses_by_position(self) -> None:
        raw = '```json\n[{"episode": "standing"}, {"episode": "nonsense"}]\n```'
        assert parse_episode_reply(raw, 2) == (EpisodeVerdict.STANDING, None)

    def test_an_unreadable_reply_is_no_answer(self) -> None:
        assert parse_episode_reply("I cannot tell.", 2) == (None, None)
        assert parse_episode_reply('{"episode": "standing"}', 1) == (None,)

    def test_the_two_writers_share_one_definition_under_two_stamps(self) -> None:
        assert EPISODE_DEFINITION in general.episode_prompt_component().rule
        assert general.episode_extractor_stamp().startswith("prompt.episode@")
        assert regeneration_stamp().startswith("regen.episode@")
        assert general.episode_extractor_stamp().split("@")[1] != regeneration_stamp().split("@")[1]

    @pytest.mark.asyncio
    async def test_the_reader_fences_the_claims_and_stamps_the_reply(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, str] = {}

        async def fake(purpose: str, prompt: str, **kw: object) -> tuple[str, str]:
            seen["prompt"], seen["system"] = prompt, str(kw.get("system"))
            return '[{"claim": 1, "episode": "session_bound"}]', "anthropic:test"

        monkeypatch.setattr("particles.llm.complete_with_provider_model", fake)
        reply = await read_episodes([EpisodeItem("PR #12 is open", "x" * 50)], passage_chars=10)
        assert reply is not None
        assert reply.verdicts == (EpisodeVerdict.SESSION_BOUND,)
        assert reply.stamp == regeneration_stamp()
        assert "Claim 1: PR #12 is open" in seen["prompt"]
        assert "x" * 11 not in seen["prompt"]  # the passage is cut to passage_chars

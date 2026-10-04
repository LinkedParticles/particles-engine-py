# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Grounded answers: per-sentence citation and attribution.

The parser tier is pure. The op tier drives the real query operation over an
in-memory store with the Anthropic client mocked at the shared ``set_client``
seam (tests/AGENTS.md § Mocking strategy), so the composer's prompt is
inspected exactly as it would be sent. The surface tier patches the backend.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.config import get_config
from particles.core.schema import (
    AnswerAttribution,
    AttributedSentence,
    AttributionKind,
    CalibrationSource,
    Confidence,
    Particle,
    ParticleType,
    ProvenanceRef,
    ProvenanceRefType,
    QueryRequest,
    QueryResponse,
    Status,
    UncertaintyNature,
)
from particles.operations.query.grounding import (
    GROUNDED_RULES,
    composed_particles,
    handle_for,
    parse_grounded,
    particle_handles,
)
from particles.operations.query.respond import REFUSAL_MARKER

cli = CliRunner()

_A = "1a2b3c4d-0000-4000-8000-000000000001"
_B = "5e6f7a8b-0000-4000-8000-000000000002"
_HANDLES = {_A: "p-1a2b3c4d", _B: "p-5e6f7a8b"}


def _particle(content: str, pid: str | None = None, **kwargs: Any) -> Particle:
    extra: dict[str, Any] = {"id": pid} if pid else {}
    return Particle(
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test-agent",
        status=Status.ACTIVE,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e", snapshot_id="s")
        ],
        **extra,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Handles
# ---------------------------------------------------------------------------


class TestHandles:
    def test_a_handle_is_the_display_id_form(self) -> None:
        assert handle_for(_A) == "p-1a2b3c4d"
        assert particle_handles([_particle("a", _A), _particle("b", _B)]) == _HANDLES

    def test_a_shared_prefix_never_names_two_particles(self) -> None:
        twin = "1a2b3c4d-ffff-4000-8000-000000000009"
        handles = particle_handles([_particle("a", _A), _particle("t", twin)])
        assert len(set(handles.values())) == 2
        assert all(len(h) == len("p-") + 32 for h in handles.values())

    def test_narrative_constituents_are_shown_and_handled(self) -> None:
        narrative = _particle("A memory.", particle_type=ParticleType.NARRATIVE)
        step = _particle("A step of it.", _B)
        shown = composed_particles([narrative, _particle("x", _A)], {narrative.id: [step]})
        assert [p.id for p in shown] == [narrative.id, _B, _A]


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


class TestParse:
    def test_every_unit_is_kept_and_labelled_as_declared(self) -> None:
        parsed = parse_grounded(
            "The floor defaults to 0.25 [p-1a2b3c4d]. It was added to stop off-topic "
            "answers [inference]. Pluto is far away [background].",
            _HANDLES,
        )
        kinds = [s.kind for s in parsed.attribution.sentences]
        assert kinds == [
            AttributionKind.CITED,
            AttributionKind.INFERENCE,
            AttributionKind.BACKGROUND,
        ]
        first = parsed.attribution.sentences[0]
        assert first.text == "The floor defaults to 0.25."
        assert first.cited_ids == [_A]
        assert parsed.answer == (
            "The floor defaults to 0.25 [p-1a2b3c4d]. It was added to stop off-topic "
            "answers [inference]. Pluto is far away [background]."
        )

    def test_a_cited_id_not_in_the_retrieved_set_is_recorded_never_trusted(self) -> None:
        """The composer cites an id it was never given: the parser does not take its word."""
        parsed = parse_grounded("Mars has two moons [p-deadbeef].", _HANDLES)
        (unit,) = parsed.attribution.sentences
        assert unit.kind is AttributionKind.UNATTRIBUTED
        assert unit.cited_ids == []
        assert unit.invalid_citations == ["p-deadbeef"]
        assert parsed.attribution.invalid_citation_count == 1
        assert parsed.answer == "Mars has two moons [unattributed]."

    def test_a_partly_invalid_citation_keeps_the_valid_ids(self) -> None:
        parsed = parse_grounded("Pluto was reclassified [p-5e6f7a8b, p-00000000].", _HANDLES)
        (unit,) = parsed.attribution.sentences
        assert unit.kind is AttributionKind.CITED
        assert unit.cited_ids == [_B]
        assert unit.invalid_citations == ["p-00000000"]
        assert parsed.answer == "Pluto was reclassified [p-5e6f7a8b]."

    def test_full_ids_and_other_prefix_forms_resolve(self) -> None:
        parsed = parse_grounded(f"One [{_A}]. Two [P:5E6F7A8B].", _HANDLES)
        assert [s.cited_ids for s in parsed.attribution.sentences] == [[_A], [_B]]

    def test_untagged_text_is_unattributed_not_dropped(self) -> None:
        parsed = parse_grounded("Cited [p-1a2b3c4d]. Then an untagged remark.", _HANDLES)
        tail = parsed.attribution.sentences[-1]
        assert (tail.kind, tail.text) == (AttributionKind.UNATTRIBUTED, "Then an untagged remark.")
        assert parsed.answer.endswith("Then an untagged remark. [unattributed]")

    def test_a_reply_with_no_tags_is_one_unattributed_unit(self) -> None:
        parsed = parse_grounded("The composer ignored the rules.", _HANDLES)
        assert [s.kind for s in parsed.attribution.sentences] == [AttributionKind.UNATTRIBUTED]

    def test_bracketed_prose_and_links_are_not_tags(self) -> None:
        parsed = parse_grounded(
            "See [the docs](https://x.org) and [sic] here [p-1a2b3c4d].", _HANDLES
        )
        (unit,) = parsed.attribution.sentences
        assert unit.kind is AttributionKind.CITED
        assert "[the docs](https://x.org)" in unit.text and "[sic]" in unit.text

    def test_adjacent_tags_are_one_attribution(self) -> None:
        parsed = parse_grounded("Both [p-1a2b3c4d][p-5e6f7a8b].", _HANDLES)
        (unit,) = parsed.attribution.sentences
        assert unit.cited_ids == [_A, _B]

    def test_a_declaration_beside_citations_keeps_them_as_premises(self) -> None:
        parsed = parse_grounded("So it was a dev default [p-1a2b3c4d] [inference].", _HANDLES)
        (unit,) = parsed.attribution.sentences
        assert unit.kind is AttributionKind.INFERENCE
        assert unit.cited_ids == [_A]

    def test_clause_level_citations_split_the_sentence(self) -> None:
        parsed = parse_grounded(
            "It reported CLEAN [p-1a2b3c4d], and stayed CLEAN later [p-5e6f7a8b].", _HANDLES
        )
        texts = [s.text for s in parsed.attribution.sentences]
        assert texts == ["It reported CLEAN", "and stayed CLEAN later."]

    def test_list_items_are_units_without_their_markers(self) -> None:
        parsed = parse_grounded("- First [p-1a2b3c4d].\n- Second [background].", _HANDLES)
        assert [s.text for s in parsed.attribution.sentences] == ["First.", "Second."]
        assert parsed.answer == "- First [p-1a2b3c4d].\n- Second [background]."


# ---------------------------------------------------------------------------
# The query op, Anthropic client mocked
# ---------------------------------------------------------------------------


class _Encoder:
    """Every text embeds to the same unit vector: every hit clears the floor."""

    def encode(self, texts: list[str], **_: Any) -> Any:
        return [np.ones(4, dtype=np.float32) / 2.0 for _ in texts]


def _client(answer: str) -> MagicMock:
    import anthropic

    content = MagicMock()
    content.text = answer
    content.type = "text"
    reply = MagicMock()
    reply.content = [content]
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages = MagicMock()
    client.messages.create = MagicMock(return_value=reply)
    return client


@pytest.fixture
async def two_beliefs(db_session: Any) -> Any:
    from particles import embeddings as ep
    from particles.store.particle_store import insert_particle

    for content, pid in (
        ("The relevance floor defaults to 0.25.", _A),
        ("Pluto was reclassified as a dwarf planet in 2006.", _B),
    ):
        await insert_particle(db_session, _particle(content, pid), [0.5, 0.5, 0.5, 0.5])
    await db_session.commit()
    original = ep._embedding_model
    ep.set_embedding_model(_Encoder())  # type: ignore[arg-type]
    try:
        yield db_session
    finally:
        ep.set_embedding_model(original)


async def _ask(session: Any, answer: str, **request: Any) -> tuple[QueryResponse, MagicMock]:
    from particles.llm import set_client
    from particles.operations.query import query

    client = _client(answer)
    set_client(client)
    try:
        response = await query(session, QueryRequest(question="What is the floor?", **request))
    finally:
        set_client(None)
    return response, client


def _sent(client: MagicMock) -> tuple[str, str]:
    """``(system, user)`` of the one composer call."""
    kwargs = client.messages.create.call_args.kwargs
    system = kwargs["system"]
    system_text = system if isinstance(system, str) else " ".join(b["text"] for b in system)
    user = kwargs["messages"][0]["content"]
    user_text = user if isinstance(user, str) else " ".join(b["text"] for b in user)
    return system_text, user_text


class TestQueryOp:
    async def test_a_grounded_request_cites_handles_and_returns_the_labels(
        self, two_beliefs: Any
    ) -> None:
        response, client = await _ask(
            two_beliefs,
            "The floor is 0.25 [p-1a2b3c4d]. It keeps answers on topic [inference].",
            grounded=True,
        )
        system, user = _sent(client)
        assert GROUNDED_RULES in system
        assert "[p-1a2b3c4d] The relevance floor defaults to 0.25." in user
        assert "[p-5e6f7a8b] Pluto was reclassified" in user
        attribution = response.answer_attribution
        assert attribution is not None
        assert [s.kind for s in attribution.sentences] == [
            AttributionKind.CITED,
            AttributionKind.INFERENCE,
        ]
        assert attribution.sentences[0].cited_ids == [_A]
        assert response.answer.endswith("It keeps answers on topic [inference].")

    async def test_a_composer_citing_an_unretrieved_id_is_caught(self, two_beliefs: Any) -> None:
        response, _ = await _ask(two_beliefs, "Mars has two moons [p-0badf00d].", grounded=True)
        assert response.answer_attribution is not None
        (unit,) = response.answer_attribution.sentences
        assert unit.kind is AttributionKind.UNATTRIBUTED
        assert unit.invalid_citations == ["p-0badf00d"]
        assert response.answer == "Mars has two moons [unattributed]."

    async def test_ungrounded_is_unchanged(self, two_beliefs: Any) -> None:
        response, client = await _ask(two_beliefs, "The floor is 0.25.", grounded=False)
        system, user = _sent(client)
        assert GROUNDED_RULES not in system and "[p-1a2b3c4d]" not in user
        assert response.answer == "The floor is 0.25."
        assert response.answer_attribution is None

    async def test_the_request_defers_to_the_configured_default(self, two_beliefs: Any) -> None:
        get_config().query.grounded_answers = True
        response, _ = await _ask(two_beliefs, "The floor is 0.25 [p-1a2b3c4d].")
        assert response.answer_attribution is not None
        response, _ = await _ask(two_beliefs, "The floor is 0.25.", grounded=False)
        assert response.answer_attribution is None

    async def test_a_refusal_is_unchanged_and_unattributed(self, two_beliefs: Any) -> None:
        response, _ = await _ask(
            two_beliefs, f"{REFUSAL_MARKER}\nNothing here bears on it.", grounded=True
        )
        assert response.answer_refused is True
        assert response.answer == "Nothing here bears on it."
        assert response.answer_attribution is None

    async def test_nothing_the_composer_adds_is_stored(self, two_beliefs: Any) -> None:
        from particles.store.particle_store import count_particles_by_status

        before = await count_particles_by_status(two_beliefs)
        await _ask(two_beliefs, "Pluto is cold [background].", grounded=True)
        assert await count_particles_by_status(two_beliefs) == before


# ---------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------


def _grounded_response() -> QueryResponse:
    return QueryResponse(
        answer="The floor is 0.25 [p-1a2b3c4d]. Mars has moons [unattributed].",
        particles=[_particle("The relevance floor defaults to 0.25.", _A)],
        effective_confidences=[0.9],
        answer_attribution=AnswerAttribution(
            sentences=[
                AttributedSentence(
                    text="The floor is 0.25.", kind=AttributionKind.CITED, cited_ids=[_A]
                ),
                AttributedSentence(
                    text="Mars has moons.",
                    kind=AttributionKind.UNATTRIBUTED,
                    invalid_citations=["p-0badf00d"],
                ),
            ]
        ),
    )


@pytest.fixture
def backend() -> Iterator[MagicMock]:
    be = MagicMock()
    be.remote = False
    be.query = AsyncMock(return_value=_grounded_response())
    be.inconsistency_backrefs = AsyncMock(return_value={})
    with patch("particles.api.cli.query.get_backend", return_value=be):
        yield be


class TestSurfaces:
    def test_cli_renders_the_labels_and_warns_on_an_invalid_citation(
        self, backend: MagicMock
    ) -> None:
        result = cli.invoke(app, ["query", "What is the floor?", "--grounded"])
        assert result.exit_code == 0, result.output
        assert backend.query.await_args.args[0].grounded is True
        assert "The floor is 0.25 [p-1a2b3c4d]." in result.stdout
        assert "Attribution: 1 cited · 0 inference · 0 background · 1 unattributed" in result.stdout
        assert "[p-1a2b3c4d] The relevance floor defaults to 0.25." in result.stdout
        assert "1 citation(s) named a particle that was not retrieved" in result.stderr

    def test_cli_leaves_the_mode_to_config_unless_asked(self, backend: MagicMock) -> None:
        cli.invoke(app, ["query", "What is the floor?"])
        assert backend.query.await_args.args[0].grounded is None
        cli.invoke(app, ["query", "What is the floor?", "--ungrounded"])
        assert backend.query.await_args.args[0].grounded is False

    async def test_mcp_passes_the_mode_and_returns_the_attribution(
        self, backend: MagicMock
    ) -> None:
        from particles.mcp.tools.query import query as mcp_query

        with patch("particles.api.client.get_backend", return_value=backend):
            out = await mcp_query(question="What is the floor?", grounded=True, summary=True)
        assert backend.query.await_args.args[0].grounded is True
        kinds = [s["kind"] for s in out["answer_attribution"]["sentences"]]
        assert kinds == ["cited", "unattributed"]

# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The MCP ``query`` tool's result: the compact shape and the cited list.

The query operation is replaced at ``particles.operations.query.query`` (the
deferred-import seam, tests/AGENTS.md § Mocking strategy), so each test fixes
the ``QueryResponse`` and reads what the tool makes of it and of the request.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest

from particles.config import get_config
from particles.core.schema import (
    AnswerAttribution,
    AttributedSentence,
    AttributionKind,
    ClaimTerm,
    Confidence,
    ContestedBadge,
    HitSource,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    QueryRequest,
    QueryResponse,
    StructuredClaim,
    TermKind,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status


def _particle(content: str, *, sources: int = 1, **fields: Any) -> Particle:
    return Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        status=Status.ACTIVE,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id=f"e{i}", snapshot_id=f"s{i}"
            )
            for i in range(sources)
        ],
        **fields,
    )


def _cited(*groups: list[Particle]) -> AnswerAttribution:
    """One cited sentence per group, citing that group's particles."""
    return AnswerAttribution(
        sentences=[
            AttributedSentence(
                text=f"Sentence {i}.", kind=AttributionKind.CITED, cited_ids=[p.id for p in group]
            )
            for i, group in enumerate(groups)
        ]
    )


def _response(particles: list[Particle], **fields: Any) -> QueryResponse:
    return QueryResponse(
        answer=fields.pop("answer", "The answer."),
        particles=particles,
        effective_confidences=fields.pop("effective_confidences", [0.5] * len(particles)),
        **fields,
    )


async def _run(resp: QueryResponse, **kwargs: Any) -> tuple[dict[str, Any], QueryRequest]:
    from particles.mcp.tools.query import query as mcp_query

    captured: list[QueryRequest] = []

    async def _fake(session: Any, request: QueryRequest) -> QueryResponse:
        captured.append(request)
        return resp.model_copy(deep=True)

    kwargs.setdefault("question", "What is it?")
    with patch("particles.operations.query.query", new=_fake):
        out = await mcp_query(**kwargs)
    return out, captured[0]


@pytest.mark.asyncio
class TestCompactShape:
    async def test_a_compact_hit_carries_only_what_is_set(self, db_session: Any) -> None:
        bare = _particle("A bare claim.", sources=0)
        rich = _particle("A rich claim.", sources=3, valid_from=datetime(2026, 9, 20, tzinfo=UTC))
        rich.contested = None
        badge = ContestedBadge(bases=["inconsistency"], inconsistency_id="inc-1")
        resp = _response(
            [rich, bare],
            effective_confidences=[0.9, 0.4],
            sources=[
                HitSource(
                    source_type="WEB_PAGE",
                    uri="https://example.org/a",
                    published=datetime(2026, 8, 23, tzinfo=UTC),
                ),
                None,
            ],
            contested=[badge, None],
            truncation_warning="More hits sit near the cut.",
        )

        out, _ = await _run(resp, grounded=False)

        assert out["particles"] == [
            {
                "id": rich.id,
                "content": "A rich claim.",
                "effective_confidence": 0.9,
                "source": "WEB_PAGE https://example.org/a 2026-08-23 (+2 more)",
                "valid_from": "2026-09-20T00:00:00+00:00",
                "contested": "inc-1",
                "contested_bases": ["inconsistency"],
            },
            {"id": bare.id, "content": "A bare claim.", "effective_confidence": 0.4},
        ]
        # The disclosures that are set stay; the lists the hits now carry, and
        # every null, empty, zero or false key, are gone.
        assert out["truncation_warning"] == "More hits sit near the cut."
        for gone in ("effective_confidences", "sources", "contested", "content_published_ats"):
            assert gone not in out
        assert "answer_refused" not in out and "withheld_candidates" not in out
        assert "sources_unavailable" not in out

    async def test_a_structural_listing_adds_predicate_and_object(self, db_session: Any) -> None:
        claim = StructuredClaim(
            subject=ClaimTerm(kind=TermKind.TOKEN, value="user"),
            predicate=ClaimTerm(kind=TermKind.TOKEN, value="lives_in"),
            object=ClaimTerm(kind=TermKind.TOKEN, value="Porto"),
            structurizer_id="test",
            structurizer_version="1",
        )
        p = _particle("The user lives in Porto.", structured_claim=claim)

        out, request = await _run(
            _response([p], sources=[None]), question=None, predicate="lives_in"
        )

        assert request.grounded is None
        assert out["particles"][0]["predicate"] == "lives_in"
        assert out["particles"][0]["object"] == "Porto"

    async def test_no_sources_from_the_engine_is_disclosed(self, db_session: Any) -> None:
        p = _particle("A claim.")

        out, _ = await _run(_response([p]))  # an engine that predates ``sources``

        assert out["sources_unavailable"] is True
        assert "source" not in out["particles"][0]

    async def test_coverage_gaps_are_a_count_in_compact_and_ids_elsewhere(
        self, db_session: Any
    ) -> None:
        # the store-wide gap list is not scoped to the question, so
        # compact carries its size only; full and summary keep the ids.
        p = _particle("A claim.")
        gaps = [str(uuid.uuid4()) for _ in range(300)]
        resp = _response([p], sources=[None], coverage_gaps=gaps)

        compact, _ = await _run(resp, grounded=False)
        full, _ = await _run(resp, detail="full")
        summary, _ = await _run(resp, detail="summary")

        assert compact["coverage_gaps_count"] == 300
        assert "coverage_gaps" not in compact
        assert full["coverage_gaps"] == gaps and "coverage_gaps_count" not in full
        assert summary["coverage_gaps"] == gaps and "coverage_gaps_count" not in summary

    async def test_no_coverage_gaps_leaves_no_count(self, db_session: Any) -> None:
        out, _ = await _run(_response([_particle("A claim.")], sources=[None]), grounded=False)

        assert "coverage_gaps_count" not in out and "coverage_gaps" not in out


@pytest.mark.asyncio
class TestCitedList:
    async def test_lists_the_cited_hits_in_first_citation_order(self, db_session: Any) -> None:
        a, b, c, d = (_particle(f"Claim {x}.") for x in "abcd")
        resp = _response(
            [a, b, c, d],
            answer="First [p-x]. Then [p-y].",
            answer_attribution=_cited([c], [a, c]),
            sources=[None] * 4,
        )

        out, request = await _run(resp)

        assert request.grounded is True  # the cited list asks for citations
        assert [h["id"] for h in out["particles"]] == [c.id, a.id]
        assert out["hits_omitted"] == 2
        assert "answer_attribution" not in out

    async def test_strips_the_citation_tags_from_the_answer(self, db_session: Any) -> None:
        p = _particle("Claim.")
        resp = _response(
            [p],
            answer="The user lives in Porto [p-0e2c943a]. Probably for work [inference].",
            answer_attribution=_cited([p]),
            sources=[None],
        )

        out, _ = await _run(resp)

        assert out["answer"] == "The user lives in Porto. Probably for work."
        full, _ = await _run(resp, detail="full")
        assert "[p-0e2c943a]" in full["answer"] and full["answer_attribution"] is not None

    async def test_the_cited_list_is_capped_at_hits(self, db_session: Any) -> None:
        ps = [_particle(f"Claim {i}.") for i in range(5)]
        resp = _response(ps, answer_attribution=_cited(ps[::-1]), sources=[None] * 5)

        out, _ = await _run(resp, hits=2)

        assert [h["id"] for h in out["particles"]] == [ps[4].id, ps[3].id]
        assert out["hits_omitted"] == 3

    @pytest.mark.parametrize(
        "attribution",
        [
            None,  # a failed answer, grounded=False, or an engine that sent none
            AnswerAttribution(
                sentences=[AttributedSentence(text="All mine.", kind=AttributionKind.INFERENCE)]
            ),
        ],
    )
    async def test_an_answer_citing_nothing_falls_back_to_the_top_hits(
        self, db_session: Any, attribution: AnswerAttribution | None
    ) -> None:
        ps = [_particle(f"Claim {i}.") for i in range(12)]
        resp = _response(ps, answer_attribution=attribution, sources=[None] * 12)

        out, _ = await _run(resp)

        assert [h["id"] for h in out["particles"]] == [p.id for p in ps[:10]]
        assert out["hits_omitted"] == 2

    async def test_the_ranked_list_asks_for_no_grounding(self, db_session: Any) -> None:
        get_config().mcp.query_list = "ranked"
        ps = [_particle(f"Claim {i}.") for i in range(3)]
        resp = _response(ps, answer_attribution=_cited([ps[2]]), sources=[None] * 3)

        out, request = await _run(resp)

        assert request.grounded is None
        assert [h["id"] for h in out["particles"]] == [p.id for p in ps]

    async def test_a_withheld_refusal_lists_nothing_and_omits_nothing(
        self, db_session: Any
    ) -> None:
        resp = _response([], answer_refused=True, withheld_candidates=7)

        out, _ = await _run(resp)

        assert out["particles"] == []
        assert out["withheld_candidates"] == 7 and out["answer_refused"] is True
        assert "hits_omitted" not in out

    async def test_a_refusal_with_its_candidates_gets_the_fallback(self, db_session: Any) -> None:
        ps = [_particle(f"Claim {i}.") for i in range(3)]
        resp = _response(ps, answer_refused=True, sources=[None] * 3)

        out, request = await _run(resp, include_refused_candidates=True)

        assert request.include_refused_candidates is True
        assert [h["id"] for h in out["particles"]] == [p.id for p in ps]


@pytest.mark.asyncio
class TestCountAndShapes:
    async def test_hits_zero_returns_the_answer_alone(self, db_session: Any) -> None:
        ps = [_particle(f"Claim {i}.") for i in range(3)]

        out, _ = await _run(_response(ps, sources=[None] * 3), hits=0)

        assert out["particles"] == []
        assert out["hits_omitted"] == 3

    async def test_full_lists_every_hit_and_matches_the_response(self, db_session: Any) -> None:
        ps = [_particle(f"Claim {i}.") for i in range(12)]
        resp = _response(ps, sources=[None] * 12)

        out, _ = await _run(resp, detail="full")

        assert out == resp.model_dump(mode="json")
        assert "hits_omitted" not in out

    async def test_full_with_hits_cuts_every_parallel_list(self, db_session: Any) -> None:
        ps = [_particle(f"Claim {i}.") for i in range(4)]
        resp = _response(ps, effective_confidences=[0.4, 0.3, 0.2, 0.1], sources=[None] * 4)

        out, _ = await _run(resp, detail="full", hits=2)

        assert [p["id"] for p in out["particles"]] == [ps[0].id, ps[1].id]
        assert out["effective_confidences"] == [0.4, 0.3]
        assert out["sources"] == [None, None]
        assert out["hits_omitted"] == 2

    async def test_summary_is_an_alias_both_ways(self, db_session: Any) -> None:
        ps = [_particle(f"Claim {i}.") for i in range(12)]
        resp = _response(ps, sources=[None] * 12)

        slim, _ = await _run(resp, summary=True)
        named, _ = await _run(resp, detail="summary")
        full, _ = await _run(resp, summary=False)

        assert slim == named
        assert len(slim["particles"]) == 12 and "confidence" in slim["particles"][0]
        assert full == resp.model_dump(mode="json")

    async def test_top_k_defaults_to_the_configured_depth(self, db_session: Any) -> None:
        get_config().query.default_top_k = 25

        _, request = await _run(_response([]))
        _, explicit = await _run(_response([]), top_k=7)

        assert (request.top_k, explicit.top_k) == (25, 7)

    async def test_an_unknown_detail_or_negative_hits_is_refused(self, db_session: Any) -> None:
        with pytest.raises(ValueError, match="Unknown detail"):
            await _run(_response([]), detail="tiny")
        with pytest.raises(ValueError, match="hits must be 0 or more"):
            await _run(_response([]), hits=-1)


@pytest.mark.asyncio
class TestRemoteEngine:
    """``sources`` crosses ``HttpBackend``, and an older engine sends none."""

    @pytest.mark.parametrize("engine_sends_sources", [True, False])
    async def test_sources_through_the_http_backend(self, engine_sends_sources: bool) -> None:
        from unittest.mock import AsyncMock

        from particles.api.client.http import HttpBackend
        from particles.mcp.tools.query import query as mcp_query

        p = _particle("A claim.")
        body = _response(
            [p], sources=[HitSource(source_type="CONVERSATION", uri="urn:session:1")]
        ).model_dump(mode="json")
        if not engine_sends_sources:
            del body["sources"]
        get_config().engine.base_url = "http://engine.test"
        backend = HttpBackend()
        with (
            patch.object(backend, "_post", new=AsyncMock(return_value=body)),
            patch.object(backend, "inconsistency_backrefs", new=AsyncMock(return_value={})),
            patch("particles.api.client.get_backend", return_value=backend),
        ):
            out = await mcp_query(question="What is it?", grounded=False)

        hit = out["particles"][0]
        if engine_sends_sources:
            assert hit["source"] == "CONVERSATION urn:session:1"
            assert "sources_unavailable" not in out
        else:
            assert "source" not in hit
            assert out["sources_unavailable"] is True

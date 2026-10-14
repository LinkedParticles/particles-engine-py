# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for operations/query/window.py: the window route (§4).

Covers the ADR's acceptance item 5 at the Engine: the semantic path's window
prefilter (an earlier event is not a candidate; a state in force is), the
deterministic listing (date order, the readings, zero LLM and embedding
calls), the store loader's new query shape, a retired state admitted for the
period it held and closed at its successor's start (the earliest-of rule),
the restatement stand-in, the as-of composition, the structural-filter
intersection, the disclosed counts, and the CLI / MCP parameter surfaces.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    QueryRequest,
    TimeWindow,
    UncertaintyNature,
    ValidKind,
    WindowReading,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.core.validity import (
    VALID_FROM_BASIS_KEY,
    ValidFromBasis,
    window_end_of_day,
    window_for_day,
)

EMB = (np.ones(4, dtype=np.float32) / 2.0).tolist()


def D(y: int, m: int, d: int) -> datetime:  # noqa: N802
    return datetime(y, m, d, tzinfo=UTC)


WEEK = TimeWindow(start=D(2026, 10, 10), end=window_end_of_day(D(2026, 10, 16)))


def _particle(
    content: str,
    *,
    valid_from: datetime | None = None,
    kind: ValidKind | None = None,
    valid_until: datetime | None = None,
    basis: str | None = ValidFromBasis.PROSE,
    status: Status = Status.ACTIVE,
    status_reason: StatusReason | None = None,
    supersedes: str | None = None,
    confidence: float = 0.9,
) -> Particle:
    props: dict[str, object] | None = None
    if valid_from is not None and basis is not None:
        props = {VALID_FROM_BASIS_KEY: basis}
    return Particle(
        content=content,
        confidence=Confidence(
            value=confidence, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test-agent",
        status=status,
        status_reason=status_reason,
        valid_from=valid_from,
        valid_kind=kind,
        valid_until=valid_until,
        supersedes=supersedes,
        properties=props,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1", snapshot_id="s1")
        ],
    )


def _mock_embeddings() -> MagicMock:
    model = MagicMock()
    model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
    return model


async def _run(session: Any, monkeypatch: Any, request: QueryRequest) -> Any:
    import particles.operations.query.main as qmain
    from particles import embeddings as ep

    original = ep._embedding_model
    ep.set_embedding_model(_mock_embeddings())
    monkeypatch.setattr(qmain, "_generate_response", AsyncMock(return_value="answer"))
    try:
        return await qmain.query(session, request)
    finally:
        ep.set_embedding_model(original)


async def _seed_week(session: Any) -> dict[str, Particle]:
    """The trip-vs-move shape: a residence, a move, a week's events, a distractor."""
    from particles.store.particle_store import insert_particle

    rows = {
        "keys": _particle(
            "Picked up the keys to the Pali Hill flat on 2026-10-10.",
            valid_from=D(2026, 10, 10),
            kind=ValidKind.EVENT,
        ),
        "office": _particle(
            "First day at the Kamala Mills office on 2026-10-12.",
            valid_from=D(2026, 10, 12),
            kind=ValidKind.EVENT,
        ),
        "earlier": _particle(
            "Walked Lodhi Garden on 2026-10-02.", valid_from=D(2026, 10, 2), kind=ValidKind.EVENT
        ),
        "flat": _particle(
            "The user lives in Pali Hill since 2026-10-10.",
            valid_from=D(2026, 10, 10),
            kind=ValidKind.STATE,
        ),
        "veg": _particle(
            "The user has been vegetarian since March 2024.",
            valid_from=D(2024, 3, 1),
            kind=ValidKind.STATE,
        ),
        "undated": _particle("The user likes thali."),
        "unkinded": _particle("Something on 2026-10-14.", valid_from=D(2026, 10, 14), kind=None),
    }
    for p in rows.values():
        await insert_particle(session, p, EMB)
    await session.commit()
    return rows


# ---------------------------------------------------------------------------
# The semantic path: the window prefilters before ranking.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSemanticWindow:
    async def test_window_prefilters_candidates(self, db_session: Any, monkeypatch: Any) -> None:
        rows = await _seed_week(db_session)
        result = await _run(
            db_session, monkeypatch, QueryRequest(question="what happened?", window=WEEK, top_k=40)
        )
        ids = {p.id for p in result.particles}
        assert rows["keys"].id in ids and rows["office"].id in ids
        assert rows["flat"].id in ids and rows["veg"].id in ids  # in force
        assert rows["unkinded"].id in ids  # BEGAN only
        assert rows["earlier"].id not in ids  # an earlier event is not in force
        assert rows["undated"].id not in ids
        hits = {p.id: h for p, h in zip(result.particles, result.window_hits, strict=True)}
        assert hits[rows["keys"].id] is not None
        assert hits[rows["keys"].id].reading is WindowReading.BEGAN
        assert hits[rows["veg"].id].reading is WindowReading.IN_FORCE
        assert hits[rows["flat"].id].reading is WindowReading.BEGAN
        assert result.window == WEEK
        cov = result.window_coverage
        assert cov is not None
        assert cov.undated == 1 and cov.unkinded == 1 and cov.retired_admitted == 0

    async def test_no_window_is_unchanged(self, db_session: Any, monkeypatch: Any) -> None:
        rows = await _seed_week(db_session)
        result = await _run(db_session, monkeypatch, QueryRequest(question="q?", top_k=40))
        assert len(result.particles) == len(rows)
        assert result.window is None and result.window_hits == []
        assert result.window_coverage is None

    async def test_point_form(self, db_session: Any, monkeypatch: Any) -> None:
        rows = await _seed_week(db_session)
        result = await _run(
            db_session,
            monkeypatch,
            QueryRequest(question="where?", window=window_for_day(D(2026, 10, 12)), top_k=40),
        )
        ids = {p.id for p in result.particles}
        assert ids == {rows["office"].id, rows["flat"].id, rows["veg"].id}


# ---------------------------------------------------------------------------
# The listing: deterministic, date-ordered, no model.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestListing:
    async def test_listing_orders_by_valid_from(self, db_session: Any) -> None:
        from particles import embeddings as ep
        from particles.llm import set_client
        from particles.operations.query.main import query

        rows = await _seed_week(db_session)
        embed = MagicMock()
        llm = MagicMock()
        original = ep._embedding_model
        ep.set_embedding_model(embed)
        set_client(llm)
        try:
            result = await query(db_session, QueryRequest(window=WEEK))
        finally:
            ep.set_embedding_model(original)
            set_client(None)
        embed.encode.assert_not_called()
        llm.messages.create.assert_not_called()
        order = [p.id for p in result.particles]
        assert order[0] == rows["veg"].id  # in force since 2024
        assert order[1:] == [
            rows["keys"].id,
            rows["flat"].id,
            rows["office"].id,
            rows["unkinded"].id,
        ]
        assert "began in it" in result.answer
        readings = [h.reading for h in result.window_hits if h is not None]
        assert readings[0] is WindowReading.IN_FORCE


def test_request_validation() -> None:
    req = QueryRequest(window=WEEK)
    assert req.is_window_listing and req.is_structural_mode
    with pytest.raises(ValueError, match="predicate or vocabulary"):
        QueryRequest(window=WEEK, list_predicates=True)
    with pytest.raises(ValueError, match="needs a question"):
        QueryRequest()


# ---------------------------------------------------------------------------
# Retired claims: admitted for the period they held, closed, stood in for.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRetired:
    async def test_retired_state_closes_at_successor_start(
        self, db_session: Any, monkeypatch: Any
    ) -> None:
        from particles.store.particle_store import insert_particle, update_particle_status

        delhi = _particle(
            "The user lives in Lajpat Nagar, Delhi since 2026-09-01.",
            valid_from=D(2026, 9, 1),
            kind=ValidKind.STATE,
        )
        await insert_particle(db_session, delhi, EMB)
        await update_particle_status(
            db_session, delhi.id, Status.PROVENANCE_STALE, StatusReason.SUPERSEDED_BY_UPDATE
        )
        # The successor's anchor is an as-of one (the move was said on the
        # 9th, its start not stated): it still bounds the predecessor's end.
        mumbai = _particle(
            "The user lives in Mumbai, as of 2026-10-09.",
            valid_from=D(2026, 10, 9),
            kind=ValidKind.STATE,
            basis=ValidFromBasis.AS_OF,
            supersedes=delhi.id,
        )
        await insert_particle(db_session, mumbai, EMB)
        await db_session.commit()

        # 2026-10-01: the Delhi residence held, retired since, closed by the move.
        on_oct_1 = await _run(
            db_session,
            monkeypatch,
            QueryRequest(question="where?", window=window_for_day(D(2026, 10, 1))),
        )
        assert [p.id for p in on_oct_1.particles] == [delhi.id]
        hit = on_oct_1.window_hits[0]
        assert hit is not None
        assert hit.reading is WindowReading.IN_FORCE
        assert hit.closed_at == D(2026, 10, 9) and hit.close_basis == "successor"
        assert hit.retired is not None and hit.retired.status is Status.PROVENANCE_STALE
        assert hit.retired.successor is not None and hit.retired.successor.id == mumbai.id
        assert on_oct_1.window_coverage is not None
        assert on_oct_1.window_coverage.retired_admitted == 1

        # The week after the move: the retired residence is NOT in force.
        week = await _run(db_session, monkeypatch, QueryRequest(question="where?", window=WEEK))
        assert [p.id for p in week.particles] == [mumbai.id]

        # No window: a retired claim never surfaces.
        now = await _run(db_session, monkeypatch, QueryRequest(question="where?"))
        assert [p.id for p in now.particles] == [mumbai.id]

    async def test_retired_closes_at_retired_at_without_dated_successor(
        self, db_session: Any, monkeypatch: Any
    ) -> None:
        from particles.store.particle_store import insert_particle, update_particle_status

        trip = _particle(
            "The user is in Mumbai for a week, as of 2026-10-08.",
            valid_from=D(2026, 10, 8),
            kind=ValidKind.STATE,
            basis=ValidFromBasis.AS_OF,
        )
        await insert_particle(db_session, trip, EMB)
        await update_particle_status(
            db_session, trip.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
        )
        await db_session.commit()
        on = await _run(
            db_session,
            monkeypatch,
            QueryRequest(question="where?", window=window_for_day(D(2026, 10, 8))),
        )
        assert [p.id for p in on.particles] == [trip.id]
        hit = on.window_hits[0]
        assert hit is not None and hit.reading is WindowReading.HELD_AT
        assert hit.close_basis == "retired_at" and hit.closed_at is not None

    async def test_restatement_stands_in_for_original(
        self, db_session: Any, monkeypatch: Any
    ) -> None:
        from particles.store.particle_store import insert_particle, update_particle_status

        original = _particle(
            "The user flies back to Delhi on 2026-10-15.",
            valid_from=D(2026, 10, 15),
            kind=ValidKind.EVENT,
        )
        await insert_particle(db_session, original, EMB)
        await update_particle_status(
            db_session, original.id, Status.SUPERSEDED, StatusReason.SUPERSEDED_BY_REANCHOR
        )
        restated = _particle(
            "When the user was on a work trip, they planned to fly back on 2026-10-15.",
            valid_from=D(2026, 10, 15),
            kind=ValidKind.EVENT,
            supersedes=original.id,
        )
        await insert_particle(db_session, restated, EMB)
        await db_session.commit()
        result = await _run(db_session, monkeypatch, QueryRequest(question="flight?", window=WEEK))
        assert [p.id for p in result.particles] == [restated.id]
        hit = result.window_hits[0]
        assert hit is not None and hit.restates == original.id
        cov = result.window_coverage
        assert cov is not None and cov.restated == 1 and cov.retired_admitted == 0

    async def test_born_retired_never_admitted(self, db_session: Any, monkeypatch: Any) -> None:
        from particles.store.particle_store import insert_particle

        record = _particle(
            "INCONSISTENCY: a vs b on 2026-10-12.",
            valid_from=D(2026, 10, 12),
            kind=ValidKind.EVENT,
            status=Status.INCONSISTENCY,
        )
        await insert_particle(db_session, record, EMB)
        await db_session.commit()
        result = await _run(db_session, monkeypatch, QueryRequest(question="q?", window=WEEK))
        assert result.particles == []


# ---------------------------------------------------------------------------
# Composition with as-of and the structural filters; the loader.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestComposition:
    async def test_as_of_filters_what_the_window_selected(
        self, db_session: Any, monkeypatch: Any
    ) -> None:
        from particles.store.particle_store import insert_particle

        # Both claims are about the first week of October; the store learned
        # one of them only on the 9th. As of the 5th, the window holds one.
        early_oct = TimeWindow(start=D(2026, 10, 1), end=window_end_of_day(D(2026, 10, 7)))
        late = _particle("Learned late.", valid_from=D(2026, 10, 3), kind=ValidKind.EVENT)
        late = late.model_copy(update={"asserted_at": D(2026, 10, 9)})
        early = _particle("Learned early.", valid_from=D(2026, 10, 2), kind=ValidKind.EVENT)
        early = early.model_copy(update={"asserted_at": D(2026, 10, 2)})
        for p in (late, early):
            await insert_particle(db_session, p, EMB)
        await db_session.commit()
        result = await _run(
            db_session,
            monkeypatch,
            QueryRequest(question="q?", window=early_oct, as_of=D(2026, 10, 5)),
        )
        assert [p.id for p in result.particles] == [early.id]

    async def test_structural_filter_intersects(self, db_session: Any) -> None:
        from particles.core.schema import ClaimTerm, StructuredClaim, TermKind
        from particles.operations.query.main import query
        from particles.store.particle_store import insert_particle

        def claim(obj: str) -> StructuredClaim:
            return StructuredClaim(
                subject=ClaimTerm(kind=TermKind.TOKEN, value="user"),
                predicate=ClaimTerm(kind=TermKind.TOKEN, value="visited"),
                object=ClaimTerm(kind=TermKind.LITERAL, value=obj),
                structurizer_id="t",
                structurizer_version="1",
            )

        inside = _particle(
            "Visited Gauri's on 2026-10-11.", valid_from=D(2026, 10, 11), kind=ValidKind.EVENT
        ).model_copy(update={"structured_claim": claim("Gauri's")})
        outside = _particle(
            "Visited Sandeep's on 2026-10-02.", valid_from=D(2026, 10, 2), kind=ValidKind.EVENT
        ).model_copy(update={"structured_claim": claim("Sandeep's")})
        for p in (inside, outside):
            await insert_particle(db_session, p, EMB)
        await db_session.commit()
        result = await query(db_session, QueryRequest(predicate="visited", window=WEEK))
        assert [p.id for p in result.particles] == [inside.id]
        assert result.window_coverage is not None
        assert result.window_hits and result.window_hits[0] is not None

    async def test_loader_shape(self, db_session: Any) -> None:
        from particles.store.particle_store import (
            count_active_undated,
            get_particles_in_window_scope,
            insert_particle,
            load_valid_from_by_ids,
        )

        rows = await _seed_week(db_session)
        no_emb = _particle(
            "Dated, no vector, on 2026-10-13.", valid_from=D(2026, 10, 13), kind=ValidKind.EVENT
        )
        await insert_particle(db_session, no_emb)
        await db_session.commit()
        everything = await get_particles_in_window_scope(db_session)
        assert {p.id for p, _, _ in everything} == {
            p.id for p in rows.values() if p.valid_from is not None
        } | {no_emb.id}
        with_emb = await get_particles_in_window_scope(db_session, require_embedding=True)
        assert no_emb.id not in {p.id for p, _, _ in with_emb}
        assert all(emb is not None for _, emb, _ in with_emb)
        assert await count_active_undated(db_session) == 1
        starts = await load_valid_from_by_ids(db_session, [rows["keys"].id, rows["undated"].id])
        assert starts[rows["keys"].id] == (D(2026, 10, 10), ValidFromBasis.PROSE)
        assert starts[rows["undated"].id] == (None, None)


# ---------------------------------------------------------------------------
# Surfaces.
# ---------------------------------------------------------------------------


class TestCliWindow:
    def test_during_needs_dots(self) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        result = CliRunner().invoke(app, ["query", "q?", "--during", "2026-10-10"])
        assert result.exit_code == 1
        assert "START..END" in result.output

    def test_during_and_on_exclusive(self) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        result = CliRunner().invoke(
            app, ["query", "q?", "--during", "2026-10-10..", "--on", "2026-10-10"]
        )
        assert result.exit_code == 1
        assert "alternatives" in result.output

    def test_during_reversed_rejected(self) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        result = CliRunner().invoke(app, ["query", "q?", "--during", "2026-10-16..2026-10-10"])
        assert result.exit_code == 1
        assert "start must not be after" in result.output

    def test_invalid_on_rejected(self) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        result = CliRunner().invoke(app, ["query", "q?", "--on", "nope"])
        assert result.exit_code == 1
        assert "Invalid --on" in result.output


@pytest.mark.asyncio
class TestMcpWindow:
    async def test_during_parsed_into_request(self, db_session: Any) -> None:
        from particles.core.schema import QueryResponse
        from particles.mcp.tools.query import query as mcp_query

        captured: list[QueryRequest] = []

        async def _fake(session: Any, request: QueryRequest) -> QueryResponse:
            captured.append(request)
            return QueryResponse(answer="ok", particles=[], effective_confidences=[])

        with patch("particles.operations.query.query", new=_fake):
            out = await mcp_query("q?", during_start="2026-10-10", during_end="2026-10-16")
            await mcp_query(on="2026-10-12")
        assert captured[0].window is not None
        assert captured[0].window.start == D(2026, 10, 10)
        assert captured[0].window.end == window_end_of_day(D(2026, 10, 16))
        assert captured[1].window is not None and captured[1].window.is_point
        assert captured[1].question is None
        assert "window_hits" in out and "window_coverage" in out

    async def test_on_and_during_exclusive(self) -> None:
        from particles.mcp.tools.query import query as mcp_query

        with pytest.raises(ValueError, match="alternatives"):
            await mcp_query("q?", on="2026-10-12", during_start="2026-10-10")

    async def test_invalid_window_raises(self) -> None:
        from particles.mcp.tools.query import query as mcp_query

        with pytest.raises(ValueError, match="Invalid window"):
            await mcp_query("q?", during_start="not-a-date")

# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The memory-server façade's provenance and the MCP rate cap (security review 2026-10-24-2).

The façade recognises its records by its own provenance as well as the tag, so
a native agent's tag cannot hide or annotate a Subject in the façade's view
(P50, beside N30). The rate cap charges one token per paid-capable
§6.6 assertion, not per tool call, and covers ``graph_view`` (P26, D30).
Adopted from the review's proof tests.
"""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from particles.config import get_config
from particles.db import DEFAULT_STORE
from particles.mcp.memory_compat import ops


@pytest.fixture
def facade() -> Any:
    from particles import embeddings as ep

    def encode(texts: Any, **_kwargs: Any) -> Any:
        items = [texts] if isinstance(texts, str) else list(texts)
        rows = []
        for text in items:
            digest = hashlib.sha256(str(text).encode()).digest()[:16]
            rows.append([(b / 127.5) - 1.0 for b in digest])
        return np.array(rows, dtype=np.float32)

    model = MagicMock()
    model.encode = MagicMock(side_effect=encode)
    original = ep._embedding_model
    ep.set_embedding_model(model)
    get_config().mcp.write.enabled_stores = [DEFAULT_STORE]
    try:
        yield get_config().mcp.memory_compat
    finally:
        ep.set_embedding_model(original)


async def _foreign(session: Any) -> Any:
    from particles.core.schema import Subject
    from particles.store.subject_store import insert_subject

    subject = Subject(canonical_name="Pluto", asserted_by="operator", subject_class="dwarf_planet")
    await insert_subject(session, subject)
    await session.commit()
    return subject


def _names(graph: dict[str, Any]) -> list[str]:
    return sorted(e["name"] for e in graph["entities"])


@pytest.mark.asyncio
async def test_n30_the_facade_cannot_tombstone_a_foreign_subject(
    db_session: Any, facade: Any
) -> None:
    """N30 as aimed (fixed): delete_entities refuses a Subject the facade did not create."""
    await _foreign(db_session)
    with pytest.raises(ValueError, match="can be changed only by an operator: Pluto"):
        await ops.delete_entities(db_session, store=DEFAULT_STORE, entity_names=["Pluto"])
    graph, _ = await ops.read_graph(db_session)
    assert _names(graph) == ["Pluto"]


@pytest.mark.asyncio
async def test_a_native_agent_tag_cannot_tombstone_a_foreign_subject_in_the_facade_view(
    db_session: Any, facade: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P50: the façade does not recognise a tombstone by particle tag alone.

    Safe property: a Subject the facade did not create stays visible to facade
    reads unless the facade's own principal tombstoned it; a belief another
    principal tagged with the facade's tombstone tag does not hide it.
    """
    import particles.ingest.subject_resolver as sr
    from particles.extraction.mcp_memory import TOMBSTONE_TAG
    from particles.mcp.tools.write import particle_assert

    pluto = await _foreign(db_session)
    # The native resolver resolves the name to the existing local Subject.
    monkeypatch.setattr(sr, "resolve_subjects", AsyncMock(return_value=[pluto.id]))
    res = await particle_assert(
        "Pluto is no longer tracked here.",
        ["Pluto"],
        0.5,
        source_excerpt="drop pluto",
        tags=[TOMBSTONE_TAG],
    )
    assert res["verdict"] == "ASSERTED"
    graph, _ = await ops.read_graph(db_session)
    assert _names(graph) == ["Pluto"], "a native agent's tag hid an operator Subject"


@pytest.mark.asyncio
async def test_the_rate_cap_bounds_paid_capable_assertions_not_tool_calls(
    db_session: Any, facade: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D30 residual: one admitted facade call fans out into one §6.6 run per observation.

    Safe property: with ``mcp.rate_limit_per_minute`` at 1, the facade runs at
    most one paid-capable §6.6 assertion in the minute.
    """
    import mcp.types as types

    from particles.mcp.memory_compat import server

    get_config().mcp.rate_limit_per_minute = 1
    calls = 0
    real = ops._assert

    async def counting(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return await real(*args, **kwargs)

    monkeypatch.setattr(ops, "_assert", counting)
    srv = server.build_server()
    call = srv.request_handlers[types.CallToolRequest]
    entities = [
        {"name": f"E{i}", "entityType": "thing", "observations": [f"fact number {i}"]}
        for i in range(5)
    ]
    result = await call(
        types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(
                name="create_entities", arguments={"entities": entities}
            ),
        )
    )
    assert calls <= 1, f"{calls} §6.6 assertions ran under one token of a 1-per-minute cap"
    # The charge is taken before anything runs: the over-cap call is refused
    # whole, so it neither probes nor writes a partial batch.
    assert result.root.isError
    assert "runs 5 assertions" in result.root.content[0].text
    graph, _ = await ops.read_graph(db_session)
    assert graph["entities"] == []


@pytest.mark.asyncio
async def test_a_facade_write_spends_one_token_per_assertion(db_session: Any, facade: Any) -> None:
    """P26: a batch within the cap lands, and spends its assertions from the bucket."""
    import mcp.types as types

    from particles.mcp.memory_compat import server

    get_config().mcp.rate_limit_per_minute = 3
    srv = server.build_server()
    call = srv.request_handlers[types.CallToolRequest]

    def _create(name: str, observations: list[str]) -> Any:
        return call(
            types.CallToolRequest(
                method="tools/call",
                params=types.CallToolRequestParams(
                    name="create_entities",
                    arguments={
                        "entities": [
                            {"name": name, "entityType": "thing", "observations": observations}
                        ]
                    },
                ),
            )
        )

    first = await _create("Alpha", ["one", "two"])
    assert not first.root.isError
    second = await _create("Beta", ["three", "four"])
    assert second.root.isError
    assert "Rate limit reached" in second.root.content[0].text


def test_assertion_cost_counts_what_each_tool_can_assert() -> None:
    from particles.mcp.memory_compat.server import _assertion_cost

    entities = [{"name": "A", "observations": ["x", "y"]}, {"name": "B"}, "junk"]
    assert _assertion_cost("create_entities", {"entities": entities}) == 2
    observations = [{"entityName": "A", "contents": ["x", "y", "z"]}]
    assert _assertion_cost("add_observations", {"observations": observations}) == 3
    assert _assertion_cost("create_relations", {"relations": [{}, {}]}) == 2
    assert _assertion_cost("delete_entities", {"entityNames": ["A", "B"]}) == 2
    assert _assertion_cost("delete_observations", {"deletions": [{}, {}]}) == 1
    assert _assertion_cost("create_relations", {"relations": "not a list"}) == 0


@pytest.mark.asyncio
async def test_graph_view_is_admitted_by_the_rate_cap() -> None:
    """D30: ``graph_view`` spends a token, like the engine's rate-limited /graph route."""
    from unittest.mock import patch

    from particles.mcp.tools.graph import graph_view

    get_config().mcp.rate_limit_per_minute = 1
    backend = MagicMock()
    backend.graph = AsyncMock(side_effect=RuntimeError("reached the backend"))
    with patch("particles.api.client.get_backend", return_value=backend):
        with pytest.raises(RuntimeError):
            await graph_view(query="anything")
        with pytest.raises(ValueError, match="Rate limit reached"):
            await graph_view(query="anything else")


@pytest.mark.asyncio
async def test_d30_a_second_particle_assert_in_the_minute_is_refused(db_session: Any) -> None:
    """D30 as aimed (fixed for the native write tools): the per-process bucket refuses."""
    from unittest.mock import patch

    from particles.mcp.tools.write import particle_assert

    get_config().mcp.write.enabled_stores = [DEFAULT_STORE]
    get_config().mcp.rate_limit_per_minute = 1
    backend = MagicMock()
    backend.particle_assert = AsyncMock(side_effect=RuntimeError("reached the backend"))
    with patch("particles.api.client.get_backend", return_value=backend):
        with pytest.raises(RuntimeError):
            await particle_assert("X.", ["X"], 0.5, source_excerpt="x")
        with pytest.raises(ValueError, match="Rate limit reached"):
            await particle_assert("Y.", ["Y"], 0.5, source_excerpt="y")


@pytest.mark.asyncio
async def test_a_native_agent_tag_cannot_add_an_observation_in_the_facade_view(
    db_session: Any, facade: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P50: an observation-tagged native belief is not read back as a façade observation."""
    import particles.ingest.subject_resolver as sr
    from particles.extraction.mcp_memory import OBSERVATION_TAG
    from particles.mcp.tools.write import particle_assert

    pluto = await _foreign(db_session)
    monkeypatch.setattr(sr, "resolve_subjects", AsyncMock(return_value=[pluto.id]))
    res = await particle_assert(
        "Pluto is a planet again.",
        ["Pluto"],
        0.5,
        source_excerpt="pluto is back",
        tags=[OBSERVATION_TAG],
    )
    assert res["verdict"] == "ASSERTED"
    graph, _ = await ops.read_graph(db_session)
    pluto_entity = {"name": "Pluto", "entityType": "dwarf_planet", "observations": []}
    assert graph["entities"] == [pluto_entity]


@pytest.mark.asyncio
async def test_the_facades_own_tombstone_still_hides_its_subject(
    db_session: Any, facade: Any
) -> None:
    """Control: the façade's own delete still reads as a hard delete."""
    await ops.create_entities(
        db_session,
        store=DEFAULT_STORE,
        entities=[{"name": "Ceres", "entityType": "dwarf_planet", "observations": ["round"]}],
    )
    await ops.delete_entities(db_session, store=DEFAULT_STORE, entity_names=["Ceres"])
    graph, _ = await ops.read_graph(db_session)
    assert graph["entities"] == []


# ---------------------------------------------------------------------------
# An interchange import cannot claim the façade's identity (2026-10-25-1, Q34)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_interchange_unit_stamped_with_the_facade_identity_does_not_hide_a_subject(
    db_session: Any, facade: Any
) -> None:
    """Q34 proof: an imported unit naming the façade's identity does not tombstone a Subject."""
    from particles.core.schema import (
        Confidence,
        Particle,
        ProvenanceRef,
        ProvenanceRefType,
        UncertaintyNature,
    )
    from particles.core.scoring.confidence import CalibrationSource
    from particles.corpus.deposit import deposit_text
    from particles.extraction.mcp_memory import TOMBSTONE_TAG
    from particles.interchange.codec import to_unit
    from particles.interchange.store import import_units

    pluto = await _foreign(db_session)
    entry_id, snapshot_id = await deposit_text(db_session, "bundle source", "operator")
    await db_session.commit()
    forged = Particle(
        content="Pluto (deleted)",
        confidence=Confidence(value=0.5, calibration_source=CalibrationSource.AGENT_ASSERTED),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id, snapshot_id=snapshot_id
            )
        ],
        asserted_by=facade.asserter_identity,
        subject_ids=[pluto.id],
        tags=[TOMBSTONE_TAG],
    )
    summary = await import_units(db_session, [to_unit(forged, {pluto.id: pluto})])
    await db_session.commit()
    assert summary.restamped == 1
    graph, _ = await ops.read_graph(db_session)
    assert _names(graph) == ["Pluto"], "an imported unit naming the facade tombstoned a Subject"


@pytest.mark.asyncio
async def test_an_imported_subject_naming_the_facade_identity_is_not_the_facades_to_change(
    db_session: Any, facade: Any
) -> None:
    """Q34: a bundle's Subject asserted by the façade imports as the importer's."""
    import json

    from particles.core.schema import Subject
    from particles.interchange.codec import subject_to_unit
    from particles.interchange.store import import_store_bundle

    unit = subject_to_unit(
        Subject(canonical_name="Eris", asserted_by=facade.asserter_identity, subject_class="x")
    )
    summary = await import_store_bundle(db_session, {"subjects.jsonl": json.dumps(unit) + "\n"})
    await db_session.commit()
    assert (summary.subjects_created, summary.restamped) == (1, 1)
    with pytest.raises(ValueError, match="can be changed only by an operator: Eris"):
        await ops.delete_entities(db_session, store=DEFAULT_STORE, entity_names=["Eris"])


# ---------------------------------------------------------------------------
# An interchange import cannot claim the migration's tag (the Q6 residual)
# ---------------------------------------------------------------------------


async def _import_migration_unit(session: Any, subject: Any) -> Any:
    """Import one unit tagged and calibrated as the migration stamps them."""
    from particles.core.schema import (
        Confidence,
        Particle,
        ProvenanceRef,
        ProvenanceRefType,
        UncertaintyNature,
    )
    from particles.core.scoring.confidence import CalibrationSource
    from particles.corpus.deposit import deposit_text
    from particles.extraction.mcp_memory import IMPORT_TAG, OBSERVATION_TAG
    from particles.interchange.codec import to_unit
    from particles.interchange.store import import_units

    entry_id, snapshot_id = await deposit_text(session, "bundle source", "operator")
    await session.commit()
    forged = Particle(
        content="Pluto is the ninth planet.",
        confidence=Confidence(value=0.5, calibration_source=CalibrationSource.IMPORTED),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id, snapshot_id=snapshot_id
            )
        ],
        asserted_by="someone-else",
        subject_ids=[subject.id],
        tags=[IMPORT_TAG, OBSERVATION_TAG, "kept"],
    )
    summary = await import_units(session, [to_unit(forged, {subject.id: subject})])
    await session.commit()
    return summary


@pytest.mark.asyncio
async def test_an_imported_unit_tagged_by_the_migration_is_not_facade_memory(
    db_session: Any, facade: Any
) -> None:
    """The interchange variant of Q6: a bundle cannot inject an observation the façade reads."""
    pluto = await _foreign(db_session)
    summary = await _import_migration_unit(db_session, pluto)
    assert (summary.imported, summary.retagged) == (1, 1)
    graph, _ = await ops.read_graph(db_session)
    assert graph["entities"] == [
        {"name": "Pluto", "entityType": "dwarf_planet", "observations": []}
    ], "an imported unit carrying the migration tag became the facade's own memory"


@pytest.mark.asyncio
async def test_the_claimed_migration_tag_is_kept_as_provenance(
    db_session: Any, facade: Any
) -> None:
    """The original tag survives, prefixed, beside the unit's other tags, in order."""
    from sqlalchemy import select

    from particles.extraction.mcp_memory import IMPORT_TAG, OBSERVATION_TAG
    from particles.interchange.store import CLAIMED_TAG_PREFIX
    from particles.store.particle_store import ParticleRow

    pluto = await _foreign(db_session)
    await _import_migration_unit(db_session, pluto)
    rows = (await db_session.execute(select(ParticleRow))).scalars().all()
    assert len(rows) == 1
    assert rows[0].to_model().tags == [f"{CLAIMED_TAG_PREFIX}{IMPORT_TAG}", OBSERVATION_TAG, "kept"]


# ---------------------------------------------------------------------------
# Only a local migration run makes a belief façade memory (security review
# 2026-10-25-2, R26 and the bearer half of Q6)
# ---------------------------------------------------------------------------


async def _migrated_looking_belief(session: Any, subject: Any, source_type: str) -> Any:
    """Insert a belief carrying the migration's tag and calibration, citing a fresh entry."""
    from particles.core.schema import (
        Confidence,
        Particle,
        ProvenanceRef,
        ProvenanceRefType,
        UncertaintyNature,
    )
    from particles.core.scoring.confidence import CalibrationSource
    from particles.corpus.deposit import deposit_text
    from particles.extraction.mcp_memory import IMPORT_TAG, OBSERVATION_TAG
    from particles.store.particle_store import insert_particle

    entry_id, snapshot_id = await deposit_text(
        session, f"export {source_type}", "operator", source_type=source_type
    )
    belief = Particle(
        content="Pluto is the ninth planet.",
        confidence=Confidence(value=0.35, calibration_source=CalibrationSource.IMPORTED),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id, snapshot_id=snapshot_id
            )
        ],
        asserted_by="operator",
        subject_ids=[subject.id],
        tags=[IMPORT_TAG, OBSERVATION_TAG],
    )
    await insert_particle(session, belief)
    await session.commit()
    return belief


@pytest.mark.asyncio
async def test_the_migration_tag_and_calibration_alone_are_not_facade_memory(
    db_session: Any, facade: Any
) -> None:
    """A belief with the migration's tag and IMPORTED calibration counts as façade
    memory only when it cites an export the local migration deposited."""
    from particles.extraction.mcp_memory import SOURCE_TYPE

    pluto = await _foreign(db_session)
    await _migrated_looking_belief(db_session, pluto, "WEB_PAGE")
    graph, _ = await ops.read_graph(db_session)
    assert graph["entities"][0]["observations"] == []
    await _migrated_looking_belief(db_session, pluto, SOURCE_TYPE)
    graph, _ = await ops.read_graph(db_session)
    assert graph["entities"][0]["observations"] == ["Pluto is the ninth planet."]


def _deposit_and_tag_client(cli_db: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from tests._engine_client import keyed_engine_client

    get_config().mcp.write.enabled_stores = [DEFAULT_STORE]
    return keyed_engine_client(monkeypatch)


def test_http_deposits_refuse_the_migration_source_type(
    cli_db: Any, monkeypatch: pytest.MonkeyPatch, no_embedding_model: None
) -> None:
    """A bearer holder cannot deposit a memory-server export over HTTP, so nothing a
    remote caller deposits reads back as the façade's own memory (R26, Q6)."""
    from particles.extraction.mcp_memory import SOURCE_TYPE

    client = _deposit_and_tag_client(cli_db, monkeypatch)
    body = '{"type": "entity", "name": "Pluto", "entityType": "x", "observations": ["o"]}'
    text = client.post(
        "/corpus/deposit/text",
        json={"text": body, "source_type": SOURCE_TYPE, "deposited_by": "api-user"},
    )
    assert text.status_code == 400, text.text
    assert "import mcp-memory" in text.json()["detail"]
    upload = client.post(
        "/corpus/deposit/file",
        files={"file": ("memory.jsonl", body.encode(), "application/jsonl")},
        data={"source_type": SOURCE_TYPE},
    )
    assert upload.status_code == 400, upload.text
    url = client.post(
        "/corpus/deposit/url",
        json={"url": "https://example.com/memory.jsonl", "source_type": SOURCE_TYPE},
    )
    assert url.status_code == 400, url.text
    assert client.get("/corpus").json() == []


def test_the_operator_tag_route_refuses_the_migration_tag(
    cli_db: Any, monkeypatch: pytest.MonkeyPatch, facade: Any
) -> None:
    """The operator tag route cannot add the migration tag, so it cannot turn an
    imported belief into the façade's own memory (R26, Y3)."""
    import asyncio

    from particles.extraction.mcp_memory import IMPORT_TAG

    async def _seed() -> str:
        from particles.db import get_engine, session_scope

        try:
            # A plain session: the import commits its own writes under the
            # writer lock (security review 2026-10-31-1, AD15).
            async with session_scope() as s:
                pluto = await _foreign(s)
                await _import_migration_unit(s, pluto)
            async with session_scope() as s:
                from particles.store.particle_store import get_active_particles

                return (await get_active_particles(s))[0].id
        finally:
            await get_engine().dispose()

    pid = asyncio.run(_seed())
    client = _deposit_and_tag_client(cli_db, monkeypatch)
    resp = client.post(f"/particles/{pid}/tags", json={"tags": [IMPORT_TAG]})
    assert resp.status_code == 400, resp.text
    assert "migration" in resp.json()["detail"]
    assert client.post(f"/particles/{pid}/tags", json={"tags": ["fine"]}).status_code == 200


@pytest.mark.asyncio
async def test_no_tag_verb_adds_the_migration_tag(db_session: Any, facade: Any) -> None:
    """The store's one tag function refuses it, so the agent verb does too."""
    from particles.extraction.mcp_memory import IMPORT_TAG
    from particles.operations.agent_write import tag_belief
    from particles.store.particle_store import get_active_particles
    from particles.store.taxonomy_store import ReservedTagError, add_particle_tags

    pluto = await _foreign(db_session)
    await _import_migration_unit(db_session, pluto)
    pid = (await get_active_particles(db_session))[0].id
    with pytest.raises(ReservedTagError):
        await add_particle_tags(db_session, pid, ["ok", IMPORT_TAG])
    with pytest.raises(ValueError):
        await tag_belief(db_session, particle_id=pid, tags=[IMPORT_TAG])
    graph, _ = await ops.read_graph(db_session)
    assert graph["entities"][0]["observations"] == []


def test_the_cli_tag_verb_refuses_the_migration_tag(cli_db: Any) -> None:
    from typer.testing import CliRunner

    from particles.api.cli import app
    from particles.extraction.mcp_memory import IMPORT_TAG

    res = CliRunner().invoke(app, ["particle", "tag", "abcd", "--tag", IMPORT_TAG, "--force"])
    assert res.exit_code == 1
    assert "only `particles import mcp-memory` stamps that tag" in res.output


# ---------------------------------------------------------------------------
# The façade does not write a local store while an engine is configured
# (security review 2026-10-25-2, M2 / R44)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_facade_refuses_while_an_engine_is_configured(
    db_session: Any, facade: Any
) -> None:
    import mcp.types as types
    from pydantic import AnyUrl

    from particles.mcp.memory_compat import server
    from particles.store.subject_store import find_by_name

    get_config().engine.base_url = "http://127.0.0.1:9"  # nothing listens; nothing is sent
    srv = server.build_server()
    call = srv.request_handlers[types.CallToolRequest]
    for name, arguments in (
        (
            "create_entities",
            {"entities": [{"name": "Local_Only", "entityType": "t", "observations": ["o"]}]},
        ),
        ("read_graph", {}),
    ):
        res = await call(
            types.CallToolRequest(
                method="tools/call",
                params=types.CallToolRequestParams(name=name, arguments=arguments),
            )
        )
        assert res.root.isError, name
        assert "engine.base_url" in res.root.content[0].text
    assert await find_by_name(db_session, "Local_Only") is None
    read = srv.request_handlers[types.ReadResourceRequest]
    with pytest.raises(ValueError, match="engine.base_url"):
        await read(
            types.ReadResourceRequest(
                method="resources/read",
                params=types.ReadResourceRequestParams(uri=AnyUrl("memory://knowledge-graph")),
            )
        )

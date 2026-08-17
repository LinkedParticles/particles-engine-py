"""Fidelity suite for the reference memory-server façade.

This is the parity tripwire. It mirrors the reference's own test suite
(``src/memory/__tests__/knowledge-graph.test.ts``, 42 cases) case-for-case,
using the reference's own fixture payloads — ``Alice`` / ``Bob`` / ``Charlie`` /
``Acme Corp``, and the README's ``John_Smith`` → ``works_at`` → ``Anthropic``.

    Fixture payloads reproduced from
    https://github.com/modelcontextprotocol/servers, ``src/memory``
    (``@modelcontextprotocol/server-memory`` v0.6.3), MIT licensed.
    Copyright (c) Model Context Protocol, a Series of LF Projects, LLC.

Beyond behaviour, it asserts **response-shape** compatibility: the exact
``structuredContent`` keys per tool, the ``delete_*`` bare-sentence text, the
tool annotations, and the resource contract. A client that worked against the
reference must work here unmodified, and a regression must fail the build
rather than reach a user's agent.

Keyless and offline: the façade resolves subjects bare-locally (never the live
authority ladder), so only the embedding model needs a stub.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from particles.config import get_config
from particles.core.status import Status
from particles.db import DEFAULT_STORE
from particles.mcp.memory_compat import ops, server
from particles.store.particle_store import get_particles_by_ids

# -- reference fixtures (MIT, see module docstring) ----------------------

ALICE = {"name": "Alice", "entityType": "person", "observations": ["works at Acme Corp"]}
BOB = {"name": "Bob", "entityType": "person", "observations": ["likes programming"]}
CHARLIE = {"name": "Charlie", "entityType": "person", "observations": []}
ACME = {"name": "Acme Corp", "entityType": "company", "observations": ["tech company"]}
ALICE_KNOWS_BOB = {"from": "Alice", "to": "Bob", "relationType": "knows"}


def _bare(name: str, entity_type: str = "person") -> dict[str, Any]:
    return {"name": name, "entityType": entity_type, "observations": []}


# -- fixtures -----------------------------------------------------------


@pytest.fixture
def distinct_embeddings() -> Any:
    """Content-derived vectors so unrelated observations never collide.

    A constant vector (the pattern in ``test_mcp_write.py``) would make every
    pair cosine-1.0 and trip the §6.6 conflict ladder on ordinary reference
    input, which is not what these tests are measuring.
    """
    from particles import embeddings as ep

    def encode(texts: Any, **_kwargs: Any) -> Any:
        items = [texts] if isinstance(texts, str) else list(texts)
        rows = []
        for text in items:
            digest = hashlib.sha256(str(text).encode()).digest()[:16]
            # Centred on zero, not [0, 1]: all-positive vectors are mutually
            # similar (cosine ≈ 0.75), which would trip the §6.6 conflict
            # ladder on unrelated observations and mask real behaviour.
            rows.append([(b / 127.5) - 1.0 for b in digest])
        return np.array(rows, dtype=np.float32)

    model = MagicMock()
    model.encode = MagicMock(side_effect=encode)
    original = ep._embedding_model
    ep.set_embedding_model(model)
    try:
        yield
    finally:
        ep.set_embedding_model(original)


@pytest.fixture
def facade(distinct_embeddings: Any) -> Any:
    """Write-enable the shared test store and reset the façade knobs to defaults."""
    get_config().mcp.write.enabled_stores = [DEFAULT_STORE]
    cfg = get_config().mcp.memory_compat
    cfg.read_graph_max_entities = 250
    cfg.read_graph_max_observations_per_entity = 25
    cfg.search_max_entities = 100
    return cfg


# -- helpers ------------------------------------------------------------


async def _create(session: Any, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return await ops.create_entities(session, store=DEFAULT_STORE, entities=entities)


async def _relate(session: Any, relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return await ops.create_relations(session, store=DEFAULT_STORE, relations=relations)


async def _graph(session: Any) -> dict[str, Any]:
    graph, _notes = await ops.read_graph(session)
    return graph


def _names(graph: dict[str, Any]) -> list[str]:
    return sorted(e["name"] for e in graph["entities"])


def _entity(graph: dict[str, Any], name: str) -> dict[str, Any]:
    return next(e for e in graph["entities"] if e["name"] == name)


# -- createEntities -----------------------------------------------------


class TestCreateEntities:
    async def test_creates_new_entities(self, db_session: Any, facade: Any) -> None:
        created = await _create(db_session, [ALICE, BOB])
        assert created == [ALICE, BOB]
        graph = await _graph(db_session)
        assert _names(graph) == ["Alice", "Bob"]
        assert _entity(graph, "Alice")["observations"] == ["works at Acme Corp"]
        assert _entity(graph, "Alice")["entityType"] == "person"

    async def test_does_not_create_duplicate_entities(self, db_session: Any, facade: Any) -> None:
        await _create(db_session, [ALICE])
        created = await _create(db_session, [ALICE, BOB])
        # Only the genuinely-new entity comes back, per the reference.
        assert created == [BOB]
        assert _names(await _graph(db_session)) == ["Alice", "Bob"]

    async def test_handles_empty_entity_arrays(self, db_session: Any, facade: Any) -> None:
        assert await _create(db_session, []) == []

    async def test_preserves_name_verbatim(self, db_session: Any, facade: Any) -> None:
        """The whole swap rests on a name coming back exactly as it went in.

        Guards the §3 decision to resolve bare-locally: the authority ladder
        would happily rewrite ``the society of mind`` to ``Society of Mind``.
        """
        await _create(db_session, [_bare("the society of mind", "book")])
        assert _names(await _graph(db_session)) == ["the society of mind"]


# -- createRelations ----------------------------------------------------


class TestCreateRelations:
    async def test_creates_new_relations(self, db_session: Any, facade: Any) -> None:
        await _create(db_session, [_bare("Alice"), _bare("Bob")])
        created = await _relate(db_session, [ALICE_KNOWS_BOB])
        assert created == [ALICE_KNOWS_BOB]
        assert (await _graph(db_session))["relations"] == [ALICE_KNOWS_BOB]

    async def test_does_not_create_duplicate_relations(self, db_session: Any, facade: Any) -> None:
        await _create(db_session, [_bare("Alice"), _bare("Bob")])
        await _relate(db_session, [ALICE_KNOWS_BOB])
        assert await _relate(db_session, [ALICE_KNOWS_BOB]) == []
        assert len((await _graph(db_session))["relations"]) == 1

    async def test_handles_empty_relation_arrays(self, db_session: Any, facade: Any) -> None:
        assert await _relate(db_session, []) == []

    async def test_arbitrary_relation_type_round_trips(self, db_session: Any, facade: Any) -> None:
        """Reference relationTypes are free-form; nothing is coerced onto RelationType."""
        weird = {"from": "John_Smith", "to": "Anthropic", "relationType": "works at / for"}
        await _relate(db_session, [weird])
        assert (await _graph(db_session))["relations"] == [weird]

    async def test_direction_is_preserved(self, db_session: Any, facade: Any) -> None:
        """`particle_subjects` has no ordering column — direction rides the tags."""
        await _relate(db_session, [ALICE_KNOWS_BOB])
        relation = (await _graph(db_session))["relations"][0]
        assert relation["from"] == "Alice"
        assert relation["to"] == "Bob"


# -- addObservations ----------------------------------------------------


class TestAddObservations:
    async def test_adds_observations_to_existing_entities(
        self, db_session: Any, facade: Any
    ) -> None:
        await _create(db_session, [ALICE])
        results = await ops.add_observations(
            db_session,
            store=DEFAULT_STORE,
            observations=[{"entityName": "Alice", "contents": ["likes coffee"]}],
        )
        assert results == [{"entityName": "Alice", "addedObservations": ["likes coffee"]}]
        assert _entity(await _graph(db_session), "Alice")["observations"] == [
            "works at Acme Corp",
            "likes coffee",
        ]

    async def test_does_not_add_duplicate_observations(self, db_session: Any, facade: Any) -> None:
        await _create(db_session, [ALICE])
        results = await ops.add_observations(
            db_session,
            store=DEFAULT_STORE,
            observations=[{"entityName": "Alice", "contents": ["works at Acme Corp"]}],
        )
        assert results == [{"entityName": "Alice", "addedObservations": []}]
        assert _entity(await _graph(db_session), "Alice")["observations"] == ["works at Acme Corp"]

    async def test_raises_for_non_existent_entity(self, db_session: Any, facade: Any) -> None:
        with pytest.raises(ValueError, match="Entity with name NonExistent not found"):
            await ops.add_observations(
                db_session,
                store=DEFAULT_STORE,
                observations=[{"entityName": "NonExistent", "contents": ["x"]}],
            )

    async def test_long_observation_is_stored_verbatim(self, db_session: Any, facade: Any) -> None:
        """A reference observation has no length limit; the 320-char agent-write
        granularity ceiling must not reject one, and it is never truncated (§4)."""
        long_text = "Alice " + "really " * 80 + "likes coffee."
        assert len(long_text) > get_config().mcp.write.max_assertion_chars
        await _create(db_session, [_bare("Alice")])
        results = await ops.add_observations(
            db_session,
            store=DEFAULT_STORE,
            observations=[{"entityName": "Alice", "contents": [long_text]}],
        )
        assert results[0]["addedObservations"] == [long_text]
        assert _entity(await _graph(db_session), "Alice")["observations"] == [long_text]


# -- deleteEntities -----------------------------------------------------


class TestDeleteEntities:
    async def test_deletes_entities(self, db_session: Any, facade: Any) -> None:
        await _create(db_session, [_bare("Alice"), _bare("Bob")])
        await ops.delete_entities(db_session, store=DEFAULT_STORE, entity_names=["Alice"])
        assert _names(await _graph(db_session)) == ["Bob"]

    async def test_cascade_deletes_relations(self, db_session: Any, facade: Any) -> None:
        await _create(db_session, [_bare("Alice"), _bare("Bob"), _bare("Charlie")])
        await _relate(
            db_session,
            [ALICE_KNOWS_BOB, {"from": "Bob", "to": "Charlie", "relationType": "knows"}],
        )
        await ops.delete_entities(db_session, store=DEFAULT_STORE, entity_names=["Alice"])
        graph = await _graph(db_session)
        assert graph["relations"] == [{"from": "Bob", "to": "Charlie", "relationType": "knows"}]

    async def test_handles_deleting_non_existent_entities(
        self, db_session: Any, facade: Any
    ) -> None:
        await _create(db_session, [_bare("Alice")])
        await ops.delete_entities(db_session, store=DEFAULT_STORE, entity_names=["Nobody"])
        assert _names(await _graph(db_session)) == ["Alice"]

    async def test_entity_without_observations_disappears(
        self, db_session: Any, facade: Any
    ) -> None:
        """The tombstone path: nothing to retract, yet the entity must vanish."""
        await _create(db_session, [CHARLIE])
        await ops.delete_entities(db_session, store=DEFAULT_STORE, entity_names=["Charlie"])
        assert _names(await _graph(db_session)) == []

    async def test_delete_then_recreate_revives_the_entity(
        self, db_session: Any, facade: Any
    ) -> None:
        await _create(db_session, [CHARLIE])
        await ops.delete_entities(db_session, store=DEFAULT_STORE, entity_names=["Charlie"])
        created = await _create(db_session, [CHARLIE])
        assert created == [CHARLIE]
        assert _names(await _graph(db_session)) == ["Charlie"]

    async def test_delete_retracts_rather_than_destroys(self, db_session: Any, facade: Any) -> None:
        """the observable contract matches, the history survives."""
        await _create(db_session, [ALICE])
        graph_before = await _graph(db_session)
        assert _entity(graph_before, "Alice")["observations"] == ["works at Acme Corp"]

        sub = await ops.load_subgraph(db_session)
        subject = sub.subject_by_name("Alice")
        assert subject is not None
        particle_ids = [p.id for p in sub.observation_particles(subject.id)]

        await ops.delete_entities(db_session, store=DEFAULT_STORE, entity_names=["Alice"])
        assert _names(await _graph(db_session)) == []

        surviving = await get_particles_by_ids(db_session, particle_ids)
        assert surviving, "the particle row must still exist after a façade delete"
        assert all(p.status == Status.RETRACTED for p in surviving.values())
        assert all(p.content == "works at Acme Corp" for p in surviving.values())


# -- deleteObservations -------------------------------------------------


class TestDeleteObservations:
    async def test_deletes_observations(self, db_session: Any, facade: Any) -> None:
        await _create(db_session, [ALICE])
        await ops.delete_observations(
            db_session,
            store=DEFAULT_STORE,
            deletions=[{"entityName": "Alice", "observations": ["works at Acme Corp"]}],
        )
        assert _entity(await _graph(db_session), "Alice")["observations"] == []

    async def test_handles_deleting_from_non_existent_entities(
        self, db_session: Any, facade: Any
    ) -> None:
        await ops.delete_observations(
            db_session,
            store=DEFAULT_STORE,
            deletions=[{"entityName": "Ghost", "observations": ["nope"]}],
        )

    async def test_handles_deleting_absent_observation(self, db_session: Any, facade: Any) -> None:
        await _create(db_session, [ALICE])
        await ops.delete_observations(
            db_session,
            store=DEFAULT_STORE,
            deletions=[{"entityName": "Alice", "observations": ["never said this"]}],
        )
        assert _entity(await _graph(db_session), "Alice")["observations"] == ["works at Acme Corp"]


# -- deleteRelations ----------------------------------------------------


class TestDeleteRelations:
    async def test_deletes_specific_relations(self, db_session: Any, facade: Any) -> None:
        await _create(db_session, [_bare("Alice"), _bare("Bob")])
        await _relate(
            db_session,
            [ALICE_KNOWS_BOB, {"from": "Alice", "to": "Bob", "relationType": "manages"}],
        )
        await ops.delete_relations(db_session, store=DEFAULT_STORE, relations=[ALICE_KNOWS_BOB])
        assert (await _graph(db_session))["relations"] == [
            {"from": "Alice", "to": "Bob", "relationType": "manages"}
        ]

    async def test_handles_absent_relation(self, db_session: Any, facade: Any) -> None:
        await ops.delete_relations(db_session, store=DEFAULT_STORE, relations=[ALICE_KNOWS_BOB])


# -- readGraph ----------------------------------------------------------


class TestReadGraph:
    async def test_returns_empty_graph_when_store_is_empty(
        self, db_session: Any, facade: Any
    ) -> None:
        assert await _graph(db_session) == {"entities": [], "relations": []}

    async def test_returns_complete_graph(self, db_session: Any, facade: Any) -> None:
        await _create(db_session, [ALICE, BOB, ACME])
        await _relate(db_session, [ALICE_KNOWS_BOB])
        graph = await _graph(db_session)
        assert _names(graph) == ["Acme Corp", "Alice", "Bob"]
        assert graph["relations"] == [ALICE_KNOWS_BOB]

    async def test_entity_cap_truncates_and_discloses(self, db_session: Any, facade: Any) -> None:
        """capped, and never silently so."""
        facade.read_graph_max_entities = 2
        await _create(db_session, [_bare("Alice"), _bare("Bob"), _bare("Charlie")])
        graph, notes = await ops.read_graph(db_session)
        assert len(graph["entities"]) == 2
        assert notes and "truncated" in notes[0]
        assert "read_graph_max_entities" in notes[0]

    async def test_observation_cap_truncates_and_discloses(
        self, db_session: Any, facade: Any
    ) -> None:
        facade.read_graph_max_observations_per_entity = 1
        await _create(db_session, [_bare("Alice")])
        await ops.add_observations(
            db_session,
            store=DEFAULT_STORE,
            observations=[{"entityName": "Alice", "contents": ["one", "two", "three"]}],
        )
        graph, notes = await ops.read_graph(db_session)
        assert _entity(graph, "Alice")["observations"] == ["one"]
        assert notes and "2 observation(s) withheld" in notes[0]


# -- searchNodes --------------------------------------------------------


class TestSearchNodes:
    @pytest.fixture(autouse=True)
    async def _seed(self, db_session: Any, facade: Any) -> None:
        await _create(
            db_session,
            [
                {
                    "name": "Alice",
                    "entityType": "person",
                    "observations": ["works at Acme Corp", "likes programming"],
                },
                {"name": "Bob", "entityType": "person", "observations": ["works at TechCo"]},
                {"name": "Acme Corp", "entityType": "company", "observations": ["tech company"]},
            ],
        )
        await _relate(db_session, [ALICE_KNOWS_BOB])

    async def _search(self, session: Any, query: str) -> dict[str, Any]:
        graph, _notes = await ops.search_nodes(session, query=query)
        return graph

    async def test_searches_by_entity_name(self, db_session: Any) -> None:
        assert _names(await self._search(db_session, "Alice")) == ["Alice"]

    async def test_searches_by_entity_type(self, db_session: Any) -> None:
        assert _names(await self._search(db_session, "company")) == ["Acme Corp"]

    async def test_searches_by_observation_content(self, db_session: Any) -> None:
        assert _names(await self._search(db_session, "programming")) == ["Alice"]

    async def test_is_case_insensitive(self, db_session: Any) -> None:
        assert _names(await self._search(db_session, "ALICE")) == ["Alice"]

    async def test_is_substring_not_semantic(self, db_session: Any) -> None:
        """The reference matches substrings; a synonym must NOT match (§8)."""
        assert _names(await self._search(db_session, "coding")) == []

    async def test_includes_relations_with_one_endpoint_matching(self, db_session: Any) -> None:
        graph = await self._search(db_session, "Alice")
        assert graph["relations"] == [ALICE_KNOWS_BOB]

    async def test_returns_empty_graph_for_no_matches(self, db_session: Any) -> None:
        graph = await self._search(db_session, "zzzz-no-such-thing")
        assert graph == {"entities": [], "relations": []}


# -- openNodes ----------------------------------------------------------


class TestOpenNodes:
    @pytest.fixture(autouse=True)
    async def _seed(self, db_session: Any, facade: Any) -> None:
        await _create(db_session, [_bare("Alice"), _bare("Bob"), _bare("Charlie")])
        await _relate(
            db_session,
            [ALICE_KNOWS_BOB, {"from": "Charlie", "to": "Alice", "relationType": "mentors"}],
        )

    async def _open(self, session: Any, names: list[str]) -> dict[str, Any]:
        graph, _notes = await ops.open_nodes(session, names=names)
        return graph

    async def test_opens_specific_nodes_by_name(self, db_session: Any) -> None:
        assert _names(await self._open(db_session, ["Alice"])) == ["Alice"]

    async def test_includes_outgoing_relation_to_node_outside_the_set(
        self, db_session: Any
    ) -> None:
        graph = await self._open(db_session, ["Alice"])
        assert ALICE_KNOWS_BOB in graph["relations"]

    async def test_includes_incoming_relation_from_node_outside_the_set(
        self, db_session: Any
    ) -> None:
        graph = await self._open(db_session, ["Alice"])
        assert {"from": "Charlie", "to": "Alice", "relationType": "mentors"} in graph["relations"]

    async def test_handles_non_existent_nodes(self, db_session: Any) -> None:
        assert _names(await self._open(db_session, ["Nobody"])) == []

    async def test_handles_empty_node_list(self, db_session: Any) -> None:
        assert await self._open(db_session, []) == {"entities": [], "relations": []}


# -- response-shape parity ----------------------------------------------


class TestResponseEnvelope:
    """The wire contract a reference client actually parses."""

    async def test_read_graph_envelope(self, db_session: Any, facade: Any) -> None:
        await _create(db_session, [ALICE])
        result = await server._dispatch_read(db_session, "read_graph", {})
        payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
        assert set(payload) == {"entities", "relations"}
        assert result.structuredContent == payload

    async def test_truncation_note_is_appended_not_prepended(
        self, db_session: Any, facade: Any
    ) -> None:
        """A client doing JSON.parse(content[0].text) must keep working (§7)."""
        facade.read_graph_max_entities = 1
        await _create(db_session, [_bare("Alice"), _bare("Bob")])
        result = await server._dispatch_read(db_session, "read_graph", {})
        assert len(result.content) == 2
        json.loads(result.content[0].text)  # type: ignore[union-attr]
        assert "truncated" in result.content[1].text  # type: ignore[union-attr]
        assert result.structuredContent is not None
        assert set(result.structuredContent) == {"entities", "relations"}

    async def test_create_entities_envelope(self, db_session: Any, facade: Any) -> None:
        result = await server._dispatch_write(
            db_session, DEFAULT_STORE, "create_entities", {"entities": [ALICE]}
        )
        assert result.structuredContent == {"entities": [ALICE]}
        assert json.loads(result.content[0].text) == {"entities": [ALICE]}  # type: ignore[union-attr]

    async def test_create_relations_envelope(self, db_session: Any, facade: Any) -> None:
        result = await server._dispatch_write(
            db_session, DEFAULT_STORE, "create_relations", {"relations": [ALICE_KNOWS_BOB]}
        )
        assert result.structuredContent == {"relations": [ALICE_KNOWS_BOB]}

    async def test_add_observations_envelope(self, db_session: Any, facade: Any) -> None:
        await _create(db_session, [_bare("Alice")])
        result = await server._dispatch_write(
            db_session,
            DEFAULT_STORE,
            "add_observations",
            {"observations": [{"entityName": "Alice", "contents": ["hi"]}]},
        )
        assert result.structuredContent == {
            "results": [{"entityName": "Alice", "addedObservations": ["hi"]}]
        }

    @pytest.mark.parametrize(
        ("tool", "arguments", "message"),
        [
            ("delete_entities", {"entityNames": []}, "Entities deleted successfully"),
            ("delete_observations", {"deletions": []}, "Observations deleted successfully"),
            ("delete_relations", {"relations": []}, "Relations deleted successfully"),
        ],
    )
    async def test_delete_envelope_is_a_bare_sentence(
        self, db_session: Any, facade: Any, tool: str, arguments: dict[str, Any], message: str
    ) -> None:
        """The reference's text content here is a sentence, NOT the JSON."""
        result = await server._dispatch_write(db_session, DEFAULT_STORE, tool, arguments)
        assert result.content[0].text == message  # type: ignore[union-attr]
        assert result.structuredContent == {"success": True, "message": message}


class TestToolSurface:
    """The nine tools, their order, annotations, and schemas."""

    def test_tool_names_and_order_match_the_reference(self) -> None:
        assert [t["name"] for t in server.tool_surface()] == [
            "create_entities",
            "create_relations",
            "add_observations",
            "delete_entities",
            "delete_observations",
            "delete_relations",
            "read_graph",
            "search_nodes",
            "open_nodes",
        ]

    def test_every_tool_declares_an_output_schema_and_title(self) -> None:
        for tool in server.tool_surface():
            assert tool["title"], tool["name"]
            assert tool["outputSchema"], tool["name"]

    @pytest.mark.parametrize(
        ("name", "read_only", "destructive", "idempotent"),
        [
            ("create_entities", False, False, False),
            ("create_relations", False, False, False),
            ("add_observations", False, False, False),
            ("delete_entities", False, True, True),
            ("delete_observations", False, True, True),
            ("delete_relations", False, True, True),
            ("read_graph", True, False, True),
            ("search_nodes", True, False, True),
            ("open_nodes", True, False, True),
        ],
    )
    def test_annotations_match_the_reference(
        self, name: str, read_only: bool, destructive: bool, idempotent: bool
    ) -> None:
        tool = next(t for t in server.tool_surface() if t["name"] == name)
        annotations = tool["annotations"]
        assert isinstance(annotations, dict)
        assert annotations["readOnlyHint"] is read_only
        assert annotations["destructiveHint"] is destructive
        assert annotations["idempotentHint"] is idempotent
        assert annotations["openWorldHint"] is False

    def test_deviations_are_disclosed_in_tool_descriptions(self) -> None:
        """A reading agent must learn the deviations from the tool list itself."""
        by_name = {t["name"]: str(t["description"]) for t in server.tool_surface()}
        for name in ("delete_entities", "delete_observations", "delete_relations"):
            assert "retracts" in by_name[name]
        for name in ("read_graph", "search_nodes", "open_nodes"):
            assert "capped" in by_name[name]

    def test_entity_and_relation_schemas_match_the_reference_fields(self) -> None:
        tools = {t["name"]: t for t in server.tool_surface()}
        entity = tools["create_entities"]["inputSchema"]["properties"]["entities"]["items"]  # type: ignore[index]
        assert entity["required"] == ["name", "entityType", "observations"]
        relation = tools["create_relations"]["inputSchema"]["properties"]["relations"]["items"]  # type: ignore[index]
        assert relation["required"] == ["from", "to", "relationType"]


class TestToolSchemaGolden:
    """Pins the whole surface byte-for-byte.

    The behavioural tests above check what the façade *does*; this checks the
    contract a client *sees* — every schema, title, description, and
    annotation. Regenerate deliberately with::

        uv run particles memory tools > tests/mcp/memory-tool-schema.json
    """

    def test_surface_matches_the_golden(self) -> None:
        from pathlib import Path

        golden_path = Path(__file__).parent / "mcp" / "memory-tool-schema.json"
        golden = json.loads(golden_path.read_text())
        assert server.tool_surface() == golden, (
            "The memory-compat tool surface changed. A reference client sees this "
            "contract — confirm the change is intended, then regenerate the golden "
            "with `uv run particles memory tools > tests/mcp/memory-tool-schema.json`."
        )


class TestWriteGating:
    """default-deny, and the parity-driven deviation from it (§9)."""

    async def test_all_nine_tools_listed_even_when_writes_are_disabled(self) -> None:
        get_config().mcp.write.enabled_stores = []
        assert len(server.tool_surface()) == 9

    async def test_write_refuses_with_an_actionable_message(self) -> None:
        get_config().mcp.write.enabled_stores = []
        with pytest.raises(ValueError, match="mcp.write.enabled_stores"):
            server._resolve_write_store(None)

    async def test_unknown_store_is_refused(self, facade: Any) -> None:
        with pytest.raises(ValueError, match="not write-enabled"):
            server._resolve_write_store("no-such-store")


class TestProvenance:
    """The reason to switch: an observation is a claim with a source."""

    async def test_observation_carries_deposit_provenance_and_facade_identity(
        self, db_session: Any, facade: Any
    ) -> None:
        await _create(db_session, [ALICE])
        sub = await ops.load_subgraph(db_session)
        subject = sub.subject_by_name("Alice")
        assert subject is not None
        particle = sub.observation_particles(subject.id)[0]

        assert particle.asserted_by == get_config().mcp.memory_compat.asserter_identity
        assert particle.provenance, "a façade observation must carry provenance"
        assert particle.provenance[0].corpus_entry_id

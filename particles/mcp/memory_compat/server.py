"""The drop-in memory-server MCP surface.

Mirrors ``@modelcontextprotocol/server-memory`` v0.6.3 closely enough that a
client which worked against the reference works here unmodified: same nine tool
names, same input schemas, same output schemas, same annotations, same
``memory://knowledge-graph`` resource with subscriptions, and the same response
envelope — including the detail that the three ``delete_*`` tools return a bare
sentence as their text content but a ``{success, message}`` object as their
structured content.

This module uses the **low-level** ``mcp.server.lowlevel.Server`` rather than
FastMCP (which the native ``particles`` server uses), for two reasons that are
not stylistic: FastMCP has no resource-subscription support at all, so it
cannot implement the capability the reference has shipped since 2026-06-17;
and returning a ``CallToolResult`` directly is the only way to control the
envelope byte-for-byte (§2).

Three behaviours differ from the reference and are disclosed in the tool
descriptions the client actually reads, not only in the ADR: deletes retract
rather than destroy, ``read_graph`` is capped, and the write tools refuse with
an actionable message on a store that is not write-enabled.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import mcp.types as types
from mcp.server.lowlevel import Server
from pydantic import AnyUrl

from particles.db import DEFAULT_STORE
from particles.mcp.memory_compat import ops

log = logging.getLogger(__name__)

SERVER_NAME = "memory-server"
RESOURCE_URI = "memory://knowledge-graph"

_ENTITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "The name of the entity"},
        "entityType": {"type": "string", "description": "The type of the entity"},
        "observations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "An array of observation contents associated with the entity",
        },
    },
    "required": ["name", "entityType", "observations"],
}

_RELATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "from": {
            "type": "string",
            "description": "The name of the entity where the relation starts",
        },
        "to": {"type": "string", "description": "The name of the entity where the relation ends"},
        "relationType": {"type": "string", "description": "The type of the relation"},
    },
    "required": ["from", "to", "relationType"],
}

_GRAPH_OUTPUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {"type": "array", "items": _ENTITY_SCHEMA},
        "relations": {"type": "array", "items": _RELATION_SCHEMA},
    },
    "required": ["entities", "relations"],
}

_DELETE_OUTPUT: dict[str, Any] = {
    "type": "object",
    "properties": {"success": {"type": "boolean"}, "message": {"type": "string"}},
    "required": ["success", "message"],
}

# Disclosed deviations, appended to the reference's own tool descriptions so a
# reading agent learns them from the tool list rather than from the ADR.
_RETRACT_NOTE = (
    " Backed by a Particles store: this retracts with an audit trail rather than "
    "hard-deleting, so the item stops appearing in reads while its history survives."
)
_CAP_NOTE = (
    " Backed by a Particles store: results are capped for context safety and any "
    "truncation is disclosed in an extra text block."
)

_WRITE_TOOLS = frozenset(
    {
        "create_entities",
        "create_relations",
        "add_observations",
        "delete_entities",
        "delete_observations",
        "delete_relations",
    }
)


def _tools() -> list[types.Tool]:
    """The nine tools, in the reference's own registration order."""
    return [
        types.Tool(
            name="create_entities",
            title="Create Entities",
            description="Create multiple new entities in the knowledge graph",
            inputSchema={
                "type": "object",
                "properties": {"entities": {"type": "array", "items": _ENTITY_SCHEMA}},
                "required": ["entities"],
            },
            outputSchema={
                "type": "object",
                "properties": {"entities": {"type": "array", "items": _ENTITY_SCHEMA}},
                "required": ["entities"],
            },
            annotations=types.ToolAnnotations(
                readOnlyHint=False,
                destructiveHint=False,
                idempotentHint=False,
                openWorldHint=False,
            ),
        ),
        types.Tool(
            name="create_relations",
            title="Create Relations",
            description=(
                "Create multiple new relations between entities in the knowledge graph. "
                "Relations should be in active voice"
            ),
            inputSchema={
                "type": "object",
                "properties": {"relations": {"type": "array", "items": _RELATION_SCHEMA}},
                "required": ["relations"],
            },
            outputSchema={
                "type": "object",
                "properties": {"relations": {"type": "array", "items": _RELATION_SCHEMA}},
                "required": ["relations"],
            },
            annotations=types.ToolAnnotations(
                readOnlyHint=False,
                destructiveHint=False,
                idempotentHint=False,
                openWorldHint=False,
            ),
        ),
        types.Tool(
            name="add_observations",
            title="Add Observations",
            description="Add new observations to existing entities in the knowledge graph",
            inputSchema={
                "type": "object",
                "properties": {
                    "observations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entityName": {
                                    "type": "string",
                                    "description": (
                                        "The name of the entity to add the observations to"
                                    ),
                                },
                                "contents": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "An array of observation contents to add",
                                },
                            },
                            "required": ["entityName", "contents"],
                        },
                    }
                },
                "required": ["observations"],
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entityName": {"type": "string"},
                                "addedObservations": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["entityName", "addedObservations"],
                        },
                    }
                },
                "required": ["results"],
            },
            annotations=types.ToolAnnotations(
                readOnlyHint=False,
                destructiveHint=False,
                idempotentHint=False,
                openWorldHint=False,
            ),
        ),
        types.Tool(
            name="delete_entities",
            title="Delete Entities",
            description=(
                "Delete multiple entities and their associated relations from the "
                "knowledge graph" + _RETRACT_NOTE
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "entityNames": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "An array of entity names to delete",
                    }
                },
                "required": ["entityNames"],
            },
            outputSchema=_DELETE_OUTPUT,
            annotations=types.ToolAnnotations(
                readOnlyHint=False,
                destructiveHint=True,
                idempotentHint=True,
                openWorldHint=False,
            ),
        ),
        types.Tool(
            name="delete_observations",
            title="Delete Observations",
            description=(
                "Delete specific observations from entities in the knowledge graph" + _RETRACT_NOTE
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "deletions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entityName": {
                                    "type": "string",
                                    "description": (
                                        "The name of the entity containing the observations"
                                    ),
                                },
                                "observations": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "An array of observations to delete",
                                },
                            },
                            "required": ["entityName", "observations"],
                        },
                    }
                },
                "required": ["deletions"],
            },
            outputSchema=_DELETE_OUTPUT,
            annotations=types.ToolAnnotations(
                readOnlyHint=False,
                destructiveHint=True,
                idempotentHint=True,
                openWorldHint=False,
            ),
        ),
        types.Tool(
            name="delete_relations",
            title="Delete Relations",
            description="Delete multiple relations from the knowledge graph" + _RETRACT_NOTE,
            inputSchema={
                "type": "object",
                "properties": {
                    "relations": {
                        "type": "array",
                        "items": _RELATION_SCHEMA,
                        "description": "An array of relations to delete",
                    }
                },
                "required": ["relations"],
            },
            outputSchema=_DELETE_OUTPUT,
            annotations=types.ToolAnnotations(
                readOnlyHint=False,
                destructiveHint=True,
                idempotentHint=True,
                openWorldHint=False,
            ),
        ),
        types.Tool(
            name="read_graph",
            title="Read Graph",
            description="Read the entire knowledge graph" + _CAP_NOTE,
            inputSchema={"type": "object", "properties": {}},
            outputSchema=_GRAPH_OUTPUT,
            annotations=types.ToolAnnotations(
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
        ),
        types.Tool(
            name="search_nodes",
            title="Search Nodes",
            description="Search for nodes in the knowledge graph based on a query" + _CAP_NOTE,
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The search query to match against entity names, types, "
                            "and observation content"
                        ),
                    }
                },
                "required": ["query"],
            },
            outputSchema=_GRAPH_OUTPUT,
            annotations=types.ToolAnnotations(
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
        ),
        types.Tool(
            name="open_nodes",
            title="Open Nodes",
            description="Open specific nodes in the knowledge graph by their names" + _CAP_NOTE,
            inputSchema={
                "type": "object",
                "properties": {
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "An array of entity names to retrieve",
                    }
                },
                "required": ["names"],
            },
            outputSchema=_GRAPH_OUTPUT,
            annotations=types.ToolAnnotations(
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
        ),
    ]


def tool_surface() -> list[dict[str, Any]]:
    """The tool surface as plain JSON — the shape pinned by the parity golden.

    Docstrings are not involved (these schemas are declared, not introspected),
    so the output is stable across Python versions without the ``cleandoc``
    normalisation the native ``particles mcp tools`` dumper needs.
    """
    return [
        {
            "name": tool.name,
            "title": tool.title,
            "description": tool.description,
            "inputSchema": tool.inputSchema,
            "outputSchema": tool.outputSchema,
            "annotations": tool.annotations.model_dump(exclude_none=True)
            if tool.annotations
            else None,
        }
        for tool in _tools()
    ]


def _json_result(payload: dict[str, Any], notes: list[str] | None = None) -> types.CallToolResult:
    """The reference envelope: pretty JSON text plus matching structured content.

    Truncation notes are **appended** as a second text block, never prepended —
    a client doing ``JSON.parse(content[0].text)`` must keep working (§7).
    """
    content: list[types.ContentBlock] = [
        types.TextContent(type="text", text=json.dumps(payload, indent=2))
    ]
    for note in notes or ():
        content.append(types.TextContent(type="text", text=note))
        log.warning("%s", note)
    return types.CallToolResult(content=content, structuredContent=payload)


def _delete_result(message: str) -> types.CallToolResult:
    """The reference's delete envelope: a bare sentence as text, an object as structure."""
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=message)],
        structuredContent={"success": True, "message": message},
    )


def _error_result(message: str) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=message)], isError=True
    )


def _resolve_write_store(store: str | None) -> str:
    """Validate the target store against the write allowlist.

    Deliberately raises rather than no-ops: an agent that believes it stored a
    memory and did not is worse off than one told plainly that it could not
    (§9). The message names the config key to change.
    """
    from particles.mcp.tools.write import _resolve_store

    return _resolve_store(store)


def build_server(store: str | None = None) -> Server[Any, Any]:
    """Construct the façade server bound to ``store``.

    Separated from :func:`main` so tests can drive the handlers without
    spawning the stdio transport.
    """
    server: Server[Any, Any] = Server(SERVER_NAME)
    read_store = store or DEFAULT_STORE
    subscribers: set[str] = set()

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        # All nine are always listed, even when the store is not write-enabled:
        # a reference client that discovers only the read tools breaks with
        # "unknown tool" rather than degrading (§9).
        return _tools()

    @server.list_resources()
    async def list_resources() -> list[types.Resource]:
        return [
            types.Resource(
                uri=AnyUrl(RESOURCE_URI),
                name="knowledge-graph",
                title="Knowledge Graph",
                description="The full knowledge graph with all entities and relations",
                mimeType="application/json",
            )
        ]

    @server.read_resource()
    async def read_resource(uri: AnyUrl) -> str:
        from particles.db import session_scope

        async with session_scope(read_store) as session:
            graph, _notes = await ops.read_graph(session)
        return json.dumps(graph, indent=2)

    @server.subscribe_resource()
    async def subscribe_resource(uri: AnyUrl) -> None:
        subscribers.add(str(uri))

    @server.unsubscribe_resource()
    async def unsubscribe_resource(uri: AnyUrl) -> None:
        subscribers.discard(str(uri))

    async def _notify_graph_updated() -> None:
        """Emit resources/updated, but only to a client that asked for it."""
        if RESOURCE_URI not in subscribers:
            return
        try:
            ctx = server.request_context
        except LookupError:  # pragma: no cover - no active request (direct call in tests)
            return
        await ctx.session.send_resource_updated(AnyUrl(RESOURCE_URI))

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        from particles.db import session_scope

        if name in _WRITE_TOOLS:
            try:
                write_store = _resolve_write_store(store)
            except ValueError as exc:
                return _error_result(str(exc))
            async with session_scope(write_store, write=True) as session:
                try:
                    result = await _dispatch_write(session, write_store, name, arguments)
                except ValueError as exc:
                    return _error_result(str(exc))
                await session.commit()
            await _notify_graph_updated()
            return result

        async with session_scope(read_store) as session:
            return await _dispatch_read(session, name, arguments)

    return server


async def _dispatch_write(
    session: Any, store: str, name: str, arguments: dict[str, Any]
) -> types.CallToolResult:
    match name:
        case "create_entities":
            created = await ops.create_entities(
                session, store=store, entities=arguments.get("entities", [])
            )
            return _json_result({"entities": created})
        case "create_relations":
            created_rel = await ops.create_relations(
                session, store=store, relations=arguments.get("relations", [])
            )
            return _json_result({"relations": created_rel})
        case "add_observations":
            results = await ops.add_observations(
                session, store=store, observations=arguments.get("observations", [])
            )
            return _json_result({"results": results})
        case "delete_entities":
            await ops.delete_entities(
                session, store=store, entity_names=arguments.get("entityNames", [])
            )
            return _delete_result(ops.DELETE_ENTITIES_MESSAGE)
        case "delete_observations":
            await ops.delete_observations(
                session, store=store, deletions=arguments.get("deletions", [])
            )
            return _delete_result(ops.DELETE_OBSERVATIONS_MESSAGE)
        case "delete_relations":
            await ops.delete_relations(
                session, store=store, relations=arguments.get("relations", [])
            )
            return _delete_result(ops.DELETE_RELATIONS_MESSAGE)
    raise ValueError(f"Unknown tool: {name}")


async def _dispatch_read(
    session: Any, name: str, arguments: dict[str, Any]
) -> types.CallToolResult:
    match name:
        case "read_graph":
            graph, notes = await ops.read_graph(session)
            return _json_result(graph, notes)
        case "search_nodes":
            graph, notes = await ops.search_nodes(session, query=arguments.get("query", ""))
            return _json_result(graph, notes)
        case "open_nodes":
            graph, notes = await ops.open_nodes(session, names=arguments.get("names", []))
            return _json_result(graph, notes)
    return _error_result(f"Unknown tool: {name}")


def main(store: str | None = None) -> None:
    """Entry point for ``particles memory serve`` — run the façade over stdio."""
    import anyio
    from mcp.server.stdio import stdio_server

    from particles.observability import setup_observability

    setup_observability()
    server = build_server(store)

    async def _run() -> None:
        init_options = server.create_initialization_options()
        # The SDK hardcodes subscribe=False when deriving capabilities; the
        # reference declares {resources: {subscribe: true}} and clients check it
        # before subscribing, so assert it explicitly (§2).
        if init_options.capabilities.resources is not None:
            init_options.capabilities.resources.subscribe = True
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, init_options)

    anyio.run(_run)

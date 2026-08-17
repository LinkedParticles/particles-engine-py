"""mcp sub-Typer — run the read-only MCP server, inspect its tool surface.

``particles mcp serve`` is what an MCP client (Claude Code, Claude
Desktop, etc.) launches as a child process. ``particles mcp tools`` is a
debugging aid that prints the registered tool surface as JSON without
spawning the stdio transport — useful for reviewing the contract or
diffing against a previous version when an ``operations/`` signature
changes.
"""

from __future__ import annotations

import asyncio
import inspect
import json

import typer

from particles.api.cli import app

mcp_app = typer.Typer(
    help="Model Context Protocol server (read-only).",
    no_args_is_help=True,
)
app.add_typer(mcp_app, name="mcp")


@mcp_app.command("serve")
def mcp_serve_cmd() -> None:
    """Run the read-only Particles MCP server over stdio.

    Typical install path on the operator's machine::

        claude mcp add particles -- uv run particles mcp serve
    """
    from particles.mcp import main as serve_main

    serve_main()


@mcp_app.command("tools")
def mcp_tools_cmd(
    output_format: str = typer.Option(
        "json", "--format", help='Output format: "json" (default) or "text".'
    ),
) -> None:
    """Print the registered MCP tool surface (name, description, input schema).

    Used to verify the contract without spawning an MCP client. The JSON
    output is what ``tests/mcp/tool-schema.json`` should match — drift
    here means an ``operations/`` signature changed and the MCP surface
    needs review.
    """
    from particles.mcp import build_server

    async def _dump() -> list[dict[str, object]]:
        server = build_server()
        tools = await server.list_tools()
        # Normalize docstring whitespace so the golden file is stable across
        # Python versions: 3.13+ strips common leading whitespace from
        # docstrings at compile time (gh-81283); 3.11 (the CI floor) does
        # not. ``inspect.cleandoc`` does the same job at runtime, giving us
        # a consistent description regardless of which Python the dumper runs on.
        return [
            {
                "name": t.name,
                "description": inspect.cleandoc(t.description) if t.description else None,
                "inputSchema": t.inputSchema,
            }
            for t in tools
        ]

    surface = asyncio.run(_dump())

    if output_format == "text":
        for entry in surface:
            typer.echo(f"{entry['name']}")
            desc = entry["description"] or ""
            first_line = str(desc).splitlines()[0] if desc else ""
            typer.echo(f"  {first_line}")
        return

    typer.echo(json.dumps(surface, indent=2, sort_keys=True))


@mcp_app.command("resources")
def mcp_resources_cmd(
    output_format: str = typer.Option(
        "json", "--format", help='Output format: "json" (default) or "text".'
    ),
) -> None:
    """Print the registered MCP resource surface (digest).

    The sibling of ``particles mcp tools`` for the *resources* primitive: the
    ``particles://digest/{store}`` template plus any concrete per-store digests
    listed for the write-enabled / opted-in memory stores. The JSON output is
    what ``tests/mcp/resource-schema.json`` should match — drift here means the
    resource contract MCP clients see has changed.
    """
    from particles.mcp import build_server

    async def _dump() -> dict[str, list[dict[str, object]]]:
        server = build_server()
        resources = await server.list_resources()
        templates = await server.list_resource_templates()
        # Normalize docstring whitespace for cross-Python golden stability, the
        # same reason mcp_tools_cmd does (3.13+ strips docstring indentation).
        return {
            "resources": [
                {
                    "name": r.name,
                    "uri": str(r.uri),
                    "description": inspect.cleandoc(r.description) if r.description else None,
                    "mimeType": r.mimeType,
                }
                for r in resources
            ],
            "templates": [
                {
                    "name": t.name,
                    "uriTemplate": t.uriTemplate,
                    "description": inspect.cleandoc(t.description) if t.description else None,
                    "mimeType": t.mimeType,
                }
                for t in templates
            ],
        }

    surface = asyncio.run(_dump())

    if output_format == "text":
        for kind in ("templates", "resources"):
            for entry in surface[kind]:
                ident = entry.get("uriTemplate") or entry.get("uri")
                typer.echo(f"{ident}  [{entry['name']}]")
                desc = str(entry["description"] or "")
                first_line = desc.splitlines()[0] if desc else ""
                typer.echo(f"  {first_line}")
        return

    typer.echo(json.dumps(surface, indent=2, sort_keys=True))

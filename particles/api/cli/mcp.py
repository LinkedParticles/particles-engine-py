# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""mcp sub-Typer — run the read-only MCP server, inspect its tool surface.

``particles mcp serve`` is what an MCP client (Claude Code, Claude
Desktop, etc.) launches as a child process. ``particles mcp tools`` is a
debugging aid that prints the registered tool surface as JSON without
spawning the stdio transport — useful for reviewing the contract or
diffing against a previous version when an ``operations/`` signature
changes.
"""

from __future__ import annotations

import inspect
import json

import typer

from particles.api.cli import app, prepare_stdio_server_store, run
from particles.db import DEFAULT_STORE

mcp_app = typer.Typer(
    help="Model Context Protocol server (read-only).",
    no_args_is_help=True,
)
app.add_typer(mcp_app, name="mcp")


@mcp_app.command("serve")
def mcp_serve_cmd(
    project_observer: str | None = typer.Option(
        None,
        "--project-observer",
        metavar="cwd",
        help=(
            "Bind this server to the project of its working directory (the only value is "
            "`cwd`): reads see global beliefs plus that project's, and writes are attributed "
            "to it. Omit for the store-wide server."
        ),
    ),
    codex_destination: str | None = typer.Option(None, "--codex-destination", hidden=True),
    codex_store: str = typer.Option("default", "--codex-store", hidden=True),
) -> None:
    """Run the Particles MCP server over stdio.

    Typical install path on the operator's machine::

        claude mcp add particles -- uv run particles mcp serve --project-observer cwd

    `--project-observer cwd` is a launch flag, not configuration, on purpose:
    `config.yaml` is shared by every MCP client on the machine, and a client
    started from your home directory should not be bound to it.
    """
    from particles.egress import UncoveredEventLoop

    # One import edge into `particles.mcp`, the one the acyclic contract pins.
    from particles.mcp import bind_project
    from particles.mcp import main as serve_main

    _check_codex_mcp_pin(codex_store, codex_destination)
    if project_observer is not None:
        if project_observer != "cwd":
            typer.echo("Error: --project-observer accepts only `cwd`.", err=True)
            raise typer.Exit(2)
        from pathlib import Path

        from particles.api.cli._claude_code import server_project_key

        key = server_project_key(Path.cwd())
        if key is None:
            typer.echo(
                "Particles MCP: started at the filesystem root, so this server is not bound "
                "to a project; reads are store-wide.",
                err=True,
            )
        else:
            bind_project(key)
    # Named stores are reached per tool call, so every configured one is held to
    # the store rule here, before serving (security review P15).
    if codex_destination is None:
        prepare_stdio_server_store(DEFAULT_STORE, every_store=True)
    _refuse_an_unusable_token_file()
    try:
        serve_main()
    except UncoveredEventLoop as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


def _check_codex_mcp_pin(store: str, destination: str | None) -> None:
    import os
    from pathlib import Path

    from particles.config import config_file_path, environment_engine_url, get_config

    cfg = get_config()
    if destination is None:
        # An existing local Codex registration pins the state directory.
        # Identify it by its own manifest, without claiming other MCP sessions.
        if (
            cfg.engine.base_url
            and not environment_engine_url()
            and os.environ.get("PARTICLES_CODEX_STATE_DIR")
        ):
            from particles.api.cli._hook_state import read_json

            state = Path(cfg.codex.state_dir).expanduser().resolve()
            for owned in (state / "installations").glob("*.json"):
                try:
                    manifest = read_json(owned, {})
                except (ValueError, OSError):
                    continue
                if (
                    manifest.get("config") == str(config_file_path())
                    and not manifest.get("engine")
                    and manifest.get("mcp")
                ):
                    typer.echo(
                        "Codex MCP needs an engine install. Re-run particles init codex.", err=True
                    )
                    raise typer.Exit(1)
        return

    import asyncio

    from particles.api._keys import Role, role_from_prefix
    from particles.api.cli._codex_remote import (
        CodexKeyRefused,
        installed_destination,
        prove_installed_key,
        token_path,
    )
    from particles.api.client.http import HttpBackend
    from particles.egress import run_armed
    from particles.secrets import engine_token_file_path, get_engine_token_optional

    try:
        url, _, manifest = installed_destination(store, destination)
        state = Path(cfg.codex.state_dir).expanduser().resolve()
        path = str(token_path(state, "mcp"))
        if manifest.get("tokens", {}).get("mcp") != path or engine_token_file_path() != path:
            raise CodexKeyRefused("Codex MCP credential pin changed. Re-run particles init codex.")
        # Use the same cached value that the serving process will send.
        key = get_engine_token_optional()
        if key is None or role_from_prefix(key) is not Role.AGENT:
            raise CodexKeyRefused("Codex MCP needs its enrolled agent key.")
        backend = HttpBackend(token=key, base_url=url)
        run_armed(
            asyncio.wait_for(
                prove_installed_key(backend, "mcp", manifest), cfg.codex.hook_deadline_seconds
            )
        )
    except (ValueError, OSError, RuntimeError) as exc:
        message = (
            str(exc)
            if isinstance(exc, CodexKeyRefused)
            else (
                "Codex MCP could not verify its engine and key pins. "
                "Run hook doctor --harness codex."
            )
        )
        typer.echo(message, err=True)
        raise typer.Exit(1) from None


def _refuse_an_unusable_token_file() -> None:
    """Refuse to serve when the engine key's token file fails a check.

    Every tool reaches a remote engine through the backend seam, so a
    server with no usable key has nothing to serve. Claude Code shows the exit
    as a failed server and keeps the message in its MCP log.
    """
    from particles.config import get_config
    from particles.secrets import EngineTokenFileError, get_engine_token_optional

    if not get_config().engine.base_url:
        return
    try:
        get_engine_token_optional()
    except EngineTokenFileError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None


@mcp_app.command("tools")
def mcp_tools_cmd(
    output_format: str = typer.Option(
        "json", "--format", help='Output format: "json" (default) or "text".'
    ),
) -> None:
    """Print the registered MCP tool surface (name, description, input schema).

    Used to verify the contract without spawning an MCP client. The JSON
    output is what ``tests/mcp/tool-schema.json`` should match; drift
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

    surface = run(_dump())

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
    what ``tests/mcp/resource-schema.json`` should match; drift here means the
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

    surface = run(_dump())

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

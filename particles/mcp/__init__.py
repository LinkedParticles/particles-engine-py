"""Read-only Model Context Protocol (MCP) server for Particles.

A fourth SDK front-end alongside the Typer CLI and FastAPI app. Lets AI
agents (Claude Code, Claude Desktop, other MCP-aware clients) consult the
particle store with typed JSON tool calls, no shell quoting and no
per-call permission prompts.

Read-only in v1 — mutating verbs (deposit, extract, tag/untag, status
transitions) defer to a separate Phase B ADR that designs the audit
trail + confirmation prompt + cost-control story.
"""

from particles.mcp.server import build_server, main

__all__ = ["build_server", "main"]

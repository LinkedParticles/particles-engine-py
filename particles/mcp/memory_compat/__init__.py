"""Reference memory-server compatibility façade.

A drop-in MCP surface mirroring ``@modelcontextprotocol/server-memory`` so an
existing client works unmodified, backed by the Particles store. Served by
``particles memory serve``.
"""

from particles.mcp.memory_compat.server import RESOURCE_URI, SERVER_NAME, build_server, main

__all__ = ["RESOURCE_URI", "SERVER_NAME", "build_server", "main"]

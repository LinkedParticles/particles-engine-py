"""Per-domain MCP tool handlers.

Each module here exports one or more module-level ``async def`` tool
functions; :func:`particles.mcp.server.build_server` imports each one
and registers it via ``server.tool()(fn)`` in the order shown in
:data:`particles.mcp.server._TOOL_REGISTRATION_ORDER`.

The split mirrors the per-verb layout under :mod:`particles.api.cli`
(established by the M2 CLI split): one file per domain, with the tool
function's docstring serving as the MCP-client-visible description.
``inspect.cleandoc`` normalisation happens at the ``particles mcp tools``
dumper layer (see :mod:`particles.api.cli.mcp`) so docstring shape is
stable across Python versions.
"""

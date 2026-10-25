# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The per-process MCP rate cap on model-calling tool calls (security review D30)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from particles.config import get_config, reset_config
from particles.db import DEFAULT_STORE
from particles.mcp import limits


def test_the_bucket_admits_its_rate_then_refuses() -> None:
    get_config().mcp.rate_limit_per_minute = 3
    for _ in range(3):
        limits.admit("query")
    with pytest.raises(ValueError, match="allows 3 such calls a minute"):
        limits.admit("query")


def test_the_bucket_refills_over_a_minute() -> None:
    get_config().mcp.rate_limit_per_minute = 2
    with patch.object(limits.time, "monotonic", return_value=1000.0):
        limits.admit("query")
        limits.admit("query")
        with pytest.raises(ValueError):
            limits.admit("query")
    with patch.object(limits.time, "monotonic", return_value=1030.0):
        limits.admit("query")  # half a minute refills one of two tokens


def test_zero_disables_the_cap() -> None:
    get_config().mcp.rate_limit_per_minute = 0
    for _ in range(500):
        limits.admit("query")


def test_reset_config_clears_the_bucket() -> None:
    get_config().mcp.rate_limit_per_minute = 1
    limits.admit("query")
    reset_config()
    get_config().mcp.rate_limit_per_minute = 1
    limits.admit("query")


def _fake_backend() -> Any:
    backend = MagicMock()
    resp = MagicMock()
    resp.particles = []
    resp.model_dump = MagicMock(return_value={})
    backend.query = AsyncMock(return_value=resp)
    backend.inconsistency_backrefs = AsyncMock(return_value={})
    return backend


@pytest.mark.asyncio
async def test_a_query_with_a_question_is_counted_and_one_without_is_not() -> None:
    from particles.mcp.tools.query import query

    get_config().mcp.rate_limit_per_minute = 1
    backend = _fake_backend()
    with patch("particles.api.client.get_backend", return_value=backend):
        await query(question="What rotates monthly?")
        with pytest.raises(ValueError, match="Rate limit reached: query"):
            await query(question="And weekly?")
        await query(list_predicates=True)  # a structural read drives no model call
    assert backend.query.await_count == 2


@pytest.mark.asyncio
async def test_particle_assert_is_counted(db_session: Any) -> None:
    from particles.mcp.tools.write import particle_assert

    get_config().mcp.write.enabled_stores = [DEFAULT_STORE]
    get_config().mcp.rate_limit_per_minute = 1
    backend = MagicMock()
    backend.particle_assert = AsyncMock(side_effect=RuntimeError("reached the backend"))
    with patch("particles.api.client.get_backend", return_value=backend):
        with pytest.raises(RuntimeError, match="reached the backend"):
            await particle_assert("X.", ["X"], 0.5, source_excerpt="x")
        with pytest.raises(ValueError, match="Rate limit reached: particle_assert"):
            await particle_assert("Y.", ["Y"], 0.5, source_excerpt="y")


@pytest.mark.asyncio
async def test_facade_writes_are_counted(db_session: Any) -> None:
    from particles.mcp.memory_compat import server

    get_config().mcp.write.enabled_stores = [DEFAULT_STORE]
    get_config().mcp.rate_limit_per_minute = 1
    srv = server.build_server()
    handler = srv.request_handlers
    import mcp.types as types

    call = handler[types.CallToolRequest]

    def request(name: str) -> types.CallToolRequest:
        return types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name=name, arguments={"entities": []}),
        )

    first = await call(request("create_entities"))
    assert not first.root.isError
    second = await call(request("create_entities"))
    assert second.root.isError
    assert "Rate limit reached: create_entities" in second.root.content[0].text

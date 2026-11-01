# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The MCP rate cap on model-calling tool calls (security review D30).

One allowance per store, shared by every server process on it (security review
2026-10-25-2, U11 / R24).
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path
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


# ---------------------------------------------------------------------------
# One allowance per store, however many server processes (U11, R24)
# ---------------------------------------------------------------------------

_CHILD = """
from particles.mcp.limits import admit
admitted = 0
for _ in range(5):
    try:
        admit("particle_assert")
        admitted += 1
    except ValueError:
        pass
print(admitted)
"""


def _child_admitted(home: Path, config: Path) -> int:
    env = {"HOME": str(home), "PARTICLES_CONFIG": str(config), "PATH": os.environ.get("PATH", "")}
    out = subprocess.run(
        [sys.executable, "-I", "-c", _CHILD],
        cwd=str(home),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return int(out.stdout.strip().splitlines()[-1])


def test_several_server_processes_on_one_store_share_one_allowance(tmp_path: Path) -> None:
    """Two server processes on one store admit no more than the cap between them.

    Each child imports only the limiter, under a temporary ``HOME`` and config, and
    resolves the default store to ``./particles.db`` in that ``HOME``; no store is
    opened and nothing reaches the network."""
    home = tmp_path / "home"
    home.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("network:\n  egress: none\nmcp:\n  rate_limit_per_minute: 2\n")
    first = _child_admitted(home, config)
    second = _child_admitted(home, config)
    assert (first, second) == (2, 0)
    assert (home / f"particles.db{limits.STATE_SUFFIX}").is_file()


def test_the_shared_bucket_sits_beside_a_file_store(tmp_path: Path) -> None:
    """A file-backed store keeps its bucket beside the store file, owner-only, so a
    restarted server (process state dropped) still spends from it."""
    db = tmp_path / "store.db"
    get_config().storage.database_url = f"sqlite+aiosqlite:///{db}"
    get_config().mcp.rate_limit_per_minute = 1
    limits.admit("query")
    limits.reset_limits()  # what a fresh process starts with
    with pytest.raises(ValueError, match="across all its MCP servers"):
        limits.admit("query")
    state = Path(f"{db}{limits.STATE_SUFFIX}")
    assert limits.shared_state_path() == state
    assert stat.S_IMODE(state.stat().st_mode) == 0o600


def test_each_store_has_its_own_allowance(tmp_path: Path) -> None:
    get_config().storage.database_url = f"sqlite+aiosqlite:///{tmp_path / 'a.db'}"
    get_config().storage.stores = {"other": f"sqlite+aiosqlite:///{tmp_path / 'b.db'}"}
    get_config().mcp.rate_limit_per_minute = 1
    limits.admit("query")
    limits.admit("query", store="other")
    with pytest.raises(ValueError):
        limits.admit("query", store="other")


def test_the_shared_bucket_refills_on_the_wall_clock(tmp_path: Path) -> None:
    get_config().storage.database_url = f"sqlite+aiosqlite:///{tmp_path / 'store.db'}"
    get_config().mcp.rate_limit_per_minute = 2
    with patch.object(limits.time, "time", return_value=1000.0):
        limits.admit("query")
        limits.admit("query")
        with pytest.raises(ValueError):
            limits.admit("query")
    with patch.object(limits.time, "time", return_value=1030.0):
        limits.admit("query")  # half a minute refills one of two tokens
    with patch.object(limits.time, "time", return_value=900.0), pytest.raises(ValueError):
        limits.admit("query")  # a clock stepped back refills nothing


def test_a_store_with_no_file_or_a_remote_engine_keeps_the_bucket_in_process(
    tmp_path: Path,
) -> None:
    """In-memory SQLite is not shared across processes, and a server routed to an
    engine is bounded by the engine's own HTTP rate limit, so neither writes a file."""
    assert limits.shared_state_path() is None  # the suite's in-memory store
    get_config().storage.database_url = f"sqlite+aiosqlite:///{tmp_path / 'store.db'}"
    get_config().engine.base_url = "http://127.0.0.1:9"
    assert limits.shared_state_path() is None
    get_config().mcp.rate_limit_per_minute = 1
    limits.admit("query")
    assert not list(tmp_path.iterdir())


def test_the_shared_allowance_write_follows_no_planted_link(tmp_path: Path) -> None:
    """A link planted at ``<db>.mcp-rate.tmp`` beside a file-backed store never
    makes an MCP tool call truncate or overwrite the file it points at
    (security review 2026-10-26-1, W40)."""
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    db = store_dir / "particles.db"
    db.write_bytes(b"")
    get_config().storage.database_url = f"sqlite+aiosqlite:///{db}"
    get_config().engine.base_url = None
    get_config().mcp.rate_limit_per_minute = 30

    victim = tmp_path / "victim.txt"
    victim.write_text("owner data that must survive\n")
    os.symlink(victim, str(db) + limits.STATE_SUFFIX + ".tmp")
    os.symlink(victim, str(db) + limits.STATE_SUFFIX)

    limits.admit("particle_assert", store=DEFAULT_STORE)
    assert victim.read_text() == "owner data that must survive\n"
    state = Path(str(db) + limits.STATE_SUFFIX)
    assert not state.is_symlink()
    assert stat.S_IMODE(state.stat().st_mode) == 0o600
    assert not [p for p in store_dir.iterdir() if p.name.endswith(".tmp") and not p.is_symlink()]


def test_two_spellings_of_one_store_file_share_one_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative DSN, a ``..`` and a link to the store file name one bucket (R24)."""
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    db = store_dir / "particles.db"
    db.write_bytes(b"")
    (tmp_path / "alias.db").symlink_to(db)
    get_config().engine.base_url = None
    monkeypatch.chdir(tmp_path)
    paths = set()
    for spelling in (
        str(db),
        "store/particles.db",
        f"{store_dir}/../store/particles.db",
        "alias.db",
    ):
        get_config().storage.database_url = f"sqlite+aiosqlite:///{spelling}"
        paths.add(limits.shared_state_path(DEFAULT_STORE))
    assert paths == {db.resolve().with_name("particles.db" + limits.STATE_SUFFIX)}

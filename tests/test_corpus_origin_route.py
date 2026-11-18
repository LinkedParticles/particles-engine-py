# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The engine's side of a client's push: stamped by the key, joined by host.

A scratch store and key file under ``tmp_path``, configuration only through the
loader, no model call and no outside connection.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from tests._corpus_rows import CLIENT_HOST_ID

MEMORY_ROOT = "/home/u/.claude/projects"
ARCHIVE_ROOT = "/home/u/.particles/claude-code/projects"
MEMORY_URI = f"file://{MEMORY_ROOT}/repo/memory/notes.md"
OTHER_CLIENT = "0c1e7a5e-0000-4000-8000-0000000b0b00"


def _create_tables() -> None:
    async def _go() -> None:
        import particles._orm_modules  # noqa: F401
        from particles.db import Base, get_engine

        async with get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await get_engine().dispose()

    asyncio.run(_go())


@pytest.fixture
def engine_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    from particles.api import _keys
    from particles.config import reset_config

    home = tmp_path / "home"
    (home / ".claude" / "projects").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    keydir = tmp_path / "keys"
    keydir.mkdir(mode=0o700)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("network:\n  egress: none\n", encoding="utf-8")
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'r.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(keydir / "engine-keys.json"))
    monkeypatch.delenv("PARTICLES_API_KEY", raising=False)
    reset_config()
    _keys.reset_key_cache()
    _keys.forget_key_files()
    _create_tables()
    yield home
    _keys.reset_key_cache()
    _keys.forget_key_files()
    reset_config()


def _client() -> TestClient:
    from particles.api.app import app

    return TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))


def _hook(host_id: str | None = CLIENT_HOST_ID) -> str:
    from particles.api import _keys
    from particles.api._keys import Role

    return _keys.mint(
        Role.AGENT,
        "claude-code-hook",
        reserved_ok=True,
        memory_roots=[MEMORY_ROOT],
        archive_roots=[ARCHIVE_ROOT],
        host_id=host_id,
    )[0]


def _push(token: str, uri: str = MEMORY_URI, text: str = "a memory") -> Any:
    return _client().post(
        "/corpus/deposit/text",
        # As HttpBackend sends it: a file:// URI-R is never LAZY over HTTP.
        json={
            "text": text,
            "uri_r": uri,
            "source_type": "LOCAL_MARKDOWN",
            "mutability": "MUTABLE",
            "fetch_policy": "NEVER",
        },
        headers={"Authorization": f"Bearer {token}"},
    )


def _local_entry(uri: str, *, origin: str | None | bool = False, lazy: bool = True) -> str:
    """An entry as a local harvest made it: this host's, or ``origin`` when given."""
    from particles.core.schema import FetchPolicy, Mutability
    from particles.corpus.deposit import deposit_text_versioned
    from particles.corpus.origin import deposit_origin
    from particles.db import session_scope

    async def _go() -> str:
        async with session_scope(write=True) as session:
            kwargs: dict[str, Any] = {
                "text": "v1",
                "uri_r": uri,
                "source_type": "LOCAL_MARKDOWN",
                "mutability": Mutability.MUTABLE,
                "deposited_by": "claude-code-hook",
                "fetch_policy": FetchPolicy.LAZY if lazy else FetchPolicy.NEVER,
            }
            if origin is False:
                entry_id, _, _ = await deposit_text_versioned(session, **kwargs)
            else:
                assert not isinstance(origin, bool)
                with deposit_origin(origin):
                    entry_id, _, _ = await deposit_text_versioned(session, **kwargs)
            await session.commit()
            return entry_id

    return asyncio.run(_go())


def _row(entry_id: str) -> Any:
    from particles.corpus.store import CorpusEntryRow
    from particles.db import session_scope

    async def _go() -> Any:
        async with session_scope() as session:
            return await session.get(CorpusEntryRow, entry_id)

    return asyncio.run(_go())


def test_a_push_is_stamped_with_the_keys_host_never_the_body(engine_home: Path) -> None:
    resp = _push(_hook())
    assert resp.status_code == 200, resp.text
    assert _row(resp.json()["entry_id"]).origin_host == CLIENT_HOST_ID


def test_a_push_into_its_own_hosts_lazy_entry_is_accepted_and_set_never(
    engine_home: Path,
) -> None:
    """The §2 refusal: a LAZY memory file the local harvest enrolled, pushed after a cutover."""
    entry_id = _local_entry(MEMORY_URI, origin=CLIENT_HOST_ID)
    resp = _push(_hook(), text="edited")
    assert resp.status_code == 200, resp.text
    row = _row(entry_id)
    assert row.fetch_policy == "NEVER" and row.origin_host == CLIENT_HOST_ID


def test_a_push_into_another_clients_entry_is_refused_naming_both(engine_home: Path) -> None:
    _local_entry(MEMORY_URI, origin=OTHER_CLIENT)
    resp = _push(_hook())
    assert resp.status_code == 409
    assert OTHER_CLIENT in resp.text and CLIENT_HOST_ID in resp.text


def test_a_push_onto_the_engines_own_file_is_refused_with_the_repair(engine_home: Path) -> None:
    _local_entry(MEMORY_URI)
    resp = _push(_hook())
    assert resp.status_code == 409
    assert "corpus reorigin" in resp.text


def test_a_key_with_no_host_names_no_file_but_keeps_its_transcripts(engine_home: Path) -> None:
    token = _hook(host_id=None)
    resp = _push(token)
    assert resp.status_code == 403 and "--host-id" in resp.text
    ok = _client().post(
        "/corpus/deposit/text",
        json={
            "text": "a session",
            "uri_r": "claude-code://session/abc",
            "mutability": "APPEND_ONLY",
            "tags": ["claude-code"],
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert ok.status_code == 200, ok.text


def test_a_key_carrying_the_engines_own_host_is_refused_at_mint_and_at_the_route(
    engine_home: Path,
) -> None:
    from particles.api import _keys
    from particles.api._keys import KeyRuleError
    from particles.host_identity import local_host_id

    own = local_host_id()
    assert own is not None
    with pytest.raises(KeyRuleError, match="engine host's own id"):
        _hook(host_id=own)
    # A hand-carried key file: the route refuses it too.
    token = _hook()
    path = _keys.keys_file_path()
    data = json.loads(path.read_text())
    data["keys"][0]["host_id"] = own
    path.write_text(json.dumps(data))
    path.chmod(0o600)
    _keys.reset_key_cache()
    resp = _push(token)
    assert resp.status_code == 403 and "engine host's own id" in resp.text


def test_whoami_returns_the_keys_host(engine_home: Path) -> None:
    body = _client().get("/auth/whoami", headers={"Authorization": f"Bearer {_hook()}"}).json()
    assert body["host_id"] == CLIENT_HOST_ID


def test_keys_create_records_the_host_and_refuses_one_on_another_role(engine_home: Path) -> None:
    from particles.api.cli import app

    runner = CliRunner()
    ok = runner.invoke(
        app,
        [
            "engine", "keys", "create", "--role", "agent", "--principal", "claude-code-hook",
            "--reserved-principal", "--host-id", CLIENT_HOST_ID.upper(),
            "--memory-root", MEMORY_ROOT, "--output", str(engine_home / "hook.key"),
        ],
    )  # fmt: skip
    assert ok.exit_code == 0, ok.output
    assert f"client host {CLIENT_HOST_ID}" in ok.output
    bad = runner.invoke(
        app,
        [
            "engine", "keys", "create", "--role", "agent", "--principal", "agent:ci-bot",
            "--host-id", CLIENT_HOST_ID, "--output", str(engine_home / "bot.key"),
        ],
    )  # fmt: skip
    assert bad.exit_code == 1 and "harvester" in bad.output


def test_db_init_stamps_once_previews_without_writing_and_reorigin_repairs(
    engine_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.api.cli import app
    from particles.api.cli import db as db_cli
    from particles.host_identity import local_host_id

    async def _schema_is_current() -> None:
        """The fixture built the schema with create_all; the stamp is what is under test."""

    monkeypatch.setattr(db_cli, "create_tables", _schema_is_current)

    entry_id = _local_entry("file:///home/u/AGENTS.md", origin=None)
    runner = CliRunner()
    preview = runner.invoke(app, ["db", "init", "--origin-preview"])
    assert preview.exit_code == 0, preview.output
    assert "Would stamp 1 file:// entr(ies)" in preview.output
    assert _row(entry_id).origin_host is None
    first = runner.invoke(app, ["db", "init"])
    assert first.exit_code == 0, first.output
    assert "Stamped 1 file:// entr(ies)" in first.output
    me = local_host_id()
    assert _row(entry_id).origin_host == me
    second = runner.invoke(app, ["db", "init"])
    assert "Stamped" not in second.output
    dry = runner.invoke(
        app, ["corpus", "reorigin", "--from", str(me), "--to", CLIENT_HOST_ID, "--dry-run"]
    )
    assert dry.exit_code == 0 and "1 file:// entr(ies)" in dry.output
    assert _row(entry_id).origin_host == me
    moved = runner.invoke(
        app, ["corpus", "reorigin", "--from", str(me), "--to", CLIENT_HOST_ID, "--yes"]
    )
    assert moved.exit_code == 0, moved.output
    assert _row(entry_id).origin_host == CLIENT_HOST_ID


def test_the_startup_migration_never_stamps(
    engine_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine's own startup migration (a container's) leaves unstamped entries alone."""
    from particles.config import reset_config
    from particles.db import create_tables

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'migrated.db'}")
    reset_config()
    asyncio.run(create_tables())
    entry_id = _local_entry("file:///home/u/AGENTS.md", origin=None)
    asyncio.run(create_tables())
    assert _row(entry_id).origin_host is None


def test_the_doctor_names_a_hook_key_recording_another_client_host(
    engine_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mistyped --host-id shows before any push."""
    from particles.api.cli import hook
    from particles.api.cli._claude_code import local_harvest_roots
    from particles.api.client import http
    from particles.config import get_config
    from particles.host_identity import local_host_id

    memory, archive = local_harvest_roots()
    monkeypatch.setattr(get_config().engine, "base_url", "http://engine.test:8000")
    monkeypatch.setenv("PARTICLES_ENGINE_TOKEN", "ptk_agent_0123456789ab_" + "A" * 43)
    monkeypatch.setattr(hook, "_hook_pinned_tokens", lambda: [])

    async def _whoami(_self: object) -> dict[str, Any]:
        return {
            "principal": "claude-code-hook",
            "role": "agent",
            "memory_roots": memory,
            "archive_roots": archive,
            "host_id": OTHER_CLIENT,
        }

    monkeypatch.setattr(http.HttpBackend, "whoami", _whoami)
    line = hook._harvest_roots_line()
    assert line is not None and "⚠" in line
    assert OTHER_CLIENT in line and f"--host-id {local_host_id()}" in line


def test_the_doctor_counts_the_stores_files_by_origin(engine_home: Path) -> None:
    from particles.api.cli import hook

    _local_entry("file:///home/u/a.md")
    _local_entry("file:///home/u/b.md", origin=OTHER_CLIENT)
    _local_entry("file:///home/u/c.md", origin=None)
    lines = hook._check_refresh_roots("default")
    assert any(
        line.startswith("file origins:")
        and "1 file:// entr(ies) are this host's, 1 another host's, 1 unstamped" in line
        for line in lines
    ), lines

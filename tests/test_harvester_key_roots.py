# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""A harvester key's file URIs lie under the client directories it records.

A scratch store and key file under ``tmp_path``, a temporary ``HOME``,
configuration only through the loader, no model call and no outside
connection. The doctor's engine probe is stubbed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator, Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

MEMORY_ROOT = "/home/u/.claude/projects"
ARCHIVE_ROOT = "/home/u/.particles/claude-code/projects"
MEMORY_URI = f"file://{MEMORY_ROOT}/repo/memory/notes.md"
ARCHIVE_URI = f"file://{ARCHIVE_ROOT}/repo/MEMORY.archive.md"


def _create_tables() -> None:
    async def _go() -> None:
        import particles._orm_modules  # noqa: F401
        from particles.db import Base, get_engine

        async with get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await get_engine().dispose()

    asyncio.run(_go())


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    """A scratch engine on per-client keys, with ``HOME`` under ``tmp_path``."""
    from particles.api import _keys
    from particles.config import reset_config

    home = tmp_path / "home"
    (home / ".claude" / "projects").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    keydir = tmp_path / "keys"
    keydir.mkdir(mode=0o700)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "network:\n  egress: none\n"
        f"claude_code:\n  state_dir: {home / '.particles' / 'claude-code'}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'r.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(keydir / "engine-keys.json"))
    monkeypatch.delenv("PARTICLES_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDECODE", raising=False)
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


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _hook(**roots: Any) -> str:
    from particles.api import _keys
    from particles.api._keys import Role

    return _keys.mint(Role.AGENT, "claude-code-hook", reserved_ok=True, **roots)[0]


def _rooted() -> str:
    return _hook(memory_roots=[MEMORY_ROOT], archive_roots=[ARCHIVE_ROOT])


def _deposit(token: str, uri: str) -> Any:
    return _client().post(
        "/corpus/deposit/text",
        json={"text": "a memory", "uri_r": uri, "source_type": "LOCAL_MARKDOWN"},
        headers=_bearer(token),
    )


@pytest.fixture
def lock_log(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    import particles.db as db

    taken: list[str] = []
    real = db.write_lock

    @contextlib.asynccontextmanager
    async def _recording(store: str = "default") -> AsyncIterator[None]:
        taken.append(store)
        async with real(store):
            yield

    monkeypatch.setattr(db, "write_lock", _recording)
    return taken


# --- the route -----------------------------------------------------------------


@pytest.mark.parametrize("uri", [MEMORY_URI, ARCHIVE_URI], ids=["memory", "archive"])
def test_a_file_under_the_keys_roots_is_accepted(home: Path, uri: str) -> None:
    resp = _deposit(_rooted(), uri)
    assert resp.status_code == 200, resp.text


@pytest.mark.parametrize(
    "uri",
    [
        "file:///anywhere/projects/victim/memory/notes.md",
        "file:///home/u/.claude/projects-evil/projects/repo/memory/notes.md",
        "file:///home/u/.Claude/projects/repo/memory/notes.md",
        "file:///home/u/.claude/projects/repo/MEMORY.archive.md",
        f"file://{ARCHIVE_ROOT}/repo/memory/notes.md",
    ],
    ids=["elsewhere", "string-prefix", "case", "archive-under-memory", "memory-under-archive"],
)
def test_a_file_outside_the_keys_roots_is_refused_before_the_writer_lock(
    home: Path, lock_log: list[str], uri: str
) -> None:
    token = _rooted()
    lock_log.clear()
    resp = _deposit(token, uri)
    assert resp.status_code == 403, resp.text
    assert "lies under none" in resp.text
    assert lock_log == []


def test_a_key_with_no_roots_names_no_file_and_keeps_its_transcripts(home: Path) -> None:
    token = _hook()
    refused = _deposit(token, MEMORY_URI)
    assert refused.status_code == 403
    assert "records no harvest roots" in refused.text
    transcript = _client().post(
        "/corpus/deposit/text",
        json={"text": "a session", "uri_r": "claude-code://session/s-1"},
        headers=_bearer(token),
    )
    assert transcript.status_code == 200, transcript.text


def test_a_harvester_key_names_no_memory_file_outside_the_engines_memory_root(
    home: Path, tmp_path: Path
) -> None:
    """The audit's proof (L40, AA46): a harvest-shaped path outside every root is refused."""
    from particles.api.app import _is_harvest_uri
    from particles.api.cli._claude_code import local_harvest_roots

    outside = tmp_path / "anywhere" / "projects" / "victim" / "memory" / "notes.md"
    uri = f"file://{outside}"
    assert _is_harvest_uri(uri)
    for token in (
        _hook(),
        _hook(**dict(zip(("memory_roots", "archive_roots"), local_harvest_roots(), strict=True))),
    ):
        assert _deposit(token, uri).status_code in {400, 403}


# --- the mint and the record rule ---------------------------------------------


@pytest.mark.parametrize("field", ["memory_roots", "archive_roots"])
def test_a_non_harvester_key_takes_no_roots(home: Path, field: str) -> None:
    from particles.api import _keys
    from particles.api._keys import KeyRuleError, Role

    with pytest.raises(KeyRuleError, match="harvester"):
        _keys.mint(Role.AGENT, "agent:ci-bot", **{field: [MEMORY_ROOT]})


@pytest.mark.parametrize("roots", [[], ["relative/dir"], ["/"], ["/a/../b"], ["/a//b"]])
def test_a_root_that_is_not_an_absolute_directory_is_refused(home: Path, roots: list[str]) -> None:
    from particles.api._keys import KeyRuleError

    with pytest.raises(KeyRuleError):
        _hook(memory_roots=roots)


def test_roots_are_stored_normalised_and_once(home: Path) -> None:
    from particles.api import _keys

    _hook(memory_roots=[MEMORY_ROOT + "/", MEMORY_ROOT], archive_roots=[ARCHIVE_ROOT])
    (record,) = _keys.live_keys()
    assert record.memory_roots == (MEMORY_ROOT,)
    assert record.archive_roots == (ARCHIVE_ROOT,)


def _rewrite(path: Path, **fields: object) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["keys"][0].update(fields)
    path.write_text(json.dumps(data), encoding="utf-8")
    os.chmod(path, 0o600)


@pytest.mark.parametrize(
    ("principal", "fields"),
    [
        ("agent:ci-bot", {"memory_roots": [MEMORY_ROOT]}),
        ("claude-code-hook", {"memory_roots": []}),
        ("claude-code-hook", {"archive_roots": ["/home/u/x/"]}),
    ],
    ids=["non-harvester", "empty", "not-normalised"],
)
def test_a_hand_written_record_breaking_the_root_rule_never_authenticates(
    home: Path, principal: str, fields: dict[str, object]
) -> None:
    from particles.api import _keys
    from particles.api._keys import Role

    token, _ = _keys.mint(Role.AGENT, principal, reserved_ok=principal == "claude-code-hook")
    _rewrite(_keys.keys_file_path(), **fields)
    _keys.reset_key_cache()
    assert _keys.lookup(token) is None


def test_a_record_with_no_roots_is_written_as_an_older_engine_wrote_it(home: Path) -> None:
    from particles.api import _keys

    _hook()
    (item,) = json.loads(_keys.keys_file_path().read_text(encoding="utf-8"))["keys"]
    assert "memory_roots" not in item and "archive_roots" not in item


# --- keys create, whoami, the doctor, the install and startup ---------------


def test_keys_create_records_this_hosts_roots_by_default(home: Path) -> None:
    from particles.api import _keys
    from particles.api.cli import app

    result = CliRunner().invoke(
        app,
        [
            "engine",
            "keys",
            "create",
            "--role",
            "agent",
            "--principal",
            "claude-code-hook",
            "--reserved-principal",
            "--print-token",
        ],
        env={"CLAUDECODE": None},
    )
    assert result.exit_code == 0, result.output
    (record,) = _keys.live_keys()
    assert record.memory_roots == (str((home / ".claude" / "projects").resolve()),)
    assert record.archive_roots == (
        str((home / ".particles" / "claude-code" / "projects").resolve()),
    )
    assert "Harvests memory files under" in result.output


def test_keys_create_records_the_roots_it_is_given(home: Path) -> None:
    from particles.api import _keys
    from particles.api.cli import app

    result = CliRunner().invoke(
        app,
        [
            "engine",
            "keys",
            "create",
            "--role",
            "agent",
            "--principal",
            "claude-code-hook",
            "--reserved-principal",
            "--print-token",
            "--memory-root",
            MEMORY_ROOT,
            "--archive-root",
            ARCHIVE_ROOT,
        ],
        env={"CLAUDECODE": None},
    )
    assert result.exit_code == 0, result.output
    (record,) = _keys.live_keys()
    assert (record.memory_roots, record.archive_roots) == ((MEMORY_ROOT,), (ARCHIVE_ROOT,))


def test_whoami_returns_the_keys_roots(home: Path) -> None:
    body = _client().get("/auth/whoami", headers=_bearer(_rooted())).json()
    assert body["memory_roots"] == [MEMORY_ROOT]
    assert body["archive_roots"] == [ARCHIVE_ROOT]


@pytest.mark.parametrize("recorded", ["same", "other"])
def test_the_doctor_names_a_client_root_the_hook_key_lacks(
    home: Path, monkeypatch: pytest.MonkeyPatch, recorded: str
) -> None:
    from particles.api.cli import hook
    from particles.api.cli._claude_code import local_harvest_roots
    from particles.api.client import http
    from particles.config import get_config

    memory, archive = local_harvest_roots()
    monkeypatch.setattr(get_config().engine, "base_url", "http://engine.test:8000")
    monkeypatch.setenv("PARTICLES_ENGINE_TOKEN", "ptk_agent_0123456789ab_" + "A" * 43)
    monkeypatch.setattr(hook, "_hook_pinned_tokens", lambda: [])

    async def _whoami(_self: object) -> dict[str, Any]:
        return {
            "principal": "claude-code-hook",
            "role": "agent",
            "memory_roots": memory if recorded == "same" else [MEMORY_ROOT],
            "archive_roots": archive,
        }

    monkeypatch.setattr(http.HttpBackend, "whoami", _whoami)
    line = hook._harvest_roots_line()
    assert line is not None
    if recorded == "same":
        assert "⚠" not in line
    else:
        assert "⚠" in line and memory[0] in line and "--memory-root" in line


def test_the_install_prints_the_hook_key_command_for_a_remote_engine(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.api.cli import init
    from particles.api.cli._claude_code import local_harvest_roots
    from particles.config import get_config

    result: dict[str, Any] = {"next_steps": []}
    init._note_harvest_key_command(result)
    assert result["next_steps"] == []
    monkeypatch.setattr(get_config().engine, "base_url", "http://engine.test:8000")
    init._note_harvest_key_command(result)
    (step,) = result["next_steps"]
    memory, archive = local_harvest_roots()
    assert f"--memory-root {memory[0]}" in step and f"--archive-root {archive[0]}" in step


def test_startup_names_a_harvester_key_with_no_roots(
    home: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from particles.api.auth import enforce_key_material_on_startup

    monkeypatch.setenv("PARTICLES_API_KEY", "a-real-secret")
    _hook()
    with caplog.at_level(logging.WARNING, logger="particles.api.auth"):
        enforce_key_material_on_startup()
    assert any("records no harvest roots" in r.getMessage() for r in caplog.records)

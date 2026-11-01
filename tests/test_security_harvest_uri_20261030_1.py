# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""A harvester's memory URI has one spelling (security review 2026-10-30-1, AB52).

The route judges a harvester's ``file://`` URI by its percent-decoded path,
while the deposit stores and looks up the raw string and ``entry_project_key``
reads the slug from it. The route therefore accepts only the spelling the
harvest sends (``Path.as_uri()``), so the path the roots judged, the URI the
store keys on and the slug every consumer reads are one. Scratch store and key
file under tmp_path, a temporary HOME, configuration through the loader, no
model call and no outside connection.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

LEGACY = "s9-legacy-operator-key"


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
    from particles.api import _keys
    from particles.config import reset_config

    home = tmp_path / "home"
    (home / ".claude" / "projects").mkdir(parents=True)
    (home / ".particles" / "claude-code" / "projects").mkdir(parents=True)
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
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 's9.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(keydir / "engine-keys.json"))
    monkeypatch.setenv("PARTICLES_API_KEY", LEGACY)
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


def _roots(home: Path) -> tuple[str, str]:
    from particles.api.cli._claude_code import local_harvest_roots

    memory, archive = local_harvest_roots()
    return memory[0], archive[0]


def _hook_key(home: Path) -> str:
    from particles.api import _keys
    from particles.api._keys import Role

    memory, archive = _roots(home)
    return _keys.mint(
        Role.AGENT,
        "claude-code-hook",
        reserved_ok=True,
        memory_roots=[memory],
        archive_roots=[archive],
    )[0]


def _deposit(token: str, uri: str, text: str = "a memory") -> Any:
    return _client().post(
        "/corpus/deposit/text",
        json={
            "text": text,
            "uri_r": uri,
            "source_type": "LOCAL_MARKDOWN",
            "tags": ["claude-code", "memory-file"],
        },
        headers={"Authorization": f"Bearer {token}"},
    )


# --- block R: the segment comparison -------------------------------------------


def test_the_slug_every_consumer_reads_is_the_slug_the_root_check_judged(home: Path) -> None:
    """An accepted harvester memory URI yields the same project slug to the route and to rescope.

    A percent-encoded spelling of a path under the roots is refused, so no
    stored URI reads as a slug the route did not judge.
    """
    from particles.api.cli._claude_code import entry_project_key

    memory, _ = _roots(home)
    token = _hook_key(home)
    tags = ["claude-code", "memory-file"]
    plain = f"file://{memory}/repo/memory/n.md"
    resp = _deposit(token, plain)
    assert resp.status_code == 200, resp.text
    assert entry_project_key(plain, tags, Path(memory)) == "repo"
    for uri in (f"file://{memory}/re%70o/memory/n.md", f"file://{memory}/repo/%6Demory/m.md"):
        resp = _deposit(token, uri)
        if resp.status_code == 200:
            assert entry_project_key(uri, tags, Path(memory)) == "repo", uri
        else:
            assert resp.status_code == 403, (uri, resp.status_code, resp.text)


def test_one_file_has_one_stored_uri(home: Path) -> None:
    """Two spellings of one memory file under the key's roots never land on two corpus entries."""
    memory, _ = _roots(home)
    token = _hook_key(home)
    plain = _deposit(token, f"file://{memory}/repo/memory/n.md", "first version")
    encoded = _deposit(token, f"file://{memory}/repo/memory/%6E.md", "second version")
    assert plain.status_code == 200, plain.text
    if encoded.status_code == 200:
        assert plain.json()["entry_id"] == encoded.json()["entry_id"]
    else:
        assert encoded.status_code == 403, encoded.text

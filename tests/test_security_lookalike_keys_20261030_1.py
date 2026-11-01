# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""A project key that reads as another is refused (security review 2026-10-30-1, AB32).

Each key below renders as, or is easily read as, a plain key a reader already
knows, and none is a whole-script spelling. Each is driven at the mint and at
an unbound agent key's deposit tag, the two accept sites that refuse a key that
reads as another. A scratch store and key file under ``tmp_path``;
configuration only through the loader; no model call.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

LEGACY = "s6-legacy-operator-key-audit10"

LOOKALIKES = {
    "c0-control": "ac\x01me",
    "c1-control": "ac\x90me",
    "private-use": "ac\ue000me",
    "unassigned": "ac͸me",
    "line-separator": "ac me",
    "latin-alpha": "ɑcme",
    "dotless-i": "bılling",
    "cyrillic-unicode-hyphen": "проект‐тест",
}


def _write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "network:\n  egress: none\nmcp:\n  write:\n    enabled_stores: [default]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))


def _create_tables() -> None:
    async def _go() -> None:
        import particles._orm_modules  # noqa: F401
        from particles.db import Base, get_engine

        async with get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await get_engine().dispose()

    asyncio.run(_go())


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    from particles.api import _keys
    from particles.config import reset_config

    keydir = tmp_path / "keys"
    keydir.mkdir(mode=0o700)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 's6.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(keydir / "engine-keys.json"))
    monkeypatch.setenv("PARTICLES_API_KEY", LEGACY)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    _write_config(tmp_path, monkeypatch)
    reset_config()
    _keys.reset_key_cache()
    _keys.forget_key_files()
    _create_tables()
    yield keydir
    _keys.reset_key_cache()
    _keys.forget_key_files()
    reset_config()


def _client() -> TestClient:
    from particles.api.app import app

    return TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))


@pytest.mark.parametrize("key", list(LOOKALIKES.values()), ids=list(LOOKALIKES))
def test_a_key_that_reads_as_another_is_never_minted(store: Path, key: str) -> None:
    """``keys create --project`` refuses a key holding a hidden or look-alike character."""
    from particles.api import _keys
    from particles.api._keys import KeyRuleError, Role

    with pytest.raises(KeyRuleError):
        _keys.mint(Role.AGENT, "agent:ci-bot", project=key)


@pytest.mark.parametrize(
    ("case", "rule"),
    [
        ("c0-control", "a control, private-use, unassigned or separator character"),
        ("line-separator", "a control, private-use, unassigned or separator character"),
        ("latin-alpha", "a letter that reads as an ASCII letter"),
        ("cyrillic-unicode-hyphen", "a punctuation mark or symbol that reads as an ASCII one"),
    ],
)
def test_a_mint_refusal_names_the_rule_that_fired(store: Path, case: str, rule: str) -> None:
    """The ``--project`` refusal names the class the key broke, not always an invisible one."""
    from particles.api import _keys
    from particles.api._keys import KeyRuleError, Role

    with pytest.raises(KeyRuleError) as refused:
        _keys.mint(Role.AGENT, "agent:ci-bot", project=LOOKALIKES[case])
    assert rule in str(refused.value)
    assert "invisible" not in str(refused.value)


@pytest.mark.parametrize("key", list(LOOKALIKES.values()), ids=list(LOOKALIKES))
def test_an_agent_deposit_tag_that_reads_as_another_is_refused(store: Path, key: str) -> None:
    """An unbound agent key's ``project:`` deposit tag holding such a character is refused."""
    from particles.api import _keys
    from particles.api._keys import Role

    token, _ = _keys.mint(Role.AGENT, "agent:ci-bot")
    resp = _client().post(
        "/corpus/deposit/text",
        json={"text": "a note", "tags": [f"project:{key}"]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code in (403, 422), (resp.status_code, resp.text)

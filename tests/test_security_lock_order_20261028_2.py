# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""A refused request takes no store writer lock (security review 2026-10-28-2, AA6, AA43, E24).

The generic text and file deposits, the operator tag and link mirrors and a
body that does not validate are refused before the writer lock is taken. Each
test uses a scratch store and key file under tmp_path, sets configuration
only through the loader, and makes no model call or outside connection. The
lock recorder wraps particles.db.write_lock, which session_scope(write=True)
resolves at call time.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

LEGACY = "lock-order-legacy-operator-key"


def _write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str = "") -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("network:\n  egress: none\n" + body, encoding="utf-8")
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))


def _create_tables() -> None:
    async def _go() -> None:
        import particles._orm_modules  # noqa: F401
        from particles.db import Base, get_engine

        async with get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await get_engine().dispose()

    asyncio.run(_go())


def _client() -> TestClient:
    from particles.api.app import app

    return TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    from particles.api import _keys
    from particles.config import reset_config

    keydir = tmp_path / "keys"
    keydir.mkdir(mode=0o700)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 's1.db'}")
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


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _agent() -> str:
    from particles.api import _keys
    from particles.api._keys import Role

    return _keys.mint(Role.AGENT, "agent:ci-bot")[0]


def _harvester() -> str:
    from particles.api import _keys
    from particles.api._keys import Role

    return _keys.mint(Role.AGENT, "claude-code-hook", reserved_ok=True)[0]


def _reader() -> str:
    from particles.api import _keys
    from particles.api._keys import Role

    return _keys.mint(Role.READER, "viewer")[0]


# ---------------------------------------------------------------------------
# Control: a refusal the pre-body gate makes takes no lock.
# ---------------------------------------------------------------------------


def test_control_a_reader_text_deposit_takes_no_writer_lock(
    store: Path, lock_log: list[str]
) -> None:
    token = _reader()
    lock_log.clear()
    resp = _client().post("/corpus/deposit/text", json={"text": "x"}, headers=_bearer(token))
    assert resp.status_code == 403, resp.text
    assert lock_log == []


# ---------------------------------------------------------------------------
# E24: the agent-gated generic deposits refuse under the lock.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"text": "x", "deposited_by": "someone-else"},
        {"text": "x", "uri_r": "https://example.org/a"},
        {"text": "x", "source_type": "LOCAL_MARKDOWN"},
        {"text": "x", "tags": ["claude-code"]},
        {"text": "x", "tags": ["rule-file"]},
    ],
    ids=["depositor", "uri", "source-type", "harvest-tag", "rule-mark"],
)
def test_an_agent_text_deposit_refusal_takes_no_writer_lock(
    store: Path, lock_log: list[str], body: dict[str, object]
) -> None:
    """An agent key's refused text deposit answers 403 without taking the writer lock."""
    token = _agent()
    lock_log.clear()
    resp = _client().post("/corpus/deposit/text", json=body, headers=_bearer(token))
    assert resp.status_code == 403, resp.text
    assert lock_log == [], "the store writer lock was taken before the 403"


def test_an_agent_file_deposit_refusal_takes_no_writer_lock(
    store: Path, lock_log: list[str]
) -> None:
    """An agent key's refused file upload answers 403 without taking the writer lock."""
    token = _agent()
    lock_log.clear()
    resp = _client().post(
        "/corpus/deposit/file",
        files={"file": ("note.md", b"# a note\n", "text/markdown")},
        data={"deposited_by": "someone-else"},
        headers=_bearer(token),
    )
    assert resp.status_code == 403, resp.text
    assert lock_log == [], "the store writer lock was taken before the 403"


def test_an_agent_file_deposit_type_refusal_takes_no_writer_lock(
    store: Path, lock_log: list[str]
) -> None:
    """An agent key's upload typed against its bytes is refused without the writer lock."""
    token = _agent()
    lock_log.clear()
    resp = _client().post(
        "/corpus/deposit/file",
        files={"file": ("note.md", b"# a note\n", "text/markdown")},
        data={"source_type": "CONVERSATION"},
        headers=_bearer(token),
    )
    assert resp.status_code == 403, resp.text
    assert lock_log == [], "the upload was copied and typed under the store writer lock"


# ---------------------------------------------------------------------------
# E24 / B14: the operator tag and link mirrors refuse belief writes under the lock.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", "/particles/p-1/tags", {"tags": ["t"]}),
        ("DELETE", "/particles/p-1/tags", {"tags": ["t"]}),
        ("POST", "/links", {"particle_a": "a", "particle_b": "b"}),
        ("DELETE", "/links", {"particle_a": "a", "particle_b": "b"}),
    ],
    ids=["tag", "untag", "link", "unlink"],
)
def test_an_operator_belief_write_refusal_on_a_disabled_store_takes_no_writer_lock(
    store: Path, lock_log: list[str], method: str, path: str, body: dict[str, object]
) -> None:
    """On a store with belief writes disabled (the default) the 403 comes before the lock."""
    lock_log.clear()
    resp = _client().request(method, path, json=body, headers=_bearer(LEGACY))
    assert resp.status_code == 403, resp.text
    assert "Belief writes are disabled" in resp.text
    assert lock_log == [], "the store writer lock was taken before the belief-write 403"


def test_an_unknown_relation_type_takes_no_writer_lock(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock_log: list[str]
) -> None:
    from particles.config import get_config, reset_config

    _write_config(tmp_path, monkeypatch, "mcp:\n  write:\n    enabled_stores: [default]\n")
    reset_config()
    assert get_config().mcp.write.enabled_stores == ["default"]
    lock_log.clear()
    resp = _client().post(
        "/links",
        json={"particle_a": "a", "particle_b": "b", "relation_type": "NOT_A_KIND"},
        headers=_bearer(LEGACY),
    )
    assert resp.status_code == 400, resp.text
    assert lock_log == [], "the 400 for an unknown relation_type came after the writer lock"


def test_an_agent_text_deposit_with_a_malformed_body_takes_no_writer_lock(
    store: Path, lock_log: list[str]
) -> None:
    """A 422 for a body that does not validate comes back without the writer lock."""
    token = _agent()
    lock_log.clear()
    resp = _client().post("/corpus/deposit/text", json={"text": 1}, headers=_bearer(token))
    assert resp.status_code == 422, resp.text
    assert lock_log == [], "the writer lock was taken before the body was validated"


def test_an_accepted_agent_text_deposit_takes_the_writer_lock(
    store: Path, lock_log: list[str]
) -> None:
    """Control: a deposit that is admitted still writes under the store's writer lock."""
    token = _agent()
    lock_log.clear()
    resp = _client().post("/corpus/deposit/text", json={"text": "x"}, headers=_bearer(token))
    assert resp.status_code == 200, resp.text
    assert lock_log == ["default"]


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/curation/affirm", {"particle_id": "p-1"}),
        ("/curation/snooze", {"card_key": "contested:p-1", "snooze_days": 3}),
    ],
    ids=["affirm", "snooze"],
)
def test_an_operator_curation_write_commits_under_the_writer_lock(
    store: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lock_log: list[str],
    path: str,
    body: dict[str, object],
) -> None:
    """A curation gesture's event commits under the store's writer lock (Z47)."""
    from particles.config import reset_config

    _write_config(tmp_path, monkeypatch, "mcp:\n  write:\n    enabled_stores: [default]\n")
    reset_config()
    lock_log.clear()
    resp = _client().post(path, json=body, headers=_bearer(LEGACY))
    assert resp.status_code == 200, resp.text
    assert lock_log == ["default"]


def test_an_operator_subject_assign_refusal_takes_no_writer_lock(
    store: Path, lock_log: list[str]
) -> None:
    """On a store with belief writes disabled the subject assign's 403 comes before the lock."""
    lock_log.clear()
    resp = _client().post(
        "/particles/p-1/subjects", json={"subject_id": "s-1"}, headers=_bearer(LEGACY)
    )
    assert resp.status_code == 403, resp.text
    assert lock_log == []

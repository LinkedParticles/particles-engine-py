# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""An operator request refused for its body takes no writer lock.

Security review 2026-10-30-1, AB54.

The per-client operator and ``PARTICLES_API_KEY`` cells: a body refusal that
reads no stored row (an unknown trust-rule scope, a trust rank out of range,
an empty split) and a body that does not validate are both answered before the
store writer lock is taken. Scratch store and key file under tmp_path,
configuration through the loader, no model call and no outside connection.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

LEGACY = "z47-legacy-operator-key"


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
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'z47.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(keydir / "engine-keys.json"))
    monkeypatch.setenv("PARTICLES_API_KEY", LEGACY)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("network:\n  egress: none\n", encoding="utf-8")
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
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


def _client() -> TestClient:
    from particles.api.app import app

    return TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))


def _operator_token() -> str:
    from particles.api import _keys
    from particles.api._keys import Role
    from particles.config import get_config

    return _keys.mint(Role.OPERATOR, get_config().curation.operator_identity)[0]


_REFUSED = [
    ("/trust/rules", {"scope": "nonsense", "pattern": "example.org", "score": 0.5}),
    (
        "/trust/statements",
        {
            "domain": "general",
            "source_ref_type": "AUTHOR",
            "source_ref_value": "x",
            "trust_rank": 7.0,
        },
    ),
    ("/subjects/no-such-subject/split", {"particle_ids": [], "new_name": "x"}),
]


@pytest.mark.parametrize("caller", ["per-client-operator", "PARTICLES_API_KEY"])
@pytest.mark.parametrize(("path", "body"), _REFUSED)
def test_an_operator_body_refusal_that_reads_no_row_takes_no_writer_lock(
    store: Path, lock_log: list[str], caller: str, path: str, body: dict[str, object]
) -> None:
    """Safe property: a request refused for its body alone, by any operator
    caller, is refused before the store writer lock is taken."""
    token = _operator_token() if caller == "per-client-operator" else LEGACY
    lock_log.clear()
    resp = _client().post(path, json=body, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code in (400, 422), resp.text
    assert lock_log == [], f"{path} refused {resp.status_code} under the writer lock"


@pytest.mark.parametrize("path", ["/trust/rules", "/trust/statements", "/subjects/s/aliases"])
def test_an_operator_body_that_does_not_validate_takes_no_writer_lock(
    store: Path, lock_log: list[str], path: str
) -> None:
    """Safe property: a 422 for a body that does not validate is given before
    the store writer lock is taken, for the per-client operator key."""
    token = _operator_token()
    lock_log.clear()
    resp = _client().post(
        path, json={"unexpected": True}, headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 422, resp.text
    assert lock_log == [], f"{path} answered 422 under the writer lock"

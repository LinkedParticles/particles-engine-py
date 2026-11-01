# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0
"""An HTTP file upload cannot write onto an entry the engine enrolled from its own disk.

Security review 2026-10-24 (third delta), P19: the upload route writes the
bytes to a temporary path, so the deposit fell through to a content-hash match
and appended to, and retyped, a ``rules sync`` entry holding the same bytes.
The upload is also copied in blocks under ``local_files.max_bytes`` rather than
read whole into memory.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests._engine_client import keyed_engine_client


@pytest.fixture
def api_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    db_path = tmp_path / "api.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    from particles.config import reset_config

    reset_config()

    async def _create() -> None:
        import particles._orm_modules  # noqa: F401
        from particles.db import Base, get_engine

        async with get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_create())
    yield db_path

    async def _dispose() -> None:
        from particles.db import get_engine

        await get_engine().dispose()

    asyncio.run(_dispose())
    reset_config()


@pytest.fixture
def client(api_db: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    return keyed_engine_client(monkeypatch)


async def _snapshot_count(entry_id: str) -> int:
    from particles.corpus.store import list_snapshots_for_entry
    from particles.db import session_scope

    async with session_scope() as session:
        return len(await list_snapshots_for_entry(session, entry_id))


async def _rule_sync_entry(uri: str, text: str) -> str:
    from particles.core.schema import FetchPolicy, Mutability
    from particles.corpus.deposit import deposit_text_versioned
    from particles.db import session_scope

    async with session_scope() as session:
        entry_id, _snap, _unchanged = await deposit_text_versioned(
            session,
            text=text,
            uri_r=uri,
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.MUTABLE,
            deposited_by="rule-sync",
            fetch_policy=FetchPolicy.LAZY,
        )
        await session.commit()
        return entry_id


async def _entry_source_type(entry_id: str) -> str:
    from particles.corpus.store import get_entry
    from particles.db import session_scope

    async with session_scope() as session:
        entry = await get_entry(session, entry_id)
        assert entry is not None
        return entry.source_type


def test_http_upload_cannot_retype_an_engine_enrolled_entry(client: TestClient) -> None:
    body = "the owner's rule, also published upstream\n"
    entry_id = asyncio.run(_rule_sync_entry("file:///Users/someone/src/repo/AGENTS.md", body))
    resp = client.post(
        "/corpus/deposit/file",
        files={"file": ("upload.md", body.encode(), "text/markdown")},
        data={"source_type": "ACADEMIC_PAPER"},
    )
    assert resp.status_code in (200, 409), resp.text
    assert asyncio.run(_entry_source_type(entry_id)) == "LOCAL_MARKDOWN"
    assert asyncio.run(_snapshot_count(entry_id)) == 1


def test_http_upload_from_the_same_caller_still_deduplicates(client: TestClient) -> None:
    body = b"an uploaded note\n"
    first = client.post("/corpus/deposit/file", files={"file": ("a.md", body, "text/markdown")})
    second = client.post("/corpus/deposit/file", files={"file": ("b.md", body, "text/markdown")})
    assert first.status_code == 200 and second.status_code == 200
    assert second.json()["entry_id"] == first.json()["entry_id"]
    assert second.json()["unchanged"] is True


def test_http_upload_over_the_local_file_cap_is_refused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.config import get_config

    monkeypatch.setattr(get_config().local_files, "max_bytes", 1024)
    resp = client.post(
        "/corpus/deposit/file", files={"file": ("big.md", b"x" * 4096, "text/markdown")}
    )
    assert resp.status_code == 413, resp.text

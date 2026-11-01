# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The URL deposit never holds the writer lock across a fetch (security review 2026-10-25-2, R4).

``POST /corpus/deposit/url`` used to take the writer-locked session, so the lock
every other writer queues on was held for as long as the origin took to answer.
Under :func:`particles.corpus.deposit.writes_committed_under_lock` each write
unit takes the lock and commits; the fetch runs outside it. No request leaves
the process: the fetch is an in-process stand-in.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

_ENGINE = "http://127.0.0.1:8000"
_KEY = "a-real-secret"


@pytest.fixture
def lock_state(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Count acquisitions of the writer lock and say whether it is held now."""
    import particles.db as db

    state: dict[str, Any] = {"taken": 0, "held": False}
    original = db.write_lock

    @asynccontextmanager
    async def _spy(*args: Any, **kwargs: Any) -> AsyncIterator[None]:
        state["taken"] += 1
        async with original(*args, **kwargs):
            state["held"] = True
            try:
                yield
            finally:
                state["held"] = False

    monkeypatch.setattr(db, "write_lock", _spy)
    return state


def _keyed(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from particles.api.app import app

    monkeypatch.setenv("PARTICLES_API_KEY", _KEY)
    return TestClient(
        app,
        base_url=_ENGINE,
        client=("127.0.0.1", 50000),
        headers={"Authorization": f"Bearer {_KEY}"},
    )


class _Response:
    status_code = 200
    content = b"<html><body>A page about ports.</body></html>"
    headers = {"content-type": "text/html"}

    def raise_for_status(self) -> None:
        return None


def test_a_failed_fetch_holds_no_lock(
    cli_db: Path, monkeypatch: pytest.MonkeyPatch, lock_state: dict[str, Any]
) -> None:
    """The route's whole deposit call runs outside the writer lock."""
    import particles.api.app as app_module

    seen: dict[str, bool] = {}

    async def _fetching_deposit(session: Any, url: str, **kwargs: Any) -> tuple[str, str]:
        seen["lock_held_during_fetch"] = lock_state["held"]
        await asyncio.sleep(0)
        raise app_module.SourceFetchError("stand-in fetch failed")

    monkeypatch.setattr(app_module, "deposit_url", _fetching_deposit)
    resp = _keyed(monkeypatch).post(
        "/corpus/deposit/url", json={"url": "https://example.invalid/a"}
    )
    assert resp.status_code == 502, resp.text
    assert seen == {"lock_held_during_fetch": False}


def test_a_deposit_fetches_unlocked_and_writes_under_the_lock(
    cli_db: Path, monkeypatch: pytest.MonkeyPatch, lock_state: dict[str, Any]
) -> None:
    """The fetch sees no lock, the write takes it, and the entry is committed."""
    import particles.corpus.deposit as deposit

    seen: list[bool] = []

    async def _get_capped(_client: Any, _url: str) -> _Response:
        seen.append(lock_state["held"])
        return _Response()

    @asynccontextmanager
    async def _client() -> AsyncIterator[None]:
        yield None

    monkeypatch.setattr(deposit, "get_capped", _get_capped)
    monkeypatch.setattr(deposit, "particles_client", _client)
    monkeypatch.setattr("particles.url_safety.validate_fetch_url", lambda _url: None)
    client = _keyed(monkeypatch)
    resp = client.post("/corpus/deposit/url", json={"url": "https://example.org/ports"})
    assert resp.status_code == 200, resp.text
    assert seen == [False]
    assert lock_state["taken"] >= 1
    entry = client.get(f"/corpus/{resp.json()['entry_id']}")
    assert entry.status_code == 200, entry.text


@pytest.mark.asyncio
async def test_outside_the_scope_a_deposit_writes_on_the_callers_session(
    db_session: Any, lock_state: dict[str, Any]
) -> None:
    """Every other caller keeps its own transaction: nothing commits or locks for it."""
    from particles.core.schema import FetchPolicy, Mutability, WarcRecordType
    from particles.corpus.deposit import write_entry_and_snapshot

    await write_entry_and_snapshot(
        session=db_session,
        uri_r="https://example.org/x",
        source_type="WEB_PAGE",
        mutability=Mutability.STABLE,
        fetch_policy=FetchPolicy.NEVER,
        content=b"x",
        content_hash="2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881",
        archive_path="unused",
        deposited_by="test",
        tags=[],
        warc_record_type=WarcRecordType.RESPONSE,
    )
    assert lock_state["taken"] == 0
    assert db_session.in_transaction()


def test_the_local_backend_fetches_while_another_writer_writes(
    cli_db: Path, monkeypatch: pytest.MonkeyPatch, lock_state: dict[str, Any]
) -> None:
    """A slow fetch on the local backend never makes a concurrent writer wait.

    The in-process URL deposit held the writer lock around the whole call, its
    fetch included, so every other writer queued behind the origin. It now
    commits each write unit under the lock, as the HTTP route does.
    """
    import particles.corpus.deposit as deposit
    from particles.api.client.local import LocalBackend
    from particles.db import get_engine, session_scope, write_lock

    seen: list[bool] = []

    async def _go() -> tuple[bool, str]:
        release = asyncio.Event()

        async def _slow_get_capped(_client: Any, _url: str) -> _Response:
            seen.append(lock_state["held"])
            await release.wait()
            return _Response()

        @asynccontextmanager
        async def _client() -> AsyncIterator[None]:
            yield None

        monkeypatch.setattr(deposit, "get_capped", _slow_get_capped)
        monkeypatch.setattr(deposit, "particles_client", _client)
        monkeypatch.setattr("particles.url_safety.validate_fetch_url", lambda _url: None)
        task = asyncio.create_task(
            LocalBackend().deposit_url(
                "https://example.org/slow",
                deposited_by="cli",
                source_type=None,
                tags=[],
                follow_post_links=False,
                follow_comment_links=False,
            )
        )
        while not seen:
            await asyncio.sleep(0.01)

        async def _other_writer() -> None:
            async with write_lock():
                pass

        try:
            await asyncio.wait_for(_other_writer(), timeout=2.0)
            other_wrote = True
        except TimeoutError:
            other_wrote = False
        release.set()
        outcome = await task
        async with session_scope() as s:
            from particles.corpus.store import get_entry

            assert await get_entry(s, outcome.entry_id) is not None
        await get_engine().dispose()
        return other_wrote, outcome.entry_id

    other_wrote, _entry = asyncio.run(_go())
    assert seen == [False], "the fetch ran under the writer lock"
    assert other_wrote, "a concurrent writer waited for the local backend's fetch"
    assert lock_state["taken"] >= 2

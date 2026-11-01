# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The inbox pass keeps every share and never blocks on the file (security review 2026-10-28-2).

AA4: a URL shared while the pass runs survives a rewrite that cannot re-read
the file. AA7: a FIFO, a link or an oversized file at the inbox path is refused
promptly rather than read. The URL fetch is stubbed; no network call is made.
"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

import pytest


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


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    from particles.config import reset_config

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'i.db'}")
    _write_config(tmp_path, monkeypatch, "inbox:\n  max_bytes: 4096\n")
    reset_config()
    _create_tables()
    yield
    reset_config()


def _run_pass(inbox: Path) -> dict[str, list[str]]:
    async def _go() -> dict[str, list[str]]:
        from particles.db import get_engine, session_scope
        from particles.operations.inbox import process_inbox

        async with session_scope() as s:
            summary = await process_inbox(s, inbox)
        await get_engine().dispose()
        return summary

    return asyncio.run(_go())


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-0 file")
def test_a_url_shared_during_the_pass_survives_a_failed_re_read(
    store: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A share appended while the pass runs is kept when the rewrite cannot re-read the file."""
    import particles.corpus.deposit as deposit_mod

    inbox = tmp_path / "inbox.txt"
    inbox.write_text("https://example.org/a\n", encoding="utf-8")

    async def _fake_deposit_url(session: object, url: str, **_kw: object) -> tuple[str, str]:
        with inbox.open("a", encoding="utf-8") as fh:
            fh.write("https://example.org/b\n")
        os.chmod(inbox, 0)
        return "e-1", "s-1"

    monkeypatch.setattr(deposit_mod, "deposit_url", _fake_deposit_url)
    try:
        _run_pass(inbox)
    finally:
        os.chmod(inbox, 0o600)
    text = inbox.read_text(encoding="utf-8")
    assert "https://example.org/b" in text, "a URL shared during the pass was dropped"


def test_an_inbox_replaced_by_a_fifo_does_not_block_the_pass(store: None, tmp_path: Path) -> None:
    """process_inbox returns promptly when the inbox path is a FIFO with no writer."""
    fifo = tmp_path / "inbox.txt"
    os.mkfifo(fifo)
    done = threading.Event()
    result: list[dict[str, list[str]]] = []

    def _run() -> None:
        try:
            result.append(_run_pass(fifo))
        finally:
            done.set()

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    try:
        assert done.wait(3.0), "the inbox pass blocked reading a FIFO"
    finally:
        if not done.is_set():
            fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
            os.close(fd)
            done.wait(3.0)
    assert result and not any(result[0].values())


def test_an_inbox_over_its_size_cap_is_not_read(store: None, tmp_path: Path) -> None:
    """A file over inbox.max_bytes deposits nothing and is left as it is."""
    from particles.operations.inbox import inbox_has_pending

    inbox = tmp_path / "inbox.txt"
    body = "https://example.org/a\n" + "#" * 8192 + "\n"
    inbox.write_text(body, encoding="utf-8")
    assert not any(_run_pass(inbox).values())
    assert inbox.read_text(encoding="utf-8") == body
    assert not inbox_has_pending(inbox)


def test_an_inbox_path_that_is_a_link_is_not_followed(store: None, tmp_path: Path) -> None:
    """A link at the inbox path is refused, so the pass never reads or rewrites its target."""
    target = tmp_path / "elsewhere.txt"
    target.write_text("https://example.org/a\n", encoding="utf-8")
    inbox = tmp_path / "inbox.txt"
    inbox.symlink_to(target)
    assert not any(_run_pass(inbox).values())
    assert inbox.is_symlink()
    assert target.read_text(encoding="utf-8") == "https://example.org/a\n"

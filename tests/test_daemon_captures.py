# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The daemon's captures tick: per-capture failures and the mtime gate.

Security review 2026-10-27-1: G9 and the W11 remainder (one capture the scan
cannot read never keeps the folder's other captures out) and G8 (the gate
never advances past a capture written while the scan ran).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from particles.config import get_config

GOOD = "---\nsource: https://example.org/a\ntitle: A\n---\nA clipped page body.\n"
DEEP_HEADER = "---\nk: " + "[" * 3000 + "]" * 3000 + "\n---\nBody.\n"


def _entry_count() -> int:
    async def go() -> int:
        from sqlalchemy import func, select

        from particles.corpus.store import CorpusEntryRow
        from particles.db import get_engine, session_scope

        try:
            async with session_scope() as session:
                count = await session.execute(select(func.count()).select_from(CorpusEntryRow))
                return int(count.scalar_one())
        finally:
            await get_engine().dispose()

    return asyncio.run(go())


def _run(tick: Any) -> str:
    async def go() -> str:
        from particles.db import get_engine

        try:
            return str(await tick())
        finally:
            await get_engine().dispose()

    return asyncio.run(go())


def _tick(captures: Path) -> Any:
    from particles.api.daemon import _make_web_clipper_tick

    return _make_web_clipper_tick(captures)


@pytest.fixture
def captures(cli_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(get_config().network, "egress", "none")
    root = tmp_path / "captures"
    root.mkdir()
    (root / "a.md").write_text(GOOD, encoding="utf-8")
    return root


def test_a_single_good_capture_is_deposited(captures: Path) -> None:
    assert _run(_tick(captures)) == "deposited 1"
    assert _entry_count() == 1


def test_one_refused_capture_does_not_block_every_other_capture(
    captures: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_config().local_files, "max_bytes", 2048)
    (captures / "z-big.md").write_text("x" * 4096, encoding="utf-8")
    assert _run(_tick(captures)) == "deposited 1, skipped 1"
    assert _entry_count() == 1


def test_a_symlinked_or_vanished_capture_does_not_block_the_rest(captures: Path) -> None:
    from particles.corpus.deposit import deposit_web_clipper
    from particles.db import get_engine, session_scope, write_transaction

    outside = captures.parent / "outside.md"
    outside.write_text(GOOD.replace("/a", "/o"), encoding="utf-8")
    os.link(outside, captures / "b-hardlink.md")

    async def go() -> tuple[int, list[Path]]:
        skipped: list[Path] = []
        try:
            async with session_scope() as session, write_transaction(session):
                results = await deposit_web_clipper(session, captures, skipped=skipped)
        finally:
            await get_engine().dispose()
        return len(results), skipped

    deposited, skipped = asyncio.run(go())
    assert deposited == 1
    assert [p.name for p in skipped] == ["b-hardlink.md"]
    assert _entry_count() == 1


def test_one_unbuildable_capture_does_not_stop_the_tick_committing_the_rest(
    captures: Path,
) -> None:
    """A header too deep to parse degrades to a plain local-file deposit."""
    (captures / "b.md").write_text(DEEP_HEADER, encoding="utf-8")
    assert _run(_tick(captures)) == "deposited 2"
    assert _entry_count() == 2


def test_a_capture_written_during_a_scan_is_deposited_by_a_later_tick(
    captures: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import particles.operations.deposit as ops

    real = ops.deposit_web_clipper
    seen: list[list[str]] = []

    async def _scan_then_late_write(session: Any, root: Path, **kw: Any) -> Any:
        seen.append(sorted(p.name for p in root.iterdir()))
        out = await real(session, root, **kw)
        if len(seen) == 1:
            late = root / "b.md"
            late.write_text(GOOD.replace("/a", "/b").replace("page", "late page"), encoding="utf-8")
            st = os.stat(root / "a.md")
            os.utime(late, (st.st_atime + 5, st.st_mtime + 5))
        return out

    monkeypatch.setattr(ops, "deposit_web_clipper", _scan_then_late_write)
    tick = _tick(captures)
    _run(tick)
    second = _run(tick)
    assert second != "unchanged"
    assert any("b.md" in names for names in seen[1:])
    assert _entry_count() == 2
    assert _run(tick) == "unchanged"


def test_a_capture_failing_with_a_store_error_does_not_keep_the_later_captures_out(
    captures: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O4 (G9's remainder): an error outside the read refusals skips one capture only."""
    from sqlalchemy.exc import OperationalError

    import particles.corpus.deposit as dep

    for name in ("b", "c"):
        body = GOOD.replace("/a", f"/{name}").replace("page", f"page {name}")
        (captures / f"{name}.md").write_text(body, encoding="utf-8")
    real = dep.write_entry_and_snapshot

    async def _flaky(*args: Any, **kwargs: Any) -> Any:
        if str(kwargs.get("uri_r", "")).endswith("/b"):
            raise OperationalError("INSERT", {}, Exception("database is locked"))
        return await real(*args, **kwargs)

    monkeypatch.setattr(dep, "write_entry_and_snapshot", _flaky)
    assert _run(_tick(captures)) == "deposited 2, skipped 1"
    assert _entry_count() == 2


def test_a_failed_scan_keeps_none_of_the_captures_it_deposited(captures: Path) -> None:
    """O52: a released first savepoint does not commit on its own, so a rollback undoes it."""
    from particles.corpus.deposit import deposit_web_clipper
    from particles.db import get_engine, session_scope, write_transaction

    class _Abort(Exception):
        pass

    async def go() -> None:
        try:
            async with session_scope() as session, write_transaction(session):
                await deposit_web_clipper(session, captures)
                raise _Abort
        except _Abort:
            pass
        finally:
            await get_engine().dispose()

    asyncio.run(go())
    assert _entry_count() == 0


def test_a_skipped_capture_is_retried_while_the_tree_is_unchanged(
    captures: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A capture skipped for a passing error is stored on a later tick with no new capture (O20)."""
    import particles.corpus.deposit as dep

    real = dep._deposit_capture
    failures = {"b.md": 1}

    async def _fails_once(session: Any, path: Path, **kw: Any) -> Any:
        if failures.get(path.name):
            failures[path.name] -= 1
            raise RuntimeError("transient")
        return await real(session, path, **kw)

    monkeypatch.setattr(dep, "_deposit_capture", _fails_once)
    (captures / "b.md").write_text(GOOD.replace("/a", "/b").replace("page", "other"))
    tick = _tick(captures)
    assert _run(tick) == "deposited 1, skipped 1"
    assert _run(tick) == "deposited 2"
    assert _run(tick) == "unchanged"
    assert _entry_count() == 2

# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""An unchanged inbox is not fetched again on every daemon tick (security review 2026-10-30-1, AB2).

The inbox lives in a directory the engine can read but not write (a real
``chmod`` under tmp_path, not a patch), so the pass's atomic rewrite raises.
The tick still closes its mtime gate, so the URLs the pass fetched are not
fetched again while the file is unchanged. The deposit is an in-process
stand-in that counts calls; no request leaves the process.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    from particles.config import reset_config

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'inbox.db'}")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("network:\n  egress: none\n", encoding="utf-8")
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    reset_config()
    yield tmp_path
    reset_config()


def test_a_failed_rewrite_does_not_refetch_every_tick(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Safe property: while the inbox file is unchanged, a URL a pass already
    fetched is not fetched again on every later tick because the rewrite failed."""
    import particles.corpus.deposit as deposit
    from particles.api.daemon import _make_inbox_tick

    inbox_dir = scratch / "inbox"
    inbox_dir.mkdir()
    inbox = inbox_dir / "inbox.txt"
    inbox.write_text("https://example.org/a\nhttps://example.org/b\n", encoding="utf-8")
    os.chmod(inbox_dir, 0o500)  # readable, not writable: mkstemp in the rewrite fails

    calls: list[str] = []

    async def _fake_deposit_url(_session: Any, url: str, **_kw: Any) -> tuple[str, str]:
        calls.append(url)
        return "e" * 32, "s" * 32

    monkeypatch.setattr(deposit, "deposit_url", _fake_deposit_url)
    tick = _make_inbox_tick(inbox)

    async def _go() -> int:
        from particles.db import get_engine

        failures = 0
        for _ in range(3):
            try:
                await tick()
            except OSError:
                failures += 1
        await get_engine().dispose()
        return failures

    try:
        failures = asyncio.run(_go())
    finally:
        os.chmod(inbox_dir, 0o700)
    assert failures >= 1, "the rewrite did not fail; the setup is not exercised"
    assert len(calls) <= 2, (
        f"{len(calls)} fetches over three ticks of an unchanged two-URL inbox: "
        "each tick fetched every URL again"
    )

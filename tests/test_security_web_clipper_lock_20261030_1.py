# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""``import web-clipper`` opens its transaction under the writer lock.

Security review 2026-10-30-1, AB61.

``_open_sqlite_transaction`` documents that ``BEGIN IMMEDIATE`` is issued
under the writer lock, so it waits on no cooperating writer. The web-clipper
scan calls it before each capture's savepoint, and ``import web-clipper`` now
runs that scan inside ``write_transaction``, as the daemon's tick does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner


def test_import_web_clipper_opens_its_transaction_under_the_writer_lock(
    tmp_path: Path, cli_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles import db as dbmod
    from particles.api.cli import app
    from particles.config import get_config
    from particles.corpus import deposit as dep

    assert get_config().storage.write_lock.enabled
    root = tmp_path / "clippings"
    root.mkdir()
    for i in range(3):
        (root / f"c{i}.md").write_text(f"---\nsource: https://example.com/{i}\n---\nBody {i}.\n")

    held: list[bool] = []
    original = dep._open_sqlite_transaction

    async def recording(session: Any) -> None:
        flock = dbmod._file_write_locks.get(dbmod.DEFAULT_STORE)
        held.append(bool(flock is not None and flock.is_locked))
        await original(session)

    monkeypatch.setattr(dep, "_open_sqlite_transaction", recording)
    result = CliRunner().invoke(app, ["import", "web-clipper", str(root)], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert held, "the scan never opened a transaction"
    assert all(held), held

# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The writer-lock recorder the lock-order tests install sees every acquisition.

Security review 2026-10-30-1, AB55.

``tests/test_security_lock_order_20261028_2.py`` and
``tests/test_harvester_key_roots.py`` patch ``particles.db.write_lock``. Every
Engine module takes the lock through the ``db`` module, so no module keeps a
reference of its own that the recorder would miss.
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import AsyncIterator

import pytest

_BY_NAME = [
    "particles.ingest.pipeline",
    "particles.ingest.absence",
    "particles.ingest.pending_collapse",
    "particles.operations.consolidation",
    "particles.operations.contradiction_disclosure",
]


@pytest.mark.parametrize("module", _BY_NAME)
def test_the_lock_recorder_sees_every_module_s_writer_lock(
    monkeypatch: pytest.MonkeyPatch, module: str
) -> None:
    """Safe property: a recorder installed where the lock-order tests install it
    is the writer lock every module takes."""
    import particles.db as db

    @contextlib.asynccontextmanager
    async def _recording(store: str = "default") -> AsyncIterator[None]:
        yield

    monkeypatch.setattr(db, "write_lock", _recording)
    mod = importlib.import_module(module)
    assert getattr(mod, "write_lock", db.write_lock) is db.write_lock, (
        f"{module} holds its own reference to write_lock; the recorder never sees it"
    )

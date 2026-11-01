# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""The writer-lock guard fails on the regressions it exists to catch.

Each test reverts one change in process (by patching, never by editing the
checkout) and runs one row of ``tests/test_writer_lock_guard.py`` with that
row's own assertions. The safe property is that the guard row fails on the
regression (security review 2026-10-31-2, AE15).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from tests.test_writer_lock_guard import (
    WRITERS,
    Recorder,
    _curate_apply_supersede,
    _local_backend_assert_and_supersede,
)
from tests.test_writer_lock_guard import recorder as recorder  # noqa: F401 - the guard's fixture


def _guard_row_passes(rec: Recorder, reaches: frozenset[str]) -> bool:
    reached = {
        kind
        for kind, calls in (
            ("model", rec.model),
            ("authority", rec.authority),
            ("embed", rec.embed),
        )
        if calls
    }
    return reaches <= reached and bool(rec.writes) and rec.held() == {}


@pytest.fixture
def writes_without_the_lock(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """The regression: a belief write's apply takes no writer lock at all."""
    import particles.operations.agent_write as aw
    from particles.db import write_lock_held

    commits: list[bool] = []

    @asynccontextmanager
    async def _no_lock(session: Any, store: str, commit_under_lock: bool) -> Any:
        yield
        commits.append(write_lock_held("default"))
        await session.commit()

    monkeypatch.setattr(aw, "_apply_phase", _no_lock)
    return commits


def test_the_guard_fails_when_curate_supersede_writes_with_no_writer_lock(
    recorder: Recorder,  # noqa: F811
    writes_without_the_lock: list[bool],
) -> None:
    """The guard's curate-supersede row fails when the gesture commits its belief with
    the writer lock free."""
    _curate_apply_supersede()
    assert writes_without_the_lock and not any(writes_without_the_lock), "regression not applied"
    assert not _guard_row_passes(recorder, frozenset({"model", "authority", "embed"})), (
        "the guard's row passes a curate supersede that committed its belief with no "
        "writer lock at all"
    )


def test_the_guard_fails_when_localbackend_writes_with_no_writer_lock(
    recorder: Recorder,  # noqa: F811
    writes_without_the_lock: list[bool],
) -> None:
    """The guard's LocalBackend row fails when the stdio MCP write path commits its
    belief with the writer lock free."""
    _local_backend_assert_and_supersede()
    assert writes_without_the_lock and not any(writes_without_the_lock), "regression not applied"
    assert not _guard_row_passes(recorder, frozenset({"model", "authority", "embed"})), (
        "the guard's row passes a LocalBackend write that committed with no writer lock"
    )


@pytest.mark.parametrize(
    "name", ["extraction write phase", "interchange import", "local backend assert and supersede"]
)
def test_the_guard_fails_when_a_writer_cannot_search_the_store_under_the_lock(
    name: str,
    recorder: Recorder,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The search column fails when a belief writer's apply does not read the store's
    changes under the lock (here: its gather records no change marker, so the apply
    falls back to a read the guard does not accept)."""
    import particles.ingest.pipeline as pipeline

    async def _no_marker(_session: Any) -> None:
        return None

    monkeypatch.setattr(pipeline, "read_conflict_watermark", _no_marker)
    writer = next(w for w in WRITERS if w.name == name)
    if writer.needs_tmp:
        writer.drive(tmp_path)
    else:
        writer.drive()
    assert recorder.searches, "the writer never reached the search"
    assert not all(recorder.searches), (
        f"the guard's search column passes {name} although its apply read no change marker"
    )

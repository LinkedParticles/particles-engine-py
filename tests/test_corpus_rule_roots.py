# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Recorded rule-source roots and the refresh that reads them.

``rules sync`` and ``init claude-code`` record the roots they resolve; the
local refresh admits rule files under those roots and never under the
directory it runs from (security review N38). A source outside every allowed
root is refused, counted, and reported once per run.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import pytest

from particles.core.schema import CorpusEntry, FetchPolicy, Mutability
from particles.corpus.fetch import count_refused_local_entries
from particles.corpus.refresh_outcome import RefreshOutcome
from particles.corpus.rule_roots import list_rule_source_roots, record_rule_source_roots
from particles.corpus.rule_sources import sync_rule_sources
from particles.corpus.store import CorpusEntryRow
from particles.operations.corpus_refresh import refresh_entries


async def _lazy_local_entry(session: Any, path: Path, *, by: str = "rule-sync") -> str:
    entry = CorpusEntry(
        entry_id=str(uuid.uuid4()),
        source_type="LOCAL_MARKDOWN",
        uri_r=path.resolve().as_uri(),
        fetch_policy=FetchPolicy.LAZY,
        mutability=Mutability.MUTABLE,
        deposited_by=by,
    )
    session.add(CorpusEntryRow.from_model(entry))
    await session.flush()
    return entry.entry_id


def _repo(root: Path) -> Path:
    root = root.resolve()
    (root / ".git").mkdir(parents=True)
    (root / "AGENTS.md").write_text("# Rules\n\nRun the gates before every commit.\n")
    return root


@pytest.mark.asyncio
async def test_recording_is_idempotent_and_keeps_the_first_recorder(
    db_session: Any, tmp_path: Path
) -> None:
    root = _repo(tmp_path / "project")
    await record_rule_source_roots(db_session, [root], by="rule-sync")
    await record_rule_source_roots(db_session, [root, root / "."], by="init")
    await db_session.commit()

    assert await list_rule_source_roots(db_session) == [root]
    from particles.corpus.rule_roots import RuleSourceRootRow

    row = await db_session.get(RuleSourceRootRow, str(root))
    assert row.recorded_by == "rule-sync"


@pytest.mark.asyncio
async def test_sync_records_its_roots_and_a_dry_run_records_none(
    db_session: Any, tmp_path: Path
) -> None:
    root = _repo(tmp_path / "project")

    await sync_rule_sources(db_session, [str(root)], dry_run=True)
    await db_session.commit()
    assert await list_rule_source_roots(db_session) == []

    await sync_rule_sources(db_session, [str(root), str(tmp_path / "absent")])
    await db_session.commit()
    assert await list_rule_source_roots(db_session) == [root]


@pytest.mark.asyncio
async def test_the_refresh_refuses_and_counts_a_source_outside_every_root(
    db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """Run from inside the clone, the clone's rule file is still refused (N38)."""
    project = _repo(tmp_path / "project")
    clone = _repo(tmp_path / "clone")
    admitted = await _lazy_local_entry(db_session, project / "AGENTS.md")
    refused_a = await _lazy_local_entry(db_session, clone / "AGENTS.md")
    (clone / "CLAUDE.md").write_text("x")
    refused_b = await _lazy_local_entry(db_session, clone / "CLAUDE.md")
    await record_rule_source_roots(db_session, [project], by="rule-sync")
    await db_session.commit()
    monkeypatch.chdir(clone)

    assert await count_refused_local_entries(db_session) == 2
    with caplog.at_level(logging.WARNING, logger="particles.operations.corpus_refresh"):
        results = [r async for r in refresh_entries(db_session, [admitted, refused_a, refused_b])]

    outcomes = {r.entry_id: r.outcome for r in results}
    assert outcomes[refused_a] is RefreshOutcome.REFUSED
    assert outcomes[refused_b] is RefreshOutcome.REFUSED
    assert outcomes[admitted] is not RefreshOutcome.REFUSED
    warnings = [r for r in caplog.records if "outside every allowed root" in r.getMessage()]
    assert len(warnings) == 1
    assert "2 local source(s)" in warnings[0].getMessage()
    assert "particles rules sync" in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_a_moved_root_admits_nothing_from_its_old_place(
    db_session: Any, tmp_path: Path
) -> None:
    old = _repo(tmp_path / "old")
    entry = await _lazy_local_entry(db_session, old / "AGENTS.md")
    await record_rule_source_roots(db_session, [old], by="rule-sync")
    await db_session.commit()
    (old / "AGENTS.md").unlink()

    (result,) = [r async for r in refresh_entries(db_session, [entry])]
    assert result.outcome is RefreshOutcome.MISSING


@pytest.mark.asyncio
async def test_a_recorded_root_still_refuses_an_entry_no_local_depositor_made(
    db_session: Any, tmp_path: Path
) -> None:
    """Recording a root widens nothing for an HTTP-enrolled entry (security review N21)."""
    project = _repo(tmp_path / "project")
    entry = await _lazy_local_entry(db_session, project / "AGENTS.md", by="api-user")
    await record_rule_source_roots(db_session, [project], by="rule-sync")
    await db_session.commit()

    assert await count_refused_local_entries(db_session) == 1
    (result,) = [r async for r in refresh_entries(db_session, [entry])]
    assert result.outcome is RefreshOutcome.REFUSED

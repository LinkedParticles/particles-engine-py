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
import os
import shutil
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from particles.config import reset_config
from particles.core.schema import CorpusEntry, FetchPolicy, Mutability
from particles.corpus.fetch import (
    RefreshRefusal,
    RefusalCounts,
    count_refused_local_entries,
    cwd_relative_store,
    local_refusal,
)
from particles.corpus.refresh_outcome import RefreshOutcome
from particles.corpus.rule_roots import (
    RuleSourceRootRow,
    list_rule_source_roots,
    record_rule_source_roots,
)
from particles.corpus.rule_sources import sync_rule_sources
from particles.corpus.store import CorpusEntryRow
from particles.operations.corpus_refresh import refresh_entries
from tests._upstream import upstream_only


@pytest.fixture
def compiled_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Load compiled defaults before the store session opens.

    A test that makes the refresher another user would also make the suite's
    config file another user's, and the refresh judges the file the settings
    were loaded from, so the settings are loaded from none. Requested ahead of
    ``db_session``, since reloading the config discards the store's engine.
    """
    monkeypatch.setenv("HOME", str((tmp_path / "home").resolve()))
    monkeypatch.delenv("PARTICLES_CONFIG", raising=False)
    reset_config()


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

    assert await count_refused_local_entries(db_session) == RefusalCounts(outside_roots=2)
    with caplog.at_level(logging.WARNING, logger="particles.operations.corpus_refresh"):
        results = [r async for r in refresh_entries(db_session, [admitted, refused_a, refused_b])]

    outcomes = {r.entry_id: r.outcome for r in results}
    assert outcomes[refused_a] is RefreshOutcome.REFUSED
    assert outcomes[refused_b] is RefreshOutcome.REFUSED
    assert outcomes[admitted] is not RefreshOutcome.REFUSED
    assert {r.refusal for r in results if r.outcome is RefreshOutcome.REFUSED} == {
        RefreshRefusal.OUTSIDE_ROOTS
    }
    warnings = [r for r in caplog.records if "outside every allowed root" in r.getMessage()]
    assert len(warnings) == 1
    assert "2 LAZY local source(s)" in warnings[0].getMessage()
    assert "particles rules sync" in warnings[0].getMessage()
    assert not [r for r in caplog.records if "untrusted" in r.getMessage()]


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

    assert await count_refused_local_entries(db_session) == RefusalCounts(untrusted_depositor=1)
    (result,) = [r async for r in refresh_entries(db_session, [entry])]
    assert result.outcome is RefreshOutcome.REFUSED
    assert result.refusal is RefreshRefusal.UNTRUSTED_DEPOSITOR


@pytest.mark.asyncio
async def test_each_cause_is_counted_and_warned_with_its_own_fix(
    db_session: Any, tmp_path: Path, caplog: Any
) -> None:
    """An outside-root entry and an untrusted depositor's entry get different advice."""
    project = _repo(tmp_path / "project")
    clone = _repo(tmp_path / "clone")
    untrusted = await _lazy_local_entry(db_session, project / "AGENTS.md", by="api-user")
    outside = await _lazy_local_entry(db_session, clone / "AGENTS.md")
    await record_rule_source_roots(db_session, [project], by="rule-sync")
    await db_session.commit()

    counts = await count_refused_local_entries(db_session)
    assert counts == RefusalCounts(outside_roots=1, untrusted_depositor=1)
    assert counts.total == 2
    outside_line, untrusted_line = counts.lines()
    assert outside_line.startswith("1 LAZY local source(s) lie outside every allowed root")
    assert "particles rules sync" in outside_line
    assert untrusted_line.startswith("1 LAZY local source(s) were enrolled by a depositor")
    # Re-enrolment keeps the first depositor, so the advice is the allowed root.
    assert "does not change this" in untrusted_line
    assert "local_refresh.allowed_roots" in untrusted_line

    with caplog.at_level(logging.WARNING, logger="particles.operations.corpus_refresh"):
        results = [r async for r in refresh_entries(db_session, [untrusted, outside])]
    assert {r.entry_id: r.refusal for r in results} == {
        untrusted: RefreshRefusal.UNTRUSTED_DEPOSITOR,
        outside: RefreshRefusal.OUTSIDE_ROOTS,
    }
    assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
        outside_line,
        untrusted_line,
    ]


@pytest.mark.asyncio
async def test_an_audit_enrolled_memory_file_is_refreshed(
    db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``particles audit`` deposits the memory files; the harvest makes them LAZY."""
    home = tmp_path / "home"
    memory = home / ".claude" / "projects" / "-repo" / "memory"
    memory.mkdir(parents=True)
    note = memory / "feedback.md"
    note.write_text("Commit with -s.\n")
    monkeypatch.setenv("HOME", str(home))
    entry = await _lazy_local_entry(db_session, note, by="audit")
    await record_rule_source_roots(db_session, [home / ".claude"], by="init")
    await db_session.commit()

    assert await count_refused_local_entries(db_session) == RefusalCounts()
    (result,) = [r async for r in refresh_entries(db_session, [entry])]
    assert result.outcome is RefreshOutcome.CHANGED
    assert result.refusal is None


@pytest.mark.asyncio
async def test_audit_is_trusted_only_under_the_memory_file_root(
    db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``deposited_by`` is free text over HTTP, so ``audit`` reaches no rule-file root."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = _repo(tmp_path / "project")
    entry = await _lazy_local_entry(db_session, project / "AGENTS.md", by="audit")
    await record_rule_source_roots(db_session, [project], by="rule-sync")
    await db_session.commit()

    assert await count_refused_local_entries(db_session) == RefusalCounts(untrusted_depositor=1)
    (result,) = [r async for r in refresh_entries(db_session, [entry])]
    assert result.refusal is RefreshRefusal.UNTRUSTED_DEPOSITOR


@pytest.mark.asyncio
async def test_a_planted_gitdir_line_does_not_admit_another_checkout(
    db_session: Any, tmp_path: Path
) -> None:
    """A recorded root's ``.git`` file admits a main checkout only when that worktree exists.

    Security review 2026-10-24-2, P12 (K6, refresh half): the ``gitdir:`` line
    of an archive-shipped ``.git`` file named another checkout's worktree
    directory, which did not exist, and that checkout's rule files were
    refreshed.
    """
    victim = tmp_path / "private-project"
    (victim / ".git").mkdir(parents=True)
    (victim / "AGENTS.md").write_text("# Private rules\n")
    planted = tmp_path / "download"
    planted.mkdir()
    (planted / ".git").write_text(f"gitdir: {victim / '.git' / 'worktrees' / 'x'}\n")
    entry_id = await _lazy_local_entry(db_session, victim / "AGENTS.md")
    await record_rule_source_roots(db_session, [planted], by="rule-sync")
    await db_session.commit()

    (result,) = [r async for r in refresh_entries(db_session, [entry_id])]
    assert result.outcome is RefreshOutcome.REFUSED


# ---------------------------------------------------------------------------
# the local refresh reads no host file for a store the operator did
# not name (security review 2026-10-24-2, P2).
# ---------------------------------------------------------------------------


def _root_row(path: Path) -> RuleSourceRootRow:
    """A row the store carries, as a planted store would: no sync wrote it."""
    return RuleSourceRootRow(
        path=str(path.resolve()), recorded_at=datetime.now(UTC), recorded_by="rule-sync"
    )


@pytest_asyncio.fixture
async def cwd_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncSession]:
    """A session on ``./particles.db`` in a cloned repository: the CLI's unpinned default."""
    import particles._orm_modules  # noqa: F401
    from particles.db import Base

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    clone = tmp_path / "clone"
    clone.mkdir()
    monkeypatch.chdir(clone)
    engine = create_async_engine("sqlite+aiosqlite:///./particles.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_a_store_carried_root_naming_a_non_rule_file_admits_nothing(
    db_session: Any, tmp_path: Path
) -> None:
    """A recorded root that is itself a non-rule file is refused, on any store."""
    secret = tmp_path / "home" / ".ssh" / "id_ed25519"
    secret.parent.mkdir(parents=True)
    secret.write_text("PRIVATE KEY MATERIAL\n")
    entry_id = await _lazy_local_entry(db_session, secret)
    db_session.add(_root_row(secret))
    await db_session.commit()

    (result,) = [r async for r in refresh_entries(db_session, [entry_id])]
    assert result.outcome is RefreshOutcome.REFUSED
    assert result.refusal is RefreshRefusal.OUTSIDE_ROOTS


@pytest.mark.asyncio
async def test_a_recorded_rule_file_admits_only_through_a_directory(
    db_session: Any, tmp_path: Path
) -> None:
    """A recorded file root admits nothing, even a rule-named one; its directory would."""
    project = _repo(tmp_path / "project")
    entry_id = await _lazy_local_entry(db_session, project / "AGENTS.md")
    db_session.add(_root_row(project / "AGENTS.md"))
    await db_session.commit()

    (result,) = [r async for r in refresh_entries(db_session, [entry_id])]
    assert result.outcome is RefreshOutcome.REFUSED


@pytest.mark.asyncio
async def test_a_cwd_store_reads_no_rule_file_under_its_recorded_directory(
    cwd_store: AsyncSession, tmp_path: Path
) -> None:
    """A planted store's directory row admits nothing: the store chose no root."""
    victim = _repo(tmp_path / "private-project")
    entry_id = await _lazy_local_entry(cwd_store, victim / "AGENTS.md")
    cwd_store.add(_root_row(victim))
    await cwd_store.commit()

    (result,) = [r async for r in refresh_entries(cwd_store, [entry_id])]
    assert result.outcome is RefreshOutcome.REFUSED
    assert result.refusal is RefreshRefusal.CWD_STORE
    assert await count_refused_local_entries(cwd_store) == RefusalCounts(cwd_store=1)


@pytest.mark.asyncio
async def test_a_cwd_store_reads_no_file_under_claude_either(
    cwd_store: AsyncSession, tmp_path: Path
) -> None:
    """The older reach through the store's own entries, with no row at all, is closed too."""
    rules = Path.home() / ".claude" / "CLAUDE.md"
    rules.parent.mkdir(parents=True)
    rules.write_text("# user rules\n")
    memory = Path.home() / ".claude" / "projects" / "-other" / "memory" / "note.md"
    memory.parent.mkdir(parents=True)
    memory.write_text("another project's memory\n")
    ids = [
        await _lazy_local_entry(cwd_store, rules),
        await _lazy_local_entry(cwd_store, memory, by="claude-code-hook"),
    ]
    await cwd_store.commit()

    results = [r async for r in refresh_entries(cwd_store, ids)]
    assert [r.refusal for r in results] == [RefreshRefusal.CWD_STORE] * 2


@pytest.mark.asyncio
async def test_an_absolutely_named_store_still_refreshes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import particles._orm_modules  # noqa: F401
    from particles.db import Base

    monkeypatch.chdir(tmp_path)
    project = _repo(tmp_path / "project")
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'named.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            assert cwd_relative_store(session) is None
            entry_id = await _lazy_local_entry(session, project / "AGENTS.md")
            await record_rule_source_roots(session, [project], by="rule-sync")
            await session.commit()
            (result,) = [r async for r in refresh_entries(session, [entry_id])]
    finally:
        await engine.dispose()
    assert result.outcome is not RefreshOutcome.REFUSED


@pytest.mark.asyncio
async def test_consolidation_records_a_skipped_refresh_on_a_cwd_store(
    cwd_store: AsyncSession, tmp_path: Path
) -> None:
    from particles.operations.consolidation import ConsolidationReport, _pass_refresh

    project = _repo(tmp_path / "project")
    await _lazy_local_entry(cwd_store, project / "AGENTS.md")
    await record_rule_source_roots(cwd_store, [project], by="rule-sync")
    await cwd_store.commit()

    report = ConsolidationReport()
    assert await _pass_refresh(cwd_store, report) == 0
    assert report.refresh_skipped == "cwd_store"
    assert report.refresh_checked == 0


@pytest.mark.asyncio
async def test_sync_records_no_root_for_a_file_path(db_session: Any, tmp_path: Path) -> None:
    """A file passed to ``rules sync`` is deposited, not recorded."""
    note = tmp_path / "notes" / "MY_RULES.md"
    note.parent.mkdir()
    note.write_text("# My rules\n\nAlways run the gates.\n")

    report = await sync_rule_sources(db_session, [str(note)])
    await db_session.commit()

    assert [path for path, _ in report.deposited] == [note.resolve()]
    assert await list_rule_source_roots(db_session) == []


def test_corpus_refresh_exits_naming_the_fix_on_a_cwd_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``corpus refresh`` on the CLI's unpinned default reads nothing and says why."""
    import asyncio

    from typer.testing import CliRunner

    import particles._orm_modules  # noqa: F401
    from particles.api.cli import app
    from particles.config import reset_config
    from particles.db import Base, get_engine, session_scope

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    clone = tmp_path / "clone"
    clone.mkdir()
    monkeypatch.chdir(clone)
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./particles.db")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    reset_config()
    secret = home / ".claude" / "CLAUDE.md"
    secret.parent.mkdir()
    secret.write_text("# user rules\n")

    async def _seed() -> None:
        async with get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with session_scope() as session:
            await _lazy_local_entry(session, secret)
            await session.commit()
        await get_engine().dispose()

    asyncio.run(_seed())
    result = CliRunner().invoke(app, ["corpus", "refresh"])

    assert result.exit_code == 1
    assert "resolves against the working directory" in result.stderr
    assert "absolute DATABASE_URL" in result.stderr


# ---------------------------------------------------------------------------
# a recorded root admits only files the refreshing user owns
# ---------------------------------------------------------------------------
#
# A test cannot create a file as another user, so each test gives the
# directory or file a foreign owner by moving the refresher's effective uid;
# the comparison is what they exercise. The audit reproduced the real
# cross-user case (security review 2026-10-25-1, Q19).


def _refresh_as_another_user(monkeypatch: pytest.MonkeyPatch, *, files_too: bool) -> None:
    """Make every directory this test created look owned by someone else (and its files)."""
    import particles.corpus.fetch as fetch_module
    import particles.corpus.rule_roots as rule_roots_module

    other = os.geteuid() + 1
    monkeypatch.setattr(rule_roots_module, "effective_uid", lambda: other)
    if files_too:
        monkeypatch.setattr(fetch_module, "effective_uid", lambda: other)


@pytest.mark.asyncio
async def test_a_recreated_recorded_root_admits_nothing(
    db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The audit's proof: a root removed and created again by someone else admits nothing."""
    from particles.corpus.fetch import is_refresh_allowed, refresh_refusal

    shared = (tmp_path / "shared").resolve()
    shared.mkdir()
    root = shared / "proj"
    root.mkdir()
    rule = root / "CLAUDE.md"
    rule.write_text("owner's rule\n")
    await record_rule_source_roots(db_session, [root], by="rule-sync")
    await db_session.commit()

    # The directory goes away, and someone else creates one at the same path.
    shutil.rmtree(root)
    root.mkdir()
    rule.write_text("written by whoever recreated the directory\n")
    _refresh_as_another_user(monkeypatch, files_too=True)

    recorded = await list_rule_source_roots(db_session)
    assert not is_refresh_allowed(rule, recorded_roots=recorded, deposited_by="rule-sync")
    assert (
        refresh_refusal(rule, recorded_roots=recorded, deposited_by="rule-sync")
        is RefreshRefusal.ROOT_NOT_OWNED
    )


@pytest.mark.asyncio
async def test_a_root_another_user_owns_is_counted_with_its_fix(
    db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal is counted under its own cause and worded with both fixes."""
    project = _repo(tmp_path / "project")
    entry = await _lazy_local_entry(db_session, project / "AGENTS.md")
    await record_rule_source_roots(db_session, [project], by="rule-sync")
    await db_session.commit()
    _refresh_as_another_user(monkeypatch, files_too=False)

    counts = await count_refused_local_entries(db_session)
    assert counts == RefusalCounts(root_not_owned=1)
    (line,) = counts.lines()
    assert line.startswith("1 LAZY local source(s) lie under a recorded root another user owns")
    assert "local_refresh.allowed_roots" in line
    assert "particles rules forget" in line
    (result,) = [r async for r in refresh_entries(db_session, [entry])]
    assert result.outcome is RefreshOutcome.REFUSED
    assert result.refusal is RefreshRefusal.ROOT_NOT_OWNED


@pytest.mark.asyncio
async def test_a_rule_file_another_user_wrote_under_an_owned_root_is_not_read(
    compiled_defaults: None,
    db_session: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: Any,
) -> None:
    """The root is the operator's, the file is not: the read refuses it."""
    monkeypatch.setenv("HOME", str((tmp_path / "home").resolve()))
    import particles.corpus.fetch as fetch_module
    from particles.corpus.fetch import maybe_refetch
    from particles.corpus.store import get_latest_snapshot

    project = _repo(tmp_path / "project")
    rule = project / "AGENTS.md"
    entry = await _lazy_local_entry(db_session, rule)
    await record_rule_source_roots(db_session, [project], by="rule-sync")
    await db_session.commit()
    first = await maybe_refetch(db_session, entry, force=True)
    await db_session.commit()
    assert first is not None

    rule.write_text("# Rules\n\nWritten by another user.\n")
    stamp = os.stat(rule).st_mtime + 5
    os.utime(rule, (stamp, stamp))
    monkeypatch.setattr(fetch_module, "effective_uid", lambda: os.geteuid() + 1)

    with caplog.at_level(logging.WARNING, logger="particles.corpus.fetch"):
        await maybe_refetch(db_session, entry, force=True)
    latest = await get_latest_snapshot(db_session, entry)
    assert latest is not None and latest.snapshot_id == first.snapshot_id
    assert any("owned by uid" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_main_checkout_another_user_owns_admits_nothing(
    db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The main checkout derived from a recorded worktree must be the refresher's too."""
    import particles.corpus.fetch as fetch_module
    from particles.corpus.fetch import refresh_refusal

    worktree = (tmp_path / "worktree").resolve()
    worktree.mkdir()
    main = (tmp_path / "main").resolve()
    (main / ".git" / "worktrees" / "w").mkdir(parents=True)
    (worktree / ".git").write_text(f"gitdir: {main}/.git/worktrees/w\n")
    (main / ".git" / "worktrees" / "w" / "gitdir").write_text(f"{worktree / '.git'}\n")
    (main / "AGENTS.md").write_text("main checkout rule\n")
    (worktree / "AGENTS.md").write_text("worktree rule\n")
    await record_rule_source_roots(db_session, [worktree], by="rule-sync")
    await db_session.commit()
    monkeypatch.setattr(fetch_module, "owned_by_refresher", lambda p: Path(p) != main)

    recorded = await list_rule_source_roots(db_session)
    assert refresh_refusal(worktree / "AGENTS.md", recorded_roots=recorded) is None
    assert (
        refresh_refusal(main / "AGENTS.md", recorded_roots=recorded)
        is RefreshRefusal.ROOT_NOT_OWNED
    )


@pytest.mark.asyncio
async def test_recording_skips_a_directory_another_user_owns(
    db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """``rules sync`` from a swapped-in directory records nothing."""
    project = _repo(tmp_path / "project")
    _refresh_as_another_user(monkeypatch, files_too=False)

    with caplog.at_level(logging.WARNING, logger="particles.corpus.rule_roots"):
        await record_rule_source_roots(db_session, [project], by="rule-sync")
    assert await list_rule_source_roots(db_session) == []
    (warning,) = [r.getMessage() for r in caplog.records]
    assert "not by you" in warning and "local_refresh.allowed_roots" in warning


@pytest.mark.asyncio
async def test_a_missing_root_stays_outside_the_roots(
    db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absence is judged before ownership: a missing root is OUTSIDE_ROOTS."""
    from particles.corpus.fetch import refresh_refusal

    project = _repo(tmp_path / "project")
    await record_rule_source_roots(db_session, [project], by="rule-sync")
    await db_session.commit()
    shutil.rmtree(project)
    _refresh_as_another_user(monkeypatch, files_too=True)

    recorded = await list_rule_source_roots(db_session)
    assert (
        refresh_refusal(project / "AGENTS.md", recorded_roots=recorded)
        is RefreshRefusal.OUTSIDE_ROOTS
    )


def test_the_rules_listing_marks_a_root_another_user_owns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.api.cli.rules import _root_line

    project = _repo(tmp_path / "project")
    assert _root_line(project) == f"  {project}"
    _refresh_as_another_user(monkeypatch, files_too=False)
    line = _root_line(project)
    assert "(owned by " in line and "admits nothing" in line and "rules forget" in line
    assert "(missing;" in _root_line(tmp_path / "gone")


def test_no_effective_uid_admits_and_warns_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """With no effective uid, recording goes on and the warning is said once."""
    import particles.corpus.rule_roots as rule_roots_module

    monkeypatch.delattr(os, "geteuid")
    monkeypatch.setattr(rule_roots_module, "_EUID_WARNED", False)
    with caplog.at_level(logging.WARNING, logger="particles.corpus.rule_roots"):
        assert rule_roots_module.owned_by_refresher(tmp_path)
        assert rule_roots_module.owned_by_refresher(tmp_path)
    assert len([r for r in caplog.records if "no effective uid" in r.getMessage()]) == 1


@pytest.mark.asyncio
async def test_owner_is_judged_on_the_route_of_the_opened_file(
    db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory swapped after the route lookup never chooses the owner check (R15).

    The name routes through ``~/.claude`` when it is looked up and through the
    recorded root when it is opened; the descriptor's route decides, so the
    file another user owns is refused. The swap is modelled by wrapping the
    read, so it lands after every check on the name and before the open (the
    security review's proof wrapped ``admission``, which the refresh no longer
    consults before the open).
    """
    import particles.corpus.fetch as fetch_module
    from particles.corpus.deposit import load_blob
    from particles.corpus.fetch import maybe_refetch
    from particles.corpus.store import list_snapshots_for_entry

    home = (tmp_path / "home").resolve()
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / "AGENTS.md").write_text("the owner's user-level rules\n")
    monkeypatch.setenv("HOME", str(home))
    project = _repo(tmp_path / "project")
    sub = project / "sub"
    sub.mkdir()
    (sub / "AGENTS.md").write_text("nested rules\n")
    evil = project / "evil"
    evil.mkdir()
    (evil / "AGENTS.md").write_text("written by another local user\n")
    entry = await _lazy_local_entry(db_session, sub / "AGENTS.md")
    await record_rule_source_roots(db_session, [project], by="rule-sync")
    await db_session.commit()
    # The files are not the refresher's; the roots still are (only the file
    # owner check's uid moves).
    monkeypatch.setattr(fetch_module, "effective_uid", lambda: os.geteuid() + 1)

    os.rename(sub, project / "sub.orig")
    os.symlink(home / ".claude", sub)
    real_read = fetch_module.read_local_file

    def _swap_then_read(*args: Any, **kwargs: Any) -> Any:
        # Between the checks on the name and the open, ``sub`` is repointed.
        os.unlink(sub)
        os.symlink(evil, sub)
        return real_read(*args, **kwargs)

    monkeypatch.setattr(fetch_module, "read_local_file", _swap_then_read)
    await maybe_refetch(db_session, entry, force=True)
    snapshots = await list_snapshots_for_entry(db_session, entry)
    read = [load_blob(s.content_hash) for s in snapshots if s.archive_path]
    assert b"written by another local user\n" not in read
    assert snapshots == []


@pytest.mark.asyncio
async def test_the_same_file_without_the_swap_is_refused(
    db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control for R15: a name that already routes to the recorded root is refused."""
    import particles.corpus.fetch as fetch_module
    from particles.corpus.fetch import maybe_refetch
    from particles.corpus.store import list_snapshots_for_entry

    home = (tmp_path / "home").resolve()
    (home / ".claude").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    project = _repo(tmp_path / "project")
    evil = project / "evil"
    evil.mkdir()
    (evil / "AGENTS.md").write_text("written by another local user\n")
    sub = project / "sub"
    os.symlink(evil, sub)
    entry = await _lazy_local_entry(db_session, sub / "AGENTS.md")
    await record_rule_source_roots(db_session, [project], by="rule-sync")
    await db_session.commit()
    monkeypatch.setattr(fetch_module, "effective_uid", lambda: os.geteuid() + 1)

    await maybe_refetch(db_session, entry, force=True)
    assert await list_snapshots_for_entry(db_session, entry) == []


@pytest.mark.asyncio
async def test_a_foreign_owned_rule_file_is_counted_as_refused(
    compiled_defaults: None, db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The counts agree with the read: a file the owner check refuses is counted (R42)."""
    monkeypatch.setenv("HOME", str((tmp_path / "home").resolve()))
    import particles.corpus.fetch as fetch_module
    from particles.corpus.fetch import maybe_refetch
    from particles.corpus.store import list_snapshots_for_entry

    monkeypatch.setenv("HOME", str((tmp_path / "home").resolve()))
    project = _repo(tmp_path / "project")
    entry = await _lazy_local_entry(db_session, project / "AGENTS.md")
    await record_rule_source_roots(db_session, [project], by="rule-sync")
    await db_session.commit()
    monkeypatch.setattr(fetch_module, "effective_uid", lambda: os.geteuid() + 1)

    await maybe_refetch(db_session, entry, force=True)
    assert await list_snapshots_for_entry(db_session, entry) == []

    counts = await count_refused_local_entries(db_session)
    assert counts == RefusalCounts(file_not_owned=1)
    (line,) = counts.lines()
    assert line.startswith("1 LAZY local source(s) under a recorded root are owned by another")
    (result,) = [r async for r in refresh_entries(db_session, [entry])]
    assert result.outcome is RefreshOutcome.REFUSED
    assert result.refusal is RefreshRefusal.FILE_NOT_OWNED


@pytest.mark.asyncio
async def test_a_hard_linked_rule_file_is_counted_as_refused(
    db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rule file with a second link is counted, as the contained read refuses it (R42)."""
    monkeypatch.setenv("HOME", str((tmp_path / "home").resolve()))
    project = _repo(tmp_path / "project")
    os.link(project / "AGENTS.md", tmp_path / "elsewhere.md")
    await _lazy_local_entry(db_session, project / "AGENTS.md")
    await record_rule_source_roots(db_session, [project], by="rule-sync")
    await db_session.commit()

    assert await count_refused_local_entries(db_session) == RefusalCounts(hard_linked=1)


@pytest.mark.asyncio
async def test_a_file_under_a_configured_root_needs_no_owner_in_the_counts(
    compiled_defaults: None, db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a recorded or derived route checks the file's owner."""
    monkeypatch.setenv("HOME", str((tmp_path / "home").resolve()))
    import particles.corpus.fetch as fetch_module
    from particles.config import get_config

    monkeypatch.setenv("HOME", str((tmp_path / "home").resolve()))
    project = _repo(tmp_path / "project")
    await _lazy_local_entry(db_session, project / "AGENTS.md")
    await db_session.commit()
    monkeypatch.setattr(get_config().local_refresh, "allowed_roots", [str(project)])
    monkeypatch.setattr(fetch_module, "effective_uid", lambda: os.geteuid() + 1)

    assert await count_refused_local_entries(db_session) == RefusalCounts()


# --- File-level counts and allowed roots (security review 2026-10-26-1) ------


def _rule_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from particles.config import reset_config

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    reset_config()
    root = tmp_path / "repo"
    (root / "sub").mkdir(parents=True)
    return root


def test_the_counts_refuse_a_symlinked_rule_file_the_read_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With follow_symlinks off the read refuses a symlinked rule file, so the counts do (W66)."""
    root = _rule_repo(tmp_path, monkeypatch)
    (root / "sub" / "CLAUDE.md").write_text("# rules\n")
    (root / "AGENTS.md").symlink_to(root / "sub" / "CLAUDE.md")
    assert (
        local_refusal(root / "AGENTS.md", recorded_roots=[root], deposited_by="rule-sync")
        is RefreshRefusal.SYMLINKED
    )


def test_the_counts_refuse_a_rule_file_over_the_size_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The counts name a file the read refuses for its size (R42)."""
    from particles.config import get_config

    root = _rule_repo(tmp_path, monkeypatch)
    (root / "CLAUDE.md").write_text("x" * 200)
    monkeypatch.setattr(get_config().local_files, "max_bytes", 64)
    assert (
        local_refusal(root / "CLAUDE.md", recorded_roots=[root], deposited_by="rule-sync")
        is RefreshRefusal.UNREADABLE
    )


def test_a_rule_file_another_user_can_write_is_not_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rule file under a recorded root that other users can write is refused by the
    counts and by the read (W23)."""
    from particles.corpus.local_read import LocalReadRefused, read_local_file

    root = _rule_repo(tmp_path, monkeypatch)
    rule = root / "CLAUDE.md"
    rule.write_text("# rules\n")
    os.chmod(rule, 0o666)
    assert (
        local_refusal(rule, recorded_roots=[root], deposited_by="rule-sync")
        is RefreshRefusal.WRITABLE_BY_OTHERS
    )
    with pytest.raises(LocalReadRefused):
        read_local_file(rule, owner=os.geteuid())


def test_a_broad_allowed_root_admits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """local_refresh.allowed_roots naming the home directory admits no file under it (W64)."""
    from particles.config import get_config
    from particles.corpus.fetch import broad_allowed_roots, refresh_refusal

    _rule_repo(tmp_path, monkeypatch)
    home = Path.home()
    secret = home / ".ssh" / "config"
    secret.parent.mkdir()
    secret.write_text("Host x\n")
    monkeypatch.setattr(get_config().local_refresh, "allowed_roots", [str(home)])
    assert refresh_refusal(secret, deposited_by="api-user") is not None
    assert broad_allowed_roots() == [str(home)]


# --- Stores named by an absolute path (security review 2026-10-26-1, W15) ----


def _store_session_refusal(store_file: Path) -> Path | None:
    import asyncio

    async def go() -> Path | None:
        engine = create_async_engine(f"sqlite+aiosqlite:///{store_file}")
        try:
            async with AsyncSession(engine) as session:
                return cwd_relative_store(session)
        finally:
            await engine.dispose()

    return asyncio.run(go())


def test_store_inside_a_clone_named_absolutely_chooses_no_host_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store file inside a git work tree, opened by its absolute path, admits no
    file:// row naming another project's memory file."""
    from types import SimpleNamespace

    from particles.config import reset_config
    from particles.corpus.fetch import entry_refusal

    home = tmp_path / "home"
    target = home / ".claude" / "projects" / "-other-project" / "memory" / "MEMORY.md"
    target.parent.mkdir(parents=True)
    target.write_text("owner's private memory\n")
    monkeypatch.setenv("HOME", str(home))
    reset_config()
    clone = tmp_path / "clone"
    (clone / ".git").mkdir(parents=True)

    store = _store_session_refusal(clone / "particles.db")
    assert store is not None
    row = SimpleNamespace(uri_r=target.as_uri(), deposited_by="claude-code-hook")
    assert entry_refusal(row, [], cwd_store=True) is RefreshRefusal.CWD_STORE  # type: ignore[arg-type]


def test_an_absolute_store_outside_any_repository_is_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "stores").mkdir()
    assert _store_session_refusal(tmp_path / "stores" / "particles.db") is None


def test_a_git_directory_at_home_does_not_refuse_every_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dotfiles repository at home is broad and does not count."""
    home = tmp_path / "home"
    (home / ".git").mkdir(parents=True)
    (home / ".particles").mkdir()
    monkeypatch.setenv("HOME", str(home))
    assert _store_session_refusal(home / ".particles" / "particles.db") is None


def test_a_store_file_other_users_may_write_is_not_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store the operator owns but others may write chooses no host file, and says so (G16)."""
    import asyncio
    import os

    from particles.corpus.fetch import RefusalCounts, store_refusal

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    shared = tmp_path / "shared"
    shared.mkdir()
    store = shared / "particles.db"
    store.write_bytes(b"")
    os.chmod(store, 0o666)
    assert _store_session_refusal(store) is not None

    async def reason() -> RefreshRefusal | None:
        engine = create_async_engine(f"sqlite+aiosqlite:///{store}")
        try:
            async with AsyncSession(engine) as session:
                return store_refusal(session)
        finally:
            await engine.dispose()

    assert asyncio.run(reason()) is RefreshRefusal.STORE_WRITABLE_BY_OTHERS
    (line,) = RefusalCounts(store_writable_by_others=1).lines()
    assert "chmod go-w" in line
    os.chmod(store, 0o600)
    assert _store_session_refusal(store) is None


def test_a_store_file_that_cannot_be_judged_is_not_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store file that exists but cannot be opened to read its owner and mode is refused."""
    import os

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "stores").mkdir()
    store = tmp_path / "stores" / "particles.db"
    store.write_bytes(b"")
    os.chmod(store, 0o000)
    try:
        if os.access(store, os.R_OK):
            pytest.skip("running as a user every mode admits")
        assert _store_session_refusal(store) is not None
    finally:
        os.chmod(store, 0o600)


def test_a_rule_file_under_a_root_others_can_write_into_is_not_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rule file under a recorded root whose directory others may write into is refused (G20)."""
    import os

    from particles.config import reset_config
    from particles.corpus.fetch import local_refusal

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    reset_config()
    root = tmp_path / "shared-checkout"
    (root / "docs").mkdir(parents=True)
    rule = root / "docs" / "CLAUDE.md"
    rule.write_text("# rules\n")
    os.chmod(rule, 0o644)
    try:
        assert local_refusal(rule, recorded_roots=[root], deposited_by="rule-sync") is None
        os.chmod(root, 0o777)
        assert (
            local_refusal(rule, recorded_roots=[root], deposited_by="rule-sync")
            is RefreshRefusal.WRITABLE_BY_OTHERS
        )
        os.chmod(root, 0o755)
        os.chmod(root / "docs", 0o775)
        assert (
            local_refusal(rule, recorded_roots=[root], deposited_by="rule-sync")
            is RefreshRefusal.WRITABLE_BY_OTHERS
        )
        os.chmod(root / "docs", 0o1777)
        assert local_refusal(rule, recorded_roots=[root], deposited_by="rule-sync") is None
    finally:
        os.chmod(root, 0o755)
        os.chmod(root / "docs", 0o755)


@pytest.mark.asyncio
async def test_the_refresh_reads_no_rule_file_from_a_directory_others_can_write_into(
    db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refresh read itself refuses the file, not only the counts (G20)."""
    import os

    project = _repo(tmp_path / "project")
    entry_id = await _lazy_local_entry(db_session, project / "AGENTS.md")
    await record_rule_source_roots(db_session, [project], by="rule-sync")
    await db_session.commit()
    os.chmod(project, 0o777)
    try:
        (result,) = [r async for r in refresh_entries(db_session, [entry_id])]
    finally:
        os.chmod(project, 0o755)
    assert result.outcome is RefreshOutcome.REFUSED
    assert result.refusal is RefreshRefusal.WRITABLE_BY_OTHERS


@pytest.mark.skipif(__import__("sys").platform != "darwin", reason="macOS extended ACLs")
def test_a_rule_file_with_an_inherited_acl_write_grant_is_not_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rule file every user may write through an inherited ACL entry is refused (O13)."""
    import subprocess

    from particles.config import reset_config
    from particles.corpus.local_read import acl_lets_others_write

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    reset_config()
    root = tmp_path / "shared-checkout"
    root.mkdir()
    # Applies to the files created below it only, not to the directory itself.
    subprocess.run(
        ["/bin/chmod", "+a", "everyone allow write,file_inherit,only_inherit", str(root)],
        check=True,
    )
    rule = root / "CLAUDE.md"
    rule.write_text("# rules\n")
    os.chmod(rule, 0o644)
    listing = subprocess.run(["/bin/ls", "-le", str(rule)], capture_output=True, text=True).stdout
    assert "inherited allow write" in listing
    fd = os.open(rule, os.O_RDONLY)
    try:
        assert acl_lets_others_write(fd, os.getuid())
    finally:
        os.close(fd)
    assert (
        local_refusal(rule, recorded_roots=[root], deposited_by="rule-sync")
        is RefreshRefusal.WRITABLE_BY_OTHERS
    )


@pytest.mark.skipif(__import__("sys").platform != "darwin", reason="macOS extended ACLs")
def test_a_directory_whose_inheritable_acl_lets_others_add_files_is_open_to_others(
    tmp_path: Path,
) -> None:
    """A directory whose flagged ACL entry lets every user add files is named (O22)."""
    import subprocess

    from particles.corpus.fetch import dir_open_to_others

    folder = tmp_path / "repo"
    folder.mkdir()
    os.chmod(folder, 0o755)
    path = folder / "CLAUDE.md"
    path.write_text("# rules\n")
    assert dir_open_to_others(Path(os.path.realpath(path))) is None
    subprocess.run(
        ["chmod", "+a", "everyone allow add_file,delete_child,file_inherit", str(folder)],
        check=True,
    )
    assert dir_open_to_others(Path(os.path.realpath(path))) is not None


def test_a_rule_file_whose_acl_cannot_be_judged_is_counted_as_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The counts refuse a rule file they cannot open to read its ACL (G20)."""
    from particles.config import reset_config
    from particles.corpus import fetch

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    reset_config()
    root = tmp_path / "checkout"
    root.mkdir()
    rule = root / "CLAUDE.md"
    rule.write_text("# rules\n")
    os.chmod(rule, 0o644)
    assert local_refusal(rule, recorded_roots=[root], deposited_by="rule-sync") is None
    real_open = os.open

    def no_open(path: Any, flags: int, *args: Any) -> int:
        if Path(path) == rule:
            raise PermissionError(13, "denied", str(path))
        return real_open(path, flags, *args)

    monkeypatch.setattr(fetch.os, "open", no_open)
    assert (
        local_refusal(rule, recorded_roots=[root], deposited_by="rule-sync")
        is RefreshRefusal.UNREADABLE
    )


def _private_store(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    os.chmod(folder, 0o700)
    store = folder / "particles.db"
    store.write_bytes(b"")
    os.chmod(store, 0o600)
    return store


def test_the_operators_store_alone_is_trusted(tmp_path: Path) -> None:
    """Control: an operator-owned 0600 store in an 0700 directory is trusted."""
    from particles.corpus.fetch import _store_file_refusal

    assert _store_file_refusal(_private_store(tmp_path / "home-store")) is None


@pytest.mark.parametrize("mode", [0o777, 0o1777, 0o770])
def test_a_store_in_a_directory_others_may_write_into_is_not_trusted(
    tmp_path: Path, mode: int
) -> None:
    """A store whose directory others may create names in is refused, sticky or not (O18, O23)."""
    from particles.corpus.fetch import _store_file_refusal

    store = _private_store(tmp_path / "shared")
    os.chmod(store.parent, mode)
    try:
        assert _store_file_refusal(store) is RefreshRefusal.STORE_WRITABLE_BY_OTHERS
    finally:
        os.chmod(store.parent, 0o700)


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_a_store_whose_sidecar_others_may_write_is_not_trusted(tmp_path: Path, suffix: str) -> None:
    """A store whose SQLite sidecar its group or other users may write is refused (O18, O23)."""
    from particles.corpus.fetch import _store_file_refusal

    store = _private_store(tmp_path / "wal")
    sidecar = store.with_name(store.name + suffix)
    sidecar.write_bytes(b"")
    os.chmod(sidecar, 0o600)
    assert _store_file_refusal(store) is None
    os.chmod(sidecar, 0o666)
    assert _store_file_refusal(store) is RefreshRefusal.STORE_WRITABLE_BY_OTHERS


def test_a_store_whose_rows_come_from_a_planted_wal_is_not_trusted(tmp_path: Path) -> None:
    """A store SQLite reads from a ``-wal`` other users may write chooses no host file (O18)."""
    import sqlite3

    from particles.corpus.fetch import _store_file_refusal

    def wal_store(path: Path, value: str) -> sqlite3.Connection:
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("CREATE TABLE t (x TEXT)")
        conn.execute("INSERT INTO t VALUES (?)", (value,))
        conn.commit()
        return conn

    stores = tmp_path / "stores"
    stores.mkdir()
    os.chmod(stores, 0o700)
    store = stores / "particles.db"
    wal_store(store, "operator").close()
    os.chmod(store, 0o600)
    assert _store_file_refusal(store) is None
    other = tmp_path / "other"
    other.mkdir()
    conn = wal_store(other / "x.db", "planted")
    try:
        shutil.copyfile(other / "x.db-wal", stores / "particles.db-wal")
    finally:
        conn.close()
    os.chmod(stores / "particles.db-wal", 0o666)
    reader = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    try:
        assert reader.execute("SELECT x FROM t").fetchall() == [("planted",)]
    finally:
        reader.close()
    assert _store_file_refusal(store) is RefreshRefusal.STORE_WRITABLE_BY_OTHERS


@pytest.mark.skipif(__import__("sys").platform != "darwin", reason="macOS extended ACLs")
def test_a_store_file_other_users_may_write_through_an_acl_is_not_trusted(
    tmp_path: Path,
) -> None:
    """A 0600 store an ACL entry lets every user write is refused as others-writable (O19)."""
    import subprocess

    from particles.corpus.fetch import _store_file_refusal

    store = _private_store(tmp_path / "stores")
    assert _store_file_refusal(store) is None
    subprocess.run(["/bin/chmod", "+a", "everyone allow write", str(store)], check=True)
    assert _store_file_refusal(store) is RefreshRefusal.STORE_WRITABLE_BY_OTHERS


@upstream_only
def test_the_refresh_rule_record_states_the_rule_the_code_applies() -> None:
    """The active record for the store rule names the work-tree, owner and mode refusals (G54)."""
    from tests._upstream import REPO_ROOT

    adr = next((REPO_ROOT / "docs" / "ADR" / "active").glob("0344-*.md")).read_text()
    assert "git work tree" in adr
    assert "another user owns the file" in adr
    assert "STORE_WRITABLE_BY_OTHERS" in adr
    assert "no setting overrides the rule" in adr
    assert "`-wal`, `-shm` or `-journal` sidecar" in adr
    assert "sticky bit counting for nothing" in adr
    assert "ACL entry lets" in adr

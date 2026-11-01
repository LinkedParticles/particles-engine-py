# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Forgetting a recorded rule-source root.

``particles rules forget`` removes ``rule_source_roots`` rows only, says which
sources under each root stop being refreshed and which another route still
admits, and records one operator event per removal. A dry run changes nothing.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

from particles.config import get_config
from particles.core.schema import CorpusEntry, FetchPolicy, Mutability
from particles.corpus.fetch import admission_route, is_refresh_allowed
from particles.corpus.rule_roots import (
    find_rule_source_root,
    list_rule_source_roots,
    record_rule_source_roots,
)
from particles.corpus.store import CorpusEntryRow
from particles.operations.rule_roots import forget_rule_source_roots
from particles.store.event_store import OperatorEventType, list_events


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


async def _lazy_entry(session: Any, path: Path, *, by: str = "rule-sync") -> str:
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


def _project(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "AGENTS.md").write_text("# Rules\n")
    return root.resolve()


async def test_forget_removes_the_row_and_the_refresh_refuses_its_files(
    db_session: Any, tmp_path: Path
) -> None:
    project = _project(tmp_path / "repo")
    await record_rule_source_roots(db_session, [project], by="rule-sync")
    await _lazy_entry(db_session, project / "AGENTS.md")

    report = await forget_rule_source_roots(db_session, [str(project)])
    await db_session.commit()

    assert await list_rule_source_roots(db_session) == []
    ((plan,),) = [report.plans]
    assert plan.becomes_refused == 1 and plan.still_admitted == {}
    assert not is_refresh_allowed(project / "AGENTS.md", recorded_roots=[])


async def test_forget_records_one_event_with_the_counts(db_session: Any, tmp_path: Path) -> None:
    project = _project(tmp_path / "repo")
    await record_rule_source_roots(db_session, [project], by="init")
    await _lazy_entry(db_session, project / "AGENTS.md")

    await forget_rule_source_roots(db_session, [str(project)])
    await db_session.commit()

    (event,) = await list_events(
        db_session, event_type=OperatorEventType.RULE_SOURCE_ROOT_FORGOTTEN
    )
    assert event.actor == "cli:rules-forget"
    assert event.payload is not None
    assert event.payload["path"] == str(project)
    assert event.payload["recorded_by"] == "init"
    assert event.payload["becomes_refused"] == 1
    assert event.payload["still_admitted"] == 0


async def test_a_dry_run_changes_nothing(db_session: Any, tmp_path: Path) -> None:
    project = _project(tmp_path / "repo")
    await record_rule_source_roots(db_session, [project], by="rule-sync")
    await _lazy_entry(db_session, project / "AGENTS.md")

    report = await forget_rule_source_roots(db_session, [str(project)], dry_run=True)
    await db_session.commit()

    assert report.plans[0].becomes_refused == 1
    assert await list_rule_source_roots(db_session) == [project]
    assert (
        await list_events(db_session, event_type=OperatorEventType.RULE_SOURCE_ROOT_FORGOTTEN) == []
    )


async def test_a_file_another_route_admits_is_reported_by_that_route(
    db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path / "repo")
    (project / "CLAUDE.md").write_text("# More rules\n")
    await record_rule_source_roots(db_session, [project], by="rule-sync")
    await _lazy_entry(db_session, project / "AGENTS.md")
    await _lazy_entry(db_session, project / "CLAUDE.md")
    monkeypatch.setattr(get_config().local_refresh, "allowed_roots", [str(project)])

    report = await forget_rule_source_roots(db_session, [str(project)], dry_run=True)

    plan = report.plans[0]
    assert plan.becomes_refused == 0
    assert plan.still_admitted == {f"local_refresh.allowed_roots ({project})": 2}


async def test_a_missing_root_is_matched_by_its_stored_key(db_session: Any, tmp_path: Path) -> None:
    gone = _project(tmp_path / "gone")
    await record_rule_source_roots(db_session, [gone], by="rule-sync")
    (gone / "AGENTS.md").unlink()
    gone.rmdir()

    assert (await find_rule_source_root(db_session, str(gone))) is not None
    report = await forget_rule_source_roots(db_session, [str(gone)])
    await db_session.commit()

    assert [p.path for p in report.plans] == [str(gone)]
    assert await list_rule_source_roots(db_session) == []


async def test_a_root_is_matched_through_a_symlinked_alias(db_session: Any, tmp_path: Path) -> None:
    project = _project(tmp_path / "real" / "repo")
    await record_rule_source_roots(db_session, [project], by="rule-sync")
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "real", target_is_directory=True)

    report = await forget_rule_source_roots(db_session, [str(alias / "repo")], dry_run=True)

    assert [p.path for p in report.plans] == [str(project)]


async def test_an_unrecorded_path_is_reported_and_changes_nothing(
    db_session: Any, tmp_path: Path
) -> None:
    project = _project(tmp_path / "repo")
    await record_rule_source_roots(db_session, [project], by="rule-sync")
    stranger = tmp_path / "elsewhere"

    report = await forget_rule_source_roots(db_session, [str(stranger), str(project)])
    await db_session.commit()

    assert report.unmatched == [str(stranger)]
    assert [p.path for p in report.plans] == [str(project)]
    assert await list_rule_source_roots(db_session) == []


def test_admission_route_names_the_memory_file_rule(_isolated_home: Path) -> None:
    memory = _isolated_home / ".claude" / "projects" / "-repo" / "memory"
    memory.mkdir(parents=True)
    note = memory / "note.md"
    note.write_text("a memory\n")

    assert admission_route(note, deposited_by="claude-code-hook") == (
        "the ~/.claude/projects memory-file rule"
    )
    assert admission_route(note, deposited_by="api-user") is None


class TestRulesForgetCli:
    def test_forget_prints_the_counts_and_the_next_step(self, cli_db: Path, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        rules = tmp_path / "proj" / "AGENTS.md"
        rules.parent.mkdir()
        rules.write_text("# Rules\n\nRun the gates before every commit.\n")
        runner = CliRunner()
        assert runner.invoke(app, ["rules", "sync", str(rules.parent)]).exit_code == 0

        dry = runner.invoke(app, ["rules", "forget", str(rules.parent.resolve()), "--dry-run"])
        assert dry.exit_code == 0, dry.output
        assert "would forget" in dry.output
        assert "1 LAZY source(s) it admitted stop being refreshed" in dry.output

        done = runner.invoke(app, ["rules", "forget", str(rules.parent.resolve())])
        assert done.exit_code == 0, done.output
        assert "particles corpus retract" in done.output

    def test_forgetting_nothing_recorded_exits_one(self, cli_db: Path, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        result = CliRunner().invoke(app, ["rules", "forget", str(tmp_path / "never")])
        assert result.exit_code == 1
        assert "Not a recorded refresh root" in result.output

    def test_the_report_prints_the_forget_command_beside_a_missing_root(
        self, cli_db: Path, tmp_path: Path
    ) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        project = tmp_path / "proj"
        project.mkdir()
        (project / "AGENTS.md").write_text("# Rules\n\nStage specific paths.\n")
        runner = CliRunner()
        assert runner.invoke(app, ["rules", "sync", str(project)]).exit_code == 0
        (project / "AGENTS.md").unlink()
        project.rmdir()

        report = runner.invoke(app, ["rules"])
        assert report.exit_code == 0, report.output
        assert f"particles rules forget {project.resolve()}" in report.output


def _worktree(tmp_path: Path) -> tuple[Path, Path]:
    """A recorded worktree root and the main checkout git links it to."""
    worktree = (tmp_path / "worktree").resolve()
    worktree.mkdir()
    main = (tmp_path / "main").resolve()
    (main / ".git" / "worktrees" / "w").mkdir(parents=True)
    (worktree / ".git").write_text(f"gitdir: {main}/.git/worktrees/w\n")
    (main / ".git" / "worktrees" / "w" / "gitdir").write_text(f"{worktree / '.git'}\n")
    (main / "AGENTS.md").write_text("main checkout rule\n")
    (worktree / "AGENTS.md").write_text("worktree rule\n")
    return worktree, main


async def test_forget_counts_the_main_checkout_sources_it_stops(
    db_session: Any, tmp_path: Path
) -> None:
    """Forgetting a worktree root counts its derived main checkout's sources (Q49)."""
    worktree, main = _worktree(tmp_path)
    await _lazy_entry(db_session, worktree / "AGENTS.md")
    await _lazy_entry(db_session, main / "AGENTS.md")
    await record_rule_source_roots(db_session, [worktree], by="rule-sync")
    await db_session.commit()
    recorded = await list_rule_source_roots(db_session)
    assert is_refresh_allowed(main / "AGENTS.md", recorded_roots=recorded, deposited_by="rule-sync")

    report = await forget_rule_source_roots(db_session, [str(worktree)])
    await db_session.commit()

    after = await list_rule_source_roots(db_session)
    stopped = sum(
        not is_refresh_allowed(p, recorded_roots=after, deposited_by="rule-sync")
        for p in (worktree / "AGENTS.md", main / "AGENTS.md")
    )
    assert stopped == 2
    assert [plan.becomes_refused for plan in report.plans] == [2]
    (event,) = await list_events(
        db_session, event_type=OperatorEventType.RULE_SOURCE_ROOT_FORGOTTEN
    )
    assert event.payload["becomes_refused"] == 2


async def test_forget_names_the_route_that_still_admits_a_main_checkout_source(
    db_session: Any, tmp_path: Path
) -> None:
    """A main checkout recorded in its own right keeps its sources, and the count says so (Q49)."""
    worktree, main = _worktree(tmp_path)
    await _lazy_entry(db_session, main / "AGENTS.md")
    await record_rule_source_roots(db_session, [worktree, main], by="rule-sync")
    await db_session.commit()

    report = await forget_rule_source_roots(db_session, [str(worktree)], dry_run=True)

    (plan,) = report.plans
    assert plan.becomes_refused == 0
    assert plan.still_admitted == {f"the recorded root {main}": 1}


async def test_forget_counts_a_foreign_owned_file_as_already_refused(
    db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file the owner check refuses today is not reported as one the forget stops (R42)."""
    import os

    import particles.corpus.fetch as fetch_module

    project = _project(tmp_path / "repo")
    await record_rule_source_roots(db_session, [project], by="rule-sync")
    await _lazy_entry(db_session, project / "AGENTS.md")
    monkeypatch.setattr(fetch_module, "effective_uid", lambda: os.geteuid() + 1)

    report = await forget_rule_source_roots(db_session, [str(project)], dry_run=True)

    plan = report.plans[0]
    assert plan.already_refused == 1
    assert plan.becomes_refused == 0

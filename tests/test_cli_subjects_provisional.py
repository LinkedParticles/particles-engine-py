# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""``subjects accept``, ``subjects list --provisional`` and the backfill verb."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from typer.testing import CliRunner

from particles.api.cli import app

runner = CliRunner()


def _seed(name: str, principal: str, *, provisional: bool) -> str:
    from particles.core.schema import Subject
    from particles.db import get_engine, session_scope
    from particles.store.subject_provisional import mark_provisional
    from particles.store.subject_store import insert_subject

    async def _go() -> str:
        try:
            async with session_scope() as s:
                subject = Subject(canonical_name=name, asserted_by=principal)
                await insert_subject(s, subject)
                if provisional:
                    await mark_provisional(s, subject.id, principal)
                await s.commit()
                return subject.id
        finally:
            await get_engine().dispose()

    return asyncio.run(_go())


def test_list_provisional_shows_the_queue_and_accept_empties_it(cli_db: Path) -> None:
    subject_id = _seed("the server", "agent:ci-bot", provisional=True)
    _seed("unrelated", "general-extractor", provisional=False)

    listed = runner.invoke(app, ["subjects", "list", "--provisional"])
    assert listed.exit_code == 0, listed.output
    assert "1 provisional subject(s)" in listed.output
    assert "the server" in listed.output and "agent:ci-bot" in listed.output
    assert "unrelated" not in listed.output

    shown = runner.invoke(app, ["subjects", "show", subject_id])
    assert "Provisional: agent:ci-bot" in shown.output

    accepted = runner.invoke(app, ["subjects", "accept", subject_id])
    assert accepted.exit_code == 0, accepted.output
    assert "Accepted" in accepted.output
    again = runner.invoke(app, ["subjects", "accept", subject_id])
    assert "already confirmed" in again.output
    assert (
        "No provisional subjects."
        in runner.invoke(app, ["subjects", "list", "--provisional"]).output
    )


@pytest.mark.parametrize("args", [[], ["one", "two"]])
def test_accept_needs_exactly_one_id(cli_db: Path, args: list[str]) -> None:
    result = runner.invoke(app, ["subjects", "accept", *args])
    assert result.exit_code == 1
    assert "Usage: subjects accept SUBJECT_ID" in result.output


def test_accept_of_an_unknown_subject_fails(cli_db: Path) -> None:
    result = runner.invoke(app, ["subjects", "accept", "no-such-subject"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_the_backfill_dry_run_writes_nothing(cli_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from particles.config import get_config

    get_config().mcp.write.asserter_identity = "mcp:agent"
    _seed("an agent's", "mcp:agent", provisional=False)

    dry = runner.invoke(app, ["subjects", "provisional-backfill", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "Would mark 1 subject(s)" in dry.output and "(no changes made)" in dry.output
    assert (
        "No provisional subjects."
        in runner.invoke(app, ["subjects", "list", "--provisional"]).output
    )

    live = runner.invoke(app, ["subjects", "provisional-backfill"])
    assert live.exit_code == 0, live.output
    assert "Marked 1 subject(s)" in live.output
    assert (
        "1 provisional subject(s)"
        in runner.invoke(app, ["subjects", "list", "--provisional"]).output
    )

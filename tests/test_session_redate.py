# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""``particles memory redate``: a harvested session's own start is its reference date.

* The op gives every undated snapshot of a session entry the session's start,
  re-anchors the entry's ``session``-basis claims to that day, and leaves
  every other basis, status and entry alone.
* A session the resolver cannot date is listed, not guessed; ``--dry-run``
  writes nothing; a second run changes nothing.
* The CLI verb reads the start from the transcript's first timestamp under
  the Claude Code projects root.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.core.schema import Mutability, ValidKind
from particles.core.status import Status
from particles.core.validity import (
    VALID_FROM_BASIS_KEY,
    VALID_FROM_CONFIDENCE_KEY,
    VALID_FROM_CUE_KEY,
    ValidFromBasis,
)
from tests._observer_scope import belief

runner = CliRunner()

WRONG = datetime(2026, 10, 15, tzinfo=UTC)
START = datetime(2026, 10, 9, 5, 12, tzinfo=UTC)


async def _session_entry(session: Any, sid: str) -> tuple[str, str]:
    from particles.corpus.deposit import deposit_text_versioned

    entry_id, snapshot_id, _ = await deposit_text_versioned(
        session,
        text=f"user: run the tests\nassistant: the tests passed ({sid})",
        uri_r=f"claude-code://session/{sid}",
        source_type="CONVERSATION",
        mutability=Mutability.APPEND_ONLY,
        tags=["claude-code", f"session:{sid}"],
    )
    return entry_id, snapshot_id


def _crumbs(basis: str, cue: str = "the tests passed") -> dict[str, object]:
    return {VALID_FROM_BASIS_KEY: basis, VALID_FROM_CUE_KEY: cue, VALID_FROM_CONFIDENCE_KEY: 0.9}


async def _seed(session: Any) -> dict[str, str]:
    """One datable session, one with no transcript; claims of every relevant kind."""
    e1, s1 = await _session_entry(session, "s1")
    e2, s2 = await _session_entry(session, "s2")
    ids: dict[str, str] = {"e1": e1, "e2": e2}
    ids["wrong"] = (
        await belief(
            session,
            "The tests passed.",
            (e1, s1),
            valid_from=WRONG,
            valid_kind=ValidKind.EVENT,
            properties=_crumbs("session"),
        )
    ).id
    ids["right"] = (
        await belief(
            session,
            "The cherry-pick landed.",
            (e1, s1),
            valid_from=datetime(2026, 10, 9, tzinfo=UTC),
            valid_kind=ValidKind.EVENT,
            properties=_crumbs("session", "landed"),
        )
    ).id
    ids["stated"] = (
        await belief(
            session,
            "The owner approved the design on 2026-10-10.",
            (e1, s1),
            valid_from=datetime(2026, 10, 10, tzinfo=UTC),
            valid_kind=ValidKind.EVENT,
            properties=_crumbs("stated", "on 2026-10-10"),
        )
    ).id
    ids["retired"] = (
        await belief(
            session,
            "A retired act.",
            (e1, s1),
            status=Status.SUPERSEDED,
            valid_from=WRONG,
            valid_kind=ValidKind.EVENT,
            properties=_crumbs("session"),
        )
    ).id
    ids["orphan"] = (
        await belief(
            session,
            "An act in a session with no transcript.",
            (e2, s2),
            valid_from=WRONG,
            valid_kind=ValidKind.EVENT,
            properties=_crumbs("session"),
        )
    ).id
    await session.commit()
    return ids


def _read(particle_id: str) -> tuple[datetime | None, str | None, dict[str, object]]:
    async def _get() -> tuple[datetime | None, str | None, dict[str, object]]:
        from particles.db import session_scope
        from particles.store.particle_store import ParticleRow

        async with session_scope() as session:
            row = await session.get(ParticleRow, particle_id)
            assert row is not None
            props = json.loads(row.properties_json) if row.properties_json else {}
            vf = row.valid_from
            if vf is not None and vf.tzinfo is None:
                vf = vf.replace(tzinfo=UTC)
            return vf, row.valid_kind, props

    return asyncio.run(_get())


def _published(entry_id: str) -> list[datetime | None]:
    async def _get() -> list[datetime | None]:
        from particles.corpus.store import list_snapshots_for_entry
        from particles.db import session_scope

        async with session_scope() as session:
            snaps = await list_snapshots_for_entry(session, entry_id)
        return [
            s.content_published_at.replace(tzinfo=UTC)
            if s.content_published_at is not None and s.content_published_at.tzinfo is None
            else s.content_published_at
            for s in snaps
        ]

    return asyncio.run(_get())


def _resolver(entry_id: str, uri_r: str | None, tags: list[str]) -> datetime | None:
    return START if (uri_r or "").endswith("/s1") else None


class TestRedateSessions:
    def test_redates_session_claims_and_snapshots(self, cli_db: Path) -> None:
        from particles.db import session_scope
        from particles.operations.validity import redate_sessions

        async def _run() -> tuple[dict[str, str], Any]:
            async with session_scope() as session:
                ids = await _seed(session)
            async with session_scope(write=True) as session:
                report = await redate_sessions(session, started_at_for=_resolver)
                await session.commit()
            return ids, report

        ids, report = asyncio.run(_run())
        assert report.examined == 2 and report.resolved == 1
        assert report.unresolved == [ids["e2"]]
        assert report.snapshots_dated == 1
        assert report.claims_examined == 2  # the wrong one and the right one
        assert report.claims_redated == 1 and report.claims_unchanged == 1
        assert report.claims_failed == []

        vf, kind, props = _read(ids["wrong"])
        assert vf == datetime(2026, 10, 9, tzinfo=UTC) and kind == "EVENT"
        assert props[VALID_FROM_BASIS_KEY] == ValidFromBasis.SESSION
        assert props[VALID_FROM_CUE_KEY] == "the tests passed"
        assert props[VALID_FROM_CONFIDENCE_KEY] == 0.9
        # Untouched: already on its day, another basis, retired, no transcript.
        assert _read(ids["right"])[0] == datetime(2026, 10, 9, tzinfo=UTC)
        assert _read(ids["stated"])[0] == datetime(2026, 10, 10, tzinfo=UTC)
        assert _read(ids["retired"])[0] == WRONG
        assert _read(ids["orphan"])[0] == WRONG
        assert _published(ids["e1"]) == [datetime(2026, 10, 9, tzinfo=UTC)]
        assert _published(ids["e2"]) == [None]

    def test_dry_run_writes_nothing_and_rerun_is_idempotent(self, cli_db: Path) -> None:
        from particles.db import session_scope
        from particles.operations.validity import redate_sessions
        from particles.store.event_store import OperatorEventType, list_events

        async def _run() -> tuple[dict[str, str], Any, Any, Any, int]:
            async with session_scope() as session:
                ids = await _seed(session)
            async with session_scope() as session:
                dry = await redate_sessions(session, started_at_for=_resolver, dry_run=True)
            async with session_scope(write=True) as session:
                first = await redate_sessions(session, started_at_for=_resolver)
                await session.commit()
            async with session_scope(write=True) as session:
                second = await redate_sessions(session, started_at_for=_resolver)
                await session.commit()
                events = await list_events(
                    session, event_type=OperatorEventType.SESSION_DATES_REPAIRED
                )
            return ids, dry, first, second, len(events)

        ids, dry, first, second, events = asyncio.run(_run())
        assert dry.claims_redated == 1 and dry.snapshots_dated == 0
        assert first.claims_redated == 1
        assert second.claims_redated == 0 and second.claims_unchanged == 2
        assert second.snapshots_dated == 0
        assert events == 2  # one per real run, none for the dry run
        assert _read(ids["wrong"])[0] == datetime(2026, 10, 9, tzinfo=UTC)


class TestRedateCli:
    def test_reads_the_session_start_from_the_transcript(
        self, cli_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.cli import memory as memory_cli
        from particles.db import session_scope

        ids = asyncio.run(_seed_only())
        root = tmp_path / "projects"
        (root / "-p").mkdir(parents=True)
        (root / "-p" / "s1.jsonl").write_text(
            json.dumps({"type": "user", "timestamp": START.isoformat().replace("+00:00", "Z")})
            + "\n"
        )
        monkeypatch.setattr(memory_cli, "_claude_projects_root", lambda: root)

        result = runner.invoke(app, ["memory", "redate", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "would set the date on 0 snapshot(s) and move 1 of 2" in result.output
        assert "Dry run" in result.output
        assert _read(ids["wrong"])[0] == WRONG

        result = runner.invoke(app, ["memory", "redate"])
        assert result.exit_code == 0, result.output
        assert "did set the date on 1 snapshot(s) and move 1 of 2" in result.output
        assert "1 session(s) have no transcript" in result.output
        assert ids["e2"] in result.output
        assert _read(ids["wrong"])[0] == datetime(2026, 10, 9, tzinfo=UTC)

        async def _event_count() -> int:
            from particles.store.event_store import OperatorEventType, list_events

            async with session_scope() as session:
                return len(
                    await list_events(session, event_type=OperatorEventType.SESSION_DATES_REPAIRED)
                )

        assert asyncio.run(_event_count()) == 1


async def _seed_only() -> dict[str, str]:
    from particles.db import session_scope

    async with session_scope() as session:
        return await _seed(session)

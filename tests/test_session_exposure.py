# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""the SessionStart hook records what each session was shown.

The acceptance tests 1 to 8 of the record, through the real ``hook
session-start`` verb over the ``cli_db`` file store (the harness of
``tests/test_memory_projection.py``), plus the HTTP backend routed through the
in-process engine (the ``tests/test_mcp_routing.py`` pattern).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.config import get_config
from particles.core.status import Status
from particles.render.markdown import format_memory_bullet, parse_memory_bullet
from particles.store.session_exposure_store import (
    SessionExposure,
    ShownBelief,
    session_exposures_for,
)
from tests._claude_projects import bind_hooks_to_store
from tests.test_memory_projection import (
    _MANIFEST_YAML,
    _last_log,
    _project_dir,
    _seed,
    _session_end,
)

# ---------------------------------------------------------------------------
# fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def hook_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    bind_hooks_to_store(tmp_path, f"sqlite+aiosqlite:///{tmp_path / 'cli.db'}")
    return tmp_path


@pytest.fixture
def projection_state(hook_home: Path) -> Path:
    state = hook_home / ".particles" / "claude-code"
    state.mkdir(parents=True, exist_ok=True)
    (state / "memory.yaml").write_text(_MANIFEST_YAML, encoding="utf-8")
    return state


def _start(
    runner: CliRunner, transcript: Path, *, session_id: str = "sess", source: str = "startup"
) -> Any:
    payload = {
        "session_id": session_id,
        "transcript_path": str(transcript),
        "cwd": "/p",
        "source": source,
    }
    result = runner.invoke(
        app, ["hook", "session-start", "--store", "default"], input=json.dumps(payload)
    )
    assert result.exit_code == 0
    return result


def _context(result: Any) -> str:
    return str(json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"])


def _rows(session_id: str = "sess") -> list[SessionExposure]:
    async def _run() -> list[SessionExposure]:
        from particles.db import session_scope

        async with session_scope() as session:
            return await session_exposures_for(session, session_id)

    return asyncio.run(_run())


def _count_rows() -> int:
    async def _run() -> int:
        from sqlalchemy import func, select

        from particles.db import session_scope
        from particles.store.session_exposure_store import SessionExposureRow

        async with session_scope() as session:
            return int(
                (await session.execute(select(func.count(SessionExposureRow.id)))).scalar_one()
            )

    return asyncio.run(_run())


def _retract(particle_id: str) -> None:
    async def _run() -> None:
        from particles.core.status import StatusReason
        from particles.db import session_scope
        from particles.store.particle_store import update_particle_status

        async with session_scope() as session:
            await update_particle_status(
                session, particle_id, Status.RETRACTED, StatusReason.SOURCE_RETRACTED
            )
            await session.commit()

    asyncio.run(_run())


def _ids(row: SessionExposure, shown_as: str) -> list[str]:
    return [b.particle_id for b in row.beliefs if b.shown_as == shown_as]


def _rendered(runner: CliRunner, tmp_path: Path, name: str) -> Path:
    """Run a SessionEnd so the project's MEMORY.md holds a rendered region."""
    transcript, _ = _project_dir(tmp_path, name)
    _session_end(runner, transcript)
    return transcript


# ---------------------------------------------------------------------------
# 1. full: the loaded trailer plus exactly the digest lines that survived
# ---------------------------------------------------------------------------


class TestFull:
    def test_records_exactly_the_digest_lines_that_survived_the_cut(
        self, runner: CliRunner, cli_db: Path, hook_home: Path, tmp_path: Path
    ) -> None:
        ids = _seed(*((f"Belief number {i:02d} " + "x" * 120, 0.5 + i / 100) for i in range(30)))
        content = {pid: f"Belief number {i:02d} " for i, pid in enumerate(ids)}
        get_config().claude_code.digest_max_bytes = 1500  # the cut binds
        transcript, _ = _project_dir(tmp_path, "-full")

        context = _context(_start(runner, transcript))

        (row,) = _rows()
        assert row.action == "full"
        assert row.source == "startup"
        recorded = _ids(row, "digest")
        shown = [pid for pid in ids if content[pid] in context]
        assert "digest truncated" in context
        assert 0 < len(shown) < len(ids)
        assert set(recorded) == set(shown)  # none of the cut lines
        assert recorded == sorted(shown, key=lambda pid: context.index(content[pid]))
        assert _ids(row, "projection") == []  # no projected region was loaded

    def test_full_after_a_loaded_region_records_its_trailer_first(
        self,
        runner: CliRunner,
        cli_db: Path,
        projection_state: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (a, b) = _seed(("DCO is enforced.", 0.9), ("Commits are signed.", 0.8))
        transcript = _rendered(runner, tmp_path, "-full2")

        async def _no_trailer(store: str, observer: str | None = None) -> str:
            return "- an unparseable fresh render\n"

        # A fresh render with no trailer degrades to the full push, but the
        # harness still loaded the region.
        monkeypatch.setattr("particles.api.cli._memory_projection._render_region_body", _no_trailer)
        _start(runner, transcript)

        (row,) = _rows()
        assert row.action == "full"
        assert _ids(row, "projection") == [a, b]
        assert set(_ids(row, "digest")) == {a, b}
        assert [x.shown_as for x in row.beliefs] == ["projection"] * 2 + ["digest"] * 2


# ---------------------------------------------------------------------------
# 2. diff and 3. skip
# ---------------------------------------------------------------------------


class TestProjection:
    def test_diff_records_the_loaded_region_plus_the_diff_lines(
        self, runner: CliRunner, cli_db: Path, projection_state: Path, tmp_path: Path
    ) -> None:
        a, b = _seed(("DCO is enforced.", 0.9), ("Commits are signed.", 0.8))
        transcript = _rendered(runner, tmp_path, "-diff")
        _retract(a)  # drops out of the fresh selection, still in the loaded file
        (c,) = _seed(("A brand new belief learned elsewhere.", 0.7))

        context = _context(_start(runner, transcript))
        assert "A brand new belief" in context

        (row,) = _rows()
        assert row.action == "diff"
        assert [x.particle_id for x in row.beliefs] == [a, b, c]
        assert all(x.shown_as == "projection" for x in row.beliefs)

    def test_diff_lines_cut_by_the_budget_are_not_recorded(
        self, runner: CliRunner, cli_db: Path, projection_state: Path, tmp_path: Path
    ) -> None:
        (a,) = _seed(("DCO is enforced.", 0.9))
        transcript = _rendered(runner, tmp_path, "-diffcut")
        new = _seed(*((f"Fresh belief {i:02d} " + "y" * 150, 0.5) for i in range(20)))
        get_config().claude_code.digest_max_bytes = 1200

        context = _context(_start(runner, transcript))

        (row,) = _rows()
        assert row.action == "diff"
        recorded = [x.particle_id for x in row.beliefs]
        assert recorded[0] == a
        survivors = recorded[1:]
        assert 0 < len(survivors) < len(new)
        for pid in survivors:
            assert f"p-{pid[:8]}" in context

    def test_skip_records_the_loaded_trailer(
        self, runner: CliRunner, cli_db: Path, projection_state: Path, tmp_path: Path
    ) -> None:
        a, b = _seed(("DCO is enforced.", 0.9), ("Commits are signed.", 0.8))
        transcript = _rendered(runner, tmp_path, "-skip")

        assert _start(runner, transcript).stdout == ""

        (row,) = _rows()
        assert row.action == "skip"
        assert [x.particle_id for x in row.beliefs] == [a, b]

    def test_shrink_is_a_skip_that_records_the_loaded_trailer(
        self, runner: CliRunner, cli_db: Path, projection_state: Path, tmp_path: Path
    ) -> None:
        a, b = _seed(("DCO is enforced.", 0.9), ("Commits are signed.", 0.8))
        transcript = _rendered(runner, tmp_path, "-shrink")
        _retract(a)  # the selection only shrank

        assert _start(runner, transcript).stdout == ""

        (row,) = _rows()
        assert row.action == "skip"
        assert [x.particle_id for x in row.beliefs] == [a, b]


# ---------------------------------------------------------------------------
# 4. resume and compact
# ---------------------------------------------------------------------------


class TestSources:
    def test_resume_writes_no_row(
        self, runner: CliRunner, cli_db: Path, hook_home: Path, tmp_path: Path
    ) -> None:
        _seed(("DCO is enforced.", 0.9))
        transcript, _ = _project_dir(tmp_path, "-resume")
        _start(runner, transcript, source="resume")
        assert _rows() == []

    def test_compact_writes_a_second_row_and_leaves_the_first(
        self, runner: CliRunner, cli_db: Path, hook_home: Path, tmp_path: Path
    ) -> None:
        _seed(("DCO is enforced.", 0.9))
        transcript, _ = _project_dir(tmp_path, "-compact")
        _start(runner, transcript)
        (first,) = _rows()
        _seed(("Learned before the compact.", 0.8))

        _start(runner, transcript, source="compact")

        rows = _rows()
        assert len(rows) == 2
        assert rows[0] == first
        assert [r.source for r in rows] == ["startup", "compact"]
        assert len(rows[1].beliefs) == 2

    def test_row_carries_the_project_key_and_the_observer_flag(
        self, runner: CliRunner, cli_db: Path, hook_home: Path, tmp_path: Path
    ) -> None:
        _seed(("DCO is enforced.", 0.9))
        transcript, _ = _project_dir(tmp_path, "-keyed")
        _start(runner, transcript, session_id="store-wide")
        get_config().claude_code.observer_scope = "project"
        _start(runner, transcript, session_id="scoped")

        (wide,) = _rows("store-wide")
        (scoped,) = _rows("scoped")
        assert wide.project_key == scoped.project_key == "-keyed"
        assert wide.observer_scope_applied is False
        assert scoped.observer_scope_applied is True


# ---------------------------------------------------------------------------
# 5. fail-open
# ---------------------------------------------------------------------------


class TestFailOpen:
    def test_failed_write_logs_and_still_returns_the_context(
        self,
        runner: CliRunner,
        cli_db: Path,
        hook_home: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _seed(("DCO is enforced.", 0.9))
        transcript, _ = _project_dir(tmp_path, "-failopen")
        expected = _context(_start(runner, transcript, session_id="before"))

        async def _boom(self: object, store: str, exposure: SessionExposure) -> None:
            raise RuntimeError("store unavailable")

        monkeypatch.setattr(
            "particles.api.client.local.LocalBackend.record_session_exposure", _boom
        )
        result = _start(runner, transcript, session_id="after")

        assert _context(result) == expected
        log = _last_log(tmp_path)
        assert log["outcome"] == "ok"
        assert log["exposure_error"] == "RuntimeError: store unavailable"
        assert _rows("after") == []

    def test_no_time_left_skips_the_write_and_keeps_the_context(
        self,
        runner: CliRunner,
        cli_db: Path,
        hook_home: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _seed(("DCO is enforced.", 0.9))
        transcript, _ = _project_dir(tmp_path, "-deadline")
        monkeypatch.setattr("particles.api.cli.hook._EXPOSURE_DEADLINE_MARGIN_SECONDS", 1e6)

        result = _start(runner, transcript)

        assert "DCO is enforced." in _context(result)
        assert "TimeoutError" in _last_log(tmp_path)["exposure_error"]
        assert _rows() == []


# ---------------------------------------------------------------------------
# 6. the HTTP backend
# ---------------------------------------------------------------------------


@pytest.fixture
def engine(cli_db: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[list[str], None, None]:
    """Route ``HttpBackend`` through the in-process engine; yield the recorded paths."""
    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "http://engine.test")
    monkeypatch.delenv("PARTICLES_ENGINE_TOKEN", raising=False)
    from particles.config import reset_config

    reset_config()
    from particles.api.app import app as fastapi_app

    recorded: list[str] = []

    async def _recorder(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            recorded.append(scope["path"])
        await fastapi_app(scope, receive, send)

    real_async_client = httpx.AsyncClient

    def _asgi_client(
        *, base_url: str, timeout: float, headers: dict[str, str]
    ) -> httpx.AsyncClient:
        return real_async_client(
            transport=httpx.ASGITransport(app=_recorder), base_url=base_url, headers=headers
        )

    monkeypatch.setattr(httpx, "AsyncClient", _asgi_client)
    yield recorded


class TestRemote:
    def test_http_row_lands_in_the_engine_store_with_digest_ids_only(
        self, runner: CliRunner, engine: list[str], projection_state: Path, tmp_path: Path
    ) -> None:
        a, b = _seed(("DCO is enforced.", 0.9), ("Commits are signed.", 0.8))
        transcript, memory_dir = _project_dir(tmp_path, "-remote")
        # A loaded region the remote path never reads.
        memory_dir.mkdir(parents=True)
        (memory_dir / "MEMORY.md").write_text(
            "<!-- BEGIN PROJECTED: memory-index (manifest: m) -->\n"
            f"- DCO is enforced. `p-{a[:8]}`\n<!-- sources: p-{a[:8]} -->\n"
            "<!-- END PROJECTED: memory-index -->\n"
        )

        _start(runner, transcript)

        assert "/digest/default" in engine
        assert "/session-exposures" in engine
        (row,) = _rows()
        assert row.action == "full"
        assert [(x.particle_id, x.shown_as) for x in row.beliefs] == [
            (a, "digest"),
            (b, "digest"),
        ]

    def test_engine_resolves_short_ids_on_write(self, engine: list[str]) -> None:
        (a,) = _seed(("DCO is enforced.", 0.9))
        exposure = SessionExposure(
            session_id="direct",
            recorded_at=datetime.now(UTC),
            source="startup",
            action="skip",
            project_key="-k",
            observer_scope_applied=False,
            beliefs=[
                ShownBelief(particle_id=f"p-{a[:8]}", shown_as="projection"),
                ShownBelief(particle_id="p-ffffffff", shown_as="projection"),
            ],
        )
        from particles.api.client import get_backend

        asyncio.run(get_backend().record_session_exposure("default", exposure))

        (row,) = _rows("direct")
        assert [x.particle_id for x in row.beliefs] == [a, "ffffffff"]


# ---------------------------------------------------------------------------
# 7. outside every rebuild path
# ---------------------------------------------------------------------------


class TestPrimaryRecord:
    def test_rebuild_utility_and_db_reset_leave_the_table(
        self, runner: CliRunner, cli_db: Path, hook_home: Path, tmp_path: Path
    ) -> None:
        _seed(("DCO is enforced.", 0.9))
        transcript, _ = _project_dir(tmp_path, "-rebuild")
        _start(runner, transcript)
        before = _rows()
        assert len(before) == 1

        async def _clear_and_rebuild() -> None:
            from particles.db import session_scope
            from particles.operations.utility_mining import rebuild_store_utility
            from particles.store.utility_store import clear_utility_events

            async with session_scope() as session:
                await clear_utility_events(session)
                await session.commit()
            await rebuild_store_utility("default")

        asyncio.run(_clear_and_rebuild())
        assert _rows() == before

        from particles.api.cli.db import _db_init_force

        asyncio.run(_db_init_force())
        assert _rows() == before

    def test_consolidation_run_leaves_the_table(
        self, runner: CliRunner, cli_db: Path, hook_home: Path, tmp_path: Path
    ) -> None:
        _seed(("DCO is enforced.", 0.9))
        transcript, _ = _project_dir(tmp_path, "-consolidate")
        _start(runner, transcript)
        before = _rows()
        rows_before = _count_rows()

        result = runner.invoke(app, ["memory", "consolidate"])
        # A real structural cycle: the LLM-priced passes skip with no API key.
        assert result.exit_code == 0, result.output
        assert "Consolidated store 'default'" in result.output

        assert _count_rows() == rows_before
        assert _rows() == before


# ---------------------------------------------------------------------------
# 8. contested bases
# ---------------------------------------------------------------------------


class TestContestedBases:
    def test_digest_lines_record_the_bases_they_displayed(
        self,
        runner: CliRunner,
        cli_db: Path,
        hook_home: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from particles.core.schema import ContestedBadge

        a, b = _seed(("DCO is enforced.", 0.9), ("Commits are signed.", 0.8))

        async def _badges(session: object, targets: list[Any], **_: object) -> list[Any]:
            return [
                ContestedBadge(bases=["stance", "divergence"]) if p.id == b else None
                for p in targets
            ]

        monkeypatch.setattr("particles.operations.digest.compute_contested_badges", _badges)
        transcript, _ = _project_dir(tmp_path, "-contested")

        context = _context(_start(runner, transcript))

        assert "contested (stance, divergence)" in context
        (row,) = _rows()
        bases = {x.particle_id: x.contested_bases for x in row.beliefs}
        assert bases == {a: [], b: ["stance", "divergence"]}

    @pytest.mark.parametrize(
        ("bases", "contested_by"),
        [((), None), (("stance",), None), (("stance", "inconsistency"), "abcd1234"), ((), "ab")],
    )
    def test_memory_bullets_round_trip(
        self, bases: tuple[str, ...], contested_by: str | None
    ) -> None:
        line = format_memory_bullet("A belief — with a dash", "1a2b3c4d", contested_by, bases)
        assert parse_memory_bullet(line) == ("1a2b3c4d", list(bases))

    def test_diff_lines_record_the_bases_they_displayed(self) -> None:
        from particles.api.cli._memory_projection import bullet_beliefs

        text = "\n".join(
            [
                "# Memory digest update — default",
                format_memory_bullet("Plain.", "aaaaaaaa"),
                format_memory_bullet("Disputed.", "bbbbbbbb", None, ("stance",)),
            ]
        )
        assert [(x.particle_id, x.contested_bases) for x in bullet_beliefs(text)] == [
            ("aaaaaaaa", []),
            ("bbbbbbbb", ["stance"]),
        ]
